use crate::core::backend::Backend;
use crate::core::device::Device;
use crate::core::dtype::DType;
use crate::core::error::CoreError;
use crate::core::shape::Shape;
use crate::core::tensor::TensorOps;
use crate::model::config::ModelConfig;
use crate::model::model::LlamaModel;
use crate::weights::gguf::{GgufDType, GgufFile, GgufValue};
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("gguf format error: {0}")]
    Gguf(String),
    #[error("missing metadata key: {0}")]
    MissingMetadata(String),
    #[error("missing required tensor: {0}")]
    MissingTensor(String),
    #[error("unsupported data type: {0:?}")]
    UnsupportedDType(GgufDType),
    #[error("tensor shape mismatch for {name}: expected {expected:?}, found {found:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<u64>,
        found: Vec<u64>,
    },
    #[error("backend error: {0}")]
    Backend(String),
    #[error("conversion error: {0}")]
    Conversion(String),
}

impl From<CoreError> for LoaderError {
    fn from(e: CoreError) -> Self {
        LoaderError::Backend(e.to_string())
    }
}

impl From<crate::weights::gguf::GgufError> for LoaderError {
    fn from(e: crate::weights::gguf::GgufError) -> Self {
        LoaderError::Gguf(e.to_string())
    }
}

//Which sub-layer within a transformer block.
#[derive(Debug, PartialEq, Eq)]
pub enum BlockLayer {
    AttnNorm,
    AttnQ,
    AttnK,
    AttnV,
    AttnOutput,
    FfnNorm,
    FfnGate,
    FfnUp,
    FfnDown,
}

//Identifies a named weight in the LLaMA architecture.
#[derive(Debug, PartialEq, Eq)]
pub enum LlamaTensor {
    TokenEmbd,
    OutputNorm,
    Output,
    Block(usize, BlockLayer),
}

impl LlamaTensor {
    //Parse a GGUF tensor key into a typed identifier.
    pub fn parse(key: &str) -> Option<Self> {
        match key {
            "token_embd.weight" => return Some(Self::TokenEmbd),
            "output_norm.weight" => return Some(Self::OutputNorm),
            "output.weight" => return Some(Self::Output),
            _ => {}
        }

        if key.starts_with("blk.") {
            let parts: Vec<&str> = key.split('.').collect();
            // "blk" . "{i}" . "{layer}" . "weight"
            if parts.len() < 4 {
                return None;
            }
            let i = parts[1].parse::<usize>().ok()?;
            let layer = match parts[2] {
                "attn_norm" => BlockLayer::AttnNorm,
                "attn_q" => BlockLayer::AttnQ,
                "attn_k" => BlockLayer::AttnK,
                "attn_v" => BlockLayer::AttnV,
                "attn_output" => BlockLayer::AttnOutput,
                "ffn_norm" => BlockLayer::FfnNorm,
                "ffn_gate" => BlockLayer::FfnGate,
                "ffn_up" => BlockLayer::FfnUp,
                "ffn_down" => BlockLayer::FfnDown,
                _ => return None,
            };
            return Some(Self::Block(i, layer));
        }
        None
    }
}

//Build a ModelConfig from GGUF metadata.
pub fn config_from_gguf(gguf: &GgufFile) -> Result<ModelConfig, LoaderError> {
    let req_u32 = |key: &str| -> Result<usize, LoaderError> {
        gguf.get_u32(key)
            .map(|v| v as usize)
            .ok_or_else(|| LoaderError::MissingMetadata(key.to_string()))
    };
    let req_f32 = |key: &str| -> Result<f32, LoaderError> {
        gguf.get_f32(key)
            .ok_or_else(|| LoaderError::MissingMetadata(key.to_string()))
    };

    let dtype_str = gguf
        .get_string("general.file_type")
        .unwrap_or("float32")
        .to_string();

    //map GGUF file_type integers to dtype strings when the field is numeric
    let torch_dtype = match dtype_str.as_str() {
        "0" | "float32" => "float32",
        "1" | "float16" => "float16",
        "32" | "bfloat16" => "bfloat16",
        _ => "float32",
    }
    .to_string();

    Ok(ModelConfig {
        hidden_size: req_u32("llama.embedding_length")?,
        num_hidden_layers: req_u32("llama.block_count")?,
        num_attention_heads: req_u32("llama.attention.head_count")?,
        num_key_value_heads: req_u32("llama.attention.head_count_kv")
            .unwrap_or_else(|_| req_u32("llama.attention.head_count").unwrap()),
        intermediate_size: req_u32("llama.feed_forward_length")?,
        vocab_size: req_u32("llama.vocab_size").or_else(|_| {
            // fallback: count from tokenizer vocab array
            match gguf.metadata.get("tokenizer.ggml.tokens") {
                Some(GgufValue::Array(arr)) => Ok(arr.len()),
                _ => Err(LoaderError::MissingMetadata("llama.vocab_size".into())),
            }
        })?,
        max_position_embeddings: req_u32("llama.context_length")?,
        rms_norm_eps: req_f32("llama.attention.layer_norm_rms_epsilon")
            .map(|v| v as f64)
            .unwrap_or(1e-5),
        hidden_act: "silu".to_string(),
        rope_theta: gguf
            .get_f32("llama.rope.freq_base")
            .map(|v| v as f64)
            .unwrap_or(10000.0),
        rope_scaling: None,
        torch_dtype,
        num_local_experts: None,
        num_experts_per_tok: None,
        num_shared_experts: None,
        expert_interval: None,
        prefetch_threshold: None,
        architectures: Some(vec!["LlamaForCausalLM".to_string()]),
        model_type: Some("llama".to_string()),
    })
}

//Convert a raw byte slice (f32 little-endian) into a backend tensor.
fn bytes_to_f32_tensor<B: Backend>(
    data: &[u8],
    shape: &Shape,
    device: &Device,
) -> Result<B::Tensor, LoaderError> {
    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    B::Tensor::from_slice(&floats, shape, device).map_err(|e| LoaderError::Backend(e.to_string()))
}

//Convert f16 bytes to f32 tensor.
fn bytes_f16_to_f32_tensor<B: Backend>(
    data: &[u8],
    shape: &Shape,
    device: &Device,
) -> Result<B::Tensor, LoaderError> {
    let floats: Vec<f32> = data
        .chunks_exact(2)
        .map(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            half::f16::from_bits(bits).to_f32()
        })
        .collect();
    B::Tensor::from_slice(&floats, shape, device).map_err(|e| LoaderError::Backend(e.to_string()))
}

//Load raw bytes from GGUF and produce a F32 backend tensor.
//Quantized types are stubbed with TODO — they need dequantization kernels.
pub fn load_bytes_as_tensor<B: Backend>(
    data: &[u8],
    info: &crate::weights::gguf::GgufTensorInfo,
    device: &Device,
) -> Result<B::Tensor, LoaderError> {
    //GGUF stores weights in column-major (reversed) dimension order
    let shape_dims: Vec<usize> = info.shape.iter().rev().map(|&d| d as usize).collect();
    let shape = Shape::new(&shape_dims);

    match info.dtype {
        GgufDType::F32 => bytes_to_f32_tensor::<B>(data, &shape, device),
        GgufDType::F16 => bytes_f16_to_f32_tensor::<B>(data, &shape, device),
        GgufDType::BF16 => {
            let floats: Vec<f32> = data
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect();
            B::Tensor::from_slice(&floats, &shape, device)
                .map_err(|e| LoaderError::Backend(e.to_string()))
        }
        // Quantized types — stub, dequantization kernels come in Phase 4/8
        dtype => {
            // TODO: implement dequantization for Q4_0, Q4_1, Q8_0, Q4_K, etc.
            // For now return zeros with correct shape so model structure loads
            eprintln!(
                "WARNING: tensor {} dtype {:?} — loading as zeros (dequant not yet implemented)",
                info.name, dtype
            );
            B::Tensor::zeros(&shape, DType::F32, device)
                .map_err(|e| LoaderError::Backend(e.to_string()))
        }
    }
}

//Load a LlamaModel from a GGUF file.
//Weights are assigned directly; quantized tensors are stubbed with zeros.
pub fn load_from_gguf<B: Backend>(
    path: &Path,
    device: &Device,
) -> Result<LlamaModel<B>, LoaderError> {
    let gguf = GgufFile::from_file(path)?;
    let config = config_from_gguf(&gguf)?;
    let mut model =
        LlamaModel::<B>::new(&config, device).map_err(|e| LoaderError::Backend(e.to_string()))?;

    // Load each known tensor into the model
    for (name, info) in &gguf.tensors {
        let kind = match LlamaTensor::parse(name) {
            Some(k) => k,
            None => continue, // skip unknown / rope tensors etc.
        };
        let data = gguf
            .get_tensor_data(name)
            .ok_or_else(|| LoaderError::MissingTensor(name.clone()))?;
        let tensor = load_bytes_as_tensor::<B>(data, info, device)?;
        model
            .set_tensor(&kind, tensor)
            .map_err(|e| LoaderError::Backend(e.to_string()))?;
    }

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── LlamaTensor::parse ──

    #[test]
    fn test_parse_token_embd() {
        assert_eq!(
            LlamaTensor::parse("token_embd.weight"),
            Some(LlamaTensor::TokenEmbd)
        );
    }

    #[test]
    fn test_parse_output_norm() {
        assert_eq!(
            LlamaTensor::parse("output_norm.weight"),
            Some(LlamaTensor::OutputNorm)
        );
    }

    #[test]
    fn test_parse_output() {
        assert_eq!(
            LlamaTensor::parse("output.weight"),
            Some(LlamaTensor::Output)
        );
    }

    #[test]
    fn test_parse_block_attn_q() {
        assert_eq!(
            LlamaTensor::parse("blk.0.attn_q.weight"),
            Some(LlamaTensor::Block(0, BlockLayer::AttnQ))
        );
    }

    #[test]
    fn test_parse_block_ffn_down_layer_3() {
        assert_eq!(
            LlamaTensor::parse("blk.3.ffn_down.weight"),
            Some(LlamaTensor::Block(3, BlockLayer::FfnDown))
        );
    }

    #[test]
    fn test_parse_all_block_layers() {
        let cases = vec![
            ("blk.0.attn_norm.weight", BlockLayer::AttnNorm),
            ("blk.0.attn_q.weight", BlockLayer::AttnQ),
            ("blk.0.attn_k.weight", BlockLayer::AttnK),
            ("blk.0.attn_v.weight", BlockLayer::AttnV),
            ("blk.0.attn_output.weight", BlockLayer::AttnOutput),
            ("blk.0.ffn_norm.weight", BlockLayer::FfnNorm),
            ("blk.0.ffn_gate.weight", BlockLayer::FfnGate),
            ("blk.0.ffn_up.weight", BlockLayer::FfnUp),
            ("blk.0.ffn_down.weight", BlockLayer::FfnDown),
        ];
        for (key, expected) in cases {
            match LlamaTensor::parse(key) {
                Some(LlamaTensor::Block(0, layer)) => {
                    assert_eq!(layer, expected, "failed for {}", key)
                }
                other => panic!("unexpected for {}: {:?}", key, other),
            }
        }
    }

    #[test]
    fn test_parse_high_block_index() {
        assert_eq!(
            LlamaTensor::parse("blk.31.attn_q.weight"),
            Some(LlamaTensor::Block(31, BlockLayer::AttnQ))
        );
    }

    #[test]
    fn test_parse_unknown_returns_none() {
        assert_eq!(LlamaTensor::parse("unknown.tensor"), None);
        assert_eq!(LlamaTensor::parse("blk.abc.attn_q.weight"), None);
        assert_eq!(LlamaTensor::parse("blk.0.unknown_layer.weight"), None);
        assert_eq!(LlamaTensor::parse("blk.0"), None);
    }

    // ── config_from_gguf ──

    #[test]
    fn test_config_from_gguf_minimal() {
        use crate::weights::gguf::{parse_gguf, GgufValue};

        fn kv_u32(buf: &mut Vec<u8>, key: &str, val: u32) {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&val.to_le_bytes());
        }
        fn kv_f32(buf: &mut Vec<u8>, key: &str, val: f32) {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&6u32.to_le_bytes());
            buf.extend_from_slice(&val.to_le_bytes());
        }

        let mut kvs = Vec::new();
        kv_u32(&mut kvs, "llama.embedding_length", 64);
        kv_u32(&mut kvs, "llama.block_count", 2);
        kv_u32(&mut kvs, "llama.attention.head_count", 4);
        kv_u32(&mut kvs, "llama.attention.head_count_kv", 2);
        kv_u32(&mut kvs, "llama.feed_forward_length", 128);
        kv_u32(&mut kvs, "llama.vocab_size", 1000);
        kv_u32(&mut kvs, "llama.context_length", 512);
        kv_f32(&mut kvs, "llama.attention.layer_norm_rms_epsilon", 1e-5);

        let mut buf = Vec::new();
        buf.extend_from_slice(&0x46554747u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&8u64.to_le_bytes());
        buf.extend_from_slice(&kvs);

        let gguf = parse_gguf(&buf).unwrap();
        let config = config_from_gguf(&gguf).unwrap();

        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_hidden_layers, 2);
        assert_eq!(config.num_attention_heads, 4);
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.intermediate_size, 128);
        assert_eq!(config.vocab_size, 1000);
        assert_eq!(config.max_position_embeddings, 512);
    }

    // ── load_from_gguf with real file ──

    #[test]
    fn test_load_from_gguf_real_file() {
        let path = Path::new("testdata/tinyllama.gguf");
        if !path.exists() {
            return;
        }

        use crate::core::backend::CandleBackend;
        let result = load_from_gguf::<CandleBackend>(path, &Device::Cpu);
        assert!(result.is_ok(), "load failed: {:?}", result.err());
    }

    #[test]
    fn test_config_from_real_gguf() {
        let path = Path::new("testdata/tinyllama.gguf");
        if !path.exists() {
            return;
        }

        let gguf = GgufFile::from_file(path).unwrap();
        let config = config_from_gguf(&gguf).unwrap();

        assert!(config.hidden_size > 0);
        assert!(config.num_hidden_layers > 0);
        assert!(config.num_attention_heads > 0);
        assert!(config.vocab_size > 0);
    }
}
