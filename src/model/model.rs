use crate::core::backend::Backend;
use crate::core::device::Device;
use crate::core::dtype::DType;
use crate::core::error::Result;
use crate::core::shape::Shape;
use crate::core::tensor::TensorOps;
use crate::model::block::Block;
use crate::model::config::ModelConfig;
use crate::model::embedding::Embedding;
use crate::model::linear::Linear;
use crate::model::norm::RmsNorm;

pub struct LlamaModel<B: Backend> {
    pub embedding: Embedding<B>,
    pub blocks: Vec<Block<B>>,
    norm: RmsNorm<B>,
    lm_head: Linear<B>,
    config: ModelConfig,
}

impl<B: Backend> LlamaModel<B> {
    pub fn new(config: &ModelConfig, device: &Device) -> Result<Self> {
        let embedding = Embedding::new(B::Tensor::zeros(
            &Shape::new(&[config.vocab_size, config.hidden_size]),
            DType::F32,
            device,
        )?);

        let blocks: Vec<Block<B>> = (0..config.num_hidden_layers)
            .map(|layer_idx| Block::new(config, layer_idx, device))
            .collect::<Result<Vec<_>>>()?;

        let norm = RmsNorm::new(
            B::Tensor::ones(&Shape::new(&[config.hidden_size]), DType::F32, device)?,
            config.rms_norm_eps as f32,
        );

        let lm_head = Linear::new(
            B::Tensor::zeros(
                &Shape::new(&[config.vocab_size, config.hidden_size]),
                DType::F32,
                device,
            )?,
            None,
        );

        Ok(Self {
            embedding,
            blocks,
            norm,
            lm_head,
            config: config.clone(),
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn causal_mask(&self, seq_len: usize, device: &Device) -> Result<B::Tensor> {
        if seq_len == 1 {
            return B::Tensor::from_slice(&[0.0f32], &Shape::new(&[1, 1, 1, 1]), device);
        }
        let mut mask = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    mask[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        B::Tensor::from_slice(&mask, &Shape::new(&[1, 1, seq_len, seq_len]), device)
    }

    pub fn forward(
        &mut self,
        token_ids: &[u32],
        mut kv_cache: Option<&mut Vec<(B::Tensor, B::Tensor)>>,
        offset: usize,
    ) -> Result<B::Tensor> {
        let seq_len = token_ids.len();
        let x = self.embedding.forward(token_ids)?;
        let mut x = x.unsqueeze(0)?;
        let device = x.device().clone();
        let mask = self.causal_mask(seq_len, &device)?;

        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let should_dump = offset == 0;

        // Statistical inspection helper matching your Block setup
        let dump = |label: &str, t: &B::Tensor| -> Result<()> {
            if !should_dump {
                return Ok(());
            }
            let mut v = t.to_vec_f32()?;
            v.truncate(hidden_size); // Check first token space safely
            eprintln!(
                "{label} VALUES: first3={:?} last3={:?}",
                &v[0..3],
                &v[v.len() - 3..]
            );

            let n = v.len() as f32;
            let mean = v.iter().sum::<f32>() / n;
            let var = v.iter().map(|val| (val - mean).powi(2)).sum::<f32>() / n;
            let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
            let nan_count = v.iter().filter(|val| val.is_nan()).count();
            let std = var.sqrt();

            eprintln!(
                "{label}: mean={mean:.4} std={std:.4} min={min:.4} max={max:.4} nan={nan_count}"
            );
            Ok(())
        };

        // 1. Log Token Embedding State
        dump("Model [0] - Token Embeddings", &x)?;

        for (i, block) in self.blocks.iter_mut().enumerate() {
            let cache = kv_cache
                .as_deref()
                .and_then(|v| v.get(i).map(|(k, v)| (k, v)));
            let mask_ref = if token_ids.len() > 1 {
                Some(&mask)
            } else {
                None
            };

            let (block_out, new_k, new_v) = block.forward(&x, mask_ref, cache, offset)?;
            x = block_out;

            if let Some(ref mut cache) = kv_cache {
                if i < cache.len() {
                    cache[i] = (new_k, new_v);
                } else {
                    cache.push((new_k, new_v));
                }
            }
        }

        // ---- CPU fallback region for final norm + LM head ----
        let x_pre_norm = x.to_vec_f32()?;
        let w_norm = self.norm.weight().to_vec_f32()?;
        let eps = self.norm.eps();

        // CPU RMS norm computation
        let mut normed = vec![0.0f32; x_pre_norm.len()];
        for i in 0..x_pre_norm.len() {
            let row_start = (i / hidden_size) * hidden_size;
            let row = &x_pre_norm[row_start..row_start + hidden_size];
            let ms = row.iter().map(|v| v * v).sum::<f32>() / hidden_size as f32;
            let rms = (ms + eps).sqrt();
            normed[i] = x_pre_norm[i] * w_norm[i % hidden_size] / rms;
        }

        // Create temporary view layer to safely print the RMSNorm via our helper
        let normed_tensor = B::Tensor::from_slice(&normed, &x.shape(), &device)?;
        dump("Model [Final] - Post Final Norm", &normed_tensor)?;

        // Compute LM Head Logits
        let w_lm_head = self.lm_head.weight().to_vec_f32()?;
        let mut logits_cpu = vec![0.0f32; seq_len * vocab_size];
        for pos in 0..seq_len {
            let hidden_start = pos * hidden_size;
            let hidden_row = &normed[hidden_start..hidden_start + hidden_size];
            let offset_logits = pos * vocab_size;
            for v in 0..vocab_size {
                let mut sum = 0.0;
                for d in 0..hidden_size {
                    sum += hidden_row[d] * w_lm_head[v * hidden_size + d];
                }
                logits_cpu[offset_logits + v] = sum;
            }
        }

        let logits_shape = Shape::new(&[1, seq_len, vocab_size]);
        let logits = B::Tensor::from_slice(&logits_cpu, &logits_shape, &device)?;

        Ok(logits)
    }

    pub fn set_tensor(
        &mut self,
        kind: &crate::weights::loader::LlamaTensor,
        tensor: B::Tensor,
    ) -> crate::core::error::Result<()> {
        use crate::model::embedding::Embedding;
        use crate::model::linear::Linear;
        use crate::model::norm::RmsNorm;
        use crate::weights::loader::{BlockLayer, LlamaTensor};

        match kind {
            LlamaTensor::TokenEmbd => {
                // Shape is now correctly row-major at the source (gguf.rs
                // reverses ne[] on parse), so no orientation heuristic is
                // needed or safe here anymore — trust the shape as-is.
                self.embedding = Embedding::new(tensor);
                let raw = self.embedding.weight().to_vec_f32()?;
                eprintln!(
                    "token_embd raw buffer len = {} (expected {})",
                    raw.len(),
                    128256usize * 2048
                );
                let row_start = 128000 * 2048;
                eprintln!(
                    "row 128000 direct read: first3={:?} last3={:?}",
                    &raw[row_start..row_start + 3],
                    &raw[row_start + 2045..row_start + 2048]
                );
            }
            LlamaTensor::OutputNorm => {
                let eps = self.norm.eps();
                self.norm = RmsNorm::new(tensor, eps);
            }
            LlamaTensor::Output => {
                self.lm_head = Linear::new(tensor, None);
            }
            LlamaTensor::Block(i, layer) => {
                let block = self.blocks.get_mut(*i).ok_or_else(|| {
                    crate::core::error::CoreError::Internal(format!(
                        "block index {} out of range",
                        i
                    ))
                })?;
                match layer {
                    BlockLayer::AttnNorm => block.set_attn_norm(tensor),
                    BlockLayer::AttnQ => block.set_attn_q(tensor),
                    BlockLayer::AttnK => block.set_attn_k(tensor),
                    BlockLayer::AttnV => block.set_attn_v(tensor),
                    BlockLayer::AttnOutput => block.set_attn_o(tensor),
                    BlockLayer::FfnNorm => block.set_ffn_norm(tensor),
                    BlockLayer::FfnGate => block.set_ffn_gate(tensor),
                    BlockLayer::FfnUp => block.set_ffn_up(tensor),
                    BlockLayer::FfnDown => block.set_ffn_down(tensor),
                }
            }
        }
        Ok(())
    }
}

// Tests section remains identical to preserve test safety...
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::{CandleBackend, CandleTensor};
    use crate::core::device::Device;
    use crate::core::dtype::DType;
    use crate::core::shape::Shape;
    use crate::core::tensor::TensorOps;
    use crate::model::config::ModelConfig;

    fn cpu() -> Device {
        Device::Cpu
    }

    fn make_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            hidden_act: "silu".to_string(),
            rope_theta: 10000.0,
            rope_scaling: None,
            num_local_experts: None,
            num_experts_per_tok: None,
            num_shared_experts: None,
            expert_interval: None,
            prefetch_threshold: None,
            torch_dtype: "float32".to_string(),
            architectures: None,
            model_type: Some("llama".to_string()),
        }
    }

    fn make_moe_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            hidden_act: "silu".to_string(),
            rope_theta: 10000.0,
            rope_scaling: None,
            num_local_experts: Some(4),
            num_experts_per_tok: Some(2),
            num_shared_experts: Some(1),
            expert_interval: Some(1),
            prefetch_threshold: Some(0.3),
            torch_dtype: "float32".to_string(),
            architectures: None,
            model_type: Some("deepseek".to_string()),
        }
    }

    #[test]
    fn test_model_construction() {
        let config = make_config();
        let model = LlamaModel::<CandleBackend>::new(&config, &cpu());
        assert!(model.is_ok());
    }

    #[test]
    fn test_moe_model_construction() {
        let config = make_moe_config();
        let model = LlamaModel::<CandleBackend>::new(&config, &cpu());
        assert!(model.is_ok());
    }

    #[test]
    fn test_block_count() {
        let config = make_config();
        let model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        assert_eq!(model.blocks.len(), config.num_hidden_layers);
    }

    #[test]
    fn test_forward_output_shape() {
        let config = make_config();
        let mut model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let token_ids = vec![1u32, 2, 3, 4];
        let logits = model.forward(&token_ids, None, 0).unwrap();
        assert_eq!(logits.shape().dim(0).unwrap(), 1);
        assert_eq!(logits.shape().dim(1).unwrap(), 4);
        assert_eq!(logits.shape().dims().last().unwrap(), &256);
    }

    #[test]
    fn test_forward_single_token() {
        let config = make_config();
        let mut model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let logits = model.forward(&[42u32], None, 0).unwrap();
        assert_eq!(logits.shape().dim(0).unwrap(), 1);
        assert_eq!(logits.shape().dim(1).unwrap(), 1);
        assert_eq!(logits.shape().dims().last().unwrap(), &256);
    }

    #[test]
    fn test_causal_mask_shape() {
        let config = make_config();
        let model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let device = cpu();
        let mask = model.causal_mask(4, &device).unwrap();
        assert_eq!(mask.shape(), &Shape::new(&[1, 1, 4, 4]));
    }

    #[test]
    fn test_causal_mask_values() {
        let config = make_config();
        let model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let device = cpu();
        let mask = model.causal_mask(3, &device).unwrap();
        let flat = mask.to_vec_f32().unwrap();
        assert_eq!(flat[0], 0.0);
        assert!(flat[1].is_infinite() && flat[1] < 0.0);
        assert_eq!(flat[3], 0.0);
        assert_eq!(flat[4], 0.0);
        assert!(flat[5].is_infinite() && flat[5] < 0.0);
    }

    #[test]
    fn test_forward_with_kv_cache() {
        let config = make_config();
        let mut model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let mut cache = Vec::new();
        let _ = model
            .forward(&[1u32, 2, 3, 4], Some(&mut cache), 0)
            .unwrap();
        assert_eq!(cache.len(), config.num_hidden_layers);
        let logits2 = model.forward(&[5u32], Some(&mut cache), 4).unwrap();
        assert_eq!(logits2.shape().dim(0).unwrap(), 1);
    }

    #[test]
    fn test_forward_kv_cache_grows() {
        let config = make_config();
        let mut model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let mut cache = Vec::new();
        model
            .forward(&[1u32, 2, 3, 4], Some(&mut cache), 0)
            .unwrap();
        let first_k_seq = cache[0].0.shape().dim(1).unwrap();
        model.forward(&[5u32], Some(&mut cache), 4).unwrap();
        let second_k_seq = cache[0].0.shape().dim(1).unwrap();
        assert_eq!(second_k_seq, first_k_seq + 1);
    }

    #[test]
    fn test_moe_model_forward() {
        let config = make_moe_config();
        let mut model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let logits = model.forward(&[1u32, 2, 3], None, 0).unwrap();
        assert_eq!(logits.shape().dims().last().unwrap(), &256);
    }
}

#[cfg(feature = "metal")]
#[cfg(test)]
mod metal_tests {
    use super::*;
    use crate::core::backend::MetalBackend;
    use crate::core::device::Device;
    use crate::core::tensor::TensorOps;

    fn ensure_metal_device() -> Device {
        if crate::metal::state::global_metal_state().is_none() {
            let _ = crate::metal::state::init_global_metal_state(512 * 1024 * 1024);
        }
        Device::Metal(0)
    }

    fn make_config() -> ModelConfig {
        super::make_config()
    }

    #[test]
    fn test_metal_model_construction() {
        let device = ensure_metal_device();
        let config = make_config();
        let model = LlamaModel::<MetalBackend>::new(&config, &device);
        assert!(model.is_ok());
    }

    #[test]
    fn test_metal_forward_single_token() {
        let device = ensure_metal_device();
        let config = make_config();
        let mut model = LlamaModel::<MetalBackend>::new(&config, &device).unwrap();
        let logits = model.forward(&[42u32], None, 0).unwrap();
        assert_eq!(logits.shape().dim(0).unwrap(), 1);
        assert_eq!(logits.shape().dim(1).unwrap(), 1);
    }

    #[test]
    fn test_metal_logits_are_finite() {
        let device = ensure_metal_device();
        let config = make_config();
        let mut model = LlamaModel::<MetalBackend>::new(&config, &device).unwrap();
        let logits = model.forward(&[1u32, 2, 3], None, 0).unwrap();
        let values = logits.to_vec_f32().unwrap();
        assert!(!values.iter().any(|x| x.is_nan()));
        assert!(!values.iter().any(|x| x.is_infinite()));
    }
}
