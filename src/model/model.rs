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
    embedding: Embedding<B>,
    blocks: Vec<Block<B>>,
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
            None, // no bias
        ); // [vocab_size , hidden_size]

        Ok(Self {
            embedding: embedding,
            blocks,
            norm: norm,
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
        let emb_vec = x.to_vec_f32()?;
        let emb_max = emb_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let emb_min = emb_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let device = x.device().clone();
        let mask = self.causal_mask(seq_len, &device)?;

        for (i, block) in self.blocks.iter_mut().enumerate() {
            if i == 0 {
                let b_vec = x.to_vec_f32()?;
                let b_max = b_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let b_min = b_vec.iter().cloned().fold(f32::INFINITY, f32::min);
            }

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

        // check hidden state after all blocks
        static MODEL_CHECKED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !MODEL_CHECKED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let x_vec = x.to_vec_f32()?;
            let x_max = x_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let x_min = x_vec.iter().cloned().fold(f32::INFINITY, f32::min);
            eprintln!(
                "DEBUG post-all-blocks max={:.4} min={:.4} first4={:?}",
                x_max,
                x_min,
                &x_vec[..4]
            );

            // also check embedding output for comparison
            let emb = self.embedding.forward(&[token_ids[0]])?;
            let emb_vec = emb.to_vec_f32()?;
            eprintln!("DEBUG embedding[0] first4={:?}", &emb_vec[..4]);
        }

        let norm_w = self.norm.weight().to_vec_f32()?;
        eprintln!("DEBUG output_norm first 4: {:?}", &norm_w[..4]);
        let x = self.norm.forward(&x)?;

        // temporary
        static LM_CHECKED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !LM_CHECKED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "DEBUG lm_head weight shape: {:?}",
                self.lm_head.weight().shape()
            );
        }

        let logits = self.lm_head.forward(&x)?;

        static LOGIT_CHECKED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !LOGIT_CHECKED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let l_vec = logits.to_vec_f32()?;
            let seq = logits.shape().dim(1)?;
            let vocab = logits.shape().dim(2)?;
            // last token logits
            let last = &l_vec[(seq - 1) * vocab..seq * vocab];
            let mut indexed: Vec<(usize, f32)> = last.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            eprintln!("DEBUG top5 logits: {:?}", &indexed[..5]);
        }

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
                // GGUF shape after reversal is [hidden, vocab] = [2048, 128256]
                // Embedding::forward needs [vocab, hidden] = [128256, 2048]
                let tensor = if tensor.shape().dim(0)? < tensor.shape().dim(1)? {
                    tensor.transpose(0, 1)?.contiguous()?
                } else {
                    tensor
                };
                self.embedding = Embedding::new(tensor);
            }
            LlamaTensor::OutputNorm => {
                // RmsNorm::new takes (weight, eps) — keep existing eps
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
        // [seq_len, vocab_size] — no batch dim since embedding returns [seq, hidden]
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
        // [0][0] = 0.0 — attend to self
        assert_eq!(flat[0], 0.0);
        // [0][1] = -inf — future position blocked
        assert!(flat[1].is_infinite() && flat[1] < 0.0);
        // [1][0] = 0.0 — attend to past
        assert_eq!(flat[3], 0.0);
        // [1][1] = 0.0 — attend to self
        assert_eq!(flat[4], 0.0);
        // [1][2] = -inf — future blocked
        assert!(flat[5].is_infinite() && flat[5] < 0.0);
    }

    #[test]
    fn test_forward_with_kv_cache() {
        let config = make_config();
        let mut model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();

        // first forward — no cache
        let mut cache: Vec<(CandleTensor, CandleTensor)> = Vec::new();
        let _ = model
            .forward(&[1u32, 2, 3, 4], Some(&mut cache), 0)
            .unwrap();

        // cache should have one entry per layer
        assert_eq!(cache.len(), config.num_hidden_layers);

        // second forward — single new token with cache
        let logits2 = model.forward(&[5u32], Some(&mut cache), 4).unwrap();
        assert_eq!(logits2.shape().dim(0).unwrap(), 1);
    }

    #[test]
    fn test_forward_kv_cache_grows() {
        let config = make_config();
        let mut model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let mut cache: Vec<(CandleTensor, CandleTensor)> = Vec::new();

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

    #[test]
    fn test_config_stored() {
        let config = make_config();
        let model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        assert_eq!(model.config.hidden_size, 64);
        assert_eq!(model.config.vocab_size, 256);
    }
}
