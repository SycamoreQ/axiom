use crate::core::backend::{Backend, CandleTensor};
use crate::core::device::Device;
use crate::core::dtype::DType;
use crate::core::error::{CoreError, Result};
use crate::core::shape::Shape;
use crate::core::tensor::TensorOps;
use crate::core::tensor::{TopKLastDimOp, TopKOutput};
use crate::model::attention::Attention;
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

    fn causal_mask(&self, seq_len: usize, device: &Device) -> Result<B::Tensor> {
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
        let mut x = self.embedding.forward(token_ids)?;
        let mut x = x.unsqueeze(0)?;
        let device = x.device().clone();
        let mask = self.causal_mask(seq_len, &device)?;

        for (i, block) in self.blocks.iter_mut().enumerate() {
            let cache = kv_cache
                .as_deref()
                .and_then(|v| v.get(i).map(|(k, v)| (k, v)));
            let (block_out, new_k, new_v) = block.forward(&x, Some(&mask), cache, offset)?;
            x = block_out;
            if let Some(ref mut cache) = kv_cache {
                if i < cache.len() {
                    cache[i] = (new_k, new_v);
                } else {
                    cache.push((new_k, new_v));
                }
            }
        }

        let x = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;
        Ok(logits)
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
