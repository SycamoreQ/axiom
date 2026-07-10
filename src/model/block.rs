use crate::core::backend::Backend;
use crate::core::device::Device;
use crate::core::dtype::DType;
use crate::core::error::Result;
use crate::core::shape::Shape;
use crate::core::tensor::TensorOps;
use crate::model::attention::Attention;
use crate::model::config::ModelConfig;
use crate::model::feedforward::FeedForward;
use crate::model::linear::Linear;
use crate::model::moe::MoeLayer;
use crate::model::norm::RmsNorm;

use std::sync::atomic::{AtomicBool, Ordering};
static BLOCK0_CHECKED: AtomicBool = AtomicBool::new(false); // for debug

pub enum FeedForwardLayer<B: Backend> {
    Dense(FeedForward<B>),
    Moe(MoeLayer<B>),
}

pub struct Block<B: Backend> {
    attn_norm: RmsNorm<B>,
    pub attn: Attention<B>,
    ffn_norm: RmsNorm<B>,
    ffn: FeedForwardLayer<B>,
    layer_idx: usize,
}

impl<B: Backend> Block<B> {
    pub fn new(config: &ModelConfig, layer_idx: usize, device: &Device) -> Result<Self> {
        let hidden_size = config.hidden_size;

        let attn_norm = RmsNorm::new(
            B::Tensor::ones(&Shape::new(&[hidden_size]), DType::F32, device)?,
            config.rms_norm_eps as f32,
        );

        let ffn_norm = RmsNorm::new(
            B::Tensor::ones(&Shape::new(&[hidden_size]), DType::F32, device)?,
            config.rms_norm_eps as f32,
        );

        let attn = Attention::new(config, device)?;

        let ffn = if config.is_moe_layer(layer_idx) {
            FeedForwardLayer::Moe(MoeLayer::new(config, None, device)?)
        } else {
            FeedForwardLayer::Dense(FeedForward::new(config, device)?)
        };

        Ok(Self {
            attn_norm,
            attn,
            ffn_norm,
            ffn,
            layer_idx,
        })
    }

    pub fn forward(
        &mut self,
        x: &B::Tensor,
        mask: Option<&B::Tensor>,
        kv_cache: Option<(&B::Tensor, &B::Tensor)>,
        offset: usize,
    ) -> Result<(B::Tensor, B::Tensor, B::Tensor)> {
        //attention with pre-norm and residual
        let h = self.attn_norm.forward(x)?;
        let (attn_out, new_k, new_v) = self.attn.forward(&h, mask, kv_cache, offset)?;

        let ao_vec = attn_out.to_vec_f32()?;
        let ao_max = ao_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let ao_min = ao_vec.iter().cloned().fold(f32::INFINITY, f32::min);

        let x = x.add(&attn_out)?;

        let after_vec = x.to_vec_f32()?;
        let after_max = after_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        //ffn with pre-norm and residual
        let h = self.ffn_norm.forward(&x)?;
        let ffn_out = match &mut self.ffn {
            FeedForwardLayer::Dense(ff) => ff.forward(&h)?,
            FeedForwardLayer::Moe(moe) => moe.forward(&h, offset)?.hidden_states,
        };
        let x = x.add(&ffn_out)?;

        Ok((x, new_k, new_v))
    }

    pub fn set_attn_norm(&mut self, w: B::Tensor) {
        let eps = self.attn_norm.eps();
        self.attn_norm = RmsNorm::new(w, eps);
    }
    pub fn set_ffn_norm(&mut self, w: B::Tensor) {
        let eps = self.ffn_norm.eps();
        self.ffn_norm = RmsNorm::new(w, eps);
    }
    pub fn set_attn_q(&mut self, w: B::Tensor) {
        self.attn.set_q_proj(Linear::new(w, None));
    }

    pub fn set_attn_k(&mut self, w: B::Tensor) {
        self.attn.set_k_proj(Linear::new(w, None));
    }

    pub fn set_attn_v(&mut self, w: B::Tensor) {
        self.attn.set_v_proj(Linear::new(w, None));
    }

    pub fn set_attn_o(&mut self, w: B::Tensor) {
        self.attn.set_o_proj(Linear::new(w, None));
    }
    pub fn set_ffn_gate(&mut self, w: B::Tensor) {
        match &mut self.ffn {
            FeedForwardLayer::Dense(ff) => ff.set_gate(Linear::new(w, None)),
            FeedForwardLayer::Moe(_) => {}
        }
    }
    pub fn set_ffn_up(&mut self, w: B::Tensor) {
        match &mut self.ffn {
            FeedForwardLayer::Dense(ff) => ff.set_up(Linear::new(w, None)),
            FeedForwardLayer::Moe(_) => {}
        }
    }
    pub fn set_ffn_down(&mut self, w: B::Tensor) {
        match &mut self.ffn {
            FeedForwardLayer::Dense(ff) => ff.set_down(Linear::new(w, None)),
            FeedForwardLayer::Moe(_) => {}
        }
    }
    pub fn attn_norm_weight(&self) -> &B::Tensor {
        &self.attn_norm.weight
    }
    pub fn ffn_norm_weight(&self) -> &B::Tensor {
        &self.ffn_norm.weight
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

    fn make_dense_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 128,
            vocab_size: 1000,
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
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 128,
            vocab_size: 1000,
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
    fn test_dense_block_construction() {
        let config = make_dense_config();
        let block = Block::<CandleBackend>::new(&config, 0, &cpu());
        assert!(block.is_ok());
    }

    #[test]
    fn test_moe_block_construction() {
        let config = make_moe_config();
        let block = Block::<CandleBackend>::new(&config, 0, &cpu());
        assert!(block.is_ok());
    }

    #[test]
    fn test_dense_block_is_dense() {
        let config = make_dense_config();
        let block = Block::<CandleBackend>::new(&config, 0, &cpu()).unwrap();
        assert!(matches!(block.ffn, FeedForwardLayer::Dense(_)));
    }

    #[test]
    fn test_moe_block_is_moe() {
        let config = make_moe_config();
        let block = Block::<CandleBackend>::new(&config, 0, &cpu()).unwrap();
        assert!(matches!(block.ffn, FeedForwardLayer::Moe(_)));
    }

    #[test]
    fn test_layer_idx_stored() {
        let config = make_dense_config();
        let block = Block::<CandleBackend>::new(&config, 3, &cpu()).unwrap();
        assert_eq!(block.layer_idx, 3);
    }

    #[test]
    fn test_dense_forward_output_shape() {
        let config = make_dense_config();
        let mut block = Block::<CandleBackend>::new(&config, 0, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 64]), DType::F32, &cpu()).unwrap();
        let (out, k, v) = block.forward(&x, None, None, 0).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 4, 64]));
        assert_eq!(k.shape().dim(0).unwrap(), 1);
        assert_eq!(v.shape().dim(0).unwrap(), 1);
    }

    #[test]
    fn test_moe_forward_output_shape() {
        let config = make_moe_config();
        let mut block = Block::<CandleBackend>::new(&config, 0, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 64]), DType::F32, &cpu()).unwrap();
        let (out, _, _) = block.forward(&x, None, None, 0).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 4, 64]));
    }

    #[test]
    fn test_forward_with_mask() {
        let config = make_dense_config();
        let mut block = Block::<CandleBackend>::new(&config, 0, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 64]), DType::F32, &cpu()).unwrap();
        let mask = CandleTensor::zeros(&Shape::new(&[1, 1, 4, 4]), DType::F32, &cpu()).unwrap();
        let (out, _, _) = block.forward(&x, Some(&mask), None, 0).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 4, 64]));
    }

    #[test]
    fn test_forward_with_kv_cache() {
        let config = make_dense_config();
        let mut block = Block::<CandleBackend>::new(&config, 0, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 64]), DType::F32, &cpu()).unwrap();
        let (_, k, v) = block.forward(&x, None, None, 0).unwrap();

        let x2 = CandleTensor::zeros(&Shape::new(&[1, 1, 64]), DType::F32, &cpu()).unwrap();
        let (out2, k2, _) = block.forward(&x2, None, Some((&k, &v)), 4).unwrap();
        assert_eq!(out2.shape(), &Shape::new(&[1, 1, 64]));
        assert_eq!(k2.shape().dim(1).unwrap(), 5);
    }

    #[test]
    fn test_forward_single_token() {
        let config = make_dense_config();
        let mut block = Block::<CandleBackend>::new(&config, 0, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 1, 64]), DType::F32, &cpu()).unwrap();
        let (out, _, _) = block.forward(&x, None, None, 0).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 1, 64]));
    }

    #[test]
    fn test_expert_interval_alternates() {
        // with expert_interval=2, even layers are MoE, odd are dense
        let mut config = make_moe_config();
        config.expert_interval = Some(2);

        let block_0 = Block::<CandleBackend>::new(&config, 0, &cpu()).unwrap();
        let block_1 = Block::<CandleBackend>::new(&config, 1, &cpu()).unwrap();
        let block_2 = Block::<CandleBackend>::new(&config, 2, &cpu()).unwrap();

        assert!(matches!(block_0.ffn, FeedForwardLayer::Moe(_)));
        assert!(matches!(block_1.ffn, FeedForwardLayer::Dense(_)));
        assert!(matches!(block_2.ffn, FeedForwardLayer::Moe(_)));
    }
}

#[cfg(feature = "metal")]
#[cfg(test)]
mod metal_tests {
    use super::*;
    use crate::core::backend::{MetalBackend, MetalTensor};
    use crate::core::device::Device;
    use crate::core::dtype::DType;
    use crate::core::shape::Shape;
    use crate::core::tensor::TensorOps;

    // Helper to safely initialize and retrieve the global Metal state for tests.
    fn ensure_metal_device() -> Device {
        if crate::metal::state::global_metal_state().is_none() {
            let pool_size = 1024 * 1024 * 100; // 100 MB
            let _ = crate::metal::state::init_global_metal_state(pool_size);
        }
        Device::Metal(0)
    }

    fn make_dense_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 128,
            vocab_size: 1000,
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
        let mut config = make_dense_config();
        config.num_local_experts = Some(4);
        config.num_experts_per_tok = Some(2);
        config.num_shared_experts = Some(1);
        config.expert_interval = Some(1);
        config.model_type = Some("deepseek".to_string());
        config
    }

    #[test]
    fn test_metal_dense_block_forward() {
        let device = ensure_metal_device();
        let config = make_dense_config();
        let mut block = Block::<MetalBackend>::new(&config, 0, &device)
            .expect("Failed to create Metal Dense Block");

        let x = MetalTensor::zeros(&Shape::new(&[1, 4, 64]), DType::F32, &device).unwrap();

        let (out, k, v) = block
            .forward(&x, None, None, 0)
            .expect("Metal block forward failed");

        assert_eq!(out.shape(), &Shape::new(&[1, 4, 64]));
        assert_eq!(k.shape().dim(0).unwrap(), 1);
        assert_eq!(v.shape().dim(0).unwrap(), 1);
    }

    #[test]
    fn test_metal_moe_block_forward() {
        let device = ensure_metal_device();
        let config = make_moe_config();
        let mut block = Block::<MetalBackend>::new(&config, 0, &device)
            .expect("Failed to create Metal MoE Block");

        let x = MetalTensor::zeros(&Shape::new(&[1, 4, 64]), DType::F32, &device).unwrap();

        let (out, _, _) = block
            .forward(&x, None, None, 0)
            .expect("Metal MoE forward failed");
        assert_eq!(out.shape(), &Shape::new(&[1, 4, 64]));
    }

    #[test]
    fn test_metal_block_with_mask() {
        let device = ensure_metal_device();
        let config = make_dense_config();
        let mut block = Block::<MetalBackend>::new(&config, 0, &device).unwrap();

        let x = MetalTensor::zeros(&Shape::new(&[1, 4, 64]), DType::F32, &device).unwrap();
        let mask = MetalTensor::zeros(&Shape::new(&[1, 1, 4, 4]), DType::F32, &device).unwrap();

        let (out, _, _) = block.forward(&x, Some(&mask), None, 0).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 4, 64]));
    }

    #[test]
    fn test_metal_block_kv_cache() {
        let device = ensure_metal_device();
        let config = make_dense_config();
        let mut block = Block::<MetalBackend>::new(&config, 0, &device).unwrap();

        // Initial prefill phase (4 tokens)
        let x_prefill = MetalTensor::zeros(&Shape::new(&[1, 4, 64]), DType::F32, &device).unwrap();
        let (_, k_cache, v_cache) = block.forward(&x_prefill, None, None, 0).unwrap();

        // Decode phase (1 token), using the cache from the prefill phase
        let x_decode = MetalTensor::zeros(&Shape::new(&[1, 1, 64]), DType::F32, &device).unwrap();
        let (out_decode, new_k, new_v) = block
            .forward(&x_decode, None, Some((&k_cache, &v_cache)), 4)
            .expect("KV Cache forward pass failed on Metal");

        assert_eq!(out_decode.shape(), &Shape::new(&[1, 1, 64]));

        // The KV cache should now hold 5 tokens (4 from prefill + 1 from decode)
        assert_eq!(
            new_k.shape().dim(1).unwrap(),
            5,
            "Metal KV Cache did not concatenate correctly"
        );
        assert_eq!(
            new_v.shape().dim(1).unwrap(),
            5,
            "Metal KV Cache did not concatenate correctly"
        );
    }
}
