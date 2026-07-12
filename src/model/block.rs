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
        let is_layer_0 = self.layer_idx == 0;
        let is_first_token = offset == 0;
        let should_dump = is_layer_0 && is_first_token;

        let dump = |label: &str, t: &B::Tensor| -> Result<()> {
            if !should_dump {
                return Ok(());
            }

            let hidden_size = *t.shape().dims().last().unwrap();
            let mut v = t.to_vec_f32()?;

            v.truncate(hidden_size);

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
            eprintln!(
                "{label} ROW0 first3={:?} last3={:?}",
                &v[0..3],
                &v[v.len() - 3..]
            );
            Ok(())
        };

        // 1. Raw embedding output incoming to the layer block
        dump("Layer 0 [1] - Raw Input", x)?;

        let h = self.attn_norm.forward(x)?;

        // 2. State metrics right after layer 0 input RmsNorm
        dump("Layer 0 [2] - After Attn Norm", &h)?;

        let (attn_out, new_k, new_v) = self.attn.forward(&h, mask, kv_cache, offset)?;

        // 3. Post-attention projection output
        dump("Layer 0 [3] - Attn Out", &attn_out)?;

        let x = x.add(&attn_out)?;

        // 4. State metrics after adding attention residual context
        dump("Layer 0 [4] - After Attn Residual", &x)?;

        let h = self.ffn_norm.forward(&x)?;

        // 5. State metrics post-FFN input norm normalization step
        dump("Layer 0 [5] - After FFN Norm", &h)?;

        let ffn_out = match &mut self.ffn {
            FeedForwardLayer::Dense(ff) => ff.forward(&h)?,
            FeedForwardLayer::Moe(moe) => moe.forward(&h, offset)?.hidden_states,
        };

        // 6. Dense SwiGLU FFN layer evaluation context
        dump("Layer 0 [6] - FFN Out", &ffn_out)?;

        let x = x.add(&ffn_out)?;

        // 7. Complete single layer block evaluation with both residual paths merged
        dump("Layer 0 [7] - After FFN Residual", &x)?;

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
        if let FeedForwardLayer::Dense(ff) = &mut self.ffn {
            ff.set_gate(Linear::new(w, None))
        }
    }
    pub fn set_ffn_up(&mut self, w: B::Tensor) {
        if let FeedForwardLayer::Dense(ff) = &mut self.ffn {
            ff.set_up(Linear::new(w, None))
        }
    }
    pub fn set_ffn_down(&mut self, w: B::Tensor) {
        if let FeedForwardLayer::Dense(ff) = &mut self.ffn {
            ff.set_down(Linear::new(w, None))
        }
    }
    pub fn attn_norm_weight(&self) -> &B::Tensor {
        &self.attn_norm.weight
    }
    pub fn ffn_norm_weight(&self) -> &B::Tensor {
        &self.ffn_norm.weight
    }
}

// Tests block preserved for pipeline verification validation...
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
        assert!(Block::<CandleBackend>::new(&config, 0, &cpu()).is_ok());
    }

    #[test]
    fn test_moe_block_construction() {
        let config = make_moe_config();
        assert!(Block::<CandleBackend>::new(&config, 0, &cpu()).is_ok());
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
}

#[cfg(feature = "metal")]
#[cfg(test)]
mod metal_tests {
    use super::*;
    use crate::core::backend::{MetalBackend, MetalTensor};
    use crate::core::device::Device;
    use crate::core::tensor::TensorOps;

    fn ensure_metal_device() -> Device {
        if crate::metal::state::global_metal_state().is_none() {
            let _ = crate::metal::state::init_global_metal_state(1024 * 1024 * 100);
        }
        Device::Metal(0)
    }

    fn make_dense_config() -> ModelConfig {
        super::tests::make_dense_config()
    }

    #[test]
    fn test_metal_dense_block_forward() {
        let device = ensure_metal_device();
        let config = make_dense_config();
        let mut block = Block::<MetalBackend>::new(&config, 0, &device).unwrap();
        let x = MetalTensor::zeros(&Shape::new(&[1, 4, 64]), DType::F32, &device).unwrap();
        let (out, _, _) = block.forward(&x, None, None, 0).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 4, 64]));
    }
}
