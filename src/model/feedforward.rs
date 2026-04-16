use candle_core::Tensor;

use crate::core::backend::Backend;
use crate::core::device::Device;
use crate::core::dtype::DType;
use crate::core::error::CoreError;
use crate::core::error::Result;
use crate::core::shape::Shape;
use crate::core::tensor::TensorOps;
use crate::model::config::ModelConfig;
use crate::model::linear::Linear;
use crate::model::rope::RotaryEmbedding;

pub struct FeedForward<B: Backend> {
    gate_proj: Linear<B>, // [intermediate_size, hidden_size]
    up_proj: Linear<B>,   // [intermediate_size, hidden_size]
    down_proj: Linear<B>, // [hidden_size, intermediate_size]
}

impl<B: Backend> FeedForward<B> {
    pub fn new(config: &ModelConfig, device: &Device) -> Result<Self> {
        let make_linear = |out: usize, inp: usize| -> Result<Linear<B>> {
            let w = B::Tensor::zeros(&Shape::new(&[out, inp]), DType::F32, device)?;
            Ok(Linear::new(w, None))
        };

        let gate_proj = make_linear(config.intermediate_size, config.hidden_size)?;
        let up_proj = make_linear(config.intermediate_size, config.hidden_size)?;
        let down_proj = make_linear(config.hidden_size, config.intermediate_size)?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, x: &B::Tensor) -> Result<B::Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        let fused = gate.mul(&up)?;
        self.down_proj.forward(&fused)
    }

    pub fn set_gate(&mut self, l: Linear<B>) {
        self.gate_proj = l;
    }
    pub fn set_up(&mut self, l: Linear<B>) {
        self.up_proj = l;
    }
    pub fn set_down(&mut self, l: Linear<B>) {
        self.down_proj = l;
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

    #[test]
    fn test_forward_shape_2d() {
        let ff = FeedForward::<CandleBackend>::new(&make_config(), &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[4, 64]), DType::F32, &cpu()).unwrap();
        let out = ff.forward(&x).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[4, 64]));
    }

    #[test]
    fn test_forward_shape_3d() {
        let ff = FeedForward::<CandleBackend>::new(&make_config(), &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[2, 8, 64]), DType::F32, &cpu()).unwrap();
        let out = ff.forward(&x).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[2, 8, 64]));
    }

    #[test]
    fn test_forward_single_token() {
        let ff = FeedForward::<CandleBackend>::new(&make_config(), &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 1, 64]), DType::F32, &cpu()).unwrap();
        let out = ff.forward(&x).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 1, 64]));
    }

    #[test]
    fn test_output_hidden_size_matches_input() {
        let ff = FeedForward::<CandleBackend>::new(&make_config(), &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 16, 64]), DType::F32, &cpu()).unwrap();
        let out = ff.forward(&x).unwrap();
        assert_eq!(out.shape().dim(2).unwrap(), 64);
    }

    #[test]
    fn test_proj_shapes() {
        let config = make_config();
        let ff = FeedForward::<CandleBackend>::new(&config, &cpu()).unwrap();
        assert_eq!(ff.gate_proj.in_features(), config.hidden_size);
        assert_eq!(ff.gate_proj.out_features(), config.intermediate_size);
        assert_eq!(ff.up_proj.in_features(), config.hidden_size);
        assert_eq!(ff.up_proj.out_features(), config.intermediate_size);
        assert_eq!(ff.down_proj.in_features(), config.intermediate_size);
        assert_eq!(ff.down_proj.out_features(), config.hidden_size);
    }

    #[test]
    fn test_different_batch_sizes() {
        let ff = FeedForward::<CandleBackend>::new(&make_config(), &cpu()).unwrap();
        for batch in [1, 2, 4, 8] {
            let x = CandleTensor::zeros(&Shape::new(&[batch, 4, 64]), DType::F32, &cpu()).unwrap();
            let out = ff.forward(&x).unwrap();
            assert_eq!(out.shape(), &Shape::new(&[batch, 4, 64]));
        }
    }
}
