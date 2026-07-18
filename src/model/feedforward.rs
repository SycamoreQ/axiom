use crate::core::backend::Backend;
#[cfg(feature = "metal")]
use crate::core::backend::MetalTensor;
use crate::core::device::Device;
use crate::core::dtype::DType;
use crate::core::error::Result;
use crate::core::shape::Shape;
use crate::core::tensor::TensorOps;
use crate::model::config::ModelConfig;
use crate::model::linear::Linear;

pub struct FeedForward<B: Backend> {
    pub gate_proj: Linear<B>, // [intermediate_size, hidden_size]
    pub up_proj: Linear<B>,   // [intermediate_size, hidden_size]
    pub down_proj: Linear<B>, // [hidden_size, intermediate_size]
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
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;

        // CPU SwiGLU – remove the Metal kernel usage
        let gate_vec = gate.to_vec_f32()?;
        let up_vec = up.to_vec_f32()?;
        let mut fused = vec![0.0f32; gate_vec.len()];
        for i in 0..gate_vec.len() {
            let silu = gate_vec[i] / (1.0 + (-gate_vec[i]).exp());
            fused[i] = silu * up_vec[i];
        }
        let fused_tensor = B::Tensor::from_slice(&fused, &gate.shape(), &x.device())?;
        self.down_proj.forward(&fused_tensor)
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

    pub fn gate_proj(&self) -> &Linear<B> {
        &self.gate_proj
    }
    pub fn up_proj(&self) -> &Linear<B> {
        &self.up_proj
    }
    pub fn down_proj(&self) -> &Linear<B> {
        &self.down_proj
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::{CandleBackend, CandleTensor, MetalBackend, MetalTensor};
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
            rope_freqs: None,
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

#[cfg(feature = "metal")]
#[cfg(test)]
mod metal_tests {
    use super::*;
    use crate::core::device::Device;
    use crate::core::dtype::DType;

    use crate::core::shape::Shape;
    use std::sync::Arc;

    // Helper to safely initialize and retrieve the global Metal state for tests.
    fn ensure_metal_device() -> Device {
        if crate::metal::state::global_metal_state().is_none() {
            let pool_size = 1024 * 1024 * 100; // 100 MB
            let _ = crate::metal::state::init_global_metal_state(pool_size);
        }
        Device::Metal(0)
    }

    // Helper to quickly create an F32 MetalTensor
    fn make_metal_tensor(data: &[f32], shape: &[usize]) -> MetalTensor {
        let device = ensure_metal_device();
        MetalTensor::from_slice(data, &Shape::new(shape), &device)
            .expect("Failed to create MetalTensor from slice")
    }

    #[test]
    fn test_metal_zeros_and_ones() {
        let device = ensure_metal_device();
        let shape = Shape::new(&[2, 3]);

        let z = MetalTensor::zeros(&shape, DType::F32, &device).unwrap();
        let z_vec = z.to_vec_f32().unwrap();
        assert_eq!(z_vec, vec![0.0; 6]);

        let o = MetalTensor::ones(&shape, DType::F32, &device).unwrap();
        let o_vec = o.to_vec_f32().unwrap();
        assert_eq!(o_vec, vec![1.0; 6]);
    }

    #[test]
    fn test_metal_arithmetic() {
        let a = make_metal_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = make_metal_tensor(&[2.0, 2.0, 2.0, 2.0], &[2, 2]);
        let add = a.add(&b).unwrap().to_vec_f32().unwrap();
        assert_eq!(add, vec![3.0, 4.0, 5.0, 6.0]);

        let sub = a.sub(&b).unwrap().to_vec_f32().unwrap();
        assert_eq!(sub, vec![-1.0, 0.0, 1.0, 2.0]);

        let mul = a.mul(&b).unwrap().to_vec_f32().unwrap();
        assert_eq!(mul, vec![2.0, 4.0, 6.0, 8.0]);

        let div = a.div(&b).unwrap().to_vec_f32().unwrap();
        assert_eq!(div, vec![0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_metal_transpose_and_contiguous() {
        let a = make_metal_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

        let transposed = a.transpose(0, 1).unwrap();
        assert_eq!(transposed.shape(), &Shape::new(&[3, 2]));

        let contig = transposed.contiguous().unwrap();
        let vec = contig.to_vec_f32().unwrap();
        assert_eq!(vec, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_metal_matmul() {
        let a = make_metal_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = make_metal_tensor(&[7.0, 8.0, 9.0, 1.0, 2.0, 3.0], &[3, 2]);

        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &Shape::new(&[2, 2]));

        let c_vec = c.to_vec_f32().unwrap();
        assert_eq!(c_vec, vec![31.0, 19.0, 85.0, 55.0]);
    }

    #[test]
    fn test_metal_broadcast_add() {
        let a = make_metal_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]); // Matrix
        let b = make_metal_tensor(&[10.0, 20.0], &[2]); // Bias

        let c = a.broadcast_add(&b).unwrap();
        let c_vec = c.to_vec_f32().unwrap();
        assert_eq!(c_vec, vec![11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn test_metal_reductions() {
        let a = make_metal_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

        // Sum along columns (dim 1)
        let sum = a.sum(1).unwrap();
        assert_eq!(sum.shape(), &Shape::new(&[2]));
        assert_eq!(sum.to_vec_f32().unwrap(), vec![6.0, 15.0]); // 1+2+3, 4+5+6

        // Mean along rows (dim 0)
        let mean = a.mean(0).unwrap();
        assert_eq!(mean.shape(), &Shape::new(&[3]));
        assert_eq!(mean.to_vec_f32().unwrap(), vec![2.5, 3.5, 4.5]); // (1+4)/2, (2+5)/2, (3+6)/2
    }

    #[test]
    fn test_metal_narrow_and_cat() {
        let a = make_metal_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        // Narrow: take 1 element along dim 0, starting at index 1 (second row)
        let narrow = a.narrow(0, 1, 1).unwrap();
        assert_eq!(narrow.shape(), &Shape::new(&[1, 2]));
        assert_eq!(narrow.to_vec_f32().unwrap(), vec![3.0, 4.0]);

        // Cat: concatenate a and narrow along dim 0
        let tensors = vec![&a, &narrow];
        let cat = MetalTensor::cat(&tensors, 0).unwrap();
        assert_eq!(cat.shape(), &Shape::new(&[3, 2]));
        assert_eq!(
            cat.to_vec_f32().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 3.0, 4.0]
        );
    }

    #[test]
    fn test_metal_activations() {
        let a = make_metal_tensor(&[0.0, 1.0, -1.0], &[3]);

        // Sigmoid
        let sig = a.sigmoid().unwrap().to_vec_f32().unwrap();
        assert!((sig[0] - 0.5).abs() < 1e-5);
        assert!((sig[1] - 0.73105).abs() < 1e-4);

        // Exp
        let e = a.exp().unwrap().to_vec_f32().unwrap();
        assert!((e[0] - 1.0).abs() < 1e-5);
        assert!((e[1] - std::f32::consts::E).abs() < 1e-4);
    }
}
