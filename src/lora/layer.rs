use crate::core::backend::{Backend, CandleBackend, CandleTensor};
use crate::core::error::Result;
use crate::core::shape::Shape;
use crate::core::tensor::TensorOps;
use crate::model::linear::Linear;

/*
The LoraLinear<B> layer — a drop-in replacement for Linear<B> that adds the LoRA residual path.

y = x @ W.T                     // base linear (frozen)
  + (x @ A.T) @ B.T * scaling   // lora residual
Where:

W is [out, in] — frozen base weight
A is [rank, in] — down projection, initialized with Kaiming uniform
B is [out, rank] — up projection, initialized with zeros
 */

pub struct LoraLinear<B: Backend> {
    // frozen base weight
    base: Linear<B>,
    lora_a: Option<B::Tensor>,
    lora_b: Option<B::Tensor>,
    scaling: f32,
    in_features: usize,
    out_features: usize,
    rank: usize,
}

impl<B: Backend> LoraLinear<B> {
    // create from existing Linear with no adapter loaded
    pub fn from_linear(base: Linear<B>, rank: usize, scaling: f32) -> Self {
        let in_features = base.in_features();
        let out_features = base.out_features();
        Self {
            base: base,
            lora_a: None,
            lora_b: None,
            scaling,
            in_features: in_features,
            out_features: out_features,
            rank,
        }
    }

    // load adapter weights
    pub fn load_adapter(&mut self, a: B::Tensor, b: B::Tensor) -> Result<()> {
        self.lora_a = Some(a);
        self.lora_b = Some(b);
        Ok(())
    }

    pub fn has_adapter(&self) -> bool {
        self.lora_a.is_some() && self.lora_b.is_some()
    }

    pub fn unload_adapter(&mut self) {
        self.lora_a = None;
        self.lora_b = None;
    }

    pub fn forward(&self, x: &B::Tensor) -> Result<B::Tensor> {
        let base_out = self.base.forward(x)?;

        match (&self.lora_a, &self.lora_b) {
            (Some(a), Some(b)) => {
                let a_t = a.transpose(0, 1)?;
                let lora_mid = x.broadcast_matmul(&a_t)?;
                let b_t = b.transpose(0, 1)?;
                let lora_out = lora_mid.broadcast_matmul(&b_t)?;
                let scaled = lora_out.scale(self.scaling as f64)?;
                base_out.add(&scaled)
            }
            _ => Ok(base_out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::CandleBackend;
    use crate::core::device::Device;
    use crate::core::dtype::DType;
    use crate::core::shape::Shape;
    use crate::core::tensor::TensorOps;
    use crate::model::linear::Linear;

    fn cpu() -> Device {
        Device::Cpu
    }

    fn make_linear(in_f: usize, out_f: usize) -> Linear<CandleBackend> {
        use crate::core::backend::CandleTensor;
        Linear::new(
            CandleTensor::zeros(&Shape::new(&[out_f, in_f]), DType::F32, &cpu()).unwrap(),
            None,
        )
    }

    #[test]
    fn test_from_linear_no_adapter() {
        let linear = make_linear(32, 64);
        let lora = LoraLinear::<CandleBackend>::from_linear(linear, 8, 0.5);
        assert!(!lora.has_adapter());
        assert_eq!(lora.rank, 8);
        assert_eq!(lora.in_features, 32);
        assert_eq!(lora.out_features, 64);
    }

    #[test]
    fn test_load_adapter() {
        use crate::core::backend::CandleTensor;
        let linear = make_linear(32, 64);
        let mut lora = LoraLinear::<CandleBackend>::from_linear(linear, 8, 1.0);
        let a = CandleTensor::zeros(&Shape::new(&[8, 32]), DType::F32, &cpu()).unwrap();
        let b = CandleTensor::zeros(&Shape::new(&[64, 8]), DType::F32, &cpu()).unwrap();
        lora.load_adapter(a, b).unwrap();
        assert!(lora.has_adapter());
    }

    #[test]
    fn test_unload_adapter() {
        use crate::core::backend::CandleTensor;
        let linear = make_linear(32, 64);
        let mut lora = LoraLinear::<CandleBackend>::from_linear(linear, 8, 1.0);
        let a = CandleTensor::zeros(&Shape::new(&[8, 32]), DType::F32, &cpu()).unwrap();
        let b = CandleTensor::zeros(&Shape::new(&[64, 8]), DType::F32, &cpu()).unwrap();
        lora.load_adapter(a, b).unwrap();
        assert!(lora.has_adapter());
        lora.unload_adapter();
        assert!(!lora.has_adapter());
    }

    #[test]
    fn test_forward_no_adapter_shape() {
        use crate::core::backend::CandleTensor;
        let linear = make_linear(32, 64);
        let lora = LoraLinear::<CandleBackend>::from_linear(linear, 8, 1.0);
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 32]), DType::F32, &cpu()).unwrap();
        let out = lora.forward(&x).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 4, 64]));
    }

    #[test]
    fn test_forward_with_adapter_shape() {
        use crate::core::backend::CandleTensor;
        let linear = make_linear(32, 64);
        let mut lora = LoraLinear::<CandleBackend>::from_linear(linear, 8, 1.0);
        let a = CandleTensor::zeros(&Shape::new(&[8, 32]), DType::F32, &cpu()).unwrap();
        let b = CandleTensor::zeros(&Shape::new(&[64, 8]), DType::F32, &cpu()).unwrap();
        lora.load_adapter(a, b).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 32]), DType::F32, &cpu()).unwrap();
        let out = lora.forward(&x).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 4, 64]));
    }

    #[test]
    fn test_forward_with_adapter_zero_weights_matches_base() {
        use crate::core::backend::CandleTensor;
        // With zero lora weights, output should equal base output
        let linear = make_linear(32, 64);
        let mut lora = LoraLinear::<CandleBackend>::from_linear(linear, 8, 1.0);
        let a = CandleTensor::zeros(&Shape::new(&[8, 32]), DType::F32, &cpu()).unwrap();
        let b = CandleTensor::zeros(&Shape::new(&[64, 8]), DType::F32, &cpu()).unwrap();
        lora.load_adapter(a, b).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 2, 32]), DType::F32, &cpu()).unwrap();
        let out = lora.forward(&x).unwrap();
        let flat = out.to_vec_f32().unwrap();
        // zero base + zero lora = all zeros
        assert!(flat.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_scaling_stored() {
        let linear = make_linear(32, 64);
        let lora = LoraLinear::<CandleBackend>::from_linear(linear, 8, 0.5);
        assert!((lora.scaling - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_forward_single_token() {
        use crate::core::backend::CandleTensor;
        let linear = make_linear(64, 64);
        let lora = LoraLinear::<CandleBackend>::from_linear(linear, 16, 1.0);
        let x = CandleTensor::zeros(&Shape::new(&[1, 1, 64]), DType::F32, &cpu()).unwrap();
        let out = lora.forward(&x).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 1, 64]));
    }
}
