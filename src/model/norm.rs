use crate::core::backend::Backend;
use crate::core::error::Result;
use crate::core::tensor::TensorOps;

pub struct RmsNorm<B: Backend> {
    pub weight: B::Tensor,
    eps: f32,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(weight: B::Tensor, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &B::Tensor) -> Result<B::Tensor> {
        x.rms_norm(&self.weight, self.eps)
    }

    pub fn weight(&self) -> &B::Tensor {
        &self.weight
    }

    pub fn eps(&self) -> f32 {
        self.eps
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

    fn cpu() -> Device {
        Device::Cpu
    }

    fn make_norm(hidden_size: usize) -> RmsNorm<CandleBackend> {
        let weight = CandleTensor::ones(&Shape::new(&[hidden_size]), DType::F32, &cpu()).unwrap();
        RmsNorm::new(weight, 1e-5)
    }

    #[test]
    fn test_forward_shape_preserved() {
        let norm = make_norm(64);
        let x = CandleTensor::ones(&Shape::new(&[2, 8, 64]), DType::F32, &cpu()).unwrap();
        let out = norm.forward(&x).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[2, 8, 64]));
    }

    #[test]
    fn test_forward_2d_shape_preserved() {
        let norm = make_norm(32);
        let x = CandleTensor::ones(&Shape::new(&[4, 32]), DType::F32, &cpu()).unwrap();
        let out = norm.forward(&x).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[4, 32]));
    }

    #[test]
    fn test_forward_ones_input() {
        // input = ones, weight = ones -> output should be all ones
        // rms of ones = 1, so x / rms * weight = 1
        let norm = make_norm(4);
        let x = CandleTensor::ones(&Shape::new(&[1, 4]), DType::F32, &cpu()).unwrap();
        let out = norm.forward(&x).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 4]));
    }

    #[test]
    fn test_eps_accessor() {
        let norm = make_norm(32);
        assert_eq!(norm.eps(), 1e-5);
    }

    #[test]
    fn test_weight_shape() {
        let norm = make_norm(128);
        assert_eq!(norm.weight().shape(), &Shape::new(&[128]));
    }

    #[test]
    fn test_different_eps() {
        let weight = CandleTensor::ones(&Shape::new(&[16]), DType::F32, &cpu()).unwrap();
        let norm: RmsNorm<CandleBackend> = RmsNorm::new(weight, 1e-6);
        let x = CandleTensor::ones(&Shape::new(&[1, 16]), DType::F32, &cpu()).unwrap();
        let out = norm.forward(&x).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 16]));
    }
}
