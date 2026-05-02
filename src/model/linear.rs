use crate::core::backend::Backend;
use crate::core::backend::CandleBackend;
use crate::core::error::Result;
use crate::core::tensor::TensorOps;

pub struct Linear<B: Backend> {
    weight: B::Tensor,
    bias: Option<B::Tensor>,
}

impl<B: Backend> Linear<B> {
    pub fn new(weight: B::Tensor, bias: Option<B::Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &B::Tensor) -> Result<B::Tensor> {
        let wt = self.weight.transpose(0, 1)?;
        let out = x.broadcast_matmul(&wt)?;
        match &self.bias {
            Some(bias) => out.broadcast_add(bias),
            None => Ok(out),
        }
    }

    pub fn weight(&self) -> &B::Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&B::Tensor> {
        self.bias.as_ref()
    }

    pub fn in_features(&self) -> usize {
        self.weight.shape().dim(1).unwrap_or(0)
    }

    pub fn out_features(&self) -> usize {
        self.weight.shape().dim(0).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::CandleTensor;
    use crate::core::device::Device;
    use crate::core::dtype::DType;
    use crate::core::shape::Shape;
    use crate::core::tensor::TensorOps;

    fn cpu() -> Device {
        Device::Cpu
    }

    fn make_linear(in_f: usize, out_f: usize, bias: bool) -> Linear<CandleBackend> {
        let weight = CandleTensor::zeros(&Shape::new(&[out_f, in_f]), DType::F32, &cpu()).unwrap();
        let b = if bias {
            Some(CandleTensor::zeros(&Shape::new(&[out_f]), DType::F32, &cpu()).unwrap())
        } else {
            None
        };
        Linear::new(weight, b)
    }

    #[test]
    fn test_in_out_features() {
        let l = make_linear(64, 128, false);
        assert_eq!(l.in_features(), 64);
        assert_eq!(l.out_features(), 128);
    }

    #[test]
    fn test_forward_no_bias_shape() {
        let l = make_linear(64, 128, false);
        let x = CandleTensor::zeros(&Shape::new(&[2, 64]), DType::F32, &cpu()).unwrap();
        let out = l.forward(&x).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[2, 128]));
    }

    #[test]
    fn test_forward_with_bias_shape() {
        let l = make_linear(64, 128, true);
        let x = CandleTensor::zeros(&Shape::new(&[2, 64]), DType::F32, &cpu()).unwrap();
        let out = l.forward(&x).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[2, 128]));
    }

    #[test]
    fn test_forward_3d_input() {
        let l = make_linear(64, 128, false);
        let x = CandleTensor::zeros(&Shape::new(&[2, 8, 64]), DType::F32, &cpu()).unwrap();
        let out = l.forward(&x).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[2, 8, 128]));
    }

    #[test]
    fn test_forward_identity_values() {
        let weight_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let weight = CandleTensor::from_slice(&weight_data, &Shape::new(&[2, 2]), &cpu()).unwrap();
        let l: Linear<CandleBackend> = Linear::new(weight, None);
        let x = CandleTensor::from_slice(&[1.0f32, 1.0], &Shape::new(&[1, 2]), &cpu()).unwrap();
        let out = l.forward(&x).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 2]));
    }

    #[test]
    fn test_weight_accessor() {
        let l = make_linear(32, 64, false);
        assert_eq!(l.weight().shape(), &Shape::new(&[64, 32]));
    }

    #[test]
    fn test_bias_accessor_none() {
        let l = make_linear(32, 64, false);
        assert!(l.bias().is_none());
    }

    #[test]
    fn test_bias_accessor_some() {
        let l = make_linear(32, 64, true);
        assert!(l.bias().is_some());
        assert_eq!(l.bias().unwrap().shape(), &Shape::new(&[64]));
    }

    #[test]
    fn test_forward_wrong_input_shape_fails() {
        let l = make_linear(64, 128, false);
        let x = CandleTensor::zeros(&Shape::new(&[2, 32]), DType::F32, &cpu()).unwrap();
        assert!(l.forward(&x).is_err());
    }

    #[test]
    fn test_forward_bias_values() {
        // weight = identity, bias = ones -> output = input + 1
        let weight_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let bias_data: Vec<f32> = vec![1.0, 1.0];
        let weight = CandleTensor::from_slice(&weight_data, &Shape::new(&[2, 2]), &cpu()).unwrap();
        let bias = CandleTensor::from_slice(&bias_data, &Shape::new(&[2]), &cpu()).unwrap();
        let l: Linear<CandleBackend> = Linear::new(weight, Some(bias));
        let x = CandleTensor::from_slice(&[1.0f32, 1.0], &Shape::new(&[1, 2]), &cpu()).unwrap();
        let out = l.forward(&x).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 2]));
    }
}
