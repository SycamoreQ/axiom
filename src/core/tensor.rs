use crate::core::backend::{CandleTensor, CudarcTensor};
use crate::core::device::Device;
use crate::core::dtype::{DType, Element};
use crate::core::error::{CoreError, Result};
use crate::core::shape::Shape;
use candle_core::{self, Tensor};
use candle_nn;

fn candle_device_from(device: &Device) -> Result<candle_core::Device> {
    match device {
        Device::Cpu => Ok(candle_core::Device::Cpu),
        Device::Cuda(n) => Ok(candle_core::Device::new_cuda(*n)?),
    }
}

pub trait TensorOps: Clone + Send + Sync + Sized {
    fn shape(&self) -> &Shape;
    fn dtype(&self) -> DType;
    fn device(&self) -> &Device;
    fn rank(&self) -> usize {
        self.shape().rank()
    }
    fn numel(&self) -> usize {
        self.shape().numel()
    }

    // creation
    fn zeros(shape: &Shape, dtype: DType, device: &Device) -> crate::core::error::Result<Self>;
    fn from_u32_slice(data: &[u32], shape: &Shape, device: &Device) -> Result<Self>;
    fn ones(shape: &Shape, dtype: DType, device: &Device) -> crate::core::error::Result<Self>;
    fn from_slice<E: Element + candle_core::WithDType>(
        data: &[E],
        shape: &Shape,
        device: &Device,
    ) -> crate::core::error::Result<Self>;

    // movement
    fn to_device(&self, device: &Device) -> crate::core::error::Result<Self>;
    fn to_dtype(&self, dtype: DType) -> crate::core::error::Result<Self>;
    fn contiguous(&self) -> crate::core::error::Result<Self>;

    // shape ops
    fn reshape(&self, shape: &Shape) -> crate::core::error::Result<Self>;
    fn transpose(&self, dim1: usize, dim2: usize) -> crate::core::error::Result<Self>;
    fn squeeze(&self, dim: usize) -> crate::core::error::Result<Self>;
    fn unsqueeze(&self, dim: usize) -> crate::core::error::Result<Self>;

    // arithmetic
    fn add(&self, other: &Self) -> crate::core::error::Result<Self>;
    fn sub(&self, other: &Self) -> crate::core::error::Result<Self>;
    fn mul(&self, other: &Self) -> crate::core::error::Result<Self>;
    fn div(&self, other: &Self) -> crate::core::error::Result<Self>;
    fn scale(&self, scalar: f64) -> crate::core::error::Result<Self>;

    // linear algebra
    fn matmul(&self, other: &Self) -> crate::core::error::Result<Self>;

    // reductions
    fn sum(&self, dim: usize) -> crate::core::error::Result<Self>;
    fn mean(&self, dim: usize) -> crate::core::error::Result<Self>;

    // activations
    fn silu(&self) -> crate::core::error::Result<Self>;
    fn gelu(&self) -> crate::core::error::Result<Self>;
    fn softmax(&self, dim: usize) -> crate::core::error::Result<Self>;
    fn sqrt(&self) -> crate::core::error::Result<Self>;

    // model ops — backends can fuse these
    fn rms_norm(&self, weight: &Self, eps: f32) -> crate::core::error::Result<Self>;
    fn broadcast_add(&self, other: &Self) -> Result<Self>;
    fn broadcast_matmul(&self, other: &Self) -> Result<Self>;
    fn index_select(&self, indexes: &Self, dim: usize) -> Result<Self>;
}

impl TensorOps for CandleTensor {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn zeros(shape: &Shape, dtype: DType, device: &Device) -> Result<Self> {
        let candle_shape = candle_core::Shape::from_dims(shape.dims());
        let candle_dtype: candle_core::DType = dtype.into();
        let candle_device = candle_device_from(device)?;
        let inner = candle_core::Tensor::zeros(candle_shape, candle_dtype, &candle_device)?;
        Ok(CandleTensor {
            shape: shape.clone(),
            dtype,
            device: device.clone(),
            inner,
        })
    }

    fn ones(shape: &Shape, dtype: DType, device: &Device) -> crate::core::error::Result<Self> {
        let candle_shape = candle_core::Shape::from_dims(shape.dims());
        let candle_dtype: candle_core::DType = dtype.into();
        let candle_device = candle_device_from(device)?;
        let inner = candle_core::Tensor::ones(candle_shape, candle_dtype, &candle_device)?;
        Ok(CandleTensor {
            shape: shape.clone(),
            dtype,
            device: device.clone(),
            inner,
        })
    }

    // TODO: remove candle_core::WithDType bound from trait — leaks backend detail
    // revisit in Phase 3 with a cleaner element conversion strategy
    fn from_slice<E: Element + candle_core::WithDType>(
        data: &[E],
        shape: &Shape,
        device: &Device,
    ) -> Result<Self> {
        let candle_shape = candle_core::Shape::from_dims(shape.dims());
        let candle_device = candle_device_from(device)?;
        let inner = candle_core::Tensor::from_slice(data, candle_shape, &candle_device)?;
        Ok(CandleTensor {
            shape: shape.clone(),
            dtype: E::dtype(),
            device: device.clone(),
            inner,
        })
    }

    fn from_u32_slice(data: &[u32], shape: &Shape, device: &Device) -> Result<Self> {
        let candle_device = candle_device_from(device)?;
        let inner = candle_core::Tensor::from_slice(data, shape.dims(), &candle_device)?;
        Ok(CandleTensor {
            shape: shape.clone(),
            dtype: DType::F32, // u32 stored but we mark F32 — fix in Phase 4
            device: device.clone(),
            inner,
        })
    }

    fn to_device(&self, device: &Device) -> Result<Self> {
        let candle_device = candle_device_from(device)?;
        let inner = self.inner.to_device(&candle_device)?;
        Ok(CandleTensor {
            inner,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: device.clone(),
        })
    }

    fn to_dtype(&self, dtype: DType) -> crate::core::error::Result<Self> {
        let candle_dtype: candle_core::DType = dtype.into();
        let inner = self.inner.to_dtype(candle_dtype)?;
        Ok(CandleTensor {
            inner,
            shape: self.shape.clone(),
            dtype: dtype.clone(),
            device: self.device.clone(),
        })
    }

    fn contiguous(&self) -> crate::core::error::Result<Self> {
        let inner = self.inner.contiguous()?;
        Ok(CandleTensor {
            inner: inner,
            shape: self.shape.clone(),
            dtype: self.dtype.clone(),
            device: self.device.clone(),
        })
    }

    fn reshape(&self, shape: &Shape) -> crate::core::error::Result<Self> {
        let inner = self.inner.reshape(shape.dims())?;

        Ok(CandleTensor {
            inner,
            shape: shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        let inner = self.inner.transpose(dim1, dim2)?;
        let mut new_dims = self.shape.dims().to_vec();
        new_dims.swap(dim1, dim2);
        Ok(CandleTensor {
            inner,
            shape: Shape::new(&new_dims),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    // squeeze removes a dimension
    fn squeeze(&self, dim: usize) -> Result<Self> {
        let inner = self.inner.squeeze(dim)?;
        let new_shape = Shape::new(&inner.dims());
        Ok(CandleTensor {
            inner,
            shape: new_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    // unsqueeze adds a dimension of size 1
    fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let inner = self.inner.unsqueeze(dim)?;
        let new_shape = Shape::new(&inner.dims());
        Ok(CandleTensor {
            inner,
            shape: new_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn add(&self, other: &Self) -> Result<Self> {
        Device::check_same("add", &self.device, &other.device)?;
        Shape::elementwise_check(&self.shape, &other.shape)?;
        let inner = self.inner.add(&other.inner)?;
        Ok(CandleTensor {
            inner,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn sub(&self, other: &Self) -> crate::core::error::Result<Self> {
        Device::check_same("add", &self.device, &other.device)?;
        Shape::elementwise_check(&self.shape, &other.shape)?;
        let inner = self.inner.sub(&other.inner)?;
        Ok(CandleTensor {
            inner,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn mul(&self, other: &Self) -> crate::core::error::Result<Self> {
        Device::check_same("mul", &self.device, &other.device)?;
        Shape::elementwise_check(&self.shape, &other.shape)?;
        let inner = self.inner.mul(&other.inner)?;
        Ok(CandleTensor {
            inner,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn div(&self, other: &Self) -> crate::core::error::Result<Self> {
        Device::check_same("div", &self.device, &other.device)?;
        Shape::elementwise_check(&self.shape, &other.shape)?;
        let inner = self.inner.div(&other.inner)?;
        Ok(CandleTensor {
            inner,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn scale(&self, scalar: f64) -> Result<Self> {
        let inner = self.inner.affine(scalar, 0.0)?;
        Ok(CandleTensor {
            inner,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn matmul(&self, other: &Self) -> Result<Self> {
        Device::check_same("matmul", &self.device, &other.device)?;
        let out_shape = Shape::matmul_check(&self.shape, &other.shape)?;
        let inner = self.inner.matmul(&other.inner)?;
        Ok(CandleTensor {
            inner,
            shape: out_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn sum(&self, dim: usize) -> Result<Self> {
        let inner = self.inner.sum_keepdim(dim)?;
        let new_shape = Shape::new(&inner.dims());
        Ok(CandleTensor {
            inner,
            shape: new_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn mean(&self, dim: usize) -> Result<Self> {
        let inner = self.inner.mean_keepdim(dim)?;
        let new_shape = Shape::new(&inner.dims());
        Ok(CandleTensor {
            inner,
            shape: new_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn silu(&self) -> crate::core::error::Result<Self> {
        let inner = self.inner.silu()?;
        Ok(CandleTensor {
            inner,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn gelu(&self) -> crate::core::error::Result<Self> {
        let inner = self.inner.gelu()?;
        Ok(CandleTensor {
            inner,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn softmax(&self, dim: usize) -> crate::core::error::Result<Self> {
        let inner = candle_nn::ops::softmax(&self.inner, dim)?;

        Ok(CandleTensor {
            inner,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn sqrt(&self) -> crate::core::error::Result<Self> {
        let inner = self.inner.sqrt()?;

        Ok(CandleTensor {
            inner,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn rms_norm(&self, weight: &Self, eps: f32) -> Result<Self> {
        let x_sq = self.inner.sqr()?;
        let mean_sq = x_sq.mean_keepdim(x_sq.rank() - 1)?;
        let mean_sq_eps = mean_sq.affine(1.0, eps as f64)?;
        // rsqrt = 1 / sqrt(x)
        let sqrt = mean_sq_eps.sqrt()?;
        let rsqrt = sqrt.recip()?;
        let normed = self.inner.broadcast_mul(&rsqrt)?;
        let inner = normed.broadcast_mul(&weight.inner)?;
        let new_shape = Shape::new(&inner.dims());
        Ok(CandleTensor {
            inner,
            shape: new_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn broadcast_add(&self, other: &Self) -> Result<Self> {
        let inner = self.inner.broadcast_add(&other.inner)?;
        Ok(CandleTensor {
            inner,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn broadcast_matmul(&self, other: &Self) -> Result<Self> {
        let inner = self.inner.broadcast_matmul(&other.inner)?;
        let new_shape = Shape::new(&inner.dims());
        Ok(CandleTensor {
            inner,
            shape: new_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn index_select(&self, indexes: &Self, dim: usize) -> Result<Self> {
        let inner = self.inner.index_select(&indexes.inner, dim)?;
        let new_shape = Shape::new(&inner.dims());
        Ok(CandleTensor {
            inner,
            shape: new_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
}

// todo component as Cudarc tensors yet to implemented in itslf for kernel logic
// will write after phase 4 or 5
impl TensorOps for CudarcTensor {
    fn shape(&self) -> &Shape {
        &self.shape
    }
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn device(&self) -> &Device {
        &self.device
    }

    fn zeros(_: &Shape, _: DType, _: &Device) -> Result<Self> {
        todo!("Phase 4")
    }
    fn ones(_: &Shape, _: DType, _: &Device) -> Result<Self> {
        todo!("Phase 4")
    }
    fn from_slice<E: Element + candle_core::WithDType>(
        _: &[E],
        _: &Shape,
        _: &Device,
    ) -> Result<Self> {
        todo!("Phase 4")
    }

    fn from_u32_slice(_: &[u32], _: &Shape, _: &Device) -> Result<Self> {
        todo!("Phase 4")
    }

    fn to_device(&self, _: &Device) -> Result<Self> {
        todo!("Phase 4")
    }
    fn to_dtype(&self, _: DType) -> Result<Self> {
        todo!("Phase 4")
    }
    fn contiguous(&self) -> Result<Self> {
        todo!("Phase 4")
    }
    fn reshape(&self, _: &Shape) -> Result<Self> {
        todo!("Phase 4")
    }
    fn transpose(&self, _: usize, _: usize) -> Result<Self> {
        todo!("Phase 4")
    }
    fn squeeze(&self, _: usize) -> Result<Self> {
        todo!("Phase 4")
    }
    fn unsqueeze(&self, _: usize) -> Result<Self> {
        todo!("Phase 4")
    }
    fn add(&self, _: &Self) -> Result<Self> {
        todo!("Phase 4")
    }
    fn sub(&self, _: &Self) -> Result<Self> {
        todo!("Phase 4")
    }
    fn mul(&self, _: &Self) -> Result<Self> {
        todo!("Phase 4")
    }
    fn div(&self, _: &Self) -> Result<Self> {
        todo!("Phase 4")
    }
    fn scale(&self, _: f64) -> Result<Self> {
        todo!("Phase 4")
    }
    fn matmul(&self, _: &Self) -> Result<Self> {
        todo!("Phase 4")
    }
    fn broadcast_add(&self, _: &Self) -> Result<Self> {
        todo!("Phase 4")
    }
    fn sum(&self, _: usize) -> Result<Self> {
        todo!("Phase 4")
    }
    fn mean(&self, _: usize) -> Result<Self> {
        todo!("Phase 4")
    }
    fn silu(&self) -> Result<Self> {
        todo!("Phase 4")
    }
    fn gelu(&self) -> Result<Self> {
        todo!("Phase 4")
    }
    fn softmax(&self, _: usize) -> Result<Self> {
        todo!("Phase 4")
    }
    fn sqrt(&self) -> Result<Self> {
        todo!("Phase 4")
    }
    fn rms_norm(&self, _: &Self, _: f32) -> Result<Self> {
        todo!("Phase 4")
    }

    fn broadcast_matmul(&self, _: &Self) -> Result<Self> {
        todo!("Phase 4")
    }
    fn index_select(&self, indexes: &Self, dim: usize) -> Result<Self> {
        todo!("phase 4")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::device::Device;
    use crate::core::dtype::DType;
    use crate::core::shape::Shape;

    fn cpu() -> Device {
        Device::Cpu
    }

    fn make_tensor(data: &[f32], shape: &[usize]) -> CandleTensor {
        CandleTensor::from_slice(data, &Shape::new(shape), &cpu()).unwrap()
    }

    #[test]
    fn test_zeros() {
        let t = CandleTensor::zeros(&Shape::new(&[2, 3]), DType::F32, &cpu()).unwrap();
        assert_eq!(t.shape(), &Shape::new(&[2, 3]));
        assert_eq!(t.dtype(), DType::F32);
        assert!(t.device().is_cpu());
    }

    #[test]
    fn test_ones() {
        let t = CandleTensor::ones(&Shape::new(&[2, 3]), DType::F32, &cpu()).unwrap();
        assert_eq!(t.shape(), &Shape::new(&[2, 3]));
    }

    #[test]
    fn test_from_slice() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let t = make_tensor(&data, &[2, 2]);
        assert_eq!(t.shape(), &Shape::new(&[2, 2]));
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.rank(), 2);
        assert_eq!(t.numel(), 4);
    }

    #[test]
    fn test_reshape() {
        let t = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let r = t.reshape(&Shape::new(&[4])).unwrap();
        assert_eq!(r.shape(), &Shape::new(&[4]));
    }

    #[test]
    fn test_transpose() {
        let t = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let r = t.transpose(0, 1).unwrap();
        assert_eq!(r.shape(), &Shape::new(&[3, 2]));
    }

    #[test]
    fn test_squeeze() {
        let t = make_tensor(&[1.0, 2.0, 3.0], &[1, 3]);
        let r = t.squeeze(0).unwrap();
        assert_eq!(r.shape(), &Shape::new(&[3]));
    }

    #[test]
    fn test_unsqueeze() {
        let t = make_tensor(&[1.0, 2.0, 3.0], &[3]);
        let r = t.unsqueeze(0).unwrap();
        assert_eq!(r.shape(), &Shape::new(&[1, 3]));
    }

    #[test]
    fn test_add() {
        let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = make_tensor(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape(), &Shape::new(&[2, 2]));
    }

    #[test]
    fn test_add_shape_mismatch() {
        let a = make_tensor(&[1.0, 2.0], &[2]);
        let b = make_tensor(&[1.0, 2.0, 3.0], &[3]);
        assert!(a.add(&b).is_err());
    }

    #[test]
    fn test_sub() {
        let a = make_tensor(&[2.0, 4.0], &[2]);
        let b = make_tensor(&[1.0, 1.0], &[2]);
        let c = a.sub(&b).unwrap();
        assert_eq!(c.shape(), &Shape::new(&[2]));
    }

    #[test]
    fn test_mul() {
        let a = make_tensor(&[2.0, 3.0], &[2]);
        let b = make_tensor(&[4.0, 5.0], &[2]);
        let c = a.mul(&b).unwrap();
        assert_eq!(c.shape(), &Shape::new(&[2]));
    }

    #[test]
    fn test_div() {
        let a = make_tensor(&[4.0, 6.0], &[2]);
        let b = make_tensor(&[2.0, 3.0], &[2]);
        let c = a.div(&b).unwrap();
        assert_eq!(c.shape(), &Shape::new(&[2]));
    }

    #[test]
    fn test_scale() {
        let t = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let r = t.scale(2.0).unwrap();
        assert_eq!(r.shape(), &Shape::new(&[4]));
    }

    #[test]
    fn test_matmul() {
        let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = make_tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &Shape::new(&[2, 2]));
    }

    #[test]
    fn test_matmul_shape_mismatch() {
        let a = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert!(a.matmul(&b).is_err());
    }

    #[test]
    fn test_sum() {
        let t = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let r = t.sum(0).unwrap();
        assert_eq!(r.shape().dims()[r.rank() - 1], 2);
    }

    #[test]
    fn test_mean() {
        let t = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let r = t.mean(0).unwrap();
        assert_eq!(r.shape().dims()[r.rank() - 1], 2);
    }

    #[test]
    fn test_silu() {
        let t = make_tensor(&[1.0, -1.0, 0.0, 2.0], &[4]);
        let r = t.silu().unwrap();
        assert_eq!(r.shape(), &Shape::new(&[4]));
    }

    #[test]
    fn test_gelu() {
        let t = make_tensor(&[1.0, -1.0, 0.0, 2.0], &[4]);
        let r = t.gelu().unwrap();
        assert_eq!(r.shape(), &Shape::new(&[4]));
    }

    #[test]
    fn test_softmax() {
        let t = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let r = t.softmax(1).unwrap();
        assert_eq!(r.shape(), &Shape::new(&[1, 4]));
    }

    #[test]
    fn test_sqrt() {
        let t = make_tensor(&[4.0, 9.0, 16.0, 25.0], &[4]);
        let r = t.sqrt().unwrap();
        assert_eq!(r.shape(), &Shape::new(&[4]));
    }

    #[test]
    fn test_to_dtype() {
        let t = make_tensor(&[1.0, 2.0, 3.0], &[3]);
        let r = t.to_dtype(DType::F16).unwrap();
        assert_eq!(r.dtype(), DType::F16);
        assert_eq!(r.shape(), &Shape::new(&[3]));
    }

    #[test]
    fn test_contiguous() {
        let t = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let r = t.contiguous().unwrap();
        assert_eq!(r.shape(), &Shape::new(&[2, 2]));
    }

    #[test]
    fn test_rms_norm() {
        let t = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let w = make_tensor(&[1.0, 1.0, 1.0, 1.0], &[1, 4]);
        let r = t.rms_norm(&w, 1e-5).unwrap();
        assert_eq!(r.shape(), &Shape::new(&[1, 4]));
    }

    #[test]
    fn test_rank_and_numel() {
        let t = make_tensor(&[1.0; 24], &[2, 3, 4]);
        assert_eq!(t.rank(), 3);
        assert_eq!(t.numel(), 24);
    }
}
