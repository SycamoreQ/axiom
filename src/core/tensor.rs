use crate::core::backend::{CandleTensor, CudarcTensor, MetalTensor};
use crate::core::device::Device;
use crate::core::dtype::{DType, Element};
use crate::core::error::{CoreError, Result};
use crate::core::shape::Shape;
use candle_core::{self};
use candle_nn;

fn candle_device_from(device: &Device) -> Result<candle_core::Device> {
    match device {
        Device::Cpu => Ok(candle_core::Device::Cpu),
        Device::Cuda(n) => Ok(candle_core::Device::new_cuda(*n)?),
        // CandleTensor is never constructed on a Metal device in axiom today —
        // MetalTensor (its own struct) handles Metal directly without going
        // through candle_core at all. This arm exists only so the match stays
        // exhaustive; it errors loudly if CandleTensor::* is ever called with
        // a Metal device by mistake, rather than silently doing the wrong thing.
        Device::Metal(_) => Err(CoreError::Internal(
            "CandleTensor does not support Device::Metal — use MetalTensor instead".to_string(),
        )),
    }
}

pub struct TopKOutput<T> {
    pub values: T,
    pub indices: T,
}

pub trait TopKLastDimOp {
    fn topk(&self, k: usize) -> Result<TopKOutput<Self>>
    where
        Self: Sized;
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
    fn from_slice<E: Element>(data: &[E], shape: &Shape, device: &Device) -> Result<Self>;

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

    fn cos(&self) -> Result<Self>;
    fn sin(&self) -> Result<Self>;
    fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self>; // candle name
    fn chunk(&self, chunks: usize, dim: usize) -> Result<Vec<Self>>; // split into N equal parts
    fn cat(tensors: &[&Self], dim: usize) -> Result<Self>; // concatenate — note: associated fn
    fn neg(&self) -> Result<Self>; // negate — needed for rotation
    fn broadcast_mul(&self, other: &Self) -> Result<Self>;
    fn repeat(&self, shape: &Shape) -> Result<Self>;
    fn sigmoid(&self) -> Result<Self>;
    fn to_vec_u32(&self) -> Result<Vec<u32>>;
    fn zeros_like(&self) -> Result<Self>;
    fn sum_keepdim(&self, dim: usize) -> Result<Self>;
    fn broadcast_div(&self, other: &Self) -> Result<Self>;
    fn to_vec_f32(&self) -> Result<Vec<f32>>;
    fn exp(&self) -> Result<Self>;
    fn log(&self) -> Result<Self>;
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

    // No longer bounded by candle_core::WithDType at the trait level (see TensorOps::from_slice).
    // We reinterpret E's bytes and hand Candle a raw buffer + our own DType -> candle_core::DType
    // mapping instead, so the WithDType requirement stays an internal Candle-impl detail rather
    // than leaking into every backend (MetalTensor in particular has no reason to know about it).
    //
    // NOTE: Tensor::from_raw_buffer(data: &[u8], dtype: DType, shape: &[usize], device: &Device)
    // is confirmed present in candle-core's safetensors.rs as of recent releases. This crate is
    // pinned to candle-core = "0.6" in Cargo.toml — verify the method exists at that exact pin
    // (`cargo doc --open -p candle-core` or docs.rs/candle-core/0.6.0) before relying on this.
    // If it's missing at 0.6, the fallback is a small per-dtype match calling the existing
    // WithDType-bounded `candle_core::Tensor::from_slice` internally (still trait-clean, since
    // the match — not a generic bound — would live entirely inside this impl).
    fn from_slice<E: Element>(data: &[E], shape: &Shape, device: &Device) -> Result<Self> {
        let candle_device = candle_device_from(device)?;
        let candle_dtype: candle_core::DType = E::dtype().into();

        // SAFETY: E: Copy + Send + Sync + 'static (via Element), and we only ever read
        // size_of_val(data) bytes — exactly the slice's own backing memory, no overrun.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };

        let inner = candle_core::Tensor::from_raw_buffer(
            bytes,
            candle_dtype,
            shape.dims(),
            &candle_device,
        )?;

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
        let new_shape = Shape::new(&inner.dims());
        Ok(CandleTensor {
            inner,
            shape: new_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn sub(&self, other: &Self) -> crate::core::error::Result<Self> {
        Device::check_same("add", &self.device, &other.device)?;
        Shape::elementwise_check(&self.shape, &other.shape)?;
        let inner = self.inner.sub(&other.inner)?;
        let new_shape = Shape::new(&inner.dims());
        Ok(CandleTensor {
            inner,
            shape: new_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn mul(&self, other: &Self) -> crate::core::error::Result<Self> {
        Device::check_same("mul", &self.device, &other.device)?;
        Shape::elementwise_check(&self.shape, &other.shape)?;
        let inner = self.inner.mul(&other.inner)?;
        let new_shape = Shape::new(&inner.dims());
        Ok(CandleTensor {
            inner,
            shape: new_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn div(&self, other: &Self) -> crate::core::error::Result<Self> {
        Device::check_same("div", &self.device, &other.device)?;
        Shape::elementwise_check(&self.shape, &other.shape)?;
        let inner = self.inner.div(&other.inner)?;
        let new_shape = Shape::new(&inner.dims());
        Ok(CandleTensor {
            inner,
            shape: new_shape,
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
        let new_shape = Shape::new(&inner.dims());
        Ok(CandleTensor {
            inner,
            shape: new_shape,
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

    fn cos(&self) -> Result<Self> {
        let inner = self.inner.cos()?;
        Ok(CandleTensor {
            inner,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn sin(&self) -> Result<Self> {
        let inner = self.inner.sin()?;
        Ok(CandleTensor {
            inner,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let inner = self.inner.narrow(dim, start, len)?;
        let new_shape = Shape::new(&inner.dims());
        Ok(CandleTensor {
            inner,
            shape: new_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn chunk(&self, chunks: usize, dim: usize) -> Result<Vec<Self>> {
        let chunks_inner = self.inner.chunk(chunks, dim)?;
        chunks_inner
            .into_iter()
            .map(|c| {
                let shape = Shape::new(&c.dims());
                Ok(CandleTensor {
                    shape,
                    dtype: self.dtype,
                    device: self.device.clone(),
                    inner: c,
                })
            })
            .collect()
    }

    fn cat(tensors: &[&Self], dim: usize) -> Result<Self> {
        let inners: Vec<&candle_core::Tensor> = tensors.iter().map(|t| &t.inner).collect();
        let inner = candle_core::Tensor::cat(&inners, dim)?;
        let shape = Shape::new(&inner.dims());
        Ok(CandleTensor {
            inner,
            shape,
            dtype: tensors[0].dtype,
            device: tensors[0].device.clone(),
        })
    }

    fn neg(&self) -> Result<Self> {
        let inner = self.inner.neg()?;

        Ok(CandleTensor {
            inner,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn broadcast_mul(&self, other: &Self) -> Result<Self> {
        let inner = self.inner.broadcast_mul(&other.inner)?;
        let new_shape = Shape::new(&inner.dims());
        Ok(CandleTensor {
            inner,
            shape: new_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn repeat(&self, shape: &Shape) -> Result<Self> {
        let inner = self.inner.repeat(shape.dims())?;
        let new_shape = Shape::new(&inner.dims());
        Ok(CandleTensor {
            inner,
            shape: new_shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn sigmoid(&self) -> Result<Self> {
        let inner = candle_nn::ops::sigmoid(&self.inner)?;
        Ok(CandleTensor {
            inner,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn to_vec_u32(&self) -> Result<Vec<u32>> {
        self.inner
            .flatten_all()?
            .to_vec1::<u32>()
            .map_err(|e| CoreError::Candle(e))
    }
    fn zeros_like(&self) -> Result<Self> {
        let inner = self.inner.zeros_like()?;
        Ok(CandleTensor {
            inner,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn sum_keepdim(&self, dim: usize) -> Result<Self> {
        let inner = self.inner.sum_keepdim(dim)?;
        Ok(CandleTensor {
            inner,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn broadcast_div(&self, other: &Self) -> Result<Self> {
        let inner = self.inner.broadcast_div(&other.inner)?;
        let shape = Shape::new(&inner.dims());
        Ok(CandleTensor {
            inner,
            shape,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn to_vec_f32(&self) -> Result<Vec<f32>> {
        self.inner
            .flatten_all()?
            .to_vec1::<f32>()
            .map_err(|e| CoreError::Candle(e))
    }

    fn exp(&self) -> Result<Self> {
        Ok(CandleTensor {
            inner: self.inner.exp()?,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn log(&self) -> Result<Self> {
        Ok(CandleTensor {
            inner: self.inner.log()?,
            shape: self.shape.clone(),
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
    fn from_slice<E: Element>(_: &[E], _: &Shape, _: &Device) -> Result<Self> {
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
    fn index_select(&self, _indexes: &Self, _dim: usize) -> Result<Self> {
        todo!("phase 4")
    }

    fn cos(&self) -> Result<Self> {
        todo!("Phase 4")
    }
    fn sin(&self) -> Result<Self> {
        todo!("Phase 4")
    }

    fn chunk(&self, _: usize, _: usize) -> Result<Vec<Self>> {
        todo!("Phase 4")
    }

    fn cat(_: &[&Self], _: usize) -> Result<Self> {
        todo!("Phase 4")
    }

    fn neg(&self) -> Result<Self> {
        todo!("Phase 4")
    }

    fn narrow(&self, _dim: usize, _start: usize, _len: usize) -> Result<Self> {
        todo!("Phase 4")
    }

    fn broadcast_mul(&self, _: &Self) -> Result<Self> {
        todo!("Phase 4")
    }

    fn repeat(&self, _: &Shape) -> Result<Self> {
        todo!("Phase 4")
    }

    fn sigmoid(&self) -> Result<Self> {
        todo!("Phase 4 ")
    }

    fn to_vec_u32(&self) -> Result<Vec<u32>> {
        todo!("phase 4")
    }

    fn zeros_like(&self) -> Result<Self> {
        todo!("phase 4")
    }

    fn sum_keepdim(&self, _dim: usize) -> Result<Self> {
        todo!("Phase 4")
    }
    fn broadcast_div(&self, _other: &Self) -> Result<Self> {
        todo!("phase 4")
    }

    fn to_vec_f32(&self) -> Result<Vec<f32>> {
        todo!("phase 4")
    }

    fn exp(&self) -> Result<Self> {
        todo!("phase 4")
    }

    fn log(&self) -> Result<Self> {
        todo!("phase 4")
    }
}

// MetalTensor's TensorOps impl. Stubbed identically to CudarcTensor for now — every
// method becomes a real MTLBuffer-backed op as the Metal kernel work lands. Unlike
// CudarcTensor, this backend never routes through candle_core: from_slice etc. will
// do a direct byte-copy into an MTLBuffer (unified memory means this is close to free
// on Apple Silicon — no separate host/device transfer step the way CUDA needs).
impl TensorOps for MetalTensor {
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
        todo!("Metal Phase 1")
    }
    fn ones(_: &Shape, _: DType, _: &Device) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn from_slice<E: Element>(_: &[E], _: &Shape, _: &Device) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn from_u32_slice(_: &[u32], _: &Shape, _: &Device) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn to_device(&self, _: &Device) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn to_dtype(&self, _: DType) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn contiguous(&self) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn reshape(&self, _: &Shape) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn transpose(&self, _: usize, _: usize) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn squeeze(&self, _: usize) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn unsqueeze(&self, _: usize) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn add(&self, _: &Self) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn sub(&self, _: &Self) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn mul(&self, _: &Self) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn div(&self, _: &Self) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn scale(&self, _: f64) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn matmul(&self, _: &Self) -> Result<Self> {
        // The foundation kernel — sequencing notes from project planning apply here first.
        todo!("Metal Phase 1")
    }
    fn broadcast_add(&self, _: &Self) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn sum(&self, _: usize) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn mean(&self, _: usize) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn silu(&self) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn gelu(&self) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn softmax(&self, _: usize) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn sqrt(&self) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn rms_norm(&self, _: &Self, _: f32) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn broadcast_matmul(&self, _: &Self) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn index_select(&self, _indexes: &Self, _dim: usize) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn cos(&self) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn sin(&self) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn chunk(&self, _: usize, _: usize) -> Result<Vec<Self>> {
        todo!("Metal Phase 1")
    }
    fn cat(_: &[&Self], _: usize) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn neg(&self) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn narrow(&self, _dim: usize, _start: usize, _len: usize) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn broadcast_mul(&self, _: &Self) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn repeat(&self, _: &Shape) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn sigmoid(&self) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn to_vec_u32(&self) -> Result<Vec<u32>> {
        todo!("Metal Phase 1")
    }
    fn zeros_like(&self) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn sum_keepdim(&self, _dim: usize) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn broadcast_div(&self, _other: &Self) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn to_vec_f32(&self) -> Result<Vec<f32>> {
        todo!("Metal Phase 1")
    }
    fn exp(&self) -> Result<Self> {
        todo!("Metal Phase 1")
    }
    fn log(&self) -> Result<Self> {
        todo!("Metal Phase 1")
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
