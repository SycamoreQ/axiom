use crate::core::backend::MetalTensor;
use crate::core::backend::{CandleTensor, CudarcTensor};
use crate::core::device::Device;
use crate::core::dtype::{DType, Element};
use crate::core::error::{CoreError, Result};
use crate::core::shape::Shape;
use crate::core::tensor;
use crate::metal::{allocator::BlockHandle, state::global_metal_state, MetalState};
use candle_core::{self, Tensor};
use candle_nn;
#[cfg(feature = "metal")]
use objc2_metal::MTLCommandBuffer;
use std::sync::Arc;

fn candle_device_from(device: &Device) -> Result<candle_core::Device> {
    match device {
        Device::Cpu => Ok(candle_core::Device::Cpu),
        Device::Cuda(n) => Ok(candle_core::Device::new_cuda(*n)?),
        // CandleTensor is never constructed on a Metal device in axiom today —
        // MetalTensor (its own struct) handles Metal directly without going
        // through candle_core at all. This arm exists only so the match stays
        // exhaustive; it errors loudly if CandleTensor::* is ever called with
        // a Metal device by mistake, rather than silently doing the wrong thing.
        #[cfg(feature = "metal")]
        Device::Metal(_) => Err(CoreError::Internal(
            "CandleTensor does not support Device::Metal — use MetalTensor instead".to_string(),
        )),
    }
}

#[cfg(feature = "metal")]
impl MetalTensor {
    //Helper to create a standard, contiguous tensor layout
    pub(crate) fn new_contiguous(
        state: Arc<MetalState>,
        block: Arc<BlockHandle>,
        shape: Shape,
        dtype: DType,
        device: Device,
    ) -> Self {
        // Calculate standard row-major strides based on the shape.
        // (If your Shape struct already has a method for this, use it!)
        let dims = shape.dims();
        let mut strides = vec![1; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }

        Self {
            state,
            block,
            shape,
            strides,
            offset_bytes: 0,
            dtype,
            device,
        }
    }

    pub(crate) fn contiguous_copy(&self) -> Result<Self> {
        // read through strides into a flat f32 vec, write to a fresh allocation
        let data = self.to_vec_f32()?; // already handles offset_bytes
        Self::from_slice(&data, &self.shape, &self.device)
    }

    fn compute_strides(dims: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        strides
    }

    pub(crate) fn from_bytes_direct(
        state: Arc<MetalState>,
        data: &[u8],
        shape: Shape,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        use objc2::rc::Retained;
        use objc2_metal::MTLBuffer;
        use objc2_metal::{MTLDevice, MTLResourceOptions};
        use std::ptr::NonNull;

        let raw_device = state.ctx.device.raw();

        let buffer = unsafe {
            raw_device.newBufferWithBytes_length_options(
                NonNull::new_unchecked(data.as_ptr() as *mut std::ffi::c_void),
                data.len(),
                MTLResourceOptions(0), // StorageModeShared = 0
            )
        }
        .ok_or_else(|| {
            CoreError::Metal(format!(
                "MTLBuffer allocation failed for {} bytes",
                data.len()
            ))
        })?;

        let ptr = unsafe { buffer.contents().as_ptr() as *mut u8 };

        let block = BlockHandle {
            index: 0,
            ptr,
            offset_bytes: 0,
            size_bytes: data.len(),
            owned_buffer: Some(buffer),
        };

        Ok(Self::new_contiguous(
            state,
            Arc::new(block),
            shape,
            dtype,
            device,
        ))
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

/*  MetalTensor's TensorOps impl. Unlike
CudarcTensor, this backend never routes through candle_core: from_slice etc. will
do a direct byte-copy into an MTLBuffer (unified memory means this is close to free
on Apple Silicon — no separate host/device transfer step the way CUDA needs).
*/

#[cfg(feature = "metal")]
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

    fn zeros(shape: &Shape, dtype: DType, device: &Device) -> Result<Self> {
        let n = shape.numel() * dtype.size_in_bytes();
        let data = vec![0u8; n];
        let state = global_metal_state()
            .ok_or_else(|| CoreError::Internal("Metal state not initialized".into()))?;
        Self::from_bytes_direct(state, &data, shape.clone(), dtype, device.clone())
    }

    fn ones(shape: &Shape, dtype: DType, device: &Device) -> Result<Self> {
        let n = shape.numel();
        let state = global_metal_state()
            .ok_or_else(|| CoreError::Internal("Metal state not initialized".into()))?;

        let bytes: Vec<u8> = match dtype {
            DType::F32 => {
                let mut buf = vec![0u8; n * 4];
                let ptr = buf.as_mut_ptr() as *mut f32;
                unsafe {
                    for i in 0..n {
                        ptr.add(i).write(1.0f32);
                    }
                }
                buf
            }
            DType::F16 => {
                let mut buf = vec![0u8; n * 2];
                let ptr = buf.as_mut_ptr() as *mut half::f16;
                unsafe {
                    for i in 0..n {
                        ptr.add(i).write(half::f16::from_f32(1.0));
                    }
                }
                buf
            }
            _ => return Err(CoreError::Internal("ones: unsupported dtype".into())),
        };

        Self::from_bytes_direct(state, &bytes, shape.clone(), dtype, device.clone())
    }

    fn from_slice<E: Element>(data: &[E], shape: &Shape, device: &Device) -> Result<Self> {
        let state = global_metal_state()
            .ok_or_else(|| CoreError::Internal("Metal state not initialized".into()))?;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        let tensor =
            Self::from_bytes_direct(state, bytes, shape.clone(), E::dtype(), device.clone())?;

        Ok(tensor)
    }

    fn from_u32_slice(data: &[u32], shape: &Shape, device: &Device) -> Result<Self> {
        Self::from_slice(data, shape, device)
    }

    fn to_device(&self, device: &Device) -> Result<Self> {
        match device {
            Device::Metal(_) => Ok(self.clone()),
            _ => Err(CoreError::Internal(
                "MetalTensor can only live on Metal device".into(),
            )),
        }
    }

    fn to_dtype(&self, dtype: DType) -> Result<Self> {
        let src_f32 = self.to_vec_f32()?;
        match dtype {
            DType::F32 => Self::from_slice(&src_f32, &self.shape, &self.device),
            DType::F16 => {
                let converted: Vec<half::f16> =
                    src_f32.iter().map(|&x| half::f16::from_f32(x)).collect();
                Self::from_slice(&converted, &self.shape, &self.device)
            }
            _ => Err(CoreError::Internal("to_dtype: unsupported dtype".into())),
        }
    }

    fn contiguous(&self) -> Result<Self> {
        let expected = Self::compute_strides(self.shape.dims());
        if self.strides == expected && self.offset_bytes == 0 {
            Ok(self.clone())
        } else {
            self.contiguous_copy()
        }
    }

    fn reshape(&self, shape: &Shape) -> Result<Self> {
        assert_eq!(
            shape.numel(),
            self.shape.numel(),
            "reshape: element count must match"
        );
        Ok(Self::new_contiguous(
            self.state.clone(),
            self.block.clone(),
            shape.clone(),
            self.dtype,
            self.device.clone(),
        ))
    }

    fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        let mut new_shape = self.shape.dims().to_vec();
        let mut new_strides = self.strides.clone();

        new_shape.swap(dim1, dim2);
        new_strides.swap(dim1, dim2);

        Ok(Self {
            state: self.state.clone(),
            block: self.block.clone(),
            shape: Shape::new(&new_shape),
            strides: new_strides,
            offset_bytes: self.offset_bytes,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn squeeze(&self, dim: usize) -> Result<Self> {
        let mut dims = self.shape.dims().to_vec();
        assert_eq!(dims[dim], 1, "squeeze: dim must be size 1");
        dims.remove(dim);
        self.reshape(&Shape::new(&dims))
    }

    fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let mut dims = self.shape.dims().to_vec();
        dims.insert(dim, 1);
        self.reshape(&Shape::new(&dims))
    }

    fn add(&self, other: &Self) -> Result<Self> {
        let n = self.shape.numel();
        let output = Self::zeros(&self.shape, self.dtype, &self.device)?;
        unsafe {
            match self.dtype {
                DType::F32 => {
                    let a =
                        (self.block.as_ref().ptr as *const u8).add(self.offset_bytes) as *const f32;
                    let b = (other.block.as_ref().ptr as *const u8).add(other.offset_bytes)
                        as *const f32;
                    let c =
                        (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes) as *mut f32;
                    for i in 0..n {
                        c.add(i).write(a.add(i).read() + b.add(i).read());
                    }
                }
                DType::F16 => {
                    let a = (self.block.as_ref().ptr as *const u8).add(self.offset_bytes)
                        as *const half::f16;
                    let b = (other.block.as_ref().ptr as *const u8).add(other.offset_bytes)
                        as *const half::f16;
                    let c = (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes)
                        as *mut half::f16;
                    for i in 0..n {
                        let sum = a.add(i).read().to_f32() + b.add(i).read().to_f32();
                        c.add(i).write(half::f16::from_f32(sum));
                    }
                }
                _ => return Err(CoreError::Internal("add: unsupported dtype".into())),
            }
        }
        Ok(output)
    }

    fn sub(&self, other: &Self) -> Result<Self> {
        if self.shape != other.shape {
            return Err(CoreError::Internal("sub: shape mismatch".into()));
        }
        let n = self.shape.numel();
        let output = Self::zeros(&self.shape, self.dtype, &self.device)?;
        unsafe {
            match self.dtype {
                DType::F32 => {
                    let a =
                        (self.block.as_ref().ptr as *const u8).add(self.offset_bytes) as *const f32;
                    let b = (other.block.as_ref().ptr as *const u8).add(other.offset_bytes)
                        as *const f32;
                    let c =
                        (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes) as *mut f32;
                    for i in 0..n {
                        c.add(i).write(a.add(i).read() - b.add(i).read());
                    }
                }
                DType::F16 => {
                    let a = (self.block.as_ref().ptr as *const u8).add(self.offset_bytes)
                        as *const half::f16;
                    let b = (other.block.as_ref().ptr as *const u8).add(other.offset_bytes)
                        as *const half::f16;
                    let c = (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes)
                        as *mut half::f16;
                    for i in 0..n {
                        let diff = a.add(i).read().to_f32() - b.add(i).read().to_f32();
                        c.add(i).write(half::f16::from_f32(diff));
                    }
                }
                _ => return Err(CoreError::Internal("sub: unsupported dtype".into())),
            }
        }
        Ok(output)
    }

    fn mul(&self, other: &Self) -> Result<Self> {
        let n = self.shape.numel();
        let output = Self::zeros(&self.shape, self.dtype, &self.device)?;
        unsafe {
            match self.dtype {
                DType::F32 => {
                    let a =
                        (self.block.as_ref().ptr as *const u8).add(self.offset_bytes) as *const f32;
                    let b = (other.block.as_ref().ptr as *const u8).add(other.offset_bytes)
                        as *const f32;
                    let c =
                        (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes) as *mut f32;
                    for i in 0..n {
                        c.add(i).write(a.add(i).read() * b.add(i).read());
                    }
                }
                DType::F16 => {
                    let a = (self.block.as_ref().ptr as *const u8).add(self.offset_bytes)
                        as *const half::f16;
                    let b = (other.block.as_ref().ptr as *const u8).add(other.offset_bytes)
                        as *const half::f16;
                    let c = (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes)
                        as *mut half::f16;
                    for i in 0..n {
                        let prod = a.add(i).read().to_f32() * b.add(i).read().to_f32();
                        c.add(i).write(half::f16::from_f32(prod));
                    }
                }
                _ => return Err(CoreError::Internal("mul: unsupported dtype".into())),
            }
        }
        Ok(output)
    }

    fn div(&self, other: &Self) -> Result<Self> {
        let n = self.shape.numel();
        let output = Self::zeros(&self.shape, self.dtype, &self.device)?;
        unsafe {
            match self.dtype {
                DType::F32 => {
                    let a =
                        (self.block.as_ref().ptr as *const u8).add(self.offset_bytes) as *const f32;
                    let b = (other.block.as_ref().ptr as *const u8).add(other.offset_bytes)
                        as *const f32;
                    let c =
                        (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes) as *mut f32;
                    for i in 0..n {
                        c.add(i).write(a.add(i).read() / b.add(i).read());
                    }
                }
                DType::F16 => {
                    let a = (self.block.as_ref().ptr as *const u8).add(self.offset_bytes)
                        as *const half::f16;
                    let b = (other.block.as_ref().ptr as *const u8).add(other.offset_bytes)
                        as *const half::f16;
                    let c = (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes)
                        as *mut half::f16;
                    for i in 0..n {
                        let div = a.add(i).read().to_f32() / b.add(i).read().to_f32();
                        c.add(i).write(half::f16::from_f32(div));
                    }
                }
                _ => return Err(CoreError::Internal("div: unsupported dtype".into())),
            }
        }
        Ok(output)
    }

    fn scale(&self, scalar: f64) -> Result<Self> {
        let n = self.shape.numel();
        let output = Self::zeros(&self.shape, self.dtype, &self.device)?;
        let scalar_f32 = scalar as f32;
        unsafe {
            match self.dtype {
                DType::F32 => {
                    let src =
                        (self.block.as_ref().ptr as *const u8).add(self.offset_bytes) as *const f32;
                    let dst =
                        (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes) as *mut f32;
                    for i in 0..n {
                        dst.add(i).write(src.add(i).read() * scalar_f32);
                    }
                }
                DType::F16 => {
                    let src = (self.block.as_ref().ptr as *const u8).add(self.offset_bytes)
                        as *const half::f16;
                    let dst = (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes)
                        as *mut half::f16;
                    for i in 0..n {
                        let scaled = src.add(i).read().to_f32() * scalar_f32;
                        dst.add(i).write(half::f16::from_f32(scaled));
                    }
                }
                _ => return Err(CoreError::Internal("scale: unsupported dtype".into())),
            }
        }
        Ok(output)
    }

    fn matmul(&self, other: &Self) -> Result<Self> {
        let m = self.shape.dims()[0];
        let k = self.shape.dims()[1];
        let n = other.shape.dims()[1];

        let a = self.to_vec_f32()?;
        let b = other.to_vec_f32()?;
        let mut c = vec![0.0f32; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        let out_tensor = Self::from_slice(&c, &Shape::new(&[m, n]), &self.device)?;
        out_tensor.to_dtype(self.dtype)
    }

    fn broadcast_add(&self, other: &Self) -> Result<Self> {
        let n = self.shape.numel();
        let bias_len = other.shape.numel();
        let output = Self::zeros(&self.shape, self.dtype, &self.device)?;
        unsafe {
            let a = (self.block.as_ref().ptr as *const u8).add(self.offset_bytes) as *const f32;
            let b = (other.block.as_ref().ptr as *const u8).add(other.offset_bytes) as *const f32;
            let c = (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes) as *mut f32;
            for i in 0..n {
                c.add(i).write(a.add(i).read() + b.add(i % bias_len).read());
            }
        }
        Ok(output)
    }

    fn sum(&self, dim: usize) -> Result<Self> {
        let src = self.to_vec_f32()?;
        let dims = self.shape.dims();
        let mut new_dims = dims.to_vec();
        new_dims.remove(dim);

        let stride = self.strides[dim];
        let dim_size = dims[dim];
        let num_elements = new_dims.iter().product::<usize>();

        let mut dst = vec![0.0f32; num_elements];
        for i in 0..num_elements {
            let mut val = 0.0;
            for j in 0..dim_size {
                let index = (i / stride) * (stride * dim_size) + (i % stride) + (j * stride);
                val += src[index];
            }
            dst[i] = val;
        }

        let out_tensor = Self::from_slice(&dst, &Shape::new(&new_dims), &self.device)?;
        out_tensor.to_dtype(self.dtype)
    }

    fn mean(&self, dim: usize) -> Result<Self> {
        let sum_tensor = self.sum(dim)?;
        let dim_size = self.shape.dims()[dim] as f64;
        sum_tensor.scale(1.0 / dim_size)
    }

    fn silu(&self) -> Result<Self> {
        let n = self.shape.numel();
        let output = Self::zeros(&self.shape, self.dtype, &self.device)?;
        unsafe {
            let src = (self.block.as_ref().ptr as *const u8).add(self.offset_bytes) as *const f32;
            let dst = (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes) as *mut f32;
            for i in 0..n {
                let x = src.add(i).read();
                dst.add(i).write(x / (1.0 + (-x).exp()));
            }
        }
        Ok(output)
    }

    fn gelu(&self) -> Result<Self> {
        let n = self.shape.numel();
        let output = Self::zeros(&self.shape, self.dtype, &self.device)?;
        unsafe {
            match self.dtype {
                DType::F32 => {
                    let src =
                        (self.block.as_ref().ptr as *const u8).add(self.offset_bytes) as *const f32;
                    let dst =
                        (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes) as *mut f32;
                    for i in 0..n {
                        let x = src.add(i).read();
                        let g = 0.5 * x * (1.0 + f32::tanh(0.797884 * (x + 0.044715 * x * x * x)));
                        dst.add(i).write(g);
                    }
                }
                DType::F16 => {
                    let src = (self.block.as_ref().ptr as *const u8).add(self.offset_bytes)
                        as *const half::f16;
                    let dst = (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes)
                        as *mut half::f16;
                    for i in 0..n {
                        let x = src.add(i).read().to_f32();
                        let g = 0.5 * x * (1.0 + f32::tanh(0.797884 * (x + 0.044715 * x * x * x)));
                        dst.add(i).write(half::f16::from_f32(g));
                    }
                }
                _ => return Err(CoreError::Internal("gelu: unsupported dtype".into())),
            }
        }
        Ok(output)
    }

    fn softmax(&self, dim: usize) -> Result<Self> {
        let dims = self.shape.dims();
        let row_size = dims[dim];
        let n = self.shape.numel();
        let num_rows = n / row_size;
        let src = self.to_vec_f32()?;
        let mut dst = vec![0.0f32; n];
        for r in 0..num_rows {
            let row = &src[r * row_size..(r + 1) * row_size];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (i, &e) in exps.iter().enumerate() {
                dst[r * row_size + i] = e / sum;
            }
        }
        let out = Self::from_slice(&dst, &self.shape, &self.device)?;
        out.to_dtype(self.dtype)
    }

    fn sqrt(&self) -> Result<Self> {
        let n = self.shape.numel();
        let output = Self::zeros(&self.shape, self.dtype, &self.device)?;
        unsafe {
            match self.dtype {
                DType::F32 => {
                    let src =
                        (self.block.as_ref().ptr as *const u8).add(self.offset_bytes) as *const f32;
                    let dst =
                        (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes) as *mut f32;
                    for i in 0..n {
                        dst.add(i).write(src.add(i).read().sqrt());
                    }
                }
                DType::F16 => {
                    let src = (self.block.as_ref().ptr as *const u8).add(self.offset_bytes)
                        as *const half::f16;
                    let dst = (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes)
                        as *mut half::f16;
                    for i in 0..n {
                        dst.add(i)
                            .write(half::f16::from_f32(src.add(i).read().to_f32().sqrt()));
                    }
                }
                _ => return Err(CoreError::Internal("sqrt: unsupported dtype".into())),
            }
        }
        Ok(output)
    }

    /// NON APPLE FOR NOW
    fn rms_norm(&self, weight: &Self, eps: f32) -> Result<Self> {
        let hidden = *self.shape.dims().last().unwrap();
        let num_tokens = self.shape.numel() / hidden;
        let src = self.to_vec_f32()?;
        let wgt = weight.to_vec_f32()?;

        let mut out = vec![0.0f32; src.len()];
        for t in 0..num_tokens {
            let row_start = t * hidden;
            let row = &src[row_start..row_start + hidden];
            let ms = row.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
            let rms = (ms + eps).sqrt();
            for i in 0..hidden {
                out[row_start + i] = row[i] * wgt[i] / rms;
            }
        }
        Self::from_slice(&out, &self.shape, &self.device)
    }

    fn broadcast_matmul(&self, other: &Self) -> Result<Self> {
        // Read both tensors into CPU memory
        let a = self.to_vec_f32()?;
        let w_orig = other.to_vec_f32()?;

        let rank = self.shape.rank();
        let other_rank = other.shape.rank();

        // Inner dimension
        let k = self.shape.dims()[rank - 1];

        // `other` is always expected in [..., K, N] form — every call site in
        // this codebase (Linear::forward, attention QK^T/softmax@V) already
        // explicitly transposes its second operand into this form before
        // calling broadcast_matmul. The previous version tried to guess
        // orientation by comparing shape dims against k, which is genuinely
        // ambiguous whenever K == N (e.g. attn_q/attn_o's square 2048x2048
        // weight) and silently picked the wrong branch in that case.
        let n = if other_rank == 2 {
            if other.shape.dims()[0] != k {
                return Err(CoreError::Internal(format!(
                    "matmul shape mismatch: self.shape={:?} (k={}), other.shape={:?}",
                    self.shape.dims(),
                    k,
                    other.shape.dims()
                )));
            }
            other.shape.dims()[1]
        } else {
            other.shape.dims()[other_rank - 1]
        };
        let w_is_transposed = true;

        let m_per = self.shape.dims()[rank - 2];
        let batch_self: usize = self.shape.dims()[..rank - 2].iter().product();
        let batch_other: usize = if other_rank > 2 {
            other.shape.dims()[..other_rank - 2].iter().product()
        } else {
            1
        };
        let batch_out = batch_self.max(batch_other);

        let mut out = vec![0.0f32; batch_out * m_per * n];

        for batch_idx in 0..batch_out {
            let self_b = if batch_self == 1 { 0 } else { batch_idx };
            let other_b = if batch_other == 1 { 0 } else { batch_idx };

            let a_offset = self_b * m_per * k;
            let w_offset = other_b * k * n; // only used when w_is_transposed
            let c_offset = batch_idx * m_per * n;

            for i in 0..m_per {
                for j in 0..n {
                    let mut sum = 0.0;
                    for p in 0..k {
                        let a_val = a[a_offset + i * k + p];
                        let w_val = if w_is_transposed {
                            // w_orig is [..., K, N] → element (p, j) is at w_offset + p * n + j
                            w_orig[w_offset + p * n + j]
                        } else {
                            // w_orig is [N, K] → element (p, j) is at other_b * (n*k) + j * k + p
                            w_orig[other_b * n * k + j * k + p]
                        };
                        sum += a_val * w_val;
                    }
                    out[c_offset + i * n + j] = sum;
                }
            }
        }

        // Determine output shape
        let mut final_shape = Vec::new();
        if batch_out > 1 {
            final_shape.extend(self.shape.dims()[..rank - 2].iter());
            if final_shape.is_empty() {
                final_shape.push(batch_out);
            }
        }
        final_shape.push(m_per);
        final_shape.push(n);
        let shape = Shape::new(&final_shape);

        Self::from_slice(&out, &shape, &self.device)
    }

    fn index_select(&self, indexes: &Self, dim: usize) -> Result<Self> {
        let idx = indexes.to_vec_u32()?;
        let src = self.to_vec_f32()?;
        let dims = self.shape.dims();
        let slice_size: usize = dims[dim + 1..].iter().product();
        let outer_size: usize = dims[..dim].iter().product();

        let out_len = outer_size * idx.len() * slice_size;
        let mut out = vec![0.0f32; out_len];

        for o in 0..outer_size {
            for (out_i, &idx_val) in idx.iter().enumerate() {
                let src_offset = (o * dims[dim] + idx_val as usize) * slice_size;
                let dst_offset = (o * idx.len() + out_i) * slice_size;
                out[dst_offset..dst_offset + slice_size]
                    .copy_from_slice(&src[src_offset..src_offset + slice_size]);
            }
        }

        let mut out_dims = dims.to_vec();
        out_dims[dim] = idx.len();
        let nonzero = out.iter().filter(|&&x| x != 0.0).count();
        eprintln!(
            "index_select: idx={:?} out_shape={:?} nonzero={}/{}",
            &idx[..idx.len().min(4)],
            out_dims,
            nonzero,
            out.len()
        );

        Self::from_slice(&out, &Shape::new(&out_dims), &self.device)
    }

    fn cos(&self) -> Result<Self> {
        let n = self.shape.numel();
        let output = Self::zeros(&self.shape, self.dtype, &self.device)?;
        unsafe {
            match self.dtype {
                DType::F32 => {
                    let src =
                        (self.block.as_ref().ptr as *const u8).add(self.offset_bytes) as *const f32;
                    let dst =
                        (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes) as *mut f32;
                    for i in 0..n {
                        dst.add(i).write(src.add(i).read().cos());
                    }
                }
                DType::F16 => {
                    let src = (self.block.as_ref().ptr as *const u8).add(self.offset_bytes)
                        as *const half::f16;
                    let dst = (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes)
                        as *mut half::f16;
                    for i in 0..n {
                        dst.add(i)
                            .write(half::f16::from_f32(src.add(i).read().to_f32().cos()));
                    }
                }
                _ => return Err(CoreError::Internal("cos: unsupported dtype".into())),
            }
        }
        Ok(output)
    }

    fn sin(&self) -> Result<Self> {
        let n = self.shape.numel();
        let output = Self::zeros(&self.shape, self.dtype, &self.device)?;
        unsafe {
            match self.dtype {
                DType::F32 => {
                    let src =
                        (self.block.as_ref().ptr as *const u8).add(self.offset_bytes) as *const f32;
                    let dst =
                        (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes) as *mut f32;
                    for i in 0..n {
                        dst.add(i).write(src.add(i).read().sin());
                    }
                }
                DType::F16 => {
                    let src = (self.block.as_ref().ptr as *const u8).add(self.offset_bytes)
                        as *const half::f16;
                    let dst = (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes)
                        as *mut half::f16;
                    for i in 0..n {
                        dst.add(i)
                            .write(half::f16::from_f32(src.add(i).read().to_f32().sin()));
                    }
                }
                _ => return Err(CoreError::Internal("sin: unsupported dtype".into())),
            }
        }
        Ok(output)
    }

    fn chunk(&self, n: usize, dim: usize) -> Result<Vec<Self>> {
        let dim_size = self.shape().dims()[dim];
        let chunk_size = dim_size / n;
        let remainder = dim_size % n;

        let mut new_chunks = Vec::with_capacity(n);
        let mut current_offset = 0;

        for i in 0..n {
            let current_chunk_size = if i < remainder {
                chunk_size + 1
            } else {
                chunk_size
            };
            let chunk_tensor = self.narrow(dim, current_offset, current_chunk_size)?;
            new_chunks.push(chunk_tensor);
            current_offset += current_chunk_size;
        }

        Ok(new_chunks)
    }

    fn cat(tensors: &[&Self], dim: usize) -> Result<Self> {
        let mut new_dims = tensors[0].shape.dims().to_vec();
        let total_dim_size: usize = tensors.iter().map(|t| t.shape.dims()[dim]).sum();
        new_dims[dim] = total_dim_size;
        let output = Self::zeros(&Shape::new(&new_dims), tensors[0].dtype, &tensors[0].device)?;

        let mut write_offset_bytes = 0usize;
        for t in tensors {
            let size_bytes = t.shape.numel() * t.dtype.size_in_bytes();
            unsafe {
                let src = (t.block.as_ref().ptr as *const u8).add(t.offset_bytes);
                let dst = (output.block.as_ref().ptr as *mut u8).add(write_offset_bytes);
                std::ptr::copy_nonoverlapping(src, dst, size_bytes);
            }
            write_offset_bytes += size_bytes;
        }
        Ok(output)
    }

    fn neg(&self) -> Result<Self> {
        let n = self.shape.numel();
        let output = Self::zeros(&self.shape, self.dtype, &self.device)?;
        unsafe {
            match self.dtype {
                DType::F32 => {
                    let src =
                        (self.block.as_ref().ptr as *const u8).add(self.offset_bytes) as *const f32;
                    let dst =
                        (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes) as *mut f32;
                    for i in 0..n {
                        dst.add(i).write(-src.add(i).read());
                    }
                }
                DType::F16 => {
                    let src = (self.block.as_ref().ptr as *const u8).add(self.offset_bytes)
                        as *const half::f16;
                    let dst = (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes)
                        as *mut half::f16;
                    for i in 0..n {
                        dst.add(i)
                            .write(half::f16::from_f32(-src.add(i).read().to_f32()));
                    }
                }
                _ => return Err(CoreError::Internal("neg: unsupported dtype".into())),
            }
        }
        Ok(output)
    }

    fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let current_shape = self.shape.dims();
        if start + len > current_shape[dim] {
            return Err(CoreError::OutOfBounds {
                op: "narrow",
                index: start + len,
                size: current_shape[dim],
            });
        }

        let element_size = self.dtype.size_in_bytes();
        let stride = self.strides[dim];
        let byte_offset = start * stride * element_size;

        let mut new_dims = current_shape.to_vec();
        new_dims[dim] = len;
        let new_shape = Shape::new(&new_dims);

        Ok(Self {
            state: self.state.clone(),
            block: self.block.clone(),
            shape: new_shape,
            strides: self.strides.clone(),
            offset_bytes: self.offset_bytes + byte_offset,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn broadcast_mul(&self, other: &Self) -> Result<Self> {
        let n = self.shape.numel();
        let bias_len = other.shape.numel();
        let output = Self::zeros(&self.shape, self.dtype, &self.device)?;
        unsafe {
            let a = (self.block.as_ref().ptr as *const u8).add(self.offset_bytes) as *const f32;
            let b = (other.block.as_ref().ptr as *const u8).add(other.offset_bytes) as *const f32;
            let c = (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes) as *mut f32;
            for i in 0..n {
                c.add(i).write(a.add(i).read() * b.add(i % bias_len).read());
            }
        }
        Ok(output)
    }

    fn repeat(&self, repeats: &Shape) -> Result<Self> {
        let out_dims: Vec<usize> = self
            .shape
            .dims()
            .iter()
            .zip(repeats.dims().iter())
            .map(|(&d, &r)| d * r)
            .collect();
        let out_shape = Shape::new(&out_dims);
        let src = self.to_vec_f32()?;
        let out_strides = MetalTensor::compute_strides(&out_dims);
        let in_strides = MetalTensor::compute_strides(self.shape.dims());
        let n = out_shape.numel();
        let mut dst = vec![0.0f32; n];

        for i in 0..n {
            let mut remaining = i;
            let mut src_idx = 0usize;
            for d in 0..out_dims.len() {
                let coord = remaining / out_strides[d];
                remaining %= out_strides[d];
                let src_coord = coord % self.shape.dims()[d];
                src_idx += src_coord * in_strides[d];
            }
            dst[i] = src[src_idx];
        }

        let out = Self::from_slice(&dst, &out_shape, &self.device)?;
        out.to_dtype(self.dtype)
    }

    fn sigmoid(&self) -> Result<Self> {
        let n = self.shape.numel();
        let output = Self::zeros(&self.shape, self.dtype, &self.device)?;
        unsafe {
            match self.dtype {
                DType::F32 => {
                    let src =
                        (self.block.as_ref().ptr as *const u8).add(self.offset_bytes) as *const f32;
                    let dst =
                        (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes) as *mut f32;
                    for i in 0..n {
                        let x = src.add(i).read();
                        dst.add(i).write(1.0 / (1.0 + (-x).exp()));
                    }
                }
                DType::F16 => {
                    let src = (self.block.as_ref().ptr as *const u8).add(self.offset_bytes)
                        as *const half::f16;
                    let dst = (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes)
                        as *mut half::f16;
                    for i in 0..n {
                        let x = src.add(i).read().to_f32();
                        dst.add(i)
                            .write(half::f16::from_f32(1.0 / (1.0 + (-x).exp())));
                    }
                }
                _ => return Err(CoreError::Internal("sigmoid: unsupported dtype".into())),
            }
        }
        Ok(output)
    }

    fn to_vec_u32(&self) -> Result<Vec<u32>> {
        let n = self.shape.numel();
        let mut out = vec![0u32; n];
        let dims = self.shape.dims();
        let rank = dims.len();

        let expected_strides = Self::compute_strides(dims);
        let is_contiguous = self.strides == expected_strides;

        unsafe {
            let base_ptr =
                (self.block.as_ref().ptr as *const u8).add(self.offset_bytes) as *const u32;

            if is_contiguous {
                std::ptr::copy_nonoverlapping(base_ptr, out.as_mut_ptr(), n);
            } else {
                for i in 0..n {
                    let mut physical_idx = 0;
                    let mut linear_idx = i;
                    for d in (0..rank).rev() {
                        let coord = linear_idx % dims[d];
                        linear_idx /= dims[d];
                        physical_idx += coord * self.strides[d];
                    }
                    out[i] = base_ptr.add(physical_idx).read();
                }
            }
        }
        Ok(out)
    }

    fn zeros_like(&self) -> Result<Self> {
        Self::zeros(&self.shape, self.dtype, &self.device)
    }

    fn sum_keepdim(&self, dim: usize) -> Result<Self> {
        let summed = self.sum(dim)?;
        let mut dims = self.shape.dims().to_vec();
        dims[dim] = 1;
        summed.reshape(&Shape::new(&dims))
    }

    fn broadcast_div(&self, other: &Self) -> Result<Self> {
        let n = self.shape.numel();
        let bias_len = other.shape.numel();
        let output = Self::zeros(&self.shape, self.dtype, &self.device)?;
        unsafe {
            let a = (self.block.as_ref().ptr as *const u8).add(self.offset_bytes) as *const f32;
            let b = (other.block.as_ref().ptr as *const u8).add(other.offset_bytes) as *const f32;
            let c = (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes) as *mut f32;
            for i in 0..n {
                c.add(i).write(a.add(i).read() / b.add(i % bias_len).read());
            }
        }
        Ok(output)
    }

    fn to_vec_f32(&self) -> Result<Vec<f32>> {
        if let Some(state) = crate::metal::state::global_metal_state() {
            let cmd_buf = state.ctx.command_buffer()?;
            cmd_buf.commit(); // 1. Flush all previous work to the GPU
            cmd_buf.waitUntilCompleted(); // 2. Safely wait for it all to finish
        }
        let n = self.shape.numel();
        let mut out = vec![0.0f32; n];
        let dims = self.shape.dims();
        let rank = dims.len();

        let expected_strides = Self::compute_strides(dims);
        let is_contiguous = self.strides == expected_strides;

        unsafe {
            match self.dtype {
                DType::F32 => {
                    let base_ptr =
                        (self.block.as_ref().ptr as *const u8).add(self.offset_bytes) as *const f32;

                    if is_contiguous {
                        // Fast path: physical memory matches logical shape
                        std::ptr::copy_nonoverlapping(base_ptr, out.as_mut_ptr(), n);
                    } else {
                        // Slow path: strided physical memory read
                        for i in 0..n {
                            let mut physical_idx = 0;
                            let mut linear_idx = i;
                            for d in (0..rank).rev() {
                                let coord = linear_idx % dims[d];
                                linear_idx /= dims[d];
                                physical_idx += coord * self.strides[d];
                            }
                            out[i] = base_ptr.add(physical_idx).read();
                        }
                    }
                }
                DType::F16 => {
                    let base_ptr = (self.block.as_ref().ptr as *const u8).add(self.offset_bytes)
                        as *const half::f16;

                    if is_contiguous {
                        for i in 0..n {
                            out[i] = base_ptr.add(i).read().to_f32();
                        }
                    } else {
                        for i in 0..n {
                            let mut physical_idx = 0;
                            let mut linear_idx = i;
                            for d in (0..rank).rev() {
                                let coord = linear_idx % dims[d];
                                linear_idx /= dims[d];
                                physical_idx += coord * self.strides[d];
                            }
                            out[i] = base_ptr.add(physical_idx).read().to_f32();
                        }
                    }
                }
                _ => return Err(CoreError::Internal("to_vec_f32: unsupported dtype".into())),
            }
        }
        Ok(out)
    }

    fn exp(&self) -> Result<Self> {
        let n = self.shape.numel();
        let output = Self::zeros(&self.shape, self.dtype, &self.device)?;
        unsafe {
            match self.dtype {
                DType::F32 => {
                    let src =
                        (self.block.as_ref().ptr as *const u8).add(self.offset_bytes) as *const f32;
                    let dst =
                        (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes) as *mut f32;
                    for i in 0..n {
                        dst.add(i).write(src.add(i).read().exp());
                    }
                }
                DType::F16 => {
                    let src = (self.block.as_ref().ptr as *const u8).add(self.offset_bytes)
                        as *const half::f16;
                    let dst = (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes)
                        as *mut half::f16;
                    for i in 0..n {
                        dst.add(i)
                            .write(half::f16::from_f32(src.add(i).read().to_f32().exp()));
                    }
                }
                _ => return Err(CoreError::Internal("exp: unsupported dtype".into())),
            }
        }
        Ok(output)
    }

    fn log(&self) -> Result<Self> {
        let n = self.shape.numel();
        let output = Self::zeros(&self.shape, self.dtype, &self.device)?;
        unsafe {
            match self.dtype {
                DType::F32 => {
                    let src =
                        (self.block.as_ref().ptr as *const u8).add(self.offset_bytes) as *const f32;
                    let dst =
                        (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes) as *mut f32;
                    for i in 0..n {
                        dst.add(i).write(src.add(i).read().ln());
                    }
                }
                DType::F16 => {
                    let src = (self.block.as_ref().ptr as *const u8).add(self.offset_bytes)
                        as *const half::f16;
                    let dst = (output.block.as_ref().ptr as *mut u8).add(output.offset_bytes)
                        as *mut half::f16;
                    for i in 0..n {
                        dst.add(i)
                            .write(half::f16::from_f32(src.add(i).read().to_f32().ln()));
                    }
                }
                _ => return Err(CoreError::Internal("log: unsupported dtype".into())),
            }
        }
        Ok(output)
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
