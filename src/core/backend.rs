use crate::core::device::Device;
use crate::core::dtype::DType;
use crate::core::error::{CoreError, Result};
use crate::core::shape::Shape;
use crate::core::tensor::TensorOps;
use crate::core::tensor::{TopKLastDimOp, TopKOutput};
#[cfg(feature = "metal")]
use crate::metal::allocator::BlockHandle;
#[cfg(feature = "metal")]
use crate::metal::state::MetalState;

use candle_core;
use std::sync::Arc;

/*
defines the compile-time abstraction that ties everything together.
 */

#[derive(Debug, Clone)]
pub struct CandleTensor {
    pub(crate) inner: candle_core::Tensor,
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
}

#[derive(Debug, Clone)]
pub struct CudarcTensor {
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
}

#[cfg(feature = "metal")]
#[derive(Clone)]
pub struct MetalTensor {
    pub(crate) state: Arc<MetalState>,
    pub(crate) block: Arc<BlockHandle>,
    pub(crate) shape: Shape,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset_bytes: usize,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
}

pub trait Backend: Clone + Send + Sync + 'static {
    type Tensor: TensorOps + TopKLastDimOp + Clone + Send + Sync;
    type Device: Into<Device> + Clone + Send + Sync;
    type Error: std::error::Error + Send + Sync + From<CoreError>;
}

#[derive(Debug, Clone, Copy)]
pub struct CandleBackend;

#[derive(Debug, Clone, Copy)]
pub struct CudarcBackend;

#[cfg(feature = "metal")]
#[derive(Debug, Clone)]
pub struct MetalBackend {
    pub state: Arc<MetalState>,
}

impl Backend for CandleBackend {
    type Tensor = CandleTensor;
    type Device = Device;
    type Error = CoreError;
}

impl Backend for CudarcBackend {
    type Tensor = CudarcTensor;
    type Device = Device;
    type Error = CoreError;
}

#[cfg(feature = "metal")]
impl Backend for MetalBackend {
    type Tensor = MetalTensor;
    type Device = Device;
    type Error = CoreError;
}

impl TopKLastDimOp for CandleTensor {
    fn topk(&self, k: usize) -> Result<TopKOutput<Self>> {
        let sorted_indices = self.inner.arg_sort_last_dim(false)?;
        let topk_indices = sorted_indices
            .narrow(candle_core::D::Minus1, 0, k)?
            .contiguous()?;
        let values = self.inner.gather(&topk_indices, candle_core::D::Minus1)?;

        let values_dims = values.dims().to_vec();
        let indices_dims = topk_indices.dims().to_vec();

        Ok(TopKOutput {
            values: CandleTensor {
                inner: values,
                shape: Shape::new(&values_dims),
                dtype: self.dtype,
                device: self.device.clone(),
            },
            indices: CandleTensor {
                inner: topk_indices,
                shape: Shape::new(&indices_dims),
                dtype: DType::U32,
                device: self.device.clone(),
            },
        })
    }
}

impl TopKLastDimOp for CudarcTensor {
    fn topk(&self, _k: usize) -> Result<TopKOutput<Self>> {
        todo!("Phase 4")
    }
}

#[cfg(feature = "metal")]
impl TopKLastDimOp for MetalTensor {
    fn topk(&self, k: usize) -> Result<TopKOutput<Self>> {
        let dims = self.shape().dims().to_vec();
        let last_dim = *dims
            .last()
            .ok_or_else(|| CoreError::Internal("topk: input tensor must have rank >= 1".into()))?;
        if k > last_dim {
            return Err(CoreError::Internal(format!(
                "topk: k={} exceeds last dimension size {}",
                k, last_dim
            )));
        }

        let num_rows = self.shape().numel() / last_dim;
        let flat = self.to_vec_f32()?;

        let mut values = Vec::with_capacity(num_rows * k);
        let mut indices = Vec::with_capacity(num_rows * k);

        for row in 0..num_rows {
            let row_start = row * last_dim;
            let row_slice = &flat[row_start..row_start + last_dim];

            let mut order: Vec<usize> = (0..last_dim).collect();
            // Descending by value -- matches CandleTensor's arg_sort_last_dim(false).
            order.sort_by(|&a, &b| row_slice[b].total_cmp(&row_slice[a]));

            for &idx in order.iter().take(k) {
                values.push(row_slice[idx]);
                indices.push(idx as u32);
            }
        }

        let mut out_dims = dims.clone();
        *out_dims.last_mut().unwrap() = k;
        let out_shape = Shape::new(&out_dims);

        Ok(TopKOutput {
            values: Self::from_slice(&values, &out_shape, self.device())?,
            indices: Self::from_u32_slice(&indices, &out_shape, self.device())?,
        })
    }
}
