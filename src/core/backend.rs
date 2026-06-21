use crate::core::device::Device;
use crate::core::dtype::DType;
use crate::core::error::{CoreError, Result};
use crate::core::shape::Shape;
use crate::core::tensor::TensorOps;
use crate::core::tensor::{TopKLastDimOp, TopKOutput};
use candle_core;

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

// todo(), its a stub for until cuda kernels come
#[derive(Debug, Clone)]
pub struct CudarcTensor {
    // stub for Phase 3 — cudarc tensors live here
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
}

// todo(), its a stub until hand-written Metal kernels come.
// Mirrors CudarcTensor's shape — a separate sibling type, not a CandleTensor variant,
// so MetalBackend never has to route through candle_core at all (unified memory means
// alloc/copy semantics here will look quite different from both CandleTensor and CudarcTensor).
#[derive(Debug, Clone)]
pub struct MetalTensor {
    // stub — actual MTLBuffer-backed storage lands with the first Metal kernel work
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
}

impl Backend for CudarcBackend {
    type Tensor = CudarcTensor;
    type Device = Device;
    type Error = CoreError;
}

impl Backend for MetalBackend {
    type Tensor = MetalTensor;
    type Device = Device;
    type Error = CoreError;
}

pub trait Backend: Clone + Send + Sync + 'static {
    type Tensor: TensorOps + TopKLastDimOp + Clone + Send + Sync;
    type Device: Into<Device> + Clone + Send + Sync;
    type Error: std::error::Error + Send + Sync + From<CoreError>;
}

// two compile time tags
#[derive(Debug, Clone, Copy)]
pub struct CandleBackend;

#[derive(Debug, Clone, Copy)]
pub struct CudarcBackend;

#[derive(Debug, Clone, Copy)]
pub struct MetalBackend;

impl Backend for CandleBackend {
    type Tensor = CandleTensor;
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

        // extract dims BEFORE moving into struct
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

// MoE routing (top-k expert selection) lives on the critical path for the Metal MoE
// dispatch work — this stub is one of the first real kernels to implement, not an
// afterthought. CandleTensor's impl (sort + narrow + gather) is a reasonable reference
// for *what* to compute; the Metal version will likely want a fused threadgroup top-k
// rather than three separate passes.
impl TopKLastDimOp for MetalTensor {
    fn topk(&self, _k: usize) -> Result<TopKOutput<Self>> {
        todo!("Metal Phase 1")
    }
}
