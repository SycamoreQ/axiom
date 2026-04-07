use crate::core::device::Device;
use crate::core::dtype::DType;
use crate::core::error::{CoreError, Result};
use crate::core::shape::Shape;
use crate::core::tensor::TensorOps;
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

impl Backend for CudarcBackend {
    type Tensor = CudarcTensor;
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

impl Backend for CandleBackend {
    type Tensor = CandleTensor;
    type Device = Device;
    type Error = CoreError;
}
