use cudarc::driver::DriverError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaError {
    #[error("cudarc driver error: {0}")]
    Driver(#[from] DriverError),

    #[error("no CUDA device at ordinal {0}")]
    NoDevice(usize),

    #[error("kernel not loaded: {0}")]
    KernelNotLoaded(&'static str),

    #[error("invalid launch config: {0}")]
    InvalidConfig(String),

    #[error("block allocator out of memory: requested {requested} blocks, {available} available")]
    OutOfBlocks { requested: usize, available: usize },

    #[error("invalid block index: {0}")]
    InvalidBlock(usize),

    #[error("shape mismatch in {op}: {detail}")]
    ShapeMismatch { op: &'static str, detail: String },

    #[error("internal cuda error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, CudaError>;
