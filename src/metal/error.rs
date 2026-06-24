use thiserror::Error;

// Mirrors cuda::error::CudaError's shape. The Driver(#[from] ...) variant has
// no direct analog yet — once device.rs/context.rs land and we know which
// objc2-metal error types actually propagate (MTLDevice/MTLLibrary/
// MTLCommandBuffer calls mostly return Option or set an NSError out-param
// rather than a uniform Result type the way cudarc does), add a #[from]
// variant wrapping that here rather than stringly-typing it via Internal.

#[derive(Error, Debug)]
pub enum MetalError {
    #[error("no Metal device at ordinal {0}")]
    NoDevice(usize),

    #[error("failed to create Metal command queue")]
    NoCommandQueue,

    #[error("failed to create Metal command buffer")]
    NoCommandBuffer,

    #[error("kernel not loaded: {0}")]
    KernelNotLoaded(&'static str),

    #[error("MSL compilation failed: {0}")]
    LibraryCompilation(String),

    #[error("invalid dispatch config: {0}")]
    InvalidConfig(String),

    #[error("block allocator out of memory: requested {requested} blocks, {available} available")]
    OutOfBlocks { requested: usize, available: usize },

    #[error("invalid block index: {0}")]
    InvalidBlock(usize),

    #[error("shape mismatch in {op}: {detail}")]
    ShapeMismatch { op: &'static str, detail: String },

    #[error("internal metal error: {0}")]
    Internal(String),

    #[error("session not found: {0}")]
    InvalidSession(u64),

    #[error("Metal buffer allocation failed")]
    AllocationFailed,

    #[error("allocator out of memory: requested {requested} bytes, {available} available")]
    OutOfMemory { requested: usize, available: usize },

    #[error("cannot build pipeline state")]
    PipelineStateInvalid,
}

pub type Result<T> = std::result::Result<T, MetalError>;
