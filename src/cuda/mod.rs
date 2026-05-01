// src/cuda/mod.rs
//
// The cuda module is gated behind the `cuda` feature flag.
// Add to Cargo.toml:
//
// [features]
// cuda = ["cudarc", "half"]
//
// [dependencies]
// cudarc = { version = "0.19", features = ["cuda-version-from-build-system"], optional = true }
// half   = { version = "2",    features = ["num-traits"],                      optional = true }
//
// The kernels crate (in /kernel) must be compiled to PTX before this module
// can be used. The build.rs in the kernels crate handles this when the
// `cuda` feature is enabled.
//
// cudarc 0.19 notes:
//   - CudaDevice is gone. cudarc::driver::CudaContext (aliased as GPUContext
//     inside context.rs) now owns both the device and the CUDA context.
//   - Memory allocation moves from device.alloc_zeros() to stream.alloc_zeros().
//   - Host-to-device copies use stream.clone_htod() instead of htod_sync_copy().
//   - Modules loaded via ctx.load_module(ptx) -> Arc<CudaModule>.
//   - Functions loaded via module.load_function(name) -> CudaFunction.
//   - Kernel launch uses stream.launch_builder(&f).arg(...).launch(cfg).
//
// Typical usage:
//
//   let ctx = CudaContext::new(0, include_str!(...axiom_kernels.ptx...))?;
//   let alloc = PagedBlockAllocator::new(&ctx, num_blocks, block_size, ...)?;
//   launch_rms_norm_f16(&ctx, &mut output, &input, &weight, eps, tokens, hidden)?;
//   ctx.synchronize()?;

#[cfg(feature = "cuda")]
pub mod allocator;
#[cfg(feature = "cuda")]
pub mod context;
#[cfg(feature = "cuda")]
pub mod error;
#[cfg(feature = "cuda")]
pub mod fork_manager;
#[cfg(feature = "cuda")]
pub mod kernels;
#[cfg(feature = "cuda")]
pub mod paged_session;

#[cfg(feature = "cuda")]
pub use allocator::{BlockId, BlockTable, PagedBlockAllocator};
#[cfg(feature = "cuda")]
pub use context::CudaContext;
#[cfg(feature = "cuda")]
pub use error::CudaError;
#[cfg(feature = "cuda")]
pub use kernels::{
    launch_argmax_f16, launch_copy_blocks_f16, launch_embedding_gather_f16,
    launch_flash_attention_3, launch_flash_attention_3_gqa, launch_flash_attention_4,
    launch_fused_residual_rmsnorm_f16, launch_reshape_and_cache_f16, launch_residual_attention,
    launch_rms_norm_f16, launch_rotary_embedding_f16,
};
