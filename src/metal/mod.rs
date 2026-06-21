// src/metal/mod.rs
//
// The metal module is gated behind the `metal` feature flag.
// Add to Cargo.toml:
//
// [features]
// metal = ["dep:objc2-metal", "dep:objc2-foundation", "dep:objc2"]
//
// [dependencies]
// objc2-metal      = { version = "0.2", optional = true }
// objc2-foundation = { version = "0.2", optional = true }
// objc2            = { version = "0.5", optional = true }
//
// Unlike the cuda module, there is no PTX-style precompile step here yet —
// Metal Shading Language (.metal) source is compiled to a metallib either at
// build time (xcrun metal / metallib) or at runtime via
// MTLDevice::newLibraryWithSource. Once kernel work starts, this mod.rs
// should document which strategy axiom uses and why (build-time is faster
// at startup and catches MSL compile errors in CI; runtime compile is
// faster to iterate on during kernel development).
//
// Why this is a separate module from src/cuda, not a parallel impl of the
// same traits: axiom's CUDA kernels (FA3/FA4, ForkKV's dual-cache attention,
// MoE speculative pre-gating) lean on Hopper/Ada-specific primitives with no
// Metal equivalent — TMA async bulk copy, cuda::pipeline double-buffering,
// warp-level cooperative groups sized and scheduled the CUDA way. The Metal
// kernels reimplement the same *algorithms* (documented in the top-level
// README's CUDA Kernels section) against Metal's own primitives:
// MTLComputeCommandEncoder, threadgroup memory, SIMD-groups, and Apple
// Silicon's unified memory (which removes the explicit host/device copy
// step CUDA requires — see core::tensor::MetalTensor's doc comment).
//
// Planned shape, mirroring src/cuda's module layout:
//   device.rs    — MTLDevice acquisition, capability queries
//   context.rs   — command queue / command buffer lifecycle (cuda::context analog)
//   allocator.rs — MTLBuffer-backed paged block allocator (unified memory —
//                  no separate host/device pool the way CUDA's allocator needs)
//   kernels.rs   — kernel dispatch wrappers (cuda::kernels analog), starting
//                  with the foundation kernels (matmul, RMSNorm, RoPE, SwiGLU)
//                  before attention and MoE routing/dispatch
//   error.rs     — MetalError, mirroring CudaError's shape
//
// Typical usage (once implemented), mirroring cuda::context's shape:
//
//   let ctx = MetalContext::new(0)?;
//   let alloc = PagedBlockAllocator::new(&ctx, num_blocks, block_size, ...)?;
//   launch_rms_norm_f16(&ctx, &mut output, &input, &weight, eps, tokens, hidden)?;
//   ctx.synchronize()?;

#[cfg(feature = "metal")]
pub mod error;

#[cfg(feature = "metal")]
pub use error::MetalError;

// device.rs, context.rs, allocator.rs, kernels.rs land with the first real
// Metal kernel work (see core::backend::MetalTensor / MetalBackend stubs).
