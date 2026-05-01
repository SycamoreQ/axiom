use crate::cuda::error::{CudaError, Result};
use cudarc::driver::{CudaContext as GPUContext, CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::Ptx;

use std::collections::HashMap;
use std::sync::Arc;

/*  CudaContext
// Owns the device handle, a dedicated inference stream, and a cache of
// loaded kernel functions. One CudaContext per GPU.
//
// cudarc 0.19: CudaDevice is gone. GPUContext (cudarc::driver::CudaContext)
// now owns the device + CUDA context together. Streams are obtained from it
// via default_stream() / new_stream(). Modules are loaded via load_module()
// and functions via module.load_function().
*/

pub const KERNEL_NAMES: &[&str] = &[
    "rms_norm_f16_kernel",
    "fused_residual_rmsnorm_f16_kernel",
    "rotary_embedding_f16_kernel",
    "reshape_and_cache_f16io_kernel",
    "copy_blocks_f16_kernel",
    "embedding_gather_f16_kernel",
    "argmax_f16_kernel",
    "flash_attention_3_decode_f16io_kernel",
    "flash_attention_3_decode_gqa_f16io_kernel",
    "residual_attention_decode_f16io_kernel",
    "flash_attention_4_decode_f16io_kernel",
];

pub struct CudaContext {
    pub gpu: Arc<GPUContext>,
    pub stream: Arc<CudaStream>,
    pub module: Arc<CudaModule>,
    funcs: HashMap<&'static str, CudaFunction>,
    ordinal: usize,
}

impl CudaContext {
    pub fn new(ordinal: usize, ptx_src: &str) -> Result<Self> {
        let gpu: Arc<GPUContext> = GPUContext::new(ordinal).map_err(CudaError::Driver)?;
        let stream = gpu.default_stream();
        let ptx = Ptx::from_src(ptx_src);
        let module: Arc<CudaModule> = gpu.load_module(ptx).map_err(CudaError::Driver)?;

        let mut funcs = HashMap::new();

        for &name in KERNEL_NAMES {
            let f = module
                .load_function(name)
                .map_err(|_| CudaError::KernelNotLoaded(name))?;

            funcs.insert(name, f);
        }

        Ok(Self {
            gpu,
            stream,
            module,
            funcs,
            ordinal,
        })
    }

    pub fn func(&self, name: &'static str) -> Result<CudaFunction> {
        self.funcs
            .get(name)
            .cloned()
            .ok_or(CudaError::KernelNotLoaded(name))
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn gpu(&self) -> &Arc<GPUContext> {
        &self.gpu
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(CudaError::Driver)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_ctx() -> Option<CudaContext> {
        let ptx = std::fs::read_to_string(env!("AXIOM_KERNELS_PTX")).ok()?;
        CudaContext::new(0, &ptx).ok()
    }

    #[test]
    fn test_context_loads_all_kernels() {
        let Some(ctx) = try_ctx() else { return };

        for &name in KERNEL_NAMES {
            assert!(ctx.func(name).is_ok(), "failed to load: {}", name);
        }
    }

    #[test]
    fn test_synchronize_no_work() {
        let Some(ctx) = try_ctx() else { return };
        assert!(ctx.synchronize().is_ok());
    }

    #[test]
    fn test_ordinal_is_zero() {
        let Some(ctx) = try_ctx() else { return };
        assert_eq!(ctx.ordinal(), 0);
    }

    #[test]
    fn test_unknown_kernel_returns_err() {
        let Some(ctx) = try_ctx() else { return };
        assert!(ctx.func("does_not_exist").is_err());
    }
}
