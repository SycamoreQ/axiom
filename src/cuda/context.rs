use std::collections::HashMap;
use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaFunction, CudaStream};
use crate::cuda::error::{CudaError, Result};

// =============================================================================
// CudaContext
//
// Owns the device handle, a dedicated inference stream, and a cache of
// loaded kernel functions. One CudaContext per GPU.
//
// The kernel function cache avoids re-loading PTX on every call.
// Functions are looked up by name after the PTX module is loaded once
// at construction time.
//
// Important cudarc notes:
//   - CudaDevice::load_ptx takes (Ptx, module_name: &str, fn_names: &[&str])
//   - The fn_names slice must exactly match extern "C" kernel names in the .cu
//   - CudaDevice::get_func(module_name, fn_name) retrieves a loaded function
//   - CudaFunction is Clone — cheap to hand out
// =============================================================================

/// Names of every kernel function exposed by the kernels crate.
/// These must match the extern "C" names in the .cu files exactly.
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

pub const MODULE_NAME: &str = "axiom_kernels";

pub struct CudaContext {
    pub device:  Arc<CudaDevice>,
    pub stream:  CudaStream,
    funcs:       HashMap<&'static str, CudaFunction>,
    ordinal:     usize,
}

impl CudaContext {
    /// Create a context for device `ordinal`, load all kernels from the
    /// embedded PTX string, and cache their function handles.
    ///
    /// `ptx_src` is the PTX text produced by compiling the kernels crate.
    /// Typically: include_str!(concat!(env!("OUT_DIR"), "/axiom_kernels.ptx"))
    ///
    /// Hint: cudarc::driver::Ptx::from_src(ptx_src) wraps a &str into a Ptx.
    /// Then device.load_ptx(ptx, MODULE_NAME, KERNEL_NAMES) loads the module.
    /// Then device.get_func(MODULE_NAME, name) for each name in KERNEL_NAMES.
    pub fn new(ordinal: usize, ptx_src: &str) -> Result<Self> {
        let device = CudaDevice::new(ordinal).map_err(CudaError::Driver)?;
        let device = Arc::new(device);

        let stream = device.fork_default_stream().map_err(CudaError::Driver)?;

        let ptx = cudarc::driver::Ptx::from_src(ptx_src);
        device.load_ptx(ptx, MODULE_NAME, KERNEL_NAMES)
            .map_err(CudaError::Driver)?;

        let mut funcs = HashMap::new();
        for &name in KERNEL_NAMES {
            let f = device.get_func(MODULE_NAME, name)
                .ok_or(CudaError::KernelNotLoaded(name))?;
            funcs.insert(name, f);
        }

        Ok(Self { device, stream, funcs, ordinal })
    }

    /// Retrieve a kernel function by name.
    /// Panics if the name is not in KERNEL_NAMES — this is a programmer error.
    pub fn func(&self, name: &'static str) -> Result<CudaFunction> {
        self.funcs.get(name)
            .cloned()
            .ok_or(CudaError::KernelNotLoaded(name))
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Synchronise the inference stream — blocks until all pending GPU work
    /// on this context's stream has completed.
    /// Call before reading device->host results.
    pub fn synchronize(&self) -> Result<()> {
        self.device.synchronize().map_err(CudaError::Driver)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_ptx() -> &'static str {
        // Minimal valid PTX that defines no kernels.
        // Real tests would embed the compiled axiom_kernels.ptx.
        ".version 7.0\n.target sm_89\n.address_size 64\n"
    }

    #[test]
    fn test_context_no_device() {
        // Ordinal 99 should not exist on any machine.
        let result = CudaContext::new(99, dummy_ptx());
        assert!(result.is_err());
    }

    #[test]
    fn test_func_unknown_name() {
        // Without a real device we can only test the lookup logic.
        // With a device, load_ptx would succeed first.
        // This test verifies the HashMap path.
        let ctx = CudaContext {
            device: {
                // Skip if no GPU
                let Ok(d) = CudaDevice::new(0) else { return };
                Arc::new(d)
            },
            stream: {
                let Ok(d) = CudaDevice::new(0) else { return };
                let Ok(s) = Arc::new(d).fork_default_stream() else { return };
                s
            },
            funcs: HashMap::new(),
            ordinal: 0,
        };
        assert!(ctx.func("nonexistent_kernel").is_err());
    }
}
