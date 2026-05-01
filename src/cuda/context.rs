use cudarc::driver::{CudaContext as CudaDevice, CudaFunction, CudaStream};
use cudarc::driver::{CudaSlice, CudaView, CudaViewMut, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::sync::Arc;

/*  CudaContext
// Owns the device handle, a dedicated inference stream, and a cache of
// loaded kernel functions. One CudaContext per GPU.
// The kernel function cache avoids re-loading PTX on every call.
// Functions are looked up by name after the PTX module is loaded once
// at construction time.
// Important cudarc notes:
//   - CudaDevice::load_ptx takes (Ptx, module_name: &str, fn_names: &[&str])
//   - The fn_names slice must exactly match extern "C" kernel names in the .cu
//   - CudaDevice::get_func(module_name, fn_name) retrieves a loaded function
//   - CudaFunction is Clone — cheap to hand out
*/

//Names of every kernel function exposed by the kernels crate.
//These must match the extern "C" names in the .cu files exactly.
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
    pub device: Arc<CudaDevice>,
    pub stream: CudaStream,
    funcs: HashMap<&'static str, CudaFunction>,
    ordinal: usize,
}

impl AxiomContext {
    //Create a context for device `ordinal`, load all kernels from the
    //embedded PTX string, and cache their function handles.
    pub fn new(ordinal: usize, ptx_src: &str) -> Result<Self> {
        let device = CudaDevice::new(ordinal).map_err(CudaError::Driver)?;
        let device = Arc::new(device);

        let stream = device.fork_default_stream().map_err(CudaError::Driver)?;

        let ptx = cudarc::driver::Ptx::from_src(ptx_src);
        device
            .load_ptx(ptx, MODULE_NAME, KERNEL_NAMES)
            .map_err(CudaError::Driver)?;

        let mut funcs = HashMap::new();
        for &name in KERNEL_NAMES {
            let f = device
                .get_func(MODULE_NAME, name)
                .ok_or(CudaError::KernelNotLoaded(name))?;
            funcs.insert(name, f);
        }

        Ok(Self {
            device,
            stream,
            funcs,
            ordinal,
        })
    }

    //Retrieve a kernel function by name.
    pub fn func(&self, name: &'static str) -> Result<CudaFunction> {
        self.funcs
            .get(name)
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

    //Synchronise the inference stream — blocks until all pending GPU work
    //on this context's stream has completed.
    //Call before reading device->host results.
    pub fn synchronize(&self) -> Result<()> {
        self.device.synchronize().map_err(CudaError::Driver)
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
            assert!(ctx.func(name).is_ok(), "failed to load kernel: {}", name);
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

    #[test]
    fn test_context_no_device() {
        let result = CudaContext::new(99, ".version 7.0\n.target sm_80\n.address_size 64\n");
        assert!(result.is_err());
    }
}
