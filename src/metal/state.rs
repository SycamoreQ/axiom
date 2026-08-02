use std::sync::{Arc, Mutex, OnceLock};
static METAL_STATE: OnceLock<Arc<MetalState>> = OnceLock::new();
use crate::core::error::{CoreError, Result};
use crate::metal::allocator::MetalAllocator;
use crate::metal::context::MetalContext;
use crate::metal::MetalDevice;
use crate::metal::MetalKernels;

#[derive(Debug)]
pub struct MetalState {
    pub ctx: MetalContext,
    pub alloc: MetalAllocator, // Mutex because alloc is &mut for alloc/free
    pub kernels: MetalKernels,
}

unsafe impl Send for MetalState {}
unsafe impl Sync for MetalState {}

pub fn init_global_metal_state(pool_size: usize) -> Result<Arc<MetalState>> {
    let state = Arc::new(MetalState::new(pool_size)?);
    METAL_STATE.set(state.clone()).ok();
    Ok(state)
}

pub fn global_metal_state() -> Option<Arc<MetalState>> {
    METAL_STATE.get().cloned()
}

impl MetalState {
    pub fn new(pool_size_bytes: usize) -> Result<Self> {
        let device = MetalDevice::system_default()?;
        let ctx = MetalContext::new(device)?;
        let alloc = MetalAllocator::new(&ctx, pool_size_bytes)?;
        let kernels = MetalKernels::new(ctx.device.raw())?;
        Ok(Self {
            ctx,
            alloc: alloc,
            kernels,
        })
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;

    #[test]
    fn test_metal_state_initialization() {
        let pool_size = 1024 * 1024;
        let state = init_global_metal_state(pool_size).unwrap();
        assert!(global_metal_state().is_some());
    }

    #[test]
    fn test_allocator_direct() {
        let state = global_metal_state().unwrap();
        let block = state.alloc.alloc(128, 16).unwrap();
        assert!(!block.ptr.is_null());
    }
}
