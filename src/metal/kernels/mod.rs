pub mod rms_norm_f16;
pub mod rope_f16;

pub use rms_norm_f16::RmsNormKernel;
pub use rope_f16::RopeKernel;

use crate::metal::context::MetalContext;
use crate::metal::error::Result;

pub struct MetalKernels {
    pub rms_norm: RmsNormKernel,
    pub rope: RopeKernel,
}

impl MetalKernels {
    pub fn new(ctx: &MetalContext) -> Result<Self> {
        let device = ctx.device.raw();

        Ok(Self {
            rms_norm: RmsNormKernel::new(&device)?,
            rope: RopeKernel::new(&device)?,
        })
    }
}
