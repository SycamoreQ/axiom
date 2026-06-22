use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use crate::metal::error;

pub struct MetalDevice {
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
}

impl MetalDevice {
    pub fn system_default() -> Result<Self, MetalError> {
        let raw = objc_metal::MTLCreateSystemDefaultDevice();
        if raw.is_null() {
            // 2. Return your custom error variant
            return Err(MetalError::NoDevice(0));
        }

        Ok(Self { raw })
    }

    pub fn name(&self) -> String {
        self.raw.name().to_string()
    }
    pub fn recommended_max_working_set_size(&self) -> u64 {
        self.raw.recommendMaxWorkingSetSize()
    }
    pub fn has_unified_memory(&self) -> bool {
        self.raw.hasUnifiedMemory()
    }
}
