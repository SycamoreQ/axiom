use crate::metal::error::{MetalError, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

pub struct MetalDevice {
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
}

impl MetalDevice {
    pub fn system_default() -> Result<Self> {
        let raw = objc2_metal::MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice(0))?;

        Ok(Self { raw })
    }

    pub fn name(&self) -> String {
        self.raw.name().to_string()
    }
    pub fn recommended_max_working_set_size(&self) -> u64 {
        self.raw.recommendedMaxWorkingSetSize()
    }
    pub fn has_unified_memory(&self) -> bool {
        self.raw.hasUnifiedMemory()
    }

    pub fn raw(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_acquisition() {
        // will only pass on Apple Silicon — expected to fail on Linux CI
        match MetalDevice::system_default() {
            Ok(dev) => {
                println!("Metal device: {}", dev.name());
                println!("Unified memory: {}", dev.has_unified_memory());
                println!(
                    "Max working set: {} MB",
                    dev.recommended_max_working_set_size() / 1024 / 1024
                );
                assert!(
                    dev.has_unified_memory(),
                    "expected unified memory on Apple Silicon"
                );
            }
            Err(MetalError::NoDevice(_)) => {
                println!("No Metal device available (expected on non-Apple hardware)");
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
}
