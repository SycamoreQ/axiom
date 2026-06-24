use crate::metal::device::MetalDevice;
use crate::metal::error::{MetalError, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer;
use objc2_metal::MTLCommandBufferStatus;
use objc2_metal::MTLCommandQueue;
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

pub struct MetalContext {
    pub device: MetalDevice,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl MetalContext {
    pub fn new(device: MetalDevice) -> Result<Self> {
        let queue = device
            .raw()
            .newCommandQueue()
            .ok_or(MetalError::NoCommandQueue)?;
        Ok(Self { device, queue })
    }

    pub fn command_buffer(&self) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        self.queue
            .commandBuffer()
            .ok_or(MetalError::NoCommandBuffer)
    }

    pub fn synchronize(&self) -> Result<()> {
        let buffer = self.command_buffer()?;

        buffer.commit();
        buffer.waitUntilCompleted();

        unsafe {
            if buffer.status() == MTLCommandBufferStatus::Error {
                if let Some(err) = buffer.error() {
                    return Err(MetalError::Internal(err.localizedDescription().to_string()));
                }
                return Err(MetalError::Internal("Unknown Metal error".to_string()));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::device::MetalDevice;

    fn make_context() -> MetalContext {
        let device = MetalDevice::system_default().expect("no Metal device — run on Apple Silicon");
        MetalContext::new(device).expect("failed to create MetalContext")
    }

    #[test]
    fn test_context_creation() {
        let ctx = make_context();
        println!("context created on: {}", ctx.device.name());
    }

    #[test]
    fn test_command_buffer_creation() {
        let ctx = make_context();
        let buf = ctx
            .command_buffer()
            .expect("failed to create command buffer");
        println!("command buffer status: {:?}", buf.status());
    }

    #[test]
    fn test_synchronize_empty() {
        // empty fence — should complete instantly and not error
        let ctx = make_context();
        ctx.synchronize().expect("synchronize failed");
    }

    #[test]
    fn test_synchronize_repeated() {
        // synchronize should be callable multiple times without breaking queue state
        let ctx = make_context();
        for i in 0..5 {
            ctx.synchronize()
                .unwrap_or_else(|e| panic!("sync {i} failed: {e}"));
        }
    }
}
