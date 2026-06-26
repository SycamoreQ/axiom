use crate::metal::allocator::{BlockHandle, MetalAllocator};
use crate::metal::context::MetalContext;
use crate::metal::error::{MetalError, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{ns_string, NSString};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLDevice, MTLLibrary, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;

const SWIGLU_MSL: &str = include_str!("swiglu_f16.metal");

pub struct SwigluKernel {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl SwigluKernel {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let source_str = NSString::from_str(SWIGLU_MSL);

        let library = device
            .newLibraryWithSource_options_error(&source_str, None)
            .map_err(|e| MetalError::LibraryCompilation(e.localizedDescription().to_string()))?;

        let function = library
            .newFunctionWithName(ns_string!("swiglu_f16"))
            .ok_or(MetalError::KernelNotLoaded("swiglu_f16"))?;

        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalError::Internal(e.localizedDescription().to_string()))?;

        Ok(Self { pipeline })
    }

    pub fn swiglu_f16(
        &self,
        ctx: &MetalContext,
        allocator: &MetalAllocator,
        gate: &BlockHandle,
        up: &BlockHandle,
        output: &BlockHandle,
        num_elements: u32,
    ) -> Result<()> {
        let cmd_buf = ctx.command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| MetalError::Internal("failed to create compute encoder".into()))?;

        encoder.setComputePipelineState(&self.pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(allocator.buffer()), gate.offset_bytes, 0);
            encoder.setBuffer_offset_atIndex(Some(allocator.buffer()), up.offset_bytes, 1);
            encoder.setBuffer_offset_atIndex(Some(allocator.buffer()), output.offset_bytes, 2);

            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&num_elements as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                3,
            );
        }

        let grid = MTLSize {
            width: num_elements as usize,
            height: 1,
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };

        unsafe {
            encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
        }

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::device::MetalDevice;
    use half::f16;

    fn setup() -> (MetalContext, MetalAllocator, SwigluKernel) {
        let device = MetalDevice::system_default().unwrap();
        let ctx = MetalContext::new(device).unwrap();
        let alloc = MetalAllocator::new(&ctx, 2 * 1024 * 1024).unwrap();
        let kernel = SwigluKernel::new(ctx.device.raw()).unwrap();
        (ctx, alloc, kernel)
    }

    fn swiglu_ref(gate: &[f32], up: &[f32]) -> Vec<f32> {
        gate.iter()
            .zip(up.iter())
            .map(|(&g, &u)| {
                let silu_up = u * (1.0 / (1.0 + (-u).exp()));
                g * silu_up
            })
            .collect()
    }

    #[test]
    fn test_swiglu_f16_execution() {
        let (ctx, mut alloc, kernel) = setup();
        let num_elements = 8usize;

        let gate_f32 = vec![1.0f32, -2.0, 3.0, -4.0, 0.0, 0.5, -0.5, 2.0];
        let up_f32 = vec![2.0f32, 1.5, -1.0, 2.0, 4.0, -2.0, 1.0, 0.5];

        let block_size = num_elements * std::mem::size_of::<f16>();
        let gate_block = alloc.alloc(block_size, 16).unwrap();
        let up_block = alloc.alloc(block_size, 16).unwrap();
        let out_block = alloc.alloc(block_size, 16).unwrap();

        unsafe {
            let gate_ptr = gate_block.ptr as *mut f16;
            let up_ptr = up_block.ptr as *mut f16;
            for i in 0..num_elements {
                gate_ptr.add(i).write(f16::from_f32(gate_f32[i]));
                up_ptr.add(i).write(f16::from_f32(up_f32[i]));
            }
        }

        kernel
            .swiglu_f16(
                &ctx,
                &alloc,
                &gate_block,
                &up_block,
                &out_block,
                num_elements as u32,
            )
            .unwrap();

        let mut output_f32 = vec![0.0f32; num_elements];
        unsafe {
            let out_ptr = out_block.ptr as *const f16;
            for i in 0..num_elements {
                output_f32[i] = out_ptr.add(i).read().to_f32();
            }
        }

        let expected = swiglu_ref(&gate_f32, &up_f32);
        for i in 0..num_elements {
            let mut all_ok = true;
            for i in 0..num_elements {
                let diff = (output_f32[i] - expected[i]).abs();
                println!(
                    "i={i}: gate={:.4}, up={:.4}, got={:.6}, expected={:.6}, diff={:.6}",
                    gate_f32[i], up_f32[i], output_f32[i], expected[i], diff
                );
                if diff >= 1e-2 {
                    all_ok = false;
                }
            }
            assert!(all_ok, "swiglu values out of tolerance — see above");
        }
    }
}
