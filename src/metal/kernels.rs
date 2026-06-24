use crate::metal::allocator::{BlockHandle, MetalAllocator};
use crate::metal::context::MetalContext;
use crate::metal::error::{MetalError, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_foundation::NSString;
use objc2_metal::MTLCommandBuffer;
use objc2_metal::MTLCommandEncoder;
use objc2_metal::MTLComputeCommandEncoder;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize};
use std::ffi::c_void;
use std::ptr::NonNull;

const RMS_NORM_MSL: &str = include_str!("rms_norm_f16.metal");

pub struct MetalKernels {
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    rms_norm_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl MetalKernels {
    pub fn new(ctx: &MetalContext) -> Result<Self> {
        let device = ctx.device.raw();
        use objc2_foundation::NSString;
        let src = NSString::from_str(RMS_NORM_MSL);
        let library = device
            .newLibraryWithSource_options_error(&src, None)
            .map_err(|e| MetalError::LibraryCompilation(e.localizedDescription().to_string()))?;

        let function = library
            .newFunctionWithName(ns_string!("rms_norm_f16"))
            .ok_or(MetalError::KernelNotLoaded("rms_norm_f16"))?;

        let rms_norm_pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalError::Internal(e.localizedDescription().to_string()))?;

        Ok(Self {
            library,
            rms_norm_pipeline,
        })
    }

    pub fn rms_norm_f16(
        &self,
        ctx: &MetalContext,
        allocator: &MetalAllocator,
        input: &BlockHandle,
        weight: &BlockHandle,
        output: &BlockHandle,
        num_tokens: u32,
        hidden: u32,
        eps: f32,
    ) -> Result<()> {
        let cmd_buf = ctx.command_buffer()?;
        let encoder = cmd_buf.computeCommandEncoder().ok_or(MetalError::Internal(
            "failed to create compute encoder".into(),
        ))?;

        encoder.setComputePipelineState(&self.rms_norm_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(allocator.buffer()),
                input.offset_bytes,
                0, // [[buffer(0)]]
            );
            encoder.setBuffer_offset_atIndex(Some(allocator.buffer()), weight.offset_bytes, 1);
            encoder.setBuffer_offset_atIndex(Some(allocator.buffer()), output.offset_bytes, 2);

            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&hidden as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                3, // [[buffer(3)]]
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&eps as *const f32 as *mut c_void),
                std::mem::size_of::<f32>(),
                4, // [[buffer(4)]]
            );
        }

        let grid = MTLSize {
            width: num_tokens as usize,
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
    use crate::metal::allocator::MetalAllocator;
    use crate::metal::context::MetalContext;
    use crate::metal::device::MetalDevice;
    use half::f16;

    fn setup() -> (MetalContext, MetalAllocator, MetalKernels) {
        let device = MetalDevice::system_default().unwrap();
        let ctx = MetalContext::new(device).unwrap();
        let alloc = MetalAllocator::new(&ctx, 4 * 1024 * 1024).unwrap(); // 4MB pool
        let kernels = MetalKernels::new(&ctx).unwrap();
        (ctx, alloc, kernels)
    }

    // reference RMSNorm in f32 on CPU — this is the ground truth
    fn rms_norm_ref(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let mean_sq = input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        input
            .iter()
            .zip(weight.iter())
            .map(|(x, w)| x * scale * w)
            .collect()
    }

    #[test]
    fn test_rms_norm_single_token() {
        let (ctx, mut alloc, kernels) = setup();

        let hidden = 8usize;
        let num_tokens = 1usize;
        let eps = 1e-5f32;

        // known input and weight
        let input_f32 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight_f32 = vec![1.0f32; hidden];

        // allocate GPU buffers
        let block_size = hidden * std::mem::size_of::<f16>();
        let in_block = alloc.alloc(block_size, 16).unwrap();
        let wt_block = alloc.alloc(block_size, 16).unwrap();
        let out_block = alloc.alloc(block_size, 16).unwrap();

        // write input and weight via CPU pointer (unified memory — no copy needed)
        unsafe {
            let in_ptr = in_block.ptr as *mut f16;
            let wt_ptr = wt_block.ptr as *mut f16;
            for i in 0..hidden {
                in_ptr.add(i).write(f16::from_f32(input_f32[i]));
                wt_ptr.add(i).write(f16::from_f32(weight_f32[i]));
            }
        }

        // run the kernel
        kernels
            .rms_norm_f16(
                &ctx,
                &alloc,
                &in_block,
                &wt_block,
                &out_block,
                num_tokens as u32,
                hidden as u32,
                eps,
            )
            .unwrap();

        // read output back (unified memory — already visible on CPU after sync)
        let mut output_f32 = vec![0.0f32; hidden];
        unsafe {
            let out_ptr = out_block.ptr as *const f16;
            for i in 0..hidden {
                output_f32[i] = out_ptr.add(i).read().to_f32();
            }
        }

        // compare against CPU reference
        let expected = rms_norm_ref(&input_f32, &weight_f32, eps);
        for i in 0..hidden {
            let diff = (output_f32[i] - expected[i]).abs();
            assert!(
                diff < 1e-2,
                "token 0 element {i}: got {}, expected {}, diff {diff}",
                output_f32[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_rms_norm_multi_token() {
        let (ctx, mut alloc, kernels) = setup();

        let hidden = 16usize;
        let num_tokens = 4usize;
        let eps = 1e-5f32;

        let input_f32: Vec<f32> = (0..num_tokens * hidden)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let weight_f32 = vec![1.0f32; hidden];

        let block_size = num_tokens * hidden * std::mem::size_of::<f16>();
        let in_block = alloc.alloc(block_size, 16).unwrap();
        let wt_block = alloc
            .alloc(hidden * std::mem::size_of::<f16>(), 16)
            .unwrap();
        let out_block = alloc.alloc(block_size, 16).unwrap();

        unsafe {
            let in_ptr = in_block.ptr as *mut f16;
            let wt_ptr = wt_block.ptr as *mut f16;
            for i in 0..num_tokens * hidden {
                in_ptr.add(i).write(f16::from_f32(input_f32[i]));
            }
            for i in 0..hidden {
                wt_ptr.add(i).write(f16::from_f32(weight_f32[i]));
            }
        }

        kernels
            .rms_norm_f16(
                &ctx,
                &alloc,
                &in_block,
                &wt_block,
                &out_block,
                num_tokens as u32,
                hidden as u32,
                eps,
            )
            .unwrap();

        unsafe {
            let out_ptr = out_block.ptr as *const f16;
            for t in 0..num_tokens {
                let row_in: Vec<f32> = (0..hidden).map(|i| input_f32[t * hidden + i]).collect();
                let expected = rms_norm_ref(&row_in, &weight_f32, eps);
                for i in 0..hidden {
                    let got = out_ptr.add(t * hidden + i).read().to_f32();
                    let diff = (got - expected[i]).abs();
                    assert!(
                        diff < 1e-2,
                        "token {t} element {i}: got {got}, expected {}, diff {diff}",
                        expected[i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_rms_norm_weight_scaling() {
        // verify that weight scaling is actually applied, not just normalization
        let (ctx, mut alloc, kernels) = setup();

        let hidden = 4usize;
        let eps = 1e-5f32;

        let input_f32 = vec![1.0f32, 1.0, 1.0, 1.0]; // uniform input
        let weight_f32 = vec![2.0f32, 0.5, 1.0, 3.0]; // non-trivial weights

        let block_size = hidden * std::mem::size_of::<f16>();
        let in_block = alloc.alloc(block_size, 16).unwrap();
        let wt_block = alloc.alloc(block_size, 16).unwrap();
        let out_block = alloc.alloc(block_size, 16).unwrap();

        unsafe {
            let in_ptr = in_block.ptr as *mut f16;
            let wt_ptr = wt_block.ptr as *mut f16;
            for i in 0..hidden {
                in_ptr.add(i).write(f16::from_f32(input_f32[i]));
                wt_ptr.add(i).write(f16::from_f32(weight_f32[i]));
            }
        }

        kernels
            .rms_norm_f16(
                &ctx,
                &alloc,
                &in_block,
                &wt_block,
                &out_block,
                1,
                hidden as u32,
                eps,
            )
            .unwrap();

        let expected = rms_norm_ref(&input_f32, &weight_f32, eps);
        unsafe {
            let out_ptr = out_block.ptr as *const f16;
            for i in 0..hidden {
                let got = out_ptr.add(i).read().to_f32();
                let diff = (got - expected[i]).abs();
                assert!(
                    diff < 1e-2,
                    "element {i}: got {got}, expected {}",
                    expected[i]
                );
            }
        }
    }
}
