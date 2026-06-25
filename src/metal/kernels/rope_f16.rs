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

const ROPE_MSL: &str = include_str!("rope_f16.metal");

pub struct RopeKernel {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl RopeKernel {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let source_str = NSString::from_str(ROPE_MSL);

        let library = device
            .newLibraryWithSource_options_error(&source_str, None)
            .map_err(|e| MetalError::LibraryCompilation(e.localizedDescription().to_string()))?;

        let function = library
            .newFunctionWithName(ns_string!("rope_f16"))
            .ok_or(MetalError::KernelNotLoaded("rope_f16"))?;

        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalError::Internal(e.localizedDescription().to_string()))?;

        Ok(Self { pipeline })
    }

    pub fn launch(
        &self,
        ctx: &MetalContext,
        allocator: &MetalAllocator,
        x: &BlockHandle,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<()> {
        let cmd_buf = ctx.command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| MetalError::Internal("failed to create compute encoder".into()))?;

        encoder.setComputePipelineState(&self.pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(allocator.buffer()), x.offset_bytes, 0);

            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&seq_len as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                1,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&n_heads as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                2,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&head_dim as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&theta as *const f32 as *mut c_void),
                std::mem::size_of::<f32>(),
                4,
            );
        }

        let grid = MTLSize {
            width: seq_len as usize,
            height: n_heads as usize,
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

    fn setup() -> (MetalContext, MetalAllocator, RopeKernel) {
        let device = MetalDevice::system_default().unwrap();
        let ctx = MetalContext::new(device).unwrap();
        let alloc = MetalAllocator::new(&ctx, 2 * 1024 * 1024).unwrap();
        let kernel = RopeKernel::new(ctx.device.raw()).unwrap();
        (ctx, alloc, kernel)
    }

    fn rope_ref(
        input: &[f32],
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
    ) -> Vec<f32> {
        let mut out = input.to_vec();
        for token in 0..seq_len {
            for head in 0..n_heads {
                let idx = (token * n_heads + head) * head_dim;
                let row = &mut out[idx..idx + head_dim];
                for i in 0..head_dim / 2 {
                    let freq = 1.0f32 / theta.powf(2.0 * i as f32 / head_dim as f32);
                    let angle = token as f32 * freq;
                    let (sin_a, cos_a) = angle.sin_cos();
                    let x0 = row[i];
                    let x1 = row[i + head_dim / 2];
                    row[i] = x0 * cos_a - x1 * sin_a;
                    row[i + head_dim / 2] = x0 * sin_a + x1 * cos_a;
                }
            }
        }
        out
    }

    #[test]
    fn test_rope_position_zero_is_identity() {
        let (ctx, mut alloc, kernel) = setup();
        let n_heads = 2usize;
        let head_dim = 8usize;
        let seq_len = 1usize;

        let input_f32: Vec<f32> = (0..seq_len * n_heads * head_dim)
            .map(|i| (i as f32 + 1.0) * 0.5)
            .collect();
        let block_size = input_f32.len() * std::mem::size_of::<f16>();
        let x_block = alloc.alloc(block_size, 16).unwrap();

        unsafe {
            let ptr = x_block.ptr as *mut f16;
            for i in 0..input_f32.len() {
                ptr.add(i).write(f16::from_f32(input_f32[i]));
            }
        }

        kernel
            .launch(
                &ctx,
                &alloc,
                &x_block,
                seq_len as u32,
                n_heads as u32,
                head_dim as u32,
                10000.0,
            )
            .unwrap();

        unsafe {
            let ptr = x_block.ptr as *const f16;
            for i in 0..input_f32.len() {
                let got = ptr.add(i).read().to_f32();
                assert!((got - input_f32[i]).abs() < 1e-2);
            }
        }
    }

    #[test]
    fn test_rope_multi_token() {
        let (ctx, mut alloc, kernel) = setup();
        let n_heads = 2usize;
        let head_dim = 8usize;
        let seq_len = 4usize;

        let input_f32: Vec<f32> = (0..seq_len * n_heads * head_dim)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let block_size = input_f32.len() * std::mem::size_of::<f16>();
        let x_block = alloc.alloc(block_size, 16).unwrap();

        unsafe {
            let ptr = x_block.ptr as *mut f16;
            for i in 0..input_f32.len() {
                ptr.add(i).write(f16::from_f32(input_f32[i]));
            }
        }

        kernel
            .launch(
                &ctx,
                &alloc,
                &x_block,
                seq_len as u32,
                n_heads as u32,
                head_dim as u32,
                10000.0,
            )
            .unwrap();

        let expected = rope_ref(&input_f32, seq_len, n_heads, head_dim, 10000.0);
        unsafe {
            let ptr = x_block.ptr as *const f16;
            for i in 0..input_f32.len() {
                let got = ptr.add(i).read().to_f32();
                assert!((got - expected[i]).abs() < 1e-2);
            }
        }
    }

    #[test]
    fn test_rope_different_positions_differ() {
        let (ctx, mut alloc, kernel) = setup();
        let n_heads = 1usize;
        let head_dim = 8usize;
        let seq_len = 2usize;

        let input_f32 = vec![1.0f32; seq_len * n_heads * head_dim];
        let block_size = input_f32.len() * std::mem::size_of::<f16>();
        let x_block = alloc.alloc(block_size, 16).unwrap();

        unsafe {
            let ptr = x_block.ptr as *mut f16;
            for i in 0..input_f32.len() {
                ptr.add(i).write(f16::from_f32(input_f32[i]));
            }
        }

        kernel
            .launch(
                &ctx,
                &alloc,
                &x_block,
                seq_len as u32,
                n_heads as u32,
                head_dim as u32,
                10000.0,
            )
            .unwrap();

        unsafe {
            let ptr = x_block.ptr as *const f16;
            let t0: Vec<f32> = (0..head_dim).map(|i| ptr.add(i).read().to_f32()).collect();
            let t1: Vec<f32> = (0..head_dim)
                .map(|i| ptr.add(head_dim + i).read().to_f32())
                .collect();
            assert_ne!(t0, t1);
        }
    }
}
