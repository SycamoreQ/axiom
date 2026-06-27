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

const MATMUL_MSL: &str = include_str!("matmul_f16.metal");

pub struct MatmulKernel {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl MatmulKernel {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let source_str = NSString::from_str(MATMUL_MSL);

        let library = device
            .newLibraryWithSource_options_error(&source_str, None)
            .map_err(|e| MetalError::LibraryCompilation(e.localizedDescription().to_string()))?;

        let function = library
            .newFunctionWithName(ns_string!("matmul_f16"))
            .ok_or(MetalError::KernelNotLoaded("matmul_f16"))?;

        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalError::Internal(e.localizedDescription().to_string()))?;

        Ok(Self { pipeline })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_f16(
        &self,
        ctx: &MetalContext,
        allocator: &MetalAllocator,
        a: &BlockHandle,
        b: &BlockHandle,
        c: &BlockHandle,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let cmd_buf = ctx.command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| MetalError::Internal("failed to create compute encoder".into()))?;

        encoder.setComputePipelineState(&self.pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(allocator.buffer()), a.offset_bytes, 0);
            encoder.setBuffer_offset_atIndex(Some(allocator.buffer()), b.offset_bytes, 1);
            encoder.setBuffer_offset_atIndex(Some(allocator.buffer()), c.offset_bytes, 2);

            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&m as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&n as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&k as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                5,
            );
        }

        let blocksize = 16usize;

        // grid = number of threadgroups needed to cover (N, M)
        let grid = MTLSize {
            width: (n as usize + blocksize - 1) / blocksize,
            height: (m as usize + blocksize - 1) / blocksize,
            depth: 1,
        };

        // threadgroup = one full 16*16 tile
        let threadgroup = MTLSize {
            width: blocksize,
            height: blocksize,
            depth: 1,
        };

        unsafe {
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threadgroup);
            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::device::MetalDevice;
    use half::f16;

    fn setup() -> (MetalContext, MetalAllocator, MatmulKernel) {
        let device = MetalDevice::system_default().unwrap();
        let ctx = MetalContext::new(device).unwrap();
        let alloc = MetalAllocator::new(&ctx, 16 * 1024 * 1024).unwrap();
        let kernel = MatmulKernel::new(ctx.device.raw()).unwrap();
        (ctx, alloc, kernel)
    }

    fn matmul_ref(a: &[f16], b: &[f16], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p].to_f32() * b[p * n + j].to_f32();
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    #[test]
    fn test_matmul_f32_execution() {
        let (ctx, mut alloc, kernel) = setup();

        let m = 64usize;
        let n = 64usize;
        let k = 64usize;

        let a_f16: Vec<f16> = (0..m * k)
            .map(|i| f16::from_f32((i % 10) as f32 * 0.1))
            .collect();
        let b_f16: Vec<f16> = (0..k * n)
            .map(|i| f16::from_f32((i % 10) as f32 * 0.1))
            .collect();
        let c_f16 = vec![f16::from_f32(0.0); m * n];

        let a_bytes = m * k * std::mem::size_of::<f16>();
        let b_bytes = k * n * std::mem::size_of::<f16>();
        let c_bytes = m * n * std::mem::size_of::<f16>();

        let a_block = alloc.alloc(a_bytes, 16).unwrap();
        let b_block = alloc.alloc(b_bytes, 16).unwrap();
        let c_block = alloc.alloc(c_bytes, 16).unwrap();

        unsafe {
            let a_ptr = a_block.ptr as *mut f16;
            let b_ptr = b_block.ptr as *mut f16;
            let c_ptr = c_block.ptr as *mut f16;

            for i in 0..m * k {
                a_ptr.add(i).write(a_f16[i]);
            }
            for i in 0..k * n {
                b_ptr.add(i).write(b_f16[i]);
            }
            for i in 0..m * n {
                c_ptr.add(i).write(c_f16[i]);
            }
        }

        kernel
            .matmul_f16(
                &ctx, &alloc, &a_block, &b_block, &c_block, m as u32, n as u32, k as u32,
            )
            .unwrap();

        let expected_c = matmul_ref(&a_f16, &b_f16, m, n, k);
        matmul_ref(&a_f16, &b_f16, m, n, k);

        unsafe {
            let out_ptr = c_block.ptr as *const f16;
            for i in 0..m * n {
                let got = out_ptr.add(i).read().to_f32();
                assert!((got - expected_c[i]).abs() < 0.5);
            }
        }
    }

    #[test]
    fn test_matmul_non_multiple_of_blocksize() {
        let (ctx, mut alloc, kernel) = setup();
        let m = 17usize;
        let n = 17usize;
        let k = 17usize;

        let a_f16: Vec<f16> = (0..m * k)
            .map(|i| f16::from_f32((i % 10) as f32 * 0.1))
            .collect();
        let b_f16: Vec<f16> = (0..k * n)
            .map(|i| f16::from_f32((i % 10) as f32 * 0.1))
            .collect();
        let c_f16 = vec![f16::from_f32(0.0); m * n];

        let a_bytes = m * k * std::mem::size_of::<f16>();
        let b_bytes = k * n * std::mem::size_of::<f16>();
        let c_bytes = m * n * std::mem::size_of::<f16>();

        let a_block = alloc.alloc(a_bytes, 16).unwrap();
        let b_block = alloc.alloc(b_bytes, 16).unwrap();
        let c_block = alloc.alloc(c_bytes, 16).unwrap();

        unsafe {
            let a_ptr = a_block.ptr as *mut f16;
            let b_ptr = b_block.ptr as *mut f16;
            let c_ptr = c_block.ptr as *mut f16;

            for i in 0..m * k {
                a_ptr.add(i).write(a_f16[i]);
            }
            for i in 0..k * n {
                b_ptr.add(i).write(b_f16[i]);
            }
            for i in 0..m * n {
                c_ptr.add(i).write(c_f16[i]);
            }
        }

        kernel
            .matmul_f16(
                &ctx, &alloc, &a_block, &b_block, &c_block, m as u32, n as u32, k as u32,
            )
            .unwrap();

        let expected_c = matmul_ref(&a_f16, &b_f16, m, n, k);
        matmul_ref(&a_f16, &b_f16, m, n, k);

        unsafe {
            let out_ptr = c_block.ptr as *const f16;
            for i in 0..m * n {
                let got = out_ptr.add(i).read().to_f32();
                assert!((got - expected_c[i]).abs() < 0.5);
            }
        }
    }
}
