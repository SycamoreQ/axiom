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

const ATTN_PV_MSL: &str = include_str!("attention_pv_f16.metal");

pub struct AttentionPVKernel {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl AttentionPVKernel {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let source_str = NSString::from_str(ATTN_PV_MSL);

        let library = device
            .newLibraryWithSource_options_error(&source_str, None)
            .map_err(|e| MetalError::LibraryCompilation(e.localizedDescription().to_string()))?;

        // Note: Make sure the function name here perfectly matches your kernel signature
        let function = library
            .newFunctionWithName(ns_string!("attention_pv_float"))
            .ok_or(MetalError::KernelNotLoaded("attention_pv_float"))?;

        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalError::Internal(e.localizedDescription().to_string()))?;

        Ok(Self { pipeline })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_pv_f16(
        &self,
        ctx: &MetalContext,
        allocator: &MetalAllocator,
        scores: &BlockHandle, // [n_heads, seq_len] (Your kernel maps this to buffer 0)
        v_cache: &BlockHandle, // [seq_len, n_heads, head_dim]
        out: &BlockHandle,    // [n_heads, head_dim] — output
        n_heads: u32,
        seq_len: u32,
        head_dim: u32,
        current_pos: u32,
    ) -> Result<()> {
        let cmd_buf = ctx.command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| MetalError::Internal("failed to create compute encoder".into()))?;

        encoder.setComputePipelineState(&self.pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(allocator.buffer()), scores.offset_bytes, 0);
            encoder.setBuffer_offset_atIndex(Some(allocator.buffer()), v_cache.offset_bytes, 1);
            encoder.setBuffer_offset_atIndex(Some(allocator.buffer()), out.offset_bytes, 2);

            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&n_heads as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&seq_len as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&head_dim as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&current_pos as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                6,
            );
        }

        // Your kernel uses THREADGROUP_SIZE = 32 and assigns 1 threadgroup per head.
        let threadgroup = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        let grid = MTLSize {
            width: (n_heads * 32) as usize,
            height: 1,
            depth: 1,
        };

        unsafe {
            encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
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

    fn setup() -> (MetalContext, MetalAllocator, AttentionPVKernel) {
        let device = MetalDevice::system_default().unwrap();
        let ctx = MetalContext::new(device).unwrap();
        let alloc = MetalAllocator::new(&ctx, 16 * 1024 * 1024).unwrap();
        let kernel = AttentionPVKernel::new(ctx.device.raw()).unwrap();
        (ctx, alloc, kernel)
    }

    fn attention_pv_ref(
        scores: &[f16],
        v_cache: &[f16],
        head_dim: usize,
        n_heads: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n_heads * head_dim];

        for h in 0..n_heads {
            // 1. Softmax Reference
            let mut max_val = f32::NEG_INFINITY;
            for t in 0..seq_len {
                let s = scores[h * seq_len + t].to_f32();
                if s > max_val {
                    max_val = s;
                }
            }

            let mut sum_exp = 0.0f32;
            let mut weights = vec![0.0f32; seq_len];
            for t in 0..seq_len {
                let s = scores[h * seq_len + t].to_f32();
                let e = (s - max_val).exp();
                sum_exp += e;
                weights[t] = e;
            }

            for t in 0..seq_len {
                weights[t] /= sum_exp;
            }

            // 2. PV Dot Reference
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for t in 0..seq_len {
                    let weight = weights[t];
                    let v_idx = (t * n_heads + h) * head_dim + d;
                    let v_val = v_cache[v_idx].to_f32();
                    acc += weight * v_val;
                }
                out[h * head_dim + d] = acc;
            }
        }
        out
    }

    #[test]
    fn test_attention_pv_execution() {
        let (ctx, mut alloc, kernel) = setup();

        let head_dim = 64usize;
        let seq_len = 64usize;
        let n_heads = 12usize;
        let current_pos = 63usize; // Max pos for sequence length 64

        let scores_f16: Vec<f16> = (0..(n_heads * seq_len))
            .map(|i| f16::from_f32((i % 10) as f32 * 0.1))
            .collect();

        let v_f16: Vec<f16> = (0..(seq_len * n_heads * head_dim))
            .map(|i| f16::from_f32((i % 5) as f32 * 0.1))
            .collect();

        let out_f16 = vec![f16::from_f32(0.0); n_heads * head_dim];

        let scores_bytes = n_heads * seq_len * std::mem::size_of::<f16>();
        let v_bytes = seq_len * n_heads * head_dim * std::mem::size_of::<f16>();
        let out_bytes = n_heads * head_dim * std::mem::size_of::<f16>();

        let scores_block = alloc.alloc(scores_bytes, 16).unwrap();
        let v_block = alloc.alloc(v_bytes, 16).unwrap();
        let out_block = alloc.alloc(out_bytes, 16).unwrap();

        unsafe {
            std::ptr::copy_nonoverlapping(
                scores_f16.as_ptr(),
                scores_block.ptr as *mut f16,
                scores_f16.len(),
            );
            std::ptr::copy_nonoverlapping(v_f16.as_ptr(), v_block.ptr as *mut f16, v_f16.len());
            std::ptr::copy_nonoverlapping(
                out_f16.as_ptr(),
                out_block.ptr as *mut f16,
                out_f16.len(),
            );
        }

        kernel
            .attention_pv_f16(
                &ctx,
                &alloc,
                &scores_block,
                &v_block,
                &out_block,
                n_heads as u32,
                seq_len as u32,
                head_dim as u32,
                current_pos as u32,
            )
            .unwrap();

        let expected_out = attention_pv_ref(&scores_f16, &v_f16, head_dim, n_heads, seq_len);

        unsafe {
            let out_ptr = out_block.ptr as *const f16;
            for i in 0..(n_heads * head_dim) {
                let got = out_ptr.add(i).read().to_f32();
                assert!(
                    (got - expected_out[i]).abs() < 0.05, // F16 exponential operations require wider tolerance
                    "Mismatch at index {}: got {}, expected {}",
                    i,
                    got,
                    expected_out[i]
                );
            }
        }
    }

    #[test]
    fn test_attention_pv_asymmetrical_dims() {
        let (ctx, mut alloc, kernel) = setup();

        let head_dim = 48usize;
        let seq_len = 35usize;
        let n_heads = 8usize;
        let current_pos = 34usize;

        let scores_f16: Vec<f16> = (0..(n_heads * seq_len))
            .map(|i| f16::from_f32((i % 7) as f32 * 0.1))
            .collect();

        let v_f16: Vec<f16> = (0..(seq_len * n_heads * head_dim))
            .map(|i| f16::from_f32((i % 7) as f32 * 0.1))
            .collect();

        let out_f16 = vec![f16::from_f32(0.0); n_heads * head_dim];

        let scores_block = alloc.alloc(n_heads * seq_len * 2, 16).unwrap();
        let v_block = alloc.alloc(seq_len * n_heads * head_dim * 2, 16).unwrap();
        let out_block = alloc.alloc(n_heads * head_dim * 2, 16).unwrap();

        unsafe {
            std::ptr::copy_nonoverlapping(
                scores_f16.as_ptr(),
                scores_block.ptr as *mut f16,
                scores_f16.len(),
            );
            std::ptr::copy_nonoverlapping(v_f16.as_ptr(), v_block.ptr as *mut f16, v_f16.len());
            std::ptr::copy_nonoverlapping(
                out_f16.as_ptr(),
                out_block.ptr as *mut f16,
                out_f16.len(),
            );
        }

        kernel
            .attention_pv_f16(
                &ctx,
                &alloc,
                &scores_block,
                &v_block,
                &out_block,
                n_heads as u32,
                seq_len as u32,
                head_dim as u32,
                current_pos as u32,
            )
            .unwrap();

        let expected_out = attention_pv_ref(&scores_f16, &v_f16, head_dim, n_heads, seq_len);

        unsafe {
            let out_ptr = out_block.ptr as *const f16;
            for i in 0..(n_heads * head_dim) {
                let got = out_ptr.add(i).read().to_f32();
                assert!(
                    (got - expected_out[i]).abs() < 0.05,
                    "Asymmetric mismatch at index {}: got {}, expected {}",
                    i,
                    got,
                    expected_out[i]
                );
            }
        }
    }
}
