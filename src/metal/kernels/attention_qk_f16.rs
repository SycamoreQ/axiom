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

const ATTN_QK_MSL: &str = include_str!("attention_qk_f16.metal");

pub struct AttentionQKKernel {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl AttentionQKKernel {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let source_str = NSString::from_str(ATTN_QK_MSL);

        let library = device
            .newLibraryWithSource_options_error(&source_str, None)
            .map_err(|e| MetalError::LibraryCompilation(e.localizedDescription().to_string()))?;

        let function = library
            .newFunctionWithName(ns_string!("attention_qk_f16"))
            .ok_or(MetalError::KernelNotLoaded("attention_qk_f16"))?;

        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalError::Internal(e.localizedDescription().to_string()))?;

        Ok(Self { pipeline })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_qk_f16(
        &self,
        ctx: &MetalContext,
        allocator: &MetalAllocator,
        q: &BlockHandle,       // [n_heads, head_dim]
        k_cache: &BlockHandle, // [seq_len, n_heads, head_dim]
        scores: &BlockHandle,  // [n_heads, seq_len] — output
        n_heads: u32,
        head_dim: u32,
        seq_len: u32,
        current_pos: u32,
    ) -> Result<()> {
        let cmd_buf = ctx.command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| MetalError::Internal("failed to create compute encoder".into()))?;

        encoder.setComputePipelineState(&self.pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(allocator.buffer()), q.offset_bytes, 0);
            encoder.setBuffer_offset_atIndex(Some(allocator.buffer()), k_cache.offset_bytes, 1);
            encoder.setBuffer_offset_atIndex(Some(allocator.buffer()), scores.offset_bytes, 2);

            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&n_heads as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&head_dim as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&seq_len as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&current_pos as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                6,
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

    fn setup() -> (MetalContext, MetalAllocator, AttentionQKKernel) {
        let device = MetalDevice::system_default().unwrap();
        let ctx = MetalContext::new(device).unwrap();
        let alloc = MetalAllocator::new(&ctx, 16 * 1024 * 1024).unwrap();
        let kernel = AttentionQKKernel::new(ctx.device.raw()).unwrap();
        (ctx, alloc, kernel)
    }

    fn attention_qk_ref(
        q: &[f16],
        k_cache: &[f16],
        head_dim: usize,
        n_heads: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let mut scores = vec![0.0f32; n_heads * seq_len];
        let scale_fac = 1.0f32 / (head_dim as f32).sqrt();
        for h in 0..n_heads {
            for t in 0..seq_len {
                let mut sum = 0.0f32;
                for d in 0..head_dim {
                    let q_idx = h * head_dim + d;

                    let k_idx = t * (n_heads * head_dim) + h * head_dim + d;

                    sum += q[q_idx].to_f32() * k_cache[k_idx].to_f32();
                }

                let score_idx = h * seq_len + t;
                scores[score_idx] = sum * scale_fac;
            }
        }

        scores
    }

    #[test]
    fn test_attention_f32_execution() {
        let (ctx, mut alloc, kernel) = setup();

        let head_dim = 64usize;
        let seq_len = 64usize;
        let n_heads = 64usize;
        let current_pos = 64usize;

        let q_f16: Vec<f16> = (0..(n_heads * head_dim))
            .map(|i| f16::from_f32((i % 10) as f32 * 0.1))
            .collect(); // initlalized as 0.1 0.2 ..

        let k_f16: Vec<f16> = (0..(seq_len * n_heads * head_dim))
            .map(|i| f16::from_f32((i % 10) as f32 * 0.1))
            .collect();

        let scores_f16: Vec<f16> = vec![f16::from_f32(0.0); seq_len * n_heads];

        let q_bytes = n_heads * head_dim * std::mem::size_of::<f16>();
        let k_bytes = seq_len * n_heads * head_dim * std::mem::size_of::<f16>();
        let scores_bytes = seq_len * n_heads * std::mem::size_of::<f16>();

        let q_block = alloc.alloc(q_bytes, 16).unwrap();
        let k_block = alloc.alloc(k_bytes, 16).unwrap();
        let scores_block = alloc.alloc(scores_bytes, 16).unwrap();

        unsafe {
            let q_ptr = q_block.ptr as *mut f16;
            let k_ptr = k_block.ptr as *mut f16;
            let scores_ptr = scores_block.ptr as *mut f16;

            for i in 0..n_heads * head_dim {
                q_ptr.add(i).write(q_f16[i]);
            }
            for i in 0..(seq_len * head_dim * n_heads) {
                k_ptr.add(i).write(k_f16[i]);
            }

            for i in 0..(seq_len * n_heads) {
                scores_ptr.add(i).write(scores_f16[i]);
            }
        }

        kernel
            .attention_qk_f16(
                &ctx,
                &alloc,
                &q_block,
                &k_block,
                &scores_block,
                n_heads as u32,
                head_dim as u32,
                seq_len as u32,
                current_pos as u32,
            )
            .unwrap();

        let expected_score = attention_qk_ref(&q_f16, &k_f16, head_dim, n_heads, seq_len);
        attention_qk_ref(&q_f16, &k_f16, head_dim, n_heads, seq_len);

        unsafe {
            let out_ptr = scores_block.ptr as *const f16;
            for i in 0..(seq_len * n_heads) {
                let got = out_ptr.add(i).read().to_f32();
                assert!((got - expected_score[i].abs() < 0.5));
            }
        }
    }

    #[test]
    fn test_attention_causal_mask() {
        let (ctx, mut alloc, kernel) = setup();

        let head_dim = 64usize;
        let seq_len = 64usize;
        let n_heads = 64usize;
        let current_pos = 31usize;

        let q_f16: Vec<f16> = (0..(n_heads * head_dim))
            .map(|i| f16::from_f32((i % 10) as f32 * 0.1))
            .collect();
        let k_f16: Vec<f16> = (0..(seq_len * n_heads * head_dim))
            .map(|i| f16::from_f32((i % 10) as f32 * 0.1))
            .collect();
        let scores_f16 = vec![f16::from_f32(0.0); seq_len * n_heads];

        let q_block = alloc.alloc(n_heads * head_dim * 2, 16).unwrap();
        let k_block = alloc.alloc(seq_len * n_heads * head_dim * 2, 16).unwrap();
        let scores_block = alloc.alloc(seq_len * n_heads * 2, 16).unwrap();

        unsafe {
            std::ptr::copy_nonoverlapping(q_f16.as_ptr(), q_block.ptr as *mut f16, q_f16.len());
            std::ptr::copy_nonoverlapping(k_f16.as_ptr(), k_block.ptr as *mut f16, k_f16.len());
            std::ptr::copy_nonoverlapping(
                scores_f16.as_ptr(),
                scores_block.ptr as *mut f16,
                scores_f16.len(),
            );
        }

        kernel
            .attention_qk_f16(
                &ctx,
                &alloc,
                &q_block,
                &k_block,
                &scores_block,
                n_heads as u32,
                head_dim as u32,
                seq_len as u32,
                current_pos as u32,
            )
            .unwrap();

        let expected_score = attention_qk_ref(&q_f16, &k_f16, head_dim, n_heads, seq_len);

        unsafe {
            let out_ptr = scores_block.ptr as *const f16;
            for i in 0..(seq_len * n_heads) {
                let got = out_ptr.add(i).read().to_f32();

                // The memory layout is [n_heads, seq_len], so i % seq_len gives us the current pos/token
                let pos = i % seq_len;

                if pos > current_pos {
                    assert_eq!(
                        got,
                        f32::NEG_INFINITY,
                        "Index {} (pos {}) should be -INFINITY",
                        i,
                        pos
                    );
                } else {
                    assert!(
                        (got - expected_score[i]).abs() < 0.5,
                        "Mismatch at index {}",
                        i
                    );
                }
            }
        }
    }

    #[test]
    fn test_attention_asymmetrical_dims() {
        let (ctx, mut alloc, kernel) = setup();

        // Purposely weird, non-matching numbers
        let head_dim = 48usize;
        let seq_len = 35usize;
        let n_heads = 12usize;
        let current_pos = 35usize;

        let q_f16: Vec<f16> = (0..(n_heads * head_dim))
            .map(|i| f16::from_f32((i % 7) as f32 * 0.1))
            .collect();
        let k_f16: Vec<f16> = (0..(seq_len * n_heads * head_dim))
            .map(|i| f16::from_f32((i % 7) as f32 * 0.1))
            .collect();
        let scores_f16 = vec![f16::from_f32(0.0); seq_len * n_heads];

        let q_block = alloc.alloc(n_heads * head_dim * 2, 16).unwrap();
        let k_block = alloc.alloc(seq_len * n_heads * head_dim * 2, 16).unwrap();
        let scores_block = alloc.alloc(seq_len * n_heads * 2, 16).unwrap();

        unsafe {
            std::ptr::copy_nonoverlapping(q_f16.as_ptr(), q_block.ptr as *mut f16, q_f16.len());
            std::ptr::copy_nonoverlapping(k_f16.as_ptr(), k_block.ptr as *mut f16, k_f16.len());
            std::ptr::copy_nonoverlapping(
                scores_f16.as_ptr(),
                scores_block.ptr as *mut f16,
                scores_f16.len(),
            );
        }

        kernel
            .attention_qk_f16(
                &ctx,
                &alloc,
                &q_block,
                &k_block,
                &scores_block,
                n_heads as u32,
                head_dim as u32,
                seq_len as u32,
                current_pos as u32,
            )
            .unwrap();

        let expected_score = attention_qk_ref(&q_f16, &k_f16, head_dim, n_heads, seq_len);

        unsafe {
            let out_ptr = scores_block.ptr as *const f16;
            for i in 0..(seq_len * n_heads) {
                let got = out_ptr.add(i).read().to_f32();
                assert!(
                    (got - expected_score[i]).abs() < 0.5,
                    "Asymmetric mismatch at index {}",
                    i
                );
            }
        }
    }
}
