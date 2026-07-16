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

const RMS_NORM_MSL: &str = include_str!("kernels/rms_norm_f16.metal");
const RMS_NORM_F32_MSL: &str = include_str!("kernels/rms_norm_f32.metal");
const ROPE_MSL: &str = include_str!("kernels/rope_f16.metal");
const ROPE_F32_MSL: &str = include_str!("kernels/rope_f32.metal");
const SWIGLU_MSL: &str = include_str!("kernels/swiglu_f16.metal");
const SWIGLU_F32_MSL: &str = include_str!("kernels/swiglu_f32.metal");
const MATMUL_MSL: &str = include_str!("kernels/matmul_f16.metal");
const MATMUL_F32_MSL: &str = include_str!("kernels/matmul_f32.metal");
const ATTN_QK_MSL: &str = include_str!("kernels/attention_qk_f16.metal");
const ATTN_PV_MSL: &str = include_str!("kernels/attention_pv_f16.metal");
const SOFTMAX_F32_MSL: &str = include_str!("kernels/softmax_f32.metal");
const SOFTMAX_F16_MSL: &str = include_str!("kernels/softmax_f16.metal");

pub struct MetalKernels {
    pub rms_norm_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub rms_norm_f32_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub rope_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub rope_f32_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub swiglu_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub swiglu_f32_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub matmul_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub matmul_f32_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub attention_qk_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub attention_pv_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub softmax_f32_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub softmax_f16_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl std::fmt::Debug for MetalKernels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalKernels")
            .field("rms_norm_pipeline", &"MTLComputePipelineState")
            .field("rm_norm_f32_pipeline", &"MTLComputePipelineState")
            .field("rope_pipeline", &"MTLComputePipelineState")
            .field("rope_f32_pipeline", &"MTLComputePipelineState")
            .field("swiglu_pipeline", &"MTLComputePipelineState")
            .field("swiglu_f32_pipeline", &"MTLComputePipelineState")
            .field("matmul_pipeline", &"MTLComputePipelineState")
            .field("matmul_f32_pipeline", &"MTLComputePipelineState")
            .field("attention_qk_pipeline", &"MTLComputePipelineState")
            .field("attention_pv_pipeline", &"MTLComputePipelineState")
            .field("softmax_f32_pipeline", &"MTLComputePipelineState")
            .field("softmax_f16_pipeline", &"MTLComputePipelineState")
            .finish()
    }
}

impl MetalKernels {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let build_pipeline =
            |source: &str,
             func_name: &'static str|
             -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
                let source_str = NSString::from_str(source);
                let library = device
                    .newLibraryWithSource_options_error(&source_str, None)
                    .map_err(|e| {
                        MetalError::LibraryCompilation(e.localizedDescription().to_string())
                    })?;

                let func_name_ns = NSString::from_str(func_name);
                let function = library
                    .newFunctionWithName(&func_name_ns)
                    .ok_or(MetalError::KernelNotLoaded(func_name))?;

                device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|e| MetalError::Internal(e.localizedDescription().to_string()))
            };

        Ok(Self {
            rms_norm_pipeline: build_pipeline(RMS_NORM_MSL, "rms_norm_f16")?,
            rms_norm_f32_pipeline: build_pipeline(RMS_NORM_F32_MSL, "rms_norm_f32")?,
            rope_pipeline: build_pipeline(ROPE_MSL, "rope_f16")?,
            rope_f32_pipeline: build_pipeline(ROPE_F32_MSL, "rope_f32")?,
            swiglu_pipeline: build_pipeline(SWIGLU_MSL, "swiglu_f16")?,
            swiglu_f32_pipeline: build_pipeline(SWIGLU_F32_MSL, "swiglu_f32")?,
            matmul_pipeline: build_pipeline(MATMUL_MSL, "matmul_f16")?,
            matmul_f32_pipeline: build_pipeline(MATMUL_F32_MSL, "matmul_f32")?,
            attention_qk_pipeline: build_pipeline(ATTN_QK_MSL, "attention_qk_f16")?,
            attention_pv_pipeline: build_pipeline(ATTN_PV_MSL, "attention_pv_float")?,
            softmax_f32_pipeline: build_pipeline(SOFTMAX_F32_MSL, "softmax_f32")?,
            softmax_f16_pipeline: build_pipeline(SOFTMAX_F16_MSL, "softmax_f16")?,
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
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| MetalError::Internal("failed to create compute encoder".into()))?;

        encoder.setComputePipelineState(&self.rms_norm_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(input.metal_buffer(allocator)),
                input.offset_bytes,
                0,
            );
            encoder.setBuffer_offset_atIndex(
                Some(weight.metal_buffer(allocator)),
                weight.offset_bytes,
                1,
            );
            encoder.setBuffer_offset_atIndex(
                Some(output.metal_buffer(allocator)),
                output.offset_bytes,
                2,
            );

            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&hidden as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&eps as *const f32 as *mut c_void),
                std::mem::size_of::<f32>(),
                4,
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

    pub fn softmax_f32(
        &self,
        ctx: &MetalContext,
        allocator: &MetalAllocator,
        input: &BlockHandle,
        output: &BlockHandle,
        num_rows: u32,
        row_size: u32,
    ) -> Result<()> {
        let cmd_buf = ctx.command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| MetalError::Internal("failed to create compute encoder".into()))?;

        encoder.setComputePipelineState(&self.softmax_f32_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(input.metal_buffer(allocator)),
                input.offset_bytes,
                0,
            );
            encoder.setBuffer_offset_atIndex(
                Some(output.metal_buffer(allocator)),
                output.offset_bytes,
                1,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&row_size as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                2,
            );
        }

        let grid = MTLSize {
            width: num_rows as usize,
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

    pub fn softmax_f16(
        &self,
        ctx: &MetalContext,
        allocator: &MetalAllocator,
        input: &BlockHandle,
        output: &BlockHandle,
        num_rows: u32,
        row_size: u32,
    ) -> Result<()> {
        let cmd_buf = ctx.command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| MetalError::Internal("failed to create compute encoder".into()))?;

        encoder.setComputePipelineState(&self.softmax_f16_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(input.metal_buffer(allocator)),
                input.offset_bytes,
                0,
            );
            encoder.setBuffer_offset_atIndex(
                Some(output.metal_buffer(allocator)),
                output.offset_bytes,
                1,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&row_size as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                2,
            );
        }

        let grid = MTLSize {
            width: num_rows as usize,
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

    pub fn rms_norm_f32(
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
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| MetalError::Internal("failed to create compute encoder".into()))?;

        encoder.setComputePipelineState(&self.rms_norm_f32_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(input.metal_buffer(allocator)),
                input.offset_bytes,
                0,
            );
            encoder.setBuffer_offset_atIndex(
                Some(weight.metal_buffer(allocator)),
                weight.offset_bytes,
                1,
            );
            encoder.setBuffer_offset_atIndex(
                Some(output.metal_buffer(allocator)),
                output.offset_bytes,
                2,
            );

            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&hidden as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&eps as *const f32 as *mut c_void),
                std::mem::size_of::<f32>(),
                4,
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

    pub fn rope_f16(
        &self,
        ctx: &MetalContext,
        allocator: &MetalAllocator,
        x: &BlockHandle,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        theta: f32,
        offset: u32,
    ) -> Result<()> {
        let cmd_buf = ctx.command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| MetalError::Internal("failed to create compute encoder".into()))?;

        encoder.setComputePipelineState(&self.rope_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.metal_buffer(allocator)), x.offset_bytes, 0);

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
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&offset as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                5,
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

    pub fn rope_f32(
        &self,
        ctx: &MetalContext,
        allocator: &MetalAllocator,
        x: &BlockHandle,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        theta: f32,
        offset: u32,
    ) -> Result<()> {
        let cmd_buf = ctx.command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| MetalError::Internal("failed to create compute encoder".into()))?;

        encoder.setComputePipelineState(&self.rope_f32_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.metal_buffer(allocator)), x.offset_bytes, 0);

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
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(&offset as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                5,
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

        encoder.setComputePipelineState(&self.swiglu_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(gate.metal_buffer(allocator)),
                gate.offset_bytes,
                0,
            );
            encoder.setBuffer_offset_atIndex(Some(up.metal_buffer(allocator)), up.offset_bytes, 1);
            encoder.setBuffer_offset_atIndex(
                Some(output.metal_buffer(allocator)),
                output.offset_bytes,
                2,
            );

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

    pub fn swiglu_f32(
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

        encoder.setComputePipelineState(&self.swiglu_f32_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(gate.metal_buffer(allocator)),
                gate.offset_bytes,
                0,
            );
            encoder.setBuffer_offset_atIndex(Some(up.metal_buffer(allocator)), up.offset_bytes, 1);
            encoder.setBuffer_offset_atIndex(
                Some(output.metal_buffer(allocator)),
                output.offset_bytes,
                2,
            );

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

        encoder.setComputePipelineState(&self.matmul_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.metal_buffer(allocator)), a.offset_bytes, 0);
            encoder.setBuffer_offset_atIndex(Some(b.metal_buffer(allocator)), b.offset_bytes, 1);
            encoder.setBuffer_offset_atIndex(Some(c.metal_buffer(allocator)), c.offset_bytes, 2);

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

        let grid = MTLSize {
            width: (n as usize + blocksize - 1) / blocksize,
            height: (m as usize + blocksize - 1) / blocksize,
            depth: 1,
        };

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

    pub fn matmul_f32(
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

        encoder.setComputePipelineState(&self.matmul_f32_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.metal_buffer(allocator)), a.offset_bytes, 0);
            encoder.setBuffer_offset_atIndex(Some(b.metal_buffer(allocator)), b.offset_bytes, 1);
            encoder.setBuffer_offset_atIndex(Some(c.metal_buffer(allocator)), c.offset_bytes, 2);

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
        let block_size = 16;

        let threads_per_threadgroup = MTLSize {
            width: block_size as usize,
            height: block_size as usize,
            depth: 1,
        };

        let threadgroups_per_grid = MTLSize {
            width: ((n + block_size - 1) / block_size) as usize,
            height: ((m + block_size - 1) / block_size) as usize,
            depth: 1,
        };

        unsafe {
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                threadgroups_per_grid,
                threads_per_threadgroup,
            );
        }

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_qk_f16(
        &self,
        ctx: &MetalContext,
        allocator: &MetalAllocator,
        q: &BlockHandle,
        k_cache: &BlockHandle,
        scores: &BlockHandle,
        n_heads: u32,
        head_dim: u32,
        seq_len: u32,
        current_pos: u32,
    ) -> Result<()> {
        let cmd_buf = ctx.command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| MetalError::Internal("failed to create compute encoder".into()))?;

        encoder.setComputePipelineState(&self.attention_qk_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.metal_buffer(allocator)), q.offset_bytes, 0);
            encoder.setBuffer_offset_atIndex(
                Some(k_cache.metal_buffer(allocator)),
                k_cache.offset_bytes,
                1,
            );
            encoder.setBuffer_offset_atIndex(
                Some(scores.metal_buffer(allocator)),
                scores.offset_bytes,
                2,
            );

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

    #[allow(clippy::too_many_arguments)]
    pub fn attention_pv_f16(
        &self,
        ctx: &MetalContext,
        allocator: &MetalAllocator,
        scores: &BlockHandle,
        v_cache: &BlockHandle,
        out: &BlockHandle,
        n_heads: u32,
        seq_len: u32,
        head_dim: u32,
        current_pos: u32,
    ) -> Result<()> {
        let cmd_buf = ctx.command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| MetalError::Internal("failed to create compute encoder".into()))?;

        encoder.setComputePipelineState(&self.attention_pv_pipeline);

        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(scores.metal_buffer(allocator)),
                scores.offset_bytes,
                0,
            );
            encoder.setBuffer_offset_atIndex(
                Some(v_cache.metal_buffer(allocator)),
                v_cache.offset_bytes,
                1,
            );
            encoder.setBuffer_offset_atIndex(
                Some(out.metal_buffer(allocator)),
                out.offset_bytes,
                2,
            );

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

    fn setup() -> (MetalContext, MetalAllocator, MetalKernels) {
        let device = MetalDevice::system_default().unwrap();
        let ctx = MetalContext::new(device).unwrap();
        let alloc = MetalAllocator::new(&ctx, 64 * 1024 * 1024).unwrap();
        let kernels = MetalKernels::new(ctx.device.raw()).unwrap();
        (ctx, alloc, kernels)
    }

    fn rms_norm_ref(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let mean_sq = input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        input
            .iter()
            .zip(weight.iter())
            .map(|(x, w)| x * scale * w)
            .collect()
    }

    fn rope_ref(
        input: &[f32],
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
    ) -> Vec<f32> {
        let mut out = input.to_vec();
        for pos in 0..seq_len {
            for head in 0..n_heads {
                let row = &mut out[(pos * n_heads + head) * head_dim..];
                for i in 0..head_dim / 2 {
                    let freq = 1.0f32 / theta.powf(2.0 * i as f32 / head_dim as f32);
                    let angle = pos as f32 * freq;
                    let (sin_a, cos_a) = angle.sin_cos();
                    let x0 = row[2 * i];
                    let x1 = row[2 * i + 1];
                    row[2 * i] = x0 * cos_a - x1 * sin_a;
                    row[2 * i + 1] = x0 * sin_a + x1 * cos_a;
                }
            }
        }
        out
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
                scores[h * seq_len + t] = sum * scale_fac;
            }
        }
        scores
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
                let e = (scores[h * seq_len + t].to_f32() - max_val).exp();
                sum_exp += e;
                weights[t] = e;
            }
            for t in 0..seq_len {
                weights[t] /= sum_exp;
            }
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for t in 0..seq_len {
                    acc += weights[t] * v_cache[(t * n_heads + h) * head_dim + d].to_f32();
                }
                out[h * head_dim + d] = acc;
            }
        }
        out
    }

    #[test]
    fn test_rms_norm_single_token() {
        let (ctx, mut alloc, kernels) = setup();
        let hidden = 8usize;
        let eps = 1e-5f32;

        let input_f32 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight_f32 = vec![1.0f32; hidden];

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

        let mut output_f32 = vec![0.0f32; hidden];
        unsafe {
            let out_ptr = out_block.ptr as *const f16;
            for i in 0..hidden {
                output_f32[i] = out_ptr.add(i).read().to_f32();
            }
        }

        let expected = rms_norm_ref(&input_f32, &weight_f32, eps);
        for i in 0..hidden {
            assert!((output_f32[i] - expected[i]).abs() < 1e-2);
        }
    }

    #[test]
    fn test_rms_norm_f32_execution() {
        let (ctx, mut alloc, kernels) = setup();
        let hidden = 8usize;
        let eps = 1e-5f32;

        let input_f32 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight_f32 = vec![1.0f32; hidden];

        let block_size = hidden * std::mem::size_of::<f32>();
        let in_block = alloc.alloc(block_size, 16).unwrap();
        let wt_block = alloc.alloc(block_size, 16).unwrap();
        let out_block = alloc.alloc(block_size, 16).unwrap();

        unsafe {
            let in_ptr = in_block.ptr as *mut f32;
            let wt_ptr = wt_block.ptr as *mut f32;
            for i in 0..hidden {
                in_ptr.add(i).write(input_f32[i]);
                wt_ptr.add(i).write(weight_f32[i]);
            }
        }

        kernels
            .rms_norm_f32(
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

        let mut output_f32 = vec![0.0f32; hidden];
        unsafe {
            let out_ptr = out_block.ptr as *const f32;
            for i in 0..hidden {
                output_f32[i] = out_ptr.add(i).read();
            }
        }

        let expected = rms_norm_ref(&input_f32, &weight_f32, eps);
        for i in 0..hidden {
            assert!(
                (output_f32[i] - expected[i]).abs() < 1e-2,
                "index {}: got {}, want {} -- if this fails but \
                 test_rms_norm_single_token (f16) passes, the bug is in the \
                 F32 kernel/pipeline specifically",
                i,
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
                    assert!((got - expected[i]).abs() < 1e-2);
                }
            }
        }
    }

    #[test]
    fn test_rms_norm_weight_scaling() {
        let (ctx, mut alloc, kernels) = setup();
        let hidden = 4usize;
        let eps = 1e-5f32;

        let input_f32 = vec![1.0f32, 1.0, 1.0, 1.0];
        let weight_f32 = vec![2.0f32, 0.5, 1.0, 3.0];

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
                assert!((got - expected[i]).abs() < 1e-2);
            }
        }
    }

    #[test]
    fn test_rope_position_zero_is_identity() {
        let (ctx, mut alloc, kernels) = setup();
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

        kernels
            .rope_f16(
                &ctx,
                &alloc,
                &x_block,
                seq_len as u32,
                n_heads as u32,
                head_dim as u32,
                10000.0,
                0, // offset — these tests exercise prefill (offset=0)
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
        let (ctx, mut alloc, kernels) = setup();
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

        kernels
            .rope_f16(
                &ctx,
                &alloc,
                &x_block,
                seq_len as u32,
                n_heads as u32,
                head_dim as u32,
                10000.0,
                0, // offset — these tests exercise prefill (offset=0)
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
        let (ctx, mut alloc, kernels) = setup();
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

        kernels
            .rope_f16(
                &ctx,
                &alloc,
                &x_block,
                seq_len as u32,
                n_heads as u32,
                head_dim as u32,
                10000.0,
                0, // offset — these tests exercise prefill (offset=0)
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

    #[test]
    fn test_swiglu_f16_execution() {
        let (ctx, mut alloc, kernels) = setup();
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

        kernels
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
                if diff >= 1e-2 {
                    all_ok = false;
                }
            }
            assert!(all_ok, "swiglu values out of tolerance");
        }
    }

    #[test]
    fn test_matmul_f32_execution() {
        let (ctx, mut alloc, kernels) = setup();

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

        kernels
            .matmul_f16(
                &ctx, &alloc, &a_block, &b_block, &c_block, m as u32, n as u32, k as u32,
            )
            .unwrap();

        let expected_c = matmul_ref(&a_f16, &b_f16, m, n, k);

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
        let (ctx, mut alloc, kernels) = setup();
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

        kernels
            .matmul_f16(
                &ctx, &alloc, &a_block, &b_block, &c_block, m as u32, n as u32, k as u32,
            )
            .unwrap();

        let expected_c = matmul_ref(&a_f16, &b_f16, m, n, k);

        unsafe {
            let out_ptr = c_block.ptr as *const f16;
            for i in 0..m * n {
                let got = out_ptr.add(i).read().to_f32();
                assert!((got - expected_c[i]).abs() < 0.5);
            }
        }
    }

    #[test]
    fn test_attention_qk_execution() {
        let (ctx, mut alloc, kernels) = setup();

        let head_dim = 64usize;
        let seq_len = 64usize;
        let n_heads = 64usize;
        let current_pos = 64usize;

        let q_f16: Vec<f16> = (0..(n_heads * head_dim))
            .map(|i| f16::from_f32((i % 10) as f32 * 0.1))
            .collect();
        let k_f16: Vec<f16> = (0..(seq_len * n_heads * head_dim))
            .map(|i| f16::from_f32((i % 10) as f32 * 0.1))
            .collect();
        let scores_f16: Vec<f16> = vec![f16::from_f32(0.0); seq_len * n_heads];

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

        kernels
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
                assert!((got - expected_score[i]).abs() < 0.5);
            }
        }
    }

    #[test]
    fn test_attention_pv_execution() {
        let (ctx, mut alloc, kernels) = setup();

        let head_dim = 64usize;
        let seq_len = 64usize;
        let n_heads = 12usize;
        let current_pos = 63usize;

        let scores_f16: Vec<f16> = (0..(n_heads * seq_len))
            .map(|i| f16::from_f32((i % 10) as f32 * 0.1))
            .collect();
        let v_f16: Vec<f16> = (0..(seq_len * n_heads * head_dim))
            .map(|i| f16::from_f32((i % 5) as f32 * 0.1))
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

        kernels
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
                assert!((got - expected_out[i]).abs() < 0.05);
            }
        }
    }

    #[test]
    fn test_rope_ref_matches_llama_cpp_ground_truth() {
        let head_dim = 8usize;
        let seq_len = 2usize;
        let n_heads = 1usize;
        let mut input = vec![0.0f32; seq_len * n_heads * head_dim];
        input[head_dim] = 0.4288;
        input[head_dim + 1] = -0.2099;

        let out = rope_ref(&input, seq_len, n_heads, head_dim, 500000.0);

        assert!(
            (out[head_dim] - 0.4083).abs() < 1e-3,
            "channel 0 @ pos 1: got {}, want 0.4083 (llama.cpp ground truth)",
            out[head_dim]
        );
        assert!(
            (out[head_dim + 1] - 0.2474).abs() < 1e-3,
            "channel 1 @ pos 1: got {}, want 0.2474 (llama.cpp ground truth)",
            out[head_dim + 1]
        );
    }

    #[test]
    fn test_rope_f16_kernel_matches_llama_cpp_ground_truth() {
        let (ctx, mut alloc, kernels) = setup();
        let n_heads = 1usize;
        let head_dim = 8usize;
        let seq_len = 2usize;

        let mut input_f32 = vec![0.0f32; seq_len * n_heads * head_dim];
        input_f32[head_dim] = 0.4288;
        input_f32[head_dim + 1] = -0.2099;

        let block_size = input_f32.len() * std::mem::size_of::<f16>();
        let x_block = alloc.alloc(block_size, 16).unwrap();
        unsafe {
            let ptr = x_block.ptr as *mut f16;
            for i in 0..input_f32.len() {
                ptr.add(i).write(f16::from_f32(input_f32[i]));
            }
        }

        kernels
            .rope_f16(
                &ctx,
                &alloc,
                &x_block,
                seq_len as u32,
                n_heads as u32,
                head_dim as u32,
                500000.0,
                0, // offset — these tests exercise prefill (offset=0)
            )
            .unwrap();

        unsafe {
            let ptr = x_block.ptr as *const f16;
            let got0 = ptr.add(head_dim).read().to_f32();
            let got1 = ptr.add(head_dim + 1).read().to_f32();
            assert!(
                (got0 - 0.4083).abs() < 1e-2,
                "channel 0 @ pos 1: got {}, want 0.4083",
                got0
            );
            assert!(
                (got1 - 0.2474).abs() < 1e-2,
                "channel 1 @ pos 1: got {}, want 0.2474",
                got1
            );
        }
    }
}
