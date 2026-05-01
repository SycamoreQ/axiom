use cudarc::driver::PushKernelArg;
use cudarc::driver::{CudaView, CudaViewMut, LaunchConfig};
use half::f16;

use crate::cuda::context::CudaContext;
use crate::cuda::error::{CudaError, Result};

/*  Kernel launch wrappers
Each function here is a thin Rust wrapper around one CUDA kernel.
Responsibilities:
1. Compute grid / block dims from runtime shapes
2. Compute dynamic shared memory size if needed
3. Retrieve the CudaFunction from context
4. Call launch with the argument tuple via the device
*/

//rms_norm_f16_kernel
pub fn launch_rms_norm_f16(
    ctx: &CudaContext,
    output: &mut CudaViewMut<f16>,
    input: &CudaView<f16>,
    weight: &CudaView<f16>,
    eps: f32,
    num_tokens: usize,
    hidden_size: usize,
) -> Result<()> {
    let block = hidden_size.min(1024) as u32;
    let grid = num_tokens as u32;
    let smem = (block as usize * std::mem::size_of::<f32>()) as u32;

    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: smem,
    };

    let f = ctx.func("rms_norm_f16_kernel")?;
    unsafe {
        ctx.stream()
            .launch_builder(&f)
            .arg(output)
            .arg(input)
            .arg(weight)
            .arg(eps)
            .arg(hidden_size as i32)
            .launch(cfg)
            .map(|_| ())
    }
    .map_err(CudaError::Driver)
}

//fused_residual_rmsnorm_f16_kernel
pub fn launch_fused_residual_rmsnorm_f16(
    ctx: &CudaContext,
    output: &mut CudaViewMut<f16>,
    residual: &mut CudaViewMut<f16>,
    input: &CudaView<f16>,
    add: &CudaView<f16>,
    weight: &CudaView<f16>,
    eps: f32,
    num_tokens: usize,
    hidden_size: usize,
) -> Result<()> {
    let block = hidden_size.min(1024) as u32;
    let grid = num_tokens as u32;
    let smem = (block as usize * std::mem::size_of::<f32>()) as u32;

    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: smem,
    };

    let f = ctx.func("fused_residual_rmsnorm_f16_kernel")?;
    unsafe {
        ctx.stream()
            .launch_builder(&f)
            .arg(output)
            .arg(residual)
            .arg(input)
            .arg(add)
            .arg(weight)
            .arg(eps)
            .arg(hidden_size as i32)
            .launch(cfg)
            .map(|_| ())
    }
    .map_err(CudaError::Driver)
}

//rotary_embedding_f16_kernel
pub fn launch_rotary_embedding_f16(
    ctx: &CudaContext,
    query: &mut CudaViewMut<f16>,
    key: &mut CudaViewMut<f16>,
    cos_cache: &CudaView<f32>,
    sin_cache: &CudaView<f32>,
    positions: &CudaView<i32>,
    num_tokens: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (num_tokens as u32, num_heads as u32, 1),
        block_dim: (head_dim as u32 / 2, 1, 1),
        shared_mem_bytes: 0,
    };

    let f = ctx.func("rotary_embedding_f16_kernel")?;
    unsafe {
        ctx.stream()
            .launch_builder(&f)
            .arg(query)
            .arg(key)
            .arg(cos_cache)
            .arg(sin_cache)
            .arg(positions)
            .arg(num_tokens as i32)
            .arg(num_heads as i32)
            .arg(num_kv_heads as i32)
            .arg(head_dim as i32)
            .launch(cfg)
            .map(|_| ())
    }
    .map_err(CudaError::Driver)
}

//reshape_and_cache_f16io_kernel
pub fn launch_reshape_and_cache_f16(
    ctx: &CudaContext,
    key_cache: &mut CudaViewMut<f16>,
    value_cache: &mut CudaViewMut<f16>,
    key: &CudaView<f16>,
    value: &CudaView<f16>,
    slot_mapping: &CudaView<i32>,
    num_tokens: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let block = (num_kv_heads * head_dim).min(1024) as u32;
    let cfg = LaunchConfig {
        grid_dim: (num_tokens as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    let f = ctx.func("reshape_and_cache_f16io_kernel")?;
    unsafe {
        ctx.stream()
            .launch_builder(&f)
            .arg(key_cache)
            .arg(value_cache)
            .arg(key)
            .arg(value)
            .arg(slot_mapping)
            .arg(num_tokens as i32)
            .arg(num_kv_heads as i32)
            .arg(head_dim as i32)
            .launch(cfg)
            .map(|_| ())
    }
    .map_err(CudaError::Driver)
}

//copy_blocks_f16_kernel
pub fn launch_copy_blocks_f16(
    ctx: &CudaContext,
    key_cache: &mut CudaViewMut<f16>,
    value_cache: &mut CudaViewMut<f16>,
    block_mapping: &CudaView<i64>,
    num_pairs: usize,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let block = (block_size * num_kv_heads * head_dim).min(1024) as u32;
    let cfg = LaunchConfig {
        grid_dim: (num_pairs as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    let f = ctx.func("copy_blocks_f16_kernel")?;
    unsafe {
        ctx.stream()
            .launch_builder(&f)
            .arg(key_cache)
            .arg(value_cache)
            .arg(block_mapping)
            .arg(num_pairs as i32)
            .arg(block_size as i32)
            .arg(num_kv_heads as i32)
            .arg(head_dim as i32)
            .launch(cfg)
            .map(|_| ())
    }
    .map_err(CudaError::Driver)
}

//embedding_gather_f16_kernel
pub fn launch_embedding_gather_f16(
    ctx: &CudaContext,
    output: &mut CudaViewMut<f16>,
    embed_table: &CudaView<f16>,
    token_ids: &CudaView<i32>,
    num_tokens: usize,
    hidden_size: usize,
    vocab_size: usize,
) -> Result<()> {
    let block = hidden_size.min(1024) as u32;
    let cfg = LaunchConfig {
        grid_dim: (num_tokens as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    let f = ctx.func("embedding_gather_f16_kernel")?;
    unsafe {
        ctx.stream()
            .launch_builder(&f)
            .arg(output)
            .arg(embed_table)
            .arg(token_ids)
            .arg(hidden_size as i32)
            .arg(vocab_size as i32)
            .launch(cfg)
            .map(|_| ())
    }
    .map_err(CudaError::Driver)
}

//argmax_f16_kernel
pub fn launch_argmax_f16(
    ctx: &CudaContext,
    output: &mut CudaViewMut<i32>,
    logits: &CudaView<f16>,
    num_tokens: usize,
    vocab_size: usize,
) -> Result<()> {
    let block = vocab_size.min(1024) as u32;
    let cfg = LaunchConfig {
        grid_dim: (num_tokens as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    let f = ctx.func("argmax_f16_kernel")?;
    unsafe {
        ctx.stream()
            .launch_builder(&f)
            .arg(logits)
            .arg(output)
            .arg(vocab_size as i32)
            .launch(cfg)
            .map(|_| ())
    }
    .map_err(CudaError::Driver)
}

//flash_attention_3_decode_f16io_kernel
pub fn launch_flash_attention_3(
    ctx: &CudaContext,
    output: &mut CudaViewMut<f16>,
    query: &CudaView<f16>,
    key_cache: &CudaView<f16>,
    value_cache: &CudaView<f16>,
    block_tables: &CudaView<i32>,
    context_lens: &CudaView<i32>,
    scale: f32,
    num_seqs: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks_per_seq: usize,
) -> Result<()> {
    const FA3_BC: usize = 64;
    const FA3_THREADS: u32 = 256;

    let smem = (FA3_BC * (head_dim + 2) * std::mem::size_of::<u16>() * 2
        + FA3_BC * std::mem::size_of::<f32>()
        + 8 * std::mem::size_of::<f32>()) as u32;

    let cfg = LaunchConfig {
        grid_dim: (num_seqs as u32, num_heads as u32, 1),
        block_dim: (FA3_THREADS, 1, 1),
        shared_mem_bytes: smem,
    };

    let f = ctx.func("flash_attention_3_decode_f16io_kernel")?;
    unsafe {
        ctx.stream()
            .launch_builder(&f)
            .arg(output)
            .arg(query)
            .arg(key_cache)
            .arg(value_cache)
            .arg(block_tables)
            .arg(context_lens)
            .arg(scale)
            .arg(num_heads as i32)
            .arg(num_kv_heads as i32)
            .arg(head_dim as i32)
            .arg(block_size as i32)
            .arg(max_blocks_per_seq as i32)
            .launch(cfg)
            .map(|_| ())
    }
    .map_err(CudaError::Driver)
}

//flash_attention_3_decode_gqa_f16io_kernel
pub fn launch_flash_attention_3_gqa(
    ctx: &CudaContext,
    output: &mut CudaViewMut<f16>,
    query: &CudaView<f16>,
    key_cache: &CudaView<f16>,
    value_cache: &CudaView<f16>,
    block_tables: &CudaView<i32>,
    context_lens: &CudaView<i32>,
    scale: f32,
    num_seqs: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_context_len: usize,
    max_blocks_per_seq: usize,
) -> Result<()> {
    const FA3_BC: usize = 64;
    const FA3_THREADS: u32 = 256;

    let heads_per_group = num_heads / num_kv_heads;

    let smem = (FA3_BC * (head_dim + 2) * std::mem::size_of::<u16>() * 2
        + heads_per_group * (FA3_BC + 1) * std::mem::size_of::<f32>()
        + 8 * std::mem::size_of::<f32>()) as u32;

    let cfg = LaunchConfig {
        grid_dim: (num_seqs as u32, num_kv_heads as u32, 1),
        block_dim: (FA3_THREADS, 1, 1),
        shared_mem_bytes: smem,
    };

    let f = ctx.func("flash_attention_3_decode_gqa_f16io_kernel")?;
    unsafe {
        ctx.stream()
            .launch_builder(&f)
            .arg(output)
            .arg(query)
            .arg(key_cache)
            .arg(value_cache)
            .arg(block_tables)
            .arg(context_lens)
            .arg(scale)
            .arg(num_heads as i32)
            .arg(num_kv_heads as i32)
            .arg(head_dim as i32)
            .arg(block_size as i32)
            .arg(max_context_len as i32)
            .arg(max_blocks_per_seq as i32)
            .launch(cfg)
            .map(|_| ())
    }
    .map_err(CudaError::Driver)
}

//residual_attention_decode_f16io_kernel
pub fn launch_residual_attention(
    ctx: &CudaContext,
    output: &mut CudaViewMut<f16>,
    query: &CudaView<f16>,
    b_key_cache: &CudaView<f16>,
    b_val_cache: &CudaView<f16>,
    b_block_table: &CudaView<i32>,
    base_context_len: usize,
    r_key_cache: &CudaView<f16>,
    r_val_cache: &CudaView<f16>,
    r_block_table: &CudaView<i32>,
    residual_context_len: usize,
    scale: f32,
    num_seqs: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_base_blocks: usize,
    max_residual_blocks: usize,
) -> Result<()> {
    const RA_BC: usize = 64;
    const RA_THREADS: u32 = 256;

    let heads_per_group = num_heads / num_kv_heads;
    let kv_stride = head_dim + 2;

    let smem = (RA_BC * kv_stride * std::mem::size_of::<u16>()
        + heads_per_group * (RA_BC + 1) * std::mem::size_of::<f32>()
        + 8 * std::mem::size_of::<f32>()) as u32;

    let cfg = LaunchConfig {
        grid_dim: (num_seqs as u32, num_kv_heads as u32, 1),
        block_dim: (RA_THREADS, 1, 1),
        shared_mem_bytes: smem,
    };

    let f = ctx.func("residual_attention_decode_f16io_kernel")?;
    unsafe {
        ctx.stream()
            .launch_builder(&f)
            .arg(output)
            .arg(query)
            .arg(b_key_cache)
            .arg(b_val_cache)
            .arg(b_block_table)
            .arg(base_context_len as i32)
            .arg(r_key_cache)
            .arg(r_val_cache)
            .arg(r_block_table)
            .arg(residual_context_len as i32)
            .arg(scale)
            .arg(num_heads as i32)
            .arg(num_kv_heads as i32)
            .arg(head_dim as i32)
            .arg(block_size as i32)
            .arg(max_base_blocks as i32)
            .arg(max_residual_blocks as i32)
            .launch(cfg)
            .map(|_| ())
    }
    .map_err(CudaError::Driver)
}

//flash_attention_4_decode_f16io_kernel
pub fn launch_flash_attention_4(
    ctx: &CudaContext,
    output: &mut CudaViewMut<f16>,
    query: &CudaView<f16>,
    key_cache: &CudaView<f16>,
    value_cache: &CudaView<f16>,
    block_tables: &CudaView<i32>,
    context_lens: &CudaView<i32>,
    d_tile_counter: &mut CudaViewMut<i32>,
    scale: f32,
    num_seqs: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks_per_seq: usize,
    sm_count: usize,
) -> Result<()> {
    const FA4_BC: usize = 64;
    const FA4_KV_PAD: usize = 2;
    const FA4_PIPE_DEPTH: usize = 2;
    const FA4_GQA_MAX_HPG: usize = 8;
    const FA4_WARPS: usize = 16;
    const FA4_SCORE_PAD: usize = 1;

    let stage_size = FA4_BC * (128 + FA4_KV_PAD) * std::mem::size_of::<u16>() * 2;

    let smem = FA4_PIPE_DEPTH * stage_size
        + FA4_GQA_MAX_HPG * 128 * std::mem::size_of::<u16>()
        + FA4_GQA_MAX_HPG * (FA4_BC + FA4_SCORE_PAD) * std::mem::size_of::<f32>()
        + FA4_WARPS * std::mem::size_of::<f32>()
        + std::mem::size_of::<i32>();

    let cfg = LaunchConfig {
        grid_dim: ((sm_count * 2) as u32, 1, 1),
        block_dim: (512, 1, 1),
        shared_mem_bytes: smem as u32,
    };

    ctx.device()
        .memset_zeros(d_tile_counter)
        .map_err(CudaError::Driver)?;

    let f = ctx.func("flash_attention_4_decode_f16io_kernel")?;
    unsafe {
        ctx.stream()
            .launch_builder(&f)
            .arg(output)
            .arg(query)
            .arg(key_cache)
            .arg(value_cache)
            .arg(block_tables)
            .arg(context_lens)
            .arg(d_tile_counter)
            .arg(scale)
            .arg(num_seqs as i32)
            .arg(num_heads as i32)
            .arg(num_kv_heads as i32)
            .arg(head_dim as i32)
            .arg(block_size as i32)
            .arg(max_blocks_per_seq as i32)
            .launch(cfg)
            .map(|_| ())
    }
    .map_err(CudaError::Driver)
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::context::CudaContext;

    fn try_ctx() -> Option<CudaContext> {
        let ptx = std::fs::read_to_string(env!("AXIOM_KERNELS_PTX")).ok()?;
        CudaContext::new(0, &ptx).ok()
    }

    #[test]
    fn test_context_loads_all_kernels() {
        let Some(ctx) = try_ctx() else { return };

        let kernel_names = [
            "rms_norm_f16_kernel",
            "fused_residual_rmsnorm_f16_kernel",
            "rotary_embedding_f16_kernel",
            "reshape_and_cache_f16io_kernel",
            "copy_blocks_f16_kernel",
            "embedding_gather_f16_kernel",
            "argmax_f16_kernel",
            "flash_attention_3_decode_f16io_kernel",
            "flash_attention_3_decode_gqa_f16io_kernel",
            "residual_attention_decode_f16io_kernel",
            "flash_attention_4_decode_f16io_kernel",
        ];

        for &name in kernel_names.iter() {
            assert!(ctx.func(name).is_ok(), "failed to load kernel: {}", name);
        }
    }

    #[test]
    fn test_synchronize_no_work() {
        let Some(ctx) = try_ctx() else { return };
        assert!(ctx.synchronize().is_ok());
    }

    #[test]
    fn test_ordinal_is_zero() {
        let Some(ctx) = try_ctx() else { return };
        assert_eq!(ctx.ordinal(), 0);
    }

    #[test]
    fn test_unknown_kernel_returns_err() {
        let Some(ctx) = try_ctx() else { return };
        assert!(ctx.func("does_not_exist").is_err());
    }

    #[test]
    fn test_context_no_device() {
        let result = CudaContext::new(99, ".version 7.0\n.target sm_80\n.address_size 64\n");

        assert!(result.is_err());
    }
}
