// flash_attention_4.cu
// FlashAttention-4: Persistent producer/consumer kernel with TMA K/V loads
//
// Thread / warpgroup layout (512 threads = 4 warpgroups of 128):
//   WG 0-1  : producers  — issue TMA / cp.async K and V tile loads
//   WG 2-3  : consumers  — QK^T, online softmax, P@V
//
// Ping-pong double buffer (FA4_PIPE_DEPTH = 2):
//   Producer fills stage (tile+1) % 2 while consumer processes stage tile % 2.
//   Synchronised via cuda::pipeline arrive/wait tokens.
//
// Persistent scheduling:
//   Each block atomicAdds into d_tile_counter to claim the next
//   (seq, kv_head) work item. Blocks loop until the counter exceeds
//   num_seqs * num_kv_heads.
//
// Shared memory layout (two copies, one per pipeline stage):
//   stage[s].s_k     [FA4_BC * (head_dim + FA4_KV_PAD)]   __half
//   stage[s].s_v     [FA4_BC * (head_dim + FA4_KV_PAD)]   __half
//   s_q              [FA4_GQA_MAX_HPG * head_dim]           __half  (loaded once)
//   s_scores         [FA4_GQA_MAX_HPG * FA4_SCORE_STRIDE]  float
//   s_warp           [FA4_WARPS]                            float
//
// References:
//   Dao et al., FlashAttention-3 (2024)
//   NVIDIA Hopper TMA White Paper
//   NVIDIA Blackwell Architecture Technical Brief

#include <cfloat>
#include <float.h>
#include <cuda_fp16.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

#if __CUDA_ARCH__ >= 900
    #include <cuda/barrier>
#endif

#define FA4_BC           64       // KV tile width
#define FA4_THREADS      512      // threads per block
#define FA4_WARPS        16       // FA4_THREADS / 32
#define FA4_WGS          4        // warpgroups per block
#define FA4_PROD_WGS     2        // producer warpgroups (WG 0-1)
#define FA4_CONS_WGS     2        // consumer warpgroups (WG 2-3)
#define FA4_PIPE_DEPTH   2        // pipeline stages (double buffer)
#define FA4_GQA_MAX_HPG  8        // max query heads per KV head
#define FA4_KV_PAD       2        // smem bank-conflict padding, KV rows
#define FA4_SCORE_PAD    1        // smem bank-conflict padding, score rows
#define FA4_SCORE_STRIDE (FA4_BC + FA4_SCORE_PAD)

#define FA4_WG_ID        (threadIdx.x / 128)
#define FA4_IS_PRODUCER  (FA4_WG_ID < FA4_PROD_WGS)
#define FA4_IS_CONSUMER  (FA4_WG_ID >= FA4_PROD_WGS)
#define FA4_WARP_ID      (threadIdx.x / 32)
#define FA4_LANE_ID      (threadIdx.x % 32)
// Warp id local to the consumer warpgroups (0..7)
#define FA4_CONS_WARP_ID (FA4_WARP_ID - FA4_PROD_WGS * 4)

// Shared memory layout
// Declared as a struct so the compiler can reason about alignment.
// Instantiated as a static extern __shared__ array in the kernel.
struct FA4Stage {
    __half k[FA4_BC][128 + FA4_KV_PAD];   // [tile_pos][head_dim], dim capped at 128
    __half v[FA4_BC][128 + FA4_KV_PAD];
};

struct FA4Smem {
    FA4Stage  stages[FA4_PIPE_DEPTH];                              // double buffer
    __half    q[FA4_GQA_MAX_HPG][128];                             // Q for all heads in group
    float     scores[FA4_GQA_MAX_HPG][FA4_SCORE_STRIDE];          // per-head scores
    float     warp[FA4_WARPS];                                     // warp reduction scratch
    int       pipe_tokens[FA4_PIPE_DEPTH];                         // arrive/wait tokens
};


// Architecture-dispatched KV tile loader
//
// On sm_89:  uses cp.async.bulk (no TMA descriptor)
// On sm_90+: uses TMA descriptor issued by the producer warpgroup
//
// Signature is identical across architectures so the consumer side
// never needs to know which path was taken.
//
// Parameters:
//   stage    — which ping-pong buffer to fill (0 or 1)
//   tile     — tile index within the sequence (0-based)
//   is_key   — true: load K, false: load V
//   smem     — pointer to FA4Smem
//   cache    — global K or V cache pointer
//   block_table — physical block indices for this seq
//   kv_head_idx, tile_start, tile_len, head_dim, num_kv_heads, block_size

// Only producer warpgroup threads call this.
//
#if __CUDA_ARCH__ >= 900
    __device__ __forceinline__ void fa4_producer_load_tile(
        int            stage,
        bool           is_key,
        FA4Smem*       smem,
        const __half*  cache,   // K or V cache: [num_blocks, block_size, num_kv_heads, head_dim]
        const int*     block_table, // [max_blocks_per_seq] for this seq
        int            kv_head_idx,
        int            tile_start,
        int            tile_len,
        int            head_dim,
        int            num_kv_heads,
        int            block_size,
        int            wg_tid,
        cuda::pipeline<cuda::thread_scope_block>& pipe   // passed from kernel
    ) {
        __half (*dst)[128 + FA4_KV_PAD] = is_key
            ? smem->stages[stage].k
            : smem->stages[stage].v;

        pipe.producer_acquire();

        for (int t = wg_tid; t < tile_len; t += 128) {
            int kv_pos   = tile_start + t;
            int page_idx = kv_pos / block_size;
            int page_off = kv_pos % block_size;
            int phys     = block_table[page_idx];               // was: cache[page_id] — wrong pointer
            const __half* src = &cache[
                ((phys * block_size + page_off) * num_kv_heads  // was: phys + block_size + page_off
                    + kv_head_idx) * head_dim
            ];
            cuda::memcpy_async(
                dst[t],
                src,
                cuda::aligned_size_t<16>(head_dim * sizeof(__half)),
                pipe
            );
        }

        pipe.producer_commit();
    }

#else

    // sm_89 variant — uses cp.async.cg, no pipeline object needed.
    __device__ __forceinline__ void fa4_producer_load_tile(
        int            stage,
        bool           is_key,
        FA4Smem*       smem,
        const __half*  cache,
        const int*     block_table,
        int            kv_head_idx,
        int            tile_start,
        int            tile_len,
        int            head_dim,
        int            num_kv_heads,
        int            block_size,
        int            wg_tid
    ) {
        __half (*dst)[128 + FA4_KV_PAD] = is_key
            ? smem->stages[stage].k
            : smem->stages[stage].v;

        for (int t = wg_tid; t < tile_len; t += 128) {
            int kv_pos   = tile_start + t;
            int page_idx = kv_pos / block_size;
            int page_off = kv_pos % block_size;
            int phys     = block_table[page_idx];

            const __half* src_ptr = &cache[
                ((phys * block_size + page_off) * num_kv_heads
                    + kv_head_idx) * head_dim
            ];

            // __cvta_generic_to_shared converts the smem pointer to the 32-bit
            // address form required by the cp.async inline asm "r" constraint.
            uint32_t smem_addr = __cvta_generic_to_shared(&dst[t][0]);

            // cp.async.cg requires 16 or 32 byte transfers.
            // Each iteration: 16 bytes = 8 x __half. Stride d += 8.
            #pragma unroll
            for (int d = 0; d < head_dim; d += 8) {
                uint32_t dst_addr = smem_addr + (uint32_t)(d * sizeof(__half));
                const __half* src = src_ptr + d;
                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], 16;"
                    :
                    : "r"(dst_addr), "l"(src)
                    : "memory"
                );
            }
        }

        asm volatile("cp.async.commit_group;");
    }

#endif

// Pipeline wait helper (consumer side)
// Blocks the consumer warpgroup until the producer has committed
// the tile for the given stage into shared memory.
#if __CUDA_ARCH__ >= 900

    __device__ __forceinline__ void fa4_consumer_wait(
        int stage,
        cuda::pipeline<cuda::thread_scope_block>& pipe
    ) {
        pipe.consumer_wait();
        __syncwarp();
        (void)stage;
    }

    __device__ __forceinline__ void fa4_consumer_arrive(
        int stage, cuda::pipeline<cuda::thread_scope_block>& pipe
    ) {
        pipe.consumer_release();
        (void)stage;
    }

#else

    __device__ __forceinline__ void fa4_consumer_wait(int stage) {
        asm volatile("cp.async.wait_group %0;" :: "n"(FA4_PIPE_DEPTH - 1));
        __syncwarp();
        (void)stage;
    }

    __device__ __forceinline__ void fa4_consumer_arrive(int stage) {
        (void)stage;
    }

#endif



// Q loader (consumer side, runs once per (seq, kv_head) work item)
// All query heads in the GQA group are loaded into smem->q[g][d].
// This amortises the global memory read across the entire tile loop.
// Only consumer threads participate.

__device__ __forceinline__ void fa4_load_q(
    FA4Smem*       smem,
    const __half*  query,
    int            seq_idx,
    int            kv_head_idx,
    int            num_heads,
    int            num_kv_heads,
    int            head_dim,
    int            cons_tid
) {
    const int heads_per_grp = num_heads / num_kv_heads;
    for (int g = 0; g < heads_per_grp; g++) {
        const int g_head = kv_head_idx * heads_per_grp + g;
        const int q_base = (seq_idx * num_heads + g_head) * head_dim;
        for (int d = cons_tid; d < head_dim; d += FA4_CONS_WGS * 128) {
            smem->q[g][d] = query[q_base + d];
        }
    }
    __syncthreads();
}

// QK^T dot product (consumer side, one tile, one head)
// Reads Q from smem->q[g] and K from smem->stages[stage].k.
// Each consumer warp handles FA4_CONS_WGS warp positions (8 positions/round).
// Writes dot products into smem->scores[g][t].
// This is structurally identical to ra_qk_dot_tile / fa3 QK^T but reads
// Q from shared memory rather than registers, since Q is shared across
// multiple tile iterations in the persistent loop.

__device__ __forceinline__ void fa4_qk_dot(
    FA4Smem*  smem,
    int       stage,
    int       g,
    int       tile_len,
    int       head_dim,
    int       cons_warp_id,
    int       lane_id
) {
    const int half2_iters = (head_dim + 63) / 64;

    for (int base_t = 0; base_t < tile_len; base_t += FA4_CONS_WGS * 4) {
        int t = base_t + cons_warp_id;
        if (t < tile_len) {
            float dot = 0.0f;
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                if (r >= half2_iters) break;
                int d = lane_id * 2 + r * 64;
                if (d < head_dim)
                    dot += __half2float(smem->q[g][d]) *
                           __half2float(smem->stages[stage].k[t][d]);
                if (d + 1 < head_dim)
                    dot += __half2float(smem->q[g][d + 1]) *
                           __half2float(smem->stages[stage].k[t][d + 1]);
            }
            dot = fa4_warp_sum(dot);
            if (lane_id == 0) smem->scores[g][t] = dot;
        }
    }
}

// Online softmax (consumer side, one tile, one head)
// Reads smem->scores[g][0..tile_len), updates row_max / row_sum / acc,
// writes softmax weights back into smem->scores[g].
// The correction-before-update pattern is identical to FA3 and
// residual_attention: guard row_max > -FLT_MAX on first tile.

__device__ __forceinline__ void fa4_online_softmax(
    float&    row_max,
    float&    row_sum,
    float     acc[4],
    FA4Smem*  smem,
    int       g,
    int       tile_len,
    int       head_dim,
    int       tid,
    int       lane_id,
    int       cons_warp_id,
    float*    s_warp
) {
    const int acc_dims = (head_dim + 255) / 256;

    float tile_max = fa4_block_reduce_max(
        (tid < tile_len) ? smem->scores[g][tid] : -FLT_MAX,
        tid, lane_id, cons_warp_id, s_warp
    );

    float new_max = fmaxf(row_max, tile_max);

    if (new_max > row_max && row_max > -FLT_MAX) {
        float correction = expf(row_max - new_max);
        #pragma unroll
        for (int r = 0; r < acc_dims && r < 4; r++)
            acc[r] *= correction;
        row_sum *= correction;
    }
    row_max = new_max;

    float my_exp = (tid < tile_len) ? expf(smem->scores[g][tid] - new_max) : 0.0f;
    if (tid < tile_len) smem->scores[g][tid] = my_exp;

    row_sum += fa4_block_reduce_sum(my_exp, tid, lane_id, cons_warp_id, s_warp);
}

// P @ V accumulation (consumer side, one tile, one head)
// smem->scores[g] holds softmax weights after fa4_online_softmax.
// smem->stages[stage].v holds the value tile.
// V is reused across all GQA heads (same optimization as FA3 GQA kernel).

__device__ __forceinline__ void fa4_pv_accumulate(
    float     acc[4],
    FA4Smem*  smem,
    int       stage,
    int       g,
    int       tile_len,
    int       head_dim,
    int       cons_tid
) {
    const int acc_dims = (head_dim + 255) / 256;

    #pragma unroll
    for (int r = 0; r < acc_dims && r < 4; r++) {
        int d = cons_tid + r * 256;
        if (d < head_dim) {
            float v_acc = 0.0f;
            for (int t = 0; t < tile_len; t++) {
                v_acc += smem->scores[g][t] *
                         __half2float(smem->stages[stage].v[t][d]);
            }
            acc[r] += v_acc;
        }
    }
}


// Output writer (consumer side)
// Normalizes acc by row_sum, writes f16 to output buffer.
// output layout: [num_seqs, num_heads, head_dim]

__device__ __forceinline__ void fa4_write_output(
    __half*      output,
    const float  acc[4],
    float        row_sum,
    int          seq_idx,
    int          g_head_idx,
    int          num_heads,
    int          head_dim,
    int          cons_tid
) {
    const int   acc_dims = (head_dim + 255) / 256;
    const int   out_base = (seq_idx * num_heads + g_head_idx) * head_dim;
    const float inv_sum  = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    #pragma unroll
    for (int r = 0; r < acc_dims && r < 4; r++) {
        int d = cons_tid + r * 256;
        if (d < head_dim) {
            output[out_base + d] = __float2half(acc[r] * inv_sum);
        }
    }
}

__device__ __forceinline__ float fa4_warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float fa4_warp_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

// Block-level reduction over the FA4_WARPS warp slots.
// Uses smem->warp[] as scratch — only consumer threads call this.
__device__ __forceinline__ float fa4_block_reduce_max(
    float val, int tid, int lane_id, int warp_id, float* s_warp
) {
    val = fa4_warp_max(val);
    if (lane_id == 0) s_warp[warp_id] = val;
    __syncthreads();
    if (tid == 0) {
        float m = s_warp[0];
        #pragma unroll
        for (int w = 1; w < FA4_WARPS; w++) m = fmaxf(m, s_warp[w]);
        s_warp[0] = m;
    }
    __syncthreads();
    return s_warp[0];
}

__device__ __forceinline__ float fa4_block_reduce_sum(
    float val, int tid, int lane_id, int warp_id, float* s_warp
) {
    val = fa4_warp_sum(val);
    if (lane_id == 0) s_warp[warp_id] = val;
    __syncthreads();
    if (tid == 0) {
        float s = s_warp[0];
        #pragma unroll
        for (int w = 1; w < FA4_WARPS; w++) s += s_warp[w];
        s_warp[0] = s;
    }
    __syncthreads();
    return s_warp[0];
}


// Main kernel: flash_attention_4_decode_f16io_kernel
//
// Persistent: each block loops over work items claimed from d_tile_counter.
// One work item = one (seq_idx, kv_head_idx) pair.
//
// Producer warpgroups (WG 0-1):
//   Loop over tiles, issue async K and V loads into the next pipeline stage,
//   commit the pipeline stage, then wait for the consumer to release it.
//
// Consumer warpgroups (WG 2-3):
//   Wait for producer to commit each stage, process QK^T + softmax + P@V,
//   release the stage back to the producer.
//   After all tiles: write normalized output.
//
// Synchronization between producer and consumer uses cuda::pipeline
// (sm_90+) or cp.async.wait_group (sm_89). Within each warpgroup,
// __syncwarp() is sufficient. Between warpgroups, __syncthreads() is
// used at stage boundaries.

extern "C"
__global__ void __launch_bounds__(FA4_THREADS, 1)
flash_attention_4_decode_f16io_kernel(
    __half* __restrict__       output,           // [num_seqs, num_heads, head_dim]

    const __half* __restrict__ query,            // [num_seqs, num_heads, head_dim]
    const __half* __restrict__ key_cache,        // [num_blocks, block_size, num_kv_heads, head_dim]
    const __half* __restrict__ value_cache,      // [num_blocks, block_size, num_kv_heads, head_dim]
    const int*    __restrict__ block_tables,     // [num_seqs, max_blocks_per_seq]
    const int*    __restrict__ context_lens,     // [num_seqs]

    int*          __restrict__ d_tile_counter,   // device-side global work counter (init to 0)

    float scale,
    int   num_seqs,
    int   num_heads,
    int   num_kv_heads,
    int   head_dim,
    int   block_size,
    int   max_blocks_per_seq
) {
    const int tid           = threadIdx.x;
    const int wg_id         = FA4_WG_ID;
    const int is_producer   = FA4_IS_PRODUCER;
    const int is_consumer   = FA4_IS_CONSUMER;
    const int warp_id       = FA4_WARP_ID;
    const int lane_id       = FA4_LANE_ID;
    const int wg_tid        = tid % 128;         // thread index within warpgroup
    const int cons_tid      = tid - FA4_PROD_WGS * 128;  // 0..255 for consumers, <0 for producers
    const int cons_warp_id  = warp_id - FA4_PROD_WGS * 4;

    const int heads_per_grp = num_heads / num_kv_heads;
    const int total_work    = num_seqs * num_kv_heads;

    extern __shared__ char smem_raw[];
    FA4Smem* smem = reinterpret_cast<FA4Smem*>(smem_raw);

    #if __CUDA_ARCH__ >= 900
        __shared__ cuda::pipeline_shared_state<
            cuda::thread_scope_block, FA4_PIPE_DEPTH> pipe_state;
        auto pipe = cuda::make_pipeline(
            cooperative_groups::this_thread_block(), &pipe_state);
    #endif

    while (true) {
        // Claim the next work item
        int work_idx = -1;
        if (tid == 0) {
            work_idx = atomicAdd(d_tile_counter, 1);
        }
        // Broadcast work_idx to all threads in the block
        //
        smem->pipe_tokens[0] = work_idx at tid==0,
        __syncthreads(), work_idx = smem->pipe_tokens[0];

        if (work_idx >= total_work) break;

        const int seq_idx = work_idx / num_kv_heads;
        const int kv_head_idx = work_idx % num_kv_heads;
        const int ctx_len = context_lens[seq_idx];

        if (ctx_len == 0) {__syncthreads(); continue};

        const int num_tiles = (ctx_len + FA4_BC - 1) / FA4_BC;
        const int* blk_table = block_tables + seq_idx * max_blocks_per_seq;

        float q_row_max[FA4_GQA_MAX_HPG];
        float q_row_sum[FA4_GQA_MAX_HPG];
        float q_acc    [FA4_GQA_MAX_HPG][4];

        if (is_consumer) {
            fa4_load_q(smem, query, seq_idx, kv_head_idx,
                       num_heads, num_kv_heads, head_dim, cons_tid);
            for (int g = 0; g < heads_per_grp && g < FA4_GQA_MAX_HPG; g++) {
                q_row_max[g] = -FLT_MAX;
                q_row_sum[g] = 0.0f;
                #pragma unroll
                for (int r = 0; r < 4; r++) q_acc[g][r] = 0.0f;
            }
        }
        __syncthreads();

        for (int tile = 0; tile < num_tiles; tile++) {
            const int stage      = tile % FA4_PIPE_DEPTH;
            const int tile_start = tile * FA4_BC;
            const int tile_len   = min(FA4_BC, ctx_len - tile_start);

            if (is_producer) {
                #if __CUDA_ARCH__ >= 900
                    fa4_producer_load_tile(stage, true,  smem, key_cache,
                        blk_table, kv_head_idx, tile_start, tile_len,
                        head_dim, num_kv_heads, block_size, wg_tid, pipe);
                    fa4_producer_load_tile(stage, false, smem, value_cache,
                        blk_table, kv_head_idx, tile_start, tile_len,
                        head_dim, num_kv_heads, block_size, wg_tid, pipe);
                #else
                    fa4_producer_load_tile(stage, true,  smem, key_cache,
                        blk_table, kv_head_idx, tile_start, tile_len,
                        head_dim, num_kv_heads, block_size, wg_tid);
                    fa4_producer_load_tile(stage, false, smem, value_cache,
                        blk_table, kv_head_idx, tile_start, tile_len,
                        head_dim, num_kv_heads, block_size, wg_tid);
                #endif
            }

            if (is_consumer) {
                #if __CUDA_ARCH__ >= 900
                    fa4_consumer_wait(stage, pipe);
                #else
                    fa4_consumer_wait(stage);
                #endif

                for (int g = 0; g < heads_per_grp && g < FA4_GQA_MAX_HPG; g++) {
                    fa4_qk_dot(smem, stage, g, tile_len, head_dim,
                               cons_warp_id, lane_id);
                }
                __syncthreads();

                for (int g = 0; g < heads_per_grp && g < FA4_GQA_MAX_HPG; g++) {
                    fa4_online_softmax(q_row_max[g], q_row_sum[g], q_acc[g],
                                       smem, g, tile_len, head_dim,
                                       cons_tid, lane_id, cons_warp_id,
                                       smem->warp);
                }

                #if __CUDA_ARCH__ >= 900
                    fa4_consumer_arrive(stage, pipe);
                #else
                    fa4_consumer_arrive(stage);
                #endif

                for (int g = 0; g < heads_per_grp && g < FA4_GQA_MAX_HPG; g++) {
                    fa4_pv_accumulate(q_acc[g], smem, stage, g,
                                      tile_len, head_dim, cons_tid);
                }
            }

            __syncthreads();

        }

        if (is_consumer) {
            for (int g = 0; g < heads_per_grp && g < FA4_GQA_MAX_HPG; g++) {
                const int g_head = kv_head_idx * heads_per_grp + g;
                fa4_write_output(output, q_acc[g], q_row_sum[g],
                                 seq_idx, g_head, num_heads, head_dim,
                                 cons_tid);
            }
        }

        __syncthreads();

    }
}

// Host-side launch helper
// (call from cudarc Rust bindings in Arc 2)
//
// Notes:
//   - d_tile_counter must be zeroed before each call (cudaMemset to 0)
//   - smem size is sizeof(FA4Smem); pass to <<< >>> as third argument
//   - grid dim: enough blocks to saturate the SM count.
//     A good starting point: min(num_seqs * num_kv_heads, sm_count * 2)
//   - For sm_90+ set cudaFuncAttributeMaxDynamicSharedMemorySize if
//     sizeof(FA4Smem) > 48KB
//
// void launch_flash_attention_4(
//     __half*       output,
//     const __half* query,
//     const __half* key_cache,
//     const __half* value_cache,
//     const int*    block_tables,
//     const int*    context_lens,
//     int*          d_tile_counter,
//     float scale,
//     int num_seqs, int num_heads, int num_kv_heads,
//     int head_dim, int block_size, int max_blocks_per_seq,
//     int sm_count,
//     cudaStream_t stream
// ) {
//     int grid  = min(num_seqs * num_kv_heads, sm_count * 2);
//     int block = FA4_THREADS;
//     size_t smem = sizeof(FA4Smem);
//
//     cudaMemsetAsync(d_tile_counter, 0, sizeof(int), stream);
//
//     flash_attention_4_decode_f16io_kernel<<<grid, block, smem, stream>>>(
//         output, query, key_cache, value_cache,
//         block_tables, context_lens, d_tile_counter,
//         scale, num_seqs, num_heads, num_kv_heads,
//         head_dim, block_size, max_blocks_per_seq
//     );
// }
