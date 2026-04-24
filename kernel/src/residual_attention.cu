// residual_attention.cu
// ForkKV Algorithm 1: ResidualAttention
//
// Attends over TWO separate paged KV caches in a single kernel:
//   bCache  — shared base prefix (read-only, shared across forked agents)
//   rCache  — per-agent residual suffix (agent-specific tokens after the fork)
//
// The key invariant: both phases share ONE continuous online softmax state.
// max/sum accumulators are NOT reset between phase 1 and phase 2.
// The final output is correctly normalized over the full (base + residual) context.
//
// Reference: arXiv:2604.06370 — ForkKV, Algorithm 1
//
// Shared memory layout (per block):
//   s_kv     [RA_BC * (head_dim + RA_KV_PAD)]   __half
//   s_scores [HPG * (RA_BC + RA_SCORE_PAD)]      float
//   s_warp   [RA_WARPS]                           float
//
// Launch config:
//   Grid:  (num_seqs, num_kv_heads, 1)
//   Block: (RA_THREADS, 1, 1)
//   Shared: RA_BC*(head_dim+2)*sizeof(__half)
//         + HPG*(RA_BC+1)*sizeof(float)
//         + RA_WARPS*sizeof(float)

#include <cfloat>
#include <float.h>
#include <cuda_fp16.h>


#define RA_BC           64
#define RA_THREADS      256
#define RA_WARPS        8
#define RA_GQA_MAX_HPG  8
#define RA_KV_PAD       2
#define RA_SCORE_PAD    1
#define RA_SCORE_STRIDE (RA_BC + RA_SCORE_PAD)

__device__ __forceinline__ float ra_warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float ra_warp_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float ra_block_reduce_max(
    float val, int tid, int lane_id, int warp_id, float* s_warp
) {
    val = ra_warp_max(val);
    if (lane_id == 0) s_warp[warp_id] = val;
    __syncthreads();
    if (tid == 0) {
        float m = s_warp[0];
        #pragma unroll
        for (int w = 1; w < RA_WARPS; w++) m = fmaxf(m, s_warp[w]);
        s_warp[0] = m;
    }
    __syncthreads();
    return s_warp[0];
}

__device__ __forceinline__ float ra_block_reduce_sum(
    float val, int tid, int lane_id, int warp_id, float* s_warp
) {
    val = ra_warp_sum(val);
    if (lane_id == 0) s_warp[warp_id] = val;
    __syncthreads();
    if (tid == 0) {
        float s = s_warp[0];
        #pragma unroll
        for (int w = 1; w < RA_WARPS; w++) s += s_warp[w];
        s_warp[0] = s;
    }
    __syncthreads();
    return s_warp[0];
}

// KV tile loader — base cache (bCache)
//
// Cache layout: [num_blocks, block_size, num_kv_heads, head_dim]
// Shared layout: [tile_len, head_dim + RA_KV_PAD]
// Uses __half2 vectorized loads. Handles odd element count with scalar fallback.
__device__ __forceinline__ void ra_load_base_tile(
    __half*       s_buf,
    const __half* b_cache,
    const int*    b_block_table,
    int           kv_head_idx,
    int           tile_start,
    int           tile_len,
    int           head_dim,
    int           num_kv_heads,
    int           block_size,
    int           tid
) {
    const int kv_stride   = head_dim + RA_KV_PAD;
    const int total_elems = tile_len * head_dim;
    const int total_h2    = total_elems / 2;

    for (int idx = tid; idx < total_h2; idx += RA_THREADS) {
        int elem      = idx * 2;
        int t         = elem / head_dim;
        int d         = elem % head_dim;
        int kv_pos    = tile_start + t;
        int page_idx  = kv_pos / block_size;
        int page_off  = kv_pos % block_size;
        int phys      = b_block_table[page_idx];
        int g_offset  = ((phys * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d;
        __half2 h2    = *reinterpret_cast<const __half2*>(&b_cache[g_offset]);
        s_buf[t * kv_stride + d]     = h2.x;
        s_buf[t * kv_stride + d + 1] = h2.y;
    }

    // scalar fallback for odd total_elems
    if ((total_elems & 1) && tid == 0) {
        int e        = total_elems - 1;
        int t        = e / head_dim;
        int d        = e % head_dim;
        int kv_pos   = tile_start + t;
        int pi       = kv_pos / block_size;
        int po       = kv_pos % block_size;
        int pb       = b_block_table[pi];
        int g_offset = ((pb * block_size + po) * num_kv_heads + kv_head_idx) * head_dim + d;
        s_buf[t * kv_stride + d] = b_cache[g_offset];
    }
}

// KV tile loader — residual cache (rCache)
//
// Identical contract to ra_load_base_tile but reads from rCache.

__device__ __forceinline__ void ra_load_residual_tile(
    __half*       s_buf,
    const __half* r_cache,
    const int*    r_block_table,
    int           kv_head_idx,
    int           tile_start,
    int           tile_len,
    int           head_dim,
    int           num_kv_heads,
    int           block_size,
    int           tid
) {
    const int kv_stride   = head_dim + RA_KV_PAD;
    const int total_elems = tile_len * head_dim;
    const int total_h2    = total_elems / 2;

    for (int idx = tid; idx < total_h2; idx += RA_THREADS) {
        int elem      = idx * 2;
        int t         = elem / head_dim;
        int d         = elem % head_dim;
        int kv_pos    = tile_start + t;
        int page_idx  = kv_pos / block_size;
        int page_off  = kv_pos % block_size;
        int phys      = r_block_table[page_idx];
        int g_offset  = ((phys * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d;
        __half2 h2    = *reinterpret_cast<const __half2*>(&r_cache[g_offset]);
        s_buf[t * kv_stride + d]     = h2.x;
        s_buf[t * kv_stride + d + 1] = h2.y;
    }

    if ((total_elems & 1) && tid == 0) {
        int e        = total_elems - 1;
        int t        = e / head_dim;
        int d        = e % head_dim;
        int kv_pos   = tile_start + t;
        int pi       = kv_pos / block_size;
        int po       = kv_pos % block_size;
        int pb       = r_block_table[pi];
        int g_offset = ((pb * block_size + po) * num_kv_heads + kv_head_idx) * head_dim + d;
        s_buf[t * kv_stride + d] = r_cache[g_offset];
    }
}

// QK^T dot product for one tile, one head
//
// Each warp handles one KV position t via ra_warp_sum across lanes.
// 8 warps -> 8 positions computed in parallel per round.
// Lane 0 writes result to s_scores[t].

__device__ __forceinline__ void ra_qk_dot_tile(
    float*        s_scores,
    const float   q_reg[4],
    const __half* s_kv,
    int           tile_len,
    int           head_dim,
    int           warp_id,
    int           lane_id
) {
    const int kv_stride   = head_dim + RA_KV_PAD;
    const int half2_iters = (head_dim + 63) / 64;

    for (int base_t = 0; base_t < tile_len; base_t += RA_WARPS) {
        int t = base_t + warp_id;
        if (t < tile_len) {
            float dot = 0.0f;
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                if (r >= half2_iters) break;
                int d = lane_id * 2 + r * 64;
                if (d + 1 < head_dim) {
                    dot += q_reg[r * 2]     * __half2float(s_kv[t * kv_stride + d]);
                    dot += q_reg[r * 2 + 1] * __half2float(s_kv[t * kv_stride + d + 1]);
                } else if (d < head_dim) {
                    dot += q_reg[r * 2] * __half2float(s_kv[t * kv_stride + d]);
                }
            }
            dot = ra_warp_sum(dot);
            if (lane_id == 0) s_scores[t] = dot;
        }
    }
}

// Online softmax update for one tile, one head
//
// CRITICAL: row_max and row_sum are passed by pointer and persist across
// both bCache and rCache phases. Never reset them between Phase 1 and 2.
//
// On the very first tile where row_max == -FLT_MAX, the correction branch
// is skipped (guarded by row_max > -FLT_MAX), so acc stays at zero and
// row_sum starts accumulating correctly from the first exp.
//
// After this call, s_scores[t] holds the softmax weight (exp value),
// ready for the P@V accumulation step.
__device__ __forceinline__ void ra_online_softmax(
    float&  row_max,
    float&  row_sum,
    float   acc[4],
    float*  s_scores,
    int     tile_len,
    int     head_dim,
    int     tid,
    int     lane_id,
    int     warp_id,
    float*  s_warp
) {
    const int acc_dims = (head_dim + RA_THREADS - 1) / RA_THREADS;

    // Step 1: find tile max across all threads
    float tile_max = ra_block_reduce_max(
        (tid < tile_len) ? s_scores[tid] : -FLT_MAX,
        tid, lane_id, warp_id, s_warp
    );

    // Step 2: new global max
    float new_max = fmaxf(row_max, tile_max);

    // Step 3: rescale existing acc and sum if max increased.
    // The guard (row_max > -FLT_MAX) ensures we skip on the very first tile
    // where there is nothing accumulated yet, avoiding expf(-inf - new_max).
    if (new_max > row_max && row_max > -FLT_MAX) {
        float correction = expf(row_max - new_max);
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            if (r < acc_dims) acc[r] *= correction;
        }
        row_sum *= correction;
    }
    row_max = new_max;

    // Step 4: compute exp weights, overwrite s_scores, accumulate denominator
    float my_exp = (tid < tile_len) ? expf(s_scores[tid] - new_max) : 0.0f;
    if (tid < tile_len) s_scores[tid] = my_exp;
    row_sum += ra_block_reduce_sum(my_exp, tid, lane_id, warp_id, s_warp);
}

// ---------------------------------------------------------------------------
// P @ V accumulation for one tile, one head
//
// s_scores[t] holds softmax weights after ra_online_softmax.
// s_kv holds the VALUE tile loaded into the same shared buffer after QK^T.
// V is reused across all GQA heads: caller loads V once, all g iterate here.
// This saves (HPG-1) * tile_len smem reads of V per tile.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void ra_pv_accumulate(
    float         acc[4],
    const float*  s_scores,
    const __half* s_kv,
    int           tile_len,
    int           head_dim,
    int           tid
) {
    const int kv_stride = head_dim + RA_KV_PAD;
    const int acc_dims  = (head_dim + RA_THREADS - 1) / RA_THREADS;

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        if (r >= acc_dims) break;
        int d = tid + r * RA_THREADS;
        if (d < head_dim) {
            float v_acc = 0.0f;
            for (int t = 0; t < tile_len; t++) {
                v_acc += s_scores[t] * __half2float(s_kv[t * kv_stride + d]);
            }
            acc[r] += v_acc;
        }
    }
}

// Q register loader
//
// Loads and scales query values for one head into q_reg[0..4).
// Uses lane_id * 2 + r * 64 indexing matching FA3's warp-parallel dot layout.

__device__ __forceinline__ void ra_load_q_reg(
    float         q_reg[4],
    const __half* query,
    int           seq_idx,
    int           g_head_idx,
    int           num_heads,
    int           head_dim,
    int           lane_id,
    float         scale
) {
    const int half2_iters = (head_dim + 63) / 64;
    const int q_base      = (seq_idx * num_heads + g_head_idx) * head_dim;

    #pragma unroll
    for (int r = 0; r < 2; r++) {
        if (r >= half2_iters) {
            q_reg[r * 2]     = 0.0f;
            q_reg[r * 2 + 1] = 0.0f;
            continue;
        }
        int d = lane_id * 2 + r * 64;
        if (d + 1 < head_dim) {
            q_reg[r * 2]     = __half2float(query[q_base + d])     * scale;
            q_reg[r * 2 + 1] = __half2float(query[q_base + d + 1]) * scale;
        } else if (d < head_dim) {
            q_reg[r * 2]     = __half2float(query[q_base + d]) * scale;
            q_reg[r * 2 + 1] = 0.0f;
        } else {
            q_reg[r * 2]     = 0.0f;
            q_reg[r * 2 + 1] = 0.0f;
        }
    }
}


// Output writer
//
// Normalizes acc by row_sum and writes f16 output for one head.
// output layout: [num_seqs, num_heads, head_dim]

__device__ __forceinline__ void ra_write_output(
    __half*      output,
    const float  acc[4],
    float        row_sum,
    int          seq_idx,
    int          g_head_idx,
    int          num_heads,
    int          head_dim,
    int          tid
) {
    const int acc_dims  = (head_dim + RA_THREADS - 1) / RA_THREADS;
    const int out_base  = (seq_idx * num_heads + g_head_idx) * head_dim;
    const float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        if (r >= acc_dims) break;
        int d = tid + r * RA_THREADS;
        if (d < head_dim) {
            output[out_base + d] = __float2half(acc[r] * inv_sum);
        }
    }
}


// Tile loop helper macro
//
// Encapsulates the 5-step tile processing sequence used identically in both
// Phase 1 (bCache) and Phase 2 (rCache). Eliminates ~40 lines of duplication.
//
// LOAD_K / LOAD_V are expression-statements: the caller passes the appropriate
// ra_load_base_tile(...) or ra_load_residual_tile(...) call.
//
// The macro requires these names in scope:
//   s_kv, s_scores, s_warp, q_regs, head_acc, head_row_max, head_row_sum,
//   heads_per_grp, head_dim, tid, warp_id, lane_id

#define RA_TILE_LOOP(LOAD_K, LOAD_V, context_len, n_tiles)                      \
do {                                                                            \
    for (int _tile = 0; _tile < (n_tiles); _tile++) {                          \
        const int tile_start = _tile * RA_BC;                                  \
        const int tile_len   = min(RA_BC, (context_len) - tile_start);         \
                                                                                \
        /* 1. Load K tile into s_kv */                                          \
        (LOAD_K);                                                               \
        __syncthreads();                                                        \
                                                                                \
        /* 2. QK^T: each head writes its scores into s_scores[g * STRIDE..] */ \
        for (int g = 0; g < heads_per_grp && g < RA_GQA_MAX_HPG; g++) {        \
            ra_qk_dot_tile(                                                     \
                s_scores + g * RA_SCORE_STRIDE,                                 \
                q_regs[g], s_kv, tile_len, head_dim, warp_id, lane_id          \
            );                                                                  \
        }                                                                       \
        __syncthreads();                                                        \
                                                                                \
        /* 3. Online softmax per head — state (row_max, row_sum) persists */   \
        for (int g = 0; g < heads_per_grp && g < RA_GQA_MAX_HPG; g++) {        \
            ra_online_softmax(                                                  \
                head_row_max[g], head_row_sum[g], head_acc[g],                  \
                s_scores + g * RA_SCORE_STRIDE,                                 \
                tile_len, head_dim, tid, lane_id, warp_id, s_warp              \
            );                                                                  \
        }                                                                       \
                                                                                \
        /* 4. Load V tile into s_kv — K is fully consumed after step 2 */      \
        (LOAD_V);                                                               \
        __syncthreads();                                                        \
                                                                                \
        /* 5. P@V — single smem V read reused across all GQA heads */          \
        for (int g = 0; g < heads_per_grp && g < RA_GQA_MAX_HPG; g++) {        \
            ra_pv_accumulate(                                                   \
                head_acc[g],                                                    \
                s_scores + g * RA_SCORE_STRIDE,                                 \
                s_kv, tile_len, head_dim, tid                                   \
            );                                                                  \
        }                                                                       \
        __syncthreads();                                                        \
    }                                                                           \
} while (0)

// Main kernel: residual_attention_decode_f16io_kernel
//
// One block per (seq_idx, kv_head_idx) pair.
// Processes all query heads in the GQA group for that kv_head.
//
// Phase 1: bCache tiles  (shared base prefix, read-only)
// Phase 2: rCache tiles  (per-agent residual, same softmax state)
// =============================================================================
extern "C"
__global__ void __launch_bounds__(RA_THREADS, 2)
residual_attention_decode_f16io_kernel(
    __half* __restrict__       output,              // [num_seqs, num_heads, head_dim]

    const __half* __restrict__ query,               // [num_seqs, num_heads, head_dim]

    // bCache — shared base prefix
    const __half* __restrict__ b_key_cache,         // [num_base_blocks,     block_size, num_kv_heads, head_dim]
    const __half* __restrict__ b_val_cache,         // [num_base_blocks,     block_size, num_kv_heads, head_dim]
    const int*    __restrict__ b_block_table,       // [num_seqs, max_base_blocks]
    int                        base_context_len,

    // rCache — per-agent residual
    const __half* __restrict__ r_key_cache,         // [num_residual_blocks, block_size, num_kv_heads, head_dim]
    const __half* __restrict__ r_val_cache,         // [num_residual_blocks, block_size, num_kv_heads, head_dim]
    const int*    __restrict__ r_block_table,       // [num_seqs, max_residual_blocks]
    int                        residual_context_len,

    float scale,
    int   num_heads,
    int   num_kv_heads,
    int   head_dim,
    int   block_size,
    int   max_base_blocks,
    int   max_residual_blocks
) {
    const int seq_idx       = blockIdx.x;
    const int kv_head_idx   = blockIdx.y;
    const int tid           = threadIdx.x;
    const int warp_id       = tid / 32;
    const int lane_id       = tid % 32;
    const int heads_per_grp = num_heads / num_kv_heads;
    const int kv_stride     = head_dim + RA_KV_PAD;

    // Nothing to attend over — write zeros and return
    if (base_context_len == 0 && residual_context_len == 0) {
        for (int g = 0; g < heads_per_grp && g < RA_GQA_MAX_HPG; g++) {
            int g_head   = kv_head_idx * heads_per_grp + g;
            int out_base = (seq_idx * num_heads + g_head) * head_dim;
            const int acc_dims = (head_dim + RA_THREADS - 1) / RA_THREADS;
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                if (r >= acc_dims) break;
                int d = tid + r * RA_THREADS;
                if (d < head_dim) output[out_base + d] = __float2half(0.0f);
            }
        }
        return;
    }

    // ------------------------------------------------------------------
    // Shared memory partition
    //   s_kv     : KV tile buffer, reused for K then V each tile
    //   s_scores : per-head attention scores, padded to avoid bank conflicts
    //   s_warp   : warp-level reduction scratch
    // ------------------------------------------------------------------
    extern __shared__ char smem_raw[];
    __half* s_kv     = (__half*)smem_raw;
    float*  s_scores = (float*)(s_kv + RA_BC * kv_stride);
    float*  s_warp   = s_scores + heads_per_grp * RA_SCORE_STRIDE;

    // ------------------------------------------------------------------
    // Per-head register state
    // These persist across both phases — never reset between Phase 1 and 2.
    // ------------------------------------------------------------------
    float q_regs      [RA_GQA_MAX_HPG][4];
    float head_row_max[RA_GQA_MAX_HPG];
    float head_row_sum[RA_GQA_MAX_HPG];
    float head_acc    [RA_GQA_MAX_HPG][4];

    for (int g = 0; g < heads_per_grp && g < RA_GQA_MAX_HPG; g++) {
        int g_head = kv_head_idx * heads_per_grp + g;
        ra_load_q_reg(q_regs[g], query, seq_idx, g_head,
                      num_heads, head_dim, lane_id, scale);
        head_row_max[g] = -FLT_MAX;   // sentinel: no tile seen yet
        head_row_sum[g] = 0.0f;
        #pragma unroll
        for (int r = 0; r < 4; r++) head_acc[g][r] = 0.0f;
    }

    // ------------------------------------------------------------------
    // Phase 1: attend over bCache (shared base prefix, read-only)
    // ------------------------------------------------------------------
    if (base_context_len > 0) {
        const int* b_row   = b_block_table + seq_idx * max_base_blocks;
        const int  n_tiles = (base_context_len + RA_BC - 1) / RA_BC;

        RA_TILE_LOOP(
            ra_load_base_tile(s_kv, b_key_cache, b_row,
                kv_head_idx, tile_start, tile_len,
                head_dim, num_kv_heads, block_size, tid),
            ra_load_base_tile(s_kv, b_val_cache, b_row,
                kv_head_idx, tile_start, tile_len,
                head_dim, num_kv_heads, block_size, tid),
            base_context_len,
            n_tiles
        );
    }

    // ------------------------------------------------------------------
    // Phase 2: attend over rCache (per-agent residual)
    //
    // head_row_max / head_row_sum / head_acc are NOT reset here.
    // The online softmax continues over the full (base + residual) context,
    // producing a single correctly normalized output.
    // ------------------------------------------------------------------
    if (residual_context_len > 0) {
        const int* r_row   = r_block_table + seq_idx * max_residual_blocks;
        const int  n_tiles = (residual_context_len + RA_BC - 1) / RA_BC;

        RA_TILE_LOOP(
            ra_load_residual_tile(s_kv, r_key_cache, r_row,
                kv_head_idx, tile_start, tile_len,
                head_dim, num_kv_heads, block_size, tid),
            ra_load_residual_tile(s_kv, r_val_cache, r_row,
                kv_head_idx, tile_start, tile_len,
                head_dim, num_kv_heads, block_size, tid),
            residual_context_len,
            n_tiles
        );
    }

    // ------------------------------------------------------------------
    // Write normalized output for all heads in the GQA group
    // ------------------------------------------------------------------
    for (int g = 0; g < heads_per_grp && g < RA_GQA_MAX_HPG; g++) {
        int g_head = kv_head_idx * heads_per_grp + g;
        ra_write_output(output, head_acc[g], head_row_sum[g],
                        seq_idx, g_head, num_heads, head_dim, tid);
    }
}

// =============================================================================
// Host-side launch helper
// (call this from the cudarc Rust bindings in Arc 2)
//
// void launch_residual_attention(
//     __half*       output,
//     const __half* query,
//     const __half* b_key_cache,   const __half* b_val_cache,
//     const int*    b_block_table, int base_context_len,
//     const __half* r_key_cache,   const __half* r_val_cache,
//     const int*    r_block_table, int residual_context_len,
//     float scale,
//     int num_seqs, int num_heads, int num_kv_heads,
//     int head_dim, int block_size,
//     int max_base_blocks, int max_residual_blocks,
//     cudaStream_t stream
// ) {
//     const int heads_per_grp = num_heads / num_kv_heads;
//     const int kv_stride     = head_dim + RA_KV_PAD;
//
//     size_t smem =
//         (size_t)RA_BC * kv_stride               * sizeof(__half)  // s_kv
//       + (size_t)heads_per_grp * RA_SCORE_STRIDE  * sizeof(float)  // s_scores
//       + (size_t)RA_WARPS                          * sizeof(float); // s_warp
//
//     dim3 grid(num_seqs, num_kv_heads);
//     dim3 block(RA_THREADS);
//
//     residual_attention_decode_f16io_kernel<<<grid, block, smem, stream>>>(
//         output, query,
//         b_key_cache, b_val_cache, b_block_table, base_context_len,
//         r_key_cache, r_val_cache, r_block_table, residual_context_len,
//         scale, num_heads, num_kv_heads, head_dim, block_size,
//         max_base_blocks, max_residual_blocks
//     );
// }
// =============================================================================
