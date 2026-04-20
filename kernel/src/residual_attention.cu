// =============================================================================
// residual_attention.cu
// ForkKV Algorithm 1: ResidualAttention
// Fuses bCache + rCache into SRAM, deferred RoPE, online softmax
// arXiv:2604.06370
// =============================================================================

#include <float.h>
#include <cuda_fp16.h>

#define RA_BC        64     // KV tile width
#define RA_THREADS  256     // threads per block
#define RA_WARPS      8     // warps per block
#define RA_GQA_MAX_HPG 8

// Launch config:
//   Grid:  (num_seqs, num_kv_heads, 1)
//   Block: (RA_THREADS, 1, 1)
//   Shared: s_bK[BC*HD] + s_bV[BC*HD] + s_rK[BC*HD] + s_rV[BC*HD]
//           + s_scores[BC] + s_warp[WARPS]  -- all __half except s_scores/s_warp (float)

__device__ __forceinline__ float ra_warp_sum(float val) {
    #pragma unroll
   for (int offset = 16 ;  offset > 0; offset >>=1)
       val += __shfl_xor_sync(0xfffffff, val , offset);
   return val
}

__device__ __forceinline__ float ra_warp_max(float val) {
    // TODO: warp shuffle max reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ void ra_load_base_tile(
    __half* s_buf,
    const __half* b_cache,           // [num_base_blocks, block_size, num_kv_heads, head_dim]
    const int* base_block_table,     // physical block indices for bCache
    int kv_head_idx,
    int tile_start, int tile_len,
    int head_dim, int num_kv_heads, int block_size,
    int tid
) {
    const int total_h2 = (tile_len * head_dim) / 2;
    for (int idx = tid; idx < total_h2; idx += RA_THREADS) {
        int elem = idx * 2;
        int t = elem / head_dim;
        int d = elem % head_dim;
        int kv_pos = tile_start + t;
        int page_idx = kv_pos / block_size;
        int page_off = kv_pos % block_size;
        int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
        int base = ((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d;
        __half2 h2 = *reinterpret_cast<const __half2*>(&b_cache[base]);
        s_kv[t * kv_stride + d]     = h2.x;
        s_kv[t * kv_stride + d + 1] = h2.y;
    }
    int total_elems = tile_len * head_dim;
    if ((total_elems & 1) && tid == 0) {
        int e = total_elems - 1;
        int t = e / head_dim, d = e % head_dim;
        int kv_pos = tile_start + t;
        int pi = kv_pos / block_size, po = kv_pos % block_size;
        int pb = block_tables[seq_idx * max_blocks_per_seq + pi];
        s_kv[t * kv_stride + d] = b_cache[((pb * block_size + po) * num_kv_heads + kv_head_idx) * head_dim + d];
    }
}

__device__ __forceinline__ void ra_load_residual_tile(
    __half* s_buf,
    const __half* r_cache,           // [num_blocks, block_size, num_kv_heads, head_dim]
    const int* residual_block_table, // physical block indices for rCache
    int kv_head_idx,
    int tile_start, int tile_len,
    int head_dim, int num_kv_heads, int block_size,
    int tid,
    int stride_s_buf                 // stride for shared memory buffer
) {
    // We use __half2 to load 32 bits at a time for better coalescing
    const int total_elems = tile_len * head_dim;
    const int total_h2 = total_elems / 2;

    for (int idx = tid; idx < total_h2; idx += blockDim.x) {
        //Map linear index to tile-local (t, d)
        int t = (idx * 2) / head_dim;
        int d = (idx * 2) % head_dim;

        //Map tile-local to global sequence position
        int kv_pos = tile_start + t;

        //Paged Memory Logic
        int page_idx = kv_pos / block_size;
        int page_off = kv_pos % block_size;

        // Get the physical block from the table passed in
        int phys_block = residual_block_table[page_idx];

        //Calculate Global Memory Address
        //Layout: [block][page_offset][head][dim]
        int g_offset = (((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d);

        //Vectorized Load from Global Memory
        __half2 h2_val = *reinterpret_cast<const __half2*>(&r_cache[g_offset]);

        //Store to Shared Memory
        //Ensure s_buf is indexed to avoid bank conflicts if possible
        s_buf[t * stride_s_buf + d]     = h2_val.x;
        s_buf[t * stride_s_buf + d + 1] = h2_val.y;
    }

    //Handle odd element if tile_len * head_dim is not even
    if ((total_elems & 1) && tid == 0) {
        int e = total_elems - 1;
        int t = e / head_dim;
        int d = e % head_dim;
        int kv_pos = tile_start + t;
        int pb = residual_block_table[kv_pos / block_size];
        int g_offset = (((pb * block_size + (kv_pos % block_size)) * num_kv_heads + kv_head_idx) * head_dim + d);
        s_buf[t * stride_s_buf + d] = r_cache[g_offset];
    }
}

extern "C"
__global__ void residual_attention_decode_f16io_kernel(
    __half* __restrict__ output,              // [num_seqs, num_heads, head_dim]
    const __half* __restrict__ query,         // [num_seqs, num_heads, head_dim]
    // bCache: shared prefix KV (read-only, shared across agents)
    const __half* __restrict__ b_key_cache,   // [num_base_blocks, block_size, num_kv_heads, head_dim]
    const __half* __restrict__ b_val_cache,
    const int*    __restrict__ b_block_table, // [num_seqs, max_base_blocks]
    int base_context_len,                     // how many tokens in bCache
    // rCache: per-agent residual KV (agent-specific suffix)
    const __half* __restrict__ r_key_cache,   // [num_residual_blocks, block_size, num_kv_heads, head_dim]
    const __half* __restrict__ r_val_cache,
    const int*    __restrict__ r_block_table, // [num_seqs, max_residual_blocks]
    int residual_context_len,                 // how many tokens in rCache
    // attention config
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_base_blocks,
    int max_residual_blocks
) {
    const int seq_idx     = blockIdx.x;
    const int kv_head_idx = blockIdx.y;
    const int tid         = threadIdx.x;
    const int warp_id     = tid / 32;
    const int lane_id     = tid % 32;
    const int heads_per_group = num_heads / num_kv_heads;

    // --- shared memory ---
    extern __shared__ char smem_raw[];
    __half* s_bK    = (__half*)smem_raw;
    __half* s_bV    = s_bK + RA_BC * head_dim;
    __half* s_rK    = s_bV + RA_BC * head_dim;
    __half* s_rV    = s_rK + RA_BC * head_dim;
    float*  s_score = (float*)(s_rV + RA_BC * head_dim);
    float*  s_warp  = s_score + RA_BC;

    // --- load Q registers ---
    // TODO: load all heads in the GQA group into q_reg[heads_per_group][4]

    // --- online softmax state ---
    // TODO: declare head_row_max[], head_row_sum[], head_acc[][]

    // --- phase 1: attend over bCache tiles ---
    // TODO: iterate base_context_len / RA_BC tiles
    //   ra_load_base_tile(s_bK, ...)
    //   QK^T per head
    //   online softmax update
    //   ra_load_base_tile(s_bV, ...)  [reuse s_bK slot]
    //   P @ V accumulate

    // --- phase 2: attend over rCache tiles ---
    // TODO: iterate residual_context_len / RA_BC tiles
    //   ra_load_residual_tile(s_rK, ...)
    //   QK^T per head (continuing the SAME online softmax state from phase 1)
    //   online softmax update
    //   ra_load_residual_tile(s_rV, ...)
    //   P @ V accumulate

    // --- write output ---
    // TODO: normalize head_acc by head_row_sum and write f16 output
    //

    const num_heads_per_group = num_heads/num_kv_heads;
    #pragma unroll
    for (int h = 0; h <= 4 ; ++h){
        int q_head_idx = (kv_head_idx * heads_per_group) + h;

        const __half* q_ptr = Q + (seq_idx * q_stride_seq)
                                    + (q_head_idx * q_stride_head)
                                    + d_offset;

            // 3. Vectorized 64-bit load (4 * 16-bit __half = 64 bits)
            // Using int2 tells the hardware to perform a single LDG.E.64 instruction
            *reinterpret_cast<int2*>(&q_reg[h][0]) = *reinterpret_cast<const int2*>(q_ptr);

    }

    float head_row_max[RA_GQA_MAX_HPG];
    float head_row_sum[RA_GQA_MAX_HPG];
    float head_acc[RA_GQA_MAX_HPG][4];

    for (int g = 0; g < heads_per_group && g < RA_GQA_MAX_HPG ; g++) {
        int g_head = kv_head_idx* heads_per_group + g;
        int q_base = (seq_idx * num_heads + g_head) * head_dim;
        #pragma unroll 
        for (int r = 0; r < half2_iters )
    }
}


// =============================================================================
// rms_norm.cu
// =============================================================================

#include <cuda_fp16.h>

// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(hidden_size, 1024), 1, 1)
//   Shared: blockDim.x * sizeof(float)

__device__ __forceinline__ float rms_warp_sum(float val) {
    // TODO: warp shuffle sum
}

extern "C"
__global__ void rms_norm_f16_kernel(
    __half* __restrict__ output,           // [num_tokens, hidden_size]
    const __half* __restrict__ input,      // [num_tokens, hidden_size]
    const __half* __restrict__ weight,     // [hidden_size]
    float eps,
    int hidden_size
) {
    const int token_idx = blockIdx.x;
    const int tid       = threadIdx.x;
    const int stride    = blockDim.x;

    extern __shared__ float sdata[];

    // TODO: pass 1 -- sum of squares (warp reduce into sdata[])
    // TODO: compute rms_scale = rsqrtf(sdata[0] / hidden_size + eps)
    // TODO: pass 2 -- output[i] = input[i] * weight[i] * rms_scale
}

// Fused variant: residual_out = input + add, then RMSNorm
// Launch: same as above
extern "C"
__global__ void fused_residual_rms_norm_f16_kernel(
    __half* __restrict__ output,           // [num_tokens, hidden_size]  normalized
    __half* __restrict__ residual_out,     // [num_tokens, hidden_size]  input + add (pre-norm)
    const __half* __restrict__ input,      // [num_tokens, hidden_size]
    const __half* __restrict__ add,        // [num_tokens, hidden_size]
    const __half* __restrict__ weight,     // [hidden_size]
    float eps,
    int hidden_size
) {
    const int token_idx = blockIdx.x;
    const int tid       = threadIdx.x;
    const int stride    = blockDim.x;

    extern __shared__ float sdata[];

    // TODO: pass 1 -- val = input[i] + add[i], write residual_out[i], accumulate ss
    // TODO: compute rms_scale
    // TODO: pass 2 -- output[i] = residual_out[i] * weight[i] * rms_scale
}


// =============================================================================
// rotary_embedding.cu
// =============================================================================

#include <cuda_fp16.h>

// Launch config:
//   Grid:  (num_tokens, num_heads, 1)
//   Block: (head_dim / 2, 1, 1)   -- one thread per rotation pair
//   Shared: none

extern "C"
__global__ void rotary_embedding_f16_kernel(
    __half* __restrict__ query,            // [num_tokens, num_heads, head_dim]   in-place
    __half* __restrict__ key,              // [num_tokens, num_kv_heads, head_dim] in-place
    const float* __restrict__ cos_cache,   // [max_position, head_dim/2]
    const float* __restrict__ sin_cache,   // [max_position, head_dim/2]
    const int*   __restrict__ positions,   // [num_tokens]
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int pair_idx  = threadIdx.x;   // 0 .. head_dim/2 - 1
    const int half_dim  = head_dim / 2;

    const int pos       = positions[token_idx];
    const float cos_val = cos_cache[pos * half_dim + pair_idx];
    const float sin_val = sin_cache[pos * half_dim + pair_idx];

    // TODO: apply to query -- q[2*i], q[2*i+1] rotation
    // TODO: apply to key if head_idx < num_kv_heads -- same rotation
}


// =============================================================================
// reshape_and_cache.cu
// Scatter per-token KV into the paged cache at slot_mapping positions
// =============================================================================

#include <cuda_fp16.h>

// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(num_kv_heads * head_dim, 1024), 1, 1)
//   Shared: none

extern "C"
__global__ void reshape_and_cache_f16_kernel(
    __half* __restrict__ key_cache,        // [num_blocks, block_size, num_kv_heads, head_dim]
    __half* __restrict__ val_cache,        // [num_blocks, block_size, num_kv_heads, head_dim]
    const __half* __restrict__ key,        // [num_tokens, num_kv_heads, head_dim]
    const __half* __restrict__ val,        // [num_tokens, num_kv_heads, head_dim]
    const int*    __restrict__ slot_mapping, // [num_tokens]  slot = block*block_size + offset
    int num_tokens,
    int num_kv_heads,
    int head_dim
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const int tid    = threadIdx.x;
    const int kv_dim = num_kv_heads * head_dim;
    const int slot   = slot_mapping[token_idx];
    if (slot < 0) return;

    // TODO: copy key[token_idx * kv_dim .. +kv_dim] -> key_cache[slot * kv_dim .. +kv_dim]
    // TODO: same for val
}


// =============================================================================
// copy_blocks.cu
// Copy KV cache blocks -- used by ForkKV copy-on-write
// =============================================================================

#include <cuda_fp16.h>

// Launch config:
//   Grid:  (num_pairs, 1, 1)
//   Block: (min(block_size * num_kv_heads * head_dim, 1024), 1, 1)
//   Shared: none

extern "C"
__global__ void copy_blocks_f16_kernel(
    __half* __restrict__ key_cache,          // [num_blocks, block_size, num_kv_heads, head_dim]
    __half* __restrict__ val_cache,
    const long* __restrict__ block_mapping,  // [num_pairs, 2]  (src, dst) physical block indices
    int num_pairs,
    int block_size,
    int num_kv_heads,
    int head_dim
) {
    const int pair_idx = blockIdx.x;
    if (pair_idx >= num_pairs) return;

    const long src_block = block_mapping[pair_idx * 2];
    const long dst_block = block_mapping[pair_idx * 2 + 1];
    const int  elems     = block_size * num_kv_heads * head_dim;

    // TODO: strided copy key_cache[src_block*elems .. ] -> key_cache[dst_block*elems .. ]
    // TODO: same for val
}


// =============================================================================
// argmax.cu
// GPU-side argmax -- avoids full logits DtoH for greedy decoding
// =============================================================================

#include <cuda_fp16.h>
#include <float.h>

// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(vocab_size, 1024), 1, 1)
//   Shared: uses static arrays -- no dynamic smem needed

extern "C"
__global__ void argmax_f16_kernel(
    const __half* __restrict__ logits,     // [num_tokens, vocab_size]
    int*          __restrict__ output,     // [num_tokens]
    int vocab_size
) {
    const int row    = blockIdx.x;
    const int tid    = threadIdx.x;
    const int stride = blockDim.x;
    const int n      = blockDim.x;

    const __half* x = logits + (long long)row * vocab_size;

    __shared__ float s_val[1024];
    __shared__ int   s_idx[1024];

    // TODO: pass 1 -- thread-local max+argmax over strided elements
    // TODO: tree reduction in shared memory to find block-level argmax
    // TODO: thread 0 writes output[row] = s_idx[0]
}


// =============================================================================
// embedding_gather.cu
// GPU embedding lookup -- avoids CPU round-trip for token embedding
// =============================================================================

#include <cuda_fp16.h>

// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(hidden_size, 1024), 1, 1)
//   Shared: none

extern "C"
__global__ void embedding_gather_f16_kernel(
    __half* __restrict__ output,             // [num_tokens, hidden_size]
    const __half* __restrict__ embed_table,  // [vocab_size, hidden_size]
    const int*    __restrict__ token_ids,    // [num_tokens]
    int hidden_size,
    int vocab_size
) {
    const int token_idx = blockIdx.x;
    const int tid       = threadIdx.x;
    const int stride    = blockDim.x;

    const int token_id   = token_ids[token_idx];
    const int out_offset = token_idx * hidden_size;

    if (token_id < 0 || token_id >= vocab_size) {
        // TODO: fill output row with zeros
        return;
    }

    // TODO: copy embed_table[token_id * hidden_size .. +hidden_size]
    //       -> output[out_offset .. +hidden_size]
}
