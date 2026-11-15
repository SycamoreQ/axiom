#include <metal_stdlib>
using namespace metal;

// GPU-Side Penalties Application Kernel
// Directly modifies logits for repetition penalty and banned n-grams
kernel void apply_penalties_f32(
    device float*          logits            [[buffer(0)]], // [vocab_size]
    constant const uint*   penalized_tokens  [[buffer(1)]],
    constant float&        penalty           [[buffer(2)]],
    constant uint&         num_penalized     [[buffer(3)]],
    constant const uint*   banned_tokens     [[buffer(4)]],
    constant uint&         num_banned        [[buffer(5)]],
    constant uint&         vocab_size        [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < num_penalized) {
        uint tok = penalized_tokens[tid];
        if (tok < vocab_size) {
            float val = logits[tok];
            if (val > 0.0f) {
                logits[tok] = val / penalty;
            } else {
                logits[tok] = val * penalty;
            }
        }
    } else {
        uint ban_idx = tid - num_penalized;
        if (ban_idx < num_banned) {
            uint tok = banned_tokens[ban_idx];
            if (tok < vocab_size) {
                logits[tok] = -INFINITY;
            }
        }
    }
}

// GPU-Side Argmax Kernel
// Finds the index of the maximum value in logits[num_rows, vocab_size]
// Vectorized with float4 loads for maximum memory bandwidth
//
// Grid dispatch:
// - Threadgroup: (256, 1, 1) -> 256 threads (8 SIMD groups)
// - Grid: (1, num_rows, 1) -> gid.y handles row m
kernel void argmax_f32(
    device const float* logits      [[buffer(0)]], // [num_rows, vocab_size]
    device       uint*  output_idx  [[buffer(1)]], // [num_rows]
    constant     uint&  vocab_size  [[buffer(2)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    uint row = gid.y;
    uint tid = lid.x;
    device const float* row_logits = logits + row * vocab_size;

    float max_val = -INFINITY;
    uint  max_idx = 0;

    // 128-bit vectorized scan (float4)
    uint vocab_size_v4 = vocab_size / 4;
    device const float4* row_logits_v4 = (device const float4*)row_logits;

    for (uint i = tid; i < vocab_size_v4; i += 256) {
        float4 v = row_logits_v4[i];
        uint base = i * 4;
        if (v.x > max_val) { max_val = v.x; max_idx = base; }
        if (v.y > max_val) { max_val = v.y; max_idx = base + 1; }
        if (v.z > max_val) { max_val = v.z; max_idx = base + 2; }
        if (v.w > max_val) { max_val = v.w; max_idx = base + 3; }
    }

    // Scalar tail for any remainder
    for (uint i = vocab_size_v4 * 4 + tid; i < vocab_size; i += 256) {
        float val = row_logits[i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    // Threadgroup reduction in shared memory
    threadgroup float tg_vals[256];
    threadgroup uint  tg_indices[256];

    tg_vals[tid] = max_val;
    tg_indices[tid] = max_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            if (tg_vals[tid + s] > tg_vals[tid]) {
                tg_vals[tid] = tg_vals[tid + s];
                tg_indices[tid] = tg_indices[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output_idx[row] = tg_indices[0];
    }
}
