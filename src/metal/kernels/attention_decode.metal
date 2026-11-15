#include <metal_stdlib>
using namespace metal;

// Fused FlashDecoding Attention Kernel using Online Softmax
// dao et al. / Milakov & Gimelshein online softmax recurrence
//
// Grid dispatch:
// - Threadgroup: (32, 1, 1) -> 1 SIMD group of 32 threads
// - Grid: (n_heads, 1, 1) -> threadgroup gid.x handles query head h
template <typename T_Q, typename T_KV, typename T_Out>
inline void attention_decode_impl(
    device const T_Q*   Q           [[buffer(0)]], // [n_heads, head_dim]
    device const T_KV*  K_cache     [[buffer(1)]], // [max_seq_len, n_kv_heads, head_dim]
    device const T_KV*  V_cache     [[buffer(2)]], // [max_seq_len, n_kv_heads, head_dim]
    device       T_Out* output      [[buffer(3)]], // [n_heads, head_dim]
    constant     uint&  n_heads     [[buffer(4)]],
    constant     uint&  n_kv_heads  [[buffer(5)]],
    constant     uint&  head_dim    [[buffer(6)]],
    constant     uint&  cache_len   [[buffer(7)]],
    uint2 gid,
    uint2 lid
) {
    uint head = gid.x;
    uint lane = lid.x;
    if (head >= n_heads) return;

    uint repeat_factor = n_heads / n_kv_heads;
    uint kv_head = head / repeat_factor;
    uint kv_stride = n_kv_heads * head_dim;

    // Up to head_dim = 256 (32 threads * 8 elements = 256)
    const uint MAX_D_PER_LANE = 8;
    uint num_d = (head_dim + 31) / 32;

    float q_reg[MAX_D_PER_LANE];
    float acc[MAX_D_PER_LANE];

    // Load Q vector into local SIMD registers (coalesced 128-byte read)
    for (uint k = 0; k < num_d; k++) {
        uint d = lane + k * 32;
        q_reg[k] = (d < head_dim) ? float(Q[head * head_dim + d]) : 0.0f;
        acc[k] = 0.0f;
    }

    float m = -INFINITY;
    float l = 0.0f;
    float scale = 1.0f / sqrt(float(head_dim));

    // Online Softmax loop across cached tokens
    for (uint t = 0; t < cache_len; t++) {
        uint base_kv = t * kv_stride + kv_head * head_dim;
        device const T_KV* k_ptr = K_cache + base_kv;

        float dot = 0.0f;
        for (uint k = 0; k < num_d; k++) {
            uint d = lane + k * 32;
            if (d < head_dim) {
                dot += q_reg[k] * float(k_ptr[d]);
            }
        }

        // Intra-SIMD group butterfly reduction across 32 lanes (1 cycle)
        float score = simd_sum(dot) * scale;

        // Online Softmax update
        float m_new = max(m, score);
        float alpha = (m == -INFINITY) ? 0.0f : exp(m - m_new);
        float beta = exp(score - m_new);

        l = l * alpha + beta;
        m = m_new;

        // Accumulate V directly in registers
        device const T_KV* v_ptr = V_cache + base_kv;
        for (uint k = 0; k < num_d; k++) {
            uint d = lane + k * 32;
            if (d < head_dim) {
                acc[k] = acc[k] * alpha + beta * float(v_ptr[d]);
            }
        }
    }

    // Final normalization
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (uint k = 0; k < num_d; k++) {
        uint d = lane + k * 32;
        if (d < head_dim) {
            output[head * head_dim + d] = static_cast<T_Out>(acc[k] * inv_l);
        }
    }
}

kernel void attention_decode_f16(
    device const half*  Q           [[buffer(0)]],
    device const half*  K_cache     [[buffer(1)]],
    device const half*  V_cache     [[buffer(2)]],
    device       half*  output      [[buffer(3)]],
    constant     uint&  n_heads     [[buffer(4)]],
    constant     uint&  n_kv_heads  [[buffer(5)]],
    constant     uint&  head_dim    [[buffer(6)]],
    constant     uint&  cache_len   [[buffer(7)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    attention_decode_impl<half, half, half>(
        Q, K_cache, V_cache, output, n_heads, n_kv_heads, head_dim, cache_len, gid, lid
    );
}

kernel void attention_decode_f32(
    device const float* Q           [[buffer(0)]],
    device const float* K_cache     [[buffer(1)]],
    device const float* V_cache     [[buffer(2)]],
    device       float* output      [[buffer(3)]],
    constant     uint&  n_heads     [[buffer(4)]],
    constant     uint&  n_kv_heads  [[buffer(5)]],
    constant     uint&  head_dim    [[buffer(6)]],
    constant     uint&  cache_len   [[buffer(7)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    attention_decode_impl<float, float, float>(
        Q, K_cache, V_cache, output, n_heads, n_kv_heads, head_dim, cache_len, gid, lid
    );
}

kernel void attention_decode_f32_kv_f16(
    device const float* Q           [[buffer(0)]],
    device const half*  K_cache     [[buffer(1)]],
    device const half*  V_cache     [[buffer(2)]],
    device       float* output      [[buffer(3)]],
    constant     uint&  n_heads     [[buffer(4)]],
    constant     uint&  n_kv_heads  [[buffer(5)]],
    constant     uint&  head_dim    [[buffer(6)]],
    constant     uint&  cache_len   [[buffer(7)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    attention_decode_impl<float, half, float>(
        Q, K_cache, V_cache, output, n_heads, n_kv_heads, head_dim, cache_len, gid, lid
    );
}

