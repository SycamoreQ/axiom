#include <metal_stdlib>
using namespace metal;

// Fused RoPE + FP16 KV Cache Write Kernel
// Computes Rotary Position Embedding on K and writes directly into k_cache as FP16
//
// Grid dispatch:
// - Threadgroup: (32, 1, 1) -> 1 SIMD group of 32 threads
// - Grid: (write_len, n_kv_heads, 1) -> gid.x is token position pos, gid.y is kv_head
kernel void rope_cache_write_f32_to_f16(
    device const float* src        [[buffer(0)]], // [write_len, n_kv_heads, head_dim]
    device       half*  k_cache    [[buffer(1)]], // [max_seq_len, n_kv_heads, head_dim]
    constant     uint&  write_pos  [[buffer(2)]],
    constant     uint&  n_kv_heads [[buffer(3)]],
    constant     uint&  head_dim   [[buffer(4)]],
    constant     uint&  write_len  [[buffer(5)]],
    constant     float& theta      [[buffer(6)]],
    constant     uint&  is_neox    [[buffer(7)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    uint pos = gid.x;
    uint kv_head = gid.y;
    uint lane = lid.x;

    if (pos >= write_len || kv_head >= n_kv_heads) return;

    uint current_token = write_pos + pos;
    uint half_dim = head_dim / 2;

    device const float* src_row = src + (pos * n_kv_heads + kv_head) * head_dim;
    device       half*  dst_row = k_cache + (current_token * n_kv_heads + kv_head) * head_dim;

    if (is_neox) {
        // NeoX / Qwen style RoPE: pair (i, i + half_dim)
        for (uint i = lane; i < half_dim; i += 32) {
            float freq = 1.0f / pow(theta, float(2 * i) / float(head_dim));
            float angle = float(current_token) * freq;
            float cos_a = cos(angle);
            float sin_a = sin(angle);

            float x0 = src_row[i];
            float x1 = src_row[i + half_dim];

            dst_row[i]            = half(x0 * cos_a - x1 * sin_a);
            dst_row[i + half_dim] = half(x0 * sin_a + x1 * cos_a);
        }
    } else {
        // Standard / Llama style RoPE: pair (2*i, 2*i + 1)
        for (uint i = lane; i < half_dim; i += 32) {
            float freq = 1.0f / pow(theta, float(2 * i) / float(head_dim));
            float angle = float(current_token) * freq;
            float cos_a = cos(angle);
            float sin_a = sin(angle);

            float x0 = src_row[2 * i];
            float x1 = src_row[2 * i + 1];

            dst_row[2 * i]     = half(x0 * cos_a - x1 * sin_a);
            dst_row[2 * i + 1] = half(x0 * sin_a + x1 * cos_a);
        }
    }
}

