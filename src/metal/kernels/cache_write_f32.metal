#include <metal_stdlib>
using namespace metal;

kernel void cache_write_f32(
    device const float* src        [[buffer(0)]], // [write_len, n_kv_heads, head_dim]
    device       float* cache      [[buffer(1)]], // [max_seq_len, n_kv_heads, head_dim]
    constant     uint&  write_pos  [[buffer(2)]],
    constant     uint&  n_kv_heads [[buffer(3)]],
    constant     uint&  head_dim   [[buffer(4)]],
    constant     uint&  write_len  [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint pos = tid.x;
    uint kv_head = tid.y;
    uint d = tid.z;
    if (pos >= write_len || kv_head >= n_kv_heads || d >= head_dim) return;

    uint src_idx = (pos * n_kv_heads + kv_head) * head_dim + d;
    uint dst_idx = ((write_pos + pos) * n_kv_heads + kv_head) * head_dim + d;
    cache[dst_idx] = src[src_idx];
}
