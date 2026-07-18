#include <metal_stdlib>
using namespace metal;

kernel void attention_qk_f32(
    device const float* Q           [[buffer(0)]], // [n_heads, head_dim]
    device const float* K_cache     [[buffer(1)]], // [seq_len, n_kv_heads, head_dim]
    device       float* scores      [[buffer(2)]], // [n_heads, seq_len]
    constant     uint&  n_heads     [[buffer(3)]],
    constant     uint&  n_kv_heads  [[buffer(4)]],
    constant     uint&  head_dim    [[buffer(5)]],
    constant     uint&  seq_len     [[buffer(6)]],
    constant     uint&  current_pos [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint pos  = tid.x;
    uint head = tid.y;
    if (pos >= seq_len || head >= n_heads) return;

    uint repeat_factor = n_heads / n_kv_heads;
    uint kv_head = head / repeat_factor;

    float scale = 1.0f / sqrt(float(head_dim));
    float score = 0.0f;
    for (uint i = 0; i < head_dim; i++) {
        float q_val = Q[head * head_dim + i];
        float k_val = K_cache[(pos * n_kv_heads + kv_head) * head_dim + i];
        score += q_val * k_val;
    }
    score *= scale;

    if (pos > current_pos) score = -INFINITY;
    scores[head * seq_len + pos] = score;
}
