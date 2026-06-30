#include <metal_stdlib>
using namespace metal;

kernel void attention_qk_f16(
    device const half*  Q           [[buffer(0)]], // [n_heads, head_dim]
    device const half*  K_cache     [[buffer(1)]], // [seq_len, n_heads, head_dim]
    device       half*  scores      [[buffer(2)]], // [n_heads, seq_len]
    constant     uint&  n_heads     [[buffer(3)]],
    constant     uint&  head_dim    [[buffer(4)]],
    constant     uint&  seq_len     [[buffer(5)]],
    constant     uint&  current_pos [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]] // tid.x = pos, tid.y = head
) {
    uint pos  = tid.x;
    uint head = tid.y;

    if (pos >= seq_len || head >= n_heads) return;

    // dot Q[head, :] against K_cache[pos, head, :]
    float scale = 1.0f / sqrt(float(head_dim));
    float score = 0.0f;
    for (uint i = 0; i < head_dim; i++) {
        float q_val = float(Q[head * head_dim + i]);
        float k_val = float(K_cache[(pos * n_heads + head) * head_dim + i]);
        score += q_val * k_val;
    }
    score *= scale;

    // causal mask
    if (pos > current_pos) {
        score = -INFINITY;
    }

    scores[head * seq_len + pos] = half(score);
}
