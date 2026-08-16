#include <metal_stdlib>
using namespace metal;

kernel void rope_neox_f32(
    device float* x         [[buffer(0)]],
    constant uint& seq_len  [[buffer(1)]],
    constant uint& n_heads  [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant float& theta   [[buffer(4)]],
    constant uint& offset   [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint token = tid.x;
    uint head = tid.y;
    if (token >= seq_len || head >= n_heads) return;

    device float* row = x + (token * n_heads + head) * head_dim;
    uint half_dim = head_dim / 2;

    for (uint i = 0; i < half_dim; i++) {
        float freq = 1.0f / pow(theta, float(2 * i) / float(head_dim));
        float angle = float(offset + token) * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);
        float x0 = row[i];
        float x1 = row[i + half_dim];
        row[i]            = x0 * cos_a - x1 * sin_a;
        row[i + half_dim] = x0 * sin_a + x1 * cos_a;
    }
}
