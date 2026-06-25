#include <metal_stdlib>
using namespace metal;

kernel void rope_f16(
    device half*  x        [[buffer(0)]],  // modified in-place
    constant uint& seq_len  [[buffer(1)]],
    constant uint& n_heads  [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant float& theta    [[buffer(4)]],  // base frequency, typically 10000.0
    uint2 tid [[thread_position_in_grid]]          // tid.x = token, tid.y = head
) {

    uint token = tid.x;
    uint head = tid.y;

    if (token >= seq_len || head >= n_heads) return;

    device half* row = x + (token * n_heads + head) * head_dim;

    for (uint i = 0; i < head_dim / 2; i++) {
        float freq = 1.0f / pow(theta, float(2 * i) / float(head_dim));
        float angle = float(token) * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);

        float x0 = float(row[i]);
        float x1 = float(row[i + head_dim / 2]);

        row[i] = half(x0 * cos_a - x1 * sin_a);
        row[i + head_dim/2] = half(x0 * sin_a + x1 * cos_a);
    }
}
