#include <metal_stdlib>
using namespace metal;

// GEMV for Q8_0 weights: y = W * x
// W: [N, K] in packed Q8_0 format (each block of 32 weights is 34 bytes)
// x: [M, K] in float (M >= 1)
// y: [M, N] in float
//
// Grid dispatch:
// - Threadgroup: (32, 1, 1) -> 1 SIMD-group of 32 threads
// - Grid: (N, M, 1) -> threadgroup gid.x handles output row n, gid.y handles token m
kernel void gemv_q8_0_f32(
    device const uchar*  W [[buffer(0)]],
    device const float*  x [[buffer(1)]],
    device       float*  y [[buffer(2)]],
    constant     uint&   K [[buffer(3)]],
    constant     uint&   N [[buffer(4)]],
    constant     uint&   M [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    uint n = gid.x;
    uint m = gid.y;
    uint lane = lid.x;
    if (n >= N || m >= M) return;

    uint num_blocks_per_row = K / 32;
    device const uchar* row_data = W + (n * num_blocks_per_row) * 34;
    device const float* x_vec = x + m * K;

    float acc = 0.0f;

    for (uint b = 0; b < num_blocks_per_row; b++) {
        device const uchar* block = row_data + b * 34;
        uint k_base = b * 32;

        half d_h = as_type<half>(ushort(ushort(block[0]) | (ushort(block[1]) << 8)));
        float d = float(d_h);

        device const char* qs = (device const char*)(block + 2);
        int q = int(qs[lane]);
        float x_val = x_vec[k_base + lane];

        acc += d * float(q) * x_val;
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        y[m * N + n] = acc;
    }
}
