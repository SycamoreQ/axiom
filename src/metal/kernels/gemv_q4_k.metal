#include <metal_stdlib>
using namespace metal;

// GEMV for Q4_K weights: y = W * x
// W: [N, K] in packed Q4_K format (each block of 256 weights is 144 bytes)
// x: [M, K] in float (M >= 1)
// y: [M, N] in float
//
// Grid dispatch:
// - Threadgroup: (32, 1, 1) -> 1 SIMD-group of 32 threads
// - Grid: (N, M, 1) -> threadgroup gid.x handles output row n, gid.y handles token m
kernel void gemv_q4_k_f32(
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

    uint num_blocks_per_row = K / 256;
    device const uchar* row_data = W + (n * num_blocks_per_row) * 144;
    device const float* x_vec = x + m * K;

    float acc = 0.0f;

    for (uint b = 0; b < num_blocks_per_row; b++) {
        device const uchar* block = row_data + b * 144;
        uint k_base = b * 256;

        half d_h = as_type<half>(ushort(ushort(block[0]) | (ushort(block[1]) << 8)));
        half dmin_h = as_type<half>(ushort(ushort(block[2]) | (ushort(block[3]) << 8)));
        float d = float(d_h);
        float dmin = float(dmin_h);

        device const uchar* sc = block + 4;
        device const uchar* qs = block + 16;

        uchar scales[8];
        uchar mins[8];
        scales[0] = sc[0] & 0x3F;
        scales[1] = sc[1] & 0x3F;
        scales[2] = sc[2] & 0x3F;
        scales[3] = sc[3] & 0x3F;
        scales[4] = (sc[8] & 0x0F) | ((sc[0] >> 6) << 4);
        scales[5] = (sc[9] & 0x0F) | ((sc[1] >> 6) << 4);
        scales[6] = (sc[10] & 0x0F) | ((sc[2] >> 6) << 4);
        scales[7] = (sc[11] & 0x0F) | ((sc[3] >> 6) << 4);

        mins[0] = sc[4] & 0x3F;
        mins[1] = sc[5] & 0x3F;
        mins[2] = sc[6] & 0x3F;
        mins[3] = sc[7] & 0x3F;
        mins[4] = (sc[8] >> 4) | ((sc[4] >> 6) << 4);
        mins[5] = (sc[9] >> 4) | ((sc[5] >> 6) << 4);
        mins[6] = (sc[10] >> 4) | ((sc[6] >> 6) << 4);
        mins[7] = (sc[11] >> 4) | ((sc[7] >> 6) << 4);

        for (uint c = 0; c < 4; c++) {
            float d1 = d * float(scales[2 * c]);
            float m1 = dmin * float(mins[2 * c]);
            float d2 = d * float(scales[2 * c + 1]);
            float m2 = dmin * float(mins[2 * c + 1]);

            uchar q_byte = qs[c * 32 + lane];
            float w1 = d1 * float(q_byte & 0x0F) - m1;
            float w2 = d2 * float(q_byte >> 4) - m2;

            float x1 = x_vec[k_base + c * 64 + lane];
            float x2 = x_vec[k_base + c * 64 + 32 + lane];

            acc += w1 * x1 + w2 * x2;
        }
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        y[m * N + n] = acc;
    }
}
