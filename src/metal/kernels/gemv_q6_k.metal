#include <metal_stdlib>
using namespace metal;

// GEMV for Q6_K weights: y = W * x
// W: [N, K] in packed Q6_K format (each block of 256 weights is 210 bytes)
// x: [M, K] in float (M >= 1)
// y: [M, N] in float
//
// Grid dispatch:
// - Threadgroup: (32, 1, 1) -> 1 SIMD-group of 32 threads
// - Grid: (N, M, 1) -> threadgroup gid.x handles output row n, gid.y handles token m
kernel void gemv_q6_k_f32(
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
    device const uchar* row_data = W + (n * num_blocks_per_row) * 210;
    device const float* x_vec = x + m * K;

    float acc = 0.0f;

    for (uint b = 0; b < num_blocks_per_row; b++) {
        device const uchar* block = row_data + b * 210;
        uint k_base = b * 256;

        half d_h = as_type<half>(ushort(ushort(block[208]) | (ushort(block[209]) << 8)));
        float d = float(d_h);

        device const uchar* ql_all = block;
        device const uchar* qh_all = block + 128;
        device const char* sc_all = (device const char*)(block + 192);

        uint is = lane / 16;
        float s1 = float(sc_all[is]);
        float s2 = float(sc_all[is + 2]);
        float s3 = float(sc_all[is + 4]);
        float s4 = float(sc_all[is + 6]);
        float s5 = float(sc_all[is + 8]);
        float s6 = float(sc_all[is + 10]);
        float s7 = float(sc_all[is + 12]);
        float s8 = float(sc_all[is + 14]);

        // First half: 128 elements (g = 0)
        uchar ql1 = ql_all[lane];
        uchar ql2 = ql_all[lane + 32];
        uchar qh0 = qh_all[lane];

        int q1 = int((ql1 & 0x0F) | (((qh0 >> 0) & 3) << 4)) - 32;
        int q2 = int((ql2 & 0x0F) | (((qh0 >> 2) & 3) << 4)) - 32;
        int q3 = int((ql1 >> 4)   | (((qh0 >> 4) & 3) << 4)) - 32;
        int q4 = int((ql2 >> 4)   | (((qh0 >> 6) & 3) << 4)) - 32;

        float x1 = x_vec[k_base + lane];
        float x2 = x_vec[k_base + 32 + lane];
        float x3 = x_vec[k_base + 64 + lane];
        float x4 = x_vec[k_base + 96 + lane];

        acc += d * (s1 * float(q1) * x1 + s2 * float(q2) * x2 + s3 * float(q3) * x3 + s4 * float(q4) * x4);

        // Second half: 128 elements (g = 1)
        uchar ql3 = ql_all[64 + lane];
        uchar ql4 = ql_all[96 + lane];
        uchar qh1 = qh_all[32 + lane];

        int q5 = int((ql3 & 0x0F) | (((qh1 >> 0) & 3) << 4)) - 32;
        int q6 = int((ql4 & 0x0F) | (((qh1 >> 2) & 3) << 4)) - 32;
        int q7 = int((ql3 >> 4)   | (((qh1 >> 4) & 3) << 4)) - 32;
        int q8 = int((ql4 >> 4)   | (((qh1 >> 6) & 3) << 4)) - 32;

        float x5 = x_vec[k_base + 128 + lane];
        float x6 = x_vec[k_base + 160 + lane];
        float x7 = x_vec[k_base + 192 + lane];
        float x8 = x_vec[k_base + 224 + lane];

        acc += d * (s5 * float(q5) * x5 + s6 * float(q6) * x6 + s7 * float(q7) * x7 + s8 * float(q8) * x8);
    }

    acc = simd_sum(acc);

    if (lane == 0) {
        y[m * N + n] = acc;
    }
}
