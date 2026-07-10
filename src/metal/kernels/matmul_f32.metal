#include <metal_stdlib>
using namespace metal;

kernel void matmul_f32(
    device const float* A       [[buffer(0)]],
    device const float* B       [[buffer(1)]],
    device       float* C       [[buffer(2)]],
    constant     uint&  M       [[buffer(3)]],
    constant     uint&  N       [[buffer(4)]],
    constant     uint&  K       [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    const uint BLOCKSIZE = 16;
    threadgroup float tile_a[16][16];
    threadgroup float tile_b[16][16];

    uint row = gid.y * BLOCKSIZE + lid.y;
    uint col = gid.x * BLOCKSIZE + lid.x;
    float acc = 0.0f;

    uint num_tiles = (K + BLOCKSIZE - 1) / BLOCKSIZE;
    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * BLOCKSIZE + lid.x;
        uint b_row = t * BLOCKSIZE + lid.y;

        tile_a[lid.y][lid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        tile_b[lid.y][lid.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < BLOCKSIZE; i++) {
            acc += tile_a[lid.y][i] * tile_b[i][lid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
