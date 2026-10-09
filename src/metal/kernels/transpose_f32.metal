#include <metal_stdlib>
using namespace metal;

kernel void transpose_f32(
    device const float* in       [[buffer(0)]],
    device       float* out      [[buffer(1)]],
    constant     uint&  rows     [[buffer(2)]],
    constant     uint&  cols     [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= cols || gid.y >= rows) return;
    out[gid.x * rows + gid.y] = in[gid.y * cols + gid.x];
}
