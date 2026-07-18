#include <metal_stdlib>
using namespace metal;

kernel void add_f32(
    device const float* a           [[buffer(0)]],
    device const float* b           [[buffer(1)]],
    device       float* output      [[buffer(2)]],
    constant     uint& num_elements [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_elements) return;
    output[tid] = a[tid] + b[tid];
}
