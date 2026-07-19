#include <metal_stdlib>
using namespace metal;

kernel void swiglu_f32(
    device const float* gate        [[buffer(0)]],  // Changed half* to float*
    device const float* up          [[buffer(1)]],  // Changed half* to float*
    device       float* output      [[buffer(2)]],  // Changed half* to float*
    constant     uint& num_elements [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_elements) return;

    // No more float() casts needed
    float gate_val = gate[tid];
    float up_val = up[tid];

    float silu_val = gate_val * (1.0f / (1.0f + exp(-gate_val)));
    output[tid] = silu_val * up_val;
}
