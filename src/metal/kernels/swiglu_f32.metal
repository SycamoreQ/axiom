#include <metal_stdlib>
using namespace metal;

kernel void swiglu_f32(
    device const float* gate        [[buffer(0)]],
    device const float* up          [[buffer(1)]],
    device       float* output      [[buffer(2)]],
    constant     uint&  num_elements [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint idx = tid * 4;
    if (idx + 3 < num_elements) {
        float4 g = *reinterpret_cast<device const float4*>(gate + idx);
        float4 u = *reinterpret_cast<device const float4*>(up + idx);
        float4 silu = g * (1.0f / (1.0f + exp(-g)));
        *reinterpret_cast<device float4*>(output + idx) = silu * u;
    } else {
        for (uint i = idx; i < num_elements; i++) {
            float g = gate[i];
            float u = up[i];
            float silu = g * (1.0f / (1.0f + exp(-g)));
            output[i] = silu * u;
        }
    }
}
