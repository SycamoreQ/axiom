#include <metal_stdlib>
using namespace metal;

kernel void swiglu_f16(
    device const half* gate         [[buffer(0)]],
    device const half* up           [[buffer(1)]],
    device       half* output       [[buffer(2)]],
    constant     uint& num_elements [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint idx = tid * 4;
    if (idx + 3 < num_elements) {
        half4 g = *reinterpret_cast<device const half4*>(gate + idx);
        half4 u = *reinterpret_cast<device const half4*>(up + idx);
        float4 gf = float4(g);
        float4 silu = gf * (1.0f / (1.0f + exp(-gf)));
        *reinterpret_cast<device half4*>(output + idx) = half4(silu * float4(u));
    } else {
        for (uint i = idx; i < num_elements; i++) {
            float g = float(gate[i]);
            float u = float(up[i]);
            float silu = g * (1.0f / (1.0f + exp(-g)));
            output[i] = half(silu * u);
        }
    }
}
