#include <metal_stdlib>
using namespace metal;

kernel void swiglu_f16(
    device const half* gate         [[buffer(0)]],
    device const half* up           [[buffer(1)]],
    device       half* output       [[buffer(2)]],
    constant     uint& num_elements [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_elements) return;

    float gate_val = float(gate[tid]);
    float up_val = float(up[tid]);

    float silu_val = gate_val * (1.0f / (1.0f + exp(-gate_val)));
    output[tid] = silu_val * up_val;
}
