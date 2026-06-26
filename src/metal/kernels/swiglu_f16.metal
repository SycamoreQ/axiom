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

    // compute silu(up_val) — you fill this in
    float silu_val = up_val * (1.0f / (1.0f + exp(-up_val)));

    // fuse: gate * silu(up)
    output[tid] = half(gate_val * silu_val);
}
