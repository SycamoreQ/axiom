#include <metal_stdlib>
using namespace metal;

kernel void softmax_f16(
    device const half*  input    [[buffer(0)]],
    device       half*  output   [[buffer(1)]],
    constant     uint&  row_size [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    device const half* row = input + tid * row_size;
    device       half* out_row = output + tid * row_size;

    // Accumulate in float even though storage is half, same reasoning as
    // rms_norm_f16 — avoids compounding half-precision rounding error
    // across the row-wide sum.
    float max_val = -INFINITY;
    for (uint i = 0; i < row_size; i++) {
        max_val = max(max_val, float(row[i]));
    }

    float sum = 0.0f;
    for (uint i = 0; i < row_size; i++) {
        float e = exp(float(row[i]) - max_val);
        out_row[i] = half(e);
        sum += e;
    }

    for (uint i = 0; i < row_size; i++) {
        out_row[i] = half(float(out_row[i]) / sum);
    }
}
