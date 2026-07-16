#include <metal_stdlib>
using namespace metal;

kernel void softmax_f32(
    device const float* input    [[buffer(0)]],
    device       float* output   [[buffer(1)]],
    constant     uint&  row_size [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    device const float* row = input + tid * row_size;
    device       float* out_row = output + tid * row_size;

    // Numerically stable softmax: subtract row max before exponentiating,
    // so exp() never sees a large positive input.
    float max_val = -INFINITY;
    for (uint i = 0; i < row_size; i++) {
        max_val = max(max_val, row[i]);
    }

    float sum = 0.0f;
    for (uint i = 0; i < row_size; i++) {
        float e = exp(row[i] - max_val);
        out_row[i] = e;
        sum += e;
    }

    for (uint i = 0; i < row_size; i++) {
        out_row[i] = out_row[i] / sum;
    }
}
