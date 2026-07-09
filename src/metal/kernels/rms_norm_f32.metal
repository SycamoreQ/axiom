#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_f32(
    device const float* input    [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    device       float* output   [[buffer(2)]],
    constant     uint&  hidden   [[buffer(3)]],
    constant     float& eps      [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    device const float* row = input + tid * hidden;

    float mean_sq = 0.0f;
    for (uint i = 0; i < hidden; i++) {
        float x = row[i];
        mean_sq += x * x;
    }
    mean_sq /= float(hidden);

    float scale = rsqrt(mean_sq + eps);

    device float* out_row = output + tid * hidden;
    for (uint i = 0; i < hidden; i++) {
        out_row[i] = row[i] * scale * weight[i];
    }
}
