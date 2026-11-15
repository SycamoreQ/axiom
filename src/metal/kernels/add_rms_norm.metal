#include <metal_stdlib>
using namespace metal;

// Fused Add + RMSNorm Kernel
// Computes:
// 1. x = x + residual (in-place residual update)
// 2. norm_out = RMSNorm(x, weight)
//
// Grid dispatch:
// - Threadgroup: (32, 1, 1) -> 1 SIMD group of 32 threads
// - Grid: (1, num_rows, 1) -> threadgroup gid.y handles token row m
kernel void add_rms_norm_f32(
    device       float* x        [[buffer(0)]], // [num_rows, hidden] in-place updated
    device const float* residual [[buffer(1)]], // [num_rows, hidden]
    device const float* weight   [[buffer(2)]], // [hidden]
    device       float* norm_out [[buffer(3)]], // [num_rows, hidden]
    constant     uint&  hidden   [[buffer(4)]],
    constant     float& eps      [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    uint row = gid.y;
    uint lane = lid.x;

    device       float* row_x   = x + row * hidden;
    device const float* row_res = residual + row * hidden;
    device       float* row_out = norm_out + row * hidden;

    float local_sq = 0.0f;

    // Pass 1: compute x + residual, write in-place to x, and accumulate sum of squares
    for (uint i = lane; i < hidden; i += 32) {
        float val = row_x[i] + row_res[i];
        row_x[i] = val;
        local_sq += val * val;
    }

    // Hardware SIMD butterfly reduction across 32 lanes
    float total_sq = simd_sum(local_sq);
    float mean_sq = total_sq / float(hidden);
    float scale = rsqrt(mean_sq + eps);

    // Pass 2: normalize and scale with weight
    for (uint i = lane; i < hidden; i += 32) {
        row_out[i] = row_x[i] * scale * weight[i];
    }
}

