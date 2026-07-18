#include <metal_stdlib>
using namespace metal;

kernel void attention_pv_f32(
    device const float* scores       [[buffer(0)]],
    device const float* V_cache      [[buffer(1)]],
    device       float* output       [[buffer(2)]],
    constant uint& n_heads           [[buffer(3)]],
    constant uint& n_kv_heads        [[buffer(4)]],
    constant uint& seq_len           [[buffer(5)]],
    constant uint& head_dim          [[buffer(6)]],
    constant uint& current_pos       [[buffer(7)]],
    uint head                        [[threadgroup_position_in_grid]],
    uint lid                         [[thread_position_in_threadgroup]]
) {
    const uint THREADGROUP_SIZE = 32;
    threadgroup float scratch[THREADGROUP_SIZE];
    uint repeat_factor = n_heads / n_kv_heads;
    uint kv_head = head / repeat_factor;

    float local_max = -INFINITY;
    for (uint i = lid; i < seq_len; i += THREADGROUP_SIZE)
        local_max = max(local_max, scores[head * seq_len + i]);
    scratch[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = THREADGROUP_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = max(scratch[lid], scratch[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_max = scratch[0];

    float local_sum = 0.0f;
    for (uint i = lid; i < seq_len; i += THREADGROUP_SIZE)
        local_sum += exp(scores[head * seq_len + i] - global_max);
    scratch[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = THREADGROUP_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_sum = scratch[0];

    for (uint d = lid; d < head_dim; d += THREADGROUP_SIZE) {
        float acc = 0.0f;
        for (uint i = 0; i < seq_len; i++) {
            float weight = exp(scores[head * seq_len + i] - global_max) / global_sum;
            float v_val  = V_cache[(i * n_kv_heads + kv_head) * head_dim + d];
            acc += weight * v_val;
        }
        output[head * head_dim + d] = acc;
    }
}
