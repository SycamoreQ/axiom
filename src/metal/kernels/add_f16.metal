#include <metal_stdlib>
using namespace metal;

kernel void add_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c [[buffer(2)]],
    constant uint& num_elements [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < num_elements) {
        c[index] = a[index] + b[index];
    }
}
