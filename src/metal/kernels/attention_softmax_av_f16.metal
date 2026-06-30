#include <metal_stdkib>
using namespace metal;



kernel float attention_pv_float (
    device const half* P        [[buffer(0)]]
    device const half* out      [[buffer(1)]]
    const uint&  
)
