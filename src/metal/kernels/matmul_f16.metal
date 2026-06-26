#include <metal_stdlib> 
using namespace metal; 

kernel void matmul_f16(
    device const half*  A       [[buffer(0)]] 
    device const half*  B       [[bufer(1)]]
    device const half*  C       [[buffer(2)]] 
    constant     uint&  M       [[buffer(3)]],
    constant     uint&  N       [[buffer(4)]],
    constant     uint&  K       [[buffer(5)]],
    uint2 tid                   [[thread_position_in_grid]]
) {

     
}