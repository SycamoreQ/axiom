// Residual attention decode kernel with f16 shared memory -- v1
// From the paper: ForkKV: Scaling Multi-LoRA Agent Serving via Copy-on-Write Disaggregated KV Cache\


#include <float.h>
#include <cuda_fp16.h>

#define RA_BC 128
#define RA_THREADS 128
#define RA_WARPS 4
#define HEAD_DIM 128
#define RANK 16
#define BLOCK_SIZE_M 128
#define BLOCK_SIZE_N 64

__device__ __force
