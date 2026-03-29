// Broadcast add bias to [N, dim] tensor in-place.
// bias[dim] is added to each of the N rows.
// Grid: (ceil(N * dim / 256), 1, 1)  Block: (256, 1, 1)

#include <cuda_fp16.h>

extern "C"
__global__ void add_bias_broadcast_f16_kernel(
    __half* __restrict__ tensor,     // [N * dim], modified in-place
    const __half* __restrict__ bias, // [dim]
    int N,
    int dim
) {
    const int total = N * dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int elem = idx % dim;
    float val = __half2float(tensor[idx]) + __half2float(bias[elem]);
    tensor[idx] = __float2half(val);
}
