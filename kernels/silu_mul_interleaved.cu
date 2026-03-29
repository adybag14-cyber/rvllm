// SiLU*Mul on interleaved [N, 2*I] layout from fused gate+up GEMM.
// Input:  src[N, 2*I] where each row is [gate_i(I), up_i(I)]
// Output: dst[N, I] where each element is silu(gate) * up
// Eliminates transpose + 2x memcpy that the old path needed.

#include <cuda_fp16.h>

extern "C"
__global__ void silu_mul_interleaved_f16_kernel(
    __half* __restrict__ dst,        // [N * I]
    const __half* __restrict__ src,  // [N * 2*I]
    int N,
    int intermediate_size
) {
    const int total = N * intermediate_size;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int token = idx / intermediate_size;
    const int elem = idx % intermediate_size;
    const int row_offset = token * intermediate_size * 2;

    float gate = __half2float(src[row_offset + elem]);
    float up   = __half2float(src[row_offset + intermediate_size + elem]);

    float silu_gate = gate / (1.0f + expf(-gate));
    dst[idx] = __float2half(silu_gate * up);
}
