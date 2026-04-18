// Fused GELU(tanh) * mul + per-tensor FP8 E4M3 quantization.
//
// Replaces fused_silu_mul_fp8_quant for Gemma 4 (uses GELU(tanh) activation).
//
// Input:  gate_up [num_tokens, 2 * intermediate] f16
//         Layout: first `intermediate` elements = gate, next = up
// Output: out_fp8 [num_tokens, intermediate] FP8 E4M3
//         scale   [num_tokens] f32 per-tensor scale
//
// GELU(tanh)(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

__device__ __forceinline__ float gelu_tanh(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

extern "C"
__global__ void fused_gelu_mul_fp8_quant_kernel(
    __nv_fp8_e4m3* __restrict__ out_fp8,    // [num_tokens, intermediate]
    float* __restrict__ scale,               // [num_tokens]
    const __half* __restrict__ gate_up,      // [num_tokens, 2 * intermediate]
    int intermediate
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    // Shared memory for warp-level amax reduction
    extern __shared__ float smem[];

    const __half* gate_row = gate_up + row * 2 * intermediate;
    const __half* up_row = gate_row + intermediate;

    float local_amax = 0.0f;

    for (int i = tid; i < intermediate; i += blockDim.x) {
        float g = __half2float(gate_row[i]);
        float u = __half2float(up_row[i]);
        float val = gelu_tanh(g) * u;
        float aval = fabsf(val);
        if (aval > local_amax) local_amax = aval;
    }

    // Warp reduce
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, local_amax, offset);
        if (other > local_amax) local_amax = other;
    }

    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    if (lane == 0) smem[warp_id] = local_amax;
    __syncthreads();

    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (warp_id == 0) {
        float v = (lane < num_warps) ? smem[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffff, v, offset);
            if (other > v) v = other;
        }
        if (lane == 0) {
            float s = fmaxf(v, 1e-12f) / 448.0f;
            smem[0] = s;
            scale[row] = s;
        }
    }
    __syncthreads();

    float inv_scale = 1.0f / smem[0];

    __nv_fp8_e4m3* out_row = out_fp8 + row * intermediate;
    for (int i = tid; i < intermediate; i += blockDim.x) {
        float g = __half2float(gate_row[i]);
        float u = __half2float(up_row[i]);
        float val = gelu_tanh(g) * u;
        out_row[i] = __nv_fp8_e4m3(val * inv_scale);
    }
}
