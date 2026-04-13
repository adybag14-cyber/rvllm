// Fused SiLU(gate)*up + per-token FP8 E4M3 quantization for SM90.
// Input layout: [num_tokens, 2 * intermediate_size] as [gate | up] per row.
// Output: [num_tokens, intermediate_size] in FP8 with per-row scales.
// Compile: nvcc -ptx -arch=sm_90 -O3 --use_fast_math

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define FP8_E4M3_MAX 448.0f
#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    val = warp_reduce_max(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
    if (warp_id == 0) val = warp_reduce_max(val);
    return val;
}

// SiLU(gate) * up + per-token FP8 quantization.
// grid=(num_tokens), block=(min(intermediate_size, 1024))
// shared mem: WARPS_MAX * sizeof(float)
extern "C" __global__ void __launch_bounds__(1024)
fused_silu_mul_fp8_quant_kernel(
    __nv_fp8_storage_t* __restrict__ output_fp8,
    float*              __restrict__ output_scales,
    const __half*       __restrict__ gate_up,
    int intermediate_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int src_row_offset = row * 2 * intermediate_size;
    const int dst_row_offset = row * intermediate_size;

    __shared__ float smem[WARPS_MAX];

    // Pass 1: compute silu(gate)*up, find absmax
    float local_max = 0.0f;
    for (int i = tid; i < intermediate_size; i += stride) {
        float g = __half2float(gate_up[src_row_offset + i]);
        float u = __half2float(gate_up[src_row_offset + intermediate_size + i]);
        float silu_g = g / (1.0f + expf(-g));
        float v = silu_g * u;
        local_max = fmaxf(local_max, fabsf(v));
    }
    float absmax = block_reduce_max(local_max, smem);
    __syncthreads();

    float scale = absmax / FP8_E4M3_MAX;
    scale = fmaxf(scale, 1e-12f);
    if (tid == 0) output_scales[row] = scale;
    float inv_scale = 1.0f / scale;

    // Pass 2: recompute and quantize
    for (int i = tid; i < intermediate_size; i += stride) {
        float g = __half2float(gate_up[src_row_offset + i]);
        float u = __half2float(gate_up[src_row_offset + intermediate_size + i]);
        float silu_g = g / (1.0f + expf(-g));
        float v = silu_g * u;
        output_fp8[dst_row_offset + i] = __nv_cvt_float_to_fp8(v * inv_scale, __NV_SATFINITE, __NV_E4M3);
    }
}
