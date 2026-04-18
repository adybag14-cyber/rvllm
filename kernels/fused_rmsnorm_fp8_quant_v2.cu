// Fused RMSNorm + per-TENSOR FP8 E4M3 quantization kernels for SM90.
// v2: uses atomicMax + spin counter for cross-block global absmax so
// cuBLASLt gets a single correct scale in scales[0]. Eliminates the
// separate rescale kernel that caused 47% throughput regression.
//
// Scratch: scales buffer must be (num_tokens + 2) floats. Last two
// are atomicMax scratch and spin counter (caller zeros them via
// cuMemsetD32Async before launch).
//
// All kernels: 1 block per token, warp-shuffle reductions.

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define FP8_E4M3_MAX 448.0f
#define WARPS_MAX 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    val = warp_reduce_sum(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);
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

// Grid-wide barrier using atomicAdd counter.
// scales[num_rows]   = global absmax (atomicMax target, pre-zeroed)
// scales[num_rows+1] = done counter (pre-zeroed)
__device__ __forceinline__ float grid_absmax_barrier(
    float local_absmax, float* scales, int num_rows
) {
    if (threadIdx.x == 0) {
        atomicMax((unsigned int*)&scales[num_rows],
                  __float_as_uint(local_absmax));
        __threadfence();
        atomicAdd((unsigned int*)&scales[num_rows + 1], 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        while (atomicAdd((unsigned int*)&scales[num_rows + 1], 0u)
               < (unsigned int)num_rows) {
            __nanosleep(32);
        }
    }
    __syncthreads();
    return __uint_as_float(
        atomicAdd((unsigned int*)&scales[num_rows], 0u));
}

// Kernel A: RMSNorm + per-tensor FP8 E4M3 quantization.
extern "C" __global__ void __launch_bounds__(1024)
fused_rmsnorm_fp8_quant_kernel(
    __nv_fp8_storage_t* __restrict__ output_fp8,
    float*              __restrict__ output_scales,
    const __half*       __restrict__ input,
    const __half*       __restrict__ weight,
    float eps,
    int hidden_size
) {
    const int row = blockIdx.x;
    const int num_rows = gridDim.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int row_offset = row * hidden_size;

    __shared__ float smem[WARPS_MAX];

    // Pass 1: sum of squares
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float v = __half2float(input[row_offset + i]);
        local_ss += v * v;
    }
    float sum_sq = block_reduce_sum(local_ss, smem);
    __syncthreads();

    float rms = rsqrtf(sum_sq / (float)hidden_size + eps);

    // Pass 2: compute normed values, find per-row absmax
    float local_max = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float v = __half2float(input[row_offset + i]) * rms * __half2float(weight[i]);
        local_max = fmaxf(local_max, fabsf(v));
    }
    float row_absmax = block_reduce_max(local_max, smem);
    __syncthreads();

    // Cross-block reduction to global absmax via atomicMax + barrier.
    float global_absmax = grid_absmax_barrier(row_absmax, output_scales, num_rows);

    float scale = global_absmax / FP8_E4M3_MAX;
    scale = fmaxf(scale, 1e-12f);
    if (row == 0 && tid == 0) output_scales[0] = scale;
    float inv_scale = 1.0f / scale;

    // Pass 3: quantize to FP8 with the global scale
    for (int i = tid; i < hidden_size; i += stride) {
        float v = __half2float(input[row_offset + i]) * rms * __half2float(weight[i]);
        output_fp8[row_offset + i] = __nv_cvt_float_to_fp8(v * inv_scale, __NV_SATFINITE, __NV_E4M3);
    }
}

// Kernel B: Residual add + RMSNorm + per-tensor FP8 E4M3 quantization.
extern "C" __global__ void __launch_bounds__(1024)
fused_add_rmsnorm_fp8_quant_kernel(
    __nv_fp8_storage_t* __restrict__ output_fp8,
    float*              __restrict__ output_scales,
    __half*             __restrict__ residual_out,
    const __half*       __restrict__ input,
    const __half*       __restrict__ residual,
    const __half*       __restrict__ weight,
    float eps,
    int hidden_size
) {
    const int row = blockIdx.x;
    const int num_rows = gridDim.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int row_offset = row * hidden_size;

    __shared__ float smem[WARPS_MAX];

    // Pass 1: residual add + sum of squares
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float v = __half2float(input[row_offset + i]) + __half2float(residual[row_offset + i]);
        residual_out[row_offset + i] = __float2half(v);
        local_ss += v * v;
    }
    float sum_sq = block_reduce_sum(local_ss, smem);
    __syncthreads();

    float rms = rsqrtf(sum_sq / (float)hidden_size + eps);

    // Pass 2: normed values + per-row absmax
    float local_max = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float v = __half2float(residual_out[row_offset + i]) * rms * __half2float(weight[i]);
        local_max = fmaxf(local_max, fabsf(v));
    }
    float row_absmax = block_reduce_max(local_max, smem);
    __syncthreads();

    float global_absmax = grid_absmax_barrier(row_absmax, output_scales, num_rows);

    float scale = global_absmax / FP8_E4M3_MAX;
    scale = fmaxf(scale, 1e-12f);
    if (row == 0 && tid == 0) output_scales[0] = scale;
    float inv_scale = 1.0f / scale;

    // Pass 3: quantize
    for (int i = tid; i < hidden_size; i += stride) {
        float v = __half2float(residual_out[row_offset + i]) * rms * __half2float(weight[i]);
        output_fp8[row_offset + i] = __nv_cvt_float_to_fp8(v * inv_scale, __NV_SATFINITE, __NV_E4M3);
    }
}

// Kernel C: Plain per-tensor FP8 E4M3 quantization (no norm).
extern "C" __global__ void __launch_bounds__(1024)
quantize_fp8_per_token_kernel(
    __nv_fp8_storage_t* __restrict__ output_fp8,
    float*              __restrict__ output_scales,
    const __half*       __restrict__ input,
    int dim
) {
    const int row = blockIdx.x;
    const int num_rows = gridDim.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int row_offset = row * dim;

    __shared__ float smem[WARPS_MAX];

    // Pass 1: find row absmax
    float local_max = 0.0f;
    for (int i = tid; i < dim; i += stride) {
        local_max = fmaxf(local_max, fabsf(__half2float(input[row_offset + i])));
    }
    float row_absmax = block_reduce_max(local_max, smem);
    __syncthreads();

    float global_absmax = grid_absmax_barrier(row_absmax, output_scales, num_rows);

    float scale = global_absmax / FP8_E4M3_MAX;
    scale = fmaxf(scale, 1e-12f);
    if (row == 0 && tid == 0) output_scales[0] = scale;
    float inv_scale = 1.0f / scale;

    // Pass 2: quantize
    for (int i = tid; i < dim; i += stride) {
        float v = __half2float(input[row_offset + i]) * inv_scale;
        output_fp8[row_offset + i] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    }
}
