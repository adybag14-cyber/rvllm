// Per-tensor FP8 rescaling kernel for SM90.
// After fused_rmsnorm_fp8_quant writes per-TOKEN scales, this kernel
// reduces them to a single per-TENSOR scale and adjusts the FP8 data
// to match. Required because cuBLASLt A_SCALE_POINTER reads a single
// scalar -- using per-token scales causes exponential error compounding.
//
// Launch: grid=(num_tokens), block=(min(dim, 1024)), smem=0
// Requires: scales buffer has (num_tokens + 2) floats. The last two
// are scratch for atomicMax and block counter (must be zeroed before launch).

#include <cuda_fp16.h>
#include <cuda_fp8.h>

extern "C" __global__ void __launch_bounds__(1024)
rescale_fp8_to_per_tensor_kernel(
    __nv_fp8_storage_t* __restrict__ data,
    float*              __restrict__ scales,
    int dim
) {
    const int row = blockIdx.x;
    const int num_rows = gridDim.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int row_offset = row * dim;

    float my_scale = scales[row];

    // Phase 1: atomicMax to find global max across all rows.
    // scales[num_rows]   = scratch for global max (caller must zero).
    // scales[num_rows+1] = scratch for done counter (caller must zero).
    if (tid == 0) {
        atomicMax((unsigned int*)&scales[num_rows],
                  __float_as_uint(my_scale));
        __threadfence();
        atomicAdd((unsigned int*)&scales[num_rows + 1], 1u);
    }
    __syncthreads();

    // Phase 2: spin until all blocks have contributed their max.
    if (tid == 0) {
        while (atomicAdd((unsigned int*)&scales[num_rows + 1], 0u)
               < (unsigned int)num_rows) {
            __nanosleep(32);
        }
    }
    __syncthreads();

    float global_scale = __uint_as_float(
        atomicAdd((unsigned int*)&scales[num_rows], 0u));

    // Phase 3: rescale FP8 data in this row.
    // old: fp8 = round(val / my_scale)
    // new: fp8 = round(val / global_scale) = round(fp8_old * my_scale / global_scale)
    float ratio = my_scale / global_scale;

    if (fabsf(ratio - 1.0f) > 1e-6f) {
        for (int i = tid; i < dim; i += stride) {
            // FP8 -> f16 -> f32, rescale, f32 -> FP8
            __half h = __nv_cvt_fp8_to_halfraw(data[row_offset + i],
                                                __NV_E4M3);
            float v = __half2float(h) * ratio;
            data[row_offset + i] = __nv_cvt_float_to_fp8(
                v, __NV_SATFINITE, __NV_E4M3);
        }
    }

    // Write global scale to scales[0] for cuBLASLt.
    if (row == 0 && tid == 0) {
        scales[0] = global_scale;
    }
}
