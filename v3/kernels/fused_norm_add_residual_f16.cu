// Fused: channelscale(f32) + rmsnorm + add-to-residual(f16)
//
// Like fused_norm_add_residual but also applies per-channel weight scale.
// Eliminates the separate scale_cols_f32 kernel.
//
// Input:  f32* gemm_out [num_tokens, hidden]  (FP8 GEMM F32 output)
//         f32* channelscale [hidden]           (per-channel weight scale)
//         f16* gamma    [hidden]               (norm weight)
//         f16* residual [num_tokens, hidden]   (read+write)
//         float eps
//
// Grid: (num_tokens), Block: (min(hidden, 1024))
// Shared memory: hidden * sizeof(float)

#include <cuda_fp16.h>

extern "C" __global__ void fused_norm_add_residual_f16_kernel(
    const float* __restrict__ gemm_out,
    const float* __restrict__ channelscale,
    const half*  __restrict__ gamma,
    half*        __restrict__ residual,
    int hidden,
    float eps
) {
    extern __shared__ float svals[];

    int token = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* row = gemm_out + (size_t)token * hidden;
    half* res = residual + (size_t)token * hidden;

    // Pass 1: read f32, apply channelscale, cache in smem, accumulate sum_sq
    float local_ss = 0.0f;
    for (int i = tid; i < hidden; i += stride) {
        float v = row[i] * channelscale[i];
        svals[i] = v;
        local_ss += v * v;
    }

    // Warp reduce
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_ss += __shfl_xor_sync(0xffffffff, local_ss, offset);

    __shared__ float warp_ss[32];
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    if (lane == 0) warp_ss[warp_id] = local_ss;
    __syncthreads();

    if (tid == 0) {
        int nw = (stride + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (int w = 0; w < nw; w++) total += warp_ss[w];
        warp_ss[0] = total;
    }
    __syncthreads();
    float rms_inv = rsqrtf(warp_ss[0] / (float)hidden + eps);

    // Pass 2: normalize, scale by gamma, add to residual
    for (int i = tid; i < hidden; i += stride) {
        float normed = svals[i] * rms_inv * __half2float(gamma[i]);
        float r = __half2float(res[i]) + normed;
        res[i] = __float2half(r);
    }
}
