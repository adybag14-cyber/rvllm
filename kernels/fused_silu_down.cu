// Fused SiLU*Mul + Down-projection GEMV for M=1 decode (f16 I/O, f32 accumulation).
//
// Eliminates the intermediate activation buffer and one kernel launch by
// computing silu(gate)*up inline during the dot product with down_weight.
//
//   activation[i] = silu(gate[i]) * up[i]
//   output[j] = sum_i( activation[i] * down_weight[j, i] )
//
// Strategy:
//   Each block computes one output element output[j].
//   Threads stride over the intermediate dimension, computing:
//     partial_sum += silu(gate[i]) * up[i] * down_weight[j, i]
//   Then reduce via warp shuffles + shared memory.
//
//   No shared memory buffer for the activation is needed because each element
//   is consumed exactly once (no reuse across output rows). Each block reads
//   gate/up independently -- this is bandwidth-bound on the weight matrix anyway,
//   so the extra gate/up reads are overlapped by the weight reads.
//
// Launch config:
//   Grid:  (hidden_size, 1, 1)   -- one block per output element
//   Block: (THREADS, 1, 1)       -- THREADS=256
//   Shared memory: (THREADS/32) * sizeof(float)
//
// For Qwen2.5-1.5B:
//   hidden_size=1536, intermediate_size=8960
//   down_weight: [1536, 8960]

#include <cuda_fp16.h>

#define FUSED_SILU_DOWN_THREADS 256

__device__ __forceinline__ float warp_reduce_sum_fsd(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

extern "C"
__global__ void __launch_bounds__(FUSED_SILU_DOWN_THREADS)
fused_silu_down_f16_kernel(
    __half* __restrict__ output,          // [hidden_size]
    const __half* __restrict__ gate,      // [intermediate_size]
    const __half* __restrict__ up,        // [intermediate_size]
    const __half* __restrict__ down_weight, // [hidden_size, intermediate_size] row-major
    int hidden_size,
    int intermediate_size
) {
    const int row = blockIdx.x;
    if (row >= hidden_size) return;

    const int tid = threadIdx.x;
    const __half* w_row = down_weight + (long long)row * intermediate_size;

    float acc = 0.0f;

    // Vectorized half2 loads: process 2 elements at a time
    const int k2 = intermediate_size / 2;
    const half2* gate2 = (const half2*)gate;
    const half2* up2   = (const half2*)up;
    const half2* w2    = (const half2*)w_row;

    for (int i = tid; i < k2; i += FUSED_SILU_DOWN_THREADS) {
        half2 g = gate2[i];
        half2 u = up2[i];
        half2 w = w2[i];

        // silu(gate) * up * weight, computed in f32
        float g0 = __half2float(g.x);
        float g1 = __half2float(g.y);
        float u0 = __half2float(u.x);
        float u1 = __half2float(u.y);
        float w0 = __half2float(w.x);
        float w1 = __half2float(w.y);

        acc += silu_f32(g0) * u0 * w0;
        acc += silu_f32(g1) * u1 * w1;
    }

    // Handle odd intermediate_size
    if ((intermediate_size & 1) && tid == 0) {
        int last = intermediate_size - 1;
        float g = __half2float(gate[last]);
        acc += silu_f32(g) * __half2float(up[last]) * __half2float(w_row[last]);
    }

    // Warp-level reduction
    acc = warp_reduce_sum_fsd(acc);

    __shared__ float warp_sums[FUSED_SILU_DOWN_THREADS / 32];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = acc;
    }
    __syncthreads();

    // Final reduction in first warp -> write output
    if (warp_id == 0) {
        constexpr int NUM_WARPS = FUSED_SILU_DOWN_THREADS / 32;
        float val = (lane_id < NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum_fsd(val);
        if (lane_id == 0) {
            output[row] = __float2half(val);
        }
    }
}

// Variant with bias: output[j] = dot(silu(gate)*up, down_weight[j,:]) + bias[j]
extern "C"
__global__ void __launch_bounds__(FUSED_SILU_DOWN_THREADS)
fused_silu_down_bias_f16_kernel(
    __half* __restrict__ output,          // [hidden_size]
    const __half* __restrict__ gate,      // [intermediate_size]
    const __half* __restrict__ up,        // [intermediate_size]
    const __half* __restrict__ down_weight, // [hidden_size, intermediate_size] row-major
    const __half* __restrict__ bias,       // [hidden_size]
    int hidden_size,
    int intermediate_size
) {
    const int row = blockIdx.x;
    if (row >= hidden_size) return;

    const int tid = threadIdx.x;
    const __half* w_row = down_weight + (long long)row * intermediate_size;

    float acc = 0.0f;

    const int k2 = intermediate_size / 2;
    const half2* gate2 = (const half2*)gate;
    const half2* up2   = (const half2*)up;
    const half2* w2    = (const half2*)w_row;

    for (int i = tid; i < k2; i += FUSED_SILU_DOWN_THREADS) {
        half2 g = gate2[i];
        half2 u = up2[i];
        half2 w = w2[i];

        float g0 = __half2float(g.x);
        float g1 = __half2float(g.y);
        float u0 = __half2float(u.x);
        float u1 = __half2float(u.y);
        float w0 = __half2float(w.x);
        float w1 = __half2float(w.y);

        acc += silu_f32(g0) * u0 * w0;
        acc += silu_f32(g1) * u1 * w1;
    }

    if ((intermediate_size & 1) && tid == 0) {
        int last = intermediate_size - 1;
        float g = __half2float(gate[last]);
        acc += silu_f32(g) * __half2float(up[last]) * __half2float(w_row[last]);
    }

    acc = warp_reduce_sum_fsd(acc);

    __shared__ float warp_sums[FUSED_SILU_DOWN_THREADS / 32];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        constexpr int NUM_WARPS = FUSED_SILU_DOWN_THREADS / 32;
        float val = (lane_id < NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum_fsd(val);
        if (lane_id == 0) {
            output[row] = __float2half(val + __half2float(bias[row]));
        }
    }
}
