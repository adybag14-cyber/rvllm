// Mega-fused O-projection + residual-add + RMSNorm + gate_up GEMV kernel (M=1 decode).
//
// Eliminates the intermediate attn_proj HBM write+read by computing:
//   oproj[h] = dot(o_weight[h, :], attn_out)      -- O projection
//   residual_out = residual + oproj                 -- residual add
//   normed = rmsnorm(residual_out, norm_weight, eps) -- RMSNorm
//   gate_up_out[j] = dot(gateup_weight[j, :], normed) -- gate_up GEMV
//
// Phase 1: All blocks redundantly compute O-proj into shared memory.
//   O_weight fits in L2 (~4.7MB for hidden=1536), so redundant reads are cheap.
// Phase 2: Residual add + RMSNorm in shared memory.
// Phase 3: Each block computes RPB=8 rows of gate_up GEMV.
//
// Launch config:
//   Grid:  ((gate_up_dim + RPB - 1) / RPB, 1, 1)
//   Block: (256, 1, 1)
//   Shared mem: hidden_size * sizeof(float) + 8 * sizeof(float)

#include <cuda_fp16.h>

#define THREADS 256
#define RPB 8

__device__ __forceinline__ float warp_reduce_sum_opag(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_oproj_add_norm_gateup_gemv(
    __half* __restrict__ gate_up_out,       // [gate_up_dim]
    __half* __restrict__ residual_out,      // [hidden_size]
    const __half* __restrict__ attn_out,    // [q_dim]
    const __half* __restrict__ o_weight,    // [hidden_size, q_dim] row-major
    const __half* __restrict__ residual,    // [hidden_size]
    const __half* __restrict__ norm_weight, // [hidden_size]
    const __half* __restrict__ gateup_weight, // [gate_up_dim, hidden_size] row-major
    float eps,
    int q_dim,
    int hidden_size,
    int gate_up_dim
) {
    const int block_row_base = blockIdx.x * RPB;
    if (block_row_base >= gate_up_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    // Shared memory layout:
    //   [0 .. hidden_size-1]      : O-proj result / normed hidden state (f32)
    //   [hidden_size .. hidden_size+7] : warp partial sums scratch
    extern __shared__ float smem[];
    float* s_hidden = smem;
    float* s_warp   = smem + hidden_size;

    // ---- Phase 1: O-projection GEMV into shared memory ----
    // Each block computes all hidden_size output elements.
    // For each output element h: oproj[h] = dot(o_weight[h,:], attn_out)
    // attn_out is small (q_dim * 2 bytes), will be cached in L1 after first pass.
    // o_weight rows are streamed; the full matrix fits in L2 for small models.

    const int q2 = q_dim / 2;
    const half2* attn2 = (const half2*)attn_out;

    for (int h = 0; h < hidden_size; h++) {
        const half2* ow2 = (const half2*)(o_weight + (long long)h * q_dim);
        float acc = 0.0f;

        for (int i = tid; i < q2; i += THREADS) {
            half2 a = attn2[i];
            half2 w = ow2[i];
            acc += __half2float(a.x) * __half2float(w.x)
                 + __half2float(a.y) * __half2float(w.y);
        }
        // Handle odd q_dim
        if ((q_dim & 1) && tid == 0) {
            acc += __half2float(attn_out[q_dim - 1]) * __half2float(o_weight[(long long)h * q_dim + q_dim - 1]);
        }

        // Warp reduction
        acc = warp_reduce_sum_opag(acc);
        if (lane_id == 0) s_warp[warp_id] = acc;
        __syncthreads();

        // Cross-warp reduction in first warp
        if (warp_id == 0) {
            float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
            val = warp_reduce_sum_opag(val);
            if (lane_id == 0) {
                s_hidden[h] = val;
            }
        }
        // Must sync before next iteration reuses s_warp
        if (h + 1 < hidden_size) __syncthreads();
    }
    __syncthreads();

    // ---- Phase 2: Residual add + RMSNorm ----
    const int h2 = hidden_size / 2;
    const half2* res2 = (const half2*)residual;

    float local_ss = 0.0f;
    for (int i = tid; i < h2; i += THREADS) {
        half2 r = res2[i];
        float v0 = s_hidden[i * 2]     + __half2float(r.x);
        float v1 = s_hidden[i * 2 + 1] + __half2float(r.y);
        s_hidden[i * 2]     = v0;
        s_hidden[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        int last = hidden_size - 1;
        float v = s_hidden[last] + __half2float(residual[last]);
        s_hidden[last] = v;
        local_ss += v * v;
    }

    // Warp reduction of sum-of-squares
    local_ss = warp_reduce_sum_opag(local_ss);
    if (lane_id == 0) s_warp[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
        val = warp_reduce_sum_opag(val);
        if (lane_id == 0) {
            s_warp[0] = rsqrtf(val / (float)hidden_size + eps);
        }
    }
    __syncthreads();

    float rms_scale = s_warp[0];

    // Block 0 writes residual_out (pre-norm residual for next layer)
    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += THREADS) {
            residual_out[i] = __float2half(s_hidden[i]);
        }
    }

    // Apply norm weights in-place in smem
    for (int i = tid; i < hidden_size; i += THREADS) {
        s_hidden[i] = s_hidden[i] * __half2float(norm_weight[i]) * rms_scale;
    }
    __syncthreads();

    // ---- Phase 3: gate_up GEMV -- RPB=8 dot products per block ----
    const int rows_this_block = min(RPB, gate_up_dim - block_row_base);

    float acc[RPB];
    #pragma unroll
    for (int r = 0; r < RPB; r++) acc[r] = 0.0f;

    for (int i = tid; i < h2; i += THREADS) {
        float sn0 = s_hidden[i * 2];
        float sn1 = s_hidden[i * 2 + 1];
        #pragma unroll
        for (int r = 0; r < RPB; r++) {
            if (r < rows_this_block) {
                const half2* w2 = (const half2*)(gateup_weight + (long long)(block_row_base + r) * hidden_size);
                half2 w = w2[i];
                acc[r] += __half2float(w.x) * sn0 + __half2float(w.y) * sn1;
            }
        }
    }

    // Handle odd hidden_size
    if ((hidden_size & 1) && tid == 0) {
        int last = hidden_size - 1;
        float sn = s_hidden[last];
        #pragma unroll
        for (int r = 0; r < RPB; r++) {
            if (r < rows_this_block) {
                const __half* w_row = gateup_weight + (long long)(block_row_base + r) * hidden_size;
                acc[r] += __half2float(w_row[last]) * sn;
            }
        }
    }

    // Reduce each row's dot product
    #pragma unroll
    for (int r = 0; r < RPB; r++) {
        if (r >= rows_this_block) break;

        float val = warp_reduce_sum_opag(acc[r]);
        if (lane_id == 0) s_warp[warp_id] = val;
        __syncthreads();

        if (warp_id == 0) {
            float v = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
            v = warp_reduce_sum_opag(v);
            if (lane_id == 0) {
                gate_up_out[block_row_base + r] = __float2half(v);
            }
        }
        __syncthreads();
    }
}
