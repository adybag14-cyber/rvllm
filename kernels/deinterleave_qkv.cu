// Deinterleave fused QKV output from [M, qkv_dim] (interleaved per-token)
// to [M*q_dim, M*kv_dim, M*kv_dim] (split layout).
//
// Input:  [T0_q T0_k T0_v | T1_q T1_k T1_v | ...]  (M rows of qkv_dim)
// Output: [Q0 Q1 ... QM | K0 K1 ... KM | V0 V1 ... VM]
//
// Launch: grid(ceil(total/256)), block(256)

#include <cuda_fp16.h>

extern "C"
__global__ void deinterleave_qkv_f16_kernel(
    __half* __restrict__ output,       // [M * qkv_dim] split layout
    const __half* __restrict__ input,  // [M * qkv_dim] interleaved layout
    int num_tokens,
    int q_dim,
    int kv_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int qkv_dim = q_dim + kv_dim + kv_dim;
    const int total = num_tokens * qkv_dim;
    if (idx >= total) return;

    // Which token and which element within that token's qkv
    const int token = idx / qkv_dim;
    const int elem = idx % qkv_dim;

    // Read from interleaved layout
    __half val = input[idx];

    // Write to split layout
    int out_idx;
    if (elem < q_dim) {
        // Q region: token * q_dim + elem
        out_idx = token * q_dim + elem;
    } else if (elem < q_dim + kv_dim) {
        // K region: M*q_dim + token * kv_dim + (elem - q_dim)
        out_idx = num_tokens * q_dim + token * kv_dim + (elem - q_dim);
    } else {
        // V region: M*q_dim + M*kv_dim + token * kv_dim + (elem - q_dim - kv_dim)
        out_idx = num_tokens * (q_dim + kv_dim) + token * kv_dim + (elem - q_dim - kv_dim);
    }

    output[out_idx] = val;
}
