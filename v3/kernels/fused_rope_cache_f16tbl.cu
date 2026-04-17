// Same as fused_rope_cache.cu but cos/sin tables are __half (not float).
// Lets v3 keep zero f32 activations/constants in the decode path.

#include <cuda_fp16.h>

extern "C"
__global__ void fused_rope_cache_f16tbl_kernel(
    __half* __restrict__ q,
    __half* __restrict__ k,
    const __half* __restrict__ v,
    __half* __restrict__ key_cache,
    __half* __restrict__ value_cache,
    const __half* __restrict__ cos_table,
    const __half* __restrict__ sin_table,
    const int* __restrict__ positions,
    const int* __restrict__ slot_mapping,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int half_dim  = head_dim / 2;
    const int tid       = threadIdx.x;
    if (tid >= half_dim) return;

    const int pos = positions[token_idx];
    const float cos_val = __half2float(cos_table[pos * half_dim + tid]);
    const float sin_val = __half2float(sin_table[pos * half_dim + tid]);

    if (head_idx < num_heads) {
        int q_base = (token_idx * num_heads + head_idx) * head_dim;
        float q0 = __half2float(q[q_base + 2 * tid]);
        float q1 = __half2float(q[q_base + 2 * tid + 1]);
        q[q_base + 2 * tid]     = __float2half(q0 * cos_val - q1 * sin_val);
        q[q_base + 2 * tid + 1] = __float2half(q0 * sin_val + q1 * cos_val);
    }

    if (head_idx < num_kv_heads) {
        int k_base = (token_idx * num_kv_heads + head_idx) * head_dim;
        float k0 = __half2float(k[k_base + 2 * tid]);
        float k1 = __half2float(k[k_base + 2 * tid + 1]);
        float k0_rot = k0 * cos_val - k1 * sin_val;
        float k1_rot = k0 * sin_val + k1 * cos_val;
        k[k_base + 2 * tid]     = __float2half(k0_rot);
        k[k_base + 2 * tid + 1] = __float2half(k1_rot);

        int slot = slot_mapping[token_idx];
        if (slot >= 0) {
            int cache_offset = (slot * num_kv_heads + head_idx) * head_dim;
            key_cache[cache_offset + 2 * tid]     = __float2half(k0_rot);
            key_cache[cache_offset + 2 * tid + 1] = __float2half(k1_rot);
            int v_base = (token_idx * num_kv_heads + head_idx) * head_dim;
            value_cache[cache_offset + 2 * tid]     = v[v_base + 2 * tid];
            value_cache[cache_offset + 2 * tid + 1] = v[v_base + 2 * tid + 1];
        }
    }
}
