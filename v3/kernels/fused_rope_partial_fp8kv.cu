// Partial RoPE + FP8 paged-KV-cache write (Gemma 4).
//
// Same as fused_rope_cache_fp8kv.cu but with `rotary_dim` parameter.
// Only the first `rotary_dim` elements of each head get RoPE rotation;
// elements [rotary_dim..head_dim) pass through unchanged (still
// quantized to FP8 and written to cache).
//
// Gemma 4 global attention layers use partial_rotary_factor=0.25,
// meaning only 64 of 256 dims are rotated. Sliding layers use 0.5
// (128 of 256). The cos/sin tables are pre-sized to rotary_dim/2.

#include <cuda_fp16.h>
#include <cuda_fp8.h>

extern "C"
__global__ void fused_rope_partial_fp8kv_kernel(
    const __half* __restrict__ q_in,
    const __half* __restrict__ k_in,
    const __half* __restrict__ v_in,
    __nv_fp8_e4m3* __restrict__ q_fp8_out,
    __nv_fp8_e4m3* __restrict__ key_cache,
    __nv_fp8_e4m3* __restrict__ value_cache,
    const __half* __restrict__ cos_table,
    const __half* __restrict__ sin_table,
    const int* __restrict__ positions,
    const int* __restrict__ slot_mapping,
    const float* __restrict__ q_scale_ptr,
    const float* __restrict__ kv_scale_ptr,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int half_rotary = rotary_dim / 2;
    const int half_head   = head_dim / 2;
    const int tid         = threadIdx.x;
    if (tid >= half_head) return;

    const float q_scale_inv = 1.0f / (*q_scale_ptr);
    const float kv_scale_inv = 1.0f / (*kv_scale_ptr);

    const int pos = positions[token_idx];

    // Q head processing
    if (head_idx < num_heads) {
        int q_base = (token_idx * num_heads + head_idx) * head_dim;

        if (tid < half_rotary) {
            // Rotary dims: apply RoPE
            float cos_val = __half2float(cos_table[pos * half_rotary + tid]);
            float sin_val = __half2float(sin_table[pos * half_rotary + tid]);
            float q0 = __half2float(q_in[q_base + 2 * tid]);
            float q1 = __half2float(q_in[q_base + 2 * tid + 1]);
            float q0r = q0 * cos_val - q1 * sin_val;
            float q1r = q0 * sin_val + q1 * cos_val;
            q_fp8_out[q_base + 2 * tid]     = __nv_fp8_e4m3(q0r * q_scale_inv);
            q_fp8_out[q_base + 2 * tid + 1] = __nv_fp8_e4m3(q1r * q_scale_inv);
        } else {
            // Pass-through dims: just quantize
            float q0 = __half2float(q_in[q_base + 2 * tid]);
            float q1 = __half2float(q_in[q_base + 2 * tid + 1]);
            q_fp8_out[q_base + 2 * tid]     = __nv_fp8_e4m3(q0 * q_scale_inv);
            q_fp8_out[q_base + 2 * tid + 1] = __nv_fp8_e4m3(q1 * q_scale_inv);
        }
    }

    // K head: partial RoPE + FP8 cache. V head: FP8 cache only.
    if (head_idx < num_kv_heads) {
        int k_base = (token_idx * num_kv_heads + head_idx) * head_dim;
        int slot = slot_mapping[token_idx];

        if (slot >= 0) {
            int cache_offset = (slot * num_kv_heads + head_idx) * head_dim;

            if (tid < half_rotary) {
                float cos_val = __half2float(cos_table[pos * half_rotary + tid]);
                float sin_val = __half2float(sin_table[pos * half_rotary + tid]);
                float k0 = __half2float(k_in[k_base + 2 * tid]);
                float k1 = __half2float(k_in[k_base + 2 * tid + 1]);
                float k0r = k0 * cos_val - k1 * sin_val;
                float k1r = k0 * sin_val + k1 * cos_val;
                key_cache[cache_offset + 2 * tid]     = __nv_fp8_e4m3(k0r * kv_scale_inv);
                key_cache[cache_offset + 2 * tid + 1] = __nv_fp8_e4m3(k1r * kv_scale_inv);
            } else {
                float k0 = __half2float(k_in[k_base + 2 * tid]);
                float k1 = __half2float(k_in[k_base + 2 * tid + 1]);
                key_cache[cache_offset + 2 * tid]     = __nv_fp8_e4m3(k0 * kv_scale_inv);
                key_cache[cache_offset + 2 * tid + 1] = __nv_fp8_e4m3(k1 * kv_scale_inv);
            }

            // V: no rotation, just quantize to cache
            int v_base = (token_idx * num_kv_heads + head_idx) * head_dim;
            float v0 = __half2float(v_in[v_base + 2 * tid]);
            float v1 = __half2float(v_in[v_base + 2 * tid + 1]);
            value_cache[cache_offset + 2 * tid]     = __nv_fp8_e4m3(v0 * kv_scale_inv);
            value_cache[cache_offset + 2 * tid + 1] = __nv_fp8_e4m3(v1 * kv_scale_inv);
        }
    }
}
