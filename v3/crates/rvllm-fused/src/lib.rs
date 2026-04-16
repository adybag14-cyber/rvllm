// rvllm-fused — scaffold only. See v3/specs/12-fused.md.
//   pub mod embedding;        // embedding_gather
//   pub mod add_norm_quant;   // fused_add_rmsnorm_fp8_quant
//   pub mod norm_quant;       // fused_rmsnorm_fp8_quant (no residual variant)
//   pub mod quant;            // quantize_fp8_per_token (post-attention)
//   pub mod rope_kv;          // fused_rope_kv_write (writes K/V into cache)
//   pub mod silu_mul;         // fused_silu_mul_fp8_quant
//   pub mod argmax;           // argmax_kernel (f32 logits -> i32 token)
//   pub mod residual;         // residual_add_f16 (NOT fused with GEMM)
//   pub mod reference;        // pure-Rust f32 reference impls for tests
