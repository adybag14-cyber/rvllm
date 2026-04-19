# FP8 GEMM 27x Bug -- Investigation Spec

## Problem
Same FP8 bytes + scales through PyTorch `_scaled_mm` = correct.
Same FP8 bytes + scales through rvLLM's cuBLASLt `fp8_gemm_inner` = 27x too large.
The bug is 100% in how `fp8_gemm_inner` configures cuBLASLt.

## Key file
`v3/crates/rvllm-cutlass/src/cublaslt.rs` -- `fp8_gemm_inner` at line 268

## Agent Tasks

### Agent A: Trace the call chain
Read `v3/crates/rvllm-runtime/src/gemma4_layer_exec.rs` to find:
1. How fp8_gemm / fp8_gemm_f32 is called
2. What M, N, K values are passed
3. What a_scale and b_scale values are (device pointers to what?)
4. How activations get into FP8 format -- is there an online quantization step?
5. If activation is F16, how does it become FP8 for the FP8 GEMM?
Report the exact call sites and what flows into a_scale/b_scale.

### Agent B: Trace weight scale computation  
Read `v3/crates/rvllm-loader/src/gemma4_load.rs` starting at line 611 (`fuse_fp8_direct_channelscale`).
1. What scale value ends up as the per-tensor weight scale?
2. Is it the original model's per-tensor scale, or a recomputed one?
3. What is the numeric value approximately?
4. After fusing, do the FP8 bytes already incorporate the scale (making scale=1.0 correct), or do they need the scale applied by cuBLASLt?
Report the exact scale semantics.

### Agent C: PyTorch _scaled_mm cuBLASLt config
Search PyTorch source code (web) for how `_scaled_mm` configures cuBLASLt.
Focus on: `aten/src/ATen/native/cuda/Blas.cpp` or `ScaledMM.cpp`.
Key questions:
1. What matmul descriptor attributes does PyTorch set?
2. Does it set FAST_ACCUM?
3. How does it handle the TN trick (row-major to col-major)?
4. Does it set D_SCALE?
5. What are the layout dimensions and leading dims?
Report a line-by-line comparison checklist vs rvLLM's fp8_gemm_inner.
