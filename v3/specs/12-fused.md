# 12 — fused kernels

## Scope
Single-composite fused kernels: norm+quant, RoPE+KV-write, silu+quant, lm-head argmax, embed gather, residual add. Pointer-only signatures, pre-allocated workspaces, alignment guards, Rust-side validation. **No megakernels.**

## v2 problems
- `kernels/quantize_activation_fp8.cu` — today (Apr 16) a vectorized `quantize_fp8_per_token` rewrite crashed `CUDA_ERROR_ILLEGAL_ADDRESS`; no correctness harness existed, failure surfaced inside a captured graph.
- `kernels/fused_oproj_add_norm_gateup_gemv.cu` — **abandoned megakernel** fusing 4 ops (O-proj GEMV + residual + RMSNorm + gate_up GEMV). Required `hidden·4B` smem, mixed shared-mem layouts, redundant L2 reads. Undebuggable once it desynced; **v3 forbids this fusion shape.**
- `kernels/fused_rmsnorm_fp8_quant.cu:95-98` — three full HBM passes, no register caching; `fused_silu_fp8_quant.cu` *does* register-cache. Two siblings, two strategies, no shared spec.
- 6 fused `.cu` files; per-token-scale layout (`f32[T]`) implicit, redefined per-kernel.
- `kernels/fused_rope_cache.cu:31` — predicated early-return inside captured graph; v3 sizes `block.x = half_dim` exactly.

## v3 contract
All kernels: `extern "C"`, pointer-only args, no in-kernel allocation. Workspace via `<kernel>_workspace_size(meta) -> usize`. Per-token scale: `f32[T]` contiguous.

| # | Name | Outputs | Fused ops | Grid | Block | Smem | Reg |
|---|---|---|---|---|---|---|---|
| 1 | `embedding_gather_f16` | `hidden:f16[T,H]` | gather | `(T)` | `min(H,256)` | 0 | ≤32 |
| 2 | `fused_add_rmsnorm_fp8_quant` | `new_resid`,`fp8_act`,`scale[T]` | add+norm+quant | `(T)` | `min(H,1024)` | `WMAX·4B` | ≤64 |
| 3 | `fused_rmsnorm_fp8_quant` | `fp8_act`,`scale[T]` | norm+quant | `(T)` | `min(H,1024)` | `WMAX·4B` | ≤48 |
| 4 | `quantize_fp8_per_token` | `fp8`,`scale[T]` | absmax+quant | `(T)` | `min(D,1024)` | `WMAX·4B` | ≤32 |
| 5 | `fused_rope_kv_write` | `q` in-place; K,V→slots | rope-q+rope-k+kv-write | `(T,max(Hq,Hkv))` | `Dh/2` | 0 | ≤48 |
| 6 | `fused_silu_mul_fp8_quant` | `fp8:e4m3[T,I]`,`scale[T]` | silu·mul+quant | `(T)` | `min(I,1024)` | `WMAX·4B` | ≤80 (reg-cached) |
| 7 | `argmax_f32` | `tokens:i32[T]` | tile-argmax+reduce | `(⌈V/256⌉,T)`→`(1,T)` | `256` | 0 | ≤32 |
| 8 | `residual_add_f16` | `x += y` (in-place) | add | `(⌈T·H/1024⌉)` | `1024` | 0 | ≤16 |

`WMAX=32`. Inputs match agent 09's pipeline. Kernel 8 stays unfused: v2's fused-residual GEMM (`cutlass_fp8_gemm_residual_v0`) crashed inside captured graphs (SPEC.md line 9), and the 4-op megakernel was abandoned. Agent 11 owns any future GEMM-side fusion.

### Vectorization rules
- Rust binding: `debug_assert!(dim % 8 == 0)`. Refuse loads that violate.
- `uint4` loads (8×f16), `uint2` stores (8×fp8) **only** when aligned. Misaligned `uint4` causes `MISALIGNED_ADDRESS` inside captured graphs (silent eagerly).
- `block.x · VEC ≤ dim`, must divide evenly. **No tail loops.** No predicated early-return inside captured kernels.

### Rust binding contract
```rust
pub fn fused_add_rmsnorm_fp8_quant(
    stream: &Stream,
    hidden: &Tensor<f16>, residual: &Tensor<f16>, gamma: &Tensor<f16>, eps: f32,
    new_residual: &mut Tensor<f16>,
    fp8_act:      &mut Tensor<fp8e4m3>,
    scale:        &mut Tensor<f32>,
) -> Result<(), RvllmError>;
```
Binding validates shape, dtype, alignment, non-aliasing of `&mut` args before launch. Kernel signature is pointer-only.

### Numeric contract
Every kernel ships a pure-Rust f32 reference at `rvllm-kernels/tests/ref_<name>.rs`. CUDA output must match: cosine ≥ `1 − 1e-3` (f16 outs), abs ≤ `1e-5` (f32 outs).

## Failure modes
- **Panic (debug_assert)**: shape mismatch, `dim%8≠0`, `scale.len()≠T`, KV slot OOB.
- **Err**: launch / async error → `RvllmError::FusedKernel { name, code }`.
- **Compile-error**: aliased `&mut Tensor`, dtype mismatch.

## Test plan
- Per-kernel: 100 random shapes, cosine ≥ 0.999 / abs ≤ 1e-5 vs reference.
- `<kernel>_workspace_size` cross-checked by `compute-sanitizer --tool memcheck`.
- Lint denies kernel filenames naming ≥4 composites (megakernel guard).
- Each kernel launched 1000× inside captured graph; sanitizer clean.
- Vectorization regression: today's IMA reproduces with alignment guard removed → CI keeps it.
- E2E: 8-kernel pipeline matches HF Qwen2.5-7B per-layer cosine ≥ 0.999 (agent 15).

## Cross-cutting deps
- 04-memory: pre-allocs `Tensor`s, scale buffers, KV slab.
- 09-layer: sequences kernels 1–6, 8 inside `LayerForward::forward`.
- 11-cutlass-fp8: GEMM-side fusion; residual epilogue stays default-off.
- 13-sampling: consumer of kernel 7 (`argmax_f32`).
- 15-validation: hosts cosine/abs-err harness and golden traces.
