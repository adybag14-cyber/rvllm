# AutoKernel Fit for rvLLM

AutoKernel is useful for `rvllm` as a workflow, not as a drop-in runtime. Its value is the loop:

1. profile the real workload
2. isolate the hottest kernel family
3. optimize one thing at a time
4. keep or revert based on fixed benchmarks
5. verify end-to-end before moving on

That fits `rvllm` well for FA3, FFN, and GEMM/autotune work. The part that does **not** fit cleanly is AutoKernel's PyTorch-first extraction stack. `rvllm` already has standalone CUDA kernels, harnesses, and runner-level profiling hooks, so we should use those directly.

## Supported Kernel Mapping

| AutoKernel kernel | rvLLM equivalent | Primary files | How to measure it here |
|---|---|---|---|
| `matmul` | batched QKV, O-proj, down-proj GEMMs | `crates/rvllm-model-runner/src/gpu_layer/batched.rs`, `crates/rvllm-model-runner/src/gpu_runner.rs`, `crates/rvllm-gpu/src/cublas_autotune.rs`, `kernels/cutlass_gemm.cu`, `kernels/cutlass_qkv_bias.cu`, `kernels/cutlass_oproj_residual.cu` | `rvllm benchmark` at `N=64,128` plus `RVLLM_BATCHED_GEMM_STRATEGY` sweeps and `RVLLM_PROFILE=1` |
| `softmax` | decode attention online softmax inside FA3 / split-KV | `kernels/flash_attention_3.cu`, `kernels/flash_attention_3_v3.cu`, `kernels/split_kv_attention.cu`, `kernels/paged_attention.cu` | `RVLLM_PROFILE=1` and `nsys` kernel summaries; there is no separate standalone hot softmax path worth optimizing in isolation first |
| `layernorm` | fused residual + RMSNorm path | `kernels/fused_residual_rmsnorm.cu`, `kernels/fused_residual_rmsnorm_f16.cu` | `RVLLM_PROFILE=1` at `N=1,64,128`; if this rises above low-single-digit time share, isolate it |
| `rmsnorm` | RMSNorm and fused residual RMSNorm | `kernels/rms_norm.cu`, `kernels/rms_norm_f16.cu`, `kernels/fused_residual_rmsnorm*.cu` | same as above; this is the actual normalization target for Qwen/LLaMA-style decode |
| `flash_attention` | FA3 decode / prefill attention and split-KV attention | `kernels/flash_attention_3.cu`, `kernels/flash_attention_3_v3.cu`, `kernels/flash_attention_3_prefill.cu`, `kernels/split_kv_attention.cu`, `kernels/test_persistent_v2.cu` | real `rvllm benchmark`, `nsys`, and the local harnesses in `kernels/` |
| `fused_mlp` | GateUp + SiLU + down-proj fused work | `kernels/cutlass_gateup_silu.cu`, `kernels/fused_add_norm_gateup_gemv.cu`, `kernels/fused_oproj_add_norm_gateup_gemv.cu`, `kernels/fused_silu_down*.cu`, `kernels/persistent_layer_v3.cu` | `RVLLM_PROFILE=1` and direct harnesses; this is one of the best fits for the AutoKernel loop |
| `cross_entropy` | sampling / lm-head, not current decode bottleneck | `kernels/fused_lm_head_argmax.cu`, `kernels/fused_lm_head_argmax_f16.cu`, `kernels/argmax*.cu` | only prioritize if benchmark/profiles show it materially growing at high batch |
| `rotary_embedding` | RoPE and fused RoPE + cache write | `kernels/rotary_embedding.cu`, `kernels/rotary_embedding_f16.cu`, `kernels/fused_rope_cache.cu` | `RVLLM_PROFILE=1` and `nsys`; usually smaller than FA3/MLP/GEMM but still directly mappable |
| `reduce` | warp/block reductions inside RMSNorm, softmax, and sampling | `kernels/rms_norm*.cu`, `kernels/softmax*.cu`, `kernels/fused_lm_head_argmax*.cu` | treat as a sub-problem inside the kernel above it, not as a first-class optimization target |

## What This Means for rvLLM

AutoKernel's strongest fit here is:

- `flash_attention` -> FA3 decode/prefill and split-KV attention
- `fused_mlp` -> CUTLASS gateup + SiLU and fused FFN kernels
- `matmul` -> batched GEMM policy plus cublasLt autotune quality
- `rmsnorm` -> fused residual RMSNorm kernels
- `rotary_embedding` -> fused RoPE + cache write

The weaker fit is:

- `softmax` as a standalone target, because our real softmax hot path is already embedded in attention kernels
- `cross_entropy`, because decode throughput is currently dominated elsewhere
- `reduce` as a standalone target, because the useful work is inside FA3/RMSNorm/sampling rather than a separate reduction kernel

## Recommended Priority Order

For current `rvllm`, the AutoKernel-style order should be:

1. `flash_attention`
2. `fused_mlp`
3. `matmul`
4. `rmsnorm`
5. `rotary_embedding`

That order matches the actual end-to-end decode bottlenecks much better than trying to optimize isolated toy kernels.

## In-Repo Workflow

Use:

```bash
scripts/autokernel_loop.sh --model /root/models/Qwen2.5-7B
```

That script does the `rvllm`-native version of the AutoKernel loop:

- release build
- fixed benchmark at `N=1,64,128`
- optional `nsys` capture if installed
- per-run artifact directory under `results/autokernel/`

It is intentionally narrow. The point is to make FA3, FFN, and GEMM/autotune iteration cheap and repeatable on the real workload, not to recreate AutoKernel's whole PyTorch extraction stack.
