# Gemma 4 31B FP8 — Implementation Status

## Model
- Source: RedHatAI/gemma-4-31B-it-FP8-Dynamic
- On H100: /workspace/models/gemma4-31b-fp8 (32GB)
- 60 layers (50 sliding + 10 full), hidden=5376, vocab=262144

## Architecture Summary
| Field | Sliding (50 layers) | Full/Global (10 layers) |
|-------|--------------------|-----------------------|
| num_heads | 32 | 16 |
| num_kv_heads | 16 | 4 |
| head_dim | 256 | 512 |
| q_dim | 8192 | 8192 (uniform) |
| kv_dim | 4096 | 2048 (but weight is 4096) |
| rotary_dim | 256 (full) | 128 (partial, 25%) |
| rope_theta | 10000 | 1000000 |
| window | 1024 tokens | full context |
| activation | gelu_pytorch_tanh | same |

## Blocker Status (10-agent review)

| # | Blocker | Status | What's needed |
|---|---------|--------|--------------|
| 1 | head_dim 256/512 | Scaffolded | Remove head_dim==128 gate (5 files). Recompile FA3 for 256. SM89 fallback for 512. |
| 2 | GELU activation | Done | Kernel rewritten with vectorized loads. MlpActivation enum + runtime dispatch in layer_exec.rs. GELU PTX optional. |
| 3 | QK-norm | Done | Kernel, launcher, loader, layer_exec all implemented in gemma4_* files. |
| 4 | Dual RoPE | Done | Two tables precomputed. Partial rotation kernel exists. Just wire per-layer cos/sin selection. |
| 5 | Sliding window | Designed | FA3 has native window_size_left support. Thread one int through FFI. |
| 6 | Tied embeddings | Needs work | BF16 embed_tokens must be re-quantized to FP8 for lm_head. Need minimal BF16->FP8 path for this one tensor. |
| 7 | Extra norms + layer_scalar | Done | 4 norms + scalar all implemented. residual_scale_f16.cu written. Module exports wired. |
| 8 | Dual KV heads | Done | Uniform GEMM shapes (q_dim=8192, kv_dim=4096 always). Per-layer dims via Gemma4LayerDims. |
| 9 | Logit softcapping | Done | Kernel, launcher, reference, config parsing all exist. Insert one call. |
| 10 | Config parsing | Scaffolded | ModelArch extended with Gemma 4 fields. Parse rope_parameters nested dict. |

## What exists in gemma4_* files
- `v3/crates/rvllm-loader/src/gemma4_arch.rs` — Gemma4Arch with per-layer accessors
- `v3/crates/rvllm-loader/src/gemma4_weights.rs` — Gemma4LayerWeights (4 norms, QK-norm, layer_scalar)
- `v3/crates/rvllm-loader/src/gemma4_load.rs` — FP8 loader with dual RoPE, tied embeddings
- `v3/crates/rvllm-runtime/src/gemma4_bring_up.rs` — Gemma4Bringup engine constructor
- `v3/crates/rvllm-runtime/src/gemma4_layer_exec.rs` — 15-launch forward pass
- `v3/crates/rvllm-fused/src/gemma4_launcher.rs` — GELU, QK-norm, partial RoPE, softcap, residual_scale launchers
- `v3/crates/rvllm-fused/src/gemma4_reference.rs` — CPU reference implementations
- `v3/kernels/fused_gelu_mul_fp8_quant.cu` — GELU(tanh) + mul + FP8 quant
- `v3/kernels/fused_qk_rmsnorm.cu` — per-head QK RMSNorm
- `v3/kernels/fused_rope_partial_fp8kv.cu` — partial rotation RoPE + FP8 KV cache
- `v3/kernels/logit_softcap.cu` — element-wise tanh softcap
- `v3/kernels/residual_scale_f16.cu` — per-layer scalar multiply

## Remaining work to get Gemma 4 running

### Phase 1: Unblock loading (1-2 hours)
1. Remove head_dim==128 gate in load.rs (line 110)
2. Fix config parser: parse rope_parameters nested dict, sliding_window, layer_types with "sliding_attention"
3. Handle tied embeddings for FP8 lm_head (minimal BF16->FP8 for one tensor)
4. Detect Gemma4 arch in Bringup::load and delegate to Gemma4Bringup

### Phase 2: Compile kernels (30 min)
1. Compile all new .cu files to PTX for sm_90
2. Add to kernel manifest
3. Recompile FA3 .so with head_dim=256 template instantiation

### Phase 3: Wire and test (2-4 hours)
1. Thread window_size_left through FA3 FFI (one int param)
2. Wire per-layer cos/sin selection in gemma4_layer_exec
3. Run rvllm-bench with Gemma4 at batch=1 (smoke test)
4. Run rvllm-ppl with Gemma4 (correctness)
5. Full batch sweep

### Phase 4: FA3 head_dim=512 (1-2 days)
The 10 global layers need head_dim=512 attention. Options:
A. Recompile FA3 with head_dim=512 (may exceed SM90 shared memory)
B. Use SM89-style register-only kernel for global layers
C. Use cuDNN flash attention for global layers
This is the hardest remaining item.

## Memory budget (H100 80GB)
| Component | Size |
|-----------|------|
| Weights (FP8) | ~31 GB |
| Embedding (BF16) | 2.6 GB |
| KV cache (FP8, 8k ctx) | ~4 GB |
| RoPE tables (4x, 262k pos) | ~192 MB |
| Scratch + workspace | ~4 GB |
| **Total** | **~42 GB** |

Fits on H100 80GB with ~38 GB headroom.
