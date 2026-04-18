# Gemma 4 31B Support -- rvLLM v3

Target: google/gemma-4-31B-it (Apache 2.0, ungated)
Target GPU: H200 141GB (also fits H100 80GB at FP8)

## Real config.json (google/gemma-4-31B-it)

Architecture: `Gemma4ForConditionalGeneration` (multimodal, text decoder only)

| Field | Value |
|---|---|
| hidden_size | 5376 |
| num_attention_heads | 32 |
| num_hidden_layers | 60 |
| intermediate_size | 21504 |
| vocab_size | 262144 |
| head_dim (sliding) | 256 |
| global_head_dim | 512 |
| num_key_value_heads (sliding) | 16 |
| num_global_key_value_heads | 4 |
| max_position_embeddings | 262144 (256k) |
| sliding_window | 1024 |
| hidden_activation | gelu_pytorch_tanh |
| final_logit_softcapping | 30.0 |
| tie_word_embeddings | true |
| attention_k_eq_v | true (global layers) |

## Critical finding: weight shapes are UNIFORM across layers

Both sliding (layer 0) and global (layer 5) have identical tensor shapes:
```
q_proj:    [8192, 5376]   = 32*256 = 16*512
k_proj:    [4096, 5376]   = 16*256 = 8*512
v_proj:    [4096, 5376]   = 16*256 = 8*512
o_proj:    [5376, 8192]
q_norm:    [256]
k_norm:    [256]
layer_scalar: [1]          (per-layer residual multiplier)
```

The dual head_dim (256 vs 512) is a RUNTIME RESHAPE, not a weight
difference. Global layers reinterpret the same weight matrices as
fewer, wider heads: 32x256 -> 16x512 for Q, 16x256 -> 8x512 for KV.

## Architecture delta vs Llama/Qwen baseline

| Feature | Llama/Qwen (current) | Gemma 4 31B |
|---|---|---|
| Activation | SiLU | GELU(tanh) |
| head_dim | 128 (derived) | 256 sliding / 512 global (runtime reshape) |
| Norms/layer | 2 (input, post_attn) | 4 + layer_scalar |
| QK-Norm | none | RMSNorm on Q,K before RoPE |
| RoPE | single theta | dual: theta=10k (sliding), theta=1M (global) |
| Partial RoPE | full | full (sliding), 0.25 (global: 128 of 512 rotated) |
| Attention | all full | 5 sliding + 1 global, repeating |
| KV heads | uniform | 16 sliding, 8 global (runtime reshape) |
| LM head | separate weight | tied to embed_tokens |
| Weight prefix | model.layers.N | model.language_model.layers.N |
| Logit post | none | softcap: 30*tanh(x/30) |
| attention_k_eq_v | n/a | true for global layers |

## File layout (new files, no edits to existing)

```
v3/
  GEMMA4_SPEC.md                          <-- this file
  crates/
    rvllm-loader/src/
      gemma4_arch.rs                      <-- Gemma4 ModelArch parser
      gemma4_load.rs                      <-- weight loader (prefix mapping, tied embed, 4 norms)
      gemma4_weights.rs                   <-- Gemma4LayerWeights, Gemma4LoadedModel
    rvllm-runtime/src/
      gemma4_layer_exec.rs                <-- 16-launch forward (GELU, QK-norm, dual RoPE, 4 norms)
      gemma4_bring_up.rs                  <-- Gemma4Bringup (scratch, plans, dual FA3)
    rvllm-fused/src/
      gemma4_launcher.rs                  <-- FusedGeluMulFp8Quant, QkRmsnorm, LogitSoftcap launchers
  kernels/
    fused_gelu_mul_fp8_quant.cu           <-- GELU(tanh) * up + FP8 quant
    fused_qk_rmsnorm.cu                   <-- per-head RMSNorm on Q and K
    fused_rope_partial_fp8kv.cu           <-- partial RoPE (only rotate first rotary_dim dims)
    logit_softcap.cu                      <-- 30*tanh(logits/30) in-place on f16
```

## Integration points (minimal edits to existing code)

1. `rvllm-loader/src/load.rs` ModelArch::from_dir():
   - detect `"Gemma3ForCausalLM"` or `"Gemma4ForCausalLM"` in architectures
   - delegate to `gemma4_arch::Gemma4Arch::from_dir()`

2. `rvllm-runtime/src/bring_up.rs` Bringup::load():
   - if arch is Gemma4, construct Gemma4Bringup instead

3. `rvllm-fused/src/lib.rs`:
   - re-export gemma4_launcher types

4. `rvllm-attention/src/lib.rs` Fa3Kernels::load():
   - accept head_dim 256 when loading a Gemma4 FA3 .so
   - need FA3 .so compiled for head_dim=256

## FA3 build

FA3 already has head_dim=256 template instantiations in flash_api.cpp.
No special flags needed -- just build flash-attention normally.

Build command (on H200 with CUDA 12.4):
```
cd /workspace/flash-attention
export FLASH_ATTN_CUDA_ARCHS="90"   # skip compute_120 (needs CUDA 12.8+)
MAX_JOBS=8 python3 setup.py build_ext --inplace
```

The .so includes head_dim 64, 128, 192, 256, and FP8 variants.

## Kernel artifact storage

Compiled kernels stored at: `and-y/rvllm-kernels` (private HF repo)
```
ptx/sm90a/           -- PTX kernels for Hopper (H100/H200)
src/                  -- CUDA source for provenance
fa3/sm90/             -- FA3 .so (after build + verification)
policy/               -- CUTLASS autotune policies per GPU
```

## Kernel launch sequence (Gemma4, 16 launches/layer)

```
 1. fused_rmsnorm_fp8_quant           (input_layernorm)
 2. fp8_gemm                          (fused Q||K||V projection)
 3. fused_qk_rmsnorm                  (RMSNorm on Q heads, RMSNorm on K heads)
 4. fused_rope_partial_fp8kv          (partial RoPE + FP8 Q + paged KV write)
 5. paged_decode / paged_prefill      (FA3, head_dim=256, sliding or full)
 6. quantize_fp8_per_token            (attn_out -> fp8)
 7. fp8_gemm_residual                 (O proj, += residual)
 8. fused_rmsnorm                     (post_attention_layernorm -- norm only, no quant)
 9. residual_scale_f16                (multiply residual by layer_scalar)
10. fused_rmsnorm_fp8_quant           (pre_feedforward_layernorm)
11. fp8_gemm                          (gate||up fused proj)
12. fused_gelu_mul_fp8_quant          (GELU(tanh)(gate) * up -> FP8)
13. fp8_gemm_residual                 (down proj, += residual)
14. fused_rmsnorm                     (post_feedforward_layernorm -- norm only, no quant)
15. residual_scale_f16                (multiply residual by layer_scalar)
16. (implicit: residual carries forward to next layer)
```

Post-model:
- final_norm (RMSNorm)
- lm_head GEMM (using embed_tokens as tied weight)
- logit_softcap (30 * tanh(logits / 30))
- argmax

## Memory budget (H200 144GB, FP8 weights)

| Component | Size |
|---|---|
| Weights (31B * 1 byte FP8) | ~31 GB |
| Embedding (262144 * 5376 * 2 bytes f16) | ~2.7 GB |
| KV cache (60 layers, uniform 4096-dim KV, 8k ctx, FP8) | ~4 GB |
| KV cache (60 layers, uniform 4096-dim KV, 128k ctx, FP8) | ~60 GB |
| Scratch + workspace | ~3 GB |
| **Total (8k ctx)** | **~41 GB** |
| **Total (128k ctx)** | **~97 GB** |

H200 (144 GB) handles 128k context. H100 80GB caps at ~16k context.

## Sliding window attention

Layers with sliding attention use a 1024-token window. Two approaches:

A) **FA3 native sliding window** -- FA3 Hopper supports `window_left` param.
   Pass window_size=1024 for sliding layers, window_size=-1 for global.
   This is the clean path but requires verifying/adding the param to the
   C API wrapper.

B) **Masking via context_lens** -- For sliding layers, clamp
   `context_lens[i] = min(real_context, 1024)` so FA3 only reads the
   last 1024 KV entries. Approximate but works with existing FA3 API.
   Loses tokens older than the window. Good enough for initial bringup.

Recommend: start with (B) for bringup, switch to (A) for production.

## Dual RoPE

Two precomputed cos/sin tables:
- `rope_sliding`: theta=10000, rotary_dim = head_dim (256)
- `rope_global`: theta=1000000, rotary_dim = head_dim * 0.25 = 64

The partial RoPE kernel only rotates the first `rotary_dim` elements
of each head, leaving the rest unchanged. For global layers that means
only 64 of 256 dims get rotation; the remaining 192 are passed through.

## Per-layer variation

Gemma 4 alternates layers: most are sliding, every 4th is global.
The layer type determines:
- Which RoPE table (sliding vs global cos/sin)
- Which attention mode (sliding window vs full)
- How many KV heads (16 for sliding, 4 for global)
- Rotary dim (full for sliding, partial for global)

This is encoded as `Gemma4LayerType` in the per-layer config.
