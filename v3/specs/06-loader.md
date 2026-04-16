# 06 — Model Loader

## Scope
Materialize a typed, immutable `LoadedModel` on GPU from HF safetensors at engine init, with optional FP8 quantization performed GPU-side from a single staging buffer per tensor.

## v2 problems
- `crates/rvllm-v2/src/runner.rs:37-67` — `ModelWeightsStore` holds **13 separate `Vec<CudaSlice<…>>`** indexed by raw layer index (`fused_qkv[i]`, `fp8_qkv[i]`, `fp8_qkv_scales[i]`, `gemv_qkv_scales[i]`, …). Type-erased, coupled, easy to skew when one vec is shorter than another.
- `crates/rvllm-v2/src/runner.rs:290-388` — `enable_fp8_weights()` does **DtoH → CPU `quantize_weight_fp8_per_tensor` → HtoD** per tensor, per layer. On a 7B model this is many seconds of `FP8 per-tensor quantization` log spam and a full PCIe round-trip per weight.
- `crates/rvllm-v2/src/runner.rs:395-411` — `shrink_weight_vecs` overwrites freed weights with 1-element stubs to keep the `Vec` indices valid. Hidden-state hack: `fused_qkv[i].len()` lies after FP8 enable.
- No clamp accounting. CPU quantizer silently saturates outliers; mis-scaled tensors load successfully and produce garbage logits.
- No deterministic placement. HBM addresses depend on prior allocation history; captured graphs are not reproducible across runs.

## v3 contract

```rust
pub struct LoadedModel {
    pub embed_tokens:      WeightFp16,        // [vocab, hidden]
    pub final_norm:        WeightFp16,        // [hidden]            (RMSNorm gamma)
    pub lm_head:           WeightFp8OrFp16,   // [vocab, hidden]
    pub rope_cos:          WeightFp32,        // [max_pos, head_dim/2]
    pub rope_sin:          WeightFp32,        // [max_pos, head_dim/2]
    pub layers:            Box<[LoadedLayer]>,// .len() == num_layers, fixed
    pub manifest:          ModelManifest,     // sha256 per tensor, dtype, shape
}

pub struct LoadedLayer {
    pub input_norm:        WeightFp16,        // [hidden]
    pub post_attn_norm:    WeightFp16,        // [hidden]
    pub qkv:               WeightFp8OrFp16,   // [(nh+2*nkv)*hd, hidden]   fused
    pub o_proj:            WeightFp8OrFp16,   // [hidden, nh*hd]
    pub gate_up:           WeightFp8OrFp16,   // [2*inter, hidden]         fused
    pub down_proj:         WeightFp8OrFp16,   // [hidden, inter]
}

pub enum WeightFp8OrFp16 {
    Fp16(WeightFp16),
    Fp8 { data: DevicePtr<u8>, scale: DevicePtr<f32>, shape: [usize;2], stride: [usize;2] },
}

// LayerWeights is a borrow, never owned. No clones, no Arc.
pub struct LayerWeights<'m> { pub layer: &'m LoadedLayer, pub idx: usize }
```

Loader pipeline (engine init only, never during decode):

```
HF safetensors mmap
    └── per-tensor: validate shape against ModelConfig (agent 02)
        └── HtoD into HBM staging buffer (FP16, owned by HbmAllocator agent 04)
            └── if quantize_fp8: launch GPU kernel quantize_e4m3_per_tensor
                  ├── pass 1: reduce abs-max → scale = abs_max / 448.0
                  ├── pass 2: write FP8 buffer + clamp_count atomic
                  └── free staging FP16 (allocator returns HBM)
            └── record final {ptr, scale, shape, stride, sha256} in manifest
```

FP8 policy (E4M3, max=448):
- One f32 scale per tensor.
- After kernel pass 2, DtoH the `clamp_count`. Compute `pct = clamp_count / numel`.
- `tracing::warn!(name, scale, clamp_count, pct, "fp8 clamp")` for every tensor.
- If `pct > 1e-5` (0.001%), return `RvllmError::Fp8MisScaled { tensor, pct }`. No fallback.

Deterministic placement: `HbmAllocator::for_model(config, gpu)` (agent 04) hands out arenas in a fixed order driven by a sorted weight load plan (alphabetical tensor name). Same `(ModelConfig, RuntimeConfig, GPU SKU)` triple → identical HBM offsets.

## Failure modes
- Missing tensor / shape mismatch → `Err(RvllmError::WeightShape{ name, expected, got })`.
- Dtype not supported (BF16 in, FP8 requested) → return `Err`, no silent cast.
- FP8 clamp >0.001% → `Err(RvllmError::Fp8MisScaled)`.
- Lazy/partial loads → **forbidden**. `Engine::new` returns only after every weight is resident and verified.
- `LoadedModel::Drop` releases its arena. Holding a `LayerWeights<'m>` past that lifetime is a compile error.

## Test plan
- Unit: synthetic FP16 tensor with known abs-max → assert scale, clamp_count exact.
- Unit: inject one outlier → assert `Fp8MisScaled` is raised.
- Integration: load Qwen2.5-7B twice in same process, assert byte-identical HBM offsets per tensor.
- Integration: cosine ≥ 0.9999 between FP16 reference and FP8 dequantized weights, all tensors.
- Integration: `Engine::new` time ≤ 3s for 7B FP8 (vs v2's many-second CPU round-trip).

## Cross-cutting deps
- agent 02 (config): `ModelConfig` provides shapes, `RuntimeConfig` provides quant policy.
- agent 03 (errors): `RvllmError::{WeightShape, Fp8MisScaled, SafetensorsIo}`.
- agent 04 (memory): `HbmAllocator::for_model` deterministic arena, staging-buffer free.
