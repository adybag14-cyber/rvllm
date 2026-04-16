# 09 — layer execution

## Scope
One transformer block as a **pure function** of `(input, weights, kv, scratch, meta)`. No `&self`, no Arc-field mutation, no inter-layer flags. Same inputs → same outputs.

## v2 problems
- `crates/rvllm-v2/src/layer.rs:803` — `forward_batched_v2` takes 12 args inc. 4 `&mut CudaSlice<f16>`. Cross-layer aliasing implicit; nothing forbids overlap with captured graph buffer.
- `layer.rs:845-862` — `residual_from_fused: bool` per-layer, threaded into N+1 as `prev_mlp_out: Option<_>`. Inter-layer global state. Bug commits `67170fd31`, `ef5b20d62`, `7c212c13c` all touched it.
- `layer.rs:824-831` — `RVLLM_GEMV_FP8` env bifurcates the path; capture vs replay kernel counts diverge.
- 2,197 lines, 8 fused/unfused permutations, no module boundary — Frankenstein dispatch (SPEC.md L11).
- `layer.rs:849-853` — fused norm+quant takes `&mut scratch.residual_tmp` aliased into N+1's input position; compiler accepts.

## v3 contract

```rust
pub struct LayerWeights<'w> {
    pub idx: u32,
    pub input_norm: Tensor<'w, f16>, pub post_norm: Tensor<'w, f16>,
    pub qkv: Fp8Weight<'w>, pub o_proj: Fp8Weight<'w>,
    pub gate_up: Fp8Weight<'w>, pub down_proj: Fp8Weight<'w>,
}
/// One per worker, allocated ONCE at engine init at worst-case shape.
pub struct LayerScratch<'a> {
    pub fp8_act: Tensor<'a, fp8e4m3>,  // [max_tok, max(hidden,intermediate)]
    pub fp8_scale: Tensor<'a, f32>,    // [max_tok]
    pub qkv_buf: Tensor<'a, f16>, pub attn_out: Tensor<'a, f16>,
    pub o_out: Tensor<'a, f16>, pub gate_up: Tensor<'a, f16>,
    pub mlp_out: Tensor<'a, f16>, pub fa3_ws: Tensor<'a, f16>,
}

pub fn forward<'a>(
    input:   &Tensor<'a, f16>,
    weights: &LayerWeights<'_>,
    kv:      &mut KvSlab<'_>,
    scratch: &mut LayerScratch<'_>,
    meta:    &Metadata<'_>,
    out:     &mut Tensor<'a, f16>,        // distinct, non-aliased
) -> Result<(), RvllmError>;
```

### Decode pipeline (12 launches)
1. `fused_add_rmsnorm_fp8_quant(input, residual=input, w.input_norm) → fp8_act + scale; new residual into out`
2. `cutlass_fp8_gemm(fp8_act, w.qkv) → qkv_buf:f16`
3. `fused_rope_kv_write(qkv_buf, meta.positions, meta.slot_mapping)` — Q in qkv_buf; K,V → kv
4. `fa3_paged_decode(qkv_buf.q, kv, meta.block_tables, meta.context_lens, fa3_ws) → attn_out:f16`
5. `quantize_fp8_per_token(attn_out) → fp8_act + scale`
6. `cutlass_fp8_gemm(fp8_act, w.o_proj) → o_out:f16`
7. `residual_add(out, o_out)` — separate; fused-residual epilogue v0 crashed today (SPEC.md L9; commit `ef5b20d62`)
8. `fused_rmsnorm_fp8_quant(out, w.post_norm) → fp8_act + scale`
9. `cutlass_fp8_gemm(fp8_act, w.gate_up) → gate_up:f16`
10. `fused_silu_mul_fp8_quant(gate_up) → fp8_act + scale`
11. `cutlass_fp8_gemm(fp8_act, w.down_proj) → mlp_out:f16`
12. `residual_add(out, mlp_out) → out`

12 launches/layer. vLLM hits 9-11 by fusing o-proj+residual and down+residual; we tried and crashed. Fused epilogues remain owned by agent 11, gated by config flag, off in v3 v1.

### Buffer rotation: eliminated
One residual buffer per slot. Engine swaps **bindings** `(input, out)` between layers — no `residual_a/b` ping-pong. `&mut Tensor<'a, f16>` makes aliasing in `CaptureScope` a compile error (agents 04/05). In-place updates allowed inside the capture region: borrow checker proves no other binding aliases.

### Self-contained: no `prev_fused`
Step 1 always reads residual from `input` (= prior layer's `out`). v2's "fused vs non-fused after residual" branch (`layer.rs:849-862`) is gone — one path, decided by which input arrived.

### Scratch lifetime
`LayerScratch` is `&mut`-borrowed for one `forward`. Worst-case dims pinned at engine init in `HbmArena` (agent 04). Layers 0..N-1 share one `LayerScratch` (sequential, agent 05). One per worker.

## Failure modes
- Panic (debug_assert): `out.shape != input.shape`; `kv` lacks layer `weights.idx`; scratch dims `< meta.num_tokens`.
- Err: GEMM/attention/fused errors as `RvllmError::{Cutlass,Attention,Cuda}` with `layer = weights.idx`.
- Compile-error: aliased `input == out`; `LayerScratch` reborrowed across concurrent layers; `LayerScratch` dropped before captured graph replays.

## Test plan
- Per-layer cosine vs HF Qwen2.5-7B ≥ 0.999 (agent 15).
- Property: snapshot kv, call `forward` twice, byte-exact (purity).
- `trybuild` compile-fail: aliasing `input == out`; reusing `LayerScratch`.
- `compute-sanitizer --tool racecheck` on 32-layer step: zero hazards.
- nsys-counted launches: 12 × num_layers per step; manifest gate fails on drift.

## Cross-cutting deps
- 04-memory: `Tensor`, `HbmArena`, `KvSlab`, `LayerScratch` alloc + aliasing rules.
- 05-concurrency: single-stream; capture-region `&mut` rules.
- 08-metadata: `Metadata` consumed read-only.
- 10-attention: `fa3_paged_decode` / `fa3_prefill` (step 4).
- 11-cutlass-fp8: `cutlass_fp8_gemm` variants (steps 2,6,9,11), workspace pre-bound on `Fp8Weight`.
- 12-fused-kernels: steps 1, 3, 5, 8, 10 — kernel sigs and per-token scale layout.
