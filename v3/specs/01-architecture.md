# 01 — Architecture

## Scope
Crate boundaries, acyclic DAG, per-crate module list, public-API surface preventing the cross-cutting state coupling that wrecked v2.

## v2 problems (file:line)
- **God-object runner.** `runner.rs:86-135` packs weights, scratch, pinned argmax, packed metadata, autotune, graph flags (`last_meta_offsets`, `last_padded_batch`). Two writers (`runner.rs:919` greedy, `runner.rs:1061` padded) race; prefill overwrites the layout the captured graph reads → today's `CUDA_ERROR_ILLEGAL_ADDRESS`.
- **Layer monolith.** `layer.rs` 2,197 lines: GEMM dispatch, attention meta, plan enums, scratch lifetimes, weight refs. New FP8 variant ships beside RoPE meta — no boundary forces compat → WS-mainloop / cooperative-epilogue crash.
- **Integration coupling.** `integration.rs:1-256` parses config, sizes KV, spawns tokio, inits CUDA, loads weights, builds Engine in one path. `max_workspace_size` (`runner.rs:191`) covers autotuned variants only; fallback `fp8_gemm`/`fp8_gemm_small` paths have no WS contract.
- **Hidden-state leak.** `worker.rs:282` comments "patch_metadata_decode skips" — worker reaches into runner internals because no opaque meta handle exists. `lib.rs:1-52` exposes 18 modules; nothing prevents `scheduler` importing `runner::PackedMetaOffsets`.

## v3 contract — crate list, modules, DAG

```
rvllm-core      (none)            error.rs ids.rs dtype.rs shape.rs
rvllm-config    core              runtime.rs model.rs validate.rs
rvllm-mem       core              hbm.rs pinned.rs kv_layout.rs alloc.rs
rvllm-stream    core,mem          stream.rs event.rs capture_token.rs
rvllm-kernels   core,mem,stream   loader.rs sigs.rs artifacts.rs
rvllm-cutlass   kernels           plan.rs autotune.rs workspace.rs variants.rs
rvllm-attention kernels           fa3_decode.rs fa3_prefill.rs gqa.rs
rvllm-fused     kernels           add_norm_quant.rs rope_kv.rs silu_mul.rs lm_argmax.rs
rvllm-metadata  core,mem,stream   pack.rs handle.rs layout.rs
rvllm-graph     stream,metadata   capture.rs replay.rs buckets.rs validate.rs
rvllm-loader    core,config,mem   safetensors.rs fp8_quant.rs placement.rs
rvllm-sampling  core,stream,fused greedy.rs topk.rs topp.rs dtoh_pinned.rs
rvllm-runtime   cutlass,attention,
                fused,metadata,
                graph,loader,
                sampling          layer_exec.rs scheduler.rs lifecycle.rs sched_state.rs
rvllm-serve     runtime           http.rs grpc.rs openai.rs
rvllm-bench     runtime           harness.rs gates.rs profile.rs
rvllm-deploy    (script)          tarball.rs deploy_and_bench.rs
rvllm-zig       core (FFI)        bpe.zig topk.zig metapack.zig
```

Cycles forbidden by `cargo deny`. `runtime` is the only crate naming cutlass+attention+fused+graph+loader+sampling together.

## Public API sketch

```rust
// rvllm-core
pub use error::{RvllmError, Result};
pub use ids::{RequestId, SeqId, BlockId};
pub use dtype::{DType, Shape};

// rvllm-mem
pub fn alloc_hbm(s:&Stream, bytes:usize)->Result<HbmBuf>;
pub struct KvLayout { pub block_size:u32, pub num_kv_heads:u32, pub head_dim:u32 }

// rvllm-cutlass — workspace is a value, not a query
pub struct Fp8GemmPlan;
pub fn plan_fp8_gemm(shape:&GemmShape, v:Fp8Variant)->Result<Fp8GemmPlan>;
pub fn workspace_bytes(p:&Fp8GemmPlan)->usize;
pub unsafe fn launch(p:&Fp8GemmPlan, a:&Fp8Args, ws:HbmView, s:&Stream)->Result<()>;

// rvllm-metadata — opaque, capture-safe
pub struct MetaPack;
impl MetaPack {
    pub fn upload(&mut self, l:MetaLayout, b:&BatchInput, s:&Stream)->Result<MetaHandle>;
    pub fn handle(&self)->MetaHandle;
}

// rvllm-runtime
pub fn execute_layer(w:&LayerWeights, kv:&mut KvBlocks, m:MetaHandle,
                     sc:&mut LayerScratch, s:&Stream)->Result<()>;
```

## Cross-crate example
`rvllm-runtime` calls `rvllm-cutlass` without naming any `rvllm-attention` type:

```rust
// rvllm-runtime/src/layer_exec.rs
use rvllm_cutlass::{plan_fp8_gemm, workspace_bytes, launch as fp8_launch, Fp8Variant};
use rvllm_mem::HbmView; use rvllm_stream::Stream;
let plan = plan_fp8_gemm(&shape, Fp8Variant::ResidualV0)?;
let ws   = scratch.workspace_view(workspace_bytes(&plan));
unsafe { fp8_launch(&plan, &args, ws, stream)?; }
```
No `Fa3*` symbol in scope. `rvllm-attention` consumes `rvllm-kernels` + `MetaHandle` independently.

## Failure modes
- Reverse-edge import → `cargo deny` CI fail.
- Crate >800 LoC or any file >500 LoC → lint fail.
- Two structs holding `&mut` to one buffer → compile error.
- Capture-region API receiving non-`CaptureSafe` arg → trait-bound fail.

## Test plan
- `tests/dag.rs` parses every `Cargo.toml`, asserts DAG matches.
- `tests/loc_budget.rs` enforces per-crate/per-file ceilings.
- Doctests prove every pub fn callable without naming a forbidden sibling type.
- `cargo +nightly udeps` rejects silent over-coupling.

## Cross-cutting deps
- 02-config → `rvllm-config`. 03-errors → `RvllmError` in `rvllm-core`. 04-memory → `KvLayout`/`HbmBuf`/`Pinned<T>`. 05-concurrency → `Stream`/`Event`/`CaptureToken`. 08-metadata → `MetaPack` opacity fixes `runner.rs:919/1061`. 09-layer → `execute_layer` sig. 14-graph → capture-region trait.
