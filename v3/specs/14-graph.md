# 14 — Graph runtime

## Scope
CUDA graph capture/replay: precaptured `(bucket, max_blocks)` pool, a `CaptureScope` that is the only entry to the capture region, fingerprint + numeric validation harness.

## v2 problems
- `runner.rs:~1090` lazy-captures during warmup against the **first** request's metadata layout; tied to that request's `is_all_decode`/`padded_batch`/block-table layout. Later steps with different layout read garbage.
- `engine.rs:288 step_pipelined` and `engine.rs:156 step()` feed the same captured graph but upload metadata via different paths (`upload_metadata_padded` vs `forward_greedy_launch`'s non-padded). Today's `CUDA_ERROR_ILLEGAL_ADDRESS` in `fa3_v3_decode_gqa_kernel` is 100% this: graph captured at padded offsets; prefill ran non-padded; `last_meta_offsets` overwritten; `patch_metadata_decode` then wrote at non-padded offsets while the graph read padded offsets.
- `runner.rs:1349` `read_graph_output` calls `stream.synchronize()` inside replay — kills overlap, hides errors as silent stalls.
- `runner.rs:~1432` graph instance lifetime is interleaved with realloc-capable allocators — captured pointers can be freed while the graph is live.
- No fingerprint of captured nodes; no way to tell which kernel/arg moved after a regression.

## v3 contract

```rust
pub struct GraphPool<'r> { /* one instance per (bucket, max_blocks) */ }
pub struct GraphHandle { bucket: u32, max_blocks: u32 }

impl<'r> GraphPool<'r> {
    /// Called ONCE at engine init, after Metadata::for_bucket is frozen.
    pub fn capture_all(
        rt: &'r Runtime,
        buckets: &[(u32, u32)],
        body: impl Fn(&mut CaptureScope, &Metadata, &LayerScratch, &mut KvSlab)
                -> Result<&Tensor<i32>, RvllmError>,
    ) -> Result<Self, RvllmError>;

    pub fn replay<'s>(&'s self, h: GraphHandle, scope: &mut CaptureScope<'s>)
        -> Result<&'s Tensor<i32>, RvllmError>;

    pub fn destroy(self); // shape change / model reload only
}

impl<'g> CaptureScope<'g> {
    /// The only handle that can launch into the capture region.
    pub fn record<F, R>(self, f: F) -> Result<(R, CapturedGraph), RvllmError>
        where F: FnOnce(&mut Self) -> Result<R, RvllmError>;
}
```

Rules:
- **One graph per bucket, captured at engine init.** No lazy capture, no warmup-driven capture. `capture_all` enumerates every `(bucket, max_blocks)` from `RuntimeConfig`.
- **Capture region = ONE forward pass:** embedding → 28 layers → final norm → LM head → argmax. Excludes metadata HtoD (pre) and token DtoH (post); agent 13 owns those.
- **Capture-pointer-stable metadata.** `Metadata::for_bucket(rt, b)` (agent 08) is computed before `capture_all` and never moves. Captured nodes embed those exact device offsets; every step uploads new bytes into those EXACT offsets via `upload_for_bucket`. There is no "non-padded" upload variant — that variant **does not exist** in v3, making today's bug unrepresentable.
- **Inside `record(|scope| …)`** only `&GraphSafe` borrows are in scope (agent 04). `&mut HbmArena` is not in scope: realloc cannot occur. `&mut Tensor` aliasing across launches is rejected by the compiler (agent 05).
- **`replay` returns `&Tensor<i32>`** — the captured argmax output; caller (`step_collect`) issues DtoH from there.

## Validation harness
1. **Fingerprint walk** post-capture: `cudaGraphGetNodes` + `cudaGraphKernelNodeGetParams`; record `{idx, kernel_name, grid, block, shmem, arg_ptrs[], arg_scalars[]}`. Persist `v3/.fingerprints/<model_sha>/<bucket>.json`.
2. **Replay assert** (debug builds): re-walk; mismatch → `RvllmError::GraphFingerprintDrift{bucket, node_idx, field}`.
3. **Numeric reference** (CI only, `cfg(feature="ref_check")`): post-replay eager forward on same input; per-tensor cosine ≥ 0.999 (agent 15). Off in production.

## Failure modes
- Bucket > max captured → `Err(RvllmError::BucketNotCaptured{bucket})`. **No fallback to non-graph.**
- Capture failure (realloc inside scope, etc.) → `RvllmError::CaptureFailed{bucket, cause}`; engine init aborts.
- Fingerprint mismatch on replay (debug) → typed `Err`; release builds skip the walk.
- Model reload / shape change → `destroy` then `capture_all`. **No partial invalidation, no `setNodeParams` hot-patching.**
- Replay before `capture_all` returns → compile error (`GraphPool` only constructable post-capture).
- Wrong stream → `debug_assert!`.

## Test plan
- Integration: capture all configured buckets; replay 256 steps/bucket with random `BatchPlan`s; compute-sanitizer memcheck clean. Direct regression for today's IMA.
- Fingerprint stability: capture twice from clean engine; JSONs byte-identical.
- Drift detection: hand-mutate one arg pointer between capture and replay (test hook); expect `GraphFingerprintDrift`.
- Numeric: `ref_check` on, N=1 vs eager; cosine ≥ 0.999 per layer for 32 tokens.
- Negative: bucket=256 when max=128 → `BucketNotCaptured`, **never silent fallback**.
- Bench: replay overhead < 50 µs/step at N=128.

## Cross-cutting deps
- 04-memory: `GraphSafe`, `Tensor`, `CaptureScope` lifetime; no realloc inside.
- 05-concurrency: single compute stream; `CaptureScope` borrow rules; `Worker::fence` only outside steady state.
- 08-metadata: `Metadata::for_bucket` is the sole capture-pointer-stable layout; `upload_for_bucket` the sole HtoD path.
- 09-layer: `forward` is the captured body — 12 launches × 28 layers per replay.
- 13-sampling: argmax is the final node inside capture; DtoH happens outside.
- 15-validation: owns the eager reference / cosine harness used by `ref_check`.
