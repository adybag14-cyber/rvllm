# 05 — Concurrency

## Scope
Single-stream model, one async pipeline (`step_launch`/`step_collect`), DtoH double-buffering, graph aliasing rules, Drop safety.

## v2 problems
- Two execution paths, different invariants. `engine.rs:156` `step()` (sync) vs `engine.rs:288` `step_pipelined()` (async). Workers diverge: `worker.rs:181` builds via `input_builder.build`; `worker.rs:258` `step_launch` builds via `build_decode_only` and uses `upload_metadata_padded`, while non-padded `forward_greedy_launch` clobbers `last_meta_offsets` — root cause of today's graph IMA.
- Hidden steady-state sync: `runner.rs:1358` `stream.synchronize()` inside `read_graph_output` kills CPU/GPU overlap on every replay.
- Two DtoH paths coexist: bare `cuMemcpyDtoHAsync_v2`+sync (`runner.rs:1349-1359`) and event-coordinated `launch_dtoh`/`wait_dtoh` (`runner.rs:1366-1406`).
- Drop ordering bug fixed in `7c212c13c` (`runner.rs:1435`): the rule "sync stream before destroying events" is implicit; nothing prevents regression.
- Defensive ad-hoc syncs leaked into APIs (`kv_cache.rs:91`, `worker.rs:632`).

## v3 contract

```rust
pub struct Worker { stream: ComputeStream }   // !Send !Sync
pub struct PendingStep<'w> { _w: &'w mut Worker, evt: EventHandle, buf: u8 }
pub struct CollectedTokens { pub ids: Vec<TokenId> }

impl Worker {
    pub fn step_launch(&mut self, batch: &StepBatch) -> Result<PendingStep<'_>>;
    pub fn step_collect(&mut self, p: PendingStep<'_>) -> Result<CollectedTokens>;
    pub fn fence(&mut self) -> Result<()>; // init / shutdown / test only
}
```

- **One stream per worker.** No multi-stream parallelism. **Decision: NO copy stream.** KV HtoD copies are rare (block evictions); a copy stream re-introduces the cross-phase event isolation issues noted at `integration.rs:471`. Add later only if profiling demands it.
- **One codepath.** Graph replay is hidden inside `step_launch`; no sync vs pipelined fork. Capture is a transparent runtime property.
- **DtoH double buffer.** Two pinned host buffers + two `CUevent`s owned by `Worker`. `step_launch` enqueues into buffer N+1; CPU reads buffer N. `PendingStep` carries the buffer index and event handle.
- **Type-state ordering.** `PendingStep<'w>` borrows `&mut Worker`, so a second `step_launch` cannot start until `step_collect` consumes it. Reading raw token bytes outside `step_collect` is unreachable.
- **No `synchronize()` in steady state.** `Worker::fence` is the only stream-sync API; clippy lint `rvllm::no_steady_state_fence` denies it outside init/shutdown/test modules.
- **Aliasing for graph capture.** `CaptureScope::record(&mut self, F)` requires every buffer used inside `F` to satisfy `GraphSafe` (agent 04) and be borrowed via `&GraphBuf<T>`. Compiler rejects `&mut` aliasing across captured launches. Realloc-capable allocators are forbidden inside.
- **HTTP serving.** Tokio runtime owns request queue and admission. A single worker thread runs `loop { batch=rx.recv(); p=step_launch(batch); tx.send(step_collect(p)) }`. `Worker: !Send` enforces single-thread.
- **Drop rule (re-derived from `7c212c13c`).** *Any object owning a CUDA handle must guarantee its stream is idle before destroying the handle.* Encoded by `trait CudaOwned: Drop` whose default impl fences the stream; events, graphs, buffers implement it. `Worker::drop` fences first.

## Failure modes
- Second `step_launch` while `PendingStep` live → compile error.
- DtoH wait error → `Err(RvllmError::Cuda{..})`. No silent path.
- `&mut` buffer aliased into `CaptureScope` → compile error.
- Stream busy at handle destroy → debug `assert!`, release fences then proceeds.
- `fence()` called from runtime modules → clippy denied.

## Test plan
- Compile-fail tests for borrow violations of `PendingStep` and `CaptureScope`.
- Loom model: 2 launches × 2 collects; assert no DtoH buffer overwritten before read.
- compute-sanitizer: 1000 graph-replay decode steps, clean (regression for removed `runner.rs:1358` sync).
- Drop test: `Worker` teardown after launched-but-uncollected step is leak-clean.
- HTTP integration: client disconnect mid-step does not leak events.
- Bench: ≥ v2 `step_pipelined` throughput at N=128 with one codepath.

## Cross-cutting deps
- Agent 04 (memory): provides `GraphSafe`/`GraphBuf` that gate `CaptureScope::record`.
- Agent 13 (sampling): owns pinned argmax buffers; `step_collect` reads via its DtoH slot.
- Agent 14 (graph-runtime): owns `CaptureScope`/replay; depends on the no-aliasing rule declared here.
