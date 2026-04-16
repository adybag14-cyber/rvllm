# 03 — Error Model

## Scope
One typed `RvllmError` in `rvllm-core` with structured per-subsystem context. No `String` errors, no `Box<dyn Error>`, no `anyhow` in libraries.

## v2 problems
- `runner.rs:1359` — `LLMError::GpuError(format!("stream sync: {e}"))` swallows the underlying `DriverError`. Today's `cuGraphLaunch failed → CUDA_ERROR_ILLEGAL_ADDRESS` reached the worker as opaque text with zero kernel/stream/launch context.
- `worker.rs:205,291` — `format!("graph replay: {e}")` loses bucket id, captured graph handle, stream, and which kernel inside the graph faulted.
- `worker.rs:587,602` — `format!("begin_capture: {e}")` and `end_capture: {e}` drop padded_batch, max_ctx, and the metadata layout uploaded just before capture — the exact state needed to debug today's "captured padded, replayed unpadded" regression.
- `kv_cache.rs:63,66` — `format!("CUDA key/val cache alloc failed layer {layer}: {e}")` stringifies an OOM that should report bytes requested, free HBM, allocator running tally.
- Today's CUTLASS regression: `cutlass_fp8_gemm_residual_v0` crashed inside graph replay because workspace was sized for autotuned variants only. Surfaced as `format!("graph replay: {e}")` — no variant id, no (m,n,k), no required-vs-given workspace.

## v3 contract

```rust
// rvllm-core/src/error.rs
pub enum RvllmError {
    Cuda      { kind: CudaErrorKind, op: &'static str, ctx: CudaCtx, bt: Backtrace },
    Cutlass   { err: CutlassError,   ctx: CutlassCtx, bt: Backtrace },
    Attention { err: AttentionError, ctx: AttnCtx,    bt: Backtrace },
    Loader    { err: LoaderError,    ctx: LoaderCtx,  bt: Backtrace },
    Config    { err: ConfigError,    field: &'static str },
    Scheduler { err: SchedulerError, req_id: ReqId },
    Graph     { err: GraphError, bucket: u32, captured_sha: [u8;32], bt: Backtrace },
    Sampling  { err: SamplingError,  ctx: SampleCtx },
    Io        { err: IoError, path: PathBuf, source: io::Error },
}
pub struct CudaCtx { stream: u64, kernel: &'static str, launch: Option<Launch>, device: i32 }
pub struct Launch  { grid: (u32,u32,u32), block: (u32,u32,u32), smem: u32 }

pub enum CutlassError {
    WorkspaceTooSmall { variant: u32, m: usize, n: usize, k: usize, needed: usize, given: usize },
    EpilogueScheduleMismatch { variant: u32, mainloop: ScheduleId, epilogue: ScheduleId },
    AutotuneCacheMiss { m: usize, n: usize, k: usize, dtype: DType },
    KernelLaunchFailed { variant: u32, cuda: CudaErrorKind },
}
pub enum GraphError {
    CaptureMetadataMismatch { captured: MetaLayoutHash, replay: MetaLayoutHash },
    ReallocInsideCapture { allocator: &'static str, bytes: usize },
    BucketMissing { padded_batch: u32 },
    ReplayFailed { cuda: CudaErrorKind, kernel_at_fault: Option<&'static str> },
}
pub type Result<T> = core::result::Result<T, RvllmError>;
```

`Display` walks the chain `subsystem → op → kernel → stream → launch → backtrace`. `Debug` dumps every field. No `From<String>`/`From<&str>`. `From<io::Error>` only on `Io`, with explicit path.

## Failure modes
- **Panic** (our invariant): scratch not pre-allocated, captured-graph metadata mismatch, allocator called inside capture, debug_assert mismatch on shape/dtype, autotune cache missing the chosen variant, kernel signature mismatch at load.
- **Err** (outside our control): HF download fail, model file corrupt, steady-state OOM, request validation, scheduler queue full.
- `debug_assert!` every kernel pre/post (shapes, dtypes, alignment, stream identity).
- CUDA driver errors never silently `Err`'d. Inside compute paths → panic with `CudaCtx`. During setup (load, alloc, capture init) → `RvllmError::Cuda` with full `CudaCtx`.
- `unwrap`/`expect` forbidden outside `#[cfg(test)]`. `anyhow` allowed only in `rvllm-bench`/`rvllm-deploy`. Every `?` at a crate boundary maps via typed `From`, never `.map_err(format!)`.

## Test plan
- compile-fail: `Box<dyn Error>` and `String` in any variant rejected by clippy lint.
- panic test: capture decode, mutate metadata layout, replay → asserts `GraphError::CaptureMetadataMismatch`.
- round-trip: `WorkspaceTooSmall { variant:7, m:1024, n:4096, k:8192, needed:8MiB, given:2MiB }` survives `Display` losslessly.
- backtrace test: synthetic launch failure names call site, kernel, stream id.
- CI grep gate: `format!(".*: \{e\}").*GpuError` returns zero hits across `rvllm-*` (excl. `bench`/`deploy`).

## Cross-cutting deps
- `01-architecture` — `RvllmError` in `rvllm-core`; every crate re-exports `Result`.
- `04-memory` — allocator returns `Cuda { op:"alloc" }` with bytes-requested + free HBM.
- `05-concurrency` — stream/event id into `CudaCtx`.
- `08-metadata` — owns `MetaLayoutHash`, consumed by `GraphError`.
- `11-cutlass-fp8` — owns `ScheduleId`, variant ids, autotune schema.
- `14-graph` — emits `GraphError`; asserts no allocator call inside capture (panic).
- `15-validation` — CI grep gate + compile-fail suite.
