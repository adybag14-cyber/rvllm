# 04 — memory

## Scope
HBM arena, KV layout, pinned buffers, and a `Tensor<'a>` regime making "no realloc inside graph capture" a compile-time invariant.

## v2 problems
- `runner.rs:878-885` reallocates `meta_packed`/`pinned_meta` mid-step when `total_elems > self.meta_packed.len()`. Captured graphs hold the old device pointer; replay reads freed memory → `CUDA_ERROR_ILLEGAL_ADDRESS`.
- `runner.rs:1034-1041` repeats the realloc in `upload_metadata_padded`, with no guard for capture region.
- `layer.rs:555` hardcodes `max_splits = 16` decoupled from `RuntimeConfig`. If FA3 picks `num_splits > 16`, workspace is OOB.
- `kv_cache.rs:60-69` allocates K and V as **two separate** `CudaSlice<f16>` per layer. FA3 expects interleaved `[2, ...]` stride; v2 papers over this with manual pointer math.
- `runner.rs:197-211` does ~10 independent `alloc_zeros` at init, fragmenting the address space with no arena cap.

## v3 contract

```rust
pub struct HbmArena<'ctx> { /* opaque, one cuMemAlloc */ }
impl<'ctx> HbmArena<'ctx> {
    pub fn new(ctx: &'ctx CudaContext, bytes: usize) -> Result<Self, RvllmError>;
    pub fn region(&self, name: &'static str, bytes: usize, align: usize)
        -> Result<Region<'ctx>, RvllmError>;
    pub fn used(&self) -> usize;
    pub fn capacity(&self) -> usize;
}
pub struct Region<'a> { /* &'a HbmArena, offset, len, name */ }
pub struct Tensor<'a, T> { region: &'a Region<'a>, shape: Shape, dtype: DType }

pub unsafe trait GraphSafe {}
unsafe impl<'a, T> GraphSafe for Tensor<'a, T> where 'a: 'static {}

pub struct CaptureScope<'g> { /* opaque */ }
impl<'g> CaptureScope<'g> {
    pub fn bind<'t: 'g, T: GraphSafe>(&mut self, t: &'t T) -> BoundHandle<'g>;
}

impl<'a, T> Tensor<'a, T> {
    /// ONLY way to obtain a raw device pointer. Borrows self.
    pub fn device_ptr(&self) -> u64;
}
```

KV per layer: contiguous `[2, num_blocks, block_size, num_kv_heads, head_dim]` in one `Region`. K at offset 0, V at `num_blocks*block_size*num_kv_heads*head_dim`. Strides match FA3 page-table descriptor exactly — no per-launch pointer math, HBM coalesces along `head_dim`. `block_size` from `RuntimeConfig::kv_block_size`; loader panics if absent.

Pinned host: `PinnedPool` per stream wraps `cuMemAllocHost`. Argmax double-buffered `[A,B]`; even step writes A, odd writes B; per-buffer DtoH event fences. `Vec<T>` forbidden in HtoD/DtoH — enforced by `HostStaging: !From<Vec<T>>` and `clippy.toml` denying `cuMemcpy*` whose source is not `PinnedSlice`.

Pre-flight: `Engine::init` sweeps `(cutlass_variant × bucket × gemm_shape)` from config, sums worst-case workspace + KV + scratch + pinned, prints `total_hbm_mib`, calls `HbmArena::new` once. `cudaMemGetInfo` < total → `RvllmError::HbmInsufficient { needed, free }`, refuse to start.

## Failure modes
- `HbmArena::region` past capacity → `RvllmError::ArenaExhausted`.
- Realloc inside `CaptureScope` → fails to type-check (no `&mut HbmArena`).
- `Tensor` outliving its `Region` → borrow checker rejects.
- `PinnedPool` exhaustion → Err; no silent malloc.
- `kv_block_size` missing → loader panics at config time.

## Test plan
- Unit: arena bump + alignment; `Drop`-only frees.
- Compile-fail (`trybuild`): `arena.region(...)` inside `CaptureScope` rejected.
- Integration: `compute-sanitizer --tool memcheck` over 100 decode replays; zero errors.
- Soak: arena `used()` constant after warmup at N=128 ctx=4096.
- KV stride golden: layout matches FA3 descriptor byte-for-byte.

## Cross-cutting deps
- 02-config: `max_batch`, `max_context`, `kv_block_size`, `cutlass_variant_set`.
- 03-errors: `RvllmError::{HbmInsufficient, ArenaExhausted, PinnedExhausted, KvLayoutMismatch}`.
- 14-graph: `CaptureScope` consumes `GraphSafe` handles; agent 14 owns the capture lifetime.
