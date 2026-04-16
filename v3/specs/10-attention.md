# 10 — attention

## Scope
FA3 SM90 paged attention: exactly two kernels (`paged_decode`, `paged_prefill`) shipped from one statically-resolved `libfa3_kernels.so`. No PTX path, no v2/v3 coexistence, no silent fallback.

## v2 problems
- `crates/rvllm-v2/src/layer.rs:1716-1777` — `decode_attention` has TWO entries: `decode_attention_fa3_sm90` if `fa3.is_some()`, else PTX `decode_attention_gqa_v3` / `decode_attention_standard`. Today's deploy missed `libfa3_kernels.so`, fell through to PTX, crashed with `CUDA_ERROR_ILLEGAL_ADDRESS` in `fa3_v3_decode_gqa_kernel`.
- `crates/rvllm-v2/src/integration.rs:672-699` — three `.so` candidate paths checked with `path.exists()`; on miss logs `"FA3 .so not found, using PTX FA3"` at info and continues. No engine-init failure.
- `crates/rvllm-v2/src/layer.rs:1744` — `head_dim == 128` gates FA3; otherwise silently degrades to PTX.
- `crates/rvllm-v2/src/layer.rs:1761-1764` — `num_splits` and `max_blocks_per_seq` computed host-side from `attn.max_context_len`; PTX reads `block_tables[seq * max_blocks_per_seq + idx]` past `ceil(context_lens[seq]/block_size)` on padded slots → today's -10MB OOB.
- `crates/rvllm-v2/src/layer.rs:1652-1714` — prefill GQA launch reuses decode-style scalars with cumulative `seq_start_pos`; one signature serves two semantics.

## v3 contract

One C ABI in `libfa3_kernels.so`, built from FlashAttention-3 SM90 source under `kernels/fa3/`, SHA-pinned (tenet 10). `dlopen`'d once at engine init.

```rust
pub struct Fa3Kernels { /* Library + symbol fns */ }

impl Fa3Kernels {
    pub fn load(so: &Path, cfg: &RuntimeConfig) -> Result<Self, RvllmError>;
    pub fn decode_workspace_bytes(&self, num_seqs: u32, num_kv_heads: u32) -> usize;
    pub fn prefill_workspace_bytes(&self, num_tokens: u32, num_kv_heads: u32) -> usize;
    pub unsafe fn paged_decode(&self, a: PagedDecodeArgs, s: CuStream)
        -> Result<(), RvllmError>;
    pub unsafe fn paged_prefill(&self, a: PagedPrefillArgs, s: CuStream)
        -> Result<(), RvllmError>;
}

#[repr(C)]
pub struct PagedDecodeArgs {
    pub out: u64, pub q: u64,
    pub k_cache: u64, pub v_cache: u64,
    pub block_tables: u64, pub context_lens: u64,
    pub workspace: u64,
    pub scale: f32,
    pub num_seqs: i32, pub num_heads: i32, pub num_kv_heads: i32,
    pub head_dim: i32, pub block_size: i32,
    pub max_blocks_per_seq: i32, pub num_blocks_total: i32,
}

#[repr(C)]
pub struct PagedPrefillArgs {
    pub out: u64, pub q: u64,
    pub k_cache: u64, pub v_cache: u64,
    pub block_tables: u64,
    pub cu_seqlens_q: u64, pub cu_seqlens_k: u64,
    pub workspace: u64,
    pub scale: f32,
    pub num_tokens: i32, pub num_seqs: i32,
    pub num_heads: i32, pub num_kv_heads: i32,
    pub head_dim: i32, pub block_size: i32,
    pub max_blocks_per_seq: i32, pub num_blocks_total: i32,
}
```

Note: NO `max_context_len`, NO `num_splits`. Per-seq adaptive split is computed inside the kernel from `context_lens[seq]`.

`Fa3Kernels::load` enforces:
- `head_dim == 128`.
- `num_heads % num_kv_heads == 0`, ratio ∈ {1,2,4,7,8,14}; GQA handled inside the kernel via the runtime ratio.
- `block_size ∈ {16, 32}` from `RuntimeConfig`.

Kernel-internal predication (compute-sanitizer-asserted):
- `context_lens[i] == 0` → write zeros to `out[i,:,:]`; never load `block_tables[i,*]`, never load KV.
- For seq `i`, kernel reads `block_tables[i * max_blocks_per_seq + idx]` only for `idx < ceil(context_lens[i] / block_size)`.

Workspace pre-allocated once by `rvllm-mem` arena from `decode_workspace_bytes` / `prefill_workspace_bytes` over the worst-case bucket; pointer stable across capture.

## Failure modes
- `.so` missing or `dlopen` fails → `RvllmError::Fa3KernelMissing { path }`. Engine refuses to start.
- Symbol missing → `RvllmError::Fa3SymbolMissing { name }`.
- `head_dim != 128` → `RvllmError::UnsupportedHeadDim { got }` at `load`.
- GQA ratio invalid → `RvllmError::UnsupportedGqaRatio { num_heads, num_kv_heads }`.
- Workspace too small → `debug_assert!`; release returns `RvllmError::WorkspaceUndersized`.
- Pointer mutation inside capture → forbidden by `GraphSafe` borrow (agent 04).

## Test plan
- Unit: `load` rejects missing path, head_dim=64, GQA ratio=3.
- Parity vs HF at every bucket × ctx ∈ {1,64,512,2048,4096}; cosine ≥ 0.999 per layer.
- Padded slots: `context_lens = [128,0,256,0]` → zero outputs at slots 1,3 and zero memcheck OOBs on their `block_tables` rows.
- `compute-sanitizer --tool memcheck` over 1k decode replays at N ∈ {1,16,64,128}; zero errors to merge.
- Prefill chunked: (2k+1k+512+256) tokens vs HF, cosine ≥ 0.999.
- Capture replay 1k iters; pointer stability via `cudaGraphExecUpdate`.

## Cross-cutting deps
- 04-memory: KV interleaved `[2, num_blocks, block_size, num_kv_heads, head_dim]`; workspace in arena.
- 08-metadata: `block_tables` `[num_seqs, max_blocks_per_seq] i32`, `context_lens` `[num_seqs] i32`, `cu_seqlens_*` `[num_seqs+1] i32`.
- 11-cutlass-fp8: shares no kernels and no workspace region.
- 14-graph: `paged_decode` is the sole attention op inside the captured decode graph; `paged_prefill` runs outside capture.
- 16-deploy: `libfa3_kernels.so` is a SHA-pinned artifact; deploy aborts on checksum drift or missing file.
