# 11 — cutlass-fp8

## Scope
FP8 (E4M3) GEMM variant catalog, schedule/epilogue compatibility, autotune-as-build-artifact, exact workspace contract, single graph-safe entry point. Plain FP8 GEMM and residual-fused FP8 GEMM are two separate catalogs sharing the same discipline.

## v2 problems
- `kernels/cutlass_fp8_gemm_autotune.cu:81` (and residual `:100`) hardcode `cutlass::epilogue::TmaWarpSpecializedCooperative` for the epilogue, but variants v0/v2/v3/v4/v6/v8/v11/v12/v13/v14 (and residual v0/v2/v4/v5/v6) use `KernelTmaWarpSpecialized` (non-Coop) for the mainloop. CUTLASS does not reject this at compile time; result: SM-occupancy/scratch mismatch crashes only inside captured graphs as `CUDA_ERROR_ILLEGAL_ADDRESS` (today's `_v0` failure).
- `crates/rvllm-gpu/src/cutlass_autotune.rs:786-835` `max_workspace_size` queried autotuned-variant workspaces only. Today's commit `12f4e175c` added `fp8_gemm_workspace_size` and `fp8_gemm_small_workspace_size` (lines 821-824) — necessary because dispatch falls through them when autotune cache is empty — but **did not unblock** N=128: the WS↔Coop epilogue mismatch is independent of workspace sizing.
- 4-tier dispatch (`cutlass_fp8_gemm_dispatch`: autotune → small → default → cuBLASLt) hides which kernel actually ran; same shape can land on different kernels across runs.
- Residual-fused variants — every v0..v9 — crashed under graph replay today (same epilogue mismatch root cause as plain FP8).
- Variant IDs are bare integers in CUDA and Rust (`FP8_GEMM_VARIANTS = 40`, residual `= 10`); adding a variant requires editing two files independently. No compile-time link between them.

## v3 contract

**Variant catalog** (initial; build script is the source of truth — adding a row regenerates both Rust enum and CUDA `extern "C"` block):

| Name | Tile (M,N,K) | Cluster | Mainloop sched | Epilogue sched | FastAccum |
|---|---|---|---|---|---|
| `wgmma_64x128x128_coop_e4m3` | 64,128,128 | 1,1,1 | Cooperative | Cooperative | no |
| `wgmma_64x128x128_pp_e4m3` | 64,128,128 | 1,1,1 | Pingpong | Pingpong | no |
| `wgmma_64x256x128_coop_e4m3` | 64,256,128 | 1,2,1 | Cooperative | Cooperative | no |
| `wgmma_128x128x128_coop_e4m3` | 128,128,128 | 1,1,1 | Cooperative | Cooperative | no |
| `wgmma_128x256x128_coop_e4m3` | 128,256,128 | 1,2,1 | Cooperative | Cooperative | no |
| `wgmma_128x128x256_coop_e4m3_fa` | 128,128,256 | 1,1,1 | Cooperative | Cooperative | yes |
| `wgmma_128x256x128_pp_e4m3_fa` | 128,256,128 | 1,2,1 | Pingpong | Pingpong | yes |
| `wgmma_64x128x256_coop_e4m3_fa` | 64,128,256 | 1,1,1 | Cooperative | Cooperative | yes |
| `wgmma_streamk_128x128x128_coop_e4m3_fa` | 128,128,128 | 1,1,1 | Cooperative+StreamK | Cooperative | yes |

Same table for residual catalog with prefix `res_`. **No WS-only mainloop variants ship.** Pingpong epilogue exists; if Pingpong proves graph-unsafe in CI, drop the row, don't add an exclusion list.

**Compatibility rule** (build-time enforced):
```
mainloop_sched ∈ {Coop, Pingpong}
epilogue_sched MUST equal mainloop_sched (Coop↔Coop, PP↔PP)
StreamK requires Coop↔Coop
```
`build.rs` parses the catalog table and `static_assert`s in generated `.cu` per variant: `is_same_v<EpilogueSchedule, MainloopScheduleEpiloguePair<MainloopSchedule>::type>`.

**Single API:**
```rust
pub struct VariantId(u16);  // index into static catalog
pub struct Cutlass;
impl Cutlass {
    pub fn variant(name: &'static str) -> VariantId;        // panics if unknown
    pub fn workspace_size(v: VariantId, m: u32, n: u32, k: u32) -> usize; // exact bytes
    pub fn run(v: VariantId, m: u32, n: u32, k: u32, args: &Fp8GemmArgs,
               workspace: DevPtr, stream: Stream) -> Result<(), RvllmError>;
}
```
`run` does **no fallback**. CUTLASS `Status != kSuccess` → `RvllmError::CutlassFailed { variant, m, n, k, status }`. There is no `cutlass_fp8_gemm_dispatch` chain; no "default" or "small" kernel exists outside the catalog.

**Policy file** (`autotune.policy.json`, SHA-pinned, shipped with binary):
```
{ "fp8_gemm":          { "(m,n,k)" : "wgmma_128x256x128_coop_e4m3_fa", ... },
  "fp8_gemm_residual": { "(m,n,k)" : "res_wgmma_128x128x128_coop_e4m3_fa", ... },
  "catalog_sha":  "<sha256 of catalog table at build time>",
  "kernel_sha":   "<sha256 of libcutlass_fp8.so>" }
```
Engine init reads the policy, intersects with `(buckets × layer-shapes)` from config, and:
1. If any required `(catalog, m, n, k)` is missing → `RvllmError::AutotunePolicyMissing`. **No silent fallback.**
2. Computes `max_workspace_size_for_runtime = max over all *callable* (variant, m, n, k)` — "callable" = appears in policy. Single allocation in HBM arena (agent 04).
3. Verifies `policy.catalog_sha == compile-time CATALOG_SHA` and `policy.kernel_sha == sha256(loaded .so)`. Mismatch → `RvllmError::ArtifactDrift`.

Autotune is a build-step (agent 16): runs once per `(model, gpu_sku, kernel_sha)`, output checked into the deploy tarball. Never runs on a serving box.

## Failure modes
- Unknown variant name → `panic!` at startup (programmer error).
- Missing policy entry for a required shape → `RvllmError::AutotunePolicyMissing { catalog, m, n, k }`, refuse to serve.
- Workspace under-budget at runtime → `debug_assert!`; release builds get `RvllmError::WorkspaceTooSmall`.
- Schedule mismatch → fails to build (CUDA `static_assert`).
- CUTLASS `can_implement` returns non-success → `RvllmError::CutlassFailed`; no fallback attempt.

## Test plan
- Build-fail (`trybuild`-style for `.cu`): hand-craft a WS-mainloop + Coop-epilogue row, expect `static_assert` failure.
- Unit: `workspace_size(v, m, n, k)` matches `Gemm::get_workspace_size` for every variant × representative shapes.
- Integration: `compute-sanitizer --tool memcheck` runs each catalog variant inside a captured graph at every bucket size, 100 replays, zero errors. **Includes every residual variant.**
- Numeric: per-variant cosine vs cuBLASLt FP16 reference ≥ 0.999 on Llama-3-8B / Qwen2.5-7B layer-0 weights.
- CI gate: regenerate `autotune.policy.json` on H100, fail PR if `kernel_sha` drifts without policy refresh.

## Cross-cutting deps
- 04-memory: `Cutlass::run` takes `DevPtr` from `HbmArena`; `max_workspace_size_for_runtime` feeds arena pre-flight.
- 09-layer-exec: only caller of `Cutlass::run`; passes `VariantId` resolved once at engine init.
- 14-graph: every catalog variant must capture+replay clean — the test plan above is the gate.
- 16-deploy: owns `autotune.policy.json` lifecycle (build, pin, ship, verify).
