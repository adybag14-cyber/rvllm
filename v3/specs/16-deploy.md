# 16 — deploy + bench

## Scope
SHA-pinned tarball, autotune in CI, kernel build outside the box, deploy-and-bench in one shot.

## v2 problems
- Tarball came from local git but launch script ran `make` for CUTLASS on the prod GPU box (~10 min); `crates/rvllm-v2/build.rs:61` triggers `nvcc` if `.so` missing.
- FA3 `.so` MISSING in tarball; engine silently fell to a stale `.ptx` path (`crates/rvllm-v2/src/engine.rs:412`), crashed mid-decode with `CUDA_ERROR_ILLEGAL_ADDRESS`.
- Stale `~/.cache/rvllm/fp8_autotune.json` got picked up; dispatched to a non-existent variant (`crates/rvllm-v2/src/cutlass.rs:233`).
- `tools/deploy_v2.sh` had no manifest, no SHA check; remote and local diverged silently.
- Bench used `include_str!` for prompts but didn't pin the set hash, so "same" benches compared different inputs.

## v3 contract

### Build artifacts (SHA-pinned)
1. `bin/{rvllm-server,rvllm-bench,rvllm-autotune}` — Rust release binaries, statically linked to `rvllm-runtime`.
2. `lib/libcutlass_kernels.so` — all FP8 GEMM variants (built once, `-arch=sm_90a`).
3. `lib/libfa3_kernels.so` — REQUIRED. No `.ptx` fallback. Build fails if `nvcc` < 12.4.
4. `kernels/*.ptx` — fused norm/quant/RoPE+KV-write (compiled `nvcc --ptx`, sm_90a).
5. `policy.json` — autotune output (format owned by agent 11), immutable per build SHA.

### Build location
GitHub Actions on H100 self-hosted runner OR ephemeral via `tools/spawn-h100.sh`. **NEVER** on the production GPU box. Flow: dev `git push` → CI builds + autotunes → CI uploads to `hf://m0at/rvllm-builds/<sha>/`.

### Tarball layout (frozen)
```
rvllm-<sha>.tar
  REVISION                  # 40-char SHA
  bin/{rvllm-server,rvllm-bench,rvllm-autotune}
  lib/{libcutlass_kernels.so,libfa3_kernels.so}
  kernels/*.ptx
  policy.json
  manifest.json             # {sha, files:[{path,sha256,bytes}], built_at, runner}
  config-template.toml
  README
```

### Deploy: `tools/deploy.sh <sha> <host>`
1. `hf download m0at/rvllm-builds/<sha>/ → build/`; verify each file's sha256 against `manifest.json`.
2. `tar -cf rvllm-<sha>.tar -C build/ .`
3. `ssh: rm -rf /opt/rvllm/<sha>/; mkdir; tar -xf; chmod +x bin/*`
4. `ssh: rvllm-bench --revision <sha> --verify-only` — asserts embedded `REVISION == <sha>` and on-disk checksums match manifest.
5. If `--bench`: full bench, JSON to `/opt/rvllm/<sha>/bench-<ts>.json`.

Refuse to start on any missing artifact, checksum mismatch, or REVISION drift.

### Bench harness
- Deterministic prompts under `rvllm-bench/prompts/`, hash pinned in manifest.
- Per-bucket (N=1,4,8,16,32,64,128) tok/s + p50/p99 latency, JSON for `tools/bench-gate` (agent 15).
- `--profile` spawns under `nsys profile` + `ncu --set full`. Zero overhead in normal mode (no sampling hooks compiled in).

### Cloud abstraction
`tools/spawn-h100.sh [vast|runpod|aws] [--gpu h100|b200]` rents, returns `ssh_string` + `cleanup_token`. CI uses this for ephemeral builders.

## Failure modes
- Missing artifact → `RvllmError::ArtifactMissing { path }` (panic at startup).
- Checksum mismatch → `RvllmError::ArtifactCorrupt { path, expected, got }` (panic).
- REVISION drift → `RvllmError::RevisionDrift { binary, expected, got }` (panic).
- Stale autotune (policy.json absent or SHA mismatch) → panic; never read `~/.cache`.

## Test plan
- `tests/deploy_smoke.rs`: spawn vast.ai, deploy, `--verify-only`, exit 0.
- `tests/manifest_drift.rs`: flip one byte of `libfa3_kernels.so`, assert panic with `ArtifactCorrupt`.
- `tests/bench_determinism.rs`: run bench twice on same SHA, tok/s within 1%.
- CI: every PR builds tarball, deploys to ephemeral H100, gates on agent-15 thresholds.

## Cross-cutting deps
- agent 03: `ArtifactMissing`, `ArtifactCorrupt`, `RevisionDrift` error variants.
- agent 11: `policy.json` schema (autotune format, version field).
- agent 15: bench JSON schema consumed by `bench-gate`.
- agent 02: `--revision`, `--verify-only`, `--profile` CLI flags.
