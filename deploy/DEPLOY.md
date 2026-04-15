# Deploy Protocol

## Source Layout

```
kernels/*.cu          -- ALL kernel sources (PTX + CUTLASS), must be in git
kernels/kernel_bucket/ -- scratch/experimental kernels, NOT deployed (untracked)
kernels/build.sh      -- compiles *.cu -> kernels/<arch>/*.ptx
kernels/build_cutlass_so.sh -- compiles cutlass_*.cu -> kernels/<arch>/libcutlass_kernels.so
```

## Compiled Artifacts (built on GPU box, not in git)

```
kernels/sm_90/*.ptx                -- PTX for H100
kernels/sm_90/*.cubin              -- cubin for cooperative kernels
kernels/sm_90/libcutlass_kernels.so -- CUTLASS shared library
target/release/rvllm-v2-bench     -- bench binary
```

## Deploy Steps

1. Build tarball from git HEAD (never from dirty tree):
   ```
   SHA=$(git rev-parse --short HEAD)
   git archive --format=tar.gz --prefix=rvllm/ HEAD -o /tmp/rvllm-${SHA}.tar.gz
   ```

2. Upload to GPU box:
   ```
   scp -P <port> /tmp/rvllm-${SHA}.tar.gz root@<host>:/workspace/
   ```

3. Clean + unpack on GPU box:
   ```
   pkill -9 -f rvllm; pkill -9 -f deploy
   rm -rf /workspace/runs/*
   mkdir -p /workspace/runs/${SHA}
   cd /workspace/runs/${SHA}
   tar xzf /workspace/rvllm-${SHA}.tar.gz
   ```

4. Run deploy script from inside the unpacked tree:
   ```
   cd /workspace/runs/${SHA}/rvllm
   bash deploy/deploy_and_bench.sh --with-cutlass --model Qwen/Qwen2.5-7B --fp8
   ```

## What the Deploy Script Does

1. Compiles all kernels/*.cu -> kernels/sm_90/*.ptx (skips cutlass_*)
2. Clones CUTLASS headers, builds libcutlass_kernels.so
3. Builds rvllm-v2-bench binary (cargo, ~50s)
4. Downloads model from HuggingFace (cached at /root/.cache/huggingface/)
5. Runs benchmark: direct engine, no HTTP, FP8

## Model Cache

HuggingFace models cache at `/root/.cache/huggingface/hub/`. This persists across deploys
and does NOT need to be cleaned. Only clean if switching models or debugging weight issues.

## Cleaning the Server

Before any deploy:
```
pkill -9 -f rvllm
pkill -9 -f deploy
pkill -9 -f bench
rm -rf /workspace/runs/*
```

Do NOT delete:
- /root/.cache/huggingface/ (model cache)
- /root/.cargo/ (Rust toolchain)
- /root/cutlass/ (CUTLASS headers)

## Rules

- NEVER edit code on the server. All edits local, rebuild tarball, redeploy.
- NEVER copy files from old runs into new runs.
- NEVER run from a shared mutable path. One SHA = one /workspace/runs/<sha>/ dir.
- ALWAYS verify kernel sources are in git before deploying (git status kernels/).
- If local SHA != deployed SHA, abort and redeploy.

## Known Issues

- bench binary reloads the full model for each batch size (slow, should be fixed to reuse one engine)
- fa3_sm90_wrapper.cu fails PTX compile (needs CUTLASS headers) -- not critical, handled by .so
- CUTLASS autotune files (cutlass_fp8_gemm_autotune.cu) take ~10min to compile (133 template variants)

## Vast.ai Instances

Check running instances:
```
vastai show instances
```

SSH format: `ssh -p <port> root@<host>`

## Previous Benchmark Baseline (April 15, 2026)

See docs/benchmark-history.md for the full comparison table.
vLLM 0.19.0 FP8 numbers are already recorded there -- do NOT re-run vLLM unless params change.
