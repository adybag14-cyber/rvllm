# Updating the docs + reproducing the bench

This doc covers (a) the runbook for updating `docs/bench.html` after a new bench run, (b) exactly how to reproduce every data point in the study from a clean H100 box, and (c) profiling with nsys + ncu.

- Every measured number in `docs/bench.html` can be regenerated in ~20 minutes on one H100.
- Every projected / estimated number has its derivation called out where it appears in the page.
- All artifacts (bench logs, nsys rep files, MMLU outputs) are captured to known paths on the H100 box and can be scp'd down.

---

## 1. Keeping the docs up to date

**Source of truth:** `docs/bench.html` — hand-edited HTML with embedded React + Recharts via unpkg CDN.

### Edit flow

1. Edit `docs/bench.html` locally.
2. Open in a browser (`file:///Users/andy/rvllm/docs/bench.html`) — charts render from CDN, no build step.
3. `git commit` and push.
4. GitHub Pages rebuilds in ~60 s. Custom host (`solidsf.com/rvllm/docs/bench.html`) is served by the Worker in `deploy/cf-worker/` — deploy with:

   ```bash
   cd deploy/cf-worker && bash build.sh && wrangler deploy
   ```

### Where the numbers live (single-point-of-truth data arrays)

Inside `docs/bench.html`, every chart's data is a JavaScript array at the bottom of the page. Replace the numbers, commit, done.

| Array name         | Chart                                           | What it is                                      |
| ------------------ | ----------------------------------------------- | ----------------------------------------------- |
| `tsData`           | Throughput grouped-bar                          | tok/s per N, rvLLM vs vLLM (hot)                |
| `ttftData`         | TTFT grouped-bar                                | ms TTFT per N, rvLLM hot vs vLLM hot            |
| `nsysData`         | nsys horizontal bar                             | ms per step for top 10 kernels at N=8           |
| `footprintData`    | Deployment footprint bar                        | MB total per engine                             |
| `vramData`         | VRAM horizontal stacked bar                     | GB reserved / free per engine on 80 GB H100     |
| `scalingData`      | Projected throughput vs batch (line)            | measured + projected tok/s for N up to 16k      |
| `migData`          | MIG TP-2 step-time breakdown                    | compute + comm ms per step (hypothetical TP-2)  |
| `tpFutureData`     | Multi-GPU TP projection (model size × card count) | projected tok/s for 7B/14B/32B/70B/180B/405B  |
| `coldStartData`    | Cold-start stacked bar                          | seconds per startup phase per engine            |

Tables in the page (HTML `<table>` blocks) carry the same numbers — if you bump the chart data, bump the table too. All measured numbers are also in `docs/bench-data/` (see §4).

---

## 2. Reproducing the rvLLM bench (tok/s + cold/hot TTFT)

### Prerequisites

- H100 SXM 80 GB (vast.ai, Lambda, etc.). CUDA 12.4+ driver.
- A fresh box with the project deployed. If starting from scratch, follow the v2-style "ship one immutable payload per run" pattern described in `CLAUDE.md` (package local tree, upload to `/workspace/runs/<sha>/`, unpack, verify SHA, run).
- Kernel artifacts built: `bash kernels/build.sh && bash kernels/build_cutlass_so.sh && bash kernels/build_fa3.sh && bash kernels/build_w4a8.sh`
- Model: Qwen2.5-7B-Instruct safetensors at `/workspace/models/qwen25-7b-instruct/`
- Manifest + policy: `python3 v3/kernels/make_manifest.py /workspace/rvllm/kernels/sm_90 sm_90 $(git rev-parse HEAD)` then `python3 v3/kernels/make_policy.py /workspace/rvllm/kernels/sm_90/policy.json $(git rev-parse HEAD)`
- Rust binary: `cd v3 && cargo build --release --features cuda -p rvllm-bench`

### The exact command for the published numbers

For each `N ∈ {1, 8, 16, 64, 128, 256, 512}`:

```bash
cd /workspace/rvllm/v3
RVLLM_MODEL_DIR=/workspace/models/qwen25-7b-instruct \
RVLLM_KERNELS_DIR=/workspace/rvllm/kernels/sm_90 \
RVLLM_CUTLASS_SO=/workspace/rvllm/kernels/sm_90/libcutlass_kernels.so \
RVLLM_FA3_SO=/workspace/rvllm/kernels/sm_90/libfa3_kernels.so \
RVLLM_POLICY=/workspace/rvllm/kernels/sm_90/policy.json \
RVLLM_TTFT=1 \
RVLLM_REAL_PREFILL=1 \
RVLLM_PREFILL_LEN=16 \
RVLLM_BATCH=$N \
RVLLM_ITERS=128 \
RVLLM_WARMUP=5 \
./target/release/rvllm-bench
```

Output lines look like:

```
bench: batch=128 iters=128 -> 21,946 tok/s (5.832 ms/step) ttft_cold=80.52ms ttft_hot=83.20ms
{"batch":128,"iters":128,"tok_per_sec":21946.1,"ms_per_step":5.8324,"ttft_cold_ms":80.524,"ttft_hot_ms":83.204}
```

Save each N's JSON line to `/workspace/bench-results/rvllm_N${N}.json` for later.

### Knobs documented

| Env var              | Default  | Meaning                                                                 |
| -------------------- | -------- | ----------------------------------------------------------------------- |
| `RVLLM_BATCH`        | required | Batch size N                                                            |
| `RVLLM_ITERS`        | required | Timed decode iterations (after warmup)                                  |
| `RVLLM_WARMUP`       | required | Eager warmup decode iterations before graph capture                     |
| `RVLLM_TTFT`         | 0        | If `1`, measure and report cold + hot TTFT                              |
| `RVLLM_REAL_PREFILL` | 0        | If `1`, use real FA3 paged-prefill; else 16-step eager faux-prefill     |
| `RVLLM_PREFILL_LEN`  | 16       | Prompt tokens per sequence (real prefill only)                          |
| `RVLLM_BLOCK_SIZE`   | 64       | FA3 paged KV block size (tokens per page)                               |

### What counts as "cold" vs "hot"

Our bench runs **two timed prefill calls back-to-back** in a single process:

- **Cold TTFT** — first prefill call from a fresh process. Includes cuBLASLt per-shape heuristic cost (~200–400 ms on first-ever visit to an `(M, N, K, epilogue)` shape) and one-time memory coalescing.
- **Hot TTFT** — second prefill call with heuristics already cached. Represents per-request TTFT under steady serving load.

This is timed inside `v3/crates/rvllm-runtime/src/bring_up.rs` around the `run_real_prefill` closure.

---

## 3. Reproducing the vLLM comparison

### Start the server

```bash
nohup vllm serve /workspace/models/qwen25-7b-instruct \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9 \
  --host 127.0.0.1 --port 8000 \
  --disable-log-stats > /tmp/vllm_server.log 2>&1 < /dev/null & disown
```

Wait for `/health` to return 200 (~60–90 s).

### Run the sweep (cold + hot per N)

For each `N ∈ {1, 8, 16, 64, 128, 256, 512}`, run **twice** in sequence — first run = cold, second = hot — and capture each output:

```bash
for N in 1 8 16 64 128 256 512; do
  for PHASE in cold hot; do
    vllm bench serve \
      --backend vllm \
      --model /workspace/models/qwen25-7b-instruct \
      --dataset-name random \
      --num-prompts $N --max-concurrency $N \
      --random-input-len 16 --random-output-len 512 \
      --host 127.0.0.1 --port 8000 \
      --ignore-eos \
      > /tmp/vllm_N${N}_${PHASE}.log 2>&1
  done
done
```

Parse `Output token throughput (tok/s)` and `Mean TTFT (ms)` from each log. These are the numbers in the `tsData` / `ttftData` columns of `bench.html`.

### ⚠️ Critical quality caveat (discovered during this study)

**vLLM with `--kv-cache-dtype fp8_e4m3` and no embedded k/v_scale in the checkpoint produces garbage output.** MMLU on Qwen2.5-7B-Instruct under the above config scored **23.0%** — below random (25%) — because the KV cache runs with scale=1.0 (vLLM warns: _"Using KV cache scaling factor 1.0 for fp8_e4m3. If this is unintended, verify that k/v_scale scaling factors are properly set in the checkpoint."_).

**What this means for the benchmark:**

- The tok/s and TTFT numbers are still valid as kernel-throughput measurements — vLLM is executing the GEMMs either way.
- But the **output-quality comparison is NOT symmetric**: vLLM under the fair-matched config is broken. Our rvLLM path uses per-tensor FP8 scales computed in `fn_rope_cache_fp8kv`, so it should score near FP16 baseline (~74% MMLU per the Qwen2.5 tech report) — but we don't have a text-gen path in v3 yet to measure directly (see §6).
- To measure vLLM *meaningfully* at FP8 E4M3 KV on Qwen2.5, one would need to pre-compute `k_scale` / `v_scale` per layer, patch them into the safetensors, and re-serve. Not done in this study.

---

## 4. Where the results live

On the H100 box (`ssh -p 29220 root@ssh3.vast.ai`):

| Path                                                       | Contents                                         |
| ---------------------------------------------------------- | ------------------------------------------------ |
| `/tmp/vllm_server.log`                                     | vLLM serve log (current session)                 |
| `/tmp/vllm_N${N}_{cold,hot}.log`                           | per-N `vllm bench serve` outputs                 |
| `/workspace/bench-results/rvllm_N${N}.json`                | rvllm-bench JSON per N (create this dir)         |
| `/workspace/nsys-runs/rvllm_n8.nsys-rep`                   | nsys profile of N=8 rvllm-bench                  |
| `/workspace/nsys-runs/rvllm_n8_cuda_gpu_kern_sum.csv`      | nsys --stats=true CSV: per-kernel summary        |
| `/workspace/nsys-runs/rvllm_n8_cuda_gpu_trace.csv`         | nsys --stats=true CSV: per-launch trace          |
| `/tmp/rvllm_eval/fp8_mmlu/.../results_*.json`              | lm-eval MMLU output (Qwen2.5-7B FP8 + FP8 KV)    |

In this repo:

| Path                                 | Contents                                           |
| ------------------------------------ | -------------------------------------------------- |
| `docs/bench.html`                    | The viz page itself (canonical numbers)            |
| `docs/index.html`                    | Marketing overview with 3-batch headline           |
| `docs/paper/rvllm-body.tex`          | Paper body (mirrors bench numbers in prose)        |
| `docs/paper/rvllm{,-bw,-dark}.pdf`   | Rendered paper PDFs                                |
| `README.md`                          | Engine readme — keep the numbers consistent        |
| `docs/DOCSUPDATE.md`                 | This file                                          |

After each new bench run, `scp` the relevant results dir down and update the `docs/bench.html` data arrays + tables + paper text in one commit.

---

## 5. Profiling — nsys (summary timings) and ncu (per-kernel detail)

### nsys (the "where does each step's time go" profile)

This is what produced the N=8 breakdown in `bench.html → Where the time goes`. It captures every CUDA kernel launch in a replay window and reports cumulative time per kernel.

```bash
# Make sure no other GPU process is running — nsys shares the device.
pgrep -af vllm  # if any, kill them

mkdir -p /workspace/nsys-runs
cd /workspace/rvllm/v3

RVLLM_MODEL_DIR=/workspace/models/qwen25-7b-instruct \
RVLLM_KERNELS_DIR=/workspace/rvllm/kernels/sm_90 \
RVLLM_CUTLASS_SO=/workspace/rvllm/kernels/sm_90/libcutlass_kernels.so \
RVLLM_FA3_SO=/workspace/rvllm/kernels/sm_90/libfa3_kernels.so \
RVLLM_POLICY=/workspace/rvllm/kernels/sm_90/policy.json \
RVLLM_BATCH=8 RVLLM_ITERS=64 RVLLM_WARMUP=5 \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cudabacktrace=none \
  --output=/workspace/nsys-runs/rvllm_n8 \
  --stats=true \
  ./target/release/rvllm-bench
```

Output:

- `rvllm_n8.nsys-rep` — binary profile (open in Nsight Systems GUI)
- `rvllm_n8_cuda_gpu_kern_sum.csv` — per-kernel summary (count, avg, total ms)
- `rvllm_n8_cuda_gpu_trace.csv` — every individual launch

Parsing pattern for the top-10 kernel table:

```bash
# Total kernel time per name, sorted descending:
python3 -c '
import csv, collections
tot = collections.defaultdict(lambda: [0, 0])
with open("/workspace/nsys-runs/rvllm_n8_cuda_gpu_kern_sum.csv") as f:
    for row in csv.DictReader(f):
        name = row["Name"]; n = int(row["Instances"]); t = float(row["Total Time (ns)"])
        tot[name][0] += n; tot[name][1] += t
for name, (n, t) in sorted(tot.items(), key=lambda x: -x[1][1])[:10]:
    print(f"{t/1e6:7.3f} ms  {n:4d} launches  {name[:60]}")
'
```

Sanity check: total kernel time / number of timed steps = bench `ms/step`. If they disagree by more than 5%, there's either launch overhead or your iterations count is off.

### ncu (per-kernel deep-dive)

When nsys points at a kernel that's slow but you don't know *why* (memory-bound? launch-bound? compute-bound?), ncu gives SM utilization, HBM bandwidth, occupancy, register pressure, warp stalls.

ncu is expensive (each kernel runs several times with different counter groups enabled). Limit to one or two kernels of interest:

```bash
ncu \
  --set=full \
  --kernel-name-base=function \
  --kernel-name='regex:xmma_gemm.*fp8' \
  --launch-skip=10 --launch-count=5 \
  --export=/workspace/nsys-runs/rvllm_n8_gemm.ncu-rep \
  /workspace/rvllm/v3/target/release/rvllm-bench
```

Then open `rvllm_n8_gemm.ncu-rep` in Nsight Compute GUI or dump a text summary:

```bash
ncu --import /workspace/nsys-runs/rvllm_n8_gemm.ncu-rep --page=details
```

Key metrics to watch for the FP8 GEMMs:

- `sm__throughput.avg.pct_of_peak_sustained_active` — should be > 60% at decent batch
- `dram__bytes.sum.per_second.peak` — we want it near 3 TB/s at large batch
- `smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active` — FP8 Tensor Core utilization
- Launch stall reasons: `smsp__warp_issue_stalled_*`

The N=8 gate_up finding ("2.5× slack") came from comparing `dram__bytes.sum` per call against the HBM-BW lower bound for that shape — ncu surfaces the ratio directly.

### compute-sanitizer (correctness, not performance)

Run after any kernel edit touching FA3 or the new w4a8 path. Zero errors required before claiming a number:

```bash
compute-sanitizer --tool=memcheck ./target/release/rvllm-bench
```

---

## 6. Known gaps (what the next study would measure)

- **rvLLM text-gen path** doesn't exist yet in v3 (no tokenizer load, no detokenize). ~700–900 LoC to add: `rvllm-runtime/src/generate.rs` + a `rvllm_eval` bin + HF tokenizer via the `tokenizers` crate. Once it lands, direct MMLU / HumanEval on rvLLM becomes a 30-minute run and we can compare numerically against the Qwen2.5 baseline.
- **Long context** (2 k / 4 k / 8 k prompts): rvLLM's bench currently allocates `gate_up_out` as `N × prefill_len × 18944 × 4 bytes`, which explodes at `N=256, prefill_len=4096` (76 GiB). Fix is a trivial scratch-sizing adjustment — not landed. At `N=64, prefill_len=8192` the sweep did run, and rvLLM held at ~12 k tok/s vs vLLM 3.1 k tok/s (rvLLM +286%).
- **W4A8 numbers**: the kernel builds and the Rust FFI works, but the dispatcher isn't wired into `layer_exec.rs` yet. Projected +25–40% at bandwidth-bound batches. Tracked as task `#28` in the workspace.
- **Skinny-M cuBLASLt autotune** (commit `82b1f592e`): fix for the N=8 GEMM-slack issue lives on `main` under `v0.3.1` but has not been re-benched yet due to GPU contention during the study.

---

## 7. One-liner sanity check after any change

```bash
# Should reproduce within a few percent of the published N=128 number.
RVLLM_BATCH=128 RVLLM_ITERS=128 RVLLM_WARMUP=5 \
RVLLM_TTFT=1 RVLLM_REAL_PREFILL=1 RVLLM_PREFILL_LEN=16 \
RVLLM_MODEL_DIR=/workspace/models/qwen25-7b-instruct \
RVLLM_KERNELS_DIR=/workspace/rvllm/kernels/sm_90 \
RVLLM_CUTLASS_SO=/workspace/rvllm/kernels/sm_90/libcutlass_kernels.so \
RVLLM_FA3_SO=/workspace/rvllm/kernels/sm_90/libfa3_kernels.so \
RVLLM_POLICY=/workspace/rvllm/kernels/sm_90/policy.json \
./v3/target/release/rvllm-bench
```

Expected (v0.3.0 on H100 SXM): `21,946 tok/s ± 3%`, `TTFT cold ≈ 80 ms, hot ≈ 83 ms`.

If you're off by more than 5%, something regressed — `git bisect` it.
