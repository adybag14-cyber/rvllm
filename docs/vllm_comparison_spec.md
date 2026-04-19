# vLLM vs rvLLM Head-to-Head Comparison Spec

## Setup

- **GPU**: H100 SXM 80GB (single GPU)
- **Model**: google/gemma-4-31b-it (or RedHatAI/gemma-4-31B-it-FP8-Dynamic for FP8)
- **Batch sizes**: 1, 4, 8, 16, 32, 64, 128, 256, 512
- **Decode iterations**: 50 per batch size
- **Warmup**: 5 iterations before timing

## Token counting

Both engines must count tokens the same way. The bench binary measures:
- `tok/s = batch_size * iters / elapsed_seconds`
- `ms/step = elapsed_ms / iters` (one step = one decode token for all seqs in the batch)

For vLLM benchmark_throughput.py:
- Use `--num-prompts N` with single-token prompts, `--output-len 50`
- `tok/s = total_output_tokens / elapsed_seconds`
- Ensure `--enforce-eager` is NOT set (vLLM should use CUDA graphs too)

## vLLM commands

```bash
# Install vLLM (if not already)
pip install vllm

# Start vLLM server with FP8
python -m vllm.entrypoints.openai.api_server \
  --model RedHatAI/gemma-4-31B-it-FP8-Dynamic \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --port 8000

# Benchmark decode throughput per batch size
for B in 1 4 8 16 32 64 128 256 512; do
  python -m vllm.entrypoints.benchmarks.benchmark_throughput \
    --model RedHatAI/gemma-4-31B-it-FP8-Dynamic \
    --num-prompts $B \
    --input-len 1 \
    --output-len 50 \
    --dtype auto \
    --gpu-memory-utilization 0.95 \
    2>&1 | tee vllm_b${B}.log
done
```

## rvLLM commands

```bash
for B in 1 4 8 16 32 64 128 256 512; do
  RVLLM_BATCH=$B RVLLM_ITERS=50 RVLLM_ARENA_GB=72 \
  RVLLM_MODEL_DIR=/workspace/models/gemma4-31b-fp8 \
  RVLLM_KERNELS_DIR=... \
  RVLLM_FA3_SO=... \
  RVLLM_POLICY=... \
  RVLLM_CUTLASS_SO=... \
  RVLLM_FA_FALLBACK_SO=... \
  target/release/rvllm-bench 2>&1 | tee rvllm_b${B}.log
done
```

## Metrics to capture

| Metric | rvLLM source | vLLM source |
|--------|-------------|-------------|
| tok/s | `bench:` line in stdout | benchmark output |
| ms/step | `bench:` line in stdout | total_time / (num_prompts * output_len) * 1000 |
| GPU memory | `nvidia-smi` during run | `nvidia-smi` during run |
| Graph nodes | `[graph]` line in stderr | N/A (vLLM doesn't report) |

## Current rvLLM numbers (fresh sweep, post-fusion)

| Batch | tok/s | ms/step |
|-------|-------|---------|
| 1 | 52 | 19.2 |
| 4 | 229 | 17.5 |
| 8 | 452 | 17.7 |
| 16 | 900 | 17.8 |
| 32 | 1,723 | 18.6 |
| 64 | 3,097 | 20.7 |
| 128 | 5,114 | 25.0 |
| 256 | 6,897 | 37.1 |
| 512 | 7,943 | 64.5 |

## Previous vLLM comparison (from earlier session)

These were measured on a prior instance:

| Batch | rvLLM | vLLM | Delta |
|-------|-------|------|-------|
| 1 | 51 | 37 | +38% |
| 4 | ~148 (est) | ~148 | ~0% |
| 64 | ~2200 (est) | ~2200 | ~0% |

Need fresh vLLM numbers on the same instance for a fair comparison.

## Checklist before running

- [ ] Kill any existing vLLM/python processes on GPU
- [ ] Verify GPU memory is free (< 100 MiB)
- [ ] Run rvLLM sweep first (already done above)
- [ ] Install vLLM if not present
- [ ] Run vLLM sweep
- [ ] Record nvidia-smi memory for both
- [ ] Compare tok/s at each batch size
