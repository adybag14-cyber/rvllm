# Single-Card Max-VRAM Inference Strategy

## Core Idea

Use a B200's 192GB (or H200's 141GB) to run inference as fast as if split across multiple smaller cards -- but without the multi-GPU communication overhead. A single B200 eliminates NVLink/PCIe bottlenecks, NCCL all-reduce, and tensor parallel synchronization.

## Why This Works

Multi-GPU inference with tensor parallelism (TP) splits the model across N GPUs:
- Each GPU holds 1/N of the weights
- Every attention and MLP layer requires all-reduce across GPUs
- NVLink overhead: ~5-15% per hop depending on topology
- NCCL synchronization: ~2-5ms per step for TP=4

A single B200 eliminates ALL of that. The only limit is memory bandwidth (8 TB/s on B200 HBM3e vs 3.35 TB/s on A100 HBM2e).

## VRAM Budget: What Fits on a Single B200 (192GB)

| Model | bf16 Weights | KV Cache (2048 ctx, 64 seqs) | Total | Fits? |
|-------|-------------|------------------------------|-------|-------|
| Qwen2.5-1.5B | 3 GB | 2 GB | 5 GB | Easily (187GB free for batching) |
| Llama-3.1-8B | 16 GB | 8 GB | 24 GB | Yes (168GB free) |
| Qwen2.5-14B | 28 GB | 14 GB | 42 GB | Yes (150GB free) |
| Llama-3.1-70B | 140 GB | 35 GB | 175 GB | Yes (17GB headroom) |
| Llama-3.1-70B (int8) | 70 GB | 35 GB | 105 GB | Yes (87GB free) |
| Mixtral-8x7B | 93 GB | 16 GB | 109 GB | Yes (83GB free) |

## Techniques to Maximize Single-Card Throughput

### 1. Fill VRAM with KV Cache (Max Concurrent Sequences)

The more sequences you batch, the higher your throughput. With excess VRAM:

```
available_kv_vram = total_vram - model_weights - kernel_overhead
kv_per_seq = 2 * num_layers * num_kv_heads * head_dim * max_seq_len * sizeof(dtype)
max_concurrent_seqs = available_kv_vram / kv_per_seq
```

For Llama-3.1-8B on B200:
- Available: 192 - 16 - 2 = 174 GB
- KV per seq (2048 ctx, bf16): ~2 MB
- Max concurrent: ~87,000 sequences (absurd -- you'll hit compute limits first)
- Practical sweet spot: 128-512 concurrent sequences

For Llama-3.1-70B on B200:
- Available: 192 - 140 - 2 = 50 GB
- KV per seq (2048 ctx, bf16): ~10 MB
- Max concurrent: ~5,000 sequences
- Practical sweet spot: 64-256

### 2. Speculative Decoding (Use VRAM for Draft Model)

Load a small draft model (1-3B) alongside the main model. The draft model generates N candidate tokens, the main model verifies in one forward pass. On a B200 with a 70B main + 1.5B draft:

- 70B main: 140 GB
- 1.5B draft: 3 GB
- KV caches: 40 GB
- Total: 183 GB (fits!)
- Expected speedup: 2-3x for code generation, 1.5-2x for general text

rvLLM already has `crates/rvllm-speculative/` scaffolded for this.

### 3. Prefix Caching (Amortize System Prompts)

For chatbot workloads where many requests share the same system prompt:
- Compute KV cache for the system prompt once
- Store it in GPU memory (using the excess VRAM)
- New requests skip prefill for the shared prefix

rvLLM already has `PrefixCache` in `gpu_engine.rs`. On B200 with 150GB free after model loading, you can cache thousands of prefix variants.

### 4. Quantized Weights + Full-Precision KV Cache

Use INT8/INT4 weights to cut model memory in half, then use the freed VRAM for more KV cache:

| Config | Model VRAM | KV VRAM | Max Concurrent |
|--------|-----------|---------|----------------|
| 70B bf16 | 140 GB | 50 GB | 256 seqs |
| 70B int8 | 70 GB | 120 GB | 6,000 seqs |
| 70B int4 | 35 GB | 155 GB | 15,000 seqs |

INT4 with full-precision KV cache gives the best throughput for high-concurrency serving.

### 5. Chunked Prefill (Overlap Prefill and Decode)

Instead of blocking all decode sequences while one long prompt prefills:
- Split long prefill into chunks (e.g., 512 tokens at a time)
- Interleave prefill chunks with decode steps
- Decode sequences don't stall waiting for a 4096-token prompt to finish

This is a scheduler-level optimization. rvLLM's `FifoScheduler` already separates prefill and decode -- extending to chunked prefill is straightforward.

### 6. Memory Bandwidth Optimization

Single-card inference is memory-bandwidth bound (not compute bound) for decode:
- B200: 8 TB/s HBM3e bandwidth
- H200: 4.8 TB/s HBM3
- A100: 2 TB/s HBM2e

Decode reads all model weights once per token. For 70B bf16:
- Bytes to read: 140 GB
- B200 time: 140/8000 = 17.5ms per token = 57 tok/s (single sequence)
- A100 time: 140/2000 = 70ms per token = 14 tok/s

With N concurrent sequences, weights are read ONCE and applied to all N:
- B200, 64 seqs: 17.5ms for 64 tokens = 3,657 tok/s
- A100, 64 seqs: 70ms for 64 tokens = 914 tok/s

The B200 is 4x faster than A100 purely from bandwidth.

### 7. CUDA Graph Replay (Eliminate Launch Overhead)

With many concurrent sequences, kernel launch overhead per token becomes significant. CUDA graphs capture the entire decode step (all 28+ layers) and replay as a single graph launch:

- Without graphs: ~10us per kernel * 10 kernels per layer * N layers = overhead
- With graphs: single graph launch, ~5us total

rvLLM has `CudaGraphPool` and `GraphRunner` already implemented, gated behind `cuda-graphs` feature.

## Implementation Priority for rvLLM

| Priority | Feature | Impact | Status |
|----------|---------|--------|--------|
| 1 | Max concurrent sequences (fill KV cache) | 3-5x throughput | Working (N=16 verified) |
| 2 | GPU-side sampling (skip logits DtoH) | ~2x decode speed | Implemented (argmax kernel) |
| 3 | CUDA graph replay | ~1.3x decode speed | Infrastructure ready, needs wiring |
| 4 | INT8 weight quantization | 2x more concurrent seqs | `rvllm-quant` exists, needs GPU dequant |
| 5 | Speculative decoding | 2-3x per-request latency | `rvllm-speculative` scaffolded |
| 6 | Chunked prefill | Better tail latency | Scheduler change |
| 7 | Prefix caching | Amortize system prompts | `PrefixCache` exists |

## B200 vs Multi-A100 for Inference

| Metric | 1x B200 | 4x A100 (TP=4) |
|--------|---------|-----------------|
| Total VRAM | 192 GB | 320 GB |
| Bandwidth | 8 TB/s | 8 TB/s (but with NVLink overhead) |
| Effective bandwidth | 8 TB/s | ~6.5 TB/s (after TP sync) |
| 70B bf16 fits? | Yes (single card) | Yes (split across 4) |
| Communication overhead | 0 | ~10-15% for TP all-reduce |
| Cost | ~$3/hr (vast.ai) | ~$4/hr (4x $1/hr) |
| Complexity | Single process | NCCL, TP, error handling |

**Bottom line: A single B200 is faster AND cheaper than 4x A100 for models up to 70B.**

## Next Steps

1. Test rvLLM on B200 with Qwen2.5-14B or Llama-3.1-8B
2. Measure max concurrent sequences before throughput plateaus
3. Enable CUDA graph replay for decode
4. Implement INT8 weight dequantization on GPU
5. Wire speculative decoding with a small draft model
