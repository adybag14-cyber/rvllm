# SUPERDEPLOY: Gemma 4 31B FP8 on RTX 6000 Ada (SM89)

Running rvLLM v3 on Ada Lovelace instead of Hopper. This doc captures
every detail for provisioning, building, and operating the instance.

## Hardware

- **GPU**: NVIDIA RTX 6000 Ada Generation (46 GB GDDR6X, SM89)
- **Instance**: vast.ai offer 26855835, Germany, $0.52/hr
- **Contract ID**: 35172200
- **SSH**: `ssh -p 12200 root@ssh5.vast.ai`
- **Image**: `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel`
- **vCPUs**: 12, **RAM**: 64 GB, **Disk**: 500 GB

Note: vast.ai `Q_RTX_6000` is Turing (SM75, 24GB), NOT Ada. Search
for `RTX_6000Ada` or filter by `compute_cap>=8.9` to get the real Ada.

## SM89 vs SM90: What Changed

rvLLM v3 was built for Hopper (SM90a). Ada (SM89) lacks:
- TMA (Tensor Memory Accelerator) -- used by Flash Attention 3
- Async warp specialization -- used by FA3's producer/consumer pipeline
- WGMMA instructions -- used by CUTLASS Hopper kernels

What works unchanged on Ada:
- **cuBLASLt FP8 GEMM** -- native Ada support, no code changes
- **Fused PTX kernels** -- element-wise ops, just recompile with `-arch=sm_89`
- **All Rust code** -- GPU-agnostic orchestration layer
- **FP8 E4M3 tensor cores** -- 4th gen tensor cores on Ada support FP8

### Attention: FA3 replaced with custom SM89 kernel

`v3/kernels/paged_attention_sm89.cu` implements paged decode + prefill
for SM89. Same C ABI as the FA3 .so (function pointer signatures match).
The Rust `Fa3Kernels::load()` auto-detects SM89 symbols when SM90
symbols are missing.

Design: one thread block per (batch, head), HEAD_DIM threads. Online
softmax over KV pages. Single-split (no workspace). Correct for
head_dim 128/256/512 and GQA.

Build:
```bash
nvcc -shared -o libfa_sm89_kernels.so paged_attention_sm89.cu \
     -arch=sm_89 -O3 --use_fast_math -Xcompiler -fPIC
```

### CUTLASS: stubbed out, cuBLASLt handles all GEMMs

The forward path (`layer_exec.rs`) exclusively uses `cublaslt.fp8_gemm()`
/ `fp8_gemm_bias()` / `fp8_gemm_residual()`. CUTLASS is loaded at
bringup but never called during inference. A stub .so satisfies the
loader.

## Memory Budget (48 GB RTX 6000 Ada)

| Component | Size |
|---|---|
| Weights (FP8 E4M3) | ~31 GB |
| Embeddings (f16, tied) | ~2.7 GB |
| KV cache (60 layers, FP8, 2k ctx) | ~1 GB |
| KV cache (60 layers, FP8, 8k ctx) | ~4 GB |
| Scratch + workspace | ~3 GB |
| **Total (2k context)** | **~38 GB** |
| **Total (8k context)** | **~41 GB** |

Max practical context: ~12k tokens before OOM. For longer contexts,
use an H100/H200.

## Model

Using a pre-quantized FP8 checkpoint avoids on-device quantization:

```bash
# If a pre-quant FP8 exists on HF:
huggingface-cli download neuralmagic/gemma-3-27b-it-FP8 --local-dir /workspace/models/gemma-3-27b-it-fp8

# Otherwise use the f16 checkpoint (rvLLM quantizes weights at load time):
huggingface-cli download google/gemma-3-27b-it --local-dir /workspace/models/gemma-3-27b-it \
    --exclude "*.gguf" --exclude "*.bin"
```

## Deploy Procedure

### 1. Provision instance

```bash
vastai create instance 32085919 \
    --image pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel \
    --disk 250
```

### 2. Run deploy script

```bash
./v3/scripts/gemma4_ada_deploy.sh <instance_id>
```

This does: package tarball, upload, install Rust, download model,
compile SM89 attention .so, compile PTX for sm_89, build rvLLM.

### 3. Manual deploy (if script fails)

SSH in, then:

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Download model
pip install huggingface-hub
huggingface-cli download google/gemma-3-27b-it \
    --local-dir /workspace/models/gemma-3-27b-it

# Upload repo tarball (from local machine)
# scp -P <port> /tmp/rvllm-ada-<sha>.tar.gz root@<host>:/workspace/

# Unpack
mkdir -p /workspace/runs/deploy && cd /workspace/runs/deploy
tar xzf /workspace/rvllm-ada-*.tar.gz

# Build attention .so
cd v3/kernels
nvcc -shared -o /workspace/kernels/sm_89/libfa_sm89_kernels.so \
    paged_attention_sm89.cu -arch=sm_89 -O3 --use_fast_math -Xcompiler -fPIC

# Build PTX kernels
mkdir -p /workspace/kernels/sm_89
for cu in fused_rmsnorm_fp8_quant.cu fused_rope_cache_fp8kv.cu \
          fused_silu_fp8_quant.cu argmax.cu add_bias_f16.cu fp8_rescale.cu \
          fused_gelu_mul_fp8_quant.cu fused_qk_rmsnorm.cu \
          fused_rope_partial_fp8kv.cu logit_softcap.cu; do
    [ -f "$cu" ] && nvcc -ptx -arch=sm_89 -O3 --use_fast_math "$cu" \
        -o "/workspace/kernels/sm_89/${cu%.cu}.ptx"
done

# Manifest
python3 make_manifest.py /workspace/kernels/sm_89 sm_89

# Stub CUTLASS .so (exports all variant symbols for CutlassLib::load())
gcc -shared -fPIC -o /workspace/kernels/sm_89/libcutlass_kernels.so \
    v3/kernels/cutlass_stub_sm89.c

# Empty policy (cuBLASLt handles all GEMMs)
echo '{"arch":"sm_89","entries":{}}' > /workspace/kernels/sm_89/policy.json

# Build rvLLM
cd /workspace/runs/deploy/v3
cargo build --release --features cuda
```

## Running Inference

```bash
export RVLLM_MODEL_DIR=/workspace/models/gemma-3-27b-it
export RVLLM_KERNELS_DIR=/workspace/kernels/sm_89
export RVLLM_CUTLASS_SO=/workspace/kernels/sm_89/libcutlass_kernels.so
export RVLLM_FA3_SO=/workspace/kernels/sm_89/libfa_sm89_kernels.so
export RVLLM_POLICY=/workspace/kernels/sm_89/policy.json
export RVLLM_NO_GRAPH=1
export RVLLM_BATCH=32
export RVLLM_KV_SCALE_ABSMAX=418.0

# Benchmark
./target/release/rvllm-bench

# Text generation
export RVLLM_PROMPT="Explain quantum entanglement in simple terms."
export RVLLM_MAX_TOKENS=256
./target/release/rvllm-eval
```

### Environment Variables

| Var | Default | Notes |
|---|---|---|
| RVLLM_MODEL_DIR | (required) | HF model directory |
| RVLLM_KERNELS_DIR | (required) | PTX + manifest directory |
| RVLLM_CUTLASS_SO | (required) | Stub .so on Ada |
| RVLLM_FA3_SO | (required) | `libfa_sm89_kernels.so` on Ada |
| RVLLM_POLICY | (required) | Empty policy on Ada |
| RVLLM_BATCH | 128 | Decode batch size |
| RVLLM_NO_GRAPH | 0 | Set 1 to skip CUDA graph capture |
| RVLLM_KV_SCALE_ABSMAX | 418.0 | FP8 KV quantization range |
| RVLLM_BLOCK_SIZE | 64 | KV cache page size (tokens) |
| RVLLM_NAN_CHECK | 0 | Set 1 for NaN diagnostics |
| RVLLM_DUMP_TOKENS | 0 | Set 1 to print sampled token IDs |

## Known Limitations on Ada

1. **No CUDA graph capture** -- use `RVLLM_NO_GRAPH=1` initially.
   Graph capture may work but is untested on Ada path.

2. **Attention kernel is single-split** -- for long contexts (>4k tokens),
   each thread block iterates sequentially over all KV pages. Performance
   degrades linearly with context length. A split-K version (multiple
   blocks per head with workspace reduction) would fix this.

3. **No CUTLASS** -- cuBLASLt handles all GEMMs. Performance is good
   but CUTLASS with Ada-tuned tiles could be faster for specific shapes.

4. **Memory ceiling** -- 48 GB limits batch size and context length.
   At batch=32, context=2k: ~40 GB. At batch=128: likely OOM.

5. **Policy.json is empty** -- Fp8GemmPlan::from_policy() will fail
   for shapes not routed through cuBLASLt. The current forward path
   doesn't use the policy for GEMM dispatch, but bench variants do.

## Troubleshooting

**"no sm90 or sm89 symbols found"**
Wrong .so file. Point RVLLM_FA3_SO at `libfa_sm89_kernels.so`.

**CUDA_ERROR_OUT_OF_MEMORY**
Reduce RVLLM_BATCH or RVLLM_ARENA_GB. At 48 GB VRAM, arena should
be ~8-10 GB after weights.

**NaN in residual after layer N**
Run with `RVLLM_NAN_CHECK=1`. Check RVLLM_KV_SCALE_ABSMAX -- Gemma 4
may need a different scale than Qwen2.5.

**Kernel launch failed (attention)**
Check `nvidia-smi` for compute capability. Must be sm_89.
Check the .so was compiled with `-arch=sm_89`.

## Performance Expectations

RTX 6000 Ada vs H200 for Gemma 4 31B FP8 decode:

| Metric | RTX 6000 Ada (48GB) | H200 (144GB) |
|---|---|---|
| Memory BW | 960 GB/s | 4.8 TB/s |
| FP8 Tensor TFLOPS | ~330 | ~1979 |
| Expected tok/s (B=32) | ~2000-4000 | ~20000-40000 |
| Max context | ~12k | ~128k |
| $/hr (vast.ai) | $0.13 | $3-5 |

The Ada path is 5-10x slower than H200 but 25-40x cheaper per hour.
Good for dev/test, not for production throughput.
