# Decode Correctness Fix Checklist

## Fix 1: Add QKV biases to GPU layer path

### Problem
Qwen2.5 has `q_proj.bias`, `k_proj.bias`, `v_proj.bias`. The CPU worker path applies them (`gpu_worker.rs:807-815`). The GPU layer path (`gpu_layer.rs`) does bare `X @ W^T` with no bias. `GpuLayerWeights` has no bias fields. Error compounds through 28 layers.

### Steps
- [ ] 1a. Add optional bias fields to `GpuLayerWeights` in `gpu_layer.rs`
  - `q_proj_bias: Option<&'a CudaSlice<f32>>`
  - `k_proj_bias: Option<&'a CudaSlice<f32>>`
  - `v_proj_bias: Option<&'a CudaSlice<f32>>`
- [ ] 1b. Update `layer_weights()` in `gpu_runner.rs` to fetch biases
  - Weight names: `model.layers.{i}.self_attn.q_proj.bias` etc.
  - Use `weights.get(name)` (returns Option, no error if missing)
- [ ] 1c. After QKV sgemm in `GpuTransformerLayer::forward()`, add bias if present
  - Use existing `add_bias` kernel: `kernels/add_bias.cu`
  - Kernel: `add_bias_kernel(data, bias, num_tokens, dim)` -- adds bias[j] to data[i*dim + j]
  - Launch: grid=(num_tokens), block=(min(dim, 1024))
- [ ] 1d. Verify: `add_bias` kernel is in the KernelLoader function table
- [ ] 1e. Test: fresh server, 5-token prompt, 30 tokens generated -- should not degrade to `!`

### Files to edit
- `crates/rvllm-model-runner/src/gpu_layer.rs` (add bias fields, apply after projection)
- `crates/rvllm-model-runner/src/gpu_runner.rs` (fetch biases in layer_weights)

---

## Fix 2: Block allocator recycling

### Problem
`next_block_id` in `gpu_engine.rs` only increments. Freed blocks are never recycled. After request 1, its block IDs are orphaned. If `next_block_id >= num_gpu_blocks`, `reshape_and_cache` writes out of bounds.

### Steps
- [ ] 2a. Add `free_blocks: Vec<u32>` field to `GpuLLMEngine`
- [ ] 2b. In block cleanup (line ~591), push freed block IDs onto `free_blocks`
- [ ] 2c. In `build_metadata` block allocation (line ~640), pop from `free_blocks` before incrementing `next_block_id`
- [ ] 2d. Test: send 3 sequential requests, verify all produce coherent output

### Files to edit
- `crates/rvllm-engine/src/gpu_engine.rs`

---

## Fix 3: rms_norm_eps from config

### Problem
`gpu_runner.rs` hardcodes `rms_norm_eps: 1e-5`. Qwen2.5 uses `1e-6`. Not catastrophic but affects accuracy.

### Steps
- [ ] 3a. Add `rms_norm_eps: f32` to `ModelRunnerConfig` in `runner.rs`
- [ ] 3b. Set it from `WorkerConfig` in `config.rs` (needs field there too, or default 1e-6)
- [ ] 3c. Use `config.rms_norm_eps` instead of `1e-5_f32` in `gpu_runner.rs`
- [ ] 3d. Pass it to `GpuLayerConfig` for per-layer RMSNorm
- [ ] 3e. Update test configs with `rms_norm_eps: 1e-6`

### Files to edit
- `crates/rvllm-model-runner/src/runner.rs`
- `crates/rvllm-model-runner/src/gpu_runner.rs`
- `crates/rvllm-model-runner/src/gpu_layer.rs`
- `crates/rvllm-worker/src/config.rs`

---

## Fix 4: FA2 prefill causal mask

### Problem
`flash_attention.cu` prefill kernel causal mask: `kv_pos > (context_len - q_len + qi)`. For single-sequence prefill where `context_len == q_len`, this simplifies to `kv_pos > qi` which IS correct. But if `context_len > q_len` (decode-phase token appended to cache before prefill reads), the mask breaks. The naive CPU attention bypass works but is slow.

### Steps
- [ ] 4a. Verify the causal mask math for the single-sequence case
- [ ] 4b. If correct, the FA2 prefill bug is elsewhere (tile loading, output indexing)
- [ ] 4c. Once biases are fixed, re-test FA2 prefill (it may work now that layer outputs are correct)
- [ ] 4d. If still broken, write a standalone CUDA test for the FA2 prefill kernel

### Files to edit
- `kernels/flash_attention.cu` (only if mask is confirmed wrong)
- `crates/rvllm-model-runner/src/gpu_layer.rs` (switch back from naive to FA2 prefill)

---

## Fix 5: Remove debug probes and unnecessary syncs

### Steps
- [ ] 5a. Remove all `eprintln!` probes from `gpu_runner.rs`
- [ ] 5b. Remove `CALL_COUNT`, `DECODE_PROBED` statics
- [ ] 5c. Remove 4x `device.synchronize()` per layer in `gpu_layer.rs`
- [ ] 5d. Keep only the pre-LM-head sync in `gpu_runner.rs` (needed for stream mismatch)
- [ ] 5e. Test: verify output still correct after removing syncs

### Files to edit
- `crates/rvllm-model-runner/src/gpu_runner.rs`
- `crates/rvllm-model-runner/src/gpu_layer.rs`

---

## Test plan (after each fix)

```bash
# Fresh server start
ssh -p PORT root@HOST "kill -9 \$(pgrep rvllm) 2>/dev/null; sleep 2"
ssh -p PORT root@HOST "nohup /root/vllm-rs/target/release/rvllm serve --model Qwen/Qwen2.5-1.5B --port 8000 > /tmp/rvllm.log 2>&1 &"
sleep 20

# Test 1: 5-token prompt, 30 generated tokens (crosses block boundary)
curl -s -X POST http://HOST:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B","prompt":"The capital of France is","max_tokens":30,"temperature":0}'

# Test 2: Second request (cross-request contamination check)
curl -s -X POST http://HOST:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B","prompt":"Hello, how are you?","max_tokens":20,"temperature":0.7}'

# Test 3: Third request (block recycling check)
curl -s -X POST http://HOST:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B","prompt":"Write a Python function","max_tokens":30,"temperature":0}'
```

### Pass criteria
- Test 1: coherent text for all 30 tokens, no `!` degradation
- Test 2: coherent text, different from test 1
- Test 3: coherent text, no crash or garbage
