# Perplexity Triage — Layer 0 produces 30,000x residual blowup

## The bug
All models (Mistral-7B, Qwen3-8B, Llama 3.1 8B) produce garbage perplexity.
Throughput bench is unaffected (all-zero input hides the issue).

## Observed values (Mistral-7B, first token of WikiText-2)

```
                    rvLLM           PyTorch ref (f16)
embedding[0:4]      -0.0012, ...    -0.0044, ...      (different tokens, OK)
after L0[0:4]       -30.3, 65.3     -0.0008, 0.0012   << 30,000x too large
after L31[0:4]      -461, -320      -1.43, 1.01
logit range         [-140, 126]     [-11.4, 12.9]      << 10x too wide
perplexity          3.8e28          ~8 expected
```

The blowup happens in layer 0 itself — not compounding across layers.

## What's been eliminated (12-agent swarm)

| Component | Verdict | Notes |
|-----------|---------|-------|
| Embedding gather (f16) | Clean | BF16->f16 conversion correct, kernel correct |
| BF16 weight loading | Clean | bf16->f16->f32->FP8 chain verified |
| RMSNorm + FP8 quant kernel | Clean | Per-token scale math correct |
| QKV GEMM (cuBLASLt) | Clean | Scale swap is commutative for scalars, output ~1-10 |
| cuBLASLt descriptor config | Clean | alpha/beta/layout/transpose all verified |
| Weight scale convention | Clean | amax/448 matches cuBLASLt expectation |
| RoPE kernel | Clean | 16 params match deployed PTX, rotation preserves magnitude |
| FA3 FP8 attention | Clean | context_len=1 -> output = V_real exactly |
| quantize_fp8_per_token | Clean | Kernel math correct for dim=4096 |
| MLP path (steps 9-12) | Clean | RMSNorm normalizes, MLP adds O(1) |
| KV cache init | Clean | Zeroed to 0x00, FA3 masks OOB |
| cuBLASLt A/B scale math | Clean | Scalar multiply is commutative |

## What hasn't been eliminated

1. **Buffer aliasing / pointer wiring** — every component is individually correct,
   but are the right buffers connected between steps? Could scratch.q_out overlap
   with scratch.attn_out? Could the QKV output land in the wrong buffer?

2. **O proj residual add (step 8)** — fp8_gemm_residual does D = A*B^T + C.
   If the GEMM output is ~1 and residual is ~0.001, new residual should be ~1.
   But we see ~30. Could the residual pointer be wrong (pointing to a different
   buffer that already has large values)?

3. **Arena region ordering** — run_ppl allocates scratch regions sequentially.
   If a region is undersized, later regions overlap it. The QKV output buffer
   uses qkv_rows * max_tokens * 2 bytes — is qkv_rows computed correctly?

4. **The QKV pointer math** — q_base, k_base, v_base are offsets into the
   packed QKV buffer. For batch=1 these are correct by coincidence (single
   column in col-major = contiguous [Q|K|V]). But what if the buffer
   allocation is wrong?

## Next steps to try

### A. Add intermediate dumps after each sub-step of layer 0
Dump residual/hidden values after: rmsnorm, QKV GEMM, RoPE, FA3, quantize,
O proj, second rmsnorm, gate_up, silu, down proj. Find exactly which step
introduces the 30x factor.

### B. Run a single-layer PyTorch reference with exact same FP8 pipeline
Use torch to load the same weights, do rmsnorm -> fp8_quant -> cuBLASLt GEMM
with the same scale values, compare output.

### C. Check buffer sizes and arena layout
Print every arena.region allocation (name, ptr, size) and verify no overlaps.

### D. Check if the issue is model-specific
The embedding values differ between rvLLM and PyTorch ref (different tokens).
Verify rvLLM tokenizes the same way — maybe a tokenizer mismatch causes the
first token to be different, and the model behaves differently.

## Key insight from rmsnorm agent
The signs don't match between rvLLM and reference (ref: -0.1989, rvLLM: +31.2).
This rules out a uniform scale error. The computation is producing STRUCTURALLY
wrong results — wrong dot products, not just amplified correct ones.

Note: different tokenizers produce different token IDs, so the embeddings differ.
Can't directly compare signs. But magnitudes should be O(0.1-1.0) for any valid
token after layer 0, not O(30-65).

## Refined top suspects (post-swarm)

1. **cuBLASLt TN layout producing wrong dot products** — the A/B swap +
   column-major trick might compute the wrong matrix multiply. Would produce
   arbitrary wrong values with wrong signs. Agent 8 verified layout dims are
   correct, but the ACTUAL memory layout of the weight tensor might not match
   what cuBLASLt expects after the swap.

2. **Wrong weight data** — layer.qkv.offset_bytes points to wrong memory.
   The loader packs Q||K||V weights sequentially; if the offsets are wrong,
   the GEMM reads from the wrong weight block.

3. **FP8 weight encoding bug** — the manual fp8_e4m3_encode in fp8_quant.rs
   has hand-coded bit manipulation. If the exponent bias or mantissa rounding
   is wrong, all FP8 weight values are systematically corrupted.

## Pending agent results
- O proj + residual add (agent 4)
- weight scale values (agent 5)
- PyTorch reference intermediates (agent 9)
- cuBLASLt scale math deep dive (agent 10)
