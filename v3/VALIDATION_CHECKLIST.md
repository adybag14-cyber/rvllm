# PPL Validation Checklist

Two semantic bugs fixed, not yet validated on real weights.

## Fixes applied

1. **Gemma RMSNorm 1+gamma** -- loader pre-adds 1.0 to all norm weights
2. **Attention scale** -- `query_pre_attn_scalar ** -0.5` instead of `1.0`

## Step 1: Confirm config value

```bash
python3 -c "
import json, sys
cfg = json.load(open(sys.argv[1]))
tc = cfg.get('text_config', cfg)
print('query_pre_attn_scalar:', tc.get('query_pre_attn_scalar', 'MISSING'))
print('final_logit_softcapping:', tc.get('final_logit_softcapping', 'MISSING'))
" /path/to/model/config.json
```

Expected: `query_pre_attn_scalar: 256`

## Step 2: HF norm weight stats

```bash
python3 scripts/gemma4_hf_probes.py \
  --model-path /path/to/model \
  --norm-stats-only
```

Expected: norm weights centered near 0.0 (confirming 1+gamma is correct).
If centered near 1.0, the 1+gamma fix is wrong and should be reverted.

## Step 3: GPU full 60-layer PPL

```bash
RVLLM_DBG_LAYER=1 cargo run --release -p rvllm-ppl -- \
  --model-dir /path/to/model \
  --max-tokens 128
```

Log will print `attn_scale=0.062500 (query_pre_attn_scalar=256)`.
Target: PPL < 20 on reasonable text (Bible passage, Wikipedia, etc).

## Step 4: If PPL still bad -- single-token HF parity

```bash
python3 scripts/gemma4_hf_probes.py \
  --model-path /path/to/model \
  --token-id 2 \
  --probe-layers 0,1
```

Compare first4 values at each layer against GPU DBG_LAYER output.
Divergence at layer 0 = loader/norm/GEMM bug.
Divergence at layer 1+ = attention or accumulation bug.

## Step 5: If step 4 matches -- TPU parity

```bash
python3 tpu/harness/gemma4_tpu_infer.py \
  --model-dir /path/to/model \
  --perplexity --max-ctx 256
```

Both TPU and GPU now have the same two fixes.
PPL should be within ~10% of each other and both < 20.

## Not yet proven (defer unless steps 3-4 fail)

- FP8-Dynamic per-channel scale interpretation
- cuBLASLt OUTER_VEC_32F mode correctness
- SM89 global attention fallback (head_dim=512)
