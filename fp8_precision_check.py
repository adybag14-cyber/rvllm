import torch
from safetensors import safe_open
import json, os

model_dir = "/workspace/models/gemma4-31b-fp8/"

index = json.load(open(model_dir + "model.safetensors.index.json"))

# weight_scale tensors are in the shard files but not in the index
# Load from the same shard as the weight itself
def load_tensor(name):
    # Try index first, then scan shards
    if name in index["weight_map"]:
        shard = index["weight_map"][name]
        f = safe_open(model_dir + shard, framework="pt")
        return f.get_tensor(name)
    # For weight_scale, use the shard of the base weight
    base = name.replace("_scale", "")
    if base in index["weight_map"]:
        shard = index["weight_map"][base]
        f = safe_open(model_dir + shard, framework="pt")
        return f.get_tensor(name)
    raise KeyError(f"Cannot find {name}")

q_w = load_tensor("model.language_model.layers.0.self_attn.q_proj.weight")
q_s = load_tensor("model.language_model.layers.0.self_attn.q_proj.weight_scale")
k_w = load_tensor("model.language_model.layers.0.self_attn.k_proj.weight")
k_s = load_tensor("model.language_model.layers.0.self_attn.k_proj.weight_scale")
v_w = load_tensor("model.language_model.layers.0.self_attn.v_proj.weight")
v_s = load_tensor("model.language_model.layers.0.self_attn.v_proj.weight_scale")

print(f"q_w: {q_w.dtype} {q_w.shape}")
print(f"q_s: {q_s.dtype} {q_s.shape}")
print(f"k_w: {k_w.dtype} {k_w.shape}")
print(f"k_s: {k_s.dtype} {k_s.shape}")
print(f"v_w: {v_w.dtype} {v_w.shape}")
print(f"v_s: {v_s.dtype} {v_s.shape}")

print(f"\nq_s range: [{q_s.min().item():.6e}, {q_s.max().item():.6e}]")
print(f"k_s range: [{k_s.min().item():.6e}, {k_s.max().item():.6e}]")
print(f"v_s range: [{v_s.min().item():.6e}, {v_s.max().item():.6e}]")

# Global max across Q, K, V scales
all_scales = torch.cat([q_s.float().flatten(), k_s.float().flatten(), v_s.float().flatten()])
global_max = all_scales.max().item()
print(f"\nglobal_max = {global_max:.6e}")

# Histogram of ratios
q_s_f = q_s.float().flatten()
k_s_f = k_s.float().flatten()
v_s_f = v_s.float().flatten()
all_ratios = all_scales / global_max
print(f"ratio range: [{all_ratios.min().item():.6f}, {all_ratios.max().item():.6f}]")
print(f"ratio < 0.1: {(all_ratios < 0.1).sum().item()} / {all_ratios.numel()}")
print(f"ratio < 0.01: {(all_ratios < 0.01).sum().item()} / {all_ratios.numel()}")
print(f"ratio median: {all_ratios.median().item():.6f}")

# For each Q row, compute ratio and precision loss
print("\n=== Per-row analysis (Q proj, first 10 rows) ===")
for row in range(min(10, q_w.shape[0])):
    ratio = q_s_f[row].item() / global_max
    # Correct dequant: fp8_val * per_channel_scale
    correct = q_w[row].float() * q_s_f[row].item()
    # Rescaled: fp8_val * ratio -> re-encode to fp8 -> decode * global_max
    rescaled_f32 = q_w[row].float() * ratio
    rescaled_fp8 = rescaled_f32.to(torch.float8_e4m3fn)
    rescaled_dequant = rescaled_fp8.float() * global_max

    # Compare
    err = (rescaled_dequant - correct).abs()
    nonzero_mask = correct.abs() > 1e-10
    rel_err = err[nonzero_mask] / correct[nonzero_mask].abs()

    zeros_orig = (q_w[row].float() == 0).sum().item()
    zeros_rescaled = (rescaled_fp8 == 0).sum().item()

    print(f"\nRow {row}: scale={q_s_f[row].item():.6e}, ratio={ratio:.6f}")
    print(f"  correct[:6]  = {[f'{x:.6f}' for x in correct[:6].tolist()]}")
    print(f"  rescaled[:6] = {[f'{x:.6f}' for x in rescaled_dequant[:6].tolist()]}")
    if rel_err.numel() > 0:
        print(f"  max_rel_err  = {rel_err.max().item():.6f}")
        print(f"  mean_rel_err = {rel_err.mean().item():.6f}")
        print(f"  p99_rel_err  = {rel_err.quantile(0.99).item():.6f}")
    print(f"  zeros: orig={zeros_orig}, rescaled={zeros_rescaled}, flushed={zeros_rescaled - zeros_orig} / {q_w.shape[1]}")

# Full matrix stats
print("\n=== Full Q matrix rescaling stats ===")
total_flushed = 0
total_elements = 0
all_rel_errs = []
for row in range(q_w.shape[0]):
    ratio = q_s_f[row].item() / global_max
    correct = q_w[row].float() * q_s_f[row].item()
    rescaled_f32 = q_w[row].float() * ratio
    rescaled_fp8 = rescaled_f32.to(torch.float8_e4m3fn)
    rescaled_dequant = rescaled_fp8.float() * global_max

    zeros_orig = (q_w[row].float() == 0).sum().item()
    zeros_rescaled = (rescaled_fp8 == 0).sum().item()
    total_flushed += (zeros_rescaled - zeros_orig)
    total_elements += q_w.shape[1]

    err = (rescaled_dequant - correct).abs()
    nonzero_mask = correct.abs() > 1e-10
    if nonzero_mask.sum() > 0:
        rel = err[nonzero_mask] / correct[nonzero_mask].abs()
        all_rel_errs.append(rel)

all_rel = torch.cat(all_rel_errs)
print(f"Total elements: {total_elements}")
print(f"Total flushed to zero: {total_flushed} ({100*total_flushed/total_elements:.4f}%)")
print(f"Rel error - mean: {all_rel.mean().item():.6f}")

# Use sorted sample for percentiles to avoid memory issues
sample_idx = torch.randperm(all_rel.numel())[:min(1000000, all_rel.numel())]
sample = all_rel[sample_idx].sort().values
n = sample.numel()
print(f"Rel error - median (sampled): {sample[n//2].item():.6f}")
print(f"Rel error - p95 (sampled): {sample[int(n*0.95)].item():.6f}")
print(f"Rel error - p99 (sampled): {sample[int(n*0.99)].item():.6f}")
print(f"Rel error - max: {all_rel.max().item():.6f}")
print(f"Rel error > 10%: {(all_rel > 0.1).sum().item()} ({100*(all_rel > 0.1).float().mean().item():.4f}%)")
print(f"Rel error > 50%: {(all_rel > 0.5).sum().item()} ({100*(all_rel > 0.5).float().mean().item():.4f}%)")
print(f"Rel error > 100%: {(all_rel > 1.0).sum().item()} ({100*(all_rel > 1.0).float().mean().item():.4f}%)")

# Also check: what is the worst-case row (lowest ratio)?
worst_row = q_s_f.argmin().item()
worst_ratio = q_s_f[worst_row].item() / global_max
print(f"\nWorst row: {worst_row}, scale={q_s_f[worst_row].item():.6e}, ratio={worst_ratio:.6f}")
