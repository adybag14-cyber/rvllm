#!/usr/bin/env python3
"""Gemma 4 31B inference on TPU v6e-4 via JAX SPMD (TP=4).

60 layers = 10 groups of (5 sliding + 1 global).
Nested scan: outer over 10 groups, inner over 5 sliding layers per group.
Global layer called explicitly between inner scans.

Weight shapes differ between sliding and global:
  Sliding: q=[8192,5376] k=[4096,5376] v=[4096,5376] o=[5376,8192] qn/kn=[256]
  Global:  q=[16384,5376] k=[2048,5376] v=MISSING     o=[5376,16384] qn/kn=[512]
  FFN: identical for all layers

Usage:
    python3 gemma4_tpu_infer.py --model-dir /path/to/gemma-4-31B-it \
        --max-tokens 32 --prompt "2,3257"
"""
import argparse, json, os, struct, sys, time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# ── constants ──
H      = 5376
NH     = 32
INTER  = 21504
VOCAB  = 262144
NL     = 60
N_GROUPS = 10
WINDOW = 1024
SOFTCAP_VAL = 30.0
EPS    = 1e-6
B      = 1

# sliding attention
S_Q_DIM  = 8192   # 32 * 256
S_KV_DIM = 4096   # 16 * 256
S_HD     = 256
S_KVH    = 16
S_GQA    = NH // S_KVH  # 2

# global attention
G_Q_DIM  = 16384  # 32 * 512
G_KV_DIM = 2048   # 4 * 512
G_HD     = 512
G_KVH    = 4
G_GQA    = NH // G_KVH  # 8

# ── mesh ──
def make_mesh():
    devs = jax.devices()
    assert len(devs) >= 4, f"Need 4 TPU chips, got {len(devs)}"
    return Mesh(np.array(devs[:4]), ('tp',))

# ── primitives ──
def rms_norm(x, g):
    x32 = x.astype(jnp.float32)
    return (x * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + EPS).astype(x.dtype)) * g

def head_norm(h, g):
    h32 = h.astype(jnp.float32)
    return (h * jax.lax.rsqrt(jnp.mean(h32 * h32, axis=-1, keepdims=True) + EPS).astype(h.dtype)) * g

def rope(x, cos, sin, rot_dim):
    half = rot_dim // 2
    xr, xp = x[..., :rot_dim], x[..., rot_dim:]
    x0, x1 = xr[..., :half], xr[..., half:]
    rotated = jnp.concatenate([
        x0 * cos.astype(x.dtype) - x1 * sin.astype(x.dtype),
        x0 * sin.astype(x.dtype) + x1 * cos.astype(x.dtype),
    ], axis=-1)
    return jnp.concatenate([rotated, xp], axis=-1)

def precompute_rope(theta, rot_dim, max_pos):
    half = rot_dim // 2
    freqs = 1.0 / (theta ** (np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim))
    angles = np.outer(np.arange(max_pos, dtype=np.float32), freqs)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)

def decode_attn(q, k, v, k_cache, v_cache, pos, ctx, scale, gqa_ratio, max_ctx, window=-1):
    """Generic decode attention. q:(B,NH,HD) k:(B,KVH,HD) v:(B,KVH,HD)"""
    kvh = k.shape[1]
    hd = k.shape[2]
    kv_dim = kvh * hd

    k_cache = k_cache.at[pos].set(k.reshape(B, kv_dim)[0])
    v_cache = v_cache.at[pos].set(v.reshape(B, kv_dim)[0])

    k_ctx = jnp.repeat(k_cache.reshape(max_ctx, kvh, hd), gqa_ratio, axis=1)
    v_ctx = jnp.repeat(v_cache.reshape(max_ctx, kvh, hd), gqa_ratio, axis=1)

    sc = jnp.einsum('bnh,tnh->bnt', q.astype(jnp.float32),
                     k_ctx.astype(jnp.float32)) * scale
    t = jnp.arange(max_ctx)
    if window > 0:
        valid = (t < ctx) & (t >= jnp.maximum(0, pos - window + 1))
    else:
        valid = t < ctx
    sc = jnp.where(valid[None, None, :], sc, jnp.float32(-1e9))
    p = jax.nn.softmax(sc, axis=-1).astype(q.dtype)
    out = jnp.einsum('bnt,tnh->bnh', p, v_ctx)
    return out, k_cache, v_cache

# ── FFN (shared by both layer types) ──
def ffn_block(x, ln3, ln4, gw, uw, dw, ls):
    residual = x
    h = rms_norm(x, ln3)
    gate = h @ gw.T
    up = h @ uw.T
    h = jax.nn.gelu(gate, approximate=True) * up
    h = h @ dw.T
    h = rms_norm(h, ln4)
    return residual + h * ls

# ── sliding layer ──
def sliding_layer(carry, xs):
    x, pos, ctx, cos_s, sin_s = carry
    max_ctx = xs['skc'].shape[0]

    residual = x
    h = rms_norm(x, xs['ln1'])
    q = (h @ xs['sqw'].T).reshape(B, NH, S_HD)
    k = (h @ xs['skw'].T).reshape(B, S_KVH, S_HD)
    v = (h @ xs['svw'].T).reshape(B, S_KVH, S_HD)

    q = head_norm(q, xs['sqn'])
    k = head_norm(k, xs['skn'])

    c = cos_s[pos][None, None, :]
    s = sin_s[pos][None, None, :]
    q = rope(q, c, s, S_HD)
    k = rope(k, c, s, S_HD)

    scale = 1.0 / jnp.sqrt(jnp.float32(S_HD))
    attn_out, skc, svc = decode_attn(q, k, v, xs['skc'], xs['svc'], pos, ctx,
                                      scale, S_GQA, max_ctx, window=WINDOW)
    h = attn_out.reshape(B, S_Q_DIM) @ xs['sow'].T
    h = rms_norm(h, xs['ln2'])
    x = residual + h * xs['ls']

    x = ffn_block(x, xs['ln3'], xs['ln4'], xs['gw'], xs['uw'], xs['dw'], xs['ls'])
    return (x, pos, ctx, cos_s, sin_s), {'skc': skc, 'svc': svc}

# ── global layer ──
def global_layer_fn(carry, gp):
    x, pos, ctx, cos_g, sin_g = carry
    max_ctx = gp['gkc'].shape[0]

    residual = x
    h = rms_norm(x, gp['ln1'])
    q = (h @ gp['gqw'].T).reshape(B, NH, G_HD)
    k = (h @ gp['gkw'].T).reshape(B, G_KVH, G_HD)
    v = k  # k_eq_v

    q = head_norm(q, gp['gqn'])
    k = head_norm(k, gp['gkn'])

    c = cos_g[pos][None, None, :]
    s = sin_g[pos][None, None, :]
    rot_dim = 128  # 512 * 0.25
    q = rope(q, c, s, rot_dim)
    k = rope(k, c, s, rot_dim)

    scale = 1.0 / jnp.sqrt(jnp.float32(G_HD))
    attn_out, gkc, gvc = decode_attn(q, k, v, gp['gkc'], gp['gvc'], pos, ctx,
                                      scale, G_GQA, max_ctx, window=-1)
    h = attn_out.reshape(B, G_Q_DIM) @ gp['gow'].T
    h = rms_norm(h, gp['ln2'])
    x = residual + h * gp['ls']

    x = ffn_block(x, gp['ln3'], gp['ln4'], gp['gw'], gp['uw'], gp['dw'], gp['ls'])
    return (x, pos, ctx, cos_g, sin_g), {'gkc': gkc, 'gvc': gvc}

# ── one group (5 sliding + 1 global) ──
def one_group(carry, group_xs):
    x, pos, ctx, cos_s, sin_s, cos_g, sin_g = carry

    s_carry = (x, pos, ctx, cos_s, sin_s)
    s_carry, s_out = jax.lax.scan(sliding_layer, s_carry, group_xs['sliding'])
    x = s_carry[0]

    g_carry = (x, pos, ctx, cos_g, sin_g)
    g_carry, g_out = global_layer_fn(g_carry, group_xs['global'])
    x = g_carry[0]

    return (x, pos, ctx, cos_s, sin_s, cos_g, sin_g), {'s': s_out, 'g': g_out}

# ── forward ──
def forward(token_id, pos, ctx, embed, final_norm, groups, cos_s, sin_s, cos_g, sin_g):
    x = embed[token_id].reshape(B, H) * jnp.sqrt(jnp.float32(H))
    init = (x, pos, ctx, cos_s, sin_s, cos_g, sin_g)

    final, scan_out = jax.lax.scan(one_group, init, groups)
    x = final[0]
    x = rms_norm(x, final_norm)
    logits = x.astype(jnp.float32) @ embed.astype(jnp.float32).T
    logits = SOFTCAP_VAL * jnp.tanh(logits / SOFTCAP_VAL)
    return jnp.argmax(logits, axis=-1).astype(jnp.int32), scan_out

# ── safetensors reader ──
def read_safetensors(path):
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        data = np.memmap(path, dtype=np.uint8, mode='r', offset=data_start)
    tensors = {}
    for name, info in header.items():
        if name == '__metadata__':
            continue
        shape = tuple(info['shape'])
        dtype_str = info['dtype']
        start, end = info['data_offsets']
        raw = np.array(data[start:end])
        if dtype_str in ('BF16', 'bf16', 'bfloat16'):
            tensors[name] = raw.view(np.uint16).reshape(shape)
        elif dtype_str in ('F16', 'f16', 'float16'):
            tensors[name] = raw.view(np.float16).reshape(shape)
        elif dtype_str in ('F32', 'f32', 'float32'):
            tensors[name] = raw.view(np.float32).reshape(shape)
    return tensors

import ml_dtypes

def to_np_bf16(arr):
    if arr.dtype == np.uint16:
        return arr.view(ml_dtypes.bfloat16)
    if arr.dtype == np.float16:
        return arr.astype(np.float32).astype(ml_dtypes.bfloat16)
    if arr.dtype == np.float32:
        return arr.astype(ml_dtypes.bfloat16)
    if arr.dtype == ml_dtypes.bfloat16:
        return arr
    raise ValueError(f"unsupported dtype {arr.dtype}")

# ── weight loading ──
def load_model(model_dir, mesh, max_ctx):
    idx_path = os.path.join(model_dir, 'model.safetensors.index.json')
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            index = json.load(f)
        weight_map = index['weight_map']
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = ['model.safetensors']
        weight_map = None

    prefix = 'model'
    if weight_map:
        for k in weight_map:
            if k.startswith('model.language_model.'):
                prefix = 'model.language_model'
                break

    print(f"loading from {model_dir}, prefix={prefix}, {len(shard_names)} shard(s)", file=sys.stderr)
    all_t = {}
    for sn in shard_names:
        print(f"  reading {sn}...", file=sys.stderr)
        all_t.update(read_safetensors(os.path.join(model_dir, sn)))
    print(f"  {len(all_t)} tensors", file=sys.stderr)

    def get(name):
        return all_t[name]

    def put(arr, spec):
        return jax.device_put(to_np_bf16(arr), NamedSharding(mesh, spec))

    embed = put(get(f'{prefix}.embed_tokens.weight'), P(None, None))
    final_norm = put(get(f'{prefix}.norm.weight'), P(None))

    # Layer indices: 0-4=sliding, 5=global, 6-10=sliding, 11=global, ...
    sliding_indices = [i for i in range(NL) if (i + 1) % 6 != 0]  # 50 layers
    global_indices  = [i for i in range(NL) if (i + 1) % 6 == 0]  # 10 layers
    assert len(sliding_indices) == 50 and len(global_indices) == 10

    # Stack sliding weights: [10, 5, ...]
    print("  stacking sliding layers...", file=sys.stderr)
    s_keys = ['sqw','skw','svw','sow','sqn','skn','ln1','ln2','ln3','ln4','gw','uw','dw','ls']
    s_stacked = {k: [] for k in s_keys}
    for group in range(N_GROUPS):
        group_layers = []
        for j in range(5):
            layer_idx = sliding_indices[group * 5 + j]
            lp = f'{prefix}.layers.{layer_idx}'
            group_layers.append({
                'sqw': get(f'{lp}.self_attn.q_proj.weight'),
                'skw': get(f'{lp}.self_attn.k_proj.weight'),
                'svw': get(f'{lp}.self_attn.v_proj.weight'),
                'sow': get(f'{lp}.self_attn.o_proj.weight'),
                'sqn': get(f'{lp}.self_attn.q_norm.weight'),
                'skn': get(f'{lp}.self_attn.k_norm.weight'),
                'ln1': get(f'{lp}.input_layernorm.weight'),
                'ln2': get(f'{lp}.post_attention_layernorm.weight'),
                'ln3': get(f'{lp}.pre_feedforward_layernorm.weight'),
                'ln4': get(f'{lp}.post_feedforward_layernorm.weight'),
                'gw':  get(f'{lp}.mlp.gate_proj.weight'),
                'uw':  get(f'{lp}.mlp.up_proj.weight'),
                'dw':  get(f'{lp}.mlp.down_proj.weight'),
                'ls':  get(f'{lp}.layer_scalar'),
            })
        for k in s_keys:
            s_stacked[k].append(np.stack([gl[k] for gl in group_layers]))
        if group % 3 == 0:
            print(f"    sliding group {group}", file=sys.stderr)

    # Stack to [10, 5, ...] and shard
    s_sharding = {
        'sqw': P(None, None, 'tp', None),  # [10,5,S_Q_DIM,H]
        'skw': P(None, None, 'tp', None),
        'svw': P(None, None, 'tp', None),
        'sow': P(None, None, None, 'tp'),
        'sqn': P(None, None, None),
        'skn': P(None, None, None),
        'ln1': P(None, None, None),
        'ln2': P(None, None, None),
        'ln3': P(None, None, None),
        'ln4': P(None, None, None),
        'gw':  P(None, None, 'tp', None),
        'uw':  P(None, None, 'tp', None),
        'dw':  P(None, None, None, 'tp'),
        'ls':  P(None, None, None),
    }
    sliding_xs = {}
    for k in s_keys:
        arr = np.stack(s_stacked[k])
        sliding_xs[k] = put(arr, s_sharding[k])
        print(f"    s/{k}: {arr.shape}", file=sys.stderr)

    # Sliding KV caches: [10, 5, max_ctx, S_KV_DIM]
    zero_skc = jnp.zeros((N_GROUPS, 5, max_ctx, S_KV_DIM), dtype=jnp.bfloat16)
    zero_svc = jnp.zeros((N_GROUPS, 5, max_ctx, S_KV_DIM), dtype=jnp.bfloat16)
    sliding_xs['skc'] = jax.device_put(zero_skc, NamedSharding(mesh, P(None, None, None, 'tp')))
    sliding_xs['svc'] = jax.device_put(zero_svc, NamedSharding(mesh, P(None, None, None, 'tp')))

    # Stack global weights: [10, ...]
    print("  stacking global layers...", file=sys.stderr)
    g_keys = ['gqw','gkw','gow','gqn','gkn','ln1','ln2','ln3','ln4','gw','uw','dw','ls']
    g_stacked = {k: [] for k in g_keys}
    for gi, layer_idx in enumerate(global_indices):
        lp = f'{prefix}.layers.{layer_idx}'
        g_stacked['gqw'].append(get(f'{lp}.self_attn.q_proj.weight'))
        g_stacked['gkw'].append(get(f'{lp}.self_attn.k_proj.weight'))
        g_stacked['gow'].append(get(f'{lp}.self_attn.o_proj.weight'))
        g_stacked['gqn'].append(get(f'{lp}.self_attn.q_norm.weight'))
        g_stacked['gkn'].append(get(f'{lp}.self_attn.k_norm.weight'))
        g_stacked['ln1'].append(get(f'{lp}.input_layernorm.weight'))
        g_stacked['ln2'].append(get(f'{lp}.post_attention_layernorm.weight'))
        g_stacked['ln3'].append(get(f'{lp}.pre_feedforward_layernorm.weight'))
        g_stacked['ln4'].append(get(f'{lp}.post_feedforward_layernorm.weight'))
        g_stacked['gw'].append(get(f'{lp}.mlp.gate_proj.weight'))
        g_stacked['uw'].append(get(f'{lp}.mlp.up_proj.weight'))
        g_stacked['dw'].append(get(f'{lp}.mlp.down_proj.weight'))
        g_stacked['ls'].append(get(f'{lp}.layer_scalar'))

    g_sharding = {
        'gqw': P(None, 'tp', None),
        'gkw': P(None, 'tp', None),
        'gow': P(None, None, 'tp'),
        'gqn': P(None, None),
        'gkn': P(None, None),
        'ln1': P(None, None),
        'ln2': P(None, None),
        'ln3': P(None, None),
        'ln4': P(None, None),
        'gw':  P(None, 'tp', None),
        'uw':  P(None, 'tp', None),
        'dw':  P(None, None, 'tp'),
        'ls':  P(None, None),
    }
    global_xs = {}
    for k in g_keys:
        arr = np.stack(g_stacked[k])
        global_xs[k] = put(arr, g_sharding[k])
        print(f"    g/{k}: {arr.shape}", file=sys.stderr)

    # Global KV caches: [10, max_ctx, G_KV_DIM]
    zero_gkc = jnp.zeros((N_GROUPS, max_ctx, G_KV_DIM), dtype=jnp.bfloat16)
    zero_gvc = jnp.zeros((N_GROUPS, max_ctx, G_KV_DIM), dtype=jnp.bfloat16)
    global_xs['gkc'] = jax.device_put(zero_gkc, NamedSharding(mesh, P(None, None, 'tp')))
    global_xs['gvc'] = jax.device_put(zero_gvc, NamedSharding(mesh, P(None, None, 'tp')))

    # Combine into groups pytree: {'sliding': {...}, 'global': {...}}
    groups = {'sliding': sliding_xs, 'global': global_xs}

    del all_t, s_stacked, g_stacked
    print("  done loading", file=sys.stderr)
    return embed, final_norm, groups

# ── main ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--max-tokens', type=int, default=32)
    parser.add_argument('--max-ctx', type=int, default=2048)
    parser.add_argument('--prompt', default='2')
    args = parser.parse_args()

    mesh = make_mesh()
    print(f"mesh: {mesh}", file=sys.stderr)

    max_ctx = args.max_ctx
    embed, final_norm, groups = load_model(args.model_dir, mesh, max_ctx)

    cos_s, sin_s = precompute_rope(10000.0, S_HD, max_ctx)   # (max_ctx, 128)
    cos_g, sin_g = precompute_rope(1000000.0, 128, max_ctx)   # (max_ctx, 64) partial rot
    cos_s = jax.device_put(jnp.array(cos_s), NamedSharding(mesh, P(None, None)))
    sin_s = jax.device_put(jnp.array(sin_s), NamedSharding(mesh, P(None, None)))
    cos_g = jax.device_put(jnp.array(cos_g), NamedSharding(mesh, P(None, None)))
    sin_g = jax.device_put(jnp.array(sin_g), NamedSharding(mesh, P(None, None)))

    prompt_ids = [int(x.strip()) for x in args.prompt.split(',')]
    print(f"prompt: {len(prompt_ids)} tokens {prompt_ids[:10]}", file=sys.stderr)

    fwd_jit = jax.jit(forward)

    generated = []
    last_sampled = None
    total_steps = len(prompt_ids) + args.max_tokens
    t_start = time.time()
    ttft = None

    for step in range(total_steps):
        if step < len(prompt_ids):
            token_id = prompt_ids[step]
        else:
            token_id = last_sampled
        pos = jnp.int32(step)
        ctx = jnp.int32(step + 1)
        tok_arr = jnp.array([token_id], dtype=jnp.int32)

        t0 = time.time()
        next_tok, scan_out = fwd_jit(
            tok_arr, pos, ctx, embed, final_norm, groups, cos_s, sin_s, cos_g, sin_g)
        next_tok.block_until_ready()
        dt = time.time() - t0

        # Update KV caches from scan output
        groups = {
            'sliding': {**groups['sliding'],
                        'skc': scan_out['s']['skc'],
                        'svc': scan_out['s']['svc']},
            'global':  {**groups['global'],
                        'gkc': scan_out['g']['gkc'],
                        'gvc': scan_out['g']['gvc']},
        }

        sampled = int(next_tok[0])
        last_sampled = sampled

        if step < len(prompt_ids):
            print('.', end='', file=sys.stderr, flush=True)
            if step == len(prompt_ids) - 1:
                ttft = time.time() - t_start
                print(f"\nTTFT: {ttft*1000:.1f}ms ({len(prompt_ids)} prompt tokens)", file=sys.stderr)
        else:
            generated.append(sampled)
            if sampled in (1, 2, 107):
                print(f"\n[EOS tok={sampled} at step {step}]", file=sys.stderr)
                break
            print(f"[{sampled}]", end='', file=sys.stderr, flush=True)
            if step < len(prompt_ids) + 3:
                print(f" ({dt*1000:.1f}ms)", end='', file=sys.stderr, flush=True)

    total = time.time() - t_start
    print(file=sys.stderr)
    print("=== Results ===", file=sys.stderr)
    print(f"prompt tokens:    {len(prompt_ids)}", file=sys.stderr)
    print(f"generated tokens: {len(generated)}", file=sys.stderr)
    if ttft:
        print(f"TTFT:             {ttft*1000:.1f}ms", file=sys.stderr)
    if len(generated) > 1 and ttft:
        decode_time = total - ttft
        tps = len(generated) / decode_time
        print(f"decode tok/s:     {tps:.1f}", file=sys.stderr)
    print(f"total time:       {total:.1f}s", file=sys.stderr)
    print(f"generated:        {generated[:20]}", file=sys.stderr)

if __name__ == '__main__':
    main()
