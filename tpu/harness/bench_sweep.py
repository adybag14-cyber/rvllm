#!/usr/bin/env python3
"""Batch sweep benchmark with 3 separate clocks: compile, TTFT, decode throughput.

Usage:
    LIBTPU_INIT_ARGS="--xla_tpu_enable_async_collective_fusion=true \
      --xla_tpu_dot_dot_fusion_duplicated=true" \
    python3 bench_sweep.py --model-dir ~/models/gemma-4-31B-it
"""
import argparse, sys, time, os
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gemma4_tpu_infer import (
    load_model, make_mesh, precompute_rope, forward_step, make_decode_loop,
    H, S_HD, NL, WINDOW, BLOCK_K, N_SLIDING, N_GLOBAL,
    S_KV, G_KV, S_KVH, G_KVH, MAX_KV, _sharded_zeros,
)
from jax.sharding import NamedSharding, PartitionSpec as P
import ml_dtypes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--max-ctx', type=int, default=2048)
    parser.add_argument('--decode-tokens', type=int, default=64)
    parser.add_argument('--batches', default='1,2,4,8,16,32,48,64,96,128,256,512,768')
    args = parser.parse_args()

    batches = [int(b) for b in args.batches.split(',')]
    mesh = make_mesh()
    max_ctx = args.max_ctx
    if max_ctx % BLOCK_K != 0:
        max_ctx = ((max_ctx + BLOCK_K - 1) // BLOCK_K) * BLOCK_K

    print("loading model...", file=sys.stderr)
    embed, final_norm, sl_w, gl_w, sl_c, gl_c = load_model(args.model_dir, mesh, max_ctx)

    cos_s_np, sin_s_np = precompute_rope(10000.0, S_HD, max_ctx)
    cos_g_np, sin_g_np = precompute_rope(1000000.0, 128, max_ctx)
    rsh = NamedSharding(mesh, P(None, None))
    cos_s = jax.device_put(jnp.array(cos_s_np), rsh)
    sin_s = jax.device_put(jnp.array(sin_s_np), rsh)
    cos_g = jax.device_put(jnp.array(cos_g_np), rsh)
    sin_g = jax.device_put(jnp.array(sin_g_np), rsh)

    print("", file=sys.stderr)
    print("=== rvLLM TPU Sweep: Gemma 4 31B, int8, split-cache ===")
    print("%6s %10s %10s %10s %10s" % ("batch", "compile_s", "ttft_ms", "decode_tps", "ms/step"))
    sys.stdout.flush()

    for B in batches:
        # Set batch size
        import gemma4_tpu_infer
        gemma4_tpu_infer.B = B

        num_prompt = 2
        num_decode = args.decode_tokens
        total = num_prompt + num_decode

        loop_fn = make_decode_loop(num_prompt, total)
        prompt_arr = jnp.array([2, 3257] + [0] * (total - num_prompt), dtype=jnp.int32)

        # Fresh caches
        kv_sh = NamedSharding(mesh, P(None, None, 'tp'))
        kvs_sh = NamedSharding(mesh, P(None, None, None))
        sl_c = {
            'kc': _sharded_zeros((N_SLIDING, WINDOW, S_KV), np.int8, kv_sh),
            'vc': _sharded_zeros((N_SLIDING, WINDOW, S_KV), np.int8, kv_sh),
            'kc_s': _sharded_zeros((N_SLIDING, WINDOW, S_KVH), ml_dtypes.bfloat16, kvs_sh),
            'vc_s': _sharded_zeros((N_SLIDING, WINDOW, S_KVH), ml_dtypes.bfloat16, kvs_sh),
        }
        gl_c = {
            'kc': _sharded_zeros((N_GLOBAL, max_ctx, G_KV), np.int8, kv_sh),
            'vc': _sharded_zeros((N_GLOBAL, max_ctx, G_KV), np.int8, kv_sh),
            'kc_s': _sharded_zeros((N_GLOBAL, max_ctx, G_KVH), ml_dtypes.bfloat16, kvs_sh),
            'vc_s': _sharded_zeros((N_GLOBAL, max_ctx, G_KVH), ml_dtypes.bfloat16, kvs_sh),
        }

        try:
            fused_jit = jax.jit(loop_fn, donate_argnums=(4, 5))

            # Clock 1: Compile (first run)
            t_compile_start = time.time()
            gen = fused_jit(prompt_arr, embed, final_norm, sl_w, gl_w, sl_c, gl_c,
                            cos_s, sin_s, cos_g, sin_g)
            gen.block_until_ready()
            compile_time = time.time() - t_compile_start

            # Clock 2+3: Pure run (cached compile) -- TTFT + decode
            sl_c2 = {
                'kc': _sharded_zeros((N_SLIDING, WINDOW, S_KV), np.int8, kv_sh),
                'vc': _sharded_zeros((N_SLIDING, WINDOW, S_KV), np.int8, kv_sh),
                'kc_s': _sharded_zeros((N_SLIDING, WINDOW, S_KVH), ml_dtypes.bfloat16, kvs_sh),
                'vc_s': _sharded_zeros((N_SLIDING, WINDOW, S_KVH), ml_dtypes.bfloat16, kvs_sh),
            }
            gl_c2 = {
                'kc': _sharded_zeros((N_GLOBAL, max_ctx, G_KV), np.int8, kv_sh),
                'vc': _sharded_zeros((N_GLOBAL, max_ctx, G_KV), np.int8, kv_sh),
                'kc_s': _sharded_zeros((N_GLOBAL, max_ctx, G_KVH), ml_dtypes.bfloat16, kvs_sh),
                'vc_s': _sharded_zeros((N_GLOBAL, max_ctx, G_KVH), ml_dtypes.bfloat16, kvs_sh),
            }

            t_pure_start = time.time()
            gen2 = fused_jit(prompt_arr, embed, final_norm, sl_w, gl_w, sl_c2, gl_c2,
                             cos_s, sin_s, cos_g, sin_g)
            gen2.block_until_ready()
            pure_time = time.time() - t_pure_start

            # TTFT = time for prompt steps / total * pure_time (approximate)
            ttft_ms = pure_time / total * num_prompt * 1000
            # Decode throughput = decode tokens * batch / decode time
            decode_time = pure_time - (ttft_ms / 1000)
            decode_tps = num_decode * B / decode_time if decode_time > 0 else 0
            ms_step = pure_time / total * 1000

            print("%6d %10.1f %10.1f %10.1f %10.2f" % (
                B, compile_time, ttft_ms, decode_tps, ms_step))

        except Exception as e:
            print("%6d     ERROR: %s" % (B, str(e)[:60]))

        sys.stdout.flush()

    print("=== DONE ===")

if __name__ == '__main__':
    main()
