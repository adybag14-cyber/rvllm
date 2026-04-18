# Spec 22: Variable head_dim support (128/256/512) for Gemma 4

## Problem

rvLLM v3 hardcodes `head_dim == 128` in 5 enforcement points. Gemma 4 31B
uses head_dim=256 for sliding attention layers and head_dim=512 for global
attention layers. The current code rejects anything != 128 at load time,
attention param validation, and FA3 .so load.

The Gemma4 bringup code (`gemma4_bring_up.rs`) already passes
`arch.head_dim_sliding` (256) to `Fa3Kernels::load()`, but that call
immediately fails on the `head_dim != 128` gate in `lib.rs:146`.

## Current enforcement points

| # | File | Line | Gate |
|---|------|------|------|
| 1 | `rvllm-loader/src/load.rs` | 92 | `if head_dim != 128 { return Err }` in `ModelArch::from_dir` |
| 2 | `rvllm-attention/src/decode.rs` | 31 | `if self.head_dim != 128` in `PagedDecodeParams::validate` |
| 3 | `rvllm-attention/src/prefill.rs` | 30 | `if self.head_dim != 128` in `PagedPrefillParams::validate` |
| 4 | `rvllm-attention/src/lib.rs` | 146 | `if head_dim != 128` in `Fa3Kernels::load` |
| 5 | `rvllm-core/src/error.rs` | 154 | `UnsupportedHeadDim { got: u32, required: u32 }` -- type itself is fine, `required` field becomes misleading |

## Solution overview

1. Replace all `== 128` gates with an allowed-set check: `{128, 256, 512}`
2. Recompile FA3 SM90 .so with head_dim=256 and head_dim=512 instantiations
3. The SM89 fallback kernel already handles 128/256/512 (no changes needed)
4. The RoPE kernels already handle variable head_dim via runtime params (no changes needed)
5. The Gemma4 forward path passes head_dim per-layer, so the dual-dim dispatch already works

---

## File changes

### 1. `v3/crates/rvllm-core/src/error.rs` -- error type cleanup

Change `UnsupportedHeadDim` to carry the allowed set instead of a single required value.

```rust
// OLD
UnsupportedHeadDim { got: u32, required: u32 },

// NEW
UnsupportedHeadDim { got: u32 },
```

Display impl: `"unsupported head_dim={got}, must be one of [128, 256, 512]"`.

The allowed set is a policy in the validation code, not the error type. The
error just says "you gave me X, it was wrong."

### 2. `v3/crates/rvllm-loader/src/load.rs` -- ModelArch loader

```rust
// OLD (line 92-105)
if head_dim != 128 {
    return Err(RvllmError::Loader {
        err: LoaderError::Corrupt {
            detail: format!(
                "v3 requires head_dim == 128, got {head_dim} ..."
            ),
        },
        ...
    });
}

// NEW
const ALLOWED_HEAD_DIMS: &[usize] = &[128, 256, 512];
if !ALLOWED_HEAD_DIMS.contains(&head_dim) {
    return Err(RvllmError::Loader {
        err: LoaderError::Corrupt {
            detail: format!(
                "v3 requires head_dim in {ALLOWED_HEAD_DIMS:?}, got {head_dim} \
                 (hidden={hidden_size}, heads={num_attention_heads})"
            ),
        },
        ctx: LoaderCtx { path: p, tensor: None },
        bt: std::backtrace::Backtrace::capture(),
    });
}
```

Note: this only gates the Llama/Qwen path. Gemma4 uses `Gemma4Arch::from_dir`
which has no head_dim gate (it reads `head_dim_sliding` and `head_dim_global`
as separate fields). No change needed there.

### 3. `v3/crates/rvllm-attention/src/lib.rs` -- Fa3Kernels::load

```rust
// OLD (line 146-160)
if head_dim != 128 {
    return Err(RvllmError::Attention {
        err: AttentionError::UnsupportedHeadDim {
            got: head_dim,
            required: 128,
        },
        ...
    });
}

// NEW
const ALLOWED: &[u32] = &[128, 256, 512];
if !ALLOWED.contains(&head_dim) {
    return Err(RvllmError::Attention {
        err: AttentionError::UnsupportedHeadDim { got: head_dim },
        ctx: AttnCtx {
            op: "Fa3Kernels::load",
            stream: 0,
            num_seqs: 0,
            head_dim,
        },
        bt: std::backtrace::Backtrace::capture(),
    });
}
```

Also update the module doc comment at line 9:

```rust
// OLD
//! - `head_dim == 128` hard gate at construction

// NEW
//! - `head_dim in {128, 256, 512}` gate at construction
```

### 4. `v3/crates/rvllm-attention/src/decode.rs` -- PagedDecodeParams::validate

```rust
// OLD (line 31-39)
if self.head_dim != 128 {
    return Err(RvllmError::Attention {
        err: AttentionError::UnsupportedHeadDim {
            got: self.head_dim,
            required: 128,
        },
        ...
    });
}

// NEW
const ALLOWED: &[u32] = &[128, 256, 512];
if !ALLOWED.contains(&self.head_dim) {
    return Err(RvllmError::Attention {
        err: AttentionError::UnsupportedHeadDim { got: self.head_dim },
        ctx: ctx(),
        bt: std::backtrace::Backtrace::capture(),
    });
}
```

Update the doc comment at line 63:

```rust
// OLD
/// loaded and head_dim is 128.

// NEW
/// loaded and head_dim is in {128, 256, 512}.
```

Update tests:
- `rejects_head_dim_64` stays as-is (64 is still rejected)
- `good()` stays at head_dim=128 (128 is still valid)
- Add new test:

```rust
#[test]
fn accepts_head_dim_256() {
    let mut p = good();
    p.head_dim = 256;
    p.scale = 1.0 / (256f32).sqrt();
    assert!(p.validate().is_ok());
}

#[test]
fn accepts_head_dim_512() {
    let mut p = good();
    p.head_dim = 512;
    p.scale = 1.0 / (512f32).sqrt();
    assert!(p.validate().is_ok());
}
```

### 5. `v3/crates/rvllm-attention/src/prefill.rs` -- PagedPrefillParams::validate

Same pattern as decode.rs:

```rust
// OLD (line 30-39)
if self.head_dim != 128 { ... }

// NEW
const ALLOWED: &[u32] = &[128, 256, 512];
if !ALLOWED.contains(&self.head_dim) {
    return Err(RvllmError::Attention {
        err: AttentionError::UnsupportedHeadDim { got: self.head_dim },
        ctx: ctx(),
        bt: std::backtrace::Backtrace::capture(),
    });
}
```

Update the test `prefill_validates_head_dim` -- it uses head_dim=64 which
is still rejected. Add a passing test for 256.

### 6. `v3/crates/rvllm-runtime/src/gemma4_bring_up.rs` -- dual FA3 loading

Current code loads a single FA3 .so with `head_dim_sliding` (256). For Gemma 4
the global layers use head_dim=512. Two options:

**Option A: single .so, runtime head_dim dispatch (chosen)**

The FA3 .so already receives `head_dim` as a runtime parameter per launch call
(see the C ABI: `head_dim: i32`). The SM89 kernel dispatches on it at runtime
(`if head_dim == 256 ... else if head_dim == 512`). For FA3 SM90, the .so must
contain template instantiations for both 256 and 512 -- but the dlopen'd .so
is the same file.

Change `Fa3Kernels::load()` to accept the *maximum* head_dim the engine will
use, and validate that the .so was compiled for it. The per-launch `head_dim`
param handles the rest.

```rust
// gemma4_bring_up.rs line 83-84
// OLD
let fa3 = Fa3Kernels::load(paths.fa3_so.clone(), arch.head_dim_sliding as u32)?;

// NEW -- pass max head_dim so the load gate accepts it
let fa3 = Fa3Kernels::load(paths.fa3_so.clone(), arch.max_head_dim() as u32)?;
```

This works because:
- The `.so` contains template instantiations for all head_dims it supports
- `head_dim` is passed as a runtime i32 to every kernel launch
- The `Fa3Kernels::load()` gate only needs to verify the .so supports the largest head_dim

**No second Fa3Kernels instance needed.** One .so, one load, per-layer dispatch
via the `head_dim` parameter already threaded through `PagedDecodeParams`.

The Gemma4 forward path (`gemma4_layer_exec.rs`) already sets
`dims.head_dim` per-layer from `arch.head_dim_for_layer(layer_idx)`, so
sliding layers dispatch at 256 and global layers at 512 automatically.

---

## FA3 SM90 recompilation

### What needs to happen

The current `libfa3_kernels.so` was compiled with head_dim=128 only. FA3
Hopper kernels are CUTLASS template-specialized on head_dim. The template
parameter flows through:

```
flash_fwd_kernel.h
  -> CollectiveMainloopFwd (CUTLASS 3.x)
    -> TiledMMA shape depends on head_dim (kHeadDim template param)
    -> TMA descriptor tile sizes depend on head_dim
    -> Shared memory allocation depends on head_dim
```

The key CUTLASS template params affected by head_dim:

| Template param | head_dim=128 | head_dim=256 | head_dim=512 |
|---|---|---|---|
| `kHeadDim` | 128 | 256 | 512 |
| TMA tile N-mode | 128 | 256 | 512 |
| Shared memory per stage (Q tile) | num_heads * 128 * sizeof(T) | num_heads * 256 * sizeof(T) | num_heads * 512 * sizeof(T) |
| Shared memory per stage (KV tile) | block_size * 128 * sizeof(T) | block_size * 256 * sizeof(T) | block_size * 512 * sizeof(T) |
| Register file per thread | ~128 regs | ~192 regs | needs occupancy check |

### FA3 instantiation files

FA3 Hopper uses explicit template instantiations in
`hopper/instantiations/`. Each head_dim gets its own .cu file:

```
flash_fwd_hdim128_fp16_paged_sm90.cu
flash_fwd_hdim128_e4m3_paged_sm90.cu
flash_fwd_hdim256_fp16_paged_sm90.cu      <-- need this
flash_fwd_hdim256_e4m3_paged_sm90.cu      <-- need this
flash_fwd_hdim512_fp16_paged_sm90.cu      <-- need this (global layers)
flash_fwd_hdim512_e4m3_paged_sm90.cu      <-- need this (global layers)
```

The instantiation file content is mechanical:

```cpp
// flash_fwd_hdim256_e4m3_paged_sm90.cu
#include "flash_fwd_launch_template.h"

template void run_mha_fwd_<cutlass::float_e4m3_t, 256, true>(
    Flash_fwd_params &params, cudaStream_t stream);
```

The `true` is the `Is_paged` template bool.

### Build command

```bash
cd /workspace/flash-attention/hopper
# Ensure instantiation files exist for hdim 256 and 512 (paged, e4m3 + fp16)
# Then build:
python setup.py build_ext --inplace \
    --head-dims 128,256,512 \
    --cuda-archs 90
```

If the upstream setup.py does not support `--head-dims`, manually add
the instantiation .cu files and include them in the build. The FA3
`flash_api.cpp` C wrapper dispatches on head_dim at runtime:

```cpp
// Already in flash_api.cpp (pseudocode):
if (params.d == 128) { run_mha_fwd_<T, 128, Is_paged>(...); }
else if (params.d == 256) { run_mha_fwd_<T, 256, Is_paged>(...); }
// ADD:
else if (params.d == 512) { run_mha_fwd_<T, 512, Is_paged>(...); }
```

### Shared memory concern for head_dim=512

FA3 Hopper uses 227+ KB shared memory for head_dim=128. At head_dim=512:
- Q tile shared memory: 4x larger
- KV tile shared memory: 4x larger
- H100 maximum: 228 KB per SM (with `cudaFuncSetAttribute` opt-in)

head_dim=512 will likely exceed 228 KB with the default FA3 pipeline
depth (num_stages). Solutions:

1. **Reduce pipeline stages** for head_dim=512: FA3 uses 2 stages by
   default. Dropping to 1 stage halves shared memory at the cost of
   losing double-buffering. Still faster than non-fused attention.

2. **Reduce tile sizes**: shrink the KV block tile from e.g. 128 to 64.
   FA3's CUTLASS mainloop supports this via `kBlockN` template param.

3. **Use the SM89 fallback for head_dim=512 only**: The custom SM89
   paged attention kernel (`paged_attention_sm89.cu`) already supports
   head_dim=512 with 512 threads per block and zero shared memory
   (register-only online softmax). This avoids the FA3 shared memory
   issue entirely. Performance is lower than FA3 but correct.

**Recommendation**: Build FA3 for head_dim=256 (will fit in shared
memory). For head_dim=512 (global layers, only 10 of 60 layers), use the
SM89-style kernel even on Hopper. Global layers are 1/6 of all layers;
the throughput impact of using the simpler kernel for just those layers
is minimal.

Implementation: load both the FA3 .so and the SM89 .so. Dispatch per-layer:
- Sliding layers (head_dim=256): FA3 SM90
- Global layers (head_dim=512): SM89-style kernel

This requires `Gemma4Bringup` to hold two kernel backends.

```rust
pub struct Gemma4Bringup {
    // ...
    pub fa3: Fa3Kernels,          // for head_dim=256 (sliding)
    pub attn_sm89: Fa3Kernels,    // for head_dim=512 (global), same ABI
    // ...
}
```

The `Fa3Kernels` struct already auto-detects SM89 vs SM90 symbols, so
loading the SM89 .so "just works" through the same type.

---

## RoPE kernel analysis

### `fused_rope_cache_fp8kv.cu` (Llama/Qwen path)

```
blockDim.x = head_dim / 2 = half_dim
```

- head_dim=128: 64 threads (current)
- head_dim=256: 128 threads
- head_dim=512: 256 threads

All within the 1024-thread-per-block limit. No shared memory used (all
register computation). **No kernel changes needed** -- head_dim is a
runtime parameter, thread count is set by the Rust launcher.

Launcher change in `rvllm-fused`: the launch config must use
`head_dim / 2` as `blockDim.x` (it already does -- verify).

### `fused_rope_partial_fp8kv.cu` (Gemma 4 path)

```
blockDim.x = head_dim / 2 = half_head
```

- Sliding (head_dim=256): 128 threads, `half_rotary = 128`
- Global (head_dim=512): 256 threads, `half_rotary = 64` (partial, 0.25)

Again within limits, no shared memory, already parameterized on
`head_dim` and `rotary_dim` at runtime. **No kernel changes needed.**

---

## Per-layer varying head_dim: how it works

The Gemma 4 layer loop already handles this. The per-layer dispatch in
the engine iterates layers and reads per-layer dimensions:

```rust
// In the engine's layer loop (pseudocode):
for layer_idx in 0..arch.num_hidden_layers {
    let head_dim = arch.head_dim_for_layer(layer_idx) as u32;
    let num_kv_heads = arch.num_kv_heads_for_layer(layer_idx) as u32;
    let rotary_dim = arch.rotary_dim_for_layer(layer_idx) as u32;

    let dims = Gemma4LayerDims {
        head_dim,
        num_kv_heads,
        rotary_dim,
        // ...
    };

    // This already flows through to:
    // - RoPE kernel (uses head_dim, rotary_dim as params)
    // - FA3 launch (uses head_dim in PagedDecodeParams)
    // - KV cache offset (uses num_kv_heads * head_dim)
    gemma4_forward(dims, ...)?;
}
```

The `Gemma4Arch` already provides:
- `head_dim_for_layer(i)` -> 256 (sliding) or 512 (global)
- `num_kv_heads_for_layer(i)` -> 16 (sliding) or 4 (global)
- `rotary_dim_for_layer(i)` -> 256 (sliding, full) or 128 (global, partial)
- `rope_theta_for_layer(i)` -> 10000 (sliding) or 1M (global)

### KV cache sizing with dual head_dim

Both sliding and global layers have the same KV projection dimension:
`kv_heads * head_dim = 16 * 256 = 8 * 512 = 4096`. The KV cache pages
are uniform: `[num_blocks, block_size, num_kv_heads_sliding, head_dim_sliding]`
= `[num_blocks, block_size, 16, 256]`.

Global layers reshape the same physical buffer as
`[num_blocks, block_size, 8, 512]` (half as many heads, twice as wide).
The total bytes per page are identical. **No KV cache layout change needed.**

The cache offset math in `gemma4_layer_exec.rs` already uses
`num_kv_heads * head_dim` per-layer, which produces the same byte offset
either way.

### Attention scale

`attn_scale = 1 / sqrt(head_dim)` varies per layer:
- Sliding: `1 / sqrt(256) = 0.0625`
- Global: `1 / sqrt(512) = 0.04419...`

This is already computed per-layer in the bringup code as
`1.0 / (head_dim as f32).sqrt()`.

---

## Summary: dual-kernel attention dispatch for Gemma 4

```
gemma4_forward(layer_idx):
    head_dim = arch.head_dim_for_layer(layer_idx)  // 256 or 512

    if head_dim <= 256:
        // Use FA3 SM90 (fast, TMA-pipelined)
        fa3.paged_decode_fp8(head_dim=256, ...)
    else:
        // Use SM89-style kernel (register-only, correct for 512)
        attn_sm89.paged_decode_fp8(head_dim=512, ...)
```

### Changes to `gemma4_layer_exec.rs`

Add a kernel selector param or pass two FA3 references:

```rust
pub unsafe fn gemma4_forward(
    dims: Gemma4LayerDims,
    kernels: &Gemma4LayerKernels,
    weights: &Gemma4LayerWeightPtrs,
    scratch: &Gemma4LayerScratch,
    meta: &Gemma4MetadataPtrs,
    cublaslt: &CublasLt,
    fa3: &Fa3Kernels,           // head_dim=256 (sliding)
    fa3_global: &Fa3Kernels,    // head_dim=512 (global), SM89 .so
    residual: u64,
    stream: u64,
) -> Result<()> {
    // ...
    // Step 5: FA3 attention
    let attn_backend = if dims.head_dim <= 256 { fa3 } else { fa3_global };
    let decode = PagedDecodeFp8Launcher::new(attn_backend);
    // ...
}
```

---

## Files touched (complete list)

| File | Change |
|------|--------|
| `v3/crates/rvllm-core/src/error.rs` | `UnsupportedHeadDim` drop `required` field |
| `v3/crates/rvllm-loader/src/load.rs` | `head_dim != 128` -> `!ALLOWED.contains(head_dim)` |
| `v3/crates/rvllm-attention/src/lib.rs` | Same gate change in `Fa3Kernels::load` + doc comment |
| `v3/crates/rvllm-attention/src/decode.rs` | Same gate change in `PagedDecodeParams::validate` + tests |
| `v3/crates/rvllm-attention/src/prefill.rs` | Same gate change in `PagedPrefillParams::validate` + tests |
| `v3/crates/rvllm-runtime/src/gemma4_bring_up.rs` | Load two attention backends; pass `max_head_dim()` |
| `v3/crates/rvllm-runtime/src/gemma4_layer_exec.rs` | Accept two FA3 refs, dispatch by head_dim |
| `v3/scripts/gemma4_h200_deploy.sh` | Build FA3 for head_dim=128,256 (512 uses SM89 kernel) |

### No changes needed

| File | Reason |
|------|--------|
| `v3/kernels/fused_rope_cache_fp8kv.cu` | Already parameterized on head_dim at runtime |
| `v3/kernels/fused_rope_partial_fp8kv.cu` | Already parameterized on head_dim and rotary_dim |
| `v3/kernels/paged_attention_sm89.cu` | Already supports 128/256/512 via template dispatch |
| `v3/crates/rvllm-loader/src/gemma4_arch.rs` | Already parses dual head_dim correctly |
| `v3/crates/rvllm-loader/src/gemma4_weights.rs` | Weights are uniform, no head_dim gate |

## Build / deploy order

1. On H200: rebuild FA3 .so with head_dim=128,256 instantiations
2. Verify SM89 .so already has 128/256/512 (it does)
3. Apply Rust changes (5 files), `cargo build --release --features cuda`
4. Run `compute-sanitizer` on a single-layer forward with head_dim=256
5. Run Gemma 4 end-to-end: verify sliding layers use FA3, global layers use SM89
6. Compare output logits to a reference (vLLM or HF transformers)
