# rvllm-xla design spec

## Purpose

rvllm-xla is the TPU backend for rvLLM. It loads StableHLO .mlir modules
(emitted by tpu/harness/emit_all.py), compiles them to XLA executables, and
runs them on TPU. It is a parallel backend to rvllm-gpu (CUDA), sharing the
same trait surface but targeting a completely different device.

## Current CUDA path (rvllm-gpu)

1. KernelLoader::new() scans a directory for .ptx/.cubin files
2. Each file is loaded into a CudaModule via cuModuleLoadData
3. Functions are resolved by name (KERNEL_FUNCTIONS table)
4. Kernels launched via cuLaunchKernel with raw device pointers
5. GpuBuffer<T> wraps CudaSlice<T> for device memory
6. GpuAllocator trait provides alloc/free/device_memory_info

rvllm-xla must provide equivalent types: XlaBuffer, XlaDevice, ModuleLoader.

## FFI path evaluation

Three options for Rust to talk to XLA on TPU:

### Option A: PJRT C API (recommended)

PJRT (Pretty Just-in-time Runtime) is XLA's official plugin API. It provides:
- `PJRT_Client` -- device management, compilation, buffer creation
- `PJRT_LoadedExecutable` -- compiled HLO ready for execution
- `PJRT_Buffer` -- device memory handle with shape/dtype metadata
- `PJRT_Executable_Execute` -- run a compiled program

The C API is a stable, versioned ABI (currently v0.54+). Google ships
`libtpu.so` as a PJRT plugin for Cloud TPU v4/v5/v6. The API is defined in
`xla/pjrt/c/pjrt_c_api.h` and loaded at runtime via dlopen.

Pros:
- Stable ABI, versioned, backward-compatible
- Single dlopen of libtpu.so -- no build-time XLA dependency
- Works across TPU generations (v4, v5e, v5p, v6e)
- Same API works for GPU PJRT plugins too (future unification)
- JAX itself uses this path

Cons:
- Callback-heavy struct-of-function-pointers pattern
- Some operations require serialized protobuf (HloModuleProto)
- StableHLO text must be serialized to bytecode before compile

### Option B: libtpu direct

libtpu.so exposes lower-level TPU driver functions beyond PJRT:
- TpuDriver_Open, TpuDriver_Allocate, TpuDriver_Execute
- Direct HBM management
- TPU-specific features (megacore, ICI mesh)

Pros:
- Lower overhead for TPU-specific optimizations
- Access to TPU hardware features not exposed via PJRT

Cons:
- Unstable, undocumented internal API
- Breaks across libtpu versions
- TPU-only -- no path to GPU unification
- Would require reverse-engineering from JAX/TF source

### Option C: xla_bridge (JAX's XLA:Python bindings)

Use PyO3 to call into jax._src.xla_bridge or jaxlib.xla_client.

Pros:
- Easiest to prototype
- Full access to JAX's compilation pipeline

Cons:
- Python runtime dependency in a Rust inference engine -- unacceptable
- GIL contention
- Deployment complexity

### Decision: PJRT C API (Option A)

PJRT is the right path. It is stable, Google-supported, and what JAX uses
internally. The Rust crate will dlopen libtpu.so at runtime, resolve the
PJRT function table, and call through it. No build-time XLA dependency.

## Architecture

```
safetensors weights
      |
      v
XlaBuffer (HBM)  <--  PJRT_Buffer (via PJRT_Client_BufferFromHostBuffer)
      |
      v
ModuleLoader
  - reads .mlir text from tpu/out/
  - calls PJRT_Client_Compile (StableHLO -> XLA HLO -> TPU executable)
  - caches PJRT_LoadedExecutable per module
      |
      v
execute(executable, &[XlaBuffer]) -> Vec<XlaBuffer>
  - calls PJRT_LoadedExecutable_Execute
  - returns output buffers on device
```

## Module loading: StableHLO .mlir -> executable

1. Read .mlir text file from disk (tpu/out/*.mlir)
2. Serialize to StableHLO bytecode (MLIR serialization via stablehlo-opt or
   the PJRT_Client_Compile path that accepts MLIR text directly)
3. Call PJRT_Client_Compile with the serialized program
4. Receive PJRT_LoadedExecutable handle
5. Cache by module name in a HashMap<String, LoadedExecutable>

The compile step is expensive (seconds). It should happen once at startup,
like PTX loading in the CUDA path.

## Buffer management: XlaBuffer

XlaBuffer is the TPU analog of GpuBuffer<T>. Key differences:

- XLA buffers carry shape and dtype metadata (PJRT_Buffer has on_device_shape)
- No raw device pointers -- all access goes through PJRT_Buffer_{ToHost,FromHost}
- XLA buffers are not byte-addressable from the host; you copy whole tensors

```rust
pub struct XlaBuffer {
    inner: *mut PjrtBuffer,  // opaque PJRT handle
    shape: Vec<i64>,
    dtype: XlaDtype,
    device: XlaDeviceId,
    size_bytes: usize,
}
```

copy_from_host: PJRT_Client_BufferFromHostBuffer (async, returns event)
copy_to_host:   PJRT_Buffer_ToHostBuffer (blocks until complete)

No partial reads/writes. No pointer arithmetic. If you need a slice, you
emit a StableHLO slice op and execute it.

## Kernel dispatch

In the CUDA path, the model runner calls:
```rust
kernels.launch_raw("rms_norm_f16", "rms_norm_f16_kernel", cfg, &mut args)?;
```

In the XLA path, the equivalent is:
```rust
let exe = modules.get("rms_norm")?;
let outputs = exe.execute(&[input_buf, weight_buf])?;
```

Key differences:
- No grid/block configuration -- XLA handles parallelism
- No shared memory sizing -- XLA handles tiling
- No raw pointer args -- pass XlaBuffer handles
- Outputs are returned, not written to pre-allocated buffers
- XLA may fuse adjacent ops automatically

The model runner will need a backend trait that abstracts over CUDA launch
vs XLA execute. This trait lives in rvllm-model-runner, not here. This
crate provides the concrete XLA implementation.

## Weight loading: safetensors -> XLA buffers

1. mmap the safetensors file (existing rvllm-model-loader logic)
2. For each tensor: read raw bytes, determine dtype + shape
3. Call PJRT_Client_BufferFromHostBuffer to upload to TPU HBM
4. Store XlaBuffer in the weight map

BF16 is the native TPU dtype. Weights stored as FP16 in safetensors should
be cast to BF16 during upload (via a convert op or host-side cast).

## Integration with rvllm-model-runner

The model runner currently depends on rvllm-gpu. To support both backends:

1. Define a `Backend` trait in rvllm-model-runner (or rvllm-core) with:
   - alloc(shape, dtype) -> Buffer
   - execute(module, inputs) -> outputs
   - upload(host_data, shape, dtype) -> Buffer
   - download(buffer) -> host_data
   - sync()

2. rvllm-gpu implements Backend via CUDA
3. rvllm-xla implements Backend via PJRT
4. Model runner is generic over Backend

This refactor is out of scope for this crate. rvllm-xla provides the
concrete types; the trait unification happens later.

## PJRT FFI layer (future, not in this scaffold)

The FFI module will:
1. dlopen libtpu.so (or libpjrt_gpu.so for testing on GPU)
2. Call PJRT_Plugin_Initialize to get the function table
3. Wrap each PJRT_* function in a safe Rust function
4. Handle async events (PJRT_Event) with blocking wait or callback

The function table is a single struct with ~60 function pointers. We will
codegen the Rust bindings from pjrt_c_api.h using bindgen, then wrap them
in a safe PjrtClient struct.

## Error handling

No fallbacks. If libtpu.so is missing, fail at startup. If a module fails
to compile, fail with the XLA error message. If execution fails, propagate
the PJRT error status. All errors go through LLMError::GpuError (which
should probably be renamed to DeviceError in a future refactor).

## File layout

```
crates/rvllm-xla/
  Cargo.toml
  SPEC.md
  src/
    lib.rs          -- module declarations, re-exports, prelude
    device.rs       -- XlaDevice: TPU device enumeration and info
    buffer.rs       -- XlaBuffer: device memory handle
    module.rs       -- ModuleLoader: .mlir -> compiled executable
```

Future files (not in initial scaffold):
```
    ffi.rs          -- raw PJRT C API bindings (bindgen output)
    client.rs       -- PjrtClient: safe wrapper around PJRT_Client
    executable.rs   -- LoadedExecutable: compiled program handle
    allocator.rs    -- XlaAllocator: GpuAllocator trait impl
```
