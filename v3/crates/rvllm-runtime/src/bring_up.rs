//! Engine bring-up: assemble every subsystem from paths on disk.
//!
//! This module exists so `main.rs` in the bench + serve binaries can
//! reach for one `bring_up::Engine::load(paths)` call and get a fully
//! wired runtime back. No graph capture here (that's a separate step
//! after weights are loaded and bucket shapes are known).

use std::path::PathBuf;
use std::sync::Arc;

use rvllm_attention::Fa3Kernels;
use rvllm_core::{ConfigError, Result, RvllmError};
use rvllm_cutlass::{CutlassLib, Fp8GemmPlan, Policy};
use rvllm_kernels::{manifest::KernelManifest, KernelFn, KernelLoader, LoadedModule};
use rvllm_loader::{load_model, LoadedModel, ModelArch};
use rvllm_mem::{context::CudaContextHandle, stream::Stream, HbmArena};

/// Paths (and only paths) the engine needs at init. All other config
/// is read from `model_dir/config.json` and `kernels_dir/manifest.json`.
#[derive(Clone, Debug)]
pub struct EnginePaths {
    pub model_dir: PathBuf,
    pub kernels_dir: PathBuf,
    pub cutlass_so: PathBuf,
    pub fa3_so: PathBuf,
    pub policy_json: PathBuf,
}

/// Assembled subsystems.
pub struct Bringup {
    pub ctx: Arc<CudaContextHandle>,
    pub arena: HbmArena<'static>,
    pub stream: Stream,
    pub arch: ModelArch,
    pub model: LoadedModel,
    pub kernels: Arc<KernelLoader>,
    pub cutlass: CutlassLib,
    pub fa3: Fa3Kernels,
    pub policy: Policy,
    pub fused_modules: FusedModules,
}

/// Loaded CUDA modules + resolved kernel handles for the nine fused
/// kernels a layer uses.
pub struct FusedModules {
    pub add_rmsnorm: LoadedModule,
    pub rope_kv_write: LoadedModule,
    pub silu_mul: LoadedModule,
    pub quantize: LoadedModule,
    pub argmax: LoadedModule,
    pub fn_add_rmsnorm: KernelFn,
    pub fn_rope_kv_write: KernelFn,
    pub fn_silu_mul: KernelFn,
    pub fn_quantize: KernelFn,
    pub fn_argmax: KernelFn,
}

impl Bringup {
    pub fn load(paths: EnginePaths, arena_bytes: usize) -> Result<Self> {
        // 1. CUDA context + stream.
        let ctx = Arc::new(CudaContextHandle::init(0)?);

        // SAFETY: arena lifetime 'static via leak — engine owns it for program lifetime.
        let arena = HbmArena::new(&ctx, arena_bytes)?;
        // The 'static lifetime gymnastics: HbmArena<'ctx> borrows from ctx.
        // The Arc keeps ctx alive. We transmute the lifetime to 'static
        // because Bringup owns both. This is sound as long as `ctx`
        // outlives `arena` — which it does (they live in the same
        // struct and ctx is the last dropped).
        let arena: HbmArena<'static> = unsafe { std::mem::transmute(arena) };

        let stream = Stream::new(&ctx)?;

        // 2. Arch + model.
        let arch = ModelArch::from_dir(&paths.model_dir)?;
        let model = load_model(&paths.model_dir, &arena, &arch)?;

        // 3. Kernel manifest -> loader -> modules.
        let manifest_path = paths.kernels_dir.join("manifest.json");
        let manifest = KernelManifest::load_and_verify(&manifest_path)?;
        let kernels = Arc::new(KernelLoader::new(manifest));
        let fused_modules = load_fused(&kernels)?;

        // 4. FA3 .so.
        let fa3 = Fa3Kernels::load(paths.fa3_so.clone(), arch.head_dim as u32)?;

        // 5. Policy + CUTLASS .so (resolve every variant referenced in
        //    the policy).
        let policy_bytes = std::fs::read(&paths.policy_json).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: paths.policy_json.clone(),
            source,
        })?;
        let policy: Policy = serde_json::from_slice(&policy_bytes).map_err(|e| {
            RvllmError::config(
                ConfigError::Inconsistent {
                    reasons: vec![format!("policy.json parse: {e}")],
                },
                "policy.json",
            )
        })?;
        let variants: Vec<_> = policy
            .entries
            .values()
            .map(|e| e.variant)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let cutlass = CutlassLib::load(paths.cutlass_so.clone(), &variants)?;

        Ok(Self {
            ctx,
            arena,
            stream,
            arch,
            model,
            kernels,
            cutlass,
            fa3,
            policy,
            fused_modules,
        })
    }

    /// Resolve a GEMM plan for a (M, N, K, dtype) shape. Missing plan
    /// in the policy is a typed AutotuneCacheMiss; the engine refuses
    /// to run that shape.
    pub fn plan(&self, m: u32, n: u32, k: u32) -> Result<Fp8GemmPlan> {
        Fp8GemmPlan::from_policy(&self.policy, m, n, k, rvllm_core::DType::Fp8E4M3)
    }
}

fn load_fused(loader: &KernelLoader) -> Result<FusedModules> {
    let add_rmsnorm = loader.load_ptx("fused_add_rmsnorm_fp8_quant")?;
    let rope_kv_write = loader.load_ptx("fused_rope_kv_write")?;
    let silu_mul = loader.load_ptx("fused_silu_mul_fp8_quant")?;
    let quantize = loader.load_ptx("quantize_fp8_per_token")?;
    let argmax = loader.load_ptx("argmax")?;

    let fn_add_rmsnorm = add_rmsnorm.get_function("fused_add_rmsnorm_fp8_quant_kernel")?;
    let fn_rope_kv_write = rope_kv_write.get_function("fused_rope_kv_write_kernel")?;
    let fn_silu_mul = silu_mul.get_function("fused_silu_mul_fp8_quant_kernel")?;
    let fn_quantize = quantize.get_function("quantize_fp8_per_token_kernel")?;
    let fn_argmax = argmax.get_function("argmax_kernel")?;

    Ok(FusedModules {
        add_rmsnorm,
        rope_kv_write,
        silu_mul,
        quantize,
        argmax,
        fn_add_rmsnorm,
        fn_rope_kv_write,
        fn_silu_mul,
        fn_quantize,
        fn_argmax,
    })
}
