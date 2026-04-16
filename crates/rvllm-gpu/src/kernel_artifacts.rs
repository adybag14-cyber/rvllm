//! Kernel artifact resolution: download pre-compiled PTX/cubin/CUTLASS from HuggingFace.
//!
//! Artifacts are organized by GPU architecture in a private HF repo:
//!   m0at/rvllm-kernels/{sm_90,sm_100,...}/
//!     manifest.json
//!     ptx/*.ptx
//!     cubin/*.cubin
//!     cutlass/libcutlass_kernels.so
//!     autotune/cutlass_autotune.json
//!
//! Resolution order:
//!   1. RVLLM_KERNEL_DIR or RVLLM_PTX_DIR env var (local override)
//!   2. Download from HuggingFace

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{info, warn};

use crate::{LLMError, Result};

const HF_KERNEL_REPO: &str = "m0at/rvllm-kernels";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelManifest {
    pub git_sha: String,
    pub build_timestamp: String,
    pub cuda_version: String,
    pub gpu_arch: String,
    pub files: HashMap<String, FileEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub sha256: String,
    pub size: u64,
}

fn has_ptx_files(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    // Check top-level and ptx/ subdir
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        if entry.path().extension().map_or(false, |e| e == "ptx") {
            return true;
        }
    }
    let ptx_subdir = dir.join("ptx");
    if ptx_subdir.is_dir() {
        for entry in std::fs::read_dir(&ptx_subdir).into_iter().flatten().flatten() {
            if entry.path().extension().map_or(false, |e| e == "ptx") {
                return true;
            }
        }
    }
    false
}

fn hf_token_from_env() -> Option<String> {
    std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
        .filter(|t| !t.is_empty())
}

/// Resolve kernel artifacts: local override first, then HuggingFace download.
///
/// Returns the directory path containing kernel files ready to load.
pub fn resolve_kernel_artifacts(gpu_arch: &str) -> Result<PathBuf> {
    // 1. Local override via env var
    for var in &["RVLLM_KERNEL_DIR", "RVLLM_PTX_DIR"] {
        if let Ok(dir) = std::env::var(var) {
            let p = PathBuf::from(&dir);
            if p.is_dir() && has_ptx_files(&p) {
                info!(path = %p.display(), env = var, "using local kernel directory");
                return Ok(p);
            }
        }
    }

    // 2. Download from HuggingFace
    info!(repo = HF_KERNEL_REPO, arch = gpu_arch, "downloading kernel artifacts from HuggingFace");
    download_from_hf(gpu_arch)
}

fn download_from_hf(gpu_arch: &str) -> Result<PathBuf> {
    use hf_hub::api::sync::ApiBuilder;

    let mut builder = ApiBuilder::from_env();
    if let Some(token) = hf_token_from_env() {
        builder = builder.with_token(Some(token));
    }
    let api = builder
        .build()
        .map_err(|e| LLMError::GpuError(format!("failed to init hf-hub API: {e}")))?;

    let repo = api.model(HF_KERNEL_REPO.to_string());

    // Download manifest first
    let manifest_hf_path = format!("{gpu_arch}/manifest.json");
    let manifest_local = repo.get(&manifest_hf_path).map_err(|e| {
        LLMError::GpuError(format!(
            "failed to download {manifest_hf_path} from {HF_KERNEL_REPO}: {e}. \
             Ensure HF_TOKEN is set and the repo exists."
        ))
    })?;

    let manifest_str = std::fs::read_to_string(&manifest_local).map_err(|e| {
        LLMError::GpuError(format!("failed to read manifest: {e}"))
    })?;
    let manifest: KernelManifest = serde_json::from_str(&manifest_str).map_err(|e| {
        LLMError::GpuError(format!("invalid manifest.json: {e}"))
    })?;

    info!(
        git_sha = %manifest.git_sha,
        cuda_version = %manifest.cuda_version,
        num_files = manifest.files.len(),
        "kernel manifest loaded"
    );

    // The manifest file lives at .../snapshots/<rev>/<arch>/manifest.json
    // The arch directory is the root for all kernel files
    let arch_dir = manifest_local.parent().ok_or_else(|| {
        LLMError::GpuError("manifest path has no parent".into())
    })?;

    // Download all files listed in manifest
    let mut failed = Vec::new();
    for (filename, entry) in &manifest.files {
        let hf_path = format!("{gpu_arch}/{filename}");
        match repo.get(&hf_path) {
            Ok(local_path) => {
                // Verify checksum
                if let Err(e) = verify_sha256(&local_path, &entry.sha256) {
                    warn!(file = %filename, %e, "checksum mismatch -- file may be corrupt");
                    failed.push(filename.clone());
                }
            }
            Err(e) => {
                warn!(file = %filename, %e, "failed to download kernel file");
                failed.push(filename.clone());
            }
        }
    }

    if !failed.is_empty() {
        return Err(LLMError::GpuError(format!(
            "{} kernel file(s) failed to download or verify: {}",
            failed.len(),
            failed.join(", ")
        )));
    }

    info!(
        path = %arch_dir.display(),
        files = manifest.files.len(),
        "all kernel artifacts downloaded and verified"
    );

    Ok(arch_dir.to_path_buf())
}

fn verify_sha256(path: &Path, expected: &str) -> std::result::Result<(), String> {
    let data = std::fs::read(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    let actual = format!("{:x}", hasher.finalize());
    if actual != expected {
        return Err(format!(
            "sha256 mismatch for {}: expected {expected}, got {actual}",
            path.display()
        ));
    }
    Ok(())
}

/// Detect GPU compute capability and return arch string like "sm_90".
#[cfg(feature = "cuda")]
pub fn detect_gpu_arch() -> Result<String> {
    unsafe {
        // Ensure CUDA is initialized
        let result = cudarc::driver::sys::cuInit(0);
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(LLMError::GpuError(format!("cuInit failed: {result:?}")));
        }

        let mut major = 0i32;
        let mut minor = 0i32;
        cudarc::driver::sys::cuDeviceGetAttribute(
            &mut major,
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            0,
        );
        cudarc::driver::sys::cuDeviceGetAttribute(
            &mut minor,
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            0,
        );

        if major == 0 {
            return Err(LLMError::GpuError("failed to detect GPU compute capability".into()));
        }

        Ok(format!("sm_{}{}", major, minor))
    }
}

#[cfg(not(feature = "cuda"))]
pub fn detect_gpu_arch() -> Result<String> {
    Err(LLMError::GpuError("GPU arch detection requires cuda feature".into()))
}
