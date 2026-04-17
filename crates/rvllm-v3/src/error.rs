//! Typed, structured errors for rvllm-v3.
//!
//! No String errors across crate boundaries. Every variant carries the
//! context needed to diagnose a failure without re-reading logs: stream
//! pointer, kernel name, launch config, shape, variant id, etc.

use std::backtrace::Backtrace;

/// One top-level error type. All library functions return `Result<T, RvllmError>`.
#[derive(Debug, thiserror::Error)]
pub enum RvllmError {
    #[error("cuda: {op}: {kind:?} (kernel={kernel:?}, stream=0x{stream:x})")]
    Cuda {
        op: &'static str,
        kernel: Option<&'static str>,
        stream: u64,
        kind: CudaErrorKind,
        #[source]
        src: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    #[error("cutlass variant {variant}: workspace too small for M={m} N={n} K={k}: need {needed}, given {given}")]
    WorkspaceTooSmall {
        variant: &'static str,
        m: i32,
        n: i32,
        k: i32,
        needed: usize,
        given: usize,
    },
    #[error("cutlass: no autotune policy entry for shape M={m} N={n} K={k} ({label}). Run autotune-cutlass.")]
    AutotuneMiss {
        label: &'static str,
        m: usize,
        n: usize,
        k: usize,
    },
    #[error("graph: {op} for bucket {bucket}: {detail}")]
    Graph {
        op: &'static str,
        bucket: usize,
        detail: String,
    },
    #[error("metadata: bucket {bucket} exceeds max_batch {max_batch}")]
    BucketExceedsMax { bucket: usize, max_batch: usize },
    #[error("metadata: plan has {num_seqs} seqs but bucket is only {bucket}")]
    PlanExceedsBucket { num_seqs: usize, bucket: usize },
    #[error("loader: {kind}: {detail}")]
    Loader { kind: &'static str, detail: String },
    #[error("config: {field}: {reason}")]
    Config { field: &'static str, reason: String },
    #[error("sampling: {op}: {detail}")]
    Sampling { op: &'static str, detail: String },
    #[error("fa3 .so missing at {path}; build with kernels/build_fa3.sh and place it in the kernel dir. No PTX fallback.")]
    Fa3SoMissing { path: String },
    #[error("cutlass .so missing at {path}")]
    CutlassSoMissing { path: String },
    #[error("unsupported: {reason}")]
    Unsupported { reason: String },
}

#[derive(Debug)]
pub enum CudaErrorKind {
    AllocFailed,
    LaunchFailed,
    MemcpyFailed,
    StreamFailed,
    EventFailed,
    GraphFailed,
    ModuleLoadFailed,
    Other,
}

pub type Result<T> = std::result::Result<T, RvllmError>;

impl RvllmError {
    pub fn cuda_op(op: &'static str, kind: CudaErrorKind, src: impl std::fmt::Debug) -> Self {
        RvllmError::Cuda {
            op,
            kernel: None,
            stream: 0,
            kind,
            src: Some(Box::<dyn std::error::Error + Send + Sync>::from(format!("{src:?}"))),
        }
    }

    pub fn cuda_kernel(
        op: &'static str,
        kernel: &'static str,
        stream: u64,
        kind: CudaErrorKind,
        src: impl std::fmt::Debug,
    ) -> Self {
        RvllmError::Cuda {
            op,
            kernel: Some(kernel),
            stream,
            kind,
            src: Some(Box::<dyn std::error::Error + Send + Sync>::from(format!("{src:?}"))),
        }
    }

    pub fn loader(kind: &'static str, detail: impl Into<String>) -> Self {
        RvllmError::Loader {
            kind,
            detail: detail.into(),
        }
    }

    pub fn config(field: &'static str, reason: impl Into<String>) -> Self {
        RvllmError::Config {
            field,
            reason: reason.into(),
        }
    }

    pub fn graph(op: &'static str, bucket: usize, detail: impl Into<String>) -> Self {
        RvllmError::Graph {
            op,
            bucket,
            detail: detail.into(),
        }
    }
}

// Interop with the v2 LLMError string-wrapped variants: we accept them at
// crate boundaries so we can drive v2's kernel dispatch functions without
// rewriting them. Once v3 has its own kernel layer, remove this.
impl From<rvllm_core::prelude::LLMError> for RvllmError {
    fn from(e: rvllm_core::prelude::LLMError) -> Self {
        RvllmError::Cuda {
            op: "v2-interop",
            kernel: None,
            stream: 0,
            kind: CudaErrorKind::Other,
            src: Some(Box::new(e)),
        }
    }
}

#[allow(dead_code)]
fn _require_backtrace_available() -> Backtrace {
    // Keep the import alive for future context-with-backtrace variants.
    Backtrace::disabled()
}
