//! rvllm v3: clean-slate runtime facade over the v2 kernel stack.
//!
//! v3 is a type-state engine API with typed errors. Internally it drives
//! v2's kernel bindings, scheduler, and metadata packing. The invariants
//! v3 enforces at the API surface are:
//!
//! - One step in flight per engine. `launch()` returns a `PendingStep<'_>`
//!   that borrows the engine mutably; calling `launch()` again without
//!   `collect()`ing first is a compile error.
//! - One metadata-upload semantics. The caller cannot choose between
//!   padded/non-padded variants — the engine picks the only correct one.
//! - Typed errors. No `String`s across the API surface. `RvllmError`
//!   carries stream pointer, kernel name, variant id, and shape.
//! - No silent artifact-missing. Missing FA3 .so, CUTLASS .so, or
//!   autotune policy entry refuses to start with the missing item named.

pub mod error;
#[cfg(feature = "cuda")]
pub mod engine;

pub use error::{CudaErrorKind, Result, RvllmError};

#[cfg(feature = "cuda")]
pub use engine::{Engine, PendingStep};
