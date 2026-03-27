//! cuBLAS GEMM operations for linear algebra.

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm as _, GemmConfig, Gemv as _, GemvConfig};
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

use crate::Result;

/// Wrapper around cuBLAS for matrix operations.
pub struct CublasHandle {
    blas: CudaBlas,
    device: Arc<CudaDevice>,
}

impl CublasHandle {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        let blas = CudaBlas::new(device.clone())
            .map_err(|e| crate::LLMError::GpuError(format!("cuBLAS init failed: {e}")))?;
        Ok(Self { blas, device })
    }

    /// Returns a reference to the underlying device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// SGEMM: C = alpha * A * B + beta * C
    ///
    /// A: [m, k], B: [k, n], C: [m, n] in row-major layout.
    ///
    /// cuBLAS expects column-major, so we compute C^T = B^T * A^T which
    /// yields the correct row-major result without explicit transposes.
    ///
    /// # Safety
    /// Caller must ensure slices have the correct lengths:
    /// a >= m*k, b >= k*n, c >= m*n.
    pub fn sgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        beta: f32,
        c: &mut CudaSlice<f32>,
    ) -> Result<()> {
        // SAFETY: cuBLAS reads/writes device memory through valid CudaSlice handles.
        // Row-major trick: C^T = B^T * A^T  =>  call gemm(N, N, n, m, k, B, A, C)
        // but since the *data* is row-major and cuBLAS sees it as column-major-transposed,
        // we pass Op_T for both and swap A/B.
        unsafe {
            self.blas
                .gemm(
                    GemmConfig {
                        transa: cublasOperation_t::CUBLAS_OP_T,
                        transb: cublasOperation_t::CUBLAS_OP_N,
                        m: n as i32,
                        n: m as i32,
                        k: k as i32,
                        alpha,
                        lda: n as i32,
                        ldb: k as i32,
                        beta,
                        ldc: n as i32,
                    },
                    b, a, c,
                )
                .map_err(|e| crate::LLMError::GpuError(format!("cuBLAS sgemm failed: {e}")))?;
        }
        Ok(())
    }

    /// HGEMM: half-precision GEMM for f16.
    ///
    /// Same layout conventions as [`sgemm`](Self::sgemm) but operates on f16
    /// tensors. Internally uses f32 accumulation for numerical stability
    /// (matching cuBLAS CUBLAS_COMPUTE_32F behavior on Ampere+).
    ///
    /// This halves memory bandwidth for weight-bound operations (all linear
    /// projections in the transformer), which is the primary bottleneck for
    /// inference at moderate batch sizes.
    pub fn hgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: half::f16,
        a: &CudaSlice<half::f16>,
        b: &CudaSlice<half::f16>,
        beta: half::f16,
        c: &mut CudaSlice<half::f16>,
    ) -> Result<()> {
        // Row-major trick: C^T = B^T * A^T
        // cuBLAS sees column-major, so we swap A<->B and adjust dims.
        //
        // For f16, cudarc's Gemm trait implementation handles the GemmEx
        // dispatch automatically when T=f16.
        unsafe {
            self.blas
                .gemm(
                    GemmConfig {
                        transa: cublasOperation_t::CUBLAS_OP_T,
                        transb: cublasOperation_t::CUBLAS_OP_N,
                        m: n as i32,
                        n: m as i32,
                        k: k as i32,
                        alpha,
                        lda: n as i32,
                        ldb: k as i32,
                        beta,
                        ldc: n as i32,
                    },
                    b, a, c,
                )
                .map_err(|e| crate::LLMError::GpuError(format!("cuBLAS hgemm failed: {e}")))?;
        }
        Ok(())
    }

    /// Batched SGEMM for multiple independent matrix multiplications (e.g. multi-head attention).
    ///
    /// Each triple (a_batch[i], b_batch[i], c_batch[i]) is an independent GEMM with
    /// the same m/n/k dimensions.
    pub fn sgemm_batched(
        &self,
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f32,
        _a_batch: &[&CudaSlice<f32>],
        _b_batch: &[&CudaSlice<f32>],
        _beta: f32,
        _c_batch: &mut [&mut CudaSlice<f32>],
    ) -> Result<()> {
        // TODO: implement via cublasSgemmBatched or cublasSgemmStridedBatched
        Err(crate::LLMError::GpuError(
            "sgemm_batched not yet implemented".into(),
        ))
    }

    /// SGEMV: y = alpha * A * x + beta * y
    ///
    /// A: [m, n] row-major, x: [n], y: [m].
    ///
    /// For row-major A, cuBLAS (column-major) sees A^T, so we pass CUBLAS_OP_T
    /// to get the correct row-major matrix-vector product.
    pub fn sgemv(
        &self,
        m: usize,
        n: usize,
        alpha: f32,
        a: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        beta: f32,
        y: &mut CudaSlice<f32>,
    ) -> Result<()> {
        // SAFETY: cuBLAS reads/writes device memory through valid CudaSlice handles.
        // Row-major A stored contiguously is column-major A^T with dims (n, m).
        // We want y = A * x  =>  cublas: y = Op(A_col) * x  where A_col is (n,m).
        // Op = CUBLAS_OP_T gives us A^T_col = A_row which is what we want.
        unsafe {
            self.blas
                .gemv(
                    GemvConfig {
                        trans: cublasOperation_t::CUBLAS_OP_T,
                        m: n as i32,
                        n: m as i32,
                        alpha,
                        lda: n as i32,
                        incx: 1,
                        beta,
                        incy: 1,
                    },
                    a, x, y,
                )
                .map_err(|e| crate::LLMError::GpuError(format!("cuBLAS sgemv failed: {e}")))?;
        }
        Ok(())
    }
}
