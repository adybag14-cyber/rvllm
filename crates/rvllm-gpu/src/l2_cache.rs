//! L2 cache persistence control for SM 8.0+ (Ampere/Hopper/Blackwell).
//!
//! Configures the device-level L2 set-aside so that small, frequently-accessed
//! buffers (KV cache pages, RoPE tables, norm weights) stay resident in L2
//! across kernel launches.

use cudarc::driver::sys;

/// Reserve a fraction of L2 for persisting accesses (device-level, once at init).
/// H100 has 50 MB L2; reserving 75% = 37.5 MB for persisting data.
pub fn configure_l2_persisting_cache(fraction: f64) -> Result<(), String> {
    let mut l2_size: i32 = 0;
    let mut max_persist: i32 = 0;
    unsafe {
        sys::cuDeviceGetAttribute(
            &mut l2_size,
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
            0, // device ordinal
        ).result().map_err(|e| format!("query L2 size: {e}"))?;

        sys::cuDeviceGetAttribute(
            &mut max_persist,
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
            0,
        ).result().map_err(|e| format!("query max persist L2: {e}"))?;
    }

    if l2_size == 0 || max_persist == 0 {
        tracing::debug!(l2_size, max_persist, "L2 persistence not supported, skipping");
        return Ok(());
    }

    let desired = ((l2_size as f64) * fraction) as usize;
    let capped = desired.min(max_persist as usize);

    unsafe {
        sys::cuCtxSetLimit(
            sys::CUlimit_enum::CU_LIMIT_PERSISTING_L2_CACHE_SIZE,
            capped,
        ).result().map_err(|e| format!("set L2 persist size: {e}"))?;
    }

    tracing::info!(
        l2_total_mb = l2_size as f64 / 1048576.0,
        persist_mb = capped as f64 / 1048576.0,
        "L2 persisting cache configured"
    );
    Ok(())
}
