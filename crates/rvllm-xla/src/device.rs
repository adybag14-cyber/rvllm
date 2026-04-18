use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct XlaDeviceId(pub usize);

#[derive(Debug, Clone)]
pub struct TpuDevice {
    pub id: XlaDeviceId,
    pub chip_name: String,
    pub num_cores: usize,
    pub hbm_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct TpuMemoryInfo {
    pub total: usize,
    pub free: usize,
    pub used: usize,
}

pub fn list_tpu_devices() -> Result<Vec<TpuDevice>> {
    // Requires PJRT client initialization. Stub until FFI layer lands.
    // Will call PJRT_Client_Devices and read device attributes.
    Err(crate::LLMError::GpuError(
        "PJRT FFI not yet implemented -- cannot enumerate TPU devices".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_id_eq() {
        assert_eq!(XlaDeviceId(0), XlaDeviceId(0));
        assert_ne!(XlaDeviceId(0), XlaDeviceId(1));
    }
}
