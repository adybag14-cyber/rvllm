use crate::device::XlaDeviceId;
use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XlaDtype {
    F32,
    F16,
    BF16,
    U8,
    U16,
    U32,
    I32,
    I64,
}

impl XlaDtype {
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::U16 => 2,
            Self::U8 => 1,
            Self::I64 => 8,
        }
    }
}

pub struct XlaBuffer {
    // Will hold *mut PJRT_Buffer once FFI lands.
    // For now, the fields define the API contract.
    shape: Vec<i64>,
    dtype: XlaDtype,
    device: XlaDeviceId,
}

impl XlaBuffer {
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn dtype(&self) -> XlaDtype {
        self.dtype
    }

    pub fn device(&self) -> XlaDeviceId {
        self.device
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    pub fn size_bytes(&self) -> usize {
        self.num_elements() * self.dtype.size_bytes()
    }

    pub fn copy_to_host(&self, _dst: &mut [u8]) -> Result<()> {
        // Will call PJRT_Buffer_ToHostBuffer + PJRT_Event_Await.
        Err(crate::LLMError::GpuError(
            "PJRT FFI not yet implemented -- cannot copy buffer to host".into(),
        ))
    }

    pub fn copy_from_host(_src: &[u8], _shape: &[i64], _dtype: XlaDtype, _device: XlaDeviceId) -> Result<Self> {
        // Will call PJRT_Client_BufferFromHostBuffer.
        Err(crate::LLMError::GpuError(
            "PJRT FFI not yet implemented -- cannot create buffer from host".into(),
        ))
    }
}

unsafe impl Send for XlaBuffer {}
unsafe impl Sync for XlaBuffer {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_sizes() {
        assert_eq!(XlaDtype::F32.size_bytes(), 4);
        assert_eq!(XlaDtype::BF16.size_bytes(), 2);
        assert_eq!(XlaDtype::U8.size_bytes(), 1);
        assert_eq!(XlaDtype::I64.size_bytes(), 8);
    }

    #[test]
    fn buffer_size_calc() {
        let buf = XlaBuffer {
            shape: vec![128, 4096],
            dtype: XlaDtype::BF16,
            device: XlaDeviceId(0),
        };
        assert_eq!(buf.num_elements(), 128 * 4096);
        assert_eq!(buf.size_bytes(), 128 * 4096 * 2);
    }
}
