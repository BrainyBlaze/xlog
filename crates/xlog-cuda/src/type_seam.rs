//! GpuScalar — marker trait for Rust scalar types that round-trip through GPU column storage.
//!
//! This is an internal seam for Wave 2's generic `download_column<T>()` and
//! `create_buffer_from_slice<T>()`. It is NOT a public API.
//!
//! # Bool encoding
//!
//! Write encoding (H2D): canonical `0x00` = false, `0x01` = true.
//! Read decoding (D2H): `0x00` = false, any nonzero byte = true.
//! (See provider.rs:7075 for current D2H bool decoding.)
//!
//! The asymmetry is intentional: we always write canonical values, but tolerate
//! non-canonical GPU output during reads to match existing provider behavior.

/// Marker trait: a Rust scalar type that can round-trip through GPU column storage.
///
/// Requires `cudarc::driver::DeviceRepr` + known byte width + little-endian serialization.
///
/// Wave 2 will add `download_column::<T>()` and `create_buffer_from_slice::<T>()`
/// that replace the current type-specialized function families.
pub trait GpuScalar: cudarc::driver::DeviceRepr + Copy + Send + 'static {
    /// Size in bytes of this scalar type.
    const BYTE_WIDTH: usize;

    /// Deserialize from a little-endian byte slice.
    /// The slice length must equal `BYTE_WIDTH`.
    fn from_le_bytes(bytes: &[u8]) -> Self;

    /// Serialize into a little-endian byte buffer.
    /// The buffer length must equal `BYTE_WIDTH`.
    fn to_le_bytes_into(self, buf: &mut [u8]);
}

impl GpuScalar for u8 {
    const BYTE_WIDTH: usize = 1;
    fn from_le_bytes(bytes: &[u8]) -> Self { bytes[0] }
    fn to_le_bytes_into(self, buf: &mut [u8]) { buf[0] = self; }
}

impl GpuScalar for u32 {
    const BYTE_WIDTH: usize = 4;
    fn from_le_bytes(bytes: &[u8]) -> Self { u32::from_le_bytes(bytes.try_into().unwrap()) }
    fn to_le_bytes_into(self, buf: &mut [u8]) { buf.copy_from_slice(&self.to_le_bytes()); }
}

impl GpuScalar for u64 {
    const BYTE_WIDTH: usize = 8;
    fn from_le_bytes(bytes: &[u8]) -> Self { u64::from_le_bytes(bytes.try_into().unwrap()) }
    fn to_le_bytes_into(self, buf: &mut [u8]) { buf.copy_from_slice(&self.to_le_bytes()); }
}

impl GpuScalar for i32 {
    const BYTE_WIDTH: usize = 4;
    fn from_le_bytes(bytes: &[u8]) -> Self { i32::from_le_bytes(bytes.try_into().unwrap()) }
    fn to_le_bytes_into(self, buf: &mut [u8]) { buf.copy_from_slice(&self.to_le_bytes()); }
}

impl GpuScalar for i64 {
    const BYTE_WIDTH: usize = 8;
    fn from_le_bytes(bytes: &[u8]) -> Self { i64::from_le_bytes(bytes.try_into().unwrap()) }
    fn to_le_bytes_into(self, buf: &mut [u8]) { buf.copy_from_slice(&self.to_le_bytes()); }
}

impl GpuScalar for f32 {
    const BYTE_WIDTH: usize = 4;
    fn from_le_bytes(bytes: &[u8]) -> Self { f32::from_le_bytes(bytes.try_into().unwrap()) }
    fn to_le_bytes_into(self, buf: &mut [u8]) { buf.copy_from_slice(&self.to_le_bytes()); }
}

impl GpuScalar for f64 {
    const BYTE_WIDTH: usize = 8;
    fn from_le_bytes(bytes: &[u8]) -> Self { f64::from_le_bytes(bytes.try_into().unwrap()) }
    fn to_le_bytes_into(self, buf: &mut [u8]) { buf.copy_from_slice(&self.to_le_bytes()); }
}

/// Bool encoding:
/// - Write (H2D): `0x00` = false, `0x01` = true (canonical).
/// - Read (D2H): `0x00` = false, nonzero = true (lenient, matches provider.rs:7075).
impl GpuScalar for bool {
    const BYTE_WIDTH: usize = 1;

    fn from_le_bytes(bytes: &[u8]) -> Self {
        // Lenient read: any nonzero byte is true (matches existing D2H behavior).
        bytes[0] != 0
    }

    fn to_le_bytes_into(self, buf: &mut [u8]) {
        // Canonical write: 0x00 or 0x01.
        buf[0] = if self { 1 } else { 0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: roundtrip a value through le-bytes serialization.
    fn roundtrip<T: GpuScalar + PartialEq + std::fmt::Debug>(val: T) {
        let mut buf = vec![0u8; T::BYTE_WIDTH];
        val.to_le_bytes_into(&mut buf);
        let recovered = T::from_le_bytes(&buf);
        assert_eq!(recovered, val);
    }

    #[test]
    fn test_gpu_scalar_roundtrip_u8() { roundtrip(42u8); roundtrip(0u8); roundtrip(255u8); }

    #[test]
    fn test_gpu_scalar_roundtrip_u32() { roundtrip(0u32); roundtrip(42u32); roundtrip(u32::MAX); }

    #[test]
    fn test_gpu_scalar_roundtrip_u64() { roundtrip(0u64); roundtrip(42u64); roundtrip(u64::MAX); }

    #[test]
    fn test_gpu_scalar_roundtrip_i32() { roundtrip(0i32); roundtrip(-1i32); roundtrip(i32::MAX); }

    #[test]
    fn test_gpu_scalar_roundtrip_i64() { roundtrip(0i64); roundtrip(-1i64); roundtrip(i64::MAX); }

    #[test]
    fn test_gpu_scalar_roundtrip_f32() { roundtrip(0.0f32); roundtrip(-1.5f32); roundtrip(f32::INFINITY); }

    #[test]
    fn test_gpu_scalar_roundtrip_f64() { roundtrip(0.0f64); roundtrip(-1.5f64); roundtrip(f64::INFINITY); }

    #[test]
    fn test_gpu_scalar_roundtrip_bool() {
        roundtrip(true);
        roundtrip(false);
    }

    #[test]
    fn test_bool_canonical_write() {
        let mut buf = [0xFFu8];
        false.to_le_bytes_into(&mut buf);
        assert_eq!(buf[0], 0x00, "false must write canonical 0x00");

        true.to_le_bytes_into(&mut buf);
        assert_eq!(buf[0], 0x01, "true must write canonical 0x01");
    }

    #[test]
    fn test_bool_lenient_read() {
        // Any nonzero byte reads as true (matches provider.rs:7075 behavior).
        assert!(!bool::from_le_bytes(&[0x00]));
        assert!(bool::from_le_bytes(&[0x01]));
        assert!(bool::from_le_bytes(&[0x02]));
        assert!(bool::from_le_bytes(&[0xFF]));
    }

    #[test]
    fn test_byte_width_consistency() {
        assert_eq!(u8::BYTE_WIDTH, std::mem::size_of::<u8>());
        assert_eq!(u32::BYTE_WIDTH, std::mem::size_of::<u32>());
        assert_eq!(u64::BYTE_WIDTH, std::mem::size_of::<u64>());
        assert_eq!(i32::BYTE_WIDTH, std::mem::size_of::<i32>());
        assert_eq!(i64::BYTE_WIDTH, std::mem::size_of::<i64>());
        assert_eq!(f32::BYTE_WIDTH, std::mem::size_of::<f32>());
        assert_eq!(f64::BYTE_WIDTH, std::mem::size_of::<f64>());
        assert_eq!(bool::BYTE_WIDTH, std::mem::size_of::<bool>());
    }
}
