//! GpuScalar — marker trait for Rust scalar types that round-trip through GPU column storage.
//!
//! The trait is `pub` because external crates call turbofish generics bounded by it
//! (e.g. `provider.download_column::<u32>()`), and Rust's `private_bounds` lint
//! requires trait bounds on pub functions to be pub. However, the trait is **sealed**:
//! external crates cannot add new implementations.
//!
//! # Bool encoding
//!
//! Write encoding (H2D): canonical `0x00` = false, `0x01` = true.
//! Read decoding (D2H): `0x00` = false, any nonzero byte = true.
//!
//! The asymmetry is intentional: we always write canonical values, but tolerate
//! non-canonical GPU output during reads to match existing provider behavior.

/// Private module prevents external crates from implementing `GpuScalar`.
mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for bool {}
}

/// Marker trait: a Rust scalar type that can round-trip through GPU column storage.
///
/// Requires `cudarc::driver::DeviceRepr` + known byte width + little-endian serialization.
///
/// This trait is **sealed** — it cannot be implemented outside `xlog-cuda`.
/// The fixed set of implementations covers all GPU-compatible scalar types.
pub trait GpuScalar:
    sealed::Sealed + crate::cuda_compat::KernelScalar + Copy + Send + 'static
{
    /// Size in bytes of this scalar type.
    const BYTE_WIDTH: usize;

    /// Deserialize from a little-endian byte slice.
    /// The slice length must equal `BYTE_WIDTH`.
    fn from_le_bytes(bytes: &[u8]) -> Self;

    /// Serialize into a little-endian byte buffer.
    /// The buffer length must equal `BYTE_WIDTH`.
    fn to_le_bytes_into(self, buf: &mut [u8]);

    /// Kernel function name for const-compare mask generation.
    fn filter_compare_kernel() -> &'static str;

    /// Kernel function name for column-column comparison mask.
    fn compare_col_kernel() -> &'static str;

    /// ScalarType variants accepted for this type in filter/compare operations.
    fn allowed_scalar_types() -> &'static [xlog_core::ScalarType];

    /// Optional fused compare+scan kernel (phase 1). Only u32 and f64 have optimized
    /// fused-scan paths. Returns None for types using the mask+compact path.
    fn filter_scan_phase1_kernel() -> Option<&'static str> {
        None
    }
}

impl GpuScalar for u8 {
    const BYTE_WIDTH: usize = 1;
    fn from_le_bytes(bytes: &[u8]) -> Self {
        bytes[0]
    }
    fn to_le_bytes_into(self, buf: &mut [u8]) {
        buf[0] = self;
    }
    fn filter_compare_kernel() -> &'static str {
        "filter_compare_u8"
    }
    fn compare_col_kernel() -> &'static str {
        "filter_compare_u8_col"
    }
    fn allowed_scalar_types() -> &'static [xlog_core::ScalarType] {
        &[xlog_core::ScalarType::Bool]
    }
}

impl GpuScalar for u32 {
    const BYTE_WIDTH: usize = 4;
    fn from_le_bytes(bytes: &[u8]) -> Self {
        u32::from_le_bytes(bytes.try_into().unwrap())
    }
    fn to_le_bytes_into(self, buf: &mut [u8]) {
        buf.copy_from_slice(&self.to_le_bytes());
    }
    fn filter_compare_kernel() -> &'static str {
        "filter_compare_u32"
    }
    fn compare_col_kernel() -> &'static str {
        "filter_compare_u32_col"
    }
    fn allowed_scalar_types() -> &'static [xlog_core::ScalarType] {
        &[xlog_core::ScalarType::U32, xlog_core::ScalarType::Symbol]
    }
    fn filter_scan_phase1_kernel() -> Option<&'static str> {
        Some("filter_compare_u32_scan_phase1")
    }
}

impl GpuScalar for u64 {
    const BYTE_WIDTH: usize = 8;
    fn from_le_bytes(bytes: &[u8]) -> Self {
        u64::from_le_bytes(bytes.try_into().unwrap())
    }
    fn to_le_bytes_into(self, buf: &mut [u8]) {
        buf.copy_from_slice(&self.to_le_bytes());
    }
    fn filter_compare_kernel() -> &'static str {
        "filter_compare_u64"
    }
    fn compare_col_kernel() -> &'static str {
        "filter_compare_u64_col"
    }
    fn allowed_scalar_types() -> &'static [xlog_core::ScalarType] {
        &[xlog_core::ScalarType::U64]
    }
}

impl GpuScalar for i32 {
    const BYTE_WIDTH: usize = 4;
    fn from_le_bytes(bytes: &[u8]) -> Self {
        i32::from_le_bytes(bytes.try_into().unwrap())
    }
    fn to_le_bytes_into(self, buf: &mut [u8]) {
        buf.copy_from_slice(&self.to_le_bytes());
    }
    fn filter_compare_kernel() -> &'static str {
        "filter_compare_i32"
    }
    fn compare_col_kernel() -> &'static str {
        "filter_compare_i32_col"
    }
    fn allowed_scalar_types() -> &'static [xlog_core::ScalarType] {
        &[xlog_core::ScalarType::I32]
    }
}

impl GpuScalar for i64 {
    const BYTE_WIDTH: usize = 8;
    fn from_le_bytes(bytes: &[u8]) -> Self {
        i64::from_le_bytes(bytes.try_into().unwrap())
    }
    fn to_le_bytes_into(self, buf: &mut [u8]) {
        buf.copy_from_slice(&self.to_le_bytes());
    }
    fn filter_compare_kernel() -> &'static str {
        "filter_compare_i64"
    }
    fn compare_col_kernel() -> &'static str {
        "filter_compare_i64_col"
    }
    fn allowed_scalar_types() -> &'static [xlog_core::ScalarType] {
        &[xlog_core::ScalarType::I64]
    }
}

impl GpuScalar for f32 {
    const BYTE_WIDTH: usize = 4;
    fn from_le_bytes(bytes: &[u8]) -> Self {
        f32::from_le_bytes(bytes.try_into().unwrap())
    }
    fn to_le_bytes_into(self, buf: &mut [u8]) {
        buf.copy_from_slice(&self.to_le_bytes());
    }
    fn filter_compare_kernel() -> &'static str {
        "filter_compare_f32"
    }
    fn compare_col_kernel() -> &'static str {
        "filter_compare_f32_col"
    }
    fn allowed_scalar_types() -> &'static [xlog_core::ScalarType] {
        &[xlog_core::ScalarType::F32]
    }
}

impl GpuScalar for f64 {
    const BYTE_WIDTH: usize = 8;
    fn from_le_bytes(bytes: &[u8]) -> Self {
        f64::from_le_bytes(bytes.try_into().unwrap())
    }
    fn to_le_bytes_into(self, buf: &mut [u8]) {
        buf.copy_from_slice(&self.to_le_bytes());
    }
    fn filter_compare_kernel() -> &'static str {
        "filter_compare_f64"
    }
    fn compare_col_kernel() -> &'static str {
        "filter_compare_f64_col"
    }
    fn allowed_scalar_types() -> &'static [xlog_core::ScalarType] {
        &[xlog_core::ScalarType::F64]
    }
    fn filter_scan_phase1_kernel() -> Option<&'static str> {
        Some("filter_compare_f64_scan_phase1")
    }
}

/// Bool encoding:
/// - Write (H2D): `0x00` = false, `0x01` = true (canonical).
/// - Read (D2H): `0x00` = false, nonzero = true (lenient, matches the D2H bool decoding path in provider/transfer.rs).
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

    // Bool uses the u8 kernel on the GPU side.
    fn filter_compare_kernel() -> &'static str {
        "filter_compare_u8"
    }
    fn compare_col_kernel() -> &'static str {
        "filter_compare_u8_col"
    }
    fn allowed_scalar_types() -> &'static [xlog_core::ScalarType] {
        &[xlog_core::ScalarType::Bool]
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
    fn test_gpu_scalar_roundtrip_u8() {
        roundtrip(42u8);
        roundtrip(0u8);
        roundtrip(255u8);
    }

    #[test]
    fn test_gpu_scalar_roundtrip_u32() {
        roundtrip(0u32);
        roundtrip(42u32);
        roundtrip(u32::MAX);
    }

    #[test]
    fn test_gpu_scalar_roundtrip_u64() {
        roundtrip(0u64);
        roundtrip(42u64);
        roundtrip(u64::MAX);
    }

    #[test]
    fn test_gpu_scalar_roundtrip_i32() {
        roundtrip(0i32);
        roundtrip(-1i32);
        roundtrip(i32::MAX);
    }

    #[test]
    fn test_gpu_scalar_roundtrip_i64() {
        roundtrip(0i64);
        roundtrip(-1i64);
        roundtrip(i64::MAX);
    }

    #[test]
    fn test_gpu_scalar_roundtrip_f32() {
        roundtrip(0.0f32);
        roundtrip(-1.5f32);
        roundtrip(f32::INFINITY);
    }

    #[test]
    fn test_gpu_scalar_roundtrip_f64() {
        roundtrip(0.0f64);
        roundtrip(-1.5f64);
        roundtrip(f64::INFINITY);
    }

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
        // Any nonzero byte reads as true (matches the D2H bool decoding path in provider/transfer.rs behavior).
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

    #[test]
    fn test_filter_kernel_names_non_empty() {
        // Every GpuScalar impl must return non-empty kernel names.
        assert!(!u8::filter_compare_kernel().is_empty());
        assert!(!u8::compare_col_kernel().is_empty());
        assert!(!u32::filter_compare_kernel().is_empty());
        assert!(!u32::compare_col_kernel().is_empty());
        assert!(!u64::filter_compare_kernel().is_empty());
        assert!(!u64::compare_col_kernel().is_empty());
        assert!(!i32::filter_compare_kernel().is_empty());
        assert!(!i32::compare_col_kernel().is_empty());
        assert!(!i64::filter_compare_kernel().is_empty());
        assert!(!i64::compare_col_kernel().is_empty());
        assert!(!f32::filter_compare_kernel().is_empty());
        assert!(!f32::compare_col_kernel().is_empty());
        assert!(!f64::filter_compare_kernel().is_empty());
        assert!(!f64::compare_col_kernel().is_empty());
        assert!(!bool::filter_compare_kernel().is_empty());
        assert!(!bool::compare_col_kernel().is_empty());
    }

    #[test]
    fn test_allowed_scalar_types_non_empty() {
        assert!(!u8::allowed_scalar_types().is_empty());
        assert!(!u32::allowed_scalar_types().is_empty());
        assert!(!u64::allowed_scalar_types().is_empty());
        assert!(!i32::allowed_scalar_types().is_empty());
        assert!(!i64::allowed_scalar_types().is_empty());
        assert!(!f32::allowed_scalar_types().is_empty());
        assert!(!f64::allowed_scalar_types().is_empty());
        assert!(!bool::allowed_scalar_types().is_empty());
    }

    #[test]
    fn test_fused_scan_kernel_only_u32_and_f64() {
        // Only u32 and f64 have fused-scan phase1 kernels.
        assert!(u32::filter_scan_phase1_kernel().is_some());
        assert!(f64::filter_scan_phase1_kernel().is_some());
        // All others return None.
        assert!(u8::filter_scan_phase1_kernel().is_none());
        assert!(u64::filter_scan_phase1_kernel().is_none());
        assert!(i32::filter_scan_phase1_kernel().is_none());
        assert!(i64::filter_scan_phase1_kernel().is_none());
        assert!(f32::filter_scan_phase1_kernel().is_none());
        assert!(bool::filter_scan_phase1_kernel().is_none());
    }

    #[test]
    fn test_bool_and_u8_share_gpu_kernels() {
        // Bool is stored as u8 on the GPU, so both types share the same kernels.
        assert_eq!(u8::filter_compare_kernel(), bool::filter_compare_kernel());
        assert_eq!(u8::compare_col_kernel(), bool::compare_col_kernel());
    }

    #[test]
    fn test_u32_allowed_includes_symbol() {
        // u32 filter must accept both U32 and Symbol columns.
        let allowed = u32::allowed_scalar_types();
        assert!(allowed.contains(&xlog_core::ScalarType::U32));
        assert!(allowed.contains(&xlog_core::ScalarType::Symbol));
    }
}
