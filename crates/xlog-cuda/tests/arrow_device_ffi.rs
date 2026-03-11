mod common;
use common::setup_provider;

use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use cudarc::driver::safe::{DevicePtr, DeviceSlice};
use cudarc::driver::sys;
use xlog_core::{ScalarType, Schema};

#[repr(C)]
struct RawArrowArray {
    length: i64,
    null_count: i64,
    offset: i64,
    n_buffers: i64,
    n_children: i64,
    buffers: *mut *const std::ffi::c_void,
    children: *mut *mut FFI_ArrowArray,
    dictionary: *mut FFI_ArrowArray,
    release: Option<unsafe extern "C" fn(*mut FFI_ArrowArray)>,
    private_data: *mut std::ffi::c_void,
}

struct RawDeviceSlice {
    ptr: sys::CUdeviceptr,
    len: usize,
}

impl DeviceSlice<u8> for RawDeviceSlice {
    fn len(&self) -> usize {
        self.len
    }
}

impl DevicePtr<u8> for RawDeviceSlice {
    fn device_ptr(&self) -> &sys::CUdeviceptr {
        &self.ptr
    }
}

#[test]
fn test_arrow_device_export_no_dtoh() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![
        ("id".to_string(), ScalarType::U32),
        ("value".to_string(), ScalarType::I64),
    ]);
    let ids: Vec<u32> = vec![1, 2, 3, 4];
    let values: Vec<i64> = vec![10, 20, 30, 40];

    let buffer = provider
        .create_buffer_from_slices(
            &[bytemuck::cast_slice(&ids), bytemuck::cast_slice(&values)],
            schema,
        )
        .unwrap();

    provider.reset_host_transfer_stats();
    let device_rb = provider.to_arrow_device_record_batch(buffer).unwrap();

    let stats = provider.host_transfer_stats();
    assert_eq!(stats.dtoh_bytes, 0, "device export performed DTOH");

    unsafe {
        let ptr = device_rb.as_ptr();
        assert!(!ptr.is_null());
        let arr = (*ptr).array as *mut FFI_ArrowArray;
        let schema = (*ptr).schema as *mut FFI_ArrowSchema;
        assert!(!arr.is_null());
        assert!(!schema.is_null());
    }
}

#[test]
fn test_arrow_device_export_schema_and_buffers() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![
        ("id".to_string(), ScalarType::U32),
        ("value".to_string(), ScalarType::I64),
    ]);
    let ids: Vec<u32> = vec![1, 2, 3, 4];
    let values: Vec<i64> = vec![10, 20, 30, 40];

    let buffer = provider
        .create_buffer_from_slices(
            &[bytemuck::cast_slice(&ids), bytemuck::cast_slice(&values)],
            schema,
        )
        .unwrap();

    let device_rb = provider.to_arrow_device_record_batch(buffer).unwrap();

    unsafe {
        let dev_ptr = device_rb.as_ptr();
        let schema_ptr = (*dev_ptr).schema;
        let array_ptr = (*dev_ptr).array;
        assert!(!schema_ptr.is_null());
        assert!(!array_ptr.is_null());

        let schema = arrow::datatypes::Schema::try_from(&*schema_ptr).unwrap();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "value");

        assert!(!array_ptr.is_null());
    }
}

#[test]
fn test_arrow_device_export_bool_bitpacked() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("flag".to_string(), ScalarType::Bool)]);
    let flags: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1, 0, 1];

    let buffer = provider
        .create_buffer_from_slices(&[&flags], schema)
        .unwrap();
    let device_rb = provider.to_arrow_device_record_batch(buffer).unwrap();

    unsafe {
        let dev_ptr = device_rb.as_ptr();
        let struct_arr = &*((*dev_ptr).array as *const RawArrowArray);
        assert_eq!(struct_arr.n_children, 1);
        let child_ptr = *struct_arr.children;
        let child_arr = &*(child_ptr as *const RawArrowArray);
        assert!(child_arr.n_buffers >= 2);

        let buffers = std::slice::from_raw_parts(child_arr.buffers, child_arr.n_buffers as usize);
        let values_ptr = buffers[1] as *const u8;
        assert!(!values_ptr.is_null());

        let packed_len = (flags.len() + 7) / 8;
        let mut host = vec![0u8; packed_len];
        let device = provider.device().inner();
        let dev_slice = RawDeviceSlice {
            ptr: values_ptr as u64,
            len: packed_len,
        };
        device.dtoh_sync_copy_into(&dev_slice, &mut host).unwrap();

        assert_eq!(host[0], 0b0100_1101u8);
        assert_eq!(host[1], 0b0000_0001u8);
    }
}

#[test]
fn test_arrow_device_export_symbol_metadata() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("sym".to_string(), ScalarType::Symbol)]);
    let ids: Vec<u32> = vec![1, 2, 3];

    let buffer = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&ids)], schema)
        .unwrap();

    let device_rb = provider.to_arrow_device_record_batch(buffer).unwrap();

    unsafe {
        let dev_ptr = device_rb.as_ptr();
        let schema_ptr = (*dev_ptr).schema;
        let schema = arrow::datatypes::Schema::try_from(&*schema_ptr).unwrap();
        let field = schema.field(0);
        let meta = field.metadata();
        assert_eq!(meta.get("xlog.symbol"), Some(&"true".to_string()));
        assert_eq!(meta.get("xlog.symbol_encoding"), Some(&"u32".to_string()));
    }
}
