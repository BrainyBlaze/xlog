//! Tests for DLPack export (zero-copy interop)

mod common;
use common::setup_provider;

use xlog_core::{ScalarType, Schema};
use xlog_cuda::{dlpack, CudaBuffer, CudaKernelProvider};

fn device_row_count(
    provider: &CudaKernelProvider,
    rows: u32,
) -> xlog_cuda::memory::TrackedCudaSlice<u32> {
    let mut d_num_rows = provider.memory().alloc::<u32>(1).expect("alloc");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[rows], &mut d_num_rows)
        .expect("htod row count");
    d_num_rows
}

fn buffer_with_row_cap(
    provider: &CudaKernelProvider,
    data: &[u32],
    row_cap: u64,
    logical_rows: u32,
    schema: Schema,
) -> CudaBuffer {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    let mut col = provider.memory().alloc::<u8>(bytes.len()).expect("alloc");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&bytes, &mut col)
        .expect("htod data");

    CudaBuffer::from_columns(
        vec![col.into()],
        row_cap,
        device_row_count(provider, logical_rows),
        schema,
    )
}

#[test]
fn test_export_u32_column_to_dlpack() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
    let ids: Vec<u32> = vec![1, 2, 3, 4, 5];

    let buffer = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&ids)], schema)
        .unwrap();

    let table = provider.to_dlpack_table(buffer);
    let tensor = table.column(0).unwrap();

    let ptr = tensor.as_ptr();
    assert!(!ptr.is_null());

    // SAFETY: ptr is owned by DlpackManagedTensor for the duration of this test.
    let managed = unsafe { &*ptr };
    assert_eq!(managed.dl_tensor.device.device_type, dlpack::K_DLCUDA);
    assert_eq!(managed.dl_tensor.device.device_id, 0);
    assert_eq!(managed.dl_tensor.ndim, 1);
    assert!(!managed.dl_tensor.shape.is_null());

    // SAFETY: shape points to a 1-element array allocated in DlpackCtx.
    let shape0 = unsafe { *managed.dl_tensor.shape };
    assert_eq!(shape0, 5);

    assert_eq!(managed.dl_tensor.dtype.code, dlpack::K_DLUINT);
    assert_eq!(managed.dl_tensor.dtype.bits, 32);
    assert_eq!(managed.dl_tensor.dtype.lanes, 1);
    assert_eq!(managed.dl_tensor.byte_offset, 0);
    assert!(!managed.dl_tensor.data.is_null());
}

#[test]
fn test_roundtrip_import_u32_column_from_dlpack_zero_copy() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
    let ids: Vec<u32> = vec![10, 20, 30, 40, 50];

    let buffer = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&ids)], schema.clone())
        .unwrap();

    let table = provider.to_dlpack_table(buffer);
    let tensor = table.column(0).unwrap();

    let raw_ptr = tensor.as_ptr();
    assert!(!raw_ptr.is_null());

    // Import takes ownership of the tensor (no device↔host copy).
    let imported = provider
        .from_dlpack_tensors_with_schema(schema, vec![tensor])
        .unwrap();

    assert_eq!(imported.num_rows(), ids.len() as u64);

    // The imported column should point to the same device pointer as the DLPack tensor.
    let dl_data_ptr = unsafe {
        let managed = &*raw_ptr;
        let base = managed.dl_tensor.data as usize;
        base + managed.dl_tensor.byte_offset as usize
    } as u64;
    let imported_ptr = *imported.column(0).unwrap().device_ptr();
    assert_eq!(imported_ptr as u64, dl_data_ptr);

    // Verify contents.
    let got = provider.download_column::<u32>(&imported, 0).unwrap();
    assert_eq!(got, ids);
}

#[test]
fn test_export_dlpack_shape_uses_logical_device_row_count() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
    let buffer = buffer_with_row_cap(&provider, &[10, 20, 99, 100], 4, 2, schema);

    let table = provider.to_dlpack_table(buffer);
    let tensor = table.column(0).unwrap();

    let shape0 = unsafe { *(*tensor.as_ptr()).dl_tensor.shape };
    assert_eq!(shape0, 2);
}

#[test]
fn test_export_dlpack_rejects_logical_rows_above_row_cap() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
    let buffer = buffer_with_row_cap(&provider, &[10, 20, 30, 40], 2, 4, schema);

    let table = provider.to_dlpack_table(buffer);
    let err = match table.column(0) {
        Ok(_) => panic!("logical rows above row_cap must fail"),
        Err(err) => err,
    };
    assert!(
        format!("{err}").contains("Logical row count 4 exceeds row capacity 2"),
        "unexpected error: {err}"
    );
}
