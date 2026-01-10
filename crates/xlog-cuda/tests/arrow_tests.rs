//! Tests for Arrow/CuDF integration

use std::sync::Arc;
use arrow::array::{AsArray, BooleanArray};
use arrow::datatypes::{Int64Type, UInt32Type};
use xlog_core::{MemoryBudget, Schema, ScalarType};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn setup_provider() -> Option<CudaKernelProvider> {
    if cudarc::driver::CudaDevice::count().unwrap_or(0) == 0 {
        return None;
    }
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1024 * 1024 * 1024),
    ));
    CudaKernelProvider::new(device, memory).ok()
}

#[test]
fn test_export_to_arrow_record_batch() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Create a simple buffer with U32 and I64 columns
    let schema = Schema::new(vec![
        ("id".to_string(), ScalarType::U32),
        ("value".to_string(), ScalarType::I64),
    ]);

    let ids: Vec<u32> = vec![1, 2, 3, 4, 5];
    let values: Vec<i64> = vec![100, 200, 300, 400, 500];

    let buffer = provider.create_buffer_from_slices(
        &[
            bytemuck::cast_slice(&ids),
            bytemuck::cast_slice(&values),
        ],
        schema,
    ).unwrap();

    // Export to Arrow RecordBatch
    let record_batch = provider.to_arrow_record_batch(&buffer).unwrap();

    assert_eq!(record_batch.num_rows(), 5);
    assert_eq!(record_batch.num_columns(), 2);

    // Verify column names
    assert_eq!(record_batch.schema().field(0).name(), "id");
    assert_eq!(record_batch.schema().field(1).name(), "value");

    // Verify data values match input
    let id_array = record_batch.column(0).as_primitive::<UInt32Type>();
    let value_array = record_batch.column(1).as_primitive::<Int64Type>();

    for (i, expected_id) in ids.iter().enumerate() {
        assert_eq!(id_array.value(i), *expected_id, "id mismatch at index {}", i);
    }
    for (i, expected_value) in values.iter().enumerate() {
        assert_eq!(value_array.value(i), *expected_value, "value mismatch at index {}", i);
    }
}

#[test]
fn test_export_bool_column_to_arrow() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Create a buffer with Bool column
    let schema = Schema::new(vec![
        ("flag".to_string(), ScalarType::Bool),
    ]);

    let flags: Vec<u8> = vec![1, 0, 1, 1, 0]; // true, false, true, true, false

    let buffer = provider.create_buffer_from_slices(
        &[&flags],
        schema,
    ).unwrap();

    // Export to Arrow RecordBatch
    let record_batch = provider.to_arrow_record_batch(&buffer).unwrap();

    assert_eq!(record_batch.num_rows(), 5);
    assert_eq!(record_batch.num_columns(), 1);
    assert_eq!(record_batch.schema().field(0).name(), "flag");

    // Verify Bool values match input
    let flag_array = record_batch
        .column(0)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("Expected BooleanArray");

    let expected_bools = [true, false, true, true, false];
    for (i, expected) in expected_bools.iter().enumerate() {
        assert_eq!(flag_array.value(i), *expected, "flag mismatch at index {}", i);
    }
}

#[test]
fn test_create_buffer_from_slices_row_count_validation() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Create mismatched slices - 5 u32 values but only 3 i64 values
    let schema = Schema::new(vec![
        ("id".to_string(), ScalarType::U32),
        ("value".to_string(), ScalarType::I64),
    ]);

    let ids: Vec<u32> = vec![1, 2, 3, 4, 5]; // 5 rows
    let values: Vec<i64> = vec![100, 200, 300]; // 3 rows - mismatch!

    let result = provider.create_buffer_from_slices(
        &[
            bytemuck::cast_slice(&ids),
            bytemuck::cast_slice(&values),
        ],
        schema,
    );

    assert!(result.is_err(), "Expected error for mismatched row counts");
    let err_msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("Expected error for mismatched row counts"),
    };
    assert!(err_msg.contains("3 rows") && err_msg.contains("5 rows"),
        "Error message should mention the row count mismatch: {}", err_msg);
}

#[test]
fn test_import_from_arrow_record_batch() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    use arrow::array::*;
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

    // Create an Arrow RecordBatch
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("x", DataType::UInt32, false),
        Field::new("y", DataType::Float64, false),
    ]));

    let x_array = Arc::new(UInt32Array::from(vec![10, 20, 30])) as Arc<dyn Array>;
    let y_array = Arc::new(Float64Array::from(vec![1.5, 2.5, 3.5])) as Arc<dyn Array>;

    let record_batch = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![x_array, y_array],
    ).unwrap();

    // Import into CudaBuffer
    let buffer = provider.from_arrow_record_batch(&record_batch).unwrap();

    assert_eq!(buffer.num_rows(), 3);
    assert_eq!(buffer.arity(), 2);

    // Verify data roundtrips correctly
    let x_values = provider.download_column_u32(&buffer, 0).unwrap();
    let y_values = provider.download_column_f64(&buffer, 1).unwrap();

    assert_eq!(x_values, vec![10, 20, 30]);
    assert!((y_values[0] - 1.5).abs() < 0.001);
    assert!((y_values[1] - 2.5).abs() < 0.001);
    assert!((y_values[2] - 3.5).abs() < 0.001);
}
