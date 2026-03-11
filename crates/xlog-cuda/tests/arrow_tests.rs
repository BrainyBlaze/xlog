//! Tests for Arrow/CuDF integration

mod common;
use common::setup_provider;

use arrow::array::{AsArray, BooleanArray};
use arrow::datatypes::{Int64Type, UInt32Type};
use std::sync::Arc;
use xlog_core::{ScalarType, Schema};

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

    let buffer = provider
        .create_buffer_from_slices(
            &[bytemuck::cast_slice(&ids), bytemuck::cast_slice(&values)],
            schema,
        )
        .unwrap();

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
        assert_eq!(
            id_array.value(i),
            *expected_id,
            "id mismatch at index {}",
            i
        );
    }
    for (i, expected_value) in values.iter().enumerate() {
        assert_eq!(
            value_array.value(i),
            *expected_value,
            "value mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_export_bool_column_to_arrow() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Create a buffer with Bool column
    let schema = Schema::new(vec![("flag".to_string(), ScalarType::Bool)]);

    let flags: Vec<u8> = vec![1, 0, 1, 1, 0]; // true, false, true, true, false

    let buffer = provider
        .create_buffer_from_slices(&[&flags], schema)
        .unwrap();

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
        assert_eq!(
            flag_array.value(i),
            *expected,
            "flag mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_arrow_ipc_stream_roundtrip() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![
        ("id".to_string(), ScalarType::U32),
        ("value".to_string(), ScalarType::I64),
    ]);

    let ids: Vec<u32> = vec![10, 20, 30];
    let values: Vec<i64> = vec![111, 222, 333];

    let buffer = provider
        .create_buffer_from_slices(
            &[bytemuck::cast_slice(&ids), bytemuck::cast_slice(&values)],
            schema,
        )
        .unwrap();

    let ipc = provider.to_arrow_ipc_stream(&buffer).unwrap();
    let roundtripped = provider.from_arrow_ipc_stream(&ipc).unwrap();

    let rb = provider.to_arrow_record_batch(&roundtripped).unwrap();
    assert_eq!(rb.num_rows(), 3);

    let id_array = rb.column(0).as_primitive::<UInt32Type>();
    let value_array = rb.column(1).as_primitive::<Int64Type>();

    assert_eq!(id_array.values(), &ids);
    assert_eq!(value_array.values(), &values);
}

#[test]
fn test_arrow_ipc_stream_file_roundtrip() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
    let ids: Vec<u32> = vec![1, 2, 3, 4];

    let buffer = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&ids)], schema)
        .unwrap();

    let path = std::env::temp_dir().join(format!(
        "xlog_arrow_ipc_test_{}_{}.arrow",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));

    provider
        .write_arrow_ipc_stream_file(&buffer, &path)
        .unwrap();
    let roundtripped = provider.read_arrow_ipc_stream_file(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    let rb = provider.to_arrow_record_batch(&roundtripped).unwrap();
    let id_array = rb.column(0).as_primitive::<UInt32Type>();
    assert_eq!(id_array.values(), &ids);
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
        &[bytemuck::cast_slice(&ids), bytemuck::cast_slice(&values)],
        schema,
    );

    assert!(result.is_err(), "Expected error for mismatched row counts");
    let err_msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("Expected error for mismatched row counts"),
    };
    assert!(
        err_msg.contains("3 rows") && err_msg.contains("5 rows"),
        "Error message should mention the row count mismatch: {}",
        err_msg
    );
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

    let record_batch =
        arrow::record_batch::RecordBatch::try_new(schema, vec![x_array, y_array]).unwrap();

    // Import into CudaBuffer
    let buffer = provider.from_arrow_record_batch(&record_batch).unwrap();

    assert_eq!(buffer.num_rows(), 3);
    assert_eq!(buffer.arity(), 2);

    // Verify data roundtrips correctly
    let x_values = provider.download_column::<u32>(&buffer, 0).unwrap();
    let y_values = provider.download_column::<f64>(&buffer, 1).unwrap();

    assert_eq!(x_values, vec![10, 20, 30]);
    assert!((y_values[0] - 1.5).abs() < 0.001);
    assert!((y_values[1] - 2.5).abs() < 0.001);
    assert!((y_values[2] - 3.5).abs() < 0.001);
}

#[test]
fn test_arrow_roundtrip_all_types() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    // Create buffer with all supported types
    let schema = Schema::new(vec![
        ("bool_col".to_string(), ScalarType::Bool),
        ("u32_col".to_string(), ScalarType::U32),
        ("i32_col".to_string(), ScalarType::I32),
        ("u64_col".to_string(), ScalarType::U64),
        ("i64_col".to_string(), ScalarType::I64),
        ("f32_col".to_string(), ScalarType::F32),
        ("f64_col".to_string(), ScalarType::F64),
    ]);

    let bool_data: Vec<u8> = vec![1, 0, 1, 0];
    let u32_data: Vec<u8> = [1u32, 2, 3, 4]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let i32_data: Vec<u8> = [-1i32, -2, 3, 4]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let u64_data: Vec<u8> = [100u64, 200, 300, 400]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let i64_data: Vec<u8> = [-100i64, 200, -300, 400]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let f32_data: Vec<u8> = [1.5f32, 2.5, 3.5, 4.5]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let f64_data: Vec<u8> = [1.5f64, 2.5, 3.5, 4.5]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let buffer = provider
        .create_buffer_from_slices(
            &[
                &bool_data, &u32_data, &i32_data, &u64_data, &i64_data, &f32_data, &f64_data,
            ],
            schema,
        )
        .unwrap();

    // Export to Arrow
    let record_batch = provider.to_arrow_record_batch(&buffer).unwrap();

    // Import back
    let buffer2 = provider.from_arrow_record_batch(&record_batch).unwrap();

    // Verify roundtrip
    assert_eq!(buffer.num_rows(), buffer2.num_rows());
    assert_eq!(buffer.arity(), buffer2.arity());

    // Check bool column (index 0)
    let bool_orig = provider.download_column::<bool>(&buffer, 0).unwrap();
    let bool_round = provider.download_column::<bool>(&buffer2, 0).unwrap();
    assert_eq!(bool_orig, bool_round, "Bool column mismatch");

    // Check u32 column (index 1)
    let u32_orig = provider.download_column::<u32>(&buffer, 1).unwrap();
    let u32_round = provider.download_column::<u32>(&buffer2, 1).unwrap();
    assert_eq!(u32_orig, u32_round, "U32 column mismatch");

    // Check i32 column (index 2)
    let i32_orig = provider.download_column::<i32>(&buffer, 2).unwrap();
    let i32_round = provider.download_column::<i32>(&buffer2, 2).unwrap();
    assert_eq!(i32_orig, i32_round, "I32 column mismatch");

    // Check u64 column (index 3)
    let u64_orig = provider.download_column::<u64>(&buffer, 3).unwrap();
    let u64_round = provider.download_column::<u64>(&buffer2, 3).unwrap();
    assert_eq!(u64_orig, u64_round, "U64 column mismatch");

    // Check i64 column (index 4)
    let i64_orig = provider.download_column::<i64>(&buffer, 4).unwrap();
    let i64_round = provider.download_column::<i64>(&buffer2, 4).unwrap();
    assert_eq!(i64_orig, i64_round, "I64 column mismatch");

    // Check f32 column (index 5)
    let f32_orig = provider.download_column::<f32>(&buffer, 5).unwrap();
    let f32_round = provider.download_column::<f32>(&buffer2, 5).unwrap();
    assert_eq!(f32_orig, f32_round, "F32 column mismatch");

    // Check f64 column (index 6)
    let f64_orig = provider.download_column::<f64>(&buffer, 6).unwrap();
    let f64_round = provider.download_column::<f64>(&buffer2, 6).unwrap();
    assert_eq!(f64_orig, f64_round, "F64 column mismatch");
}
