mod common;
use common::setup_provider;

use arrow::array::AsArray;
use xlog_core::{ScalarType, Schema};

#[test]
fn test_device_row_count_tracks_host_count() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
    let ids: Vec<u32> = vec![1, 2, 3, 4];
    let buffer = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&ids)], schema)
        .unwrap();

    let mut host_count = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_count)
        .unwrap();
    assert_eq!(host_count[0], 4);
}

#[test]
fn test_union_gpu_dedups_rows() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
    let a: Vec<u32> = vec![1, 2, 2, 3];
    let b: Vec<u32> = vec![2, 4];
    let buf_a = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&a)], schema.clone())
        .unwrap();
    let buf_b = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&b)], schema)
        .unwrap();

    let out = provider.union(&buf_a, &buf_b).unwrap();
    let mut host_count = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(out.num_rows_device(), &mut host_count)
        .unwrap();
    assert_eq!(host_count[0], 4);
    let record = provider.to_arrow_record_batch(&out).unwrap();
    let col = record
        .column(0)
        .as_primitive::<arrow::datatypes::UInt32Type>();
    let mut vals: Vec<u32> = (0..record.num_rows()).map(|i| col.value(i)).collect();
    vals.sort();
    vals.dedup();
    assert_eq!(vals, vec![1, 2, 3, 4]);
}
