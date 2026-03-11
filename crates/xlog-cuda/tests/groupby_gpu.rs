mod common;
use common::setup_provider;

use xlog_core::{AggOp, ScalarType, Schema};

#[test]
fn test_groupby_agg_gpu_multi_key() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![
        ("k1".to_string(), ScalarType::U32),
        ("k2".to_string(), ScalarType::U32),
        ("v".to_string(), ScalarType::U32),
    ]);

    let k1: Vec<u32> = vec![1, 1, 2, 2, 2];
    let k2: Vec<u32> = vec![7, 7, 9, 9, 10];
    let v: Vec<u32> = vec![10, 20, 3, 4, 5];

    let buffer = provider
        .create_buffer_from_slices(
            &[
                bytemuck::cast_slice(&k1),
                bytemuck::cast_slice(&k2),
                bytemuck::cast_slice(&v),
            ],
            schema,
        )
        .unwrap();

    let out = provider
        .groupby_agg(&buffer, &[0, 1], AggOp::Sum, 2)
        .unwrap();
    let rb = provider.to_arrow_record_batch(&out).unwrap();
    assert_eq!(rb.num_columns(), 3);
    assert_eq!(rb.num_rows(), 3);
}
