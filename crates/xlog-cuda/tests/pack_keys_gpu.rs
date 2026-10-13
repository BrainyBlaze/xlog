mod common;
use common::setup_provider;
use xlog_core::{ScalarType, Schema};
use xlog_cuda::JoinType;

#[test]
fn test_pack_keys_gpu_wide_keys_no_host_transfers() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![
        ("a".to_string(), ScalarType::U32),
        ("b".to_string(), ScalarType::U32),
        ("c".to_string(), ScalarType::U32),
        ("d".to_string(), ScalarType::U32),
        ("e".to_string(), ScalarType::U32),
    ]);

    let a = vec![1u32, 2, 3];
    let b = vec![4u32, 5, 6];
    let c = vec![7u32, 8, 9];
    let d = vec![10u32, 11, 12];
    let e = vec![13u32, 14, 15];

    let buffer = provider
        .create_buffer_from_u32_columns(&[&a, &b, &c, &d, &e], schema)
        .unwrap();

    provider.reset_host_transfer_stats();

    let index = provider
        .build_join_index_v2(&buffer, &[0, 1, 2, 3, 4])
        .unwrap();
    assert!(index.estimated_bytes() > 0);

    let stats = provider.host_transfer_stats();
    assert_eq!(
        stats.dtoh_bytes, 0,
        "unexpected device-to-host transfers during GPU key packing: {} bytes",
        stats.dtoh_bytes
    );
    assert_eq!(
        stats.htod_bytes, 0,
        "unexpected key packing upload: {stats:?}"
    );
    assert_eq!(stats.htod_calls, 0);
    assert_eq!(stats.dtoh_calls, 0);
}

#[test]
fn test_pack_keys_gpu_common_path_no_host_transfers() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let schema = Schema::new(vec![
        ("a".to_string(), ScalarType::U32),
        ("b".to_string(), ScalarType::U32),
    ]);

    let a = vec![1u32, 2, 3, 4];
    let b = vec![10u32, 20, 30, 40];
    let buffer = provider
        .create_buffer_from_u32_columns(&[&a, &b], schema)
        .unwrap();

    provider.reset_host_transfer_stats();

    let index = provider.build_join_index_v2(&buffer, &[0, 1]).unwrap();
    assert!(index.estimated_bytes() > 0);

    let stats = provider.host_transfer_stats();
    assert_eq!(
        stats.dtoh_bytes, 0,
        "unexpected device-to-host transfer during common GPU key packing: {:?}",
        stats
    );
    assert_eq!(
        stats.htod_bytes, 0,
        "unexpected host-to-device transfer during common GPU key packing: {:?}",
        stats
    );
}

#[test]
fn wide_mixed_type_keys_preserve_tail_columns_without_transfers() {
    let Some(provider) = setup_provider() else {
        return;
    };
    for key_count in [1usize, 4, 5, 8, 9, 17] {
        let make_columns = |tail: [u64; 3], payload: [u32; 3]| {
            let mut columns = Vec::new();
            let mut schema = Vec::new();
            for column in 0..key_count {
                let wide = column % 2 == 1;
                schema.push((
                    format!("key_{column}"),
                    if wide {
                        ScalarType::U64
                    } else {
                        ScalarType::U32
                    },
                ));
                let values = if column + 1 == key_count {
                    tail
                } else {
                    [column as u64 + 13; 3]
                };
                columns.push(if wide {
                    values
                        .into_iter()
                        .flat_map(|v| (v << 40).to_le_bytes())
                        .collect::<Vec<_>>()
                } else {
                    values
                        .into_iter()
                        .flat_map(|v| (v as u32).to_le_bytes())
                        .collect::<Vec<_>>()
                });
            }
            schema.push(("payload".into(), ScalarType::U32));
            columns.push(payload.into_iter().flat_map(u32::to_le_bytes).collect());
            (columns, Schema::new(schema))
        };
        let (left_columns, left_schema) = make_columns([0, 1, 2], [10, 20, 30]);
        let (right_columns, right_schema) = make_columns([2, 0, 3], [100, 200, 300]);
        let left = provider
            .create_buffer_from_slices(
                &left_columns.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                left_schema,
            )
            .unwrap();
        let right = provider
            .create_buffer_from_slices(
                &right_columns.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                right_schema,
            )
            .unwrap();
        let keys = (0..key_count).collect::<Vec<_>>();
        provider.reset_host_transfer_stats();
        let result = provider
            .hash_join_v2(&left, &right, &keys, &keys, JoinType::Inner)
            .unwrap();
        let stats = provider.host_transfer_stats();
        assert_eq!(
            (stats.htod_bytes, stats.htod_calls),
            (0, 0),
            "{key_count} keys: {stats:?}"
        );
        assert_eq!(
            (stats.dtoh_bytes, stats.dtoh_calls),
            (0, 0),
            "{key_count} keys: {stats:?}"
        );

        // Export happens after the measured join, and checks the paired payloads.
        let left_values = provider.download_column::<u32>(&result, key_count).unwrap();
        let right_values = provider
            .download_column::<u32>(&result, 2 * key_count + 1)
            .unwrap();
        let mut pairs = left_values
            .into_iter()
            .zip(right_values)
            .collect::<Vec<_>>();
        pairs.sort_unstable();
        assert_eq!(pairs, [(10, 200), (30, 100)], "{key_count} keys");
    }
}
