// crates/xlog-cuda/tests/test_d2h_counter.rs
//! Tests for the column-level D2H transfer counter on CudaKernelProvider.

mod common;
use common::setup_provider;
use xlog_core::{ScalarType, Schema};

#[test]
fn d2h_counter_starts_at_zero() {
    let Some(provider) = setup_provider() else {
        return;
    };
    assert_eq!(provider.d2h_transfer_count(), 0);
}

#[test]
fn d2h_counter_increments_on_download() {
    let Some(provider) = setup_provider() else {
        return;
    };
    let schema = Schema::new(vec![("col".to_string(), ScalarType::U32)]);
    let buf = provider
        .create_buffer_from_u32_columns(&[&[42u32]], schema)
        .unwrap();
    provider.reset_d2h_transfer_count();
    let _ = provider.download_column::<u32>(&buf, 0).unwrap();
    assert_eq!(provider.d2h_transfer_count(), 1);
    let _ = provider.download_column::<u32>(&buf, 0).unwrap();
    assert_eq!(provider.d2h_transfer_count(), 2);
}

#[test]
fn d2h_counter_resets() {
    let Some(provider) = setup_provider() else {
        return;
    };
    let schema = Schema::new(vec![("col".to_string(), ScalarType::U32)]);
    let buf = provider
        .create_buffer_from_u32_columns(&[&[42u32]], schema)
        .unwrap();
    let _ = provider.download_column::<u32>(&buf, 0).unwrap();
    assert!(provider.d2h_transfer_count() > 0);
    provider.reset_d2h_transfer_count();
    assert_eq!(provider.d2h_transfer_count(), 0);
}
