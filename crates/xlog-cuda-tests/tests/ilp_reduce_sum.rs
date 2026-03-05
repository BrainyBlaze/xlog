//! Tests for the ilp_reduce_sum_f32 and ilp_reduce_sum_f64 GPU reduction kernels.
//!
//! Run with: cargo test -p xlog-cuda-tests --test ilp_reduce_sum --release -- --nocapture

use xlog_cuda_tests::harness::TestContext;

#[test]
fn f32_basic() {
    let ctx = match TestContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping f32_basic: no CUDA device ({})", e);
            return;
        }
    };

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let n = input_data.len() as u32;

    let mut d_input = ctx.memory.alloc::<f32>(input_data.len()).unwrap();
    ctx.device
        .inner()
        .htod_sync_copy_into(&input_data, &mut d_input)
        .unwrap();

    let d_result = ctx.provider.ilp_reduce_sum_f32_launch(&d_input, n).unwrap();

    let mut result_host = [0.0f32];
    ctx.device
        .inner()
        .dtoh_sync_copy_into(&d_result, &mut result_host)
        .unwrap();

    let expected = 15.0f32;
    assert!(
        (result_host[0] - expected).abs() < 1e-6,
        "f32_basic: expected {}, got {}",
        expected,
        result_host[0]
    );
}

#[test]
fn f64_basic() {
    let ctx = match TestContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping f64_basic: no CUDA device ({})", e);
            return;
        }
    };

    let input_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let n = input_data.len() as u32;

    let mut d_input = ctx.memory.alloc::<f64>(input_data.len()).unwrap();
    ctx.device
        .inner()
        .htod_sync_copy_into(&input_data, &mut d_input)
        .unwrap();

    let d_result = ctx.provider.ilp_reduce_sum_f64_launch(&d_input, n).unwrap();

    let mut result_host = [0.0f64];
    ctx.device
        .inner()
        .dtoh_sync_copy_into(&d_result, &mut result_host)
        .unwrap();

    let expected = 15.0f64;
    assert!(
        (result_host[0] - expected).abs() < 1e-12,
        "f64_basic: expected {}, got {}",
        expected,
        result_host[0]
    );
}

#[test]
fn f32_large() {
    let ctx = match TestContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping f32_large: no CUDA device ({})", e);
            return;
        }
    };

    let input_data: Vec<f32> = (1..=1000).map(|x| x as f32).collect();
    let n = input_data.len() as u32;

    let mut d_input = ctx.memory.alloc::<f32>(input_data.len()).unwrap();
    ctx.device
        .inner()
        .htod_sync_copy_into(&input_data, &mut d_input)
        .unwrap();

    let d_result = ctx.provider.ilp_reduce_sum_f32_launch(&d_input, n).unwrap();

    let mut result_host = [0.0f32];
    ctx.device
        .inner()
        .dtoh_sync_copy_into(&d_result, &mut result_host)
        .unwrap();

    let expected = 500500.0f32;
    assert!(
        (result_host[0] - expected).abs() < 1.0,
        "f32_large: expected {}, got {} (diff={})",
        expected,
        result_host[0],
        (result_host[0] - expected).abs()
    );
}

#[test]
fn f32_empty() {
    let ctx = match TestContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping f32_empty: no CUDA device ({})", e);
            return;
        }
    };

    // Allocate a dummy 1-element buffer (the kernel won't read it with n=0).
    let d_input = ctx.memory.alloc::<f32>(1).unwrap();
    let d_result = ctx.provider.ilp_reduce_sum_f32_launch(&d_input, 0).unwrap();

    let mut result_host = [f32::NAN];
    ctx.device
        .inner()
        .dtoh_sync_copy_into(&d_result, &mut result_host)
        .unwrap();

    assert!(
        (result_host[0] - 0.0).abs() < 1e-6,
        "f32_empty: expected 0.0, got {}",
        result_host[0]
    );
}
