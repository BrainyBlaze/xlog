//! Tests for the ilp_csr_histogram CUDA kernel.
//!
//! Run with: cargo test -p xlog-cuda-tests --test ilp_csr_histogram --release

xlog_cuda_tests::gpu_test_skip!(test_csr_histogram_basic, |ctx: &xlog_cuda_tests::harness::TestContext| {
    // sorted_facts = [0, 0, 1, 3, 3, 3], nnz=6, num_facts=5
    // Expected histogram = [2, 1, 0, 3, 0]
    let sorted_facts_host: Vec<u32> = vec![0, 0, 1, 3, 3, 3];
    let nnz = 6u32;
    let num_facts = 5u32;

    let mut d_sorted = ctx.memory.alloc::<u32>(sorted_facts_host.len()).expect("alloc sorted_facts");
    ctx.device.inner()
        .htod_sync_copy_into(&sorted_facts_host, &mut d_sorted)
        .expect("upload sorted_facts");

    let d_hist = ctx.provider
        .ilp_csr_histogram_launch(&d_sorted, nnz, num_facts)
        .expect("ilp_csr_histogram_launch");

    let mut hist_host = vec![0u32; num_facts as usize];
    ctx.device.inner()
        .dtoh_sync_copy_into(&d_hist, &mut hist_host)
        .expect("download histogram");

    assert_eq!(hist_host, vec![2, 1, 0, 3, 0], "histogram mismatch");
});

xlog_cuda_tests::gpu_test_skip!(test_csr_histogram_empty, |ctx: &xlog_cuda_tests::harness::TestContext| {
    // nnz=0, num_facts=3 → histogram = [0, 0, 0]
    let nnz = 0u32;
    let num_facts = 3u32;

    // Allocate a dummy sorted_facts buffer (won't be read since nnz=0)
    let d_sorted = ctx.memory.alloc::<u32>(1).expect("alloc dummy sorted_facts");

    let d_hist = ctx.provider
        .ilp_csr_histogram_launch(&d_sorted, nnz, num_facts)
        .expect("ilp_csr_histogram_launch empty");

    let mut hist_host = vec![0u32; num_facts as usize];
    ctx.device.inner()
        .dtoh_sync_copy_into(&d_hist, &mut hist_host)
        .expect("download histogram");

    assert_eq!(hist_host, vec![0, 0, 0], "empty histogram mismatch");
});

xlog_cuda_tests::gpu_test_skip!(test_csr_histogram_single_element, |ctx: &xlog_cuda_tests::harness::TestContext| {
    // sorted_facts = [2], nnz=1, num_facts=4 → histogram = [0, 0, 1, 0]
    let sorted_facts_host: Vec<u32> = vec![2];
    let nnz = 1u32;
    let num_facts = 4u32;

    let mut d_sorted = ctx.memory.alloc::<u32>(sorted_facts_host.len()).expect("alloc sorted_facts");
    ctx.device.inner()
        .htod_sync_copy_into(&sorted_facts_host, &mut d_sorted)
        .expect("upload sorted_facts");

    let d_hist = ctx.provider
        .ilp_csr_histogram_launch(&d_sorted, nnz, num_facts)
        .expect("ilp_csr_histogram_launch single");

    let mut hist_host = vec![0u32; num_facts as usize];
    ctx.device.inner()
        .dtoh_sync_copy_into(&d_hist, &mut hist_host)
        .expect("download histogram");

    assert_eq!(hist_host, vec![0, 0, 1, 0], "single element histogram mismatch");
});
