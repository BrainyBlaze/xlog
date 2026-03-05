use cudarc::driver::DeviceSlice;
use xlog_cuda_tests::TestContext;

#[test]
fn test_count_mask_device_basic() {
    let ctx = TestContext::new().expect("CUDA unavailable");
    let mask_host: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1];
    let n = mask_host.len() as u32;
    let mut d_mask = ctx.memory.alloc::<u8>(mask_host.len()).unwrap();
    ctx.device
        .inner()
        .htod_sync_copy_into(&mask_host, &mut d_mask)
        .unwrap();

    let d_count = ctx.provider.count_mask_device(&d_mask, n).unwrap();

    let mut host_count = vec![0u32; 1];
    ctx.device
        .inner()
        .dtoh_sync_copy_into(&d_count, &mut host_count)
        .unwrap();
    assert_eq!(host_count[0], 4);
}

#[test]
fn test_count_mask_device_all_zeros() {
    let ctx = TestContext::new().expect("CUDA unavailable");
    let mask_host: Vec<u8> = vec![0, 0, 0];
    let n = mask_host.len() as u32;
    let mut d_mask = ctx.memory.alloc::<u8>(mask_host.len()).unwrap();
    ctx.device
        .inner()
        .htod_sync_copy_into(&mask_host, &mut d_mask)
        .unwrap();

    let d_count = ctx.provider.count_mask_device(&d_mask, n).unwrap();

    let mut host_count = vec![0u32; 1];
    ctx.device
        .inner()
        .dtoh_sync_copy_into(&d_count, &mut host_count)
        .unwrap();
    assert_eq!(host_count[0], 0);
}

#[test]
fn test_count_mask_device_empty() {
    let ctx = TestContext::new().expect("CUDA unavailable");
    let d_mask = ctx.memory.alloc::<u8>(0).unwrap();
    assert_eq!(d_mask.len(), 0);

    let d_count = ctx.provider.count_mask_device(&d_mask, 0).unwrap();

    let mut host_count = vec![0u32; 1];
    ctx.device
        .inner()
        .dtoh_sync_copy_into(&d_count, &mut host_count)
        .unwrap();
    assert_eq!(host_count[0], 0);
}
