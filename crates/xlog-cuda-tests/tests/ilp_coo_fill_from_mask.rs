//! Test for the ilp_coo_fill_from_mask CUDA kernel.
//!
//! Run with: cargo test -p xlog-cuda-tests --test ilp_coo_fill_from_mask --release -- --nocapture

use xlog_cuda_tests::harness::TestContext;

#[test]
fn test_ilp_coo_fill_from_mask_basic() {
    let ctx = match TestContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping: CUDA unavailable: {}", e);
            return;
        }
    };

    // mask: [1, 0, 1, 1] -> 3 set bits
    let mask_data: Vec<u8> = vec![1, 0, 1, 1];
    // prefix_sum: exclusive scan of mask -> [0, 1, 1, 2]
    let prefix_sum_data: Vec<u32> = vec![0, 1, 1, 2];
    // fact_indices: [10, 20, 30, 40]
    let fact_indices_data: Vec<u32> = vec![10, 20, 30, 40];
    // offset_idx = 2, cand_value = 5
    // d_offsets = [0, 0, 7, 0, 0, 0] (offset at index 2 = 7)
    let offset_idx: u32 = 2;
    let cand_value: u32 = 5;
    let d_offsets_data: Vec<u32> = vec![0, 0, 7, 0, 0, 0];
    let num_query: u32 = 4;

    // Allocate output arrays large enough (offset 7 + 3 writes = need at least 10 elements)
    let coo_size = 16usize;

    let mut mask = ctx.memory.alloc::<u8>(num_query as usize).expect("alloc mask");
    let mut prefix_sum = ctx.memory.alloc::<u32>(num_query as usize).expect("alloc prefix_sum");
    let mut fact_indices = ctx.memory.alloc::<u32>(num_query as usize).expect("alloc fact_indices");
    let mut d_offsets = ctx.memory.alloc::<u32>(d_offsets_data.len()).expect("alloc d_offsets");
    let mut coo_fact = ctx.memory.alloc::<u32>(coo_size).expect("alloc coo_fact");
    let mut coo_cand = ctx.memory.alloc::<u32>(coo_size).expect("alloc coo_cand");

    // Upload input data
    ctx.htod_sync_copy_into(&mask_data, &mut mask).expect("upload mask");
    ctx.htod_sync_copy_into(&prefix_sum_data, &mut prefix_sum).expect("upload prefix_sum");
    ctx.htod_sync_copy_into(&fact_indices_data, &mut fact_indices).expect("upload fact_indices");
    ctx.htod_sync_copy_into(&d_offsets_data, &mut d_offsets).expect("upload d_offsets");

    // Initialize output to zeros
    let zeros = vec![0u32; coo_size];
    ctx.htod_sync_copy_into(&zeros, &mut coo_fact).expect("zero coo_fact");
    ctx.htod_sync_copy_into(&zeros, &mut coo_cand).expect("zero coo_cand");

    // Launch kernel with split offset_idx / cand_value
    ctx.provider
        .ilp_coo_fill_from_mask_launch(
            &mask,
            &prefix_sum,
            &fact_indices,
            offset_idx,
            cand_value,
            num_query,
            &d_offsets,
            &mut coo_fact,
            &mut coo_cand,
        )
        .expect("kernel launch");

    // Download results
    let coo_fact_host: Vec<u32> = ctx.dtoh_sync_copy(&coo_fact).expect("download coo_fact");
    let coo_cand_host: Vec<u32> = ctx.dtoh_sync_copy(&coo_cand).expect("download coo_cand");

    // Expected writes at indices 7, 8, 9 (offset = d_offsets[offset_idx=2] = 7):
    // mask[0]=1: write_idx = 7 + 0 = 7 -> coo_fact[7]=10, coo_cand[7]=5 (cand_value)
    // mask[1]=0: skip
    // mask[2]=1: write_idx = 7 + 1 = 8 -> coo_fact[8]=30, coo_cand[8]=5
    // mask[3]=1: write_idx = 7 + 2 = 9 -> coo_fact[9]=40, coo_cand[9]=5
    assert_eq!(coo_fact_host[7], 10, "coo_fact[7] should be 10");
    assert_eq!(coo_fact_host[8], 30, "coo_fact[8] should be 30");
    assert_eq!(coo_fact_host[9], 40, "coo_fact[9] should be 40");

    assert_eq!(coo_cand_host[7], 5, "coo_cand[7] should be cand_value=5");
    assert_eq!(coo_cand_host[8], 5, "coo_cand[8] should be cand_value=5");
    assert_eq!(coo_cand_host[9], 5, "coo_cand[9] should be cand_value=5");

    // Verify untouched positions remain zero
    for i in 0..7 {
        assert_eq!(coo_fact_host[i], 0, "coo_fact[{}] should be untouched (0)", i);
        assert_eq!(coo_cand_host[i], 0, "coo_cand[{}] should be untouched (0)", i);
    }
    for i in 10..coo_size {
        assert_eq!(coo_fact_host[i], 0, "coo_fact[{}] should be untouched (0)", i);
        assert_eq!(coo_cand_host[i], 0, "coo_cand[{}] should be untouched (0)", i);
    }

    println!("ilp_coo_fill_from_mask basic test PASSED");
}

#[test]
fn test_ilp_coo_fill_from_mask_empty() {
    let ctx = match TestContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping: CUDA unavailable: {}", e);
            return;
        }
    };

    // num_query = 0 should be a no-op
    let mask = ctx.memory.alloc::<u8>(1).expect("alloc mask");
    let prefix_sum = ctx.memory.alloc::<u32>(1).expect("alloc prefix_sum");
    let fact_indices = ctx.memory.alloc::<u32>(1).expect("alloc fact_indices");
    let d_offsets = ctx.memory.alloc::<u32>(1).expect("alloc d_offsets");
    let mut coo_fact = ctx.memory.alloc::<u32>(1).expect("alloc coo_fact");
    let mut coo_cand = ctx.memory.alloc::<u32>(1).expect("alloc coo_cand");

    ctx.provider
        .ilp_coo_fill_from_mask_launch(
            &mask,
            &prefix_sum,
            &fact_indices,
            0, // offset_idx
            0, // cand_value
            0, // num_query = 0
            &d_offsets,
            &mut coo_fact,
            &mut coo_cand,
        )
        .expect("empty launch should succeed");

    println!("ilp_coo_fill_from_mask empty test PASSED");
}

#[test]
fn test_ilp_coo_fill_from_mask_all_zeros() {
    let ctx = match TestContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping: CUDA unavailable: {}", e);
            return;
        }
    };

    // All-zero mask: no writes should occur
    let mask_data: Vec<u8> = vec![0, 0, 0, 0];
    let prefix_sum_data: Vec<u32> = vec![0, 0, 0, 0];
    let fact_indices_data: Vec<u32> = vec![10, 20, 30, 40];
    let d_offsets_data: Vec<u32> = vec![0, 5];
    let num_query: u32 = 4;
    let coo_size = 16usize;

    let mut mask = ctx.memory.alloc::<u8>(num_query as usize).expect("alloc mask");
    let mut prefix_sum = ctx.memory.alloc::<u32>(num_query as usize).expect("alloc prefix_sum");
    let mut fact_indices = ctx.memory.alloc::<u32>(num_query as usize).expect("alloc fact_indices");
    let mut d_offsets = ctx.memory.alloc::<u32>(d_offsets_data.len()).expect("alloc d_offsets");
    let mut coo_fact = ctx.memory.alloc::<u32>(coo_size).expect("alloc coo_fact");
    let mut coo_cand = ctx.memory.alloc::<u32>(coo_size).expect("alloc coo_cand");

    ctx.htod_sync_copy_into(&mask_data, &mut mask).expect("upload mask");
    ctx.htod_sync_copy_into(&prefix_sum_data, &mut prefix_sum).expect("upload prefix_sum");
    ctx.htod_sync_copy_into(&fact_indices_data, &mut fact_indices).expect("upload fact_indices");
    ctx.htod_sync_copy_into(&d_offsets_data, &mut d_offsets).expect("upload d_offsets");

    // Initialize output to sentinel value 0xDEAD
    let sentinel = vec![0xDEADu32; coo_size];
    ctx.htod_sync_copy_into(&sentinel, &mut coo_fact).expect("init coo_fact");
    ctx.htod_sync_copy_into(&sentinel, &mut coo_cand).expect("init coo_cand");

    ctx.provider
        .ilp_coo_fill_from_mask_launch(
            &mask,
            &prefix_sum,
            &fact_indices,
            1,  // offset_idx
            99, // cand_value
            num_query,
            &d_offsets,
            &mut coo_fact,
            &mut coo_cand,
        )
        .expect("kernel launch");

    let coo_fact_host: Vec<u32> = ctx.dtoh_sync_copy(&coo_fact).expect("download coo_fact");
    let coo_cand_host: Vec<u32> = ctx.dtoh_sync_copy(&coo_cand).expect("download coo_cand");

    // All positions should still be the sentinel value
    for i in 0..coo_size {
        assert_eq!(
            coo_fact_host[i], 0xDEAD,
            "coo_fact[{}] should be untouched (sentinel)",
            i
        );
        assert_eq!(
            coo_cand_host[i], 0xDEAD,
            "coo_cand[{}] should be untouched (sentinel)",
            i
        );
    }

    println!("ilp_coo_fill_from_mask all-zeros mask test PASSED");
}
