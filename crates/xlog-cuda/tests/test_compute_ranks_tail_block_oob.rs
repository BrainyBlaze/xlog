// RED repro for the compute_ranks tail-block underflow (external consumer P0.1/P0.2/P0.4).
//
// When the device row count sits below a block boundary while the launch
// grid was sized from the row cap, `block_count = block_end - block_start`
// underflows for blocks entirely past the live rows, so every thread in
// those blocks passes the `threadIdx.x < block_count` guard and writes
// ranks[gid] — out of bounds whenever gid >= ranks.len(). The union/dedup
// chain runs in exactly this count-below-cap state on every evaluate, which
// is how the overrun reaches production (silent garbage keys when the bytes
// land in pool slack, CUDA_ERROR_ILLEGAL_ADDRESS when they cross a page).
//
// The ranks buffer here is oversized past row_cap so the stray writes land
// in observable in-allocation slack instead of foreign memory.

mod common;
use common::setup_provider;
use xlog_cuda::provider::SORT_MODULE;
use xlog_cuda::{LaunchAsync, LaunchConfig};

const SENTINEL: u32 = 0xDEAD_BEEF;

#[test]
fn compute_ranks_tail_block_must_not_write_past_row_cap() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    let device = provider.device().inner();

    let block_size = 256u32;
    let row_cap = 384u32; // grid covers 2 blocks
    let actual_rows = 100u32; // below block 1's start -> tail-block underflow
    let ranks_len = 512usize; // slack [row_cap, 512) catches stray writes

    let keys: Vec<u32> = (0..row_cap).collect();
    let d_keys = device.htod_sync_copy(&keys).unwrap();
    let d_num_rows = device.htod_sync_copy(&[actual_rows]).unwrap();
    let mut d_ranks = device.htod_sync_copy(&vec![SENTINEL; ranks_len]).unwrap();

    let func = device
        .get_func(SORT_MODULE, "compute_ranks")
        .expect("compute_ranks kernel not loaded");
    let config = LaunchConfig {
        grid_dim: (row_cap.div_ceil(block_size), 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        func.launch(
            config,
            (&d_keys, &d_num_rows, row_cap, &mut d_ranks, 0u32),
        )
    }
    .expect("compute_ranks launch failed");
    device.synchronize().unwrap();

    let ranks_after = device.dtoh_sync_copy(&d_ranks).unwrap();
    let stray: Vec<usize> = (row_cap as usize..ranks_len)
        .filter(|&i| ranks_after[i] != SENTINEL)
        .collect();
    assert!(
        stray.is_empty(),
        "compute_ranks wrote past row_cap into [{}, {}): {} stray writes, first at index {}",
        row_cap,
        ranks_len,
        stray.len(),
        stray[0],
    );
}
