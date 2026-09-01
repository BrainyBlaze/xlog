//! Regression: byte offsets in the compaction and permutation kernels are
//! computed in 64 bits.
//!
//! `compact_bytes_by_mask` (filter.cu) and `apply_permutation_bytes` (sort.cu)
//! address their input and output as `row_index * elem_size + byte`. That product
//! is a BYTE count, and it passes 2^32 as soon as one column reaches 4 GiB --
//! 2^30 rows for a 4-byte column, 2^29 rows for an 8-byte one. Computed in
//! `uint32_t` it wrapped, and the kernels then read and wrote at the wrapped
//! address. The engine returned a plausible but incomplete relation with exit
//! status 0, and a different wrong answer on every launch; nothing on stderr
//! distinguished a correct run from a broken one.
//!
//! This test pins the READ side, which is deterministic: every surviving row
//! comes from beyond the 4 GiB mark, so a wrapped read returns values from the
//! start of the column instead. No two threads write the same output row, so
//! there is no race to make the failure intermittent -- the assertion either
//! holds or it does not.
//!
//! Cost: one 4 GiB device column plus its mask and prefix sum, roughly 7 GB of
//! device memory and the same again on the host. That is the floor rather than a
//! choice: the defect is defined by a buffer crossing 4 GiB and cannot be
//! reproduced below it. On a device that cannot hold it the test skips and says
//! so -- unless `XLOG_REQUIRE_CUDA=1`, which turns every skip here into a
//! failure.

use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::{CudaKernelProvider, CudaProviderBuilder};

/// 2^29 rows of `u64` is exactly 4 GiB. Every row above this index addresses
/// past the 2^32-byte mark, which is the whole point of the fixture.
const BOUNDARY_ROWS: usize = 1 << 29;
/// The rows that survive the mask. Small: the assertion is about which bytes
/// they were read from, not about volume.
const SURVIVORS: usize = 64;

fn large_budget_provider() -> Option<CudaKernelProvider> {
    CudaProviderBuilder::new(0, MemoryBudget::with_limit(16u64 * 1024 * 1024 * 1024))
        .build()
        .ok()
}

#[test]
fn compaction_reads_past_four_gibibytes() {
    let require = std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1");

    let Some(provider) = large_budget_provider() else {
        assert!(
            !require,
            "XLOG_REQUIRE_CUDA=1 but no CUDA provider was available"
        );
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let rows = BOUNDARY_ROWS + SURVIVORS;
    let schema = Schema::new(vec![("v".to_string(), ScalarType::U64)]);

    // value == row index, so a wrongly-sourced row names the row it came from.
    let values: Vec<u64> = (0..rows as u64).collect();
    let buffer = match provider.create_buffer_from_slice::<u64>(&values, schema) {
        Ok(buffer) => buffer,
        Err(error) => {
            assert!(
                !require,
                "XLOG_REQUIRE_CUDA=1 but the 4 GiB column would not allocate: {error}"
            );
            eprintln!("Skipping: device cannot hold a 4 GiB column: {error}");
            return;
        }
    };
    drop(values);

    let mut mask = vec![0u8; rows];
    for slot in mask[BOUNDARY_ROWS..].iter_mut() {
        *slot = 1;
    }

    let filtered = match provider.filter_by_mask(&buffer, &mask) {
        Ok(filtered) => filtered,
        Err(error) => {
            assert!(
                !require,
                "XLOG_REQUIRE_CUDA=1 but the 4 GiB compaction failed: {error}"
            );
            eprintln!("Skipping: 4 GiB compaction could not run: {error}");
            return;
        }
    };
    drop(mask);

    let got = provider
        .download_column::<u64>(&filtered, 0)
        .expect("download the compacted column");
    let expected: Vec<u64> = (BOUNDARY_ROWS as u64..rows as u64).collect();

    assert_eq!(
        got.len(),
        SURVIVORS,
        "the mask selects {SURVIVORS} rows; compaction returned {}",
        got.len()
    );
    assert_eq!(
        got, expected,
        "compaction read through a wrapped 32-bit byte offset: the surviving rows \
         were taken from the start of the column instead of from past the 4 GiB mark"
    );
}
