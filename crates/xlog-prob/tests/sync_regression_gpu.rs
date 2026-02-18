//! Guardrail: prevent regressions that re-introduce unnecessary device synchronize()
//! calls in the GpuXgcf evaluation code (gpu.rs).
//!
//! After the sync audit, gpu.rs should contain zero synchronize() calls.
//! Forward eval, backward eval, and builder all rely on same-stream ordering
//! and delegate sync to the caller or the batch boundary (executor.rs:355).
//!
//! Adding a new synchronize() here will cause this test to fail — make sure
//! it is truly needed before bumping the count.

use std::path::PathBuf;

/// Count `.synchronize()` calls in non-comment lines.
fn count_sync_calls(source: &str) -> usize {
    source
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                return false;
            }
            trimmed.contains(".synchronize()")
        })
        .count()
}

#[test]
fn gpu_rs_synchronize_count_is_pinned() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("gpu.rs");

    let text = std::fs::read_to_string(&path).expect("read gpu.rs");
    let count = count_sync_calls(&text);

    assert_eq!(
        count, 0,
        "gpu.rs has {} synchronize() calls (expected 0). \
         Forward eval, backward eval, and circuit builder all rely on \
         same-stream ordering. Callers sync at the batch boundary.",
        count
    );
}
