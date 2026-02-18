//! Guardrail: prevent regressions that re-introduce unnecessary device synchronize()
//! calls in the GPU circuit cache. Zero sync sites are needed:
//!
//!   - `cache_cnf_hash` — hash stays device-resident; same-stream ordering suffices
//!   - `lookup_or_insert` — slot/compile_needed stay device-resident; same-stream ordering suffices
//!
//! All cache operations rely on same-stream ordering. Adding a new
//! synchronize() here will cause this test to fail — make sure it is truly
//! needed (host reads the result) before bumping the count.

use std::path::PathBuf;

/// Count `.synchronize()` calls in non-comment lines.
fn count_sync_calls(source: &str) -> usize {
    source
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            // Skip comment-only lines (// and /// doc comments)
            if trimmed.starts_with("//") {
                return false;
            }
            trimmed.contains(".synchronize()")
        })
        .count()
}

#[test]
fn gpu_cache_synchronize_count_is_pinned() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("compilation");
    path.push("gpu_cache.rs");

    let text = std::fs::read_to_string(&path).expect("read gpu_cache.rs");
    let count = count_sync_calls(&text);

    assert_eq!(
        count, 0,
        "gpu_cache.rs has {} synchronize() calls (expected 0). \
         If you added a new sync, verify it is truly needed (host reads \
         the result immediately after). Same-stream ordering makes most \
         syncs between GPU-to-GPU ops redundant.",
        count
    );
}
