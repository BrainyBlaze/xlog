// crates/xlog-cuda/tests/runtime_cold_race.rs
//! Cold-start initialization race test for [`XlogDeviceRuntime`].
//!
//! Lives in its own integration-test binary so that no prior test
//! has warmed the per-ordinal singleton — the racing threads all hit
//! the slow path of `try_get` together. Earlier revisions of
//! `try_get` built a runtime *outside* the init lock and then called
//! `OnceLock::get_or_init`, which leaked the loser's runtime (and its
//! CUDA context handle) when two threads raced on cold start. The
//! fixed implementation gates the build on a per-ordinal mutex so
//! exactly one thread performs the device open / resource construction.
//!
//! What this test asserts:
//!   * All concurrent callers observe the **same `&'static`
//!     reference** for ordinal 0.
//!   * `device_ordinal()` is consistent across them.
//!
//! What this test does **not** assert (would require process-level
//! tracing to verify):
//!   * That only one CUDA context was actually created. We rely on
//!     the implementation owning the build path under the init mutex
//!     to make the leak structurally impossible. Code review enforces
//!     this; this test catches the observable consequence (different
//!     `&'static` addresses returned to different threads).

use std::sync::{Arc, Barrier};
use std::thread;

use xlog_cuda::device_runtime::XlogDeviceRuntime;

const RACE_THREADS: usize = 8;

#[test]
fn cold_concurrent_try_get_returns_one_singleton() {
    // No pre-warming: the first call into `try_get` for ordinal 0 in
    // this binary happens inside the racing threads.
    let barrier = Arc::new(Barrier::new(RACE_THREADS));
    let mut handles = Vec::with_capacity(RACE_THREADS);

    for _ in 0..RACE_THREADS {
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            // Skip the test on hosts without a CUDA device by
            // returning None. The assertion below treats this as a
            // clean skip (all None or all Some(addr)).
            XlogDeviceRuntime::try_get(0)
                .ok()
                .map(|rt| (rt as *const XlogDeviceRuntime as usize, rt.device_ordinal()))
        }));
    }

    let results: Vec<Option<(usize, u32)>> = handles
        .into_iter()
        .map(|h| h.join().expect("racing thread panicked"))
        .collect();

    // Either all skip (no CUDA), or all observe the same singleton.
    let any_some = results.iter().any(|r| r.is_some());
    if !any_some {
        eprintln!("Skipping cold-race: no CUDA device available on any thread");
        return;
    }

    let expected = results
        .iter()
        .find_map(|r| r.as_ref())
        .copied()
        .expect("at least one thread observed a runtime");

    for (i, r) in results.into_iter().enumerate() {
        let (addr, ord) = r.unwrap_or_else(|| panic!("thread {} unexpectedly skipped", i));
        assert_eq!(
            addr, expected.0,
            "thread {} observed a different singleton address — cold-race leak",
            i
        );
        assert_eq!(ord, expected.1, "thread {} reported wrong ordinal", i);
    }
}
