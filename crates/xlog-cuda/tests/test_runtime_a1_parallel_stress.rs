// crates/xlog-cuda/tests/test_runtime_a1_parallel_stress.rs
//! Acceptance gate **A1** for the v0.6 device-runtime allocator.
//!
//! Per the locked acceptance criteria: 8–16 host threads each
//! repeatedly allocate device blocks of varied sizes via
//! `XlogDeviceRuntime::try_get(0).allocate(...)`, write a
//! deterministic byte pattern on the GPU, read it back, verify it
//! matches, then deallocate. After all threads finish,
//! `bytes_outstanding()` must return to its baseline.
//!
//! Status: this test is expected to **pass** on `DirectCudaResource`.
//! `cuMemAlloc`/`cuMemFree` are device-wide and synchronous, so
//! parallel callers should not see overlapping allocations or
//! corrupted bookkeeping. A failure indicates a real bug in the
//! direct backend or the singleton — stop and debug, do not advance
//! to A2.
//!
//! The byte-pattern check goes through `cuMemcpyHtoD_v2` /
//! `cuMemcpyDtoH_v2` on the raw `DeviceBlock::ptr`. This is the
//! lowest-level API that exercises the block's memory directly,
//! independent of any cudarc safe wrapper, so a cross-thread pointer
//! aliasing bug would surface as a content mismatch rather than a
//! benign bookkeeping success.

use std::sync::{Arc, Barrier};
use std::thread;

use cudarc::driver::sys;
use xlog_cuda::device_runtime::{AllocTag, StreamId, XlogDeviceRuntime};

const NUM_THREADS: usize = 8;
const ITERATIONS_PER_THREAD: usize = 64;

/// Tiny LCG so we don't pull in `rand`. Per-thread seed is
/// `(thread_idx + 1) * 0x9E3779B97F4A7C15`. Deterministic across runs
/// for a fixed `NUM_THREADS`.
#[derive(Clone, Copy)]
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn next_in(&mut self, lo: usize, hi: usize) -> usize {
        let span = (hi - lo) as u64;
        lo + (self.next_u64() % span) as usize
    }
}

unsafe fn htod(ptr: u64, host: &[u8]) {
    let res = sys::cuMemcpyHtoD_v2(ptr, host.as_ptr() as *const _, host.len());
    assert_eq!(
        res,
        sys::cudaError_enum::CUDA_SUCCESS,
        "cuMemcpyHtoD_v2 returned {:?} for {} bytes at {:#x}",
        res,
        host.len(),
        ptr
    );
}

unsafe fn dtoh(ptr: u64, host: &mut [u8]) {
    let res = sys::cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, ptr, host.len());
    assert_eq!(
        res,
        sys::cudaError_enum::CUDA_SUCCESS,
        "cuMemcpyDtoH_v2 returned {:?} for {} bytes at {:#x}",
        res,
        host.len(),
        ptr
    );
}

#[test]
fn a1_parallel_stress_alloc_write_verify_dealloc() {
    let runtime = match XlogDeviceRuntime::try_get(0) {
        Ok(rt) => rt,
        Err(err) => {
            eprintln!("Skipping A1: CUDA runtime unavailable: {}", err);
            return;
        }
    };

    let baseline = runtime.bytes_outstanding();

    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let mut handles = Vec::with_capacity(NUM_THREADS);

    for thread_idx in 0..NUM_THREADS {
        let barrier = Arc::clone(&barrier);
        let runtime_ref = runtime;
        handles.push(thread::spawn(move || {
            // Wait until all threads are ready so the contention
            // window starts together.
            barrier.wait();

            let seed: u64 = (thread_idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut rng = Lcg::new(seed);
            // Per-thread tag pool: tags must be 'static; pre-pick
            // from a small const set indexed by thread.
            let tag = match thread_idx % 4 {
                0 => AllocTag("a1-thread-0"),
                1 => AllocTag("a1-thread-1"),
                2 => AllocTag("a1-thread-2"),
                _ => AllocTag("a1-thread-other"),
            };

            for iter in 0..ITERATIONS_PER_THREAD {
                let bytes = rng.next_in(64, 4096 + 1);
                let block = runtime_ref
                    .allocate(bytes, StreamId::DEFAULT, tag)
                    .unwrap_or_else(|e| {
                        panic!("thread {} iter {}: allocate({}) failed: {}", thread_idx, iter, bytes, e)
                    });
                assert_eq!(block.bytes, bytes);
                assert_eq!(block.device_ordinal, 0);

                // Deterministic per-(thread, iter) byte pattern: a
                // pointer-aliasing bug would write thread A's bytes
                // into thread B's block and the readback assertion
                // below would fail.
                let stamp: u8 = ((thread_idx as u32).wrapping_add(iter as u32) & 0xFF) as u8;
                let host_in: Vec<u8> = (0..bytes)
                    .map(|i| stamp.wrapping_add((i & 0xFF) as u8))
                    .collect();
                let mut host_out = vec![0u8; bytes];

                // SAFETY: block.ptr is a live device pointer of size
                // `bytes` returned by the runtime's allocator. The
                // host_in/host_out slices have matching length. The
                // synchronous memcpy variants block until the copy
                // completes on the default stream.
                unsafe {
                    htod(block.ptr, &host_in);
                    dtoh(block.ptr, &mut host_out);
                }
                assert_eq!(
                    host_out, host_in,
                    "thread {} iter {}: byte pattern mismatch — possible cross-thread pointer aliasing",
                    thread_idx, iter
                );

                runtime_ref
                    .deallocate(block)
                    .unwrap_or_else(|e| panic!("thread {} iter {}: deallocate failed: {}", thread_idx, iter, e));
            }
        }));
    }

    for (i, h) in handles.into_iter().enumerate() {
        h.join()
            .unwrap_or_else(|_| panic!("A1 thread {} panicked", i));
    }

    let after = runtime.bytes_outstanding();
    assert_eq!(
        after, baseline,
        "bytes_outstanding leaked: baseline {}, after {} — bookkeeping race in DirectCudaResource",
        baseline, after
    );
}
