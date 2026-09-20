//! Regression coverage for cross-stream allocation lifetime safety.
//!
//! Raw allocations are owned by the canonical private-stream reclamation
//! path. Work submitted on another stream must be registered with the
//! allocation dependency ledger before the logical block is detached. The
//! managed test below holds that foreign stream behind a CUDA host callback,
//! records a queued use, and proves that physical reclamation cannot finish
//! until the foreign use is released. This tests the lifetime contract
//! directly instead of depending on a particular CUDA memory-pool address
//! reuse policy.
//!
//! The ignored unmanaged test remains an opt-in diagnostic for the caller
//! contract: raw CUDA work that bypasses the dependency ledger is invisible
//! to XLOG and may race reclamation.

use std::sync::Arc;

use cudarc::driver::sys;
use xlog_cuda::device_runtime::{
    AllocTag, AsyncCudaResource, DeviceMemoryResource, StreamId, StreamPool,
};
use xlog_cuda::CudaDevice;

const BYTES: usize = 4096;

/// Async memset on a specific raw stream handle. Unlike async
/// memcpy from a non-pinned host buffer (which the driver may
/// stage synchronously), `cuMemsetD8Async` is genuinely
/// stream-asynchronous — it queues on the stream and returns
/// immediately. Used here to ensure the cross-stream "use" of
/// the allocation is actually pending when we drop.
unsafe fn memset_async(stream: sys::CUstream, dst: u64, value: u8, len: usize) {
    let res = sys::cuMemsetD8Async(dst, value, len, stream);
    assert_eq!(
        res,
        sys::cudaError_enum::CUDA_SUCCESS,
        "cuMemsetD8Async: {:?}",
        res
    );
}

/// Synchronous DtoH (called only after both involved streams have
/// been synchronized).
unsafe fn dtoh_sync(dst: &mut [u8], src: u64) {
    let res = sys::cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut _, src, dst.len());
    assert_eq!(
        res,
        sys::cudaError_enum::CUDA_SUCCESS,
        "cuMemcpyDtoH_v2: {:?}",
        res
    );
}

/// Registered cross-stream work must finish before physical reclamation.
#[test]
fn managed_cross_stream_use_blocks_reclamation_until_use_completes() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    struct UseGate {
        entered: AtomicBool,
        released: AtomicBool,
    }

    struct ReleaseUse(Arc<UseGate>);

    impl Drop for ReleaseUse {
        fn drop(&mut self) {
            self.0.released.store(true, Ordering::Release);
        }
    }

    unsafe extern "C" fn hold_use(data: *mut std::ffi::c_void) {
        // SAFETY: exactly one callback consumes the Arc transferred below.
        let gate = unsafe { Arc::from_raw(data.cast::<UseGate>()) };
        gate.entered.store(true, Ordering::Release);
        // CUDA host callbacks must not call CUDA. This callback only holds the
        // use stream until the CPU controller releases it.
        while !gate.released.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let resource = AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool));

    // Two distinct non-default streams. If the pool can't fork at
    // least two streams, skip — the bug class needs the
    // alloc-stream / use-stream split.
    let s_alloc_id = match pool.acquire() {
        Ok(id) => id,
        Err(e) => {
            eprintln!("Skipping: StreamPool::acquire failed: {}", e);
            return;
        }
    };
    let s_use_id = match pool.acquire() {
        Ok(id) => id,
        Err(e) => {
            eprintln!("Skipping: StreamPool::acquire (second) failed: {}", e);
            return;
        }
    };
    assert_ne!(s_alloc_id, s_use_id);
    assert_ne!(s_alloc_id, StreamId::DEFAULT);
    assert_ne!(s_use_id, StreamId::DEFAULT);

    let s_use = pool.resolve(s_use_id).expect("s_use resolves");
    let s_use_handle = s_use.cu_stream();

    const PATTERN_USE: u8 = 0xCD;
    let block = resource
        .allocate(BYTES, s_alloc_id, AllocTag("managed-use"))
        .expect("allocate managed-use block");

    let gate = Arc::new(UseGate {
        entered: AtomicBool::new(false),
        released: AtomicBool::new(false),
    });
    let release = ReleaseUse(Arc::clone(&gate));
    let callback_owner = Arc::into_raw(Arc::clone(&gate)).cast_mut().cast();
    // SAFETY: the stream is owned by the pool; the callback owns its gate and
    // coordinates CPU state only.
    assert_eq!(
        unsafe { sys::cuLaunchHostFunc(s_use_handle, Some(hold_use), callback_owner) },
        sys::cudaError_enum::CUDA_SUCCESS
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    while !gate.entered.load(Ordering::Acquire) {
        assert!(
            Instant::now() < deadline,
            "use-stream callback never started"
        );
        std::thread::sleep(Duration::from_millis(1));
    }

    // This write and the dependency event are queued behind the held callback.
    // Reclamation must therefore remain blocked until the callback is released.
    unsafe { memset_async(s_use_handle, block.ptr, PATTERN_USE, BYTES) };
    resource
        .record_block_use(&block, s_use_id)
        .expect("record managed cross-stream use");
    resource
        .deallocate(block)
        .expect("detach managed-use block");
    assert_eq!(resource.live_bytes(), 0);
    assert_eq!(resource.pending_free_bytes(), BYTES);

    std::thread::scope(|scope| {
        let (started_tx, started_rx) = mpsc::channel();
        let (returned_tx, returned_rx) = mpsc::channel();
        let resource = &resource;
        let worker = scope.spawn(move || {
            started_tx.send(()).unwrap();
            let result = resource.reap_pending();
            returned_tx.send(()).unwrap();
            result
        });
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let early = returned_rx.recv_timeout(Duration::from_millis(250));
        drop(release);
        worker.join().unwrap().expect("reap managed-use block");
        assert!(
            matches!(early, Err(mpsc::RecvTimeoutError::Timeout)),
            "physical reclamation returned before the registered use completed: {early:?}"
        );
    });

    s_use
        .synchronize()
        .expect("synchronize released use stream");
    assert_eq!(resource.pending_free_bytes(), 0);
}

/// **Unmanaged-uses path: kept `#[ignore]`d, documents the
/// contract.**
///
/// Same shape as the managed test, but the caller submits
/// cross-stream work and **does NOT** call `record_block_use`.
/// The resource has no way to infer arbitrary external CUDA work
/// — the cross-stream pending memset is invisible to it — so the
/// pool can return the address to a subsequent allocate while
/// the cross-stream write is still in flight, and corruption is
/// observed.
///
/// This test exists to lock the contract documented on
/// `record_block_use`: callers that submit raw CUDA work on a
/// stream other than `block.alloc_stream` and bypass xlog's
/// launch-builder / use-recording layer are responsible for
/// their own cross-stream synchronization. If they neither use
/// `record_block_use` nor synchronize manually, lifetime safety
/// is undefined by design.
///
/// Kept `#[ignore]`d because:
///   * It demonstrates the *expected* unsafe behavior of an
///     unmanaged caller — running it as part of default CI would
///     turn the suite red.
///   * The corruption observable here is by design; a future
///     change that "fixes" this path *automatically* by tracking
///     all CUDA work the resource never saw would be wrong, and
///     this test prevents that drift.
///
/// To verify the unmanaged path still corrupts:
///   `cargo test -p xlog-cuda --release --test \
///    test_runtime_cross_stream_use_after_free -- --ignored \
///    unmanaged_cross_stream_use`
#[test]
#[ignore = "documents the unmanaged-raw-CUDA-call contract; corruption is the *intended* outcome here"]
fn unmanaged_cross_stream_use_corrupts_reuse_by_design() {
    let Some(device) = CudaDevice::new(0).ok().map(Arc::new) else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let resource = AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool));

    let s_alloc_id = match pool.acquire() {
        Ok(id) => id,
        Err(e) => {
            eprintln!("Skipping: StreamPool::acquire failed: {}", e);
            return;
        }
    };
    let s_use_id = match pool.acquire() {
        Ok(id) => id,
        Err(e) => {
            eprintln!("Skipping: StreamPool::acquire (second) failed: {}", e);
            return;
        }
    };
    let s_alloc = pool.resolve(s_alloc_id).expect("s_alloc resolves");
    let s_use = pool.resolve(s_use_id).expect("s_use resolves");
    let s_use_handle = s_use.cu_stream();

    const ITERATIONS: usize = 64;
    const PATTERN_USE: u8 = 0xCD;
    const PATTERN_B: u8 = 0xBB;

    let mut last_writer_was_b = 0usize;
    let mut last_writer_was_use = 0usize;
    let mut reuse_observed = 0usize;

    for _ in 0..ITERATIONS {
        let block_a = resource
            .allocate(BYTES, s_alloc_id, AllocTag("rep-A"))
            .expect("alloc A");
        let ptr_a = block_a.ptr;

        // Add bounded work on the allocation stream so a reused address still
        // has a competing writer when this opt-in diagnostic is exercised.
        unsafe { memset_async(s_alloc.cu_stream(), ptr_a, 0x00, BYTES) };
        unsafe { memset_async(s_use_handle, ptr_a, PATTERN_USE, BYTES) };

        // Note: NO record_block_use call. The cross-stream memset
        // is invisible to the resource.
        resource.deallocate(block_a).expect("dealloc A");

        let block_b = resource
            .allocate(BYTES, s_alloc_id, AllocTag("rep-B"))
            .expect("alloc B");
        let ptr_b = block_b.ptr;
        if ptr_b == ptr_a {
            reuse_observed += 1;
        }

        unsafe { memset_async(s_alloc.cu_stream(), ptr_b, PATTERN_B, BYTES) };

        // Drain s_alloc first. Without a registered dependency, work queued on
        // this stream does not wait for s_use's memset.
        s_alloc.synchronize().expect("sync s_alloc");
        // Now drain s_use — its memset of PATTERN_USE finally
        // lands. If ptr_b == ptr_a, it overwrites PATTERN_B.
        s_use.synchronize().expect("sync s_use");

        let mut readback = vec![0u8; BYTES];
        unsafe { dtoh_sync(&mut readback, ptr_b) };
        if ptr_b == ptr_a {
            if readback[0] == PATTERN_USE && readback[BYTES - 1] == PATTERN_USE {
                last_writer_was_use += 1;
            } else if readback[0] == PATTERN_B && readback[BYTES - 1] == PATTERN_B {
                last_writer_was_b += 1;
            }
        }

        resource.deallocate(block_b).expect("dealloc B");
        resource.reap_pending().expect("reap");
    }

    eprintln!(
        "[unmanaged] iterations={} reuse_observed={} \
         last_writer_was_b={} last_writer_was_use={}",
        ITERATIONS, reuse_observed, last_writer_was_b, last_writer_was_use
    );

    assert!(
        reuse_observed > 0,
        "address reuse never observed across {} iterations; the \
         test cannot exercise the unmanaged-uses contract",
        ITERATIONS
    );
    // The unmanaged contract: at least one iteration must show
    // s_use's memset as the last writer. We don't require all
    // iterations to corrupt — driver scheduling can let s_alloc
    // win the race occasionally — but if NONE corrupt, the
    // documented contract has somehow been "fixed" outside the
    // managed path, which would be an unrelated bug.
    assert!(
        last_writer_was_use > 0,
        "unmanaged-uses test expected at least one iteration where \
         s_use's late memset clobbered B (proving the documented \
         contract still applies), but observed 0/{} \
         (reuse_observed={}, last_writer_was_b={}). Either driver \
         scheduling shifted unexpectedly, or some unrelated change \
         silently fixed this path; investigate before re-enabling.",
        ITERATIONS,
        reuse_observed,
        last_writer_was_b,
    );
}
