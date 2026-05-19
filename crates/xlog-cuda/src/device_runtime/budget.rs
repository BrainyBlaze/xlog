//! [`GlobalDeviceBudget`] — per-runtime byte-limit decorator.
//!
//! Wraps a [`DeviceMemoryResource`] and enforces a single byte limit
//! across all allocations that flow through it. Designed to be the
//! per-runtime singleton replacement for the v0.5 per-provider
//! `GpuMemoryManager` (which had no way to enforce a coherent budget
//! across parallel tests, multiple providers, or Python callers
//! sharing one physical GPU).
//!
//! # Accounting model
//!
//! `GlobalDeviceBudget` keeps `reserved_bytes` strictly equal to
//! `inner.bytes_outstanding()` at every quiescent moment. This is the
//! "live + retired-but-not-yet-freed" view from the trait — exactly
//! the bytes the budget should be guarding.
//!
//! To keep that invariant under both synchronous and stream-ordered
//! async inners, every public method is serialized through a single
//! `Mutex<BudgetState>` and the inner call is invoked **inside** the
//! lock. The lock window is bounded by the inner's CUDA call, which
//! is in any case the dominant cost — the budget decorator does not
//! add hot-path overhead beyond what the inner already imposes.
//!
//! ## Allocate
//!
//!   1. Lock state.
//!   2. If `reserved + bytes > limit`: return
//!      `ResourceError::OutOfBudget { requested, remaining }`.
//!   3. Optimistically reserve: `reserved += bytes`.
//!   4. Call `inner.allocate(bytes, ..)` under the lock. The inner's
//!      own bookkeeping moves `bytes` from "free" to "live".
//!   5. If inner returned `Err`, roll back the reservation:
//!      `reserved -= bytes`. Forward the error.
//!
//! ## Deallocate / Reap
//!
//! For both methods we sample `inner.bytes_outstanding()` before and
//! after the inner call (under the lock), and decrement `reserved`
//! by the observed delta. The pattern handles both backends without
//! branching:
//!
//!   * Synchronous inner (`DirectCudaResource`): `bytes_outstanding`
//!     drops by the block's bytes on `deallocate`, so the delta is
//!     `block.bytes`. `reap_pending` is a no-op (delta zero).
//!   * Stream-ordered async inner (`AsyncCudaResource`): `deallocate`
//!     moves bytes from "live" to "pending"; `bytes_outstanding`
//!     stays the same, so the delta is zero — the budget is *not*
//!     released yet. `reap_pending` drains the pending bytes whose
//!     queued `cuMemFreeAsync` has completed; `bytes_outstanding`
//!     drops by the drained total and the budget releases that
//!     same total.
//!
//! Because the inner call and the before/after samples happen under
//! the same lock, no concurrent budget op can perturb the inner's
//! `bytes_outstanding` between our reads — the delta strictly
//! reflects this call's effect on the inner.
//!
//! # Composition
//!
//! `GlobalDeviceBudget` is a normal `DeviceMemoryResource`, so it
//! plugs into [`XlogDeviceRuntime::with_resource`] and stacks under
//! / over [`LoggingResource`]. Recommended ordering for production:
//! `GlobalDeviceBudget(LoggingResource(AsyncCudaResource))`. That
//! gives the budget atomic accounting, the logger sees the
//! eventually-applied call (so `OutOfBudget` errors do not get
//! double-logged), and the underlying allocator is reached last.
//! Tests can stack either way.

use std::sync::Mutex;

use super::resource::{
    Access, AllocTag, BlockId, DeviceBlock, DeviceMemoryResource, ResourceError, ResourceResult,
    StreamId,
};

/// Internal state guarded by the budget mutex. Kept in its own
/// struct so the lock guard syntactically scopes all updates.
struct BudgetState {
    reserved: usize,
}

/// Per-runtime byte-limit decorator.
pub struct GlobalDeviceBudget {
    inner: Box<dyn DeviceMemoryResource + Send + Sync>,
    limit: usize,
    state: Mutex<BudgetState>,
}

impl GlobalDeviceBudget {
    /// Wrap `inner` with a hard `limit` in bytes. The initial
    /// reserved tally is sampled from `inner.bytes_outstanding()`
    /// so callers may compose around an inner that already has live
    /// allocations — though in practice the decorator is installed
    /// before any allocation flows through it.
    pub fn new(inner: Box<dyn DeviceMemoryResource + Send + Sync>, limit: usize) -> Self {
        let initial = inner.bytes_outstanding();
        Self {
            inner,
            limit,
            state: Mutex::new(BudgetState { reserved: initial }),
        }
    }

    /// Hard byte limit. Set at construction; not adjustable.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Bytes currently reserved against the budget (live + pending
    /// async free). Matches `inner.bytes_outstanding()` at every
    /// quiescent moment.
    pub fn reserved_bytes(&self) -> usize {
        self.state
            .lock()
            .expect("GlobalDeviceBudget poisoned")
            .reserved
    }

    /// Headroom in bytes for the next allocation. Equal to
    /// `limit - reserved_bytes`, saturating at zero.
    pub fn remaining(&self) -> usize {
        let state = self.state.lock().expect("GlobalDeviceBudget poisoned");
        self.limit.saturating_sub(state.reserved)
    }
}

impl DeviceMemoryResource for GlobalDeviceBudget {
    fn allocate(
        &self,
        bytes: usize,
        stream: StreamId,
        tag: AllocTag,
    ) -> ResourceResult<DeviceBlock> {
        // First-pass reservation attempt under the budget lock.
        // If the request fits, reserve and forward to the inner
        // immediately.
        {
            let mut state = self.state.lock().expect("GlobalDeviceBudget poisoned");
            let remaining = self.limit.saturating_sub(state.reserved);
            if bytes <= remaining {
                state.reserved = state.reserved.saturating_add(bytes);
                drop(state);
                return match self.inner.allocate(bytes, stream, tag) {
                    Ok(block) => Ok(block),
                    Err(e) => {
                        let mut state = self.state.lock().expect("GlobalDeviceBudget poisoned");
                        state.reserved = state.reserved.saturating_sub(bytes);
                        Err(e)
                    }
                };
            }
            // Genuinely oversized requests can never fit even
            // after a reap. Short-circuit before touching the
            // inner stack so the rejection stays cheap and does
            // not emit a reap log record.
            if bytes > self.limit {
                return Err(ResourceError::OutOfBudget {
                    requested: bytes,
                    remaining,
                });
            }
        }

        // Second pass: reservation didn't fit. With the
        // stream-ordered async backend, dropped buffers transit
        // through `pending_per_stream` until `reap_pending` runs;
        // their bytes still count against `state.reserved`. Tight
        // allocate-then-drop loops (cert hardware sustained
        // tests; recursive Datalog inner loops without explicit
        // reap) hit this even when the GPU has plenty of free
        // memory. Drain pending frees once and retry. If the
        // retry still fails, the budget is genuinely exhausted
        // and the caller should see `OutOfBudget` as before.
        //
        // Reap is performed WITHOUT holding `state` so the inner
        // resource's own locks can run; reap itself updates
        // `state.reserved` via `Self::reap_pending` (which takes
        // the lock) so a concurrent racing allocate sees the
        // freed bytes.
        let _ = self.reap_pending();

        let mut state = self.state.lock().expect("GlobalDeviceBudget poisoned");
        let remaining = self.limit.saturating_sub(state.reserved);
        if bytes > remaining {
            return Err(ResourceError::OutOfBudget {
                requested: bytes,
                remaining,
            });
        }
        state.reserved = state.reserved.saturating_add(bytes);
        drop(state);

        match self.inner.allocate(bytes, stream, tag) {
            Ok(block) => Ok(block),
            Err(e) => {
                let mut state = self.state.lock().expect("GlobalDeviceBudget poisoned");
                state.reserved = state.reserved.saturating_sub(bytes);
                Err(e)
            }
        }
    }

    fn deallocate(&self, block: DeviceBlock) -> ResourceResult<()> {
        let mut state = self.state.lock().expect("GlobalDeviceBudget poisoned");

        let before = self.inner.bytes_outstanding();
        let result = self.inner.deallocate(block);
        let after = self.inner.bytes_outstanding();
        let freed = before.saturating_sub(after);
        if freed > 0 {
            state.reserved = state.reserved.saturating_sub(freed);
        }
        result
    }

    fn device_ordinal(&self) -> u32 {
        self.inner.device_ordinal()
    }

    fn bytes_outstanding(&self) -> usize {
        // Authoritative view is the inner's. We could return our
        // own `reserved` instead, but matching the inner sidesteps
        // any transient skew during error rollback.
        self.inner.bytes_outstanding()
    }

    fn reap_pending(&self) -> ResourceResult<()> {
        let mut state = self.state.lock().expect("GlobalDeviceBudget poisoned");

        let before = self.inner.bytes_outstanding();
        let result = self.inner.reap_pending();
        let after = self.inner.bytes_outstanding();
        let freed = before.saturating_sub(after);
        if freed > 0 {
            state.reserved = state.reserved.saturating_sub(freed);
        }
        result
    }

    fn record_block_use(&self, block: &DeviceBlock, use_stream: StreamId) -> ResourceResult<()> {
        // Pass-through: budget enforcement does not affect
        // cross-stream lifetime tracking; the inner resource (the
        // stream-ordered backend) is the only layer that owns
        // last-use events.
        self.inner.record_block_use(block, use_stream)
    }

    fn supports_block_use_tracking(&self) -> bool {
        self.inner.supports_block_use_tracking()
    }

    fn prepare_block_use(
        &self,
        block: BlockId,
        use_stream: StreamId,
        access: Access,
    ) -> ResourceResult<()> {
        // Pass-through: cross-stream waits live in the
        // stream-ordered backend; budget accounting is unaffected.
        self.inner.prepare_block_use(block, use_stream, access)
    }

    fn finish_block_use(
        &self,
        block: BlockId,
        use_stream: StreamId,
        access: Access,
    ) -> ResourceResult<()> {
        // Pass-through: see prepare_block_use rationale above.
        self.inner.finish_block_use(block, use_stream, access)
    }
}

#[cfg(test)]
mod tests {
    use super::super::async_resource::AsyncCudaResource;
    use super::super::direct::DirectCudaResource;
    use super::super::resource::{BlockState, Generation};
    use super::super::stream_pool::StreamPool;
    use super::*;
    use std::sync::Arc;

    use crate::CudaDevice;

    fn try_device() -> Option<Arc<CudaDevice>> {
        CudaDevice::new(0).ok().map(Arc::new)
    }

    /// Test fixture that always fails `allocate` so we can exercise
    /// the rollback path without touching CUDA. `deallocate` and
    /// `reap_pending` are no-ops; `bytes_outstanding` reflects an
    /// internally tracked tally so the budget's delta-sampling logic
    /// is also exercised.
    struct AlwaysFailAllocResource {
        ord: u32,
        outstanding: std::sync::atomic::AtomicUsize,
    }

    impl AlwaysFailAllocResource {
        fn new(ord: u32) -> Self {
            Self {
                ord,
                outstanding: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    impl DeviceMemoryResource for AlwaysFailAllocResource {
        fn allocate(
            &self,
            _bytes: usize,
            _stream: StreamId,
            _tag: AllocTag,
        ) -> ResourceResult<DeviceBlock> {
            Err(ResourceError::Driver("inner always fails".into()))
        }
        fn deallocate(&self, _block: DeviceBlock) -> ResourceResult<()> {
            Ok(())
        }
        fn device_ordinal(&self) -> u32 {
            self.ord
        }
        fn bytes_outstanding(&self) -> usize {
            self.outstanding.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    #[test]
    fn allocate_within_limit_succeeds_and_updates_reserved() {
        let Some(device) = try_device() else {
            return;
        };
        let inner = Box::new(DirectCudaResource::new(Arc::clone(&device), 0));
        let budget = GlobalDeviceBudget::new(inner, 64 * 1024);

        let block = budget
            .allocate(2048, StreamId::DEFAULT, AllocTag("budget-success"))
            .expect("alloc within limit");
        assert_eq!(budget.reserved_bytes(), 2048);
        assert_eq!(budget.remaining(), 64 * 1024 - 2048);
        assert_eq!(budget.bytes_outstanding(), 2048);

        budget.deallocate(block).expect("dealloc");
        assert_eq!(budget.reserved_bytes(), 0);
        assert_eq!(budget.bytes_outstanding(), 0);
    }

    #[test]
    fn allocate_at_exact_limit_succeeds_then_next_byte_rejected() {
        let Some(device) = try_device() else {
            return;
        };
        let inner = Box::new(DirectCudaResource::new(Arc::clone(&device), 0));
        let budget = GlobalDeviceBudget::new(inner, 4096);

        let block = budget
            .allocate(4096, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .expect("alloc at exact limit");
        assert_eq!(budget.reserved_bytes(), 4096);
        assert_eq!(budget.remaining(), 0);

        let err = budget.allocate(1, StreamId::DEFAULT, AllocTag::UNTAGGED);
        assert!(
            matches!(
                err,
                Err(ResourceError::OutOfBudget {
                    requested: 1,
                    remaining: 0
                })
            ),
            "expected OutOfBudget {{1,0}}, got {:?}",
            err
        );
        // Failed alloc must not perturb reserved.
        assert_eq!(budget.reserved_bytes(), 4096);

        budget.deallocate(block).expect("dealloc");
        assert_eq!(budget.reserved_bytes(), 0);
    }

    #[test]
    fn over_limit_alloc_returns_out_of_budget_with_correct_remaining() {
        let Some(device) = try_device() else {
            return;
        };
        let inner = Box::new(DirectCudaResource::new(Arc::clone(&device), 0));
        let budget = GlobalDeviceBudget::new(inner, 1024);

        // First alloc takes 768 bytes → 256 remaining.
        let block = budget
            .allocate(768, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .expect("first alloc");
        assert_eq!(budget.remaining(), 256);

        let err = budget.allocate(512, StreamId::DEFAULT, AllocTag::UNTAGGED);
        assert!(
            matches!(
                err,
                Err(ResourceError::OutOfBudget {
                    requested: 512,
                    remaining: 256
                })
            ),
            "expected OutOfBudget {{512,256}}, got {:?}",
            err
        );

        budget.deallocate(block).expect("dealloc");
    }

    #[test]
    fn failed_inner_allocation_rolls_back_reservation() {
        // No CUDA dependency — the fake inner always errors.
        let inner = Box::new(AlwaysFailAllocResource::new(0));
        let budget = GlobalDeviceBudget::new(inner, 1024 * 1024);
        assert_eq!(budget.reserved_bytes(), 0);

        let err = budget.allocate(2048, StreamId::DEFAULT, AllocTag::UNTAGGED);
        assert!(matches!(err, Err(ResourceError::Driver(_))));
        // Reservation must be rolled back: no live or pending bytes
        // landed on the inner, so reserved stays at the pre-call
        // value (0).
        assert_eq!(budget.reserved_bytes(), 0);
        assert_eq!(budget.remaining(), 1024 * 1024);
    }

    #[test]
    fn deallocate_releases_budget_immediately_for_synchronous_inner() {
        // DirectCudaResource is treated as synchronous from the
        // budget's perspective: bytes_outstanding drops at
        // deallocate time, so the delta-based release fires there.
        let Some(device) = try_device() else {
            return;
        };
        let inner = Box::new(DirectCudaResource::new(Arc::clone(&device), 0));
        let budget = GlobalDeviceBudget::new(inner, 16 * 1024);

        let block = budget
            .allocate(8 * 1024, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .expect("alloc");
        assert_eq!(budget.reserved_bytes(), 8 * 1024);
        budget.deallocate(block).expect("dealloc");
        assert_eq!(
            budget.reserved_bytes(),
            0,
            "synchronous inner releases budget at deallocate"
        );
        // reap is a no-op for sync inners; budget unchanged.
        budget.reap_pending().expect("reap noop");
        assert_eq!(budget.reserved_bytes(), 0);
    }

    #[test]
    fn deallocate_holds_budget_for_async_inner_until_reap_pending() {
        let Some(device) = try_device() else {
            return;
        };
        let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let inner = Box::new(AsyncCudaResource::new(
            Arc::clone(&device),
            0,
            Arc::clone(&pool),
        ));
        let budget = GlobalDeviceBudget::new(inner, 32 * 1024);

        let block = budget
            .allocate(4096, StreamId::DEFAULT, AllocTag("budget-async"))
            .expect("alloc");
        assert_eq!(budget.reserved_bytes(), 4096);

        // After deallocate the cuMemFreeAsync is queued but not
        // drained; bytes_outstanding still shows 4096 (live → pending),
        // so the budget MUST NOT release yet.
        budget.deallocate(block).expect("dealloc");
        assert_eq!(
            budget.reserved_bytes(),
            4096,
            "async inner: budget must stay reserved until reap_pending drains pending free"
        );
        assert_eq!(budget.bytes_outstanding(), 4096);

        budget.reap_pending().expect("reap");
        assert_eq!(
            budget.reserved_bytes(),
            0,
            "async inner: reap_pending releases the pending bytes"
        );
        assert_eq!(budget.bytes_outstanding(), 0);
    }

    #[test]
    fn deallocate_unknown_block_does_not_release_budget() {
        let Some(device) = try_device() else {
            return;
        };
        let inner = Box::new(DirectCudaResource::new(Arc::clone(&device), 0));
        let budget = GlobalDeviceBudget::new(inner, 16 * 1024);

        let block = budget
            .allocate(2048, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .expect("alloc");
        assert_eq!(budget.reserved_bytes(), 2048);

        // Bogus block — inner returns UseAfterFree without freeing
        // anything; budget must not move.
        let bogus = DeviceBlock {
            ptr: 0xfeed_face,
            device_ordinal: 0,
            alloc_stream: StreamId::DEFAULT,
            bytes: 1024,
            align: 1,
            tag: AllocTag::UNTAGGED,
            generation: Generation::next(),
            state: BlockState::Live,
        };
        let res = budget.deallocate(bogus);
        assert!(matches!(res, Err(ResourceError::UseAfterFree { .. })));
        assert_eq!(
            budget.reserved_bytes(),
            2048,
            "bogus dealloc must not release budget"
        );

        budget.deallocate(block).expect("real dealloc");
        assert_eq!(budget.reserved_bytes(), 0);
    }

    #[test]
    fn forwards_device_ordinal() {
        let inner = Box::new(AlwaysFailAllocResource::new(7));
        let budget = GlobalDeviceBudget::new(inner, 1024);
        assert_eq!(budget.device_ordinal(), 7);
    }
}
