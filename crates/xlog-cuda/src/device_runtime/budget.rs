//! Per-runtime budget over the canonical memory resource.
//!
//! Admission reserves bytes before allocation. Definite refusal and pre-malloc
//! unwind roll back. Acquired storage carries its exact release ticket through
//! ordinary and cold reclamation. Cumulative proven-release events restore
//! capacity even when concurrent allocations or out-of-band reaping occur.

use std::sync::{Arc, Mutex};

#[cfg(test)]
use super::resource::AllocTag;
use super::resource::{
    Access, AllocationAccounting, AllocationRequest, BlockId, DeviceBlock, DeviceMemoryResource,
    ResourceBudgetSnapshot, ResourceError, ResourceResult, StreamId,
};

/// Internal state guarded by the budget mutex. Kept in its own
/// struct so the lock guard syntactically scopes all updates.
struct BudgetState {
    admitted_total: u128,
    baseline_reclaimed: u128,
}

impl BudgetState {
    fn reserved(&self, total_reclaimed: u128) -> ResourceResult<usize> {
        let reclaimed = total_reclaimed
            .checked_sub(self.baseline_reclaimed)
            .ok_or_else(|| {
                ResourceError::Driver("resource reclamation total precedes budget baseline".into())
            })?;
        let reserved = self.admitted_total.checked_sub(reclaimed).ok_or_else(|| {
            ResourceError::Driver("resource releases exceed budget admissions".into())
        })?;
        usize::try_from(reserved)
            .map_err(|_| ResourceError::Driver("global budget accounting overflow".into()))
    }
}

struct BudgetReservation<'a> {
    state: &'a Mutex<BudgetState>,
    bytes: usize,
    reclamation: Arc<crate::memory::AllocationReclamation>,
}

impl Drop for BudgetReservation<'_> {
    fn drop(&mut self) {
        if !self.reclamation.was_acquired() {
            let mut state = self.state.lock().expect("GlobalDeviceBudget poisoned");
            state.admitted_total = state
                .admitted_total
                .checked_sub(self.bytes as u128)
                .expect("budget admission includes the refused allocation");
        }
    }
}

/// Per-runtime byte-limit decorator.
pub struct GlobalDeviceBudget {
    inner: Box<dyn DeviceMemoryResource + Send + Sync>,
    limit: usize,
    state: Mutex<BudgetState>,
    accounting: Arc<AllocationAccounting>,
}

impl GlobalDeviceBudget {
    /// Wrap `inner` with a hard `limit` in bytes. The initial
    /// reserved tally and release baseline are sampled atomically from the
    /// inner resource's canonical accounting ledger,
    /// so callers may compose around an inner that already has live
    /// allocations — though in practice the decorator is installed
    /// before any allocation flows through it.
    pub fn new(inner: Box<dyn DeviceMemoryResource + Send + Sync>, limit: usize) -> Self {
        let accounting = inner.allocation_accounting();
        let (initial, baseline_reclaimed) = accounting.snapshot();
        Self {
            inner,
            limit,
            state: Mutex::new(BudgetState {
                admitted_total: initial as u128,
                baseline_reclaimed,
            }),
            accounting,
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
        let state = self.state.lock().expect("GlobalDeviceBudget poisoned");
        state
            .reserved(self.accounting.snapshot().1)
            .expect("canonical budget release accounting is consistent")
    }

    /// Headroom in bytes for the next allocation. Equal to
    /// `limit - reserved_bytes`, saturating at zero.
    pub fn remaining(&self) -> usize {
        self.limit.saturating_sub(self.reserved_bytes())
    }

    fn allocate_with_pressure(&self, request: AllocationRequest) -> ResourceResult<DeviceBlock> {
        let bytes = request.bytes;
        let reservation_pressure_bytes = request.reservation_pressure_bytes;
        // Admit against both materialized allocations and runtime promises.
        // Only materialized bytes are added to this decorator's own tally.
        {
            let mut state = self.state.lock().expect("GlobalDeviceBudget poisoned");
            let current = state
                .reserved(self.accounting.snapshot().1)?
                .checked_add(reservation_pressure_bytes)
                .ok_or_else(|| {
                    ResourceError::Driver(
                        "global budget reservation accounting overflow".to_string(),
                    )
                })?;
            let remaining = self.limit.saturating_sub(current);
            if bytes <= remaining {
                state.admitted_total =
                    state
                        .admitted_total
                        .checked_add(bytes as u128)
                        .ok_or_else(|| {
                            ResourceError::Driver("global budget accounting overflow".to_string())
                        })?;
                drop(state);
                return self.materialize_reserved(request);
            }
            if bytes > self.limit.saturating_sub(reservation_pressure_bytes) {
                return Err(ResourceError::OutOfBudget {
                    requested: bytes,
                    current,
                    remaining,
                    limit: self.limit,
                });
            }
        }

        // A request that could fit after retired asynchronous allocations are
        // reclaimed gets one reap-and-retry before it is rejected.
        let _ = self.reap_pending();

        let mut state = self.state.lock().expect("GlobalDeviceBudget poisoned");
        let current = state
            .reserved(self.accounting.snapshot().1)?
            .checked_add(reservation_pressure_bytes)
            .ok_or_else(|| {
                ResourceError::Driver("global budget reservation accounting overflow".to_string())
            })?;
        let remaining = self.limit.saturating_sub(current);
        if bytes > remaining {
            return Err(ResourceError::OutOfBudget {
                requested: bytes,
                current,
                remaining,
                limit: self.limit,
            });
        }
        state.admitted_total =
            state
                .admitted_total
                .checked_add(bytes as u128)
                .ok_or_else(|| {
                    ResourceError::Driver("global budget accounting overflow".to_string())
                })?;
        drop(state);

        self.materialize_reserved(request)
    }

    fn materialize_reserved(&self, request: AllocationRequest) -> ResourceResult<DeviceBlock> {
        let _reservation = BudgetReservation {
            state: &self.state,
            bytes: request.bytes,
            reclamation: request.reclamation(),
        };
        self.inner.materialize(request)
    }
}

impl DeviceMemoryResource for GlobalDeviceBudget {
    fn materialize(&self, request: AllocationRequest) -> ResourceResult<DeviceBlock> {
        self.allocate_with_pressure(request)
    }

    fn allocation_accounting(&self) -> Arc<AllocationAccounting> {
        Arc::clone(&self.accounting)
    }

    fn deallocate(&self, block: DeviceBlock) -> ResourceResult<()> {
        self.inner.deallocate(block)
    }

    fn device_ordinal(&self) -> u32 {
        self.inner.device_ordinal()
    }

    fn bytes_outstanding(&self) -> usize {
        self.accounting.snapshot().0
    }

    fn budget_snapshot(&self) -> Option<ResourceBudgetSnapshot> {
        Some(ResourceBudgetSnapshot {
            limit: self.limit,
            reserved: self.reserved_bytes(),
        })
    }

    fn reap_pending(&self) -> ResourceResult<()> {
        self.inner.reap_pending()
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

    fn access_dependencies(
        &self,
        block: BlockId,
        bytes: usize,
    ) -> ResourceResult<Option<std::sync::Arc<super::resource::DeviceAccessDependencies>>> {
        self.inner.access_dependencies(block, bytes)
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
pub(crate) mod tests {
    use super::super::async_resource::AsyncCudaResource;
    use super::super::direct::DirectCudaResource;
    use super::super::resource::{BlockState, Generation};
    use super::super::stream_pool::StreamPool;
    use super::*;
    use std::sync::Arc;

    use crate::CudaDevice;

    fn try_device() -> Option<Arc<CudaDevice>> {
        match CudaDevice::new(0) {
            Ok(device) => Some(Arc::new(device)),
            Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
                panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
            }
            Err(error) => {
                eprintln!("Skipping test: CUDA runtime unavailable: {error}");
                None
            }
        }
    }

    /// Test fixture that always fails `allocate` so we can exercise
    /// the rollback path without touching CUDA. `deallocate` and
    /// `reap_pending` are no-ops; `bytes_outstanding` reflects an
    /// shared physical ledger, independently of admission rollback.
    struct AlwaysFailAllocResource {
        ord: u32,
        accounting: Arc<AllocationAccounting>,
    }

    impl AlwaysFailAllocResource {
        fn new(ord: u32) -> Self {
            Self {
                ord,
                accounting: Arc::default(),
            }
        }
    }

    impl DeviceMemoryResource for AlwaysFailAllocResource {
        fn materialize(&self, _request: AllocationRequest) -> ResourceResult<DeviceBlock> {
            Err(ResourceError::Driver("inner always fails".into()))
        }
        fn allocation_accounting(&self) -> Arc<AllocationAccounting> {
            Arc::clone(&self.accounting)
        }
        fn deallocate(&self, _block: DeviceBlock) -> ResourceResult<()> {
            Ok(())
        }
        fn device_ordinal(&self) -> u32 {
            self.ord
        }
        fn bytes_outstanding(&self) -> usize {
            self.accounting.snapshot().0
        }
    }

    pub(crate) struct HostAllocation {
        storage: Option<Box<[u8]>>,
        reclamation: Arc<crate::memory::AllocationReclamation>,
    }

    impl Drop for HostAllocation {
        fn drop(&mut self) {
            drop(self.storage.take());
            self.reclamation
                .complete()
                .expect("host allocation physically freed");
        }
    }

    #[derive(Default)]
    pub(crate) struct DeferredHostResource {
        pub(crate) owners: Arc<Mutex<std::collections::HashMap<u64, HostAllocation>>>,
        pub(crate) accounting: Arc<AllocationAccounting>,
        pub(crate) release_on_detach: bool,
    }

    impl DeviceMemoryResource for DeferredHostResource {
        fn materialize(&self, request: AllocationRequest) -> ResourceResult<DeviceBlock> {
            let AllocationRequest {
                bytes,
                stream,
                tag,
                reclamation,
                ..
            } = request;
            reclamation.attach_resource(Arc::clone(&self.accounting), bytes)?;
            let storage = vec![0_u8; bytes].into_boxed_slice();
            let ptr = storage.as_ptr() as u64;
            let owner = HostAllocation {
                storage: Some(storage),
                reclamation,
            };
            owner.reclamation.acquired()?;
            self.owners.lock().unwrap().insert(ptr, owner);
            Ok(DeviceBlock {
                ptr,
                device_ordinal: 0,
                alloc_stream: stream,
                bytes,
                align: 1,
                tag,
                generation: Generation::next(),
                state: BlockState::Live,
            })
        }

        fn deallocate(&self, block: DeviceBlock) -> ResourceResult<()> {
            assert!(self.owners.lock().unwrap().contains_key(&block.ptr));
            if self.release_on_detach {
                let owner = self.owners.lock().unwrap().remove(&block.ptr);
                drop(owner);
            }
            Ok(())
        }

        fn device_ordinal(&self) -> u32 {
            0
        }

        fn bytes_outstanding(&self) -> usize {
            self.accounting.snapshot().0
        }
        fn allocation_accounting(&self) -> Arc<AllocationAccounting> {
            Arc::clone(&self.accounting)
        }
    }

    #[test]
    fn host_release_outside_resource_calls_restores_budget_capacity() {
        let resource = DeferredHostResource::default();
        let owners = Arc::clone(&resource.owners);
        let budget = GlobalDeviceBudget::new(Box::new(resource), 64);
        let block = budget
            .allocate(64, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .unwrap();
        let ptr = block.ptr;
        budget.deallocate(block).unwrap();
        assert_eq!(budget.reserved_bytes(), 64);

        let owner = owners.lock().unwrap().remove(&ptr).unwrap();
        drop(owner);

        assert_eq!(budget.bytes_outstanding(), 0);
        assert_eq!(budget.reserved_bytes(), 0);
        assert_eq!(budget.remaining(), 64);
        let replacement = budget
            .allocate(64, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .expect("physical release restores admission capacity");
        budget.deallocate(replacement).unwrap();
    }

    #[test]
    fn host_budget_construction_preserves_live_bytes_and_prior_reclamation() {
        let resource = DeferredHostResource::default();
        let owners = Arc::clone(&resource.owners);
        let old = resource
            .allocate(16, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .unwrap();
        drop(owners.lock().unwrap().remove(&old.ptr));
        let live = resource
            .allocate(64, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .unwrap();
        let inner = GlobalDeviceBudget::new(Box::new(resource), 96);
        let outer = GlobalDeviceBudget::new(Box::new(inner), 96);
        assert_eq!(outer.reserved_bytes(), 64);
        assert!(outer
            .allocate_with_reservation_pressure(1, 32, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .is_err());
        assert_eq!(outer.reserved_bytes(), 64);
        drop(owners.lock().unwrap().remove(&live.ptr));
        assert_eq!(outer.reserved_bytes(), 0);
        let next = outer
            .allocate(96, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .unwrap();
        assert_eq!(outer.reserved_bytes(), 96);
        let owner = owners.lock().unwrap().remove(&next.ptr).unwrap();
        std::thread::spawn(move || drop(owner)).join().unwrap();
        assert_eq!(outer.remaining(), 96);
    }

    #[test]
    fn host_budget_new_admission_does_not_mask_concurrent_physical_release() {
        struct PausingResource {
            inner: DeferredHostResource,
            entered: Arc<std::sync::Barrier>,
            proceed: Arc<std::sync::Barrier>,
        }
        impl DeviceMemoryResource for PausingResource {
            fn materialize(&self, request: AllocationRequest) -> ResourceResult<DeviceBlock> {
                self.entered.wait();
                self.proceed.wait();
                self.inner.materialize(request)
            }
            fn allocation_accounting(&self) -> Arc<AllocationAccounting> {
                self.inner.allocation_accounting()
            }
            fn bytes_outstanding(&self) -> usize {
                self.inner.bytes_outstanding()
            }
            fn device_ordinal(&self) -> u32 {
                0
            }
            fn deallocate(&self, block: DeviceBlock) -> ResourceResult<()> {
                self.inner.deallocate(block)
            }
        }
        let inner = DeferredHostResource::default();
        let old = inner
            .allocate(64, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .unwrap();
        let owners = Arc::clone(&inner.owners);
        let entered = Arc::new(std::sync::Barrier::new(2));
        let proceed = Arc::new(std::sync::Barrier::new(2));
        let budget = Arc::new(GlobalDeviceBudget::new(
            Box::new(PausingResource {
                inner,
                entered: Arc::clone(&entered),
                proceed: Arc::clone(&proceed),
            }),
            128,
        ));
        let worker = Arc::clone(&budget);
        let admission = std::thread::spawn(move || {
            worker
                .allocate(64, StreamId::DEFAULT, AllocTag::UNTAGGED)
                .unwrap()
        });
        entered.wait();
        assert_eq!(budget.reserved_bytes(), 128);
        drop(owners.lock().unwrap().remove(&old.ptr));
        assert_eq!(budget.reserved_bytes(), 64);
        proceed.wait();
        let new = admission.join().unwrap();
        assert_eq!(budget.bytes_outstanding(), 64);
        assert_eq!(budget.reserved_bytes(), 64);
        drop(owners.lock().unwrap().remove(&new.ptr));
        assert_eq!(budget.remaining(), 128);
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
    fn failed_initialization_refunds_only_after_physical_release() {
        struct AllocatedThenFailed {
            inner: DeferredHostResource,
            unwind: bool,
            release_in_frame: bool,
        }
        impl DeviceMemoryResource for AllocatedThenFailed {
            fn materialize(&self, request: AllocationRequest) -> ResourceResult<DeviceBlock> {
                let reclamation = request.reclamation();
                let block = self.inner.materialize(request)?;
                if self.release_in_frame {
                    let owner = self.inner.owners.lock().unwrap().remove(&block.ptr);
                    drop(owner);
                }
                if self.unwind {
                    panic!("initialization interrupted after acquisition");
                }
                Err(ResourceError::Driver("ready event failed".into())
                    .retaining(block.bytes, reclamation))
            }
            fn allocation_accounting(&self) -> Arc<AllocationAccounting> {
                self.inner.allocation_accounting()
            }
            fn deallocate(&self, _: DeviceBlock) -> ResourceResult<()> {
                unreachable!()
            }
            fn device_ordinal(&self) -> u32 {
                0
            }
            fn bytes_outstanding(&self) -> usize {
                self.inner.bytes_outstanding()
            }
        }
        for unwind in [false, true] {
            for release_in_frame in [false, true] {
                let inner = DeferredHostResource::default();
                let owners = Arc::clone(&inner.owners);
                let budget = GlobalDeviceBudget::new(
                    Box::new(AllocatedThenFailed {
                        inner,
                        unwind,
                        release_in_frame,
                    }),
                    64,
                );
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    assert!(budget
                        .allocate(64, StreamId::DEFAULT, AllocTag::UNTAGGED)
                        .is_err());
                }));
                assert_eq!(outcome.is_err(), unwind);
                assert_eq!(
                    budget.reserved_bytes(),
                    if release_in_frame { 0 } else { 64 }
                );
                let pending = std::mem::take(&mut *owners.lock().unwrap());
                std::thread::spawn(move || drop(pending)).join().unwrap();
                assert_eq!(budget.reserved_bytes(), 0);
                budget.reap_pending().unwrap();
                assert_eq!(budget.remaining(), 64);
            }
        }
    }

    #[test]
    fn allocation_unwind_before_acquisition_restores_budget_charge() {
        struct UnwindingResource(Arc<AllocationAccounting>);
        impl DeviceMemoryResource for UnwindingResource {
            fn materialize(&self, _: AllocationRequest) -> ResourceResult<DeviceBlock> {
                panic!("allocator refused before acquisition")
            }
            fn allocation_accounting(&self) -> Arc<AllocationAccounting> {
                Arc::clone(&self.0)
            }
            fn deallocate(&self, _: DeviceBlock) -> ResourceResult<()> {
                unreachable!()
            }
            fn device_ordinal(&self) -> u32 {
                0
            }
            fn bytes_outstanding(&self) -> usize {
                0
            }
        }
        let budget = GlobalDeviceBudget::new(Box::new(UnwindingResource(Arc::default())), 64);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = budget.allocate(64, StreamId::DEFAULT, AllocTag::UNTAGGED);
        }))
        .is_err());
        assert_eq!(budget.reserved_bytes(), 0);
        assert_eq!(budget.remaining(), 64);
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
                    current: 4096,
                    remaining: 0,
                    limit: 4096,
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
                    current: 768,
                    remaining: 256,
                    limit: 1024,
                })
            ),
            "expected OutOfBudget {{512,256}}, got {:?}",
            err
        );

        budget.deallocate(block).expect("dealloc");
    }

    #[test]
    fn memory_pressure_runtime_budget_reports_exact_limit() {
        let Some(device) = try_device() else {
            return;
        };
        let inner = Box::new(DirectCudaResource::new(Arc::clone(&device), 0));
        let budget = GlobalDeviceBudget::new(inner, 1024);
        let block = budget
            .allocate(768, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .expect("baseline allocation");

        let error = budget
            .allocate(512, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .expect_err("cumulative allocation must exceed the runtime budget");

        assert_eq!(
            format!("{error:?}"),
            "OutOfBudget { requested: 512, current: 768, remaining: 256, limit: 1024 }"
        );
        assert_eq!(budget.reserved_bytes(), 768);
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
    fn reservation_pressure_is_included_before_inner_allocation() {
        let inner = Box::new(AlwaysFailAllocResource::new(0));
        let budget = GlobalDeviceBudget::new(inner, 1024);

        let error = budget
            .allocate_with_reservation_pressure(600, 500, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .expect_err("runtime promises must reduce ordinary allocation headroom");
        assert!(matches!(
            error,
            ResourceError::OutOfBudget {
                requested: 600,
                current: 500,
                remaining: 524,
                limit: 1024,
            }
        ));
        assert_eq!(budget.reserved_bytes(), 0);
    }

    #[test]
    fn deallocate_releases_budget_immediately_for_synchronous_inner() {
        // DirectCudaResource is treated as synchronous from the
        // budget's perspective: physical release during deallocate updates
        // the shared ledger before the next admission snapshot.
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
