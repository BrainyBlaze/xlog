//! CUDA memory management
//!
//! This module provides GPU memory management with budget enforcement.
//! It wraps cudarc's allocation functions and tracks total allocated memory.

use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use cudarc::driver::{CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice, SyncOnDrop};
use xlog_core::{resolve_bool, MemoryBudget, Result, Schema, XlogError};

use crate::arrow_device::ArrowDeviceImport;
use crate::cuda_compat::{AsKernelParam, DeviceParamStorage, IntoKernelParamStorage};
use crate::device_runtime::resource::{
    block_use_registry, Access, AllocationAccounting, AllocationRequest, BlockUseRegistry,
    DeviceAccessDependencies, MemoryStorageOwner, MemoryUse, MemoryUseGroup, RetainedStorageUse,
};
use crate::device_runtime::{
    AllocTag, BlockId, BlockState, DeviceBlock, ResourceError, RuntimeMemoryReservation, StreamId,
    XlogDeviceRuntime,
};
use crate::dlpack::DlpackManagedTensor;
use crate::launch::RecorderTransaction;
use crate::CudaDevice;

#[cfg(test)]
type AfterLocalReservationHook = std::sync::Mutex<Option<Arc<dyn Fn(u64) + Send + Sync + 'static>>>;

/// Budget-enforced device allocation. Every view retains actual storage.
/// Runtime-backed managers use the canonical resource stack. Native allocations
/// share its raw owner: post-malloc failures retain storage and budget together.
pub struct GpuMemoryManager {
    /// The CUDA device for memory operations
    device: Arc<CudaDevice>,
    /// Memory budget configuration
    budget: MemoryBudget,
    /// Accounting shared by every allocation view over this budget.
    accounting: Arc<GpuMemoryAccounting>,
    /// Optional v0.6 device runtime. When set, [`alloc_raw`]
    /// reserves through the runtime's resource stack in addition
    /// to enforcing the local budget; both must accept for the
    /// allocation to proceed.
    runtime: Option<Arc<XlogDeviceRuntime>>,
    /// Unit-test seam used to pause a request after local reservation but
    /// before runtime admission. Production builds contain no hook.
    #[cfg(test)]
    after_local_reservation_hook: AfterLocalReservationHook,
}

#[derive(Default)]
struct GpuMemoryAccounting {
    /// Serializes accounting mutations that must validate multiple counters
    /// before changing any of them.
    mutation_lock: std::sync::Mutex<()>,
    /// Bytes reserved against the local budget, including requests that have
    /// passed the local guard but are still awaiting allocator admission.
    /// This counter is intentionally conservative so concurrent requests
    /// cannot oversubscribe the configured budget.
    budget_reserved: AtomicU64,
    /// Currently admitted bytes (tracked atomically for thread safety).
    /// Unlike `budget_reserved`, this excludes provisional and refused
    /// requests and is the value exposed by [`allocated_bytes`](Self::allocated_bytes).
    allocated: AtomicU64,
    /// High-water mark of successful manager-accounted reservations since
    /// construction or the last [`reset_peak`](Self::reset_peak). This is a
    /// reservation-lifetime metric, not a physical-memory measurement.
    peak: AtomicU64,
    /// Count of `alloc` calls (device allocation requests). Resettable; used by
    /// the GPU-resident MC engine's no-host gate to prove that **zero** device
    /// allocations happen inside the measured region (all arenas are allocated
    /// before it). Distinct from `allocated` (bytes).
    alloc_count: AtomicU64,
    /// Runtime deallocations that returned an error. Their bytes remain
    /// charged locally because physical release was not proven.
    deallocation_failure_count: AtomicU64,
    deallocation_failure_bytes: AtomicU64,
}

impl GpuMemoryAccounting {
    fn release_reserved(&self, bytes: u64) -> Result<()> {
        let _mutation = self.mutation_lock.lock().map_err(|_| {
            XlogError::Kernel("GPU memory accounting poisoned during reservation release".into())
        })?;
        let previous = self.budget_reserved.load(Ordering::SeqCst);
        let next = previous.checked_sub(bytes).ok_or_else(|| {
            XlogError::Kernel(format!(
                "GPU memory reservation release underflow: current_bytes={} requested_bytes={}",
                previous, bytes,
            ))
        })?;
        self.budget_reserved.store(next, Ordering::SeqCst);
        Ok(())
    }

    fn release_owned_allocation(&self, bytes: u64) -> Result<()> {
        let _mutation = self.mutation_lock.lock().map_err(|_| {
            XlogError::Kernel("GPU memory accounting poisoned during release".into())
        })?;
        let admitted = self.allocated.load(Ordering::SeqCst);
        let reserved = self.budget_reserved.load(Ordering::SeqCst);
        let next_admitted = admitted.checked_sub(bytes).ok_or_else(|| {
            XlogError::Kernel(format!(
                "GPU admitted allocation release underflow: current_bytes={} requested_bytes={}",
                admitted, bytes
            ))
        })?;
        let next_reserved = reserved.checked_sub(bytes).ok_or_else(|| {
            XlogError::Kernel(format!(
                "GPU local reservation release underflow: current_bytes={} requested_bytes={}",
                reserved, bytes
            ))
        })?;
        self.allocated.store(next_admitted, Ordering::SeqCst);
        self.budget_reserved.store(next_reserved, Ordering::SeqCst);
        Ok(())
    }
}

/// Accounting owned by one allocation attempt and then by its physical storage.
///
/// Resource implementations bind their accounting before malloc, call
/// [`Self::acquired`] immediately after arming the returned pointer, and retain
/// this owner until [`Self::confirm_physical_release`] settles physical release.
/// Refusal and unwind guards use the persistent acquisition proof, not the
/// shape of an error or whether a concurrent reaper already freed the storage.
#[derive(Default)]
pub struct AllocationReclamation {
    state: std::sync::Mutex<AllocationReclamationState>,
    acquired: std::sync::atomic::AtomicBool,
    released: Arc<std::sync::atomic::AtomicBool>,
}

#[derive(Default)]
struct AllocationReclamationState {
    charge: Option<LocalAllocationCharge>,
    local_published: bool,
    resource: Option<(Arc<AllocationAccounting>, usize)>,
    resource_published: bool,
    pending: Option<(Arc<std::sync::atomic::AtomicUsize>, usize)>,
}

struct LocalAllocationCharge {
    accounting: Arc<GpuMemoryAccounting>,
    bytes: u64,
}

impl AllocationReclamation {
    /// Transfer an already reserved local claim before any physical allocation.
    fn attach_local(
        &self,
        accounting: Arc<GpuMemoryAccounting>,
        bytes: u64,
    ) -> crate::device_runtime::ResourceResult<()> {
        let charge = LocalAllocationCharge { accounting, bytes };
        let mut state = self.state.lock().map_err(|_| {
            ResourceError::Driver(
                "allocation reclamation state poisoned during charge transfer".into(),
            )
        })?;
        if self.was_acquired() || self.was_released() || state.charge.is_some() {
            return Err(ResourceError::Driver(
                "allocation charge transfer requires an unreleased, unbound owner".into(),
            ));
        }
        state.charge = Some(charge);
        Ok(())
    }

    /// Bind the backend's exact accounting domain before calling malloc.
    pub fn attach_resource(
        &self,
        accounting: Arc<AllocationAccounting>,
        bytes: usize,
    ) -> crate::device_runtime::ResourceResult<()> {
        let mut state = self.state.lock().map_err(|_| {
            ResourceError::Driver("allocation accounting poisoned during binding".into())
        })?;
        if self.was_acquired() || self.was_released() || state.resource.is_some() {
            return Err(ResourceError::Driver(
                "resource accounting requires an unacquired, unbound allocation".into(),
            ));
        }
        state.resource = Some((accounting, bytes));
        Ok(())
    }

    /// Publish acquisition after the real pointer has an armed storage owner.
    /// Even an accounting error means storage exists: it must be retained for
    /// physical cleanup, never reported as a definite malloc refusal.
    pub fn acquired(&self) -> crate::device_runtime::ResourceResult<()> {
        self.acquired.store(true, Ordering::Release);
        let mut state = self.state.lock().map_err(|_| {
            ResourceError::Driver("allocation accounting poisoned during acquisition".into())
        })?;
        if self.was_released() {
            return Err(ResourceError::Driver(
                "cannot acquire released storage".into(),
            ));
        }
        if !state.resource_published {
            if let Some((accounting, bytes)) = &state.resource {
                accounting.acquire(*bytes)?;
                state.resource_published = true;
            }
        }
        if !state.local_published {
            if let Some(charge) = &state.charge {
                let _mutation = charge.accounting.mutation_lock.lock().map_err(|_| {
                    ResourceError::Driver("local allocation accounting poisoned".into())
                })?;
                let previous = charge.accounting.allocated.load(Ordering::SeqCst);
                let admitted = previous.checked_add(charge.bytes).ok_or_else(|| {
                    ResourceError::Driver("local admitted allocation accounting overflow".into())
                })?;
                charge
                    .accounting
                    .allocated
                    .store(admitted, Ordering::SeqCst);
                charge.accounting.peak.fetch_max(admitted, Ordering::SeqCst);
            }
            state.local_published = true;
        }
        Ok(())
    }

    /// Remains true after release, so an outer error/unwind cannot refund twice.
    pub fn was_acquired(&self) -> bool {
        self.acquired.load(Ordering::Acquire)
    }

    /// Transfer the physical-byte tally before logical detach. The exact raw
    /// owner settles it even if backend destruction or unwind bypasses its queue.
    pub(crate) fn attach_pending(
        &self,
        accounting: Arc<std::sync::atomic::AtomicUsize>,
        bytes: usize,
    ) -> crate::device_runtime::ResourceResult<()> {
        let mut state = self.state.lock().map_err(|_| {
            ResourceError::Driver("allocation accounting poisoned during detach".into())
        })?;
        if !self.was_acquired() || self.was_released() || state.pending.is_some() {
            return Err(ResourceError::Driver(
                "pending charge requires acquired live storage".into(),
            ));
        }
        accounting
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |value| {
                value.checked_add(bytes)
            })
            .map_err(|_| ResourceError::Driver("pending allocation accounting overflow".into()))?;
        state.pending = Some((accounting, bytes));
        Ok(())
    }

    /// Called only after the raw owner proves physical free. Accounting failure
    /// retains the exact charge for retry; no CUDA wait runs under this mutex.
    ///
    /// # Safety
    /// The exact allocation must have been physically freed after every device
    /// use completed. Keep its owner and this ticket until settlement succeeds.
    /// Publishing this proof for live storage can invalidate access exclusion.
    pub unsafe fn confirm_physical_release(&self) -> crate::device_runtime::ResourceResult<()> {
        self.complete()
    }

    // Internal storage owners already establish the physical-release boundary.
    // Custom resource implementations cross the unsafe public boundary above.
    pub(crate) fn complete(&self) -> crate::device_runtime::ResourceResult<()> {
        if self.was_acquired() && !self.was_released() {
            // Acquisition publication itself may have failed after malloc.
            // Complete its unsettled bookkeeping before recording the release.
            if let Err(error) = self.acquired() {
                // A concurrent completion may have settled while this caller
                // was acquiring the state lock. Its proof is authoritative.
                if self.was_released() {
                    return Ok(());
                }
                return Err(error);
            }
        }
        let (charge, resource, pending) = {
            let mut state = self.state.lock().map_err(|_| {
                ResourceError::Driver("allocation reclamation state poisoned during release".into())
            })?;
            if self.was_released() {
                return Ok(());
            }
            if let Some(charge) = &state.charge {
                if !state.local_published {
                    return Err(ResourceError::Driver(
                        "local allocation acquisition was not published".into(),
                    ));
                }
                charge
                    .accounting
                    .release_owned_allocation(charge.bytes)
                    .map_err(|error| ResourceError::Driver(error.to_string()))?;
            }
            // Remove each settled obligation before the next fallible step.
            // A later retry cannot refund an already settled local charge.
            let charge = state.charge.take();
            if let Some((accounting, bytes)) = &state.resource {
                if !state.resource_published {
                    return Err(ResourceError::Driver(
                        "resource allocation acquisition was not published".into(),
                    ));
                }
                accounting.release(*bytes)?;
            }
            let resource = state.resource.take();
            if let Some((accounting, bytes)) = &state.pending {
                accounting
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |value| {
                        value.checked_sub(*bytes)
                    })
                    .map_err(|_| {
                        ResourceError::Driver("pending allocation accounting underflow".into())
                    })?;
            }
            self.released.store(true, Ordering::Release);
            (charge, resource, state.pending.take())
        };
        drop((charge, resource, pending));
        Ok(())
    }

    pub(crate) fn was_released(&self) -> bool {
        self.released.load(Ordering::Acquire)
    }

    pub(crate) fn release_proof(&self) -> Arc<std::sync::atomic::AtomicBool> {
        Arc::clone(&self.released)
    }
}

/// Until malloc succeeds, a local claim belongs to this calling frame. A token
/// refusal restores unused token bytes; an ordinary refusal returns its budget.
/// After acquisition the raw owner, not this guard, settles the claim.
struct LocalAllocationAttempt<'a> {
    accounting: Arc<GpuMemoryAccounting>,
    bytes: u64,
    remaining: Option<&'a mut u64>,
    reclamation: Arc<AllocationReclamation>,
}

impl<'a> LocalAllocationAttempt<'a> {
    fn new(
        accounting: Arc<GpuMemoryAccounting>,
        bytes: u64,
        remaining: Option<&'a mut u64>,
    ) -> Self {
        let reclamation = Arc::new(AllocationReclamation::default());
        let attempt = Self {
            accounting,
            bytes,
            remaining,
            reclamation,
        };
        attempt
            .reclamation
            .attach_local(Arc::clone(&attempt.accounting), bytes)
            .expect("new allocation attempt has no bound charge");
        attempt
    }
}

impl Drop for LocalAllocationAttempt<'_> {
    fn drop(&mut self) {
        if self.reclamation.was_acquired() {
            return;
        }
        if let Some(remaining) = self.remaining.as_mut() {
            **remaining = remaining
                .checked_add(self.bytes)
                .expect("unused allocation reservation remains representable");
        } else if let Err(error) = self.accounting.release_reserved(self.bytes) {
            eprintln!("local allocation reservation rollback failed: {error}");
        }
    }
}

/// The backend remains the physical owner while a returned block is being
/// bound to a typed wrapper. Preserve logical-detach work through any error or
/// unwind, using the same cold queue as final storage retirement.
pub(crate) struct ResourceBlockRetirement {
    payload: Option<ResourceBlockRetirementPayload>,
}

struct ResourceBlockRetirementPayload {
    block: DeviceBlock,
    resource: Arc<dyn crate::device_runtime::DeviceMemoryResource + Send + Sync>,
    reclamation: Arc<AllocationReclamation>,
    detached: bool,
}

impl ResourceBlockRetirement {
    pub(crate) fn new(
        block: DeviceBlock,
        resource: Arc<dyn crate::device_runtime::DeviceMemoryResource + Send + Sync>,
        reclamation: Arc<AllocationReclamation>,
    ) -> Self {
        Self {
            payload: Some(ResourceBlockRetirementPayload {
                block,
                resource,
                reclamation,
                detached: false,
            }),
        }
    }

    pub(crate) fn block(&self) -> &DeviceBlock {
        &self
            .payload
            .as_ref()
            .expect("unpublished block present")
            .block
    }

    pub(crate) fn into_block(mut self) -> DeviceBlock {
        self.payload.take().expect("published block present").block
    }

    fn release(&mut self) -> crate::device_runtime::ResourceResult<()> {
        reclaim_allocation(&mut self.payload, |owner| {
            if owner.reclamation.was_released() {
                return Ok(());
            }
            let mut detach_error = None;
            let mut detach_panic = None;
            if !owner.detached {
                let block = &owner.block;
                // This is a retry of exact generation-checked logical detach,
                // not a second physical free. The backend owns that transition.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    owner.resource.deallocate(DeviceBlock {
                        ptr: block.ptr,
                        device_ordinal: block.device_ordinal,
                        alloc_stream: block.alloc_stream,
                        bytes: block.bytes,
                        align: block.align,
                        tag: block.tag,
                        generation: block.generation,
                        state: block.state,
                    })
                }));
                match result {
                    Ok(Ok(())) | Ok(Err(ResourceError::UseAfterFree { .. })) => {
                        owner.detached = true
                    }
                    Ok(Err(error)) => detach_error = Some(error),
                    Err(panic) => detach_panic = Some(panic),
                }
            }
            // A decorator can unwind after the backend accepted logical
            // detach. Still offer that exact backend its physical reap; retrying
            // only the decorator would strand an already-pending allocation.
            let reaped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                owner.resource.reap_pending()
            }));
            if let Some(panic) = detach_panic {
                // Both operations have been attempted. Preserve the first
                // failure even when a telemetry sink also unwinds during reap.
                drop(reaped);
                std::panic::resume_unwind(panic);
            }
            let reaped = match reaped {
                Ok(result) => result,
                Err(panic) => std::panic::resume_unwind(panic),
            };
            allocation_reclamation_outcome(
                owner.reclamation.was_released(),
                detach_error.map_or(reaped, Err),
            )
        })
    }
}

impl Drop for ResourceBlockRetirement {
    fn drop(&mut self) {
        if let Some(payload) = self.payload.take() {
            let mut owner = Self {
                payload: Some(payload),
            };
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::cuda_graph::retry_retirement_after_stream_captures(move || {
                    owner.release().is_ok()
                });
            }));
        }
    }
}

/// One atomic claim on a [`GpuMemoryManager`] budget.
///
/// The claim protects a complete multi-allocation request from competing
/// callers. Bytes remain reserved until they are transferred to returned
/// allocation owners or released when this token is dropped.
#[must_use = "dropping the reservation immediately releases its unused budget"]
pub struct GpuMemoryReservation {
    manager: Arc<GpuMemoryManager>,
    runtime_reservation: Option<RuntimeMemoryReservation>,
    total_bytes: u64,
    remaining_bytes: u64,
}

impl std::fmt::Debug for GpuMemoryReservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuMemoryReservation")
            .field("total_bytes", &self.total_bytes)
            .field("remaining_bytes", &self.remaining_bytes)
            .finish()
    }
}

impl GpuMemoryReservation {
    /// Stable address of the memory manager that admitted this reservation.
    pub fn memory_manager_ptr_value(&self) -> usize {
        Arc::as_ptr(&self.manager) as usize
    }

    /// Complete byte claim made when this reservation was created.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Bytes still available for materialization through this token.
    pub fn remaining_bytes(&self) -> u64 {
        self.remaining_bytes
    }

    /// Bytes already transferred to allocation owners.
    pub fn used_bytes(&self) -> u64 {
        self.total_bytes - self.remaining_bytes
    }

    /// Allocate typed device memory from this reservation.
    pub fn alloc<T: cudarc::driver::DeviceRepr>(
        &mut self,
        len: usize,
    ) -> Result<TrackedCudaSlice<T>> {
        self.manager
            .accounting
            .alloc_count
            .fetch_add(1, Ordering::Relaxed);
        let bytes = (len as u64)
            .checked_mul(std::mem::size_of::<T>() as u64)
            .ok_or_else(|| XlogError::Kernel("Allocation size overflow".to_string()))?;
        let used = self.used_bytes();
        if bytes > self.remaining_bytes {
            return Err(MemoryPressure {
                layer: "manager_reservation_alloc",
                current_bytes: used as u128,
                requested_bytes: bytes as u128,
                budget_bytes: self.total_bytes,
                prior_peak_bytes: self.manager.accounting.peak.load(Ordering::SeqCst),
            }
            .into_error());
        }

        self.remaining_bytes -= bytes;
        let manager = Arc::clone(&self.manager);
        let attempt = LocalAllocationAttempt::new(
            Arc::clone(&manager.accounting),
            bytes,
            Some(&mut self.remaining_bytes),
        );
        let runtime_reservation = self.runtime_reservation.as_mut();
        manager
            .alloc_after_local_reservation::<T>(
                len,
                bytes,
                runtime_reservation,
                Arc::clone(&attempt.reclamation),
            )
            .map_err(|error| map_resource_error(error, manager.peak_bytes()))
    }

    /// Allocate raw device bytes from this reservation through the attached
    /// runtime resource stack.
    pub fn alloc_raw(&mut self, bytes: usize, tag: AllocTag) -> Result<RuntimeAllocBlock> {
        let bytes_u64 = u64::try_from(bytes)
            .map_err(|_| XlogError::Kernel("Allocation size overflow".to_string()))?;
        let used = self.used_bytes();
        if bytes_u64 > self.remaining_bytes {
            return Err(MemoryPressure {
                layer: "manager_reservation_alloc_raw",
                current_bytes: used as u128,
                requested_bytes: bytes_u64 as u128,
                budget_bytes: self.total_bytes,
                prior_peak_bytes: self.manager.accounting.peak.load(Ordering::SeqCst),
            }
            .into_error());
        }

        self.remaining_bytes -= bytes_u64;
        let manager = Arc::clone(&self.manager);
        let attempt = LocalAllocationAttempt::new(
            Arc::clone(&manager.accounting),
            bytes_u64,
            Some(&mut self.remaining_bytes),
        );
        let runtime_reservation = self.runtime_reservation.as_mut();
        manager
            .alloc_raw_after_local_reservation(
                bytes,
                bytes_u64,
                tag,
                runtime_reservation,
                Arc::clone(&attempt.reclamation),
            )
            .map_err(|error| map_resource_error(error, manager.peak_bytes()))
    }
}

impl Drop for GpuMemoryReservation {
    fn drop(&mut self) {
        let unused = std::mem::take(&mut self.remaining_bytes);
        let release = self.manager.rollback_local_reservation(unused);
        debug_assert!(release.is_ok(), "reservation release must be balanced");
    }
}

struct MemoryPressure {
    layer: &'static str,
    current_bytes: u128,
    requested_bytes: u128,
    budget_bytes: u64,
    prior_peak_bytes: u64,
}

impl MemoryPressure {
    fn required_bytes(&self) -> u128 {
        self.current_bytes + self.requested_bytes
    }

    fn into_error(self) -> XlogError {
        let required_bytes = self.required_bytes();
        let required_u64_overflow = required_bytes > u64::MAX as u128;
        XlogError::ResourceExhausted {
            context: format!(
                "GPU memory pressure: layer={} current_bytes={} requested_bytes={} required_bytes={} required_u64_overflow={} budget_bytes={} prior_peak_bytes={}",
                self.layer,
                self.current_bytes,
                self.requested_bytes,
                required_bytes,
                required_u64_overflow,
                self.budget_bytes,
                self.prior_peak_bytes,
            ),
            estimated_bytes: u64::try_from(required_bytes).unwrap_or(u64::MAX),
            budget_bytes: self.budget_bytes,
        }
    }
}

/// The actual allocation owner, shared by every typed view and retained use.
struct DeviceStorage {
    backing: ManuallyDrop<Backing>,
    stream: Arc<CudaStream>,
    dependencies: Arc<DeviceAccessDependencies>,
}

enum Backing {
    Native(Arc<RawDeviceAllocation>),
    Runtime(RuntimeAllocBlock),
    Foreign {
        _owner: Box<dyn Send + Sync>,
        _source: Option<Arc<TrackedCudaSlice<u8>>>,
        bytes: u64,
    },
}

/// One native allocation owner shared by the typed and runtime allocators.
/// No cudarc CudaSlice is constructed: its post-malloc event construction and
/// implicit Drop free cannot express retained initialization/release failures.
pub(crate) struct RawDeviceAllocation {
    payload: Option<RawAllocationPayload>,
    reclamation_admission: Option<MemoryUseGroup>,
}

struct RawAllocationPayload {
    ptr: u64,
    acquired: bool,
    bytes: usize,
    stream: Arc<CudaStream>,
    allocation_stream: Arc<CudaStream>,
    dependencies: Option<Arc<DeviceAccessDependencies>>,
    reclamation: Arc<AllocationReclamation>,
    asynchronous: bool,
    free_state: DriverReleaseState,
    manager: Option<Arc<GpuMemoryManager>>,
    lifecycle_exclusion: Option<AllocationLifecyclePermit>,
}

/// Serialize only physical allocation/release address reuse within one CUDA
/// context. Operation admission and device-resident execution never enter it.
struct AllocationLifecycleExclusion {
    active: std::sync::Mutex<bool>,
    changed: std::sync::Condvar,
}

struct AllocationLifecyclePermit {
    exclusion: Arc<AllocationLifecycleExclusion>,
}

impl AllocationLifecycleExclusion {
    fn enter(self: &Arc<Self>) -> crate::device_runtime::ResourceResult<AllocationLifecyclePermit> {
        let mut active = self.active.lock().map_err(|_| {
            ResourceError::Driver("CUDA allocation lifecycle exclusion poisoned".into())
        })?;
        while *active {
            active = self.changed.wait(active).map_err(|_| {
                ResourceError::Driver("CUDA allocation lifecycle exclusion poisoned".into())
            })?;
        }
        *active = true;
        Ok(AllocationLifecyclePermit {
            exclusion: Arc::clone(self),
        })
    }
}

impl Drop for AllocationLifecyclePermit {
    fn drop(&mut self) {
        let mut active = self
            .exclusion
            .active
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *active = false;
        self.exclusion.changed.notify_one();
    }
}

fn allocation_lifecycle_exclusion(
    context: usize,
) -> crate::device_runtime::ResourceResult<Arc<AllocationLifecycleExclusion>> {
    static EXCLUSIONS: std::sync::OnceLock<
        std::sync::Mutex<
            std::collections::HashMap<usize, std::sync::Weak<AllocationLifecycleExclusion>>,
        >,
    > = std::sync::OnceLock::new();
    let mut exclusions = EXCLUSIONS
        .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
        .lock()
        .map_err(|_| ResourceError::Driver("CUDA allocation lifecycle registry poisoned".into()))?;
    if let Some(exclusion) = exclusions.get(&context).and_then(std::sync::Weak::upgrade) {
        return Ok(exclusion);
    }
    let exclusion = Arc::new(AllocationLifecycleExclusion {
        active: std::sync::Mutex::new(false),
        changed: std::sync::Condvar::new(),
    });
    exclusions.insert(context, Arc::downgrade(&exclusion));
    Ok(exclusion)
}

#[derive(Default, PartialEq, Eq)]
enum DriverReleaseState {
    #[default]
    Owned,
    OutcomeUnknown,
    Submitted,
}

impl DriverReleaseState {
    fn confirm_async(
        &self,
        recorded: &mut bool,
        record: impl FnOnce() -> crate::device_runtime::ResourceResult<()>,
        wait: impl FnOnce() -> crate::device_runtime::ResourceResult<()>,
    ) -> crate::device_runtime::ResourceResult<()> {
        if *self != Self::Submitted {
            return Err(ResourceError::Driver(
                "completion cannot prove an unconfirmed allocation free submission".into(),
            ));
        }
        if !*recorded {
            record()?;
            *recorded = true;
        }
        wait()
    }

    fn submit(
        &mut self,
        free: impl FnOnce() -> crate::device_runtime::ResourceResult<()>,
    ) -> crate::device_runtime::ResourceResult<()> {
        match self {
            Self::Submitted => return Ok(()),
            Self::OutcomeUnknown => {
                return Err(ResourceError::Driver(
                    "previous resource release outcome is unknown; refusing a second release"
                        .into(),
                ))
            }
            Self::Owned => {}
        }
        *self = Self::OutcomeUnknown;
        free()?;
        *self = Self::Submitted;
        Ok(())
    }
}

impl RawDeviceAllocation {
    pub(crate) fn allocate(
        stream: Arc<CudaStream>,
        allocation_stream: Arc<CudaStream>,
        bytes: usize,
        manager: Option<Arc<GpuMemoryManager>>,
        reclamation: Arc<AllocationReclamation>,
    ) -> crate::device_runtime::ResourceResult<Arc<Self>> {
        let _capture_exclusion = crate::cuda_graph::reserve_uncaptured_stream(&stream)?;
        let asynchronous = stream.context().has_async_alloc();
        stream.context().bind_to_thread()?;
        // Preserve the caller-prefix ordering previously carried by a private
        // ready event without retaining an event per allocation.
        stream.synchronize()?;
        // Arm both stream owners before malloc. A post-malloc failure therefore
        // reaches the canonical cold reaper with the exact device-owned stream.
        let mut allocation = initialize_allocation(
            Self {
                reclamation_admission: None,
                payload: Some(RawAllocationPayload {
                    ptr: 0,
                    acquired: false,
                    bytes,
                    stream,
                    allocation_stream,
                    dependencies: None,
                    reclamation: Arc::clone(&reclamation),
                    asynchronous,
                    free_state: DriverReleaseState::Owned,
                    manager,
                    lifecycle_exclusion: None,
                }),
            },
            bytes,
            Arc::clone(&reclamation),
            |allocation| {
                let payload = allocation
                    .payload
                    .as_mut()
                    .expect("allocation owner present");
                // Keep the permit in the armed owner so a fallible
                // post-malloc step transfers it to the cold reaper.
                payload.lifecycle_exclusion = Some(
                    allocation_lifecycle_exclusion(payload.stream.context().cu_ctx() as usize)?
                        .enter()?,
                );
                let ptr = if bytes == 0 {
                    0
                } else {
                    // SAFETY: capture is excluded and the device-owned stream
                    // is serialized by the allocation lifecycle permit.
                    unsafe {
                        if asynchronous {
                            cudarc::driver::result::malloc_async(
                                payload.allocation_stream.cu_stream(),
                                bytes,
                            )?
                        } else {
                            cudarc::driver::result::malloc_sync(bytes)?
                        }
                    }
                };
                payload.ptr = ptr;
                payload.acquired = true;
                payload.reclamation.acquired()?;
                let poison = payload.manager.is_some() && bytes != 0 && poison_alloc_enabled();
                if payload.manager.is_some() {
                    alloc_guard_insert(payload.ptr, bytes as u64);
                    if poison {
                        // SAFETY: initialized only by this armed allocation owner.
                        unsafe {
                            cudarc::driver::sys::cuMemsetD8Async(
                                payload.ptr,
                                0xDD,
                                bytes,
                                payload.allocation_stream.cu_stream(),
                            )
                            .result()?;
                        }
                    }
                }
                if (asynchronous && bytes != 0) || poison {
                    // SAFETY: the lifecycle permit excludes any other user of
                    // this device-owned allocation stream until publication.
                    unsafe {
                        cudarc::driver::result::stream::synchronize(
                            payload.allocation_stream.cu_stream(),
                        )?;
                    }
                }
                payload.dependencies = Some(Arc::new(DeviceAccessDependencies::after_ready(
                    Arc::clone(payload.stream.context()),
                    Arc::clone(&reclamation),
                )));
                Ok(())
            },
        )?;
        // Take the permit while the plain owner is still uniquely mutable.
        let pending_lifecycle_exclusion = allocation
            .payload
            .as_mut()
            .expect("initialized allocation")
            .lifecycle_exclusion
            .take()
            .expect("allocation lifecycle exclusion held");
        let allocation = Arc::new(allocation);
        // Declare the moved permit after the Arc so error or unwind releases
        // the exclusion before Arc::drop synchronously enters the cold reaper.
        let lifecycle_exclusion = pending_lifecycle_exclusion;
        let dependencies = allocation.dependencies();
        dependencies.bind_allocation(&allocation);
        {
            let owner: Arc<dyn MemoryStorageOwner> =
                Arc::clone(&allocation) as Arc<dyn MemoryStorageOwner>;
            let payload = allocation.payload.as_ref().expect("initialized allocation");
            let range = match MemoryUse::new(payload.ptr, payload.bytes, Access::ReadWrite) {
                Ok(range) => range,
                Err(error) => return Err(error.retaining(bytes, reclamation)),
            };
            block_use_registry()
                .lock()
                .expect("device block-use registry poisoned")
                .register_storage(
                    payload.stream.context().cu_ctx() as usize,
                    range,
                    Arc::downgrade(&owner),
                    dependencies.reclamation.release_proof(),
                );
        }
        drop(lifecycle_exclusion);
        Ok(allocation)
    }

    pub(crate) fn ptr(&self) -> u64 {
        self.payload.as_ref().expect("allocation live").ptr
    }
    pub(crate) fn len(&self) -> usize {
        self.payload.as_ref().expect("allocation live").bytes
    }
    pub(crate) fn dependencies(&self) -> Arc<DeviceAccessDependencies> {
        Arc::clone(
            self.payload
                .as_ref()
                .expect("allocation live")
                .dependencies
                .as_ref()
                .expect("allocation producer was recorded"),
        )
    }

    pub(crate) fn reclamation(&self) -> &Arc<AllocationReclamation> {
        &self
            .payload
            .as_ref()
            .expect("allocation owner remains live")
            .reclamation
    }

    /// Cold physical reclamation. A successful async free is never submitted a
    /// second time, even if its following completion check fails.
    pub(crate) fn release(&mut self) -> crate::device_runtime::ResourceResult<()> {
        let admission = &mut self.reclamation_admission;
        reclaim_allocation(&mut self.payload, |payload| {
            // No memory was acquired; the device owns the shared allocation
            // stream and the calling frame restores its provisional budget.
            if !payload.acquired {
                return Ok(());
            }
            if payload.lifecycle_exclusion.is_none() {
                // A failed or unknown release keeps this permit in the
                // retryable payload until physical release is proven.
                payload.lifecycle_exclusion = Some(
                    allocation_lifecycle_exclusion(payload.stream.context().cu_ctx() as usize)?
                        .enter()?,
                );
            }
            crate::device_runtime::resource::with_reclamation_admission(
                block_use_registry(),
                payload.stream.context().cu_ctx() as usize,
                MemoryUse::new(payload.ptr, payload.bytes, Access::ReadWrite)?,
                payload.reclamation.release_proof(),
                admission,
                || {
                    // The device-owned stream is serialized by the lifecycle
                    // permit and is never exposed as an execution dependency.
                    let _capture_exclusion = crate::cuda_graph::reserve_capture_exclusion()?;
                    payload.stream.context().bind_to_thread()?;
                    if payload.free_state != DriverReleaseState::Submitted {
                        if payload.free_state == DriverReleaseState::OutcomeUnknown {
                            return Err(ResourceError::Driver(
                                "previous allocation free outcome is unknown".into(),
                            ));
                        }
                        if let Some(dependencies) = &payload.dependencies {
                            dependencies.synchronize()?;
                        }
                        payload.stream.context().bind_to_thread()?;
                        let poison = payload.manager.is_some()
                            && payload.bytes != 0
                            && poison_free_enabled();
                        if poison {
                            // SAFETY: all users completed and no free was attempted.
                            // A failed poison wait may retry on the same serialized
                            // device-owned stream, where earlier poison is ordered.
                            unsafe {
                                cudarc::driver::sys::cuMemsetD8Async(
                                    payload.ptr,
                                    0xDD,
                                    payload.bytes,
                                    payload.allocation_stream.cu_stream(),
                                )
                                .result()?;
                            }
                            // SAFETY: the lifecycle permit owns the stream prefix.
                            unsafe {
                                cudarc::driver::result::stream::synchronize(
                                    payload.allocation_stream.cu_stream(),
                                )?;
                            }
                        }
                        payload.free_state.submit(|| {
                            if payload.bytes != 0 {
                                // Stop advertising a logically live range before the driver
                                // can reuse its address for another allocation. The armed
                                // owner still retains storage/budget on an unknown outcome.
                                if payload.manager.is_some() {
                                    alloc_guard_remove(payload.ptr);
                                }
                                // SAFETY: all actual-use fences completed; free matches the
                                // chosen allocation mode and the exact allocation context.
                                unsafe {
                                    if payload.asynchronous {
                                        cudarc::driver::result::free_async(
                                            payload.ptr,
                                            payload.allocation_stream.cu_stream(),
                                        )?;
                                    } else {
                                        cudarc::driver::result::free_sync(payload.ptr)?;
                                    }
                                }
                            }
                            Ok(())
                        })?;
                    }
                    if payload.asynchronous && payload.bytes != 0 {
                        // A failed synchronization retains Submitted state, so
                        // retry proves the same free prefix without resubmission.
                        unsafe {
                            cudarc::driver::result::stream::synchronize(
                                payload.allocation_stream.cu_stream(),
                            )?;
                        }
                    }
                    payload.reclamation.complete()?;
                    Ok(())
                },
            )
        })
    }
}

impl MemoryStorageOwner for RawDeviceAllocation {
    fn dependencies(&self) -> crate::device_runtime::ResourceResult<Arc<DeviceAccessDependencies>> {
        Ok(RawDeviceAllocation::dependencies(self))
    }
}

/// Keep the actual payload across a failed or unwinding reclamation attempt.
/// Only a completed physical release and its accounting may retire the owner.
pub(crate) fn reclaim_allocation<T>(
    payload: &mut Option<T>,
    release: impl FnOnce(&mut T) -> crate::device_runtime::ResourceResult<()>,
) -> crate::device_runtime::ResourceResult<()> {
    if let Some(owner) = payload.as_mut() {
        release(owner)?;
        *payload = None;
    }
    Ok(())
}

/// Reclaim only the unique physical owner. A shared owner is returned unchanged
/// to the existing pending queue. After uniqueness wins, the old Weak handles
/// can no longer mint a new lease; a failed attempt restores the real payload
/// for the next reaper, including when the driver boundary unwinds.
pub(crate) fn reclaim_shared_allocation<T>(
    pending: &mut Option<Arc<T>>,
    release: impl FnOnce(&mut T) -> crate::device_runtime::ResourceResult<()>,
) -> crate::device_runtime::ResourceResult<bool> {
    let Some(shared) = pending.take() else {
        return Ok(true);
    };
    let owner = match Arc::try_unwrap(shared) {
        Ok(owner) => owner,
        Err(shared) => {
            *pending = Some(shared);
            return Ok(false);
        }
    };
    struct RestoreOwner<'a, T> {
        pending: &'a mut Option<Arc<T>>,
        owner: Option<T>,
    }
    impl<T> Drop for RestoreOwner<'_, T> {
        fn drop(&mut self) {
            if let Some(owner) = self.owner.take() {
                *self.pending = Some(Arc::new(owner));
            }
        }
    }
    let mut guard = RestoreOwner {
        pending,
        owner: Some(owner),
    };
    release(guard.owner.as_mut().expect("physical owner present"))?;
    drop(guard.owner.take());
    Ok(true)
}

/// Restore every unfinished actual owner to its existing pending queue on
/// return or unwind. The merge moves owners and empties the drained container.
pub(crate) struct PendingReclamationBatch<'a, T> {
    queue: &'a std::sync::Mutex<T>,
    owners: ManuallyDrop<T>,
    merge: fn(&mut T, &mut T),
}

impl<'a, T: Default> PendingReclamationBatch<'a, T> {
    pub(crate) fn take(queue: &'a std::sync::Mutex<T>, merge: fn(&mut T, &mut T)) -> Self {
        let owners = {
            let mut pending = queue.lock().expect("pending allocation queue poisoned");
            ManuallyDrop::new(std::mem::take(&mut *pending))
        };
        Self {
            queue,
            owners,
            merge,
        }
    }
}

impl<T> std::ops::Deref for PendingReclamationBatch<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.owners
    }
}

impl<T> std::ops::DerefMut for PendingReclamationBatch<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.owners
    }
}

impl<T> Drop for PendingReclamationBatch<'_, T> {
    fn drop(&mut self) {
        // A failed driver wait may unwind. Return its real owner, and every
        // unvisited owner, to the same queue before the next reaper runs.
        // Merge functions move ownership only; a panic keeps any unmerged
        // remainder armed rather than dropping CUDA owners under this mutex.
        let mut queue = self.queue.lock().unwrap_or_else(|error| error.into_inner());
        (self.merge)(&mut *queue, &mut self.owners);
        drop(queue);
        // SAFETY: the merge emptied the batch; its container metadata is no
        // longer protected by the pending queue mutex.
        unsafe { ManuallyDrop::drop(&mut self.owners) };
    }
}

/// Both allocator backends reap the same actual owners. A leased entry is
/// skipped without reserving a free; failures leave every unfinished slot
/// owned by the caller's existing pending batch.
pub(crate) fn reap_raw_allocations(
    allocations: &mut Vec<Option<Arc<RawDeviceAllocation>>>,
) -> crate::device_runtime::ResourceResult<()> {
    for allocation in allocations.iter_mut() {
        reclaim_shared_allocation(allocation, RawDeviceAllocation::release)?;
    }
    allocations.retain(Option::is_some);
    Ok(())
}

impl Drop for RawDeviceAllocation {
    fn drop(&mut self) {
        if let Some(payload) = self.payload.take() {
            let mut owner = Self {
                payload: Some(payload),
                reclamation_admission: self.reclamation_admission.take(),
            };
            // The canonical cold queue retains this callable physical owner
            // across errors and unwind. Even a final Drop must not turn a
            // retryable fence failure into an unreachable ManuallyDrop leak.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::cuda_graph::retry_retirement_after_stream_captures(move || {
                    if let Err(error) = owner.release() {
                        if let Some(manager) = owner
                            .payload
                            .as_ref()
                            .filter(|p| !p.reclamation.was_released())
                            .and_then(|p| p.manager.as_ref())
                        {
                            manager.record_deallocation_failure(owner.len() as u64);
                        }
                        eprintln!(
                            "CUDA allocation cleanup incomplete; retaining unresolved memory or handle ownership in the cold reaper: {error}"
                        );
                        return false;
                    }
                    true
                });
            }));
        }
    }
}

/// Arm the real allocation before any fallible post-allocation initialization.
/// Success transfers it to the ordinary storage owner. On error or unwind its
/// dropping owner reaches cold retirement; the error shares only its ticket.
fn initialize_allocation<T: Send + Sync + 'static>(
    mut owner: T,
    bytes: usize,
    reclamation: Arc<AllocationReclamation>,
    initialize: impl FnOnce(&mut T) -> crate::device_runtime::ResourceResult<()>,
) -> crate::device_runtime::ResourceResult<T> {
    match initialize(&mut owner) {
        Ok(()) => Ok(owner),
        Err(error) if reclamation.was_acquired() => Err(error.retaining(bytes, reclamation)),
        Err(error) => Err(error),
    }
}

// DeviceStorage never exposes the opaque payload through a shared reference.
// All mutable bookkeeping is locked; its retirement payload is armed against
// unwind before release. This is also the Arrow custom-allocation contract.
impl std::panic::RefUnwindSafe for DeviceStorage {}

impl DeviceStorage {
    fn new(backing: Backing, stream: Arc<CudaStream>, raw_ptr: u64) -> Arc<Self> {
        let dependencies = match &backing {
            Backing::Native(allocation) => allocation.dependencies(),
            Backing::Runtime(allocation) => Arc::clone(&allocation.dependencies),
            Backing::Foreign { .. } => Arc::new(DeviceAccessDependencies::after_ready(
                Arc::clone(stream.context()),
                Arc::default(),
            )),
        };
        let storage = Arc::new(Self {
            backing: ManuallyDrop::new(backing),
            stream,
            dependencies,
        });
        // Native and runtime storage already register the actual raw owner.
        // A typed wrapper can disappear while its raw lease remains usable;
        // registering that wrapper would leave a false retirement tombstone.
        // Foreign storage instead owns the producer's actual deleter token.
        if matches!(&*storage.backing, Backing::Foreign { .. }) {
            let owner: Arc<dyn MemoryStorageOwner> = storage.clone();
            let range = MemoryUse::new(raw_ptr, storage.bytes() as usize, Access::ReadWrite)
                .expect("CUDA allocation must describe a representable device range");
            block_use_registry()
                .lock()
                .expect("device block-use registry poisoned")
                .register_storage(
                    storage.stream.context().cu_ctx() as usize,
                    range,
                    Arc::downgrade(&owner),
                    storage.dependencies.reclamation.release_proof(),
                );
        }
        storage
    }

    fn manager(&self) -> Option<&Arc<GpuMemoryManager>> {
        match &*self.backing {
            Backing::Native(allocation) => {
                allocation.payload.as_ref().and_then(|p| p.manager.as_ref())
            }
            Backing::Runtime(allocation) => Some(&allocation.manager),
            Backing::Foreign { .. } => None,
        }
    }

    fn bytes(&self) -> u64 {
        match &*self.backing {
            Backing::Native(allocation) => allocation.len() as u64,
            Backing::Runtime(allocation) => allocation.bytes,
            Backing::Foreign { bytes, .. } => *bytes,
        }
    }
}

impl MemoryStorageOwner for DeviceStorage {
    fn dependencies(&self) -> crate::device_runtime::ResourceResult<Arc<DeviceAccessDependencies>> {
        Ok(Arc::clone(&self.dependencies))
    }
}

impl Drop for DeviceStorage {
    fn drop(&mut self) {
        let backing = unsafe { ManuallyDrop::take(&mut self.backing) };
        let dependencies = Arc::clone(&self.dependencies);
        let stream = Arc::clone(&self.stream);
        match backing {
            Backing::Foreign { .. } => crate::cuda_graph::retire_resources_after_completion(
                (Some(backing), dependencies, stream),
                |(_, dependencies, _)| dependencies.synchronize(),
                |owners| {
                    // The producer deleter must return before publishing release.
                    // Its context and dependencies remain armed if it unwinds;
                    // neither the callback nor its deleter may run a second time.
                    drop(owners.0.take());
                    Ok(())
                },
                |owners| owners.1.mark_allocation_released(),
                |_, error| eprintln!("CUDA imported storage retirement incomplete: {error}"),
            ),
            backing => crate::cuda_graph::retire_after_stream_captures(move || {
                // Native Drop transfers the actual raw owner to cold retirement;
                // runtime Drop transfers it to the existing live/pending backend.
                // Neither pending nor error keeps a second runtime-owner cycle.
                drop(backing);
            }),
        }
    }
}

/// Debug probe: poison legacy allocations with 0xDD at drop so any
/// live alias of freed memory becomes visually distinct. Gated on
/// `XLOG_DEBUG_POISON_FREE=1`, read once per process.
fn debug_env_enabled(name: &str) -> bool {
    resolve_bool(None, name, false)
        .unwrap_or_else(|error| panic!("invalid CUDA memory debug configuration: {error}"))
}

fn poison_free_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| debug_env_enabled("XLOG_DEBUG_POISON_FREE"))
}

/// Debug probe: poison fresh legacy allocations with 0xDD so reads of
/// unwritten contents surface deterministically. Gated on
/// `XLOG_DEBUG_POISON_ALLOC=1`, read once per process.
fn poison_alloc_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| debug_env_enabled("XLOG_DEBUG_POISON_ALLOC"))
}

/// Debug probe: track live legacy allocation ranges and panic if the
/// allocator ever hands out a region overlapping one that is still
/// live (double-hand-out / use-after-free detector, timing
/// independent). Gated on `XLOG_DEBUG_ALLOC_GUARD=1`.
fn alloc_guard() -> Option<&'static std::sync::Mutex<std::collections::BTreeMap<u64, u64>>> {
    static GUARD: std::sync::OnceLock<
        Option<std::sync::Mutex<std::collections::BTreeMap<u64, u64>>>,
    > = std::sync::OnceLock::new();
    GUARD
        .get_or_init(|| {
            if debug_env_enabled("XLOG_DEBUG_ALLOC_GUARD") {
                Some(std::sync::Mutex::new(std::collections::BTreeMap::new()))
            } else {
                None
            }
        })
        .as_ref()
}

fn alloc_guard_insert(ptr: u64, bytes: u64) {
    let Some(guard) = alloc_guard() else { return };
    if bytes == 0 {
        return;
    }
    let mut live = guard.lock().unwrap();
    // Overlap check against the nearest live range at or below ptr and
    // the first live range above it.
    if let Some((&p, &b)) = live.range(..=ptr).next_back() {
        if p + b > ptr {
            panic!(
                "ALLOC GUARD: new allocation [{:#x}, {:#x}) overlaps live [{:#x}, {:#x})",
                ptr,
                ptr + bytes,
                p,
                p + b
            );
        }
    }
    if let Some((&p, _)) = live.range(ptr + 1..).next() {
        if ptr + bytes > p {
            panic!(
                "ALLOC GUARD: new allocation [{:#x}, {:#x}) overlaps live starting at {:#x}",
                ptr,
                ptr + bytes,
                p
            );
        }
    }
    live.insert(ptr, bytes);
}

fn alloc_guard_remove(ptr: u64) {
    let Some(guard) = alloc_guard() else { return };
    guard.lock().unwrap().remove(&ptr);
}

/// A typed view retaining its actual device allocation and memory budget.
///
/// Native owning slices cannot be extracted through a safe mutable borrow:
///
/// ```compile_fail
/// use xlog_cuda::memory::TrackedCudaSlice;
/// use cudarc::driver::CudaSlice;
/// fn native_owner(slice: &mut TrackedCudaSlice<u8>) -> &mut CudaSlice<u8> {
///     slice
/// }
/// ```
///
/// Device access must be prepared by XLOG; a passive allocation handle cannot
/// bypass its access reservation through cudarc's safe pointer API:
///
/// ```compile_fail
/// use xlog_cuda::memory::TrackedCudaSlice;
/// use cudarc::driver::DevicePtr;
/// fn native_read(slice: &TrackedCudaSlice<u8>) {
///     fn accepts_native<T: DevicePtr<u8>>(_: &T) {}
///     accepts_native(slice);
/// }
/// ```
pub struct TrackedCudaSlice<T: cudarc::driver::DeviceRepr> {
    storage: Arc<DeviceStorage>,
    ptr: u64,
    len: usize,
    element: std::marker::PhantomData<T>,
}

/// A checked typed device span with a strong allocation owner. Subviews and
/// reinterpretations preserve this same owner; none expose cudarc pointer traits.
pub struct DeviceMemoryView<T> {
    ptr: u64,
    len: usize,
    storage: Arc<DeviceStorage>,
    element: std::marker::PhantomData<T>,
}

/// Physical provenance of a view of an allocation owned by XLOG.
///
/// Retains the actual allocation, not an address-derived identity. This is
/// metadata only: it grants no device access, content validity, mutation
/// exclusion, publication lease, or external tensor version-counter identity.
#[derive(Clone)]
pub struct DeviceAllocationProvenance {
    allocation: Arc<RawDeviceAllocation>,
    byte_offset: u64,
    view_bytes: u64,
}

impl DeviceAllocationProvenance {
    /// Compare the retained physical owners, independently of view geometry.
    pub fn same_allocation(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.allocation, &other.allocation)
    }

    /// Complete byte extent of the actual allocation, including padding.
    pub fn allocation_bytes(&self) -> u64 {
        self.allocation.len() as u64
    }

    pub fn byte_offset(&self) -> u64 {
        self.byte_offset
    }

    pub fn view_bytes(&self) -> u64 {
        self.view_bytes
    }
}

impl<T> Clone for DeviceMemoryView<T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len,
            storage: Arc::clone(&self.storage),
            element: std::marker::PhantomData,
        }
    }
}

impl<T> DeviceMemoryView<T> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.storage.stream
    }
    pub fn device_ptr(&self) -> &u64 {
        &self.ptr
    }

    /// Preserve this view's actual native allocation origin. A foreign owner
    /// supplies only its imported span, not proof of a complete allocation;
    /// never manufacture a native allocation identity for that case.
    pub fn allocation_provenance(&self) -> Option<DeviceAllocationProvenance> {
        let allocation = match &*self.storage.backing {
            Backing::Native(allocation) => allocation,
            Backing::Runtime(block) => block.allocation.as_ref()?,
            Backing::Foreign { .. } => return None,
        };
        let byte_offset = self.ptr.checked_sub(allocation.ptr())?;
        let view_bytes = u64::try_from(self.len.checked_mul(std::mem::size_of::<T>())?).ok()?;
        let allocation_bytes = u64::try_from(allocation.len()).ok()?;
        if byte_offset.checked_add(view_bytes)? > allocation_bytes {
            return None;
        }
        Some(DeviceAllocationProvenance {
            allocation: Arc::clone(allocation),
            byte_offset,
            view_bytes,
        })
    }

    /// Complete byte view of the same actual native allocation. Callers still
    /// have to provide the access grant and producer/consumer ordering; physical
    /// ownership alone does not authorize use of these bytes.
    pub(crate) fn allocation_view(&self) -> Option<DeviceMemoryView<u8>> {
        let provenance = self.allocation_provenance()?;
        Some(DeviceMemoryView {
            ptr: provenance.allocation.ptr(),
            len: provenance.allocation.len(),
            storage: Arc::clone(&self.storage),
            element: std::marker::PhantomData,
        })
    }

    pub fn try_slice(&self, range: impl std::ops::RangeBounds<usize>) -> Option<Self> {
        use std::ops::Bound;
        let start = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n.checked_add(1)?,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&n) => n.checked_add(1)?,
            Bound::Excluded(&n) => n,
            Bound::Unbounded => self.len,
        };
        if start > end || end > self.len {
            return None;
        }
        let offset = start.checked_mul(std::mem::size_of::<T>())?;
        Some(Self {
            ptr: self.ptr.checked_add(u64::try_from(offset).ok()?)?,
            len: end - start,
            storage: Arc::clone(&self.storage),
            element: std::marker::PhantomData,
        })
    }

    pub fn slice(&self, range: impl std::ops::RangeBounds<usize>) -> Self {
        self.try_slice(range)
            .expect("device slice range is out of bounds")
    }

    pub fn slice_mut(&mut self, range: impl std::ops::RangeBounds<usize>) -> Self {
        self.slice(range)
    }

    /// Reinterpret the span without changing its allocation owner.
    ///
    /// # Safety
    /// The initialized device bytes must have the representation required by U
    /// before any read through the returned view. Alignment and size are checked.
    pub unsafe fn cast<U>(&self) -> Option<DeviceMemoryView<U>> {
        let bytes = self.len.checked_mul(std::mem::size_of::<T>())?;
        let size = std::mem::size_of::<U>();
        if size == 0
            || !bytes.is_multiple_of(size)
            || !self.ptr.is_multiple_of(std::mem::align_of::<U>() as u64)
        {
            return None;
        }
        Some(DeviceMemoryView {
            ptr: self.ptr,
            len: bytes / size,
            storage: Arc::clone(&self.storage),
            element: std::marker::PhantomData,
        })
    }
}

impl<T: DeviceRepr + 'static> DeviceMemoryView<T> {
    /// Allocate through the same native owner used by runtime resources.
    pub(crate) fn allocate(
        stream: Arc<CudaStream>,
        allocation_stream: Arc<CudaStream>,
        len: usize,
    ) -> crate::device_runtime::ResourceResult<Self> {
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| ResourceError::Driver("allocation size overflow".into()))?;
        let allocation = RawDeviceAllocation::allocate(
            Arc::clone(&stream),
            allocation_stream,
            bytes,
            None,
            Arc::default(),
        )?;
        let ptr = allocation.ptr();
        Ok(Self {
            ptr,
            len,
            storage: DeviceStorage::new(Backing::Native(allocation), stream, ptr),
            element: std::marker::PhantomData,
        })
    }
}

impl<T> DeviceSlice<T> for DeviceMemoryView<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        &self.storage.stream
    }
}

/// Safe XLOG device operations accept owner-bearing spans, not native pointer
/// adapters. The returned span must retain the actual allocation.
pub trait DeviceRead<T>: private_access::Sealed {
    fn device_view(&self) -> DeviceMemoryView<T>;
}

/// A span accepted as a destination by safe XLOG operations. Conflicting aliases
/// are excluded by operation admission, including aliases of a different type.
pub trait DeviceWrite<T>: DeviceRead<T> {}

mod private_access {
    pub trait Sealed {}
}

impl<T: DeviceRepr> private_access::Sealed for TrackedCudaSlice<T> {}
impl<T: DeviceRepr> DeviceRead<T> for TrackedCudaSlice<T> {
    fn device_view(&self) -> DeviceMemoryView<T> {
        self.view()
    }
}
impl<T: DeviceRepr> DeviceWrite<T> for TrackedCudaSlice<T> {}
impl<T> private_access::Sealed for DeviceMemoryView<T> {}
impl<T> DeviceRead<T> for DeviceMemoryView<T> {
    fn device_view(&self) -> DeviceMemoryView<T> {
        self.clone()
    }
}
impl<T> DeviceWrite<T> for DeviceMemoryView<T> {}

impl<T: DeviceRepr> AsKernelParam for &DeviceMemoryView<T> {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.ptr as *const u64).cast_mut().cast()
    }
}

impl<T: DeviceRepr> AsKernelParam for &mut DeviceMemoryView<T> {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.ptr as *const u64).cast_mut().cast()
    }
}

impl<'a, T: DeviceRepr> IntoKernelParamStorage for &'a DeviceMemoryView<T> {
    type Storage = DeviceParamStorage<'a>;

    fn into_kernel_param_storage(self) -> Self::Storage {
        DeviceParamStorage::unsynced(self.ptr)
    }
}

impl<'a, T: DeviceRepr> IntoKernelParamStorage for &'a mut DeviceMemoryView<T> {
    type Storage = DeviceParamStorage<'a>;

    fn into_kernel_param_storage(self) -> Self::Storage {
        DeviceParamStorage::unsynced(self.ptr)
    }
}

/// One checked span and its real backing owner. Obtaining this manifest is
/// passive; only admission below grants access on the operation's stream.
#[derive(Clone)]
pub(crate) struct DeviceMemoryAccess {
    storage: Arc<DeviceStorage>,
    range: MemoryUse,
}

impl DeviceMemoryAccess {
    pub(crate) fn retained_owner(&self) -> RetainedStorageUse {
        RetainedStorageUse {
            owner: self.storage.clone(),
            access: self.range.access(),
        }
    }

    pub(crate) fn context(&self) -> usize {
        self.storage.stream.context().cu_ctx() as usize
    }
}

impl<T> DeviceMemoryView<T> {
    pub(crate) fn access(
        &self,
        access: Access,
    ) -> crate::device_runtime::ResourceResult<DeviceMemoryAccess> {
        let bytes = self
            .len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| ResourceError::StreamMisuse("device view byte size overflow".into()))?;
        Ok(DeviceMemoryAccess {
            storage: Arc::clone(&self.storage),
            range: MemoryUse::new(self.ptr, bytes, access)?,
        })
    }
}

/// Passive storage ownership. A graph retains this immutable manifest, never
/// an executing stream pin, access reservation, or prepared dependency state.
#[derive(Default)]
pub(crate) struct MemoryAccessManifest {
    accesses: Vec<DeviceMemoryAccess>,
    ranges: Vec<(usize, MemoryUse)>,
    retained: Vec<RetainedStorageUse>,
    runtimes: Vec<Arc<XlogDeviceRuntime>>,
    runtime_dependencies: Vec<(Arc<DeviceAccessDependencies>, Access)>,
    runtime_allocations: Vec<Arc<RawDeviceAllocation>>,
}

impl MemoryAccessManifest {
    pub(crate) fn combine(manifests: &[Arc<Self>]) -> Arc<Self> {
        let mut combined = Self::default();
        for manifest in manifests {
            combined.accesses.extend(manifest.accesses.iter().cloned());
            combined.ranges.extend_from_slice(&manifest.ranges);
            combined
                .retained
                .extend(manifest.retained.iter().map(|retained| RetainedStorageUse {
                    owner: Arc::clone(&retained.owner),
                    access: retained.access,
                }));
            combined.runtimes.extend(manifest.runtimes.iter().cloned());
            combined
                .runtime_dependencies
                .extend(manifest.runtime_dependencies.iter().cloned());
            combined
                .runtime_allocations
                .extend(manifest.runtime_allocations.iter().cloned());
        }
        Arc::new(combined)
    }

    pub(crate) fn covers(&self, ranges: &[(usize, MemoryUse)]) -> bool {
        ranges.iter().all(|(context, range)| {
            self.ranges
                .iter()
                .any(|(owned_context, owned)| context == owned_context && owned.covers(*range))
        })
    }

    pub(crate) fn covered_by(&self, other: &Self) -> bool {
        other.covers(&self.ranges)
    }
}

/// Positive completion of one admitted, synchronous enqueue callback and all
/// its nested submissions. This proof is never reused by another admission.
/// Only a successful wait on its exact execution can publish completion.
#[derive(Debug)]
pub(crate) struct OperationCompletion {
    execution_id: u64,
    complete: std::sync::atomic::AtomicBool,
}

impl OperationCompletion {
    pub(crate) fn new(execution_id: u64) -> Self {
        Self {
            execution_id,
            complete: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub(crate) fn validate_submission(
        &self,
        execution_id: u64,
    ) -> crate::device_runtime::ResourceResult<()> {
        if execution_id != self.execution_id || self.is_complete() {
            return Err(ResourceError::StreamMisuse(
                "memory operation requires its original uncompleted stream execution".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn synchronize_with(
        &self,
        execution_id: u64,
        synchronize: impl FnOnce() -> crate::device_runtime::ResourceResult<()>,
    ) -> crate::device_runtime::ResourceResult<()> {
        if execution_id != self.execution_id {
            return Err(ResourceError::StreamMisuse(
                "memory operation cannot synchronize another stream execution".into(),
            ));
        }
        synchronize()?;
        self.complete.store(true, Ordering::Release);
        Ok(())
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }

    /// A cold barrier on the owning context completes this closed admission,
    /// including work on an originating thread that has since exited. This is
    /// not a stream-identity substitution and cannot authorize another enqueue.
    fn synchronize_retired_with(
        &self,
        synchronize_context: impl FnOnce() -> crate::device_runtime::ResourceResult<()>,
    ) -> crate::device_runtime::ResourceResult<()> {
        if !self.is_complete() {
            synchronize_context()?;
            self.complete.store(true, Ordering::Release);
        }
        Ok(())
    }
}

pub(crate) struct MemoryOperationOwner {
    stream: Arc<CudaStream>,
    manifest: Arc<MemoryAccessManifest>,
    retained: Vec<RetainedStorageUse>,
    captured: std::sync::atomic::AtomicBool,
    dependencies: std::sync::OnceLock<Vec<(Arc<DeviceAccessDependencies>, Access)>>,
    completion: Arc<OperationCompletion>,
}

impl crate::launch::RecorderCleanup<MemoryUseGroup> for MemoryOperationOwner {
    fn synchronize_retired(&self) -> crate::device_runtime::ResourceResult<()> {
        if self.captured.load(Ordering::Acquire) {
            // Captured nodes retain their own storage; this admission submitted
            // no live work and must not synchronize or end that capture.
            return Ok(());
        }
        let _ordinary = crate::cuda_graph::reserve_capture_exclusion()?;
        self.completion.synchronize_retired_with(|| {
            // Failed cleanup may reach this queue after the original PTDS host
            // thread has exited and without a successfully recorded event. Only
            // a barrier on the retained context can cover that missing fence.
            // This potentially blocking wait belongs exclusively to cold error
            // retirement; normal enqueue/commit adds no context-wide wait.
            self.stream.context().bind_to_thread()?;
            self.stream.context().synchronize().map_err(|error| {
                ResourceError::Driver(format!(
                    "retired memory operation context wait failed: {error}"
                ))
            })
        })
    }

    fn cancel_retired(&self, group: MemoryUseGroup) -> crate::device_runtime::ResourceResult<()> {
        self.cancel(&[group])
    }
}

impl MemoryOperationOwner {
    pub(crate) fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub(crate) fn manifest(&self) -> &Arc<MemoryAccessManifest> {
        &self.manifest
    }

    pub(crate) fn completion(&self) -> &Arc<OperationCompletion> {
        &self.completion
    }

    pub(crate) fn bind_submission(
        self: &Arc<Self>,
        pin: &crate::cuda_graph::StreamSubmissionPin<'_>,
    ) -> crate::device_runtime::ResourceResult<()> {
        // Check before dependency preparation, including PTDS migration between
        // admission and enqueue. Cleanup must never certify a different thread.
        self.completion
            .validate_submission(pin.validate_execution(&self.stream)?)?;
        // Retain actual storage before the first node can reach the driver.
        // No persistent owner retains the submission pin or its capture target.
        if pin.capture_memory(&self.manifest) {
            self.captured.store(true, Ordering::Release);
        }
        Ok(())
    }

    pub(crate) fn synchronize(&self) -> crate::device_runtime::ResourceResult<()> {
        if self.captured.load(Ordering::Acquire) {
            // Capture queued graph nodes, not live work. Its actual owners were
            // transferred before enqueue; only the capture owner may end it.
            return Ok(());
        }
        self.completion.synchronize_with(
            crate::cuda_graph::stream_execution_id(&self.stream)?,
            || {
                self.stream.synchronize().map_err(|error| {
                    ResourceError::Driver(format!(
                        "device memory operation synchronization failed: {error}"
                    ))
                })
            },
        )
    }

    pub(crate) fn cancel(
        &self,
        groups: &[MemoryUseGroup],
    ) -> crate::device_runtime::ResourceResult<()> {
        let [group] = groups else {
            return Err(ResourceError::StreamMisuse(
                "device memory operation must own one exact reservation".into(),
            ));
        };
        block_use_registry()
            .lock()
            .expect("device block-use registry poisoned")
            .release_memory_uses(*group)
    }

    pub(crate) fn prepare(&self) -> crate::device_runtime::ResourceResult<()> {
        if self.captured.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut dependencies = Vec::new();
        for (dependency, access) in &self.manifest.runtime_dependencies {
            if let Some((_, existing_access)) = dependencies
                .iter_mut()
                .find(|(existing, _)| Arc::ptr_eq(existing, dependency))
            {
                *existing_access = crate::launch::combine_access(*existing_access, *access);
            } else {
                dependencies.push((Arc::clone(dependency), *access));
            }
        }
        for retained in &self.retained {
            let dependency = retained.owner.dependencies()?;
            if let Some((_, access)) = dependencies
                .iter_mut()
                .find(|(existing, _)| Arc::ptr_eq(existing, &dependency))
            {
                *access = crate::launch::combine_access(*access, retained.access);
            } else {
                dependencies.push((dependency, retained.access));
            }
        }
        self.dependencies.set(dependencies).map_err(|_| {
            ResourceError::StreamMisuse(
                "device memory dependencies may be prepared only once".into(),
            )
        })?;
        for (dependency, access) in self.dependencies.get().expect("dependencies installed") {
            dependency.prepare(&self.stream, *access)?;
        }
        Ok(())
    }

    pub(crate) fn finish(
        &self,
        group: MemoryUseGroup,
    ) -> crate::device_runtime::ResourceResult<()> {
        if self.captured.load(Ordering::Acquire) {
            return self.cancel(&[group]);
        }
        // Keep the reservation until every overlapping allocation/import owner
        // has the completion dependency. A partial publication failure is
        // handled by the same synchronize/cancel/quarantine path as launches.
        let dependencies = self.dependencies.get().ok_or_else(|| {
            ResourceError::StreamMisuse(
                "device memory completion requires prepared dependencies".into(),
            )
        })?;
        DeviceAccessDependencies::record_operation_completion(
            dependencies,
            Arc::clone(&self.stream),
        )?;
        self.cancel(&[group])
    }
}

/// Native pointers are available only while this private proof is borrowed by
/// the enqueue closure. It cannot be constructed from a raw address or kept
/// after the enclosing operation has published or cancelled its reservation.
pub(crate) struct MemoryEnqueue<'a> {
    owner: &'a Arc<MemoryOperationOwner>,
    submission: &'a crate::cuda_graph::StreamSubmissionPin<'a>,
}

impl MemoryEnqueue<'_> {
    pub(crate) fn cuda_enqueue(
        &self,
    ) -> crate::device_runtime::ResourceResult<crate::launch::CudaEnqueue<'_>> {
        crate::launch::CudaEnqueue::from_admission(self.owner, self.submission)
    }

    fn validate<T>(
        &self,
        view: &DeviceMemoryView<T>,
        access: Access,
    ) -> crate::device_runtime::ResourceResult<()> {
        let range = view.access(access)?.range;
        if !self.owner.manifest.accesses.iter().any(|admitted| {
            Arc::ptr_eq(&admitted.storage, &view.storage) && admitted.range.covers(range)
        }) {
            return Err(ResourceError::StreamMisuse(
                "native device view is not covered by the admitted access manifest".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn read<'a, T>(
        &'a self,
        view: &'a DeviceMemoryView<T>,
    ) -> crate::device_runtime::ResourceResult<PreparedDeviceRead<'a, T>> {
        self.validate(view, Access::Read)?;
        self.validate_native_context(view)?;
        Ok(PreparedDeviceRead { view, proof: self })
    }

    pub(crate) fn write<'a, T>(
        &'a self,
        view: &'a DeviceMemoryView<T>,
    ) -> crate::device_runtime::ResourceResult<PreparedDeviceWrite<'a, T>> {
        self.validate(view, Access::Write)?;
        self.validate_native_context(view)?;
        Ok(PreparedDeviceWrite { view, proof: self })
    }

    fn validate_native_context<T>(
        &self,
        view: &DeviceMemoryView<T>,
    ) -> crate::device_runtime::ResourceResult<()> {
        if view.stream().context().cu_ctx() != self.owner.stream.context().cu_ctx() {
            return Err(ResourceError::StreamMisuse(
                "native device access requires the allocation context".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn copy<T>(
        &self,
        source: &DeviceMemoryView<T>,
        destination: &DeviceMemoryView<T>,
    ) -> crate::device_runtime::ResourceResult<()> {
        self.validate(source, Access::Read)?;
        self.validate(destination, Access::Write)?;
        self.validate_native_context(destination)?;
        let source_context = source.stream().context().cu_ctx();
        let destination_context = destination.stream().context().cu_ctx();
        source.access(Access::Read)?.range.validate_copy_to(
            source_context as usize,
            destination.access(Access::Write)?.range,
            destination_context as usize,
        )?;
        if source_context == destination_context {
            self.owner
                .stream
                .memcpy_dtod(&self.read(source)?, &mut self.write(destination)?)?;
        } else {
            self.owner.stream.context().bind_to_thread()?;
            // SAFETY: the one admitted operation retains both exact allocation
            // contexts and byte ranges, with a source read and destination write.
            // All prior dependencies were queued before this proof was borrowed.
            // The operation's stream belongs to the destination context.
            unsafe {
                cudarc::driver::result::memcpy_peer_async(
                    destination_context,
                    destination.ptr,
                    source_context,
                    source.ptr,
                    source
                        .len
                        .checked_mul(std::mem::size_of::<T>())
                        .ok_or_else(|| {
                            ResourceError::StreamMisuse("device copy byte size overflow".into())
                        })?,
                    self.owner.stream.cu_stream(),
                )?;
            }
        }
        Ok(())
    }
}

pub(crate) struct PreparedDeviceRead<'a, T> {
    view: &'a DeviceMemoryView<T>,
    proof: &'a MemoryEnqueue<'a>,
}

pub(crate) struct PreparedDeviceWrite<'a, T> {
    view: &'a DeviceMemoryView<T>,
    proof: &'a MemoryEnqueue<'a>,
}

impl<T> DeviceSlice<T> for PreparedDeviceRead<'_, T> {
    fn len(&self) -> usize {
        self.view.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        self.view.stream()
    }
}

impl<T> DevicePtr<T> for PreparedDeviceRead<'_, T> {
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (u64, SyncOnDrop<'a>) {
        assert!(
            std::ptr::eq(stream, self.proof.owner.stream.as_ref()),
            "prepared view used on a different stream"
        );
        (self.view.ptr, SyncOnDrop::Sync(None))
    }
}

impl<T> DeviceSlice<T> for PreparedDeviceWrite<'_, T> {
    fn len(&self) -> usize {
        self.view.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        self.view.stream()
    }
}

impl<T> DevicePtr<T> for PreparedDeviceWrite<'_, T> {
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (u64, SyncOnDrop<'a>) {
        assert!(
            std::ptr::eq(stream, self.proof.owner.stream.as_ref()),
            "prepared view used on a different stream"
        );
        (self.view.ptr, SyncOnDrop::Sync(None))
    }
}

impl<T> DevicePtrMut<T> for PreparedDeviceWrite<'_, T> {
    fn device_ptr_mut<'a>(&'a mut self, stream: &'a CudaStream) -> (u64, SyncOnDrop<'a>) {
        assert!(
            std::ptr::eq(stream, self.proof.owner.stream.as_ref()),
            "prepared view used on a different stream"
        );
        (self.view.ptr, SyncOnDrop::Sync(None))
    }
}

struct MemoryOperation {
    transaction: RecorderTransaction<MemoryOperationOwner, MemoryUseGroup>,
    // Host admission is not part of the storage quarantine on uncertain free.
    submission: crate::cuda_graph::StreamSubmissionPhase,
}

impl Drop for MemoryOperation {
    fn drop(&mut self) {
        if self.transaction.is_prepared() {
            // abort_with retains the whole owner capsule if synchronization or
            // cancellation fails, including on unwinding from the enqueue.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.transaction.abort_with(
                    MemoryOperationOwner::synchronize,
                    MemoryOperationOwner::cancel,
                )
            }));
        }
    }
}

/// One admission path for copies and recorded launches. No driver wait or
/// publication runs under the registry mutex; source and alias owners are
/// transferred to the armed transaction before that mutex is released.
pub(crate) fn admit_memory_access(
    stream: Arc<CudaStream>,
    accesses: Vec<DeviceMemoryAccess>,
    runtime: Option<Arc<XlogDeviceRuntime>>,
    runtime_uses: &[crate::device_runtime::BlockUse],
    execution_id: u64,
) -> crate::device_runtime::ResourceResult<RecorderTransaction<MemoryOperationOwner, MemoryUseGroup>>
{
    let mut ranges_by_context = std::collections::BTreeMap::<usize, Vec<MemoryUse>>::new();
    for access in &accesses {
        ranges_by_context
            .entry(access.context())
            .or_default()
            .push(access.range);
    }
    if let Some(runtime) = &runtime {
        let context = runtime.device().inner().stream().context().cu_ctx() as usize;
        for use_ in runtime_uses {
            ranges_by_context
                .entry(context)
                .or_default()
                .push(MemoryUse::new(use_.block.ptr, use_.bytes, use_.access)?);
        }
    } else if !runtime_uses.is_empty() {
        return Err(ResourceError::StreamMisuse(
            "recorded blocks require their runtime owner".into(),
        ));
    }
    let ranges = ranges_by_context
        .iter()
        .flat_map(|(context, ranges)| ranges.iter().map(|range| (*context, *range)))
        .collect::<Vec<_>>();
    let mut manifest = MemoryAccessManifest {
        // Even an empty span retains its actual owner; weak overlap discovery
        // is an alias rendezvous, not proof that the source is owned.
        retained: accesses
            .iter()
            .map(DeviceMemoryAccess::retained_owner)
            .collect(),
        accesses,
        ranges,
        runtimes: runtime.iter().cloned().collect(),
        runtime_dependencies: Vec::new(),
        runtime_allocations: Vec::new(),
    };
    let mut registry = block_use_registry()
        .lock()
        .expect("device block-use registry poisoned");
    if let Some(runtime) = &runtime {
        for use_ in runtime_uses {
            let dependencies = runtime
                .allocation_dependencies(use_.block, use_.bytes)?
                .ok_or_else(|| {
                    ResourceError::StreamMisuse(
                        "recorded block backend does not expose allocation dependencies".into(),
                    )
                })?;
            manifest
                .runtime_allocations
                .push(dependencies.retain_allocation()?);
            manifest
                .runtime_dependencies
                .push((dependencies, use_.access));
        }
    }
    for (context, ranges) in &ranges_by_context {
        registry.retain_storage_uses(*context, ranges, &mut manifest.retained)?;
    }
    let manifest = Arc::new(manifest);
    // Owner discovery and range reservation share this registry lock. A fresh
    // manifest therefore needs no second alias scan during admission.
    let group = reserve_memory_manifest_locked(&mut registry, &manifest);
    drop(registry);
    let retained = manifest
        .retained
        .iter()
        .map(|retained| RetainedStorageUse {
            owner: Arc::clone(&retained.owner),
            access: retained.access,
        })
        .collect();
    memory_operation_transaction(stream, manifest, retained, execution_id, group?)
}

pub(crate) fn admit_memory_manifest(
    stream: Arc<CudaStream>,
    manifest: Arc<MemoryAccessManifest>,
    execution_id: u64,
) -> crate::device_runtime::ResourceResult<RecorderTransaction<MemoryOperationOwner, MemoryUseGroup>>
{
    let mut retained = manifest
        .retained
        .iter()
        .map(|retained| RetainedStorageUse {
            owner: Arc::clone(&retained.owner),
            access: retained.access,
        })
        .collect::<Vec<_>>();
    let mut registry = block_use_registry()
        .lock()
        .expect("device block-use registry poisoned");
    // Passive manifests may outlive their original admission. Refresh aliases
    // once, then reserve their complete ranges under this same registry lock.
    let mut ranges_by_context = std::collections::BTreeMap::<usize, Vec<MemoryUse>>::new();
    for (context, range) in &manifest.ranges {
        ranges_by_context.entry(*context).or_default().push(*range);
    }
    for (context, ranges) in ranges_by_context {
        registry.retain_storage_uses(context, &ranges, &mut retained)?;
    }
    let group = reserve_memory_manifest_locked(&mut registry, &manifest);
    drop(registry);
    memory_operation_transaction(stream, manifest, retained, execution_id, group?)
}

fn reserve_memory_manifest_locked(
    registry: &mut BlockUseRegistry,
    manifest: &MemoryAccessManifest,
) -> crate::device_runtime::ResourceResult<MemoryUseGroup> {
    let source_proofs = manifest
        .accesses
        .iter()
        .map(|access| access.storage.dependencies.reclamation.release_proof())
        .chain(
            manifest
                .runtime_dependencies
                .iter()
                .map(|(deps, _)| deps.reclamation.release_proof()),
        )
        .collect::<Vec<_>>();
    registry.reserve_owned_memory_uses(&manifest.ranges, &source_proofs)
}

fn memory_operation_transaction(
    stream: Arc<CudaStream>,
    manifest: Arc<MemoryAccessManifest>,
    retained: Vec<RetainedStorageUse>,
    execution_id: u64,
    group: MemoryUseGroup,
) -> crate::device_runtime::ResourceResult<RecorderTransaction<MemoryOperationOwner, MemoryUseGroup>>
{
    let owner = Arc::new(MemoryOperationOwner {
        stream,
        manifest,
        retained,
        captured: std::sync::atomic::AtomicBool::new(false),
        dependencies: std::sync::OnceLock::new(),
        completion: Arc::new(OperationCompletion::new(execution_id)),
    });
    Ok(RecorderTransaction::from_admitted(owner, Box::new([group])))
}

pub(crate) fn with_memory_access<R>(
    stream: Arc<CudaStream>,
    accesses: Vec<DeviceMemoryAccess>,
    operation: impl FnOnce(&MemoryEnqueue<'_>) -> crate::device_runtime::ResourceResult<R>,
) -> crate::device_runtime::ResourceResult<R> {
    let submission = crate::cuda_graph::acquire_stream_submission_phase(&stream)?;
    let execution_id = submission.execution_id();
    let guard = MemoryOperation {
        transaction: admit_memory_access(stream, accesses, None, &[], execution_id)?,
        submission,
    };
    with_memory_operation(guard, |owner, submission| {
        operation(&MemoryEnqueue { owner, submission })
    })
}

pub(crate) fn with_memory_manifest<T>(
    stream: Arc<CudaStream>,
    manifest: Arc<MemoryAccessManifest>,
    operation: impl FnOnce(&crate::launch::CudaEnqueue<'_>) -> crate::device_runtime::ResourceResult<T>,
) -> crate::device_runtime::ResourceResult<T> {
    let submission = crate::cuda_graph::acquire_stream_submission_phase(&stream)?;
    let execution_id = submission.execution_id();
    let guard = MemoryOperation {
        transaction: admit_memory_manifest(stream, manifest, execution_id)?,
        submission,
    };
    with_memory_operation(guard, |owner, pin| {
        operation(&crate::launch::CudaEnqueue::from_admission(owner, pin)?)
    })
}

fn with_memory_operation<T>(
    mut guard: MemoryOperation,
    operation: impl FnOnce(
        &Arc<MemoryOperationOwner>,
        &crate::cuda_graph::StreamSubmissionPin<'_>,
    ) -> crate::device_runtime::ResourceResult<T>,
) -> crate::device_runtime::ResourceResult<T> {
    let owner = Arc::clone(
        guard
            .transaction
            .prepared_owner()
            .expect("memory operation admitted"),
    );
    let mut result = None;
    guard
        .transaction
        .enqueue_operation_with(
            || {
                let pin = guard.submission.pin()?;
                owner.bind_submission(&pin)?;
                Ok(pin)
            },
            MemoryOperationOwner::prepare,
            |pin| {
                result = Some(pin.with_serialized_submission(|| operation(&owner, pin))?);
                Ok::<_, ResourceError>(())
            },
            MemoryOperationOwner::synchronize,
            MemoryOperationOwner::cancel,
        )
        .map_err(|error| ResourceError::Driver(error.to_string()))?;
    guard.transaction.commit_with(
        MemoryOperationOwner::finish,
        MemoryOperationOwner::synchronize,
        MemoryOperationOwner::cancel,
    )?;
    Ok(result.expect("successful memory operation produced its result"))
}

#[derive(Clone)]
pub(crate) struct RuntimeAllocationIdentity {
    pub(crate) manager_id: usize,
    pub(crate) allocation_ptr: u64,
    pub(crate) allocation_bytes: usize,
    pub(crate) block_id: BlockId,
    pub(crate) block_bytes: usize,
    pub(crate) block_state: BlockState,
    pub(crate) context: Arc<cudarc::driver::CudaContext>,
}

impl<T: cudarc::driver::DeviceRepr> DeviceSlice<T> for TrackedCudaSlice<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn stream(&self) -> &Arc<CudaStream> {
        &self.storage.stream
    }
}

impl<T: cudarc::driver::DeviceRepr> TrackedCudaSlice<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.storage.stream
    }

    pub fn device_ptr(&self) -> &cudarc::driver::sys::CUdeviceptr {
        &self.ptr
    }

    pub fn device_ptr_value(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.ptr
    }

    /// A passive view retaining this allocation; obtaining it performs no device access.
    pub fn view(&self) -> DeviceMemoryView<T> {
        DeviceMemoryView {
            ptr: self.ptr,
            len: self.len,
            storage: Arc::clone(&self.storage),
            element: std::marker::PhantomData,
        }
    }

    /// Retain a disjoint typed span of this allocation as an owning slice.
    ///
    /// The returned slice shares the physical allocation and budget charge.
    /// Operation admission still applies to its exact byte range, so sibling
    /// spans can be recorded independently without duplicating ownership.
    pub(crate) fn try_owned_subslice<U: cudarc::driver::DeviceRepr>(
        &self,
        byte_range: std::ops::Range<usize>,
    ) -> Option<TrackedCudaSlice<U>> {
        let bytes = self.view().try_slice(byte_range)?;
        // SAFETY: callers receive an uninitialized device allocation span, as
        // they do from alloc::<U>. Alignment and extent are checked by cast;
        // the span must be initialized before any device read.
        let typed = unsafe { bytes.cast::<U>()? };
        Some(TrackedCudaSlice {
            storage: Arc::clone(&typed.storage),
            ptr: typed.ptr,
            len: typed.len,
            element: std::marker::PhantomData,
        })
    }

    pub fn try_slice(
        &self,
        range: impl std::ops::RangeBounds<usize>,
    ) -> Option<DeviceMemoryView<T>> {
        self.view().try_slice(range)
    }

    pub fn slice(&self, range: impl std::ops::RangeBounds<usize>) -> DeviceMemoryView<T> {
        self.try_slice(range)
            .expect("device slice range is out of bounds")
    }

    pub fn slice_mut(&mut self, range: impl std::ops::RangeBounds<usize>) -> DeviceMemoryView<T> {
        self.slice(range)
    }

    /// Stable address of the memory manager that owns this allocation.
    pub fn memory_manager_ptr_value(&self) -> usize {
        Arc::as_ptr(
            self.storage
                .manager()
                .expect("tracked allocation has a memory manager"),
        ) as usize
    }

    pub(crate) fn runtime_allocation_identity(&self) -> Result<Option<RuntimeAllocationIdentity>> {
        let Some(block) = self.runtime_block() else {
            return Ok(None);
        };
        let allocation_bytes = self
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| XlogError::Kernel("runtime allocation byte size overflow".into()))?;
        Ok(Some(RuntimeAllocationIdentity {
            manager_id: self.memory_manager_ptr_value(),
            allocation_ptr: self.device_ptr_value(),
            allocation_bytes,
            block_id: BlockId::from_block(block),
            block_bytes: block.bytes,
            block_state: block.state,
            context: Arc::clone(DeviceSlice::stream(self).context()),
        }))
    }

    /// Runtime block identity, when present. Device-native and foreign owners
    /// have no runtime block ID but still participate in shared range admission.
    pub fn runtime_block(&self) -> Option<&crate::device_runtime::DeviceBlock> {
        match &*self.storage.backing {
            Backing::Native(_) | Backing::Foreign { .. } => None,
            Backing::Runtime(allocation) => Some(allocation.device_block()),
        }
    }

    /// Reinterpret this typed allocation as a raw byte allocation.
    ///
    /// This is a zero-copy conversion used by XLOG's columnar
    /// `CudaBuffer` representation, which stores device memory as
    /// untyped bytes + a schema. The conversion preserves the
    /// underlying [`Backing`] — runtime-routed slices remain
    /// runtime-routed, legacy cudarc slices remain cudarc-routed —
    /// so deallocation continues to match the original allocator.
    pub fn into_bytes(self) -> TrackedCudaSlice<u8> {
        let len = self
            .len
            .checked_mul(std::mem::size_of::<T>())
            .expect("tracked slice byte size must fit into usize");
        TrackedCudaSlice {
            storage: self.storage,
            ptr: self.ptr,
            len,
            element: std::marker::PhantomData,
        }
    }
}

impl<T: cudarc::driver::DeviceRepr> AsKernelParam for &TrackedCudaSlice<T> {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        ((*self).device_ptr() as *const cudarc::driver::sys::CUdeviceptr)
            .cast_mut()
            .cast()
    }
}

impl<T: cudarc::driver::DeviceRepr> AsKernelParam for &mut TrackedCudaSlice<T> {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        ((self.device_ptr()) as *const cudarc::driver::sys::CUdeviceptr)
            .cast_mut()
            .cast()
    }
}

impl<'a, T: cudarc::driver::DeviceRepr> IntoKernelParamStorage for &'a TrackedCudaSlice<T> {
    type Storage = DeviceParamStorage<'a>;

    fn into_kernel_param_storage(self) -> Self::Storage {
        DeviceParamStorage::unsynced(self.ptr)
    }
}

impl<T: cudarc::driver::DeviceRepr> IntoKernelParamStorage for &mut TrackedCudaSlice<T> {
    type Storage = DeviceParamStorage<'static>;

    fn into_kernel_param_storage(self) -> Self::Storage {
        DeviceParamStorage::unsynced(self.ptr)
    }
}

impl GpuMemoryManager {
    /// Create a new GPU memory manager
    ///
    /// # Arguments
    /// * `device` - The CUDA device to allocate memory on
    /// * `budget` - Memory budget configuration
    pub(crate) fn new(device: Arc<CudaDevice>, budget: MemoryBudget) -> Self {
        Self {
            device,
            budget,
            accounting: Arc::new(GpuMemoryAccounting::default()),
            runtime: None,
            #[cfg(test)]
            after_local_reservation_hook: std::sync::Mutex::new(None),
        }
    }

    /// Like [`new`], but additionally attaches a v0.6
    /// [`XlogDeviceRuntime`]. The runtime mediates **both**
    /// [`alloc::<T>`](Self::alloc) and [`alloc_raw`](Self::alloc_raw)
    /// through the v0.6 resource stack: typed `alloc::<T>` returns a
    /// [`TrackedCudaSlice<T>`] whose underlying memory is owned by
    /// the runtime (typed view via cudarc's `upgrade_device_ptr::<T>`,
    /// freed through the runtime on drop). The legacy cudarc path is
    /// only used when the manager is built via [`new`] (no runtime
    /// attached). Provider construction does not yet require the
    /// runtime; callers that want runtime-routed allocations opt in
    /// here.
    pub(crate) fn with_runtime(
        device: Arc<CudaDevice>,
        budget: MemoryBudget,
        runtime: Arc<XlogDeviceRuntime>,
    ) -> Self {
        Self {
            device,
            budget,
            accounting: Arc::new(GpuMemoryAccounting::default()),
            runtime: Some(runtime),
            #[cfg(test)]
            after_local_reservation_hook: std::sync::Mutex::new(None),
        }
    }

    /// Atomically reserve `bytes` for one bounded multi-allocation request.
    ///
    /// The returned token owns the complete local-budget claim. A
    /// runtime-backed manager also reserves the same bytes against the
    /// runtime resource stack's finite global budget before touching local
    /// accounting; runtimes without a reservable global budget are refused.
    /// Creating the token performs no device allocation and does not increment
    /// [`alloc_count`](Self::alloc_count).
    pub fn reserve_bytes(self: &Arc<Self>, bytes: u64) -> Result<GpuMemoryReservation> {
        let runtime_reservation = match &self.runtime {
            Some(runtime) => {
                let bytes_usize = usize::try_from(bytes).map_err(|_| {
                    XlogError::Kernel(format!(
                        "GPU reservation size {} bytes exceeds platform usize",
                        bytes
                    ))
                })?;
                Some(runtime.reserve_memory(bytes_usize).map_err(|error| {
                    map_resource_error(error, self.accounting.peak.load(Ordering::SeqCst))
                })?)
            }
            None => None,
        };
        self.reserve_local_bytes(bytes, "manager_reserve")?;
        Ok(GpuMemoryReservation {
            manager: Arc::clone(self),
            runtime_reservation,
            total_bytes: bytes,
            remaining_bytes: bytes,
        })
    }

    fn reserve_local_bytes(&self, bytes: u64, layer: &'static str) -> Result<()> {
        let _mutation = self
            .accounting
            .mutation_lock
            .lock()
            .expect("GPU memory accounting poisoned");
        loop {
            let current = self.accounting.budget_reserved.load(Ordering::SeqCst);
            let required = current as u128 + bytes as u128;
            if required > self.budget.device_bytes as u128 {
                return Err(MemoryPressure {
                    layer,
                    current_bytes: current as u128,
                    requested_bytes: bytes as u128,
                    budget_bytes: self.budget.device_bytes,
                    prior_peak_bytes: self.accounting.peak.load(Ordering::SeqCst),
                }
                .into_error());
            }
            if self
                .accounting
                .budget_reserved
                .compare_exchange(current, required as u64, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    /// Borrow the attached device runtime, if any. `None` when the
    /// manager was constructed via [`new`].
    ///
    /// This accessor is the supported bridge for provider operations
    /// that receive runtime-owned allocations through the memory manager
    /// and must submit recorded work through
    /// [`crate::launch::LaunchRecorder`]. Code that already owns the
    /// runtime should use that `Arc<XlogDeviceRuntime>` directly.
    pub fn runtime(&self) -> Option<&Arc<XlogDeviceRuntime>> {
        self.runtime.as_ref()
    }

    /// Complete stream-ordered frees that are pending in the attached device
    /// runtime.
    ///
    /// Diagnostic transactions use this after dropping temporary buffers on
    /// an error path so a later operation observes the restored byte budget.
    pub fn reap_pending_deallocations(&self) -> Result<()> {
        let Some(runtime) = self.runtime.as_ref() else {
            crate::cuda_graph::reap_capture_retirements();
            return Ok(());
        };
        runtime
            .reap_pending()
            .map_err(|error| map_resource_error(error, self.peak_bytes()))
    }

    /// Release a local-budget reservation that was not admitted by the
    /// underlying allocator. Admitted accounting is untouched because the
    /// request was never published there.
    fn rollback_local_reservation(&self, bytes: u64) -> Result<()> {
        self.accounting.release_reserved(bytes)
    }

    fn record_deallocation_failure(&self, bytes: u64) {
        let _ = self.accounting.deallocation_failure_count.fetch_update(
            Ordering::SeqCst,
            Ordering::SeqCst,
            |current| Some(current.saturating_add(1)),
        );
        let _ = self.accounting.deallocation_failure_bytes.fetch_update(
            Ordering::SeqCst,
            Ordering::SeqCst,
            |current| Some(current.saturating_add(bytes)),
        );
    }

    /// Allocate `len` elements through the attached runtime or the shared raw
    /// allocator. Allocation completes before publication; retained failures
    /// keep their actual allocation and local byte claim.
    pub fn alloc<T: cudarc::driver::DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<TrackedCudaSlice<T>> {
        // Count every device allocation request (resettable no-host-gate counter).
        self.accounting.alloc_count.fetch_add(1, Ordering::Relaxed);

        // Fix Issue 2: Use checked_mul to prevent integer overflow before cast
        let bytes = (len as u64)
            .checked_mul(std::mem::size_of::<T>() as u64)
            .ok_or_else(|| XlogError::Kernel("Allocation size overflow".to_string()))?;

        self.reserve_local_bytes(bytes, "manager_alloc")?;
        let attempt = LocalAllocationAttempt::new(Arc::clone(&self.accounting), bytes, None);
        self.alloc_after_local_reservation::<T>(len, bytes, None, Arc::clone(&attempt.reclamation))
            .map_err(|error| map_resource_error(error, self.peak_bytes()))
    }

    fn alloc_after_local_reservation<T: cudarc::driver::DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
        bytes: u64,
        runtime_reservation: Option<&mut RuntimeMemoryReservation>,
        reclamation: Arc<AllocationReclamation>,
    ) -> crate::device_runtime::ResourceResult<TrackedCudaSlice<T>> {
        let extent = usize::try_from(bytes)
            .map_err(|_| ResourceError::Driver("allocation size exceeds platform usize".into()))?;
        if self.runtime.is_some() && bytes != 0 {
            let allocation = self.alloc_raw_after_local_reservation(
                extent,
                bytes,
                AllocTag::UNTAGGED,
                runtime_reservation,
                reclamation,
            )?;
            let raw_ptr = allocation.device_block().ptr;
            return Ok(TrackedCudaSlice {
                storage: DeviceStorage::new(
                    Backing::Runtime(allocation),
                    Arc::clone(self.device.inner().stream()),
                    raw_ptr,
                ),
                ptr: raw_ptr,
                len,
                element: std::marker::PhantomData,
            });
        }

        #[cfg(test)]
        self.run_after_local_reservation_hook(bytes);
        let allocation = RawDeviceAllocation::allocate(
            Arc::clone(self.device.inner().stream()),
            Arc::clone(self.device.inner().allocation_stream()),
            extent,
            Some(Arc::clone(self)),
            reclamation,
        )?;
        let raw_ptr = allocation.ptr();
        Ok(TrackedCudaSlice {
            storage: DeviceStorage::new(
                Backing::Native(allocation),
                Arc::clone(self.device.inner().stream()),
                raw_ptr,
            ),
            ptr: raw_ptr,
            len,
            element: std::marker::PhantomData,
        })
    }

    /// Check if an allocation of `bytes` would exceed the budget
    ///
    /// # Arguments
    /// * `bytes` - Number of bytes to allocate
    ///
    /// # Returns
    /// `Ok(())` if allocation is within budget
    ///
    /// # Errors
    /// `XlogError::ResourceExhausted` if allocation would exceed budget
    pub fn check_budget(&self, bytes: u64) -> Result<()> {
        // Include provisional local reservations: this is a budget-admission
        // query, not a sample of already admitted allocations.
        let current = self.accounting.budget_reserved.load(Ordering::SeqCst);
        let required = current as u128 + bytes as u128;

        if required > self.budget.device_bytes as u128 {
            return Err(MemoryPressure {
                layer: "manager_check_budget",
                current_bytes: current as u128,
                requested_bytes: bytes as u128,
                budget_bytes: self.budget.device_bytes,
                prior_peak_bytes: self.accounting.peak.load(Ordering::SeqCst),
            }
            .into_error());
        }

        Ok(())
    }

    /// Get the current allocated memory in bytes
    pub fn allocated_bytes(&self) -> u64 {
        self.accounting.allocated.load(Ordering::SeqCst)
    }

    /// Number of runtime deallocations that returned an error. Their bytes
    /// remain charged locally because physical release was not proven.
    pub fn deallocation_failure_count(&self) -> u64 {
        self.accounting
            .deallocation_failure_count
            .load(Ordering::SeqCst)
    }

    /// Total bytes retained in local accounting after runtime deallocation
    /// errors.
    pub fn deallocation_failure_bytes(&self) -> u64 {
        self.accounting
            .deallocation_failure_bytes
            .load(Ordering::SeqCst)
    }

    /// High-water mark of successful manager-accounted reservations since
    /// construction or the last [`reset_peak`](Self::reset_peak). Direct CUDA
    /// allocations that bypass this manager are not included.
    pub fn peak_bytes(&self) -> u64 {
        self.accounting.peak.load(Ordering::SeqCst)
    }

    /// Reset the peak high-water mark to the *current* allocated
    /// level, so a measurement window starts from live state rather
    /// than zero. Measurement-harness API.
    pub fn reset_peak(&self) {
        self.accounting.peak.store(
            self.accounting.allocated.load(Ordering::SeqCst),
            Ordering::SeqCst,
        );
    }

    /// Number of `alloc` calls issued so far (device allocation requests).
    /// The GPU-resident MC engine snapshots this around the measured region to
    /// prove `per_operator_host_allocations == 0` (all arenas pre-allocated).
    pub fn alloc_count(&self) -> u64 {
        self.accounting.alloc_count.load(Ordering::Relaxed)
    }

    /// Reset the allocation-request counter to zero.
    pub fn reset_alloc_count(&self) {
        self.accounting.alloc_count.store(0, Ordering::Relaxed);
    }

    /// Get the memory budget
    pub fn budget(&self) -> &MemoryBudget {
        &self.budget
    }

    /// Total local budget enforced jointly by this manager and any overlays.
    pub fn budget_limit_bytes(&self) -> u64 {
        self.budget.device_bytes
    }

    /// Get the underlying CUDA device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Validate a caller-requested manual accounting release.
    ///
    /// Tracked allocations release themselves through their private owner
    /// path. A public caller cannot authenticate ownership, so nonzero manual
    /// releases are refused while bytes are live and underflow is rejected
    /// without mutating either counter.
    pub fn record_free(&self, bytes: u64) -> Result<()> {
        let _mutation = self
            .accounting
            .mutation_lock
            .lock()
            .expect("GPU memory accounting poisoned");
        let admitted = self.accounting.allocated.load(Ordering::SeqCst);
        let reserved = self.accounting.budget_reserved.load(Ordering::SeqCst);
        if admitted != 0 || reserved != 0 {
            return Err(XlogError::Kernel(format!(
                "manual GPU accounting release refused with live tracked bytes: admitted_bytes={} reserved_bytes={}",
                admitted, reserved
            )));
        }
        if bytes != 0 {
            return Err(XlogError::Kernel(format!(
                "manual GPU accounting release underflow: current_bytes=0 requested_bytes={}",
                bytes
            )));
        }
        Ok(())
    }

    /// v0.6 device-runtime entry point: allocate `bytes` raw bytes
    /// through the attached [`XlogDeviceRuntime`].
    ///
    /// Returns a [`RuntimeAllocBlock`] that owns the allocation. On
    /// drop, the block deallocates through the runtime and updates
    /// both the manager's local `allocated` counter and the
    /// runtime's bookkeeping.
    ///
    /// Both budgets apply: the manager's local
    /// `MemoryBudget::device_bytes` AND any `GlobalDeviceBudget`
    /// stacked above the runtime's underlying resource. Either
    /// rejecting the request returns an `XlogError`. On runtime
    /// rejection the local reservation is rolled back so subsequent
    /// allocations see consistent state.
    ///
    /// # Errors
    /// * `XlogError::Kernel` if no runtime is attached.
    /// * `XlogError::ResourceExhausted` if the local budget cannot
    ///   accommodate the request.
    /// * `XlogError::ResourceExhausted` if the runtime's budget rejects the
    ///   request. Other runtime errors are reported as `XlogError::Kernel`.
    pub fn alloc_raw(self: &Arc<Self>, bytes: usize, tag: AllocTag) -> Result<RuntimeAllocBlock> {
        if self.runtime.is_none() {
            return Err(XlogError::Kernel(
                "GpuMemoryManager::alloc_raw called without an attached XlogDeviceRuntime; \
                 construct via with_runtime to enable runtime routing"
                    .to_string(),
            ));
        }
        let bytes_u64 = u64::try_from(bytes)
            .map_err(|_| XlogError::Kernel("Allocation size overflow".to_string()))?;
        self.reserve_local_bytes(bytes_u64, "manager_alloc_raw")?;
        let attempt = LocalAllocationAttempt::new(Arc::clone(&self.accounting), bytes_u64, None);
        self.alloc_raw_after_local_reservation(
            bytes,
            bytes_u64,
            tag,
            None,
            Arc::clone(&attempt.reclamation),
        )
        .map_err(|error| map_resource_error(error, self.peak_bytes()))
    }

    fn alloc_raw_after_local_reservation(
        self: &Arc<Self>,
        bytes: usize,
        bytes_u64: u64,
        tag: AllocTag,
        runtime_reservation: Option<&mut RuntimeMemoryReservation>,
        reclamation: Arc<AllocationReclamation>,
    ) -> crate::device_runtime::ResourceResult<RuntimeAllocBlock> {
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            ResourceError::Driver("raw allocation requires an attached device runtime".into())
        })?;
        #[cfg(test)]
        self.run_after_local_reservation_hook(bytes_u64);
        let request = AllocationRequest {
            bytes,
            stream: StreamId::DEFAULT,
            tag,
            reservation_pressure_bytes: 0,
            reclamation: Arc::clone(&reclamation),
        };
        let allocation = match runtime_reservation {
            Some(reservation) => reservation.materialize(request),
            None => runtime.materialize(request),
        };
        match allocation {
            Ok(block) => {
                // Arm the allocated block and backend before any fallible lookup.
                let owner = ResourceBlockRetirement::new(
                    block,
                    runtime.retirement_resource(),
                    Arc::clone(&reclamation),
                );
                let bind_charge: crate::device_runtime::ResourceResult<_> = (|| {
                    let block = owner.block();
                    let dependencies =
                        runtime
                            .allocation_dependencies(BlockId::from_block(block), block.bytes)?
                            .ok_or_else(|| {
                                ResourceError::Driver(
                            "runtime allocation does not provide physical release ownership".into())
                            })?;
                    if !Arc::ptr_eq(&dependencies.reclamation, &reclamation) {
                        return Err(ResourceError::Driver(
                            "runtime allocation replaced its accounting owner".into(),
                        ));
                    }
                    let allocation = dependencies.retain_allocation()?;
                    Ok((dependencies, allocation))
                })();
                let (dependencies, allocation) = match bind_charge {
                    Ok(owners) => owners,
                    Err(error) => {
                        return Err(error.retaining(bytes, reclamation));
                    }
                };
                let block = owner.into_block();
                Ok(RuntimeAllocBlock {
                    bytes: bytes_u64,
                    manager: Arc::clone(self),
                    runtime: Arc::clone(runtime),
                    block: Some(block),
                    dependencies,
                    allocation: Some(allocation),
                })
            }
            Err(error) => Err(error),
        }
    }

    /// Get remaining budget in bytes
    pub fn remaining_bytes(&self) -> u64 {
        let reserved = self.accounting.budget_reserved.load(Ordering::SeqCst);
        self.budget.device_bytes.saturating_sub(reserved)
    }

    /// Reset diagnostic tracking only when no tracked bytes are live.
    pub fn reset_tracking(&self) -> Result<()> {
        let _mutation = self
            .accounting
            .mutation_lock
            .lock()
            .expect("GPU memory accounting poisoned");
        let admitted = self.accounting.allocated.load(Ordering::SeqCst);
        let reserved = self.accounting.budget_reserved.load(Ordering::SeqCst);
        if admitted != 0 || reserved != 0 {
            return Err(XlogError::Kernel(format!(
                "GPU accounting reset refused with live tracked bytes: admitted_bytes={} reserved_bytes={}",
                admitted, reserved
            )));
        }
        self.accounting.peak.store(0, Ordering::SeqCst);
        Ok(())
    }

    #[cfg(test)]
    fn run_after_local_reservation_hook(&self, bytes: u64) {
        let hook = self
            .after_local_reservation_hook
            .lock()
            .expect("after-local-reservation test hook poisoned")
            .clone();
        if let Some(hook) = hook {
            hook(bytes);
        }
    }
}

fn map_resource_error(e: ResourceError, prior_peak_bytes: u64) -> XlogError {
    match e {
        error @ ResourceError::AllocationRetained(_) => XlogError::Kernel(error.to_string()),
        ResourceError::OutOfBudget {
            requested,
            current,
            limit,
            ..
        } => MemoryPressure {
            layer: "device_runtime",
            current_bytes: current as u128,
            requested_bytes: requested as u128,
            budget_bytes: u64::try_from(limit).unwrap_or(u64::MAX),
            prior_peak_bytes,
        }
        .into_error(),
        ResourceError::Driver(msg) => XlogError::Kernel(format!("device-runtime driver: {}", msg)),
        ResourceError::StreamMisuse(msg) => {
            XlogError::Kernel(format!("device-runtime stream misuse: {}", msg))
        }
        ResourceError::UseAfterFree { generation } => XlogError::Kernel(format!(
            "device-runtime use-after-free on generation {:?}",
            generation
        )),
        ResourceError::OutOfBounds { generation } => XlogError::Kernel(format!(
            "device-runtime out-of-bounds on generation {:?}",
            generation
        )),
    }
}

/// Owned handle for a raw allocation routed through the device runtime.
/// Both raw allocations and typed storage views retain this handle. Drop hands
/// its block identity to the runtime; the actual raw owner's successful physical
/// reclamation returns the local manager charge, including on another reaper.
pub struct RuntimeAllocBlock {
    bytes: u64,
    manager: Arc<GpuMemoryManager>,
    runtime: Arc<XlogDeviceRuntime>,
    dependencies: Arc<DeviceAccessDependencies>,
    allocation: Option<Arc<RawDeviceAllocation>>,
    /// `None` after Drop fires; `Some(_)` while the block is live.
    /// Wrapped in Option so `Drop` can move the block out and pass
    /// it by value to `runtime.deallocate`.
    block: Option<DeviceBlock>,
}

fn allocation_reclamation_outcome(
    physically_released: bool,
    queue_result: crate::device_runtime::ResourceResult<()>,
) -> crate::device_runtime::ResourceResult<()> {
    if physically_released {
        return Ok(());
    }
    queue_result?;
    Err(ResourceError::Driver(
        "allocation reclamation is still owned by another pending reaper".into(),
    ))
}

impl RuntimeAllocBlock {
    fn release(&mut self) -> crate::device_runtime::ResourceResult<()> {
        // The backend already owns the actual raw allocation. Dropping these
        // bookkeeping references must not strand a runtime after a late reap.
        // Backend Drop itself transfers unproven allocations to cold retirement.
        let Some(block) = self.block.take() else {
            return allocation_reclamation_outcome(
                self.dependencies.allocation_was_released(),
                Ok(()),
            );
        };
        // The backend still owns the live allocation. Relinquish this wrapper's
        // lease before asking the reaper whether any other real owner remains.
        drop(self.allocation.take());
        let mut retirement = ResourceBlockRetirement::new(
            block,
            self.runtime.retirement_resource(),
            Arc::clone(&self.dependencies.reclamation),
        );
        let queue_result = retirement.release();
        match allocation_reclamation_outcome(
            self.dependencies.allocation_was_released(),
            queue_result,
        ) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.manager.record_deallocation_failure(self.bytes);
                Err(error)
            }
        }
    }

    /// Raw device pointer for this allocation. Live until the
    /// block is dropped.
    pub fn ptr(&self) -> u64 {
        self.block
            .as_ref()
            .expect("RuntimeAllocBlock used after drop")
            .ptr
    }

    /// Allocation size in bytes.
    pub fn bytes(&self) -> usize {
        self.bytes as usize
    }

    /// Borrow the underlying [`DeviceBlock`] metadata. Test/
    /// diagnostic accessor.
    pub fn device_block(&self) -> &DeviceBlock {
        self.block
            .as_ref()
            .expect("RuntimeAllocBlock used after drop")
    }
}

impl std::fmt::Debug for RuntimeAllocBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("RuntimeAllocBlock");
        dbg.field("bytes", &self.bytes);
        match &self.block {
            Some(b) => {
                dbg.field("ptr", &format_args!("{:#x}", b.ptr));
                dbg.field("device_ordinal", &b.device_ordinal);
                dbg.field("alloc_stream", &b.alloc_stream);
                dbg.field("tag", &b.tag);
                dbg.field("generation", &b.generation);
                dbg.field("state", &b.state);
            }
            None => {
                dbg.field("block", &"<dropped>");
            }
        }
        dbg.finish()
    }
}

impl Drop for RuntimeAllocBlock {
    fn drop(&mut self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.release()));
    }
}

/// Column data stored in device memory.
///
/// Most columns are owned by XLOG (`Owned`) and tracked against the memory budget. Columns may
/// also be imported via DLPack (`Dlpack`) or Arrow device (`ArrowDevice`) without copies; these are
/// freed via the DLPack deleter or Arrow release callback.
pub enum CudaColumn {
    Owned(TrackedCudaSlice<u8>),
    Dlpack(DlpackColumn),
    ArrowDevice(ArrowDeviceColumn),
}

pub struct DlpackColumn {
    ptr: cudarc::driver::sys::CUdeviceptr,
    len_bytes: usize,
    stream: Arc<CudaStream>,
    storage: Arc<DeviceStorage>,
    /// `Some` when this DLPack column wraps memory that xlog
    /// itself owns through the device runtime — i.e. the
    /// caller exported an xlog-allocated slice via DLPack and
    /// kept ownership inside xlog. The strong reference keeps
    /// the source slice's [`crate::device_runtime::DeviceBlock`]
    /// reachable for runtime-block identity propagation, and
    /// keeps the underlying allocation alive across the
    /// DLPack handoff (drop order: column → tensor →
    /// `source_slice` → `runtime.deallocate`).
    ///
    /// `None` for true external DLPack producers; those
    /// columns continue to be rejected by strict-mode launch
    /// recorders.
    source_slice: Option<Arc<TrackedCudaSlice<u8>>>,
}

pub struct ArrowDeviceColumn {
    ptr: cudarc::driver::sys::CUdeviceptr,
    len_bytes: usize,
    stream: Arc<CudaStream>,
    storage: Arc<DeviceStorage>,
    /// Same role as [`DlpackColumn::source_slice`]: `Some` for
    /// xlog-owned Arrow device columns, `None` for true
    /// external Arrow producers.
    source_slice: Option<Arc<TrackedCudaSlice<u8>>>,
}

impl CudaColumn {
    pub fn owned(slice: TrackedCudaSlice<u8>) -> Self {
        Self::Owned(slice)
    }

    /// Construct a column over an authenticated foreign allocation.
    ///
    /// # Safety
    /// The byte range must be live in `stream`'s CUDA context and retained by
    /// `tensor`. The producer must have completed its work before handoff and
    /// must coordinate subsequent non-XLOG accesses for the whole import lifetime.
    pub unsafe fn dlpack(
        ptr: cudarc::driver::sys::CUdeviceptr,
        len_bytes: usize,
        stream: Arc<CudaStream>,
        tensor: DlpackManagedTensor,
    ) -> Self {
        let storage = DeviceStorage::new(
            Backing::Foreign {
                _owner: Box::new(tensor),
                _source: None,
                bytes: len_bytes as u64,
            },
            Arc::clone(&stream),
            ptr,
        );
        Self::Dlpack(DlpackColumn {
            ptr,
            len_bytes,
            stream,
            storage,
            source_slice: None,
        })
    }

    /// Construct a DLPack column that wraps memory **xlog
    /// itself owns** through the device runtime.
    ///
    /// Use this when xlog allocated `source_slice` via the
    /// runtime-backed manager and is exporting it as a DLPack
    /// tensor for inspection by external code while retaining
    /// ownership. The resulting column reports
    /// [`Self::is_external`] as `false` and
    /// [`Self::runtime_block`] returns the slice's
    /// [`crate::device_runtime::DeviceBlock`] — strict-mode
    /// launch recorders will record it normally instead of
    /// rejecting.
    ///
    /// True external DLPack producers (DLPack tensors handed
    /// to xlog by another framework) must continue to use
    /// [`Self::dlpack`].
    pub fn dlpack_xlog_owned(
        source_slice: Arc<TrackedCudaSlice<u8>>,
        stream: Arc<CudaStream>,
        tensor: DlpackManagedTensor,
    ) -> Self {
        let ptr = *source_slice.device_ptr();
        let len_bytes = source_slice.len();
        assert_eq!(
            source_slice.stream().context().cu_ctx(),
            stream.context().cu_ctx(),
            "import stream must belong to the source allocation context"
        );
        let storage = DeviceStorage::new(
            Backing::Foreign {
                _owner: Box::new(tensor),
                _source: Some(Arc::clone(&source_slice)),
                bytes: len_bytes as u64,
            },
            Arc::clone(&stream),
            ptr,
        );
        Self::Dlpack(DlpackColumn {
            ptr,
            len_bytes,
            stream,
            storage,
            source_slice: Some(source_slice),
        })
    }

    /// Construct a column over an authenticated Arrow device allocation.
    ///
    /// # Safety
    /// `import` must retain this live byte range in `stream`'s CUDA context.
    /// Producer work must be complete and subsequent non-XLOG access coordinated
    /// for the entire lifetime of this imported storage.
    pub unsafe fn arrow_device(
        ptr: cudarc::driver::sys::CUdeviceptr,
        len_bytes: usize,
        stream: Arc<CudaStream>,
        import: Arc<ArrowDeviceImport>,
    ) -> Self {
        let storage = DeviceStorage::new(
            Backing::Foreign {
                _owner: Box::new(import),
                _source: None,
                bytes: len_bytes as u64,
            },
            Arc::clone(&stream),
            ptr,
        );
        Self::ArrowDevice(ArrowDeviceColumn {
            ptr,
            len_bytes,
            stream,
            storage,
            source_slice: None,
        })
    }

    /// Construct an Arrow device column that wraps memory
    /// **xlog itself owns** through the device runtime. Same
    /// contract as [`Self::dlpack_xlog_owned`]: identity is
    /// preserved, strict recorders accept the column, and
    /// drop order keeps the underlying allocation alive
    /// through the Arrow handoff.
    ///
    /// True external Arrow device producers must continue to
    /// use [`Self::arrow_device`].
    pub fn arrow_device_xlog_owned(
        source_slice: Arc<TrackedCudaSlice<u8>>,
        stream: Arc<CudaStream>,
        import: Arc<ArrowDeviceImport>,
    ) -> Self {
        let ptr = *source_slice.device_ptr();
        let len_bytes = source_slice.len();
        assert_eq!(
            source_slice.stream().context().cu_ctx(),
            stream.context().cu_ctx(),
            "import stream must belong to the source allocation context"
        );
        let storage = DeviceStorage::new(
            Backing::Foreign {
                _owner: Box::new(import),
                _source: Some(Arc::clone(&source_slice)),
                bytes: len_bytes as u64,
            },
            Arc::clone(&stream),
            ptr,
        );
        Self::ArrowDevice(ArrowDeviceColumn {
            ptr,
            len_bytes,
            stream,
            storage,
            source_slice: Some(source_slice),
        })
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        match self {
            CudaColumn::Owned(slice) => slice.stream(),
            CudaColumn::Dlpack(col) => &col.stream,
            CudaColumn::ArrowDevice(col) => &col.stream,
        }
    }

    pub fn device_ptr(&self) -> &cudarc::driver::sys::CUdeviceptr {
        match self {
            CudaColumn::Owned(slice) => slice.device_ptr(),
            CudaColumn::Dlpack(col) => &col.ptr,
            CudaColumn::ArrowDevice(col) => &col.ptr,
        }
    }

    /// Stable identity of the xlog memory manager that owns this column.
    ///
    /// True external DLPack and Arrow device columns return `None`; wrappers
    /// retaining an xlog-owned source slice preserve that slice's identity.
    pub fn memory_manager_ptr_value(&self) -> Option<usize> {
        match self {
            CudaColumn::Owned(slice) => Some(slice.memory_manager_ptr_value()),
            CudaColumn::Dlpack(col) => col
                .source_slice
                .as_ref()
                .map(|slice| slice.memory_manager_ptr_value()),
            CudaColumn::ArrowDevice(col) => col
                .source_slice
                .as_ref()
                .map(|slice| slice.memory_manager_ptr_value()),
        }
    }

    pub(crate) fn runtime_allocation_identity(&self) -> Result<Option<RuntimeAllocationIdentity>> {
        match self {
            CudaColumn::Owned(slice) => slice.runtime_allocation_identity(),
            CudaColumn::Dlpack(col) => {
                let Some(source) = &col.source_slice else {
                    return Ok(None);
                };
                let mut identity = source.runtime_allocation_identity()?;
                if let Some(identity) = &mut identity {
                    identity.allocation_ptr = col.ptr;
                    identity.allocation_bytes = col.len_bytes;
                }
                Ok(identity)
            }
            CudaColumn::ArrowDevice(col) => {
                let Some(source) = &col.source_slice else {
                    return Ok(None);
                };
                let mut identity = source.runtime_allocation_identity()?;
                if let Some(identity) = &mut identity {
                    identity.allocation_ptr = col.ptr;
                    identity.allocation_bytes = col.len_bytes;
                }
                Ok(identity)
            }
        }
    }

    /// Borrow the underlying [`crate::device_runtime::DeviceBlock`].
    ///
    /// Returns `Some(&block)` when xlog owns the memory through
    /// the runtime — `Owned` slices that were allocated via a
    /// runtime-backed manager, AND `Dlpack` / `ArrowDevice`
    /// columns constructed via the `*_xlog_owned` constructors
    /// (where the source slice's block is reachable through
    /// the retained `Arc<TrackedCudaSlice<u8>>`).
    ///
    /// Returns `None` for legacy cudarc-backed `Owned` slices
    /// (no runtime block exists) and for true external
    /// `Dlpack` / `ArrowDevice` columns (xlog never owned the
    /// allocation). Strict-mode launch recorders reject `None`
    /// returns; permissive recorders silently skip.
    pub fn runtime_block(&self) -> Option<&crate::device_runtime::DeviceBlock> {
        match self {
            CudaColumn::Owned(slice) => slice.runtime_block(),
            CudaColumn::Dlpack(col) => col.source_slice.as_ref().and_then(|s| s.runtime_block()),
            CudaColumn::ArrowDevice(col) => {
                col.source_slice.as_ref().and_then(|s| s.runtime_block())
            }
        }
    }

    /// Whether this column wraps externally-managed device
    /// memory.
    ///
    /// Returns `true` only for `Dlpack` / `ArrowDevice` columns
    /// where xlog never owned the allocation (no `source_slice`).
    /// `Dlpack` / `ArrowDevice` columns built via
    /// `*_xlog_owned` constructors return `false` — xlog still
    /// owns the memory; the DLPack / Arrow handle is just an
    /// export wrapper.
    ///
    /// External memory has no xlog-side runtime identity;
    /// strict launch recorders reject such columns and require
    /// callers to coordinate cross-stream synchronization
    /// themselves.
    pub fn is_external(&self) -> bool {
        match self {
            CudaColumn::Owned(_) => false,
            CudaColumn::Dlpack(col) => col.source_slice.is_none(),
            CudaColumn::ArrowDevice(col) => col.source_slice.is_none(),
        }
    }
}

impl From<TrackedCudaSlice<u8>> for CudaColumn {
    fn from(value: TrackedCudaSlice<u8>) -> Self {
        CudaColumn::Owned(value)
    }
}

impl DeviceSlice<u8> for CudaColumn {
    fn len(&self) -> usize {
        match self {
            CudaColumn::Owned(slice) => slice.len(),
            CudaColumn::Dlpack(col) => col.len_bytes,
            CudaColumn::ArrowDevice(col) => col.len_bytes,
        }
    }

    fn stream(&self) -> &Arc<CudaStream> {
        self.stream()
    }
}

impl private_access::Sealed for CudaColumn {}

impl DeviceRead<u8> for CudaColumn {
    fn device_view(&self) -> DeviceMemoryView<u8> {
        match self {
            CudaColumn::Owned(slice) => slice.view(),
            CudaColumn::Dlpack(col) => DeviceMemoryView {
                ptr: col.ptr,
                len: col.len_bytes,
                storage: Arc::clone(&col.storage),
                element: std::marker::PhantomData,
            },
            CudaColumn::ArrowDevice(col) => DeviceMemoryView {
                ptr: col.ptr,
                len: col.len_bytes,
                storage: Arc::clone(&col.storage),
                element: std::marker::PhantomData,
            },
        }
    }
}

impl DeviceWrite<u8> for CudaColumn {}

impl AsKernelParam for &CudaColumn {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        ((self.device_ptr()) as *const cudarc::driver::sys::CUdeviceptr)
            .cast_mut()
            .cast()
    }
}

impl AsKernelParam for &mut CudaColumn {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        ((self.device_ptr()) as *const cudarc::driver::sys::CUdeviceptr)
            .cast_mut()
            .cast()
    }
}

impl<'a> IntoKernelParamStorage for &'a CudaColumn {
    type Storage = DeviceParamStorage<'a>;

    fn into_kernel_param_storage(self) -> Self::Storage {
        match self {
            CudaColumn::Owned(slice) => slice.into_kernel_param_storage(),
            CudaColumn::Dlpack(col) => DeviceParamStorage::unsynced(col.ptr),
            CudaColumn::ArrowDevice(col) => DeviceParamStorage::unsynced(col.ptr),
        }
    }
}

impl<'a> IntoKernelParamStorage for &'a mut CudaColumn {
    type Storage = DeviceParamStorage<'a>;

    fn into_kernel_param_storage(self) -> Self::Storage {
        match self {
            CudaColumn::Owned(slice) => slice.into_kernel_param_storage(),
            CudaColumn::Dlpack(col) => DeviceParamStorage::unsynced(col.ptr),
            CudaColumn::ArrowDevice(col) => DeviceParamStorage::unsynced(col.ptr),
        }
    }
}

/// Column-oriented GPU buffer
///
/// Holds columnar data on the GPU with an associated schema.
/// Each column is stored as a separate `CudaSlice<u8>`.
pub struct CudaBuffer {
    /// Column data stored as raw bytes
    pub(crate) columns: Vec<CudaColumn>,
    /// Row capacity for allocated columns
    pub(crate) row_cap: u64,
    /// Device-resident row count (len = 1)
    pub(crate) d_num_rows: TrackedCudaSlice<u32>,
    /// Schema describing the column types
    pub(crate) schema: Schema,
    /// Cached host-side row count (u32::MAX = not yet cached).
    /// Avoids repeated synchronous D2H transfers between explicit mutations,
    /// whose public accessors invalidate this cache before returning.
    cached_row_count: AtomicU32,
    /// True only when construction or a set operation proves that rows are
    /// lexicographically sorted by every schema column and full-row unique.
    canonical_full_row_set_certified: bool,
}

impl CudaBuffer {
    /// Create a buffer from existing columns
    ///
    /// # Arguments
    /// * `columns` - Pre-allocated column data
    /// * `row_cap` - Row capacity for the buffer
    /// * `d_num_rows` - Device-resident row count
    /// * `schema` - Schema describing the columns
    ///
    /// # Panics
    /// Panics if the number of columns doesn't match the schema arity
    pub fn from_columns(
        columns: Vec<CudaColumn>,
        row_cap: u64,
        d_num_rows: TrackedCudaSlice<u32>,
        schema: Schema,
    ) -> Self {
        assert_eq!(
            columns.len(),
            schema.arity(),
            "Number of columns ({}) must match schema arity ({})",
            columns.len(),
            schema.arity()
        );
        Self {
            columns,
            row_cap,
            d_num_rows,
            schema,
            cached_row_count: AtomicU32::new(u32::MAX),
            canonical_full_row_set_certified: false,
        }
    }

    /// Like `from_columns`, but eagerly populates the row-count cache.
    /// Use when the host already knows the exact row count (e.g., `buffer_from_columns`).
    pub fn from_columns_with_host_count(
        columns: Vec<CudaColumn>,
        row_cap: u64,
        d_num_rows: TrackedCudaSlice<u32>,
        schema: Schema,
        host_row_count: u32,
    ) -> Self {
        assert_eq!(
            columns.len(),
            schema.arity(),
            "Number of columns ({}) must match schema arity ({})",
            columns.len(),
            schema.arity()
        );
        Self {
            columns,
            row_cap,
            d_num_rows,
            schema,
            cached_row_count: AtomicU32::new(host_row_count),
            canonical_full_row_set_certified: false,
        }
    }

    /// Returns the cached row count if available (not sentinel `u32::MAX`).
    pub fn cached_row_count(&self) -> Option<u32> {
        let v = self.cached_row_count.load(Ordering::Relaxed);
        if v == u32::MAX {
            None
        } else {
            Some(v)
        }
    }

    /// Sets the cached row count if not already set (CAS from sentinel).
    /// No-op if already cached.
    pub(crate) fn set_cached_row_count_if_unset(&self, count: u32) {
        let _ = self.cached_row_count.compare_exchange(
            u32::MAX,
            count,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }

    /// Whether rows are sorted in schema-column order and full-row unique.
    pub fn canonical_full_row_set_certified(&self) -> bool {
        self.canonical_full_row_set_certified
    }

    /// Record a full-schema ordering and uniqueness proof from a set operation.
    pub(crate) fn certify_canonical_full_row_set(&mut self) {
        self.canonical_full_row_set_certified = true;
    }

    /// Borrow every column without permitting mutation of certified contents.
    pub fn columns(&self) -> &[CudaColumn] {
        &self.columns
    }

    /// Mutably borrow columns after invalidating canonical-set metadata.
    pub fn columns_mut(&mut self) -> &mut [CudaColumn] {
        self.canonical_full_row_set_certified = false;
        &mut self.columns
    }

    /// Replace the schema after invalidating canonical ordering metadata.
    pub fn set_schema(&mut self, schema: Schema) {
        assert_eq!(self.columns.len(), schema.arity());
        self.schema = schema;
        self.canonical_full_row_set_certified = false;
    }

    /// Set row capacity and invalidate all host-derived row metadata.
    pub fn set_row_capacity(&mut self, row_cap: u64) {
        self.row_cap = row_cap;
        self.cached_row_count.store(u32::MAX, Ordering::Relaxed);
        self.canonical_full_row_set_certified = false;
    }

    /// Mutably borrow the device row count after invalidating derived metadata.
    pub fn num_rows_device_mut(&mut self) -> &mut TrackedCudaSlice<u32> {
        self.cached_row_count.store(u32::MAX, Ordering::Relaxed);
        self.canonical_full_row_set_certified = false;
        &mut self.d_num_rows
    }

    /// Get the row capacity
    pub fn num_rows(&self) -> u64 {
        self.row_cap
    }

    /// Get the device-resident row count
    pub fn num_rows_device(&self) -> &TrackedCudaSlice<u32> {
        &self.d_num_rows
    }

    /// Check if the buffer has zero row capacity
    pub fn is_empty(&self) -> bool {
        self.row_cap == 0
    }

    /// Get the schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get the number of columns (arity)
    pub fn arity(&self) -> usize {
        self.schema.arity()
    }

    /// Estimated memory usage in bytes
    pub fn estimated_bytes(&self) -> u64 {
        self.row_cap * self.schema.row_size_bytes() as u64
    }

    /// Get a reference to a specific column by index
    pub fn column(&self, index: usize) -> Option<&CudaColumn> {
        self.columns.get(index)
    }
}

pub fn validate_logical_row_count(row_cap: u64, logical_rows: usize) -> Result<usize> {
    let row_cap_usize = usize::try_from(row_cap)
        .map_err(|_| XlogError::Kernel(format!("Row capacity {} exceeds usize::MAX", row_cap)))?;
    if logical_rows > row_cap_usize {
        return Err(XlogError::Kernel(format!(
            "Logical row count {} exceeds row capacity {}",
            logical_rows, row_cap
        )));
    }
    debug_assert!(logical_rows <= row_cap_usize);
    Ok(logical_rows)
}

#[cfg(test)]
pub(crate) fn test_memory_manifest(
    on_drop: impl FnOnce() + Send + Sync + 'static,
) -> (Arc<MemoryAccessManifest>, std::sync::Weak<[u8]>) {
    struct Bytes<F: FnOnce()>(Arc<[u8]>, Option<F>);
    impl<F: FnOnce()> Drop for Bytes<F> {
        fn drop(&mut self) {
            if let Some(on_drop) = self.1.take() {
                on_drop();
            }
        }
    }
    impl<F: FnOnce() + Send + Sync + 'static> MemoryStorageOwner for Bytes<F> {
        fn dependencies(
            &self,
        ) -> crate::device_runtime::ResourceResult<Arc<DeviceAccessDependencies>> {
            panic!("a passive host manifest must not touch CUDA dependencies")
        }
    }
    let bytes = Arc::<[u8]>::from(vec![7; 64]);
    let weak = Arc::downgrade(&bytes);
    let owner = Arc::new(Bytes(bytes, Some(on_drop)));
    let mut manifest = MemoryAccessManifest::default();
    manifest.ranges.push((
        17,
        MemoryUse::new(owner.0.as_ptr() as u64, owner.0.len(), Access::ReadWrite).unwrap(),
    ));
    manifest.retained.push(RetainedStorageUse {
        owner,
        access: Access::ReadWrite,
    });
    (Arc::new(manifest), weak)
}

#[cfg(test)]
mod tests {
    impl crate::launch::RecorderCleanup<u8> for OperationCompletion {
        fn synchronize_retired(&self) -> ResourceResult<()> {
            if self.is_complete() {
                Ok(())
            } else {
                Err(ResourceError::Driver("completion remains unknown".into()))
            }
        }

        fn cancel_retired(&self, _use: u8) -> ResourceResult<()> {
            Ok(())
        }
    }

    #[test]
    fn retired_context_completion_retries_unknown_wait_without_changing_execution_identity() {
        let proof = OperationCompletion::new(17);
        assert!(proof
            .synchronize_retired_with(|| Err(ResourceError::Driver("context wait failed".into())))
            .is_err());
        assert!(!proof.is_complete());
        assert_eq!(proof.execution_id, 17);
        assert!(std::panic::catch_unwind(|| {
            let _ = proof.synchronize_retired_with(|| panic!("context wait panic"));
        })
        .is_err());
        assert!(!proof.is_complete());
        proof.synchronize_retired_with(|| Ok(())).unwrap();
        proof
            .synchronize_retired_with(|| panic!("completed wait must not repeat"))
            .unwrap();
        assert!(proof.is_complete());
        assert_eq!(proof.execution_id, 17);
        assert!(proof.validate_submission(17).is_err());
        assert!(proof.validate_submission(18).is_err());
    }

    #[test]
    #[ignore = "requires authorized CUDA execution"]
    fn failed_abort_reclaims_real_allocation_after_origin_thread_exit() {
        use crate::cuda_graph::{reap_capture_retirements, CapturedCudaGraph};
        use crate::device_runtime::{AsyncCudaResource, GlobalDeviceBudget, StreamPool};
        use std::sync::mpsc;

        let _serial = crate::cuda_graph::capture_lifecycle_test_guard();
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let default_stream = Arc::clone(device.inner().stream());
        let context = Arc::clone(device.inner().stream().context());
        assert!(
            context.has_async_alloc(),
            "this lifecycle regression requires real stream-ordered allocation and free"
        );
        let capture_stream = context.new_stream().unwrap();
        let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let resource = Box::new(GlobalDeviceBudget::new(
            Box::new(AsyncCudaResource::new(
                Arc::clone(&device),
                0,
                Arc::clone(&pool),
            )),
            4096,
        ));
        let runtime = Arc::new(XlogDeviceRuntime::with_resource(
            Arc::clone(&device),
            0,
            pool,
            resource,
        ));
        let manager = Arc::new(GpuMemoryManager::with_runtime(
            device,
            MemoryBudget::with_limit(4096),
            Arc::clone(&runtime),
        ));
        let (sent, received) = mpsc::channel();
        let (abort, wait_for_abort) = mpsc::channel();
        let worker_manager = Arc::clone(&manager);
        let worker = std::thread::spawn(move || {
            // Real caller-prefix work precedes the target allocation and
            // completed publication; keep its bytes for later observation.
            let stream = Arc::clone(worker_manager.device.inner().stream());
            let prefix = DeviceMemoryView::<u8>::allocate(
                Arc::clone(&stream),
                Arc::clone(worker_manager.device.inner().allocation_stream()),
                16,
            )
            .unwrap();
            with_memory_access(
                Arc::clone(&stream),
                vec![prefix.access(Access::Write).unwrap()],
                |_| {
                    // SAFETY: the admitted prefix owns these exact bytes.
                    unsafe {
                        cudarc::driver::sys::cuMemsetD8Async(
                            *prefix.device_ptr(),
                            0x3c,
                            16,
                            stream.cu_stream(),
                        )
                        .result()?;
                    }
                    Ok(())
                },
            )
            .unwrap();
            let allocation = worker_manager.alloc::<u8>(4096).unwrap();
            let raw = allocation.storage.dependencies.retain_allocation().unwrap();
            let stream = Arc::clone(allocation.stream());
            let ptr = *allocation.device_ptr();
            let submission = crate::cuda_graph::acquire_stream_submission_phase(&stream).unwrap();
            let execution_id = submission.execution_id();
            let mut operation = MemoryOperation {
                submission,
                transaction: admit_memory_access(
                    Arc::clone(&stream),
                    vec![allocation.view().access(Access::Write).unwrap()],
                    None,
                    &[],
                    execution_id,
                )
                .unwrap(),
            };
            let admitted = Arc::clone(operation.transaction.prepared_owner().unwrap());
            let completion = Arc::clone(admitted.completion());
            let owner = Arc::downgrade(&admitted);
            operation
                .transaction
                .enqueue_operation_with(
                    || {
                        let pin = operation.submission.pin()?;
                        admitted.bind_submission(&pin)?;
                        Ok(pin)
                    },
                    MemoryOperationOwner::prepare,
                    |pin| {
                        pin.with_serialized_submission(
                            || -> crate::device_runtime::ResourceResult<()> {
                                // SAFETY: this admitted operation owns the whole allocation;
                                // its real initialization dependencies were just prepared.
                                unsafe {
                                    cudarc::driver::sys::cuMemsetD8Async(
                                        ptr,
                                        0x5a,
                                        4096,
                                        stream.cu_stream(),
                                    )
                                    .result()?;
                                }
                                Ok(())
                            },
                        )
                    },
                    MemoryOperationOwner::synchronize,
                    MemoryOperationOwner::cancel,
                )
                .unwrap();
            drop(admitted);
            drop(allocation);
            sent.send((raw, completion, owner, prefix)).unwrap();
            wait_for_abort.recv().unwrap();
            // Inject only the failed original wait, not allocation, preparation,
            // enqueue, context recovery or physical free. Active real capture
            // defers cold recovery until this originating host thread has exited.
            let error = operation
                .transaction
                .abort_with(
                    |_| {
                        Err(ResourceError::Driver(
                            "injected original completion wait failure".into(),
                        ))
                    },
                    MemoryOperationOwner::cancel,
                )
                .unwrap_err();
            assert!(error
                .to_string()
                .contains("original completion wait failure"));
        });
        let (raw, completion, owner, prefix) = received.recv().unwrap();
        assert_ne!(
            completion.execution_id,
            crate::cuda_graph::stream_execution_id(&default_stream).unwrap(),
            "the reaper must not identify its own PTDS as the exited producer's execution"
        );
        let reclamation = Arc::clone(raw.reclamation());
        let graph = CapturedCudaGraph::capture_on_stream(&capture_stream, || {
            abort.send(()).unwrap();
            worker.join().unwrap();
            reap_capture_retirements();
            assert!(!completion.is_complete());
            assert!(owner.upgrade().is_some());
            assert!(!reclamation.was_released());
            assert_eq!(manager.allocated_bytes(), 4096);
            assert_eq!(runtime.bytes_outstanding(), 4096);
            assert!(matches!(
                runtime.reserve_memory(1),
                Err(ResourceError::OutOfBudget {
                    current: 4096,
                    remaining: 0,
                    ..
                })
            ));
            Ok(())
        })
        .unwrap();
        drop(graph);
        runtime.reap_pending().unwrap();
        assert!(completion.is_complete());
        assert!(owner.upgrade().is_none());
        // The separately retained real allocation prevents physical free even
        // after successful operation recovery and logical runtime detachment.
        let pending = Arc::clone(
            &reclamation
                .state
                .lock()
                .unwrap()
                .pending
                .as_ref()
                .unwrap()
                .0,
        );
        assert_eq!(pending.load(Ordering::Acquire), 4096);
        assert!(!reclamation.was_released());
        assert_eq!(manager.allocated_bytes(), 4096);
        assert_eq!(runtime.bytes_outstanding(), 4096);
        assert!(matches!(
            runtime.reserve_memory(1),
            Err(ResourceError::OutOfBudget {
                current: 4096,
                remaining: 0,
                ..
            })
        ));
        context.bind_to_thread().unwrap();
        let mut observed = [0u8; 4096];
        // SAFETY: successful context recovery completed the admitted write;
        // raw still owns all bytes and no other operation uses this allocation.
        unsafe {
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                observed.as_mut_ptr().cast(),
                raw.ptr(),
                observed.len(),
            )
            .result()
            .unwrap();
        }
        assert!(observed.iter().all(|byte| *byte == 0x5a));
        let mut observed_prefix = [0u8; 16];
        // SAFETY: the same successful context barrier covers the actual caller
        // prefix, and its original allocation is still owned by prefix.
        unsafe {
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                observed_prefix.as_mut_ptr().cast(),
                *prefix.device_ptr(),
                observed_prefix.len(),
            )
            .result()
            .unwrap();
        }
        assert_eq!(observed_prefix, [0x3c; 16]);
        drop(prefix);
        drop(raw);
        runtime.reap_pending().unwrap();
        assert!(reclamation.was_released());
        assert_eq!(pending.load(Ordering::Acquire), 0);
        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(manager.remaining_bytes(), 4096);
        assert_eq!(runtime.bytes_outstanding(), 0);
        drop(runtime.reserve_memory(4096).unwrap());
    }

    #[test]
    fn operation_completion_rejects_moved_execution_before_dependency_prepare() {
        let proof = Arc::new(OperationCompletion::new(17));
        let mut transaction =
            RecorderTransaction::from_admitted(Arc::clone(&proof), Box::new([0u8]));
        let mut cancelled = false;
        let result = transaction.enqueue_operation_with(
            || proof.validate_submission(18),
            |_| panic!("moved execution must not prepare dependencies"),
            |_| -> std::result::Result<(), &str> { panic!("moved execution must not launch") },
            |_| panic!("refused admission must not wait on another execution"),
            |_, _| {
                cancelled = true;
                Ok(())
            },
        );
        assert!(matches!(
            result,
            Err(crate::launch::LaunchEnqueueError::Preparation(_))
        ));
        assert!(cancelled);
        assert!(!proof.is_complete());
    }

    #[test]
    fn operation_completion_requires_exact_successful_synchronization() {
        let proof = OperationCompletion::new(17);
        assert!(proof.validate_submission(18).is_err());
        assert!(proof
            .synchronize_with(18, || panic!("wrong stream must not be waited"))
            .is_err());
        assert!(!proof.is_complete());
        assert!(proof
            .synchronize_with(17, || Err(ResourceError::Driver("wait failed".into())))
            .is_err());
        assert!(!proof.is_complete());
        let unwound = std::panic::catch_unwind(|| {
            let _ = proof.synchronize_with(17, || panic!("wait unwound"));
        });
        assert!(unwound.is_err());
        assert!(!proof.is_complete());
        proof.synchronize_with(17, || Ok(())).unwrap();
        assert!(proof.is_complete());
        assert!(proof.validate_submission(17).is_err());
    }

    #[test]
    fn operation_completion_survives_failed_reservation_cancellation() {
        let proof = Arc::new(OperationCompletion::new(17));
        let mut transaction =
            RecorderTransaction::from_admitted(Arc::clone(&proof), Box::new([0u8]));
        let result = transaction.enqueue_operation_with(
            || Ok(()),
            |proof| proof.validate_submission(17),
            |_| Err("record failed"),
            |proof| proof.synchronize_with(17, || Ok(())),
            |_, _| Err(ResourceError::Driver("cancellation failed".into())),
        );
        assert!(matches!(
            result,
            Err(crate::launch::LaunchEnqueueError::OperationAndCleanup { .. })
        ));
        assert!(proof.is_complete());
    }

    #[test]
    fn passive_manifest_keeps_storage_and_context_qualified_coverage() {
        struct Bytes(Box<[u8]>);
        impl MemoryStorageOwner for Bytes {
            fn dependencies(
                &self,
            ) -> crate::device_runtime::ResourceResult<Arc<DeviceAccessDependencies>> {
                panic!("a passive manifest must not touch CUDA dependencies")
            }
        }
        let bytes = Arc::new(Bytes(vec![7; 64].into_boxed_slice()));
        assert_eq!(bytes.0[0], 7);
        let weak = Arc::downgrade(&bytes);
        let mut source = MemoryAccessManifest::default();
        source
            .ranges
            .push((17, MemoryUse::new(0x1000, 64, Access::ReadWrite).unwrap()));
        source.retained.push(RetainedStorageUse {
            owner: bytes,
            access: Access::ReadWrite,
        });
        let source = Arc::new(source);
        let combined = MemoryAccessManifest::combine(std::slice::from_ref(&source));
        drop(source);
        assert!(weak.upgrade().is_some());
        assert!(combined.covers(&[(17, MemoryUse::new(0x1010, 16, Access::Read).unwrap())]));
        assert!(!combined.covers(&[(18, MemoryUse::new(0x1010, 16, Access::Read).unwrap())]));
        assert!(!combined.covers(&[(17, MemoryUse::new(0x1030, 32, Access::Read).unwrap())]));
        drop(combined);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn shared_reclamation_preserves_leases_until_the_actual_owner_can_retire() {
        use crate::device_runtime::resource::{BlockUseRegistry, MemoryUse};
        use std::sync::atomic::AtomicUsize;

        struct PhysicalStorage {
            bytes: Box<[u8]>,
            drops: Arc<AtomicUsize>,
        }
        impl Drop for PhysicalStorage {
            fn drop(&mut self) {
                self.drops.fetch_add(1, Ordering::SeqCst);
            }
        }
        struct Payload {
            physical: Option<PhysicalStorage>,
            reclamation: Arc<AllocationReclamation>,
            admission: Option<crate::device_runtime::resource::MemoryUseGroup>,
        }

        let registry = Arc::new(std::sync::Mutex::new(BlockUseRegistry::default()));
        let accounting = Arc::new(GpuMemoryAccounting::default());
        accounting.budget_reserved.store(64, Ordering::SeqCst);
        let reclamation = Arc::new(AllocationReclamation::default());
        reclamation
            .attach_local(Arc::clone(&accounting), 64)
            .unwrap();
        reclamation.acquired().unwrap();
        let physical_drops = Arc::new(AtomicUsize::new(0));
        let physical = PhysicalStorage {
            bytes: vec![3u8; 64].into_boxed_slice(),
            drops: Arc::clone(&physical_drops),
        };
        let memory = MemoryUse::new(physical.bytes.as_ptr() as u64, 64, Access::ReadWrite).unwrap();
        let mut pending = Some(Arc::new(Payload {
            physical: Some(physical),
            reclamation: Arc::clone(&reclamation),
            admission: None,
        }));
        let lease = Arc::clone(pending.as_ref().unwrap());
        let observer = Arc::downgrade(&lease);

        // Logical detach leaves the actual bytes available to existing leases.
        // It must not reserve a physical free or invoke the driver boundary.
        assert!(!reclaim_shared_allocation(&mut pending, |_| {
            panic!("a live storage lease must prevent physical reclamation")
        })
        .unwrap());
        assert!(Arc::ptr_eq(pending.as_ref().unwrap(), &lease));
        let use_group = registry
            .lock()
            .unwrap()
            .reserve_owned_memory_uses(&[(7, memory)], &[reclamation.release_proof()])
            .unwrap();
        assert_eq!(lease.physical.as_ref().unwrap().bytes[9], 3);
        registry
            .lock()
            .unwrap()
            .release_memory_uses(use_group)
            .unwrap();
        drop(lease);

        // The sole real owner now enters physical reclamation. Failed and
        // unwinding driver waits retain both its allocation and its admission.
        let attempt = |owner: &mut Payload| {
            crate::device_runtime::resource::with_reclamation_admission(
                &registry,
                7,
                memory,
                owner.reclamation.release_proof(),
                &mut owner.admission,
                || Err(ResourceError::Driver("physical completion pending".into())),
            )
        };
        assert!(reclaim_shared_allocation(&mut pending, attempt).is_err());
        assert!(observer.upgrade().is_none());
        let queue = std::sync::Mutex::new(vec![pending.take()]);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut batch =
                PendingReclamationBatch::take(&queue, |queue, owners| queue.append(owners));
            let _ = reclaim_shared_allocation(&mut batch[0], |_| panic!("wait unwound"));
        }))
        .is_err());
        pending = queue
            .lock()
            .unwrap()
            .pop()
            .expect("failed owner returned to canonical queue");
        assert!(pending.as_ref().unwrap().physical.is_some());
        assert_eq!(physical_drops.load(Ordering::SeqCst), 0);
        assert_eq!(accounting.allocated.load(Ordering::SeqCst), 64);
        assert!(registry
            .lock()
            .unwrap()
            .reserve_memory_uses(7, &[memory])
            .is_err());

        let late_registry = Arc::clone(&registry);
        std::thread::spawn(move || {
            assert!(reclaim_shared_allocation(&mut pending, |owner| {
                crate::device_runtime::resource::with_reclamation_admission(
                    &late_registry,
                    7,
                    memory,
                    owner.reclamation.release_proof(),
                    &mut owner.admission,
                    || {
                        assert!(late_registry.try_lock().is_ok());
                        drop(owner.physical.take());
                        owner.reclamation.complete()
                    },
                )
            })
            .unwrap());
            assert!(pending.is_none());
        })
        .join()
        .unwrap();
        assert_eq!(physical_drops.load(Ordering::SeqCst), 1);
        assert!(reclamation.was_released());
        assert_eq!(accounting.allocated.load(Ordering::SeqCst), 0);
        assert_eq!(accounting.budget_reserved.load(Ordering::SeqCst), 0);
        let replacement = AllocationReclamation::default();
        let group = registry
            .lock()
            .unwrap()
            .reserve_owned_memory_uses(&[(7, memory)], &[replacement.release_proof()])
            .unwrap();
        registry.lock().unwrap().release_memory_uses(group).unwrap();
    }

    #[test]
    fn late_reclamation_retires_actual_payload_charge_and_same_range_together() {
        use crate::device_runtime::resource::{
            with_reclamation_admission, BlockUseRegistry, MemoryStorageOwner, MemoryUse,
        };
        use std::sync::atomic::AtomicUsize;

        struct PhysicalStorage {
            _bytes: Box<[u8]>,
            drops: Arc<AtomicUsize>,
        }
        impl Drop for PhysicalStorage {
            fn drop(&mut self) {
                self.drops.fetch_add(1, Ordering::SeqCst);
            }
        }
        struct Payload {
            physical: Option<PhysicalStorage>,
            reclamation: Arc<AllocationReclamation>,
            drops: Arc<AtomicUsize>,
            admission: Option<MemoryUseGroup>,
        }
        impl Drop for Payload {
            fn drop(&mut self) {
                self.drops.fetch_add(1, Ordering::SeqCst);
            }
        }
        struct StorageView;
        impl MemoryStorageOwner for StorageView {
            fn dependencies(
                &self,
            ) -> crate::device_runtime::ResourceResult<Arc<DeviceAccessDependencies>> {
                panic!("host driver boundary must not initialize CUDA");
            }
        }

        for enqueue_error in [false, true] {
            let registry = std::sync::Mutex::new(BlockUseRegistry::default());
            let accounting = Arc::new(GpuMemoryAccounting::default());
            accounting.budget_reserved.store(64, Ordering::SeqCst);
            let reclamation = Arc::new(AllocationReclamation::default());
            reclamation
                .attach_local(Arc::clone(&accounting), 64)
                .unwrap();
            reclamation.acquired().unwrap();
            let physical_drops = Arc::new(AtomicUsize::new(0));
            let payload_drops = Arc::new(AtomicUsize::new(0));
            let bytes = vec![0u8; 64].into_boxed_slice();
            let ptr = bytes.as_ptr() as u64;
            let mut live = Some(Payload {
                physical: Some(PhysicalStorage {
                    _bytes: bytes,
                    drops: Arc::clone(&physical_drops),
                }),
                reclamation: Arc::clone(&reclamation),
                drops: Arc::clone(&payload_drops),
                admission: None,
            });
            let view: Arc<dyn MemoryStorageOwner> = Arc::new(StorageView);
            let memory = MemoryUse::new(ptr, 64, Access::ReadWrite).unwrap();
            registry.lock().unwrap().register_storage(
                7,
                memory,
                Arc::downgrade(&view),
                reclamation.release_proof(),
            );
            let mut pending = live.take();
            let owner = pending.as_mut().unwrap();
            let result = with_reclamation_admission(
                &registry,
                7,
                memory,
                reclamation.release_proof(),
                &mut owner.admission,
                || {
                    assert!(registry.try_lock().is_ok());
                    if enqueue_error {
                        Err(ResourceError::Driver(
                            "pending free completion unknown".into(),
                        ))
                    } else {
                        Ok(())
                    }
                },
            );
            assert_eq!(result.is_err(), enqueue_error);
            drop(view);
            assert!(live.is_none());
            assert!(
                reclaim_allocation(&mut pending, |_| Err(ResourceError::Driver(
                    "driver completion still pending".into()
                )))
                .is_err()
            );
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = reclaim_allocation(&mut pending, |_| panic!("driver wait unwound"));
            }))
            .is_err());
            assert!(pending.as_ref().unwrap().physical.is_some());
            assert_eq!(physical_drops.load(Ordering::SeqCst), 0);
            assert_eq!(payload_drops.load(Ordering::SeqCst), 0);
            assert_eq!(accounting.allocated.load(Ordering::SeqCst), 64);
            assert_eq!(accounting.budget_reserved.load(Ordering::SeqCst), 64);
            assert!(!reclamation.was_released());
            {
                let mut registry = registry.lock().unwrap();
                assert!(registry.reserve_memory_uses(7, &[memory]).is_err());
                assert!(registry
                    .retain_storage_uses(7, &[memory], &mut Vec::new())
                    .is_err());
            }

            // The real raw-owner retirement helper runs on a different reaper.
            // Only the physical driver operation is replaced by host storage.
            std::thread::spawn(move || {
                reclaim_allocation(&mut pending, |owner| {
                    drop(owner.physical.take());
                    owner.reclamation.complete()
                })
                .unwrap();
                assert!(pending.is_none());
                reclaim_allocation(&mut pending, |_| panic!("must not free twice")).unwrap();
            })
            .join()
            .unwrap();
            assert_eq!(physical_drops.load(Ordering::SeqCst), 1);
            assert_eq!(payload_drops.load(Ordering::SeqCst), 1);
            assert!(reclamation.was_released());
            assert_eq!(accounting.allocated.load(Ordering::SeqCst), 0);
            assert_eq!(accounting.budget_reserved.load(Ordering::SeqCst), 0);
            assert_eq!(Arc::strong_count(&accounting), 1);
            let replacement: Arc<dyn MemoryStorageOwner> = Arc::new(StorageView);
            let replacement_reclamation = AllocationReclamation::default();
            let mut retained = Vec::new();
            {
                let mut registry = registry.lock().unwrap();
                registry.register_storage(
                    7,
                    memory,
                    Arc::downgrade(&replacement),
                    replacement_reclamation.release_proof(),
                );
                // Reusing the address never revalidates an old source owner.
                assert!(registry
                    .reserve_owned_memory_uses(&[(7, memory)], &[reclamation.release_proof()],)
                    .is_err());
                registry
                    .retain_storage_uses(7, &[memory], &mut retained)
                    .unwrap();
                let group = registry
                    .reserve_owned_memory_uses(
                        &[(7, memory)],
                        &[replacement_reclamation.release_proof()],
                    )
                    .unwrap();
                registry.release_memory_uses(group).unwrap();
            }
            assert_eq!(retained.len(), 1);
            assert!(Arc::ptr_eq(&retained[0].owner, &replacement));
        }
    }

    #[test]
    fn foreign_storage_retirement_retries_completion_and_releases_the_actual_owner() {
        use std::sync::atomic::{AtomicBool, AtomicUsize};

        struct Producer(Arc<AtomicUsize>);
        impl Drop for Producer {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let _capture_test = crate::cuda_graph::capture_lifecycle_test_guard();
        let drops = Arc::new(AtomicUsize::new(0));
        let producer = Arc::new(Producer(Arc::clone(&drops)));
        let retained = Arc::downgrade(&producer);
        let ready = Arc::new(AtomicBool::new(false));
        let completion = Arc::clone(&ready);
        let reclamation = Arc::new(AllocationReclamation::default());
        let proof = Arc::clone(&reclamation);
        crate::cuda_graph::retire_resources_after_completion(
            Some(producer),
            move |_| {
                if completion.load(Ordering::SeqCst) {
                    Ok(())
                } else {
                    Err(ResourceError::Driver(
                        "producer use completion pending".into(),
                    ))
                }
            },
            |owner| {
                drop(owner.take());
                Ok(())
            },
            move |_| proof.complete(),
            |_, error| panic!("unexpected producer retirement failure: {error}"),
        );
        assert!(retained.upgrade().is_some());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert!(!reclamation.was_released());
        ready.store(true, Ordering::SeqCst);
        crate::cuda_graph::reap_capture_retirements();
        assert!(retained.upgrade().is_none());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(reclamation.was_released());
        crate::cuda_graph::reap_capture_retirements();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn foreign_storage_retirement_retries_release_publication_without_repeating_deleter() {
        use std::sync::atomic::AtomicUsize;
        struct Producer(Arc<AtomicUsize>);
        impl Drop for Producer {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let _capture_test = crate::cuda_graph::capture_lifecycle_test_guard();
        let drops = Arc::new(AtomicUsize::new(0));
        let accounting = Arc::new(GpuMemoryAccounting::default());
        let reclamation = Arc::new(AllocationReclamation::default());
        reclamation
            .attach_local(Arc::clone(&accounting), 64)
            .unwrap();
        reclamation.acquired().unwrap();
        let resources = (Some(Producer(Arc::clone(&drops))), Arc::clone(&reclamation));
        crate::cuda_graph::retire_resources_after_completion(
            resources,
            |_| Ok(()),
            |owners| {
                drop(owners.0.take());
                Ok(())
            },
            |owners| owners.1.complete(),
            |_, _| {},
        );
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(!reclamation.was_released());
        accounting.allocated.store(64, Ordering::SeqCst);
        accounting.budget_reserved.store(64, Ordering::SeqCst);
        std::thread::spawn(crate::cuda_graph::reap_capture_retirements)
            .join()
            .unwrap();
        assert!(reclamation.was_released());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(accounting.allocated.load(Ordering::SeqCst), 0);
        assert_eq!(accounting.budget_reserved.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn foreign_storage_retirement_publication_unwind_keeps_retry_without_deleter() {
        use std::sync::atomic::{AtomicBool, AtomicUsize};
        let _capture_test = crate::cuda_graph::capture_lifecycle_test_guard();
        let destroyed = Arc::new(AtomicUsize::new(0));
        let destroy_probe = Arc::clone(&destroyed);
        let first = Arc::new(AtomicBool::new(true));
        let context = Arc::new(());
        let retained = Arc::downgrade(&context);
        let reclamation = Arc::new(AllocationReclamation::default());
        let proof = Arc::clone(&reclamation);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::cuda_graph::retire_resources_after_completion(
                context,
                |_| Ok(()),
                move |_| {
                    destroy_probe.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                },
                move |_| {
                    if first.swap(false, Ordering::SeqCst) {
                        panic!("release publication interrupted");
                    }
                    proof.complete()
                },
                |_, error| panic!("unexpected destructive failure: {error}"),
            );
        }));
        assert!(result.is_err());
        assert_eq!(destroyed.load(Ordering::SeqCst), 1);
        assert!(retained.upgrade().is_some());
        assert!(!reclamation.was_released());
        std::thread::spawn(crate::cuda_graph::reap_capture_retirements)
            .join()
            .unwrap();
        assert_eq!(destroyed.load(Ordering::SeqCst), 1);
        assert!(reclamation.was_released());
        assert!(retained.upgrade().is_none());
    }

    #[test]
    fn foreign_storage_retirement_unwinding_deleter_keeps_context_without_retry() {
        use std::sync::atomic::AtomicUsize;

        struct Producer(Arc<AtomicUsize>);
        impl Drop for Producer {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
                panic!("producer deleter interrupted");
            }
        }
        let _capture_test = crate::cuda_graph::capture_lifecycle_test_guard();
        let drops = Arc::new(AtomicUsize::new(0));
        let context = Arc::new(());
        let retained_context = Arc::downgrade(&context);
        let reclamation = Arc::new(AllocationReclamation::default());
        let resources = (
            Some(Producer(Arc::clone(&drops))),
            Arc::clone(&reclamation),
            context,
        );
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::cuda_graph::retire_resources_after_completion(
                resources,
                |_| Ok(()),
                |owners| {
                    drop(owners.0.take());
                    Ok(())
                },
                |owners| owners.1.complete(),
                |_, error| panic!("unexpected retirement result: {error}"),
            );
        }));
        assert!(result.is_err());
        crate::cuda_graph::reap_capture_retirements();
        crate::cuda_graph::reap_capture_retirements();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(retained_context.upgrade().is_some());
        assert!(!reclamation.was_released());
    }

    #[test]
    fn allocation_reclamation_refunds_local_charge_once_on_the_actual_reaper() {
        let accounting = Arc::new(GpuMemoryAccounting::default());
        accounting.budget_reserved.store(64, Ordering::SeqCst);
        let reclamation = Arc::new(AllocationReclamation::default());
        reclamation
            .attach_local(Arc::clone(&accounting), 64)
            .unwrap();
        reclamation.acquired().unwrap();
        let reaper = Arc::clone(&reclamation);
        std::thread::spawn(move || reaper.complete().unwrap())
            .join()
            .unwrap();
        reclamation.complete().unwrap();
        assert!(reclamation.was_released());
        assert_eq!(accounting.allocated.load(Ordering::SeqCst), 0);
        assert_eq!(accounting.budget_reserved.load(Ordering::SeqCst), 0);
        assert_eq!(Arc::strong_count(&accounting), 1);
    }

    #[test]
    fn allocation_reclamation_settles_all_byte_charges_before_handle_retirement() {
        let local = Arc::new(GpuMemoryAccounting::default());
        local.budget_reserved.store(64, Ordering::SeqCst);
        let ledger = Arc::new(AllocationAccounting::default());
        let pending = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let ticket = Arc::new(AllocationReclamation::default());
        ticket.attach_local(Arc::clone(&local), 64).unwrap();
        ticket.attach_resource(Arc::clone(&ledger), 64).unwrap();
        let storage = vec![0_u8; 64].into_boxed_slice();
        ticket.acquired().unwrap();
        ticket.attach_pending(Arc::clone(&pending), 64).unwrap();
        assert!(ticket.attach_pending(Arc::clone(&pending), 64).is_err());
        assert_eq!(pending.load(Ordering::SeqCst), 64);
        let mut handle_owner = Some((storage, Arc::clone(&ticket)));
        assert!(reclaim_allocation(&mut handle_owner, |owner| {
            // The memory allocation, not the surrounding handle owner, is freed.
            drop(std::mem::take(&mut owner.0));
            owner.1.complete()?;
            Err(ResourceError::Driver(
                "stream destruction outcome unknown".into(),
            ))
        })
        .is_err());
        assert!(handle_owner.is_some());
        assert!(ticket.was_acquired());
        assert!(ticket.was_released());
        assert_eq!(local.budget_reserved.load(Ordering::SeqCst), 0);
        assert_eq!(local.allocated.load(Ordering::SeqCst), 0);
        assert_eq!(ledger.snapshot(), (0, 64));
        assert_eq!(pending.load(Ordering::SeqCst), 0);
        ticket.complete().unwrap();
        assert_eq!(ledger.snapshot(), (0, 64));
    }

    #[test]
    fn allocation_reclamation_concurrent_completion_settles_once() {
        let ledger = Arc::new(AllocationAccounting::default());
        let ticket = Arc::new(AllocationReclamation::default());
        ticket.attach_resource(Arc::clone(&ledger), 64).unwrap();
        let storage = vec![0_u8; 64].into_boxed_slice();
        ticket.acquired().unwrap();
        drop(storage);
        let barrier = Arc::new(std::sync::Barrier::new(8));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let ticket = Arc::clone(&ticket);
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    barrier.wait();
                    for _ in 0..100 {
                        ticket.complete().unwrap();
                    }
                });
            }
        });
        assert_eq!(ledger.snapshot(), (0, 64));
    }

    #[test]
    fn allocation_reclamation_detach_panic_does_not_skip_physical_reap() {
        use crate::device_runtime::{DeviceMemoryResource, ResourceResult};
        struct InterruptedDetach(
            crate::device_runtime::budget::tests::DeferredHostResource,
            bool,
        );
        impl DeviceMemoryResource for InterruptedDetach {
            fn materialize(&self, request: AllocationRequest) -> ResourceResult<DeviceBlock> {
                self.0.materialize(request)
            }
            fn allocation_accounting(&self) -> Arc<AllocationAccounting> {
                self.0.allocation_accounting()
            }
            fn device_ordinal(&self) -> u32 {
                0
            }
            fn bytes_outstanding(&self) -> usize {
                self.0.bytes_outstanding()
            }
            fn deallocate(&self, block: DeviceBlock) -> ResourceResult<()> {
                self.0.deallocate(block)?;
                panic!("detach telemetry interrupted");
            }
            fn reap_pending(&self) -> ResourceResult<()> {
                let owners = std::mem::take(&mut *self.0.owners.lock().unwrap());
                drop(owners);
                if self.1 {
                    panic!("reap telemetry interrupted");
                }
                Ok(())
            }
        }
        for reap_panics in [false, true] {
            let resource: Arc<dyn DeviceMemoryResource + Send + Sync> =
                Arc::new(InterruptedDetach(Default::default(), reap_panics));
            let ledger = resource.allocation_accounting();
            let request = AllocationRequest::new(64, StreamId::DEFAULT, AllocTag::UNTAGGED);
            let ticket = request.reclamation();
            let block = resource.materialize(request).unwrap();
            let mut owner = ResourceBlockRetirement::new(block, resource, ticket);
            let outcome =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| owner.release()));
            let panic = outcome.expect_err("detach panic must be preserved");
            assert_eq!(
                panic.downcast_ref::<&str>(),
                Some(&"detach telemetry interrupted")
            );
            assert_eq!(ledger.snapshot(), (0, 64));
            owner.release().unwrap();
        }
    }

    #[test]
    fn allocation_reclamation_partial_settlement_does_not_refund_twice() {
        let local = Arc::new(GpuMemoryAccounting::default());
        local.budget_reserved.store(64, Ordering::SeqCst);
        let ledger = Arc::new(AllocationAccounting::default());
        let pending = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let ticket = AllocationReclamation::default();
        ticket.attach_local(Arc::clone(&local), 64).unwrap();
        ticket.attach_resource(Arc::clone(&ledger), 64).unwrap();
        ticket.acquired().unwrap();
        ticket.attach_pending(Arc::clone(&pending), 64).unwrap();
        pending.store(32, Ordering::SeqCst); // inject a failed final obligation
        assert!(ticket.complete().is_err());
        assert_eq!(local.allocated.load(Ordering::SeqCst), 0);
        assert_eq!(ledger.snapshot(), (0, 64));
        assert!(!ticket.was_released());
        pending.store(64, Ordering::SeqCst);
        ticket.complete().unwrap();
        assert_eq!(ledger.snapshot(), (0, 64));
        assert_eq!(pending.load(Ordering::SeqCst), 0);
        assert!(ticket.was_released());
    }

    #[test]
    fn allocation_reclamation_local_attempt_restores_only_unacquired_claims() {
        for from_token in [false, true] {
            for acquired in [false, true] {
                let accounting = Arc::new(GpuMemoryAccounting::default());
                accounting.budget_reserved.store(128, Ordering::SeqCst);
                let mut remaining = 64;
                let mut ticket = None;
                let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let attempt = LocalAllocationAttempt::new(
                        Arc::clone(&accounting),
                        64,
                        if from_token {
                            Some(&mut remaining)
                        } else {
                            None
                        },
                    );
                    ticket = Some(Arc::clone(&attempt.reclamation));
                    if acquired {
                        attempt.reclamation.acquired().unwrap();
                        // A concurrent physical release before the original frame
                        // unwinds must not make the frame treat malloc as refused.
                        attempt.reclamation.complete().unwrap();
                    }
                    panic!("allocation frame interrupted");
                }));
                assert!(panic.is_err());
                assert_eq!(remaining, if from_token && !acquired { 128 } else { 64 });
                assert_eq!(
                    accounting.budget_reserved.load(Ordering::SeqCst),
                    if from_token && !acquired { 128 } else { 64 }
                );
                assert_eq!(accounting.allocated.load(Ordering::SeqCst), 0);
            }
        }
    }

    #[test]
    fn allocation_reclamation_unknown_completion_retains_canonical_charge() {
        let accounting = Arc::new(GpuMemoryAccounting::default());
        accounting.allocated.store(64, Ordering::SeqCst);
        accounting.budget_reserved.store(64, Ordering::SeqCst);
        let reclamation = AllocationReclamation::default();
        accounting.allocated.store(0, Ordering::SeqCst);
        reclamation
            .attach_local(Arc::clone(&accounting), 64)
            .unwrap();
        reclamation.acquired().unwrap();
        assert!(!reclamation.was_released());
        drop(reclamation);
        assert_eq!(accounting.allocated.load(Ordering::SeqCst), 64);
        assert_eq!(accounting.budget_reserved.load(Ordering::SeqCst), 64);
    }

    #[test]
    fn allocation_reclamation_failed_refund_is_atomic_and_retryable() {
        let accounting = Arc::new(GpuMemoryAccounting::default());
        accounting.budget_reserved.store(64, Ordering::SeqCst);
        let reclamation = AllocationReclamation::default();
        reclamation
            .attach_local(Arc::clone(&accounting), 64)
            .unwrap();
        reclamation.acquired().unwrap();
        accounting.allocated.store(32, Ordering::SeqCst);
        assert!(reclamation.complete().is_err());
        assert!(!reclamation.was_released());
        assert_eq!(accounting.allocated.load(Ordering::SeqCst), 32);
        assert_eq!(accounting.budget_reserved.load(Ordering::SeqCst), 64);
        assert_eq!(Arc::strong_count(&accounting), 2);
        accounting.allocated.store(64, Ordering::SeqCst);
        reclamation.complete().unwrap();
        assert!(reclamation.was_released());
        assert_eq!(accounting.allocated.load(Ordering::SeqCst), 0);
        assert_eq!(accounting.budget_reserved.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn physical_release_proof_is_independent_of_another_pending_failure() {
        assert!(allocation_reclamation_outcome(
            true,
            Err(ResourceError::Driver("another allocation failed".into()))
        )
        .is_ok());
        assert!(allocation_reclamation_outcome(false, Ok(())).is_err());
        assert!(allocation_reclamation_outcome(
            false,
            Err(ResourceError::Driver("own free failed".into()))
        )
        .is_err());
    }
    #[test]
    fn allocation_initialization_failure_reaches_the_canonical_cold_reaper() {
        use std::sync::atomic::{AtomicBool, AtomicUsize};

        struct DeferredOwner {
            storage: Option<Box<[u8]>>,
            completed: Arc<AtomicBool>,
            releases: Arc<AtomicUsize>,
        }
        impl Drop for DeferredOwner {
            fn drop(&mut self) {
                let mut storage = self.storage.take();
                let completed = Arc::clone(&self.completed);
                let releases = Arc::clone(&self.releases);
                crate::cuda_graph::retry_retirement_after_stream_captures(move || {
                    if !completed.load(Ordering::Acquire) {
                        return false;
                    }
                    drop(storage.take());
                    releases.fetch_add(1, Ordering::SeqCst);
                    true
                });
            }
        }

        let _capture_test = crate::cuda_graph::capture_lifecycle_test_guard();
        for unwind in [false, true] {
            let completed = Arc::new(AtomicBool::new(false));
            let releases = Arc::new(AtomicUsize::new(0));
            let owner = DeferredOwner {
                storage: Some(vec![0; 64].into_boxed_slice()),
                completed: Arc::clone(&completed),
                releases: Arc::clone(&releases),
            };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let result = initialize_allocation(owner, 64, Arc::default(), |_| {
                    if unwind {
                        panic!("allocation initialization interrupted");
                    }
                    Err(ResourceError::Driver(
                        "allocation initialization failed".into(),
                    ))
                });
                assert!(result.is_err());
            }));
            assert_eq!(result.is_err(), unwind);
            crate::cuda_graph::reap_capture_retirements();
            assert_eq!(releases.load(Ordering::SeqCst), 0);
            completed.store(true, Ordering::Release);
            std::thread::spawn(crate::cuda_graph::reap_capture_retirements)
                .join()
                .unwrap();
            assert_eq!(releases.load(Ordering::SeqCst), 1);
            crate::cuda_graph::reap_capture_retirements();
            assert_eq!(releases.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn allocation_initialization_failure_transfers_actual_owner_to_drop() {
        struct Owner(Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for Owner {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let reclamation = Arc::new(AllocationReclamation::default());
        reclamation.acquired().unwrap();
        let error = initialize_allocation(Owner(Arc::clone(&drops)), 64, reclamation, |_| {
            Err(ResourceError::Driver(
                "allocation-ready record failed".into(),
            ))
        })
        .err()
        .expect("initialization must fail");
        assert_eq!(error.retained_allocation_bytes(), 64);
        drop(error);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn allocation_initialization_unwind_transfers_actual_owner_to_drop() {
        struct Owner(Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for Owner {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let result = std::panic::catch_unwind(|| {
            let _ = initialize_allocation(Owner(Arc::clone(&drops)), 64, Arc::default(), |_| {
                panic!("initialization unwound")
            });
        });
        assert!(result.is_err());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn allocation_initialization_success_transfers_owner_once() {
        let owner = Arc::new(());
        let initialized =
            initialize_allocation(Arc::clone(&owner), 64, Arc::default(), |_| Ok(())).unwrap();
        assert_eq!(Arc::strong_count(&owner), 2);
        drop(initialized);
        assert_eq!(Arc::strong_count(&owner), 1);
    }

    #[test]
    fn allocation_release_retries_only_unconfirmed_fence_steps() {
        use std::cell::Cell;

        let mut state = DriverReleaseState::default();
        let frees = Cell::new(0);
        state
            .submit(|| {
                frees.set(frees.get() + 1);
                Ok(())
            })
            .unwrap();
        let records = Cell::new(0);
        let waits = Cell::new(0);
        let mut recorded = false;
        assert!(state
            .confirm_async(
                &mut recorded,
                || {
                    records.set(records.get() + 1);
                    Err(ResourceError::Driver("record failed".into()))
                },
                || {
                    waits.set(waits.get() + 1);
                    Ok(())
                },
            )
            .is_err());
        assert!(!recorded);
        assert_eq!(waits.get(), 0, "an unrecorded event proves no free");
        assert!(state
            .confirm_async(
                &mut recorded,
                || {
                    records.set(records.get() + 1);
                    Ok(())
                },
                || {
                    waits.set(waits.get() + 1);
                    Err(ResourceError::Driver("wait failed".into()))
                },
            )
            .is_err());
        assert!(recorded);
        state
            .confirm_async(
                &mut recorded,
                || panic!("recorded fence must not be replaced"),
                || {
                    waits.set(waits.get() + 1);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(frees.get(), 1);
        assert_eq!(records.get(), 2);
        assert_eq!(waits.get(), 2);

        for state in [
            DriverReleaseState::Owned,
            DriverReleaseState::OutcomeUnknown,
        ] {
            assert!(state
                .confirm_async(
                    &mut false,
                    || panic!("cannot invent proof of an unconfirmed free submission"),
                    || panic!("cannot confirm an unsubmitted free"),
                )
                .is_err());
        }
    }

    #[test]
    fn allocation_release_does_not_resubmit_unknown_free() {
        let attempts = std::cell::Cell::new(0);
        let mut state = DriverReleaseState::default();
        assert!(state
            .submit(|| {
                attempts.set(attempts.get() + 1);
                Err(ResourceError::Driver("unknown free outcome".into()))
            })
            .is_err());
        assert!(state
            .submit(|| {
                attempts.set(attempts.get() + 1);
                Ok(())
            })
            .is_err());
        assert_eq!(attempts.get(), 1);
    }

    #[test]
    fn allocation_release_does_not_resubmit_successful_free() {
        let attempts = std::cell::Cell::new(0);
        let mut state = DriverReleaseState::default();
        for _ in 0..2 {
            state
                .submit(|| {
                    attempts.set(attempts.get() + 1);
                    Ok(())
                })
                .unwrap();
        }
        assert_eq!(attempts.get(), 1);
    }
    use super::*;
    use crate::device_runtime::{DeviceMemoryResource, DirectCudaResource, ResourceResult};
    use xlog_core::ScalarType;

    #[test]
    fn reservation_exposes_exact_memory_manager_identity() {
        let _: fn(&GpuMemoryReservation) -> usize = GpuMemoryReservation::memory_manager_ptr_value;
    }

    #[test]
    fn cuda_column_exposes_optional_memory_manager_identity() {
        let _: fn(&CudaColumn) -> Option<usize> = CudaColumn::memory_manager_ptr_value;
    }

    #[test]
    fn tracked_slice_exposes_runtime_allocation_identity_snapshot() {
        let _: fn(&TrackedCudaSlice<u32>) -> Result<Option<RuntimeAllocationIdentity>> =
            TrackedCudaSlice::<u32>::runtime_allocation_identity;
    }

    #[test]
    fn cuda_column_exposes_runtime_allocation_identity_snapshot() {
        let _: fn(&CudaColumn) -> Result<Option<RuntimeAllocationIdentity>> =
            CudaColumn::runtime_allocation_identity;
    }

    struct FailAfterDeallocateResource {
        inner: DirectCudaResource,
        deallocate_calls: Arc<AtomicU64>,
    }

    struct FailFirstAllocationResource {
        inner: DirectCudaResource,
        fail_next: std::sync::atomic::AtomicBool,
    }

    impl DeviceMemoryResource for FailFirstAllocationResource {
        fn materialize(&self, request: AllocationRequest) -> ResourceResult<DeviceBlock> {
            if self.fail_next.swap(false, Ordering::SeqCst) {
                return Err(ResourceError::Driver(
                    "injected allocation failure".to_string(),
                ));
            }
            self.inner.materialize(request)
        }

        fn allocation_accounting(&self) -> Arc<AllocationAccounting> {
            self.inner.allocation_accounting()
        }

        fn access_dependencies(
            &self,
            block: BlockId,
            bytes: usize,
        ) -> ResourceResult<Option<Arc<DeviceAccessDependencies>>> {
            self.inner.access_dependencies(block, bytes)
        }

        fn deallocate(&self, block: DeviceBlock) -> ResourceResult<()> {
            self.inner.deallocate(block)
        }

        fn device_ordinal(&self) -> u32 {
            self.inner.device_ordinal()
        }

        fn bytes_outstanding(&self) -> usize {
            self.inner.bytes_outstanding()
        }
    }

    impl DeviceMemoryResource for FailAfterDeallocateResource {
        fn materialize(&self, request: AllocationRequest) -> ResourceResult<DeviceBlock> {
            self.inner.materialize(request)
        }

        fn allocation_accounting(&self) -> Arc<AllocationAccounting> {
            self.inner.allocation_accounting()
        }

        fn access_dependencies(
            &self,
            block: BlockId,
            bytes: usize,
        ) -> ResourceResult<Option<Arc<DeviceAccessDependencies>>> {
            self.inner.access_dependencies(block, bytes)
        }

        fn deallocate(&self, block: DeviceBlock) -> ResourceResult<()> {
            self.inner.deallocate(block)?;
            self.deallocate_calls.fetch_add(1, Ordering::SeqCst);
            Err(ResourceError::Driver(
                "injected deallocation completion failure".to_string(),
            ))
        }

        fn device_ordinal(&self) -> u32 {
            self.inner.device_ordinal()
        }

        fn bytes_outstanding(&self) -> usize {
            self.inner.bytes_outstanding()
        }
    }

    fn try_device() -> Option<Arc<CudaDevice>> {
        match CudaDevice::new(0) {
            Ok(d) => Some(Arc::new(d)),
            Err(e) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
                panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {e}")
            }
            Err(e) => {
                eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
                None
            }
        }
    }

    fn assert_memory_pressure(
        error: XlogError,
        expected_context: &str,
        expected_required: u64,
        expected_budget: u64,
    ) {
        match error {
            XlogError::ResourceExhausted {
                context,
                estimated_bytes,
                budget_bytes,
            } => {
                assert_eq!(context, expected_context);
                assert_eq!(estimated_bytes, expected_required);
                assert_eq!(budget_bytes, expected_budget);
            }
            other => panic!("expected ResourceExhausted, got {other:?}"),
        }
    }

    // Test CudaBuffer without requiring a GPU
    #[test]
    fn test_cuda_buffer_empty() {
        let Some(device) = try_device() else {
            return;
        };
        let budget = MemoryBudget::with_limit(1024 * 1024);
        let manager = Arc::new(GpuMemoryManager::new(device, budget));
        let mut d_num_rows = manager.alloc::<u32>(1).unwrap();
        manager
            .device()
            .inner()
            .htod_sync_copy_into(&[0u32], &mut d_num_rows)
            .unwrap();
        let buffer = CudaBuffer::from_columns(Vec::new(), 0, d_num_rows, Schema::new(vec![]));
        assert!(buffer.is_empty());
        assert_eq!(buffer.num_rows(), 0);
        assert_eq!(buffer.arity(), 0);
        assert_eq!(buffer.estimated_bytes(), 0);
    }

    #[test]
    fn test_cuda_buffer_schema() {
        let schema = Schema::new(vec![
            ("a".to_string(), ScalarType::U32),
            ("b".to_string(), ScalarType::U64),
        ]);

        let Some(device) = try_device() else {
            return;
        };
        let budget = MemoryBudget::with_limit(1024 * 1024);
        let manager = Arc::new(GpuMemoryManager::new(device, budget));
        let mut d_num_rows = manager.alloc::<u32>(1).unwrap();
        manager
            .device()
            .inner()
            .htod_sync_copy_into(&[100u32], &mut d_num_rows)
            .unwrap();

        // Allocate dummy columns matching the schema arity (100 rows each)
        let col_a = CudaColumn::owned(manager.alloc::<u8>(100 * 4).unwrap()); // U32: 4 bytes
        let col_b = CudaColumn::owned(manager.alloc::<u8>(100 * 8).unwrap()); // U64: 8 bytes
        let buffer = CudaBuffer::from_columns(vec![col_a, col_b], 100, d_num_rows, schema.clone());

        assert_eq!(buffer.num_rows(), 100);
        assert_eq!(buffer.arity(), 2);
        // 4 bytes (U32) + 8 bytes (U64) = 12 bytes per row * 100 rows
        assert_eq!(buffer.estimated_bytes(), 1200);
        assert_eq!(buffer.schema(), &schema);
    }

    // Tests requiring GPU
    #[test]
    fn test_memory_manager_creation() {
        let Some(device) = try_device() else {
            return;
        };
        let budget = MemoryBudget::with_limit(1024 * 1024); // 1 MB
        let manager = Arc::new(GpuMemoryManager::new(device, budget));

        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(manager.budget().device_bytes, 1024 * 1024);
        assert_eq!(manager.remaining_bytes(), 1024 * 1024);
    }

    #[test]
    fn test_memory_manager_alloc() {
        let Some(device) = try_device() else {
            return;
        };
        let budget = MemoryBudget::with_limit(1024 * 1024); // 1 MB
        let manager = Arc::new(GpuMemoryManager::new(device, budget));

        // Allocate 256 u32 values = 1024 bytes
        let _slice = manager
            .alloc::<u32>(256)
            .expect("Allocation should succeed");

        assert_eq!(manager.allocated_bytes(), 1024);
        assert_eq!(manager.remaining_bytes(), 1024 * 1024 - 1024);
    }

    #[test]
    fn test_memory_manager_budget_exceeded() {
        let Some(device) = try_device() else {
            return;
        };
        let budget = MemoryBudget::with_limit(1024); // 1 KB limit
        let manager = Arc::new(GpuMemoryManager::new(device, budget));

        // Try to allocate 512 u32 values = 2048 bytes (exceeds 1KB budget)
        let result = manager.alloc::<u32>(512);

        assert!(result.is_err());
        if let Err(XlogError::ResourceExhausted {
            estimated_bytes,
            budget_bytes,
            ..
        }) = result
        {
            assert_eq!(estimated_bytes, 2048);
            assert_eq!(budget_bytes, 1024);
        } else {
            panic!("Expected ResourceExhausted error");
        }
    }

    #[test]
    fn test_memory_manager_check_budget() {
        let Some(device) = try_device() else {
            return;
        };
        let budget = MemoryBudget::with_limit(1000);
        let manager = Arc::new(GpuMemoryManager::new(device, budget));

        // Check that 500 bytes is within budget
        assert!(manager.check_budget(500).is_ok());

        // Check that 1001 bytes exceeds budget
        assert!(manager.check_budget(1001).is_err());
    }

    #[test]
    fn check_budget_does_not_reserve_a_multi_allocation_request() {
        let Some(device) = try_device() else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::new(
            device,
            MemoryBudget::with_limit(8192),
        ));
        let checked = Arc::new(std::sync::Barrier::new(2));
        let competitor_allocated = Arc::new(std::sync::Barrier::new(2));
        let release_competitor = Arc::new(std::sync::Barrier::new(2));

        let checked_in_thread = Arc::clone(&checked);
        let allocated_in_thread = Arc::clone(&competitor_allocated);
        let release_in_thread = Arc::clone(&release_competitor);
        let competitor_manager = Arc::clone(&manager);
        let competitor = std::thread::spawn(move || {
            checked_in_thread.wait();
            let allocation = competitor_manager
                .alloc::<u8>(4096)
                .expect("competing allocation must fit after the check-only query");
            allocated_in_thread.wait();
            release_in_thread.wait();
            drop(allocation);
        });

        manager
            .check_budget(8192)
            .expect("the complete request fits before the competing allocation");
        checked.wait();
        competitor_allocated.wait();
        let first = manager
            .alloc::<u8>(4096)
            .expect("the first materialized allocation must fit");
        let second = manager.alloc::<u8>(4096);
        assert!(
            matches!(second, Err(XlogError::ResourceExhausted { .. })),
            "a check-only query cannot protect later allocations from a competitor"
        );

        release_competitor.wait();
        competitor.join().expect("competitor thread panicked");
        drop(first);
        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(manager.remaining_bytes(), 8192);
    }

    #[test]
    fn reservation_rejects_the_complete_request_before_any_allocation() {
        let Some(device) = try_device() else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::new(
            device,
            MemoryBudget::with_limit(4095),
        ));
        manager.reset_alloc_count();

        let error = manager
            .reserve_bytes(4096)
            .expect_err("a request one byte above the budget must be rejected atomically");

        assert_memory_pressure(
            error,
            "GPU memory pressure: layer=manager_reserve current_bytes=0 requested_bytes=4096 required_bytes=4096 required_u64_overflow=false budget_bytes=4095 prior_peak_bytes=0",
            4096,
            4095,
        );
        assert_eq!(manager.alloc_count(), 0);
        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(manager.peak_bytes(), 0);
        assert_eq!(manager.remaining_bytes(), 4095);
    }

    #[test]
    fn competing_complete_reservations_cannot_both_claim_the_budget() {
        let Some(device) = try_device() else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::new(
            device,
            MemoryBudget::with_limit(4096),
        ));
        let attempted = Arc::new(std::sync::Barrier::new(2));

        let reserve = |manager: Arc<GpuMemoryManager>, attempted: Arc<std::sync::Barrier>| {
            std::thread::spawn(move || {
                let reservation = manager.reserve_bytes(4096);
                attempted.wait();
                let admitted = reservation.is_ok();
                drop(reservation);
                admitted
            })
        };
        let left = reserve(Arc::clone(&manager), Arc::clone(&attempted));
        let right = reserve(Arc::clone(&manager), Arc::clone(&attempted));

        let admitted = [
            left.join().expect("left reservation thread panicked"),
            right.join().expect("right reservation thread panicked"),
        ];
        assert_eq!(admitted.into_iter().filter(|value| *value).count(), 1);
        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(manager.remaining_bytes(), 4096);
    }

    #[test]
    fn competing_managers_share_one_runtime_reservation_budget() {
        let Some((device, runtime)) = try_runtime_with_budget(4096) else {
            return;
        };
        let left_manager = Arc::new(GpuMemoryManager::with_runtime(
            Arc::clone(&device),
            MemoryBudget::with_limit(4096),
            Arc::clone(&runtime),
        ));
        let right_manager = Arc::new(GpuMemoryManager::with_runtime(
            device,
            MemoryBudget::with_limit(4096),
            runtime,
        ));
        let attempted = Arc::new(std::sync::Barrier::new(2));

        let reserve = |manager: Arc<GpuMemoryManager>, attempted: Arc<std::sync::Barrier>| {
            std::thread::spawn(move || {
                let reservation = manager.reserve_bytes(4096);
                attempted.wait();
                let admitted = reservation.is_ok();
                drop(reservation);
                admitted
            })
        };
        let left = reserve(left_manager, Arc::clone(&attempted));
        let right = reserve(right_manager, attempted);

        let admitted = [
            left.join().expect("left reservation thread panicked"),
            right.join().expect("right reservation thread panicked"),
        ];
        assert_eq!(admitted.into_iter().filter(|value| *value).count(), 1);
    }

    #[test]
    fn runtime_reservation_blocks_competing_ordinary_allocation_until_materialized() {
        let Some((device, runtime)) = try_runtime_with_budget(4096) else {
            return;
        };
        let reserving_manager = Arc::new(GpuMemoryManager::with_runtime(
            Arc::clone(&device),
            MemoryBudget::with_limit(4096),
            Arc::clone(&runtime),
        ));
        let competing_manager = Arc::new(GpuMemoryManager::with_runtime(
            device,
            MemoryBudget::with_limit(4096),
            runtime,
        ));
        let mut reservation = reserving_manager
            .reserve_bytes(4096)
            .expect("complete runtime reservation");

        assert!(matches!(
            competing_manager.alloc::<u8>(1),
            Err(XlogError::ResourceExhausted { .. })
        ));
        assert_eq!(competing_manager.allocated_bytes(), 0);
        assert_eq!(competing_manager.remaining_bytes(), 4096);

        let allocation = reservation
            .alloc::<u8>(4096)
            .expect("reserved bytes cannot be stolen by an ordinary allocator");
        assert_eq!(reserving_manager.allocated_bytes(), 4096);
        drop(reservation);
        drop(allocation);
        assert_eq!(reserving_manager.allocated_bytes(), 0);
    }

    #[test]
    fn global_budget_rejects_complete_manifest_before_first_inner_allocation() {
        let Some((device, runtime, sink)) = try_runtime_with_logging_budget(4095) else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::with_runtime(
            device,
            MemoryBudget::with_limit(8192),
            runtime,
        ));

        let error = manager
            .reserve_bytes(4096)
            .expect_err("the complete request is one byte above the runtime budget");
        assert_memory_pressure(
            error,
            "GPU memory pressure: layer=device_runtime current_bytes=0 requested_bytes=4096 required_bytes=4096 required_u64_overflow=false budget_bytes=4095 prior_peak_bytes=0",
            4096,
            4095,
        );
        assert!(sink.snapshot().is_empty());
        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(manager.remaining_bytes(), 8192);
    }

    #[test]
    fn runtime_backed_reservation_requires_a_reservable_global_budget() {
        let Some((device, runtime)) = try_unbudgeted_runtime() else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::with_runtime(
            device,
            MemoryBudget::with_limit(4096),
            runtime,
        ));

        let error = manager
            .reserve_bytes(1024)
            .expect_err("an unbudgeted runtime cannot promise complete admission");
        assert!(
            matches!(error, XlogError::Kernel(ref detail) if detail.contains("reservable global budget")),
            "unexpected error: {error}"
        );
        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(manager.remaining_bytes(), 4096);
    }

    #[test]
    fn reservation_materializes_exactly_its_declared_bytes() {
        let Some(device) = try_device() else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::new(
            device,
            MemoryBudget::with_limit(4096),
        ));
        let mut reservation = manager
            .reserve_bytes(4096)
            .expect("the exact complete request must fit");
        assert_eq!(reservation.total_bytes(), 4096);
        assert_eq!(reservation.remaining_bytes(), 4096);
        assert_eq!(reservation.used_bytes(), 0);
        assert_eq!(manager.remaining_bytes(), 0);

        let words = reservation
            .alloc::<u32>(512)
            .expect("first reserved allocation");
        let bytes = reservation
            .alloc::<u8>(2048)
            .expect("second reserved allocation");
        assert_eq!(reservation.remaining_bytes(), 0);
        assert_eq!(reservation.used_bytes(), 4096);
        assert_eq!(manager.allocated_bytes(), 4096);
        assert_eq!(manager.peak_bytes(), 4096);

        let error = match reservation.alloc::<u8>(1) {
            Err(error) => error,
            Ok(_) => panic!("a reservation cannot materialize more than its declaration"),
        };
        assert_memory_pressure(
            error,
            "GPU memory pressure: layer=manager_reservation_alloc current_bytes=4096 requested_bytes=1 required_bytes=4097 required_u64_overflow=false budget_bytes=4096 prior_peak_bytes=4096",
            4097,
            4096,
        );

        drop(reservation);
        assert_eq!(manager.remaining_bytes(), 0);
        drop(bytes);
        assert_eq!(manager.remaining_bytes(), 2048);
        drop(words);
        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(manager.remaining_bytes(), 4096);
    }

    #[test]
    fn failed_reserved_raw_allocation_preserves_unused_claim_until_drop() {
        let Some((device, runtime)) = try_runtime_with_first_allocation_failure(4096) else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::with_runtime(
            device,
            MemoryBudget::with_limit(4096),
            runtime,
        ));
        let mut reservation = manager
            .reserve_bytes(4096)
            .expect("local complete request must fit");

        let error = reservation
            .alloc_raw(2048, AllocTag::UNTAGGED)
            .expect_err("the injected underlying allocation must fail");
        assert!(
            matches!(error, XlogError::Kernel(ref detail) if detail.contains("injected allocation failure")),
            "unexpected error: {error}"
        );
        assert_eq!(reservation.remaining_bytes(), 4096);
        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(manager.remaining_bytes(), 0);

        let allocation = reservation
            .alloc_raw(1024, AllocTag::UNTAGGED)
            .expect("a smaller raw suballocation must fit both budgets");
        assert_eq!(reservation.remaining_bytes(), 3072);
        assert_eq!(manager.allocated_bytes(), 1024);

        drop(reservation);
        assert_eq!(manager.remaining_bytes(), 3072);
        drop(allocation);
        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(manager.remaining_bytes(), 4096);
    }

    #[test]
    fn partial_typed_materialization_releases_each_byte_once_in_either_drop_order() {
        let Some(device) = try_device() else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::new(
            device,
            MemoryBudget::with_limit(4096),
        ));
        let mut reservation = manager.reserve_bytes(4096).expect("complete request");
        let allocation = reservation
            .alloc::<u8>(1024)
            .expect("partial materialization");
        let error = match reservation.alloc::<u8>(4096) {
            Err(error) => error,
            Ok(_) => panic!("the remaining reservation is only 3072 bytes"),
        };
        assert_memory_pressure(
            error,
            "GPU memory pressure: layer=manager_reservation_alloc current_bytes=1024 requested_bytes=4096 required_bytes=5120 required_u64_overflow=false budget_bytes=4096 prior_peak_bytes=1024",
            5120,
            4096,
        );
        assert_eq!(reservation.remaining_bytes(), 3072);

        drop(allocation);
        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(manager.remaining_bytes(), 1024);
        drop(reservation);
        assert_eq!(manager.remaining_bytes(), 4096);
    }

    #[test]
    fn ordinary_allocations_and_complete_reservations_share_one_budget() {
        let Some(device) = try_device() else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::new(
            device,
            MemoryBudget::with_limit(4096),
        ));
        let ordinary = manager
            .alloc::<u8>(1024)
            .expect("ordinary allocation before reservation");
        let mut reservation = manager
            .reserve_bytes(3072)
            .expect("reservation must fit beside the ordinary owner");
        assert_eq!(manager.remaining_bytes(), 0);
        assert!(matches!(
            manager.alloc::<u8>(1),
            Err(XlogError::ResourceExhausted { .. })
        ));

        let reserved = reservation
            .alloc::<u8>(3072)
            .expect("reserved allocation bypasses the already-paid local claim");
        assert_eq!(manager.allocated_bytes(), 4096);
        drop(reservation);
        drop(reserved);
        assert_eq!(manager.allocated_bytes(), 1024);
        assert_eq!(manager.remaining_bytes(), 3072);
        drop(ordinary);
        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(manager.remaining_bytes(), 4096);
    }

    #[test]
    fn public_accounting_mutators_refuse_live_reservations_and_allocations() {
        let Some(device) = try_device() else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::new(
            device,
            MemoryBudget::with_limit(4096),
        ));
        let reservation = manager.reserve_bytes(2048).expect("reservation");

        let free_error = manager
            .record_free(1)
            .expect_err("public accounting cannot release a live reservation");
        assert!(
            matches!(free_error, XlogError::Kernel(detail) if detail.contains("live tracked bytes"))
        );
        let reset_error = manager
            .reset_tracking()
            .expect_err("public reset cannot erase a live reservation");
        assert!(
            matches!(reset_error, XlogError::Kernel(detail) if detail.contains("live tracked bytes"))
        );
        assert_eq!(manager.remaining_bytes(), 2048);

        drop(reservation);
        let allocation = manager.alloc::<u8>(1024).expect("allocation");
        assert!(manager.record_free(1024).is_err());
        assert!(manager.reset_tracking().is_err());
        assert_eq!(manager.allocated_bytes(), 1024);
        assert_eq!(manager.remaining_bytes(), 3072);

        drop(allocation);
        manager
            .reset_tracking()
            .expect("quiescent accounting can reset diagnostic peaks");
        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(manager.remaining_bytes(), 4096);
    }

    #[test]
    fn public_record_free_refuses_underflow_without_mutation() {
        let Some(device) = try_device() else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::new(
            device,
            MemoryBudget::with_limit(4096),
        ));

        let error = manager
            .record_free(1)
            .expect_err("unowned bytes cannot be released from accounting");
        assert!(matches!(error, XlogError::Kernel(detail) if detail.contains("underflow")));
        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(manager.remaining_bytes(), 4096);
        manager
            .record_free(0)
            .expect("zero-byte release is a no-op");
    }

    #[test]
    fn typed_post_release_error_honors_physical_release_proof() {
        let Some((device, runtime, deallocate_calls)) = try_runtime_with_deallocation_failure()
        else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::with_runtime(
            device,
            MemoryBudget::with_limit(4096),
            Arc::clone(&runtime),
        ));
        let allocation = manager.alloc::<u8>(1024).expect("typed allocation");

        drop(allocation);

        assert_eq!(deallocate_calls.load(Ordering::SeqCst), 1);
        assert_eq!(runtime.bytes_outstanding(), 0);
        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(manager.remaining_bytes(), 4096);
        assert_eq!(manager.deallocation_failure_count(), 0);
        assert_eq!(manager.deallocation_failure_bytes(), 0);
        manager.reset_tracking().unwrap();
    }

    #[test]
    fn raw_post_release_error_honors_physical_release_proof() {
        let Some((device, runtime, deallocate_calls)) = try_runtime_with_deallocation_failure()
        else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::with_runtime(
            device,
            MemoryBudget::with_limit(4096),
            Arc::clone(&runtime),
        ));
        let allocation = manager
            .alloc_raw(512, AllocTag::UNTAGGED)
            .expect("raw allocation");

        drop(allocation);

        assert_eq!(deallocate_calls.load(Ordering::SeqCst), 1);
        assert_eq!(runtime.bytes_outstanding(), 0);
        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(manager.remaining_bytes(), 4096);
        assert_eq!(manager.deallocation_failure_count(), 0);
        assert_eq!(manager.deallocation_failure_bytes(), 0);
        manager.reset_tracking().unwrap();
    }

    #[test]
    fn test_memory_manager_multiple_allocs() {
        let Some(device) = try_device() else {
            return;
        };
        let budget = MemoryBudget::with_limit(4096); // 4 KB
        let manager = Arc::new(GpuMemoryManager::new(device, budget));

        // First allocation: 256 u32 = 1024 bytes
        let _slice1 = manager
            .alloc::<u32>(256)
            .expect("First allocation should succeed");
        assert_eq!(manager.allocated_bytes(), 1024);

        // Second allocation: 256 u32 = 1024 bytes
        let _slice2 = manager
            .alloc::<u32>(256)
            .expect("Second allocation should succeed");
        assert_eq!(manager.allocated_bytes(), 2048);

        // Third allocation that would exceed budget
        let result = manager.alloc::<u32>(1024); // 4096 bytes, would make total 6144
        assert!(result.is_err());

        // Allocated should still be 2048
        assert_eq!(manager.allocated_bytes(), 2048);
    }

    #[test]
    fn test_memory_manager_record_free() {
        let Some(device) = try_device() else {
            return;
        };
        let budget = MemoryBudget::with_limit(4096);
        let manager = Arc::new(GpuMemoryManager::new(device, budget));

        // Allocate
        let slice = manager
            .alloc::<u32>(256)
            .expect("Allocation should succeed");
        assert_eq!(manager.allocated_bytes(), 1024);

        // Drop should automatically update tracking
        drop(slice);
        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(manager.remaining_bytes(), 4096);
    }

    #[test]
    fn test_memory_manager_peak_tracking() {
        let Some(device) = try_device() else {
            return;
        };
        let budget = MemoryBudget::with_limit(8192);
        let manager = Arc::new(GpuMemoryManager::new(device, budget));

        let a = manager.alloc::<u32>(256).expect("alloc a"); // 1024 B
        let b = manager.alloc::<u32>(512).expect("alloc b"); // 2048 B
        assert_eq!(manager.peak_bytes(), 3072);

        // Frees lower `allocated` but never the peak.
        drop(b);
        assert_eq!(manager.allocated_bytes(), 1024);
        assert_eq!(manager.peak_bytes(), 3072);

        // reset_peak restarts the window from live state.
        manager.reset_peak();
        assert_eq!(manager.peak_bytes(), 1024);

        let c = manager.alloc::<u32>(128).expect("alloc c"); // 512 B
        assert_eq!(manager.peak_bytes(), 1536);

        drop(c);
        drop(a);
        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(manager.peak_bytes(), 1536);
    }

    #[test]
    fn memory_pressure_alloc_reports_exact_cumulative_pressure() {
        let Some(device) = try_device() else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::new(
            device,
            MemoryBudget::with_limit(4096),
        ));
        let baseline = manager.alloc::<u8>(1024).expect("baseline allocation");

        let error = match manager.alloc::<u8>(4096) {
            Err(error) => error,
            Ok(_) => panic!("cumulative allocation must exceed the local budget"),
        };

        assert_memory_pressure(
            error,
            "GPU memory pressure: layer=manager_alloc current_bytes=1024 requested_bytes=4096 required_bytes=5120 required_u64_overflow=false budget_bytes=4096 prior_peak_bytes=1024",
            5120,
            4096,
        );
        assert_eq!(manager.allocated_bytes(), 1024);
        assert_eq!(manager.peak_bytes(), 1024, "refusal must not raise peak");
        drop(baseline);
    }

    #[test]
    fn memory_pressure_check_budget_reports_exact_cumulative_pressure() {
        let Some(device) = try_device() else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::new(
            device,
            MemoryBudget::with_limit(1000),
        ));
        let baseline = manager.alloc::<u8>(512).expect("baseline allocation");

        let error = manager
            .check_budget(600)
            .expect_err("cumulative request must exceed the local budget");

        assert_memory_pressure(
            error,
            "GPU memory pressure: layer=manager_check_budget current_bytes=512 requested_bytes=600 required_bytes=1112 required_u64_overflow=false budget_bytes=1000 prior_peak_bytes=512",
            1112,
            1000,
        );
        assert_eq!(manager.allocated_bytes(), 512);
        assert_eq!(
            manager.peak_bytes(),
            512,
            "check-only refusal must not raise peak"
        );
        drop(baseline);
    }

    #[test]
    fn memory_pressure_check_budget_reports_u64_representability_overflow() {
        let Some(device) = try_device() else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::new(
            device,
            MemoryBudget::with_limit(u64::MAX),
        ));
        manager
            .accounting
            .budget_reserved
            .store(u64::MAX - 3, Ordering::SeqCst);
        manager
            .accounting
            .allocated
            .store(u64::MAX - 3, Ordering::SeqCst);
        manager
            .accounting
            .peak
            .store(u64::MAX - 3, Ordering::SeqCst);

        let error = manager
            .check_budget(8)
            .expect_err("the exact required byte count is not representable as u64");

        assert_memory_pressure(
            error,
            "GPU memory pressure: layer=manager_check_budget current_bytes=18446744073709551612 requested_bytes=8 required_bytes=18446744073709551620 required_u64_overflow=true budget_bytes=18446744073709551615 prior_peak_bytes=18446744073709551612",
            u64::MAX,
            u64::MAX,
        );
        assert_eq!(manager.allocated_bytes(), u64::MAX - 3);
        assert_eq!(manager.peak_bytes(), u64::MAX - 3);

        // Restore the synthetic accounting state before dropping the fixture.
        manager
            .accounting
            .budget_reserved
            .store(0, Ordering::SeqCst);
        manager.accounting.allocated.store(0, Ordering::SeqCst);
        manager.accounting.peak.store(0, Ordering::SeqCst);
    }

    #[test]
    fn test_cuda_buffer_from_columns() {
        let Some(device) = try_device() else {
            return;
        };
        let budget = MemoryBudget::with_limit(1024 * 1024);
        let manager = Arc::new(GpuMemoryManager::new(device, budget));

        let schema = Schema::new(vec![
            ("col1".to_string(), ScalarType::U32),
            ("col2".to_string(), ScalarType::U32),
        ]);

        // Allocate columns (100 rows * 4 bytes = 400 bytes each)
        let col1 = manager.alloc::<u8>(400).expect("Alloc col1");
        let col2 = manager.alloc::<u8>(400).expect("Alloc col2");

        let mut d_num_rows = manager.alloc::<u32>(1).expect("Alloc row count");
        manager
            .device()
            .inner()
            .htod_sync_copy_into(&[100u32], &mut d_num_rows)
            .expect("Upload row count");
        let buffer =
            CudaBuffer::from_columns(vec![col1.into(), col2.into()], 100, d_num_rows, schema);

        assert_eq!(buffer.num_rows(), 100);
        assert_eq!(buffer.arity(), 2);
        assert!(!buffer.is_empty());
        assert!(buffer.column(0).is_some());
        assert!(buffer.column(1).is_some());
        assert!(buffer.column(2).is_none());
    }

    #[test]
    fn test_cuda_buffer_from_columns_mismatch() {
        let schema = Schema::new(vec![
            ("col1".to_string(), ScalarType::U32),
            ("col2".to_string(), ScalarType::U32),
        ]);

        let Some(device) = try_device() else {
            return;
        };
        let budget = MemoryBudget::with_limit(1024 * 1024);
        let manager = Arc::new(GpuMemoryManager::new(device, budget));
        let mut d_num_rows = manager.alloc::<u32>(1).expect("Alloc row count");
        manager
            .device()
            .inner()
            .htod_sync_copy_into(&[100u32], &mut d_num_rows)
            .expect("Upload row count");

        // This should panic: 0 columns but schema has 2.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            CudaBuffer::from_columns(vec![], 100, d_num_rows, schema);
        }));
        assert!(
            result.is_err(),
            "Expected from_columns to panic on schema mismatch"
        );
    }

    fn try_runtime() -> Option<(
        Arc<CudaDevice>,
        Arc<crate::device_runtime::XlogDeviceRuntime>,
    )> {
        use crate::device_runtime::{
            AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, StreamPool,
            XlogDeviceRuntime,
        };
        let device = try_device()?;
        let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
            AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
        );
        let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
            Box::new(GlobalDeviceBudget::new(async_resource, 64 * 1024 * 1024));
        Some((
            Arc::clone(&device),
            Arc::new(XlogDeviceRuntime::with_resource(
                Arc::clone(&device),
                0,
                pool,
                budget,
            )),
        ))
    }

    fn try_unbudgeted_runtime() -> Option<(
        Arc<CudaDevice>,
        Arc<crate::device_runtime::XlogDeviceRuntime>,
    )> {
        use crate::device_runtime::{
            AsyncCudaResource, DeviceMemoryResource, StreamPool, XlogDeviceRuntime,
        };
        let device = try_device()?;
        let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
            AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
        );
        Some((
            Arc::clone(&device),
            Arc::new(XlogDeviceRuntime::with_resource(
                Arc::clone(&device),
                0,
                pool,
                resource,
            )),
        ))
    }

    fn try_runtime_with_deallocation_failure() -> Option<(
        Arc<CudaDevice>,
        Arc<crate::device_runtime::XlogDeviceRuntime>,
        Arc<AtomicU64>,
    )> {
        use crate::device_runtime::{StreamPool, XlogDeviceRuntime};
        let device = try_device()?;
        let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let deallocate_calls = Arc::new(AtomicU64::new(0));
        let resource: Box<dyn DeviceMemoryResource + Send + Sync> =
            Box::new(FailAfterDeallocateResource {
                inner: DirectCudaResource::new(Arc::clone(&device), 0),
                deallocate_calls: Arc::clone(&deallocate_calls),
            });
        Some((
            Arc::clone(&device),
            Arc::new(XlogDeviceRuntime::with_resource(
                Arc::clone(&device),
                0,
                pool,
                resource,
            )),
            deallocate_calls,
        ))
    }

    fn try_runtime_with_first_allocation_failure(
        limit: usize,
    ) -> Option<(
        Arc<CudaDevice>,
        Arc<crate::device_runtime::XlogDeviceRuntime>,
    )> {
        use crate::device_runtime::{GlobalDeviceBudget, StreamPool, XlogDeviceRuntime};
        let device = try_device()?;
        let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let failing_resource: Box<dyn DeviceMemoryResource + Send + Sync> =
            Box::new(FailFirstAllocationResource {
                inner: DirectCudaResource::new(Arc::clone(&device), 0),
                fail_next: std::sync::atomic::AtomicBool::new(true),
            });
        let budget_resource: Box<dyn DeviceMemoryResource + Send + Sync> =
            Box::new(GlobalDeviceBudget::new(failing_resource, limit));
        Some((
            Arc::clone(&device),
            Arc::new(XlogDeviceRuntime::with_resource(
                Arc::clone(&device),
                0,
                pool,
                budget_resource,
            )),
        ))
    }

    fn try_runtime_with_budget(
        limit: usize,
    ) -> Option<(
        Arc<CudaDevice>,
        Arc<crate::device_runtime::XlogDeviceRuntime>,
    )> {
        use crate::device_runtime::{
            DeviceMemoryResource, DirectCudaResource, GlobalDeviceBudget, StreamPool,
            XlogDeviceRuntime,
        };
        let device = try_device()?;
        let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let direct_resource: Box<dyn DeviceMemoryResource + Send + Sync> =
            Box::new(DirectCudaResource::new(Arc::clone(&device), 0));
        let budget_resource: Box<dyn DeviceMemoryResource + Send + Sync> =
            Box::new(GlobalDeviceBudget::new(direct_resource, limit));
        Some((
            Arc::clone(&device),
            Arc::new(XlogDeviceRuntime::with_resource(
                Arc::clone(&device),
                0,
                pool,
                budget_resource,
            )),
        ))
    }

    fn try_runtime_with_logging_budget(
        limit: usize,
    ) -> Option<(
        Arc<CudaDevice>,
        Arc<crate::device_runtime::XlogDeviceRuntime>,
        Arc<crate::device_runtime::InMemorySink>,
    )> {
        use crate::device_runtime::{
            DeviceMemoryResource, DirectCudaResource, GlobalDeviceBudget, InMemorySink,
            LoggingResource, LoggingSink, StreamPool, XlogDeviceRuntime,
        };
        let device = try_device()?;
        let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let direct_resource: Box<dyn DeviceMemoryResource + Send + Sync> =
            Box::new(DirectCudaResource::new(Arc::clone(&device), 0));
        let budget_resource: Box<dyn DeviceMemoryResource + Send + Sync> =
            Box::new(GlobalDeviceBudget::new(direct_resource, limit));
        let sink = Arc::new(InMemorySink::new());
        let logging_sink: Arc<dyn LoggingSink> = sink.clone();
        let logging_resource: Box<dyn DeviceMemoryResource + Send + Sync> =
            Box::new(LoggingResource::new(budget_resource, logging_sink));
        Some((
            Arc::clone(&device),
            Arc::new(XlogDeviceRuntime::with_resource(
                Arc::clone(&device),
                0,
                pool,
                logging_resource,
            )),
            sink,
        ))
    }

    fn pause_request_after_local_reservation(
        manager: &GpuMemoryManager,
        paused_bytes: u64,
    ) -> (Arc<std::sync::Barrier>, Arc<std::sync::Barrier>) {
        let entered = Arc::new(std::sync::Barrier::new(2));
        let release = Arc::new(std::sync::Barrier::new(2));
        let hook_entered = Arc::clone(&entered);
        let hook_release = Arc::clone(&release);
        *manager
            .after_local_reservation_hook
            .lock()
            .expect("after-local-reservation test hook poisoned") = Some(Arc::new(move |bytes| {
            if bytes == paused_bytes {
                hook_entered.wait();
                hook_release.wait();
            }
        }));
        (entered, release)
    }

    #[test]
    fn memory_pressure_concurrent_alloc_raw_refusal_excludes_provisional_bytes() {
        let Some((device, runtime)) = try_runtime_with_budget(4096) else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::with_runtime(
            device,
            MemoryBudget::with_limit(8192),
            runtime,
        ));
        let (paused, release) = pause_request_after_local_reservation(&manager, 4096);

        let refused_manager = Arc::clone(&manager);
        let refused = std::thread::spawn(move || {
            refused_manager
                .alloc_raw(4096, AllocTag::UNTAGGED)
                .expect_err("runtime must refuse the paused request")
        });
        paused.wait();

        let admitted = manager
            .alloc_raw(1024, AllocTag::UNTAGGED)
            .expect("concurrent smaller request must be admitted");
        release.wait();
        let error = refused.join().expect("paused allocation thread panicked");

        assert_memory_pressure(
            error,
            "GPU memory pressure: layer=device_runtime current_bytes=1024 requested_bytes=4096 required_bytes=5120 required_u64_overflow=false budget_bytes=4096 prior_peak_bytes=1024",
            5120,
            4096,
        );
        assert_eq!(manager.allocated_bytes(), 1024);
        assert_eq!(
            manager.peak_bytes(),
            1024,
            "refused provisional bytes must never enter the admitted peak"
        );
        drop(admitted);
    }

    #[test]
    fn memory_pressure_concurrent_typed_alloc_refusal_excludes_provisional_bytes() {
        let Some((device, runtime)) = try_runtime_with_budget(4096) else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::with_runtime(
            device,
            MemoryBudget::with_limit(8192),
            runtime,
        ));
        let (paused, release) = pause_request_after_local_reservation(&manager, 4096);

        let refused_manager = Arc::clone(&manager);
        let refused = std::thread::spawn(move || match refused_manager.alloc::<u8>(4096) {
            Err(error) => error,
            Ok(_) => panic!("runtime must refuse the paused typed request"),
        });
        paused.wait();

        let admitted = manager
            .alloc::<u8>(1024)
            .expect("concurrent smaller typed request must be admitted");
        release.wait();
        let error = refused.join().expect("paused allocation thread panicked");

        assert_memory_pressure(
            error,
            "GPU memory pressure: layer=device_runtime current_bytes=1024 requested_bytes=4096 required_bytes=5120 required_u64_overflow=false budget_bytes=4096 prior_peak_bytes=1024",
            5120,
            4096,
        );
        assert_eq!(manager.allocated_bytes(), 1024);
        assert_eq!(
            manager.peak_bytes(),
            1024,
            "refused provisional bytes must never enter the admitted peak"
        );
        drop(admitted);
    }

    #[test]
    fn memory_pressure_alloc_raw_reports_exact_local_pressure() {
        let Some((device, runtime)) = try_runtime_with_budget(64 * 1024) else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::with_runtime(
            device,
            MemoryBudget::with_limit(4096),
            runtime,
        ));
        let baseline = manager
            .alloc_raw(1024, AllocTag::UNTAGGED)
            .expect("baseline allocation");

        let error = manager
            .alloc_raw(4096, AllocTag::UNTAGGED)
            .expect_err("cumulative allocation must exceed the local budget");

        assert_memory_pressure(
            error,
            "GPU memory pressure: layer=manager_alloc_raw current_bytes=1024 requested_bytes=4096 required_bytes=5120 required_u64_overflow=false budget_bytes=4096 prior_peak_bytes=1024",
            5120,
            4096,
        );
        assert_eq!(manager.allocated_bytes(), 1024);
        assert_eq!(manager.peak_bytes(), 1024, "refusal must not raise peak");
        drop(baseline);
    }

    #[test]
    fn memory_pressure_alloc_raw_drop_restores_local_headroom() {
        let Some((device, runtime)) = try_runtime_with_budget(64 * 1024) else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::with_runtime(
            device,
            MemoryBudget::with_limit(1024),
            runtime,
        ));

        let allocation = manager
            .alloc_raw(1024, AllocTag::UNTAGGED)
            .expect("initial allocation must consume the local budget");
        assert_eq!(manager.allocated_bytes(), 1024);
        assert_eq!(manager.remaining_bytes(), 0);

        drop(allocation);
        assert_eq!(manager.allocated_bytes(), 0);
        assert_eq!(
            manager.remaining_bytes(),
            1024,
            "dropping an admitted raw allocation must restore local headroom"
        );

        let replacement = manager
            .alloc_raw(1024, AllocTag::UNTAGGED)
            .expect("restored local headroom must admit a replacement allocation");
        drop(replacement);
    }

    #[test]
    fn memory_pressure_runtime_rejection_preserves_peak() {
        let Some((device, runtime)) = try_runtime_with_budget(4096) else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::with_runtime(
            device,
            MemoryBudget::with_limit(8192),
            runtime,
        ));
        let baseline = manager
            .alloc_raw(1024, AllocTag::UNTAGGED)
            .expect("baseline allocation");

        let error = manager
            .alloc_raw(4096, AllocTag::UNTAGGED)
            .expect_err("runtime budget must reject the cumulative allocation");

        assert_memory_pressure(
            error,
            "GPU memory pressure: layer=device_runtime current_bytes=1024 requested_bytes=4096 required_bytes=5120 required_u64_overflow=false budget_bytes=4096 prior_peak_bytes=1024",
            5120,
            4096,
        );
        assert_eq!(
            manager.allocated_bytes(),
            1024,
            "local reservation must roll back"
        );
        assert_eq!(
            manager.peak_bytes(),
            1024,
            "runtime refusal must not advance the manager peak"
        );
        drop(baseline);
    }

    /// xlog-owned DLPack column constructed from a
    /// runtime-backed slice exposes its `DeviceBlock` via
    /// `runtime_block()` and reports `is_external() == false`.
    /// The recorder will record it normally instead of
    /// strict-rejecting.
    #[test]
    fn test_xlog_owned_dlpack_runtime_backed_carries_identity() {
        let Some((device, runtime)) = try_runtime() else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::with_runtime(
            Arc::clone(&device),
            MemoryBudget::with_limit(1024 * 1024),
            Arc::clone(&runtime),
        ));
        let slice = manager.alloc::<u8>(64).expect("alloc runtime-backed");
        assert!(slice.runtime_block().is_some());
        let stream = device.inner().stream().clone();
        let slice = Arc::new(slice);
        let tensor = crate::dlpack::export_slice_managed_tensor(
            Arc::clone(&slice),
            device.ordinal() as i32,
            crate::dlpack::DLDataType {
                code: crate::dlpack::K_DLUINT,
                bits: 8,
                lanes: 1,
            },
            1,
            slice.len(),
        )
        .expect("real export owner");
        stream.synchronize().expect("producer ready");
        let col = CudaColumn::dlpack_xlog_owned(slice, stream, tensor);
        assert!(
            !col.is_external(),
            "xlog-owned DLPack column must report is_external=false"
        );
        assert!(
            col.runtime_block().is_some(),
            "xlog-owned DLPack column over a runtime-backed slice must expose runtime_block"
        );
    }

    /// xlog-owned DLPack over a LEGACY (cudarc-backed) slice:
    /// `is_external()` is still false (xlog owns the
    /// allocation), but `runtime_block()` is None because the
    /// underlying slice has no runtime block. Strict recorders
    /// will reject with the "legacy cudarc-backed" message
    /// rather than the "external memory" message.
    #[test]
    fn test_xlog_owned_dlpack_legacy_backed_no_runtime_block() {
        let Some(device) = try_device() else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::new(
            Arc::clone(&device),
            MemoryBudget::with_limit(1024 * 1024),
        ));
        let slice = manager.alloc::<u8>(64).expect("alloc legacy");
        assert!(slice.runtime_block().is_none());
        let stream = device.inner().stream().clone();
        let slice = Arc::new(slice);
        let tensor = crate::dlpack::export_slice_managed_tensor(
            Arc::clone(&slice),
            device.ordinal() as i32,
            crate::dlpack::DLDataType {
                code: crate::dlpack::K_DLUINT,
                bits: 8,
                lanes: 1,
            },
            1,
            slice.len(),
        )
        .expect("real export owner");
        stream.synchronize().expect("producer ready");
        let col = CudaColumn::dlpack_xlog_owned(slice, stream, tensor);
        assert!(
            !col.is_external(),
            "xlog-owned DLPack column is owned regardless of allocator backing"
        );
        assert!(
            col.runtime_block().is_none(),
            "legacy-backed slice has no runtime block, even when wrapped xlog-owned"
        );
    }

    /// True external DLPack (no source_slice) — the existing
    /// `dlpack` constructor — keeps reporting `is_external=true`
    /// and `runtime_block=None`. Strict recorders reject with
    /// the "external memory" message.
    #[test]
    fn test_external_dlpack_remains_external() {
        let Some(device) = try_device() else {
            return;
        };
        let stream = device.inner().stream().clone();
        let manager = Arc::new(GpuMemoryManager::new(
            Arc::clone(&device),
            MemoryBudget::with_limit(1024 * 1024),
        ));
        let slice = Arc::new(manager.alloc::<u8>(1).expect("external allocation"));
        let ptr = slice.device_ptr_value();
        let tensor = crate::dlpack::export_slice_managed_tensor(
            slice,
            device.ordinal() as i32,
            crate::dlpack::DLDataType {
                code: crate::dlpack::K_DLUINT,
                bits: 8,
                lanes: 1,
            },
            1,
            1,
        )
        .expect("real external owner");
        stream.synchronize().expect("producer ready");
        // SAFETY: the token retains the live one-byte allocation on this context,
        // producer work has completed, and there are no foreign accesses.
        let col = unsafe { CudaColumn::dlpack(ptr, 1, stream, tensor) };
        assert!(
            col.is_external(),
            "true external DLPack column must report is_external=true"
        );
        assert!(
            col.runtime_block().is_none(),
            "true external DLPack column has no xlog-side runtime block"
        );
    }

    /// xlog-owned Arrow device column carries identity through
    /// `arrow_device_xlog_owned`. Mirrors the DLPack test;
    /// builds a minimal `ArrowDeviceImport` from an empty
    /// `ArrayData`.
    #[test]
    fn test_xlog_owned_arrow_device_runtime_backed_carries_identity() {
        let Some((device, runtime)) = try_runtime() else {
            return;
        };
        let manager = Arc::new(GpuMemoryManager::with_runtime(
            Arc::clone(&device),
            MemoryBudget::with_limit(1024 * 1024),
            Arc::clone(&runtime),
        ));
        let slice = manager.alloc::<u8>(64).expect("alloc runtime-backed");
        assert!(slice.runtime_block().is_some());
        let stream = device.inner().stream().clone();
        // Synthesize a minimal ArrowDeviceImport via empty
        // ArrayData; Arrow is not exercised on the data path
        // here — the recorder only reads the column metadata.
        let import = Arc::new(crate::arrow_device::ArrowDeviceImport::new(
            arrow::array::ArrayData::new_null(&arrow::datatypes::DataType::UInt8, 0),
        ));
        let col = CudaColumn::arrow_device_xlog_owned(Arc::new(slice), stream, import);
        assert!(
            !col.is_external(),
            "xlog-owned Arrow device column must report is_external=false"
        );
        assert!(
            col.runtime_block().is_some(),
            "xlog-owned Arrow column over a runtime-backed slice must expose runtime_block"
        );
    }

    /// True external Arrow device column (no source_slice)
    /// keeps reporting external + no runtime block.
    #[test]
    fn test_external_arrow_device_remains_external() {
        let Some(device) = try_device() else {
            return;
        };
        let stream = device.inner().stream().clone();
        let import = Arc::new(crate::arrow_device::ArrowDeviceImport::new(
            arrow::array::ArrayData::new_null(&arrow::datatypes::DataType::UInt8, 0),
        ));
        // SAFETY: the empty ArrayData owns this empty span; no bytes or producer
        // work exist, and its release is safe on any host thread.
        let col = unsafe { CudaColumn::arrow_device(0, 0, stream, import) };
        assert!(
            col.is_external(),
            "true external Arrow column must report is_external=true"
        );
        assert!(
            col.runtime_block().is_none(),
            "true external Arrow column has no xlog-side runtime block"
        );
    }
}
