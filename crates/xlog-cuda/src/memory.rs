//! CUDA memory management
//!
//! This module provides GPU memory management with budget enforcement.
//! It wraps cudarc's allocation functions and tracks total allocated memory.

use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceSlice, SyncOnDrop};
use xlog_core::{resolve_bool, MemoryBudget, Result, Schema, XlogError};

use crate::arrow_device::ArrowDeviceImport;
use crate::cuda_compat::{AsKernelParam, DeviceParamStorage, IntoKernelParamStorage};
use crate::device_runtime::{
    AllocTag, BlockId, BlockState, DeviceBlock, ResourceError, RuntimeMemoryReservation, StreamId,
    XlogDeviceRuntime,
};
use crate::dlpack::DlpackManagedTensor;
use crate::CudaDevice;

#[cfg(test)]
type AfterLocalReservationHook = std::sync::Mutex<Option<Arc<dyn Fn(u64) + Send + Sync + 'static>>>;

/// GPU memory manager with budget enforcement
///
/// Tracks allocated GPU memory and enforces a memory budget.
/// When the budget would be exceeded, returns `XlogError::ResourceExhausted`.
///
/// # Device-runtime routing
///
/// Canonical CUDA providers construct the manager via
/// [`GpuMemoryManager::with_runtime`]. Its [`XlogDeviceRuntime`] mediates
/// allocations through `LoggingResource` when enabled, then
/// `GlobalDeviceBudget`, then `AsyncCudaResource`. When attached:
///   * [`GpuMemoryManager::alloc::<T>`] routes the underlying
///     allocation through the runtime and produces a typed view via
///     cudarc's `upgrade_device_ptr::<T>`. The returned
///     [`TrackedCudaSlice`] frees through the runtime on drop.
///   * [`GpuMemoryManager::alloc_raw`] is the explicit raw-bytes
///     entry point (no typed view), also runtime-routed.
///
/// Both budgets apply: the manager's local `MemoryBudget` AND any
/// `GlobalDeviceBudget` stacked above the runtime's underlying
/// resource.
///
/// The crate-private runtime-free constructor exists for focused allocator
/// tests. Its typed allocations use cudarc directly, while `alloc_raw` refuses
/// to run because it cannot honor the runtime ownership contract. Production
/// provider construction never uses that path.
/// at construction sites that need it.
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
        let runtime_reservation = self.runtime_reservation.as_mut();
        match manager.alloc_after_local_reservation::<T>(len, bytes, runtime_reservation) {
            Ok(allocation) => Ok(allocation),
            Err(error) => {
                self.remaining_bytes = self
                    .remaining_bytes
                    .checked_add(bytes)
                    .expect("reservation rollback overflow");
                Err(error)
            }
        }
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
        let runtime_reservation = self.runtime_reservation.as_mut();
        match manager.alloc_raw_after_local_reservation(bytes, bytes_u64, tag, runtime_reservation)
        {
            Ok(allocation) => Ok(allocation),
            Err(error) => {
                self.remaining_bytes = self
                    .remaining_bytes
                    .checked_add(bytes_u64)
                    .expect("reservation rollback overflow");
                Err(error)
            }
        }
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

/// Selects which allocator owns the underlying device memory of a
/// [`TrackedCudaSlice`]. Internal — surfaced only via the methods
/// on `TrackedCudaSlice`. Migrated allocations carry `Runtime`
/// backing; legacy allocations stay on `Cudarc`.
enum Backing {
    /// Legacy: cudarc owns the slice. The inner `CudaSlice<T>` is
    /// the actual handle returned by `device.alloc::<T>(..)`, and
    /// dropping it invokes cudarc's free path. The
    /// `TrackedCudaSlice` `Drop` impl runs that drop explicitly so
    /// the timing is identical to pre-migration behavior.
    Cudarc,
    /// v0.6 runtime-routed: the [`XlogDeviceRuntime`] owns the
    /// allocation via its resource stack, and the inner
    /// `CudaSlice<T>` is a typed view created by
    /// `upgrade_device_ptr::<T>` over the runtime's raw pointer.
    /// On drop, the inner view must be **forgotten** (cudarc must
    /// not free) and the runtime must be told to deallocate the
    /// `DeviceBlock`. Order of operations matters: deallocate the
    /// block first, then forget the view, so the runtime sees the
    /// block in its `live` map.
    Runtime {
        runtime: Arc<XlogDeviceRuntime>,
        block: Option<DeviceBlock>,
    },
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

/// A `CudaSlice` that automatically updates `GpuMemoryManager`
/// allocation tracking on drop. Inner slice is wrapped in
/// `ManuallyDrop` so the [`Backing`] enum can choose between
/// cudarc-side free (legacy) and runtime-side deallocate (migrated)
/// without producing a double-free.
pub struct TrackedCudaSlice<T: cudarc::driver::DeviceRepr> {
    bytes: u64,
    manager: Arc<GpuMemoryManager>,
    inner: ManuallyDrop<CudaSlice<T>>,
    raw_ptr: cudarc::driver::sys::CUdeviceptr,
    backing: Backing,
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

impl<T: cudarc::driver::DeviceRepr> Deref for TrackedCudaSlice<T> {
    type Target = CudaSlice<T>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: cudarc::driver::DeviceRepr> DerefMut for TrackedCudaSlice<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<T: cudarc::driver::DeviceRepr> DeviceSlice<T> for TrackedCudaSlice<T> {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn stream(&self) -> &Arc<CudaStream> {
        self.inner.stream()
    }
}

impl<T: cudarc::driver::DeviceRepr> DevicePtr<T> for TrackedCudaSlice<T> {
    fn device_ptr<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (cudarc::driver::sys::CUdeviceptr, SyncOnDrop<'a>) {
        // Explicit `&*` deref through ManuallyDrop — the trait
        // method is not auto-resolved through the wrapper.
        DevicePtr::device_ptr(&*self.inner, stream)
    }
}

impl<T: cudarc::driver::DeviceRepr> DevicePtrMut<T> for TrackedCudaSlice<T> {
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (cudarc::driver::sys::CUdeviceptr, SyncOnDrop<'a>) {
        DevicePtrMut::device_ptr_mut(&mut *self.inner, stream)
    }
}

impl<T: cudarc::driver::DeviceRepr> TrackedCudaSlice<T> {
    pub fn device_ptr(&self) -> &cudarc::driver::sys::CUdeviceptr {
        &self.raw_ptr
    }

    pub fn device_ptr_value(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.raw_ptr
    }

    /// Stable address of the memory manager that owns this allocation.
    pub fn memory_manager_ptr_value(&self) -> usize {
        Arc::as_ptr(&self.manager) as usize
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

    /// Borrow the underlying [`DeviceBlock`] for runtime-backed
    /// allocations. Returns `None` for legacy cudarc-backed
    /// slices ([`Backing::Cudarc`]) — those are not tracked by
    /// the v0.6 device runtime and therefore have no
    /// runtime-side block to record uses against.
    ///
    /// Callers (notably [`crate::launch::LaunchRecorder`]) use
    /// this to attach cross-stream uses via
    /// [`crate::device_runtime::XlogDeviceRuntime::record_block_use`].
    /// A `None` return signals that the slice is on the legacy
    /// path and the recorder cannot track its lifetime — callers
    /// must either route the allocation through
    /// [`GpuMemoryManager::with_runtime`] or accept that no
    /// cross-stream safety applies to this buffer.
    pub fn runtime_block(&self) -> Option<&crate::device_runtime::DeviceBlock> {
        match &self.backing {
            Backing::Cudarc => None,
            Backing::Runtime { block, .. } => block.as_ref(),
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
        // Wrap `self` in `ManuallyDrop` so its `Drop` impl never
        // runs — we are doing the cleanup manually below by either
        // (a) leaving the original `inner` forgotten and reusing
        // its `backing` (Runtime mode), or (b) leaving the original
        // `inner` forgotten while the new u8 view takes ownership
        // via `upgrade_device_ptr` (Cudarc mode — same dance as
        // the pre-migration code).
        let this = ManuallyDrop::new(self);
        let bytes = this.bytes;
        let manager = Arc::clone(&this.manager);
        let ptr = this.raw_ptr;

        let len_bytes: usize = bytes
            .try_into()
            .expect("TrackedCudaSlice byte size must fit into usize");

        // SAFETY: `this` is `ManuallyDrop`, so its destructor will
        // not run. We bit-copy `backing` out of the original; the
        // original location is forgotten along with the rest of
        // `this`. This is sound because each field is owned and not
        // touched again.
        let backing: Backing = unsafe { std::ptr::read(&this.backing) };

        // SAFETY: the runtime / cudarc-side memory is still live —
        // the original `inner` ManuallyDrop never had its
        // destructor called, so cudarc has not freed. The new
        // `CudaSlice<u8>` is a typed view over the same bytes.
        // For Cudarc backing the new view will free on drop (one
        // alloc, one free, balanced — same as pre-migration).
        // For Runtime backing the new view will be `mem::forget`
        // -ed by the new `Drop` impl, and the runtime's
        // `deallocate(block)` (carried in `backing`) is the sole
        // free path.
        let new_inner = unsafe {
            manager
                .device
                .inner()
                .upgrade_device_ptr::<u8>(ptr, len_bytes)
        };

        TrackedCudaSlice {
            bytes,
            manager,
            inner: ManuallyDrop::new(new_inner),
            raw_ptr: ptr,
            backing,
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
        let (ptr, sync) = DevicePtr::device_ptr(&*self.inner, self.inner.stream());
        DeviceParamStorage::synced(ptr, sync)
    }
}

impl<T: cudarc::driver::DeviceRepr> IntoKernelParamStorage for &mut TrackedCudaSlice<T> {
    type Storage = DeviceParamStorage<'static>;

    fn into_kernel_param_storage(self) -> Self::Storage {
        let stream = self.inner.stream().clone();
        let (ptr, sync) = DevicePtrMut::device_ptr_mut(&mut *self.inner, &stream);
        std::mem::forget(sync);
        DeviceParamStorage::unsynced(ptr)
    }
}

impl<T: cudarc::driver::DeviceRepr> Drop for TrackedCudaSlice<T> {
    fn drop(&mut self) {
        let released = match &mut self.backing {
            Backing::Cudarc => {
                // Debug probe (XLOG_DEBUG_POISON_FREE=1): overwrite the
                // allocation with 0xDD before cudarc frees it, so any
                // still-live alias of this memory reads the poison
                // pattern instead of recycled contents. Diagnostic only;
                // off unless the env var is set.
                if poison_free_enabled() && self.bytes > 0 {
                    unsafe {
                        let _ = cudarc::driver::sys::cuMemsetD8_v2(
                            self.raw_ptr,
                            0xDD,
                            self.bytes as usize,
                        );
                    }
                }
                alloc_guard_remove(self.raw_ptr);
                // SAFETY: drop runs at most once per slice, and the
                // inner CudaSlice<T> has not been moved out by any
                // method (`into_bytes` consumes `self` by value and
                // leaves the original ManuallyDrop forgotten).
                unsafe { ManuallyDrop::drop(&mut self.inner) };
                true
            }
            Backing::Runtime { runtime, block } => {
                // Runtime owns the underlying memory. Tell it to
                // deallocate the block; the inner `CudaSlice<T>` is
                // a typed view that must NOT free on its own,
                // which `ManuallyDrop` ensures by simply not
                // calling its destructor here.
                match block.take() {
                    Some(block) => match runtime.deallocate(block) {
                        Ok(()) => true,
                        Err(_) => {
                            self.manager.record_deallocation_failure(self.bytes);
                            false
                        }
                    },
                    None => false,
                }
            }
        };
        if released {
            let release = self.manager.release_owned_allocation(self.bytes);
            debug_assert!(release.is_ok(), "allocation release must be balanced");
        }
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
    /// manager was constructed via [`new`]. Test/diagnostic
    /// accessor; production call sites that need the runtime own
    /// it directly.
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
        let _mutation = self
            .accounting
            .mutation_lock
            .lock()
            .expect("GPU memory accounting poisoned");
        let previous = self.accounting.budget_reserved.load(Ordering::SeqCst);
        let next = previous.checked_sub(bytes).ok_or_else(|| {
            XlogError::Kernel(format!(
                "GPU memory reservation release underflow: current_bytes={} requested_bytes={}",
                previous, bytes
            ))
        })?;
        self.accounting
            .budget_reserved
            .store(next, Ordering::SeqCst);
        Ok(())
    }

    /// Publish a successful allocator admission. The local-budget reservation
    /// already includes `bytes`; only admitted current and peak are updated.
    fn publish_admission(&self, bytes: u64) {
        let _mutation = self
            .accounting
            .mutation_lock
            .lock()
            .expect("GPU memory accounting poisoned");
        let previous = self.accounting.allocated.load(Ordering::SeqCst);
        let admitted = previous
            .checked_add(bytes)
            .expect("admitted allocation accounting overflow");
        self.accounting.allocated.store(admitted, Ordering::SeqCst);
        self.accounting.peak.fetch_max(admitted, Ordering::SeqCst);
    }

    /// Release bytes owned by a successfully destroyed tracked allocation.
    /// Only allocation-owner drop paths may call this method.
    fn release_owned_allocation(&self, bytes: u64) -> Result<()> {
        let _mutation = self
            .accounting
            .mutation_lock
            .lock()
            .expect("GPU memory accounting poisoned");
        let admitted = self.accounting.allocated.load(Ordering::SeqCst);
        let reserved = self.accounting.budget_reserved.load(Ordering::SeqCst);
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
        self.accounting
            .allocated
            .store(next_admitted, Ordering::SeqCst);
        self.accounting
            .budget_reserved
            .store(next_reserved, Ordering::SeqCst);
        Ok(())
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

    /// Allocate GPU memory for `len` elements of type `T`
    ///
    /// # Arguments
    /// * `len` - Number of elements to allocate
    ///
    /// # Returns
    /// A tracked `CudaSlice<T>` containing the allocated memory
    ///
    /// # Errors
    /// - `XlogError::ResourceExhausted` if allocation would exceed budget
    /// - `XlogError::Kernel` if CUDA allocation fails
    ///
    /// # v0.6 routing
    /// When the manager has an attached [`XlogDeviceRuntime`]
    /// (constructed via [`with_runtime`]), the underlying allocation
    /// is routed through the runtime's resource stack and a typed
    /// view is created via cudarc's `upgrade_device_ptr::<T>` over
    /// the runtime's raw pointer. The returned [`TrackedCudaSlice`]
    /// frees through the runtime on drop. Without a runtime
    /// attached, the legacy cudarc `alloc::<T>` path is used and
    /// drop frees through cudarc — bit-for-bit identical to
    /// pre-migration behavior.
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
        match self.alloc_after_local_reservation::<T>(len, bytes, None) {
            Ok(allocation) => Ok(allocation),
            Err(error) => {
                self.rollback_local_reservation(bytes)?;
                Err(error)
            }
        }
    }

    fn alloc_after_local_reservation<T: cudarc::driver::DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
        bytes: u64,
        runtime_reservation: Option<&mut RuntimeMemoryReservation>,
    ) -> Result<TrackedCudaSlice<T>> {
        #[cfg(test)]
        self.run_after_local_reservation_hook(bytes);

        if let Some(runtime) = &self.runtime {
            // Zero-byte allocations (empty Vec, empty buffer) are
            // legitimate in production code. The v0.6 resource
            // stack rejects zero-byte requests by contract
            // (DirectCudaResource and AsyncCudaResource both error
            // on `bytes == 0` because `cuMemAlloc(0)` is undefined
            // behavior in the CUDA driver). Cudarc's `alloc::<T>(0)`
            // does the right thing — returns an empty CudaSlice<T>
            // without calling the driver — so route zero-byte
            // requests through the legacy path even when a runtime
            // is attached. The resulting slice carries
            // `Backing::Cudarc`; its drop is a no-op against
            // cudarc's empty handle.
            //
            // `len == 0` and `bytes == 0` are equivalent here only
            // if `T` has nonzero size (the common case). For
            // zero-sized types (rare but valid in Rust) `bytes`
            // would also be 0 regardless of `len`; the cudarc empty
            // path handles both consistently.
            if bytes == 0 {
                let slice = unsafe {
                    self.device.inner().alloc::<T>(len).map_err(|e| {
                        XlogError::Kernel(format!("GPU allocation failed (zero-byte): {}", e))
                    })?
                };
                let (raw_ptr, sync) = DevicePtr::device_ptr(&slice, slice.stream());
                std::mem::forget(sync);
                self.publish_admission(bytes);
                return Ok(TrackedCudaSlice {
                    bytes,
                    manager: Arc::clone(self),
                    inner: ManuallyDrop::new(slice),
                    raw_ptr,
                    backing: Backing::Cudarc,
                });
            }

            // v0.6 path: route through the runtime resource stack.
            // Convert checked: `bytes` is u64 from
            // `len * size_of::<T>()`, and the runtime trait surface
            // uses `usize`. On 64-bit targets this is lossless; on
            // 32-bit a stray `bytes as usize` would silently
            // truncate and desync manager accounting (which still
            // tracks the full u64) from the runtime's view. Surface
            // the overflow as `XlogError::Kernel`; the caller retains or
            // rolls back the local reservation as appropriate.
            let bytes_usize = match usize::try_from(bytes) {
                Ok(v) => v,
                Err(_) => {
                    return Err(XlogError::Kernel(format!(
                        "GPU allocation size {} bytes exceeds platform usize",
                        bytes
                    )));
                }
            };
            let block_result = match runtime_reservation {
                Some(reservation) => {
                    reservation.allocate(bytes_usize, StreamId::DEFAULT, AllocTag::UNTAGGED)
                }
                None => runtime.allocate(bytes_usize, StreamId::DEFAULT, AllocTag::UNTAGGED),
            };
            let block = match block_result {
                Ok(b) => b,
                Err(e) => {
                    let prior_peak = self.accounting.peak.load(Ordering::SeqCst);
                    return Err(map_resource_error(e, prior_peak));
                }
            };
            self.publish_admission(bytes);
            let raw_ptr = block.ptr;
            // SAFETY: `block.ptr` is a live device pointer of size
            // `bytes` returned by the runtime; `len * size_of::<T>()`
            // == `bytes` by construction. The resulting CudaSlice<T>
            // is a typed view; the `Backing::Runtime` Drop branch
            // forgets it (via ManuallyDrop + no destructor call) so
            // cudarc never frees — the runtime's deallocate is the
            // sole free path.
            let typed_view = unsafe { self.device.inner().upgrade_device_ptr::<T>(raw_ptr, len) };
            return Ok(TrackedCudaSlice {
                bytes,
                manager: Arc::clone(self),
                inner: ManuallyDrop::new(typed_view),
                raw_ptr,
                backing: Backing::Runtime {
                    runtime: Arc::clone(runtime),
                    block: Some(block),
                },
            });
        }

        // Legacy path: cudarc allocator. SAFETY: budget reserved
        // atomically above and the device is valid; cudarc's
        // alloc returns properly aligned memory for type T.
        let slice = unsafe {
            self.device
                .inner()
                .alloc::<T>(len)
                .map_err(|e| XlogError::Kernel(format!("GPU allocation failed: {}", e)))?
        };
        let (raw_ptr, sync) = DevicePtr::device_ptr(&slice, slice.stream());
        std::mem::forget(sync);
        alloc_guard_insert(raw_ptr, bytes);
        self.publish_admission(bytes);

        // Debug probe (XLOG_DEBUG_POISON_ALLOC=1): poison fresh legacy
        // allocations with 0xDD so any read of unwritten allocation
        // contents becomes a deterministic, recognizable pattern
        // instead of whatever the recycled memory held. Diagnostic
        // only; off unless the env var is set.
        if poison_alloc_enabled() && bytes > 0 {
            unsafe {
                let _ = cudarc::driver::sys::cuMemsetD8Async(
                    raw_ptr,
                    0xDD,
                    bytes as usize,
                    std::ptr::null_mut(),
                );
            }
        }

        Ok(TrackedCudaSlice {
            bytes,
            manager: Arc::clone(self),
            inner: ManuallyDrop::new(slice),
            raw_ptr,
            backing: Backing::Cudarc,
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
        match self.alloc_raw_after_local_reservation(bytes, bytes_u64, tag, None) {
            Ok(allocation) => Ok(allocation),
            Err(error) => {
                self.rollback_local_reservation(bytes_u64)?;
                Err(error)
            }
        }
    }

    fn alloc_raw_after_local_reservation(
        self: &Arc<Self>,
        bytes: usize,
        bytes_u64: u64,
        tag: AllocTag,
        runtime_reservation: Option<&mut RuntimeMemoryReservation>,
    ) -> Result<RuntimeAllocBlock> {
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            XlogError::Kernel(
                "GpuMemoryManager::alloc_raw called without an attached XlogDeviceRuntime; \
                 construct via with_runtime to enable runtime routing"
                    .to_string(),
            )
        })?;

        #[cfg(test)]
        self.run_after_local_reservation_hook(bytes_u64);

        // Route through the runtime. Stream is the runtime's
        // default for now; once stream-aware kernel launches start
        // routing through alloc_raw the caller will pass an
        // explicit StreamId.
        let allocation = match runtime_reservation {
            Some(reservation) => reservation.allocate(bytes, StreamId::DEFAULT, tag),
            None => runtime.allocate(bytes, StreamId::DEFAULT, tag),
        };
        match allocation {
            Ok(block) => {
                self.publish_admission(bytes_u64);
                Ok(RuntimeAllocBlock {
                    bytes: bytes_u64,
                    manager: Arc::clone(self),
                    runtime: Arc::clone(runtime),
                    block: Some(block),
                })
            }
            Err(e) => {
                let prior_peak = self.accounting.peak.load(Ordering::SeqCst);
                Err(map_resource_error(e, prior_peak))
            }
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

/// Owned handle for a raw allocation routed through
/// [`GpuMemoryManager::alloc_raw`] / the v0.6 device runtime.
///
/// Manual `Debug` impl below — the runtime / manager handles
/// inside this struct are not `Debug`, so a derive would not
/// compile.
///
/// On drop, deallocates through the runtime (returning the bytes
/// to the runtime's bookkeeping — pending if the runtime's backend
/// is async) and decrements the manager's local `allocated`
/// counter. The block exposes only the raw device pointer and
/// byte length; typed views are the caller's responsibility (this
/// path is not yet wired into the typed `CudaSlice<T>` API — that
/// is a follow-up slice).
pub struct RuntimeAllocBlock {
    bytes: u64,
    manager: Arc<GpuMemoryManager>,
    runtime: Arc<XlogDeviceRuntime>,
    /// `None` after Drop fires; `Some(_)` while the block is live.
    /// Wrapped in Option so `Drop` can move the block out and pass
    /// it by value to `runtime.deallocate`.
    block: Option<DeviceBlock>,
}

impl RuntimeAllocBlock {
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
        if let Some(block) = self.block.take() {
            match self.runtime.deallocate(block) {
                Ok(()) => {
                    let release = self.manager.release_owned_allocation(self.bytes);
                    debug_assert!(release.is_ok(), "allocation release must be balanced");
                }
                Err(_) => self.manager.record_deallocation_failure(self.bytes),
            }
        }
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
    _tensor: DlpackManagedTensor,
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
    _import: Arc<ArrowDeviceImport>,
    /// Same role as [`DlpackColumn::source_slice`]: `Some` for
    /// xlog-owned Arrow device columns, `None` for true
    /// external Arrow producers.
    source_slice: Option<Arc<TrackedCudaSlice<u8>>>,
}

impl CudaColumn {
    pub fn owned(slice: TrackedCudaSlice<u8>) -> Self {
        Self::Owned(slice)
    }

    pub fn dlpack(
        ptr: cudarc::driver::sys::CUdeviceptr,
        len_bytes: usize,
        stream: Arc<CudaStream>,
        tensor: DlpackManagedTensor,
    ) -> Self {
        Self::Dlpack(DlpackColumn {
            ptr,
            len_bytes,
            stream,
            _tensor: tensor,
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
        Self::Dlpack(DlpackColumn {
            ptr,
            len_bytes,
            stream,
            _tensor: tensor,
            source_slice: Some(source_slice),
        })
    }

    pub fn arrow_device(
        ptr: cudarc::driver::sys::CUdeviceptr,
        len_bytes: usize,
        stream: Arc<CudaStream>,
        import: Arc<ArrowDeviceImport>,
    ) -> Self {
        Self::ArrowDevice(ArrowDeviceColumn {
            ptr,
            len_bytes,
            stream,
            _import: import,
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
        Self::ArrowDevice(ArrowDeviceColumn {
            ptr,
            len_bytes,
            stream,
            _import: import,
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

impl DevicePtr<u8> for CudaColumn {
    fn device_ptr<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (cudarc::driver::sys::CUdeviceptr, SyncOnDrop<'a>) {
        match self {
            CudaColumn::Owned(slice) => DevicePtr::device_ptr(slice, stream),
            CudaColumn::Dlpack(col) => (col.ptr, SyncOnDrop::Sync(None)),
            CudaColumn::ArrowDevice(col) => (col.ptr, SyncOnDrop::Sync(None)),
        }
    }
}

impl DevicePtrMut<u8> for CudaColumn {
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (cudarc::driver::sys::CUdeviceptr, SyncOnDrop<'a>) {
        match self {
            CudaColumn::Owned(slice) => DevicePtrMut::device_ptr_mut(slice, stream),
            CudaColumn::Dlpack(col) => (col.ptr, SyncOnDrop::Sync(None)),
            CudaColumn::ArrowDevice(col) => (col.ptr, SyncOnDrop::Sync(None)),
        }
    }
}

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
mod tests {
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
        fn allocate(
            &self,
            bytes: usize,
            stream: StreamId,
            tag: AllocTag,
        ) -> ResourceResult<DeviceBlock> {
            if self.fail_next.swap(false, Ordering::SeqCst) {
                return Err(ResourceError::Driver(
                    "injected allocation failure".to_string(),
                ));
            }
            self.inner.allocate(bytes, stream, tag)
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
        fn allocate(
            &self,
            bytes: usize,
            stream: StreamId,
            tag: AllocTag,
        ) -> ResourceResult<DeviceBlock> {
            self.inner.allocate(bytes, stream, tag)
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
    fn typed_deallocation_failure_retains_local_charge_and_records_failure() {
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
        assert_eq!(manager.allocated_bytes(), 1024);
        assert_eq!(manager.remaining_bytes(), 3072);
        assert_eq!(manager.deallocation_failure_count(), 1);
        assert_eq!(manager.deallocation_failure_bytes(), 1024);
        assert!(manager.reset_tracking().is_err());
    }

    #[test]
    fn raw_deallocation_failure_retains_local_charge_and_records_failure() {
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
        assert_eq!(manager.allocated_bytes(), 512);
        assert_eq!(manager.remaining_bytes(), 3584);
        assert_eq!(manager.deallocation_failure_count(), 1);
        assert_eq!(manager.deallocation_failure_bytes(), 512);
        assert!(manager.reset_tracking().is_err());
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
    ///
    /// Uses a null-pointer `DlpackManagedTensor` purely as a
    /// drop-safe placeholder — the recorder never derefs the
    /// tensor, only the source slice.
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
        // SAFETY: null-pointer DlpackManagedTensor is drop-safe
        // (the Drop impl checks for null before invoking the
        // deleter). Acceptable for a unit fixture that exercises
        // identity propagation, not the tensor lifecycle.
        let tensor = unsafe { DlpackManagedTensor::from_raw(std::ptr::null_mut()) };
        let col = CudaColumn::dlpack_xlog_owned(Arc::new(slice), stream, tensor);
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
        let tensor = unsafe { DlpackManagedTensor::from_raw(std::ptr::null_mut()) };
        let col = CudaColumn::dlpack_xlog_owned(Arc::new(slice), stream, tensor);
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
        let tensor = unsafe { DlpackManagedTensor::from_raw(std::ptr::null_mut()) };
        // Bogus ptr/len — never dereferenced in this unit test
        // (we only inspect the column metadata).
        let col = CudaColumn::dlpack(0, 0, stream, tensor);
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
        let col = CudaColumn::arrow_device(0, 0, stream, import);
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
