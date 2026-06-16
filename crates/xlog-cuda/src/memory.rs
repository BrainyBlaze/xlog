//! CUDA memory management
//!
//! This module provides GPU memory management with budget enforcement.
//! It wraps cudarc's allocation functions and tracks total allocated memory.

use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceSlice, SyncOnDrop};
use xlog_core::{MemoryBudget, Result, Schema, XlogError};

use crate::arrow_device::ArrowDeviceImport;
use crate::cuda_compat::{AsKernelParam, DeviceParamStorage, IntoKernelParamStorage};
use crate::device_runtime::{AllocTag, DeviceBlock, ResourceError, StreamId, XlogDeviceRuntime};
use crate::dlpack::DlpackManagedTensor;
use crate::CudaDevice;

/// GPU memory manager with budget enforcement
///
/// Tracks allocated GPU memory and enforces a memory budget.
/// When the budget would be exceeded, returns `XlogError::ResourceExhausted`.
///
/// # v0.6 device-runtime routing (opt-in)
///
/// Constructing via [`GpuMemoryManager::with_runtime`] attaches an
/// [`XlogDeviceRuntime`] that mediates allocations through the v0.6
/// resource stack (e.g., `GlobalDeviceBudget` → `LoggingResource` →
/// `AsyncCudaResource`). When attached:
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
/// When the manager is constructed via [`GpuMemoryManager::new`]
/// (no runtime attached), `alloc::<T>` and the rest of the public
/// API behave bit-for-bit identically to pre-migration: cudarc's
/// `device.alloc::<T>(len)` allocates and `cudarc` frees on drop.
/// `alloc_raw` returns `XlogError::Kernel` when no runtime is
/// attached (no silent fallback). `CudaKernelProvider::new`
/// continues to construct the manager via `new` for now;
/// runtime-routed providers are an opt-in through `with_runtime`
/// at construction sites that need it.
pub struct GpuMemoryManager {
    /// The CUDA device for memory operations
    device: Arc<CudaDevice>,
    /// Memory budget configuration
    budget: MemoryBudget,
    /// Currently allocated bytes (tracked atomically for thread safety)
    allocated: AtomicU64,
    /// High-water mark of `allocated` since construction or the last
    /// [`reset_peak`](Self::reset_peak). Updated at the two reservation
    /// funnels (`alloc`, `alloc_raw`); used by measurement harnesses to
    /// report true peak device-memory pressure across an operation.
    peak: AtomicU64,
    /// Count of `alloc` calls (device allocation requests). Resettable; used by
    /// the GPU-resident MC engine's no-host gate to prove that **zero** device
    /// allocations happen inside the measured region (all arenas are allocated
    /// before it). Distinct from `allocated` (bytes).
    alloc_count: AtomicU64,
    /// Optional v0.6 device runtime. When set, [`alloc_raw`]
    /// reserves through the runtime's resource stack in addition
    /// to enforcing the local budget; both must accept for the
    /// allocation to proceed.
    runtime: Option<Arc<XlogDeviceRuntime>>,
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
fn poison_free_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("XLOG_DEBUG_POISON_FREE").map(|v| v == "1") == Ok(true))
}

/// Debug probe: poison fresh legacy allocations with 0xDD so reads of
/// unwritten contents surface deterministically. Gated on
/// `XLOG_DEBUG_POISON_ALLOC=1`, read once per process.
fn poison_alloc_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("XLOG_DEBUG_POISON_ALLOC").map(|v| v == "1") == Ok(true))
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
            if std::env::var("XLOG_DEBUG_ALLOC_GUARD").map(|v| v == "1") == Ok(true) {
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
        match &mut self.backing {
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
            }
            Backing::Runtime { runtime, block } => {
                // Runtime owns the underlying memory. Tell it to
                // deallocate the block; the inner `CudaSlice<T>` is
                // a typed view that must NOT free on its own,
                // which `ManuallyDrop` ensures by simply not
                // calling its destructor here.
                if let Some(block) = block.take() {
                    let _ = runtime.deallocate(block);
                }
            }
        }
        self.manager.record_free(self.bytes);
    }
}

impl GpuMemoryManager {
    /// Create a new GPU memory manager
    ///
    /// # Arguments
    /// * `device` - The CUDA device to allocate memory on
    /// * `budget` - Memory budget configuration
    pub fn new(device: Arc<CudaDevice>, budget: MemoryBudget) -> Self {
        Self {
            device,
            budget,
            allocated: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            alloc_count: AtomicU64::new(0),
            runtime: None,
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
    pub fn with_runtime(
        device: Arc<CudaDevice>,
        budget: MemoryBudget,
        runtime: Arc<XlogDeviceRuntime>,
    ) -> Self {
        Self {
            device,
            budget,
            allocated: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            alloc_count: AtomicU64::new(0),
            runtime: Some(runtime),
        }
    }

    /// Borrow the attached device runtime, if any. `None` when the
    /// manager was constructed via [`new`]. Test/diagnostic
    /// accessor; production call sites that need the runtime own
    /// it directly.
    pub fn runtime(&self) -> Option<&Arc<XlogDeviceRuntime>> {
        self.runtime.as_ref()
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
        self.alloc_count.fetch_add(1, Ordering::Relaxed);

        // Fix Issue 2: Use checked_mul to prevent integer overflow before cast
        let bytes = (len as u64)
            .checked_mul(std::mem::size_of::<T>() as u64)
            .ok_or_else(|| XlogError::Kernel("Allocation size overflow".to_string()))?;

        // Fix Issue 1: Use compare_exchange loop to prevent TOCTOU race condition
        // Two threads could both pass check_budget() but exceed budget together
        loop {
            let current = self.allocated.load(Ordering::SeqCst);
            let new_val = current.saturating_add(bytes);
            if new_val > self.budget.device_bytes {
                return Err(XlogError::ResourceExhausted {
                    context: "GPU memory allocation".to_string(),
                    estimated_bytes: bytes,
                    budget_bytes: self.budget.device_bytes,
                });
            }
            if self
                .allocated
                .compare_exchange(current, new_val, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                self.peak.fetch_max(new_val, Ordering::SeqCst);
                break;
            }
        }

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
                        self.allocated.fetch_sub(bytes, Ordering::SeqCst);
                        XlogError::Kernel(format!("GPU allocation failed (zero-byte): {}", e))
                    })?
                };
                let (raw_ptr, sync) = DevicePtr::device_ptr(&slice, slice.stream());
                std::mem::forget(sync);
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
            // the overflow as `XlogError::Kernel` and roll back the
            // local reservation so the manager stays consistent.
            let bytes_usize = match usize::try_from(bytes) {
                Ok(v) => v,
                Err(_) => {
                    self.allocated.fetch_sub(bytes, Ordering::SeqCst);
                    return Err(XlogError::Kernel(format!(
                        "GPU allocation size {} bytes exceeds platform usize",
                        bytes
                    )));
                }
            };
            let block = match runtime.allocate(bytes_usize, StreamId::DEFAULT, AllocTag::UNTAGGED) {
                Ok(b) => b,
                Err(e) => {
                    self.allocated.fetch_sub(bytes, Ordering::SeqCst);
                    return Err(map_resource_error(e));
                }
            };
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
            self.device.inner().alloc::<T>(len).map_err(|e| {
                // Rollback the allocation tracking if CUDA allocation fails
                self.allocated.fetch_sub(bytes, Ordering::SeqCst);
                XlogError::Kernel(format!("GPU allocation failed: {}", e))
            })?
        };
        let (raw_ptr, sync) = DevicePtr::device_ptr(&slice, slice.stream());
        std::mem::forget(sync);
        alloc_guard_insert(raw_ptr, bytes);

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
        let current = self.allocated.load(Ordering::SeqCst);
        let proposed = current.saturating_add(bytes);

        if proposed > self.budget.device_bytes {
            return Err(XlogError::ResourceExhausted {
                context: "GPU memory allocation".to_string(),
                estimated_bytes: bytes,
                budget_bytes: self.budget.device_bytes,
            });
        }

        Ok(())
    }

    /// Get the current allocated memory in bytes
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated.load(Ordering::SeqCst)
    }

    /// High-water mark of allocated bytes since construction or the
    /// last [`reset_peak`](Self::reset_peak). Always ≥
    /// [`allocated_bytes`](Self::allocated_bytes) at the moment it was
    /// recorded. Measurement-harness API (S3 peak-memory gate).
    pub fn peak_bytes(&self) -> u64 {
        self.peak.load(Ordering::SeqCst)
    }

    /// Reset the peak high-water mark to the *current* allocated
    /// level, so a measurement window starts from live state rather
    /// than zero. Measurement-harness API.
    pub fn reset_peak(&self) {
        self.peak
            .store(self.allocated.load(Ordering::SeqCst), Ordering::SeqCst);
    }

    /// Number of `alloc` calls issued so far (device allocation requests).
    /// The GPU-resident MC engine snapshots this around the measured region to
    /// prove `per_operator_host_allocations == 0` (all arenas pre-allocated).
    pub fn alloc_count(&self) -> u64 {
        self.alloc_count.load(Ordering::Relaxed)
    }

    /// Reset the allocation-request counter to zero.
    pub fn reset_alloc_count(&self) {
        self.alloc_count.store(0, Ordering::Relaxed);
    }

    /// Get the memory budget
    pub fn budget(&self) -> &MemoryBudget {
        &self.budget
    }

    /// Get the underlying CUDA device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Record that memory has been freed
    ///
    /// Note: cudarc automatically frees memory when CudaSlice is dropped.
    /// This method should be called to update tracking when memory is freed.
    pub fn record_free(&self, bytes: u64) {
        self.allocated.fetch_sub(bytes, Ordering::SeqCst);
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
    /// * `XlogError::Kernel` (with the resource error rendered)
    ///   if the runtime rejects the request — including the
    ///   runtime's own `OutOfBudget`, which is mapped here so
    ///   callers see a single error surface.
    pub fn alloc_raw(self: &Arc<Self>, bytes: usize, tag: AllocTag) -> Result<RuntimeAllocBlock> {
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            XlogError::Kernel(
                "GpuMemoryManager::alloc_raw called without an attached XlogDeviceRuntime; \
                 construct via with_runtime to enable runtime routing"
                    .to_string(),
            )
        })?;

        let bytes_u64 = bytes as u64;

        // Reserve against the local budget first (preserves the
        // pre-existing semantics for callers that mix alloc and
        // alloc_raw under a single MemoryBudget).
        loop {
            let current = self.allocated.load(Ordering::SeqCst);
            let new_val = current.saturating_add(bytes_u64);
            if new_val > self.budget.device_bytes {
                return Err(XlogError::ResourceExhausted {
                    context: "GPU memory allocation (runtime path)".to_string(),
                    estimated_bytes: bytes_u64,
                    budget_bytes: self.budget.device_bytes,
                });
            }
            if self
                .allocated
                .compare_exchange(current, new_val, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                self.peak.fetch_max(new_val, Ordering::SeqCst);
                break;
            }
        }

        // Route through the runtime. Stream is the runtime's
        // default for now; once stream-aware kernel launches start
        // routing through alloc_raw the caller will pass an
        // explicit StreamId.
        match runtime.allocate(bytes, StreamId::DEFAULT, tag) {
            Ok(block) => Ok(RuntimeAllocBlock {
                bytes: bytes_u64,
                manager: Arc::clone(self),
                runtime: Arc::clone(runtime),
                block: Some(block),
            }),
            Err(e) => {
                // Roll back local reservation; runtime did not
                // accept the bytes.
                self.allocated.fetch_sub(bytes_u64, Ordering::SeqCst);
                Err(map_resource_error(e))
            }
        }
    }

    /// Get remaining budget in bytes
    pub fn remaining_bytes(&self) -> u64 {
        let allocated = self.allocated.load(Ordering::SeqCst);
        self.budget.device_bytes.saturating_sub(allocated)
    }

    /// Reset allocation tracking
    ///
    /// This should be called when GPU memory has been freed but the tracker
    /// hasn't been updated (e.g., when CudaSlice instances are dropped without
    /// calling record_free). This is a temporary workaround until proper
    /// RAII-based tracking is implemented.
    pub fn reset_tracking(&self) {
        self.allocated.store(0, Ordering::SeqCst);
        self.peak.store(0, Ordering::SeqCst);
    }
}

fn map_resource_error(e: ResourceError) -> XlogError {
    match e {
        ResourceError::OutOfBudget {
            requested,
            remaining,
        } => XlogError::ResourceExhausted {
            context: format!(
                "device-runtime budget refused allocation ({} bytes, {} remaining)",
                requested, remaining
            ),
            estimated_bytes: requested as u64,
            budget_bytes: (requested + remaining) as u64,
        },
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
            // Runtime deallocate may queue an async free (see
            // AsyncCudaResource); the local manager counter
            // releases immediately because the block.bytes are
            // no longer "live from the manager's perspective".
            // Runtime-side bookkeeping converges after
            // `runtime.reap_pending()`.
            let _ = self.runtime.deallocate(block);
            self.manager
                .allocated
                .fetch_sub(self.bytes, Ordering::SeqCst);
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
    pub columns: Vec<CudaColumn>,
    /// Row capacity for allocated columns
    pub row_cap: u64,
    /// Device-resident row count (len = 1)
    pub d_num_rows: TrackedCudaSlice<u32>,
    /// Schema describing the column types
    pub schema: Schema,
    /// Cached host-side row count (u32::MAX = not yet cached).
    /// Avoids repeated synchronous D2H transfers for the immutable row count.
    cached_row_count: AtomicU32,
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
    pub fn set_cached_row_count_if_unset(&self, count: u32) {
        let _ = self.cached_row_count.compare_exchange(
            u32::MAX,
            count,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
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
    use xlog_core::ScalarType;

    fn try_device() -> Option<Arc<CudaDevice>> {
        match CudaDevice::new(0) {
            Ok(d) => Some(Arc::new(d)),
            Err(e) => {
                eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
                None
            }
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
