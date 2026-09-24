//! CUDA device management
//!
//! This module keeps XLOG's historical single-stream device abstraction while
//! targeting cudarc's newer CUDA 13-capable context/stream APIs.
//!
//! Loading/replacing modules and releasing their last owners can wait for
//! execution-event completion, directly or through deferred capture cleanup.
//! Callers must satisfy the [cold-lifecycle progress contract](crate::cuda_graph).

use std::collections::BTreeMap;
use std::ffi::{c_void, CString};
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

use cudarc::driver::result::{self, DriverError};
use cudarc::driver::{
    sys, CudaContext as CudarcContext, CudaEvent, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    DeviceRepr, HostSlice, LaunchConfig, ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;
use sha2::{Digest, Sha256};
use xlog_core::{Result, XlogError};

use crate::device_runtime::{Access, ResourceResult};
use crate::memory::{with_memory_access, DeviceMemoryView, DeviceRead, DeviceWrite};

#[derive(Clone, Copy, Debug, Default)]
enum HostTransferCompletion {
    #[default]
    Complete,
    Unfenced,
    EventRecorded,
}

impl HostTransferCompletion {
    fn submit<E>(
        &mut self,
        transfer: impl FnOnce() -> std::result::Result<(), E>,
        mut record: impl FnMut() -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        *self = Self::Unfenced;
        // Keep the original stream/frame alive while fencing even a failed or
        // unwinding submission. Never replace its error with cleanup failure.
        let submitted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(transfer));
        let recorded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(&mut record));
        if matches!(recorded, Ok(Ok(()))) {
            *self = Self::EventRecorded;
        } else if matches!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(&mut record)),
            Ok(Ok(()))
        ) {
            // A transient record error may be repaired here, but the original
            // operation still fails. No later thread guesses a PTDS identity.
            *self = Self::EventRecorded;
        }
        match submitted {
            Err(panic) => std::panic::resume_unwind(panic),
            Ok(Err(error)) => Err(error),
            Ok(Ok(())) => match recorded {
                Ok(result) => result,
                Err(panic) => std::panic::resume_unwind(panic),
            },
        }
    }

    fn wait<E>(
        &mut self,
        event: impl FnOnce() -> std::result::Result<(), E>,
        refence: impl FnOnce() -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        match self {
            Self::Complete => return Ok(()),
            Self::Unfenced => {
                refence()?;
                *self = Self::EventRecorded;
            }
            Self::EventRecorded => {}
        }
        event()?;
        *self = Self::Complete;
        Ok(())
    }
}

/// One page-locked host allocation, shared by synchronous staging and terminal
/// receipt owners. An uncertain copy retains these bytes through cold retirement.
/// No caller-owned host address is ever placed in that retirement queue.
#[derive(Debug)]
pub(crate) struct PinnedHostBuffer {
    ptr: usize,
    bytes: usize,
    context: Arc<CudarcContext>,
    event: Option<CudaEvent>,
    completion: HostTransferCompletion,
}

impl PinnedHostBuffer {
    pub(crate) fn new(stream: &CudaStream, bytes: usize) -> std::result::Result<Self, DriverError> {
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(stream)?;
        let context = Arc::clone(stream.context());
        let event = context.new_event(None)?;
        context.bind_to_thread()?;
        let mut ptr = std::ptr::null_mut();
        // SAFETY: all fallible prerequisites precede allocation. The output is
        // immediately owned; zero-length logical buffers still own one byte.
        unsafe { sys::cuMemHostAlloc(&mut ptr, bytes.max(1), 0).result()? };
        Ok(Self {
            ptr: ptr as usize,
            bytes,
            context,
            event: Some(event),
            completion: HostTransferCompletion::Complete,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.bytes
    }

    pub(crate) fn copy_from_device(
        &mut self,
        source: &DeviceMemoryView<u8>,
        stream: &Arc<CudaStream>,
    ) -> ResourceResult<Vec<u8>> {
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(stream)?;
        if source.len() != self.bytes {
            return Err(crate::device_runtime::ResourceError::StreamMisuse(
                "pinned host destination size does not match device source".into(),
            ));
        }
        let bytes = self.bytes;
        with_memory_access(
            Arc::clone(stream),
            vec![source.access(Access::Read)?],
            |proof| {
                let source = proof.read(source)?;
                let (ptr, _guard) = source.device_ptr(stream);
                // SAFETY: the read admission retains the entire device allocation;
                // this owner retains the host destination even if abort cannot wait.
                unsafe {
                    self.enqueue(stream, |host| {
                        sys::cuMemcpyDtoHAsync_v2(host.cast(), ptr, bytes, stream.cu_stream())
                            .result()
                    })?;
                }
                Ok(self.wait()?)
            },
        )?;
        // SAFETY: the admitted byte source initialized the complete destination.
        Ok(unsafe { self.read_vec(bytes)? })
    }

    /// Enqueue a whole transfer batch, recording its fence on the submitting
    /// thread. Even a per-thread stream can then retire on a different thread.
    ///
    /// # Safety
    /// The closure must access only this allocation's `len()` host bytes. Every
    /// device range must have an admitted, retained owner for the whole batch.
    pub(crate) unsafe fn enqueue(
        &mut self,
        stream: &CudaStream,
        transfer: impl FnOnce(*mut u8) -> std::result::Result<(), DriverError>,
    ) -> std::result::Result<(), DriverError> {
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(stream)?;
        if stream.context().cu_ctx() != self.context.cu_ctx() {
            return Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_CONTEXT));
        }
        self.wait()?;
        self.context.bind_to_thread()?;
        let event = self.event.as_ref().expect("pinned owner event present");
        self.completion
            .submit(|| transfer(self.ptr as *mut u8), || event.record(stream))
    }

    pub(crate) fn wait(&mut self) -> std::result::Result<(), DriverError> {
        let _ordinary = crate::cuda_graph::reserve_capture_exclusion()?;
        self.completion.wait(
            || {
                self.event
                    .as_ref()
                    .expect("pinned owner event present")
                    .synchronize()
            },
            // Both original-frame record attempts failed. An old event or a
            // different thread's PTDS cannot prove this prefix complete. A
            // context fence could invalidate an uncatalogued external capture.
            || Err(DriverError(sys::CUresult::CUDA_ERROR_UNKNOWN)),
        )
    }

    fn write<T: DeviceRepr>(&mut self, source: &[T]) -> std::result::Result<(), DriverError> {
        if std::mem::size_of_val(source) != self.bytes {
            return Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE));
        }
        self.wait()?;
        // SAFETY: the source is initialized, destination is an equally sized
        // uniquely owned allocation, and previous DMA has completed.
        unsafe {
            std::ptr::copy_nonoverlapping(
                source.as_ptr().cast::<u8>(),
                self.ptr as *mut u8,
                self.bytes,
            );
        }
        Ok(())
    }

    /// # Safety
    /// A completed transfer must have initialized all bytes as `len` valid T
    /// values. Allocation alone does not initialize this host buffer.
    pub(crate) unsafe fn read_vec<T: DeviceRepr>(
        &mut self,
        len: usize,
    ) -> std::result::Result<Vec<T>, DriverError> {
        if len.checked_mul(std::mem::size_of::<T>()) != Some(self.bytes) {
            return Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE));
        }
        self.wait()?;
        let mut values = Vec::<T>::with_capacity(len);
        // SAFETY: caller proves initialization/type; Vec supplies alignment.
        // Publish length only after the infallible byte copy, never before DMA.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.ptr as *const u8,
                values.as_mut_ptr().cast::<u8>(),
                self.bytes,
            );
            values.set_len(len);
        }
        Ok(values)
    }

    /// # Safety
    /// A completed copy must have initialized these bytes as valid `T` values.
    unsafe fn read_into<T: DeviceRepr>(
        &mut self,
        dst: &mut [T],
    ) -> std::result::Result<(), DriverError> {
        if std::mem::size_of_val(dst) != self.bytes {
            return Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE));
        }
        self.wait()?;
        // SAFETY: caller proves the copied representation; the host destination
        // is exclusively borrowed and cannot be accessed by this owner's DMA.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.ptr as *const u8,
                dst.as_mut_ptr().cast::<u8>(),
                self.bytes,
            );
        }
        Ok(())
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        let resources = (
            self.ptr,
            Arc::clone(&self.context),
            self.event.take().expect("pinned owner event present"),
            self.completion,
        );
        crate::cuda_graph::retire_resources_after_completion(
            resources,
            |(_, context, event, completion)| {
                context.bind_to_thread()?;
                let mut completion = *completion;
                completion.wait(
                    || event.synchronize(),
                    || Err(DriverError(sys::CUresult::CUDA_ERROR_UNKNOWN)),
                )
            },
            |(ptr, _, _, _)| {
                // SAFETY: the entire copy batch is complete. The canonical
                // retirement helper never resubmits an unknown free outcome.
                unsafe { sys::cuMemFreeHost(*ptr as *mut c_void).result() }
            },
            |_| Ok(()),
            |_, error| eprintln!("CUDA pinned host allocation retirement incomplete: {error}"),
        );
    }
}

#[cfg(test)]
mod host_transfer_tests {
    use super::*;

    #[test]
    fn execution_retirement_observes_successful_abort_after_failed_event_records() {
        use crate::cuda_graph::{reap_capture_retirements, retire_resources_after_completion};
        use crate::launch::{LaunchEnqueueError, RecorderTransaction};
        use crate::memory::OperationCompletion;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let _serial = crate::cuda_graph::capture_lifecycle_test_guard();
        let proof = Arc::new(OperationCompletion::new(17));
        let fence = Arc::new(Mutex::new(ExecutionFence::default()));
        let payload = Arc::new(());
        let payload_weak = Arc::downgrade(&payload);
        let destroys = Arc::new(AtomicUsize::new(0));
        let mut transaction =
            RecorderTransaction::from_admitted(Arc::clone(&proof), Box::<[u8]>::default());
        let launches = AtomicUsize::new(0);
        let records = AtomicUsize::new(0);
        let error = DriverError(sys::CUresult::CUDA_ERROR_UNKNOWN);
        let result = transaction.enqueue_operation_with(
            || Ok(()),
            |proof| proof.validate_submission(17),
            |_| {
                let submitted = fence.lock().unwrap().submit(
                    Some(Arc::clone(&proof)),
                    || {
                        launches.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    },
                    || {
                        records.fetch_add(1, Ordering::SeqCst);
                        Err(error)
                    },
                );
                let observed = Arc::clone(&fence);
                let destroyed = Arc::clone(&destroys);
                retire_resources_after_completion(
                    payload,
                    move |_| {
                        observed
                            .lock()
                            .unwrap()
                            .wait(|| panic!("unrecorded event is not proof"))
                    },
                    move |_| {
                        destroyed.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    },
                    |_| Ok(()),
                    |_, _: DriverError| panic!("completion failure must remain retryable"),
                );
                assert!(payload_weak.upgrade().is_some());
                assert_eq!(destroys.load(Ordering::SeqCst), 0);
                submitted
            },
            |proof| proof.synchronize_with(17, || Ok(())),
            |_, _| Ok(()),
        );
        assert!(
            matches!(result, Err(LaunchEnqueueError::Operation(observed)) if observed == error)
        );
        assert!(proof.is_complete());
        reap_capture_retirements();
        assert!(payload_weak.upgrade().is_none());
        assert_eq!(destroys.load(Ordering::SeqCst), 1);
        reap_capture_retirements();
        assert_eq!(destroys.load(Ordering::SeqCst), 1);
        assert_eq!(launches.load(Ordering::SeqCst), 1);
        assert_eq!(records.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn execution_fence_replaces_previous_operation_confirmation() {
        use crate::memory::OperationCompletion;
        let old = Arc::new(OperationCompletion::new(17));
        let new = Arc::new(OperationCompletion::new(17));
        let mut fence = ExecutionFence::default();
        fence
            .submit(Some(Arc::clone(&old)), || Ok(()), || Ok(()))
            .unwrap();
        old.synchronize_with(17, || Ok(())).unwrap();
        let error = DriverError(sys::CUresult::CUDA_ERROR_UNKNOWN);
        assert_eq!(
            fence.submit(Some(Arc::clone(&new)), || Ok(()), || Err(error)),
            Err(error)
        );
        assert_eq!(
            fence.wait(|| panic!("old event cannot confirm the new execution")),
            Err(error)
        );
        new.synchronize_with(17, || Ok(())).unwrap();
        fence
            .wait(|| panic!("positive exact confirmation does not need the event"))
            .unwrap();
        assert_eq!(fence.submit(None, || Ok(()), || Err(error)), Err(error));
        assert_eq!(
            fence.wait(|| panic!("borrowed launch cannot reuse an earlier admission's proof")),
            Err(error)
        );
    }

    #[test]
    fn execution_fence_retries_a_recorded_event_without_relaunch() {
        let mut fence = ExecutionFence::default();
        let error = DriverError(sys::CUresult::CUDA_ERROR_UNKNOWN);
        let mut records = 0;
        assert_eq!(
            fence.submit(
                None,
                || Ok(()),
                || {
                    records += 1;
                    if records == 1 {
                        Err(error)
                    } else {
                        Ok(())
                    }
                }
            ),
            Err(error)
        );
        assert_eq!(fence.wait(|| Err(error)), Err(error));
        fence.wait(|| Ok(())).unwrap();
        assert_eq!(records, 2);
    }

    #[test]
    fn failed_host_submission_records_the_original_prefix_before_returning_error() {
        let mut completion = HostTransferCompletion::default();
        let result = completion.submit(|| Err("original transfer error"), || Ok(()));
        assert_eq!(result, Err("original transfer error"));
        assert!(matches!(completion, HostTransferCompletion::EventRecorded));
    }

    #[test]
    fn failed_event_record_is_repaired_without_erasing_its_error() {
        let mut completion = HostTransferCompletion::default();
        let mut attempts = 0;
        let result = completion.submit(
            || Ok(()),
            || {
                attempts += 1;
                if attempts == 1 {
                    Err("record failed")
                } else {
                    Ok(())
                }
            },
        );
        assert_eq!(result, Err("record failed"));
        assert_eq!(attempts, 2);
        assert!(matches!(completion, HostTransferCompletion::EventRecorded));
        assert!(completion
            .wait(|| Ok::<(), &str>(()), || panic!("already fenced"))
            .is_ok());
    }

    #[test]
    fn submission_unwind_preserves_original_panic_when_both_fences_fail() {
        let mut completion = HostTransferCompletion::default();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: std::result::Result<(), &str> = completion.submit(
                || panic!("original submission panic"),
                || panic!("cleanup panic"),
            );
        }));
        assert_eq!(
            *result.unwrap_err().downcast::<&str>().unwrap(),
            "original submission panic"
        );
        assert!(matches!(completion, HostTransferCompletion::Unfenced));
        assert!(completion
            .wait(|| panic!("no valid fence"), || Err("unfenced"))
            .is_err());
    }

    #[test]
    fn uncertain_host_transfer_requires_completion_before_owner_release() {
        for fail_record in [false, true] {
            let mut completion = HostTransferCompletion::default();
            let result = completion.submit(
                || {
                    if fail_record {
                        Ok(())
                    } else {
                        Err("copy failed")
                    }
                },
                || Err("event record failed"),
            );
            assert!(result.is_err());
            assert!(completion
                .wait(|| panic!("no recorded fence"), || Err("wait failed"))
                .is_err());
            assert!(matches!(completion, HostTransferCompletion::Unfenced));
            completion.wait(|| Ok(()), || Ok::<(), &str>(())).unwrap();
            assert!(matches!(completion, HostTransferCompletion::Complete));
        }
    }

    #[test]
    fn recorded_host_transfer_retries_its_exact_event_only() {
        let mut completion = HostTransferCompletion::default();
        completion.submit(|| Ok::<(), &str>(()), || Ok(())).unwrap();
        assert!(completion
            .wait(|| Err("event pending"), || panic!("recorded event exists"))
            .is_err());
        completion
            .wait(|| Ok::<(), &str>(()), || panic!("recorded event exists"))
            .unwrap();
        completion
            .wait(|| panic!("already complete"), || Err("already complete"))
            .unwrap();
    }

    #[test]
    #[ignore = "requires authorized CUDA execution"]
    fn pinned_host_buffer_retires_real_allocation_on_another_thread_after_binding_failure() {
        let _capture_test = crate::cuda_graph::capture_lifecycle_test_guard();
        let device = CudaDevice::new(0).unwrap();
        let context = Arc::clone(device.inner().stream().context());
        let stream = context.new_stream().unwrap();
        let source = device.inner().htod_sync_copy(&[0xa5_u8; 32]).unwrap();
        let mut buffer = PinnedHostBuffer::new(&stream, 32).unwrap();
        with_memory_access(
            Arc::clone(&stream),
            vec![source.access(Access::Read).unwrap()],
            |proof| {
                let source = proof.read(&source)?;
                let (ptr, _guard) = source.device_ptr(&stream);
                // SAFETY: the real source and owned pinned bank span exactly 32
                // initialized bytes and both live through this test's proven fence.
                unsafe {
                    buffer.enqueue(&stream, |host| {
                        sys::cuMemcpyDtoHAsync_v2(host.cast(), ptr, 32, stream.cu_stream()).result()
                    })?;
                }
                Ok(())
            },
        )
        .unwrap();
        // Establish real completion before injecting a retirement prerequisite
        // failure. Keep EventRecorded so production Drop still waits its event.
        buffer.event.as_ref().unwrap().synchronize().unwrap();
        assert_eq!(
            unsafe { std::slice::from_raw_parts(buffer.ptr as *const u8, 32) },
            &[0xa5; 32]
        );
        drop(source);
        drop(stream);
        drop(device);
        crate::cuda_graph::reap_capture_retirements();
        let retained = Arc::downgrade(&context);
        context.record_err::<()>(Err(DriverError(sys::CUresult::CUDA_ERROR_UNKNOWN)));
        drop(context);
        drop(buffer);
        assert!(
            retained.upgrade().is_some(),
            "failed binding lost the actual pinned owner"
        );
        std::thread::spawn(crate::cuda_graph::reap_capture_retirements)
            .join()
            .unwrap();
        assert!(
            retained.upgrade().is_none(),
            "completed retirement kept the event/context alive"
        );
        crate::cuda_graph::reap_capture_retirements();
        assert!(retained.upgrade().is_none());
    }

    #[test]
    fn unwinding_host_transfer_cannot_mark_its_storage_complete() {
        let mut completion = HostTransferCompletion::default();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = completion.submit(|| panic!("copy interrupted"), || Ok::<(), &str>(()));
        }));
        assert!(result.is_err());
        assert!(matches!(completion, HostTransferCompletion::EventRecorded));
    }
}

#[derive(Debug)]
pub(crate) struct LoadedModule {
    cu_module: sys::CUmodule,
    functions: BTreeMap<Arc<str>, sys::CUfunction>,
    module_name: Arc<str>,
    artifact_name: Arc<str>,
    artifact_sha256: Option<[u8; 32]>,
    context: Arc<CudarcContext>,
    completion: Mutex<ExecutionCompletion>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CapturedCudaLaunchBinding {
    pub(crate) module_name: Arc<str>,
    pub(crate) artifact_name: Arc<str>,
    pub(crate) artifact_sha256: Option<[u8; 32]>,
    pub(crate) function_name: Arc<str>,
    pub(crate) grid_dim: (u32, u32, u32),
    pub(crate) block_dim: (u32, u32, u32),
    pub(crate) shared_mem_bytes: u32,
    pub(crate) parameter_count: u64,
    pub(crate) cooperative: bool,
}

/// Completion events belong to the submitted executable, not its stream wrapper.
/// A failed submission/record is kept uncertain even if an older event completes.
/// Every recorded event includes the entire preceding stream and its transitive
/// dependencies, not only work using this executable.
#[derive(Debug, Default)]
struct ExecutionFence {
    state: HostTransferCompletion,
    confirmation: Option<Arc<crate::memory::OperationCompletion>>,
}

impl ExecutionFence {
    fn submit(
        &mut self,
        confirmation: Option<Arc<crate::memory::OperationCompletion>>,
        launch: impl FnOnce() -> std::result::Result<(), DriverError>,
        record: impl FnMut() -> std::result::Result<(), DriverError>,
    ) -> std::result::Result<(), DriverError> {
        // Replace the old admission's proof before any new work can be queued.
        self.confirmation = confirmation;
        self.state.submit(launch, record)
    }

    fn wait(
        &self,
        event: impl FnOnce() -> std::result::Result<(), DriverError>,
    ) -> std::result::Result<(), DriverError> {
        if self
            .confirmation
            .as_ref()
            .is_some_and(|proof| proof.is_complete())
        {
            // Abort has positively waited for this exact admitted execution,
            // even if neither private event-record attempt succeeded. This
            // permits retirement, not reuse of a failed executable.
            return Ok(());
        }
        let mut state = self.state;
        state.wait(event, || {
            Err(DriverError(sys::CUresult::CUDA_ERROR_UNKNOWN))
        })
    }
}

/// Per-stream execution fences and a separate, irreversible reuse poison.
#[derive(Debug, Default)]
pub(crate) struct ExecutionCompletion {
    events: BTreeMap<u64, (CudaEvent, ExecutionFence)>,
    spare: Option<CudaEvent>,
    uncertain: bool,
}

impl ExecutionCompletion {
    pub(crate) fn new(context: &Arc<CudarcContext>) -> std::result::Result<Self, DriverError> {
        Ok(Self {
            spare: Some(context.new_event(None)?),
            ..Self::default()
        })
    }

    pub(crate) fn submit(
        &mut self,
        stream: &CudaStream,
        confirmation: Option<Arc<crate::memory::OperationCompletion>>,
        launch: impl FnOnce() -> std::result::Result<(), DriverError>,
    ) -> std::result::Result<(), DriverError> {
        if self.uncertain {
            return Err(DriverError(sys::CUresult::CUDA_ERROR_UNKNOWN));
        }
        let key = crate::cuda_graph::stream_execution_id(stream)?;
        if let Some(proof) = &confirmation {
            proof
                .validate_submission(key)
                .map_err(|_| DriverError(sys::CUresult::CUDA_ERROR_INVALID_CONTEXT))?;
        }
        if let std::collections::btree_map::Entry::Vacant(entry) = self.events.entry(key) {
            // Allocate before launching. Failure here has not submitted work.
            let event = match self.spare.take() {
                Some(event) => event,
                None => stream.context().new_event(None)?,
            };
            entry.insert((event, ExecutionFence::default()));
        }
        // Arm before the may-submit boundary, including an unwind before the
        // completion event can be recorded. An older event cannot prove it done.
        self.uncertain = true;
        let (event, completion) = self.events.get_mut(&key).expect("execution event present");
        let submitted = completion.submit(confirmation, launch, || event.record(stream));
        if submitted.is_ok() {
            self.uncertain = false;
        }
        submitted
    }

    pub(crate) fn wait(&self) -> std::result::Result<(), DriverError> {
        let _ordinary = crate::cuda_graph::reserve_capture_exclusion()?;
        for (event, completion) in self.events.values() {
            completion.wait(|| event.synchronize())?;
        }
        Ok(())
    }
}

// The driver permits module/function handles to be used from threads with the
// owning context bound. Mutation of the module registry is separately locked.
unsafe impl Send for LoadedModule {}
unsafe impl Sync for LoadedModule {}

impl Drop for LoadedModule {
    fn drop(&mut self) {
        // Wait for events recorded after this module's submissions, outside
        // capture. They include preceding stream work and transitive waits.
        // A context-wide wait could invalidate an unrelated stream capture.
        let module = self.cu_module as usize;
        let context = self.context.clone();
        let completion = std::mem::take(
            self.completion
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
        crate::cuda_graph::retire_resources_after_completion(
            (module, context, completion),
            |(_, context, completion)| context.bind_to_thread().and_then(|()| completion.wait()),
            |(module, _, _)| {
                // SAFETY: direct submissions are complete; captured submissions
                // retained this module through their graph's retirement fence.
                unsafe { result::module::unload(*module as sys::CUmodule) }
            },
            |_| Ok(()),
            |(_, context, _), error| context.record_err::<()>(Err(error)),
        );
    }
}

/// Kernel handle bound to XLOG's default CUDA stream.
/// Retains the exact loaded module independently of its registry name.
/// Releasing the last module owner follows the potentially blocking
/// [cold-lifecycle progress contract](crate::cuda_graph).
#[derive(Debug, Clone)]
pub struct CudaFunction {
    function_name: Arc<str>,
    context: Arc<CudarcContext>,
    stream: Arc<CudaStream>,
    _module: Arc<LoadedModule>,
}

impl CudaFunction {
    // The immutable symbol table and its audited module owner retain the driver
    // handle. Do not duplicate a raw handle in this otherwise shareable owner,
    // or resolve the symbol through a possibly replaced module registry entry.
    fn cu_function(&self) -> sys::CUfunction {
        self._module.functions[self.function_name.as_ref()]
    }

    fn captured_launch_binding(
        &self,
        cfg: LaunchConfig,
        parameter_count: usize,
        cooperative: bool,
    ) -> CapturedCudaLaunchBinding {
        CapturedCudaLaunchBinding {
            module_name: Arc::clone(&self._module.module_name),
            artifact_name: Arc::clone(&self._module.artifact_name),
            artifact_sha256: self._module.artifact_sha256,
            function_name: Arc::clone(&self.function_name),
            grid_dim: cfg.grid_dim,
            block_dim: cfg.block_dim,
            shared_mem_bytes: cfg.shared_mem_bytes,
            parameter_count: parameter_count as u64,
            cooperative,
        }
    }

    fn submit(
        &self,
        stream: &CudaStream,
        binding: CapturedCudaLaunchBinding,
        launch: impl FnOnce() -> std::result::Result<(), DriverError>,
    ) -> std::result::Result<(), DriverError> {
        if stream.context().cu_ctx() != self.context.cu_ctx() {
            return Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_CONTEXT));
        }
        crate::cuda_graph::submit_with_capture(stream, |owners| {
            self.submit_admitted(stream, owners, Some(binding), None, launch)
        })
    }

    fn submit_admitted(
        &self,
        stream: &CudaStream,
        owners: Option<&crate::cuda_graph::CaptureOwners>,
        binding: Option<CapturedCudaLaunchBinding>,
        confirmation: Option<Arc<crate::memory::OperationCompletion>>,
        launch: impl FnOnce() -> std::result::Result<(), DriverError>,
    ) -> std::result::Result<(), DriverError> {
        if let Some(owners) = owners {
            let mut owners = owners
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            owners
                .modules
                .entry(Arc::as_ptr(&self._module) as usize)
                .or_insert_with(|| self._module.clone());
            if let Some(binding) = binding {
                owners.launches.push(binding);
            }
            drop(owners);
            launch()
        } else {
            self._module
                .completion
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .submit(stream, confirmation, launch)
        }
    }

    pub(crate) unsafe fn launch_raw_in(
        &self,
        enqueue: &crate::launch::CudaEnqueue<'_>,
        cfg: LaunchConfig,
        params: &mut [*mut c_void],
        cooperative: bool,
    ) -> std::result::Result<(), DriverError> {
        let stream = enqueue.stream();
        if stream.context().cu_ctx() != self.context.cu_ctx() {
            return Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_CONTEXT));
        }
        self.context.bind_to_thread()?;
        let binding = self.captured_launch_binding(cfg, params.len(), cooperative);
        enqueue.submit(|owners| {
            self.submit_admitted(
                stream,
                owners,
                Some(binding),
                Some(enqueue.completion()),
                || {
                    if cooperative {
                        result::launch_cooperative_kernel(
                            self.cu_function(),
                            cfg.grid_dim,
                            cfg.block_dim,
                            cfg.shared_mem_bytes,
                            stream.cu_stream(),
                            params,
                        )
                    } else {
                        result::launch_kernel(
                            self.cu_function(),
                            cfg.grid_dim,
                            cfg.block_dim,
                            cfg.shared_mem_bytes,
                            stream.cu_stream(),
                            params,
                        )
                    }
                },
            )
        })
    }
    pub(crate) unsafe fn launch_raw(
        &self,
        cfg: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> std::result::Result<(), DriverError> {
        self.launch_raw_on_stream(&self.stream, cfg, params)
    }

    pub(crate) unsafe fn launch_raw_on_stream(
        &self,
        stream: &CudaStream,
        cfg: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> std::result::Result<(), DriverError> {
        self.context.bind_to_thread()?;
        let parameter_count = params.len();
        let launch = || {
            result::launch_kernel(
                self.cu_function(),
                cfg.grid_dim,
                cfg.block_dim,
                cfg.shared_mem_bytes,
                stream.cu_stream(),
                params,
            )
        };
        self.submit(
            stream,
            self.captured_launch_binding(cfg, parameter_count, false),
            launch,
        )
    }

    pub(crate) unsafe fn launch_raw_cooperative(
        &self,
        cfg: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> std::result::Result<(), DriverError> {
        self.launch_raw_cooperative_on_stream(&self.stream, cfg, params)
    }

    pub(crate) unsafe fn launch_raw_cooperative_on_stream(
        &self,
        stream: &CudaStream,
        cfg: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> std::result::Result<(), DriverError> {
        self.context.bind_to_thread()?;
        let parameter_count = params.len();
        let launch = || {
            result::launch_cooperative_kernel(
                self.cu_function(),
                cfg.grid_dim,
                cfg.block_dim,
                cfg.shared_mem_bytes,
                stream.cu_stream(),
                params,
            )
        };
        self.submit(
            stream,
            self.captured_launch_binding(cfg, parameter_count, true),
            launch,
        )
    }

    pub fn occupancy_available_dynamic_smem_per_block(
        &self,
        num_blocks: u32,
        block_size: u32,
    ) -> std::result::Result<usize, DriverError> {
        self.context.bind_to_thread()?;
        let mut dynamic_smem_size: usize = 0;
        unsafe {
            sys::cuOccupancyAvailableDynamicSMemPerBlock(
                &mut dynamic_smem_size,
                self.cu_function(),
                num_blocks as std::ffi::c_int,
                block_size as std::ffi::c_int,
            )
            .result()?
        };
        Ok(dynamic_smem_size)
    }

    pub fn occupancy_max_active_blocks_per_multiprocessor(
        &self,
        block_size: u32,
        dynamic_smem_size: usize,
        flags: Option<sys::CUoccupancy_flags_enum>,
    ) -> std::result::Result<u32, DriverError> {
        self.context.bind_to_thread()?;
        let mut num_blocks: std::ffi::c_int = 0;
        let flags = flags.unwrap_or(sys::CUoccupancy_flags_enum::CU_OCCUPANCY_DEFAULT);
        unsafe {
            sys::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                &mut num_blocks,
                self.cu_function(),
                block_size as std::ffi::c_int,
                dynamic_smem_size,
                flags as std::ffi::c_uint,
            )
            .result()?
        };
        Ok(num_blocks as u32)
    }

    pub fn occupancy_max_active_clusters(
        &self,
        config: LaunchConfig,
    ) -> std::result::Result<u32, DriverError> {
        self.context.bind_to_thread()?;
        let mut num_clusters: std::ffi::c_int = 0;
        let cfg = sys::CUlaunchConfig {
            gridDimX: config.grid_dim.0,
            gridDimY: config.grid_dim.1,
            gridDimZ: config.grid_dim.2,
            blockDimX: config.block_dim.0,
            blockDimY: config.block_dim.1,
            blockDimZ: config.block_dim.2,
            sharedMemBytes: config.shared_mem_bytes,
            hStream: self.stream.cu_stream(),
            attrs: std::ptr::null_mut(),
            numAttrs: 0,
        };
        unsafe {
            sys::cuOccupancyMaxActiveClusters(&mut num_clusters, self.cu_function(), &cfg)
                .result()?
        };
        Ok(num_clusters as u32)
    }

    pub fn occupancy_max_potential_block_size(
        &self,
        block_size_to_dynamic_smem_size: extern "C" fn(block_size: std::ffi::c_int) -> usize,
        dynamic_smem_size: usize,
        block_size_limit: u32,
        flags: Option<sys::CUoccupancy_flags_enum>,
    ) -> std::result::Result<(u32, u32), DriverError> {
        self.context.bind_to_thread()?;
        let mut min_grid_size: std::ffi::c_int = 0;
        let mut block_size: std::ffi::c_int = 0;
        let flags = flags.unwrap_or(sys::CUoccupancy_flags_enum::CU_OCCUPANCY_DEFAULT);
        unsafe {
            sys::cuOccupancyMaxPotentialBlockSizeWithFlags(
                &mut min_grid_size,
                &mut block_size,
                self.cu_function(),
                Some(block_size_to_dynamic_smem_size),
                dynamic_smem_size,
                block_size_limit as std::ffi::c_int,
                flags as std::ffi::c_uint,
            )
            .result()?
        };
        Ok((min_grid_size as u32, block_size as u32))
    }

    pub fn occupancy_max_potential_cluster_size(
        &self,
        config: LaunchConfig,
    ) -> std::result::Result<u32, DriverError> {
        self.context.bind_to_thread()?;
        let mut cluster_size: std::ffi::c_int = 0;
        let cfg = sys::CUlaunchConfig {
            gridDimX: config.grid_dim.0,
            gridDimY: config.grid_dim.1,
            gridDimZ: config.grid_dim.2,
            blockDimX: config.block_dim.0,
            blockDimY: config.block_dim.1,
            blockDimZ: config.block_dim.2,
            sharedMemBytes: config.shared_mem_bytes,
            hStream: self.stream.cu_stream(),
            attrs: std::ptr::null_mut(),
            numAttrs: 0,
        };
        unsafe {
            sys::cuOccupancyMaxPotentialClusterSize(&mut cluster_size, self.cu_function(), &cfg)
                .result()?
        };
        Ok(cluster_size as u32)
    }

    pub fn get_attribute(
        &self,
        attribute: sys::CUfunction_attribute_enum,
    ) -> std::result::Result<i32, DriverError> {
        self.context.bind_to_thread()?;
        unsafe { result::function::get_function_attribute(self.cu_function(), attribute) }
    }

    pub fn num_regs(&self) -> std::result::Result<i32, DriverError> {
        self.get_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS)
    }

    pub fn shared_size_bytes(&self) -> std::result::Result<i32, DriverError> {
        self.get_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
    }

    pub fn const_size_bytes(&self) -> std::result::Result<i32, DriverError> {
        self.get_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES)
    }

    pub fn local_size_bytes(&self) -> std::result::Result<i32, DriverError> {
        self.get_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)
    }

    pub fn max_threads_per_block(&self) -> std::result::Result<i32, DriverError> {
        self.get_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    }

    pub fn ptx_version(&self) -> std::result::Result<i32, DriverError> {
        self.get_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PTX_VERSION)
    }

    pub fn binary_version(&self) -> std::result::Result<i32, DriverError> {
        self.get_attribute(sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_BINARY_VERSION)
    }

    pub fn set_attribute(
        &self,
        attribute: sys::CUfunction_attribute_enum,
        value: i32,
    ) -> std::result::Result<(), DriverError> {
        self.context.bind_to_thread()?;
        unsafe { result::function::set_function_attribute(self.cu_function(), attribute, value) }
    }

    pub fn set_function_cache_config(
        &self,
        config: sys::CUfunc_cache,
    ) -> std::result::Result<(), DriverError> {
        self.context.bind_to_thread()?;
        unsafe { result::function::set_function_cache_config(self.cu_function(), config) }
    }
}

#[derive(Debug)]
pub struct CudaDeviceInner {
    context: Arc<CudarcContext>,
    stream: Arc<CudaStream>,
    allocation_stream: Arc<CudaStream>,
    modules: RwLock<BTreeMap<String, Arc<LoadedModule>>>,
}

impl CudaDeviceInner {
    fn insert_module(
        &self,
        module_name: &str,
        artifact_name: &str,
        artifact_sha256: Option<[u8; 32]>,
        cu_module: sys::CUmodule,
        kernels: &[&str],
    ) -> std::result::Result<(), DriverError> {
        // Establish ownership before fallible symbol lookup. A rejected module
        // is retired without leaking it or disturbing the previous registry entry.
        let mut module = LoadedModule {
            cu_module,
            functions: BTreeMap::new(),
            module_name: Arc::from(module_name),
            artifact_name: Arc::from(artifact_name),
            artifact_sha256,
            context: self.context.clone(),
            completion: Mutex::default(),
        };
        for &kernel in kernels {
            let name_c = CString::new(kernel).unwrap();
            let cu_function = unsafe { result::module::get_function(cu_module, name_c) }?;
            module.functions.insert(Arc::from(kernel), cu_function);
        }

        let previous = self
            .modules
            .write()
            .unwrap()
            .insert(module_name.to_string(), Arc::new(module));
        // Do not synchronize while holding the registry lock. Existing function
        // owners (including retained session graph functions) keep this image alive.
        drop(previous);
        Ok(())
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub(crate) fn allocation_stream(&self) -> &Arc<CudaStream> {
        &self.allocation_stream
    }

    pub fn has_func(&self, module_name: &str, func_name: &str) -> bool {
        let modules = self.modules.read().unwrap();
        modules
            .get(module_name)
            .is_some_and(|module| module.functions.contains_key(func_name))
    }

    pub fn get_func(&self, module_name: &str, func_name: &str) -> Option<CudaFunction> {
        let modules = self.modules.read().unwrap();
        let module = modules.get(module_name)?;
        let (function_name, _) = module.functions.get_key_value(func_name)?;
        Some(CudaFunction {
            function_name: Arc::clone(function_name),
            context: self.context.clone(),
            stream: self.stream.clone(),
            _module: module.clone(),
        })
    }

    pub fn load_file(
        &self,
        path: &Path,
        module_name: &str,
        kernels: &[&str],
    ) -> std::result::Result<(), DriverError> {
        let artifact_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("file-artifact")
            .to_string();
        if let Ok(bytes) = std::fs::read(path) {
            let is_cubin = path.extension().and_then(|value| value.to_str()) == Some("cubin");
            return self.load_artifact_bytes(
                &bytes,
                is_cubin,
                module_name,
                &artifact_name,
                kernels,
            );
        }
        crate::cuda_graph::with_module_loading(|| {
            self.context.bind_to_thread()?;
            let name_c = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let cu_module = result::module::load(name_c)?;
            self.insert_module(module_name, &artifact_name, None, cu_module, kernels)
        })
    }

    pub(crate) fn load_artifact_bytes(
        &self,
        bytes: &[u8],
        is_cubin: bool,
        module_name: &str,
        artifact_name: &str,
        kernels: &[&str],
    ) -> std::result::Result<(), DriverError> {
        let artifact_sha256 = Sha256::digest(bytes).into();
        crate::cuda_graph::with_module_loading(|| {
            self.context.bind_to_thread()?;
            let cu_module = if is_cubin {
                unsafe { result::module::load_data(bytes.as_ptr().cast()) }?
            } else {
                let source = CString::new(bytes).unwrap();
                unsafe { result::module::load_data(source.as_ptr().cast()) }?
            };
            self.insert_module(
                module_name,
                artifact_name,
                Some(artifact_sha256),
                cu_module,
                kernels,
            )
        })
    }

    pub fn load_ptx(
        &self,
        ptx: Ptx,
        module_name: &str,
        kernels: &[&str],
    ) -> std::result::Result<(), DriverError> {
        self.load_ptx_named(ptx, module_name, "in-memory-ptx", kernels)
    }

    pub(crate) fn load_ptx_named(
        &self,
        ptx: Ptx,
        module_name: &str,
        artifact_name: &str,
        kernels: &[&str],
    ) -> std::result::Result<(), DriverError> {
        crate::cuda_graph::with_module_loading(|| {
            self.context.bind_to_thread()?;
            let (cu_module, artifact_sha256) = if let Some(bytes) = ptx.as_bytes() {
                let digest = Sha256::digest(bytes).into();
                (
                    unsafe { result::module::load_data(bytes.as_ptr() as *const _) }?,
                    digest,
                )
            } else {
                let source = ptx.to_src();
                let digest = Sha256::digest(source.as_bytes()).into();
                let src = CString::new(source).unwrap();
                (
                    unsafe { result::module::load_data(src.as_ptr() as *const _) }?,
                    digest,
                )
            };
            self.insert_module(
                module_name,
                artifact_name,
                Some(artifact_sha256),
                cu_module,
                kernels,
            )
        })
    }

    /// Allocate an uninitialized device slice on this device stream.
    ///
    /// # Safety
    ///
    /// The caller must initialize the returned allocation before any device or
    /// host read observes its contents.
    pub unsafe fn alloc<T: DeviceRepr + 'static>(
        &self,
        len: usize,
    ) -> ResourceResult<DeviceMemoryView<T>> {
        DeviceMemoryView::allocate(
            Arc::clone(&self.stream),
            Arc::clone(&self.allocation_stream),
            len,
        )
    }

    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits + 'static>(
        &self,
        len: usize,
    ) -> ResourceResult<DeviceMemoryView<T>> {
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)?;
        let mut allocation = DeviceMemoryView::allocate(
            Arc::clone(&self.stream),
            Arc::clone(&self.allocation_stream),
            len,
        )?;
        self.memset_zeros(&mut allocation)?;
        Ok(allocation)
    }

    pub fn memset_zeros<T: DeviceRepr + ValidAsZeroBits, Dst: DeviceWrite<T>>(
        &self,
        dst: &mut Dst,
    ) -> ResourceResult<()> {
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)?;
        let dst = dst.device_view();
        with_memory_access(
            Arc::clone(&self.stream),
            vec![dst.access(Access::Write)?],
            |proof| {
                self.stream.memset_zeros(&mut proof.write(&dst)?)?;
                Ok(self.stream.synchronize()?)
            },
        )
    }

    pub fn htod_sync_copy_into<T: DeviceRepr, Dst: DeviceWrite<T>, Src: HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> ResourceResult<()> {
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)?;
        let dst = dst.device_view().try_slice(..src.len()).ok_or_else(|| {
            crate::device_runtime::ResourceError::StreamMisuse(
                "host-to-device copy destination is shorter than its source".into(),
            )
        })?;
        let bytes = src
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| {
                crate::device_runtime::ResourceError::StreamMisuse(
                    "host copy byte size overflow".into(),
                )
            })?;
        let mut staging = PinnedHostBuffer::new(&self.stream, bytes)?;
        {
            // SAFETY: HostSlice supplies its recorded producer dependencies.
            // Complete those waits before reading even a pinned host slice.
            let (source, _guard) = unsafe { src.stream_synced_slice(&self.stream) };
            self.stream.synchronize()?;
            staging.write(source)?;
        }
        with_memory_access(
            Arc::clone(&self.stream),
            vec![dst.access(Access::Write)?],
            |proof| {
                let mut destination = proof.write(&dst)?;
                let (ptr, _guard) = destination.device_ptr_mut(&self.stream);
                // SAFETY: the complete destination is admitted; only owned
                // staging storage participates in DMA, including on error.
                unsafe {
                    staging.enqueue(&self.stream, |host| {
                        sys::cuMemcpyHtoDAsync_v2(ptr, host.cast(), bytes, self.stream.cu_stream())
                            .result()
                    })?;
                }
                Ok(staging.wait()?)
            },
        )
    }

    pub fn dtoh_sync_copy_into<T: DeviceRepr, Src: DeviceRead<T>, Dst: HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> ResourceResult<()> {
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)?;
        let src = src.device_view();
        let len = src.len();
        if dst.len() < len {
            return Err(crate::device_runtime::ResourceError::StreamMisuse(
                "device-to-host copy destination is shorter than its source".into(),
            ));
        }
        let bytes = len.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
            crate::device_runtime::ResourceError::StreamMisuse(
                "host copy byte size overflow".into(),
            )
        })?;
        let mut staging = PinnedHostBuffer::new(&self.stream, bytes)?;
        with_memory_access(
            Arc::clone(&self.stream),
            vec![src.access(Access::Read)?],
            |proof| {
                let source = proof.read(&src)?;
                let (ptr, _guard) = source.device_ptr(&self.stream);
                // SAFETY: admitted source and uniquely owned pinned destination
                // remain retained through the transaction's failure cleanup.
                unsafe {
                    staging.enqueue(&self.stream, |host| {
                        sys::cuMemcpyDtoHAsync_v2(host.cast(), ptr, bytes, self.stream.cu_stream())
                            .result()
                    })?;
                }
                Ok(staging.wait()?)
            },
        )?;
        // SAFETY: synchronize HostSlice's own dependencies before CPU writes.
        let (destination, _guard) = unsafe { dst.stream_synced_mut_slice(&self.stream) };
        self.stream.synchronize()?;
        // SAFETY: the admitted T source initialized every staging byte and its
        // completion was proven before the caller's storage is touched.
        unsafe { staging.read_into(&mut destination[..len])? };
        Ok(())
    }

    pub fn htod_sync_copy<T: DeviceRepr + 'static, Src: HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
    ) -> ResourceResult<DeviceMemoryView<T>> {
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)?;
        let mut dst = DeviceMemoryView::allocate(
            Arc::clone(&self.stream),
            Arc::clone(&self.allocation_stream),
            src.len(),
        )?;
        self.htod_sync_copy_into(src, &mut dst)?;
        Ok(dst)
    }

    pub fn dtoh_sync_copy<T: DeviceRepr, Src: DeviceRead<T>>(
        &self,
        src: &Src,
    ) -> ResourceResult<Vec<T>> {
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)?;
        let src = src.device_view();
        let len = src.len();
        let bytes = len.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
            crate::device_runtime::ResourceError::StreamMisuse(
                "host copy byte size overflow".into(),
            )
        })?;
        let mut staging = PinnedHostBuffer::new(&self.stream, bytes)?;
        with_memory_access(
            Arc::clone(&self.stream),
            vec![src.access(Access::Read)?],
            |proof| {
                let source = proof.read(&src)?;
                let (ptr, _guard) = source.device_ptr(&self.stream);
                // SAFETY: ownership is admitted and the pinned destination is
                // outside the callback, so unwind cannot destroy it before abort.
                unsafe {
                    staging.enqueue(&self.stream, |host| {
                        sys::cuMemcpyDtoHAsync_v2(host.cast(), ptr, bytes, self.stream.cu_stream())
                            .result()
                    })?;
                }
                Ok(staging.wait()?)
            },
        )?;
        // SAFETY: the whole admitted device source was copied and synchronized.
        Ok(unsafe { staging.read_vec(len)? })
    }

    pub fn dtod_copy<T, Src: DeviceRead<T>, Dst: DeviceWrite<T>>(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> ResourceResult<()> {
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)?;
        let src = src.device_view();
        let dst = dst.device_view().try_slice(..src.len()).ok_or_else(|| {
            crate::device_runtime::ResourceError::StreamMisuse(
                "device copy destination is shorter than its source".into(),
            )
        })?;
        with_memory_access(
            Arc::clone(&self.stream),
            vec![src.access(Access::Read)?, dst.access(Access::Write)?],
            |proof| {
                proof.copy(&src, &dst)?;
                Ok(self.stream.synchronize()?)
            },
        )
    }

    /// Enqueue a device-to-device copy without synchronizing the stream.
    ///
    /// Callers batching many copies must synchronize once after the last
    /// enqueue and before reading any destination.
    pub fn dtod_copy_async<T, Src: DeviceRead<T>, Dst: DeviceWrite<T>>(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> ResourceResult<()> {
        let src = src.device_view();
        let dst = dst.device_view().try_slice(..src.len()).ok_or_else(|| {
            crate::device_runtime::ResourceError::StreamMisuse(
                "device copy destination is shorter than its source".into(),
            )
        })?;
        with_memory_access(
            Arc::clone(&self.stream),
            vec![src.access(Access::Read)?, dst.access(Access::Write)?],
            |proof| proof.copy(&src, &dst),
        )
    }

    /// Wrap an existing CUDA device pointer in a typed cudarc slice.
    ///
    /// # Safety
    ///
    /// `cu_device_ptr` must point to a live allocation containing at least
    /// `len * size_of::<T>()` bytes, and the resulting wrapper must not outlive
    /// the allocation or alias another owner that will free it independently.
    pub unsafe fn upgrade_device_ptr<T>(
        &self,
        cu_device_ptr: sys::CUdeviceptr,
        len: usize,
    ) -> CudaSlice<T> {
        self.stream.upgrade_device_ptr(cu_device_ptr, len)
    }

    pub fn attribute(
        &self,
        attrib: sys::CUdevice_attribute,
    ) -> std::result::Result<i32, DriverError> {
        self.context.attribute(attrib)
    }

    pub fn synchronize(&self) -> std::result::Result<(), DriverError> {
        let _ordinary = crate::cuda_graph::reserve_uncaptured_stream(&self.stream)?;
        self.stream.synchronize()
    }

    pub fn ordinal(&self) -> usize {
        self.context.ordinal()
    }
}

/// CUDA device wrapper for GPU operations.
///
/// This keeps XLOG's historical "device with a built-in default stream" API,
/// but is backed by cudarc's newer `CudaContext` and `CudaStream`.
pub struct CudaDevice {
    device: Arc<CudaDeviceInner>,
}

impl CudaDevice {
    /// Create a new CUDA device on the specified GPU ordinal.
    pub fn new(ordinal: usize) -> Result<Self> {
        let context = std::panic::catch_unwind(|| CudarcContext::new(ordinal))
            .map_err(|_| {
                XlogError::Kernel(format!(
                    "Failed to create CUDA device {}: cudarc panicked during driver initialization",
                    ordinal
                ))
            })?
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to create CUDA device {}: {}", ordinal, e))
            })?;

        let stream = context.default_stream();
        let allocation_stream = context.new_stream().map_err(|error| {
            XlogError::Kernel(format!(
                "Failed to create CUDA allocation stream on device {}: {}",
                ordinal, error
            ))
        })?;
        Ok(Self {
            device: Arc::new(CudaDeviceInner {
                context,
                stream,
                allocation_stream,
                modules: RwLock::new(BTreeMap::new()),
            }),
        })
    }

    pub fn count() -> Result<i32> {
        std::panic::catch_unwind(|| {
            result::init()?;
            result::device::get_count()
        })
        .map_err(|_| {
            XlogError::Kernel(
                "Failed to count CUDA devices: cudarc panicked during driver initialization"
                    .to_string(),
            )
        })?
        .map_err(|e| XlogError::Kernel(format!("Failed to count CUDA devices: {}", e)))
    }

    pub fn synchronize(&self) -> Result<()> {
        self.device
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("Failed to synchronize device: {}", e)))
    }

    pub fn inner(&self) -> &Arc<CudaDeviceInner> {
        &self.device
    }

    pub fn ordinal(&self) -> usize {
        self.device.ordinal()
    }
}

// Compile-time assertion: CudaDevice must be Send so pyxlog can use py.allow_threads().
const _: () = {
    fn _assert_send<T: Send>() {}
    fn _check() {
        _assert_send::<CudaDevice>();
    }
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_native_owner_chain_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CudaFunction>();
        assert_send_sync::<crate::SemanticHypergraph>();
        assert_send_sync::<crate::SemanticTransitionSession>();
        assert_send_sync::<crate::SemanticPublishedLease>();
        assert_send_sync::<crate::SemanticTensorContentWitness>();
    }
    use cudarc::driver::DevicePtrMut;

    #[test]
    #[ignore = "requires authorized CUDA execution"]
    fn exact_abort_releases_real_graph_after_failed_completion_records() {
        use crate::cuda_graph::{
            reap_capture_retirements, retire_resources_after_completion, CapturedCudaGraph,
        };
        use crate::device_runtime::{
            AsyncCudaResource, ResourceError, StreamPool, XlogDeviceRuntime,
        };
        use crate::memory::{with_memory_manifest, MemoryAccessManifest};

        let _serial = crate::cuda_graph::capture_lifecycle_test_guard();
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let stream = device.inner().stream().context().new_stream().unwrap();
        let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let resource = Box::new(AsyncCudaResource::new(
            Arc::clone(&device),
            0,
            Arc::clone(&pool),
        ));
        let runtime = XlogDeviceRuntime::with_resource(Arc::clone(&device), 0, pool, resource);
        let image = || {
            Ptx::from_src(
                ".version 7.0\n.target sm_75\n.address_size 64\n.visible .entry run() { ret; }\n",
            )
        };
        device
            .inner()
            .load_ptx(image(), "abort_completion_module", &["run"])
            .unwrap();
        let function = device
            .inner()
            .get_func("abort_completion_module", "run")
            .unwrap();
        let module = Arc::downgrade(&function._module);
        let graph = CapturedCudaGraph::capture_on_stream(&stream, || {
            // SAFETY: this kernel has no arguments or memory accesses.
            unsafe {
                function.launch_raw_on_stream(
                    &stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut [],
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })
        .unwrap()
        .bind_resident_lifecycle(&runtime);
        drop(function);
        device
            .inner()
            .load_ptx(image(), "abort_completion_module", &["run"])
            .unwrap();
        assert!(module.upgrade().is_some());
        let live = runtime.resident_graph_handle_lifecycle_stats();
        assert_eq!((live.live_graphs, live.live_graph_execs), (1, 1));

        let first_error = DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE);
        let repair_error = DriverError(sys::CUresult::CUDA_ERROR_UNKNOWN);
        let mut records = 0;
        let mut launches = 0;
        let mut proof = None;
        let result: ResourceResult<()> = with_memory_manifest(
            Arc::clone(&stream),
            Arc::new(MemoryAccessManifest::default()),
            |enqueue| {
                let confirmation = enqueue.completion();
                proof = Some(Arc::clone(&confirmation));
                let event = stream.context().new_event(None).unwrap();
                let mut fence = ExecutionFence::default();
                // Inject record failures at the shared fence, around a real
                // launch. This is not public graph event-record fault injection.
                let submitted = enqueue.submit(|_| {
                    fence.submit(
                        Some(confirmation),
                        || {
                            launches += 1;
                            // SAFETY: the graph and its captured module are owned
                            // until retirement; this is their original context.
                            unsafe { sys::cuGraphLaunch(graph.exec(), stream.cu_stream()).result() }
                        },
                        || {
                            records += 1;
                            Err(if records == 1 {
                                first_error
                            } else {
                                repair_error
                            })
                        },
                    )
                });
                let unconfirmed = fence.wait(|| event.synchronize());
                retire_resources_after_completion(
                    (fence, event, Some(graph)),
                    |(fence, event, _)| fence.wait(|| event.synchronize()),
                    |(_, _, graph)| {
                        drop(graph.take());
                        Ok(())
                    },
                    |_| Ok(()),
                    |_, error| eprintln!("injected completion-record failure: {error}"),
                );
                // Assertions come after transferring the actual graph into
                // retirement, so a failed assertion cannot drop an in-flight
                // raw launch through the graph's unsubmitted ordinary fence.
                assert_eq!(submitted, Err(first_error));
                assert_eq!(unconfirmed, Err(repair_error));
                assert_eq!(runtime.resident_graph_handle_lifecycle_stats(), live);
                // Returning the original error invokes MemoryOperationOwner's
                // real exact-stream abort, not a test synchronization callback.
                submitted.map_err(|error| ResourceError::Driver(error.to_string()))
            },
        );
        let expected = crate::launch::LaunchEnqueueError::Operation(ResourceError::Driver(
            first_error.to_string(),
        ))
        .to_string();
        assert!(matches!(result, Err(ResourceError::Driver(error)) if error == expected));
        assert!(proof.unwrap().is_complete());
        assert_eq!((launches, records), (1, 2));
        reap_capture_retirements();
        let retired = runtime.resident_graph_handle_lifecycle_stats();
        assert_eq!((retired.live_graphs, retired.live_graph_execs), (0, 0));
        assert_eq!(retired.destroyed_graphs, live.destroyed_graphs + 1);
        assert_eq!(
            retired.destroyed_graph_execs,
            live.destroyed_graph_execs + 1
        );
        // This weak pointer checks Rust module ownership, not driver unload.
        assert!(module.upgrade().is_none());
        reap_capture_retirements();
        assert_eq!(runtime.resident_graph_handle_lifecycle_stats(), retired);
    }

    #[test]
    fn captured_function_outlives_replacement_and_external_owners() {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
            return;
        }
        let device = CudaDevice::new(0).unwrap();
        let stream = device.inner().stream().context().new_stream().unwrap();
        let joined = stream.context().new_stream().unwrap();
        let fork = stream.context().new_event(None).unwrap();
        let join = stream.context().new_event(None).unwrap();
        let mut output = device.inner().stream().alloc_zeros::<u32>(1).unwrap();
        let (mut address, output_record) = output.device_ptr_mut(&joined);
        drop(output_record);
        joined.synchronize().unwrap();
        let image = |value: u32| {
            Ptx::from_src(format!(
                ".version 7.0\n.target sm_75\n.address_size 64\n\
             .visible .entry write_value(.param .u64 output) {{\n\
             .reg .b64 address;\nld.param.u64 address, [output];\n\
             st.global.u32 [address], {value};\nret;\n}}\n"
            ))
        };
        device
            .inner()
            .load_ptx(image(7), "captured_module", &["write_value"])
            .unwrap();
        let function = device
            .inner()
            .get_func("captured_module", "write_value")
            .unwrap();
        let module = Arc::downgrade(&function._module);
        let graph = crate::cuda_graph::CapturedCudaGraph::capture_on_stream(&stream, || {
            fork.record(&stream).unwrap();
            joined.wait(&fork).unwrap();
            let mut params = [(&mut address as *mut u64).cast::<c_void>()];
            // SAFETY: one thread writes this live one-u32 allocation on its own
            // context. The allocation remains owned through graph replay below.
            unsafe {
                function.launch_raw_on_stream(
                    &joined,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut params,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))?;
            drop(function);
            join.record(&joined).unwrap();
            stream.wait(&join).unwrap();
            Ok(())
        })
        .unwrap();
        device
            .inner()
            .load_ptx(image(11), "captured_module", &["write_value"])
            .unwrap();
        drop(device);
        assert!(module.upgrade().is_some());
        graph.launch(&stream).unwrap();
        // No external wait: graph retirement must retain its module until its
        // submitted execution completes, then release that exact module.
        drop(graph);
        assert!(module.upgrade().is_none());
        assert_eq!(stream.clone_dtoh(&output).unwrap(), [7]);
    }

    #[test]
    #[ignore = "requires authorized CUDA execution"]
    fn submission_error_repairs_prefix_without_invalidating_external_capture() {
        struct ExternalCapture {
            stream: Arc<CudaStream>,
            active: bool,
        }
        impl ExternalCapture {
            fn end(&mut self) -> sys::CUresult {
                if !self.active {
                    return sys::CUresult::CUDA_SUCCESS;
                }
                self.active = false;
                let mut graph = std::ptr::null_mut();
                // SAFETY: this guard owns the external capture; EndCapture
                // terminates it even when the captured sequence was invalidated.
                let ended = unsafe { sys::cuStreamEndCapture(self.stream.cu_stream(), &mut graph) };
                if !graph.is_null() {
                    // SAFETY: EndCapture just returned this uninstantiated graph.
                    let destroyed = unsafe { sys::cuGraphDestroy(graph) };
                    if destroyed != sys::CUresult::CUDA_SUCCESS {
                        eprintln!("external captured graph cleanup returned {destroyed:?}");
                        if ended == sys::CUresult::CUDA_SUCCESS {
                            return destroyed;
                        }
                    }
                }
                ended
            }
        }
        impl Drop for ExternalCapture {
            fn drop(&mut self) {
                let ended = self.end();
                if ended != sys::CUresult::CUDA_SUCCESS {
                    eprintln!("external capture cleanup returned {ended:?}");
                }
            }
        }
        let _capture_test = crate::cuda_graph::capture_lifecycle_test_guard();
        let device = CudaDevice::new(0).unwrap();
        let stream = device.inner().stream().context().new_stream().unwrap();
        let external = stream.context().new_stream().unwrap();
        device.inner().load_ptx(
            Ptx::from_src(".version 7.0\n.target sm_75\n.address_size 64\n.visible .entry run() { ret; }\n"),
            "unconfirmed_module", &["run"],
        ).unwrap();
        let function = device
            .inner()
            .get_func("unconfirmed_module", "run")
            .unwrap();
        let module = Arc::downgrade(&function._module);
        // Deliberately bypass the XLOG capture registry, as another library
        // sharing this CUDA primary context can do.
        unsafe {
            sys::cuStreamBeginCapture_v2(
                external.cu_stream(),
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
            )
            .result()
            .unwrap();
        }
        let mut external_capture = ExternalCapture {
            stream: Arc::clone(&external),
            active: true,
        };
        let submitted = function
            ._module
            .completion
            .lock()
            .unwrap()
            .submit(&stream, None, || {
                // Fault injection after a real enqueue, before a completion event:
                // an older completed event would not prove this launch complete.
                unsafe {
                    result::launch_kernel(
                        function.cu_function(),
                        (1, 1, 1),
                        (1, 1, 1),
                        0,
                        stream.cu_stream(),
                        &mut [],
                    )?;
                }
                Err(DriverError(sys::CUresult::CUDA_ERROR_UNKNOWN))
            });
        assert_eq!(
            submitted,
            Err(DriverError(sys::CUresult::CUDA_ERROR_UNKNOWN))
        );
        assert!(function._module.completion.lock().unwrap().uncertain);
        drop(function);
        drop(device);
        let ended = external_capture.end();
        assert_eq!(
            ended,
            sys::CUresult::CUDA_SUCCESS,
            "retirement invalidated external capture"
        );
        stream.context().bind_to_thread().unwrap();
        assert!(
            module.upgrade().is_none(),
            "confirmed prefix repair did not retire the module"
        );
        stream.synchronize().unwrap();
    }

    #[test]
    fn retiring_unrelated_module_does_not_invalidate_stream_capture() {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
            return;
        }
        let device = CudaDevice::new(0).unwrap();
        let stream = device.inner().context.new_stream().unwrap();
        let image = || {
            Ptx::from_src(
                ".version 7.0\n.target sm_75\n.address_size 64\n\
                 .visible .entry unused_function() { ret; }\n",
            )
        };
        device
            .inner()
            .load_ptx(image(), "retired_module", &["unused_function"])
            .unwrap();
        let original = device
            .inner()
            .get_func("retired_module", "unused_function")
            .unwrap();
        device
            .inner()
            .load_ptx(image(), "retired_module", &["unused_function"])
            .unwrap();
        let graph = crate::cuda_graph::CapturedCudaGraph::capture_on_stream(&stream, || {
            drop(original);
            Ok(())
        });
        let _graph = graph.expect("module retirement invalidated capture");
        stream.synchronize().unwrap();
    }

    #[test]
    fn loaded_functions_survive_module_replacement_and_device_owner_drop() {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
            return;
        }
        let device = CudaDevice::new(0).unwrap();
        let stream = Arc::clone(device.inner().stream());
        let mut output = device.inner().stream().alloc_zeros::<u32>(1).unwrap();
        let image = |value: u32| {
            Ptx::from_src(format!(
                ".version 7.0\n.target sm_75\n.address_size 64\n\
                 .visible .entry write_value(.param .u64 output) {{\n\
                 .reg .b64 address;\nld.param.u64 address, [output];\n\
                 st.global.u32 [address], {value};\nret;\n}}\n"
            ))
        };
        let mut run = |function: &CudaFunction, expected: u32| {
            {
                let (mut address, _record) = output.device_ptr_mut(&stream);
                let mut params = [(&mut address as *mut u64).cast::<c_void>()];
                // SAFETY: this one-thread kernel writes one u32 in this live
                // allocation, on its owning context and stream. Wait below before
                // reading or reusing the allocation or releasing the function.
                unsafe {
                    function.launch_raw(
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (1, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &mut params,
                    )
                }
                .unwrap();
            }
            let result = stream.clone_dtoh(&output).unwrap();
            stream.synchronize().unwrap();
            assert_eq!(result, [expected]);
        };
        device
            .inner()
            .load_ptx(image(7), "retained_module", &["write_value"])
            .unwrap();
        let original = device
            .inner()
            .get_func("retained_module", "write_value")
            .unwrap();
        run(&original, 7);
        assert!(device
            .inner()
            .load_ptx(image(11), "retained_module", &["missing_function"])
            .is_err());
        run(
            &device
                .inner()
                .get_func("retained_module", "write_value")
                .unwrap(),
            7,
        );
        device
            .inner()
            .load_ptx(image(11), "retained_module", &["write_value"])
            .unwrap();
        let replacement = device
            .inner()
            .get_func("retained_module", "write_value")
            .unwrap();
        run(&original, 7);
        run(&replacement, 11);
        drop(device);
        run(&original, 7);
        run(&replacement, 11);
    }

    #[test]
    fn test_device_creation() {
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
                return;
            }
        };
        drop(device);
    }

    #[test]
    fn test_device_synchronize() {
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
                return;
            }
        };
        let result = device.synchronize();
        assert!(result.is_ok(), "Failed to synchronize: {:?}", result.err());
    }

    #[test]
    fn test_device_ordinal() {
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
                return;
            }
        };
        assert_eq!(device.ordinal(), 0);
    }

    #[test]
    fn test_device_inner_access() {
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
                return;
            }
        };
        let inner = device.inner();
        assert_eq!(inner.ordinal(), 0);
    }

    #[test]
    fn test_invalid_device_ordinal() {
        let result = CudaDevice::new(9999);
        assert!(result.is_err(), "Should fail with invalid ordinal");

        if let Err(XlogError::Kernel(msg)) = result {
            assert!(msg.contains("9999"), "Error should mention device ordinal");
        } else {
            panic!("Expected XlogError::Kernel");
        }
    }
}
