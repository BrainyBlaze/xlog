//! CUDA Graph RAII helpers for production graph capture/replay.
//!
//! This module intentionally stays close to the CUDA driver API. The bounded
//! CSM CUDA Graph path needs explicit graph lifetime ownership and node
//! inventory before it can safely update graph-exec parameters for runtime
//! pointers and capacity classes.
//!
//! # Cold lifecycle and progress
//!
//! Releasing the last module or graph owner can synchronously wait for recorded
//! execution events. During capture, destruction is deferred: finishing the last
//! capture can therefore wait for another owner's earlier executions. Capture
//! setup and module loading/replacement can also drain pending destruction.
//! These are cold lifecycle boundaries, outside the measured resident replay;
//! include their cost in end-to-end measurements.
//! Failed memory-operation cleanup can require a barrier on the entire retained
//! context, including after its original host thread exits without a usable
//! completion event. This recovery runs only in the cold retirement queue with
//! capture excluded; ordinary enqueue and commit do not add that barrier.
//!
//! An execution event covers all preceding work on its stream, including
//! transitive waits, not just kernels using the retiring resource. Before any
//! boundary that can drain destruction, that work must be able to finish without
//! this call returning, a new module load or capture, launching the graph still
//! being captured, or releasing a caller-held lock or Python GIL. Persistent
//! kernels need an independently progressing stop path established beforehand.
//! Do not rely on CUDA callbacks to advance this teardown, or destroy owners
//! from a CUDA callback. The registry mutex is released during waits/destruction;
//! new captures remain excluded, while already-loaded work can still be submitted.
//! Context-barrier recovery imposes that same progress requirement on all earlier
//! work in the retained context, not just work on one retiring stream.
//!
//! Recorded events or a successful barrier covering the actual submissions prove
//! completion. EndCapture and unrecorded events do not. Unknown completion retains
//! the concrete resources
//! for diagnosis; it is neither successful release nor a normal lifetime policy.
//! Raw graph/function users must retain their exact resource owners through all
//! uses and future replays, and establish completion before releasing them.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::{
    fmt, mem, ptr,
    sync::{Arc, Condvar, Mutex, OnceLock},
};

use cudarc::driver::{result::DriverError, sys, CudaContext, CudaStream};
use libloading::Library;
use xlog_core::{Result, XlogError};

use crate::device::{ExecutionCompletion, LoadedModule};
use crate::device_runtime::XlogDeviceRuntime;

pub const CSM_CUDA_GRAPH_NODE_LAYOUT_VERSION: u32 = 1;
const CONDITIONAL_GRAPH_MINIMUM_DRIVER: i32 = 12_030;

type DriverGetVersionFn = unsafe extern "C" fn(*mut i32) -> sys::CUresult;
type ConditionalHandleCreateFn = unsafe extern "C" fn(
    *mut sys::CUgraphConditionalHandle,
    sys::CUgraph,
    sys::CUcontext,
    u32,
    u32,
) -> sys::CUresult;
type GraphAddNodeFn = unsafe extern "C" fn(
    *mut sys::CUgraphNode,
    sys::CUgraph,
    *const sys::CUgraphNode,
    usize,
    *mut sys::CUgraphNodeParams,
) -> sys::CUresult;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaConditionalGraphUnavailable {
    DriverLibraryUnavailable,
    MissingDriverSymbol {
        symbol: &'static str,
    },
    DriverVersionQueryFailed {
        code: sys::CUresult,
    },
    DriverVersionTooOld {
        found: i32,
        required: i32,
    },
    DriverCallFailed {
        operation: &'static str,
        code: sys::CUresult,
    },
    NullDriverHandle {
        operation: &'static str,
    },
    ContextMismatch,
    StreamCaptureBusy,
    BodyPopulationFailed {
        detail: String,
    },
}

impl CudaConditionalGraphUnavailable {
    pub fn is_unsupported(&self) -> bool {
        matches!(
            self,
            Self::DriverLibraryUnavailable
                | Self::MissingDriverSymbol { .. }
                | Self::DriverVersionTooOld { .. }
                | Self::DriverCallFailed {
                    code: sys::CUresult::CUDA_ERROR_NOT_SUPPORTED,
                    ..
                }
        )
    }

    pub fn decline_detail(&self) -> String {
        match self {
            Self::DriverLibraryUnavailable => {
                "CUDA driver library is unavailable for conditional graphs".to_string()
            }
            Self::MissingDriverSymbol { symbol } => {
                format!("CUDA driver is missing required conditional-graph symbol {symbol}")
            }
            Self::DriverVersionQueryFailed { code } => {
                format!("CUDA driver version query failed: {code:?}")
            }
            Self::DriverVersionTooOld { found, required } => {
                format!("CUDA conditional graphs require driver API {required}, found {found}")
            }
            Self::DriverCallFailed { operation, code } => {
                format!("CUDA conditional-graph operation {operation} failed: {code:?}")
            }
            Self::NullDriverHandle { operation } => {
                format!("CUDA conditional-graph operation {operation} returned a null handle")
            }
            Self::ContextMismatch => {
                "CUDA conditional graph and stream belong to different contexts".to_string()
            }
            Self::StreamCaptureBusy => {
                "CUDA stream already has an active graph capture".to_string()
            }
            Self::BodyPopulationFailed { detail } => {
                format!("CUDA conditional graph body population failed: {detail}")
            }
        }
    }

    pub fn body_population(error: impl fmt::Display) -> Self {
        Self::BodyPopulationFailed {
            detail: error.to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct StreamCaptureKey {
    context: usize,
    stream: u64,
}

/// CUDA's process-unique ID also distinguishes per-thread default streams;
/// their raw sentinel handle is identical on every host thread.
pub(crate) fn stream_execution_id(stream: &CudaStream) -> std::result::Result<u64, DriverError> {
    stream.context().bind_to_thread()?;
    let mut id = 0;
    unsafe { sys::cuStreamGetId(stream.cu_stream(), &mut id).result()? };
    Ok(id)
}

#[derive(Default)]
struct StreamCaptureRegistry {
    active: HashSet<StreamCaptureKey>,
    modules: BTreeMap<(usize, u64), CaptureSubmissionEntry>,
    deferred: VecDeque<DeferredRetirement>,
    retiring: bool,
    capture_exclusions: usize,
}

enum DeferredRetirement {
    Once(Option<Box<dyn FnOnce() + Send>>),
    Retry(Box<dyn FnMut() -> bool + Send>),
}

impl DeferredRetirement {
    fn attempt(&mut self) -> bool {
        match self {
            Self::Once(retire) => {
                if let Some(retire) = retire.take() {
                    retire();
                }
                true
            }
            Self::Retry(retire) => retire(),
        }
    }
}

#[derive(Default)]
pub(crate) struct CapturedOwners {
    pub(crate) modules: BTreeMap<usize, Arc<LoadedModule>>,
    pub(crate) memory: Vec<Arc<crate::memory::MemoryAccessManifest>>,
    // Actual producer and stream owners, not an attestation of pointer validity
    // or execution. They must exist before capture can record their uses.
    external_resources: Vec<Arc<dyn Send + Sync>>,
    submission: Arc<Mutex<()>>,
    failed: bool,
}

impl CapturedOwners {
    fn with_external_resources(external_resources: Vec<Arc<dyn Send + Sync>>) -> Self {
        Self {
            external_resources,
            ..Self::default()
        }
    }
}

impl std::fmt::Debug for CapturedOwners {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapturedOwners")
            .field("modules", &self.modules.len())
            .field("memory_manifests", &self.memory.len())
            .field("external_resources", &self.external_resources.len())
            .finish()
    }
}

pub(crate) type CaptureOwners = Arc<Mutex<CapturedOwners>>;

fn retain_child_capture_owners(
    parent: &CaptureOwners,
    child: &CaptureOwners,
    memory: &Arc<crate::memory::MemoryAccessManifest>,
) {
    // Release the child lock before locking the parent, including when nested
    // capture already shares one owner set. Freeze the current memory binding,
    // not the child's future mutable graph parameters.
    let (modules, external_resources) = {
        let child = child.lock().unwrap_or_else(|error| error.into_inner());
        (child.modules.clone(), child.external_resources.clone())
    };
    let mut parent = parent.lock().unwrap_or_else(|error| error.into_inner());
    parent.modules.extend(modules);
    parent.external_resources.extend(external_resources);
    parent.memory.push(memory.clone());
}

struct CaptureSubmissionEntry {
    owners: CaptureOwners,
    closing: bool,
    executing: Arc<CaptureExecutingSubmissions>,
}

#[derive(Default)]
struct CaptureExecutingSubmissions {
    count: Mutex<usize>,
    idle: Condvar,
}

impl CaptureExecutingSubmissions {
    // Admission holds the registry mutex while calling this method, so closing
    // cannot pass a newly admitted submission before its count is visible.
    fn pin(self: &Arc<Self>) -> CaptureExecutingSubmission {
        *self.count.lock().unwrap_or_else(|error| error.into_inner()) += 1;
        CaptureExecutingSubmission {
            executing: self.clone(),
        }
    }

    fn wait_until_idle(&self) {
        let mut count = self.count.lock().unwrap_or_else(|error| error.into_inner());
        while *count != 0 {
            count = self
                .idle
                .wait(count)
                .unwrap_or_else(|error| error.into_inner());
        }
    }
}

struct CaptureExecutingSubmission {
    executing: Arc<CaptureExecutingSubmissions>,
}

impl Drop for CaptureExecutingSubmission {
    fn drop(&mut self) {
        let mut count = self
            .executing
            .count
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *count -= 1;
        if *count == 0 {
            self.executing.idle.notify_all();
        }
    }
}

/// Select the stream's ownership phase before resource admission. A captured
/// descriptor retains its exact target but does not keep capture executing.
pub(crate) struct StreamSubmissionPhase {
    state: StreamSubmissionState,
}

/// Keeps one actual submission admitted until its enqueue callback returns.
/// The borrowed phase retains ordinary capture exclusion or captured owners.
#[must_use]
pub(crate) struct StreamSubmissionPin<'a> {
    _executing: Option<CaptureExecutingSubmission>,
    owners: Option<&'a CaptureOwners>,
}

fn capture_outcome<T, E>(
    owners: Option<&CaptureOwners>,
    submit: impl FnOnce() -> std::result::Result<T, E>,
) -> std::result::Result<T, E> {
    struct Outcome<'a> {
        owners: Option<&'a CaptureOwners>,
        success: bool,
    }
    impl Drop for Outcome<'_> {
        fn drop(&mut self) {
            if !self.success {
                if let Some(owners) = self.owners {
                    owners
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .failed = true;
                }
            }
        }
    }
    let mut outcome = Outcome {
        owners,
        success: false,
    };
    let result = submit();
    outcome.success = result.is_ok();
    result
}

impl StreamSubmissionPin<'_> {
    pub(crate) fn with_serialized_submission<T, E>(
        &self,
        submit: impl FnOnce() -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E> {
        let gate = self.owners.map(|owners| {
            owners
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .submission
                .clone()
        });
        let _serial = gate
            .as_ref()
            .map(|gate| gate.lock().unwrap_or_else(|error| error.into_inner()));
        capture_outcome(self.owners, submit)
    }

    pub(crate) fn capture_memory(
        &self,
        manifest: &Arc<crate::memory::MemoryAccessManifest>,
    ) -> bool {
        if let Some(owners) = self.owners {
            owners
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .memory
                .push(manifest.clone());
            true
        } else {
            false
        }
    }

    pub(crate) fn with_submission<T>(
        &self,
        submit: impl FnOnce(Option<&CaptureOwners>) -> std::result::Result<T, DriverError>,
    ) -> std::result::Result<T, DriverError> {
        // The parent already authenticated and pinned the exact capture. Its
        // close waits for this borrow; nested work must not acquire admission
        // again after that close has forbidden independent submissions.
        capture_outcome(self.owners, || submit(self.owners))
    }
}

enum StreamSubmissionState {
    Ordinary {
        _reservation: CaptureExclusionReservation,
    },
    Captured {
        context: usize,
        id: u64,
        owners: CaptureOwners,
    },
}

impl StreamSubmissionPhase {
    fn from_capture_info(
        mut registry: std::sync::MutexGuard<'static, StreamCaptureRegistry>,
        context: usize,
        status: sys::CUstreamCaptureStatus,
        id: u64,
    ) -> std::result::Result<Self, DriverError> {
        let state = match status {
            sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE => {
                // Existing captures on other streams may continue. The query
                // proved this stream ordinary; exclude any new managed begin.
                StreamSubmissionState::Ordinary {
                    _reservation: CaptureExclusionReservation::acquire(&mut registry),
                }
            }
            sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE => {
                let entry = registry.modules.get(&(context, id)).ok_or(DriverError(
                    sys::CUresult::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
                ))?;
                if entry.closing {
                    return Err(DriverError(
                        sys::CUresult::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED,
                    ));
                }
                StreamSubmissionState::Captured {
                    context,
                    id,
                    owners: entry.owners.clone(),
                }
            }
            _ => {
                return Err(DriverError(
                    sys::CUresult::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED,
                ));
            }
        };
        Ok(Self { state })
    }

    /// Pin only the actual enqueue callback, after checking that the exact
    /// capture and owner target remain open. No registry mutex crosses user or
    /// driver submission code, and unwind releases the executing pin as well.
    pub(crate) fn pin(&self) -> std::result::Result<StreamSubmissionPin<'_>, DriverError> {
        match &self.state {
            StreamSubmissionState::Ordinary { .. } => Ok(StreamSubmissionPin {
                _executing: None,
                owners: None,
            }),
            StreamSubmissionState::Captured {
                context,
                id,
                owners,
            } => {
                let executing = {
                    let registry = stream_capture_registry();
                    let entry = registry
                        .modules
                        .get(&(*context, *id))
                        .filter(|entry| !entry.closing && Arc::ptr_eq(&entry.owners, owners))
                        .ok_or(DriverError(
                            sys::CUresult::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED,
                        ))?;
                    entry.executing.pin()
                };
                Ok(StreamSubmissionPin {
                    _executing: Some(executing),
                    owners: Some(owners),
                })
            }
        }
    }

    pub(crate) fn with_submission<T>(
        &self,
        submit: impl FnOnce(Option<&CaptureOwners>) -> std::result::Result<T, DriverError>,
    ) -> std::result::Result<T, DriverError> {
        let pin = self.pin()?;
        pin.with_serialized_submission(|| pin.with_submission(submit))
    }
}

pub(crate) fn acquire_stream_submission_phase(
    stream: &CudaStream,
) -> std::result::Result<StreamSubmissionPhase, DriverError> {
    stream.context().bind_to_thread()?;
    let registry = stream_capture_registry();
    let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    let mut id = 0;
    // SAFETY: the caller owns the stream and its bound context. Managed begin
    // is serialized with this query and ordinary-exclusion acquisition; captured
    // submission revalidates the identity before it obtains an executing pin.
    unsafe {
        sys::cuStreamGetCaptureInfo_v2(
            stream.cu_stream(),
            &mut status,
            &mut id,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
        .result()?;
    }
    StreamSubmissionPhase::from_capture_info(
        registry,
        stream.context().cu_ctx() as usize,
        status,
        id,
    )
}

/// Called by the common kernel and graph launch boundaries before handing an
/// executable to CUDA. The graph owns exact modules, independently of names.
pub(crate) fn submit_with_capture(
    stream: &CudaStream,
    submit: impl FnOnce(Option<&CaptureOwners>) -> std::result::Result<(), DriverError>,
) -> std::result::Result<(), DriverError> {
    acquire_stream_submission_phase(stream)?.with_submission(submit)
}

static ACTIVE_STREAM_CAPTURES: OnceLock<Mutex<StreamCaptureRegistry>> = OnceLock::new();

fn stream_capture_registry() -> std::sync::MutexGuard<'static, StreamCaptureRegistry> {
    ACTIVE_STREAM_CAPTURES
        .get_or_init(|| Mutex::new(StreamCaptureRegistry::default()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// One exclusive driver-mutation reservation in the capture lifecycle owner.
/// Never hold the registry mutex while executing driver calls or destructors.
struct DriverMutationReservation {
    active: bool,
    current: Option<DeferredRetirement>,
    retry: VecDeque<DeferredRetirement>,
}

impl Drop for DriverMutationReservation {
    fn drop(&mut self) {
        if self.active {
            // Restore unfinished tasks to the same authoritative queue, including
            // the executing retry closure if its driver boundary unwound. No
            // destructor or driver operation runs under this mutex.
            let mut registry = stream_capture_registry();
            registry.deferred.append(&mut self.retry);
            if matches!(self.current, Some(DeferredRetirement::Retry(_))) {
                registry
                    .deferred
                    .push_back(self.current.take().expect("executing retirement present"));
            }
            registry.retiring = false;
        }
    }
}

pub(crate) fn with_module_loading<T>(
    load: impl FnOnce() -> std::result::Result<T, DriverError>,
) -> std::result::Result<T, DriverError> {
    drain_capture_retirements(stream_capture_registry());
    let reservation = reserve_capture_exclusion()?;
    let loaded = load();
    drop(reservation);
    drain_capture_retirements(stream_capture_registry());
    loaded
}

pub(crate) fn reserve_capture_exclusion(
) -> std::result::Result<CaptureExclusionReservation, DriverError> {
    let mut registry = stream_capture_registry();
    if !registry.active.is_empty() {
        return Err(DriverError(
            sys::CUresult::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
        ));
    }
    Ok(CaptureExclusionReservation::acquire(&mut registry))
}

/// Excludes managed capture while an ordinary resident operation submits or waits.
/// No mutex is held by this reservation and its release does not drain the cold
/// retirement queue. It does not prevent a cold caller from retiring resources.
#[doc(hidden)]
pub struct CaptureExclusionReservation {
    _private: (),
}

impl CaptureExclusionReservation {
    fn acquire(registry: &mut StreamCaptureRegistry) -> Self {
        registry.capture_exclusions += 1;
        Self { _private: () }
    }
}

impl Drop for CaptureExclusionReservation {
    fn drop(&mut self) {
        stream_capture_registry().capture_exclusions -= 1;
    }
}

/// Admit an ordinary operation on a stream, never a captured enqueue or wait.
/// The reservation closes the check-to-submit race against managed BeginCapture.
/// Foreign/raw capture must be externally serialized with XLOG stream operations.
/// This call does not drain deferred destruction or add a blocking hot-loop wait.
#[doc(hidden)]
pub fn reserve_uncaptured_stream(
    stream: &CudaStream,
) -> std::result::Result<CaptureExclusionReservation, DriverError> {
    let reservation = reserve_capture_exclusion()?;
    stream.context().bind_to_thread()?;
    let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    // SAFETY: the stream and context remain owned by the caller; the registry
    // reservation excludes managed capture until the ordinary operation returns.
    unsafe {
        sys::cuStreamIsCapturing(stream.cu_stream(), &mut status).result()?;
    }
    match status {
        sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE => Ok(reservation),
        sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE => Err(DriverError(
            sys::CUresult::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
        )),
        _ => Err(DriverError(
            sys::CUresult::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED,
        )),
    }
}

/// Run destruction only outside all captures managed by this module. The
/// reservation also excludes new captures until destruction has returned.
/// Closures must not begin capture; nested retirement is queued. An unwind
/// releases the reservation without discarding other queued resources.
/// The call that drains the queue may wait, including a later capture release
/// or module load. Its caller must satisfy this module's cold-lifecycle contract.
#[doc(hidden)]
pub fn retire_after_stream_captures(retire: impl FnOnce() + Send + 'static) {
    enqueue_retirement(DeferredRetirement::Once(Some(Box::new(retire))));
}

/// Retain an actual resource until its release can be proved. A false result
/// leaves this same callable owner queued for the next cold lifecycle drain;
/// it is never retried in a tight loop. Unwind preserves it for a later drain.
pub(crate) fn retry_retirement_after_stream_captures(
    retire: impl FnMut() -> bool + Send + 'static,
) {
    enqueue_retirement(DeferredRetirement::Retry(Box::new(retire)));
}

fn enqueue_retirement(retire: DeferredRetirement) {
    let mut registry = stream_capture_registry();
    registry.deferred.push_back(retire);
    drain_capture_retirements(registry);
}

/// Explicit cold reaper for resources whose last allocator owner is gone.
/// Like capture release/module loading, it may wait and must run without caller
/// locks needed by pending work. Active captures leave the queue untouched.
pub(crate) fn reap_capture_retirements() {
    drain_capture_retirements(stream_capture_registry());
}

// Tests that assert a particular global capture/retirement state must not
// interleave independent scenarios. Worker threads within a scenario remain
// concurrent and exercise the production queue without this test-only lock.
#[cfg(test)]
pub(crate) fn capture_lifecycle_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner())
}

fn drain_capture_retirements(mut registry: std::sync::MutexGuard<'static, StreamCaptureRegistry>) {
    if !registry.active.is_empty() || registry.retiring || registry.deferred.is_empty() {
        return;
    }
    registry.retiring = true;
    let mut reservation = DriverMutationReservation {
        active: true,
        current: None,
        retry: VecDeque::new(),
    };
    loop {
        let Some(retire) = registry.deferred.pop_front() else {
            registry.deferred.append(&mut reservation.retry);
            registry.retiring = false;
            reservation.active = false;
            return;
        };
        // No driver calls or resource destructors under the mutex. The retiring
        // reservation excludes begin-capture across this unlock/relock window.
        drop(registry);
        reservation.current = Some(retire);
        let completed = reservation
            .current
            .as_mut()
            .expect("executing retirement present")
            .attempt();
        let retire = reservation
            .current
            .take()
            .expect("executing retirement present");
        if completed {
            drop(retire);
        } else {
            reservation.retry.push_back(retire);
        }
        registry = stream_capture_registry();
    }
}

fn release_stream_capture_key(key: StreamCaptureKey) {
    let mut registry = stream_capture_registry();
    registry.active.remove(&key);
    drain_capture_retirements(registry);
}

#[derive(Debug)]
struct StreamCaptureLease {
    key: StreamCaptureKey,
    driver_active: bool,
    modules: CaptureOwners,
    capture_id: Option<u64>,
}

impl StreamCaptureLease {
    fn register_capture(&mut self, id: u64, registry: &mut StreamCaptureRegistry) {
        self.capture_id = Some(id);
        registry.modules.insert(
            (self.key.context, id),
            CaptureSubmissionEntry {
                owners: self.modules.clone(),
                closing: false,
                executing: Arc::default(),
            },
        );
    }

    fn close_submission_admission(&self) -> Option<Arc<CaptureExecutingSubmissions>> {
        let id = self.capture_id?;
        let mut registry = stream_capture_registry();
        let entry = registry.modules.get_mut(&(self.key.context, id))?;
        if !Arc::ptr_eq(&entry.owners, &self.modules) {
            return None;
        }
        entry.closing = true;
        Some(entry.executing.clone())
    }

    fn complete_driver_capture(&mut self) {
        self.driver_active = false;
        if let Some(id) = self.capture_id.take() {
            let removed = stream_capture_registry()
                .modules
                .remove(&(self.key.context, id));
            // Release the registry entry outside its mutex. The lease and any
            // returned graph still own the captured resource target.
            drop(removed);
        }
    }
}

fn try_acquire_stream_capture_key(
    key: StreamCaptureKey,
) -> std::result::Result<StreamCaptureLease, CudaConditionalGraphUnavailable> {
    drain_capture_retirements(stream_capture_registry());
    let mut registry = stream_capture_registry();
    if registry.retiring || registry.capture_exclusions != 0 || !registry.active.insert(key) {
        return Err(CudaConditionalGraphUnavailable::StreamCaptureBusy);
    }
    Ok(StreamCaptureLease {
        key,
        driver_active: false,
        modules: Arc::default(),
        capture_id: None,
    })
}

fn try_acquire_stream_capture(
    stream: &CudaStream,
) -> std::result::Result<StreamCaptureLease, CudaConditionalGraphUnavailable> {
    let context =
        stream_context(stream).map_err(CudaConditionalGraphUnavailable::body_population)?;
    try_acquire_stream_capture_key(StreamCaptureKey {
        context: context as usize,
        stream: stream_execution_id(stream)
            .map_err(CudaConditionalGraphUnavailable::body_population)?,
    })
}

impl Drop for StreamCaptureLease {
    fn drop(&mut self) {
        if !self.driver_active {
            release_stream_capture_key(self.key);
        }
    }
}

/// Keeps the host reservation until the driver has actually left capture.
/// A callback unwind takes the same end-capture path as an ordinary return.
struct DriverStreamCapture<'a> {
    lease: StreamCaptureLease,
    stream: &'a CudaStream,
    owns_returned_graph: bool,
}

impl<'a> DriverStreamCapture<'a> {
    fn prepare(
        stream: &'a CudaStream,
        owns_returned_graph: bool,
        modules: CaptureOwners,
    ) -> std::result::Result<Self, CudaConditionalGraphUnavailable> {
        let mut lease = try_acquire_stream_capture(stream)?;
        lease.modules = modules.clone();
        Ok(Self {
            lease,
            stream,
            owns_returned_graph,
        })
    }

    fn begin(
        &mut self,
        operation: &'static str,
        begin: impl FnOnce() -> sys::CUresult,
    ) -> std::result::Result<(), CudaConditionalGraphUnavailable> {
        self.stream
            .context()
            .bind_to_thread()
            .map_err(CudaConditionalGraphUnavailable::body_population)?;
        let mut registry = stream_capture_registry();
        // An ordinary phase may have been acquired after this capture reserved
        // its stream but before reaching the driver begin boundary.
        if registry.capture_exclusions != 0 {
            return Err(CudaConditionalGraphUnavailable::StreamCaptureBusy);
        }
        conditional_driver_call(operation, begin())?;
        self.lease.driver_active = true;
        let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let mut id = 0;
        conditional_driver_call("cuStreamGetCaptureInfo_v2", unsafe {
            sys::cuStreamGetCaptureInfo_v2(
                self.stream.cu_stream(),
                &mut status,
                &mut id,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        })?;
        if status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
            return Err(CudaConditionalGraphUnavailable::body_population(
                "begin-capture did not enter active capture",
            ));
        }
        self.lease.register_capture(id, &mut registry);
        Ok(())
    }

    fn finish(&mut self) -> std::result::Result<sys::CUgraph, CudaConditionalGraphUnavailable> {
        if let Some(executing) = self.lease.close_submission_admission() {
            // Saved descriptors are passive. Only enqueues that already passed
            // admission may delay EndCapture, without blocking registry access.
            executing.wait_until_idle();
        }
        self.stream
            .context()
            .bind_to_thread()
            .map_err(CudaConditionalGraphUnavailable::body_population)?;
        let mut graph = ptr::null_mut();
        // SAFETY: BeginCapture succeeded on this stream and thread, the stream
        // is borrowed for this guard's lifetime, and its context is now bound.
        let code = unsafe { sys::cuStreamEndCapture(self.stream.cu_stream(), &mut graph) };
        let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE;
        let driver_active = code != sys::CUresult::CUDA_SUCCESS
            && !(unsafe { sys::cuStreamIsCapturing(self.stream.cu_stream(), &mut status) }
                == sys::CUresult::CUDA_SUCCESS
                && status == sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE);
        if !driver_active {
            self.lease.complete_driver_capture();
        }
        conditional_driver_call("cuStreamEndCapture", code)?;
        if self
            .lease
            .modules
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .failed
        {
            if self.owns_returned_graph && !graph.is_null() {
                drop(UninstantiatedCudaGraph {
                    raw: graph,
                    context: self.stream.context().clone(),
                    modules: self.lease.modules.clone(),
                });
            }
            return Err(CudaConditionalGraphUnavailable::body_population(
                "a captured submission failed or unwound; the graph cannot be instantiated",
            ));
        }
        Ok(graph)
    }
}

impl Drop for DriverStreamCapture<'_> {
    fn drop(&mut self) {
        if self.lease.driver_active {
            match self.finish() {
                Ok(graph) if self.owns_returned_graph && !graph.is_null() => {
                    drop(UninstantiatedCudaGraph {
                        raw: graph,
                        context: self.stream.context().clone(),
                        modules: self.lease.modules.clone(),
                    });
                }
                _ => {}
            }
        }
        if self.lease.driver_active {
            // Unknown completion is not permission to release the reservation
            // or context. Destruction queued behind this capture stays retained.
            let context = self.stream.context().clone();
            let modules = self.lease.modules.clone();
            retire_after_stream_captures(move || drop((context, modules)));
        }
    }
}

impl fmt::Display for CudaConditionalGraphUnavailable {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.decline_detail())
    }
}

impl std::error::Error for CudaConditionalGraphUnavailable {}

impl From<XlogError> for CudaConditionalGraphUnavailable {
    fn from(error: XlogError) -> Self {
        Self::body_population(error)
    }
}

struct ConditionalGraphDriverApi {
    _library: Arc<Library>,
    driver_get_version: DriverGetVersionFn,
    conditional_handle_create: ConditionalHandleCreateFn,
    graph_add_node: GraphAddNodeFn,
}

impl ConditionalGraphDriverApi {
    fn load() -> std::result::Result<Arc<Self>, CudaConditionalGraphUnavailable> {
        #[cfg(target_os = "windows")]
        const CUDA_DRIVER_LIBRARY: &str = "nvcuda.dll";
        #[cfg(not(target_os = "windows"))]
        const CUDA_DRIVER_LIBRARY: &str = "libcuda.so.1";

        let library = Arc::new(
            unsafe { Library::new(CUDA_DRIVER_LIBRARY) }
                .map_err(|_| CudaConditionalGraphUnavailable::DriverLibraryUnavailable)?,
        );
        let driver_get_version = unsafe {
            load_required_symbol(&library, b"cuDriverGetVersion\0", "cuDriverGetVersion")?
        };
        let conditional_handle_create = unsafe {
            load_required_symbol(
                &library,
                b"cuGraphConditionalHandleCreate\0",
                "cuGraphConditionalHandleCreate",
            )?
        };
        let graph_add_node =
            unsafe { load_required_symbol(&library, b"cuGraphAddNode\0", "cuGraphAddNode")? };

        let api = Arc::new(Self {
            _library: library,
            driver_get_version,
            conditional_handle_create,
            graph_add_node,
        });
        api.require_supported_driver()?;
        Ok(api)
    }

    fn require_supported_driver(&self) -> std::result::Result<(), CudaConditionalGraphUnavailable> {
        let mut version = 0;
        let code = unsafe { (self.driver_get_version)(&mut version) };
        if code != sys::CUresult::CUDA_SUCCESS {
            return Err(CudaConditionalGraphUnavailable::DriverVersionQueryFailed { code });
        }
        require_conditional_graph_driver(version)
    }
}

unsafe fn load_required_symbol<F: Copy>(
    library: &Library,
    name: &'static [u8],
    display_name: &'static str,
) -> std::result::Result<F, CudaConditionalGraphUnavailable> {
    library.get::<F>(name).map(|symbol| *symbol).map_err(|_| {
        CudaConditionalGraphUnavailable::MissingDriverSymbol {
            symbol: display_name,
        }
    })
}

fn require_conditional_graph_driver(
    version: i32,
) -> std::result::Result<(), CudaConditionalGraphUnavailable> {
    if version < CONDITIONAL_GRAPH_MINIMUM_DRIVER {
        Err(CudaConditionalGraphUnavailable::DriverVersionTooOld {
            found: version,
            required: CONDITIONAL_GRAPH_MINIMUM_DRIVER,
        })
    } else {
        Ok(())
    }
}

fn conditional_node_params(
    handle: sys::CUgraphConditionalHandle,
    ctx: sys::CUcontext,
    kind: sys::CUgraphConditionalNodeType,
) -> sys::CUgraphNodeParams {
    let mut params: sys::CUgraphNodeParams = unsafe { mem::zeroed() };
    params.type_ = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL;
    params.__bindgen_anon_1.conditional = sys::CUDA_CONDITIONAL_NODE_PARAMS {
        handle,
        type_: kind,
        size: 1,
        phGraph_out: ptr::null_mut(),
        ctx,
    };
    params
}

fn conditional_driver_call(
    operation: &'static str,
    code: sys::CUresult,
) -> std::result::Result<(), CudaConditionalGraphUnavailable> {
    if code == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CudaConditionalGraphUnavailable::DriverCallFailed { operation, code })
    }
}

struct UninstantiatedCudaGraph {
    raw: sys::CUgraph,
    context: Arc<CudaContext>,
    modules: CaptureOwners,
}

impl UninstantiatedCudaGraph {
    fn create(
        context: Arc<CudaContext>,
    ) -> std::result::Result<Self, CudaConditionalGraphUnavailable> {
        context
            .bind_to_thread()
            .map_err(CudaConditionalGraphUnavailable::body_population)?;
        let mut raw = ptr::null_mut();
        unsafe {
            conditional_driver_call("cuGraphCreate", sys::cuGraphCreate(&mut raw, 0))?;
        }
        if raw.is_null() {
            Err(CudaConditionalGraphUnavailable::NullDriverHandle {
                operation: "cuGraphCreate",
            })
        } else {
            Ok(Self {
                raw,
                context,
                modules: Arc::default(),
            })
        }
    }

    fn raw(&self) -> sys::CUgraph {
        self.raw
    }

    fn into_raw(mut self) -> sys::CUgraph {
        let raw = self.raw;
        self.raw = ptr::null_mut();
        raw
    }
}

impl Drop for UninstantiatedCudaGraph {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let graph = self.raw as usize;
            let context = self.context.clone();
            let modules = self.modules.clone();
            retire_resources_after_completion(
                (graph, context, modules),
                |(_, context, _)| context.bind_to_thread(),
                |(graph, _, _)| unsafe { sys::cuGraphDestroy(*graph as sys::CUgraph).result() },
                |_| Ok(()),
                |(_, context, _), error| context.record_err::<()>(Err(error)),
            );
        }
    }
}

/// CUDA-owned body graph produced while adding one conditional-WHILE node.
///
/// The body graph and conditional handle are owned by the parent graph. They
/// must not be destroyed independently. The numeric handle is intended to be
/// passed by value to a device kernel that calls `cudaGraphSetConditional`.
#[derive(Debug)]
pub struct ConditionalCudaGraphBody {
    graph: sys::CUgraph,
    handle: sys::CUgraphConditionalHandle,
    context: sys::CUcontext,
    modules: CaptureOwners,
}

impl ConditionalCudaGraphBody {
    pub fn graph(&self) -> sys::CUgraph {
        self.graph
    }

    pub fn handle(&self) -> sys::CUgraphConditionalHandle {
        self.handle
    }

    pub fn context(&self) -> sys::CUcontext {
        self.context
    }

    /// Return this body's actual node kinds in dependency-chain order.
    ///
    /// The body must be a single linear dependency chain. CUDA's node-list
    /// enumeration order is not used as an execution-order signal.
    pub fn linear_chain_node_kinds(
        &self,
    ) -> std::result::Result<Vec<CudaGraphNodeKind>, CudaConditionalGraphUnavailable> {
        let mut check = |operation, code| conditional_driver_call(operation, code);
        let mut shape_error = |error| CudaConditionalGraphUnavailable::BodyPopulationFailed {
            detail: format!("conditional graph body is not a linear dependency chain: {error}"),
        };
        graph_linear_chain_node_kinds_with(self.graph, &mut check, &mut shape_error)
    }

    /// Capture graph-compatible work directly into this conditional body.
    ///
    /// `stream` must be a non-default stream. The callback must not allocate or
    /// free CUDA memory, synchronize, or record/wait on events. Host-only
    /// suballocation from retained device storage is allowed if it cannot grow
    /// or remap that storage; captured addresses must remain valid and must not
    /// be recycled while graph consumers or saved backward values need them.
    /// CUDA conditional bodies allow only kernel, empty, child-graph, device
    /// memcpy/memset, and nested conditional nodes.
    /// Capture setup and cleanup may wait for deferred destruction under the
    /// [module-level lifecycle contract](crate::cuda_graph).
    pub fn capture_on_stream<F, E>(
        &self,
        stream: &CudaStream,
        record: F,
    ) -> std::result::Result<(), CudaConditionalGraphUnavailable>
    where
        F: FnOnce() -> std::result::Result<(), E>,
        E: fmt::Display,
    {
        let owners = self.modules.clone();
        capture_outcome(Some(&owners), || {
            let stream_ctx =
                stream_context(stream).map_err(CudaConditionalGraphUnavailable::body_population)?;
            if stream_ctx != self.context {
                return Err(CudaConditionalGraphUnavailable::ContextMismatch);
            }
            let mut capture = DriverStreamCapture::prepare(stream, false, self.modules.clone())?;

            unsafe {
                capture.begin("cuStreamBeginCaptureToGraph", || {
                    sys::cuStreamBeginCaptureToGraph(
                        stream.cu_stream(),
                        self.graph,
                        ptr::null(),
                        ptr::null(),
                        0,
                        sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                    )
                })?;
            }

            let record_result = capture_outcome(Some(&capture.lease.modules), record);
            let end_result = capture.finish();
            if let Err(error) = record_result {
                return Err(CudaConditionalGraphUnavailable::body_population(error));
            }
            let captured = end_result?;
            if captured.is_null() {
                return Err(CudaConditionalGraphUnavailable::NullDriverHandle {
                    operation: "cuStreamEndCapture",
                });
            }
            if captured != self.graph {
                return Err(CudaConditionalGraphUnavailable::BodyPopulationFailed {
                    detail: "capture returned a graph other than the conditional body".to_string(),
                });
            }
            Ok(())
        })
    }
}

/// Builds one dependency-ordered parent graph containing ordinary captured
/// segments, one-shot conditional-IF bodies and conditional-WHILE loops.
///
/// Stratified Datalog uses this topology so work before a
/// recursive strongly connected component executes once, only that component
/// is placed in a device-controlled WHILE, and later strata depend on its
/// completion. An IF body instead admits or skips a whole bounded step after
/// its device preflight. The finished value launches through one `cuGraphLaunch`.
pub struct ConditionalCudaGraphSequenceBuilder {
    graph: UninstantiatedCudaGraph,
    raw_context: sys::CUcontext,
    api: Arc<ConditionalGraphDriverApi>,
    frontier: Vec<sys::CUgraphNode>,
}

impl ConditionalCudaGraphSequenceBuilder {
    /// Retain original resources and the native stream before recording any
    /// segment. They follow the same graph ownership and uncertain-retirement
    /// path as captured modules and memory manifests.
    pub(crate) fn new_retaining(
        stream: &Arc<CudaStream>,
        mut external_resources: Vec<Arc<dyn Send + Sync>>,
    ) -> std::result::Result<Self, CudaConditionalGraphUnavailable> {
        let builder = Self::new(stream)?;
        external_resources.push(stream.clone());
        builder
            .graph
            .modules
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .external_resources
            .extend(external_resources);
        Ok(builder)
    }

    /// Create an empty parent graph bound to `stream`'s CUDA context.
    pub fn new(stream: &CudaStream) -> std::result::Result<Self, CudaConditionalGraphUnavailable> {
        let api = ConditionalGraphDriverApi::load()?;
        let context = stream.context().clone();
        let raw_context =
            stream_context(stream).map_err(CudaConditionalGraphUnavailable::body_population)?;
        if raw_context != context.cu_ctx() {
            return Err(CudaConditionalGraphUnavailable::ContextMismatch);
        }
        Ok(Self {
            graph: UninstantiatedCudaGraph::create(Arc::clone(&context))?,
            raw_context,
            api,
            frontier: Vec::new(),
        })
    }

    /// Capture one ordinary graph segment after the current dependency
    /// frontier. The callback may enqueue only capture-compatible operations.
    /// Capture setup and cleanup may wait for deferred destruction under the
    /// [module-level lifecycle contract](crate::cuda_graph).
    pub fn capture_segment_on_stream<F, E>(
        &mut self,
        stream: &CudaStream,
        record: F,
    ) -> std::result::Result<(), CudaConditionalGraphUnavailable>
    where
        F: FnOnce() -> std::result::Result<(), E>,
        E: fmt::Display,
    {
        let owners = self.graph.modules.clone();
        capture_outcome(Some(&owners), || {
            self.ensure_stream_context(stream)?;
            let mut capture =
                DriverStreamCapture::prepare(stream, false, self.graph.modules.clone())?;
            unsafe {
                capture.begin("cuStreamBeginCaptureToGraph", || {
                    sys::cuStreamBeginCaptureToGraph(
                        stream.cu_stream(),
                        self.graph.raw(),
                        if self.frontier.is_empty() {
                            ptr::null()
                        } else {
                            self.frontier.as_ptr()
                        },
                        ptr::null(),
                        self.frontier.len(),
                        sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                    )
                })?;
            }

            let record_result = capture_outcome(Some(&capture.lease.modules), record);
            let end_result = capture.finish();
            if let Err(error) = record_result {
                return Err(CudaConditionalGraphUnavailable::body_population(error));
            }
            let captured = end_result?;
            if captured != self.graph.raw() {
                return Err(CudaConditionalGraphUnavailable::BodyPopulationFailed {
                    detail: "segment capture returned a graph other than its parent".to_string(),
                });
            }
            self.frontier = graph_leaf_nodes(self.graph.raw())?;
            Ok(())
        })
    }

    /// Append one conditional-WHILE node after the current frontier.
    pub fn add_conditional_while<F>(
        &mut self,
        initial_value: u32,
        assign_default_on_launch: bool,
        populate_body: F,
    ) -> std::result::Result<sys::CUgraphConditionalHandle, CudaConditionalGraphUnavailable>
    where
        F: FnOnce(
            &ConditionalCudaGraphBody,
        ) -> std::result::Result<(), CudaConditionalGraphUnavailable>,
    {
        let owners = self.graph.modules.clone();
        capture_outcome(Some(&owners), || {
            let handle = self.create_conditional_handle(initial_value, assign_default_on_launch)?;
            self.append_conditional_body(
                handle,
                sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_WHILE,
                populate_body,
            )?;
            Ok(handle)
        })
    }

    /// Capture a device admission test, then execute its body at most once.
    ///
    /// The preflight receives the actual graph handle and must set it on every
    /// replay. Its nodes precede the IF node on the same dependency frontier;
    /// a rejected step never enters the body. The handle defaults to false on
    /// each launch. Neither callback runs when the finished graph is replayed.
    pub fn add_conditional_if<P, F, E>(
        &mut self,
        stream: &CudaStream,
        preflight: P,
        populate_body: F,
    ) -> std::result::Result<sys::CUgraphConditionalHandle, CudaConditionalGraphUnavailable>
    where
        P: FnOnce(sys::CUgraphConditionalHandle) -> std::result::Result<(), E>,
        E: fmt::Display,
        F: FnOnce(
            &ConditionalCudaGraphBody,
        ) -> std::result::Result<(), CudaConditionalGraphUnavailable>,
    {
        let owners = self.graph.modules.clone();
        capture_outcome(Some(&owners), || {
            self.ensure_stream_context(stream)?;
            let handle = self.create_conditional_handle(0, true)?;
            self.capture_segment_on_stream(stream, || preflight(handle))?;
            self.append_conditional_body(
                handle,
                sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_IF,
                populate_body,
            )?;
            Ok(handle)
        })
    }

    fn create_conditional_handle(
        &self,
        initial_value: u32,
        assign_default_on_launch: bool,
    ) -> std::result::Result<sys::CUgraphConditionalHandle, CudaConditionalGraphUnavailable> {
        let mut handle = 0;
        let flags = if assign_default_on_launch {
            sys::CU_GRAPH_COND_ASSIGN_DEFAULT
        } else {
            0
        };
        unsafe {
            conditional_driver_call(
                "cuGraphConditionalHandleCreate",
                (self.api.conditional_handle_create)(
                    &mut handle,
                    self.graph.raw(),
                    self.raw_context,
                    initial_value,
                    flags,
                ),
            )?;
        }
        Ok(handle)
    }

    fn append_conditional_body<F>(
        &mut self,
        handle: sys::CUgraphConditionalHandle,
        kind: sys::CUgraphConditionalNodeType,
        populate_body: F,
    ) -> std::result::Result<(), CudaConditionalGraphUnavailable>
    where
        F: FnOnce(
            &ConditionalCudaGraphBody,
        ) -> std::result::Result<(), CudaConditionalGraphUnavailable>,
    {
        let mut params = conditional_node_params(handle, self.raw_context, kind);
        let mut conditional_node = ptr::null_mut();
        unsafe {
            conditional_driver_call(
                "cuGraphAddNode",
                (self.api.graph_add_node)(
                    &mut conditional_node,
                    self.graph.raw(),
                    if self.frontier.is_empty() {
                        ptr::null()
                    } else {
                        self.frontier.as_ptr()
                    },
                    self.frontier.len(),
                    &mut params,
                ),
            )?;
        }
        if conditional_node.is_null() {
            return Err(CudaConditionalGraphUnavailable::NullDriverHandle {
                operation: "cuGraphAddNode",
            });
        }
        let conditional = unsafe { params.__bindgen_anon_1.conditional };
        if conditional.phGraph_out.is_null() {
            return Err(CudaConditionalGraphUnavailable::NullDriverHandle {
                operation: "cuGraphAddNode body array",
            });
        }
        let body_graph = unsafe { *conditional.phGraph_out };
        if body_graph.is_null() {
            return Err(CudaConditionalGraphUnavailable::NullDriverHandle {
                operation: "cuGraphAddNode conditional body",
            });
        }
        populate_body(&ConditionalCudaGraphBody {
            graph: body_graph,
            handle,
            context: self.raw_context,
            modules: self.graph.modules.clone(),
        })?;
        self.frontier.clear();
        self.frontier.push(conditional_node);
        Ok(())
    }

    /// Instantiate the complete parent graph exactly once.
    pub fn instantiate(
        self,
    ) -> std::result::Result<CapturedCudaGraph, CudaConditionalGraphUnavailable> {
        let ConditionalCudaGraphSequenceBuilder { graph, api, .. } = self;
        let mut captured = CapturedCudaGraph::instantiate_graph(graph)?;
        captured._conditional_api = Some(api);
        Ok(captured)
    }

    fn ensure_stream_context(
        &self,
        stream: &CudaStream,
    ) -> std::result::Result<(), CudaConditionalGraphUnavailable> {
        let context =
            stream_context(stream).map_err(CudaConditionalGraphUnavailable::body_population)?;
        if context == self.raw_context {
            Ok(())
        } else {
            Err(CudaConditionalGraphUnavailable::ContextMismatch)
        }
    }
}

fn raw_graph_nodes_with<E>(
    graph: sys::CUgraph,
    check: &mut impl FnMut(&'static str, sys::CUresult) -> std::result::Result<(), E>,
) -> std::result::Result<Vec<sys::CUgraphNode>, E> {
    let mut node_count = 0usize;
    unsafe {
        check(
            "cuGraphGetNodes(count)",
            sys::cuGraphGetNodes(graph, ptr::null_mut(), &mut node_count),
        )?;
    }
    let mut nodes = vec![ptr::null_mut(); node_count];
    if node_count != 0 {
        unsafe {
            check(
                "cuGraphGetNodes(nodes)",
                sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut node_count),
            )?;
        }
        nodes.truncate(node_count);
    }
    Ok(nodes)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LinearGraphChainError {
    ForeignDependency { node: usize, dependency: usize },
    DuplicateDependency { node: usize, dependency: usize },
    Cycle,
    RootCount { found: usize },
    IncomingDegree { node: usize, dependencies: usize },
    Branch { node: usize, dependents: usize },
    LeafCount { found: usize },
    Disconnected { visited: usize, total: usize },
}

impl fmt::Display for LinearGraphChainError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ForeignDependency { node, dependency } => write!(
                formatter,
                "node {node} depends on foreign enumeration index {dependency}"
            ),
            Self::DuplicateDependency { node, dependency } => write!(
                formatter,
                "node {node} repeats dependency enumeration index {dependency}"
            ),
            Self::Cycle => formatter.write_str("dependency graph contains a cycle"),
            Self::RootCount { found } => {
                write!(
                    formatter,
                    "dependency graph has {found} roots instead of one"
                )
            }
            Self::IncomingDegree { node, dependencies } => write!(
                formatter,
                "non-root node {node} has {dependencies} immediate dependencies instead of one"
            ),
            Self::Branch { node, dependents } => write!(
                formatter,
                "node {node} has {dependents} immediate dependents instead of at most one"
            ),
            Self::LeafCount { found } => {
                write!(
                    formatter,
                    "dependency graph has {found} leaves instead of one"
                )
            }
            Self::Disconnected { visited, total } => write!(
                formatter,
                "dependency chain visits {visited} of {total} enumerated nodes"
            ),
        }
    }
}

fn linear_chain_order(
    immediate_dependencies: &[Vec<usize>],
) -> std::result::Result<Vec<usize>, LinearGraphChainError> {
    let node_count = immediate_dependencies.len();
    let mut outgoing = vec![Vec::new(); node_count];
    for (node, dependencies) in immediate_dependencies.iter().enumerate() {
        let mut unique = HashSet::with_capacity(dependencies.len());
        for &dependency in dependencies {
            if dependency >= node_count {
                return Err(LinearGraphChainError::ForeignDependency { node, dependency });
            }
            if !unique.insert(dependency) {
                return Err(LinearGraphChainError::DuplicateDependency { node, dependency });
            }
            outgoing[dependency].push(node);
        }
    }

    let mut remaining_indegree = immediate_dependencies
        .iter()
        .map(Vec::len)
        .collect::<Vec<_>>();
    let mut ready = remaining_indegree
        .iter()
        .enumerate()
        .filter_map(|(node, &degree)| (degree == 0).then_some(node))
        .collect::<Vec<_>>();
    let mut acyclic_nodes = 0usize;
    while let Some(node) = ready.pop() {
        acyclic_nodes += 1;
        for &dependent in &outgoing[node] {
            remaining_indegree[dependent] -= 1;
            if remaining_indegree[dependent] == 0 {
                ready.push(dependent);
            }
        }
    }
    if acyclic_nodes != node_count {
        return Err(LinearGraphChainError::Cycle);
    }

    let roots = immediate_dependencies
        .iter()
        .enumerate()
        .filter_map(|(node, dependencies)| dependencies.is_empty().then_some(node))
        .collect::<Vec<_>>();
    if roots.len() != 1 {
        return Err(LinearGraphChainError::RootCount { found: roots.len() });
    }
    let root = roots[0];
    for (node, dependencies) in immediate_dependencies.iter().enumerate() {
        if node != root && dependencies.len() != 1 {
            return Err(LinearGraphChainError::IncomingDegree {
                node,
                dependencies: dependencies.len(),
            });
        }
    }
    for (node, dependents) in outgoing.iter().enumerate() {
        if dependents.len() > 1 {
            return Err(LinearGraphChainError::Branch {
                node,
                dependents: dependents.len(),
            });
        }
    }
    let leaf_count = outgoing
        .iter()
        .filter(|dependents| dependents.is_empty())
        .count();
    if leaf_count != 1 {
        return Err(LinearGraphChainError::LeafCount { found: leaf_count });
    }

    let mut order = Vec::with_capacity(node_count);
    let mut visited = vec![false; node_count];
    let mut current = Some(root);
    while let Some(node) = current {
        if visited[node] {
            return Err(LinearGraphChainError::Cycle);
        }
        visited[node] = true;
        order.push(node);
        current = outgoing[node].first().copied();
    }
    if order.len() != node_count {
        return Err(LinearGraphChainError::Disconnected {
            visited: order.len(),
            total: node_count,
        });
    }
    Ok(order)
}

fn raw_node_dependencies_with<E>(
    node: sys::CUgraphNode,
    check: &mut impl FnMut(&'static str, sys::CUresult) -> std::result::Result<(), E>,
) -> std::result::Result<Vec<sys::CUgraphNode>, E> {
    let mut dependency_count = 0usize;
    unsafe {
        check(
            "cuGraphNodeGetDependencies(count)",
            sys::cuGraphNodeGetDependencies(node, ptr::null_mut(), &mut dependency_count),
        )?;
    }
    let mut dependencies = vec![ptr::null_mut(); dependency_count];
    if dependency_count != 0 {
        unsafe {
            check(
                "cuGraphNodeGetDependencies(nodes)",
                sys::cuGraphNodeGetDependencies(
                    node,
                    dependencies.as_mut_ptr(),
                    &mut dependency_count,
                ),
            )?;
        }
        dependencies.truncate(dependency_count);
    }
    Ok(dependencies)
}

fn graph_nodes_with<E>(
    graph: sys::CUgraph,
    check: &mut impl FnMut(&'static str, sys::CUresult) -> std::result::Result<(), E>,
) -> std::result::Result<Vec<CudaGraphNode>, E> {
    let raw_nodes = raw_graph_nodes_with(graph, check)?;
    let mut nodes = Vec::with_capacity(raw_nodes.len());
    for (index, raw) in raw_nodes.into_iter().enumerate() {
        let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        unsafe {
            check("cuGraphNodeGetType", sys::cuGraphNodeGetType(raw, &mut ty))?;
        }
        nodes.push(CudaGraphNode {
            index,
            raw,
            kind: CudaGraphNodeKind::from_sys(ty),
        });
    }
    Ok(nodes)
}

fn graph_linear_chain_node_kinds_with<E>(
    graph: sys::CUgraph,
    check: &mut impl FnMut(&'static str, sys::CUresult) -> std::result::Result<(), E>,
    shape_error: &mut impl FnMut(LinearGraphChainError) -> E,
) -> std::result::Result<Vec<CudaGraphNodeKind>, E> {
    let nodes = graph_nodes_with(graph, check)?;
    let mut immediate_dependencies = Vec::with_capacity(nodes.len());
    for (node_index, node) in nodes.iter().enumerate() {
        let raw_dependencies = raw_node_dependencies_with(node.raw, check)?;
        let mut dependency_indices = Vec::with_capacity(raw_dependencies.len());
        for dependency in raw_dependencies {
            let Some(dependency_index) = nodes
                .iter()
                .position(|candidate| candidate.raw == dependency)
            else {
                return Err(shape_error(LinearGraphChainError::ForeignDependency {
                    node: node_index,
                    dependency: nodes.len(),
                }));
            };
            dependency_indices.push(dependency_index);
        }
        immediate_dependencies.push(dependency_indices);
    }
    let order = linear_chain_order(&immediate_dependencies).map_err(shape_error)?;
    Ok(order.into_iter().map(|index| nodes[index].kind).collect())
}

fn graph_leaf_nodes(
    graph: sys::CUgraph,
) -> std::result::Result<Vec<sys::CUgraphNode>, CudaConditionalGraphUnavailable> {
    let mut check = |_, code| conditional_driver_call("cuGraphGetNodes", code);
    let mut nodes = raw_graph_nodes_with(graph, &mut check)?;

    let mut edge_count = 0usize;
    unsafe {
        conditional_driver_call(
            "cuGraphGetEdges",
            sys::cuGraphGetEdges(graph, ptr::null_mut(), ptr::null_mut(), &mut edge_count),
        )?;
    }
    let mut from = vec![ptr::null_mut(); edge_count];
    let mut to = vec![ptr::null_mut(); edge_count];
    if edge_count != 0 {
        unsafe {
            conditional_driver_call(
                "cuGraphGetEdges",
                sys::cuGraphGetEdges(graph, from.as_mut_ptr(), to.as_mut_ptr(), &mut edge_count),
            )?;
        }
        from.truncate(edge_count);
    }
    nodes.retain(|node| !from.contains(node));
    Ok(nodes)
}

/// Instantiated CUDA Graph with owned graph + exec handles.
///
/// Destruction may wait for recorded execution events and their stream-prefix
/// dependencies, directly or when another capture drains deferred destruction.
/// Satisfy the [module-level lifecycle contract](crate::cuda_graph) before release
/// or capture. This waiting behavior belongs to the wrapper, not GraphExecDestroy.
/// Modules captured through XLOG launches remain alive until replay completes.
pub struct CapturedCudaGraph {
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    context: Arc<CudaContext>,
    _conditional_api: Option<Arc<ConditionalGraphDriverApi>>,
    _resident_lifecycle_lease: Option<Box<dyn Send + Sync>>,
    modules: CaptureOwners,
    execution: Mutex<GraphExecution>,
}

struct GraphExecution {
    completion: ExecutionCompletion,
    binding: GraphMemoryBinding,
}

#[derive(Default)]
struct GraphMemoryBinding {
    memory: Arc<crate::memory::MemoryAccessManifest>,
    unusable: bool,
}

/// Preserve both generations before the first potentially partial driver update.
/// Failure and unwind leave the executable unusable and both sets of owners live.
fn replace_graph_memory<E>(
    binding: &mut GraphMemoryBinding,
    candidate: Arc<crate::memory::MemoryAccessManifest>,
    wait: impl FnOnce() -> std::result::Result<(), E>,
    update: impl FnOnce() -> std::result::Result<(), E>,
) -> std::result::Result<Arc<crate::memory::MemoryAccessManifest>, E> {
    wait()?;
    binding.memory =
        crate::memory::MemoryAccessManifest::combine(&[binding.memory.clone(), candidate.clone()]);
    binding.unusable = true;
    update()?;
    let obsolete = mem::replace(&mut binding.memory, candidate);
    binding.unusable = false;
    Ok(obsolete)
}

pub(crate) struct KernelNodeUpdate<'a> {
    pub(crate) node: CudaGraphNode,
    pub(crate) params: &'a sys::CUDA_KERNEL_NODE_PARAMS,
}

// CUDA graph handles are context-owned driver handles. xlog stores them behind
// provider-level synchronization when caching graph executions.
unsafe impl Send for CapturedCudaGraph {}
unsafe impl Sync for CapturedCudaGraph {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaGraphNodeKind {
    Kernel,
    Memcpy,
    Memset,
    Host,
    Graph,
    Empty,
    WaitEvent,
    EventRecord,
    ExternalSemaphoresSignal,
    ExternalSemaphoresWait,
    MemAlloc,
    MemFree,
    BatchMemOp,
    Conditional,
}

#[derive(Debug, Clone, Copy)]
pub struct CudaGraphNode {
    pub index: usize,
    pub raw: sys::CUgraphNode,
    pub kind: CudaGraphNodeKind,
}

unsafe impl Send for CudaGraphNode {}
unsafe impl Sync for CudaGraphNode {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CsmCudaGraphJoinKind {
    Inner,
    IndexedInner,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScanTopology {
    pub input_len: u32,
    pub block_size: u32,
    pub scratch_lengths: Vec<u32>,
    pub kernel_node_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CsmCudaGraphKey {
    pub join_kind: CsmCudaGraphJoinKind,
    pub key_arity: u8,
    pub key_bytes: u32,
    pub probe_capacity_class: u32,
    pub output_capacity_class: u32,
    pub scan_topology: ScanTopology,
    pub node_layout_version: u32,
}

impl CsmCudaGraphKey {
    pub fn inner(
        key_arity: usize,
        key_bytes: u32,
        probe_capacity: u32,
        output_capacity: u32,
    ) -> Result<Self> {
        let key_arity = u8::try_from(key_arity).map_err(|_| {
            XlogError::Kernel(format!(
                "CSM CUDA Graph key arity {} exceeds u8::MAX",
                key_arity
            ))
        })?;
        Ok(Self {
            join_kind: CsmCudaGraphJoinKind::Inner,
            key_arity,
            key_bytes,
            probe_capacity_class: graph_capacity_class_u32(probe_capacity),
            output_capacity_class: graph_capacity_class_u32(output_capacity),
            scan_topology: scan_topology_u32(probe_capacity),
            node_layout_version: CSM_CUDA_GRAPH_NODE_LAYOUT_VERSION,
        })
    }
}

pub fn graph_capacity_class_u32(n: u32) -> u32 {
    if n <= 1 {
        1
    } else {
        n.checked_next_power_of_two().unwrap_or(u32::MAX)
    }
}

pub fn scan_topology_u32(mut n: u32) -> ScanTopology {
    let input_len = n;
    let block_size = 256u32;
    let mut scratch_lengths = Vec::new();
    let mut kernel_node_count = if n == 0 { 0 } else { 1 };
    while n > block_size {
        let num_blocks = n.div_ceil(block_size);
        scratch_lengths.push(num_blocks);
        kernel_node_count += 2;
        n = num_blocks;
    }
    ScanTopology {
        input_len,
        block_size,
        scratch_lengths,
        kernel_node_count,
    }
}

impl CapturedCudaGraph {
    /// Rebind all changed kernel nodes under one execution lock and one admitted
    /// candidate manifest. A partial driver update requires rebuilding the graph.
    ///
    /// # Safety
    /// Each argument array must match the original kernel ABI. The candidate
    /// admission must cover every allocation referenced by the resulting graph,
    /// including unchanged nodes and hidden pointers in by-value descriptors.
    pub(crate) unsafe fn rebind_kernel_nodes_in(
        &mut self,
        enqueue: &crate::launch::CudaEnqueue<'_>,
        updates: &[KernelNodeUpdate<'_>],
    ) -> Result<()> {
        enqueue
            .submit(|owners| {
                if owners.is_some() {
                    Err(cudarc::driver::DriverError(
                        sys::CUresult::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
                    ))
                } else {
                    Ok(())
                }
            })
            .map_err(|error| {
                XlogError::Kernel(format!(
                    "graph rebinding requires an ordinary admitted stream: {error}"
                ))
            })?;
        self.context
            .bind_to_thread()
            .map_err(|error| XlogError::Kernel(error.to_string()))?;
        let mut execution = self
            .execution
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.require_replay(enqueue.stream(), &execution)?;
        let nodes = self.nodes_unlocked()?;
        let mut seen = std::collections::BTreeSet::new();
        for update in updates {
            if update.node.kind != CudaGraphNodeKind::Kernel
                || !nodes.iter().any(|node| {
                    node.raw == update.node.raw && node.kind == CudaGraphNodeKind::Kernel
                })
                || !seen.insert(update.node.raw as usize)
                || self.kernel_node_params_unlocked(update.node)?.func != update.params.func
            {
                return Err(XlogError::Kernel("graph rebinding requires unique owned kernel nodes with their original functions".into()));
            }
        }
        let GraphExecution {
            completion,
            binding,
        } = &mut *execution;
        let obsolete = replace_graph_memory(
            binding,
            enqueue.manifest().clone(),
            || {
                completion.wait().map_err(|error| {
                    XlogError::Kernel(format!("graph rebinding prior completion failed: {error}"))
                })
            },
            || {
                for update in updates {
                    cuda_graph_check(
                        "cuGraphExecKernelNodeSetParams_v2",
                        sys::cuGraphExecKernelNodeSetParams_v2(
                            self.exec,
                            update.node.raw,
                            update.params,
                        ),
                    )?;
                    // Nested capture clones the graph template, not its exec.
                    // Keep both representations on the same admitted generation.
                    cuda_graph_check(
                        "cuGraphKernelNodeSetParams_v2",
                        sys::cuGraphKernelNodeSetParams_v2(update.node.raw, update.params),
                    )?;
                }
                Ok(())
            },
        )?;
        drop(execution);
        drop(obsolete);
        Ok(())
    }

    /// Tie resident lifecycle accounting to this real graph/exec owner.
    ///
    /// Binding is idempotent. The lease is created only after graph
    /// instantiation has succeeded and is dropped after this type's `Drop`
    /// implementation destroys the executable and parent graph handles.
    pub fn bind_resident_lifecycle(mut self, runtime: &XlogDeviceRuntime) -> Self {
        if self._resident_lifecycle_lease.is_none() {
            self._resident_lifecycle_lease = Some(Box::new(runtime.resident_graph_handle_lease()));
        }
        self
    }

    /// Create, populate, and instantiate a parent graph containing exactly one
    /// root conditional-WHILE node.
    ///
    /// The body is borrowed only during construction; it cannot escape the
    /// callback or modify an instantiated graph. Its numeric conditional handle
    /// remains valid until this graph is dropped. CUDA permits only one live
    /// executable instantiation of a graph containing a conditional node.
    pub fn conditional_while_on_stream<F>(
        stream: &CudaStream,
        initial_value: u32,
        assign_default_on_launch: bool,
        populate_body: F,
    ) -> std::result::Result<Self, CudaConditionalGraphUnavailable>
    where
        F: FnOnce(
            &ConditionalCudaGraphBody,
        ) -> std::result::Result<(), CudaConditionalGraphUnavailable>,
    {
        let api = ConditionalGraphDriverApi::load()?;
        let context = stream.context().clone();
        let raw_context =
            stream_context(stream).map_err(CudaConditionalGraphUnavailable::body_population)?;
        if raw_context != context.cu_ctx() {
            return Err(CudaConditionalGraphUnavailable::ContextMismatch);
        }

        let graph = UninstantiatedCudaGraph::create(Arc::clone(&context))?;
        let mut handle = 0;
        let flags = if assign_default_on_launch {
            sys::CU_GRAPH_COND_ASSIGN_DEFAULT
        } else {
            0
        };
        unsafe {
            conditional_driver_call(
                "cuGraphConditionalHandleCreate",
                (api.conditional_handle_create)(
                    &mut handle,
                    graph.raw(),
                    raw_context,
                    initial_value,
                    flags,
                ),
            )?;
        }

        let mut params = conditional_node_params(
            handle,
            raw_context,
            sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_WHILE,
        );
        let mut conditional_node = ptr::null_mut();
        unsafe {
            conditional_driver_call(
                "cuGraphAddNode",
                (api.graph_add_node)(
                    &mut conditional_node,
                    graph.raw(),
                    ptr::null(),
                    0,
                    &mut params,
                ),
            )?;
        }
        if conditional_node.is_null() {
            return Err(CudaConditionalGraphUnavailable::NullDriverHandle {
                operation: "cuGraphAddNode",
            });
        }

        let conditional = unsafe { params.__bindgen_anon_1.conditional };
        if conditional.phGraph_out.is_null() {
            return Err(CudaConditionalGraphUnavailable::NullDriverHandle {
                operation: "cuGraphAddNode body array",
            });
        }
        let body_graph = unsafe { *conditional.phGraph_out };
        if body_graph.is_null() {
            return Err(CudaConditionalGraphUnavailable::NullDriverHandle {
                operation: "cuGraphAddNode WHILE body",
            });
        }
        populate_body(&ConditionalCudaGraphBody {
            graph: body_graph,
            handle,
            context: raw_context,
            modules: graph.modules.clone(),
        })?;

        let mut instantiated = Self::instantiate_graph(graph)?;
        instantiated._conditional_api = Some(api);
        Ok(instantiated)
    }

    /// Instantiate and assume sole ownership of `graph`.
    ///
    /// The graph is destroyed on instantiation failure. On success, this value
    /// destroys the executable first and the source graph second.
    ///
    /// # Safety
    /// `graph` must be a valid, unowned graph in `context`, and no other owner
    /// may destroy or instantiate it while this value exists.
    /// The caller must retain modules and allocations referenced by this foreign
    /// graph through its last execution. Ownership of raw CUDA nodes does not
    /// transfer ownership of their external resources.
    pub unsafe fn instantiate_owned_graph(
        graph: sys::CUgraph,
        context: Arc<CudaContext>,
    ) -> std::result::Result<Self, CudaConditionalGraphUnavailable> {
        if graph.is_null() {
            return Err(CudaConditionalGraphUnavailable::NullDriverHandle {
                operation: "instantiate_owned_graph",
            });
        }
        Self::instantiate_graph(UninstantiatedCudaGraph {
            raw: graph,
            context: Arc::clone(&context),
            modules: Arc::default(),
        })
    }

    fn instantiate_graph(
        owned_graph: UninstantiatedCudaGraph,
    ) -> std::result::Result<Self, CudaConditionalGraphUnavailable> {
        if owned_graph
            .modules
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .failed
        {
            return Err(CudaConditionalGraphUnavailable::body_population(
                "a captured submission failed or unwound; rebuild the graph before instantiation",
            ));
        }
        let context = owned_graph.context.clone();
        context
            .bind_to_thread()
            .map_err(CudaConditionalGraphUnavailable::body_population)?;
        let completion = ExecutionCompletion::new(&context)
            .map_err(CudaConditionalGraphUnavailable::body_population)?;
        let mut exec = ptr::null_mut();
        conditional_driver_call(
            "cuGraphInstantiateWithFlags",
            // SAFETY: the graph is owned, uninstantiated, and its context is bound.
            unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, owned_graph.raw(), 0) },
        )?;
        if exec.is_null() {
            return Err(CudaConditionalGraphUnavailable::NullDriverHandle {
                operation: "cuGraphInstantiateWithFlags",
            });
        }
        let modules = owned_graph.modules.clone();
        let memory = {
            let mut owners = modules.lock().unwrap_or_else(|error| error.into_inner());
            crate::memory::MemoryAccessManifest::combine(&mem::take(&mut owners.memory))
        };
        Ok(Self {
            graph: owned_graph.into_raw(),
            exec,
            context,
            _conditional_api: None,
            _resident_lifecycle_lease: None,
            modules,
            execution: Mutex::new(GraphExecution {
                completion,
                binding: GraphMemoryBinding {
                    memory,
                    unusable: false,
                },
            }),
        })
    }

    /// Capture work submitted by `record` on `stream`, instantiate it, and take
    /// ownership of the resulting graph handles.
    ///
    /// Capture setup and cleanup may wait for deferred destruction under the
    /// [module-level lifecycle contract](crate::cuda_graph), including on errors
    /// and callback unwind. The resulting graph has not executed at that point.
    pub fn capture_on_stream<F>(stream: &CudaStream, record: F) -> Result<Self>
    where
        F: FnOnce() -> Result<()>,
    {
        Self::capture_with_owners(stream, Arc::default(), record)
    }

    /// Keep the actual stream and preallocated producer resources in the same
    /// graph ownership chain before the first capture operation. Unknown capture
    /// or execution completion retains them through the ordinary retirement path.
    /// Resources allocated by `record` must already be retained by these owners
    /// before being submitted; this does not validate foreign pointer aliases.
    pub(crate) fn capture_on_stream_retaining<F>(
        stream: &Arc<CudaStream>,
        mut external_resources: Vec<Arc<dyn Send + Sync>>,
        record: F,
    ) -> Result<Self>
    where
        F: FnOnce() -> Result<()>,
    {
        external_resources.push(stream.clone());
        let owners = Arc::new(Mutex::new(CapturedOwners::with_external_resources(
            external_resources,
        )));
        Self::capture_with_owners(stream, owners, record)
    }

    fn capture_with_owners<F>(
        stream: &CudaStream,
        modules: CaptureOwners,
        record: F,
    ) -> Result<Self>
    where
        F: FnOnce() -> Result<()>,
    {
        let mut capture = DriverStreamCapture::prepare(stream, true, modules.clone())
            .map_err(|error| XlogError::Kernel(error.decline_detail()))?;
        unsafe {
            capture
                .begin("cuStreamBeginCapture_v2", || {
                    sys::cuStreamBeginCapture_v2(
                        stream.cu_stream(),
                        sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                    )
                })
                .map_err(|error| XlogError::Kernel(error.decline_detail()))?;
        }

        let record_result = capture_outcome(Some(&capture.lease.modules), record);
        let end_result = capture.finish();

        if let Err(record_err) = record_result {
            if let Ok(graph) = end_result {
                if !graph.is_null() {
                    drop(UninstantiatedCudaGraph {
                        raw: graph,
                        context: stream.context().clone(),
                        modules: modules.clone(),
                    });
                }
            }
            return Err(record_err);
        }
        let graph = end_result.map_err(|error| XlogError::Kernel(error.decline_detail()))?;
        if graph.is_null() {
            return Err(XlogError::Kernel(
                "cuStreamEndCapture returned a null CUDA graph".to_string(),
            ));
        }

        // SAFETY: EndCapture returned this new graph and ownership has not been
        // exposed. The canonical owner handles all instantiation error paths.
        Self::instantiate_graph(UninstantiatedCudaGraph {
            raw: graph,
            context: stream.context().clone(),
            modules,
        })
        .map_err(|error| XlogError::Kernel(error.decline_detail()))
    }

    /// Replay after admitting the complete current memory manifest on this stream.
    /// Submission is asynchronous; dropping the graph may wait for completion.
    pub fn launch(&self, stream: &Arc<CudaStream>) -> Result<()> {
        let memory = {
            let execution = self
                .execution
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            self.require_replay(stream, &execution)?;
            execution.binding.memory.clone()
        };
        crate::memory::with_memory_manifest(stream.clone(), memory.clone(), |enqueue| {
            let mut execution = self
                .execution
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            self.require_replay(stream, &execution)
                .map_err(|error| crate::device_runtime::ResourceError::Driver(error.to_string()))?;
            if !Arc::ptr_eq(&memory, &execution.binding.memory) {
                return Err(crate::device_runtime::ResourceError::StreamMisuse(
                    "graph bindings changed during memory admission".into(),
                ));
            }
            self.submit_graph(&mut execution, enqueue)
                .map_err(|error| crate::device_runtime::ResourceError::Driver(error.to_string()))
        })
        .map_err(|error| {
            XlogError::Kernel(format!(
                "CUDA graph memory admission/replay failed: {error}"
            ))
        })
    }

    /// Replay inside an existing admission without acquiring an overlapping group
    /// or a new capture pin.
    pub fn launch_in(&self, enqueue: &crate::launch::CudaEnqueue<'_>) -> Result<()> {
        let mut execution = self
            .execution
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.require_replay(enqueue.stream(), &execution)?;
        if !execution.binding.memory.covered_by(enqueue.manifest()) {
            return Err(XlogError::Kernel(
                "CUDA graph replay lacks admission for its complete memory manifest".into(),
            ));
        }
        self.submit_graph(&mut execution, enqueue)
            .map_err(|error| XlogError::Kernel(format!("CUDA graph submission failed: {error}")))
    }

    fn require_replay(&self, stream: &CudaStream, execution: &GraphExecution) -> Result<()> {
        if execution.binding.unusable {
            return Err(XlogError::Kernel(
                "CUDA graph has an incomplete parameter update; rebuild it before replay".into(),
            ));
        }
        if stream.context().cu_ctx() != self.context.cu_ctx() {
            return Err(XlogError::Kernel(
                CudaConditionalGraphUnavailable::ContextMismatch.decline_detail(),
            ));
        }
        Ok(())
    }

    fn submit_graph(
        &self,
        execution: &mut GraphExecution,
        enqueue: &crate::launch::CudaEnqueue<'_>,
    ) -> std::result::Result<(), cudarc::driver::DriverError> {
        self.context.bind_to_thread()?;
        enqueue.submit(|owners| {
            let launch = || unsafe { sys::cuGraphLaunch(self.exec, enqueue.stream().cu_stream()).result() };
            if let Some(owners) = owners {
                // AddChildGraphNode clones the updated template. Freeze this
                // generation's owners alongside that clone in the parent.
                retain_child_capture_owners(owners, &self.modules, &execution.binding.memory);
                unsafe {
                    let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
                    let mut id = 0;
                    let mut graph = ptr::null_mut();
                    let mut dependencies = ptr::null();
                    let mut count = 0;
                    sys::cuStreamGetCaptureInfo_v2(enqueue.stream().cu_stream(), &mut status, &mut id, &mut graph, &mut dependencies, &mut count).result()?;
                    if status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE || graph.is_null() {
                        return Err(cudarc::driver::DriverError(sys::CUresult::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED));
                    }
                    let mut child = ptr::null_mut();
                    sys::cuGraphAddChildGraphNode(&mut child, graph, dependencies, count, self.graph).result()?;
                    sys::cuStreamUpdateCaptureDependencies(enqueue.stream().cu_stream(), &mut child, 1, sys::CUstreamUpdateCaptureDependencies_flags::CU_STREAM_SET_CAPTURE_DEPENDENCIES as u32).result()
                }
            } else {
                execution.completion.submit(enqueue.stream(), Some(enqueue.completion()), launch)
            }
        })
    }

    /// Number of nodes in the captured graph. Used by bounded CSM CUDA Graph
    /// cache-key and node-inventory certs to prove topology stability.
    pub fn node_count(&self) -> Result<usize> {
        let _execution = self
            .execution
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.context
            .bind_to_thread()
            .map_err(|error| XlogError::Kernel(error.to_string()))?;
        let mut count = 0usize;
        unsafe {
            cuda_graph_check(
                "cuGraphGetNodes(count)",
                sys::cuGraphGetNodes(self.graph, ptr::null_mut(), &mut count),
            )?;
        }
        Ok(count)
    }

    /// Return graph nodes in CUDA's enumeration order with their node type.
    ///
    /// CUDA does not define this list as dependency or execution order. Use
    /// [`Self::linear_chain_node_kinds`] when a linear topology is required.
    pub fn nodes(&self) -> Result<Vec<CudaGraphNode>> {
        let _execution = self
            .execution
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.context
            .bind_to_thread()
            .map_err(|error| XlogError::Kernel(error.to_string()))?;
        self.nodes_unlocked()
    }

    fn nodes_unlocked(&self) -> Result<Vec<CudaGraphNode>> {
        let mut check = |operation, code| cuda_graph_check(operation, code);
        graph_nodes_with(self.graph, &mut check)
    }

    /// Return actual node kinds in root-to-leaf dependency order.
    ///
    /// This fails unless the graph is one connected linear chain with exactly
    /// one root, one leaf, one immediate dependency per non-root node, and no
    /// branches, duplicate dependencies, foreign dependencies, or cycles.
    pub fn linear_chain_node_kinds(&self) -> Result<Vec<CudaGraphNodeKind>> {
        let _execution = self
            .execution
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.context
            .bind_to_thread()
            .map_err(|error| XlogError::Kernel(error.to_string()))?;
        let mut check = |operation, code| cuda_graph_check(operation, code);
        let mut shape_error = |error| {
            XlogError::Kernel(format!(
                "CUDA graph is not a linear dependency chain: {error}"
            ))
        };
        graph_linear_chain_node_kinds_with(self.graph, &mut check, &mut shape_error)
    }

    /// Read CUDA's raw kernel-node params for inventory/update code.
    ///
    /// The returned `kernelParams` pointer is CUDA-owned capture metadata. Treat
    /// it as read-only unless constructing a fresh params object for
    /// a complete admitted parameter update.
    pub fn kernel_node_params(&self, node: CudaGraphNode) -> Result<sys::CUDA_KERNEL_NODE_PARAMS> {
        let _execution = self
            .execution
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.context
            .bind_to_thread()
            .map_err(|error| XlogError::Kernel(error.to_string()))?;
        if !self
            .nodes_unlocked()?
            .iter()
            .any(|owned| owned.raw == node.raw && owned.kind == node.kind)
        {
            return Err(XlogError::Kernel(
                "kernel metadata requires a node owned by this graph".into(),
            ));
        }
        self.kernel_node_params_unlocked(node)
    }

    fn kernel_node_params_unlocked(
        &self,
        node: CudaGraphNode,
    ) -> Result<sys::CUDA_KERNEL_NODE_PARAMS> {
        if node.kind != CudaGraphNodeKind::Kernel {
            return Err(XlogError::Kernel(format!(
                "kernel_node_params called for non-kernel graph node {:?}",
                node.kind
            )));
        }
        let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { mem::zeroed() };
        unsafe {
            cuda_graph_check(
                "cuGraphKernelNodeGetParams_v2",
                sys::cuGraphKernelNodeGetParams_v2(node.raw, &mut params),
            )?;
        }
        Ok(params)
    }

    /// Read CUDA's raw memset-node params for inventory/update code.
    pub fn memset_node_params(&self, node: CudaGraphNode) -> Result<sys::CUDA_MEMSET_NODE_PARAMS> {
        let _execution = self
            .execution
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.context
            .bind_to_thread()
            .map_err(|error| XlogError::Kernel(error.to_string()))?;
        if !self
            .nodes_unlocked()?
            .iter()
            .any(|owned| owned.raw == node.raw && owned.kind == node.kind)
        {
            return Err(XlogError::Kernel(
                "memset metadata requires a node owned by this graph".into(),
            ));
        }
        if node.kind != CudaGraphNodeKind::Memset {
            return Err(XlogError::Kernel(format!(
                "memset_node_params called for non-memset graph node {:?}",
                node.kind
            )));
        }
        let mut params: sys::CUDA_MEMSET_NODE_PARAMS = unsafe { mem::zeroed() };
        unsafe {
            cuda_graph_check(
                "cuGraphMemsetNodeGetParams",
                sys::cuGraphMemsetNodeGetParams(node.raw, &mut params),
            )?;
        }
        Ok(params)
    }

    /// Update one memset node using a complete admitted candidate manifest.
    ///
    /// # Safety
    /// The candidate admission must cover all unchanged graph accesses as well
    /// as the new destination. Parameters must obey CUDA's memset-node ABI.
    pub unsafe fn set_memset_node_params_in(
        &mut self,
        node: CudaGraphNode,
        params: &sys::CUDA_MEMSET_NODE_PARAMS,
        enqueue: &crate::launch::CudaEnqueue<'_>,
    ) -> Result<()> {
        enqueue
            .submit(|owners| {
                if owners.is_some() {
                    Err(cudarc::driver::DriverError(
                        sys::CUresult::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
                    ))
                } else {
                    Ok(())
                }
            })
            .map_err(|error| XlogError::Kernel(error.to_string()))?;
        self.context
            .bind_to_thread()
            .map_err(|error| XlogError::Kernel(error.to_string()))?;
        let mut execution = self
            .execution
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.require_replay(enqueue.stream(), &execution)?;
        if node.kind != CudaGraphNodeKind::Memset
            || !self
                .nodes_unlocked()?
                .iter()
                .any(|owned| owned.raw == node.raw && owned.kind == node.kind)
        {
            return Err(XlogError::Kernel(
                "memset update requires an owned memset node".into(),
            ));
        }
        let bytes = params
            .width
            .checked_mul(params.elementSize as usize)
            .and_then(|row| {
                params
                    .height
                    .saturating_sub(1)
                    .checked_mul(params.pitch)
                    .and_then(|offset| offset.checked_add(row))
            })
            .ok_or_else(|| XlogError::Kernel("memset destination span overflow".into()))?;
        let context = enqueue.stream().context().cu_ctx() as usize;
        let range = crate::device_runtime::resource::MemoryUse::new(
            params.dst,
            bytes,
            crate::device_runtime::Access::Write,
        )
        .map_err(|error| XlogError::Kernel(error.to_string()))?;
        if !enqueue.manifest().covers(&[(context, range)]) {
            return Err(XlogError::Kernel(
                "memset update destination lacks an admitted write".into(),
            ));
        }
        let GraphExecution {
            completion,
            binding,
        } = &mut *execution;
        let obsolete = replace_graph_memory(
            binding,
            enqueue.manifest().clone(),
            || {
                completion
                    .wait()
                    .map_err(|error| XlogError::Kernel(error.to_string()))
            },
            || {
                cuda_graph_check(
                    "cuGraphExecMemsetNodeSetParams",
                    sys::cuGraphExecMemsetNodeSetParams(
                        self.exec,
                        node.raw,
                        params,
                        self.context.cu_ctx(),
                    ),
                )?;
                cuda_graph_check(
                    "cuGraphMemsetNodeSetParams",
                    sys::cuGraphMemsetNodeSetParams(node.raw, params),
                )
            },
        )?;
        drop(execution);
        drop(obsolete);
        Ok(())
    }

    /// Raw graph handle for low-level node inventory/update code.
    pub fn graph(&self) -> sys::CUgraph {
        self.graph
    }

    /// Raw instantiated graph handle for low-level graph-exec update code.
    /// Executions submitted outside [`Self::launch`] are not tracked by this
    /// owner. The unsafe raw-API caller must finish them before dropping it.
    pub fn exec(&self) -> sys::CUgraphExec {
        self.exec
    }
}

impl Drop for CapturedCudaGraph {
    fn drop(&mut self) {
        let graph = self.graph as usize;
        let exec = self.exec as usize;
        let context = self.context.clone();
        let modules = self.modules.clone();
        let execution = self
            .execution
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        let completion = mem::take(&mut execution.completion);
        let binding = mem::take(&mut execution.binding);
        let conditional_api = self._conditional_api.take();
        let resident_lease = self._resident_lifecycle_lease.take();
        retire_resources_after_completion(
            (
                graph,
                exec,
                context,
                modules,
                completion,
                binding,
                conditional_api,
                resident_lease,
            ),
            |(_, _, context, _, completion, _, _, _)| {
                context.bind_to_thread().and_then(|()| completion.wait())
            },
            |(graph, exec, _, _, _, _, _, _)| unsafe {
                sys::cuGraphExecDestroy(*exec as sys::CUgraphExec).result()?;
                sys::cuGraphDestroy(*graph as sys::CUgraph).result()
            },
            |_| Ok(()),
            |(_, _, context, _, _, _, _, _), error| context.record_err::<()>(Err(error)),
        );
    }
}

/// Retry completion prerequisites and post-destruction release publication.
/// The destructive callback runs only once. Its error or unwind retains the
/// entire chain without resubmitting handles of unknown validity. This is
/// quarantine, not proof of completed destruction. On success, `finish` may
/// retry independently and must not access resources consumed by `destroy`.
pub(crate) fn retire_resources_after_completion<T: Send + 'static, E: std::fmt::Display>(
    resources: T,
    ready: impl Fn(&T) -> std::result::Result<(), E> + Send + 'static,
    destroy: impl Fn(&mut T) -> std::result::Result<(), E> + Send + 'static,
    finish: impl Fn(&mut T) -> std::result::Result<(), E> + Send + 'static,
    diagnose: impl Fn(&T, E) + Send + 'static,
) {
    let mut pending = mem::ManuallyDrop::new(Some(resources));
    let mut destroyed = false;
    retry_retirement_after_stream_captures(move || {
        let Some(resources) = pending.as_ref() else {
            return true;
        };
        if !destroyed {
            if let Err(error) = ready(resources) {
                // Do not reinsert this error into the context: bind_to_thread
                // consumes stored errors, so reinsertion prevents later retry.
                eprintln!("CUDA resource retirement prerequisites incomplete: {error}");
                return false;
            }
            let mut resources =
                mem::ManuallyDrop::new(pending.take().expect("retirement owns its resources"));
            if let Err(error) = destroy(&mut resources) {
                retain_failed_resources(resources, |resources| diagnose(resources, error));
                return true;
            }
            // Restore the capsule before retryable publication, including its
            // unwind path. Consumed owners must be taken out by `destroy`.
            *pending = Some(mem::ManuallyDrop::into_inner(resources));
            destroyed = true;
        }
        if let Err(error) = finish(pending.as_mut().expect("retirement owns its resources")) {
            eprintln!("CUDA resource release publication incomplete: {error}");
            return false;
        }
        drop(pending.take());
        true
    });
}

fn retain_failed_resources<T: 'static>(resources: T, diagnose: impl FnOnce(&T)) {
    // Unknown completion cannot release dependencies, even if diagnosis unwinds.
    let resources = Box::leak(Box::new(resources));
    diagnose(resources);
}

impl CudaGraphNodeKind {
    fn from_sys(kind: sys::CUgraphNodeType) -> Self {
        match kind {
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => Self::Kernel,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => Self::Memcpy,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET => Self::Memset,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_HOST => Self::Host,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_GRAPH => Self::Graph,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY => Self::Empty,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_WAIT_EVENT => Self::WaitEvent,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EVENT_RECORD => Self::EventRecord,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL => {
                Self::ExternalSemaphoresSignal
            }
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT => Self::ExternalSemaphoresWait,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC => Self::MemAlloc,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE => Self::MemFree,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_BATCH_MEM_OP => Self::BatchMemOp,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL => Self::Conditional,
        }
    }
}

fn cuda_graph_check(label: &str, code: sys::CUresult) -> Result<()> {
    if code == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(XlogError::Kernel(format!("{label} failed: {code:?}")))
    }
}

fn stream_context(stream: &CudaStream) -> Result<sys::CUcontext> {
    stream
        .context()
        .bind_to_thread()
        .map_err(|error| XlogError::Kernel(error.to_string()))?;
    let mut ctx = ptr::null_mut();
    unsafe {
        cuda_graph_check(
            "cuStreamGetCtx",
            sys::cuStreamGetCtx(stream.cu_stream(), &mut ctx),
        )?;
    }
    if ctx.is_null() {
        Err(XlogError::Kernel(
            "cuStreamGetCtx returned a null CUDA context".to_string(),
        ))
    } else {
        Ok(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::{
        driver::{DevicePtr, LaunchConfig, PushKernelArg},
        nvrtc::compile_ptx,
    };
    use std::sync::Barrier;
    use std::thread;

    #[test]
    fn failed_graph_retirement_retains_dependencies_before_diagnostic_unwind() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct DropProbe(Arc<AtomicUsize>);

        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        for panic_diagnostic in [false, true] {
            let drops = Arc::new(AtomicUsize::new(0));
            let resources = (
                Arc::new(DropProbe(drops.clone())),
                Arc::new(DropProbe(drops.clone())),
                Arc::new(DropProbe(drops.clone())),
            );
            // No external strong owner can mask premature dependency release.
            let observers = [
                Arc::downgrade(&resources.0),
                Arc::downgrade(&resources.1),
                Arc::downgrade(&resources.2),
            ];
            let diagnosed = std::cell::Cell::new(false);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                retain_failed_resources(resources, |_| {
                    diagnosed.set(true);
                    if panic_diagnostic {
                        panic!("graph retirement diagnostic failed");
                    }
                });
            }));

            assert!(diagnosed.get());
            assert_eq!(result.is_err(), panic_diagnostic);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            assert!(observers.iter().all(|owner| owner.upgrade().is_some()));
        }
    }

    fn conditional_cuda_context_or_skip() -> Option<Arc<CudaContext>> {
        if let Err(error) = ConditionalGraphDriverApi::load() {
            if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
                panic!("XLOG_REQUIRE_CUDA=1 but CUDA conditional-graph setup failed: {error}");
            }
            eprintln!("Skipping test: {}", error.decline_detail());
            return None;
        }

        match CudaContext::new(0) {
            Ok(context) => Some(context),
            Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
                panic!("XLOG_REQUIRE_CUDA=1 but CUDA setup failed: {error}")
            }
            Err(error) => {
                eprintln!("Skipping test: CUDA unavailable: {error}");
                None
            }
        }
    }

    #[test]
    fn conditional_graphs_expose_dependency_ordered_linear_chain_inventory() {
        let _: fn(
            &ConditionalCudaGraphBody,
        )
            -> std::result::Result<Vec<CudaGraphNodeKind>, CudaConditionalGraphUnavailable> =
            ConditionalCudaGraphBody::linear_chain_node_kinds;
        let _: fn(&CapturedCudaGraph) -> Result<Vec<CudaGraphNodeKind>> =
            CapturedCudaGraph::linear_chain_node_kinds;
    }

    #[test]
    fn linear_chain_inventory_recovers_dependency_order_from_shuffled_nodes() {
        let enumerated_kinds = [
            CudaGraphNodeKind::Kernel,
            CudaGraphNodeKind::Kernel,
            CudaGraphNodeKind::Conditional,
            CudaGraphNodeKind::Kernel,
            CudaGraphNodeKind::Conditional,
        ];
        let immediate_dependencies = vec![vec![2], vec![4], vec![3], vec![], vec![0]];
        let dependency_order = linear_chain_order(&immediate_dependencies).unwrap();
        assert_eq!(dependency_order, vec![3, 2, 0, 4, 1]);
        assert_eq!(
            dependency_order
                .into_iter()
                .map(|index| enumerated_kinds[index])
                .collect::<Vec<_>>(),
            vec![
                CudaGraphNodeKind::Kernel,
                CudaGraphNodeKind::Conditional,
                CudaGraphNodeKind::Kernel,
                CudaGraphNodeKind::Conditional,
                CudaGraphNodeKind::Kernel,
            ]
        );
        assert_eq!(linear_chain_order(&[vec![]]).unwrap(), vec![0]);
    }

    #[test]
    fn linear_chain_inventory_rejects_non_linear_dependency_shapes() {
        let cases = [
            (
                "empty graph",
                Vec::new(),
                LinearGraphChainError::RootCount { found: 0 },
            ),
            (
                "branch",
                vec![vec![], vec![0], vec![0]],
                LinearGraphChainError::Branch {
                    node: 0,
                    dependents: 2,
                },
            ),
            (
                "disconnected",
                vec![vec![], vec![]],
                LinearGraphChainError::RootCount { found: 2 },
            ),
            (
                "cycle",
                vec![vec![1], vec![0]],
                LinearGraphChainError::Cycle,
            ),
            (
                "foreign dependency",
                vec![vec![], vec![2]],
                LinearGraphChainError::ForeignDependency {
                    node: 1,
                    dependency: 2,
                },
            ),
            (
                "duplicate edge",
                vec![vec![], vec![0, 0]],
                LinearGraphChainError::DuplicateDependency {
                    node: 1,
                    dependency: 0,
                },
            ),
        ];
        for (case, dependencies, expected) in cases {
            assert_eq!(linear_chain_order(&dependencies), Err(expected), "{case}");
        }
    }

    #[test]
    fn capture_retirement_waits_for_every_active_capture() {
        let _capture_test = crate::cuda_graph::capture_lifecycle_test_guard();
        use std::sync::atomic::{AtomicUsize, Ordering};
        let completed = Arc::new(AtomicUsize::new(0));
        let first = try_acquire_stream_capture_key(StreamCaptureKey {
            context: 0xc1,
            stream: 0xc2,
        })
        .unwrap();
        let second = try_acquire_stream_capture_key(StreamCaptureKey {
            context: 0xc1,
            stream: 0xc3,
        })
        .unwrap();
        let observed = completed.clone();
        retire_after_stream_captures(move || {
            observed.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(completed.load(Ordering::SeqCst), 0);
        drop(first);
        assert_eq!(completed.load(Ordering::SeqCst), 0);
        drop(second);
        assert_eq!(completed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn capture_retirement_retries_owned_work_once_per_cold_drain() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let _capture_test = capture_lifecycle_test_guard();
        let attempts = Arc::new(AtomicUsize::new(0));
        let completed = Arc::new(AtomicUsize::new(0));
        let allocation = Arc::new(vec![42_u8; 64].into_boxed_slice());
        let observer = Arc::downgrade(&allocation);
        let observed_attempts = attempts.clone();
        let observed_completed = completed.clone();
        retry_retirement_after_stream_captures(move || {
            assert_eq!(allocation[0], 42);
            let attempt = observed_attempts.fetch_add(1, Ordering::SeqCst);
            if attempt == 0 {
                let nested = observed_completed.clone();
                retire_after_stream_captures(move || {
                    nested.fetch_add(1, Ordering::SeqCst);
                });
                return false;
            }
            true
        });
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert_eq!(completed.load(Ordering::SeqCst), 1);
        assert!(observer.upgrade().is_some());

        let unrelated = completed.clone();
        thread::spawn(move || {
            retire_after_stream_captures(move || {
                unrelated.fetch_add(1, Ordering::SeqCst);
            });
        })
        .join()
        .unwrap();
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert_eq!(completed.load(Ordering::SeqCst), 2);
        assert!(observer.upgrade().is_none());
    }

    #[test]
    fn capture_retirement_retry_unwind_restores_executing_and_pending_owners() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        let _capture_test = capture_lifecycle_test_guard();
        let earlier = Arc::new(vec![3_u8; 16].into_boxed_slice());
        let earlier_observer = Arc::downgrade(&earlier);
        let ready = Arc::new(AtomicBool::new(false));
        let observed_ready = ready.clone();
        retry_retirement_after_stream_captures(move || {
            assert_eq!(earlier[0], 3);
            observed_ready.load(Ordering::SeqCst)
        });
        let allocation = Arc::new(vec![7_u8; 32].into_boxed_slice());
        let observer = Arc::downgrade(&allocation);
        let attempts = Arc::new(AtomicUsize::new(0));
        let observed_attempts = attempts.clone();
        let pending = Arc::new(AtomicUsize::new(0));
        let observed_pending = pending.clone();
        let failed = std::panic::catch_unwind(move || {
            retry_retirement_after_stream_captures(move || {
                assert_eq!(allocation[0], 7);
                if observed_attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                    let nested = observed_pending.clone();
                    retire_after_stream_captures(move || {
                        nested.fetch_add(1, Ordering::SeqCst);
                    });
                    panic!("reclamation fence wait unwound");
                }
                true
            });
        });
        assert!(failed.is_err());
        assert!(observer.upgrade().is_some());
        assert!(earlier_observer.upgrade().is_some());
        assert!(!stream_capture_registry().retiring);
        ready.store(true, Ordering::SeqCst);
        retire_after_stream_captures(|| {});
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert_eq!(pending.load(Ordering::SeqCst), 1);
        assert!(observer.upgrade().is_none());
        assert!(earlier_observer.upgrade().is_none());
    }

    fn host_capture_phase(
        key: StreamCaptureKey,
        id: u64,
    ) -> (StreamCaptureLease, StreamSubmissionPhase) {
        let mut lease = try_acquire_stream_capture_key(key).unwrap();
        lease.driver_active = true;
        lease.register_capture(id, &mut stream_capture_registry());
        let phase = StreamSubmissionPhase::from_capture_info(
            stream_capture_registry(),
            key.context,
            sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE,
            id,
        )
        .unwrap();
        (lease, phase)
    }

    fn capture_phase_owners(phase: &StreamSubmissionPhase) -> &CaptureOwners {
        match &phase.state {
            StreamSubmissionState::Captured { owners, .. } => owners,
            StreamSubmissionState::Ordinary { .. } => panic!("expected a captured phase"),
        }
    }

    #[test]
    fn capture_submission_borrowed_pin_survives_closing() {
        let _capture_test = capture_lifecycle_test_guard();
        let (mut lease, phase) = host_capture_phase(
            StreamCaptureKey {
                context: 0x301,
                stream: 0x302,
            },
            0x303,
        );
        let pin = phase.pin().unwrap();
        let executing = lease.close_submission_admission().unwrap();
        assert!(phase.pin().is_err());
        pin.with_submission(|owners| {
            assert!(Arc::ptr_eq(owners.unwrap(), capture_phase_owners(&phase)));
            assert!(ACTIVE_STREAM_CAPTURES.get().unwrap().try_lock().is_ok());
            Ok(())
        })
        .unwrap();
        drop(pin);
        executing.wait_until_idle();
        lease.complete_driver_capture();
    }

    #[test]
    fn graph_retirement_unknown_destroy_never_resubmits_or_releases_owners() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let _capture_test = capture_lifecycle_test_guard();
        for unwind in [false, true] {
            let calls = Arc::new(AtomicUsize::new(0));
            let called = calls.clone();
            let (owner, bytes) = crate::memory::test_memory_manifest(|| {});
            let resource = Arc::new(vec![71_u8]);
            let original = Arc::downgrade(&resource);
            let mut captured = CapturedOwners::with_external_resources(vec![resource]);
            captured.memory.push(owner);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                retire_resources_after_completion(
                    Arc::new(Mutex::new(captured)),
                    |_| Ok(()),
                    move |_| {
                        called.fetch_add(1, Ordering::SeqCst);
                        if unwind {
                            panic!("destructive call interrupted");
                        }
                        Err(DriverError(sys::CUresult::CUDA_ERROR_UNKNOWN))
                    },
                    |_| Ok(()),
                    |_, _| {},
                )
            }));
            assert_eq!(result.is_err(), unwind);
            reap_capture_retirements();
            reap_capture_retirements();
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert!(bytes.upgrade().is_some());
            assert!(original.upgrade().is_some());
        }
    }

    #[test]
    fn graph_retirement_precondition_unwind_keeps_retry_callable() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let _capture_test = capture_lifecycle_test_guard();
        let first = Arc::new(AtomicBool::new(true));
        let (owner, bytes) = crate::memory::test_memory_manifest(|| {});
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            retire_resources_after_completion(
                owner,
                move |_| {
                    if first.swap(false, Ordering::SeqCst) {
                        panic!("completion wait interrupted");
                    }
                    Ok(())
                },
                |_| Ok(()),
                |_| Ok(()),
                |_, _: DriverError| panic!("no terminal failure expected"),
            )
        }));
        assert!(result.is_err());
        assert!(bytes.upgrade().is_some());
        reap_capture_retirements();
        assert!(bytes.upgrade().is_none());
    }

    #[test]
    fn graph_retirement_retries_preconditions_without_resubmitting_destroy() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        let _capture_test = capture_lifecycle_test_guard();
        let ready = Arc::new(AtomicBool::new(false));
        let destroyed = Arc::new(AtomicUsize::new(0));
        let ready_probe = ready.clone();
        let destroy_probe = destroyed.clone();
        let (owner, bytes) = crate::memory::test_memory_manifest(|| {});
        let resource = Arc::new(vec![83_u8]);
        let original = Arc::downgrade(&resource);
        let mut captured = CapturedOwners::with_external_resources(vec![resource]);
        captured.memory.push(owner);
        retire_resources_after_completion(
            Arc::new(Mutex::new(captured)),
            move |_| {
                if ready_probe.load(Ordering::SeqCst) {
                    Ok(())
                } else {
                    Err(DriverError(sys::CUresult::CUDA_ERROR_NOT_READY))
                }
            },
            move |_| {
                destroy_probe.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
            |_| Ok(()),
            |_, _| panic!("no terminal failure expected"),
        );
        assert_eq!(destroyed.load(Ordering::SeqCst), 0);
        assert!(bytes.upgrade().is_some());
        assert!(original.upgrade().is_some());
        ready.store(true, Ordering::SeqCst);
        reap_capture_retirements();
        assert_eq!(destroyed.load(Ordering::SeqCst), 1);
        assert!(bytes.upgrade().is_none());
        assert!(original.upgrade().is_none());
        reap_capture_retirements();
        assert_eq!(destroyed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn graph_memory_partial_update_retains_both_owner_generations() {
        use crate::memory::test_memory_manifest;
        use std::panic::{catch_unwind, AssertUnwindSafe};
        for unwind in [false, true] {
            for failed_step in 0..6 {
                let (old, old_bytes) = test_memory_manifest(|| {});
                let (candidate, new_bytes) = test_memory_manifest(|| {});
                let mut binding = GraphMemoryBinding {
                    memory: old,
                    unusable: false,
                };
                let steps = std::cell::Cell::new(0);
                let result = catch_unwind(AssertUnwindSafe(|| {
                    replace_graph_memory(
                        &mut binding,
                        candidate,
                        || Ok::<(), ()>(()),
                        || {
                            for step in 0..6 {
                                assert!(old_bytes.upgrade().is_some());
                                assert!(new_bytes.upgrade().is_some());
                                steps.set(steps.get() + 1);
                                if step == failed_step {
                                    if unwind {
                                        panic!("parameter update interrupted");
                                    }
                                    return Err(());
                                }
                            }
                            Ok(())
                        },
                    )
                }));
                assert_eq!(result.is_err(), unwind);
                if let Ok(result) = result {
                    assert!(result.is_err());
                }
                assert_eq!(steps.get(), failed_step + 1);
                assert!(binding.unusable);
                assert!(old_bytes.upgrade().is_some());
                assert!(new_bytes.upgrade().is_some());
                drop(binding);
                assert!(old_bytes.upgrade().is_none());
                assert!(new_bytes.upgrade().is_none());
            }
        }
    }

    #[test]
    fn graph_memory_wait_failure_preserves_original_binding_without_update() {
        let (old, old_bytes) = crate::memory::test_memory_manifest(|| {});
        let (candidate, new_bytes) = crate::memory::test_memory_manifest(|| {});
        let mut binding = GraphMemoryBinding {
            memory: old,
            unusable: false,
        };
        assert!(replace_graph_memory(
            &mut binding,
            candidate,
            || Err(()),
            || panic!("update after failed wait")
        )
        .is_err());
        assert!(!binding.unusable);
        assert!(old_bytes.upgrade().is_some());
        assert!(new_bytes.upgrade().is_none());
        drop(binding);
        assert!(old_bytes.upgrade().is_none());
    }

    #[test]
    fn graph_memory_success_releases_obsolete_owners_outside_lock() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let gate = Arc::new(Mutex::new(GraphMemoryBinding::default()));
        let weak_gate = Arc::downgrade(&gate);
        let released_outside = Arc::new(AtomicBool::new(false));
        let released = released_outside.clone();
        let (old, old_bytes) = crate::memory::test_memory_manifest(move || {
            released.store(
                weak_gate.upgrade().unwrap().try_lock().is_ok(),
                Ordering::SeqCst,
            );
        });
        let (candidate, new_bytes) = crate::memory::test_memory_manifest(|| {});
        let mut guard = gate.lock().unwrap();
        guard.memory = old;
        let obsolete =
            replace_graph_memory(&mut guard, candidate, || Ok::<(), ()>(()), || Ok(())).unwrap();
        assert!(!guard.unusable);
        assert!(old_bytes.upgrade().is_some());
        assert!(new_bytes.upgrade().is_some());
        drop(guard);
        drop(obsolete);
        assert!(old_bytes.upgrade().is_none());
        assert!(released_outside.load(Ordering::SeqCst));
        assert!(new_bytes.upgrade().is_some());
        drop(gate);
        assert!(new_bytes.upgrade().is_none());
    }

    #[test]
    fn capture_submission_serializes_one_target_without_blocking_another() {
        use std::sync::mpsc::{channel, RecvTimeoutError};
        use std::time::Duration;
        let _capture_test = capture_lifecycle_test_guard();
        let (mut first, phase) = host_capture_phase(
            StreamCaptureKey {
                context: 0x191,
                stream: 0x192,
            },
            0x193,
        );
        let (mut other, other_phase) = host_capture_phase(
            StreamCaptureKey {
                context: 0x191,
                stream: 0x194,
            },
            0x195,
        );
        let (entered_tx, entered_rx) = channel();
        let (release_tx, release_rx) = channel();
        let (second_tx, second_rx) = channel();
        thread::scope(|scope| {
            let first_phase = &phase;
            let callback = scope.spawn(move || {
                let pin = first_phase.pin().unwrap();
                pin.with_serialized_submission(|| {
                    assert!(capture_phase_owners(first_phase).try_lock().is_ok());
                    assert!(ACTIVE_STREAM_CAPTURES.get().unwrap().try_lock().is_ok());
                    pin.with_submission(|_| Ok(()))?;
                    entered_tx.send(()).unwrap();
                    release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                    Ok::<(), DriverError>(())
                })
                .unwrap();
            });
            entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let same_phase = &phase;
            let gate = capture_phase_owners(&phase)
                .lock()
                .unwrap()
                .submission
                .clone();
            assert!(matches!(
                gate.try_lock(),
                Err(std::sync::TryLockError::WouldBlock)
            ));
            let second = scope.spawn(move || {
                same_phase
                    .with_submission(|_| {
                        second_tx.send(()).unwrap();
                        Ok(())
                    })
                    .unwrap();
            });
            other_phase.with_submission(|_| Ok(())).unwrap();
            assert_eq!(
                second_rx.recv_timeout(Duration::from_millis(100)),
                Err(RecvTimeoutError::Timeout)
            );
            release_tx.send(()).unwrap();
            second_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            callback.join().unwrap();
            second.join().unwrap();
        });
        first
            .close_submission_admission()
            .unwrap()
            .wait_until_idle();
        first.complete_driver_capture();
        other
            .close_submission_admission()
            .unwrap()
            .wait_until_idle();
        other.complete_driver_capture();
    }

    #[test]
    fn capture_retained_resources_survive_callback_failure_and_unwind() {
        for unwind in [false, true] {
            let resource = Arc::new(vec![17_u8, 29]);
            let original = Arc::downgrade(&resource);
            let owners = Arc::new(Mutex::new(CapturedOwners::with_external_resources(vec![
                resource,
            ])));
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                capture_outcome(Some(&owners), || {
                    assert_eq!(&**original.upgrade().unwrap(), &[17, 29]);
                    if unwind {
                        panic!("producer capture interrupted");
                    }
                    Err::<(), ()>(())
                })
            }));
            assert_eq!(result.is_err(), unwind);
            assert!(owners.lock().unwrap().failed);
            assert!(original.upgrade().is_some());
            drop(owners);
            assert!(original.upgrade().is_none());
        }
    }

    #[test]
    fn capture_retained_resources_follow_the_child_snapshot() {
        let resource = Arc::new(vec![41_u8, 53]);
        let original = Arc::downgrade(&resource);
        let child = Arc::new(Mutex::new(CapturedOwners::with_external_resources(vec![
            resource,
        ])));
        let parent = CaptureOwners::default();
        let (memory, bytes) = crate::memory::test_memory_manifest(|| {});
        retain_child_capture_owners(&parent, &child, &memory);
        drop(child);
        drop(memory);
        assert_eq!(&**original.upgrade().unwrap(), &[41, 53]);
        assert!(bytes.upgrade().is_some());
        drop(parent);
        assert!(original.upgrade().is_none());
        assert!(bytes.upgrade().is_none());
    }

    #[test]
    fn capture_submission_outer_callback_failure_marks_the_shared_target() {
        for unwind in [false, true] {
            let owners = CaptureOwners::default();
            let (manifest, bytes) = crate::memory::test_memory_manifest(|| {});
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                capture_outcome(Some(&owners), || {
                    // Successful earlier nodes have already transferred their owner.
                    owners.lock().unwrap().memory.push(manifest);
                    if unwind {
                        panic!("outer capture callback interrupted");
                    }
                    Err::<(), ()>(())
                })
            }));
            assert_eq!(result.is_err(), unwind);
            assert!(owners.lock().unwrap().failed);
            assert!(bytes.upgrade().is_some());
            drop(owners);
            assert!(bytes.upgrade().is_none());
        }
    }

    #[test]
    fn capture_submission_swallowed_nested_failure_stays_poisoned() {
        let _capture_test = capture_lifecycle_test_guard();
        for unwind in [false, true] {
            let (mut lease, phase) = host_capture_phase(
                StreamCaptureKey {
                    context: 0x171,
                    stream: 0x172,
                },
                0x173,
            );
            let pin = phase.pin().unwrap();
            pin.with_serialized_submission(|| {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    pin.with_submission(|_| {
                        if unwind {
                            panic!("nested submission interrupted");
                        }
                        Err::<(), _>(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE))
                    })
                }));
                assert_eq!(result.is_err(), unwind);
                pin.with_submission(|_| Ok(()))?;
                Ok::<(), DriverError>(())
            })
            .unwrap();
            assert!(capture_phase_owners(&phase).lock().unwrap().failed);
            drop(pin);
            lease
                .close_submission_admission()
                .unwrap()
                .wait_until_idle();
            lease.complete_driver_capture();
        }
    }

    #[test]
    fn capture_submission_child_snapshot_survives_rebind_and_child_drop() {
        let _capture_test = capture_lifecycle_test_guard();
        let (mut lease, phase) = host_capture_phase(
            StreamCaptureKey {
                context: 0x181,
                stream: 0x182,
            },
            0x183,
        );
        let (old, old_bytes) = crate::memory::test_memory_manifest(|| {});
        let (candidate, new_bytes) = crate::memory::test_memory_manifest(|| {});
        let mut binding = GraphMemoryBinding {
            memory: old,
            unusable: false,
        };
        let pin = phase.pin().unwrap();
        assert!(pin.capture_memory(&binding.memory));
        drop(pin);
        drop(
            replace_graph_memory(&mut binding, candidate, || Ok::<(), ()>(()), || Ok(())).unwrap(),
        );
        assert!(old_bytes.upgrade().is_some());
        let parent = crate::memory::MemoryAccessManifest::combine(
            &capture_phase_owners(&phase).lock().unwrap().memory,
        );
        drop(binding);
        assert!(new_bytes.upgrade().is_none());
        lease
            .close_submission_admission()
            .unwrap()
            .wait_until_idle();
        lease.complete_driver_capture();
        drop(lease);
        drop(phase);
        assert!(old_bytes.upgrade().is_some());
        drop(parent);
        assert!(old_bytes.upgrade().is_none());
    }

    #[test]
    fn capture_submission_close_waits_only_for_executing_callback_and_releases_owners() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::mpsc::{channel, RecvTimeoutError};
        use std::time::Duration;

        struct DropProbe(Arc<AtomicUsize>);
        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let _capture_test = capture_lifecycle_test_guard();
        for unwind in [false, true] {
            let (mut lease, phase) = host_capture_phase(
                StreamCaptureKey {
                    context: 0x101,
                    stream: 0x102,
                },
                0x103,
            );
            let module_owners = Arc::downgrade(capture_phase_owners(&phase));
            let drops = Arc::new(AtomicUsize::new(0));
            let owner = DropProbe(drops.clone());
            retire_after_stream_captures(move || drop(owner));
            let (entered_tx, entered_rx) = channel();
            let (release_tx, release_rx) = channel();
            let (closed_tx, closed_rx) = channel();
            let (waiting_tx, waiting_rx) = channel();
            thread::scope(|scope| {
                let callback_phase = &phase;
                let callback = scope.spawn(move || {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let pin = callback_phase.pin()?;
                        let submit = |owners: Option<&CaptureOwners>| {
                            assert!(Arc::ptr_eq(
                                owners.unwrap(),
                                capture_phase_owners(callback_phase)
                            ));
                            assert!(ACTIVE_STREAM_CAPTURES.get().unwrap().try_lock().is_ok());
                            // Reentrant submission uses the same owner and never takes a
                            // registry mutex recursively around the callback.
                            pin.with_submission(|_| Ok(()))?;
                            entered_tx.send(()).unwrap();
                            release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                            if unwind {
                                panic!("capture submission unwind");
                            }
                            Ok(())
                        };
                        pin.with_serialized_submission(|| pin.with_submission(submit))
                    }))
                });
                entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                // Closing admission is the same production transition used before
                // EndCapture, independent of waiting for the admitted callback.
                let executing = lease.close_submission_admission().unwrap();
                assert!(phase.with_submission(|_| Ok(())).is_err());
                let finish = scope.spawn(move || {
                    waiting_tx.send(()).unwrap();
                    executing.wait_until_idle();
                    lease.complete_driver_capture();
                    drop(lease);
                    closed_tx.send(()).unwrap();
                });
                waiting_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                let registry_available = ACTIVE_STREAM_CAPTURES.get().unwrap().try_lock().is_ok();
                let returned_before_completion = closed_rx.recv_timeout(Duration::from_millis(100));
                let dropped_before_completion = drops.load(Ordering::SeqCst);
                release_tx.send(()).unwrap();
                let callback = callback.join().unwrap();
                finish.join().unwrap();
                assert_eq!(callback.is_err(), unwind);
                if let Ok(result) = callback {
                    result.unwrap();
                }
                assert!(registry_available);
                assert_eq!(returned_before_completion, Err(RecvTimeoutError::Timeout));
                assert_eq!(dropped_before_completion, 0);
            });
            assert_eq!(drops.load(Ordering::SeqCst), 1);
            // The passive saved descriptor neither pins a callback nor permits a
            // post-finish enqueue to downgrade to ordinary execution.
            assert!(phase.with_submission(|_| Ok(())).is_err());
            assert!(module_owners.upgrade().is_some());
            drop(phase);
            assert!(module_owners.upgrade().is_none());
        }
    }

    #[test]
    fn capture_submission_rejects_wrong_identity_without_blocking_other_capture() {
        let _capture_test = capture_lifecycle_test_guard();
        let (mut first, first_phase) = host_capture_phase(
            StreamCaptureKey {
                context: 0x111,
                stream: 0x112,
            },
            0x113,
        );
        let (mut second, second_phase) = host_capture_phase(
            StreamCaptureKey {
                context: 0x111,
                stream: 0x114,
            },
            0x115,
        );
        let first_executing = first.close_submission_admission().unwrap();
        first_executing.wait_until_idle();
        first.complete_driver_capture();
        drop(first);
        second_phase.with_submission(|_| Ok(())).unwrap();
        let ordinary = StreamSubmissionPhase::from_capture_info(
            stream_capture_registry(),
            0x111,
            sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE,
            0,
        )
        .unwrap();
        ordinary
            .with_submission(|owners| {
                assert!(owners.is_none());
                second_phase.with_submission(|_| Ok(()))
            })
            .unwrap();
        drop(ordinary);
        for (context, id, owners) in [
            (0x111, 0x113, capture_phase_owners(&first_phase).clone()),
            (0x116, 0x115, capture_phase_owners(&second_phase).clone()),
            (0x111, 0x115, capture_phase_owners(&first_phase).clone()),
        ] {
            let stale = StreamSubmissionPhase {
                state: StreamSubmissionState::Captured {
                    context,
                    id,
                    owners,
                },
            };
            let mut called = false;
            assert!(stale
                .with_submission(|_| {
                    called = true;
                    Ok(())
                })
                .is_err());
            assert!(!called);
        }
        assert!(StreamSubmissionPhase::from_capture_info(
            stream_capture_registry(),
            0x111,
            sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE,
            0x117,
        )
        .is_err());
        second
            .close_submission_admission()
            .unwrap()
            .wait_until_idle();
        second.complete_driver_capture();
        drop(second);
    }

    #[test]
    fn capture_submission_ordinary_phase_excludes_capture_through_callback() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StreamSubmissionPhase>();
        let _capture_test = capture_lifecycle_test_guard();
        let phase = StreamSubmissionPhase::from_capture_info(
            stream_capture_registry(),
            0x121,
            sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE,
            0,
        )
        .unwrap();
        assert!(matches!(
            phase.state,
            StreamSubmissionState::Ordinary { .. }
        ));
        let key = StreamCaptureKey {
            context: 0x121,
            stream: 0x122,
        };
        let pin = phase.pin().unwrap();
        assert!(pin.owners.is_none());
        assert!(pin._executing.is_none());
        assert!(try_acquire_stream_capture_key(key).is_err());
        drop(pin);
        phase
            .with_submission(|owners| {
                assert!(owners.is_none());
                assert!(ACTIVE_STREAM_CAPTURES.get().unwrap().try_lock().is_ok());
                assert!(try_acquire_stream_capture_key(key).is_err());
                Ok(())
            })
            .unwrap();
        assert!(try_acquire_stream_capture_key(key).is_err());
        drop(phase);
        drop(try_acquire_stream_capture_key(key).unwrap());
    }

    #[test]
    fn ordinary_execution_reservation_excludes_capture_without_draining_or_locking() {
        let _capture_test = crate::cuda_graph::capture_lifecycle_test_guard();
        use std::sync::atomic::{AtomicUsize, Ordering};
        let reservation = reserve_capture_exclusion().unwrap();
        let mutex_available = ACTIVE_STREAM_CAPTURES.get().unwrap().try_lock().is_ok();
        let capture = try_acquire_stream_capture_key(StreamCaptureKey {
            context: 0xca,
            stream: 0xcb,
        });
        assert!(mutex_available);
        assert!(capture.is_err());
        let releases = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&releases);
        retire_after_stream_captures(move || {
            observed.fetch_add(1, Ordering::SeqCst);
        });
        // A cold retirement must not strand completed resources just because
        // another ordinary execution still excludes capture.
        assert_eq!(releases.load(Ordering::SeqCst), 1);
        drop(reservation);
        // Returning from a measured operation does not acquire a new cold wait.
        assert_eq!(releases.load(Ordering::SeqCst), 1);
        drain_capture_retirements(stream_capture_registry());
        assert_eq!(releases.load(Ordering::SeqCst), 1);

        let capture = try_acquire_stream_capture_key(StreamCaptureKey {
            context: 0xca,
            stream: 0xcb,
        })
        .unwrap();
        assert!(reserve_capture_exclusion().is_err());
        drop(capture);
        assert!(reserve_capture_exclusion().is_ok());
    }

    #[test]
    fn capture_retirement_release_waits_without_holding_registry_mutex() {
        let _capture_test = crate::cuda_graph::capture_lifecycle_test_guard();
        use std::sync::mpsc::{channel, TryRecvError};
        use std::sync::TryLockError;
        use std::time::Duration;

        let lease = try_acquire_stream_capture_key(StreamCaptureKey {
            context: 0xc4,
            stream: 0xc5,
        })
        .unwrap();
        let (entered_tx, entered_rx) = channel();
        let (release_tx, release_rx) = channel();
        let (returned_tx, returned_rx) = channel();
        retire_after_stream_captures(move || {
            entered_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        });
        let release = thread::spawn(move || {
            drop(lease);
            returned_tx.send(()).unwrap();
        });

        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        // Inspect without blocking, so a lock regression can still release the
        // waiting destructor and join its thread before failing the assertion.
        let registry_state = match ACTIVE_STREAM_CAPTURES.get().unwrap().try_lock() {
            Ok(registry) => Some((registry.active.is_empty(), registry.retiring)),
            Err(TryLockError::Poisoned(error)) => {
                let registry = error.into_inner();
                Some((registry.active.is_empty(), registry.retiring))
            }
            Err(TryLockError::WouldBlock) => None,
        };
        let returned_before_completion = returned_rx.try_recv();
        release_tx.send(()).unwrap();
        release.join().unwrap();
        assert_eq!(registry_state, Some((true, true)));
        assert_eq!(returned_before_completion, Err(TryRecvError::Empty));
        returned_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(!stream_capture_registry().retiring);
    }

    #[test]
    fn capture_retirement_unwind_preserves_pending_work_and_releases_reservation() {
        let _capture_test = crate::cuda_graph::capture_lifecycle_test_guard();
        use std::panic::{catch_unwind, AssertUnwindSafe};
        use std::sync::atomic::{AtomicBool, Ordering};
        let key = StreamCaptureKey {
            context: 0xe1,
            stream: 0xe2,
        };
        let lease = try_acquire_stream_capture_key(key).unwrap();
        let completed = Arc::new(AtomicBool::new(false));
        retire_after_stream_captures(|| panic!("retirement unwind"));
        let observed = completed.clone();
        retire_after_stream_captures(move || observed.store(true, Ordering::SeqCst));
        assert!(catch_unwind(AssertUnwindSafe(|| drop(lease))).is_err());
        assert!(!completed.load(Ordering::SeqCst));
        drop(try_acquire_stream_capture_key(key).expect("unwind must release reservation"));
        assert!(completed.load(Ordering::SeqCst));
    }

    #[test]
    fn module_loading_excludes_capture_and_releases_after_error_or_unwind() {
        let _capture_test = crate::cuda_graph::capture_lifecycle_test_guard();
        use std::panic::{catch_unwind, AssertUnwindSafe};
        let key = StreamCaptureKey {
            context: 0xf1,
            stream: 0xf2,
        };
        let lease = try_acquire_stream_capture_key(key).unwrap();
        let mut called = false;
        assert_eq!(
            with_module_loading(|| {
                called = true;
                Ok(())
            }),
            Err(DriverError(
                sys::CUresult::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED
            ))
        );
        assert!(!called, "capture rejection must precede module loading");
        drop(lease);
        let result = with_module_loading(|| {
            assert_eq!(
                try_acquire_stream_capture_key(key).unwrap_err(),
                CudaConditionalGraphUnavailable::StreamCaptureBusy
            );
            Err::<(), _>(DriverError(sys::CUresult::CUDA_ERROR_INVALID_PTX))
        });
        assert_eq!(
            result,
            Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_PTX))
        );
        drop(try_acquire_stream_capture_key(key).unwrap());
        assert!(catch_unwind(AssertUnwindSafe(|| {
            with_module_loading(|| -> std::result::Result<(), DriverError> {
                panic!("module load unwind");
            })
        }))
        .is_err());
        drop(try_acquire_stream_capture_key(key).unwrap());
    }

    #[test]
    fn concurrent_module_loads_do_not_report_a_nonexistent_capture() {
        let _capture_test = crate::cuda_graph::capture_lifecycle_test_guard();
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let first = thread::spawn(move || {
            with_module_loading(|| {
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok(())
            })
        });
        entered_rx.recv().unwrap();
        let second = with_module_loading(|| Ok(()));
        let capture = try_acquire_stream_capture_key(StreamCaptureKey {
            context: 0xf3,
            stream: 0xf4,
        });
        release_tx.send(()).unwrap();
        first.join().unwrap().unwrap();
        assert_eq!(second.map_err(|error| error.0), Ok(()));
        assert_eq!(
            capture.unwrap_err(),
            CudaConditionalGraphUnavailable::StreamCaptureBusy
        );
    }

    #[test]
    fn capture_retirement_excludes_new_capture_and_drains_nested_retirement() {
        let _capture_test = crate::cuda_graph::capture_lifecycle_test_guard();
        use std::sync::atomic::{AtomicUsize, Ordering};
        let completed = Arc::new(AtomicUsize::new(0));
        let observed = completed.clone();
        let key = StreamCaptureKey {
            context: 0xd1,
            stream: 0xd2,
        };
        retire_after_stream_captures(move || {
            assert_eq!(
                try_acquire_stream_capture_key(key).unwrap_err(),
                CudaConditionalGraphUnavailable::StreamCaptureBusy
            );
            retire_after_stream_captures(move || {
                observed.fetch_add(1, Ordering::SeqCst);
            });
        });
        assert_eq!(completed.load(Ordering::SeqCst), 1);
        drop(try_acquire_stream_capture_key(key).unwrap());
    }

    #[test]
    fn capture_retirement_keeps_unknown_driver_capture_reserved() {
        let _capture_test = crate::cuda_graph::capture_lifecycle_test_guard();
        use std::sync::atomic::{AtomicBool, Ordering};
        let key = StreamCaptureKey {
            context: 0xe1,
            stream: 0xe2,
        };
        let mut lease = try_acquire_stream_capture_key(key).unwrap();
        lease.driver_active = true;
        let completed = Arc::new(AtomicBool::new(false));
        let observed = completed.clone();
        retire_after_stream_captures(move || observed.store(true, Ordering::SeqCst));
        drop(lease);
        assert!(!completed.load(Ordering::SeqCst));
        assert_eq!(
            try_acquire_stream_capture_key(key).unwrap_err(),
            CudaConditionalGraphUnavailable::StreamCaptureBusy
        );
        // Only this test's host-only reservation is removed. Production removes
        // the reservation only after the CUDA driver proves capture has ended.
        release_stream_capture_key(key);
        assert!(completed.load(Ordering::SeqCst));
    }

    #[test]
    fn same_stream_capture_registry_is_deterministically_busy_until_release() {
        let _capture_test = crate::cuda_graph::capture_lifecycle_test_guard();
        let key = StreamCaptureKey {
            context: 0x51,
            stream: 0x73,
        };
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let first_entered = Arc::clone(&entered);
        let first_release = Arc::clone(&release);
        let first = thread::spawn(move || {
            let _lease = try_acquire_stream_capture_key(key).expect("first capture lease");
            first_entered.wait();
            first_release.wait();
        });
        entered.wait();
        assert_eq!(
            try_acquire_stream_capture_key(key).expect_err("same stream must be busy"),
            CudaConditionalGraphUnavailable::StreamCaptureBusy
        );
        release.wait();
        first.join().expect("first capture thread");
        drop(try_acquire_stream_capture_key(key).expect("capture after release"));
    }

    #[test]
    fn different_stream_capture_registry_entries_can_coexist() {
        let _capture_test = crate::cuda_graph::capture_lifecycle_test_guard();
        let first = try_acquire_stream_capture_key(StreamCaptureKey {
            context: 0x91,
            stream: 0x92,
        })
        .expect("first stream capture");
        let second = try_acquire_stream_capture_key(StreamCaptureKey {
            context: 0x91,
            stream: 0x93,
        })
        .expect("different stream capture");
        drop((first, second));
    }

    #[test]
    fn real_capture_helper_returns_typed_busy_before_beginning_driver_capture() {
        let Some(context) = conditional_cuda_context_or_skip() else {
            return;
        };
        let stream = context.new_stream().expect("non-default CUDA stream");
        let _lease = try_acquire_stream_capture(&stream).expect("held stream capture lease");
        let mut builder = match ConditionalCudaGraphSequenceBuilder::new(&stream) {
            Ok(builder) => builder,
            Err(error) if error.is_unsupported() => return,
            Err(error) => panic!("sequence builder failed: {error}"),
        };
        let error = builder
            .capture_segment_on_stream(&stream, || Ok::<(), XlogError>(()))
            .expect_err("same stream capture must decline before driver capture");
        assert_eq!(error, CudaConditionalGraphUnavailable::StreamCaptureBusy);
    }

    #[test]
    fn capture_callback_error_and_panic_leave_every_capture_path_reusable() {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
            return;
        }
        use std::panic::{catch_unwind, AssertUnwindSafe};
        let context = CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        for path in 0..3 {
            for unwind in [false, true] {
                let fail = || -> Result<()> {
                    if unwind {
                        panic!("capture callback unwind");
                    }
                    Err(XlogError::Kernel("capture callback error".into()))
                };
                let result = catch_unwind(AssertUnwindSafe(|| match path {
                    0 => CapturedCudaGraph::capture_on_stream(&stream, fail).map(|_| ()),
                    1 => CapturedCudaGraph::conditional_while_on_stream(&stream, 0, true, |body| {
                        body.capture_on_stream(&stream, fail)
                    })
                    .map(|_| ())
                    .map_err(|error| XlogError::Kernel(error.decline_detail())),
                    _ => ConditionalCudaGraphSequenceBuilder::new(&stream)
                        .and_then(|mut builder| builder.capture_segment_on_stream(&stream, fail))
                        .map_err(|error| XlogError::Kernel(error.decline_detail())),
                }));
                if unwind {
                    assert!(result.is_err(), "callback must unwind for path {path}");
                } else {
                    assert!(result
                        .unwrap()
                        .unwrap_err()
                        .to_string()
                        .contains("callback error"));
                }
                let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE;
                unsafe { sys::cuStreamIsCapturing(stream.cu_stream(), &mut status).result() }
                    .unwrap();
                assert_eq!(
                    status,
                    sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
                );
                let graph = CapturedCudaGraph::capture_on_stream(&stream, || Ok(())).unwrap();
                graph.launch(&stream).unwrap();
                stream.synchronize().unwrap();
            }
        }
    }

    #[test]
    fn per_thread_default_streams_have_distinct_execution_ids() {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() != Ok("1") {
            return;
        }
        let context = CudaContext::new(0).unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let mut threads = Vec::new();
        for _ in 0..2 {
            let context = context.clone();
            let barrier = barrier.clone();
            threads.push(thread::spawn(move || {
                let stream = context.per_thread_stream();
                let id = stream_execution_id(&stream).unwrap();
                barrier.wait();
                (stream.cu_stream() as usize, id)
            }));
        }
        let first = threads.remove(0).join().unwrap();
        let second = threads.remove(0).join().unwrap();
        assert_eq!(
            first.0, second.0,
            "both wrappers use the per-thread sentinel"
        );
        assert_ne!(
            first.1, second.1,
            "completion fences need distinct execution IDs"
        );
    }

    #[cfg(unix)]
    #[test]
    fn missing_driver_symbol_is_a_typed_error_instead_of_a_panic() {
        let library = libloading::Library::from(libloading::os::unix::Library::this());
        let error = unsafe {
            load_required_symbol::<ConditionalHandleCreateFn>(
                &library,
                b"xlog_missing_cuda_conditional_symbol_for_test\0",
                "xlog_missing_cuda_conditional_symbol_for_test",
            )
        }
        .expect_err("the deliberately absent symbol must fail closed");

        assert_eq!(
            error,
            CudaConditionalGraphUnavailable::MissingDriverSymbol {
                symbol: "xlog_missing_cuda_conditional_symbol_for_test",
            }
        );
        assert!(error.is_unsupported());
        assert_eq!(
            error.decline_detail(),
            "CUDA driver is missing required conditional-graph symbol \
             xlog_missing_cuda_conditional_symbol_for_test"
        );
    }

    #[test]
    fn driver_versions_before_cuda_twelve_three_decline_conditionals() {
        let error = require_conditional_graph_driver(12_020).expect_err("CUDA 12.2 is too old");
        assert_eq!(
            error,
            CudaConditionalGraphUnavailable::DriverVersionTooOld {
                found: 12_020,
                required: 12_030,
            }
        );
        assert!(error.is_unsupported());
        assert_eq!(
            error.decline_detail(),
            "CUDA conditional graphs require driver API 12030, found 12020"
        );
        require_conditional_graph_driver(12_030).expect("CUDA 12.3 is supported");
    }

    #[test]
    fn while_node_params_use_the_driver_abi_and_return_one_body() {
        let handle = 0x1234_u64;
        let ctx = 0x5678_usize as sys::CUcontext;
        let params = conditional_node_params(
            handle,
            ctx,
            sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_WHILE,
        );

        assert_eq!(
            params.type_,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL
        );
        let conditional = unsafe { params.__bindgen_anon_1.conditional };
        assert_eq!(conditional.handle, handle);
        assert_eq!(
            conditional.type_,
            sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_WHILE
        );
        assert_eq!(conditional.size, 1);
        assert!(conditional.phGraph_out.is_null());
        assert_eq!(conditional.ctx, ctx);
    }

    #[test]
    fn conditional_node_params_preserve_one_shot_and_repeating_body_semantics() {
        let handle = 0x1234_u64;
        let context = 0x5678_usize as sys::CUcontext;
        for kind in [
            sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_IF,
            sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_WHILE,
        ] {
            let params = conditional_node_params(handle, context, kind);
            assert_eq!(
                params.type_,
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL
            );
            // SAFETY: the production constructor selected the conditional ABI.
            let conditional = unsafe { params.__bindgen_anon_1.conditional };
            assert_eq!(conditional.type_, kind);
            assert_eq!(conditional.handle, handle);
            assert_eq!(conditional.ctx, context);
            assert_eq!(conditional.size, 1);
            assert!(conditional.phGraph_out.is_null());
        }
    }

    #[test]
    fn conditional_body_exposes_device_setter_handle_and_context() {
        let graph = 0x1234_usize as sys::CUgraph;
        let handle = 0x5678_u64;
        let context = 0x9abc_usize as sys::CUcontext;
        let body = ConditionalCudaGraphBody {
            graph,
            handle,
            context,
            modules: Arc::default(),
        };

        assert_eq!(body.graph(), graph);
        assert_eq!(body.handle(), handle);
        assert_eq!(body.context(), context);
    }

    #[test]
    fn real_conditional_while_graph_creates_instantiates_and_launches() {
        let Some(context) = conditional_cuda_context_or_skip() else {
            return;
        };
        let stream = context.new_stream().expect("non-default CUDA stream");
        let body_buffer = stream.alloc_zeros::<u32>(1).expect("body buffer");
        let (body_ptr, _body_sync) = body_buffer.device_ptr(&stream);
        let ptx = compile_ptx(
            r#"
            extern "C" __device__ void cudaGraphSetConditional(
                unsigned long long handle,
                unsigned int value
            );

            extern "C" __global__ void run_once(
                unsigned long long handle,
                unsigned int *counter
            ) {
                if (blockIdx.x == 0 && threadIdx.x == 0) {
                    *counter += 1;
                    cudaGraphSetConditional(handle, 0);
                }
            }
            "#,
        )
        .expect("compile conditional setter kernel");
        let module = context.load_module(ptx).expect("load setter module");
        let run_once = module
            .load_function("run_once")
            .expect("load setter function");
        let graph = CapturedCudaGraph::conditional_while_on_stream(&stream, 1, true, |body| {
            body.capture_on_stream(&stream, || {
                let handle = body.handle();
                let mut launch = stream.launch_builder(&run_once);
                launch.arg(&handle).arg(&body_ptr);
                unsafe { launch.launch(LaunchConfig::for_num_elems(1)) }
                    .map(|_| ())
                    .map_err(|error| XlogError::Kernel(error.to_string()))
            })
        });
        let graph = match graph {
            Ok(graph) => graph,
            Err(error) if error.is_unsupported() => {
                if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
                    panic!("CUDA conditional graphs are required: {error}");
                }
                eprintln!("Skipping test: {error}");
                return;
            }
            Err(error) => panic!("conditional graph construction failed: {error}"),
        };

        assert_eq!(graph.node_count().expect("node count"), 1);
        assert_eq!(
            graph.nodes().expect("nodes")[0].kind,
            CudaGraphNodeKind::Conditional
        );
        graph.launch(&stream).expect("conditional graph launch");
        stream.synchronize().expect("conditional graph completion");
        let mut observed = [0_u32; 1];
        stream
            .memcpy_dtoh(&body_buffer, &mut observed)
            .expect("read body effect");
        assert_eq!(observed, [1], "WHILE body must execute exactly once");
    }

    #[test]
    fn real_conditional_sequence_orders_segments_around_multiple_while_capability() {
        let Some(context) = conditional_cuda_context_or_skip() else {
            return;
        };
        let stream = context.new_stream().expect("non-default CUDA stream");
        let buffer = stream.alloc_zeros::<u32>(1).expect("sequence buffer");
        let (buffer_ptr, _buffer_sync) = buffer.device_ptr(&stream);
        let ptx = compile_ptx(
            r#"
            extern "C" __device__ void cudaGraphSetConditional(
                unsigned long long handle,
                unsigned int value
            );

            extern "C" __global__ void add_value(
                unsigned int *counter,
                unsigned int value
            ) {
                if (blockIdx.x == 0 && threadIdx.x == 0) *counter += value;
            }

            extern "C" __global__ void add_once(
                unsigned long long handle,
                unsigned int *counter
            ) {
                if (blockIdx.x == 0 && threadIdx.x == 0) {
                    *counter += 2;
                    cudaGraphSetConditional(handle, 0);
                }
            }
            "#,
        )
        .expect("compile sequence kernels");
        let module = context.load_module(ptx).expect("load sequence module");
        let add_value = module
            .load_function("add_value")
            .expect("load add function");
        let add_once = module
            .load_function("add_once")
            .expect("load conditional function");

        let sequence = ConditionalCudaGraphSequenceBuilder::new(&stream).and_then(|mut builder| {
            builder.capture_segment_on_stream(&stream, || {
                let value = 1_u32;
                let mut launch = stream.launch_builder(&add_value);
                launch.arg(&buffer_ptr).arg(&value);
                unsafe { launch.launch(LaunchConfig::for_num_elems(1)) }
                    .map(|_| ())
                    .map_err(|error| XlogError::Kernel(error.to_string()))
            })?;
            builder.add_conditional_while(1, true, |body| {
                body.capture_on_stream(&stream, || {
                    let handle = body.handle();
                    let mut launch = stream.launch_builder(&add_once);
                    launch.arg(&handle).arg(&buffer_ptr);
                    unsafe { launch.launch(LaunchConfig::for_num_elems(1)) }
                        .map(|_| ())
                        .map_err(|error| XlogError::Kernel(error.to_string()))
                })
            })?;
            builder.capture_segment_on_stream(&stream, || {
                let value = 4_u32;
                let mut launch = stream.launch_builder(&add_value);
                launch.arg(&buffer_ptr).arg(&value);
                unsafe { launch.launch(LaunchConfig::for_num_elems(1)) }
                    .map(|_| ())
                    .map_err(|error| XlogError::Kernel(error.to_string()))
            })?;
            builder.instantiate()
        });
        let graph = match sequence {
            Ok(graph) => graph,
            Err(error) if error.is_unsupported() => {
                if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
                    panic!("CUDA conditional graphs are required: {error}");
                }
                eprintln!("Skipping test: {error}");
                return;
            }
            Err(error) => panic!("conditional graph sequence construction failed: {error}"),
        };
        assert_eq!(graph.node_count().expect("sequence node count"), 3);
        graph.launch(&stream).expect("sequence graph launch");
        stream.synchronize().expect("sequence graph completion");
        let mut observed = [0_u32; 1];
        stream
            .memcpy_dtoh(&buffer, &mut observed)
            .expect("read sequence effect");
        assert_eq!(observed, [7]);
    }

    #[test]
    #[ignore = "requires explicitly authorized CUDA conditional graph execution"]
    fn conditional_if_skips_the_whole_body_and_rechecks_each_replay() {
        let context = conditional_cuda_context_or_skip().expect("CUDA context required");
        let stream = context.new_stream().expect("non-default CUDA stream");
        let mut admission = stream.alloc_zeros::<u32>(1).expect("admission input");
        let counters = stream
            .alloc_zeros::<u32>(2)
            .expect("body and suffix counters");
        let (admission_ptr, _admission_sync) = admission.device_ptr(&stream);
        let (counters_ptr, _counters_sync) = counters.device_ptr(&stream);
        drop(_admission_sync);
        let ptx = compile_ptx(
            r#"
            extern "C" __device__ void cudaGraphSetConditional(
                unsigned long long handle, unsigned int value);
            extern "C" __global__ void preflight(
                unsigned long long handle, const unsigned int *admission) {
                if (blockIdx.x == 0 && threadIdx.x == 0)
                    cudaGraphSetConditional(handle, *admission);
            }
            extern "C" __global__ void body(unsigned int *counters) {
                if (blockIdx.x == 0 && threadIdx.x == 0) ++counters[0];
            }
            extern "C" __global__ void suffix(unsigned int *counters) {
                if (blockIdx.x == 0 && threadIdx.x == 0) ++counters[1];
            }
        "#,
        )
        .expect("compile device admission and body kernels");
        let module = context.load_module(ptx).expect("load conditional module");
        let preflight = module.load_function("preflight").expect("preflight kernel");
        let body_kernel = module.load_function("body").expect("body kernel");
        let suffix = module.load_function("suffix").expect("suffix kernel");
        let resource = Arc::new(vec![17_u8, 29]);
        let retained_resource = Arc::downgrade(&resource);
        let mut builder =
            ConditionalCudaGraphSequenceBuilder::new_retaining(&stream, vec![resource])
                .expect("conditional graph support required");
        assert!(retained_resource.upgrade().is_some());
        builder
            .add_conditional_if(
                &stream,
                |handle| {
                    let mut launch = stream.launch_builder(&preflight);
                    launch.arg(&handle).arg(&admission_ptr);
                    unsafe { launch.launch(LaunchConfig::for_num_elems(1)) }
                        .map(|_| ())
                        .map_err(|error| XlogError::Kernel(error.to_string()))
                },
                |body| {
                    body.capture_on_stream(&stream, || {
                        let mut launch = stream.launch_builder(&body_kernel);
                        launch.arg(&counters_ptr);
                        unsafe { launch.launch(LaunchConfig::for_num_elems(1)) }
                            .map(|_| ())
                            .map_err(|error| XlogError::Kernel(error.to_string()))
                    })
                },
            )
            .expect("capture guarded body");
        builder
            .capture_segment_on_stream(&stream, || {
                let mut launch = stream.launch_builder(&suffix);
                launch.arg(&counters_ptr);
                unsafe { launch.launch(LaunchConfig::for_num_elems(1)) }
                    .map(|_| ())
                    .map_err(|error| XlogError::Kernel(error.to_string()))
            })
            .expect("capture dependent suffix");
        let graph = builder.instantiate().expect("instantiate guarded sequence");
        assert!(retained_resource.upgrade().is_some());
        assert_eq!(graph.node_count().expect("parent node count"), 3);
        for (allowed, expected) in [(0, [0, 1]), (1, [1, 2]), (0, [1, 3]), (1, [2, 4])] {
            // Explicit test input change between completed launches, outside capture.
            stream
                .memcpy_htod(&[allowed], &mut admission)
                .expect("set admission");
            graph.launch(&stream).expect("launch guarded sequence");
            stream.synchronize().expect("guarded sequence completion");
            let mut observed = [0_u32; 2];
            stream
                .memcpy_dtoh(&counters, &mut observed)
                .expect("read execution counters");
            assert_eq!(observed, expected);
        }
    }

    #[test]
    fn scan_topology_matches_recursive_multiblock_shape() {
        assert_eq!(
            scan_topology_u32(0),
            ScanTopology {
                input_len: 0,
                block_size: 256,
                scratch_lengths: vec![],
                kernel_node_count: 0,
            }
        );
        assert_eq!(scan_topology_u32(256).scratch_lengths, Vec::<u32>::new());
        assert_eq!(scan_topology_u32(256).kernel_node_count, 1);
        assert_eq!(scan_topology_u32(257).scratch_lengths, vec![2]);
        assert_eq!(scan_topology_u32(257).kernel_node_count, 3);
        assert_eq!(scan_topology_u32(65_537).scratch_lengths, vec![257, 2]);
        assert_eq!(scan_topology_u32(65_537).kernel_node_count, 5);
    }

    #[test]
    fn csm_key_uses_capacity_classes_and_layout_version() {
        let key = CsmCudaGraphKey::inner(2, 16, 257, 513).expect("key");
        assert_eq!(key.join_kind, CsmCudaGraphJoinKind::Inner);
        assert_eq!(key.key_arity, 2);
        assert_eq!(key.key_bytes, 16);
        assert_eq!(key.probe_capacity_class, 512);
        assert_eq!(key.output_capacity_class, 1024);
        assert_eq!(key.scan_topology.scratch_lengths, vec![2]);
        assert_eq!(key.node_layout_version, CSM_CUDA_GRAPH_NODE_LAYOUT_VERSION);
    }
}
