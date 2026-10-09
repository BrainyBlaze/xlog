//! Cold custody of the original process observer and source operation interval.

use super::*;
use rand::{rngs::OsRng, RngCore};
use std::ffi::c_void;
use std::mem::{align_of, size_of};
use std::thread::{self, ThreadId};

const COMPLETE: u32 = 0;
const INCOMPLETE: u32 = 1;
const UNKNOWN: u32 = 2;
const START_VALID: u64 = 1;
const END_VALID: u64 = 2;
const DELIVERY_DRAINED: u64 = 4;
const STARTED_BEFORE_CUDA: u64 = 8;
const BACKING_PEAK_VALID: u64 = 32;
const ALLOCATOR_COMPLETED: u64 = 64;
const BASELINE_COMPLETE: u64 = 128;
const OWNED_BACKING: u64 = 128;
pub(super) const MEMORY_UNIT: &str = "owned-gpu-backing-peak-bytes";
pub(super) const MEMORY_ADMISSION: &str = "whole-device-total-physical-bytes";
pub(super) const PROGRAM_FORMAT: &str = "dlm-new.scientific-learning/2";
#[cfg(feature = "semantic-policy")]
const CANCEL_GROUP: u32 = 2;

type Begin = unsafe extern "C" fn(*mut c_void, *const u8, u64, *mut *mut c_void) -> u32;
type Finish = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
type Read = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut TraceView) -> u32;
type Release = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
type BindStep = unsafe extern "C" fn(*mut c_void, *mut c_void, u64, u64, u64, u32) -> u32;
type ReadStep = unsafe extern "C" fn(*mut c_void, *mut c_void, u64, *mut TraceView) -> u32;
type ReadCertificate = unsafe extern "C" fn(*mut c_void, *const u8, *mut DeviceCertificate) -> u32;
type RecordRoot =
    unsafe extern "C" fn(*mut c_void, *const xlog_cuda::memory::GpuBackingRoot) -> u32;
type ResolveStepCancellation =
    unsafe extern "C" fn(*mut c_void, *mut c_void, u64, u64, u64) -> u32;

// These are the exact layouts of the installed xlog_resource_observer.h.
// Event storage remains producer-owned; this memory consumer never dereferences it.
#[repr(C)]
struct ApiHeader {
    abi_version: u32,
    struct_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Api {
    abi_version: u32,
    struct_size: u32,
    context: *mut c_void,
    begin: Option<Begin>,
    finish: Option<Finish>,
    read: Option<Read>,
    release: Option<Release>,
    bind_step: Option<BindStep>,
    read_step: Option<ReadStep>,
    read_certificate: Option<ReadCertificate>,
    record_root: Option<RecordRoot>,
    resolve_step_cancellation: Option<ResolveStepCancellation>,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct DeviceCertificate {
    struct_size: u32,
    state: u32,
    observer_identity: [u8; 32],
    device_uuid: [u8; 16],
    total_physical_bytes: u64,
    flags: u64,
    process_id: u64,
    driver_version: [u8; 32],
    source: u32,
    reserved: u32,
}

#[repr(C)]
struct TraceView {
    struct_size: u32,
    state: u32,
    original_attempt_nonce: [u8; 32],
    operation_ordinal: u64,
    start: u64,
    end: u64,
    boundary_flags: u64,
    requested_coverage: u64,
    enabled_coverage: u64,
    unsupported_coverage: u64,
    process_dropped_records: u64,
    process_errors: u64,
    process_buffer_overflows: u64,
    certificate_identity: [u8; 32],
    backing_baseline_bytes: u64,
    backing_peak_bytes: u64,
    host_to_device_bytes: u64,
    device_to_host_bytes: u64,
    device_to_device_bytes: u64,
    peer_copy_bytes: u64,
    unified_host_to_device_bytes: u64,
    unified_device_to_host_bytes: u64,
    events: *const c_void,
    event_count: u64,
    event_size: u64,
}

const _: () = assert!(
    size_of::<Api>() == 88
        && size_of::<TraceView>() == 240
        && size_of::<DeviceCertificate>() == 120
        && size_of::<xlog_cuda::memory::GpuBackingRoot>() == 64
);

fn checked_api(capsule: &Bound<'_, PyCapsule>) -> PyResult<(usize, Api)> {
    let pointer = capsule.pointer_checked(Some(c"xlog.resource_observer.v4"))?;
    if !(pointer.as_ptr() as usize).is_multiple_of(align_of::<Api>()) {
        return Err(invalid(
            "resource observer table has invalid C ABI alignment",
        ));
    }
    // SAFETY: original trusted capsule pins its immutable C table. Read only
    // the common header before checking the exact new layout, never an old ABI.
    let header = unsafe { &*pointer.as_ptr().cast::<ApiHeader>() };
    if header.abi_version != 4 || header.struct_size as usize != size_of::<Api>() {
        return Err(invalid(
            "resource observer requires the exact installed C ABI table",
        ));
    }
    // SAFETY: checked original header establishes this complete immutable ABI.
    let api = unsafe { *pointer.as_ptr().cast::<Api>() };
    if api.context.is_null()
        || api.record_root.is_none()
        || api.read_certificate.is_none()
        || api.resolve_step_cancellation.is_none()
    {
        return Err(invalid(
            "resource observer requires its original backing producer and device certificate",
        ));
    }
    Ok((pointer.as_ptr() as usize, api))
}

struct ProcessBackingObserver {
    _capsule: Py<PyCapsule>,
    table: usize,
    context: usize,
    creator: ThreadId,
    record: RecordRoot,
    certificate: Mutex<Option<DeviceCertificate>>,
}

static PROCESS_BACKING_OBSERVER: Mutex<Option<Arc<ProcessBackingObserver>>> = Mutex::new(None);

impl xlog_cuda::memory::GpuBackingObserver for ProcessBackingObserver {
    fn record(
        &self,
        root: &xlog_cuda::memory::GpuBackingRoot,
    ) -> xlog_cuda::device_runtime::ResourceResult<()> {
        // SAFETY: process-global custody pins the original immutable table,
        // context and function code. The collector's recording entrypoint is
        // thread-safe and never enters Python/CUDA, including on cold reapers.
        let status = unsafe { (self.record)(self.context as *mut c_void, root) };
        if status != COMPLETE {
            return Err(xlog_cuda::device_runtime::ResourceError::Driver(
                "original backing root registration is unknown; retain its owner".into(),
            ));
        }
        Ok(())
    }
}

/// Called by the original process collector before GPU-bearing model imports.
/// Repeated access to that same original table is idempotent, never a reset.
#[pyfunction]
pub(crate) fn register_process_resource_observer(capsule: &Bound<'_, PyCapsule>) -> PyResult<()> {
    let (table, api) = checked_api(capsule)?;
    let mut retained = PROCESS_BACKING_OBSERVER
        .lock()
        .map_err(|_| invalid("original process backing observer custody is poisoned"))?;
    if let Some(original) = retained.as_ref() {
        if original.table != table
            || original.context != api.context as usize
            || original.creator != thread::current().id()
        {
            return Err(invalid(
                "the original process backing observer cannot be replaced",
            ));
        }
        return Ok(());
    }
    let observer = Arc::new(ProcessBackingObserver {
        _capsule: capsule.clone().unbind(),
        table,
        context: api.context as usize,
        creator: thread::current().id(),
        record: api.record_root.expect("checked original root callback"),
        certificate: Mutex::new(None),
    });
    xlog_cuda::memory::install_gpu_backing_observer(
        Arc::clone(&observer) as Arc<dyn xlog_cuda::memory::GpuBackingObserver>
    )
    .map_err(|error| invalid(&error.to_string()))?;
    *retained = Some(observer);
    Ok(())
}

fn original_certificate(
    capsule: &Bound<'_, PyCapsule>,
    uuid: [u8; 16],
) -> PyResult<DeviceCertificate> {
    let (table, api) = checked_api(capsule)?;
    let retained = PROCESS_BACKING_OBSERVER
        .lock()
        .map_err(|_| invalid("original process backing observer custody is poisoned"))?;
    let original = retained
        .as_ref()
        .filter(|original| {
            original.table == table
                && original.context == api.context as usize
                && original.creator == thread::current().id()
        })
        .ok_or_else(|| {
            invalid(
                "device certificate requires the original collector installed before allocation",
            )
        })?;
    // SAFETY: this exact C layout consists only of integers and byte arrays.
    // Zeroed storage grants no evidence until the producer result is checked.
    let mut certificate: DeviceCertificate = unsafe { std::mem::zeroed() };
    certificate.struct_size = size_of::<DeviceCertificate>() as u32;
    certificate.state = UNKNOWN;
    // SAFETY: original immutable capsule pins this producer and output layout.
    let status = unsafe {
        api.read_certificate
            .expect("checked original certificate reader")(
            api.context,
            uuid.as_ptr(),
            &mut certificate,
        )
    };
    let driver_end = certificate
        .driver_version
        .iter()
        .position(|byte| *byte == 0);
    if status != COMPLETE
        || certificate.state != COMPLETE
        || certificate.struct_size as usize != size_of::<DeviceCertificate>()
        || certificate.observer_identity == [0; 32]
        || certificate.device_uuid != uuid
        || certificate.total_physical_bytes == 0
        || certificate.flags != 3
        || certificate.process_id != u64::from(std::process::id())
        || certificate.source != 1
        || certificate.reserved != 0
        || driver_end.is_none_or(|end| {
            end == 0
                || !certificate.driver_version[..end]
                    .iter()
                    .all(u8::is_ascii_graphic)
                || certificate.driver_version[end..]
                    .iter()
                    .any(|byte| *byte != 0)
        })
    {
        return Err(invalid("whole-device physical capacity lacks its original before-CUDA NVML certificate and actual device binding"));
    }
    let mut saved = original
        .certificate
        .lock()
        .map_err(|_| invalid("original device certificate custody is poisoned"))?;
    if saved.as_ref().is_some_and(|saved| *saved != certificate) {
        return Err(invalid(
            "the original whole-device certificate cannot change or cover a second GPU",
        ));
    }
    *saved = Some(certificate);
    Ok(certificate)
}

impl DeviceCertificate {
    fn value(&self) -> serde_json::Value {
        let hex = |bytes: &[u8]| {
            bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        let end = self
            .driver_version
            .iter()
            .position(|byte| *byte == 0)
            .expect("validated original driver version");
        serde_json::json!({
            "format": "xlog.device-capacity/1",
            "observer_identity": hex(&self.observer_identity),
            "device_uuid": hex(&self.device_uuid),
            "total_physical_bytes": self.total_physical_bytes,
            "process_id": self.process_id,
            "driver_version": std::str::from_utf8(&self.driver_version[..end])
                .expect("validated original ASCII driver version"),
            "source": "nvml-memory-v2", "whole_device": true,
        })
    }

    fn python_value<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dictionary = PyDict::new(py);
        for (name, value) in self.value().as_object().expect("certificate projection") {
            match value {
                serde_json::Value::String(value) => dictionary.set_item(name, value)?,
                serde_json::Value::Bool(value) => dictionary.set_item(name, value)?,
                serde_json::Value::Number(value) => {
                    dictionary.set_item(name, value.as_u64().expect("certificate u64"))?
                }
                _ => unreachable!("certificate has only original scalar fields"),
            }
        }
        Ok(dictionary)
    }
}

fn registered_device_certificate(
    py: Python<'_>,
    device_ordinal: usize,
) -> PyResult<DeviceCertificate> {
    let capsule = {
        let retained = PROCESS_BACKING_OBSERVER
            .lock()
            .map_err(|_| invalid("original process observer custody is poisoned"))?;
        let original = retained.as_ref().ok_or_else(|| {
            invalid("device admission requires the collector registered before CUDA")
        })?;
        if original.creator != thread::current().id() {
            return Err(invalid(
                "device admission requires the original observer creator",
            ));
        }
        original._capsule.clone_ref(py)
    };
    let ordinal = i32::try_from(device_ordinal)
        .map_err(|_| invalid("device ordinal exceeds the CUDA device range"))?;
    let mut device = 0;
    let mut uuid = cudarc::driver::sys::CUuuid { bytes: [0; 16] };
    // SAFETY: cold device lookup creates no context, stream, GPU allocation or
    // model. The original process observer was installed before cuInit.
    unsafe {
        cudarc::driver::sys::cuInit(0).result().map_err(xlog_err)?;
        cudarc::driver::sys::cuDeviceGet(&mut device, ordinal)
            .result()
            .map_err(xlog_err)?;
        cudarc::driver::sys::cuDeviceGetUuid_v2(&mut uuid, device)
            .result()
            .map_err(xlog_err)?;
    }
    original_certificate(capsule.bind(py), uuid.bytes.map(|byte| byte as u8))
}

/// Original native projection for the application's before-model admission.
/// No Python memory supplier or allocation/reservation substitutes for U.
#[pyfunction]
pub(crate) fn resource_observer_device_certificate(
    py: Python<'_>,
    device_ordinal: usize,
) -> PyResult<Py<PyDict>> {
    Ok(registered_device_certificate(py, device_ordinal)?
        .python_value(py)?
        .unbind())
}

pub(in crate::semantic_transition) fn require_restore_memory_admission(
    py: Python<'_>,
    device_ordinal: usize,
    program: &[u8],
) -> PyResult<()> {
    let certificate = registered_device_certificate(py, device_ordinal)?;
    cold_restore::require_program_memory(&cold_restore::json(program)?, &certificate.value())
}

struct StepInterval {
    ordinal: u64,
    capture_stream: u64,
    start_confirmed: bool,
    end_attempted: bool,
    capture_finished: bool,
    boundaries: Option<(u64, u64)>,
    peak: Option<u64>,
}

struct Interval {
    nonce: [u8; 32],
    ordinal: u64,
    handle: usize,
    begun: bool,
    finish_attempted: bool,
    boundaries: Option<(u64, u64)>,
    backing_peak: Option<u64>,
    release_attempted: bool,
    released: bool,
    steps: Vec<StepInterval>,
    cancellation_attempted: bool,
    cancellation_arguments: Option<(u64, u64, u64)>,
    cancellation_status: u32,
    cancelled_roster: bool,
}

/// The capsule pins the original immutable table, context and function code.
/// Stored addresses grant no cross-thread execution rights: every operation
/// checks the original creator before converting them back to native pointers.
/// There is deliberately no Drop implementation that releases unknown work.
pub(super) struct ResourceObserver {
    capsule: Py<PyCapsule>,
    table: usize,
    context: usize,
    creator: ThreadId,
    begin: Begin,
    finish: Finish,
    read: Read,
    release: Release,
    bind_step: BindStep,
    read_step: ReadStep,
    resolve_step_cancellation: ResolveStepCancellation,
    certificate: DeviceCertificate,
    original_context: u64,
    interval: Mutex<Option<Interval>>,
}

impl ResourceObserver {
    pub(super) fn capture(
        capsule: &Bound<'_, PyCapsule>,
        (uuid, original_context): ([u8; 16], u64),
    ) -> PyResult<Self> {
        let (table, api) = checked_api(capsule)?;
        let retained = PROCESS_BACKING_OBSERVER
            .lock()
            .map_err(|_| invalid("original process backing observer custody is poisoned"))?;
        if original_context == 0
            || retained.as_ref().is_none_or(|original| {
                original.table != table
                    || original.context != api.context as usize
                    || original.creator != thread::current().id()
            })
        {
            return Err(invalid("phase observation requires the collector installed before native allocation and its actual CUDA context"));
        }
        drop(retained);
        let certificate = original_certificate(capsule, uuid)?;
        let missing = || {
            invalid(
                "resource observer requires its original context, interval and step-boundary functions",
            )
        };
        if api.context.is_null() {
            return Err(missing());
        }
        Ok(Self {
            capsule: capsule.clone().unbind(),
            table,
            context: api.context as usize,
            creator: thread::current().id(),
            begin: api.begin.ok_or_else(missing)?,
            finish: api.finish.ok_or_else(missing)?,
            read: api.read.ok_or_else(missing)?,
            release: api.release.ok_or_else(missing)?,
            bind_step: api.bind_step.ok_or_else(missing)?,
            read_step: api.read_step.ok_or_else(missing)?,
            resolve_step_cancellation: api.resolve_step_cancellation.ok_or_else(missing)?,
            certificate,
            original_context,
            interval: Mutex::new(None),
        })
    }

    pub(super) fn require_original(&self, py: Python<'_>) -> PyResult<()> {
        if thread::current().id() != self.creator
            || self
                .capsule
                .bind(py)
                .pointer_checked(Some(c"xlog.resource_observer.v4"))?
                .as_ptr() as usize
                != self.table
        {
            return Err(invalid(
                "resource observer lost its original creator or capsule table",
            ));
        }
        Ok(())
    }

    pub(super) fn require_memory_admission(&self, program: &[u8]) -> PyResult<()> {
        if self.original_context == 0 {
            return Err(invalid(
                "phase memory admission lost its original CUDA context",
            ));
        }
        cold_restore::require_program_memory(
            &cold_restore::json(program)?,
            &self.certificate.value(),
        )
    }

    pub(super) fn require_observed_history(&self, program: &[u8], history: &[u8]) -> PyResult<()> {
        cold_restore::require_memory_history(
            &cold_restore::json(program)?,
            &cold_restore::json(history)?,
            Some(&self.certificate.value()),
        )
    }

    pub(super) fn observation_arguments(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        self.require_original(py)?;
        arguments.set_item("memory_unit", MEMORY_UNIT)?;
        arguments.set_item("memory_admission", MEMORY_ADMISSION)?;
        arguments.set_item("device_certificate", self.certificate.python_value(py)?)
    }

    fn interval(&self) -> PyResult<MutexGuard<'_, Option<Interval>>> {
        self.interval
            .lock()
            .map_err(|_| invalid("original resource interval custody mutex is poisoned"))
    }

    /// Called only after exact durable admission and before source native work.
    pub(super) fn begin(&self, py: Python<'_>, ordinal: u64) -> PyResult<()> {
        self.require_original(py)?;
        let mut retained = self.interval()?;
        if retained
            .as_ref()
            .is_some_and(|original| !original.released || ordinal <= original.ordinal)
        {
            return Err(invalid(
                "the next resource interval requires known release and a later original operation; an unknown interval cannot be replaced",
            ));
        }
        let mut nonce = [0; 32];
        OsRng
            .try_fill_bytes(&mut nonce)
            .map_err(|_| invalid("original resource interval nonce generation failed"))?;
        *retained = Some(Interval {
            nonce,
            ordinal,
            handle: 0,
            begun: false,
            finish_attempted: false,
            boundaries: None,
            backing_peak: None,
            release_attempted: false,
            released: false,
            steps: Vec::new(),
            cancellation_attempted: false,
            cancellation_arguments: None,
            cancellation_status: UNKNOWN,
            cancelled_roster: false,
        });
        let interval = retained.as_mut().expect("retained before original begin");
        let mut handle = std::ptr::null_mut();
        // SAFETY: the captured native table/context is pinned by the capsule;
        // nonce and output storage live through this creator-thread C call.
        let status = unsafe {
            (self.begin)(
                self.context as *mut c_void,
                interval.nonce.as_ptr(),
                ordinal,
                &mut handle,
            )
        };
        // Retain even an interval returned alongside a failing begin. Neither a
        // missing handle nor an unknown status authorizes a replacement begin.
        interval.handle = handle as usize;
        interval.begun = status == COMPLETE && !handle.is_null();
        if !interval.begun {
            return Err(invalid(
                "original resource interval begin is unknown; retain the same pending and observer",
            ));
        }
        Ok(())
    }

    /// These are cold capture delimiters, NOT execution timestamps. The same
    /// producer binds the actual captured node/correlation identities, then
    /// obtains original GPU execution boundaries in its own timestamp scale.
    pub(super) fn bind_step_capture(
        &self,
        py: Python<'_>,
        ordinal: u64,
        step: u64,
        capture_stream: u64,
        end: bool,
    ) -> PyResult<()> {
        self.require_original(py)?;
        let mut retained = self.interval()?;
        let interval = retained
            .as_mut()
            .ok_or_else(|| invalid("step capture lost its original observer interval"))?;
        if !interval.begun
            || interval.finish_attempted
            || interval.release_attempted
            || interval.cancellation_attempted
            || capture_stream == 0
        {
            return Err(invalid(
                "step capture requires its original open whole-process interval",
            ));
        }
        if end {
            let count = interval.steps.len() as u64;
            let current = interval
                .steps
                .last_mut()
                .ok_or_else(|| invalid("step capture end has no original beginning"))?;
            if current.ordinal != ordinal
                || current.capture_stream != capture_stream
                || !current.start_confirmed
                || current.end_attempted
                || step.checked_add(1) != Some(count)
            {
                return Err(invalid("step capture changed or repeated its original end"));
            }
            current.end_attempted = true;
        } else {
            if step != interval.steps.len() as u64
                || ordinal
                    != interval
                        .ordinal
                        .checked_add(step)
                        .ok_or_else(|| invalid("step interval ordinal overflowed"))?
                || interval
                    .steps
                    .last()
                    .is_some_and(|original| !original.capture_finished)
            {
                return Err(invalid(
                    "step capture changed its complete contiguous original roster",
                ));
            }
            // Retain before invoking the producer. Unknown binding cannot
            // repeat capture or issue another step-boundary owner.
            interval.steps.push(StepInterval {
                ordinal,
                capture_stream,
                start_confirmed: false,
                end_attempted: false,
                capture_finished: false,
                boundaries: None,
                peak: None,
            });
        }
        // SAFETY: original pinned ABI context and handle; these immutable
        // original ordinals and stream identify the kernel just enqueued inside
        // its actual capture. The collector resolves and retains the node here.
        let status = unsafe {
            (self.bind_step)(
                self.context as *mut c_void,
                interval.handle as *mut c_void,
                ordinal,
                step,
                capture_stream,
                u32::from(end),
            )
        };
        if status != COMPLETE {
            return Err(invalid("original graph-step boundary binding is unknown; retain the same interval and capture owners"));
        }
        let original = interval.steps.last_mut().expect("original capture binding");
        if end {
            original.capture_finished = true;
        } else {
            original.start_confirmed = true;
        }
        Ok(())
    }

    /// Only an issued native whole-roster disposition may withdraw execution
    /// subinterval obligations. Preserve every original attempt and its status;
    /// an unknown bind or handoff cannot be repaired by absence of GPU activity.
    #[cfg(feature = "semantic-policy")]
    pub(super) fn cancel_step_captures(
        &self,
        py: Python<'_>,
        ordinal: u64,
        steps: &[SemanticPreparedStep],
        capture_stream: u64,
        proof: &xlog_cuda::SemanticPreparedSegmentNonSubmission,
    ) -> PyResult<()> {
        self.require_original(py)?;
        let mut retained = self.interval()?;
        let interval = retained
            .as_mut()
            .ok_or_else(|| invalid("capture cancellation lost its original interval"))?;
        let count = u64::try_from(steps.len())
            .map_err(|_| invalid("original planned roster exceeds u64"))?;
        if !proof.matches(steps)
            || count == 0
            || capture_stream == 0
            || !interval.begun
            || interval.ordinal != ordinal
            || interval.finish_attempted
            || interval.release_attempted
            || interval.cancellation_attempted
            || interval.steps.len() > steps.len()
            || interval.steps.iter().any(|step| {
                !step.start_confirmed
                    || (step.end_attempted && !step.capture_finished)
                    || step.capture_stream != capture_stream
            })
        {
            return Err(invalid(
                "capture cancellation requires its native original roster and known binding results",
            ));
        }
        interval.cancellation_attempted = true;
        interval.cancellation_arguments = Some((ordinal, count, capture_stream));
        // SAFETY: site 2 is the installed whole-roster cancellation contract on
        // this same pinned table/handle. The native proof excludes submit after
        // known EndCapture and actual preparation joins. Here step_ordinal is
        // the original planned count, not a manufactured execution coordinate.
        let status = unsafe {
            (self.bind_step)(
                self.context as *mut c_void,
                interval.handle as *mut c_void,
                ordinal,
                count,
                capture_stream,
                CANCEL_GROUP,
            )
        };
        interval.cancellation_status = status;
        if status != COMPLETE {
            return Err(invalid(
                "original capture cancellation is unknown; retain its same interval and owners",
            ));
        }
        interval.cancelled_roster = true;
        Ok(())
    }

    /// Observe the same original cancellation, never repeat its handoff or
    /// replace the native non-submission proof with collector activity.
    #[cfg(feature = "semantic-policy")]
    pub(super) fn resolve_step_cancellation(
        &self,
        py: Python<'_>,
        ordinal: u64,
        steps: &[SemanticPreparedStep],
        capture_stream: u64,
        proof: &xlog_cuda::SemanticPreparedSegmentNonSubmission,
    ) -> PyResult<()> {
        self.require_original(py)?;
        let mut retained = self.interval()?;
        let interval = retained
            .as_mut()
            .ok_or_else(|| invalid("capture cancellation lost its original interval"))?;
        let count = Self::require_original_step_cancellation(
            interval, ordinal, steps, capture_stream, proof,
        )?;
        if interval.cancelled_roster && interval.cancellation_status == COMPLETE {
            return Ok(());
        }
        if !matches!(interval.cancellation_status, INCOMPLETE | UNKNOWN) {
            return Err(invalid("a known failed capture cancellation cannot be resolved as success"));
        }
        // SAFETY: the immutable capsule pins this exact resolver/context. The
        // retained tuple and native proof authenticate the only prior handoff.
        let status = unsafe {
            (self.resolve_step_cancellation)(
                self.context as *mut c_void,
                interval.handle as *mut c_void,
                ordinal,
                count,
                capture_stream,
            )
        };
        interval.cancellation_status = status;
        if status != COMPLETE {
            return Err(invalid("original capture cancellation is still unresolved; retain its same interval and owners"));
        }
        interval.cancelled_roster = true;
        Ok(())
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn step_cancellation_pending(
        &self,
        py: Python<'_>,
        ordinal: u64,
        steps: &[SemanticPreparedStep],
        capture_stream: u64,
        proof: &xlog_cuda::SemanticPreparedSegmentNonSubmission,
    ) -> PyResult<bool> {
        self.require_original(py)?;
        let retained = self.interval()?;
        let interval = retained.as_ref()
            .ok_or_else(|| invalid("capture cancellation lost its original interval"))?;
        Self::require_original_step_cancellation(interval, ordinal, steps, capture_stream, proof)?;
        match interval.cancellation_status {
            COMPLETE if interval.cancelled_roster => Ok(false),
            INCOMPLETE | UNKNOWN if !interval.cancelled_roster => Ok(true),
            _ => Err(invalid("capture cancellation has no genuine pending original result")),
        }
    }

    #[cfg(feature = "semantic-policy")]
    fn require_original_step_cancellation(
        interval: &Interval,
        ordinal: u64,
        steps: &[SemanticPreparedStep],
        capture_stream: u64,
        proof: &xlog_cuda::SemanticPreparedSegmentNonSubmission,
    ) -> PyResult<u64> {
        let count = u64::try_from(steps.len())
            .map_err(|_| invalid("original planned roster exceeds u64"))?;
        if !proof.matches(steps)
            || !interval.begun
            || count == 0
            || capture_stream == 0
            || interval.ordinal != ordinal
            || !interval.cancellation_attempted
            || interval.cancellation_arguments != Some((ordinal, count, capture_stream))
            || interval.finish_attempted
            || interval.release_attempted
            || interval.steps.len() > steps.len()
            || interval.steps.iter().any(|step| {
                !step.start_confirmed
                    || (step.end_attempted && !step.capture_finished)
                    || step.capture_stream != capture_stream
            })
        {
            return Err(invalid("cancellation resolution requires the exact original handoff, native proof and known capture prefix"));
        }
        Ok(count)
    }

    /// Called once, after actual source native/model joins, never by readback.
    pub(super) fn finish(&self, py: Python<'_>) -> PyResult<()> {
        self.require_original(py)?;
        let mut retained = self.interval()?;
        let interval = retained
            .as_mut()
            .ok_or_else(|| invalid("source join lost its original resource interval"))?;
        if !interval.begun
            || interval.finish_attempted
            || interval.release_attempted
            || (interval.cancellation_attempted && !interval.cancelled_roster)
        {
            return Err(invalid(
                "resource interval finish is single-attempt after its original join",
            ));
        }
        interval.finish_attempted = true;
        // SAFETY: begin issued this handle in the same retained context. The
        // source creator has joined actual consumers before entering this call.
        let status =
            unsafe { (self.finish)(self.context as *mut c_void, interval.handle as *mut c_void) };
        if status != COMPLETE && status != INCOMPLETE {
            return Err(invalid("original resource interval finish is unknown; read only the same interval without repeating finish"));
        }
        Ok(())
    }

    /// No flush, join, replay, finish or numerical substitution occurs here.
    pub(super) fn backing_peak(&self, py: Python<'_>) -> PyResult<u64> {
        self.require_original(py)?;
        let mut retained = self.interval()?;
        let interval = retained
            .as_mut()
            .ok_or_else(|| invalid("backing result lost its original resource interval"))?;
        if let Some(peak) = interval.backing_peak {
            return Ok(peak);
        }
        if !interval.begun || !interval.finish_attempted || interval.release_attempted {
            return Err(invalid(
                "backing result requires the original finished, retained interval",
            ));
        }
        // SAFETY: this C record contains only integers, byte arrays and a raw
        // nullable event pointer. All-zero storage is valid for every field;
        // availability still requires the producer's original explicit bits.
        let mut view: TraceView = unsafe { std::mem::zeroed() };
        view.struct_size = size_of::<TraceView>() as u32;
        view.state = UNKNOWN;
        // SAFETY: the immutable producer table, context and original handle
        // remain pinned. The installed producer writes precisely this C layout;
        // event pointers are never dereferenced or transferred to Python.
        let status = unsafe {
            (self.read)(
                self.context as *mut c_void,
                interval.handle as *mut c_void,
                &mut view,
            )
        };
        if status != COMPLETE {
            return Err(invalid("original resource result is unknown; retain the original interval and completed source result"));
        }
        if view.struct_size as usize != size_of::<TraceView>()
            || view.original_attempt_nonce != interval.nonce
            || view.operation_ordinal != interval.ordinal
            || view.boundary_flags & (START_VALID | END_VALID) != (START_VALID | END_VALID)
            || view.start == 0
            || view.end < view.start
            || interval
                .boundaries
                .is_some_and(|original| original != (view.start, view.end))
        {
            return Err(invalid(
                "resource view changed its original layout, nonce, ordinal or interval boundaries",
            ));
        }
        interval.boundaries = Some((view.start, view.end));
        let required = START_VALID
            | END_VALID
            | DELIVERY_DRAINED
            | STARTED_BEFORE_CUDA
            | BACKING_PEAK_VALID
            | ALLOCATOR_COMPLETED
            | BASELINE_COMPLETE;
        if !matches!(view.state, COMPLETE | INCOMPLETE)
            || view.boundary_flags & required != required
            || view.requested_coverage & OWNED_BACKING == 0
            || view.enabled_coverage & OWNED_BACKING == 0
            || view.unsupported_coverage & OWNED_BACKING != 0
            || view.process_dropped_records != 0
            || view.process_errors != 0
            || view.process_buffer_overflows != 0
            || view.certificate_identity != self.certificate.observer_identity
            || view.backing_peak_bytes < view.backing_baseline_bytes
            || view.backing_peak_bytes > self.certificate.total_physical_bytes
        {
            // In particular, COMPLETE and DELIVERY_DRAINED are trace delivery,
            // not allocator completion or proof of the backing producer.
            // Do not read the stored scalar, even when its bits happen to be 0.
            return Err(invalid("original owned GPU backing peak or allocator completion is unknown; retain the same interval and owners without numerical resource admission"));
        }
        interval.backing_peak = Some(view.backing_peak_bytes);
        Ok(view.backing_peak_bytes)
    }

    pub(super) fn step_backing_peaks(&self, py: Python<'_>, count: usize) -> PyResult<Vec<u64>> {
        let group_peak = self.backing_peak(py)?;
        let mut retained = self.interval()?;
        let interval = retained.as_mut().expect("known original backing interval");
        if count == 0
            || interval.cancelled_roster
            || interval.steps.len() != count
            || interval
                .steps
                .iter()
                .any(|step| !step.start_confirmed || !step.capture_finished)
        {
            return Err(invalid(
                "backing step results require the complete original captured roster",
            ));
        }
        let (group_start, group_end) = interval.boundaries.expect("known group boundaries");
        let mut previous_end = group_start;
        let mut peaks = Vec::with_capacity(count);
        for step in &mut interval.steps {
            if step.peak.is_none() {
                // SAFETY: the exact installed integer/pointer C record is
                // initialized; validity and original custody are checked below.
                let mut view: TraceView = unsafe { std::mem::zeroed() };
                view.struct_size = size_of::<TraceView>() as u32;
                view.state = UNKNOWN;
                // SAFETY: read only this sealed original subinterval. No
                // capture, flush, finish, launch or registration is repeated.
                let status = unsafe {
                    (self.read_step)(
                        self.context as *mut c_void,
                        interval.handle as *mut c_void,
                        step.ordinal,
                        &mut view,
                    )
                };
                let required = START_VALID
                    | END_VALID
                    | DELIVERY_DRAINED
                    | STARTED_BEFORE_CUDA
                    | BACKING_PEAK_VALID
                    | ALLOCATOR_COMPLETED
                    | BASELINE_COMPLETE;
                if status != COMPLETE
                    || view.struct_size as usize != size_of::<TraceView>()
                    || view.original_attempt_nonce != interval.nonce
                    || view.operation_ordinal != step.ordinal
                    || view.boundary_flags & (START_VALID | END_VALID) != (START_VALID | END_VALID)
                    || view.start != previous_end
                    || view.end < view.start
                    || view.end > group_end
                    || step
                        .boundaries
                        .is_some_and(|original| original != (view.start, view.end))
                {
                    return Err(invalid("original per-step execution boundaries are unknown; retain the same interval and captured roster"));
                }
                step.boundaries = Some((view.start, view.end));
                if !matches!(view.state, COMPLETE | INCOMPLETE)
                    || view.boundary_flags & required != required
                    || view.requested_coverage & OWNED_BACKING == 0
                    || view.enabled_coverage & OWNED_BACKING == 0
                    || view.unsupported_coverage & OWNED_BACKING != 0
                    || view.process_dropped_records != 0
                    || view.process_errors != 0
                    || view.process_buffer_overflows != 0
                    || view.certificate_identity != self.certificate.observer_identity
                    || view.backing_peak_bytes < view.backing_baseline_bytes
                    || view.backing_peak_bytes > self.certificate.total_physical_bytes
                {
                    return Err(invalid("original per-step backing memory or execution boundaries are unknown; group peak cannot replace them"));
                }
                step.peak = Some(view.backing_peak_bytes);
            }
            let (start, end) = step.boundaries.expect("known original step boundaries");
            if start != previous_end {
                return Err(invalid(
                    "original backing step intervals have a gap or overlap",
                ));
            }
            previous_end = end;
            peaks.push(step.peak.expect("known original step peak"));
        }
        if previous_end != group_end || peaks.iter().copied().max() != Some(group_peak) {
            return Err(invalid(
                "original backing step intervals do not cover the complete whole-process interval",
            ));
        }
        Ok(peaks)
    }

    /// Final owner release is allowed only after this original backing result
    /// was consumed. A failed release is irreversible and retains the capsule.
    pub(super) fn release(&self, py: Python<'_>) -> PyResult<()> {
        self.require_original(py)?;
        let mut retained = self.interval()?;
        let interval = retained
            .as_mut()
            .ok_or_else(|| invalid("resource release lost its original interval"))?;
        if interval.released {
            return Ok(());
        }
        if interval.backing_peak.is_none()
            || (interval.cancellation_attempted && !interval.cancelled_roster)
            || (!interval.cancelled_roster && interval.steps.iter().any(|step| step.peak.is_none()))
            || interval.release_attempted
        {
            return Err(invalid(
                "unknown backing completion or release retains its original interval and owners",
            ));
        }
        interval.release_attempted = true;
        // SAFETY: the backing result came from this same finished handle with
        // drained delivery, exact boundaries and original context/capsule.
        let status =
            unsafe { (self.release)(self.context as *mut c_void, interval.handle as *mut c_void) };
        if status != COMPLETE {
            return Err(invalid("original resource interval release is unknown; retain its original capsule and pending"));
        }
        interval.released = true;
        Ok(())
    }
}
