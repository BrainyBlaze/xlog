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
const PHYSICAL_PEAK_VALID: u64 = 32;
const ALLOCATOR_COMPLETED: u64 = 64;
const PHYSICAL_MEMORY: u64 = 128;
#[cfg(feature = "semantic-policy")]
const CANCEL_GROUP: u32 = 2;

type Begin = unsafe extern "C" fn(*mut c_void, *const u8, u64, *mut *mut c_void) -> u32;
type Finish = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
type Read = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut TraceView) -> u32;
type Release = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
type BindStep = unsafe extern "C" fn(*mut c_void, *mut c_void, u64, u64, u64, u32) -> u32;
type ReadStep = unsafe extern "C" fn(*mut c_void, *mut c_void, u64, *mut TraceView) -> u32;

// These are the exact layouts of the installed xlog_resource_observer.h.
// Event storage remains producer-owned; this memory consumer never dereferences it.
#[repr(C)]
struct ApiHeader {
    abi_version: u32,
    struct_size: u32,
}

#[repr(C)]
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
    physical_peak_bytes: u64,
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

const _: () = assert!(size_of::<Api>() == 64 && size_of::<TraceView>() == 200);

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
    physical_peak: Option<u64>,
    release_attempted: bool,
    released: bool,
    steps: Vec<StepInterval>,
    cancellation_attempted: bool,
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
    interval: Mutex<Option<Interval>>,
}

impl ResourceObserver {
    pub(super) fn capture(capsule: &Bound<'_, PyCapsule>) -> PyResult<Self> {
        let pointer = capsule.pointer_checked(Some(c"xlog.resource_observer.v2"))?;
        if !(pointer.as_ptr() as usize).is_multiple_of(align_of::<Api>()) {
            return Err(invalid(
                "resource observer table has invalid C ABI alignment",
            ));
        }
        // SAFETY: the original trusted native capsule owns an immutable C table
        // for its complete lifetime. Read only its common header before checking
        // the exact supported layout; no old-size table is read as the new ABI.
        let header = unsafe { &*pointer.as_ptr().cast::<ApiHeader>() };
        if header.abi_version != 2 || header.struct_size as usize != size_of::<Api>() {
            return Err(invalid(
                "resource observer requires the exact installed C ABI table",
            ));
        }
        // SAFETY: the checked header and the capsule's native producer contract
        // establish the complete immutable table. Function pointers are captured
        // before any issuer, store, authority or model callback can run.
        let api = unsafe { &*pointer.as_ptr().cast::<Api>() };
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
            table: pointer.as_ptr() as usize,
            context: api.context as usize,
            creator: thread::current().id(),
            begin: api.begin.ok_or_else(missing)?,
            finish: api.finish.ok_or_else(missing)?,
            read: api.read.ok_or_else(missing)?,
            release: api.release.ok_or_else(missing)?,
            bind_step: api.bind_step.ok_or_else(missing)?,
            read_step: api.read_step.ok_or_else(missing)?,
            interval: Mutex::new(None),
        })
    }

    pub(super) fn require_original(&self, py: Python<'_>) -> PyResult<()> {
        if thread::current().id() != self.creator
            || self
                .capsule
                .bind(py)
                .pointer_checked(Some(c"xlog.resource_observer.v2"))?
                .as_ptr() as usize
                != self.table
        {
            return Err(invalid(
                "resource observer lost its original creator or capsule table",
            ));
        }
        Ok(())
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
            physical_peak: None,
            release_attempted: false,
            released: false,
            steps: Vec::new(),
            cancellation_attempted: false,
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
        // SAFETY: original pinned ABI2 context and handle; these immutable
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
        if status != COMPLETE {
            return Err(invalid(
                "original capture cancellation is unknown; retain its same interval and owners",
            ));
        }
        interval.cancelled_roster = true;
        Ok(())
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
    pub(super) fn physical_peak(&self, py: Python<'_>) -> PyResult<u64> {
        self.require_original(py)?;
        let mut retained = self.interval()?;
        let interval = retained
            .as_mut()
            .ok_or_else(|| invalid("physical result lost its original resource interval"))?;
        if let Some(peak) = interval.physical_peak {
            return Ok(peak);
        }
        if !interval.begun || !interval.finish_attempted || interval.release_attempted {
            return Err(invalid(
                "physical result requires the original finished, retained interval",
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
            | PHYSICAL_PEAK_VALID
            | ALLOCATOR_COMPLETED;
        if !matches!(view.state, COMPLETE | INCOMPLETE)
            || view.boundary_flags & required != required
            || view.requested_coverage & PHYSICAL_MEMORY == 0
            || view.enabled_coverage & PHYSICAL_MEMORY == 0
            || view.unsupported_coverage & PHYSICAL_MEMORY != 0
            || view.process_dropped_records != 0
            || view.process_errors != 0
            || view.process_buffer_overflows != 0
        {
            // In particular, COMPLETE and DELIVERY_DRAINED are trace delivery,
            // not allocator completion or proof of the physical producer.
            // Do not read the stored scalar, even when its bits happen to be 0.
            return Err(invalid("original whole-process physical peak or allocator completion is unknown; retain the same interval and owners without numerical resource admission"));
        }
        interval.physical_peak = Some(view.physical_peak_bytes);
        Ok(view.physical_peak_bytes)
    }

    pub(super) fn step_physical_peaks(&self, py: Python<'_>, count: usize) -> PyResult<Vec<u64>> {
        let group_peak = self.physical_peak(py)?;
        let mut retained = self.interval()?;
        let interval = retained.as_mut().expect("known original physical interval");
        if count == 0
            || interval.cancelled_roster
            || interval.steps.len() != count
            || interval
                .steps
                .iter()
                .any(|step| !step.start_confirmed || !step.capture_finished)
        {
            return Err(invalid(
                "physical step results require the complete original captured roster",
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
                    | PHYSICAL_PEAK_VALID
                    | ALLOCATOR_COMPLETED;
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
                    || view.requested_coverage & PHYSICAL_MEMORY == 0
                    || view.enabled_coverage & PHYSICAL_MEMORY == 0
                    || view.unsupported_coverage & PHYSICAL_MEMORY != 0
                    || view.process_dropped_records != 0
                    || view.process_errors != 0
                    || view.process_buffer_overflows != 0
                {
                    return Err(invalid("original per-step physical memory or execution boundaries are unknown; group peak cannot replace them"));
                }
                step.peak = Some(view.physical_peak_bytes);
            }
            let (start, end) = step.boundaries.expect("known original step boundaries");
            if start != previous_end {
                return Err(invalid(
                    "original physical step intervals have a gap or overlap",
                ));
            }
            previous_end = end;
            peaks.push(step.peak.expect("known original step peak"));
        }
        if previous_end != group_end || peaks.iter().copied().max() != Some(group_peak) {
            return Err(invalid(
                "original physical step intervals do not cover the complete whole-process interval",
            ));
        }
        Ok(peaks)
    }

    /// Final owner release is allowed only after this original physical result
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
        if interval.physical_peak.is_none()
            || (interval.cancellation_attempted && !interval.cancelled_roster)
            || (!interval.cancelled_roster && interval.steps.iter().any(|step| step.peak.is_none()))
            || interval.release_attempted
        {
            return Err(invalid(
                "unknown physical completion or release retains its original interval and owners",
            ));
        }
        interval.release_attempted = true;
        // SAFETY: the physical result came from this same finished handle with
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
