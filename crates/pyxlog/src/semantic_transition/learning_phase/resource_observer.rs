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

type Begin = unsafe extern "C" fn(*mut c_void, *const u8, u64, *mut *mut c_void) -> u32;
type Finish = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
type Read = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut TraceView) -> u32;
type Release = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;

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

const _: () = assert!(size_of::<Api>() == 48 && size_of::<TraceView>() == 200);

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
    interval: Mutex<Option<Interval>>,
}

impl ResourceObserver {
    pub(super) fn capture(capsule: &Bound<'_, PyCapsule>) -> PyResult<Self> {
        let pointer = capsule.pointer_checked(Some(c"xlog.resource_observer.v1"))?;
        if !(pointer.as_ptr() as usize).is_multiple_of(align_of::<Api>()) {
            return Err(invalid(
                "resource observer table has invalid C ABI alignment",
            ));
        }
        // SAFETY: the original trusted native capsule owns an immutable C table
        // for its complete lifetime. Read only its common header before checking
        // the exact supported layout; no old-size table is read as the new ABI.
        let header = unsafe { &*pointer.as_ptr().cast::<ApiHeader>() };
        if header.abi_version != 1 || header.struct_size as usize != size_of::<Api>() {
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
                "resource observer requires its original context and all four interval functions",
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
            interval: Mutex::new(None),
        })
    }

    pub(super) fn require_original(&self, py: Python<'_>) -> PyResult<()> {
        if thread::current().id() != self.creator
            || self
                .capsule
                .bind(py)
                .pointer_checked(Some(c"xlog.resource_observer.v1"))?
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
        if retained.is_some() {
            return Err(invalid(
                "resource interval begin is single-attempt; retain the original interval",
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

    /// Called once, after actual source native/model joins, never by readback.
    pub(super) fn finish(&self, py: Python<'_>) -> PyResult<()> {
        self.require_original(py)?;
        let mut retained = self.interval()?;
        let interval = retained
            .as_mut()
            .ok_or_else(|| invalid("source join lost its original resource interval"))?;
        if !interval.begun || interval.finish_attempted || interval.release_attempted {
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
        if interval.physical_peak.is_none() || interval.release_attempted {
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
