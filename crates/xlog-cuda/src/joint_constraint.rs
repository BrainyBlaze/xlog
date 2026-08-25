//! Joint constraint carrier: buffer ownership, registration, and the
//! device-resident label-feasibility solve stage.
//!
//! The carrier owns every solver buffer: score, domain, constraint and
//! output memory is allocated by the xlog device runtime and exported
//! outward, never imported from an external DLPack producer. Strict
//! launch recorders therefore record every carrier column (a runtime
//! block is always present), and schema registration is once-per
//! session with a typed refusal on duplicates.
//!
//! The solve stage runs entirely on device: catalog-bound signature
//! masks upload once cold-path after registration, and the existential
//! label-feasibility kernel launches through a strict recorder with
//! fuel charged before the launch — beyond fuel the solve refuses
//! typed without touching the device.

use std::sync::Arc;

use xlog_core::MemoryBudget;

use crate::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use crate::joint_solver::{FuelMeter, SolverError};
use crate::launch::LaunchRecorder;
use crate::memory::{CudaColumn, GpuMemoryManager};
use crate::provider::JOINT_SOLVE_MODULE;
use crate::{CudaDevice, LaunchAsync, LaunchConfig};

/// Kernel entry point for the existential label-feasibility stage.
const FEASIBILITY_KERNEL: &str = "joint_label_feasibility";
/// Kernel entry point for the per-candidate exact top-two stage.
const TOP2_KERNEL: &str = "joint_label_top2";
/// Kernel entry points for device-resident candidate-component discovery.
const COMPONENT_PLAN_INIT_KERNEL: &str = "joint_component_plan_init";
const COMPONENT_ENTITY_OWNERS_KERNEL: &str = "joint_component_entity_owners";
const COMPONENT_UNION_KERNEL: &str = "joint_component_union";
const COMPONENT_COMPRESS_KERNEL: &str = "joint_component_compress";
/// Kernel entry point for the exact component-enumeration stage.
const COMPONENT_KERNEL: &str = "joint_component_enumerate";
/// Exact specialized and general component-search entry points.
const CHAIN_DP_KERNEL: &str = "joint_component_chain_dp";
const BRANCH_AND_BOUND_KERNEL: &str = "joint_component_branch_and_bound";

const JOINT_SOLVE_KERNELS: &[&str] = &[
    FEASIBILITY_KERNEL,
    TOP2_KERNEL,
    COMPONENT_PLAN_INIT_KERNEL,
    COMPONENT_ENTITY_OWNERS_KERNEL,
    COMPONENT_UNION_KERNEL,
    COMPONENT_COMPRESS_KERNEL,
    COMPONENT_KERNEL,
    CHAIN_DP_KERNEL,
    BRANCH_AND_BOUND_KERNEL,
];

/// Fixed carrier budget: device-owned working buffers are capacity-bounded
/// and small; the production capacity envelope is validated against the
/// solver's consensus thresholds.
const CARRIER_BUDGET_BYTES: u64 = 64 * 1024 * 1024;

/// Typed carrier errors. Refusals are concrete variants — callers
/// match on the variant, never on message text.
#[derive(Debug)]
pub enum CarrierError {
    /// A schema is already registered for this carrier session;
    /// registration is once-per-session and never silently rebinds
    /// live buffers.
    SchemaAlreadyRegistered {
        /// The catalog anchor the session is already bound to.
        catalog_sha: String,
        /// The solver identity the session is already bound to.
        solver_identity: String,
    },
    /// Device allocation through the runtime failed.
    Allocation(xlog_core::XlogError),
    /// A capacity dimension is zero. A carrier with no entities,
    /// lanes, candidates, or labels cannot participate in a solve;
    /// silently clamping the dimension would hide the caller's bug.
    ZeroCapacity {
        /// Name of the zero dimension.
        dimension: &'static str,
    },
    /// Signature binding or solving was attempted before schema
    /// registration; masks are catalog-bound, so the catalog anchor
    /// must be fixed first.
    SchemaNotRegistered,
    /// Signature masks are already bound for this session; rebinding
    /// live masks under a registered schema is never silent.
    SignaturesAlreadyBound,
    /// A signature mask slice does not match the carrier capacity.
    SignatureShapeMismatch {
        /// Which mask side mismatched.
        side: &'static str,
        /// Expected u64 word count (labels x lanes).
        expected_words: usize,
        /// Provided u64 word count.
        got_words: usize,
    },
    /// The solve was attempted before signature masks were bound.
    SignaturesUnbound,
    /// The top-two stage was attempted before the feasibility stage
    /// populated the feasible sets it consumes.
    FeasibilityNotSolved,
    /// The abstain label index is outside the label universe.
    AbstainOutOfRange {
        /// The offending index.
        abstain_label: u32,
        /// The label universe width.
        labels: usize,
    },
    /// The joint-solve kernel module could not be loaded or its
    /// entry point resolved on this device.
    KernelUnavailable {
        /// Load-failure detail.
        detail: String,
    },
    /// The recorded launch failed preflight, launch, or commit.
    Launch(xlog_core::XlogError),
    /// A typed solver refusal (fuel exhaustion) surfaced through the
    /// carrier solve entry.
    Solver(SolverError),
}

impl std::fmt::Display for CarrierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CarrierError::SchemaAlreadyRegistered {
                catalog_sha,
                solver_identity,
            } => write!(
                f,
                "carrier schema already registered (catalog {catalog_sha}, \
                 solver {solver_identity}); registration is once-per-session"
            ),
            CarrierError::Allocation(err) => write!(f, "carrier allocation failed: {err}"),
            CarrierError::ZeroCapacity { dimension } => write!(
                f,
                "carrier capacity dimension {dimension} is zero; refusing \
                 instead of silently clamping"
            ),
            CarrierError::SchemaNotRegistered => write!(
                f,
                "carrier schema is not registered; signature masks are \
                 catalog-bound and require the catalog anchor first"
            ),
            CarrierError::SignaturesAlreadyBound => {
                write!(f, "signature masks already bound for this session")
            }
            CarrierError::SignatureShapeMismatch {
                side,
                expected_words,
                got_words,
            } => write!(
                f,
                "{side} signature mask has {got_words} u64 words, expected \
                 {expected_words} (labels x lanes)"
            ),
            CarrierError::SignaturesUnbound => write!(
                f,
                "solve refused: signature masks are not bound for this session"
            ),
            CarrierError::FeasibilityNotSolved => write!(
                f,
                "top-two stage refused: the feasibility stage has not \
                 populated the feasible sets this session"
            ),
            CarrierError::AbstainOutOfRange {
                abstain_label,
                labels,
            } => write!(
                f,
                "abstain label {abstain_label} is outside the label universe \
                 of width {labels}"
            ),
            CarrierError::KernelUnavailable { detail } => {
                write!(f, "joint-solve kernel unavailable: {detail}")
            }
            CarrierError::Launch(err) => write!(f, "carrier solve launch failed: {err}"),
            CarrierError::Solver(err) => write!(f, "carrier solve refused: {err}"),
        }
    }
}

impl std::error::Error for CarrierError {}

/// No-op logging sink for the carrier's private resource stack.
struct SilentSink;

impl LoggingSink for SilentSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

/// The carrier buffers addressable through the outward export
/// surface, in the carrier's stable column order plus the
/// device-resident logical-counts buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarrierBufferId {
    /// Entity sort-domain bitsets, `entities x domain_lanes` u64.
    Domains,
    /// Relation candidate scores, `candidates x labels` f32.
    Scores,
    /// Padded candidate entity arguments, `candidates x max_arity` u32.
    Constraints,
    /// Declared active argument count for each candidate, `candidates` u32.
    ArgumentArities,
    /// Per-candidate feasible label counts, `candidates` u32.
    Outputs,
    /// Per-candidate feasible label bitmasks,
    /// `candidates x ceil(labels/64)` u64.
    FeasibleSets,
    /// Device-resident logical batch state, 4 u32:
    /// `[logical_entities, logical_candidates, logical_edges,
    /// overflow_flag]`. Producers write it on device; a nonzero
    /// overflow flag marks a producer that ran past capacity.
    LogicalCounts,
    /// Per-candidate exact top-two results, `candidates x 4` u32:
    /// `[best_label, ambiguous_flag, best_score_bits, margin_bits]`
    /// (f32 stored as raw bits). Authoritative as a global
    /// max-marginal ONLY for single-candidate components; a set
    /// ambiguity flag must never emit as a unique MAP label.
    MapResults,
    /// Per-candidate solve authority, `candidates` u32: 2 = exact by
    /// complete enumeration, 4 = exact by device-discovered chain DP,
    /// 5 = exact by general branch-and-bound, 3 = refused on fuel,
    /// 0xFFFFFFFF = poisoned. Rows no component stage touched keep
    /// their prior top-two authority; status 6 is internal escalation
    /// and must be consumed before this method returns.
    SolveStatus,
}

/// One buffer exported outward while xlog retains ownership. The
/// binding layer wraps `slice` in a real DLPack capsule via
/// [`CudaColumn::dlpack_xlog_owned`]; the shared `Arc` keeps the
/// runtime identity alive, so strict launch recorders keep recording
/// the exported view instead of rejecting it.
pub struct CarrierExport {
    /// The runtime-backed allocation, shared with the carrier.
    pub slice: Arc<crate::memory::TrackedCudaSlice<u8>>,
    /// The stream the export synchronizes against.
    pub stream: Arc<crate::CudaStream>,
    /// Element width in bytes (8 for u64 buffers, 4 for u32/f32).
    pub elem_bytes: usize,
    /// Logical row count of the 2-D view.
    pub rows: usize,
    /// Logical column count of the 2-D view.
    pub cols: usize,
}

/// Device-resident buffer set for the joint placement/relation
/// constraint solve. All memory is runtime-backed and xlog-owned;
/// every buffer is shared between the carrier's working columns and
/// the outward export surface, so both sides observe one allocation
/// identity.
pub struct JointConstraintCarrier {
    buffers: [Arc<crate::memory::TrackedCudaSlice<u8>>; 9],
    columns: [CudaColumn; 8],
    component_buffers: [Arc<crate::memory::TrackedCudaSlice<u8>>; 4],
    component_columns: [CudaColumn; 4],
    search_columns: [CudaColumn; 4],
    signatures: Option<CudaColumn>,
    registered_schema: Option<(String, String)>,
    feasibility_solved: bool,
    /// Producer-completion events recorded on EXTERNAL streams via
    /// [`Self::note_producer_stream`], consumed (waited then
    /// destroyed) by the next solve stage. Raw driver handles; the
    /// carrier destroys any leftovers on drop.
    pending_producer_events: Vec<cudarc::driver::sys::CUevent>,
    /// External consumer streams waiting for completion of the next
    /// successful solve stage. The carrier does not own these raw
    /// handles; registrations are cleared after handoff, solve
    /// failure, or drop.
    pending_consumer_streams: Vec<cudarc::driver::sys::CUstream>,
    entities: usize,
    domain_lanes: usize,
    candidates: usize,
    labels: usize,
    max_arity: usize,
    device: Arc<CudaDevice>,
    pool: Arc<StreamPool>,
    memory: Arc<GpuMemoryManager>,
    runtime: Arc<XlogDeviceRuntime>,
}

/// u64 words needed for one per-candidate feasible-label bitmask row.
fn label_words(labels: usize) -> usize {
    labels.div_ceil(64)
}

/// Working column over a shared runtime-backed allocation. The null
/// managed tensor is drop-safe (its deleter is null-checked) and
/// carries no capsule — real DLPack capsules are built by the
/// binding layer around [`JointConstraintCarrier::export_buffer`].
/// Ownership predicates hold: the column reports non-external and
/// resolves its runtime block through the shared slice.
fn shared_column(
    slice: &Arc<crate::memory::TrackedCudaSlice<u8>>,
    stream: &Arc<crate::CudaStream>,
) -> CudaColumn {
    let tensor = unsafe { crate::DlpackManagedTensor::from_raw(std::ptr::null_mut()) };
    CudaColumn::dlpack_xlog_owned(Arc::clone(slice), Arc::clone(stream), tensor)
}

/// Load the joint-solve kernel module onto `device` if it is not
/// already resident. Fail closed: a carrier never constructs without
/// its solve kernel resolvable.
fn ensure_joint_solve_module(device: &Arc<CudaDevice>) -> Result<(), CarrierError> {
    if JOINT_SOLVE_KERNELS
        .iter()
        .all(|k| device.inner().get_func(JOINT_SOLVE_MODULE, k).is_some())
    {
        return Ok(());
    }
    let cc = crate::provider::detect_compute_capability(device).map_err(|e| {
        CarrierError::KernelUnavailable {
            detail: e.to_string(),
        }
    })?;
    let sources = crate::provider::load_module_sources("joint_solve", cc).map_err(|e| {
        CarrierError::KernelUnavailable {
            detail: e.to_string(),
        }
    })?;
    let mut load_errors = Vec::new();
    for source in sources {
        let attempt = match source {
            crate::provider::KernelModuleSource::File { path, .. } => device
                .inner()
                .load_file(&path, JOINT_SOLVE_MODULE, JOINT_SOLVE_KERNELS)
                .map_err(|e| format!("{}: {e}", path.display())),
            crate::provider::KernelModuleSource::EmbeddedPortablePtx { ptx } => device
                .inner()
                .load_ptx(
                    cudarc::nvrtc::Ptx::from_src(ptx),
                    JOINT_SOLVE_MODULE,
                    JOINT_SOLVE_KERNELS,
                )
                .map_err(|e| format!("embedded portable PTX: {e}")),
        };
        match attempt {
            Ok(()) => return Ok(()),
            Err(detail) => load_errors.push(detail),
        }
    }
    Err(CarrierError::KernelUnavailable {
        detail: if load_errors.is_empty() {
            "no kernel artifact source available".to_string()
        } else {
            load_errors.join("; ")
        },
    })
}

impl Drop for JointConstraintCarrier {
    fn drop(&mut self) {
        // Destroy producer events never consumed by a solve stage;
        // the driver defers destruction past any in-flight work.
        for event in self.pending_producer_events.drain(..) {
            // SAFETY: created by note_producer_stream, consumed
            // nowhere else once we are in drop.
            unsafe {
                let _ = cudarc::driver::result::event::destroy(event);
            }
        }
        self.pending_consumer_streams.clear();
    }
}

impl JointConstraintCarrier {
    /// Allocate the capacity-bounded carrier buffers through the xlog
    /// device runtime: entity sort-domain bitsets, relation candidate
    /// scores, constraint slots, and solver outputs.
    pub fn allocate(
        device: Arc<CudaDevice>,
        entities: usize,
        domain_lanes: usize,
        candidates: usize,
        labels: usize,
    ) -> Result<Self, CarrierError> {
        let carrier =
            Self::allocate_with_max_arity(device, entities, domain_lanes, candidates, labels, 2)?;
        let binary_arities = vec![2u32; candidates];
        unsafe {
            cudarc::driver::result::memcpy_htod_sync(
                *carrier.buffers[3].device_ptr(),
                &binary_arities,
            )
            .map_err(|error| {
                CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                    "legacy binary arity initialization failed: {error}"
                )))
            })?;
        }
        Ok(carrier)
    }

    /// Allocate an arbitrary-arity carrier. Candidate rows are padded to
    /// `max_arity`; the active width of every row lives in its owned arity
    /// buffer and is consumed by the solver kernels.
    pub fn allocate_with_max_arity(
        device: Arc<CudaDevice>,
        entities: usize,
        domain_lanes: usize,
        candidates: usize,
        labels: usize,
        max_arity: usize,
    ) -> Result<Self, CarrierError> {
        for (dimension, value) in [
            ("entities", entities),
            ("domain_lanes", domain_lanes),
            ("candidates", candidates),
            ("labels", labels),
            ("max_arity", max_arity),
        ] {
            if value == 0 {
                return Err(CarrierError::ZeroCapacity { dimension });
            }
        }

        let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
            AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
        );
        let budget: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
            GlobalDeviceBudget::new(async_resource, CARRIER_BUDGET_BYTES as usize),
        );
        let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
            budget,
            Arc::new(SilentSink) as Arc<dyn LoggingSink>,
        ));
        let runtime = Arc::new(XlogDeviceRuntime::with_resource(
            Arc::clone(&device),
            0,
            Arc::clone(&pool),
            logging,
        ));
        let memory = Arc::new(GpuMemoryManager::with_runtime(
            Arc::clone(&device),
            MemoryBudget::with_limit(CARRIER_BUDGET_BYTES),
            Arc::clone(&runtime),
        ));

        ensure_joint_solve_module(&device)?;

        let domains = memory
            .alloc::<u64>(entities * domain_lanes)
            .map_err(CarrierError::Allocation)?;
        let scores = memory
            .alloc::<f32>(candidates * labels)
            .map_err(CarrierError::Allocation)?;
        let constraints = memory
            .alloc::<u32>(candidates * max_arity)
            .map_err(CarrierError::Allocation)?;
        let argument_arities = memory
            .alloc::<u32>(candidates)
            .map_err(CarrierError::Allocation)?;
        let outputs = memory
            .alloc::<u32>(candidates)
            .map_err(CarrierError::Allocation)?;
        let feasible_sets = memory
            .alloc::<u64>(candidates * label_words(labels))
            .map_err(CarrierError::Allocation)?;
        let logical_counts = memory.alloc::<u32>(4).map_err(CarrierError::Allocation)?;
        let map_results = memory
            .alloc::<u32>(candidates * 4)
            .map_err(CarrierError::Allocation)?;
        let solve_status = memory
            .alloc::<u32>(candidates)
            .map_err(CarrierError::Allocation)?;
        let component_parents = memory
            .alloc::<u32>(candidates)
            .map_err(CarrierError::Allocation)?;
        let component_entity_owners = memory
            .alloc::<u32>(entities)
            .map_err(CarrierError::Allocation)?;
        let component_count = memory.alloc::<u32>(1).map_err(CarrierError::Allocation)?;
        let component_fuel = memory.alloc::<u64>(1).map_err(CarrierError::Allocation)?;
        let search_assignment = memory
            .alloc::<u32>(candidates)
            .map_err(CarrierError::Allocation)?;
        let search_best_label = memory
            .alloc::<u32>(candidates)
            .map_err(CarrierError::Allocation)?;
        let search_best_total = memory
            .alloc::<f32>(candidates)
            .map_err(CarrierError::Allocation)?;
        let search_alt_total = memory
            .alloc::<f32>(candidates)
            .map_err(CarrierError::Allocation)?;

        // Every buffer is held as a shared Arc so the outward export
        // surface and the carrier's working columns observe one
        // allocation identity.
        let buffers: [Arc<crate::memory::TrackedCudaSlice<u8>>; 9] = [
            Arc::new(domains.into_bytes()),
            Arc::new(scores.into_bytes()),
            Arc::new(constraints.into_bytes()),
            Arc::new(argument_arities.into_bytes()),
            Arc::new(outputs.into_bytes()),
            Arc::new(feasible_sets.into_bytes()),
            Arc::new(logical_counts.into_bytes()),
            Arc::new(map_results.into_bytes()),
            Arc::new(solve_status.into_bytes()),
        ];
        let component_buffers: [Arc<crate::memory::TrackedCudaSlice<u8>>; 4] = [
            Arc::new(component_parents.into_bytes()),
            Arc::new(component_entity_owners.into_bytes()),
            Arc::new(component_count.into_bytes()),
            Arc::new(component_fuel.into_bytes()),
        ];
        let search_buffers: [Arc<crate::memory::TrackedCudaSlice<u8>>; 4] = [
            Arc::new(search_assignment.into_bytes()),
            Arc::new(search_best_label.into_bytes()),
            Arc::new(search_best_total.into_bytes()),
            Arc::new(search_alt_total.into_bytes()),
        ];
        // Deterministic empty session: every buffer is zeroed so a
        // fresh carrier can never read reused device memory — in
        // particular, garbage in the solve-status column could
        // otherwise accidentally read as a claimed authority.
        let stream = device.inner().stream().clone();
        for buffer in buffers
            .iter()
            .chain(component_buffers.iter())
            .chain(search_buffers.iter())
        {
            // SAFETY: each pointer is a live runtime-backed
            // allocation of exactly `len()` bytes on this device.
            unsafe {
                cudarc::driver::result::memset_d8_async(
                    *buffer.device_ptr(),
                    0,
                    buffer.len(),
                    stream.cu_stream(),
                )
                .map_err(|e| {
                    CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                        "carrier zero-init failed: {e}"
                    )))
                })?;
            }
        }
        device.inner().synchronize().map_err(|e| {
            CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                "carrier zero-init sync failed: {e}"
            )))
        })?;

        let columns = [
            shared_column(&buffers[0], &stream),
            shared_column(&buffers[1], &stream),
            shared_column(&buffers[2], &stream),
            shared_column(&buffers[3], &stream),
            shared_column(&buffers[4], &stream),
            shared_column(&buffers[5], &stream),
            shared_column(&buffers[7], &stream),
            shared_column(&buffers[8], &stream),
        ];
        let component_columns = [
            shared_column(&component_buffers[0], &stream),
            shared_column(&component_buffers[1], &stream),
            shared_column(&component_buffers[2], &stream),
            shared_column(&component_buffers[3], &stream),
        ];
        let search_columns = [
            shared_column(&search_buffers[0], &stream),
            shared_column(&search_buffers[1], &stream),
            shared_column(&search_buffers[2], &stream),
            shared_column(&search_buffers[3], &stream),
        ];

        Ok(Self {
            buffers,
            columns,
            component_buffers,
            component_columns,
            search_columns,
            signatures: None,
            registered_schema: None,
            feasibility_solved: false,
            pending_producer_events: Vec::new(),
            pending_consumer_streams: Vec::new(),
            entities,
            domain_lanes,
            candidates,
            labels,
            max_arity,
            device,
            pool,
            memory,
            runtime,
        })
    }

    /// Record a producer-completion event on an EXTERNAL stream (a
    /// raw `CUstream` handle on this device — e.g. torch's
    /// `current_stream().cuda_stream`). The next solve stage waits
    /// on every noted event BEFORE launching, so producer writes
    /// through exported views order against the solve entirely on
    /// device — no host synchronization barrier is involved, which
    /// is what keeps the measured region host-interaction-free.
    pub fn note_producer_stream(&mut self, external_stream: u64) -> Result<(), CarrierError> {
        if external_stream == 0 {
            return Err(CarrierError::Launch(xlog_core::XlogError::Kernel(
                "null producer stream handle".to_string(),
            )));
        }
        // SAFETY: the caller contract is a valid stream handle on
        // this device's context; a stale/foreign handle surfaces as
        // a typed driver error here, never undefined behavior in
        // the solve path.
        unsafe {
            let event = cudarc::driver::result::event::create(
                cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
            )
            .map_err(|e| {
                CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                    "producer event create failed: {e}"
                )))
            })?;
            if let Err(e) = cudarc::driver::result::event::record(
                event,
                external_stream as cudarc::driver::sys::CUstream,
            ) {
                let _ = cudarc::driver::result::event::destroy(event);
                return Err(CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                    "producer event record failed: {e}"
                ))));
            }
            self.pending_producer_events.push(event);
        }
        Ok(())
    }

    /// Make `cu_stream` wait on every pending producer event, then
    /// destroy and clear them. Enqueued waits capture the events, so
    /// destruction is deferred by the driver until they complete.
    fn drain_producer_waits(&mut self, cu_stream: &crate::CudaStream) -> Result<(), CarrierError> {
        for event in self.pending_producer_events.drain(..) {
            // SAFETY: event was created and recorded by
            // note_producer_stream and is consumed exactly once here.
            unsafe {
                let wait = cudarc::driver::result::stream::wait_event(
                    cu_stream.cu_stream(),
                    event,
                    cudarc::driver::sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
                );
                let _ = cudarc::driver::result::event::destroy(event);
                wait.map_err(|e| {
                    CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                        "producer event wait failed: {e}"
                    )))
                })?;
            }
        }
        Ok(())
    }

    /// Register an external CUDA stream to consume the next
    /// successful solve stage. After the solve work is enqueued, the
    /// carrier records one completion event on its internal stream and
    /// makes every registered consumer stream wait on that event.
    pub fn note_consumer_stream(&mut self, external_stream: u64) -> Result<(), CarrierError> {
        if external_stream == 0 {
            return Err(CarrierError::Launch(xlog_core::XlogError::Kernel(
                "null consumer stream handle".to_string(),
            )));
        }
        self.pending_consumer_streams
            .push(external_stream as cudarc::driver::sys::CUstream);
        Ok(())
    }

    /// Publish successful solve completion to every registered
    /// consumer stream, consuming the registrations exactly once.
    fn handoff_consumers(&mut self, cu_stream: &crate::CudaStream) -> Result<(), CarrierError> {
        let consumer_streams = std::mem::take(&mut self.pending_consumer_streams);
        if consumer_streams.is_empty() {
            return Ok(());
        }

        // SAFETY: the event is created in the carrier's current CUDA
        // context. Registered stream handles are caller-guaranteed to
        // be live streams on the same device. Event destruction is
        // deferred by the driver until every enqueued wait completes.
        unsafe {
            let event = cudarc::driver::result::event::create(
                cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
            )
            .map_err(|e| {
                CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                    "consumer event create failed: {e}"
                )))
            })?;
            if let Err(e) = cudarc::driver::result::event::record(event, cu_stream.cu_stream()) {
                let _ = cudarc::driver::result::event::destroy(event);
                return Err(CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                    "consumer event record failed: {e}"
                ))));
            }

            let wait_result = consumer_streams
                .into_iter()
                .try_for_each(|consumer_stream| {
                    cudarc::driver::result::stream::wait_event(
                        consumer_stream,
                        event,
                        cudarc::driver::sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
                    )
                    .map_err(|e| {
                        CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                            "consumer event wait failed: {e}"
                        )))
                    })
                });
            let destroy_result = cudarc::driver::result::event::destroy(event).map_err(|e| {
                CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                    "consumer event destroy failed: {e}"
                )))
            });
            wait_result?;
            destroy_result?;
        }
        Ok(())
    }

    /// Export one buffer outward while xlog retains ownership. The
    /// returned `Arc` shares the exact allocation the carrier solves
    /// on; the binding layer wraps it in a DLPack capsule via
    /// [`CudaColumn::dlpack_xlog_owned`], and strict launch recorders
    /// keep recording the exported view.
    pub fn export_buffer(&self, id: CarrierBufferId) -> CarrierExport {
        let (index, elem_bytes, rows, cols) = match id {
            CarrierBufferId::Domains => (0, 8, self.entities, self.domain_lanes),
            CarrierBufferId::Scores => (1, 4, self.candidates, self.labels),
            CarrierBufferId::Constraints => (2, 4, self.candidates, self.max_arity),
            CarrierBufferId::ArgumentArities => (3, 4, self.candidates, 1),
            CarrierBufferId::Outputs => (4, 4, self.candidates, 1),
            CarrierBufferId::FeasibleSets => (5, 8, self.candidates, label_words(self.labels)),
            CarrierBufferId::LogicalCounts => (6, 4, 1, 4),
            CarrierBufferId::MapResults => (7, 4, self.candidates, 4),
            CarrierBufferId::SolveStatus => (8, 4, self.candidates, 1),
        };
        CarrierExport {
            slice: Arc::clone(&self.buffers[index]),
            stream: self.device.inner().stream().clone(),
            elem_bytes,
            rows,
            cols,
        }
    }

    /// Bind the catalog-bound label signature masks, one cold-path
    /// upload per session after schema registration. Each mask slice
    /// is `labels x domain_lanes` u64 words.
    pub fn bind_signatures(
        &mut self,
        head_masks: &[u64],
        tail_masks: &[u64],
    ) -> Result<(), CarrierError> {
        if self.registered_schema.is_none() {
            return Err(CarrierError::SchemaNotRegistered);
        }
        if self.signatures.is_some() {
            return Err(CarrierError::SignaturesAlreadyBound);
        }
        let expected_words = self.labels * self.domain_lanes;
        for (side, masks) in [("head", head_masks), ("tail", tail_masks)] {
            if masks.len() != expected_words {
                return Err(CarrierError::SignatureShapeMismatch {
                    side,
                    expected_words,
                    got_words: masks.len(),
                });
            }
        }
        if self.max_arity != 2 {
            return Err(CarrierError::SignatureShapeMismatch {
                side: "roles",
                expected_words: self.max_arity * expected_words,
                got_words: 2 * expected_words,
            });
        }
        let mut role_masks = Vec::with_capacity(2 * expected_words);
        role_masks.extend_from_slice(head_masks);
        role_masks.extend_from_slice(tail_masks);
        self.signatures = Some(self.upload_mask(&role_masks)?);
        Ok(())
    }

    /// Bind role-indexed catalog signatures in role-major order:
    /// `max_arity x labels x domain_lanes` u64 words.
    pub fn bind_role_signatures(&mut self, role_masks: &[u64]) -> Result<(), CarrierError> {
        if self.registered_schema.is_none() {
            return Err(CarrierError::SchemaNotRegistered);
        }
        if self.signatures.is_some() {
            return Err(CarrierError::SignaturesAlreadyBound);
        }
        let expected_words = self.max_arity * self.labels * self.domain_lanes;
        if role_masks.len() != expected_words {
            return Err(CarrierError::SignatureShapeMismatch {
                side: "roles",
                expected_words,
                got_words: role_masks.len(),
            });
        }
        self.signatures = Some(self.upload_mask(role_masks)?);
        Ok(())
    }

    /// Cold-path upload of one signature mask into a runtime-backed
    /// column.
    fn upload_mask(&self, masks: &[u64]) -> Result<CudaColumn, CarrierError> {
        let mut slice = self
            .memory
            .alloc::<u64>(masks.len())
            .map_err(CarrierError::Allocation)?;
        self.device
            .inner()
            .htod_sync_copy_into(masks, &mut slice)
            .map_err(|e| {
                CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                    "signature mask upload failed: {e}"
                )))
            })?;
        Ok(CudaColumn::owned(slice.into_bytes()))
    }

    /// Run the existential label-feasibility stage on device through
    /// a strict launch recorder. Fuel is charged with one node
    /// expansion per (candidate, label) cell BEFORE the launch —
    /// beyond fuel the solve refuses typed without touching the
    /// device. Results stay device-resident in the outputs
    /// (feasible counts) and feasible-sets columns.
    pub fn solve_label_feasibility(
        &mut self,
        abstain_label: u32,
        fuel: &mut FuelMeter,
    ) -> Result<(), CarrierError> {
        let result = self.solve_label_feasibility_inner(abstain_label, fuel);
        if result.is_err() {
            self.pending_consumer_streams.clear();
        }
        result
    }

    fn solve_label_feasibility_inner(
        &mut self,
        abstain_label: u32,
        fuel: &mut FuelMeter,
    ) -> Result<(), CarrierError> {
        if self.registered_schema.is_none() {
            return Err(CarrierError::SchemaNotRegistered);
        }
        if self.signatures.is_none() {
            return Err(CarrierError::SignaturesUnbound);
        }
        if abstain_label as usize >= self.labels {
            return Err(CarrierError::AbstainOutOfRange {
                abstain_label,
                labels: self.labels,
            });
        }
        fuel.charge((self.candidates as u64) * (self.labels as u64))
            .map_err(CarrierError::Solver)?;

        let stream_id = self.pool.acquire().map_err(|e| {
            CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                "no launch stream available: {e:?}"
            )))
        })?;
        let cu_stream = self.pool.resolve(stream_id).ok_or_else(|| {
            CarrierError::Launch(xlog_core::XlogError::Kernel(
                "launch stream did not resolve".to_string(),
            ))
        })?;

        self.drain_producer_waits(&cu_stream)?;

        let Some(signatures) = &self.signatures else {
            return Err(CarrierError::SignaturesUnbound);
        };
        let [domains, _scores, arguments, argument_arities, outputs, feasible_sets, _map_results, _solve_status] =
            &self.columns;
        let role_masks = signatures;

        let mut rec = LaunchRecorder::new_strict(stream_id);
        rec.read_column(domains);
        rec.read_column(arguments);
        rec.read_column(argument_arities);
        rec.read_column(role_masks);
        rec.write_column(outputs);
        rec.write_column(feasible_sets);
        rec.preflight(&self.runtime).map_err(|e| {
            CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                "solve launch preflight failed: {e}"
            )))
        })?;

        let kernel = self
            .device
            .inner()
            .get_func(JOINT_SOLVE_MODULE, FEASIBILITY_KERNEL)
            .ok_or_else(|| CarrierError::KernelUnavailable {
                detail: format!("{FEASIBILITY_KERNEL} not resolvable after module load"),
            })?;
        let block = 256u32;
        let grid = (self.candidates as u32).div_ceil(block);
        // SAFETY: joint_label_feasibility(domains, arguments, arities,
        // role_masks, max_arity, num_entities, num_candidates, num_labels,
        // lanes, abstain, feasible_counts, feasible_sets); every pointer is a
        // live runtime-backed carrier column recorded above, and the
        // capacity metadata matches the allocation shapes. Corrupt
        // invalid arities or entity indices poison their row inside the kernel.
        unsafe {
            kernel
                .launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        *domains.device_ptr(),
                        *arguments.device_ptr(),
                        *argument_arities.device_ptr(),
                        *role_masks.device_ptr(),
                        self.max_arity as u32,
                        self.entities as u32,
                        self.candidates as u32,
                        self.labels as u32,
                        self.domain_lanes as u32,
                        abstain_label,
                        *outputs.device_ptr(),
                        *feasible_sets.device_ptr(),
                    ),
                )
                .map_err(|e| {
                    CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                        "solve launch failed: {e}"
                    )))
                })?;
        }
        rec.commit(&self.runtime).map_err(|e| {
            CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                "solve launch commit failed: {e}"
            )))
        })?;
        self.handoff_consumers(&cu_stream)?;
        self.feasibility_solved = true;
        Ok(())
    }

    /// Run the per-candidate exact top-two stage on device, consuming
    /// the feasibility stage's feasible sets — a real produce/consume
    /// chain whose cross-stream ordering rides on the recorded launch
    /// events, not on host synchronization. Fuel is charged one node
    /// expansion per (candidate, label) cell BEFORE the launch.
    ///
    /// The results are the exact global max-marginal ONLY for
    /// single-candidate components (see
    /// [`crate::joint_solver::ConstraintGraph::decompose`]); a set
    /// ambiguity flag is a typed MAP-ambiguity signal and must never
    /// emit as a unique label. Multi-candidate components stay behind
    /// the cross-candidate dynamic-programming stage.
    pub fn solve_label_map_top2(&mut self, fuel: &mut FuelMeter) -> Result<(), CarrierError> {
        let result = self.solve_label_map_top2_inner(fuel);
        if result.is_err() {
            self.pending_consumer_streams.clear();
        }
        result
    }

    fn solve_label_map_top2_inner(&mut self, fuel: &mut FuelMeter) -> Result<(), CarrierError> {
        if self.registered_schema.is_none() {
            return Err(CarrierError::SchemaNotRegistered);
        }
        if !self.feasibility_solved {
            return Err(CarrierError::FeasibilityNotSolved);
        }
        fuel.charge((self.candidates as u64) * (self.labels as u64))
            .map_err(CarrierError::Solver)?;

        let stream_id = self.pool.acquire().map_err(|e| {
            CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                "no launch stream available: {e:?}"
            )))
        })?;
        let cu_stream = self.pool.resolve(stream_id).ok_or_else(|| {
            CarrierError::Launch(xlog_core::XlogError::Kernel(
                "launch stream did not resolve".to_string(),
            ))
        })?;

        self.drain_producer_waits(&cu_stream)?;

        let [_domains, scores, _arguments, _argument_arities, _outputs, feasible_sets, map_results, _solve_status] =
            &self.columns;

        let mut rec = LaunchRecorder::new_strict(stream_id);
        rec.read_column(scores);
        rec.read_column(feasible_sets);
        rec.write_column(map_results);
        rec.preflight(&self.runtime).map_err(|e| {
            CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                "top-two launch preflight failed: {e}"
            )))
        })?;

        let kernel = self
            .device
            .inner()
            .get_func(JOINT_SOLVE_MODULE, TOP2_KERNEL)
            .ok_or_else(|| CarrierError::KernelUnavailable {
                detail: format!("{TOP2_KERNEL} not resolvable after module load"),
            })?;
        let block = 256u32;
        let grid = (self.candidates as u32).div_ceil(block);
        // SAFETY: joint_label_top2(scores, feasible_sets,
        // num_candidates, num_labels, map_results); every pointer is
        // a live runtime-backed carrier column recorded above.
        unsafe {
            kernel
                .launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        *scores.device_ptr(),
                        *feasible_sets.device_ptr(),
                        self.candidates as u32,
                        self.labels as u32,
                        *map_results.device_ptr(),
                    ),
                )
                .map_err(|e| {
                    CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                        "top-two launch failed: {e}"
                    )))
                })?;
        }
        rec.commit(&self.runtime).map_err(|e| {
            CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                "top-two launch commit failed: {e}"
            )))
        })?;
        self.handoff_consumers(&cu_stream)?;
        Ok(())
    }

    /// Discover candidate components from the carrier-owned argument
    /// matrix and solve every component exactly on the device. No
    /// component membership crosses the host boundary. Components
    /// beyond complete-enumeration capacity continue through the
    /// device-discovered exact chain DP and then general device
    /// branch-and-bound. Topology and arity never select a refusal;
    /// only measured fuel exhaustion may refuse (status 3).
    pub fn solve_components_exact(&mut self, fuel: &mut FuelMeter) -> Result<(), CarrierError> {
        let result = self.solve_components_exact_inner(fuel);
        if result.is_err() {
            self.pending_consumer_streams.clear();
        }
        result
    }

    fn solve_components_exact_inner(&mut self, fuel: &mut FuelMeter) -> Result<(), CarrierError> {
        if self.registered_schema.is_none() {
            return Err(CarrierError::SchemaNotRegistered);
        }
        if !self.feasibility_solved {
            return Err(CarrierError::FeasibilityNotSolved);
        }

        // Component count stays device-resident. Authorize the whole
        // remaining budget up front; the exact kernel divides it by
        // the device-computed count, and the device reports only the
        // actual expansions through one bounded metadata counter.
        let authorized = fuel.remaining();
        fuel.charge(authorized).map_err(CarrierError::Solver)?;

        let stream_id = self.pool.acquire().map_err(|e| {
            CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                "no launch stream available: {e:?}"
            )))
        })?;
        let cu_stream = self.pool.resolve(stream_id).ok_or_else(|| {
            CarrierError::Launch(xlog_core::XlogError::Kernel(
                "launch stream did not resolve".to_string(),
            ))
        })?;
        self.drain_producer_waits(&cu_stream)?;

        let Some(signatures) = &self.signatures else {
            return Err(CarrierError::SignaturesUnbound);
        };
        let [domains, scores, arguments, argument_arities, _outputs, feasible_sets, map_results, solve_status] =
            &self.columns;
        let [component_parents, component_entity_owners, component_count, component_fuel] =
            &self.component_columns;
        let [search_assignment, search_best_label, search_best_total, search_alt_total] =
            &self.search_columns;
        let role_masks = signatures;

        let mut rec = LaunchRecorder::new_strict(stream_id);
        rec.read_column(scores);
        rec.read_column(feasible_sets);
        rec.read_column(arguments);
        rec.read_column(argument_arities);
        rec.read_column(domains);
        rec.read_column(role_masks);
        rec.read_column(component_parents);
        rec.write_column(component_parents);
        rec.read_column(component_entity_owners);
        rec.write_column(component_entity_owners);
        rec.read_column(component_count);
        rec.write_column(component_count);
        rec.write_column(component_fuel);
        for search_column in [
            search_assignment,
            search_best_label,
            search_best_total,
            search_alt_total,
        ] {
            rec.read_column(search_column);
            rec.write_column(search_column);
        }
        rec.write_column(map_results);
        rec.write_column(solve_status);
        rec.preflight(&self.runtime).map_err(|e| {
            CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                "component solve preflight failed: {e}"
            )))
        })?;

        let plan_init_kernel = self
            .device
            .inner()
            .get_func(JOINT_SOLVE_MODULE, COMPONENT_PLAN_INIT_KERNEL)
            .ok_or_else(|| CarrierError::KernelUnavailable {
                detail: format!("{COMPONENT_PLAN_INIT_KERNEL} not resolvable after module load"),
            })?;
        let entity_owners_kernel = self
            .device
            .inner()
            .get_func(JOINT_SOLVE_MODULE, COMPONENT_ENTITY_OWNERS_KERNEL)
            .ok_or_else(|| CarrierError::KernelUnavailable {
                detail: format!(
                    "{COMPONENT_ENTITY_OWNERS_KERNEL} not resolvable after module load"
                ),
            })?;
        let union_kernel = self
            .device
            .inner()
            .get_func(JOINT_SOLVE_MODULE, COMPONENT_UNION_KERNEL)
            .ok_or_else(|| CarrierError::KernelUnavailable {
                detail: format!("{COMPONENT_UNION_KERNEL} not resolvable after module load"),
            })?;
        let compress_kernel = self
            .device
            .inner()
            .get_func(JOINT_SOLVE_MODULE, COMPONENT_COMPRESS_KERNEL)
            .ok_or_else(|| CarrierError::KernelUnavailable {
                detail: format!("{COMPONENT_COMPRESS_KERNEL} not resolvable after module load"),
            })?;
        let exact_kernel = self
            .device
            .inner()
            .get_func(JOINT_SOLVE_MODULE, COMPONENT_KERNEL)
            .ok_or_else(|| CarrierError::KernelUnavailable {
                detail: format!("{COMPONENT_KERNEL} not resolvable after module load"),
            })?;
        let chain_dp_kernel = self
            .device
            .inner()
            .get_func(JOINT_SOLVE_MODULE, CHAIN_DP_KERNEL)
            .ok_or_else(|| CarrierError::KernelUnavailable {
                detail: format!("{CHAIN_DP_KERNEL} not resolvable after module load"),
            })?;
        let branch_and_bound_kernel = self
            .device
            .inner()
            .get_func(JOINT_SOLVE_MODULE, BRANCH_AND_BOUND_KERNEL)
            .ok_or_else(|| CarrierError::KernelUnavailable {
                detail: format!("{BRANCH_AND_BOUND_KERNEL} not resolvable after module load"),
            })?;
        // SAFETY: each raw parameter array below matches its CUDA
        // kernel ABI exactly. Every pointer is a live runtime-backed
        // column recorded above, every launch uses the same stream,
        // and the scalar locals live through all enqueues.
        unsafe {
            use std::ffi::c_void;
            let scores_p = *scores.device_ptr();
            let feasible_p = *feasible_sets.device_ptr();
            let arguments_p = *arguments.device_ptr();
            let arities_p = *argument_arities.device_ptr();
            let domains_p = *domains.device_ptr();
            let role_masks_p = *role_masks.device_ptr();
            let parents_p = *component_parents.device_ptr();
            let entity_owners_p = *component_entity_owners.device_ptr();
            let component_count_p = *component_count.device_ptr();
            let component_fuel_p = *component_fuel.device_ptr();
            let search_assignment_p = *search_assignment.device_ptr();
            let search_best_label_p = *search_best_label.device_ptr();
            let search_best_total_p = *search_best_total.device_ptr();
            let search_alt_total_p = *search_alt_total.device_ptr();
            let num_candidates_v = self.candidates as u32;
            let num_entities_v = self.entities as u32;
            let num_labels_v = self.labels as u32;
            let lanes_v = self.domain_lanes as u32;
            let max_arity_v = self.max_arity as u32;
            let map_p = *map_results.device_ptr();
            let status_p = *solve_status.device_ptr();
            let planner_threads = 256u32;
            let planner_rows = num_candidates_v.max(num_entities_v);
            let planner_grid = planner_rows.div_ceil(planner_threads);
            let candidate_grid = num_candidates_v.div_ceil(planner_threads);

            let mut init_params: [*mut c_void; 6] = [
                &parents_p as *const _ as *mut c_void,
                &num_candidates_v as *const _ as *mut c_void,
                &entity_owners_p as *const _ as *mut c_void,
                &num_entities_v as *const _ as *mut c_void,
                &component_count_p as *const _ as *mut c_void,
                &component_fuel_p as *const _ as *mut c_void,
            ];
            plan_init_kernel
                .launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (planner_grid, 1, 1),
                        block_dim: (planner_threads, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut init_params[..],
                )
                .map_err(|e| {
                    CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                        "component plan initialization failed: {e}"
                    )))
                })?;

            let mut owner_params: [*mut c_void; 6] = [
                &arguments_p as *const _ as *mut c_void,
                &arities_p as *const _ as *mut c_void,
                &num_candidates_v as *const _ as *mut c_void,
                &max_arity_v as *const _ as *mut c_void,
                &num_entities_v as *const _ as *mut c_void,
                &entity_owners_p as *const _ as *mut c_void,
            ];
            entity_owners_kernel
                .launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (candidate_grid, 1, 1),
                        block_dim: (planner_threads, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut owner_params[..],
                )
                .map_err(|e| {
                    CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                        "component entity ownership failed: {e}"
                    )))
                })?;

            let mut union_params: [*mut c_void; 7] = [
                &arguments_p as *const _ as *mut c_void,
                &arities_p as *const _ as *mut c_void,
                &num_candidates_v as *const _ as *mut c_void,
                &max_arity_v as *const _ as *mut c_void,
                &num_entities_v as *const _ as *mut c_void,
                &entity_owners_p as *const _ as *mut c_void,
                &parents_p as *const _ as *mut c_void,
            ];
            union_kernel
                .launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (candidate_grid, 1, 1),
                        block_dim: (planner_threads, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut union_params[..],
                )
                .map_err(|e| {
                    CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                        "component union failed: {e}"
                    )))
                })?;

            let mut compress_params: [*mut c_void; 3] = [
                &parents_p as *const _ as *mut c_void,
                &num_candidates_v as *const _ as *mut c_void,
                &component_count_p as *const _ as *mut c_void,
            ];
            compress_kernel
                .launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (candidate_grid, 1, 1),
                        block_dim: (planner_threads, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut compress_params[..],
                )
                .map_err(|e| {
                    CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                        "component compression failed: {e}"
                    )))
                })?;

            let mut exact_params: [*mut c_void; 16] = [
                &scores_p as *const _ as *mut c_void,
                &feasible_p as *const _ as *mut c_void,
                &arguments_p as *const _ as *mut c_void,
                &arities_p as *const _ as *mut c_void,
                &domains_p as *const _ as *mut c_void,
                &role_masks_p as *const _ as *mut c_void,
                &parents_p as *const _ as *mut c_void,
                &component_count_p as *const _ as *mut c_void,
                &num_candidates_v as *const _ as *mut c_void,
                &num_labels_v as *const _ as *mut c_void,
                &lanes_v as *const _ as *mut c_void,
                &max_arity_v as *const _ as *mut c_void,
                &authorized as *const _ as *mut c_void,
                &map_p as *const _ as *mut c_void,
                &status_p as *const _ as *mut c_void,
                &component_fuel_p as *const _ as *mut c_void,
            ];
            exact_kernel
                .launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (num_candidates_v, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut exact_params[..],
                )
                .map_err(|e| {
                    CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                        "component solve launch failed: {e}"
                    )))
                })?;
            chain_dp_kernel
                .launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (num_candidates_v, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut exact_params[..],
                )
                .map_err(|e| {
                    CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                        "device-discovered chain DP launch failed: {e}"
                    )))
                })?;
            let mut branch_params: [*mut c_void; 20] = [
                &scores_p as *const _ as *mut c_void,
                &feasible_p as *const _ as *mut c_void,
                &arguments_p as *const _ as *mut c_void,
                &arities_p as *const _ as *mut c_void,
                &domains_p as *const _ as *mut c_void,
                &role_masks_p as *const _ as *mut c_void,
                &parents_p as *const _ as *mut c_void,
                &component_count_p as *const _ as *mut c_void,
                &num_candidates_v as *const _ as *mut c_void,
                &num_labels_v as *const _ as *mut c_void,
                &lanes_v as *const _ as *mut c_void,
                &max_arity_v as *const _ as *mut c_void,
                &authorized as *const _ as *mut c_void,
                &map_p as *const _ as *mut c_void,
                &status_p as *const _ as *mut c_void,
                &component_fuel_p as *const _ as *mut c_void,
                &search_assignment_p as *const _ as *mut c_void,
                &search_best_label_p as *const _ as *mut c_void,
                &search_best_total_p as *const _ as *mut c_void,
                &search_alt_total_p as *const _ as *mut c_void,
            ];
            branch_and_bound_kernel
                .launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (num_candidates_v, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut branch_params[..],
                )
                .map_err(|e| {
                    CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                        "general exact branch-and-bound launch failed: {e}"
                    )))
                })?;
        }
        rec.commit(&self.runtime).map_err(|e| {
            CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                "component solve commit failed: {e}"
            )))
        })?;

        // Bounded post-solve metadata read (num_rows class): one 8-byte
        // counter after a stream-scoped completion wait, reconciling
        // the meter to the DEVICE-measured expansions.
        let mut measured = [0u64; 1];
        unsafe {
            cudarc::driver::result::stream::synchronize(cu_stream.cu_stream()).map_err(|e| {
                CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                    "component solve completion wait failed: {e}"
                )))
            })?;
            cudarc::driver::result::memcpy_dtoh_sync(
                &mut measured,
                *self.component_buffers[3].device_ptr(),
            )
            .map_err(|e| {
                CarrierError::Launch(xlog_core::XlogError::Kernel(format!(
                    "fuel counter readback failed: {e}"
                )))
            })?;
        }
        fuel.refund(authorized.saturating_sub(measured[0]));
        self.handoff_consumers(&cu_stream)?;
        Ok(())
    }

    /// All device columns the carrier owns, in a stable order:
    /// domains, scores, constraints, outputs (feasible counts),
    /// feasible sets, map results, solve status.
    pub fn columns(&self) -> impl Iterator<Item = &CudaColumn> {
        self.columns.iter()
    }

    /// Bind the carrier session to one catalog anchor and one solver
    /// identity (see [`crate::joint_solver::SOLVER_ABI_IDENTITY`]).
    /// Registration is once-per-session: a second call refuses with
    /// the typed [`CarrierError::SchemaAlreadyRegistered`] variant
    /// carrying both bound identities.
    pub fn register_schema(
        &mut self,
        catalog_sha: &str,
        solver_identity: &str,
    ) -> Result<(), CarrierError> {
        if let Some((catalog, solver)) = &self.registered_schema {
            return Err(CarrierError::SchemaAlreadyRegistered {
                catalog_sha: catalog.clone(),
                solver_identity: solver.clone(),
            });
        }
        self.registered_schema = Some((catalog_sha.to_string(), solver_identity.to_string()));
        Ok(())
    }
}
