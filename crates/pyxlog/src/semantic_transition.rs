#![cfg_attr(
    not(feature = "semantic-policy"),
    expect(
        dead_code,
        reason = "prepared graph bindings are activated by the semantic-policy feature"
    )
)]

//! Python ownership of the canonical admitted semantic session and cold policy layout.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::thread::ThreadId;

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyInt, PyList, PyString, PyTuple};
use sha2::{Digest, Sha256};
use xlog_core::{RelId, ScalarType, Schema};
use xlog_cuda::memory::DeviceAllocationProvenance;
use xlog_cuda::{
    DlpackManagedTensor, Identity256, SemanticAdmissionLimits, SemanticAdmissionRecords,
    SemanticArgument, SemanticContinuationInput, SemanticHypergraphCapacities,
    SemanticObservedSource, SemanticParentBinding, SemanticPolarity, SemanticPredicateRecord,
    SemanticPreparedStep, SemanticPublishedLease, SemanticRecordRole, SemanticRngBinding,
    SemanticSourceMapping, SemanticStateRecord, SemanticStateRole, SemanticSupportRecord,
    SemanticTensorContentWitness, SemanticTensorInput, SemanticTensorLayout, SemanticTextSlot,
    SemanticTrainingCanary, SemanticTrainingCanaryKind, SemanticTrainingObjective,
    SemanticTrainingObjectiveGroup, SemanticTrainingObjectiveGroupKind, SemanticTrainingViewBasis,
    SemanticTrainingViewPort, SemanticTrainingViewRow, SemanticTransitionKind,
    SemanticTransitionSession, SemanticTypedRecord,
};
use xlog_cuda::{
    SemanticModelContractLayout, SemanticModelMemory, SemanticModelStorage, SemanticModelView,
};
use xlog_prob::exact::GpuConfig;

use crate::guarded_python_callback as recording_callback;
use crate::types::{val_err, xlog_err};

type PredicateInput = (u32, String, Vec<(String, u8, String)>, Vec<usize>);
type RecordInput = (u32, Vec<(u8, Py<PyAny>)>, Vec<u32>);
type SupportInput = (u32, String, u32, u32, u32, u32);

/// Own one admitted native semantic session and its acquired policy codebooks.
///
/// All constructor arguments are required and keyword-only. ``predicates`` rows
/// are ``(predicate_id, role, columns, key_columns)``. Each column is
/// ``(name, scalar_type_code, sort_label)``. Roles are ``statement``, ``qualifier``,
/// ``provenance``, ``source``, ``context``, and ``scope``.
///
/// ``records`` rows are ``(predicate_id, arguments, qualifier_indices)``; each
/// argument is ``(scalar_type_code, value)``. Canonical XLOG scalar codes are
/// 0=u32, 1=u64, 2=i32, 3=i64, 4=f32 bits, 5=f64 bits, 6=bool, 7=symbol ID.
/// Float arguments require unsigned integer IEEE bit patterns. Symbol IDs come
/// from ``intern_symbols``; canonical admission retains the accepted symbol bytes.
///
/// ``supports`` rows are ``(statement_index, polarity, provenance_index,
/// source_index, context_index, scope_index)`` with polarity ``pro`` or ``contra``.
/// All record indices address ``records``. ``capacities`` is
/// ``(roots, statements, supports, versions)``. ``admission_limits`` is
/// ``(max_records, max_terms, max_references, max_utf8_bytes)``. ``device_ordinal``
/// selects the CUDA device and ``memory_bytes`` bounds its allocation budget.
///
/// Construction allocates the native CUDA owner and admits the supplied typed
/// descriptors against its empty base. It does not insert supplied supports into
/// that base or grant task execution, publication, inference, or training authority.
/// Public orchestration and reads belong to the constructing Python thread.
/// The trusted Runtime must retain every issued witness and policy invocation,
/// their original producers, callbacks and saved autograd payloads through the
/// original backward call, then perform one guaranteed creator-thread cleanup.
/// External aliases require their owning Runtime to remain alive until release.
/// Transferable native ownership does not make arbitrary Python finalizers safe
/// on workers. Only a retained content witness may verify from a worker thread.
#[pyclass(name = "SemanticTransitionSession", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticTransitionSession {
    inner: Mutex<SemanticTransitionSession>,
    importing: Arc<AtomicBool>,
    recording: AtomicBool,
    retiring: AtomicBool,
    prepared_segment: Mutex<Option<PreparedPythonSegment>>,
    issuance: Arc<AtomicU64>,
    owner_thread: ThreadId,
    device_ordinal: usize,
}

impl PySemanticTransitionSession {
    fn require_creator(&self) -> PyResult<()> {
        require_creator_thread(self.owner_thread)
    }

    fn owner(&self) -> PyResult<MutexGuard<'_, SemanticTransitionSession>> {
        self.require_creator()?;
        if !self.recording.load(Ordering::Acquire) {
            drain_export_owners();
        }
        self.witness_owner()
    }

    // Only content verification enters this lock from an autograd worker.
    // Do not drain creator-owned Python references or run callbacks here.
    fn witness_owner(&self) -> PyResult<MutexGuard<'_, SemanticTransitionSession>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("native semantic session owner mutex is poisoned"))
    }
}

#[pymethods]
impl PySemanticTransitionSession {
    #[new]
    #[pyo3(signature = (*, predicates, records, supports, capacities, admission_limits, device_ordinal, memory_bytes))]
    #[expect(
        clippy::too_many_arguments,
        reason = "session construction receives independent admission and device budgets"
    )]
    fn new(
        py: Python<'_>,
        predicates: Vec<PredicateInput>,
        records: Vec<RecordInput>,
        supports: Vec<SupportInput>,
        capacities: (u32, u32, u32, u32),
        admission_limits: (u32, u32, u32, usize),
        device_ordinal: usize,
        memory_bytes: u64,
    ) -> PyResult<Self> {
        let capacities = SemanticHypergraphCapacities::try_new(
            capacities.0,
            capacities.1,
            capacities.2,
            capacities.3,
        )
        .map_err(val_err)?;
        let records = parse_admission(py, predicates, records, supports)?;
        let limits = SemanticAdmissionLimits {
            max_records: admission_limits.0,
            max_terms: admission_limits.1,
            max_references: admission_limits.2,
            max_utf8_bytes: admission_limits.3,
        };
        let mut config = GpuConfig::default();
        config.device_ordinal = device_ordinal;
        config.memory_bytes = memory_bytes;
        let provider = Arc::new(crate::provider_from_config(config).map_err(xlog_err)?);
        let runtime = provider.memory().runtime().cloned().ok_or_else(|| {
            PyRuntimeError::new_err("native semantic session requires a CUDA runtime")
        })?;
        let stream_id = runtime.stream_pool().acquire().map_err(xlog_err)?;
        let stream = runtime.stream_pool().resolve(stream_id).ok_or_else(|| {
            PyRuntimeError::new_err("native semantic session stream is unavailable")
        })?;
        let domain = provider
            .bind_resident_execution_domain(runtime, stream_id, stream)
            .map_err(xlog_err)?;
        let mut graph = provider
            .allocate_semantic_hypergraph(&domain, capacities)
            .map_err(xlog_err)?;
        graph
            .admit_records(graph.empty_root(), records, limits)
            .map_err(xlog_err)?;
        let session = SemanticTransitionSession::from_hypergraph(graph).map_err(xlog_err)?;
        Ok(Self {
            inner: Mutex::new(session),
            importing: Arc::new(AtomicBool::new(false)),
            recording: AtomicBool::new(false),
            retiring: AtomicBool::new(false),
            prepared_segment: Mutex::new(None),
            issuance: Arc::new(AtomicU64::new(0)),
            owner_thread: std::thread::current().id(),
            device_ordinal,
        })
    }

    /// Return this retained session's ``(generation, digest_bytes)`` binding.
    fn binding(&self, py: Python<'_>) -> PyResult<(u64, Py<PyBytes>)> {
        self.require_creator()?;
        let binding = self.owner()?.binding();
        Ok((
            binding.generation,
            PyBytes::new(py, binding.digest.as_bytes()).unbind(),
        ))
    }

    /// Return one immutable snapshot of the acquired binding and parameter layout.
    ///
    /// The result is ``(binding, z_range, recurrence_range, positions_range,
    /// fields, parameter_cells)``. ``binding`` is ``(generation, digest_bytes)``.
    /// The eighteen ordered ``fields`` are ``(cardinality, null_category,
    /// embeddings_range, biases_range)``. Every range is a half-open ``(start,
    /// end)`` tuple measured in FP32 elements. NULL is ``None`` when absent.
    /// All values come from this owner's admitted codebooks under one lock.
    fn policy_layout(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_creator()?;
        let owner = self.owner()?;
        let layout = owner.policy_layout().map_err(xlog_err)?;
        let binding = owner.binding();
        let fields = PyTuple::new(
            py,
            layout.fields.iter().map(|field| {
                (
                    field.cardinality,
                    field.null_category,
                    (field.embeddings.start, field.embeddings.end),
                    (field.biases.start, field.biases.end),
                )
            }),
        )?;
        Ok((
            (
                binding.generation,
                PyBytes::new(py, binding.digest.as_bytes()),
            ),
            (layout.z.start, layout.z.end),
            (layout.recurrence.start, layout.recurrence.end),
            (layout.positions.start, layout.positions.end),
            fields,
            layout.parameter_cells,
        )
            .into_pyobject(py)?
            .unbind())
    }
}

/// Owned primitive transport. Integers remain decimal strings until a particular
/// native index requires a checked bound; training identity integers are lossless.
#[derive(Clone, Debug, PartialEq, Eq)]
enum ColdValue {
    None,
    Bool(bool),
    Integer(String),
    Text(String),
    Sequence(Vec<ColdValue>),
    // Only the dedicated replay reader constructs bytes; general metadata
    // transport still refuses bytes and arbitrary dictionaries.
    Bytes(Arc<[u8]>),
}

fn invalid(message: &str) -> PyErr {
    PyValueError::new_err(message.to_owned())
}

#[cfg(feature = "semantic-policy")]
fn transition_refusal(
    outcome: &xlog_cuda::SemanticTransitionOutcome,
) -> Option<(&'static str, u64)> {
    use xlog_cuda::{SemanticTransitionOutcome, SemanticTransitionRefusal};
    match outcome {
        SemanticTransitionOutcome::Published(_) => None,
        SemanticTransitionOutcome::Refused(refusal) => Some(match refusal {
            SemanticTransitionRefusal::InvalidFinalSupport { completed_draws } => {
                ("invalid_final_support", *completed_draws)
            }
            SemanticTransitionRefusal::NonFinitePolicyInput { completed_draws } => {
                ("non_finite_policy_input", *completed_draws)
            }
            SemanticTransitionRefusal::WorkCounterOverflow { completed_draws } => {
                ("work_counter_overflow", *completed_draws)
            }
            SemanticTransitionRefusal::InvalidTrainingView { status } => {
                ("invalid_training_view", *status)
            }
            SemanticTransitionRefusal::RejectedModelUpdate { evidence } => {
                let reason = match evidence.reason() {
                    Some(xlog_cuda::SemanticTrainingCanaryRefusalReason::NonFiniteMeasurement) => {
                        "canary_non_finite_measurement"
                    }
                    Some(xlog_cuda::SemanticTrainingCanaryRefusalReason::OutsideBounds) => {
                        "canary_outside_bounds"
                    }
                    Some(xlog_cuda::SemanticTrainingCanaryRefusalReason::MemoryLimitExceeded) => {
                        "canary_memory_limit_exceeded"
                    }
                    Some(xlog_cuda::SemanticTrainingCanaryRefusalReason::FuelLimitExceeded) => {
                        "canary_fuel_limit_exceeded"
                    }
                    None => "invalid_canary_refusal",
                };
                (reason, evidence.row_ordinal)
            }
        }),
    }
}

#[cfg(feature = "semantic-policy")]
fn transition_refusal_report(
    py: Python<'_>,
    outcome: &xlog_cuda::SemanticTransitionOutcome,
) -> PyResult<Option<Py<PyDict>>> {
    use xlog_cuda::{SemanticTransitionOutcome, SemanticTransitionRefusal};
    let SemanticTransitionOutcome::Refused(refusal) = outcome else {
        return Ok(None);
    };
    let report = PyDict::new(py);
    let (reason, detail) = transition_refusal(outcome)
        .ok_or_else(|| invalid("native refusal has no public reason"))?;
    report.set_item("reason", reason)?;
    match refusal {
        SemanticTransitionRefusal::InvalidFinalSupport { .. }
        | SemanticTransitionRefusal::NonFinitePolicyInput { .. }
        | SemanticTransitionRefusal::WorkCounterOverflow { .. } => {
            report.set_item("completed_draws", detail)?;
        }
        SemanticTransitionRefusal::InvalidTrainingView { .. } => {
            report.set_item("training_view_status", detail)?;
        }
        SemanticTransitionRefusal::RejectedModelUpdate { evidence } => {
            report.set_item("canary_kind", evidence.kind)?;
            report.set_item("row_ordinal", evidence.row_ordinal)?;
            report.set_item("measurement_bits", evidence.measurement_bits)?;
            report.set_item("lower_bound_bits", evidence.lower_bound_bits)?;
            report.set_item("upper_bound_bits", evidence.upper_bound_bits)?;
            report.set_item("memory_used", evidence.memory_used)?;
            report.set_item("memory_limit", evidence.memory_limit)?;
            report.set_item("fuel_used", evidence.fuel_used)?;
            report.set_item("fuel_limit", evidence.fuel_limit)?;
            report.set_item("canary_identity", evidence.identity)?;
            report.set_item("selection_identity", evidence.selection_identity)?;
        }
    }
    Ok(Some(report.unbind()))
}

fn require_creator_thread(creator: ThreadId) -> PyResult<()> {
    if std::thread::current().id() != creator {
        return Err(PyRuntimeError::new_err(
            "native semantic orchestration and reads require the creating thread",
        ));
    }
    Ok(())
}

fn parse_witness_consumer_stream(value: &Bound<'_, PyAny>, budget: &mut usize) -> PyResult<u64> {
    let stream = ColdValue::read(value, budget, 0)?.unsigned()?;
    if stream == 0 || stream == 2 {
        return Err(invalid("tensor content requires an explicit consumer stream or legacy default stream 1; stream 0 and per-thread default stream 2 are invalid"));
    }
    Ok(stream)
}

fn parse_model_work(
    kind: &Bound<'_, PyAny>,
    dimensions: &Bound<'_, PyAny>,
) -> PyResult<(xlog_cuda::ModelWorkKind, [u64; 8], usize)> {
    let kind = ColdValue::read(kind, &mut 128, 0)?.unsigned()?;
    let kind = xlog_cuda::ModelWorkKind::from_code(kind)
        .ok_or_else(|| invalid("unknown model work unit category"))?;
    let geometry = ColdValue::read(dimensions, &mut 1024, 0)?;
    let dimensions = geometry.sequence()?;
    if dimensions.len() > 8 {
        return Err(invalid("model work geometry exceeds eight dimensions"));
    }
    let mut values = [0u64; 8];
    for (index, dimension) in dimensions.iter().enumerate() {
        values[index] = dimension.unsigned()?;
    }
    Ok((kind, values, dimensions.len()))
}

fn transition_kind(value: &ColdValue) -> PyResult<SemanticTransitionKind> {
    match value.text()? {
        "proposal" => Ok(SemanticTransitionKind::Proposal),
        "recompute" => Ok(SemanticTransitionKind::Recompute),
        "drain" => Ok(SemanticTransitionKind::Drain),
        "update" => Ok(SemanticTransitionKind::Update),
        _ => Err(invalid(
            "transition must be proposal, recompute, update, or drain",
        )),
    }
}

fn prepared_transition(
    value: &Bound<'_, PyAny>,
    budget: &mut usize,
) -> PyResult<SemanticTransitionKind> {
    if !value.is_exact_instance_of::<PyString>() {
        return Err(invalid(
            "a prepared transition must be an exact proposal, recompute, or update string",
        ));
    }
    let kind = transition_kind(&ColdValue::read(value, budget, 0)?)?;
    if kind == SemanticTransitionKind::Drain {
        return Err(invalid(
            "segment drain is selected only by the acquired device terminal state",
        ));
    }
    Ok(kind)
}

fn prepared_transitions<'py>(
    value: &Bound<'py, PyAny>,
) -> PyResult<impl ExactSizeIterator<Item = SemanticTransitionKind> + Clone + 'py> {
    if !value.is_exact_instance_of::<PyTuple>() {
        return Err(invalid(
            "segment transitions must be an exact tuple of proposal, recompute, or update modes",
        ));
    }
    let values = value.cast::<PyTuple>()?.clone();
    if values.is_empty() {
        return Err(invalid("segment transitions must not be empty"));
    }
    let mut budget = 16 * 1024 * 1024;
    for item in values.iter() {
        prepared_transition(&item, &mut budget)?;
    }
    // Retain only the caller's exact immutable tuple. Native reserves the full
    // segment budget before materializing any T-sized collection of modes.
    Ok((0..values.len()).map(move |index| {
        prepared_transition(
            &values.get_item(index).expect("validated exact tuple index"),
            &mut 128,
        )
        .expect("validated immutable ordinary transition")
    }))
}

// Alias retirement returns its retained Session reference before the Runtime's
// guaranteed creator-thread release. This queue is not a worker-finalizer safety
// guarantee: supported Runtime custody must retain every original Python graph
// owner through backward and close aliases before its creator-thread cleanup.
enum PublishedExportOwner {
    Session {
        _session: Py<PySemanticTransitionSession>,
    },
    Parent {
        _parent: Py<PySemanticPublishedParent>,
    },
    Prepared {
        _step: Py<PySemanticPreparedStep>,
    },
    Producers {
        _producers: Vec<Py<PyAny>>,
    },
}

/// The graph and Session share the original pool/model owner until explicit
/// creator-thread retirement after known completion and all late consumers.
struct PreparedProducerResources {
    producers: Mutex<Vec<Py<PyAny>>>,
    owner_thread: ThreadId,
}

impl Drop for PreparedProducerResources {
    fn drop(&mut self) {
        let producers = std::mem::take(
            self.producers
                .get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        // Always defer the final decrefs out of native graph/Session destruction.
        DEFERRED_EXPORT_OWNERS
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((
                self.owner_thread,
                PublishedExportOwner::Producers {
                    _producers: producers,
                },
            ));
    }
}

struct PreparedPythonSegment {
    scope: Arc<()>,
    task_use: Py<PySemanticTransitionTaskUse>,
    steps: Vec<Py<PySemanticPreparedStep>>,
    resources: Arc<PreparedProducerResources>,
    producers_retired: bool,
}

struct PreparedBuildGuard<'a> {
    session: &'a PySemanticTransitionSession,
    task_use: &'a PySemanticTransitionTaskUse,
    committed: bool,
}

struct PreparedRetirementGuard<'a>(&'a AtomicBool);

impl Drop for PreparedRetirementGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl Drop for PreparedBuildGuard<'_> {
    fn drop(&mut self) {
        if !self.committed {
            if let Ok(mut state) = self.task_use.state() {
                state.phase = TaskUsePhase::Refused;
            }
            self.session
                .inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .abort();
        }
        self.session.recording.store(false, Ordering::Release);
    }
}

/// Declared after the build guard and before the graph, so unwinding destroys
/// capture first, exits the original pool second, and finally aborts the task.
struct PreparedMemoryScope<'py> {
    py: Python<'py>,
    exit: Bound<'py, PyAny>,
    entered: bool,
}

impl<'py> PreparedMemoryScope<'py> {
    fn new(
        py: Python<'py>,
        scope: &Bound<'py, PyAny>,
        check: &dyn Fn() -> PyResult<()>,
    ) -> PyResult<Self> {
        let exit = recording_callback(check, || scope.getattr("__exit__"))?;
        Ok(Self {
            py,
            exit,
            entered: false,
        })
    }

    fn finish(&mut self, error: Option<&PyErr>) -> PyResult<()> {
        if !std::mem::replace(&mut self.entered, false) {
            return Ok(());
        }
        let result = if let Some(error) = error {
            self.exit.call1((
                error.get_type(self.py),
                error.value(self.py),
                error.traceback(self.py),
            ))?
        } else {
            self.exit
                .call1((self.py.None(), self.py.None(), self.py.None()))?
        };
        if !result.is_none()
            && (!result.is_exact_instance_of::<PyBool>() || result.extract::<bool>()?)
        {
            return Err(invalid(
                "recording memory scope must not suppress producer or capture failure",
            ));
        }
        Ok(())
    }
}

impl Drop for PreparedMemoryScope<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.finish(None) {
            error.write_unraisable(self.py, Some(&self.exit));
        }
    }
}

impl From<Py<PySemanticTransitionSession>> for PublishedExportOwner {
    fn from(session: Py<PySemanticTransitionSession>) -> Self {
        Self::Session { _session: session }
    }
}

impl From<Py<PySemanticPublishedParent>> for PublishedExportOwner {
    fn from(parent: Py<PySemanticPublishedParent>) -> Self {
        Self::Parent { _parent: parent }
    }
}

impl From<Py<PySemanticPreparedStep>> for PublishedExportOwner {
    fn from(step: Py<PySemanticPreparedStep>) -> Self {
        Self::Prepared { _step: step }
    }
}

type DeferredExportOwner = (ThreadId, PublishedExportOwner);
static DEFERRED_EXPORT_OWNERS: OnceLock<Mutex<Vec<DeferredExportOwner>>> = OnceLock::new();

fn drain_export_owners() {
    let Some(queue) = DEFERRED_EXPORT_OWNERS.get() else {
        return;
    };
    let thread = std::thread::current().id();
    let owners = {
        let mut queue = queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut owners = Vec::new();
        let mut index = 0;
        while index < queue.len() {
            if queue[index].0 == thread {
                owners.push(queue.swap_remove(index).1);
            } else {
                index += 1;
            }
        }
        owners
    };
    // Drop outside the queue and Session mutexes: a final Python decref may
    // recursively destroy other exports. PyO3 attaches on this same OS thread.
    Python::attach(|_| drop(owners));
}

struct PublishedExportContext {
    tensor: DlpackManagedTensor,
    owner: PublishedExportOwner,
    owner_thread: ThreadId,
}

unsafe extern "C" fn published_export_deleter(pointer: *mut xlog_cuda::dlpack::DLManagedTensor) {
    if pointer.is_null() {
        return;
    }
    // SAFETY: both allocations were created by retain_export_owner below, and
    // the single-consumer DLPack contract calls this deleter exactly once.
    let managed = unsafe { Box::from_raw(pointer) };
    let context = unsafe { Box::from_raw(managed.manager_ctx.cast::<PublishedExportContext>()) };
    let PublishedExportContext {
        tensor,
        owner,
        owner_thread,
    } = *context;
    // This drops the real native allocation/reader guard, whose deleter is
    // explicitly thread-safe. It does not release the Session's device lease.
    drop(tensor);
    DEFERRED_EXPORT_OWNERS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push((owner_thread, owner));
}

fn retain_export_owner(
    tensor: DlpackManagedTensor,
    owner: impl Into<PublishedExportOwner>,
    owner_thread: ThreadId,
) -> PyResult<DlpackManagedTensor> {
    if tensor.as_ptr().is_null() {
        return Err(invalid("native publication returned a null tensor owner"));
    }
    // SAFETY: metadata borrows the retained native tensor for the complete
    // lifetime of the wrapper. This is an ownership wrapper, not a new storage
    // allocation, numerical copy, or replacement tensor leaf.
    let source = unsafe { &(*tensor.as_ptr()).dl_tensor };
    let metadata = xlog_cuda::dlpack::DLTensor {
        data: source.data,
        device: source.device,
        ndim: source.ndim,
        dtype: source.dtype,
        shape: source.shape,
        strides: source.strides,
        byte_offset: source.byte_offset,
    };
    let context = Box::new(PublishedExportContext {
        tensor,
        owner: owner.into(),
        owner_thread,
    });
    let managed = Box::new(xlog_cuda::dlpack::DLManagedTensor {
        dl_tensor: metadata,
        manager_ctx: Box::into_raw(context).cast(),
        deleter: Some(published_export_deleter),
    });
    // SAFETY: the wrapper owns metadata through its real producer and its
    // deleter transfers Python ownership without off-thread decref or access.
    Ok(unsafe { DlpackManagedTensor::from_raw(Box::into_raw(managed)) })
}

impl ColdValue {
    /// Length-delimited metadata and byte-payload custody, never a grant. Byte
    /// payloads bind their exact length and SHA256 while retained separately. The
    /// parser's exact builtin integer spelling preserves arbitrary-width values.
    /// Lists and tuples deliberately share the canonical sequence representation.
    fn canonical_bytes(&self) -> Vec<u8> {
        fn append(value: &ColdValue, bytes: &mut Vec<u8>) {
            match value {
                ColdValue::None => bytes.push(0),
                ColdValue::Bool(value) => bytes.extend_from_slice(&[1, u8::from(*value)]),
                ColdValue::Integer(text) | ColdValue::Text(text) => {
                    bytes.push(if matches!(value, ColdValue::Integer(_)) {
                        2
                    } else {
                        3
                    });
                    bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
                    bytes.extend_from_slice(text.as_bytes());
                }
                ColdValue::Sequence(values) => {
                    bytes.push(4);
                    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
                    for value in values {
                        append(value, bytes);
                    }
                }
                ColdValue::Bytes(value) => {
                    bytes.push(5);
                    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
                    bytes.extend_from_slice(&Sha256::digest(value));
                }
            }
        }
        let mut bytes = b"xlog.cold.value.v1\0".to_vec();
        append(self, &mut bytes);
        bytes
    }

    fn python_value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match self {
            Self::None => py.None(),
            Self::Bool(item) => item.into_pyobject(py)?.to_owned().into_any().unbind(),
            Self::Integer(item) => py.get_type::<PyInt>().call1((item,))?.unbind(),
            Self::Text(item) => PyString::new(py, item).into_any().unbind(),
            Self::Bytes(item) => PyBytes::new(py, item).into_any().unbind(),
            Self::Sequence(items) => PyTuple::new(
                py,
                items
                    .iter()
                    .map(|item| item.python_value(py))
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any()
            .unbind(),
        })
    }

    /// Decode archived metadata in the existing format, not byte-payload hashes.
    /// This reconstructs custody data only; the authority parser and native
    /// publication verification must still validate its meaning and issuer.
    fn from_canonical_bytes(bytes: &[u8]) -> PyResult<Self> {
        fn take<'a>(remaining: &mut &'a [u8], count: usize) -> PyResult<&'a [u8]> {
            if count > remaining.len() {
                return Err(invalid("truncated cold metadata encoding"));
            }
            let (value, rest) = remaining.split_at(count);
            *remaining = rest;
            Ok(value)
        }

        fn length(remaining: &mut &[u8]) -> PyResult<usize> {
            let encoded: [u8; 8] = take(remaining, 8)?.try_into().unwrap();
            usize::try_from(u64::from_le_bytes(encoded))
                .map_err(|_| invalid("cold metadata length does not fit this host"))
        }

        fn decode(remaining: &mut &[u8], budget: &mut usize, depth: usize) -> PyResult<ColdValue> {
            if depth > 32 || *budget == 0 {
                return Err(invalid(
                    "cold task transport exceeds its depth or 16 MiB budget",
                ));
            }
            *budget -= 1;
            let tag = take(remaining, 1)?[0];
            match tag {
                0 => Ok(ColdValue::None),
                1 => match take(remaining, 1)?[0] {
                    0 => Ok(ColdValue::Bool(false)),
                    1 => Ok(ColdValue::Bool(true)),
                    _ => Err(invalid("cold metadata boolean is not canonical")),
                },
                2 | 3 => {
                    let count = length(remaining)?;
                    *budget = budget
                        .checked_sub(count)
                        .ok_or_else(|| invalid("cold task transport exceeds its 16 MiB budget"))?;
                    let text = std::str::from_utf8(take(remaining, count)?)
                        .map_err(|_| invalid("cold metadata text is not UTF-8"))?;
                    if tag == 2 {
                        let digits = text.strip_prefix('-').unwrap_or(text).as_bytes();
                        let nonzero = matches!(digits.first(), Some(b'1'..=b'9'))
                            && digits.iter().all(u8::is_ascii_digit);
                        if text != "0" && !nonzero {
                            return Err(invalid("cold metadata integer is not canonical decimal"));
                        }
                        Ok(ColdValue::Integer(text.to_owned()))
                    } else {
                        Ok(ColdValue::Text(text.to_owned()))
                    }
                }
                4 => {
                    let count = length(remaining)?;
                    // Every child consumes at least one node and one tag byte.
                    // Check both before allocating from the declared count.
                    if count > *budget || count > remaining.len() {
                        return Err(invalid(
                            "cold metadata sequence exceeds its input or budget",
                        ));
                    }
                    let mut values = Vec::with_capacity(count);
                    for _ in 0..count {
                        values.push(decode(remaining, budget, depth + 1)?);
                    }
                    Ok(ColdValue::Sequence(values))
                }
                _ => Err(invalid(
                    "cold metadata encoding contains an unsupported tag",
                )),
            }
        }

        let mut remaining = bytes
            .strip_prefix(b"xlog.cold.value.v1\0")
            .ok_or_else(|| invalid("cold metadata encoding has an incorrect domain"))?;
        let value = decode(&mut remaining, &mut (16 * 1024 * 1024), 0)?;
        if !remaining.is_empty() || value.canonical_bytes() != bytes {
            return Err(invalid(
                "cold metadata encoding is not exact canonical metadata",
            ));
        }
        Ok(value)
    }

    fn read(value: &Bound<'_, PyAny>, budget: &mut usize, depth: usize) -> PyResult<Self> {
        if depth > 32 || *budget == 0 {
            return Err(invalid(
                "cold task transport exceeds its depth or 16 MiB budget",
            ));
        }
        *budget -= 1;
        if value.is_none() {
            return Ok(Self::None);
        }
        if value.is_exact_instance_of::<PyBool>() {
            return Ok(Self::Bool(value.extract()?));
        }
        let scalar = if value.is_exact_instance_of::<PyString>() {
            Some((false, value.extract::<String>()?))
        } else if value.is_exact_instance_of::<PyInt>() {
            // Exact builtin int only: no user conversion or subclass handler.
            Some((true, value.str()?.to_str()?.to_owned()))
        } else {
            None
        };
        if let Some((integer, text)) = scalar {
            *budget = budget
                .checked_sub(text.len())
                .ok_or_else(|| invalid("cold task transport exceeds its 16 MiB budget"))?;
            return Ok(if integer {
                Self::Integer(text)
            } else {
                Self::Text(text)
            });
        }
        let elements = if value.is_exact_instance_of::<PyTuple>() {
            let tuple = value.cast::<PyTuple>()?;
            if tuple.len() > *budget {
                return Err(invalid("cold task transport exceeds its 16 MiB budget"));
            }
            tuple.iter().collect::<Vec<_>>()
        } else if value.is_exact_instance_of::<PyList>() {
            let list = value.cast::<PyList>()?;
            if list.len() > *budget {
                return Err(invalid("cold task transport exceeds its 16 MiB budget"));
            }
            list.iter().collect::<Vec<_>>()
        } else {
            return Err(PyTypeError::new_err(
                "cold task transport accepts only exact builtin str, int, bool, None, tuple and list",
            ));
        };
        Ok(Self::Sequence(
            elements
                .iter()
                .map(|item| Self::read(item, budget, depth + 1))
                .collect::<PyResult<_>>()?,
        ))
    }

    fn sequence(&self) -> PyResult<&[Self]> {
        match self {
            Self::Sequence(items) => Ok(items),
            _ => Err(invalid("expected a cold sequence")),
        }
    }

    fn fields(&self, count: usize) -> PyResult<&[Self]> {
        let items = self.sequence()?;
        if items.len() != count {
            return Err(invalid("cold record has an incorrect field count"));
        }
        Ok(items)
    }

    fn text(&self) -> PyResult<&str> {
        match self {
            Self::Text(value) if !value.is_empty() => Ok(value),
            _ => Err(invalid("expected a nonempty exact string")),
        }
    }

    fn unsigned(&self) -> PyResult<u64> {
        match self {
            Self::Integer(value) => value.parse().map_err(|_| invalid("expected a u64 integer")),
            _ => Err(invalid("expected an exact integer, excluding bool")),
        }
    }

    fn signed(&self) -> PyResult<i64> {
        match self {
            Self::Integer(value) => value
                .parse()
                .map_err(|_| invalid("expected an i64 integer")),
            _ => Err(invalid("expected an exact integer, excluding bool")),
        }
    }

    fn boolean(&self) -> PyResult<bool> {
        match self {
            Self::Bool(value) => Ok(*value),
            _ => Err(invalid("expected an exact bool")),
        }
    }

    fn strings(&self) -> PyResult<Vec<String>> {
        self.sequence()?
            .iter()
            .map(|item| Ok(item.text()?.to_owned()))
            .collect()
    }
}

fn unique(values: &[String]) -> PyResult<()> {
    if values.iter().collect::<BTreeSet<_>>().len() != values.len() {
        return Err(invalid("cold reference sequence contains duplicates"));
    }
    Ok(())
}

fn parse_text_slots(value: &ColdValue) -> PyResult<Vec<SemanticTextSlot>> {
    value
        .sequence()?
        .iter()
        .map(|row| {
            let fields = row.fields(8)?;
            Ok(SemanticTextSlot {
                token: fields[0].unsigned()?,
                logical_position: fields[1].unsigned()?,
                kind: fields[2].unsigned()?,
                provenance: fields[3].unsigned()?,
                valid: u64::from(fields[4].boolean()?),
                committed: u64::from(fields[5].boolean()?),
                recomputed: u64::from(fields[6].boolean()?),
                provenance_record: fields[7].unsigned()?,
            })
        })
        .collect()
}

fn unsigned_array<const N: usize>(value: &ColdValue) -> PyResult<[u64; N]> {
    value
        .fields(N)?
        .iter()
        .map(ColdValue::unsigned)
        .collect::<PyResult<Vec<_>>>()?
        .try_into()
        .map_err(|_| invalid("incorrect fixed-array length"))
}

fn identity_bytes(value: &Bound<'_, PyAny>) -> PyResult<Identity256> {
    if !value.is_exact_instance_of::<PyBytes>() {
        return Err(invalid("native identity requires exact builtin bytes"));
    }
    let bytes: [u8; 32] = value
        .cast::<PyBytes>()?
        .as_bytes()
        .try_into()
        .map_err(|_| invalid("native identity must contain exactly 32 bytes"))?;
    Ok(Identity256::from_bytes(bytes))
}

fn object_sequence<'py>(
    value: &Bound<'py, PyAny>,
    budget: &mut usize,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let items = if value.is_exact_instance_of::<PyTuple>() {
        let values = value.cast::<PyTuple>()?;
        if values.len() > *budget {
            return Err(invalid("cold parent transport exceeds its input bound"));
        }
        values.iter().collect::<Vec<_>>()
    } else if value.is_exact_instance_of::<PyList>() {
        let values = value.cast::<PyList>()?;
        if values.len() > *budget {
            return Err(invalid("cold parent transport exceeds its input bound"));
        }
        values.iter().collect::<Vec<_>>()
    } else {
        return Err(invalid(
            "cold parent records require an exact tuple or list",
        ));
    };
    *budget -= items.len();
    Ok(items)
}

fn parse_tensor_layout(value: &ColdValue) -> PyResult<SemanticTensorLayout> {
    let fields = value.fields(8)?;
    Ok(SemanticTensorLayout {
        role: fields[0].unsigned()?,
        index: fields[1].unsigned()?,
        element_bytes: fields[2].unsigned()?,
        scalar_type: fields[3].unsigned()?,
        rank: fields[4].unsigned()?,
        logical_axis: fields[5].unsigned()?,
        dimensions: unsigned_array(&fields[6])?,
        strides_bytes: unsigned_array(&fields[7])?,
    })
}

fn parse_content_ranges(value: &ColdValue) -> PyResult<Vec<(SemanticStateRole, u64)>> {
    let rows = value.sequence()?;
    if rows.is_empty() {
        return Err(invalid(
            "content guard requires at least one published range",
        ));
    }
    let mut seen = BTreeSet::new();
    rows.iter()
        .map(|row| {
            let fields = row.fields(2)?;
            let code = fields[0].unsigned()?;
            let role = SemanticStateRole::from_code(code)
                .ok_or_else(|| invalid("unknown native publication content role"))?;
            let index = fields[1].unsigned()?;
            if !seen.insert((code, index)) {
                return Err(invalid("content guard repeats a published range"));
            }
            Ok((role, index))
        })
        .collect()
}

// Before native admission joins producer streams, any failed Python handoff
// must retain previously consumed capsules. Guessing producer completion on an
// exception or a reentrant authority change would permit use-after-free.
static FAILED_TENSOR_HANDOFFS: OnceLock<Mutex<Vec<Box<dyn Send>>>> = OnceLock::new();
struct TensorHandoff<T: Send + 'static = SemanticTensorInput>(Vec<T>);
impl<T: Send + 'static> TensorHandoff<T> {
    fn into_native(mut self) -> Vec<T> {
        std::mem::take(&mut self.0)
    }
}
impl<T: Send + 'static> Drop for TensorHandoff<T> {
    fn drop(&mut self) {
        if !self.0.is_empty() {
            FAILED_TENSOR_HANDOFFS
                .get_or_init(|| Mutex::new(Vec::new()))
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(Box::new(std::mem::take(&mut self.0)));
        }
    }
}

// Callers destructure before taking Session/state locks, so original Python
// producers cannot run their finalizers while those mutexes are held.
struct ParsedTensorInputs {
    handoff: TensorHandoff,
    producers: Vec<Py<PyAny>>,
}

fn parse_model_storages(
    value: &Bound<'_, PyAny>,
    budget: &mut usize,
) -> PyResult<Vec<SemanticModelStorage>> {
    ColdValue::read(value, budget, 0)?
        .sequence()?
        .iter()
        .map(|row| {
            let fields = row.fields(3)?;
            Ok(SemanticModelStorage {
                allocation: fields[0].unsigned()?,
                byte_offset: fields[1].unsigned()?,
                span_bytes: fields[2].unsigned()?,
            })
        })
        .collect()
}

fn parse_model_views(
    value: &Bound<'_, PyAny>,
    budget: &mut usize,
) -> PyResult<Vec<SemanticModelView>> {
    ColdValue::read(value, budget, 0)?
        .sequence()?
        .iter()
        .map(|row| {
            let fields = row.fields(4)?;
            Ok(SemanticModelView {
                role: fields[0].unsigned()?,
                index: fields[1].unsigned()?,
                storage: fields[2].unsigned()?,
                byte_offset: fields[3].unsigned()?,
            })
        })
        .collect()
}

fn parse_tensor_inputs(
    value: &Bound<'_, PyAny>,
    budget: &mut usize,
    expected_device: usize,
) -> PyResult<ParsedTensorInputs> {
    parse_tensor_inputs_guarded(value, budget, expected_device, 1, &|| Ok(()))
}

fn parse_tensor_inputs_guarded(
    value: &Bound<'_, PyAny>,
    budget: &mut usize,
    expected_device: usize,
    consumer_stream: u64,
    check: &dyn Fn() -> PyResult<()>,
) -> PyResult<ParsedTensorInputs> {
    let consumer_stream = i64::try_from(consumer_stream)
        .map_err(|_| invalid("consumer stream exceeds DLPack address space"))?;
    let mut pending = Vec::new();
    // Check every cold row before invoking even the first producer.
    for row in object_sequence(value, budget)? {
        let fields = object_sequence(&row, budget)?;
        if fields.len() != 5 {
            return Err(invalid(
                "model tensor input needs layout, interval start/end, its real producer and native allocation or None",
            ));
        }
        let layout = parse_tensor_layout(&ColdValue::read(&fields[0], budget, 0)?)?;
        let logical_begin = ColdValue::read(&fields[1], budget, 0)?.unsigned()?;
        let logical_end = ColdValue::read(&fields[2], budget, 0)?.unsigned()?;
        if logical_end < logical_begin {
            return Err(invalid("model tensor interval is reversed"));
        }
        let native_allocation = if fields[4].is_none() {
            None
        } else {
            Some(
                fields[4]
                    .extract::<PyRef<'_, PyNativeTensorAllocation>>()?
                    .provenance
                    .clone(),
            )
        };
        pending.push((
            layout,
            logical_begin,
            logical_end,
            fields[3].clone(),
            native_allocation,
        ));
    }
    // Keep the actual producers independently of mutable caller-owned rows.
    // Their original autograd objects are distinct from DLPack storage aliases.
    for (_, _, _, producer, _) in &pending {
        validate_producer_device_guarded(producer, expected_device, check)?;
    }
    let producers = TensorHandoff(
        pending
            .iter()
            .map(|(_, _, _, producer, _)| producer.clone().unbind())
            .collect(),
    );
    let mut handoff = TensorHandoff(Vec::with_capacity(pending.len()));
    for (layout, logical_begin, logical_end, producer, native_allocation) in pending {
        // Preserve the original exception without invoking its Python __str__
        // after a producer callback has already refused authority.
        let tensor = crate::dlpack_from_py_for_stream_guarded(&producer, consumer_stream, check)?;
        handoff.0.push(SemanticTensorInput {
            layout,
            logical_begin,
            logical_end,
            tensor,
            native_allocation,
        });
    }
    Ok(ParsedTensorInputs {
        handoff,
        producers: producers.into_native(),
    })
}

#[cfg(test)]
fn validate_continuation_producer(
    producer: &Bound<'_, PyAny>,
    expected_device: usize,
) -> PyResult<()> {
    validate_continuation_producer_guarded(producer, expected_device, &|| Ok(()))
}

fn validate_continuation_producer_guarded(
    producer: &Bound<'_, PyAny>,
    expected_device: usize,
    check: &dyn Fn() -> PyResult<()>,
) -> PyResult<()> {
    if !recording_callback(check, || producer.hasattr("__dlpack__"))?
        || !recording_callback(check, || producer.hasattr("__dlpack_device__"))?
    {
        return Err(invalid(
            "continuation services require original CUDA DLPack producers",
        ));
    }
    validate_producer_device_guarded(producer, expected_device, check)
}

fn validate_producer_device_guarded(
    producer: &Bound<'_, PyAny>,
    expected_device: usize,
    check: &dyn Fn() -> PyResult<()>,
) -> PyResult<()> {
    // Reject known foreign-device producers before consuming any capsule.
    // Native admission still validates raw capsules and retains uncertain owners.
    if recording_callback(check, || producer.hasattr("__dlpack__"))? {
        let method = recording_callback(check, || producer.getattr("__dlpack_device__"))?;
        let device = recording_callback(check, || method.call0())?;
        let (_, device_id) = crate::dlpack_device_pair(&device)?;
        if usize::try_from(device_id).ok() != Some(expected_device) {
            return Err(invalid("tensor producer belongs to another device"));
        }
    }
    Ok(())
}

fn parse_state_records(
    value: &Bound<'_, PyAny>,
    budget: &mut usize,
) -> PyResult<Vec<SemanticStateRecord>> {
    object_sequence(value, budget)?.iter().map(|row| {
        let fields = object_sequence(row, budget)?;
        if fields.len() != 4 { return Err(invalid("state record needs role, index, capacity and actual bytes")); }
        let role = SemanticStateRole::from_code(ColdValue::read(&fields[0], budget, 0)?.unsigned()?)
            .ok_or_else(|| invalid("unknown native state role"))?;
        if matches!(role, SemanticStateRole::RuntimeContract | SemanticStateRole::PrefixIdentity | SemanticStateRole::TokenProvenanceRecords | SemanticStateRole::FeedbackSupportProvenance | SemanticStateRole::AuthorityEnvelope | SemanticStateRole::AuthorityDecisions | SemanticStateRole::IntentEntries | SemanticStateRole::IntentPayload) {
            return Err(invalid("native-owned runtime contract, prefix identity, authority, dependency and intent records cannot be supplied by the caller"));
        }
        let index = ColdValue::read(&fields[1], budget, 0)?.unsigned()?;
        let capacity_bytes = usize::try_from(ColdValue::read(&fields[2], budget, 0)?.unsigned()?)
            .map_err(|_| invalid("state-record capacity overflows"))?;
        if !fields[3].is_exact_instance_of::<PyBytes>() {
            return Err(invalid("state payload requires exact builtin bytes"));
        }
        let bytes = fields[3].cast::<PyBytes>()?.as_bytes();
        *budget = budget.checked_sub(bytes.len()).ok_or_else(|| invalid("cold state records exceed their input bound"))?;
        if bytes.len() > capacity_bytes { return Err(invalid("state payload exceeds declared capacity")); }
        Ok(SemanticStateRecord { role, index, bytes: bytes.to_vec(), capacity_bytes })
    }).collect()
}

fn metadata_item<'py>(metadata: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    metadata
        .get_item(name)?
        .ok_or_else(|| invalid("missing required parent metadata field"))
}

fn metadata_value(
    metadata: &Bound<'_, PyDict>,
    name: &str,
    budget: &mut usize,
) -> PyResult<ColdValue> {
    ColdValue::read(&metadata_item(metadata, name)?, budget, 0)
}

fn metadata_integer(metadata: &Bound<'_, PyDict>, name: &str, budget: &mut usize) -> PyResult<u64> {
    metadata_value(metadata, name, budget)?.unsigned()
}

fn parse_parent(
    metadata: &Bound<'_, PyAny>,
    source: &Bound<'_, PyAny>,
    prefix: &Bound<'_, PyAny>,
    records: &Bound<'_, PyAny>,
    task_use: &PySemanticTransitionTaskUse,
) -> PyResult<SemanticParentBinding> {
    const KEYS: &[&str] = &[
        "recovered_instance",
        "ring_head",
        "provenance_capacity_records",
        "prefix_capacity",
        "feedback_capacity",
        "max_position",
        "pad_token",
        "terminal_tokens",
        "final_intent_payload_bytes",
        "intent_effect",
        "intent_entry_capacity",
        "intent_payload_capacity_bytes",
        "model_generation",
        "model_numerical_mode",
        "model_contract_layout",
        "policy_generation",
        "neural_generation",
        "cache_generation",
        "training_cursor",
        "training_rng",
        "fuel",
        "rng",
        "topology_identity",
        "table_identity",
        "role_counts",
        "active_layouts",
        "authority_decisions_capacity_bytes",
    ];
    if !metadata.is_exact_instance_of::<PyDict>() {
        return Err(invalid("parent metadata requires an exact builtin dict"));
    }
    let metadata = metadata.cast::<PyDict>()?;
    if metadata.len() != KEYS.len() {
        return Err(invalid("parent metadata has missing or additional fields"));
    }
    for key in metadata.keys().iter() {
        if !key.is_exact_instance_of::<PyString>()
            || !KEYS.contains(&key.cast::<PyString>()?.to_str()?)
        {
            return Err(invalid("unknown parent metadata field"));
        }
    }
    let mut budget = 16 * 1024 * 1024;
    let source = parse_text_slots(&ColdValue::read(source, &mut budget, 0)?)?
        .try_into()
        .map_err(|_| invalid("parent active source requires exactly 32 original ring slots"))?;
    let prefix = parse_text_slots(&ColdValue::read(prefix, &mut budget, 0)?)?;
    let recovered = metadata_item(metadata, "recovered_instance")?;
    let recovered_instance = if recovered.is_none() {
        None
    } else {
        return Err(invalid(
            "fresh parent initialization cannot discard recovered history",
        ));
    };
    let effect = metadata_item(metadata, "intent_effect")?;
    if !effect.is_exact_instance_of::<PyBytes>() {
        return Err(invalid(
            "intent effect requires the actual application descriptor bytes",
        ));
    }
    let intent_effect = effect.cast::<PyBytes>()?.as_bytes();
    budget = budget
        .checked_sub(intent_effect.len())
        .ok_or_else(|| invalid("cold state records exceed their input bound"))?;
    if intent_effect.is_empty() {
        return Err(invalid("intent effect descriptor is empty"));
    }
    let role_counts: [u64; 55] =
        unsigned_array(&metadata_value(metadata, "role_counts", &mut budget)?)?;
    let mut records = parse_state_records(records, &mut budget)?;
    let decision_capacity = usize::try_from(metadata_integer(
        metadata,
        "authority_decisions_capacity_bytes",
        &mut budget,
    )?)
    .map_err(|_| invalid("authority decision capacity overflows"))?;
    let state = task_use.state()?;
    for (role, bytes) in [
        (
            SemanticStateRole::FeedbackSupportProvenance,
            &task_use.authority.canonical,
        ),
        (
            SemanticStateRole::AuthorityEnvelope,
            &task_use.authority.canonical,
        ),
        (
            SemanticStateRole::AuthorityDecisions,
            &state.snapshot.canonical,
        ),
    ] {
        if role_counts[role as usize - 1] != 1 {
            return Err(invalid(
                "parent must retain each complete controller record exactly once",
            ));
        }
        let capacity_bytes = if role == SemanticStateRole::AuthorityDecisions {
            decision_capacity
        } else {
            bytes.len()
        };
        if capacity_bytes < bytes.len() {
            return Err(invalid(
                "authority decision capacity is smaller than the current complete snapshot",
            ));
        }
        records.push(SemanticStateRecord {
            role,
            index: 0,
            capacity_bytes,
            bytes: bytes.clone(),
        });
    }
    drop(state);
    let rng_fields = unsigned_array::<4>(&metadata_value(metadata, "rng", &mut budget)?)?;
    let rng = SemanticRngBinding {
        model_generation: u32::try_from(rng_fields[0])
            .map_err(|_| invalid("RNG model generation exceeds u32"))?,
        stream_serial: rng_fields[1],
        family_id: u8::try_from(rng_fields[2]).map_err(|_| invalid("RNG family exceeds u8"))?,
        proposal: u32::try_from(rng_fields[3]).map_err(|_| invalid("RNG proposal exceeds u32"))?,
    };
    let terminal_tokens = metadata_value(metadata, "terminal_tokens", &mut budget)?
        .sequence()?
        .iter()
        .map(ColdValue::unsigned)
        .collect::<PyResult<Vec<_>>>()?;
    let active_layouts = metadata_value(metadata, "active_layouts", &mut budget)?
        .sequence()?
        .iter()
        .map(parse_tensor_layout)
        .collect::<PyResult<Vec<_>>>()?;
    let model_numerical_mode =
        metadata_value(metadata, "model_numerical_mode", &mut budget)?.canonical_bytes();
    // The producer owns numerical semantics and format versions. XLOG retains
    // the complete canonical contribution, and requires lossless cold restore
    // before admitting it; a payload hash is not reconstructible metadata.
    ColdValue::from_canonical_bytes(&model_numerical_mode)?;
    let [schema_begin, schema_bytes, schema_digest_offset, generation_offset, numerical_digest_offset, identity_offset] =
        unsigned_array::<6>(&metadata_value(
            metadata,
            "model_contract_layout",
            &mut budget,
        )?)?;
    let model_contract_layout = SemanticModelContractLayout {
        schema_begin,
        schema_bytes,
        schema_digest_offset,
        generation_offset,
        numerical_digest_offset,
        identity_offset,
    };
    let mut parent = SemanticParentBinding {
        recovered_instance,
        source,
        prefix,
        ring_head: metadata_integer(metadata, "ring_head", &mut budget)?,
        provenance_records: 0,
        prefix_capacity: metadata_integer(metadata, "prefix_capacity", &mut budget)?,
        feedback_capacity: metadata_integer(metadata, "feedback_capacity", &mut budget)?,
        max_position: metadata_integer(metadata, "max_position", &mut budget)?,
        pad_token: metadata_integer(metadata, "pad_token", &mut budget)?,
        terminal_tokens,
        final_intent_payload_bytes: metadata_integer(
            metadata,
            "final_intent_payload_bytes",
            &mut budget,
        )?,
        intent_effect: intent_effect.to_vec(),
        intent_entry_capacity: metadata_integer(metadata, "intent_entry_capacity", &mut budget)?,
        intent_payload_capacity_bytes: usize::try_from(metadata_integer(
            metadata,
            "intent_payload_capacity_bytes",
            &mut budget,
        )?)
        .map_err(|_| invalid("intent payload capacity overflows"))?,
        model_generation: metadata_integer(metadata, "model_generation", &mut budget)?,
        model_numerical_mode,
        model_contract_layout,
        policy_generation: metadata_integer(metadata, "policy_generation", &mut budget)?,
        neural_generation: metadata_integer(metadata, "neural_generation", &mut budget)?,
        cache_generation: metadata_integer(metadata, "cache_generation", &mut budget)?,
        authority_generation: task_use.task_epoch,
        training_cursor: metadata_integer(metadata, "training_cursor", &mut budget)?,
        training_rng: unsigned_array(&metadata_value(metadata, "training_rng", &mut budget)?)?,
        fuel: metadata_integer(metadata, "fuel", &mut budget)?,
        rng,
        topology_identity: identity_bytes(&metadata_item(metadata, "topology_identity")?)?,
        table_identity: identity_bytes(&metadata_item(metadata, "table_identity")?)?,
        role_counts,
        records,
        tensors: Vec::new(),
        model_memory: SemanticModelMemory {
            allocations: Vec::new(),
            storages: Vec::new(),
            views: Vec::new(),
        },
        active_layouts,
    };
    parent
        .bind_initial_sources(
            &task_use.authority.initial_sources,
            &task_use.authority.source_mapping,
            &task_use.authority.canonical,
            metadata_integer(metadata, "provenance_capacity_records", &mut budget)?,
        )
        .map_err(xlog_err)?;
    // Only after all scalar/control input checks may a real producer callback
    // execute. The caller rechecks the unchanged task and phase after handoff.
    Ok(parent)
}

fn sorted_references(value: &ColdValue) -> PyResult<Vec<String>> {
    let values = value.strings()?;
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(invalid("snapshot references must be sorted and unique"));
    }
    Ok(values)
}

fn checked_digest(value: &ColdValue) -> PyResult<()> {
    let text = value.text()?;
    if text.len() != 64
        || !text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid("expected a lowercase SHA-256 identity"));
    }
    Ok(())
}

/// The trusted application supplies the original ISO spelling and the exact UTC
/// microsecond projection produced by its existing timezone-aware datetime parser.
/// This preserves spelling without introducing a second datetime grammar here.
#[derive(Clone, Debug, PartialEq, Eq)]
struct AuthorityInstant {
    spelling: String,
    micros: i64,
}

impl AuthorityInstant {
    fn parse(value: &ColdValue) -> PyResult<Self> {
        let fields = value.fields(2)?;
        Ok(Self {
            spelling: fields[0].text()?.to_owned(),
            micros: fields[1].signed()?,
        })
    }
}

#[derive(Clone, Debug)]
struct LiveAuthority {
    canonical: ColdValue,
    retention_micros: i64,
}

fn validate_live(value: &ColdValue, scope: &[String]) -> PyResult<()> {
    let fields = value.fields(16)?;
    if fields[0].text()? != "live" {
        return Err(invalid(
            "only the explicit live authority branch is accepted",
        ));
    }
    for index in [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 15] {
        fields[index].text()?;
    }
    for (index, expected) in [4, 5, 6, 9, 10].into_iter().zip(scope) {
        if fields[index].text()? != expected {
            return Err(invalid(
                "live envelope does not match the pinned task scope",
            ));
        }
    }
    let purposes = sorted_references(&fields[8])?;
    if purposes.iter().any(|purpose| {
        !matches!(
            purpose.as_str(),
            "current-request-fast-adaptation" | "durable-slow-consolidation"
        )
    }) {
        return Err(invalid("live envelope has an invalid learning purpose"));
    }
    fields[13].boolean()?;
    fields[14].boolean()?;
    Ok(())
}

#[derive(Clone, Debug)]
struct AuthoritySnapshot {
    canonical: Vec<u8>,
    revision: u64,
    observed_at: AuthorityInstant,
    segment_end: AuthorityInstant,
    live: Vec<(ColdValue, AuthorityInstant, Vec<String>)>,
    revoked_grants: Vec<String>,
}

impl AuthoritySnapshot {
    fn parse(value: &ColdValue) -> PyResult<Self> {
        let fields = value.fields(5)?;
        let live = fields[3]
            .sequence()?
            .iter()
            .map(|value| {
                let fields = value.fields(3)?;
                Ok((
                    fields[0].clone(),
                    AuthorityInstant::parse(&fields[1])?,
                    sorted_references(&fields[2])?,
                ))
            })
            .collect::<PyResult<_>>()?;
        Ok(Self {
            canonical: value.canonical_bytes(),
            revision: fields[0].unsigned()?,
            observed_at: AuthorityInstant::parse(&fields[1])?,
            segment_end: AuthorityInstant::parse(&fields[2])?,
            live,
            revoked_grants: sorted_references(&fields[4])?,
        })
    }

    fn newer_than(&self, previous: &Self) -> PyResult<()> {
        if self.revision <= previous.revision
            || self.observed_at.micros < previous.observed_at.micros
        {
            return Err(invalid(
                "authority snapshot is not fresh relative to this task use",
            ));
        }
        if self.segment_end != previous.segment_end
            || self.live.len() != previous.live.len()
            || self
                .live
                .iter()
                .zip(&previous.live)
                .any(|(current, prior)| current.1 != prior.1)
        {
            return Err(invalid(
                "authority refresh cannot change the preregistered segment boundary",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Dependency {
    identity: String,
    kind: String,
    data_parents: Vec<String>,
    control_parents: Vec<String>,
    live_envelopes: Vec<String>,
    target: Option<(usize, u64)>,
    native_record: Option<(String, u32)>,
}

#[derive(Clone, Debug)]
struct ResolvedGrant {
    reference: String,
    issuer_provenance: String,
    allowed: bool,
    task_ref: String,
    dependencies: Vec<String>,
    expiry: AuthorityInstant,
    revocation_ref: String,
    learning_purpose: Option<String>,
}

impl ResolvedGrant {
    fn parse(value: &ColdValue, training: bool) -> PyResult<Self> {
        let fields = value.fields(if training { 8 } else { 7 })?;
        let allowed = match fields[2].text()? {
            "allow" => true,
            "deny" => false,
            _ => return Err(invalid("grant decision must be explicitly allow or deny")),
        };
        let dependencies = fields[4].strings()?;
        unique(&dependencies)?;
        if dependencies.is_empty() {
            return Err(invalid(
                "grant must name an exact nonempty dependency scope",
            ));
        }
        let learning_purpose = if training {
            let purpose = fields[7].text()?;
            if !matches!(
                purpose,
                "current-request-fast-adaptation" | "durable-slow-consolidation"
            ) {
                return Err(invalid(
                    "training grant needs an explicit canonical learning purpose",
                ));
            }
            Some(purpose.to_owned())
        } else {
            None
        };
        Ok(Self {
            reference: fields[0].text()?.to_owned(),
            issuer_provenance: fields[1].text()?.to_owned(),
            allowed,
            task_ref: fields[3].text()?.to_owned(),
            dependencies,
            expiry: AuthorityInstant::parse(&fields[5])?,
            revocation_ref: fields[6].text()?.to_owned(),
            learning_purpose,
        })
    }
}

/// JSON's ASCII string representation used by the observed-source data format.
/// This encodes transformation identity strings without invoking Python handlers.
fn observation_json_string(value: &str) -> String {
    let mut output = String::from("\"");
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{8}' => output.push_str("\\b"),
            '\u{c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            ' '..='~' => output.push(character),
            _ => {
                for unit in character.encode_utf16(&mut [0; 2]) {
                    use std::fmt::Write;
                    write!(output, "\\u{unit:04x}").expect("String writes cannot fail");
                }
            }
        }
    }
    output.push('"');
    output
}

/// Read the one full replay carrier projection. Each material and evidence has
/// its own byte cap; the separate total counts each SHA-addressed material once.
/// All caps are checked before copying payloads, independently of metadata.
fn read_replay_rows(
    value: &Bound<'_, PyAny>,
    budget: &mut usize,
    max_material_bytes: usize,
    mut total_material_bytes: usize,
    max_evidence_bytes: usize,
) -> PyResult<ColdValue> {
    if max_material_bytes == 0 || total_material_bytes == 0 || max_evidence_bytes == 0 {
        return Err(invalid("replay kit byte caps must be positive integers"));
    }
    let mut rows = Vec::new();
    let mut unique_materials = BTreeMap::<String, Bound<'_, PyAny>>::new();
    for row in object_sequence(value, budget)? {
        let fields = object_sequence(&row, budget)?;
        if fields.len() != 5 {
            return Err(invalid(
                "replay row requires basis, identity, record line, materials and evidence",
            ));
        }
        let mut materials = Vec::new();
        for material in object_sequence(&fields[3], budget)? {
            let pair = object_sequence(&material, budget)?;
            if pair.len() != 2 {
                return Err(invalid(
                    "replay material requires its reference and bytes or None",
                ));
            }
            let reference = read_material_reference(&pair[0], budget)?;
            if !pair[1].is_none() {
                let bytes = exact_replay_bytes(&pair[1])?;
                if bytes.len() > max_material_bytes {
                    return Err(invalid("replay material exceeds max_material_bytes"));
                }
                let digest = &reference.fields(4)?[2];
                checked_digest(digest)?;
                let digest = digest.text()?;
                if replay_hash(bytes) != digest {
                    return Err(invalid("replay material byte digest differs"));
                }
                if let Some(previous) = unique_materials.get(digest) {
                    if exact_replay_bytes(previous)? != bytes {
                        return Err(invalid(
                            "shared replay material digest names different bytes",
                        ));
                    }
                } else {
                    total_material_bytes = total_material_bytes
                        .checked_sub(bytes.len())
                        .ok_or_else(|| {
                            invalid("replay materials exceed max_total_material_bytes")
                        })?;
                    unique_materials.insert(digest.to_owned(), pair[1].clone());
                }
            }
            materials.push((reference, pair[1].clone()));
        }
        let evidence = exact_replay_bytes(&fields[4])?;
        if evidence.len() > max_evidence_bytes {
            return Err(invalid("replay evidence exceeds max_evidence_bytes"));
        }
        rows.push((fields, materials));
    }
    let unique_materials = unique_materials
        .into_iter()
        .map(|(digest, value)| Ok((digest, Arc::<[u8]>::from(exact_replay_bytes(&value)?))))
        .collect::<PyResult<BTreeMap<_, _>>>()?;
    Ok(ColdValue::Sequence(
        rows.into_iter()
            .map(|(fields, materials)| {
                let materials = materials
                    .into_iter()
                    .map(|(reference, payload)| {
                        let bytes = if payload.is_none() {
                            ColdValue::None
                        } else {
                            let digest = reference.fields(4)?[2].text()?;
                            ColdValue::Bytes(Arc::clone(
                                unique_materials
                                    .get(digest)
                                    .ok_or_else(|| invalid("replay material payload is absent"))?,
                            ))
                        };
                        Ok(ColdValue::Sequence(vec![reference, bytes]))
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(ColdValue::Sequence(vec![
                    ColdValue::read(&fields[0], budget, 0)?,
                    ColdValue::read(&fields[1], budget, 0)?,
                    ColdValue::read(&fields[2], budget, 0)?,
                    ColdValue::Sequence(materials),
                    ColdValue::Bytes(Arc::from(exact_replay_bytes(&fields[4])?)),
                ]))
            })
            .collect::<PyResult<Vec<_>>>()?,
    ))
}

fn exact_replay_bytes<'a>(value: &'a Bound<'_, PyAny>) -> PyResult<&'a [u8]> {
    if !value.is_exact_instance_of::<PyBytes>() {
        return Err(PyTypeError::new_err(
            "replay payloads require exact builtin bytes",
        ));
    }
    Ok(value.cast::<PyBytes>()?.as_bytes())
}

fn replay_dictionary<'py>(
    value: &Bound<'py, PyAny>,
    keys: &[&str],
    budget: &mut usize,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if !value.is_exact_instance_of::<PyDict>() {
        return Err(PyTypeError::new_err(
            "replay references require exact builtin dictionaries",
        ));
    }
    let dictionary = value.cast::<PyDict>()?;
    if dictionary.len() != keys.len() {
        return Err(invalid(
            "replay reference dictionary has an incorrect key set",
        ));
    }
    for (key, _) in dictionary.iter() {
        if !key.is_exact_instance_of::<PyString>()
            || !keys.contains(&key.extract::<String>()?.as_str())
        {
            return Err(invalid(
                "replay reference dictionary has an incorrect key set",
            ));
        }
        ColdValue::read(&key, budget, 0)?;
    }
    keys.iter()
        .map(|key| {
            dictionary
                .get_item(key)?
                .ok_or_else(|| invalid("replay reference field is absent"))
        })
        .collect()
}

fn read_material_reference(value: &Bound<'_, PyAny>, budget: &mut usize) -> PyResult<ColdValue> {
    let fields = replay_dictionary(
        value,
        &["identity", "kind", "bytes_sha256", "reconstruction"],
        budget,
    )?;
    let reconstruction = if fields[3].is_none() {
        ColdValue::None
    } else {
        let fields = replay_dictionary(&fields[3], &["operation", "inputs"], budget)?;
        ColdValue::Sequence(
            fields
                .iter()
                .map(|value| ColdValue::read(value, budget, 0))
                .collect::<PyResult<_>>()?,
        )
    };
    Ok(ColdValue::Sequence(vec![
        ColdValue::read(&fields[0], budget, 0)?,
        ColdValue::read(&fields[1], budget, 0)?,
        ColdValue::read(&fields[2], budget, 0)?,
        reconstruction,
    ]))
}

fn replay_hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

type ReplayJsonObject = BTreeMap<String, Box<serde_json::value::RawValue>>;

/// Split a JSON value using serde's parser without normalizing its scalar bytes.
fn take_replay_json<'a>(remaining: &mut &'a str) -> PyResult<&'a str> {
    let input = *remaining;
    if input
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_whitespace)
    {
        return Err(invalid("replay JSON containers require compact separators"));
    }
    let mut values =
        serde_json::Deserializer::from_str(input).into_iter::<&serde_json::value::RawValue>();
    let value = values
        .next()
        .ok_or_else(|| invalid("replay JSON value is absent"))?
        .map_err(|_| invalid("replay JSON value is invalid"))?;
    *remaining = &input[values.byte_offset()..];
    Ok(value.get())
}

/// Python compares object keys by Unicode code points. Decode keys without
/// requiring Rust UTF-8 strings: isolated UTF-16 surrogate escapes are valid
/// data-owner keys, and paired escapes sort as their one non-BMP code point.
fn replay_json_key(raw: &str) -> PyResult<Vec<u32>> {
    let mut characters = raw[1..raw.len() - 1].chars();
    let mut units = Vec::new();
    while let Some(character) = characters.next() {
        if character != '\\' {
            units.push(character as u32);
            continue;
        }
        let escaped = characters
            .next()
            .ok_or_else(|| invalid("replay JSON key escape is incomplete"))?;
        units.push(match escaped {
            '"' | '\\' | '/' => escaped as u32,
            'b' => 8,
            'f' => 12,
            'n' => 10,
            'r' => 13,
            't' => 9,
            'u' => {
                let mut unit = 0;
                for _ in 0..4 {
                    unit = unit * 16
                        + characters
                            .next()
                            .and_then(|value| value.to_digit(16))
                            .ok_or_else(|| {
                                invalid("replay JSON key Unicode escape is incomplete")
                            })?;
                }
                unit
            }
            _ => return Err(invalid("replay JSON key escape is invalid")),
        });
    }
    let mut points = Vec::new();
    let mut position = 0;
    while position < units.len() {
        let unit = units[position];
        if (0xd800..=0xdbff).contains(&unit)
            && units
                .get(position + 1)
                .is_some_and(|next| (0xdc00..=0xdfff).contains(next))
        {
            points.push(0x10000 + ((unit - 0xd800) << 10) + units[position + 1] - 0xdc00);
            position += 2;
        } else {
            points.push(unit);
            position += 1;
        }
    }
    Ok(points)
}

/// Check every container's compact structure, sorted keys and uniqueness.
/// Scalar spelling and complete episode laws remain the upstream data owner's
/// responsibility; their original bytes are preserved and content-hashed here.
fn validate_replay_json_structure(raw: &str, depth: usize) -> PyResult<()> {
    if depth > 32 {
        return Err(invalid("replay JSON exceeds its nesting bound"));
    }
    let object = raw.starts_with('{');
    if !object && !raw.starts_with('[') {
        return Ok(());
    }
    let end = if object { "}" } else { "]" };
    let mut remaining = &raw[1..];
    if remaining == end {
        return Ok(());
    }
    let mut previous: Option<Vec<u32>> = None;
    loop {
        if object {
            if !remaining.starts_with('"') {
                return Err(invalid("replay JSON objects require compact string keys"));
            }
            let key = replay_json_key(take_replay_json(&mut remaining)?)?;
            if previous.as_ref().is_some_and(|value| value >= &key) {
                return Err(invalid("replay JSON object keys must be sorted and unique"));
            }
            previous = Some(key);
            remaining = remaining
                .strip_prefix(':')
                .ok_or_else(|| invalid("replay JSON objects require compact separators"))?;
        }
        validate_replay_json_structure(take_replay_json(&mut remaining)?, depth + 1)?;
        if remaining == end {
            return Ok(());
        }
        remaining = remaining
            .strip_prefix(',')
            .ok_or_else(|| invalid("replay JSON containers require compact separators"))?;
    }
}

/// Keep uninterpreted JSON bytes, including Python float spelling and escaped
/// UTF-16. Sorted root reassembly also rejects duplicate keys and extra spacing.
fn replay_json_object(text: &str) -> PyResult<ReplayJsonObject> {
    let object: ReplayJsonObject =
        serde_json::from_str(text).map_err(|_| invalid("replay episode requires a JSON object"))?;
    if replay_json_join(&object, None) != text {
        return Err(invalid(
            "replay JSON object must preserve canonical keys and separators",
        ));
    }
    Ok(object)
}

fn replay_json_join(object: &ReplayJsonObject, omit: Option<&str>) -> String {
    format!(
        "{{{}}}",
        object
            .iter()
            .filter(|(key, _)| Some(key.as_str()) != omit)
            .map(|(key, value)| format!("{}:{}", observation_json_string(key), value.get()))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn replay_json_field<'a>(object: &'a ReplayJsonObject, key: &str) -> PyResult<&'a str> {
    object
        .get(key)
        .map(|value| value.get())
        .ok_or_else(|| invalid("replay episode binding field is absent"))
}

fn replay_json_value(text: &str) -> PyResult<serde_json::Value> {
    serde_json::from_str(text).map_err(|_| invalid("replay native binding field is invalid JSON"))
}

fn replay_identity_json(value: &ColdValue) -> PyResult<String> {
    Ok(match value {
        ColdValue::None => "null".into(),
        ColdValue::Bool(value) => value.to_string(),
        ColdValue::Integer(value) => value.clone(),
        ColdValue::Text(value) => observation_json_string(value),
        ColdValue::Sequence(values) => format!(
            "[{}]",
            values
                .iter()
                .map(replay_identity_json)
                .collect::<PyResult<Vec<_>>>()?
                .join(",")
        ),
        ColdValue::Bytes(_) => return Err(invalid("replay identity cannot contain payload bytes")),
    })
}

#[derive(Clone, Debug)]
struct ReplayReconstruction {
    operation: String,
    inputs: Vec<String>,
}

#[derive(Clone, Debug)]
struct ReplayMaterial {
    identity: String,
    kind: String,
    bytes_sha256: Option<String>,
    reconstruction: Option<ReplayReconstruction>,
    bytes: Option<Arc<[u8]>>,
}

impl ReplayMaterial {
    fn reference_json(&self) -> serde_json::Value {
        serde_json::json!({
            "identity": self.identity, "kind": self.kind, "bytes_sha256": self.bytes_sha256,
            "reconstruction": self.reconstruction.as_ref().map(|value|
                serde_json::json!({"operation": value.operation, "inputs": value.inputs})),
        })
    }

    fn parse(value: &ColdValue, earlier: &BTreeSet<String>) -> PyResult<Self> {
        let pair = value.fields(2)?;
        let fields = pair[0].fields(4)?;
        checked_digest(&fields[0])?;
        let identity = fields[0].text()?.to_owned();
        if earlier.contains(&identity) {
            return Err(invalid("duplicate replay material identity"));
        }
        let kind = fields[1].text()?.to_owned();
        if !matches!(
            kind.as_str(),
            "entry-boundary"
                | "source-tokens"
                | "topology"
                | "cache"
                | "feedback"
                | "feedback-statements"
                | "feedback-support-provenance"
                | "feedback-encoder"
                | "parameters"
                | "physical-partition"
                | "numerical-realization"
                | "logits"
                | "random-state"
                | "baseline"
                | "final-mask"
                | "pwl-cell"
                | "active-set"
                | "vjp"
                | "receipt"
                | "theory-delta"
                | "evidence"
                | "theory-state"
                | "logical-state"
                | "world-root"
                | "neural-generation"
                | "law"
                | "codebook"
                | "roster"
                | "action"
                | "result"
                | "observation"
                | "environment"
                | "provenance"
                | "downstream-snapshot"
                | "estimator-bound"
                | "training-view"
        ) {
            return Err(invalid("unknown replay material kind"));
        }
        let (bytes_sha256, reconstruction, bytes) = match (&fields[2], &fields[3], &pair[1]) {
            (ColdValue::Text(digest), ColdValue::None, ColdValue::Bytes(bytes)) => {
                checked_digest(&fields[2])?;
                if replay_hash(bytes) != *digest {
                    return Err(invalid("replay material byte digest differs"));
                }
                (Some(digest.clone()), None, Some(Arc::clone(bytes)))
            }
            (ColdValue::None, reconstruction, ColdValue::None) => {
                let fields = reconstruction.fields(2)?;
                let operation = fields[0].text()?.to_owned();
                let inputs = fields[1].strings()?;
                if inputs.is_empty() || inputs.iter().any(|value| !earlier.contains(value)) {
                    return Err(invalid(
                        "replay reconstruction inputs must name earlier materials",
                    ));
                }
                (None, Some(ReplayReconstruction { operation, inputs }), None)
            }
            _ => {
                return Err(invalid(
                    "replay material requires exactly bytes or reconstruction",
                ))
            }
        };
        if kind == "training-view" && bytes_sha256.as_ref() != Some(&identity) {
            return Err(invalid(
                "training-view identity must be the digest of its original bytes",
            ));
        }
        Ok(Self {
            identity,
            kind,
            bytes_sha256,
            reconstruction,
            bytes,
        })
    }
}

/// Full data custody is retained for the native restore join. Application
/// training identity, dimensions, targets and complete record bytes are bound
/// to the original data producer's projection, without a second identity. This does
/// not recognize opaque evidence as an authentic native publication.
#[derive(Clone, Debug)]
struct ReplayRow {
    basis: ReplayBasis,
    identity: ColdValue,
    record_line: String,
    record: ReplayJsonObject,
    materials: Vec<ReplayMaterial>,
    evidence: Arc<[u8]>,
}

#[derive(Clone, Debug)]
enum ReplayBasis {
    Episode { execution: serde_json::Value },
    CorpusAnchor { group: ReplayAnchorGroup },
}

#[derive(Clone, Copy, Debug)]
enum ReplayAnchorGroup {
    Language,
    Symbolic,
}

fn read_training_objective(value: &ColdValue) -> PyResult<Option<SemanticTrainingObjective>> {
    if *value == ColdValue::None {
        return Ok(None);
    }
    let fields = value.fields(6)?;
    let bounds = fields[1].fields(2)?;
    let coefficient_values = fields[2].fields(9)?;
    let mut coefficients = [0.0f32; 9];
    for (index, value) in coefficient_values.iter().enumerate() {
        let bits = u32::try_from(value.unsigned()?)
            .map_err(|_| invalid("training objective coefficient is not an f32 bit pattern"))?;
        coefficients[index] = f32::from_bits(bits);
    }
    let cost = fields[3].fields(2)?;
    let groups = fields[4]
        .sequence()?
        .iter()
        .map(|value| {
            let fields = value.fields(3)?;
            let kind = match fields[0].text()? {
                "masked-language" => SemanticTrainingObjectiveGroupKind::MaskedLanguage,
                "autoregressive-language" => {
                    SemanticTrainingObjectiveGroupKind::AutoregressiveLanguage
                }
                "semantic" => SemanticTrainingObjectiveGroupKind::Semantic,
                "edit" => SemanticTrainingObjectiveGroupKind::Edit,
                "execution" => SemanticTrainingObjectiveGroupKind::Execution,
                "retention-language" => SemanticTrainingObjectiveGroupKind::RetentionLanguage,
                "retention-symbolic" => SemanticTrainingObjectiveGroupKind::RetentionSymbolic,
                "actor-critic-cost" => SemanticTrainingObjectiveGroupKind::ActorCriticCost,
                _ => return Err(invalid("unknown training objective group kind")),
            };
            let row_ordinals = fields[2]
                .sequence()?
                .iter()
                .map(ColdValue::unsigned)
                .collect::<PyResult<Vec<_>>>()?;
            Ok(SemanticTrainingObjectiveGroup {
                kind,
                denominator: fields[1].unsigned()?,
                row_ordinals,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let canaries = fields[5]
        .sequence()?
        .iter()
        .map(|value| {
            let fields = value.fields(7)?;
            let kind = match fields[0].text()? {
                "symbolic-utility" => SemanticTrainingCanaryKind::SymbolicUtility,
                "retained-behavior" => SemanticTrainingCanaryKind::RetainedBehavior,
                "logit-drift" => SemanticTrainingCanaryKind::LogitDrift,
                "goal-chain" => SemanticTrainingCanaryKind::GoalChain,
                "resource-limits" => SemanticTrainingCanaryKind::ResourceLimits,
                _ => return Err(invalid("unknown training canary kind")),
            };
            Ok(SemanticTrainingCanary {
                kind,
                row_ordinal: fields[1].unsigned()?,
                lower_bound: f64::from_bits(fields[2].unsigned()?),
                upper_bound: f64::from_bits(fields[3].unsigned()?),
                memory_limit: fields[4].unsigned()?,
                fuel_limit: fields[5].unsigned()?,
                identity: replay_digest_identity(fields[6].text()?)?,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(Some(SemanticTrainingObjective {
        identity: replay_digest_identity(fields[0].text()?)?,
        evaluator_min: f64::from_bits(bounds[0].unsigned()?),
        evaluator_max: f64::from_bits(bounds[1].unsigned()?),
        coefficients,
        cost_unit: replay_digest_identity(cost[0].text()?)?,
        cost_cap: cost[1].unsigned()?,
        groups,
        canaries,
    }))
}

impl ReplayBasis {
    fn name(&self) -> &'static str {
        match self {
            Self::Episode { .. } => "episode",
            Self::CorpusAnchor { .. } => "anchor",
        }
    }
}

const NATIVE_REPLAY_OPERATION: &str = "xlog.semantic-transition.replay.v1";

struct NativeReplayBinding {
    material: xlog_cuda::SemanticReplayMaterial,
    original_decisions: Arc<[u8]>,
}

impl ReplayRow {
    fn training_view_row(&self) -> PyResult<SemanticTrainingViewRow> {
        let mut materials = self
            .materials
            .iter()
            .filter(|material| material.kind == "training-view");
        let material = materials
            .next()
            .ok_or_else(|| invalid("replay row has no admitted training view"))?;
        if materials.next().is_some() {
            return Err(invalid(
                "replay row has more than one admitted training view",
            ));
        }
        let bytes = material
            .bytes
            .as_deref()
            .ok_or_else(|| invalid("training view has no original bytes"))?;
        let identity = self.identity.fields(6)?;
        let origin = match &self.basis {
            ReplayBasis::Episode { .. } => Some(
                self.native_replay()?
                    .material
                    .training_view_origin()
                    .map_err(xlog_err)?,
            ),
            ReplayBasis::CorpusAnchor { .. } => None,
        };
        Ok(SemanticTrainingViewRow {
            basis: match &self.basis {
                ReplayBasis::Episode { .. } => SemanticTrainingViewBasis::Episode,
                ReplayBasis::CorpusAnchor {
                    group: ReplayAnchorGroup::Language,
                } => SemanticTrainingViewBasis::CorpusLanguageAnchor,
                ReplayBasis::CorpusAnchor {
                    group: ReplayAnchorGroup::Symbolic,
                } => SemanticTrainingViewBasis::CorpusSymbolicAnchor,
            },
            content_identity: replay_digest_identity(identity[2].text()?)?,
            origin,
            bytes: bytes.to_vec(),
        })
    }

    fn native_replay(&self) -> PyResult<NativeReplayBinding> {
        let ReplayBasis::Episode { execution } = &self.basis else {
            return Err(invalid(
                "a corpus anchor has admission evidence, not a native execution to replay",
            ));
        };
        let digest = |name: &str| {
            execution
                .get(name)
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| invalid("native replay material root is absent"))
        };
        let predecessor = digest("action_base_h_logical")?;
        let successor = digest("action_successor_h_logical")?;
        let envelope = replay_json_object(replay_json_field(&self.record, "envelope")?)?;
        let provenance = replay_json_value(replay_json_field(&envelope, "provenance_identity")?)?;
        let provenance = provenance
            .as_str()
            .ok_or_else(|| invalid("native replay provenance root is absent"))?;
        let resolve = |identity: &str, kind: &str| {
            self.materials
                .iter()
                .find(|material| material.identity == identity && material.kind == kind)
                .ok_or_else(|| {
                    invalid("native replay input does not resolve to its exact material")
                })
        };
        let original = resolve(predecessor, "logical-state")?;
        let result = resolve(successor, "logical-state")?;
        let provenance = resolve(provenance, "provenance")?;
        let reconstruction = result.reconstruction.as_ref().ok_or_else(|| {
            invalid("successor must be reconstructed, not loaded from its own full snapshot")
        })?;
        if reconstruction.operation != NATIVE_REPLAY_OPERATION {
            return Err(invalid(
                "successor reconstruction must use the pinned native transition operation",
            ));
        }
        let invocation = execution
            .get("invocation")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| invalid("native replay invocation is absent"))?;
        for input in [predecessor, provenance.identity.as_str()]
            .into_iter()
            .chain(invocation.values().filter_map(serde_json::Value::as_str))
        {
            if !reconstruction.inputs.iter().any(|item| item == input) {
                return Err(invalid("native successor reconstruction omits an original invocation input or provenance root"));
            }
        }
        fn bytes(material: &ReplayMaterial) -> PyResult<&[u8]> {
            material.bytes.as_deref().ok_or_else(|| {
                invalid("native predecessor and provenance require their complete original bytes")
            })
        }
        let material = xlog_cuda::SemanticReplayMaterial::decode(
            bytes(original)?,
            &self.evidence,
            replay_digest_identity(predecessor)?,
            replay_digest_identity(successor)?,
        )
        .map_err(xlog_err)?;
        ColdValue::from_canonical_bytes(material.model_numerical_mode().map_err(xlog_err)?)?;
        let original_decisions = material
            .verify_provenance(bytes(provenance)?)
            .map_err(xlog_err)?;
        // Historical metadata is reproducible input only. Never install this
        // snapshot as the current TaskUse authority or derive a present grant.
        AuthoritySnapshot::parse(&ColdValue::from_canonical_bytes(&original_decisions)?)?;
        Ok(NativeReplayBinding {
            material,
            original_decisions: original_decisions.into(),
        })
    }

    fn python_view(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let mut blobs: BTreeMap<&str, Py<PyAny>> = BTreeMap::new();
        let mut materials = Vec::with_capacity(self.materials.len());
        for material in &self.materials {
            let reference = PyDict::new(py);
            reference.set_item("identity", &material.identity)?;
            reference.set_item("kind", &material.kind)?;
            reference.set_item("bytes_sha256", &material.bytes_sha256)?;
            if let Some(reconstruction) = &material.reconstruction {
                let item = PyDict::new(py);
                item.set_item("operation", &reconstruction.operation)?;
                item.set_item("inputs", PyList::new(py, &reconstruction.inputs)?)?;
                reference.set_item("reconstruction", item)?;
            } else {
                reference.set_item("reconstruction", py.None())?;
            }
            let payload =
                if let (Some(digest), Some(bytes)) = (&material.bytes_sha256, &material.bytes) {
                    blobs
                        .entry(digest)
                        .or_insert_with(|| PyBytes::new(py, bytes).into_any().unbind())
                        .clone_ref(py)
                } else {
                    py.None()
                };
            materials.push((reference, payload));
        }
        Ok((
            self.basis.name(),
            self.identity.python_value(py)?,
            &self.record_line,
            PyTuple::new(py, materials)?,
            PyBytes::new(py, &self.evidence),
        )
            .into_pyobject(py)?
            .unbind())
    }

    fn parse(value: &ColdValue) -> PyResult<Self> {
        let fields = value.fields(5)?;
        let basis = fields[0].text()?;
        if !matches!(basis, "episode" | "anchor") {
            return Err(invalid("unknown replay row basis"));
        }
        let identity = fields[1].fields(6)?;
        checked_digest(&identity[1])?;
        checked_digest(&identity[2])?;
        let record_line = fields[2].text()?.to_owned();
        validate_replay_json_structure(&record_line, 0)?;
        let record = replay_json_object(&record_line)?;
        let payload = replay_json_join(&record, Some("content_sha256"));
        // These are the external producer's wire-domain tags, not native
        // semantic laws or an additional native identity projection.
        let content = if basis == "anchor" {
            format!("[\"dlm-new/corpus-anchor/v1\",{payload}]")
        } else {
            payload
        };
        if replay_json_value(replay_json_field(&record, "content_sha256")?)?
            != identity[2].text()?
            || replay_hash(content.as_bytes()) != identity[2].text()?
        {
            return Err(invalid(
                "replay record content digest differs from its identity",
            ));
        }
        validate_replay_training_identity(identity, &record)?;
        let ColdValue::Bytes(evidence) = &fields[4] else {
            return Err(invalid("replay evidence bytes are absent"));
        };
        if evidence.is_empty() {
            return Err(invalid("replay evidence bytes are empty"));
        }
        if basis == "anchor" {
            let (source, group) = validate_anchor_admission(identity, &record, evidence)?;
            let transported = fields[3].fields(1)?;
            let material = ReplayMaterial::parse(&transported[0], &BTreeSet::new())?;
            if material.kind != "training-view"
                || replay_json_value(replay_json_field(&record, "training_view_identity")?)?
                    != material.identity
            {
                return Err(invalid(
                    "corpus anchor requires its one original training-view material",
                ));
            }
            validate_replay_training_payload(&material, identity, &source, true)?;
            return Ok(Self {
                basis: ReplayBasis::CorpusAnchor { group },
                identity: fields[1].clone(),
                record_line,
                record,
                materials: vec![material],
                evidence: Arc::clone(evidence),
            });
        }
        let envelope = replay_json_object(replay_json_field(&record, "envelope")?)?;
        let execution = replay_json_object(replay_json_field(&envelope, "execution_materials")?)?;
        let expected = [
            "action_base_h_logical",
            "action_successor_h_logical",
            "invocation",
            "training_view_identity",
            "baseline",
            "materials",
        ];
        if execution.len() != expected.len()
            || !expected.iter().all(|key| execution.contains_key(*key))
        {
            return Err(invalid("replay execution materials field set differs"));
        }
        let declared: Vec<Box<serde_json::value::RawValue>> =
            serde_json::from_str(replay_json_field(&execution, "materials")?)
                .map_err(|_| invalid("replay material roster is not an array"))?;
        let transported = fields[3].sequence()?;
        if declared.len() != transported.len() || declared.is_empty() {
            return Err(invalid(
                "replay material roster is missing or detached from the episode",
            ));
        }
        let mut earlier = BTreeSet::new();
        let mut materials = Vec::new();
        for (declared, transported) in declared.iter().zip(transported) {
            let material = ReplayMaterial::parse(transported, &earlier)?;
            let reference = replay_json_object(declared.get())?;
            let reconstruction = replay_json_field(&reference, "reconstruction")?;
            if reconstruction != "null" {
                replay_json_object(reconstruction)?;
            }
            if replay_json_value(declared.get())? != material.reference_json() {
                return Err(invalid(
                    "replay material order or reference differs from the episode",
                ));
            }
            earlier.insert(material.identity.clone());
            materials.push(material);
        }
        let require_kind = |raw: &str, kind: &str| -> PyResult<()> {
            let value = replay_json_value(raw)?;
            let identity = value
                .as_str()
                .ok_or_else(|| invalid("replay material binding requires a digest"))?;
            if !materials
                .iter()
                .any(|material| material.identity == identity && material.kind == kind)
            {
                return Err(invalid(
                    "replay execution binding does not resolve to its material kind",
                ));
            }
            Ok(())
        };
        require_kind(
            replay_json_field(&execution, "action_base_h_logical")?,
            "logical-state",
        )?;
        require_kind(
            replay_json_field(&execution, "action_successor_h_logical")?,
            "logical-state",
        )?;
        let invocation = replay_json_object(replay_json_field(&execution, "invocation")?)?;
        let invocation_kinds = [
            ("entry_boundary_identity", "entry-boundary"),
            ("source_tokens_identity", "source-tokens"),
            ("topology_identity", "topology"),
            ("cache_identity", "cache"),
            ("feedback_identity", "feedback"),
            ("model_generation", "neural-generation"),
            ("parameters_identity", "parameters"),
            ("physical_partition_identity", "physical-partition"),
            ("numerical_realization_identity", "numerical-realization"),
            ("logits_identity", "logits"),
            ("random_state_identity", "random-state"),
        ];
        if invocation.len() != invocation_kinds.len() {
            return Err(invalid("replay invocation field set differs"));
        }
        for (name, kind) in invocation_kinds {
            require_kind(replay_json_field(&invocation, name)?, kind)?;
        }
        if replay_json_field(&invocation, "model_generation")?
            != replay_json_field(&envelope, "neural_generation")?
        {
            return Err(invalid(
                "replay invocation generation differs from its envelope",
            ));
        }
        let training = materials
            .iter()
            .filter(|material| material.kind == "training-view")
            .collect::<Vec<_>>();
        if training.len() != 1 {
            return Err(invalid(
                "replay episode requires one original training-view material",
            ));
        }
        if replay_json_value(replay_json_field(&execution, "training_view_identity")?)?
            != training[0].identity
        {
            return Err(invalid(
                "replay episode training-view identity differs from its original material",
            ));
        }
        let source = replay_json_value(replay_json_field(&invocation, "source_tokens_identity")?)?;
        validate_replay_training_payload(
            training[0],
            identity,
            source
                .as_str()
                .ok_or_else(|| invalid("replay source identity is not a digest"))?,
            false,
        )?;
        Ok(Self {
            basis: ReplayBasis::Episode {
                execution: replay_json_value(replay_json_field(&envelope, "execution_materials")?)?,
            },
            identity: fields[1].clone(),
            record_line,
            record,
            materials,
            evidence: Arc::clone(evidence),
        })
    }
}

fn validate_replay_training_identity(
    identity: &[ColdValue],
    record: &ReplayJsonObject,
) -> PyResult<()> {
    const NAMES: [&str; 9] = [
        "data_manifest_hash",
        "example_id",
        "final_split",
        "tokenizer_hash",
        "tokenizer_revision",
        "mask_policy_hash",
        "training_seed",
        "epoch_index",
        "variant_index",
    ];
    let view = identity[0].fields(NAMES.len())?;
    let original = replay_json_object(replay_json_field(record, "training_view")?)?;
    if original.len() != NAMES.len() {
        return Err(invalid("training view requires its original nine fields"));
    }
    for (index, name) in NAMES.iter().enumerate() {
        if index < 6 {
            view[index].text()?;
        } else if !matches!(&view[index], ColdValue::Integer(value) if !value.starts_with('-')) {
            return Err(invalid(
                "training view seed, epoch and variant require nonnegative integers",
            ));
        }
        if replay_json_field(&original, name)? != replay_identity_json(&view[index])? {
            return Err(invalid(
                "training identity differs from its original record projection",
            ));
        }
    }
    for index in [0, 3, 5] {
        checked_digest(&view[index])?;
    }
    if replay_json_field(record, "example_id")? != replay_identity_json(&view[1])?
        || !matches!(view[2].text()?, "train" | "dev" | "request-local-replay")
    {
        return Err(invalid(
            "replay training identity names another example or a sealed partition",
        ));
    }
    if let Some(split) = record.get("final_split") {
        if split.get() != replay_identity_json(&view[2])? {
            return Err(invalid("replay training partition differs from its record"));
        }
    }
    let canonical = replay_identity_json(&identity[0])?;
    if replay_hash(format!("[\"dlm-new/training-view-identity/v1\",{canonical}]").as_bytes())
        != identity[1].text()?
    {
        return Err(invalid("training-view identity digest differs"));
    }
    Ok(())
}

fn validate_anchor_admission(
    identity: &[ColdValue],
    record: &ReplayJsonObject,
    evidence: &[u8],
) -> PyResult<(String, ReplayAnchorGroup)> {
    let keys = [
        "content_sha256",
        "example_id",
        "group",
        "training_view",
        "training_view_identity",
    ];
    if record.len() != keys.len() || keys.iter().any(|key| !record.contains_key(*key)) {
        return Err(invalid("corpus anchor record field set differs"));
    }
    let view = identity[0].fields(9)?;
    let group = match replay_json_value(replay_json_field(record, "group")?)?.as_str() {
        Some("language") => ReplayAnchorGroup::Language,
        Some("symbolic") => ReplayAnchorGroup::Symbolic,
        _ => {
            return Err(invalid(
                "corpus anchor requires the exact language or symbolic retention group",
            ));
        }
    };
    if view[2].text()? != "train" {
        return Err(invalid(
            "corpus anchor requires its admitted train partition and original retention group",
        ));
    }
    let text = std::str::from_utf8(evidence)
        .map_err(|_| invalid("corpus admission evidence is not UTF-8"))?;
    validate_replay_json_structure(text, 0)?;
    let parts: Vec<Box<serde_json::value::RawValue>> = serde_json::from_str(text)
        .map_err(|_| invalid("corpus anchor admission evidence requires its canonical record"))?;
    if parts.len() != 3
        || replay_json_value(parts[0].get())? != "dlm-new/corpus-anchor-admission/v1"
        || parts[1].get() != replay_identity_json(&view[0])?
    {
        return Err(invalid(
            "corpus anchor admission differs from the original manifest",
        ));
    }
    let entry = replay_json_value(parts[2].get())?;
    let entry = entry
        .as_array()
        .filter(|entry| entry.len() == 10)
        .ok_or_else(|| invalid("corpus anchor admission requires the complete manifest entry"))?;
    if entry[0] != view[1].text()? || entry[1] != "train" {
        return Err(invalid(
            "corpus anchor admission names another example or partition",
        ));
    }
    for index in [2, 3, 4] {
        replay_digest_identity(
            entry[index]
                .as_str()
                .ok_or_else(|| invalid("manifest entry content digest is absent"))?,
        )?;
    }
    let provenance = entry[7]
        .as_array()
        .filter(|parts| parts.len() == 11)
        .ok_or_else(|| invalid("corpus anchor requires its complete original corpus provenance"))?;
    let admission = entry[8]
        .as_array()
        .filter(|parts| parts.len() == 3)
        .ok_or_else(|| invalid("corpus anchor requires its original admission decision"))?;
    if provenance[0] != "corpus"
        || admission[0] != "corpus"
        || admission[2] != "corpus"
        || !provenance[8]
            .as_array()
            .is_some_and(|uses| uses.iter().any(|use_| use_ == "training"))
        || !admission[1]
            .as_array()
            .is_some_and(|splits| splits.iter().any(|split| split == "train"))
    {
        return Err(invalid("corpus anchor has no train-only corpus admission"));
    }
    // This checks the transported producer record, not a signature or a grant.
    // TaskAuthority still requires the trusted application's full dependency
    // closure and fresh permission for every use of these retained bytes.
    Ok((
        entry[4]
            .as_str()
            .expect("checked original normalized record digest")
            .to_owned(),
        group,
    ))
}

fn validate_replay_training_payload(
    material: &ReplayMaterial,
    identity: &[ColdValue],
    source: &str,
    require_retention: bool,
) -> PyResult<()> {
    let bytes = material
        .bytes
        .as_deref()
        .ok_or_else(|| invalid("training view requires its original bytes"))?;
    let mut tag = [0u8; 32];
    let schema = b"dlm-new/training-view/v1";
    tag[..schema.len()].copy_from_slice(schema);
    if bytes.len() < 136
        || bytes[..32] != tag
        || bytes[32..64] != *replay_digest_identity(identity[1].text()?)?.as_bytes()
        || bytes[64..96] != *replay_digest_identity(source)?.as_bytes()
    {
        return Err(invalid(
            "training-view header differs from its original identity or source",
        ));
    }
    let word = |offset: usize| {
        u64::from_le_bytes(
            bytes[offset..offset + 8]
                .try_into()
                .expect("bounded training-view word"),
        )
    };
    let window = usize::try_from(word(96))
        .map_err(|_| invalid("training-view window exceeds host address space"))?;
    let length = identity[3].unsigned()?;
    let block = identity[4].unsigned()?;
    let answer_start = word(128);
    let expected = window
        .checked_mul(68)
        .and_then(|size| size.checked_add(136 + (window % 2) * 4));
    if window == 0
        || length == 0
        || block == 0
        || word(104) != length
        || word(112) != block
        || word(120) > length
        || answer_start == 0
        || answer_start > length
        || (require_retention && answer_start == length)
        || expected != Some(bytes.len())
        || length
            .checked_mul(2)
            .is_none_or(|extent| extent > window as u64)
    {
        return Err(invalid(
            "training-view byte extent or dimensions differ from its original row",
        ));
    }
    let padding = 136 + window * 20;
    if bytes[padding..padding + (window % 2) * 4]
        .iter()
        .any(|&value| value != 0)
    {
        return Err(invalid("training-view alignment padding is not canonical"));
    }
    let mut targets = BTreeSet::new();
    let mut previous = None;
    for target in identity[5].sequence()? {
        let target = target.fields(3)?;
        let position = target[0].unsigned()?;
        if position >= length
            || previous.is_some_and(|last| position <= last)
            || target[1].unsigned()? != position
            || Some(target[2].unsigned()?) != length.checked_add(position)
        {
            return Err(invalid(
                "training-view targets must retain ordered (p, p, L+p) coordinates",
            ));
        }
        previous = Some(position);
        targets.insert(target[2].unsigned()? as usize);
    }
    let retention = padding + (window % 2) * 4 + window * 8;
    for slot in 0..window {
        let label = word(136 + window * 8 + slot * 8) as i64;
        let offset = 136 + window * 16 + slot * 4;
        let weight = f32::from_le_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .expect("bounded training-view weight"),
        );
        if targets.contains(&slot) {
            if label < 0 || !weight.is_finite() || weight <= 0.0 {
                return Err(invalid(
                    "training-view target is absent from its original MASK supervision",
                ));
            }
        } else if label != -100 || weight != 0.0 {
            return Err(invalid(
                "training-view MASK supervision contains an undeclared target",
            ));
        }
        // Retention reads the original causal memory predictor, not the
        // permuted prediction stream or a newly drawn subset of MASK rows.
        let next = slot as u64 + 1;
        let expected_label = if next >= answer_start && next < length {
            let token = word(136 + (slot + 1) * 8) as i64;
            if token < 0 {
                return Err(invalid("training-view retention target is not a token"));
            }
            token
        } else {
            -100
        };
        if word(retention + slot * 8) as i64 != expected_label {
            return Err(invalid(
                "training-view retention labels differ from the original answer projection",
            ));
        }
    }
    Ok(())
}

fn replay_digest_identity(text: &str) -> PyResult<Identity256> {
    checked_digest(&ColdValue::Text(text.to_owned()))?;
    let mut bytes = [0u8; 32];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16)
            .map_err(|_| invalid("native replay identity is not an exact SHA256 digest"))?;
    }
    Ok(Identity256::from_bytes(bytes))
}

fn select_replay_row(rows: &[ReplayRow], selection: &ColdValue) -> PyResult<Option<usize>> {
    if selection == &ColdValue::None {
        return Ok(None);
    }
    let fields = selection.fields(2)?;
    let ordinal = usize::try_from(fields[0].unsigned()?)
        .map_err(|_| invalid("selected replay ordinal exceeds host address space"))?;
    let row = rows
        .get(ordinal)
        .ok_or_else(|| invalid("selected replay ordinal is absent"))?;
    if row.identity != fields[1] {
        return Err(invalid(
            "selected replay identity differs from the exact original row",
        ));
    }
    Ok(Some(ordinal))
}

fn decode_selected_replay(
    rows: &[ReplayRow],
    selection: &ColdValue,
) -> PyResult<(Option<usize>, Option<NativeReplayBinding>)> {
    let selection = select_replay_row(rows, selection)?;
    let mut selected = None;
    for (ordinal, row) in rows.iter().enumerate() {
        if matches!(row.basis, ReplayBasis::CorpusAnchor { .. }) {
            if selection == Some(ordinal) {
                return Err(invalid(
                    "native execution replay cannot select a corpus anchor",
                ));
            }
            continue;
        }
        let material = row.native_replay()?;
        if selection == Some(ordinal) {
            selected = Some(material);
        }
    }
    Ok((selection, selected))
}

fn read_task_evaluation_spec(
    statements: &Bound<'_, PyAny>,
    supports: &Bound<'_, PyAny>,
    source: &Bound<'_, PyAny>,
    queries: &Bound<'_, PyAny>,
    scoring: &Bound<'_, PyAny>,
    truth_masks: &Bound<'_, PyAny>,
    budget: &mut usize,
) -> PyResult<xlog_cuda::SemanticTaskEvaluationSpec> {
    let statements = ColdValue::read(statements, budget, 0)?;
    let statements = statements.fields(3)?;
    let record_index = |value: &ColdValue| {
        u32::try_from(value.unsigned()?)
            .map_err(|_| invalid("native admitted record index exceeds u32"))
    };
    let statement_records = [
        record_index(&statements[0])?,
        record_index(&statements[1])?,
        record_index(&statements[2])?,
    ];
    let allowed_support_records = ColdValue::read(supports, budget, 0)?
        .sequence()?
        .iter()
        .map(record_index)
        .collect::<PyResult<Vec<_>>>()?;
    let source = ColdValue::read(source, budget, 0)?.text()?.to_owned();
    let queries = ColdValue::read(queries, budget, 0)?;
    let queries = queries.fields(3)?;
    let query_ordinal = |value: &ColdValue| {
        usize::try_from(value.unsigned()?)
            .map_err(|_| invalid("native task query ordinal exceeds host address space"))
    };
    let query_ordinals = [
        query_ordinal(&queries[0])?,
        query_ordinal(&queries[1])?,
        query_ordinal(&queries[2])?,
    ];
    let scoring = ColdValue::read(scoring, budget, 0)?;
    let scoring = scoring.fields(6)?;
    let weight = |value: &ColdValue| {
        u32::try_from(value.unsigned()?)
            .map_err(|_| invalid("native task scoring weight exceeds u32"))
    };
    let scoring = xlog_cuda::SemanticTaskScoring {
        correct_weight: weight(&scoring[0])?,
        all_correct_weight: weight(&scoring[1])?,
        work_weight: weight(&scoring[2])?,
        improvement_weight: weight(&scoring[3])?,
        refusal_weight: weight(&scoring[4])?,
        spent_weight: weight(&scoring[5])?,
    };
    let masks = ColdValue::read(truth_masks, budget, 0)?;
    let masks = masks.fields(3)?;
    let mask = |value: &ColdValue| {
        u8::try_from(value.unsigned()?).map_err(|_| invalid("native task truth mask exceeds u8"))
    };
    let admissible_truth_masks = [mask(&masks[0])?, mask(&masks[1])?, mask(&masks[2])?];
    let program = Arc::new(
        xlog_gpu::logic::SemanticLogicTaskProgram::compile(source, query_ordinals)
            .map_err(xlog_err)?,
    );
    Ok(xlog_cuda::SemanticTaskEvaluationSpec {
        statement_records,
        allowed_support_records,
        program,
        scoring,
        admissible_truth_masks,
    })
}

fn read_task_replay_limits(
    material: &Bound<'_, PyAny>,
    total: &Bound<'_, PyAny>,
    evidence: &Bound<'_, PyAny>,
    budget: &mut usize,
) -> PyResult<[usize; 3]> {
    let mut limits = [0; 3];
    for (index, (value, name)) in [
        (material, "max_material_bytes"),
        (total, "max_total_material_bytes"),
        (evidence, "max_evidence_bytes"),
    ]
    .into_iter()
    .enumerate()
    {
        limits[index] = usize::try_from(ColdValue::read(value, budget, 0)?.unsigned()?)
            .map_err(|_| invalid(&format!("{name} exceeds host address space")))?;
    }
    Ok(limits)
}

/// The immutable retained source values and complete trusted dependency graph.
/// These are resolved decisions from privileged application code, not signatures
/// authenticated by this extension and not rights inferred from provenance refs.
#[derive(Clone, Debug)]
struct TaskAuthority {
    canonical: Vec<u8>,
    task_ref: String,
    scope: Vec<String>,
    live: Vec<LiveAuthority>,
    replay: Vec<ReplayRow>,
    dependencies: Vec<Dependency>,
    feedback_roots: Vec<String>,
    publication: Vec<ResolvedGrant>,
    inference: Vec<ResolvedGrant>,
    training: Vec<ResolvedGrant>,
    initial_sources: Vec<SemanticObservedSource>,
    source_mapping: Vec<SemanticSourceMapping>,
    sources_allow_learning: bool,
    source_partitions: Vec<(Option<String>, String)>,
    corpus_sources: BTreeSet<String>,
}

impl TaskAuthority {
    fn parse(values: &[ColdValue]) -> PyResult<Self> {
        if values.len() != 9 {
            return Err(invalid("incorrect task authority input count"));
        }
        let scope = values[1].strings()?;
        if scope.len() != 5 {
            return Err(invalid("task_scope must contain five explicit fields"));
        }
        let live = values[2]
            .sequence()?
            .iter()
            .map(|value| {
                let fields = value.fields(2)?;
                validate_live(&fields[0], &scope)?;
                Ok(LiveAuthority {
                    canonical: fields[0].clone(),
                    retention_micros: fields[1].signed()?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut dependencies = Vec::new();
        for value in values[4].sequence()? {
            let fields = value.fields(7)?;
            let target = if fields[5] == ColdValue::None {
                None
            } else {
                let target = fields[5].fields(2)?;
                Some((
                    usize::try_from(target[0].unsigned()?)
                        .map_err(|_| invalid("target row index overflow"))?,
                    target[1].unsigned()?,
                ))
            };
            let native_record = if fields[6] == ColdValue::None {
                None
            } else {
                let record = fields[6].fields(2)?;
                let kind = record[0].text()?;
                if !matches!(kind, "statement" | "support" | "observer") {
                    return Err(invalid("invalid native task read-root kind"));
                }
                let index = u32::try_from(record[1].unsigned()?)
                    .map_err(|_| invalid("native record index exceeds u32"))?;
                if kind == "observer" && index != 0 {
                    return Err(invalid("only the bound native observer zero exists"));
                }
                Some((kind.to_owned(), index))
            };
            dependencies.push(Dependency {
                identity: fields[0].text()?.to_owned(),
                kind: fields[1].text()?.to_owned(),
                data_parents: fields[2].strings()?,
                control_parents: fields[3].strings()?,
                live_envelopes: fields[4].strings()?,
                target,
                native_record,
            });
        }
        let grants = |value: &ColdValue, training| {
            value
                .sequence()?
                .iter()
                .map(|value| ResolvedGrant::parse(value, training))
                .collect::<PyResult<Vec<_>>>()
        };
        let result = Self {
            canonical: ColdValue::Sequence(values.to_vec()).canonical_bytes(),
            task_ref: values[0].text()?.to_owned(),
            scope,
            live,
            replay: values[3]
                .sequence()?
                .iter()
                .map(ReplayRow::parse)
                .collect::<PyResult<_>>()?,
            dependencies,
            feedback_roots: values[5].strings()?,
            publication: grants(&values[6], false)?,
            inference: grants(&values[7], false)?,
            training: grants(&values[8], true)?,
            initial_sources: Vec::new(),
            source_mapping: Vec::new(),
            sources_allow_learning: true,
            source_partitions: Vec::new(),
            corpus_sources: BTreeSet::new(),
        };
        result.validate_lineage()?;
        Ok(result)
    }

    fn bind_initial_sources(&mut self, values: &ColdValue, mapping: &ColdValue) -> PyResult<()> {
        let mut sources = Vec::new();
        let mut learning = true;
        let mut identities = BTreeSet::new();
        let mut partitions = Vec::new();
        let mut corpus_sources = BTreeSet::new();
        for value in values.sequence()? {
            // Exact projection of the data owner's sealed manifest entry and
            // separately observed token payload, supplied by trusted Runtime.
            let fields = value.fields(6)?;
            let identity = fields[0].text()?;
            if !identities.insert(identity.to_owned()) {
                return Err(invalid("duplicate initial source identity"));
            }
            checked_digest(&fields[1])?;
            let tokenizer = fields[2].text()?;
            let revision = fields[3].text()?;
            let entry = fields[4].fields(10)?;
            entry[0].text()?;
            for digest in &entry[2..5] {
                checked_digest(digest)?;
            }
            entry[6].strings()?;
            let observation = entry[9].fields(4)?;
            let field = observation[0].text()?;
            checked_digest(&observation[1])?;
            checked_digest(&observation[2])?;
            let tokens = fields[5]
                .sequence()?
                .iter()
                .map(ColdValue::unsigned)
                .collect::<PyResult<Vec<_>>>()?;
            if tokens.is_empty() || observation[3].unsigned()? != tokens.len() as u64 {
                return Err(invalid(
                    "observed source token payload or bound count is invalid",
                ));
            }
            let payload = format!(
                "[{}]",
                tokens
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let transform = format!("{{\"field\":{},\"template\":\"observed-source-field:v1\",\"tokenizer_hash\":{},\"tokenizer_revision\":{}}}",
                observation_json_string(field), observation_json_string(tokenizer), observation_json_string(revision));
            let hash = |bytes: &[u8]| {
                Sha256::digest(bytes)
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>()
            };
            if hash(payload.as_bytes()) != observation[1].text()?
                || hash(transform.as_bytes()) != observation[2].text()?
            {
                return Err(invalid("observed payload or tokenizer transformation differs from the admitted binding"));
            }
            let origin = entry[7].sequence()?;
            let branch = origin
                .first()
                .ok_or_else(|| invalid("source origin is absent"))?
                .text()?;
            if branch == "corpus" || entry[5] != ColdValue::None {
                checked_digest(&entry[5])?;
            }
            let source_envelopes = self.source_lineage_envelopes(identity)?;
            match branch {
                "live" => {
                    validate_live(&entry[7], &self.scope)?;
                    if !self.live.iter().any(|live| live.canonical == entry[7]) {
                        return Err(invalid(
                            "initial source live origin differs from this task's admitted envelope",
                        ));
                    }
                    if !source_envelopes.contains(origin[1].text()?) {
                        return Err(invalid(
                            "initial source lineage omits its actual live origin envelope",
                        ));
                    }
                }
                "corpus" => {
                    corpus_sources.insert(identity.to_owned());
                    let origin = entry[7].fields(11)?;
                    for index in [1, 2, 3, 4, 5, 6, 9, 10] {
                        origin[index].text()?;
                    }
                    for index in [7, 8] {
                        let capabilities = origin[index].strings()?;
                        unique(&capabilities)?;
                        if capabilities.is_empty() {
                            return Err(invalid("corpus origin has an empty capability set"));
                        }
                    }
                }
                _ => {
                    return Err(invalid(
                        "observed source requires complete corpus or live origin",
                    ))
                }
            }
            let admission = entry[8].fields(3)?;
            let permitted = admission[1].strings()?;
            unique(&permitted)?;
            if admission[0].text()? != branch
                || permitted.iter().any(|split| {
                    !matches!(
                        split.as_str(),
                        "train" | "dev" | "eval" | "request-local-replay"
                    )
                })
                || (branch == "corpus" && admission[2].text()? != "corpus")
                || (branch == "live"
                    && !matches!(admission[2].text()?, "cross-request" | "request-local"))
            {
                return Err(invalid(
                    "observed source admission does not match its full origin",
                ));
            }
            if entry[1] == ColdValue::None {
                if branch != "live"
                    || !origin[8].sequence()?.is_empty()
                    || !permitted.is_empty()
                    || admission[2].text()? != "request-local"
                {
                    return Err(invalid(
                        "observation-only source cannot carry learning or replay admission",
                    ));
                }
                learning = false;
            } else {
                let split = entry[1].text()?;
                if !permitted.iter().any(|allowed| allowed == split) {
                    return Err(invalid(
                        "source split is absent from its admitted partitions",
                    ));
                }
                learning &= split != "eval";
                if branch == "live" && origin[8].sequence()?.is_empty() {
                    return Err(invalid(
                        "live source without learning rights cannot enter a replay partition",
                    ));
                }
            }
            sources.push(SemanticObservedSource {
                identity: identity.to_owned(),
                tokens,
                origin: value.canonical_bytes(),
            });
            partitions.push((
                if entry[1] == ColdValue::None {
                    None
                } else {
                    Some(entry[1].text()?.to_owned())
                },
                admission[2].text()?.to_owned(),
            ));
        }
        let mapping = mapping
            .sequence()?
            .iter()
            .map(|entry| {
                let fields = entry.fields(3)?;
                let source = fields[0].text()?;
                let offset = fields[1].unsigned()?;
                let logical_position = fields[2].unsigned()?;
                if sources
                    .iter()
                    .find(|entry| entry.identity == source)
                    .is_none_or(|entry| offset >= entry.tokens.len() as u64)
                {
                    return Err(invalid(
                        "initial source mapping has an unknown source or out-of-range token offset",
                    ));
                }
                Ok(SemanticSourceMapping {
                    source: source.to_owned(),
                    offset,
                    logical_position,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        if mapping
            .iter()
            .map(|entry| entry.logical_position)
            .collect::<BTreeSet<_>>()
            .len()
            != mapping.len()
        {
            return Err(invalid("initial source mapping repeats a logical position"));
        }
        self.initial_sources = sources;
        self.source_mapping = mapping;
        self.sources_allow_learning = learning;
        self.source_partitions = partitions;
        self.corpus_sources = corpus_sources;
        self.validate_lineage()?;
        self.require_source_origins()?;
        // Preserve the original canonical task payload and extend that same
        // retained authority record with the complete admitted source values.
        let source_bytes = values.canonical_bytes();
        let mut binding = b"xlog.task.observed-sources.v1\0".to_vec();
        for bytes in [&self.canonical, &source_bytes] {
            binding.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            binding.extend_from_slice(bytes);
        }
        for entry in &self.source_mapping {
            binding.extend_from_slice(&(entry.source.len() as u64).to_le_bytes());
            binding.extend_from_slice(entry.source.as_bytes());
            binding.extend_from_slice(&entry.offset.to_le_bytes());
            binding.extend_from_slice(&entry.logical_position.to_le_bytes());
        }
        self.canonical = binding;
        Ok(())
    }

    fn source_lineage_envelopes(&self, identity: &str) -> PyResult<BTreeSet<String>> {
        let index = self
            .dependencies
            .iter()
            .map(|node| (node.identity.as_str(), node))
            .collect::<BTreeMap<_, _>>();
        let mut pending = vec![identity];
        let mut seen = BTreeSet::new();
        let mut envelopes = BTreeSet::new();
        let mut observed = false;
        while let Some(reference) = pending.pop() {
            if !seen.insert(reference) {
                continue;
            }
            let node = index.get(reference).ok_or_else(|| {
                invalid("initial source is absent from the complete task dependency graph")
            })?;
            if node.kind == "target_value" {
                return Err(invalid(
                    "initial source depends on a target value through data or control lineage",
                ));
            }
            observed |= node.kind == "source";
            envelopes.extend(node.live_envelopes.iter().cloned());
            pending.extend(
                node.data_parents
                    .iter()
                    .chain(&node.control_parents)
                    .map(String::as_str),
            );
        }
        if !observed {
            return Err(invalid("initial source lacks observed data ancestry"));
        }
        Ok(envelopes)
    }

    fn require_source_origins(&self) -> PyResult<()> {
        for node in &self.dependencies {
            if node.kind == "source"
                && node.live_envelopes.is_empty()
                && !self.corpus_sources.contains(&node.identity)
            {
                return Err(invalid(
                    "source root without a live envelope requires its admitted corpus observation",
                ));
            }
        }
        Ok(())
    }

    fn validate_lineage(&self) -> PyResult<()> {
        let mut targets = BTreeSet::new();
        for (row, replay) in self.replay.iter().enumerate() {
            let fields = replay.identity.fields(6)?;
            let context = fields[0].fields(9)?;
            if !matches!(context[2].text()?, "train" | "dev" | "request-local-replay") {
                return Err(invalid("replay task cannot contain sealed evaluation rows"));
            }
            checked_digest(&fields[1])?;
            checked_digest(&fields[2])?;
            let length = fields[3].unsigned()?;
            if length == 0 || fields[4].unsigned()? == 0 {
                return Err(invalid("replay dimensions must be positive"));
            }
            let mut previous = None;
            for value in fields[5].sequence()? {
                let target = value.fields(3)?;
                let position = target[0].unsigned()?;
                if position >= length
                    || previous.is_some_and(|last| position <= last)
                    || target[1].unsigned()? != position
                    || Some(target[2].unsigned()?) != length.checked_add(position)
                {
                    return Err(invalid(
                        "replay target must preserve ordered (p, p, L+p) identity",
                    ));
                }
                previous = Some(position);
                targets.insert((row, position));
            }
        }
        if self.dependencies.is_empty() {
            return Err(invalid("task dependency closure cannot be empty"));
        }
        let index = self
            .dependencies
            .iter()
            .enumerate()
            .map(|(i, node)| (node.identity.as_str(), i))
            .collect::<BTreeMap<_, _>>();
        if index.len() != self.dependencies.len() {
            return Err(invalid("duplicate dependency identity"));
        }
        let envelopes = self
            .live
            .iter()
            .map(|live| live.canonical.fields(16)?[1].text())
            .collect::<PyResult<BTreeSet<_>>>()?;
        if envelopes.len() != self.live.len() {
            return Err(invalid("duplicate live envelope identity"));
        }
        let mut masks = BTreeSet::new();
        let mut indegree = vec![0usize; self.dependencies.len()];
        let mut children = vec![Vec::new(); self.dependencies.len()];
        for (child, node) in self.dependencies.iter().enumerate() {
            unique(&node.data_parents)?;
            unique(&node.control_parents)?;
            unique(&node.live_envelopes)?;
            if node
                .live_envelopes
                .iter()
                .any(|reference| !envelopes.contains(reference.as_str()))
            {
                return Err(invalid("dependency names an unbound data-use envelope"));
            }
            match node.kind.as_str() {
                "constant"
                    if node.data_parents.is_empty()
                        && node.control_parents.is_empty()
                        && node.live_envelopes.is_empty()
                        && node.target.is_none() => {}
                "source"
                    if node.data_parents.is_empty()
                        && node.control_parents.is_empty()
                        && node.target.is_none() => {}
                "mask" | "target_value"
                    if node.target.is_some() && !node.data_parents.is_empty() =>
                {
                    let target = node.target.unwrap();
                    if !targets.contains(&target) {
                        return Err(invalid(
                            "dependency target is not in the verified replay projection",
                        ));
                    }
                    if node.kind == "mask" && !masks.insert(target) {
                        return Err(invalid("duplicate acquired MASK identity"));
                    }
                }
                "derived"
                    if node.target.is_none()
                        && (!node.data_parents.is_empty() || !node.control_parents.is_empty()) => {}
                _ => {
                    return Err(invalid(
                        "dependency kind does not match its source and target fields",
                    ))
                }
            }
            for parent in node.data_parents.iter().chain(&node.control_parents) {
                let parent = *index
                    .get(parent.as_str())
                    .ok_or_else(|| invalid("dependency closure omits a data or control parent"))?;
                indegree[child] += 1;
                children[parent].push(child);
            }
        }
        if masks != targets {
            return Err(invalid(
                "replay target lacks its exact acquired MASK provenance join",
            ));
        }
        let mut ready = indegree
            .iter()
            .enumerate()
            .filter_map(|(i, count)| (*count == 0).then_some(i))
            .collect::<Vec<_>>();
        let mut visited = 0;
        while let Some(parent) = ready.pop() {
            visited += 1;
            for &child in &children[parent] {
                indegree[child] -= 1;
                if indegree[child] == 0 {
                    ready.push(child);
                }
            }
        }
        if visited != self.dependencies.len() {
            return Err(invalid("task dependency closure contains a cycle"));
        }
        unique(&self.feedback_roots)?;
        let mut pending =
            self.feedback_roots
                .iter()
                .map(|reference| {
                    index.get(reference.as_str()).copied().ok_or_else(|| {
                        invalid("feedback root is absent from the dependency closure")
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
        // These roots cannot be omitted by selecting a smaller feedback_roots
        // tuple. bind_native_reads below requires their exact native read set.
        pending.extend(
            self.dependencies
                .iter()
                .enumerate()
                .filter_map(|(i, node)| node.native_record.as_ref().map(|_| i)),
        );
        for source in &self.initial_sources {
            let i = *index.get(source.identity.as_str()).ok_or_else(|| {
                invalid("initial source is absent from the task dependency closure")
            })?;
            if !matches!(self.dependencies[i].kind.as_str(), "source" | "derived") {
                return Err(invalid(
                    "initial observation must resolve to a source data/control lineage",
                ));
            }
            pending.push(i);
        }
        let mut seen = BTreeSet::new();
        while let Some(i) = pending.pop() {
            if !seen.insert(i) {
                continue;
            }
            let node = &self.dependencies[i];
            if node.kind == "target_value" {
                return Err(invalid(
                    "feedback or initial source depends on a target value through data or control lineage",
                ));
            }
            pending.extend(
                node.data_parents
                    .iter()
                    .chain(&node.control_parents)
                    .map(|parent| index[parent.as_str()]),
            );
        }
        for family in [&self.publication, &self.inference, &self.training] {
            unique(
                &family
                    .iter()
                    .map(|grant| grant.reference.clone())
                    .collect::<Vec<_>>(),
            )?;
            for grant in family {
                if grant.task_ref != self.task_ref
                    || grant.issuer_provenance.is_empty()
                    || grant
                        .dependencies
                        .iter()
                        .any(|reference| !index.contains_key(reference.as_str()))
                {
                    return Err(invalid(
                        "resolved grant does not match the exact task dependency scope",
                    ));
                }
            }
        }
        Ok(())
    }

    fn bind_native_reads(
        &self,
        observations: &xlog_cuda::SemanticTaskObservationRoots,
        supports: &[u32],
    ) -> PyResult<()> {
        let mut expected = observations
            .query_records
            .iter()
            .map(|&index| ("statement".to_owned(), index))
            .chain(supports.iter().map(|&index| ("support".to_owned(), index)))
            .chain(std::iter::once(("observer".to_owned(), 0)))
            .collect::<BTreeSet<_>>();
        for &(query, statement, support) in &observations.contributors {
            if query >= observations.query_records.len() as u32 {
                return Err(invalid(
                    "native observation contributor names an absent task query",
                ));
            }
            let statement = statement.ok_or_else(|| invalid(
                "native feedback contributor has a derived target without an original admitted statement"))?;
            expected.insert(("statement".to_owned(), statement));
            expected.insert(("support".to_owned(), support));
        }
        let actual = self
            .dependencies
            .iter()
            .filter_map(|node| node.native_record.clone())
            .collect::<Vec<_>>();
        let distinct = actual.iter().cloned().collect::<BTreeSet<_>>();
        if actual.len() != distinct.len() || distinct != expected {
            return Err(invalid("dependency closure must bind each actual native observer, statement and support read root exactly once"));
        }
        Ok(())
    }

    fn check_snapshot(&self, snapshot: &AuthoritySnapshot) -> PyResult<i64> {
        self.require_source_origins()?;
        if snapshot.live.len() != self.live.len() {
            return Err(invalid(
                "authority snapshot does not cover every live envelope",
            ));
        }
        for (live, (canonical, segment_end, revoked)) in self.live.iter().zip(&snapshot.live) {
            validate_live(canonical, &self.scope)?;
            if canonical != &live.canonical {
                return Err(invalid(
                    "snapshot envelope differs from the imported live provenance",
                ));
            }
            if live.retention_micros <= segment_end.micros
                || live.retention_micros <= snapshot.observed_at.micros
            {
                return Err(invalid(
                    "live data authority expires before the required boundary",
                ));
            }
            let revocation_ref = canonical.fields(16)?[15].text()?;
            if revoked.iter().any(|reference| reference == revocation_ref) {
                return Err(invalid("live data authority has been revoked"));
            }
            if segment_end != &snapshot.segment_end {
                return Err(invalid(
                    "live snapshot differs from the task's preregistered segment end",
                ));
            }
        }
        Ok(snapshot.segment_end.micros)
    }

    fn check_grants(
        &self,
        grants: &[ResolvedGrant],
        snapshot: &AuthoritySnapshot,
        horizon: i64,
    ) -> PyResult<()> {
        let mut covered = BTreeSet::new();
        for grant in grants {
            if !grant.allowed
                || grant.expiry.micros <= horizon
                || snapshot.revoked_grants.contains(&grant.revocation_ref)
            {
                return Err(invalid(
                    "required independent grant is denied, expired, or revoked",
                ));
            }
            covered.extend(grant.dependencies.iter().map(String::as_str));
            if let Some(purpose) = &grant.learning_purpose {
                for (split, scope) in &self.source_partitions {
                    if (purpose == "current-request-fast-adaptation"
                        && (split.as_deref() != Some("request-local-replay")
                            || scope != "request-local"))
                        || (purpose == "durable-slow-consolidation"
                            && (!matches!(split.as_deref(), Some("train" | "dev"))
                                || scope == "request-local"))
                    {
                        return Err(invalid("training grant is incompatible with an observed source's immutable split or replay scope"));
                    }
                }
                for live in &self.live {
                    let fields = live.canonical.fields(16)?;
                    if !fields[8].strings()?.contains(purpose)
                        || (purpose == "durable-slow-consolidation"
                            && (!fields[13].boolean()? || !fields[14].boolean()?))
                    {
                        return Err(invalid(
                            "training grant exceeds the live data-use learning purpose",
                        ));
                    }
                }
                for row in &self.replay {
                    let split = row.identity.fields(6)?[0].fields(9)?[2].text()?;
                    if (purpose == "current-request-fast-adaptation"
                        && split != "request-local-replay")
                        || (purpose == "durable-slow-consolidation"
                            && !matches!(split, "train" | "dev"))
                    {
                        return Err(invalid(
                            "training grant is incompatible with the immutable replay split",
                        ));
                    }
                }
            }
        }
        if self
            .dependencies
            .iter()
            .any(|node| !covered.contains(node.identity.as_str()))
        {
            return Err(invalid(
                "required independent grant does not cover the full dependency closure",
            ));
        }
        Ok(())
    }

    fn check_use(
        &self,
        operation: &str,
        snapshot: &AuthoritySnapshot,
        before_segment: bool,
    ) -> PyResult<()> {
        let end = self.check_snapshot(snapshot)?;
        if before_segment && snapshot.observed_at.micros >= end {
            return Err(invalid("preregistered segment end is not in the future"));
        }
        let horizon = if before_segment {
            end
        } else {
            snapshot.observed_at.micros
        };
        self.check_grants(&self.publication, snapshot, horizon)?;
        match operation {
            "inference" => self.check_grants(&self.inference, snapshot, horizon),
            // Fresh observations need both admitted material and its validated
            // position mapping. Neither alternative issues learning rights.
            "training" if !self.replay.is_empty()
                || (!self.initial_sources.is_empty() && !self.source_mapping.is_empty()) => {
                if !self.sources_allow_learning {
                    return Err(invalid("observed source admission excludes learning or replay"));
                }
                self.check_grants(&self.training, snapshot, horizon)
            }
            _ => Err(invalid(
                "operation must be inference or training with verified replay rows or admitted initial sources and mapping",
            )),
        }
    }
}

/// Privileged, constructing-thread controller for one real native Session.
///
/// The application, not generated code, must construct all primitive arguments
/// from already resolved authority and verified replay values. This boundary is
/// not a sandbox or cryptographic issuer verifier. Trusted application-supplied
/// replay callbacks restore and retire the original invocation; they cannot
/// issue rights. Cold metadata accepts closed primitive values, not custom
/// conversion handlers. The Session exposes no rights-issuing API.
#[pyclass(
    name = "SemanticTransitionController",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticTransitionController {
    session: Py<PySemanticTransitionSession>,
    identity: Arc<()>,
}

#[derive(Clone, Debug)]
enum TaskUsePhase {
    Importing {
        operation: String,
        reads: bool,
    },
    Imported,
    Recording {
        scope: Arc<()>,
        operation: Option<String>,
    },
    Prepared {
        scope: Arc<()>,
        operation: Option<String>,
    },
    Segment(String),
    Refused,
}

struct TaskUseState {
    phase: TaskUsePhase,
    snapshot: AuthoritySnapshot,
}

/// Process-local resource authority, independent of restored native epochs.
/// Every import reserves one new generation before parsing or rebinding state.
#[derive(Clone)]
struct TaskIssuance {
    counter: Arc<AtomicU64>,
    generation: u64,
}

impl TaskIssuance {
    fn issue(counter: Arc<AtomicU64>) -> PyResult<Self> {
        let previous = counter
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |generation| {
                generation.checked_add(1)
            })
            .map_err(|_| invalid("native task issuance generation is exhausted"))?;
        Ok(Self {
            counter,
            generation: previous + 1,
        })
    }

    fn require_current(&self) -> PyResult<()> {
        if self.counter.load(Ordering::Acquire) != self.generation {
            return Err(invalid("task use was invalidated by a later import"));
        }
        Ok(())
    }

    fn same_as(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.counter, &other.counter) && self.generation == other.generation
    }
}

impl TaskUseState {
    fn begin_build(&mut self) -> PyResult<Arc<()>> {
        let operation = match &self.phase {
            TaskUsePhase::Imported => None,
            TaskUsePhase::Segment(operation) => Some(operation.clone()),
            _ => {
                return Err(invalid(
                    "segment recording cannot be nested or resume a refused use",
                ))
            }
        };
        let scope = Arc::new(());
        self.phase = TaskUsePhase::Recording {
            scope: Arc::clone(&scope),
            operation,
        };
        Ok(scope)
    }

    fn prepared_content_binding(&self, scope: &Arc<()>) -> PyResult<(Option<String>, Vec<u8>)> {
        let operation = match &self.phase {
            TaskUsePhase::Recording { scope: current, .. }
            | TaskUsePhase::Prepared { scope: current, .. }
                if Arc::ptr_eq(current, scope) =>
            {
                None
            }
            // Native retained-step custody is additionally required by every
            // caller; a completed step's witnesses outlive its recording scope.
            TaskUsePhase::Segment(operation) => Some(operation.clone()),
            _ => {
                return Err(invalid(
                    "prepared content belongs to no active or completed recording",
                ))
            }
        };
        Ok((operation, self.snapshot.canonical.clone()))
    }

    fn finish_build(&mut self, scope: &Arc<()>) -> PyResult<()> {
        let TaskUsePhase::Recording {
            scope: current,
            operation,
        } = &self.phase
        else {
            return Err(invalid("only the original recording may finish"));
        };
        if !Arc::ptr_eq(current, scope) {
            return Err(invalid("segment recording scope changed"));
        }
        self.phase = TaskUsePhase::Prepared {
            scope: Arc::clone(scope),
            operation: operation.clone(),
        };
        Ok(())
    }

    fn admit_built_segment(
        &mut self,
        scope: &Arc<()>,
        authority: &TaskAuthority,
        operation: String,
        snapshot: AuthoritySnapshot,
    ) -> PyResult<()> {
        let TaskUsePhase::Prepared {
            scope: current,
            operation: original,
        } = &self.phase
        else {
            return Err(invalid(
                "built segment may be submitted exactly once after recording",
            ));
        };
        if !Arc::ptr_eq(current, scope)
            || original.as_ref().is_some_and(|value| value != &operation)
        {
            return Err(invalid(
                "built segment changed its scope or original operation",
            ));
        }
        // Reuse the ordinary authoritative snapshot check. No callback or native
        // operation may intervene before the caller's one actual submission.
        self.phase = original.as_ref().map_or(TaskUsePhase::Imported, |value| {
            TaskUsePhase::Segment(value.clone())
        });
        self.admit_segment(authority, operation, snapshot)
    }

    fn finish_import(
        &mut self,
        authority: &TaskAuthority,
        snapshot: AuthoritySnapshot,
    ) -> PyResult<()> {
        let result = match &self.phase {
            TaskUsePhase::Importing {
                operation,
                reads: false,
            } => snapshot
                .newer_than(&self.snapshot)
                .and_then(|()| authority.check_use(operation, &snapshot, false)),
            _ => Err(invalid("only the closed original import may finish")),
        };
        match result {
            Ok(()) => {
                self.snapshot = snapshot;
                self.phase = TaskUsePhase::Imported;
                Ok(())
            }
            Err(error) => {
                self.phase = TaskUsePhase::Refused;
                Err(error)
            }
        }
    }

    fn require_read(&self) -> PyResult<()> {
        match &self.phase {
            TaskUsePhase::Imported
            | TaskUsePhase::Segment(_)
            | TaskUsePhase::Importing { reads: true, .. } => Ok(()),
            _ => Err(invalid(
                "task use is not available outside its controlled import",
            )),
        }
    }

    fn require_read_during_import(&self, importing: bool) -> PyResult<()> {
        if importing != matches!(&self.phase, TaskUsePhase::Importing { .. }) {
            return Err(invalid(
                "task reader does not belong to the active import state",
            ));
        }
        self.require_read()
    }

    fn require_public_use(&self) -> PyResult<()> {
        match &self.phase {
            TaskUsePhase::Imported | TaskUsePhase::Segment(_) => Ok(()),
            _ => Err(invalid(
                "unfinished or refused import grants no public operation",
            )),
        }
    }

    fn set_import_reads(&mut self, enabled: bool) -> PyResult<()> {
        let TaskUsePhase::Importing { reads, .. } = &mut self.phase else {
            return Err(invalid("task use is not in the original unfinished import"));
        };
        if *reads == enabled {
            return Err(invalid("import read scope cannot be nested or reused"));
        }
        *reads = enabled;
        Ok(())
    }

    fn content_handoff_binding(&self) -> PyResult<(String, Vec<u8>)> {
        let operation = match &self.phase {
            TaskUsePhase::Segment(operation)
            | TaskUsePhase::Importing {
                operation,
                reads: true,
            } => operation,
            _ => {
                return Err(invalid(
                    "tensor content requires an admitted segment or controlled import",
                ))
            }
        };
        Ok((operation.clone(), self.snapshot.canonical.clone()))
    }

    fn admit_segment(
        &mut self,
        authority: &TaskAuthority,
        operation: String,
        snapshot: AuthoritySnapshot,
    ) -> PyResult<()> {
        if !matches!(&self.phase, TaskUsePhase::Imported)
            && !matches!(&self.phase, TaskUsePhase::Segment(previous) if previous == &operation)
        {
            return Err(invalid(
                "task use cannot change its original operation or resume a refused use",
            ));
        }
        let result = snapshot
            .newer_than(&self.snapshot)
            .and_then(|()| authority.check_use(&operation, &snapshot, true));
        match result {
            Ok(()) => {
                self.phase = TaskUsePhase::Segment(operation);
                self.snapshot = snapshot;
                Ok(())
            }
            Err(error) => {
                self.phase = TaskUsePhase::Refused;
                Err(error)
            }
        }
    }

    fn revalidate_activation(
        &mut self,
        authority: &TaskAuthority,
        snapshot: AuthoritySnapshot,
    ) -> PyResult<()> {
        let TaskUsePhase::Segment(operation) = &self.phase else {
            return Err(invalid("task use has no admitted segment"));
        };
        let result = snapshot
            .newer_than(&self.snapshot)
            .and_then(|()| authority.check_use(operation, &snapshot, false));
        match result {
            Ok(()) => {
                self.snapshot = snapshot;
                Ok(())
            }
            Err(error) => {
                self.phase = TaskUsePhase::Refused;
                Err(error)
            }
        }
    }
}

use xlog_cuda::SemanticFeedbackSchema as FeedbackSchema;

fn feedback_schema_object(py: Python<'_>, schema: &FeedbackSchema) -> PyResult<Py<PyTuple>> {
    let mut adapter = Sha256::new();
    adapter.update(b"xlog.python.typed-feedback-adapter.v1\0");
    adapter.update(SemanticTransitionSession::feedback_adapter_identity().as_bytes());
    adapter.update(include_bytes!("semantic_transition.rs"));
    let adapter: [u8; 32] = adapter.finalize().into();
    let fields = PyTuple::new(
        py,
        (0..schema.statement_bytes)
            .map(|offset| ("statement", offset, 8u8, false))
            .chain([("record", 32, 1, true), ("record", 40, 1, true)])
            .collect::<Vec<_>>(),
    )?;
    Ok((
        1u64,
        PyBytes::new(py, &schema.identity),
        PyBytes::new(py, &adapter),
        schema.statement_bytes,
        schema.feature_width,
        2u64,
        fields,
    )
        .into_pyobject(py)?
        .unbind())
}

/// A controller-issued, task-bound use. It cannot issue or refresh grants.
/// It retains its real Session, nonreused issuance and exact native task epoch.
/// Transferable ownership does not permit off-thread orchestration or reads.
#[pyclass(name = "SemanticTransitionTaskUse", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticTransitionTaskUse {
    session: Py<PySemanticTransitionSession>,
    controller: Arc<()>,
    issuance: TaskIssuance,
    importing: Arc<AtomicBool>,
    task_identity: [u8; 32],
    task_epoch: u64,
    authority: TaskAuthority,
    state: Mutex<TaskUseState>,
}

impl PySemanticTransitionTaskUse {
    fn state(&self) -> PyResult<MutexGuard<'_, TaskUseState>> {
        self.state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("task-use state mutex is poisoned"))
    }

    fn require_current(&self, owner: &SemanticTransitionSession) -> PyResult<()> {
        self.require_identity(owner)?;
        self.state()?
            .require_read_during_import(self.importing.load(Ordering::Acquire))
    }

    fn require_identity(&self, owner: &SemanticTransitionSession) -> PyResult<()> {
        self.issuance.require_current()?;
        if owner.is_poisoned() {
            return Err(invalid("native semantic session is poisoned"));
        }
        if owner.task_evaluation_epoch() != self.task_epoch
            || owner
                .task_evaluation_identity()
                .is_none_or(|identity| identity.as_bytes() != &self.task_identity)
        {
            return Err(invalid("task use is stale after native task rebinding"));
        }
        Ok(())
    }
}

/// The shared Session rejects reentrant imports from any controller. Only a
/// fully verified result commits this guard; all other exits poison the owner.
struct ColdImportGuard<'a> {
    session: &'a PySemanticTransitionSession,
    issuance: TaskIssuance,
    original_decisions: Option<Arc<[u8]>>,
    committed: bool,
}

impl<'a> ColdImportGuard<'a> {
    fn begin(
        session: &'a PySemanticTransitionSession,
        original_decisions: Option<Arc<[u8]>>,
    ) -> PyResult<Self> {
        let mut owner = session.owner()?;
        if session.importing.load(Ordering::Acquire)
            || session.recording.load(Ordering::Acquire)
            || session.retiring.load(Ordering::Acquire)
            || owner.is_poisoned()
        {
            owner.abort();
            return Err(invalid(
                "native session cannot enter a nested or aborted import",
            ));
        }
        let issuance = match TaskIssuance::issue(Arc::clone(&session.issuance)) {
            Ok(issuance) => issuance,
            Err(error) => {
                owner.abort();
                return Err(error);
            }
        };
        session.importing.store(true, Ordering::Release);
        Ok(Self {
            session,
            issuance,
            original_decisions,
            committed: false,
        })
    }
}

impl Drop for ColdImportGuard<'_> {
    fn drop(&mut self) {
        let mut owner = self
            .session
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !self.committed {
            owner.abort();
        }
        self.session.importing.store(false, Ordering::Release);
    }
}

/// Dynamic access to genuine restored readers, never controller execution.
/// No state lock survives into Python. Saved references lose this access on
/// every normal/error/unwind exit; a refused phase cannot be resurrected.
struct ImportReadScope<'a> {
    task_use: &'a PySemanticTransitionTaskUse,
}

impl<'a> ImportReadScope<'a> {
    fn enter(task_use: &'a PySemanticTransitionTaskUse) -> PyResult<Self> {
        task_use.issuance.require_current()?;
        if !task_use.importing.load(Ordering::Acquire) {
            return Err(invalid("controlled import reads require an active import"));
        }
        task_use.state()?.set_import_reads(true)?;
        Ok(Self { task_use })
    }
}

impl Drop for ImportReadScope<'_> {
    fn drop(&mut self) {
        if let Ok(mut state) = self.task_use.state() {
            if let TaskUsePhase::Importing { reads, .. } = &mut state.phase {
                *reads = false;
            }
        }
    }
}

/// A reader of one actually acquired native bank, never a caller-built view.
/// Keep it alive through the original model invocation and every exported alias.
/// Retire bank reads separately from the original step's final consumers when
/// late backward must survive bank reuse. Final release destroys step custody.
#[pyclass(name = "SemanticPublishedParent", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticPublishedParent {
    session: Py<PySemanticTransitionSession>,
    task_use: Py<PySemanticTransitionTaskUse>,
    inner: Mutex<SemanticPublishedLease>,
    // Policy invocations retain this actual parent through late backward. Keep
    // original autograd producers without a parent-to-witness ownership cycle.
    continuation_producers: Mutex<Vec<Py<PyAny>>>,
}

/// Native physical allocation origin of one exported tensor view. Instances
/// come only from an original parent or prepared step's tensor_allocation(),
/// never Python input.
/// Retains allocation lifetime but not reader access or immutable content.
/// Allocation ownership and external tensor version counters are independent.
#[pyclass(name = "NativeTensorAllocation", module = "pyxlog._native", frozen)]
pub(crate) struct PyNativeTensorAllocation {
    provenance: DeviceAllocationProvenance,
}

#[pymethods]
impl PyNativeTensorAllocation {
    /// Whether both views retain the same actual native allocation owner.
    fn same_allocation(&self, other: &Self) -> bool {
        self.provenance.same_allocation(&other.provenance)
    }

    #[getter]
    fn allocation_bytes(&self) -> u64 {
        self.provenance.allocation_bytes()
    }

    #[getter]
    fn byte_offset(&self) -> u64 {
        self.provenance.byte_offset()
    }

    #[getter]
    fn view_bytes(&self) -> u64 {
        self.provenance.view_bytes()
    }
}

/// Session-issued storage for one bounded recorded step, not an acquired parent.
/// No host publication identity is available before device execution. The
/// original Runtime retains this handle and its producers through backward.
#[pyclass(name = "SemanticPreparedStep", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticPreparedStep {
    session: Py<PySemanticTransitionSession>,
    task_use: Py<PySemanticTransitionTaskUse>,
    scope: Arc<()>,
    inner: SemanticPreparedStep,
    continuation_producers: Mutex<[Vec<Py<PyAny>>; 2]>,
    #[cfg(feature = "semantic-policy")]
    policy_inputs: Mutex<[Option<PreparedPolicyInputs>; 2]>,
}

#[cfg(feature = "semantic-policy")]
struct PreparedPolicyInputs {
    model_output: Py<PyAny>,
    inputs: [Py<PyAny>; 4],
    invocation_issued: bool,
}

#[cfg(feature = "semantic-policy")]
impl PreparedPolicyInputs {
    fn issue_originals(&mut self, py: Python<'_>) -> PyResult<(Py<PyAny>, [Py<PyAny>; 4])> {
        if self.invocation_issued {
            return Err(invalid("prepared policy invocation was already issued"));
        }
        // Keep the full original graph on the step for late exported adjoints.
        // No getter, tensor copy, new leaf or second numerical invocation runs.
        let output = self.model_output.clone_ref(py);
        let inputs = std::array::from_fn(|index| self.inputs[index].clone_ref(py));
        self.invocation_issued = true;
        Ok((output, inputs))
    }
}

impl PySemanticPreparedStep {
    fn require_recording_stream(
        &self,
        owner: &SemanticTransitionSession,
        stream: u64,
    ) -> PyResult<()> {
        let native = owner
            .prepared_stream(&self.inner)
            .map_err(xlog_err)?
            .cu_stream() as u64;
        let declared = if stream == 1 { 0 } else { stream };
        if declared != native {
            return Err(invalid(
                "recorded producer handoff requires the original native capture stream",
            ));
        }
        Ok(())
    }

    fn content_binding_with_owner(
        &self,
        py: Python<'_>,
        owner: &SemanticTransitionSession,
    ) -> PyResult<(Option<String>, Vec<u8>)> {
        let task_use = self.task_use.borrow(py);
        task_use.require_identity(owner)?;
        owner.prepared_stream(&self.inner).map_err(xlog_err)?;
        let binding = task_use.state()?.prepared_content_binding(&self.scope)?;
        Ok(binding)
    }

    fn completed_observation_streams(
        &self,
        py: Python<'_>,
        consumer_streams: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<u64>> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        if session.importing.load(Ordering::Acquire)
            || session.recording.load(Ordering::Acquire)
            || session.retiring.load(Ordering::Acquire)
        {
            return Err(invalid(
                "completed input observation cannot overlap recording, import or retirement",
            ));
        }
        let mut budget = 16 * 1024 * 1024;
        let streams = ColdValue::read(consumer_streams, &mut budget, 0)?;
        let streams = streams
            .sequence()?
            .iter()
            .map(ColdValue::unsigned)
            .collect::<PyResult<Vec<_>>>()?;
        if streams.is_empty() {
            return Err(invalid(
                "completed input observation requires the original consumer streams",
            ));
        }
        let owner = session.owner()?;
        if self.content_binding_with_owner(py, &owner)?.0.is_none() {
            return Err(invalid(
                "completed input observation requires the admitted original operation",
            ));
        }
        Ok(streams)
    }

    fn require_task(&self, py: Python<'_>, task_use: &PySemanticTransitionTaskUse) -> PyResult<()> {
        let issued = self.task_use.borrow(py);
        issued.issuance.require_current()?;
        if self.session.as_ptr() != task_use.session.as_ptr()
            || !Arc::ptr_eq(&issued.controller, &task_use.controller)
            || !issued.issuance.same_as(&task_use.issuance)
            || issued.task_epoch != task_use.task_epoch
            || issued.task_identity != task_use.task_identity
        {
            return Err(invalid("prepared step belongs to another task use"));
        }
        Ok(())
    }

    fn export(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
        export: impl FnOnce(
            &mut SemanticTransitionSession,
            &SemanticPreparedStep,
            u64,
        ) -> Result<DlpackManagedTensor, xlog_cuda::SemanticTransitionError>,
    ) -> PyResult<Py<PyAny>> {
        self.session.borrow(py).require_creator()?;
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        self.content_binding_with_owner(py, &owner)?;
        let tensor = export(&mut owner, &self.inner, stream).map_err(xlog_err)?;
        crate::dlpack_capsule_from_tensor(
            py,
            retain_export_owner(tensor, self.session.clone_ref(py), session.owner_thread)?,
        )
    }
}

#[pymethods]
impl PySemanticPreparedStep {
    /// The Session's immutable nominal mode, readable during cold preparation
    /// and recording. This is not the device-acquired effective mode or a grant
    /// to execute; DRAIN_REQUIRED may override either mode after submission.
    #[getter]
    fn requested_transition(&self, py: Python<'_>) -> PyResult<&'static str> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        self.content_binding_with_owner(py, &owner)?;
        match owner
            .prepared_transition_kind(&self.inner)
            .map_err(xlog_err)?
        {
            SemanticTransitionKind::Proposal => Ok("proposal"),
            SemanticTransitionKind::Recompute => Ok("recompute"),
            SemanticTransitionKind::Update => Ok("update"),
            SemanticTransitionKind::Drain => {
                Err(invalid("drain is not a cold scheduled transition"))
            }
        }
    }

    /// Cold fixed-address Source U64[32,8]; populated by this step's device acquire.
    #[pyo3(signature = (*, consumer_stream))]
    fn source(&self, py: Python<'_>, consumer_stream: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.export(
            py,
            consumer_stream,
            SemanticTransitionSession::prepared_source,
        )
    }

    /// Cold full-capacity prefix view for one capture-time bank branch. Only
    /// the device-selected branch and prefix extent are live during execution.
    #[pyo3(signature = (bank, *, consumer_stream))]
    fn prefix(
        &self,
        py: Python<'_>,
        bank: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.session.borrow(py).require_creator()?;
        let bank = usize::try_from(ColdValue::read(bank, &mut 128, 0)?.unsigned()?)
            .map_err(|_| invalid("prepared tensor bank exceeds native address space"))?;
        self.export(py, consumer_stream, |owner, step, stream| {
            owner.prepared_prefix(step, bank, stream)
        })
    }

    #[pyo3(signature = (*, consumer_stream))]
    fn prefix_extent(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.export(
            py,
            consumer_stream,
            SemanticTransitionSession::prepared_prefix_extent,
        )
    }

    #[pyo3(signature = (*, consumer_stream))]
    fn ring_head(&self, py: Python<'_>, consumer_stream: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.export(
            py,
            consumer_stream,
            SemanticTransitionSession::prepared_ring_head,
        )
    }

    /// Original-sealed U64[1] terminal state of the actual device parent.
    /// This is a private input snapshot, never an alias of the authority lease.
    #[pyo3(signature = (*, consumer_stream))]
    fn terminal_state(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.export(
            py,
            consumer_stream,
            SemanticTransitionSession::prepared_terminal,
        )
    }

    #[pyo3(signature = (role, index, bank, *, consumer_stream))]
    fn tensor(
        &self,
        py: Python<'_>,
        role: &Bound<'_, PyAny>,
        index: &Bound<'_, PyAny>,
        bank: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.session.borrow(py).require_creator()?;
        let role = ColdValue::read(role, &mut 128, 0)?.unsigned()?;
        let role = SemanticStateRole::from_code(role)
            .ok_or_else(|| invalid("unknown native publication tensor role"))?;
        let index = ColdValue::read(index, &mut 128, 0)?.unsigned()?;
        let bank = usize::try_from(ColdValue::read(bank, &mut 128, 0)?.unsigned()?)
            .map_err(|_| invalid("prepared tensor bank exceeds native address space"))?;
        self.export(py, consumer_stream, |owner, step, stream| {
            owner.prepared_tensor(step, role, index, bank, stream)
        })
    }

    /// Return (native provenance, full allocation U8 DLPack capsule) for the
    /// exact prepared tensor view. The capsule retains the original step grant;
    /// consume once and retain through every use, including original backward.
    /// Its full backing is read-only by contract, never an update destination.
    /// Capacity does not certify live input content. Version-counter origins
    /// belong to the model producer, independently of this physical owner.
    #[pyo3(signature = (role, index, bank, *, consumer_stream))]
    fn tensor_allocation(
        &self,
        py: Python<'_>,
        role: &Bound<'_, PyAny>,
        index: &Bound<'_, PyAny>,
        bank: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        let role = SemanticStateRole::from_code(ColdValue::read(role, &mut 128, 0)?.unsigned()?)
            .ok_or_else(|| invalid("unknown native publication tensor role"))?;
        let index = ColdValue::read(index, &mut 128, 0)?.unsigned()?;
        let bank = usize::try_from(ColdValue::read(bank, &mut 128, 0)?.unsigned()?)
            .map_err(|_| invalid("prepared tensor bank exceeds native address space"))?;
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        self.content_binding_with_owner(py, &owner)?;
        let (provenance, tensor) = owner
            .prepared_tensor_allocation(&self.inner, role, index, bank, stream)
            .map_err(xlog_err)?;
        let tensor = retain_export_owner(tensor, self.session.clone_ref(py), session.owner_thread)?;
        Ok((
            Py::new(py, PyNativeTensorAllocation { provenance })?,
            crate::dlpack_capsule_from_tensor(py, tensor)?,
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// Fixed device outputs of this Update step's native replay selection.
    /// The first U64 tensor is the selection/status record. It is followed by the
    /// complete roster metadata, frozen objective, reduction groups, group member
    /// ordinals and canaries. Remaining rank-two ports are token IDs, mask labels,
    /// mask weights, autoregressive labels, retention labels, source slots, logical
    /// positions, kinds, and parents for every replay row.
    #[pyo3(signature = (*, consumer_stream))]
    fn training_view(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        let ports = [
            SemanticTrainingViewPort::Selection,
            SemanticTrainingViewPort::RosterRows,
            SemanticTrainingViewPort::Objective,
            SemanticTrainingViewPort::ObjectiveGroups,
            SemanticTrainingViewPort::ObjectiveGroupMembers,
            SemanticTrainingViewPort::Canaries,
            SemanticTrainingViewPort::TokenIds,
            SemanticTrainingViewPort::MaskLabels,
            SemanticTrainingViewPort::MaskWeights,
            SemanticTrainingViewPort::AutoregressiveLabels,
            SemanticTrainingViewPort::RetentionLabels,
            SemanticTrainingViewPort::SourceSlots,
            SemanticTrainingViewPort::LogicalPositions,
            SemanticTrainingViewPort::Kinds,
            SemanticTrainingViewPort::Parents,
        ];
        let mut values = Vec::with_capacity(ports.len());
        for port in ports {
            values.push(self.export(py, consumer_stream, |owner, step, stream| {
                owner.prepared_training_view_port(step, port, stream)
            })?);
        }
        Ok(PyTuple::new(py, values)?.unbind())
    }

    /// Record the device-selected actor, critic and cost VJP for this prepared
    /// Proposal inside a later prepared Update branch. The returned parameter,
    /// full MASK and component-baseline roots remain device-resident; the
    /// native proposal lease predicates the unselected bank to exact zeros.
    #[cfg(feature = "semantic-policy")]
    #[pyo3(signature = (update_step, bank, *, consumer_stream))]
    fn temporal_selected_vjp(
        &self,
        py: Python<'_>,
        update_step: Py<PySemanticPreparedStep>,
        bank: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        let bank = usize::try_from(ColdValue::read(bank, &mut 128, 0)?.unsigned()?)
            .map_err(|_| invalid("prepared policy bank exceeds native address space"))?;
        let consumer_stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        {
            let update = update_step.borrow(py);
            if self.session.as_ptr() != update.session.as_ptr()
                || !Arc::ptr_eq(&self.scope, &update.scope)
            {
                return Err(invalid(
                    "temporal policy backward requires an Update from the same prepared segment",
                ));
            }
            let task_use = self.task_use.borrow(py);
            update.require_task(py, &task_use)?;
        }
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        self.content_binding_with_owner(py, &owner)?;
        let gradients = owner
            .record_selected_prepared_policy_vjp(
                &self.inner,
                bank,
                &update_step.borrow(py).inner,
                consumer_stream,
            )
            .map_err(xlog_err)?;
        let [parameters, text, baselines] = gradients.into_dlpack().map_err(xlog_err)?;
        let parameters =
            retain_export_owner(parameters, self.session.clone_ref(py), session.owner_thread)?;
        let text = retain_export_owner(text, self.session.clone_ref(py), session.owner_thread)?;
        let baselines =
            retain_export_owner(baselines, self.session.clone_ref(py), session.owner_thread)?;
        Ok((
            crate::dlpack_capsule_from_tensor(py, parameters)?,
            crate::dlpack_capsule_from_tensor(py, text)?,
            crate::dlpack_capsule_from_tensor(py, baselines)?,
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// Export an admitted immutable record; bank-varying records require their
    /// dedicated device-produced input port rather than a guessed cold extent.
    #[pyo3(signature = (role, index, *, consumer_stream))]
    fn record(
        &self,
        py: Python<'_>,
        role: &Bound<'_, PyAny>,
        index: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.session.borrow(py).require_creator()?;
        let role = SemanticStateRole::from_code(ColdValue::read(role, &mut 128, 0)?.unsigned()?)
            .ok_or_else(|| invalid("unknown native publication record role"))?;
        let index = ColdValue::read(index, &mut 128, 0)?.unsigned()?;
        self.export(py, consumer_stream, |owner, step, stream| {
            owner.prepared_record(step, role, index, stream)
        })
    }

    fn range_keys(&self, py: Python<'_>) -> PyResult<Vec<(u64, u64)>> {
        let session = self.session.borrow(py);
        let owner = session.owner()?;
        self.content_binding_with_owner(py, &owner)?;
        Ok(owner
            .prepared_range_keys(&self.inner)
            .map_err(xlog_err)?
            .into_iter()
            .map(|(role, index)| (role as u64, index))
            .collect())
    }

    #[pyo3(signature = (*, ranges, consumer_stream))]
    fn guard_content(
        &self,
        py: Python<'_>,
        ranges: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.session.borrow(py).require_creator()?;
        let mut budget = 16 * 1024 * 1024;
        let ranges = parse_content_ranges(&ColdValue::read(ranges, &mut budget, 0)?)?;
        let stream = parse_witness_consumer_stream(consumer_stream, &mut budget)?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        self.content_binding_with_owner(py, &owner)?;
        owner
            .guard_prepared_content(&self.inner, &ranges, stream)
            .map_err(xlog_err)
    }

    /// Immutable admitted topology/table identities, generation and capacities.
    /// Dynamic parent epochs, words, extents and terminal state are device ports.
    fn model_geometry(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let session = self.session.borrow(py);
        let owner = session.owner()?;
        self.content_binding_with_owner(py, &owner)?;
        let (topology, table, generation, prefix, feedback, position) = owner
            .prepared_model_geometry(&self.inner)
            .map_err(xlog_err)?;
        Ok((
            PyBytes::new(py, topology.as_bytes()),
            PyBytes::new(py, table.as_bytes()),
            generation,
            prefix,
            feedback,
            position,
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// Cold authenticated binding of this completed step's original model.
    /// Returns (predecessor_identity, successor_identity_or_None, model_generation,
    /// model_geometry_digest, model_numerical_digest). Identities have the same
    /// four fields as PublishedParent.identity(); they are not live readers.
    /// Join every original consumer stream. Read record 44 and complete model
    /// allocations through this step's existing exports, then call again after
    /// serialization to verify the same sealed inputs. Unknown completion and
    /// skipped steps cannot expose a binding; a refusal has no successor.
    #[pyo3(signature = (*, consumer_streams))]
    fn completed_model_binding(
        &self,
        py: Python<'_>,
        consumer_streams: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        let streams = self.completed_observation_streams(py, consumer_streams)?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        let (predecessor, successor, generation, geometry, numerical) = owner
            .prepared_model_binding(&self.inner, &streams)
            .map_err(xlog_err)?;
        let identity = |value: xlog_cuda::SemanticPublishedIdentity| {
            (
                PyBytes::new(py, value.instance.as_bytes()),
                value.word,
                PyBytes::new(py, value.logical_digest.as_bytes()),
                PyBytes::new(py, value.state_digest.as_bytes()),
            )
        };
        Ok((
            identity(predecessor),
            successor.map(identity),
            generation,
            PyBytes::new(py, geometry.as_bytes()),
            PyBytes::new(py, numerical.as_bytes()),
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// Cold original records, not encoded features or current-bank substitutes.
    /// Returns (predecessor_identity, successor_identity_or_None, records), with
    /// records ordered by (role, index). Each record is (role, index, generation,
    /// capacity_bytes, logical_begin, logical_end, native_identity, raw_bytes).
    /// Native identity is the original range seal; a carrier hashes raw_bytes
    /// separately for bytes_sha256. No new model or publication identity issues.
    #[pyo3(signature = (*, consumer_streams))]
    fn completed_feedback_records(
        &self,
        py: Python<'_>,
        consumer_streams: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        let streams = self.completed_observation_streams(py, consumer_streams)?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        let (predecessor, successor, records) = owner
            .prepared_feedback_records(&self.inner, &streams)
            .map_err(xlog_err)?;
        let identity = |value: xlog_cuda::SemanticPublishedIdentity| {
            (
                PyBytes::new(py, value.instance.as_bytes()),
                value.word,
                PyBytes::new(py, value.logical_digest.as_bytes()),
                PyBytes::new(py, value.state_digest.as_bytes()),
            )
        };
        let records = PyTuple::new(
            py,
            records.iter().map(|record| {
                (
                    record.role as u64,
                    record.index,
                    record.generation,
                    record.capacity_bytes,
                    record.logical_begin,
                    record.logical_end,
                    PyBytes::new(py, record.identity.as_bytes()),
                    PyBytes::new(py, &record.bytes),
                )
            }),
        )?;
        Ok((identity(predecessor), successor.map(identity), records)
            .into_pyobject(py)?
            .unbind())
    }

    /// Original imported task lineage, with no invented publication identity.
    fn dependency_lineage(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let session = self.session.borrow(py);
        let owner = session.owner()?;
        self.content_binding_with_owner(py, &owner)?;
        let task_use = self.task_use.borrow(py);
        Ok((
            task_use.task_epoch,
            dependency_lineage_object(py, &task_use.authority.dependencies)?,
            PyTuple::new(py, &task_use.authority.feedback_roots)?,
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// Reserve a producer-derived upper bound before capture; no device allocator
    /// runs when original autograd tensors are first bound during recording.
    fn reserve_tensor_content(&self, py: Python<'_>, capacity: &Bound<'_, PyAny>) -> PyResult<()> {
        self.session.borrow(py).require_creator()?;
        let capacity = usize::try_from(ColdValue::read(capacity, &mut 128, 0)?.unsigned()?)
            .map_err(|_| invalid("tensor witness capacity exceeds native address space"))?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        self.content_binding_with_owner(py, &owner)?;
        owner
            .reserve_tensor_content(&self.inner, capacity)
            .map_err(xlog_err)
    }

    /// Reserve event slots before capture; no additional model forward occurs.
    fn reserve_model_work(&self, py: Python<'_>, capacity: &Bound<'_, PyAny>) -> PyResult<()> {
        self.session.borrow(py).require_creator()?;
        let capacity = usize::try_from(ColdValue::read(capacity, &mut 128, 0)?.unsigned()?)
            .map_err(|_| invalid("model work capacity exceeds native address space"))?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        self.content_binding_with_owner(py, &owner)?;
        owner
            .reserve_model_work(&self.inner, capacity)
            .map_err(xlog_err)
    }

    /// Cold native-owned U64[capacity,3] work scratch, retained by this step.
    /// Rows are [actual_units, sticky_overflow, producer_writes]; the producer
    /// must count every reached category, including predicates and padding.
    #[pyo3(signature = (*, consumer_stream))]
    fn model_work_buffer(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.export(
            py,
            consumer_stream,
            SemanticTransitionSession::model_work_buffer,
        )
    }

    /// Record a bounded device-produced category and reset its original slot.
    /// Returns the row index in model_work_buffer, not a device result. Call
    /// before the producer in the same active original recording stream. A
    /// closed forward recording cannot receive work from a later invocation.
    fn record_model_device_work(
        &self,
        py: Python<'_>,
        kind: &Bound<'_, PyAny>,
        upper_dimensions: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        self.session.borrow(py).require_creator()?;
        let (kind, values, rank) = parse_model_work(kind, upper_dimensions)?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        self.content_binding_with_owner(py, &owner)?;
        owner
            .record_model_device_work(&self.inner, kind, &values[..rank])
            .map_err(xlog_err)
    }

    /// Record one original executed operation's unit category and geometry.
    /// The native transition freezes the roster when enqueue_step returns.
    fn record_model_work(
        &self,
        py: Python<'_>,
        kind: &Bound<'_, PyAny>,
        dimensions: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.session.borrow(py).require_creator()?;
        let (kind, values, rank) = parse_model_work(kind, dimensions)?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        self.content_binding_with_owner(py, &owner)?;
        owner
            .record_model_work(&self.inner, kind, &values[..rank])
            .map_err(xlog_err)
    }

    /// Charge original autograd save occurrences using retained native layouts,
    /// never a Python byte total or a later replacement tensor.
    fn record_saved_tensor_work(
        &self,
        py: Python<'_>,
        witness: PyRef<'_, PySemanticTensorContentWitness>,
    ) -> PyResult<()> {
        self.session.borrow(py).require_creator()?;
        if witness.session.as_ptr() != self.session.as_ptr() {
            return Err(invalid("saved work belongs to another Session"));
        }
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        self.content_binding_with_owner(py, &owner)?;
        owner
            .record_saved_tensor_work(&self.inner, &witness.inner)
            .map_err(xlog_err)
    }

    /// Original native feedback producer ports: schema and six DLPack capsules.
    #[pyo3(signature = (*, consumer_stream))]
    fn feedback(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        self.content_binding_with_owner(py, &owner)?;
        let (schema, tensors) = owner
            .prepared_feedback(&self.inner, stream)
            .map_err(xlog_err)?;
        let tensors = tensors
            .into_iter()
            .map(|tensor| {
                crate::dlpack_capsule_from_tensor(
                    py,
                    retain_export_owner(tensor, self.session.clone_ref(py), session.owner_thread)?,
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok((
            feedback_schema_object(py, &schema)?,
            PyTuple::new(py, tensors)?,
        )
            .into_pyobject(py)?
            .unbind())
    }
}

enum ContentStepOwner {
    Published(Py<PySemanticPublishedParent>),
    Prepared(Py<PySemanticPreparedStep>),
}

#[derive(Clone, Copy)]
enum ContentStepRef<'a> {
    Published(&'a PySemanticPublishedParent),
    Prepared(&'a PySemanticPreparedStep),
}

enum TensorContentBinding {
    Capture,
    Model(Option<usize>),
}

impl ContentStepOwner {
    fn check_producer_stream(
        &self,
        py: Python<'_>,
        owner: &SemanticTransitionSession,
        stream: u64,
    ) -> PyResult<()> {
        if let Self::Prepared(step) = self {
            let step = step.borrow(py);
            let issued = step.task_use.borrow(py);
            if matches!(
                &issued.state()?.phase,
                TaskUsePhase::Recording { .. } | TaskUsePhase::Prepared { .. }
            ) {
                step.session.borrow(py).require_creator()?;
                step.require_recording_stream(owner, stream)?;
            }
        }
        Ok(())
    }

    fn from_python(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(parent) = value.extract::<Py<PySemanticPublishedParent>>() {
            Ok(Self::Published(parent))
        } else if let Ok(step) = value.extract::<Py<PySemanticPreparedStep>>() {
            Ok(Self::Prepared(step))
        } else {
            Err(invalid(
                "content requires an authentic acquired parent or prepared step",
            ))
        }
    }

    fn binding(
        &self,
        py: Python<'_>,
        owner: &SemanticTransitionSession,
        retained: bool,
    ) -> PyResult<(Option<String>, Vec<u8>)> {
        match self {
            Self::Published(parent) => {
                let parent = parent.borrow(py);
                let (operation, snapshot) = if retained {
                    parent.retained_content_binding_with_owner(py, owner)?
                } else {
                    parent.content_binding_with_owner(py, owner)?
                };
                Ok((Some(operation), snapshot))
            }
            Self::Prepared(step) => step.borrow(py).content_binding_with_owner(py, owner),
        }
    }
}

/// Private content witness for one acquired parent and its original producers.
/// Transient capture proves stability; model binding compares the original
/// publication baseline. Neither proves derivation, provenance, replay or
/// permission to execute. Only the privileged controller can issue one.
/// The issuing Runtime must retain this complete object until its guaranteed
/// creator-thread cleanup after the original backward, not just save its native
/// aliases. Arbitrary retained Python producers must not be last-dropped by a
/// worker; thread-safe native ownership is not a producer-finalizer guarantee.
#[pyclass(
    name = "SemanticTensorContentWitness",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticTensorContentWitness {
    // Drop native ownership before the retained Python producers and Session.
    inner: SemanticTensorContentWitness,
    _inputs: Py<PyAny>,
    _producers: Vec<Py<PyAny>>,
    parent: ContentStepOwner,
    session: Py<PySemanticTransitionSession>,
}

#[pymethods]
impl PySemanticTensorContentWitness {
    /// Enqueue verification against this witness's original content baseline.
    /// Native checks their original pointers, metadata and device bytes. DLPack
    /// callbacks run without Session, reader or task-state locks; their complete
    /// handoff must leave the task and admitted authority snapshot unchanged.
    ///
    /// Call after hooks and before the first dependent read. All preceding writes
    /// must be ordered on the declared stream, with no concurrent writes through
    /// the final dependent use. A byte mismatch fatally invalidates the CUDA
    /// context. Returning None means only that the guard was enqueued; the
    /// original cold observation must propagate any device error before accepting
    /// results. Do not reset or retry a failed task. No digest or status is copied
    /// to the host, and verification grants no replay or import authority.
    /// Workers may call this one method while the issuing Runtime retains the
    /// complete witness and original graph through guaranteed creator cleanup.
    /// Use the actual consumer stream or legacy default stream 1. Per-thread
    /// default stream 2 is rejected on every thread before producer handoff.
    #[pyo3(signature = (*, tensors, consumer_stream))]
    fn verify(
        &self,
        py: Python<'_>,
        tensors: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let session = self.session.borrow(py);
        let binding = {
            let owner = session.witness_owner()?;
            self.parent.binding(py, &owner, true)?
        };
        let mut budget = 16 * 1024 * 1024;
        let stream = parse_witness_consumer_stream(consumer_stream, &mut budget)?;
        let device = session.device_ordinal;
        let check = || -> PyResult<()> {
            let owner = session.witness_owner()?;
            if self.parent.binding(py, &owner, true)? != binding {
                return Err(invalid(
                    "tensor content authority changed during producer callback",
                ));
            }
            self.parent.check_producer_stream(py, &owner, stream)
        };
        check()?;
        let ParsedTensorInputs {
            handoff: tensors,
            producers: _producers,
        } = parse_tensor_inputs_guarded(tensors, &mut budget, device, stream, &check)?;
        let mut owner = session.witness_owner()?;
        match &self.parent {
            ContentStepOwner::Published(parent) => {
                let parent = parent.borrow(py);
                let task_use = parent.task_use.borrow(py);
                task_use.require_current(&owner)?;
                let state = task_use.state()?;
                let (operation, snapshot) = state.content_handoff_binding()?;
                if (Some(operation), snapshot) != binding {
                    return Err(invalid(
                        "tensor content authority changed during producer handoff",
                    ));
                }
                let lease = parent.lease()?;
                owner.retained_step_identity(&lease).map_err(xlog_err)?;
                owner
                    .verify_tensor_content(&lease, &self.inner, tensors.into_native(), stream)
                    .map_err(xlog_err)
            }
            ContentStepOwner::Prepared(step) => {
                let step = step.borrow(py);
                let task_use = step.task_use.borrow(py);
                task_use.require_identity(&owner)?;
                let state = task_use.state()?;
                if state.prepared_content_binding(&step.scope)? != binding {
                    return Err(invalid(
                        "tensor content authority changed during producer handoff",
                    ));
                }
                owner
                    .verify_prepared_tensor_content(
                        &step.inner,
                        &self.inner,
                        tensors.into_native(),
                        stream,
                    )
                    .map_err(xlog_err)
            }
        }
    }
}

/// Issued only after the real policy launch and its native published or refused outcome.
/// The trusted model caller retains its authenticated output here alongside
/// graph-preserving packed inputs. This is not a Python result authenticator:
/// the controller's caller must have checked the original model invocation.
/// The Runtime must retain this complete invocation, its model output and packed
/// inputs through original backward or explicit final use and guaranteed creator cleanup.
#[cfg(feature = "semantic-policy")]
#[pyclass(name = "SemanticPolicyInvocation", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticPolicyInvocation {
    session: Py<PySemanticTransitionSession>,
    _task_use: Py<PySemanticTransitionTaskUse>,
    _parent: ContentStepOwner,
    _model_output: Py<PyAny>,
    _inputs: [Py<PyAny>; 4],
    rng: SemanticRngBinding,
    outcome: xlog_cuda::SemanticTransitionOutcome,
    training: bool,
    final_use_started: Mutex<bool>,
}

#[cfg(feature = "semantic-policy")]
#[pymethods]
impl PySemanticPolicyInvocation {
    /// Original native coordinates, not a successor or caller-supplied seed.
    #[getter]
    fn rng(&self, py: Python<'_>) -> PyResult<(u32, u64, u8, u32)> {
        self.session.borrow(py).require_creator()?;
        self.require_public_result(py)?;
        Ok((
            self.rng.model_generation,
            self.rng.stream_serial,
            self.rng.family_id,
            self.rng.proposal,
        ))
    }

    /// None for publication, otherwise a typed native refusal dictionary.
    /// Canary refusal dictionaries retain exact device-authored measurement,
    /// frozen-bound, resource and identity evidence.
    /// A refusal contains no published result or differentiable receipt bank.
    #[getter]
    fn refusal(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.session.borrow(py).require_creator()?;
        self.require_public_result(py)?;
        transition_refusal_report(py, &self.outcome)
    }

    /// Actual native evaluator output and full unclipped tuple weight.
    /// These cold results are not a differentiable surrogate selected-score.
    #[getter]
    fn result(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.session.borrow(py).require_creator()?;
        self.require_public_result(py)?;
        let observation = self.published_observation()?;
        let task = observation
            .task_evaluation
            .as_ref()
            .ok_or_else(|| invalid("policy invocation has no native task evaluation"))?;
        let result = PyDict::new(py);
        result.set_item("binding", PyBytes::new(py, task.binding.as_bytes()))?;
        result.set_item("winner", task.winner)?;
        result.set_item("return_value", task.return_value)?;
        result.set_item("importance_weight", observation.importance_weight)?;
        result.set_item("next_proposal", observation.next_proposal)?;
        result.set_item("query_count", task.query_count)?;
        result.set_item(
            "facts",
            PyTuple::new(
                py,
                task.facts.iter().map(|fact| {
                    (
                        (fact.correct[0], fact.correct[1]),
                        fact.g,
                        fact.p,
                        fact.c,
                        fact.v,
                        fact.eligible,
                    )
                }),
            )?,
        )?;
        Ok(result.unbind())
    }

    /// Fresh copies of all 136 original receipts. Mutating these Python copies
    /// cannot replace the native receipt/support buffers used by backward.
    #[getter]
    fn receipts(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        self.require_public_result(py)?;
        let receipts = self.published_observation()?.components.iter().map(|r| {
            let row = PyDict::new(py);
            macro_rules! scalar { ($($name:ident),+ $(,)?) => { $(row.set_item(stringify!($name), r.$name)?;)+ }; }
            scalar!(proposal, catalogue_generation, ordinal, lane, slot, field, kind, choice,
                legal_count, active_count, draw, cdf_start, cdf_end, mass, active_fields, null_fields);
            row.set_item("catalogue_digest", PyBytes::new(py, r.catalogue_digest.as_bytes()))?;
            row.set_item("admission_binding", PyBytes::new(py, r.admission_binding.as_bytes()))?;
            row.set_item("key", (r.key[0], r.key[1]))?;
            row.set_item("counter", (r.counter[0], r.counter[1], r.counter[2], r.counter[3]))?;
            row.set_item("random_words", (r.random_words[0], r.random_words[1], r.random_words[2], r.random_words[3]))?;
            row.set_item("p", (r.p[0], r.p[1], r.p[2]))?;
            row.set_item("q", (r.q[0], r.q[1], r.q[2]))?;
            row.set_item("factor_denominator", (r.factor_denominator[0], r.factor_denominator[1], r.factor_denominator[2]))?;
            Ok(row)
        }).collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, receipts)?.unbind())
    }

    /// Supply the original loss's contiguous FP64[136] selected-score
    /// cotangents. Returns real FP32 parameter and full MASK adjoint capsules,
    /// in that order, to be applied to the retained original graph. No model
    /// getter, second forward, Python selected-score or new primal leaf is used.
    /// Adjoint consumers enqueue on ``consumer_stream`` after this returns;
    /// readiness is ordered by the native producer event without a host wait.
    /// Stream 1 names the legacy default; streams 0 and 2 are refused.
    /// This is a single-use handoff, including uncertain producer failures.
    #[pyo3(signature = (score_cotangents, *, consumer_stream))]
    fn backward(
        &self,
        py: Python<'_>,
        score_cotangents: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        self.require_public_result(py)?;
        let expected = {
            let session = self.session.borrow(py);
            let owner = session.owner()?;
            self.final_use_binding(py, &owner)?
        };
        if !self.training {
            return Err(invalid(
                "policy backward requires the original training use",
            ));
        }
        self.published_observation()?;
        let mut budget = 4096;
        let consumer_stream = parse_witness_consumer_stream(consumer_stream, &mut budget)?;
        let producer_stream = i64::try_from(consumer_stream)
            .map_err(|_| invalid("consumer stream exceeds the DLPack signed stream range"))?;
        self.start_final_use()?;
        let check = || {
            let session = self.session.borrow(py);
            let owner = session.owner()?;
            if self.final_use_binding(py, &owner)? != expected {
                return Err(invalid(
                    "policy backward authority changed during producer handoff",
                ));
            }
            Ok(())
        };
        validate_producer_device_guarded(
            score_cotangents,
            self.session.borrow(py).device_ordinal,
            &check,
        )?;
        let mut handoff = TensorHandoff(vec![crate::dlpack_from_py_for_stream_guarded(
            score_cotangents,
            producer_stream,
            &check,
        )?]);
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        // A producer callback may have terminally aborted the shared owner.
        // Recheck the actual original parent and exact authority before
        // relinquishing the quarantined handoff or launching the native VJP.
        if self.final_use_binding(py, &owner)? != expected {
            return Err(invalid(
                "policy backward authority changed during producer handoff",
            ));
        }
        let tensor = handoff.0.pop().expect("one original cotangent producer");
        let gradients = match &self._parent {
            ContentStepOwner::Published(_) => {
                owner.backward_policy_dlpack(self.rng, tensor, consumer_stream)
            }
            ContentStepOwner::Prepared(step) => owner.backward_prepared_policy_dlpack(
                &step.borrow(py).inner,
                self.rng,
                tensor,
                consumer_stream,
            ),
        }
        .map_err(xlog_err)?;
        let [parameters, text] = gradients.into_dlpack().map_err(xlog_err)?;
        // The actual parent also owns the original continuation producers and
        // Session. Both capsule owners preserve them through late consumers,
        // with final Python decrefs deferred to the creating thread.
        let export_owner = || -> PublishedExportOwner {
            match &self._parent {
                ContentStepOwner::Published(parent) => parent.clone_ref(py).into(),
                ContentStepOwner::Prepared(step) => step.clone_ref(py).into(),
            }
        };
        let parameters = retain_export_owner(parameters, export_owner(), session.owner_thread)?;
        let text = retain_export_owner(text, export_owner(), session.owner_thread)?;
        Ok((
            crate::dlpack_capsule_from_tensor(py, parameters)?,
            crate::dlpack_capsule_from_tensor(py, text)?,
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// Consume this prepared proposal's exact retained tape against the
    /// prepared Update step's resident actor/critic/cost objective. The caller
    /// invokes this method for every proposal invocation in the frozen segment
    /// and joins every returned root into one GraphTask. Native selection makes
    /// unselected invocations graph-connected zeros; Python never reads the
    /// selected ordinal, reconstructs selected scores, or runs a second primal.
    ///
    /// Returns FP32 parameter, full MASK and component-baseline adjoint
    /// capsules, in that order. This is a single-use handoff, including
    /// uncertain native failures.
    #[pyo3(signature = (update_step, *, consumer_stream))]
    fn backward_selected(
        &self,
        py: Python<'_>,
        update_step: Py<PySemanticPreparedStep>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        self.require_public_result(py)?;
        let policy_step = match &self._parent {
            ContentStepOwner::Prepared(step) => step,
            ContentStepOwner::Published(_) => {
                return Err(invalid(
                    "device-selected policy backward requires an original prepared proposal",
                ));
            }
        };
        let expected = {
            let session = self.session.borrow(py);
            let owner = session.owner()?;
            self.final_use_binding(py, &owner)?
        };
        if !self.training {
            return Err(invalid(
                "device-selected policy backward requires the original training use",
            ));
        }
        self.published_observation()?;
        let mut budget = 4096;
        let consumer_stream = parse_witness_consumer_stream(consumer_stream, &mut budget)?;
        {
            let update = update_step.borrow(py);
            if self.session.as_ptr() != update.session.as_ptr() {
                return Err(invalid(
                    "device-selected policy backward requires an Update from the same Session",
                ));
            }
            let task_use = self._task_use.borrow(py);
            update.require_task(py, &task_use)?;
        }
        self.start_final_use()?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        if self.final_use_binding(py, &owner)? != expected {
            return Err(invalid(
                "policy backward authority changed before resident objective use",
            ));
        }
        let gradients = owner
            .backward_selected_prepared_policy(
                &policy_step.borrow(py).inner,
                &update_step.borrow(py).inner,
                self.rng,
                consumer_stream,
            )
            .map_err(xlog_err)?;
        let [parameters, text, baselines] = gradients.into_dlpack().map_err(xlog_err)?;
        let export_owner = || policy_step.clone_ref(py);
        let parameters = retain_export_owner(parameters, export_owner(), session.owner_thread)?;
        let text = retain_export_owner(text, export_owner(), session.owner_thread)?;
        let baselines = retain_export_owner(baselines, export_owner(), session.owner_thread)?;
        Ok((
            crate::dlpack_capsule_from_tensor(py, parameters)?,
            crate::dlpack_capsule_from_tensor(py, text)?,
            crate::dlpack_capsule_from_tensor(py, baselines)?,
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// Finish the original published or refused inference use after consumers are enqueued.
    ///
    /// ``consumer_stream`` must be the explicit CUDA stream containing those
    /// consumers (legacy default stream 1 is allowed; 0 and 2 are not). Native
    /// finalization joins that stream without a host wait and retires only this
    /// invocation's retained policy tape. It does not release the parent.
    /// Published training invocations require ``backward``; refused training
    /// invocations require ``finish_refusal``. Final use is one-shot, including
    /// uncertain native failures.
    #[pyo3(signature = (*, consumer_stream))]
    fn finish_inference(&self, py: Python<'_>, consumer_stream: &Bound<'_, PyAny>) -> PyResult<()> {
        self.session.borrow(py).require_creator()?;
        self.require_public_result(py)?;
        if self.training {
            return Err(invalid(
                "policy inference completion requires the original inference use",
            ));
        }
        self.finish_final_use(py, consumer_stream)
    }

    /// Finish an actual native refusal from inference or training without a VJP.
    ///
    /// Enqueue final consumers on the explicit ``consumer_stream`` first.
    /// Stream 1 is allowed; 0 and 2 are refused. Published outcomes are not
    /// admitted here. This shares the one-shot final-use owner with ``backward``
    /// and ``finish_inference`` and retains the parent for its later release.
    #[pyo3(signature = (*, consumer_stream))]
    fn finish_refusal(&self, py: Python<'_>, consumer_stream: &Bound<'_, PyAny>) -> PyResult<()> {
        self.session.borrow(py).require_creator()?;
        self.require_public_result(py)?;
        if !matches!(
            &self.outcome,
            xlog_cuda::SemanticTransitionOutcome::Refused(_)
        ) {
            return Err(invalid(
                "policy refusal completion requires an actual native refusal",
            ));
        }
        self.finish_final_use(py, consumer_stream)
    }
}

#[cfg(feature = "semantic-policy")]
impl PySemanticPolicyInvocation {
    fn finish_final_use(&self, py: Python<'_>, consumer_stream: &Bound<'_, PyAny>) -> PyResult<()> {
        let expected = {
            let session = self.session.borrow(py);
            let owner = session.owner()?;
            self.final_use_binding(py, &owner)?
        };
        let mut budget = 4096;
        let consumer_stream = parse_witness_consumer_stream(consumer_stream, &mut budget)?;
        self.start_final_use()?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        if self.final_use_binding(py, &owner)? != expected {
            return Err(invalid("policy authority changed before final use"));
        }
        match &self._parent {
            ContentStepOwner::Published(parent) => {
                let parent = parent.borrow(py);
                let lease = parent.lease()?;
                owner.finish_policy_invocation(&lease, self.rng, consumer_stream)
            }
            ContentStepOwner::Prepared(step) => owner.finish_prepared_policy_invocation(
                &step.borrow(py).inner,
                self.rng,
                consumer_stream,
            ),
        }
        .map_err(xlog_err)
    }

    fn published_observation(&self) -> PyResult<&xlog_cuda::SemanticTransitionObservation> {
        match &self.outcome {
            xlog_cuda::SemanticTransitionOutcome::Published(observation) => Ok(observation),
            xlog_cuda::SemanticTransitionOutcome::Refused(_) => Err(invalid(
                "refused policy invocation has no published result or differentiable receipts",
            )),
        }
    }

    fn final_use_binding(
        &self,
        py: Python<'_>,
        owner: &SemanticTransitionSession,
    ) -> PyResult<(String, Vec<u8>)> {
        let task_use = self._task_use.borrow(py);
        if self.session.as_ptr() != task_use.session.as_ptr() {
            return Err(invalid(
                "policy final use belongs to another native Session",
            ));
        }
        let binding = match &self._parent {
            ContentStepOwner::Published(parent) => {
                let parent = parent.borrow(py);
                if self.session.as_ptr() != parent.session.as_ptr() {
                    return Err(invalid(
                        "policy final use belongs to another native Session",
                    ));
                }
                parent.require_task(py, &task_use)?;
                task_use.require_current(owner)?;
                owner
                    .retained_step_identity(&*parent.lease()?)
                    .map_err(xlog_err)?;
                let state = task_use.state()?;
                state.require_public_use()?;
                state.content_handoff_binding()?
            }
            ContentStepOwner::Prepared(step) => {
                let step = step.borrow(py);
                if self.session.as_ptr() != step.session.as_ptr() {
                    return Err(invalid(
                        "policy final use belongs to another native Session",
                    ));
                }
                step.require_task(py, &task_use)?;
                task_use.require_identity(owner)?;
                owner.prepared_stream(&step.inner).map_err(xlog_err)?;
                let state = task_use.state()?;
                state.require_public_use()?;
                let (operation, snapshot) = state.prepared_content_binding(&step.scope)?;
                (
                    operation.ok_or_else(|| {
                        invalid("policy final use requires actual admitted execution")
                    })?,
                    snapshot,
                )
            }
        };
        let operation = if self.training {
            "training"
        } else {
            "inference"
        };
        if binding.0 != operation {
            return Err(invalid(
                "policy final use requires the original admitted operation",
            ));
        }
        Ok(binding)
    }

    fn start_final_use(&self) -> PyResult<()> {
        let mut started = self
            .final_use_started
            .lock()
            .map_err(|_| invalid("policy final-use owner is poisoned"))?;
        if *started {
            return Err(invalid("policy final use has already started"));
        }
        *started = true;
        Ok(())
    }

    fn require_public_result(&self, py: Python<'_>) -> PyResult<()> {
        let session = self.session.borrow(py);
        let owner = session.owner()?;
        if matches!(&self._parent, ContentStepOwner::Prepared(_)) {
            return self.final_use_binding(py, &owner).map(|_| ());
        }
        let task_use = self._task_use.borrow(py);
        task_use.require_current(&owner)?;
        let result = task_use.state()?.require_public_use();
        result
    }
}

fn descriptor_meaning_object(
    py: Python<'_>,
    descriptor: &xlog_cuda::SemanticActionDescriptor,
) -> PyResult<Py<PyAny>> {
    use xlog_cuda::SemanticActionDescriptor;
    let value = match descriptor {
        SemanticActionDescriptor::Target { predicate } => {
            ("target", predicate.0).into_pyobject(py)?.into_any()
        }
        SemanticActionDescriptor::Operand {
            scalar_type,
            sort_label,
            encoded_value,
        } => (
            "operand",
            scalar_type.to_code(),
            sort_label.as_str(),
            PyBytes::new(py, encoded_value),
        )
            .into_pyobject(py)?
            .into_any(),
        SemanticActionDescriptor::QualifierBundle { records } => (
            "qualifier_bundle",
            PyTuple::new(py, records.iter().copied())?,
        )
            .into_pyobject(py)?
            .into_any(),
        SemanticActionDescriptor::SupportEvent { source_record } => {
            ("support_event", *source_record)
                .into_pyobject(py)?
                .into_any()
        }
    };
    Ok(value.unbind())
}

fn dependency_lineage_object(py: Python<'_>, dependencies: &[Dependency]) -> PyResult<Py<PyTuple>> {
    let rows = dependencies
        .iter()
        .map(|dependency| {
            (
                dependency.identity.as_str(),
                dependency.kind.as_str(),
                PyTuple::new(py, &dependency.data_parents)?,
                PyTuple::new(py, &dependency.control_parents)?,
                PyTuple::new(py, &dependency.live_envelopes)?,
                dependency.target,
                dependency
                    .native_record
                    .as_ref()
                    .map(|(kind, index)| (kind.as_str(), *index)),
            )
                .into_pyobject(py)
                .map(Bound::unbind)
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, rows)?.unbind())
}

fn release_python_producers(producers: &Mutex<Vec<Py<PyAny>>>) {
    let producers = std::mem::take(
        &mut *producers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    );
    drop(producers);
}

impl PySemanticPublishedParent {
    fn lease(&self) -> PyResult<MutexGuard<'_, SemanticPublishedLease>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("published parent lease mutex is poisoned"))
    }

    // The native reader has already joined its consumers and released. Drop
    // original autograd owners only after releasing every native/state lock:
    // their Python finalizers may call the controller again.
    fn release_continuation_producers(&self) {
        release_python_producers(&self.continuation_producers);
    }

    fn require_task(&self, py: Python<'_>, task_use: &PySemanticTransitionTaskUse) -> PyResult<()> {
        let issued = self.task_use.borrow(py);
        issued.issuance.require_current()?;
        if !Arc::ptr_eq(&issued.controller, &task_use.controller)
            || !issued.issuance.same_as(&task_use.issuance)
            || issued.task_epoch != task_use.task_epoch
            || issued.task_identity != task_use.task_identity
        {
            return Err(invalid("published parent belongs to another task use"));
        }
        Ok(())
    }

    fn content_binding_with_owner(
        &self,
        py: Python<'_>,
        owner: &SemanticTransitionSession,
    ) -> PyResult<(String, Vec<u8>)> {
        let task_use = self.task_use.borrow(py);
        task_use.require_current(owner)?;
        owner
            .published_identity(&*self.lease()?)
            .map_err(xlog_err)?;
        let binding = task_use.state()?.content_handoff_binding()?;
        Ok(binding)
    }

    fn retained_content_binding_with_owner(
        &self,
        py: Python<'_>,
        owner: &SemanticTransitionSession,
    ) -> PyResult<(String, Vec<u8>)> {
        let task_use = self.task_use.borrow(py);
        task_use.require_current(owner)?;
        owner
            .retained_step_identity(&*self.lease()?)
            .map_err(xlog_err)?;
        let binding = task_use.state()?.content_handoff_binding()?;
        Ok(binding)
    }

    fn release_native_ownership(
        &self,
        py: Python<'_>,
        consumer_streams: &Bound<'_, PyAny>,
        reader_only: bool,
    ) -> PyResult<()> {
        self.session.borrow(py).require_creator()?;
        let session = self.session.borrow(py);
        if session.importing.load(Ordering::Acquire) {
            return Err(invalid(
                "public parent release cannot interrupt a native import",
            ));
        }
        let mut budget = 4096;
        let streams = ColdValue::read(consumer_streams, &mut budget, 0)?;
        let streams = streams
            .sequence()?
            .iter()
            .map(ColdValue::unsigned)
            .collect::<PyResult<Vec<_>>>()?;
        let mut owner = session.owner()?;
        if session.importing.load(Ordering::Acquire) {
            return Err(invalid(
                "public parent release cannot interrupt a native import",
            ));
        }
        // Cleanup uses the opaque original lease, not a new use grant. Bank
        // retirement preserves Python producers and their original graph until
        // native final release confirms all late consumers have completed.
        if reader_only {
            owner
                .release_published_reader(&mut *self.lease()?, &streams)
                .map_err(xlog_err)?;
        } else {
            owner
                .release(&mut *self.lease()?, &streams)
                .map_err(xlog_err)?;
        }
        drop(owner);
        drop(session);
        if !reader_only {
            self.release_continuation_producers();
        }
        Ok(())
    }
}

#[pymethods]
impl PySemanticPublishedParent {
    /// Enqueue fatal device guards for the original sealed native ranges.
    /// ``ranges`` is an exact tuple/list of exact ``(role, index)`` rows. Role 1
    /// checks the original raw prefix range. Arbitrary aliases outside these
    /// ranges are not attested. Use the same acquired parent throughout.
    ///
    /// Call after hooks and before the first dependent read, ordering all prior
    /// writes on ``consumer_stream`` and preventing concurrent writes through
    /// final use. The Session joins the named stream, guards the sealed bytes,
    /// then joins guard completion back to the consumer without a host copy or
    /// synchronization. A mismatch fatally invalidates the CUDA context; returning
    /// None only means enqueue succeeded. The original cold observation must
    /// propagate failure before accepting results. Never reset or retry that task.
    #[pyo3(signature = (*, ranges, consumer_stream))]
    fn guard_content(
        &self,
        py: Python<'_>,
        ranges: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.session.borrow(py).require_creator()?;
        let mut budget = 16 * 1024 * 1024;
        let ranges = parse_content_ranges(&ColdValue::read(ranges, &mut budget, 0)?)?;
        let stream = ColdValue::read(consumer_stream, &mut budget, 0)?.unsigned()?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        let task_use = self.task_use.borrow(py);
        task_use.require_current(&owner)?;
        let state = task_use.state()?;
        state.content_handoff_binding()?;
        owner
            .guard_published_content(&*self.lease()?, &ranges, stream)
            .map_err(xlog_err)
    }

    /// The sealed identity returned by this lease's single native Acquire.
    fn identity(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        let session = self.session.borrow(py);
        let owner = session.owner()?;
        self.task_use.borrow(py).require_current(&owner)?;
        let identity = owner
            .published_identity(&*self.lease()?)
            .map_err(xlog_err)?;
        Ok((
            PyBytes::new(py, identity.instance.as_bytes()),
            identity.word,
            PyBytes::new(py, identity.logical_digest.as_bytes()),
            PyBytes::new(py, identity.state_digest.as_bytes()),
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// Return the complete accepted ``(role, index)`` roster as an immutable
    /// tuple, preserving this acquired lease's native directory order.
    /// These are cached host metadata from the original Acquire; listing them
    /// performs no device payload, digest or status copy and no synchronization.
    /// Actual access through tensor(), record() or prefix() still checks content
    /// readiness. This roster grants no new use, import or replay authority.
    fn range_keys(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        let session = self.session.borrow(py);
        let owner = session.owner()?;
        self.task_use.borrow(py).require_current(&owner)?;
        let ranges = owner
            .published_range_keys(&*self.lease()?)
            .map_err(xlog_err)?;
        Ok(PyTuple::new(
            py,
            ranges.into_iter().map(|(role, index)| (role as u64, index)),
        )?
        .unbind())
    }

    /// Project model metadata from this exact acquired parent. Returns:
    /// (parent_identity, descriptor_digest, semantic_root, prefix, ring,
    /// generations, contracts, capacities, numerical_state).
    ///
    /// ``semantic_root`` is (owner, slot, generation, digest, extents), with
    /// statement/support/version extents. It identifies the acquired XLOG root;
    /// it is not the parent state digest or the action catalogue binding.
    /// ``prefix`` contains CUDA DLPack capsules (U8[32] identity, U64[1] extent).
    /// ``ring`` contains capsules (U64[1] head, U64[32,8] original source slots).
    /// They view this acquired bank without copying, sorting or rotating rows.
    /// ``generations`` is (model, neural_bank, neural, cache, authority).
    /// ``contracts`` is (topology_identity, position_table_identity).
    /// ``capacities`` is (prefix, feedback, exclusive_position_bound).
    /// ``numerical_state`` is (model_geometry_digest, model_numerical_digest),
    /// both 32-byte identities from this same acquired header. The first binds
    /// full allocation/storage/view geometry and model layouts; the second
    /// folds that identity with actual typed and full-backing seals for roles
    /// 18..25. Neither substitutes for the full ModelContract or parent identity.
    /// Other digests are 32-byte bytes. The descriptor digest binds the publication
    /// instance/word; it is not the action catalogue's descriptor identity.
    ///
    /// The retained parent remains the owner: matching these values does not
    /// authenticate unrelated model inputs or issue a new use permission. Device
    /// views are read-only by contract and consumed once; retain their framework
    /// tensors until final use. ``consumer_stream`` follows tensor()'s protocol.
    #[pyo3(signature = (*, consumer_stream))]
    fn model_context(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        let mut budget = 128;
        let stream = ColdValue::read(consumer_stream, &mut budget, 0)?.unsigned()?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        self.task_use.borrow(py).require_current(&owner)?;
        let context = owner
            .published_model_context(&*self.lease()?, stream)
            .map_err(xlog_err)?;
        let parent = context.parent;
        let root = context.semantic_root;
        let capsule = |tensor| {
            crate::dlpack_capsule_from_tensor(
                py,
                retain_export_owner(tensor, self.session.clone_ref(py), session.owner_thread)?,
            )
        };
        Ok((
            (
                PyBytes::new(py, parent.instance.as_bytes()),
                parent.word,
                PyBytes::new(py, parent.logical_digest.as_bytes()),
                PyBytes::new(py, parent.state_digest.as_bytes()),
            ),
            PyBytes::new(py, context.descriptor_digest.as_bytes()),
            (
                root.0,
                root.1,
                root.2,
                PyBytes::new(py, root.3.as_bytes()),
                (root.4[0], root.4[1], root.4[2]),
            ),
            (capsule(context.prefix.0)?, capsule(context.prefix.1)?),
            (capsule(context.ring_head)?, capsule(context.source)?),
            context.generations,
            (
                PyBytes::new(py, context.topology_identity.as_bytes()),
                PyBytes::new(py, context.table_identity.as_bytes()),
            ),
            context.capacities,
            (
                PyBytes::new(py, context.numerical_state.0.as_bytes()),
                PyBytes::new(py, context.numerical_state.1.as_bytes()),
            ),
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// Export the exact selected resident neural bank without a device copy.
    /// Returns (parent_identity, generations, numerical_state, allocations,
    /// storages, views). Each allocation row is (native_provenance, U8 DLPack)
    /// and occurs exactly once. Storage rows are (allocation, byte_offset,
    /// span_bytes). View rows are (role, index, storage, byte_offset, layout),
    /// where layout preserves element size, scalar type, rank, logical axis,
    /// four dimensions and four byte strides.
    ///
    /// These aliases are read-only views of the actually selected bank, not the
    /// private late-backward copies returned by tensor_allocation(). Retain the
    /// parent and all framework aliases through final use, then release with all
    /// consumer streams. The retained reader prevents native bank reuse; matching
    /// metadata alone grants no authority and must not be rebound to other bytes.
    #[pyo3(signature = (*, consumer_stream))]
    fn resident_model_memory(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        let stream = ColdValue::read(consumer_stream, &mut 128, 0)?.unsigned()?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        self.task_use.borrow(py).require_current(&owner)?;
        let memory = owner
            .published_resident_model_memory(&*self.lease()?, stream)
            .map_err(xlog_err)?;
        let parent = memory.parent;
        let mut allocations = Vec::with_capacity(memory.allocations.len());
        for (provenance, tensor) in memory.allocations {
            let tensor =
                retain_export_owner(tensor, self.session.clone_ref(py), session.owner_thread)?;
            allocations.push(
                (
                    Py::new(py, PyNativeTensorAllocation { provenance })?,
                    crate::dlpack_capsule_from_tensor(py, tensor)?,
                )
                    .into_pyobject(py)?
                    .unbind(),
            );
        }
        let storages = PyTuple::new(
            py,
            memory
                .storages
                .into_iter()
                .map(|storage| (storage.allocation, storage.byte_offset, storage.span_bytes)),
        )?;
        let views = PyTuple::new(
            py,
            memory.views.into_iter().map(|(view, layout)| {
                (
                    view.role,
                    view.index,
                    view.storage,
                    view.byte_offset,
                    (
                        layout.element_bytes,
                        layout.scalar_type,
                        layout.rank,
                        layout.logical_axis,
                        (
                            layout.dimensions[0],
                            layout.dimensions[1],
                            layout.dimensions[2],
                            layout.dimensions[3],
                        ),
                        (
                            layout.strides_bytes[0],
                            layout.strides_bytes[1],
                            layout.strides_bytes[2],
                            layout.strides_bytes[3],
                        ),
                    ),
                )
            }),
        )?;
        Ok((
            (
                PyBytes::new(py, parent.instance.as_bytes()),
                parent.word,
                PyBytes::new(py, parent.logical_digest.as_bytes()),
                PyBytes::new(py, parent.state_digest.as_bytes()),
            ),
            memory.generations,
            (
                PyBytes::new(py, memory.numerical_state.0.as_bytes()),
                PyBytes::new(py, memory.numerical_state.1.as_bytes()),
            ),
            PyTuple::new(py, allocations)?,
            storages,
            views,
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// Resolve one descriptor in this live parent's native catalogue.
    ///
    /// Returns ((catalogue_generation, digest), meaning). Descriptor fields are
    /// 2 (target), 3..6 (operand), 11 (qualifier bundle), and 17 (support event).
    /// Category zero in these fields is NULL, represented by None; an empty
    /// qualifier bundle remains ("qualifier_bundle", ()). Operands retain the
    /// native scalar type code, sort label and exact encoded bytes. A support
    /// event refers to the admission's support array, not its statement array.
    ///
    /// This read-only projection grants no use authority and does not establish
    /// the causal origin of a model operand from equality of its bytes.
    fn descriptor_meaning(
        &self,
        py: Python<'_>,
        field: &Bound<'_, PyAny>,
        category: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        if !field.is_exact_instance_of::<pyo3::types::PyInt>()
            || !category.is_exact_instance_of::<pyo3::types::PyInt>()
        {
            return Err(invalid("descriptor coordinates must be exact integers"));
        }
        let field = field.extract::<u32>()?;
        let category = category.extract::<u32>()?;
        if !matches!(field, 2 | 3..=6 | 11 | 17) {
            return Err(invalid("field does not contain descriptor categories"));
        }
        let session = self.session.borrow(py);
        let owner = session.owner()?;
        self.task_use.borrow(py).require_current(&owner)?;
        owner
            .published_identity(&*self.lease()?)
            .map_err(xlog_err)?;
        let cardinality = owner
            .components()
            .iter()
            .find(|entry| entry.field == field)
            .ok_or_else(|| invalid("descriptor field is absent from the native catalogue"))?
            .cardinality;
        if category >= cardinality {
            return Err(invalid("descriptor category exceeds the native catalogue"));
        }
        let meaning = if category == 0 {
            py.None()
        } else {
            descriptor_meaning_object(
                py,
                owner
                    .descriptor_meaning(field, category)
                    .ok_or_else(|| invalid("native descriptor category has no meaning"))?,
            )?
        };
        let binding = owner.binding();
        Ok((
            (
                binding.generation,
                PyBytes::new(py, binding.digest.as_bytes()),
            ),
            meaning,
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// Return the retained task's checked dependency graph for this live parent.
    ///
    /// The result is (parent_identity, task_epoch, nodes, feedback_roots).
    /// Each node preserves the imported closed-schema order:
    /// (identity, kind, data_parents, control_parents, live_envelopes, target,
    /// native_record). A native_record is ("statement" | "support" | "observer",
    /// index); target is an optional (replay_batch_row, logical_position).
    ///
    /// These are the actual dependencies retained after import validation,
    /// including native-read roots, not a synthesized allowed-support set.
    /// They do not replace device query receipts, current feedback validity,
    /// token origin records, or a model operand's actual dependency edges.
    fn dependency_lineage(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        let session = self.session.borrow(py);
        let owner = session.owner()?;
        let task_use = self.task_use.borrow(py);
        task_use.require_current(&owner)?;
        let identity = owner
            .published_identity(&*self.lease()?)
            .map_err(xlog_err)?;
        let parent_identity = (
            PyBytes::new(py, identity.instance.as_bytes()),
            identity.word,
            PyBytes::new(py, identity.logical_digest.as_bytes()),
            PyBytes::new(py, identity.state_digest.as_bytes()),
        );
        Ok((
            parent_identity,
            task_use.task_epoch,
            dependency_lineage_object(py, &task_use.authority.dependencies)?,
            PyTuple::new(py, &task_use.authority.feedback_roots)?,
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// Cold read of the complete original model contribution from this parent's
    /// native RuntimeContract. Its producer owns format and numerical validation.
    /// Use before reconstruction, never within a resident loop. No current model
    /// settings are sampled or substituted.
    fn model_numerical_mode(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.session.borrow(py).require_creator()?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        self.task_use.borrow(py).require_current(&owner)?;
        let bytes = owner
            .published_model_numerical_mode(&*self.lease()?)
            .map_err(xlog_err)?;
        let value = ColdValue::from_canonical_bytes(&bytes)?;
        value.python_value(py)
    }

    /// Original ring slots as a read-only CUDA DLPack capsule U64[32,8], element
    /// strides (8,1), without sorting, repacking or rotating by the ring head.
    /// Columns: token, logical_position, kind, provenance, valid, committed,
    /// recomputed, provenance_record. Consume once; unused slots are not rows.
    /// The capsule retains this reader and Session. Final consumer streams must
    /// be joined at release; ``consumer_stream`` follows tensor()'s protocol.
    #[pyo3(signature = (*, consumer_stream))]
    fn source(&self, py: Python<'_>, consumer_stream: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.session.borrow(py).require_creator()?;
        let mut budget = 128;
        let stream = ColdValue::read(consumer_stream, &mut budget, 0)?.unsigned()?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        self.task_use.borrow(py).require_current(&owner)?;
        let tensor = owner
            .published_source(&*self.lease()?, stream)
            .map_err(xlog_err)?;
        crate::dlpack_capsule_from_tensor(
            py,
            retain_export_owner(tensor, self.session.clone_ref(py), session.owner_thread)?,
        )
    }

    /// Original terminal flag from this acquired parent's private step inputs,
    /// as a single-consumer, read-only-by-contract CUDA DLPack capsule U64[1]
    /// with element stride 1. Zero means active; one means Drain is required.
    /// This is not a host observation or a replacement for native eligibility.
    /// Use for cold reconstruction with the same acquired source/model context.
    /// The capsule retains its native step owner and Session; retain the imported
    /// tensor until final use and join its consumer streams at release.
    /// ``consumer_stream`` follows tensor()'s protocol.
    #[pyo3(signature = (*, consumer_stream))]
    fn terminal_state(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.session.borrow(py).require_creator()?;
        let mut budget = 128;
        let stream = ColdValue::read(consumer_stream, &mut budget, 0)?.unsigned()?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        self.task_use.borrow(py).require_current(&owner)?;
        let tensor = owner
            .published_terminal(&*self.lease()?, stream)
            .map_err(xlog_err)?;
        crate::dlpack_capsule_from_tensor(
            py,
            retain_export_owner(tensor, self.session.clone_ref(py), session.owner_thread)?,
        )
    }

    /// Return (computed, rows) from this acquired bank's sealed active-cache
    /// record. Each row is (physical_row, source_slot, logical_position, kind).
    /// Computed empty rows are valid; an uncomputed initial cache is not.
    fn active_rows(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        self.task_use.borrow(py).require_current(&owner)?;
        let active = owner
            .published_active_rows(&*self.lease()?)
            .map_err(xlog_err)?;
        let rows = PyTuple::new(
            py,
            active.rows.into_iter().map(|row| {
                (
                    row.physical_row,
                    row.source_slot,
                    row.logical_position,
                    row.kind,
                )
            }),
        )?;
        Ok((active.computed, rows).into_pyobject(py)?.unbind())
    }

    /// Return (parent_identity, rows) for this acquired parent's committed prefix.
    /// ``rows`` is a single-consumer CUDA DLPack capsule with dtype U64, shape
    /// (prefix_capacity, 8), and element strides (8, 1). Committed row i is logical position i;
    /// columns are (token, logical_position, kind, provenance, valid, committed,
    /// recomputed, provenance_record). Kind is FILLED=1; provenance preserves
    /// SOURCE=1 or GENERATED=2, and provenance_record is its retained native ledger
    /// index, not an authority bit. Flags remain u64 values, not a numeric cast.
    /// The identity is exactly the four-component tuple returned by identity().
    ///
    /// Only [0, prefix_extent) is committed; even an empty prefix retains its
    /// cold capacity shape. Unused rows must not enter model computation.
    /// There is no second Acquire, ring reconstruction, or host prefix copy.
    /// The read-only-by-contract view retains this reader and the Session until
    /// its final DLPack deleter. Do not modify it. ``consumer_stream`` follows
    /// tensor()'s protocol; release must name every stream after its final use.
    #[pyo3(signature = (*, consumer_stream))]
    fn prefix(&self, py: Python<'_>, consumer_stream: &Bound<'_, PyAny>) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        let mut budget = 128;
        let stream = ColdValue::read(consumer_stream, &mut budget, 0)?.unsigned()?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        self.task_use.borrow(py).require_current(&owner)?;
        let (identity, tensor) = owner
            .published_prefix(&*self.lease()?, stream)
            .map_err(xlog_err)?;
        let parent_identity = (
            PyBytes::new(py, identity.instance.as_bytes()),
            identity.word,
            PyBytes::new(py, identity.logical_digest.as_bytes()),
            PyBytes::new(py, identity.state_digest.as_bytes()),
        );
        let tensor = retain_export_owner(tensor, self.session.clone_ref(py), session.owner_thread)?;
        Ok((
            parent_identity,
            crate::dlpack_capsule_from_tensor(py, tensor)?,
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// Export a real bank tensor with its cold physical-capacity shape as a
    /// single-consumer DLPack capsule. ``consumer_stream`` is legacy default 1
    /// or an explicit live CUDA stream; zero and per-thread default 2 refuse.
    /// It is not a publication identity. The native owner orders producer work
    /// onto it without a host wait. Role/index must exist and have computed
    /// content; capacity does not make unused rows valid model inputs.
    #[pyo3(signature = (*, role, index, consumer_stream))]
    fn tensor(
        &self,
        py: Python<'_>,
        role: &Bound<'_, PyAny>,
        index: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.session.borrow(py).require_creator()?;
        let mut budget = 128;
        let role = SemanticStateRole::from_code(ColdValue::read(role, &mut budget, 0)?.unsigned()?)
            .ok_or_else(|| invalid("unknown native publication tensor role"))?;
        let index = ColdValue::read(index, &mut budget, 0)?.unsigned()?;
        let stream = ColdValue::read(consumer_stream, &mut budget, 0)?.unsigned()?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        self.task_use.borrow(py).require_current(&owner)?;
        let tensor = owner
            .published_tensor(&*self.lease()?, role, index, stream)
            .map_err(xlog_err)?;
        let tensor = retain_export_owner(tensor, self.session.clone_ref(py), session.owner_thread)?;
        crate::dlpack_capsule_from_tensor(py, tensor)
    }

    /// Return (native provenance, full allocation U8 DLPack capsule) for
    /// tensor(role, index). Use during cold binding; consume the capsule once.
    /// Geometry is in bytes relative to the complete native allocation, and
    /// same_allocation compares retained owners, not addresses or consumer storage.
    /// The capsule keeps the original reader/step grant and stream ordering;
    /// its full backing is read-only by contract, never an update destination.
    /// capacity does not certify unused content. Retain the original owners
    /// through every use and backward. Model version-counter origins are separate.
    #[pyo3(signature = (*, role, index, consumer_stream))]
    fn tensor_allocation(
        &self,
        py: Python<'_>,
        role: &Bound<'_, PyAny>,
        index: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        let mut budget = 128;
        let role = SemanticStateRole::from_code(ColdValue::read(role, &mut budget, 0)?.unsigned()?)
            .ok_or_else(|| invalid("unknown native publication tensor role"))?;
        let index = ColdValue::read(index, &mut budget, 0)?.unsigned()?;
        let stream = ColdValue::read(consumer_stream, &mut budget, 0)?.unsigned()?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        self.task_use.borrow(py).require_current(&owner)?;
        let (provenance, tensor) = owner
            .published_tensor_allocation(&*self.lease()?, role, index, stream)
            .map_err(xlog_err)?;
        let tensor = retain_export_owner(tensor, self.session.clone_ref(py), session.owner_thread)?;
        Ok((
            Py::new(py, PyNativeTensorAllocation { provenance })?,
            crate::dlpack_capsule_from_tensor(py, tensor)?,
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// Export a sealed native record's actual device bytes, including raw
    /// feedback (role15) and its qualified typed-statement payload (role16).
    /// These are U8 bytes with the same reader lifetime as model tensors;
    /// neither an encoded model feature nor a caller-supplied feedback object.
    #[pyo3(signature = (*, role, index, consumer_stream))]
    fn record(
        &self,
        py: Python<'_>,
        role: &Bound<'_, PyAny>,
        index: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.session.borrow(py).require_creator()?;
        let mut budget = 128;
        let role = SemanticStateRole::from_code(ColdValue::read(role, &mut budget, 0)?.unsigned()?)
            .ok_or_else(|| invalid("unknown native publication record role"))?;
        let index = ColdValue::read(index, &mut budget, 0)?.unsigned()?;
        let stream = ColdValue::read(consumer_stream, &mut budget, 0)?.unsigned()?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        self.task_use.borrow(py).require_current(&owner)?;
        let tensor = owner
            .published_record(&*self.lease()?, role, index, stream)
            .map_err(xlog_err)?;
        let tensor = retain_export_owner(tensor, self.session.clone_ref(py), session.owner_thread)?;
        crate::dlpack_capsule_from_tensor(py, tensor)
    }

    /// Produce (parent_identity, schema, features, device_validity, lineage).
    /// The two device values are single-consumer DLPack capsules with exact
    /// FP32[capacity, 9*B+4] and Bool8[capacity] shapes. Validity remains on the
    /// device; provenance errors trap before the consumer readiness event.
    /// No slots are packed. Each byte contributes eight LSB-first bits followed
    /// by presence; the tail is (PRO, 1, CONTRA, 1) for valid records. Invalid
    /// rows are entirely zero, before any payload address or bit computation.
    ///
    /// Lineage is (task_epoch, support_fields, current_receipts,
    /// original_statements, support_offsets, support_rows). These four additional
    /// single-consumer capsules are U64[capacity,42], U64[capacity],
    /// U64[capacity+1], U64[support_capacity,12]. Receipts are opaque production
    /// query receipts from this acquired root, not archived runtime handles.
    /// A slot owns rows [offset[s],offset[s+1]); support_fields names their
    /// columns. Rows are actual newest-to-oldest contributions, not the allowed
    /// mutation roster. Invalid slots have index U64::MAX and no rows; valid
    /// Neither retains its original statement index and an empty row interval.
    /// Service values are not learned feature columns and do not replace the
    /// original dependency graph or current task epoch check.
    ///
    /// Requires before_segment for this parent. All exports retain its reader
    /// until their final deleters, including any original encoder backward.
    /// Release must name every consumer stream after the final use of all six.
    #[pyo3(signature = (*, consumer_stream))]
    fn feedback(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        let mut budget = 128;
        let stream = ColdValue::read(consumer_stream, &mut budget, 0)?.unsigned()?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        let task_use = self.task_use.borrow(py);
        task_use.require_current(&owner)?;
        task_use.state()?.content_handoff_binding()?;
        let feedback = owner
            .published_feedback(&*self.lease()?, stream)
            .map_err(xlog_err)?;
        let identity = feedback.parent;
        let parent_identity = (
            PyBytes::new(py, identity.instance.as_bytes()),
            identity.word,
            PyBytes::new(py, identity.logical_digest.as_bytes()),
            PyBytes::new(py, identity.state_digest.as_bytes()),
        );
        let schema = feedback_schema_object(py, &feedback.schema)?;
        let [features, device_validity, queries, originals, offsets, supports] =
            feedback.into_dlpack();
        let features =
            retain_export_owner(features, self.session.clone_ref(py), session.owner_thread)?;
        let device_validity = retain_export_owner(
            device_validity,
            self.session.clone_ref(py),
            session.owner_thread,
        )?;
        let export = |tensor| {
            let tensor =
                retain_export_owner(tensor, self.session.clone_ref(py), session.owner_thread)?;
            crate::dlpack_capsule_from_tensor(py, tensor)
        };
        let lineage = (
            task_use.task_epoch,
            PyTuple::new(py, xlog_cuda::SEMANTIC_FEEDBACK_SUPPORT_FIELDS)?,
            export(queries)?,
            export(originals)?,
            export(offsets)?,
            export(supports)?,
        );
        Ok((
            parent_identity,
            schema,
            crate::dlpack_capsule_from_tensor(py, features)?,
            crate::dlpack_capsule_from_tensor(py, device_validity)?,
            lineage,
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// Join the actual bank consumers and retire only this exact bank reader.
    /// Every bank alias must have returned its managed owner. Original private
    /// witnesses, model baselines, feedback outputs, continuation producers and
    /// the original autograd graph remain owned for late verification/backward.
    /// No further publication reads are allowed through this parent; original
    /// witnesses and policy final use still require current task authority.
    /// Finish with ``release`` after all late consumers and witnesses are gone.
    /// Failure preserves native and Python custody; never retry an uncertain use.
    #[pyo3(signature = (*, consumer_streams))]
    fn release_reader(&self, py: Python<'_>, consumer_streams: &Bound<'_, PyAny>) -> PyResult<()> {
        self.release_native_ownership(py, consumer_streams, true)
    }

    /// Join all named consumer streams and finally release this original step.
    /// Accepts either a live bank reader or one retired by ``release_reader``.
    /// Outstanding content witnesses or externally held DLPack aliases refuse
    /// without destroying retained custody. A failed device release is quarantined by
    /// Session, never retried automatically.
    #[pyo3(signature = (*, consumer_streams))]
    fn release(&self, py: Python<'_>, consumer_streams: &Bound<'_, PyAny>) -> PyResult<()> {
        self.release_native_ownership(py, consumer_streams, false)
    }
}

#[pymethods]
impl PySemanticTransitionTaskUse {
    /// Read the retained Session binding without exposing the controller.
    fn binding(&self, py: Python<'_>) -> PyResult<(u64, Py<PyBytes>)> {
        self.session.borrow(py).require_creator()?;
        let session = self.session.borrow(py);
        self.require_current(&*session.owner()?)?;
        session.binding(py)
    }

    /// Read the same native Session layout; this does not grant execution.
    fn policy_layout(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        let session = self.session.borrow(py);
        self.require_current(&*session.owner()?)?;
        session.policy_layout(py)
    }

    /// Return the cold typed-feedback layout without granting execution or
    /// pretending that a result bank has been acquired.
    ///
    /// The tuple is ``(abi, schema_digest, adapter_digest, statement_capacity,
    /// feature_width, minimum_slots, fields)``. Each field is ``(source,
    /// byte_offset, bit_width, required)``. ``statement`` names the native
    /// qualified-value payload: u64 atom-length, exact predicate/arity/typed
    /// values, u32 qualifier-count, then each u64 qualifier-length and its exact
    /// typed values, in admitted order. Lengths and values are little-endian;
    /// symbols contain the originally admitted UTF-8, not registry IDs.
    /// Schema digests, addresses, observer answers, and receipt seals are not
    /// part of that learned payload. A statement byte is present exactly below
    /// the acquired record's statement length; every absent byte is canonical
    /// zero. ``record`` offsets 32 and 40 name the independent pro/contra u64
    /// values, each restricted to zero or one. Bits are least-significant-first.
    /// Invalid slots are masked before encoding, independently of payload bytes.
    /// The actual fixed slot capacity and validity come only from parent Acquire.
    /// Flat coordinates append each field's LSB-first bits and its presence:
    /// byte i occupies [9*i, 9*i+9); the final four are PRO, presence(PRO),
    /// CONTRA, presence(CONTRA). Required support presence is one only for valid
    /// records. Use parent.feedback for the authenticated device producer.
    fn feedback_schema(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        let session = self.session.borrow(py);
        let owner = session.owner()?;
        self.require_current(&owner)?;
        let schema = FeedbackSchema::new(owner.feedback_statement_bytes().map_err(xlog_err)?)
            .map_err(xlog_err)?;
        feedback_schema_object(py, &schema)
    }

    /// Native support packing in fixed catalogue order.
    /// Returns (binding, 136 half-open byte spans, total Bool8 cells).
    /// Inactive TEXT keeps its reserved span, which the device does not read.
    fn policy_support_layout(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        let session = self.session.borrow(py);
        let owner = session.owner()?;
        self.require_current(&owner)?;
        let binding = owner.binding();
        let (spans, cells) = owner.policy_support_layout().map_err(xlog_err)?;
        let spans: Vec<_> = spans
            .into_iter()
            .map(|span| (span.start, span.end))
            .collect();
        Ok((
            (
                binding.generation,
                PyBytes::new(py, binding.digest.as_bytes()),
            ),
            spans,
            cells,
        )
            .into_pyobject(py)?
            .unbind())
    }
}

#[pymethods]
impl PySemanticTransitionController {
    #[new]
    fn new(py: Python<'_>, session: Py<PySemanticTransitionSession>) -> PyResult<Self> {
        session.borrow(py).require_creator()?;
        Ok(Self {
            session,
            identity: Arc::new(()),
        })
    }

    /// Record one fixed bounded segment without submitting it. The original
    /// producer prepares cold storage on consumer_stream and returns an owner
    /// with memory_scope, enqueue_step(step, bank), and finish_segment() methods.
    /// memory_scope spans allocation/recording through EndCapture/instantiate;
    /// enqueue_step records both fixed bank branches per slot and returns None.
    /// Each bank binds its own original model witness, continuation, policy and
    /// update outputs through this controller. Device admission selects exactly
    /// one retained branch; no branch aliases the other's primal or tape owners.
    /// Python receives neither the graph nor authority to report its completion.
    /// transitions is a nonempty exact tuple of "proposal" or "recompute" modes;
    /// its length fixes the bound and each mode is retained before any callback.
    /// Device DRAIN_REQUIRED overrides either nominal mode. Recompute binds the
    /// original model continuation without a policy or backward obligation.
    #[cfg(feature = "semantic-policy")]
    #[pyo3(signature = (task_use, *, transitions, producer))]
    fn build_segment(
        &self,
        py: Python<'_>,
        task_use: Py<PySemanticTransitionTaskUse>,
        transitions: &Bound<'_, PyAny>,
        producer: Py<PyAny>,
    ) -> PyResult<()> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        let issued = task_use.borrow(py);
        self.require_issued(&issued)?;
        if self.session.as_ptr() != issued.session.as_ptr() {
            return Err(invalid("segment task belongs to another Session"));
        }
        let transitions = prepared_transitions(transitions)?;
        if session.importing.load(Ordering::Acquire)
            || session.retiring.load(Ordering::Acquire)
            || session.recording.swap(true, Ordering::AcqRel)
        {
            return Err(invalid(
                "segment recording cannot overlap another recording or import",
            ));
        }
        let mut guard = PreparedBuildGuard {
            session: &session,
            task_use: &issued,
            committed: false,
        };
        let (scope, snapshot, handles, stream) = {
            let mut owner = session.owner()?;
            issued.require_identity(&owner)?;
            if session
                .prepared_segment
                .lock()
                .map_err(|_| invalid("prepared segment mutex is poisoned"))?
                .is_some()
            {
                return Err(invalid("previous prepared segment has not been retired"));
            }
            let mut state = issued.state()?;
            let scope = state.begin_build()?;
            let handles = owner.prepare_segment_steps(transitions).map_err(xlog_err)?;
            let stream = owner.prepared_stream(&handles[0]).map_err(xlog_err)?;
            (scope, state.snapshot.canonical.clone(), handles, stream)
        };
        let mut steps = Vec::with_capacity(handles.len());
        for inner in handles {
            steps.push(Py::new(
                py,
                PySemanticPreparedStep {
                    session: self.session.clone_ref(py),
                    task_use: task_use.clone_ref(py),
                    scope: Arc::clone(&scope),
                    inner,
                    continuation_producers: Mutex::new(std::array::from_fn(|_| Vec::new())),
                    policy_inputs: Mutex::new(std::array::from_fn(|_| None)),
                },
            )?);
        }
        let resources = Arc::new(PreparedProducerResources {
            producers: Mutex::new(vec![producer.clone_ref(py)]),
            owner_thread: session.owner_thread,
        });
        *session
            .prepared_segment
            .lock()
            .map_err(|_| invalid("prepared segment mutex is poisoned"))? =
            Some(PreparedPythonSegment {
                scope: Arc::clone(&scope),
                task_use: task_use.clone_ref(py),
                steps: steps.iter().map(|step| step.clone_ref(py)).collect(),
                resources: Arc::clone(&resources),
                producers_retired: false,
            });
        let check = || self.check_recording(py, &issued, &scope, &snapshot);
        let kwargs = PyDict::new(py);
        kwargs.set_item(
            "steps",
            PyTuple::new(py, steps.iter().map(|step| step.clone_ref(py)))?,
        )?;
        kwargs.set_item("consumer_stream", stream.cu_stream() as u64)?;
        let prepare = recording_callback(check, || producer.bind(py).getattr("prepare_segment"))?;
        let prepared = recording_callback(check, || {
            prepare.call((task_use.clone_ref(py),), Some(&kwargs))
        })?;
        resources
            .producers
            .lock()
            .map_err(|_| invalid("prepared producer owner mutex is poisoned"))?
            .push(prepared.clone().unbind());
        let memory = recording_callback(check, || prepared.getattr("memory_scope"))?;
        resources
            .producers
            .lock()
            .map_err(|_| invalid("prepared producer owner mutex is poisoned"))?
            .push(memory.clone().unbind());
        let mut memory_owner = PreparedMemoryScope::new(py, &memory, &check)?;
        let enter = recording_callback(check, || memory.getattr("__enter__"))?;
        recording_callback(check, || {
            let entered = enter.call0()?;
            memory_owner.entered = true;
            Ok(entered)
        })?;
        let recorded = (|| -> PyResult<()> {
            let external: Arc<dyn Send + Sync> = resources.clone();
            let (mut capture, capture_stream) = {
                let mut owner = session.owner()?;
                owner
                    .begin_prepared_segment(vec![external])
                    .map_err(xlog_err)?
            };
            if capture_stream.cu_stream() != stream.cu_stream() {
                return Err(invalid(
                    "native recording stream changed after cold preparation",
                ));
            }
            for step in &steps {
                check()?;
                let native = step.borrow(py).inner.clone();
                let admission_error = std::cell::RefCell::new(None);
                let admission = capture.add_conditional_if(
                    &stream,
                    |handle| -> PyResult<()> {
                        let result = session
                            .owner()?
                            .record_prepared_step_admission(&native, handle)
                            .map_err(xlog_err);
                        if let Err(value) = &result {
                            *admission_error.borrow_mut() = Some(value.clone_ref(py));
                        }
                        result
                    },
                    |body| {
                        body.capture_on_stream(&stream, || -> PyResult<()> {
                            let result = session
                                .owner()?
                                .record_prepared_step_inputs(&native)
                                .map_err(xlog_err);
                            if let Err(value) = &result {
                                *admission_error.borrow_mut() = Some(value.clone_ref(py));
                            }
                            result
                        })
                    },
                );
                if let Err(graph_error) = admission {
                    return Err(admission_error
                        .into_inner()
                        .unwrap_or_else(|| xlog_err(graph_error)));
                }

                for bank in 0..2 {
                    let requested_error = std::cell::RefCell::new(None);
                    let requested = capture.add_conditional_if(
                        &stream,
                        |handle| -> PyResult<()> {
                            let result = session
                                .owner()?
                                .record_prepared_step_requested_bank_gate(&native, bank, handle)
                                .map_err(xlog_err);
                            if let Err(value) = &result {
                                *requested_error.borrow_mut() = Some(value.clone_ref(py));
                            }
                            result
                        },
                        |body| {
                            body.capture_on_stream(&stream, || -> PyResult<()> {
                                let result = (|| {
                                    session
                                        .owner()?
                                        .begin_prepared_step_bank_capture(&native, bank)
                                        .map_err(xlog_err)?;
                                    session
                                        .owner()?
                                        .record_prepared_bank_model_content(&native, bank)
                                        .map_err(xlog_err)?;
                                    let enqueue = recording_callback(check, || {
                                        prepared.getattr("enqueue_step")
                                    })?;
                                    recording_callback(check, || {
                                        let value = enqueue.call1((step.clone_ref(py), bank))?;
                                        if !value.is_none() {
                                            return Err(invalid(
                                                "enqueue_step must return None, not a host result",
                                            ));
                                        }
                                        Ok(())
                                    })?;
                                    session
                                        .owner()?
                                        .enqueue_prepared_transition(&native, bank)
                                        .map_err(xlog_err)
                                })();
                                if let Err(value) = &result {
                                    *requested_error.borrow_mut() = Some(value.clone_ref(py));
                                }
                                result
                            })
                        },
                    );
                    if let Err(graph_error) = requested {
                        return Err(requested_error
                            .into_inner()
                            .unwrap_or_else(|| xlog_err(graph_error)));
                    }
                }

                let drain_error = std::cell::RefCell::new(None);
                let drain = capture.add_conditional_if(
                    &stream,
                    |handle| -> PyResult<()> {
                        let result = session
                            .owner()?
                            .record_prepared_step_drain_gate(&native, handle)
                            .map_err(xlog_err);
                        if let Err(value) = &result {
                            *drain_error.borrow_mut() = Some(value.clone_ref(py));
                        }
                        result
                    },
                    |body| {
                        body.capture_on_stream(&stream, || -> PyResult<()> {
                            let result = session
                                .owner()?
                                .enqueue_prepared_drain(&native)
                                .map_err(xlog_err);
                            if let Err(value) = &result {
                                *drain_error.borrow_mut() = Some(value.clone_ref(py));
                            }
                            result
                        })
                    },
                );
                if let Err(graph_error) = drain {
                    return Err(drain_error
                        .into_inner()
                        .unwrap_or_else(|| xlog_err(graph_error)));
                }

                let release_error = std::cell::RefCell::new(None);
                let release = capture.add_conditional_if(
                    &stream,
                    |handle| -> PyResult<()> {
                        let result = session
                            .owner()?
                            .record_prepared_step_active_gate(&native, handle)
                            .map_err(xlog_err);
                        if let Err(value) = &result {
                            *release_error.borrow_mut() = Some(value.clone_ref(py));
                        }
                        result
                    },
                    |body| {
                        body.capture_on_stream(&stream, || -> PyResult<()> {
                            let result = session
                                .owner()?
                                .record_prepared_step_release(&native)
                                .map_err(xlog_err);
                            if let Err(value) = &result {
                                *release_error.borrow_mut() = Some(value.clone_ref(py));
                            }
                            result
                        })
                    },
                );
                if let Err(graph_error) = release {
                    return Err(release_error
                        .into_inner()
                        .unwrap_or_else(|| xlog_err(graph_error)));
                }
                check()?;
            }
            // Instantiation and any graph destructor run outside the owner lock.
            let mut executable = Some(capture.instantiate().map_err(xlog_err)?);
            check()?;
            let installed = session
                .owner()?
                .finish_prepared_segment(&mut executable)
                .map_err(xlog_err);
            drop(executable);
            installed
        })();
        // Cleanup is mandatory even when a callback invalidated the task. It
        // cannot hide that refusal, replace the graph, or suppress an exception.
        let before_cleanup = check();
        let cleanup = memory_owner.finish(recorded.as_ref().err());
        let cleanup = finish_with_cleanup(py, before_cleanup, cleanup);
        let cleanup = finish_with_cleanup(py, cleanup, check());
        finish_with_cleanup(py, recorded, cleanup)?;
        issued.state()?.finish_build(&scope)?;
        guard.committed = true;
        Ok(())
    }

    /// Submit the original built graph exactly once, await actual completion,
    /// and refresh original rights before exposing any result (including refusal
    /// and drain). No producer callback runs between submission and completion.
    /// Return (genuine final parent, ordered step rows). Each row is
    /// (prepared_step, transition_or_None, skipped_status, refusal_or_None,
    /// policy_invocation_or_None, selected_bank_or_None). Refusal is a typed native
    /// evidence dictionary. Every completed row carries the validated publication bank.
    /// A completed drain has no policy invocation or fabricated RNG draw.
    #[cfg(feature = "semantic-policy")]
    #[pyo3(signature = (task_use, *, operation, snapshot, refresh_snapshot))]
    fn execute_segment(
        &self,
        py: Python<'_>,
        task_use: Py<PySemanticTransitionTaskUse>,
        operation: &Bound<'_, PyAny>,
        snapshot: &Bound<'_, PyAny>,
        refresh_snapshot: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        let issued = task_use.borrow(py);
        self.require_read_issued(&issued)?;
        if session.recording.load(Ordering::Acquire) || session.retiring.load(Ordering::Acquire) {
            return Err(invalid(
                "segment submission cannot overlap a producer ownership scope",
            ));
        }
        if !refresh_snapshot.is_callable() {
            return Err(invalid(
                "completion requires the original current authority callback",
            ));
        }
        let mut budget = 16 * 1024 * 1024;
        let operation = ColdValue::read(operation, &mut budget, 0)?
            .text()?
            .to_owned();
        let snapshot = AuthoritySnapshot::parse(&ColdValue::read(snapshot, &mut budget, 0)?)?;
        let (scope, steps) = {
            let stored = session
                .prepared_segment
                .lock()
                .map_err(|_| invalid("prepared segment mutex is poisoned"))?;
            let stored = stored
                .as_ref()
                .ok_or_else(|| invalid("no original built segment"))?;
            if stored.task_use.as_ptr() != task_use.as_ptr() {
                return Err(invalid("built segment belongs to another task use"));
            }
            (
                Arc::clone(&stored.scope),
                stored
                    .steps
                    .iter()
                    .map(|step| step.clone_ref(py))
                    .collect::<Vec<_>>(),
            )
        };
        let mut guard = PreparedBuildGuard {
            session: &session,
            task_use: &issued,
            committed: false,
        };
        let expected = {
            let mut owner = session.owner()?;
            issued.require_identity(&owner)?;
            let mut state = issued.state()?;
            state.admit_built_segment(&scope, &issued.authority, operation.clone(), snapshot)?;
            let expected = Self::execution_binding(&state, false)?;
            owner
                .launch_prepared_segment(state.snapshot.canonical.clone())
                .map_err(xlog_err)?;
            expected
        };
        let shared = &*session;
        let outcomes = py.detach(|| {
            let mut owner = shared
                .inner
                .lock()
                .map_err(|_| invalid("native semantic session owner mutex is poisoned"))?;
            owner.complete_prepared_segment().map_err(xlog_err)
        })?;
        let check = || -> PyResult<()> {
            self.require_read_issued(&issued)?;
            let owner = session.owner()?;
            issued.require_identity(&owner)?;
            if !owner.prepared_segment_completion_known()
                || Self::execution_binding(&*issued.state()?, false)? != expected
            {
                return Err(invalid(
                    "completed segment authority changed during result handoff",
                ));
            }
            Ok(())
        };
        let fresh = recording_callback(check, || refresh_snapshot.call0())?;
        let fresh = AuthoritySnapshot::parse(&ColdValue::read(&fresh, &mut budget, 0)?)?;
        let lease = {
            let mut owner = session.owner()?;
            issued.require_identity(&owner)?;
            let mut state = issued.state()?;
            if Self::execution_binding(&state, false)? != expected {
                return Err(invalid(
                    "completion snapshot no longer belongs to this original operation",
                ));
            }
            state.revalidate_activation(&issued.authority, fresh)?;
            owner.acquire().map_err(xlog_err)?
        };
        let parent = Py::new(
            py,
            PySemanticPublishedParent {
                session: self.session.clone_ref(py),
                task_use: task_use.clone_ref(py),
                inner: Mutex::new(lease),
                continuation_producers: Mutex::new(Vec::new()),
            },
        )?;
        if outcomes.len() != steps.len() {
            return Err(invalid("native completion changed the original step count"));
        }
        let mut rows = Vec::with_capacity(steps.len());
        for (step, completed) in steps.into_iter().zip(outcomes) {
            let (transition, status, refusal, invocation, bank) = match completed {
                xlog_cuda::SemanticPreparedStepOutcome::Skipped { status, .. } => {
                    (None, status, None, py.None(), None)
                }
                xlog_cuda::SemanticPreparedStepOutcome::Completed {
                    invocation,
                    bank,
                    transition,
                    outcome,
                    ..
                } => {
                    let label = match transition {
                        SemanticTransitionKind::Proposal => "proposal",
                        SemanticTransitionKind::Recompute => "recompute",
                        SemanticTransitionKind::Update => "update",
                        SemanticTransitionKind::Drain => "drain",
                    };
                    let refusal = transition_refusal_report(py, &outcome)?;
                    let invocation = if transition == SemanticTransitionKind::Proposal {
                        Py::new(
                            py,
                            self.issue_prepared_policy_invocation(
                                py,
                                task_use.clone_ref(py),
                                step.clone_ref(py),
                                invocation,
                                outcome,
                                operation == "training",
                            )?,
                        )?
                        .into_any()
                    } else {
                        py.None()
                    };
                    (Some(label), 0, refusal, invocation, Some(bank))
                }
            };
            rows.push(
                (step, transition, status, refusal, invocation, bank)
                    .into_pyobject(py)?
                    .unbind(),
            );
        }
        let result = (parent, PyTuple::new(py, rows)?)
            .into_pyobject(py)?
            .unbind();
        guard.committed = true;
        Ok(result)
    }

    /// Creator-thread terminal cleanup after every original backward/final-use.
    /// Join exact consumer streams before releasing the graph or invoking the
    /// original producer's finish_segment(). External aliases/invocations must
    /// be retired by their owners; native custody refuses premature release.
    /// Unknown completion retains all original resources and cannot enter here.
    #[cfg(feature = "semantic-policy")]
    #[pyo3(signature = (task_use, *, consumer_streams))]
    fn retire_segment(
        &self,
        py: Python<'_>,
        task_use: Py<PySemanticTransitionTaskUse>,
        consumer_streams: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        self.require_read_issued(&task_use.borrow(py))?;
        if session.importing.load(Ordering::Acquire)
            || session.recording.load(Ordering::Acquire)
            || session.retiring.swap(true, Ordering::AcqRel)
        {
            return Err(invalid(
                "segment retirement cannot be nested or overlap recording/import",
            ));
        }
        let _retirement = PreparedRetirementGuard(&session.retiring);
        let snapshot = task_use.borrow(py).state()?.snapshot.canonical.clone();
        let mut budget = 16 * 1024 * 1024;
        let streams = ColdValue::read(consumer_streams, &mut budget, 0)?;
        let streams = streams
            .sequence()?
            .iter()
            .map(ColdValue::unsigned)
            .collect::<PyResult<Vec<_>>>()?;
        if streams.is_empty()
            || streams
                .iter()
                .any(|&stream| stream == 0 || stream == 2 || stream > i64::MAX as u64)
        {
            return Err(invalid(
                "final segment retirement requires exact supported consumer streams",
            ));
        }
        let (steps, resources, producers_retired) = {
            let stored = session
                .prepared_segment
                .lock()
                .map_err(|_| invalid("prepared segment mutex is poisoned"))?;
            let stored = stored
                .as_ref()
                .ok_or_else(|| invalid("no original prepared segment to retire"))?;
            if stored.task_use.as_ptr() != task_use.as_ptr() {
                return Err(invalid("segment retirement belongs to another task use"));
            }
            (
                stored
                    .steps
                    .iter()
                    .map(|step| step.clone_ref(py))
                    .collect::<Vec<_>>(),
                Arc::clone(&stored.resources),
                stored.producers_retired,
            )
        };
        let graph = {
            let mut owner = session.owner()?;
            for step in &steps {
                owner
                    .quiesce_prepared_step(&step.borrow(py).inner, &streams)
                    .map_err(xlog_err)?;
            }
            owner
                .take_prepared_executable_for_retirement()
                .map_err(xlog_err)?
        };
        drop(graph);
        let prepared = {
            let retained = resources
                .producers
                .lock()
                .map_err(|_| invalid("prepared producer owner mutex is poisoned"))?;
            retained
                .get(1)
                .ok_or_else(|| invalid("original prepared producer is absent"))?
                .clone_ref(py)
        };
        if !producers_retired {
            let check = || -> PyResult<()> {
                let issued = task_use.borrow(py);
                self.require_read_issued(&issued)?;
                if issued.state()?.snapshot.canonical != snapshot {
                    return Err(invalid(
                        "original task authority changed during producer retirement",
                    ));
                }
                Ok(())
            };
            let finish = recording_callback(check, || prepared.bind(py).getattr("finish_segment"))?;
            let finished = recording_callback(check, || finish.call0())?;
            if !finished.is_none() {
                return Err(invalid(
                    "finish_segment must return None after retiring original producers",
                ));
            }
            session
                .prepared_segment
                .lock()
                .map_err(|_| invalid("prepared segment mutex is poisoned"))?
                .as_mut()
                .ok_or_else(|| invalid("original segment was removed during producer retirement"))?
                .producers_retired = true;
        }
        for step in &steps {
            let step = step.borrow(py);
            let continuation = std::mem::take(
                &mut *step
                    .continuation_producers
                    .lock()
                    .map_err(|_| invalid("continuation producer ownership mutex is poisoned"))?,
            );
            let policy = std::mem::take(
                &mut *step
                    .policy_inputs
                    .lock()
                    .map_err(|_| invalid("prepared policy inputs mutex is poisoned"))?,
            );
            drop(policy);
            drop(continuation);
        }
        drain_export_owners();
        for step in &steps {
            session
                .owner()?
                .release_prepared_step(&step.borrow(py).inner, &streams)
                .map_err(xlog_err)?;
            // Preserve only unretired steps if another external alias delays a
            // later release. A retry never re-releases an already removed owner.
            let removed = {
                let mut stored = session
                    .prepared_segment
                    .lock()
                    .map_err(|_| invalid("prepared segment mutex is poisoned"))?;
                let stored = stored
                    .as_mut()
                    .ok_or_else(|| invalid("original segment disappeared during retirement"))?;
                let index = stored
                    .steps
                    .iter()
                    .position(|original| original.as_ptr() == step.as_ptr())
                    .ok_or_else(|| invalid("original retirement step changed"))?;
                stored.steps.remove(index)
            };
            drop(removed);
        }
        let native_resources = session
            .owner()?
            .take_prepared_resources_for_retirement()
            .map_err(xlog_err)?;
        drop(native_resources);
        let retained = session
            .prepared_segment
            .lock()
            .map_err(|_| invalid("prepared segment mutex is poisoned"))?
            .take();
        drop(retained);
        drop(resources);
        drop(prepared);
        drop(steps);
        drain_export_owners();
        Ok(())
    }

    /// Permanently abort this shared native Session after trusted application
    /// validation fails, including verification after bind_parent has published.
    /// Saved task uses, parents and other controllers of this same Session can
    /// no longer continue execution. No current task use is required, and
    /// repeated abort calls are safe. Resources remain retained; abort performs
    /// no device synchronization, resource release or reset. A fatal CUDA error
    /// still requires terminating the task process.
    fn abort(&self, py: Python<'_>) -> PyResult<()> {
        self.session.borrow(py).require_creator()?;
        let session = self.session.borrow(py);
        session.owner()?.abort();
        Ok(())
    }

    /// Inspect cold native observation coverage before constructing the task's
    /// authority graph. The replay transport, exact selection and byte limits
    /// are identical to import_task; no replay is executed or restored here.
    /// Returns (root_digest, root_extents, query_records, contributors), where
    /// each contributor is (query_ordinal, original_target_or_None, original_support).
    /// These are original admission coordinates, not learned features, current
    /// device receipts or use grants. A derived target remains None even when
    /// its value equals an admitted query. import_task independently recomputes
    /// this coverage and requires original targets before issuing its TaskUse.
    #[pyo3(signature = (*, statement_records, allowed_support_records, task_program_source, task_query_ordinals, task_scoring, admissible_truth_masks, replay_rows, replay_selection, max_material_bytes, max_total_material_bytes, max_evidence_bytes))]
    #[expect(
        clippy::too_many_arguments,
        reason = "observation coverage retains every independent task and replay input"
    )]
    fn task_observation_roots(
        &self,
        py: Python<'_>,
        statement_records: &Bound<'_, PyAny>,
        allowed_support_records: &Bound<'_, PyAny>,
        task_program_source: &Bound<'_, PyAny>,
        task_query_ordinals: &Bound<'_, PyAny>,
        task_scoring: &Bound<'_, PyAny>,
        admissible_truth_masks: &Bound<'_, PyAny>,
        replay_rows: &Bound<'_, PyAny>,
        replay_selection: &Bound<'_, PyAny>,
        max_material_bytes: &Bound<'_, PyAny>,
        max_total_material_bytes: &Bound<'_, PyAny>,
        max_evidence_bytes: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        let mut budget = 16 * 1024 * 1024;
        let [material_limit, total_material_limit, evidence_limit] = read_task_replay_limits(
            max_material_bytes,
            max_total_material_bytes,
            max_evidence_bytes,
            &mut budget,
        )?;
        let spec = read_task_evaluation_spec(
            statement_records,
            allowed_support_records,
            task_program_source,
            task_query_ordinals,
            task_scoring,
            admissible_truth_masks,
            &mut budget,
        )?;
        let rows = read_replay_rows(
            replay_rows,
            &mut budget,
            material_limit,
            total_material_limit,
            evidence_limit,
        )?;
        let rows = rows
            .sequence()?
            .iter()
            .map(ReplayRow::parse)
            .collect::<PyResult<Vec<_>>>()?;
        let (_, selected) =
            decode_selected_replay(&rows, &ColdValue::read(replay_selection, &mut budget, 0)?)?;
        let session = self.session.borrow(py);
        let roots = session
            .owner()?
            .task_observation_roots(&spec, selected.as_ref().map(|binding| &binding.material))
            .map_err(xlog_err)?;
        Ok((
            PyBytes::new(py, roots.root_digest.as_bytes()),
            PyTuple::new(py, roots.root_extents)?,
            PyTuple::new(py, roots.query_records)?,
            PyTuple::new(py, roots.contributors)?,
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// Import resolved rights and lineage into this controller's real Session.
    ///
    /// All arguments are mandatory. ``task_scope`` is ``(controller, tenant,
    /// security_scope, request_scope, repository_scope)``. ``statement_records``
    /// contains the three admitted native observer statement indices;
    /// ``allowed_support_records`` contains admitted support indices.
    /// ``task_program_source`` is the complete authored XLOG observer program,
    /// including all concrete input facts. ``task_query_ordinals`` maps its three
    /// selected zero-arity query results to the ordered statement records.
    /// The native owner executes that program during every cold task binding.
    /// ``task_scoring`` supplies (correct, all_correct, work, improvement,
    /// refusal, spent) unsigned weights. ``admissible_truth_masks`` supplies three
    /// bit masks over (neither, true, false, both), in that bit order.
    ///
    /// ``training_objective`` is None only when replay_rows is empty. Otherwise it
    /// is ``(identity, evaluator_f64_bits, coefficient_f32_bits, cost, groups,
    /// canaries)``. The evaluator pair and nine coefficient words are exact IEEE
    /// bit patterns. Cost is ``(unit_identity, positive_cap)``. Each group is
    /// ``(kind, positive_denominator, strictly_increasing_row_ordinals)`` and the
    /// complete mandatory group roster must cover every row. Each canary is
    /// ``(kind, row_ordinal, lower_f64_bits, upper_f64_bits, memory_limit,
    /// fuel_limit, identity)``. Native recomputes the objective identity and owns
    /// the fixed-shape roster on device.
    ///
    /// ``live_authorities`` rows are ``(LiveProvenance.canonical(),
    /// retention_deadline_utc_us)``. ``replay_rows`` rows are
    /// ``(basis, identity, canonical_record_line, materials, evidence_bytes)``.
    /// ``basis`` is ``episode`` or ``anchor``. ``identity`` is the original
    /// ``(training_view, training_view_digest, content_sha256, source_length,
    /// block_size, target_triples)``; training_view retains the producer's nine
    /// canonical fields (manifest, example, split, tokenizer hash/revision,
    /// mask policy, seed, epoch, variant), including exact unbounded integers.
    /// No additional native identity or rewritten episode record is required.
    /// Targets retain ``(p, p, source_length+p)`` from the original projection,
    /// not a physical row permutation. The original training-view material
    /// retains its 136-byte header, answer boundary, supervision and geometry.
    /// An anchor retains its original admission evidence and retention group;
    /// that evidence cannot authorize native execution replay or actor credit.
    /// Materials are the complete ordered ``(MaterialReference.payload(),
    /// bytes_or_None)`` closure. Evidence is the original opaque bytes, not a
    /// caller-decoded receipt. Identity-only rows are not a replay carrier.
    /// The exact positive integers ``max_material_bytes``,
    /// ``max_total_material_bytes`` and ``max_evidence_bytes`` come from the
    /// trusted replay-kit caps. Material and evidence limits apply per item;
    /// the total counts each unique material SHA256 once across all rows.
    /// Equal material digests must name equal bytes and share retained storage.
    /// All limits are checked before copying payloads, and are not CUDA limits.
    ///
    /// ``dependencies`` rows are ``(identity, kind, data_parents, control_parents,
    /// live_envelope_refs, target_key_or_None, native_record_or_None)``. Kinds are ``constant``, ``source``,
    /// ``mask``, ``target_value`` and ``derived``. A target key is
    /// ``(replay_row_index, logical_position)``. The trusted caller supplies the
    /// complete graph and exact acquired MASK join; the importer rejects missing
    /// parents, cycles, missing MASK targets, and target-value ancestors through
    /// either edge kind. ``native_record`` is ``('statement', index)``,
    /// ``('support', index)`` or ``('observer', 0)``. The exact native observer,
    /// both queried statements, every allowed mutation support, and every actual
    /// queried-head contributor's original target/support must each have one
    /// distinct node. The cold task_observation_roots getter exposes that native
    /// coverage without granting mutation rights or claiming current causality.
    /// Their full closures are always joined with ``feedback_roots``, regardless
    /// of which feedback roots the caller explicitly lists. Observer zero names
    /// the retained program and the actual native execution bound to this task.
    ///
    /// ``initial_sources`` rows are ``(source_id, manifest_hash, tokenizer_hash,
    /// tokenizer_revision, manifest_entry, tokens)``. The trusted application
    /// obtains observed tokens from its data reader and the complete entry from
    /// the same sealed manifest. The ten-field entry retains example, split,
    /// acquired/transform/normalized digests, decontamination, leakage keys,
    /// full corpus/live origin, admission and the observed-field binding.
    /// That final binding is ``(field, payload_sha256, transformation_sha256,
    /// token_count)``. The normalized mixed training-row digest is never used
    /// as the observation digest. Native verifies the actual observed payload
    /// and transformation; provenance references alone grant no operation.
    /// ``source_mapping`` rows are ``(source_id, token_offset, logical_position)``;
    /// no second token ID is accepted. Each source resolves through the same
    /// complete data/control graph; target-value ancestry is rejected even when
    /// the source is not explicitly listed as a feedback root.
    ///
    /// Each separate grant family contains ``(grant_ref, issuer_provenance,
    /// decision, task_ref, dependency_ids, expiry, revocation_ref)``. Decision is
    /// explicitly ``allow`` or ``deny``; training records additionally contain the
    /// canonical learning purpose as their eighth field. Grant references and
    /// issuer strings retain provenance, not cryptographic authentication. The
    /// resolved decision originates in the trusted controlling application.
    /// Empty explicit families grant no corresponding use; no rights are defaulted.
    /// A training use requires one compatible learning purpose across its whole
    /// dependency closure. Independently scoped grants are intersected for this
    /// shared use, not combined into a wider model-use permission.
    ///
    /// ``snapshot`` is ``(revision, observed_at, preregistered_segment_end, live_snapshots,
    /// revoked_grant_refs)``; each live snapshot is ``(bound_envelope.canonical(),
    /// segment_end, revoked_refs)``. Times are ``(original_ISO_string, utc_us)``;
    /// the application uses the canonical timezone-aware parser before import.
    /// The task-level end is mandatory even for corpus-only tasks; each live
    /// deadline must equal it exactly and refresh cannot change it.
    /// Snapshot revision is supplied by the trusted current authority source,
    /// not inferred from a hash or native Session creation. Revoked references
    /// must be sorted and unique. This bounded transport accepts admitted observed data,
    /// exact builtin primitive containers, at most 32 nesting levels and
    /// 16,777,216 primitive-input units: one per node plus scalar UTF-8 bytes.
    /// This is an input bound, not a native heap or CUDA allocation budget.
    /// ``replay_selection`` is None for a fresh task, otherwise the original
    /// (row_ordinal, complete_row_identity). All rows retain their original order;
    /// validating their material does not claim to have executed every row.
    /// Fresh import requires replay_operation and all callbacks to be None.
    /// Selected replay requires an explicit current inference/training operation
    /// and trusted application callbacks, not permissions inferred from archive.
    ///
    /// restore_invocation(task_use, restored_parent, full_row, transition) returns
    /// (original_model_output, continuation_arguments), where the second tuple
    /// is (text_rows, text_row_count, selected_text, active_rows, active_row_count,
    /// numerical_admissibility, tensors, producer_witness, consumer_stream).
    /// The six services are original CUDA DLPack producers; the witness is the
    /// original transient capture for this restored parent and exact roster.
    /// For a proposal, pack_policy(task_use, restored_parent, original_model_output)
    /// returns (binding, text_logits, parameters, product_support), computed by
    /// the same Runtime after the genuine continuation has been bound. For
    /// recompute/drain pack_policy must be None; no policy draw is synthesized.
    /// These callbacks receive genuine task-bound readers only within a dynamic
    /// import read scope; ordinary controller execution/backward remains closed.
    ///
    /// After the first full native successor comparison, finish_invocation(task_use,
    /// restored_parent) destroys the Runtime's temporary graph/aliases/witnesses
    /// and returns its exact consumer-stream tuple. refresh_snapshot() then
    /// obtains a newer current authority snapshot. Native joins those streams,
    /// verifies full P/S again after all callbacks, and releases both readers.
    /// Only then does this same TaskUse become Imported. Any failed import
    /// permanently aborts the shared Session, including saved callback references.
    /// No Session, task-state or reader mutex is held across a Python callback.
    #[pyo3(signature = (*, task_ref, task_scope, statement_records, allowed_support_records, task_program_source, task_query_ordinals, task_scoring, admissible_truth_masks, live_authorities, replay_rows, training_objective, max_material_bytes, max_total_material_bytes, max_evidence_bytes, dependencies, feedback_roots, publication_grants, inference_grants, training_grants, initial_sources, source_mapping, snapshot, replay_selection, replay_operation, restore_invocation, pack_policy, finish_invocation, refresh_snapshot))]
    #[expect(
        clippy::too_many_arguments,
        reason = "task import binds the complete authority, replay, and callback contract"
    )]
    fn import_task(
        &self,
        py: Python<'_>,
        task_ref: &Bound<'_, PyAny>,
        task_scope: &Bound<'_, PyAny>,
        statement_records: &Bound<'_, PyAny>,
        allowed_support_records: &Bound<'_, PyAny>,
        task_program_source: &Bound<'_, PyAny>,
        task_query_ordinals: &Bound<'_, PyAny>,
        task_scoring: &Bound<'_, PyAny>,
        admissible_truth_masks: &Bound<'_, PyAny>,
        live_authorities: &Bound<'_, PyAny>,
        replay_rows: &Bound<'_, PyAny>,
        training_objective: &Bound<'_, PyAny>,
        max_material_bytes: &Bound<'_, PyAny>,
        max_total_material_bytes: &Bound<'_, PyAny>,
        max_evidence_bytes: &Bound<'_, PyAny>,
        dependencies: &Bound<'_, PyAny>,
        feedback_roots: &Bound<'_, PyAny>,
        publication_grants: &Bound<'_, PyAny>,
        inference_grants: &Bound<'_, PyAny>,
        training_grants: &Bound<'_, PyAny>,
        initial_sources: &Bound<'_, PyAny>,
        source_mapping: &Bound<'_, PyAny>,
        snapshot: &Bound<'_, PyAny>,
        replay_selection: &Bound<'_, PyAny>,
        replay_operation: &Bound<'_, PyAny>,
        restore_invocation: &Bound<'_, PyAny>,
        pack_policy: &Bound<'_, PyAny>,
        finish_invocation: &Bound<'_, PyAny>,
        refresh_snapshot: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PySemanticTransitionTaskUse>> {
        self.session.borrow(py).require_creator()?;
        let session = self.session.borrow(py);
        let mut import = ColdImportGuard::begin(&session, None)?;
        let mut budget = 16 * 1024 * 1024;
        let [material_limit, total_material_limit, evidence_limit] = read_task_replay_limits(
            max_material_bytes,
            max_total_material_bytes,
            max_evidence_bytes,
            &mut budget,
        )?;
        let values = [
            task_ref,
            task_scope,
            live_authorities,
            replay_rows,
            dependencies,
            feedback_roots,
            publication_grants,
            inference_grants,
            training_grants,
        ]
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            if index == 3 {
                read_replay_rows(
                    value,
                    &mut budget,
                    material_limit,
                    total_material_limit,
                    evidence_limit,
                )
            } else {
                ColdValue::read(value, &mut budget, 0)
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
        let mut authority = TaskAuthority::parse(&values)?;
        let training_views = authority
            .replay
            .iter()
            .map(ReplayRow::training_view_row)
            .collect::<PyResult<Vec<_>>>()?;
        let training_objective =
            read_training_objective(&ColdValue::read(training_objective, &mut budget, 0)?)?;
        if training_views.is_empty() != training_objective.is_none() {
            return Err(invalid(
                "training objective must exist exactly when replay rows exist",
            ));
        }
        authority.bind_initial_sources(
            &ColdValue::read(initial_sources, &mut budget, 0)?,
            &ColdValue::read(source_mapping, &mut budget, 0)?,
        )?;
        let snapshot = AuthoritySnapshot::parse(&ColdValue::read(snapshot, &mut budget, 0)?)?;
        authority.check_snapshot(&snapshot)?;
        let spec = read_task_evaluation_spec(
            statement_records,
            allowed_support_records,
            task_program_source,
            task_query_ordinals,
            task_scoring,
            admissible_truth_masks,
            &mut budget,
        )?;
        let (selection, selected_material) = decode_selected_replay(
            &authority.replay,
            &ColdValue::read(replay_selection, &mut budget, 0)?,
        )?;
        let observations = session
            .owner()?
            .task_observation_roots(
                &spec,
                selected_material.as_ref().map(|binding| &binding.material),
            )
            .map_err(xlog_err)?;
        authority.bind_native_reads(&observations, &spec.allowed_support_records)?;
        let operation = ColdValue::read(replay_operation, &mut budget, 0)?;
        let (phase, transition) = if let Some(material) = &selected_material {
            let operation = operation.text()?.to_owned();
            authority.check_use(&operation, &snapshot, true)?;
            for callback in [restore_invocation, finish_invocation, refresh_snapshot] {
                if !callback.is_callable() {
                    return Err(invalid("selected replay requires trusted restore, cleanup and current-authority callbacks"));
                }
            }
            let kind = material.material.transition_kind();
            if kind == SemanticTransitionKind::Proposal {
                if !pack_policy.is_callable() {
                    return Err(invalid(
                        "proposal replay requires its real Runtime policy packer",
                    ));
                }
                #[cfg(not(feature = "semantic-policy"))]
                return Err(invalid("proposal replay requires a semantic-policy build"));
            } else if !pack_policy.is_none() {
                return Err(invalid("no-draw replay must not supply a policy packer"));
            }
            import.original_decisions = Some(Arc::clone(&material.original_decisions));
            (
                TaskUsePhase::Importing {
                    operation,
                    reads: false,
                },
                Some(kind),
            )
        } else {
            if operation != ColdValue::None
                || [
                    restore_invocation,
                    pack_policy,
                    finish_invocation,
                    refresh_snapshot,
                ]
                .iter()
                .any(|callback| !callback.is_none())
            {
                return Err(invalid(
                    "fresh import requires no replay operation or callbacks",
                ));
            }
            (TaskUsePhase::Imported, None)
        };
        let (identity, task_epoch) = {
            let mut owner = session.owner()?;
            owner.bind_task_evaluation(spec).map_err(xlog_err)?;
            if let Some(objective) = training_objective {
                owner
                    .bind_training_view_arena(training_views, objective)
                    .map_err(xlog_err)?;
            }
            if let Some(material) = &selected_material {
                owner
                    .restore_replay_material(&material.material)
                    .map_err(xlog_err)?;
            }
            (
                *owner
                    .task_evaluation_identity()
                    .ok_or_else(|| invalid("native import lost its task binding"))?
                    .as_bytes(),
                owner.task_evaluation_epoch(),
            )
        };
        let task_use = Py::new(
            py,
            PySemanticTransitionTaskUse {
                session: self.session.clone_ref(py),
                controller: Arc::clone(&self.identity),
                issuance: import.issuance.clone(),
                importing: Arc::clone(&session.importing),
                task_identity: identity,
                task_epoch,
                authority,
                state: Mutex::new(TaskUseState { phase, snapshot }),
            },
        )?;
        if let (Some(ordinal), Some(material), Some(kind)) =
            (selection, selected_material, transition)
        {
            let replay = self.replay_import(
                py,
                &task_use,
                ordinal,
                material,
                kind,
                restore_invocation,
                pack_policy,
                finish_invocation,
                refresh_snapshot,
                &import,
            );
            if let Err(error) = replay {
                if let Ok(mut state) = task_use.borrow(py).state() {
                    state.phase = TaskUsePhase::Refused;
                }
                return Err(error);
            }
        }
        import.committed = true;
        Ok(task_use)
    }

    /// Export one genuinely held transition into the existing replay carrier.
    /// Returns (predecessor_identity, successor_identity, predecessor_material,
    /// provenance_material, publication_evidence). Identities have the same four
    /// fields as parent.identity(). Each material is exact native bytes; the
    /// caller binds them to the canonical ExecutionMaterials references and
    /// keeps publication_evidence outside the episode's own hashed closure.
    /// The successor's full snapshot is deliberately not an episode material.
    ///
    /// Both original readers must remain held. A fresh trusted snapshot must
    /// authorize the same admitted operation across the full dependency closure.
    /// Export does not refresh execution authority, publish, or grant restore
    /// permission. This cold operation intentionally copies complete bytes.
    /// The Runtime must stop further writers during export and have registered
    /// every consumer stream through the real native reader operations. Export
    /// completes their submitted work and guards every live P/S range, including
    /// tensors, before copying. Both later reader releases must include legacy
    /// stream 1, used by this controller's complete-content guards, in addition
    /// to the Runtime's original consumer streams. Aliases may remain retained.
    #[pyo3(signature = (task_use, *, predecessor, successor, snapshot))]
    fn export_replay_materials(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        predecessor: &PySemanticPublishedParent,
        successor: &PySemanticPublishedParent,
        snapshot: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        self.require_issued(task_use)?;
        predecessor.require_task(py, task_use)?;
        successor.require_task(py, task_use)?;
        let mut budget = 16 * 1024 * 1024;
        let snapshot = AuthoritySnapshot::parse(&ColdValue::read(snapshot, &mut budget, 0)?)?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        task_use.require_current(&owner)?;
        let state = task_use.state()?;
        let TaskUsePhase::Segment(operation) = &state.phase else {
            return Err(invalid(
                "replay export requires its admitted original operation",
            ));
        };
        snapshot.newer_than(&state.snapshot)?;
        task_use.authority.check_use(operation, &snapshot, false)?;
        if std::ptr::eq(predecessor, successor) {
            return Err(invalid(
                "replay export requires distinct predecessor and successor readers",
            ));
        }
        let predecessor = predecessor.lease()?;
        let successor = successor.lease()?;
        let predecessor_identity = owner.published_identity(&predecessor).map_err(xlog_err)?;
        let successor_identity = owner.published_identity(&successor).map_err(xlog_err)?;
        let evidence = owner
            .published_replay_evidence(&predecessor, &successor)
            .map_err(xlog_err)?;
        let provenance = owner
            .published_replay_provenance(&predecessor, &successor)
            .map_err(xlog_err)?;
        let material = owner
            .published_state_material(&predecessor)
            .map_err(xlog_err)?;
        let identity = |value: xlog_cuda::SemanticPublishedIdentity| {
            (
                PyBytes::new(py, value.instance.as_bytes()),
                value.word,
                PyBytes::new(py, value.logical_digest.as_bytes()),
                PyBytes::new(py, value.state_digest.as_bytes()),
            )
        };
        Ok((
            identity(predecessor_identity),
            identity(successor_identity),
            PyBytes::new(py, &material),
            PyBytes::new(py, &provenance),
            PyBytes::new(py, &evidence),
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// Revalidate exact rights before each encoder or model forward and native transition.
    /// ``operation`` is explicitly ``inference`` or ``training``. A newer trusted
    /// snapshot must cover the unchanged preregistered worst-case segment end.
    /// The native fuel/terminal admission reads this exact acquired parent
    /// before encoder or model execution; this check does not itself spend fuel or publish.
    /// Later transitions use the same original operation and a fresh snapshot.
    #[pyo3(signature = (task_use, *, parent, transition, operation, snapshot))]
    fn before_segment(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
        transition: &Bound<'_, PyAny>,
        operation: &Bound<'_, PyAny>,
        snapshot: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.session.borrow(py).require_creator()?;
        self.require_issued(task_use)?;
        parent.require_task(py, task_use)?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        task_use.require_current(&owner)?;
        let mut budget = 16 * 1024 * 1024;
        let kind = transition_kind(&ColdValue::read(transition, &mut budget, 0)?)?;
        let operation = ColdValue::read(operation, &mut budget, 0)?
            .text()?
            .to_owned();
        let snapshot = AuthoritySnapshot::parse(&ColdValue::read(snapshot, &mut budget, 0)?)?;
        let mut state = task_use.state()?;
        state.admit_segment(&task_use.authority, operation, snapshot)?;
        if let Err(error) = owner.admit_transition(&*parent.lease()?, kind) {
            state.phase = TaskUsePhase::Refused;
            return Err(xlog_err(error));
        }
        Ok(())
    }

    /// Acquire exactly one sealed native bank. Cold import, layout/schema
    /// access and a pending transition are not substitutes for this operation.
    #[pyo3(signature = (task_use))]
    fn acquire(
        &self,
        py: Python<'_>,
        task_use: Py<PySemanticTransitionTaskUse>,
    ) -> PyResult<PySemanticPublishedParent> {
        self.session.borrow(py).require_creator()?;
        let issued = task_use.borrow(py);
        self.require_issued(&issued)?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        issued.require_current(&owner)?;
        let lease = owner.acquire().map_err(xlog_err)?;
        drop(issued);
        Ok(PySemanticPublishedParent {
            session: self.session.clone_ref(py),
            task_use,
            inner: Mutex::new(lease),
            continuation_producers: Mutex::new(Vec::new()),
        })
    }

    /// Capture the current bytes of real original tensors for this same parent.
    /// ``tensors`` uses the original typed layout/interval/producer rows accepted
    /// by bind_parent, including I64=7 and Bool8=8. Original Python producers and
    /// their autograd objects remain retained independently of mutable input rows.
    /// Role 0 is the private transient namespace, never a publication role.
    /// Its index is the operand's ordinal in the complete supplied roster,
    /// including repeated operands; it is not derived from values or storage.
    /// Nonpositional operands use logical_axis=u64::MAX and interval (0, 0).
    /// Verification requires the same complete roster and original storage.
    ///
    /// The trusted controller must call immediately after its real forward or
    /// transform and before public hooks, then bind the private witness to that
    /// original execution. Capture proves later content stability; it does not
    /// establish derivation, provenance or permission to import/replay a model.
    /// The issuing Runtime must retain this complete witness and every issued
    /// policy invocation, original producer, callback and saved autograd payload
    /// through native cotangents and the original Python backward. One guaranteed
    /// creating-thread cleanup must release that custody after backward finishes.
    /// External aliases require an explicit owner kept alive through their last
    /// use. Worker access does not make arbitrary Python finalizers thread-safe;
    /// neither a native alias alone nor a future deferred-release drain suffices.
    /// The witness retains the step, so final release refuses until it is destroyed.
    /// Bank-only retirement does not destroy its original baseline or producers.
    /// Producer and consumer streams join through the Session without a host
    /// payload/digest/status copy or synchronization. Verification and every
    /// dependent forward/backward/result/publication must preserve that ordering.
    /// Capture and verification require legacy default stream 1 or an explicit
    /// consumer stream; per-thread default stream 2 is rejected before handoff.
    /// Capture requires CUDA stream-ordered allocation support; native refuses
    /// a context that would require a synchronizing allocation fallback.
    /// Native feedback outputs already carry their producer's original seal.
    /// Capturing those aliases verifies that seal and requires their original
    /// type, shape, interval and roster coordinate; it never reseals changed bytes.
    #[pyo3(signature = (task_use, *, parent, tensors, consumer_stream))]
    fn capture_tensor_content(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        parent: &Bound<'_, PyAny>,
        tensors: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<PySemanticTensorContentWitness> {
        self.bind_tensor_content(
            py,
            task_use,
            ContentStepOwner::from_python(parent)?,
            tensors,
            consumer_stream,
            TensorContentBinding::Capture,
        )
    }

    /// Bind all actual model tensors to the original native model baseline.
    /// ``tensors`` has the existing typed layout/interval/producer rows, covering
    /// exactly roles 18/19/20 present in the acquired parent. Optional adapters
    /// must be present exactly when the original roster contains them. Native
    /// checks the admitted generation and sealed model contract (role 44).
    /// Expected content comes from the original device publication seals, never
    /// from a new capture of current model bytes. Initial producer addresses and
    /// valid physical strides may differ; type, shape and logical interval may
    /// not. Later witness.verify requires these same live storage/layout owners.
    /// A prepared parent requires ``bank`` so each recorded branch seals the
    /// resident model storage that its producer actually reads. Published parents
    /// reject ``bank``. Call before model reads and verify after callbacks and
    /// before late backward. This does not certify Python identity, hooks,
    /// configuration or numerical derivation; those remain the model owner's
    /// checks. Successful return is stream-ordered enqueue, not GPU acceptance.
    /// Retain the complete witness through creator-thread cleanup, as for
    /// capture_tensor_content; no first-reader lifetime is used as a baseline.
    #[pyo3(signature = (task_use, *, parent, tensors, consumer_stream, bank=None))]
    fn bind_model_content(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        parent: &Bound<'_, PyAny>,
        tensors: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
        bank: Option<usize>,
    ) -> PyResult<PySemanticTensorContentWitness> {
        self.bind_tensor_content(
            py,
            task_use,
            ContentStepOwner::from_python(parent)?,
            tensors,
            consumer_stream,
            TensorContentBinding::Model(bank),
        )
    }

    /// Retain the actual original-forward outputs for this acquired parent.
    /// ``tensors`` uses the typed tensor layout from cold parent import. The
    /// six service arguments are contiguous CUDA DLPack producers: text_rows
    /// U64[32,2], text_row_count U64[1], selected_text Bool8[32], active_rows
    /// U64[A,4], active_row_count U64[1], and numerical_admissibility Bool8[1].
    /// Selection is indexed by original source slot. Active rows contain
    /// (physical_row, source_slot, logical_position, kind), with FILLED=1,
    /// MASK=2 and FEEDBACK=3; invalid feedback slots leave physical gaps.
    /// The original transient ``producer_witness`` captures model tensors in
    /// their original order followed by these six services at role 0 indices
    /// N through N+5. Model-content witnesses cannot substitute for this capture.
    /// ``consumer_stream`` is explicit (or legacy stream 1); 0 and 2 are refused.
    /// Native validation checks the acquired source, prefix, services and
    /// generations before pending ranges enter the sole publication CAS. A
    /// prepared parent requires ``bank`` and retains a distinct continuation for
    /// each recorded branch; published parents reject it.
    #[pyo3(signature = (task_use, *, parent, text_rows, text_row_count, selected_text, active_rows, active_row_count, numerical_admissibility, tensors, producer_witness, consumer_stream, bank=None))]
    #[expect(
        clippy::too_many_arguments,
        reason = "continuation binding retains each typed producer and its witness"
    )]
    fn bind_continuation(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        parent: &Bound<'_, PyAny>,
        text_rows: &Bound<'_, PyAny>,
        text_row_count: &Bound<'_, PyAny>,
        selected_text: &Bound<'_, PyAny>,
        active_rows: &Bound<'_, PyAny>,
        active_row_count: &Bound<'_, PyAny>,
        numerical_admissibility: &Bound<'_, PyAny>,
        tensors: &Bound<'_, PyAny>,
        producer_witness: &PySemanticTensorContentWitness,
        consumer_stream: &Bound<'_, PyAny>,
        bank: Option<usize>,
    ) -> PyResult<()> {
        self.session.borrow(py).require_creator()?;
        self.require_read_issued(task_use)?;
        match ContentStepOwner::from_python(parent)? {
            ContentStepOwner::Published(parent) => {
                if bank.is_some() {
                    return Err(invalid(
                        "published continuation does not accept a prepared bank",
                    ));
                }
                self.require_issued(task_use)?;
                self.bind_continuation_in_execution(
                    py,
                    task_use,
                    ContentStepRef::Published(&parent.borrow(py)),
                    text_rows,
                    text_row_count,
                    selected_text,
                    active_rows,
                    active_row_count,
                    numerical_admissibility,
                    tensors,
                    producer_witness,
                    consumer_stream,
                    None,
                    None,
                )
            }
            ContentStepOwner::Prepared(step) => {
                let bank =
                    bank.ok_or_else(|| invalid("prepared continuation requires bank zero or one"))?;
                self.bind_continuation_in_execution(
                    py,
                    task_use,
                    ContentStepRef::Prepared(&step.borrow(py)),
                    text_rows,
                    text_row_count,
                    selected_text,
                    active_rows,
                    active_row_count,
                    numerical_admissibility,
                    tensors,
                    producer_witness,
                    consumer_stream,
                    Some(bank),
                    None,
                )
            }
        }
    }

    /// Bind the complete original model backing produced by this recorded
    /// update. The transient witness must cover allocations followed by typed
    /// model views, Bool8[1] numerical admissibility and the five U64[5,9]
    /// frozen canary result records produced after the selected-view backward,
    /// optimizer update and candidate cache rebuild. ``bank`` identifies the
    /// recorded branch that owns these original output producers.
    #[pyo3(signature = (task_use, *, step, bank, tensors, model_allocations, model_storages, model_views, numerical_admissibility, canary_results, allocation_witness, consumer_stream))]
    #[expect(
        clippy::too_many_arguments,
        reason = "prepared update binding retains model geometry and numerical admissibility"
    )]
    fn bind_prepared_update_output(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        step: &PySemanticPreparedStep,
        bank: usize,
        tensors: &Bound<'_, PyAny>,
        model_allocations: &Bound<'_, PyAny>,
        model_storages: &Bound<'_, PyAny>,
        model_views: &Bound<'_, PyAny>,
        numerical_admissibility: &Bound<'_, PyAny>,
        canary_results: &Bound<'_, PyAny>,
        allocation_witness: &PySemanticTensorContentWitness,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.session.borrow(py).require_creator()?;
        self.require_read_issued(task_use)?;
        if bank > 1 {
            return Err(invalid("prepared model update bank must be zero or one"));
        }
        let parent = ContentStepRef::Prepared(step);
        let mut budget = 16 * 1024 * 1024;
        let stream = parse_witness_consumer_stream(consumer_stream, &mut budget)?;
        let storages = parse_model_storages(model_storages, &mut budget)?;
        let views = parse_model_views(model_views, &mut budget)?;
        let expected = {
            let session = self.session.borrow(py);
            let owner = session.owner()?;
            let state = self.continuation_state(py, task_use, parent, &owner, None)?;
            step.require_recording_stream(&owner, stream)?;
            self.require_continuation_witness(py, parent, allocation_witness)?;
            if owner
                .prepared_transition_kind(&step.inner)
                .map_err(xlog_err)?
                != SemanticTransitionKind::Update
            {
                return Err(invalid(
                    "model update output belongs only to a prepared update",
                ));
            }
            self.continuation_binding(&state, parent, false)?
        };
        let check = || -> PyResult<()> {
            let session = self.session.borrow(py);
            let owner = session.owner()?;
            let state = self.continuation_state(py, task_use, parent, &owner, None)?;
            step.require_recording_stream(&owner, stream)?;
            self.require_continuation_witness(py, parent, allocation_witness)?;
            if self.continuation_binding(&state, parent, false)? != expected {
                return Err(invalid(
                    "prepared update authority changed during producer handoff",
                ));
            }
            Ok(())
        };
        let device = self.session.borrow(py).device_ordinal;
        let ParsedTensorInputs {
            handoff: allocation_handoff,
            producers: allocation_producers,
        } = parse_tensor_inputs_guarded(model_allocations, &mut budget, device, stream, &check)?;
        let mut producer_owners = TensorHandoff(vec![
            allocation_witness._inputs.clone_ref(py),
            model_allocations.clone().unbind(),
            tensors.clone().unbind(),
            canary_results.clone().unbind(),
        ]);
        producer_owners.0.extend(
            allocation_witness
                ._producers
                .iter()
                .map(|producer| producer.clone_ref(py)),
        );
        producer_owners.0.extend(allocation_producers);
        let ParsedTensorInputs {
            handoff: tensor_handoff,
            producers: tensor_producers,
        } = parse_tensor_inputs_guarded(tensors, &mut budget, device, stream, &check)?;
        producer_owners.0.extend(tensor_producers);
        validate_producer_device_guarded(numerical_admissibility, device, &check)?;
        producer_owners
            .0
            .push(numerical_admissibility.clone().unbind());
        let allocation_index = u64::try_from(allocation_handoff.0.len())
            .map_err(|_| invalid("model update allocation roster is too large"))?;
        let numerical_admissibility = SemanticTensorInput {
            tensor: crate::dlpack_from_py_for_stream_guarded(
                numerical_admissibility,
                i64::try_from(stream)
                    .map_err(|_| invalid("consumer stream exceeds DLPack address space"))?,
                &check,
            )?,
            layout: SemanticTensorLayout {
                role: 0,
                index: allocation_index,
                element_bytes: 1,
                scalar_type: 8,
                rank: 1,
                logical_axis: u64::MAX,
                dimensions: [1, 0, 0, 0],
                strides_bytes: [1, 0, 0, 0],
            },
            logical_begin: 0,
            logical_end: 0,
            native_allocation: None,
        };
        validate_producer_device_guarded(canary_results, device, &check)?;
        let canary_index = allocation_index
            .checked_add(1)
            .ok_or_else(|| invalid("model update canary index exceeds native address space"))?;
        let canary_results = SemanticTensorInput {
            tensor: crate::dlpack_from_py_for_stream_guarded(
                canary_results,
                i64::try_from(stream)
                    .map_err(|_| invalid("consumer stream exceeds DLPack address space"))?,
                &check,
            )?,
            layout: SemanticTensorLayout {
                role: 0,
                index: canary_index,
                element_bytes: 8,
                scalar_type: 3,
                rank: 2,
                logical_axis: u64::MAX,
                dimensions: [5, 9, 0, 0],
                strides_bytes: [72, 8, 0, 0],
            },
            logical_begin: 0,
            logical_end: 0,
            native_allocation: None,
        };
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        let state = self.continuation_state(py, task_use, parent, &owner, None)?;
        self.require_continuation_witness(py, parent, allocation_witness)?;
        if self.continuation_binding(&state, parent, false)? != expected {
            return Err(invalid(
                "prepared update authority changed during model output handoff",
            ));
        }
        owner
            .bind_prepared_update_output(
                &step.inner,
                bank,
                SemanticModelMemory {
                    allocations: allocation_handoff.into_native(),
                    storages,
                    views,
                },
                tensor_handoff.into_native(),
                numerical_admissibility,
                canary_results,
                &allocation_witness.inner,
                stream,
            )
            .map_err(xlog_err)?;
        drop(state);
        drop(owner);
        drop(session);
        step.continuation_producers
            .lock()
            .map_err(|_| invalid("prepared update producer ownership mutex is poisoned"))?[bank]
            .append(&mut producer_owners.0);
        Ok(())
    }

    /// Bind the original policy producers while this genuine step is recording.
    /// The combined transient witness covers text logits, product support,
    /// parameters and the model-issued component baselines in that exact order.
    /// This records no host publication or RNG identity and returns no invocation;
    /// only actual completed execution can expose the original result and its
    /// retained late-backward tape. ``bank`` identifies the recorded branch whose
    /// original numerical producers remain owned until native result selection.
    #[cfg(feature = "semantic-policy")]
    #[pyo3(signature = (task_use, *, step, bank, binding, model_output, text_logits, product_support, parameters, component_baselines, producer_witness, consumer_stream))]
    #[expect(
        clippy::too_many_arguments,
        reason = "parent binding retains complete publication and model ownership"
    )]
    fn bind_prepared_policy(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        step: &PySemanticPreparedStep,
        bank: usize,
        binding: &Bound<'_, PyAny>,
        model_output: Py<PyAny>,
        text_logits: Py<PyAny>,
        product_support: Py<PyAny>,
        parameters: Py<PyAny>,
        component_baselines: Py<PyAny>,
        producer_witness: &PySemanticTensorContentWitness,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.session.borrow(py).require_creator()?;
        if bank > 1 {
            return Err(invalid("prepared policy bank must be zero or one"));
        }
        let mut budget = 16 * 1024 * 1024;
        let stream = parse_witness_consumer_stream(consumer_stream, &mut budget)?;
        let fields = object_sequence(binding, &mut budget)?;
        if fields.len() != 2 {
            return Err(invalid(
                "policy binding requires generation and exact digest bytes",
            ));
        }
        let binding = xlog_cuda::SemanticCatalogueBinding {
            generation: ColdValue::read(&fields[0], &mut budget, 0)?.unsigned()?,
            digest: identity_bytes(&fields[1])?,
        };
        let parent = ContentStepRef::Prepared(step);
        let expected = {
            let session = self.session.borrow(py);
            let owner = session.owner()?;
            let state = self.continuation_state(py, task_use, parent, &owner, None)?;
            step.require_recording_stream(&owner, stream)?;
            self.require_continuation_witness(py, parent, producer_witness)?;
            if binding != owner.binding() {
                return Err(invalid("policy packing belongs to another native binding"));
            }
            if step
                .policy_inputs
                .lock()
                .map_err(|_| invalid("prepared policy producer owner is poisoned"))?[bank]
                .is_some()
            {
                return Err(invalid("prepared policy producers were already bound"));
            }
            self.continuation_binding(&state, parent, false)?
        };
        let inputs = [
            text_logits,
            product_support,
            parameters,
            component_baselines,
        ];
        let device = self.session.borrow(py).device_ordinal;
        let mut handoff = TensorHandoff(Vec::with_capacity(inputs.len()));
        let producer_stream = i64::try_from(stream)
            .map_err(|_| invalid("consumer stream exceeds the DLPack signed stream range"))?;
        let check = || {
            let session = self.session.borrow(py);
            let owner = session.owner()?;
            let state = self.continuation_state(py, task_use, parent, &owner, None)?;
            step.require_recording_stream(&owner, stream)?;
            self.require_continuation_witness(py, parent, producer_witness)?;
            if self.continuation_binding(&state, parent, false)? != expected {
                return Err(invalid(
                    "prepared policy authority changed during producer handoff",
                ));
            }
            Ok(())
        };
        for input in &inputs {
            validate_producer_device_guarded(input.bind(py), device, &check)?;
            handoff.0.push(crate::dlpack_from_py_for_stream_guarded(
                input.bind(py),
                producer_stream,
                &check,
            )?);
        }
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        let state = self.continuation_state(py, task_use, parent, &owner, None)?;
        self.require_continuation_witness(py, parent, producer_witness)?;
        if self.continuation_binding(&state, parent, false)? != expected {
            return Err(invalid(
                "prepared policy authority changed during producer handoff",
            ));
        }
        let [text, support, parameters, component_baselines]: [_; 4] = handoff
            .into_native()
            .try_into()
            .unwrap_or_else(|_| unreachable!("exact policy producer roster"));
        owner
            .bind_prepared_policy(
                &step.inner,
                bank,
                binding,
                text,
                support,
                parameters,
                component_baselines,
                &producer_witness.inner,
                stream,
            )
            .map_err(xlog_err)?;
        drop(state);
        drop(owner);
        drop(session);
        let mut retained = step
            .policy_inputs
            .lock()
            .map_err(|_| invalid("prepared policy producer owner is poisoned"))?;
        if retained[bank].is_some() {
            return Err(invalid("prepared policy producers were already bound"));
        }
        retained[bank] = Some(PreparedPolicyInputs {
            model_output,
            inputs,
            invocation_issued: false,
        });
        Ok(())
    }

    /// Snapshot one original full-MASK policy, execute all draws and the sole device
    /// publication, and retain its late tape. ``binding`` is the exact pair
    /// returned with the packing layout. Inputs are contiguous one-dimensional
    /// F32[32,V], Bool8[support_cells], FP32[parameter_cells] and
    /// FP32[1,136], respectively.
    /// ``model_output`` is the caller-authenticated original output, retained
    /// without invoking a getter; rows/selected bits come only from the already
    /// admitted continuation. Acquire of the result remains explicit.
    #[cfg(feature = "semantic-policy")]
    #[pyo3(signature = (task_use, *, parent, binding, model_output, text_logits, product_support, parameters, component_baselines))]
    #[expect(
        clippy::too_many_arguments,
        reason = "replay import carries independent material and lifecycle callbacks"
    )]
    fn execute_policy(
        &self,
        py: Python<'_>,
        task_use: Py<PySemanticTransitionTaskUse>,
        parent: Py<PySemanticPublishedParent>,
        binding: &Bound<'_, PyAny>,
        model_output: Py<PyAny>,
        text_logits: Py<PyAny>,
        product_support: Py<PyAny>,
        parameters: Py<PyAny>,
        component_baselines: Py<PyAny>,
    ) -> PyResult<PySemanticPolicyInvocation> {
        self.session.borrow(py).require_creator()?;
        self.require_issued(&task_use.borrow(py))?;
        self.execute_policy_in_execution(
            py,
            task_use,
            parent,
            binding,
            model_output,
            text_logits,
            product_support,
            parameters,
            component_baselines,
            None,
        )
    }

    /// Publish the already admitted recompute/drain continuation without any
    /// policy draw. A proposal cannot use this entry: native preflight requires
    /// the actual bound policy. Returns the observed next proposal counter;
    /// acquire the new parent with the separate native Acquire operation.
    #[pyo3(signature = (task_use, *, parent))]
    fn execute_continuation(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
    ) -> PyResult<u64> {
        self.session.borrow(py).require_creator()?;
        self.require_issued(task_use)?;
        self.execute_continuation_in_execution(py, task_use, parent, None)
    }

    /// Materialize the initial native parent before any model continuation.
    /// This cold operation does not consume ``before_segment`` and returns only
    /// the device-validated ``(instance, word, logical_digest, state_digest)``;
    /// a model must separately acquire the resulting parent lease.
    ///
    /// ``source`` contains exactly 32 original ring rows and ``prefix`` the
    /// existing committed source, each ``(token, logical_position, kind,
    /// provenance, valid, committed, recomputed, provenance_record)``. Flags are
    /// exact bools, kinds are PAD=0/FILLED=1/MASK=2 and provenance is
    /// NONE=0/SOURCE=1/GENERATED=2. This fresh initializer accepts only observed
    /// SOURCE for FILLED; GENERATED needs actual published history restoration.
    /// Initial provenance_record values are zero; the controller derives actual
    /// ledger indices from the task's source_mapping. Neither sequence is sorted
    /// or repacked, and supplied FILLED tokens must equal their mapped payload.
    ///
    /// ``metadata`` is an exact dict with every key required: recovered_instance
    /// (None for this fresh initializer; recovery must preserve actual history), ring_head, provenance_capacity_records, prefix_capacity,
    /// feedback_capacity, max_position, pad_token, terminal_tokens,
    /// final_intent_payload_bytes, intent_effect (actual application bytes),
    /// intent_entry_capacity, intent_payload_capacity_bytes, model_generation, policy_generation,
    /// model_numerical_mode (complete, restorable producer-owned cold metadata),
    /// model_contract_layout (S_begin, S_bytes, S_digest_offset, generation_offset,
    /// N_offset, identity_offset), six u64 byte coordinates in record 44,
    /// neural_generation, cache_generation, training_cursor, training_rng
    /// (four u64 words), fuel, rng (model u32, stream u64, family u8, proposal u32),
    /// topology_identity and table_identity (bytes32), role_counts (55 u64s),
    /// active_layouts, authority_decisions_capacity_bytes. Active layouts are
    /// reserved shapes, not fabricated forward
    /// outputs; their initial content is uncomputed and cannot be acquired as a
    /// valid tensor. Decision capacity must fit the complete current and planned
    /// future snapshots; no implicit growth occurs. Authority generation comes
    /// from this native task use. Provenance capacity is in records and must fit
    /// the mapped initial records plus 32 possible appended generated tokens.
    ///
    /// ``records`` rows are ``(role, index, capacity_bytes, actual_bytes)``;
    /// generated native records are not caller inputs. In particular, role 3
    /// (PrefixIdentity) is derived from the actual PrefixSource by fresh native
    /// initialization; caller bytes are rejected, even for an empty prefix.
    /// RuntimeContract (role 43) is also native-generated: it joins the actual
    /// cold contract, capacities and layouts with the complete model contribution.
    /// Its own capacity is computed by the native producer, not supplied here.
    /// Checkpoint/replay restoration retains and validates the original bytes.
    /// Complete dependency and
    /// authority records are inserted from this controller's retained import.
    /// ``tensors`` rows are ``(layout, logical_begin, logical_end, producer)``.
    /// A layout is ``(role, index, element_bytes, scalar_type, rank, logical_axis,
    /// dimensions[4], strides_bytes[4])``. Scalar codes are U8=1/U32=2/U64=3/
    /// F16=4/BF16=5/F32=6/I64=7/Bool8=8. The original DLPack dtype and layout must agree;
    /// no numerical conversion or detached replacement leaf is performed.
    /// ``model_allocations`` uses the same tensor-input rows with complete U8
    /// backing vectors, role zero and index equal to allocation ordinal.
    /// ``model_storages`` rows are ``(allocation, byte_offset, span_bytes)``;
    /// their ordinal is the producer storage identity. ``model_views`` rows are
    /// ``(role, index, storage, byte_offset)`` in role/index order. View offsets
    /// are relative to that storage, independently of tensor version identity.
    #[pyo3(signature = (task_use, *, metadata, source, prefix, records, tensors, model_allocations, model_storages, model_views))]
    #[expect(
        clippy::too_many_arguments,
        reason = "execution continuation retains each typed producer and callback guard"
    )]
    fn bind_parent(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        metadata: &Bound<'_, PyAny>,
        source: &Bound<'_, PyAny>,
        prefix: &Bound<'_, PyAny>,
        records: &Bound<'_, PyAny>,
        tensors: &Bound<'_, PyAny>,
        model_allocations: &Bound<'_, PyAny>,
        model_storages: &Bound<'_, PyAny>,
        model_views: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        self.session.borrow(py).require_creator()?;
        self.require_issued(task_use)?;
        {
            let session = self.session.borrow(py);
            task_use.require_current(&*session.owner()?)?;
            if !matches!(task_use.state()?.phase, TaskUsePhase::Imported) {
                return Err(invalid(
                    "initial parent must be bound before segment admission",
                ));
            }
        }
        let mut parent = parse_parent(metadata, source, prefix, records, task_use)?;
        let device = self.session.borrow(py).device_ordinal;
        let mut budget = 16 * 1024 * 1024;
        parent.model_memory.storages = parse_model_storages(model_storages, &mut budget)?;
        parent.model_memory.views = parse_model_views(model_views, &mut budget)?;
        let ParsedTensorInputs {
            handoff: tensors,
            producers: _producers,
        } = parse_tensor_inputs(tensors, &mut budget, device)?;
        let ParsedTensorInputs {
            handoff: allocations,
            producers: _allocation_producers,
        } = parse_tensor_inputs(model_allocations, &mut budget, device)?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        task_use.require_current(&owner)?;
        let state = task_use.state()?;
        if !matches!(state.phase, TaskUsePhase::Imported) {
            return Err(invalid("task phase changed during parent producer handoff"));
        }
        parent.tensors = tensors.into_native();
        parent.model_memory.allocations = allocations.into_native();
        let identity = owner.bind_parent(parent).map_err(xlog_err)?;
        Ok((
            PyBytes::new(py, identity.instance.as_bytes()),
            identity.word,
            PyBytes::new(py, identity.logical_digest.as_bytes()),
            PyBytes::new(py, identity.state_digest.as_bytes()),
        )
            .into_pyobject(py)?
            .unbind())
    }

    /// Revalidate after the segment, before any externally visible activation.
    /// Fresh independent rights and this actual acquired FINAL bank's native
    /// terminal intent are both required. This check does not deliver an effect.
    #[pyo3(signature = (task_use, *, parent, snapshot))]
    fn before_external_activation(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
        snapshot: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.session.borrow(py).require_creator()?;
        self.require_issued(task_use)?;
        parent.require_task(py, task_use)?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        task_use.require_current(&owner)?;
        let mut budget = 16 * 1024 * 1024;
        let snapshot = AuthoritySnapshot::parse(&ColdValue::read(snapshot, &mut budget, 0)?)?;
        task_use
            .state()?
            .revalidate_activation(&task_use.authority, snapshot)?;
        owner
            .admit_external_activation(&*parent.lease()?)
            .map_err(xlog_err)
    }
}

impl PySemanticTransitionController {
    fn bind_tensor_content(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        parent: ContentStepOwner,
        tensors: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
        binding_kind: TensorContentBinding,
    ) -> PyResult<PySemanticTensorContentWitness> {
        self.session.borrow(py).require_creator()?;
        self.require_read_issued(task_use)?;
        match &parent {
            ContentStepOwner::Published(acquired) => {
                acquired.borrow(py).require_task(py, task_use)?
            }
            ContentStepOwner::Prepared(step) => step.borrow(py).require_task(py, task_use)?,
        }
        let binding = {
            let session = self.session.borrow(py);
            let owner = session.owner()?;
            parent.binding(py, &owner, false)?
        };
        let mut budget = 16 * 1024 * 1024;
        let stream = parse_witness_consumer_stream(consumer_stream, &mut budget)?;
        let inputs = tensors.clone().unbind();
        let device = self.session.borrow(py).device_ordinal;
        let check = || -> PyResult<()> {
            let session = self.session.borrow(py);
            let owner = session.owner()?;
            if parent.binding(py, &owner, false)? != binding {
                return Err(invalid(
                    "original content scope changed during producer callback",
                ));
            }
            parent.check_producer_stream(py, &owner, stream)
        };
        check()?;
        let ParsedTensorInputs { handoff, producers } =
            parse_tensor_inputs_guarded(tensors, &mut budget, device, stream, &check)?;
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        task_use.require_identity(&owner)?;
        let state = task_use.state()?;
        let inner = match &parent {
            ContentStepOwner::Published(acquired) => {
                let acquired = acquired.borrow(py);
                let (operation, snapshot) = state.content_handoff_binding()?;
                if (Some(operation), snapshot) != binding {
                    return Err(invalid(
                        "tensor content authority changed during producer handoff",
                    ));
                }
                let lease = acquired.lease()?;
                owner.published_identity(&lease).map_err(xlog_err)?;
                match binding_kind {
                    TensorContentBinding::Capture => {
                        owner.capture_tensor_content(&lease, handoff.into_native(), stream)
                    }
                    TensorContentBinding::Model(None) => {
                        owner.bind_model_content(&lease, handoff.into_native(), stream)
                    }
                    TensorContentBinding::Model(Some(_)) => {
                        return Err(invalid(
                            "published model content does not accept a prepared bank",
                        ));
                    }
                }
            }
            ContentStepOwner::Prepared(step) => {
                let step = step.borrow(py);
                if state.prepared_content_binding(&step.scope)? != binding {
                    return Err(invalid(
                        "tensor content authority changed during producer handoff",
                    ));
                }
                match binding_kind {
                    TensorContentBinding::Model(Some(bank)) => owner.bind_prepared_model_content(
                        &step.inner,
                        bank,
                        handoff.into_native(),
                        stream,
                    ),
                    TensorContentBinding::Model(None) => {
                        return Err(invalid("prepared model content requires bank zero or one"));
                    }
                    TensorContentBinding::Capture => owner.capture_prepared_tensor_content(
                        &step.inner,
                        handoff.into_native(),
                        stream,
                    ),
                }
            }
        }
        .map_err(xlog_err)?;
        drop(state);
        drop(owner);
        Ok(PySemanticTensorContentWitness {
            inner,
            _inputs: inputs,
            _producers: producers,
            parent,
            session: self.session.clone_ref(py),
        })
    }

    fn require_issued(&self, task_use: &PySemanticTransitionTaskUse) -> PyResult<()> {
        self.require_read_issued(task_use)?;
        task_use.state()?.require_public_use()
    }

    fn require_read_issued(&self, task_use: &PySemanticTransitionTaskUse) -> PyResult<()> {
        task_use.issuance.require_current()?;
        if !Arc::ptr_eq(&self.identity, &task_use.controller) {
            return Err(invalid("task use belongs to another controller"));
        }
        Ok(())
    }

    fn check_recording(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        scope: &Arc<()>,
        snapshot: &[u8],
    ) -> PyResult<()> {
        self.require_read_issued(task_use)?;
        if self.session.as_ptr() != task_use.session.as_ptr() {
            return Err(invalid("recording belongs to another Session"));
        }
        let session = self.session.borrow(py);
        let owner = session.owner()?;
        task_use.require_identity(&owner)?;
        let state = task_use.state()?;
        if !session.recording.load(Ordering::Acquire)
            || !matches!(&state.phase, TaskUsePhase::Recording { scope: current, .. }
                if Arc::ptr_eq(scope, current))
            || state.snapshot.canonical != snapshot
        {
            return Err(invalid(
                "original recording scope or authority changed during callback",
            ));
        }
        Ok(())
    }
}

impl PySemanticTransitionController {
    // Pure phase projection; the caller derives `importing` only after
    // validating the private guard. This grants no native execution by itself.
    fn execution_binding(state: &TaskUseState, importing: bool) -> PyResult<(String, Vec<u8>)> {
        let operation = match (&state.phase, importing) {
            (TaskUsePhase::Segment(operation), false)
            | (
                TaskUsePhase::Importing {
                    operation,
                    reads: false,
                },
                true,
            ) => operation,
            _ => {
                return Err(invalid(
                    "controller execution requires its admitted phase with import reads closed",
                ))
            }
        };
        Ok((operation.clone(), state.snapshot.canonical.clone()))
    }

    fn execution_state<'a>(
        &self,
        py: Python<'_>,
        task_use: &'a PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
        owner: &SemanticTransitionSession,
        import: Option<&ColdImportGuard<'_>>,
    ) -> PyResult<MutexGuard<'a, TaskUseState>> {
        self.require_read_issued(task_use)?;
        parent.require_task(py, task_use)?;
        if self.session.as_ptr() != task_use.session.as_ptr()
            || self.session.as_ptr() != parent.session.as_ptr()
        {
            return Err(invalid(
                "controller execution owners belong to another native Session",
            ));
        }
        task_use.require_identity(owner)?;
        if let Some(guard) = import {
            let session = self.session.borrow(py);
            if !std::ptr::eq(&*session, guard.session)
                || guard.committed
                || !guard.issuance.same_as(&task_use.issuance)
                || !session.importing.load(Ordering::Acquire)
            {
                return Err(invalid(
                    "controller execution requires this Session's active uncommitted import guard",
                ));
            }
        }
        let state = task_use.state()?;
        Self::execution_binding(&state, import.is_some())?;
        Ok(state)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "replay import carries independent material and lifecycle callbacks"
    )]
    fn replay_import(
        &self,
        py: Python<'_>,
        task_use: &Py<PySemanticTransitionTaskUse>,
        ordinal: usize,
        material: NativeReplayBinding,
        kind: SemanticTransitionKind,
        restore_invocation: &Bound<'_, PyAny>,
        _pack_policy: &Bound<'_, PyAny>,
        finish_invocation: &Bound<'_, PyAny>,
        refresh_snapshot: &Bound<'_, PyAny>,
        import: &ColdImportGuard<'_>,
    ) -> PyResult<()> {
        let issued = task_use.borrow(py);
        let lease = {
            let mut owner = import.session.owner()?;
            issued.require_identity(&owner)?;
            let lease = owner.acquire().map_err(xlog_err)?;
            owner.admit_transition(&lease, kind).map_err(xlog_err)?;
            lease
        };
        let parent = Py::new(
            py,
            PySemanticPublishedParent {
                session: self.session.clone_ref(py),
                task_use: task_use.clone_ref(py),
                inner: Mutex::new(lease),
                continuation_producers: Mutex::new(Vec::new()),
            },
        )?;
        let acquired = parent.borrow(py);
        let expected = {
            let owner = import.session.owner()?;
            let state = self.execution_state(py, &issued, &acquired, &owner, Some(import))?;
            Self::execution_binding(&state, true)?
        };
        // All temporary original outputs and policy invocation owners leave this
        // block before the Runtime is asked to retire its remaining aliases.
        let replay_result = (|| -> PyResult<()> {
            let row = issued.authority.replay[ordinal].python_view(py)?;
            let transition = match kind {
                SemanticTransitionKind::Proposal => "proposal",
                SemanticTransitionKind::Recompute => "recompute",
                SemanticTransitionKind::Update => "update",
                SemanticTransitionKind::Drain => "drain",
            };
            let restored = {
                let _reads = ImportReadScope::enter(&issued)?;
                restore_invocation.call1((
                    task_use.clone_ref(py),
                    parent.clone_ref(py),
                    row,
                    transition,
                ))?
            };
            self.check_import_callback(py, &issued, &acquired, import, &expected)?;
            let mut budget = 16 * 1024 * 1024;
            let restored = object_sequence(&restored, &mut budget)?;
            if restored.len() != 2 {
                return Err(invalid(
                    "original invocation restore must return its output and continuation arguments",
                ));
            }
            // Retain the original output even without the policy feature.
            let _original_output = restored[0].clone().unbind();
            let arguments = object_sequence(&restored[1], &mut budget)?;
            if arguments.len() != 9 {
                return Err(invalid("original continuation requires six CUDA services, tensors, its captured producer witness and consumer stream"));
            }
            let producer_witness =
                arguments[7].extract::<PyRef<'_, PySemanticTensorContentWitness>>()?;
            self.bind_continuation_in_execution(
                py,
                &issued,
                ContentStepRef::Published(&acquired),
                &arguments[0],
                &arguments[1],
                &arguments[2],
                &arguments[3],
                &arguments[4],
                &arguments[5],
                &arguments[6],
                &producer_witness,
                &arguments[8],
                None,
                Some(import),
            )?;
            if kind == SemanticTransitionKind::Proposal {
                #[cfg(feature = "semantic-policy")]
                {
                    let policy = {
                        let _reads = ImportReadScope::enter(&issued)?;
                        _pack_policy.call1((
                            task_use.clone_ref(py),
                            parent.clone_ref(py),
                            _original_output.clone_ref(py),
                        ))?
                    };
                    self.check_import_callback(py, &issued, &acquired, import, &expected)?;
                    let policy = object_sequence(&policy, &mut budget)?;
                    if policy.len() != 5 {
                        return Err(invalid("Runtime replay policy requires binding, logits, parameters, actual product support and component baselines"));
                    }
                    let invocation = self.execute_policy_in_execution(
                        py,
                        task_use.clone_ref(py),
                        parent.clone_ref(py),
                        &policy[0],
                        _original_output,
                        policy[1].clone().unbind(),
                        policy[3].clone().unbind(),
                        policy[2].clone().unbind(),
                        policy[4].clone().unbind(),
                        Some(import),
                    )?;
                    // Keep the complete original invocation in existing failure
                    // custody until native final use succeeds. Import replay
                    // verifies history; it neither runs a backward nor opens
                    // the public policy final-use API during Importing.
                    let retained = TensorHandoff(vec![invocation]);
                    let invocation = &retained.0[0];
                    let comparison = invocation
                        .outcome
                        .as_published()
                        .map_err(xlog_err)
                        .and_then(|_| {
                            self.verify_import_successor(
                                py,
                                &issued,
                                &acquired,
                                &material.material,
                                import,
                            )
                        });
                    let completion = (|| -> PyResult<()> {
                        let mut owner = import.session.owner()?;
                        let state =
                            self.execution_state(py, &issued, &acquired, &owner, Some(import))?;
                        if Self::execution_binding(&state, true)? != expected {
                            return Err(invalid(
                                "policy import authority changed before final use",
                            ));
                        }
                        invocation.start_final_use()?;
                        // Policy handoff and predecessor/successor comparison
                        // use this controller-owned legacy stream. Native final
                        // use also checks the original continuation's stream.
                        owner
                            .finish_policy_invocation(&*acquired.lease()?, invocation.rng, 1)
                            .map_err(xlog_err)
                    })();
                    if completion.is_ok() {
                        drop(retained.into_native());
                    }
                    // Finalization is attempted once even for a refusal or a
                    // failed comparison. Uncertain completion retains invocation,
                    // tape and reader; reader release will not retry that use.
                    finish_with_cleanup(py, comparison, completion)?;
                }
                #[cfg(not(feature = "semantic-policy"))]
                return Err(invalid("proposal replay requires a semantic-policy build"));
            } else {
                self.execute_continuation_in_execution(py, &issued, &acquired, Some(import))?;
                self.verify_import_successor(py, &issued, &acquired, &material.material, import)?;
            }
            Ok(())
        })();
        // Cleanup runs exactly once even if restore, packing, execution or the
        // first comparison failed. Its sole permission is retiring the original
        // Runtime's temporary owners; an aborted native owner still rejects reads.
        let cleanup_result = (|| -> PyResult<Vec<u64>> {
            let streams = {
                let _reads = ImportReadScope::enter(&issued)?;
                finish_invocation.call1((task_use.clone_ref(py), parent.clone_ref(py)))?
            };
            let mut budget = 16 * 1024 * 1024;
            let mut streams = ColdValue::read(&streams, &mut budget, 0)?
                .sequence()?
                .iter()
                .map(ColdValue::unsigned)
                .collect::<PyResult<Vec<_>>>()?;
            // Complete-state verification uses this actual controller-owned
            // legacy stream in addition to the Runtime's exact consumer roster.
            if !streams.contains(&1) {
                streams.push(1);
            }
            Ok(streams)
        })();
        let streams = match (replay_result, cleanup_result) {
            (Err(error), cleanup) => {
                let release = cleanup
                    .and_then(|streams| Self::release_import_reader(&acquired, import, &streams));
                return finish_with_cleanup(py, Err(error), release);
            }
            (Ok(()), Err(error)) => return Err(error),
            (Ok(()), Ok(streams)) => streams,
        };
        let mut release_allowed = true;
        let final_result = (|| -> PyResult<AuthoritySnapshot> {
            self.check_import_callback(py, &issued, &acquired, import, &expected)?;
            // Current authority refresh is not a model callback and opens no
            // unfinished import reads. Historical role39 never supplies rights.
            let snapshot = refresh_snapshot.call0()?;
            let mut budget = 16 * 1024 * 1024;
            let snapshot = AuthoritySnapshot::parse(&ColdValue::read(&snapshot, &mut budget, 0)?)?;
            {
                let mut owner = import.session.owner()?;
                let state = self.execution_state(py, &issued, &acquired, &owner, Some(import))?;
                if Self::execution_binding(&state, true)? != expected {
                    return Err(invalid(
                        "current authority changed outside the importer's refresh",
                    ));
                }
                snapshot.newer_than(&state.snapshot)?;
                issued.authority.check_use(&expected.0, &snapshot, false)?;
                // Join all callback writes to P/S shared storage and retire
                // aliases, but retain P's reader for final full comparison.
                // An uncertain completion is quarantined, never retried here.
                release_allowed = false;
                owner
                    .quiesce_published_reader(&*acquired.lease()?, &streams)
                    .map_err(xlog_err)?;
                release_allowed = true;
            }
            self.verify_import_successor(py, &issued, &acquired, &material.material, import)?;
            Ok(snapshot)
        })();
        let release = if release_allowed {
            Self::release_import_reader(&acquired, import, &streams)
        } else {
            Ok(())
        };
        let snapshot = finish_with_cleanup(py, final_result, release)?;
        let owner = import.session.owner()?;
        let mut state = self.execution_state(py, &issued, &acquired, &owner, Some(import))?;
        state.finish_import(&issued.authority, snapshot)
    }

    fn release_import_reader(
        parent: &PySemanticPublishedParent,
        import: &ColdImportGuard<'_>,
        streams: &[u64],
    ) -> PyResult<()> {
        let mut owner = import.session.owner()?;
        if owner.is_poisoned() {
            return Err(invalid(
                "failed native completion retains the import reader in quarantine",
            ));
        }
        owner
            .release(&mut *parent.lease()?, streams)
            .map_err(xlog_err)?;
        drop(owner);
        parent.release_continuation_producers();
        Ok(())
    }

    fn check_import_callback(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
        import: &ColdImportGuard<'_>,
        expected: &(String, Vec<u8>),
    ) -> PyResult<()> {
        let owner = import.session.owner()?;
        let state = self.execution_state(py, task_use, parent, &owner, Some(import))?;
        if Self::execution_binding(&state, true)? != *expected {
            return Err(invalid(
                "import callback changed the task's original current authority",
            ));
        }
        owner
            .published_identity(&*parent.lease()?)
            .map_err(xlog_err)?;
        Ok(())
    }

    fn verify_import_successor(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        predecessor: &PySemanticPublishedParent,
        material: &xlog_cuda::SemanticReplayMaterial,
        import: &ColdImportGuard<'_>,
    ) -> PyResult<()> {
        let mut owner = import.session.owner()?;
        let _state = self.execution_state(py, task_use, predecessor, &owner, Some(import))?;
        let mut successor = owner.acquire().map_err(xlog_err)?;
        let comparison = predecessor.lease().and_then(|lease| {
            owner
                .verify_replay_publication(material, &lease, &successor)
                .map_err(xlog_err)
        });
        // This private successor was never exported to Python; native full
        // content verification registered the controller's legacy stream 1.
        // Retire its actual native reader even on comparison failure; preserve
        // both errors if cleanup also fails and let import abort own it.
        let release = owner.release(&mut successor, &[1]).map_err(xlog_err);
        // Normalizing an exception may execute Python; release native locks first.
        drop(_state);
        drop(owner);
        finish_with_cleanup(py, comparison, release)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "execution continuation retains each typed producer and callback guard"
    )]
    fn bind_continuation_in_execution(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        parent: ContentStepRef<'_>,
        text_rows: &Bound<'_, PyAny>,
        text_row_count: &Bound<'_, PyAny>,
        selected_text: &Bound<'_, PyAny>,
        active_rows: &Bound<'_, PyAny>,
        active_row_count: &Bound<'_, PyAny>,
        numerical_admissibility: &Bound<'_, PyAny>,
        tensors: &Bound<'_, PyAny>,
        producer_witness: &PySemanticTensorContentWitness,
        consumer_stream: &Bound<'_, PyAny>,
        prepared_bank: Option<usize>,
        import: Option<&ColdImportGuard<'_>>,
    ) -> PyResult<()> {
        match parent {
            ContentStepRef::Published(_) if prepared_bank.is_some() => {
                return Err(invalid(
                    "published continuation does not accept a prepared bank",
                ));
            }
            ContentStepRef::Prepared(_) if prepared_bank.is_none_or(|bank| bank > 1) => {
                return Err(invalid("prepared continuation requires bank zero or one"));
            }
            _ => {}
        }
        let expected = {
            let session = self.session.borrow(py);
            let owner = session.owner()?;
            let state = self.continuation_state(py, task_use, parent, &owner, import)?;
            self.require_continuation_witness(py, parent, producer_witness)?;
            self.continuation_binding(&state, parent, import.is_some())?
        };
        let mut budget = 16 * 1024 * 1024;
        let consumer_stream = parse_witness_consumer_stream(consumer_stream, &mut budget)?;
        if let ContentStepRef::Prepared(step) = parent {
            let session = self.session.borrow(py);
            let owner = session.owner()?;
            step.require_recording_stream(&owner, consumer_stream)?;
        }
        let check = || -> PyResult<()> {
            let session = self.session.borrow(py);
            let owner = session.owner()?;
            let state = self.continuation_state(py, task_use, parent, &owner, import)?;
            if self.continuation_binding(&state, parent, import.is_some())? != expected {
                return Err(invalid(
                    "original continuation scope changed during producer callback",
                ));
            }
            Ok(())
        };
        let services = [
            text_rows,
            text_row_count,
            selected_text,
            active_rows,
            active_row_count,
            numerical_admissibility,
        ];
        // Retain the original Python producers independently of mutable input
        // rows. Native code retains the captured witness without a Python cycle.
        // Any failed callback leaves these original autograd owners quarantined.
        let mut producer_owners = TensorHandoff(vec![producer_witness._inputs.clone_ref(py)]);
        producer_owners.0.extend(
            producer_witness
                ._producers
                .iter()
                .map(|value| value.clone_ref(py)),
        );
        producer_owners
            .0
            .extend(services.iter().map(|value| (*value).clone().unbind()));
        let device = self.session.borrow(py).device_ordinal;
        for producer in services {
            validate_continuation_producer_guarded(producer, device, &check)?;
        }
        let ParsedTensorInputs {
            handoff: tensors,
            producers,
        } = parse_tensor_inputs_guarded(tensors, &mut budget, device, consumer_stream, &check)?;
        producer_owners.0.extend(producers);
        let mut service_handoff = TensorHandoff(Vec::with_capacity(services.len()));
        for producer in services {
            service_handoff
                .0
                .push(crate::dlpack_from_py_for_stream_guarded(
                    producer,
                    i64::try_from(consumer_stream)
                        .map_err(|_| invalid("consumer stream exceeds DLPack address space"))?,
                    &check,
                )?);
        }
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        let state = self.continuation_state(py, task_use, parent, &owner, import)?;
        self.require_continuation_witness(py, parent, producer_witness)?;
        if self.continuation_binding(&state, parent, import.is_some())? != expected {
            return Err(invalid(
                "task phase changed during continuation producer handoff",
            ));
        }
        let authority_decisions = match import {
            Some(guard) => guard
                .original_decisions
                .as_ref()
                .ok_or_else(|| {
                    invalid("cold replay continuation lacks original native authority decisions")
                })?
                .as_ref()
                .to_vec(),
            None => state.snapshot.canonical.clone(),
        };
        let [text_rows, text_row_count, selected_text, active_rows,
            active_row_count, numerical_admissibility]: [_; 6] = service_handoff
                .into_native().try_into()
                .unwrap_or_else(|_| unreachable!("exact continuation service producer roster"));
        let input = SemanticContinuationInput {
            text_rows,
            text_row_count,
            selected_text,
            active_rows,
            active_row_count,
            numerical_admissibility,
            tensors: tensors.into_native(),
            authority_decisions,
        };
        let result = match parent {
            ContentStepRef::Published(parent) => owner.bind_continuation(
                &*parent.lease()?,
                input,
                &producer_witness.inner,
                consumer_stream,
            ),
            ContentStepRef::Prepared(step) => owner.bind_prepared_continuation(
                &step.inner,
                prepared_bank.expect("validated prepared bank"),
                input,
                &producer_witness.inner,
                consumer_stream,
            ),
        }
        .map_err(xlog_err);
        drop(state);
        drop(owner);
        drop(session);
        result?;
        match parent {
            ContentStepRef::Published(parent) => parent
                .continuation_producers
                .lock()
                .map_err(|_| {
                    PyRuntimeError::new_err("continuation producer ownership mutex is poisoned")
                })?
                .append(&mut producer_owners.0),
            ContentStepRef::Prepared(step) => step.continuation_producers.lock().map_err(|_| {
                PyRuntimeError::new_err("continuation producer ownership mutex is poisoned")
            })?[prepared_bank.expect("validated prepared bank")]
            .append(&mut producer_owners.0),
        }
        Ok(())
    }

    fn require_continuation_witness(
        &self,
        py: Python<'_>,
        parent: ContentStepRef<'_>,
        witness: &PySemanticTensorContentWitness,
    ) -> PyResult<()> {
        if witness.session.as_ptr() != self.session.as_ptr()
            || !match (parent, &witness.parent) {
                (ContentStepRef::Published(parent), ContentStepOwner::Published(issued)) => {
                    std::ptr::eq(&*issued.borrow(py), parent)
                }
                (ContentStepRef::Prepared(step), ContentStepOwner::Prepared(issued)) => {
                    std::ptr::eq(&*issued.borrow(py), step)
                }
                _ => false,
            }
        {
            return Err(invalid(
                "continuation producer witness belongs to another Session or published parent",
            ));
        }
        Ok(())
    }

    fn continuation_state<'a>(
        &self,
        py: Python<'_>,
        task_use: &'a PySemanticTransitionTaskUse,
        parent: ContentStepRef<'_>,
        owner: &SemanticTransitionSession,
        import: Option<&ColdImportGuard<'_>>,
    ) -> PyResult<MutexGuard<'a, TaskUseState>> {
        match parent {
            ContentStepRef::Published(parent) => {
                self.execution_state(py, task_use, parent, owner, import)
            }
            ContentStepRef::Prepared(step) => {
                if import.is_some() {
                    return Err(invalid("prepared recording cannot replace replay import"));
                }
                self.require_read_issued(task_use)?;
                step.require_task(py, task_use)?;
                task_use.require_identity(owner)?;
                owner.prepared_stream(&step.inner).map_err(xlog_err)?;
                let state = task_use.state()?;
                if !matches!(&state.phase, TaskUsePhase::Recording { scope, .. }
                    if Arc::ptr_eq(scope, &step.scope))
                {
                    return Err(invalid(
                        "continuation requires this step's original recording",
                    ));
                }
                Ok(state)
            }
        }
    }

    fn continuation_binding(
        &self,
        state: &TaskUseState,
        parent: ContentStepRef<'_>,
        importing: bool,
    ) -> PyResult<(Option<String>, Vec<u8>)> {
        match parent {
            ContentStepRef::Published(_) => Self::execution_binding(state, importing)
                .map(|(operation, snapshot)| (Some(operation), snapshot)),
            ContentStepRef::Prepared(step) => state.prepared_content_binding(&step.scope),
        }
    }

    #[cfg(feature = "semantic-policy")]
    fn issue_prepared_policy_invocation(
        &self,
        py: Python<'_>,
        task_use: Py<PySemanticTransitionTaskUse>,
        step: Py<PySemanticPreparedStep>,
        rng: SemanticRngBinding,
        outcome: xlog_cuda::SemanticTransitionOutcome,
        training: bool,
    ) -> PyResult<PySemanticPolicyInvocation> {
        self.session.borrow(py).require_creator()?;
        let original = step.borrow(py);
        let issued = task_use.borrow(py);
        let session = self.session.borrow(py);
        let owner = session.owner()?;
        self.require_issued(&issued)?;
        original.require_task(py, &issued)?;
        issued.require_current(&owner)?;
        let bank = owner
            .require_prepared_policy_invocation(&original.inner, rng)
            .map_err(xlog_err)?;
        let state = issued.state()?;
        state.require_public_use()?;
        let (operation, _) = state.prepared_content_binding(&original.scope)?;
        if operation.as_deref() != Some(if training { "training" } else { "inference" }) {
            return Err(invalid(
                "prepared policy result changed its admitted operation",
            ));
        }
        let mut retained = original
            .policy_inputs
            .lock()
            .map_err(|_| invalid("prepared policy producer owner is poisoned"))?;
        let inputs = retained[bank]
            .as_mut()
            .ok_or_else(|| invalid("prepared policy has no original numerical producers"))?;
        let (model_output, packed) = inputs.issue_originals(py)?;
        drop(retained);
        drop(state);
        drop(owner);
        drop(session);
        drop(issued);
        drop(original);
        Ok(PySemanticPolicyInvocation {
            session: self.session.clone_ref(py),
            _task_use: task_use,
            _parent: ContentStepOwner::Prepared(step),
            _model_output: model_output,
            _inputs: packed,
            rng,
            outcome,
            training,
            final_use_started: Mutex::new(false),
        })
    }

    #[cfg(feature = "semantic-policy")]
    #[expect(
        clippy::too_many_arguments,
        reason = "policy execution binds each typed numerical producer explicitly"
    )]
    fn execute_policy_in_execution(
        &self,
        py: Python<'_>,
        task_use: Py<PySemanticTransitionTaskUse>,
        parent: Py<PySemanticPublishedParent>,
        binding: &Bound<'_, PyAny>,
        model_output: Py<PyAny>,
        text_logits: Py<PyAny>,
        product_support: Py<PyAny>,
        parameters: Py<PyAny>,
        component_baselines: Py<PyAny>,
        import: Option<&ColdImportGuard<'_>>,
    ) -> PyResult<PySemanticPolicyInvocation> {
        let issued = task_use.borrow(py);
        let acquired = parent.borrow(py);
        let mut budget = 16 * 1024 * 1024;
        let fields = object_sequence(binding, &mut budget)?;
        if fields.len() != 2 {
            return Err(invalid(
                "policy binding requires generation and exact digest bytes",
            ));
        }
        let binding = xlog_cuda::SemanticCatalogueBinding {
            generation: ColdValue::read(&fields[0], &mut budget, 0)?.unsigned()?,
            digest: identity_bytes(&fields[1])?,
        };
        let (rng, expected, training) = {
            let session = self.session.borrow(py);
            let owner = session.owner()?;
            let state = self.execution_state(py, &issued, &acquired, &owner, import)?;
            if binding != owner.binding() {
                return Err(invalid("policy packing belongs to another native binding"));
            }
            let expected = Self::execution_binding(&state, import.is_some())?;
            let training = expected.0 == "training";
            (
                owner
                    .continuation_rng(&*acquired.lease()?)
                    .map_err(xlog_err)?,
                expected,
                training,
            )
        };
        let inputs = [
            text_logits,
            product_support,
            parameters,
            component_baselines,
        ];
        let device = self.session.borrow(py).device_ordinal;
        let check = || {
            let session = self.session.borrow(py);
            let owner = session.owner()?;
            let state = self.execution_state(py, &issued, &acquired, &owner, import)?;
            if Self::execution_binding(&state, import.is_some())? != expected
                || owner
                    .continuation_rng(&*acquired.lease()?)
                    .map_err(xlog_err)?
                    != rng
            {
                return Err(invalid("policy invocation changed during producer handoff"));
            }
            Ok(())
        };
        for input in &inputs {
            validate_producer_device_guarded(input.bind(py), device, &check)?;
        }
        let mut handoff = TensorHandoff(Vec::with_capacity(inputs.len()));
        for input in &inputs {
            handoff.0.push(crate::dlpack_from_py_for_stream_guarded(
                input.bind(py),
                1,
                &check,
            )?);
        }
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        let state = self.execution_state(py, &issued, &acquired, &owner, import)?;
        if Self::execution_binding(&state, import.is_some())? != expected
            || owner
                .continuation_rng(&*acquired.lease()?)
                .map_err(xlog_err)?
                != rng
        {
            return Err(invalid("policy invocation changed during producer handoff"));
        }
        let [text, support, parameters, component_baselines]: [_; 4] = handoff
            .into_native()
            .try_into()
            .unwrap_or_else(|_| unreachable!("exact policy producer roster"));
        owner
            .bind_policy_dlpack(binding, rng, text, support, parameters, component_baselines)
            .map_err(xlog_err)?;
        owner.capture().map_err(xlog_err)?;
        owner.launch().map_err(xlog_err)?;
        let outcome = owner.observe(rng.proposal).map_err(xlog_err)?;
        drop(state);
        drop(acquired);
        drop(issued);
        Ok(PySemanticPolicyInvocation {
            session: self.session.clone_ref(py),
            _task_use: task_use,
            _parent: ContentStepOwner::Published(parent),
            _model_output: model_output,
            _inputs: inputs,
            rng,
            outcome,
            training,
            final_use_started: Mutex::new(false),
        })
    }

    fn execute_continuation_in_execution(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
        import: Option<&ColdImportGuard<'_>>,
    ) -> PyResult<u64> {
        let session = self.session.borrow(py);
        let mut owner = session.owner()?;
        let _state = self.execution_state(py, task_use, parent, &owner, import)?;
        let rng = owner
            .continuation_rng(&*parent.lease()?)
            .map_err(xlog_err)?;
        owner.capture().map_err(xlog_err)?;
        owner.launch().map_err(xlog_err)?;
        Ok(owner
            .observe(rng.proposal)
            .map_err(xlog_err)?
            .into_published()
            .map_err(xlog_err)?
            .next_proposal)
    }
}

fn finish_with_cleanup<T>(
    py: Python<'_>,
    result: PyResult<T>,
    cleanup: PyResult<()>,
) -> PyResult<T> {
    match (result, cleanup) {
        (Err(error), Err(cleanup_error)) => {
            // Retain the original cause order, then append each distinct cleanup
            // exception. Identity deduplication also breaks pre-existing cycles
            // or overlap between a callback's cause and a later cleanup failure.
            let mut causes = Vec::new();
            let mut seen = BTreeSet::new();
            for first in [error.clone_ref(py), cleanup_error] {
                let mut next = Some(first);
                while let Some(cause) = next {
                    if !seen.insert(cause.value(py).as_ptr()) {
                        break;
                    }
                    next = cause.cause(py);
                    causes.push(cause);
                }
            }
            for (index, cause) in causes.iter().enumerate() {
                let next = causes.get(index + 1);
                let previous = cause.cause(py);
                if previous.as_ref().map(|value| value.value(py).as_ptr())
                    != next.map(|value| value.value(py).as_ptr())
                {
                    cause.set_cause(py, next.map(|value| value.clone_ref(py)));
                }
            }
            Err(error)
        }
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(value), Ok(())) => Ok(value),
    }
}

fn scalar_type(code: u8) -> PyResult<ScalarType> {
    ScalarType::from_code(code)
        .ok_or_else(|| PyValueError::new_err(format!("invalid XLOG scalar type code {code}")))
}

fn parse_argument(code: u8, value: &Bound<'_, PyAny>) -> PyResult<SemanticArgument> {
    let scalar = scalar_type(code)?;
    if scalar == ScalarType::Bool {
        if !value.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "semantic bool argument requires a Python bool",
            ));
        }
    } else if !value.is_instance_of::<PyInt>() || value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "semantic numeric bits and symbol IDs require a Python int, excluding bool",
        ));
    }
    Ok(match scalar {
        ScalarType::U32 => SemanticArgument::U32(value.extract()?),
        ScalarType::U64 => SemanticArgument::U64(value.extract()?),
        ScalarType::I32 => SemanticArgument::I32(value.extract()?),
        ScalarType::I64 => SemanticArgument::I64(value.extract()?),
        ScalarType::F32 => SemanticArgument::F32Bits(value.extract()?),
        ScalarType::F64 => SemanticArgument::F64Bits(value.extract()?),
        ScalarType::Bool => SemanticArgument::Bool(value.extract()?),
        ScalarType::Symbol => SemanticArgument::Symbol(value.extract()?),
    })
}

fn parse_admission(
    py: Python<'_>,
    predicates: Vec<PredicateInput>,
    records: Vec<RecordInput>,
    supports: Vec<SupportInput>,
) -> PyResult<SemanticAdmissionRecords> {
    let predicates = predicates
        .into_iter()
        .map(|(predicate, role, columns, key_columns)| {
            let role = match role.as_str() {
                "statement" => SemanticRecordRole::Statement,
                "qualifier" => SemanticRecordRole::Qualifier,
                "provenance" => SemanticRecordRole::Provenance,
                "source" => SemanticRecordRole::Source,
                "context" => SemanticRecordRole::Context,
                "scope" => SemanticRecordRole::Scope,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "invalid semantic record role {role:?}"
                    )))
                }
            };
            let mut labels = Vec::with_capacity(columns.len());
            let columns = columns
                .into_iter()
                .map(|(name, code, label)| {
                    labels.push(label);
                    Ok((name, scalar_type(code)?))
                })
                .collect::<PyResult<Vec<_>>>()?;
            let mut schema = Schema::new(columns)
                .with_sort_labels(labels)
                .map_err(val_err)?;
            schema.key_columns = key_columns;
            Ok(SemanticPredicateRecord {
                predicate: RelId(predicate),
                role,
                schema,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let records = records
        .into_iter()
        .map(|(predicate, arguments, qualifiers)| {
            Ok(SemanticTypedRecord {
                predicate: RelId(predicate),
                arguments: arguments
                    .into_iter()
                    .map(|(code, value)| parse_argument(code, value.bind(py)))
                    .collect::<PyResult<Vec<_>>>()?,
                qualifiers,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let supports = supports
        .into_iter()
        .map(
            |(statement, polarity, provenance, source, context, scope)| {
                Ok(SemanticSupportRecord {
                    statement,
                    polarity: match polarity.as_str() {
                        "pro" => SemanticPolarity::Pro,
                        "contra" => SemanticPolarity::Contra,
                        _ => {
                            return Err(PyValueError::new_err(format!(
                                "invalid semantic support polarity {polarity:?}"
                            )))
                        }
                    },
                    provenance,
                    source,
                    context,
                    scope,
                })
            },
        )
        .collect::<PyResult<Vec<_>>>()?;
    Ok(SemanticAdmissionRecords {
        predicates,
        records,
        supports,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        object_sequence, parse_admission, parse_text_slots, read_replay_rows, select_replay_row,
        AuthoritySnapshot, ColdValue, FeedbackSchema, PredicateInput, RecordInput, ReplayBasis,
        ReplayRow, SupportInput, TaskAuthority, TaskUsePhase, TaskUseState,
    };
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyTuple};
    use std::ffi::CString;
    use xlog_cuda::{SemanticArgument, SemanticPolarity, SemanticRecordRole};

    #[test]
    fn task_program_transport_preserves_explicit_configuration_without_cuda() {
        Python::initialize();
        Python::attach(|py| {
            let arguments = py.eval(c"((9,4,2),(8,3),'pred rainfall(u32). rainfall(8). ?- rainfall(8). ?- rainfall(2). ?- rainfall(9).',(1,0,2),(5,4,3,2,1,0),(15,7,3))", None, None).unwrap();
            let arguments = arguments.cast::<PyTuple>().unwrap();
            let fields = arguments.iter().collect::<Vec<_>>();
            let spec = super::read_task_evaluation_spec(
                &fields[0],
                &fields[1],
                &fields[2],
                &fields[3],
                &fields[4],
                &fields[5],
                &mut (16 * 1024 * 1024),
            )
            .unwrap();
            assert_eq!(spec.statement_records, [9, 4, 2]);
            assert_eq!(spec.allowed_support_records, [8, 3]);
            assert_eq!(spec.admissible_truth_masks, [15, 7, 3]);
            assert_eq!(spec.scoring.correct_weight, 5);
            assert_eq!(spec.scoring.spent_weight, 0);
        });
    }

    #[test]
    fn task_program_transport_rejects_answer_arrays_and_invalid_query_selection() {
        Python::initialize();
        Python::attach(|py| {
            for expression in [
                "((0,1),(),(True,False),(0,1),(1,1,1,1,1,1),(7,7))",
                "((0,1),(),'pred rainfall(u32). rainfall(8). ?- rainfall(X). ?- rainfall(8).',(0,1),(1,1,1,1,1,1),(7,7))",
                "((0,1),(),'pred rainfall(u32). rainfall(8). ?- rainfall(8). ?- rainfall(2).',(0,9),(1,1,1,1,1,1),(7,7))",
                "((0,1),(),'pred rainfall(u32). rainfall(8). ?- rainfall(8). ?- rainfall(2).',(0,1),(True,1,1,1,1,1),(7,7))",
            ] {
                let value = py.eval(&CString::new(expression).unwrap(), None, None).unwrap();
                let fields = value.cast::<PyTuple>().unwrap().iter().collect::<Vec<_>>();
                assert!(super::read_task_evaluation_spec(&fields[0], &fields[1], &fields[2],
                    &fields[3], &fields[4], &fields[5], &mut (16 * 1024 * 1024)).is_err());
            }
        });
    }

    #[test]
    fn prepared_step_cannot_be_constructed_from_python() {
        Python::initialize();
        Python::attach(|py| {
            let class = <super::PySemanticPreparedStep as pyo3::PyTypeInfo>::type_object(py);
            assert!(class.call0().is_err());
        });
    }

    #[test]
    fn prepared_schedule_accepts_only_frozen_ordinary_modes() {
        use super::SemanticTransitionKind;
        Python::initialize();
        Python::attach(|py| {
            let modes = PyTuple::new(py, ["proposal", "recompute", "proposal"]).unwrap();
            assert_eq!(
                super::prepared_transitions(modes.as_any())
                    .unwrap()
                    .collect::<Vec<_>>(),
                [
                    SemanticTransitionKind::Proposal,
                    SemanticTransitionKind::Recompute,
                    SemanticTransitionKind::Proposal
                ]
            );
            for expression in [
                "()",
                "('drain',)",
                "('proposal', True)",
                "('update',)",
                "['proposal']",
                "3",
                "type('Schedule', (tuple,), {})(('proposal',))",
            ] {
                let expression = CString::new(expression).unwrap();
                let value = py.eval(&expression, None, None).unwrap();
                assert!(super::prepared_transitions(&value).is_err());
            }
        });
    }

    #[test]
    fn cold_segment_build_does_not_admit_execution_or_allow_scope_substitution() {
        let (inputs, snapshot) = task_inputs("");
        let authority = TaskAuthority::parse(&inputs).unwrap();
        let mut state = TaskUseState {
            phase: TaskUsePhase::Imported,
            snapshot: snapshot.clone(),
        };
        let scope = state.begin_build().unwrap();
        let foreign = std::sync::Arc::new(());
        assert!(state.require_public_use().is_err());
        assert!(state.content_handoff_binding().is_err());
        assert!(state.begin_build().is_err());
        assert!(state.prepared_content_binding(&scope).is_ok());
        assert!(state.prepared_content_binding(&foreign).is_err());
        assert!(state.finish_build(&foreign).is_err());
        assert!(state
            .admit_segment(&authority, "inference".into(), snapshot.clone())
            .is_err());
        state.finish_build(&scope).unwrap();
        assert!(state.finish_build(&scope).is_err());
        assert!(state.require_public_use().is_err());
        assert!(state
            .admit_built_segment(&foreign, &authority, "inference".into(), snapshot.clone())
            .is_err());
        // Even the authentic completed recording cannot reuse the import snapshot.
        assert!(state
            .admit_built_segment(&scope, &authority, "inference".into(), snapshot)
            .is_err());
        assert!(matches!(state.phase, TaskUsePhase::Refused));
        assert!(state.prepared_content_binding(&scope).is_err());
    }

    #[test]
    fn recording_callback_cannot_hide_refusal_or_authority_change() {
        let (_, snapshot) = task_inputs("");
        for refuse in [false, true] {
            let state = std::sync::Mutex::new(TaskUseState {
                phase: TaskUsePhase::Imported,
                snapshot: snapshot.clone(),
            });
            let scope = state.lock().unwrap().begin_build().unwrap();
            let expected = state
                .lock()
                .unwrap()
                .prepared_content_binding(&scope)
                .unwrap();
            let result = super::recording_callback(
                || {
                    let binding = state.lock().unwrap().prepared_content_binding(&scope)?;
                    if binding != expected {
                        return Err(super::invalid("recording changed"));
                    }
                    Ok(())
                },
                || {
                    // The real callback boundary permits reentry: no state lock
                    // may span the callback, and its result cannot hide mutation.
                    let mut state = state.try_lock().expect("callback ran under state lock");
                    if refuse {
                        state.phase = TaskUsePhase::Refused;
                    } else {
                        state.snapshot.canonical.push(0);
                    }
                    Ok(())
                },
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn original_recording_memory_scope_exits_after_post_callback_refusal() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(c"class Scope:\n    def __enter__(self): events.append('enter')\n    def __exit__(self, *args): events.append('exit')\nevents = []\nscope = Scope()", Some(&locals), None).unwrap();
            let original = locals.get_item("scope").unwrap().unwrap();
            let mut scope = super::PreparedMemoryScope::new(py, &original, &|| Ok(())).unwrap();
            let checks = std::cell::Cell::new(0);
            let result = super::recording_callback(
                || {
                    checks.set(checks.get() + 1);
                    if checks.get() > 1 {
                        Err(super::invalid("authority refused during pool entry"))
                    } else {
                        Ok(())
                    }
                },
                || {
                    original.call_method0("__enter__")?;
                    scope.entered = true;
                    Ok(())
                },
            );
            assert!(result.is_err());
            drop(scope);
            assert_eq!(
                locals
                    .get_item("events")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                ["enter", "exit"]
            );
        });
    }

    #[test]
    fn original_recording_memory_scope_retains_exit_before_entry() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(c"class Scope:\n    def __enter__(self):\n        events.append('enter')\n        Scope.__exit__ = lambda *args: events.append('replacement')\n    def __exit__(self, *args): events.append('original')\nevents = []\nscope = Scope()", Some(&locals), None).unwrap();
            let original = locals.get_item("scope").unwrap().unwrap();
            let mut scope = super::PreparedMemoryScope::new(py, &original, &|| Ok(())).unwrap();
            original.call_method0("__enter__").unwrap();
            scope.entered = true;
            scope.finish(None).unwrap();
            assert_eq!(
                locals
                    .get_item("events")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                ["enter", "original"]
            );
        });
    }

    #[cfg(feature = "semantic-policy")]
    #[test]
    fn segment_completion_preserves_native_refusal_without_a_policy_invocation() {
        use xlog_cuda::{SemanticTransitionOutcome, SemanticTransitionRefusal};
        let refused =
            SemanticTransitionOutcome::Refused(SemanticTransitionRefusal::NonFinitePolicyInput {
                completed_draws: 0,
            });
        assert_eq!(
            super::transition_refusal(&refused),
            Some(("non_finite_policy_input", 0))
        );
        let refused =
            SemanticTransitionOutcome::Refused(SemanticTransitionRefusal::InvalidFinalSupport {
                completed_draws: 17,
            });
        assert_eq!(
            super::transition_refusal(&refused),
            Some(("invalid_final_support", 17))
        );
    }

    #[test]
    fn captured_original_producers_retire_only_after_last_owner_and_creator_drain() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(c"import weakref\nclass Original:\n    def __del__(self): events.append('released')\nevents = []\npayload = Original()\nreference = weakref.ref(payload)", Some(&locals), None).unwrap();
            let payload = locals.get_item("payload").unwrap().unwrap().unbind();
            locals.del_item("payload").unwrap();
            let resources = std::sync::Arc::new(super::PreparedProducerResources {
                producers: std::sync::Mutex::new(vec![payload]),
                owner_thread: std::thread::current().id(),
            });
            let recorded_owner = std::sync::Arc::clone(&resources);
            drop(resources);
            let reference = locals.get_item("reference").unwrap().unwrap();
            assert!(!reference.call0().unwrap().is_none());
            drop(recorded_owner);
            assert!(!reference.call0().unwrap().is_none());
            super::drain_export_owners();
            assert!(reference.call0().unwrap().is_none());
            assert_eq!(
                locals
                    .get_item("events")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                ["released"]
            );
        });
    }

    #[test]
    fn prepared_handoff_negotiates_the_actual_native_stream_before_capsule_validation() {
        Python::initialize();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            py.run(c"class Producer:\n def __dlpack_device__(self): return (2,0)\n def __dlpack__(self, *, stream):\n  self.seen_stream = stream\n  return None\nproducer = Producer()", Some(&globals), None).unwrap();
            let producer = globals.get_item("producer").unwrap().unwrap();
            let error = crate::dlpack_from_py_for_stream(&producer, 73)
                .err()
                .unwrap();
            assert!(error.to_string().contains("Invalid DLPack capsule"));
            assert_eq!(
                producer
                    .getattr("seen_stream")
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                73
            );
        });
    }

    #[test]
    fn prepared_device_reader_accepts_enum_payload_without_widening_cold_metadata() {
        Python::initialize();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            py.run(
                cr#"
from enum import IntEnum
class DeviceKind(IntEnum):
    CUDA = 2
    CPU = 1
    def __int__(self):
        raise AssertionError('device kind conversion invoked')
    def __index__(self):
        raise AssertionError('device kind index invoked')
calls = []
class Producer:
    def __dlpack_device__(self):
        calls.append('device')
        return device
    def __dlpack__(self, *, stream):
        raise AssertionError('capsule consumed by device preflight')
producer = Producer()
device = (DeviceKind.CUDA, 0)
"#,
                Some(&globals),
                None,
            )
            .unwrap();
            let producer = globals.get_item("producer").unwrap().unwrap();
            let device = globals.get_item("device").unwrap().unwrap();
            assert!(ColdValue::read(&device, &mut 128, 0).is_err());
            super::validate_producer_device_guarded(&producer, 0, &|| Ok(())).unwrap();
            super::validate_continuation_producer(&producer, 0).unwrap();
            assert!(super::validate_producer_device_guarded(&producer, 1, &|| Ok(())).is_err());
            assert_eq!(
                globals
                    .get_item("calls")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                ["device", "device", "device"]
            );
            for expression in [
                "(DeviceKind.CPU, 0)",
                "(DeviceKind.CUDA, -1)",
                "(DeviceKind.CUDA, False)",
                "(DeviceKind.CUDA, 1 << 31)",
                "[DeviceKind.CUDA, 0]",
                "(True, 0)",
            ] {
                let device = py
                    .eval(&CString::new(expression).unwrap(), Some(&globals), None)
                    .unwrap();
                globals.set_item("device", device).unwrap();
                assert!(
                    super::validate_producer_device_guarded(&producer, 0, &|| Ok(())).is_err(),
                    "{expression}"
                );
            }
        });
    }

    #[test]
    fn prepared_device_enum_does_not_bypass_post_callback_refusal() {
        Python::initialize();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            py.run(
                cr#"
from enum import IntEnum
class DeviceKind(IntEnum):
    CUDA = 2
refused = False
calls = []
class Producer:
    def __getattribute__(self, name):
        if refused:
            calls.append('late_lookup')
        return object.__getattribute__(self, name)
    def __dlpack_device__(self):
        global refused
        refused = True
        calls.append('device')
        return (DeviceKind.CUDA, 0)
    def __dlpack__(self, *, stream):
        calls.append('export')
        return None
producer = Producer()
"#,
                Some(&globals),
                None,
            )
            .unwrap();
            let producer = globals.get_item("producer").unwrap().unwrap();
            let check = || {
                if globals.get_item("refused")?.unwrap().extract::<bool>()? {
                    Err(super::invalid("original producer authority was refused"))
                } else {
                    Ok(())
                }
            };
            let error = super::validate_producer_device_guarded(&producer, 0, &check).unwrap_err();
            assert!(error
                .to_string()
                .contains("original producer authority was refused"));
            assert_eq!(
                globals
                    .get_item("calls")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                ["device"]
            );
        });
    }

    #[test]
    fn continuation_services_require_real_cuda_producers_before_capsule_consumption() {
        Python::initialize();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            py.run(c"class ForeignProducer:\n    def __init__(self, device):\n        self.device = device\n    def __dlpack_device__(self):\n        return self.device\n    def __dlpack__(self, **kwargs):\n        raise AssertionError('capsule consumed before CUDA producer validation')\nclass MissingDevice:\n    def __dlpack__(self, **kwargs):\n        raise AssertionError('capsule consumed without device declaration')", Some(&globals), None).unwrap();
            for expression in [
                "None",
                "()",
                "1",
                "MissingDevice()",
                "ForeignProducer((1, 0))",
                "ForeignProducer((2, 1))",
            ] {
                let value = py
                    .eval(&CString::new(expression).unwrap(), Some(&globals), None)
                    .unwrap();
                assert!(
                    super::validate_continuation_producer(&value, 0).is_err(),
                    "{expression}"
                );
            }
        });
    }

    #[test]
    fn tensor_layout_transport_preserves_all_scalar_codes_and_coordinates() {
        Python::initialize();
        Python::attach(|py| {
            for (scalar, width) in [
                (1, 1),
                (2, 4),
                (3, 8),
                (4, 2),
                (5, 2),
                (6, 4),
                (7, 8),
                (8, 1),
            ] {
                let expression = format!(
                    "(4, 11, {width}, {scalar}, 3, 1, (2, 3, 5, 0), ({}, {}, {width}, 0))",
                    15 * width,
                    5 * width
                );
                let value = py
                    .eval(&CString::new(expression).unwrap(), None, None)
                    .unwrap();
                let cold = ColdValue::read(&value, &mut 4096, 0).unwrap();
                let layout = super::parse_tensor_layout(&cold).unwrap();
                assert_eq!(
                    (
                        layout.role,
                        layout.index,
                        layout.element_bytes,
                        layout.scalar_type,
                        layout.rank,
                        layout.logical_axis
                    ),
                    (4, 11, width, scalar, 3, 1)
                );
                assert_eq!(layout.dimensions, [2, 3, 5, 0]);
                assert_eq!(layout.strides_bytes, [15 * width, 5 * width, width, 0]);
            }
            for expression in [
                "(4, 0, 8, True, 1, 0, (2, 0, 0, 0), (8, 0, 0, 0))",
                "(4, -1, 8, 7, 1, 0, (2, 0, 0, 0), (8, 0, 0, 0))",
                "(4, 0, 8, 7, 1, 0, (2, 0, 0), (8, 0, 0, 0))",
            ] {
                let value = py
                    .eval(&CString::new(expression).unwrap(), None, None)
                    .unwrap();
                assert!(
                    ColdValue::read(&value, &mut 4096, 0)
                        .and_then(|cold| super::parse_tensor_layout(&cold))
                        .is_err(),
                    "{expression}"
                );
            }
        });
    }

    #[test]
    fn malformed_tensor_rows_are_rejected_before_any_producer_callback() {
        Python::initialize();
        Python::attach(|py| {
            let globals = pyo3::types::PyDict::new(py);
            py.run(c"class Producer:\n    def __dlpack_device__(self):\n        raise AssertionError('producer callback ran before cold validation')\n    def __dlpack__(self, **kwargs):\n        raise AssertionError('producer callback ran before cold validation')\np = Producer()\nlayout = (4, 0, 8, 7, 1, 0, (2, 0, 0, 0), (8, 0, 0, 0))", Some(&globals), None).unwrap();
            for expression in [
                "((layout, 0, 2, p, None), (layout, 3, 2, p, None))",
                "((layout, 0, 2, p, None), (layout, 0, 2, p))",
                "((layout, 0, 2, p, None), (layout, True, 2, p, None))",
            ] {
                let value = py
                    .eval(&CString::new(expression).unwrap(), Some(&globals), None)
                    .unwrap();
                let error = match super::parse_tensor_inputs(&value, &mut 4096, 0) {
                    Ok(_) => panic!("accepted malformed tensor input: {expression}"),
                    Err(error) => error,
                };
                assert!(
                    error.is_instance_of::<pyo3::exceptions::PyValueError>(py),
                    "{error}"
                );
            }
        });
    }

    #[test]
    fn content_guard_ranges_preserve_exact_roles_and_reject_coercion() {
        Python::initialize();
        Python::attach(|py| {
            let value = py
                .eval(c"((1, 0), (4, 18446744073709551615))", None, None)
                .unwrap();
            let cold = ColdValue::read(&value, &mut 4096, 0).unwrap();
            let ranges = super::parse_content_ranges(&cold).unwrap();
            assert_eq!(
                ranges,
                vec![
                    (xlog_cuda::SemanticStateRole::from_code(1).unwrap(), 0),
                    (
                        xlog_cuda::SemanticStateRole::from_code(4).unwrap(),
                        u64::MAX
                    ),
                ]
            );
            for expression in [
                "()",
                "((1, 0), (1, 0))",
                "((0, 0),)",
                "((999, 0),)",
                "((True, 0),)",
                "((1, False),)",
                "((1, -1),)",
                "((1, 18446744073709551616),)",
                "((1, 0, 0),)",
            ] {
                let value = py
                    .eval(&CString::new(expression).unwrap(), None, None)
                    .unwrap();
                assert!(
                    ColdValue::read(&value, &mut 4096, 0)
                        .and_then(|cold| super::parse_content_ranges(&cold))
                        .is_err(),
                    "{expression}"
                );
            }
        });
    }

    fn run_python(script: &str) {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_native").unwrap();
            crate::pyxlog(py, &module).unwrap();
            py.run(&CString::new(script).unwrap(), Some(&module.dict()), None)
                .unwrap();
        });
    }

    #[test]
    fn initial_source_ledger_cannot_be_supplied_as_private_record_bytes() {
        Python::initialize();
        Python::attach(|py| {
            let records = py
                .eval(c"[(2, 0, 4096, b'caller-authored-origin')]", None, None)
                .unwrap();
            assert!(super::parse_state_records(&records, &mut 4096).is_err());
        });
    }

    #[test]
    fn initial_prefix_identity_cannot_be_supplied_as_caller_bytes() {
        Python::initialize();
        Python::attach(|py| {
            for expression in [c"[(3, 0, 32, bytes(32))]", c"[(3, 0, 64, b'x' * 32)]"] {
                let records = py.eval(expression, None, None).unwrap();
                assert!(
                    super::parse_state_records(&records, &mut 4096).is_err(),
                    "fresh prefix identity belongs to the native prefix source owner"
                );
            }
            let records = py.eval(c"[]", None, None).unwrap();
            assert!(super::parse_state_records(&records, &mut 4096)
                .unwrap()
                .is_empty());
        });
    }

    #[test]
    fn runtime_contract_cannot_be_supplied_as_caller_record() {
        Python::initialize();
        Python::attach(|py| {
            let records = py
                .eval(c"[(43, 0, 4096, b'caller-runtime-contract')]", None, None)
                .unwrap();
            assert!(
                super::parse_state_records(&records, &mut 4096).is_err(),
                "the complete runtime contract belongs to the native publication owner"
            );
        });
    }

    #[test]
    fn dlpack_storage_export_keeps_the_original_nonleaf_graph() {
        // PyTorch retains Python objects in native thread-local storage. Run
        // this embedding check in its own interpreter process: Rust's test
        // workers otherwise tear down CPython thread states before Torch's
        // TLS destructors, including when --test-threads=1 is selected.
        const CHILD: &str = "XLOG_DLPACK_GRAPH_TEST_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "semantic_transition::tests::dlpack_storage_export_keeps_the_original_nonleaf_graph",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("CUDA_VISIBLE_DEVICES", "")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated DLPack graph check failed: {}\n{}\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        Python::initialize();
        Python::attach(|py| {
            let globals = pyo3::types::PyDict::new(py);
            py.run(c"import torch\nx = torch.arange(3, dtype=torch.float32, requires_grad=True)\ny = x.square()\noriginal_fn = y.grad_fn", Some(&globals), None).unwrap();
            let original = globals.get_item("y").unwrap().unwrap();
            // CPU exercises the production exporter only. The public importer
            // still requires CUDA, whose stream/publication path is separate.
            let capsule = crate::dlpack_export_for_stream(&original, None).unwrap();
            globals.set_item("capsule", capsule).unwrap();
            py.run(c"alias = torch.utils.dlpack.from_dlpack(capsule)\nassert alias.data_ptr() == y.data_ptr()\nassert alias.dtype == y.dtype and alias.shape == y.shape and alias.stride() == y.stride()\nassert y.requires_grad and not y.is_leaf and y.grad_fn is original_fn\ny.backward(torch.tensor([1., 2., 3.]))\nassert torch.equal(x.grad, torch.tensor([0., 4., 12.]))", Some(&globals), None).unwrap();
            py.run(
                c"x.grad = None\nempty = x.square()[:0]\nempty_fn = empty.grad_fn",
                Some(&globals),
                None,
            )
            .unwrap();
            let empty = globals.get_item("empty").unwrap().unwrap();
            globals
                .set_item(
                    "empty_capsule",
                    crate::dlpack_export_for_stream(&empty, None).unwrap(),
                )
                .unwrap();
            py.run(c"empty_alias = torch.utils.dlpack.from_dlpack(empty_capsule)\nassert empty_alias.shape == (0,) and empty_alias.numel() == 0\nassert empty.requires_grad and empty.grad_fn is empty_fn\nempty.backward(torch.empty(0))\nassert torch.equal(x.grad, torch.zeros(3))", Some(&globals), None).unwrap();
            py.run(
                c"x.grad = None\nscalar = x.square().sum()\nscalar_fn = scalar.grad_fn",
                Some(&globals),
                None,
            )
            .unwrap();
            let scalar = globals.get_item("scalar").unwrap().unwrap();
            globals
                .set_item(
                    "scalar_capsule",
                    crate::dlpack_export_for_stream(&scalar, None).unwrap(),
                )
                .unwrap();
            py.run(c"scalar_alias = torch.utils.dlpack.from_dlpack(scalar_capsule)\nassert scalar_alias.shape == () and scalar_alias.stride() == ()\nassert scalar_alias.numel() == 1 and scalar_alias.data_ptr() == scalar.data_ptr()\nassert scalar_alias.dtype == scalar.dtype\nassert scalar.requires_grad and not scalar.is_leaf and scalar.grad_fn is scalar_fn\nscalar.backward()\nassert torch.equal(x.grad, torch.tensor([0., 2., 4.]))", Some(&globals), None).unwrap();
        });
    }

    const TASK_INPUT: &str = r#"
import hashlib, json, struct
def replay_fixture(split='request-local-replay'):
    digest = lambda value: hashlib.sha256(value).hexdigest()
    hid = lambda text: digest(text.encode())
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'))
    view = [hid('manifest'), 'sample', split, hid('tokenizer'), 'revision', hid('mask-policy'), 2**100, 0, 0]
    names = ('data_manifest_hash', 'example_id', 'final_split', 'tokenizer_hash', 'tokenizer_revision',
             'mask_policy_hash', 'training_seed', 'epoch_index', 'variant_index')
    identity_digest = digest(canonical(['dlm-new/training-view-identity/v1', view]).encode())
    invocation_kinds = dict(entry_boundary_identity='entry-boundary', source_tokens_identity='source-tokens',
        topology_identity='topology', cache_identity='cache', feedback_identity='feedback',
        model_generation='neural-generation', parameters_identity='parameters',
        physical_partition_identity='physical-partition', numerical_realization_identity='numerical-realization',
        random_state_identity='random-state', logits_identity='logits')
    invocation = {name:hid(name) for name in invocation_kinds}
    materials = []
    def add(identity, kind, reconstruction=None):
        payload = None if reconstruction else b'carrier:' + identity.encode()
        reference = dict(identity=identity, kind=kind,
                         bytes_sha256=None if payload is None else digest(payload), reconstruction=reconstruction)
        materials.append((reference, payload))
    add(hid('base'), 'logical-state'); add(hid('successor'), 'logical-state')
    for name, kind in invocation_kinds.items():
        reconstruction = dict(operation='forward-fp32-logits/v1', inputs=[invocation['parameters_identity'],
            invocation['entry_boundary_identity'], invocation['numerical_realization_identity']]) if name == 'logits_identity' else None
        add(invocation[name], kind, reconstruction)
    add(hid('provenance'), 'provenance')
    training = (b'dlm-new/training-view/v1'.ljust(32, b'\0') + bytes.fromhex(identity_digest)
        + bytes.fromhex(invocation['source_tokens_identity']) + struct.pack('<5Q', 8, 4, 2, 0, 4)
        + struct.pack('<8q', *range(8)) + struct.pack('<8q', -100, -100, -100, -100, -100, 3, -100, -100)
        + struct.pack('<8f', 0, 0, 0, 0, 0, 1, 0, 0) + struct.pack('<8q', *([-100]*8))
        + struct.pack('<8q', *([-100]*8))
        + struct.pack('<8q', *range(8)) + struct.pack('<8q', 0, 1, 2, 3, 0, 1, 2, 3)
        + struct.pack('<8q', 4, 4, 4, 4, 5, 2, 5, 5) + struct.pack('<8q', -1, 0, 1, 2, -1, 4, 5, 6))
    materials.append((dict(identity=digest(training), kind='training-view',
        bytes_sha256=digest(training), reconstruction=None), training))
    envelope = dict(neural_generation=invocation['model_generation'], provenance_identity=hid('provenance'),
        execution_materials=dict(action_base_h_logical=hid('base'), action_successor_h_logical=hid('successor'),
            invocation=invocation, training_view_identity=digest(training), baseline=None,
            materials=[reference for reference, _ in materials]))
    episode = dict(example_id='sample', final_split=split, training_view=dict(zip(names, view)),
        envelope=envelope, leakage_keys=['request-key'],
        intent=dict(task_text='retain original \u03c0\ud83d\ude00 material', constraints=[],
            filter_identity=None, budgets=dict(fuel=1)),
        action_trace=dict(total_return=-0.0))
    episode['content_sha256'] = digest(canonical(episode).encode())
    identity = (view, identity_digest, episode['content_sha256'], 4, 2, [[1, 1, 5]])
    # Complete data-carrier fixture only; these bytes are not native publication evidence.
    return ('episode', identity, canonical(episode), tuple(materials), b'carrier-evidence')
task_ref = 'task'
scope = ('controller', 'tenant', 'security', 'request', 'repository')
envelope = ['live', 'envelope', 'data-issuer', 'original acquisition spelling',
            'controller', 'tenant', 'security', 'confidential',
            ['current-request-fast-adaptation'], 'request', 'repository',
            '1970-01-01T00:00:10Z', 'delete', False, False, 'live-revoke']
replay = (replay_fixture(),)
nodes = (('source', 'source', (), (), ('envelope',), None, ('statement', 0)),
         ('mask', 'mask', ('source',), (), (), (0, 1), None),
         ('target', 'target_value', ('source',), (), (), (0, 1), None),
         ('receipt', 'derived', ('mask',), (), (), None, ('statement', 1)),
         ('feedback', 'derived', ('receipt',), (), (), None, None),
         ('observer', 'derived', ('source',), (), (), None, ('observer', 0)))
ids = tuple(node[0] for node in nodes)
def grant(name):
    return (name, name+'-issuer', 'allow', 'task', ids,
            ('1970-01-01T00:00:08Z', 8_000_000), name+'-revoke')
inputs = ('task', scope, ((envelope, 10_000_000),), replay, nodes,
          ('feedback',), (grant('publication'),), (grant('inference'),),
          (grant('training')+('current-request-fast-adaptation',),))
snapshot = (0, ('1970-01-01T00:00:01Z', 1_000_000),
            ('1970-01-01T00:00:05Z', 5_000_000),
            ((envelope, ('1970-01-01T00:00:05Z', 5_000_000), ()),), ())
"#;

    fn task_inputs(change: &str) -> (Vec<ColdValue>, AuthoritySnapshot) {
        Python::initialize();
        Python::attach(|py| {
            let globals = pyo3::types::PyDict::new(py);
            py.run(
                &CString::new(format!("{TASK_INPUT}\n{change}")).unwrap(),
                Some(&globals),
                None,
            )
            .unwrap();
            let mut budget = 16 * 1024 * 1024;
            let inputs =
                object_sequence(&globals.get_item("inputs").unwrap().unwrap(), &mut budget)
                    .unwrap();
            let inputs = inputs
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    if index == 3 {
                        read_replay_rows(value, &mut budget, 1 << 20, 1 << 20, 1024)
                    } else {
                        ColdValue::read(value, &mut budget, 0)
                    }
                })
                .collect::<PyResult<Vec<_>>>()
                .unwrap();
            let snapshot = ColdValue::read(
                &globals.get_item("snapshot").unwrap().unwrap(),
                &mut budget,
                0,
            )
            .unwrap();
            (inputs, AuthoritySnapshot::parse(&snapshot).unwrap())
        })
    }

    fn replay_transport(
        change: &str,
        material_limit: usize,
        evidence_limit: usize,
    ) -> PyResult<ColdValue> {
        Python::initialize();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            py.run(
                &CString::new(format!("{TASK_INPUT}\nrow = list(replay[0])\n{change}")).unwrap(),
                Some(&globals),
                None,
            )?;
            let row = globals.get_item("row")?.unwrap();
            let rows = PyTuple::new(py, [row])?;
            read_replay_rows(
                rows.as_any(),
                &mut (16 * 1024 * 1024),
                material_limit,
                1 << 20,
                evidence_limit,
            )
        })
    }

    #[test]
    fn replay_carrier_preserves_uninterpreted_metadata_with_original_training_identity() {
        let rows = replay_transport(
            r#"
episode = json.loads(row[2])
episode['application_metadata'] = dict(resource='edge', labels=['observed', 'reviewed'])
episode.pop('content_sha256')
canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'))
digest = hashlib.sha256(canonical(episode).encode()).hexdigest()
episode['content_sha256'] = digest
row[1] = (*row[1][:2], digest, *row[1][3:])
row[2] = canonical(episode)
"#,
            1 << 20,
            1024,
        )
        .unwrap();
        let row = ReplayRow::parse(&rows.sequence().unwrap()[0]).unwrap();
        assert!(row.record.contains_key("application_metadata"));
        assert!(!row.record.contains_key("replay_identity"));
        assert!(
            row.native_replay().is_err(),
            "carrier integrity must not issue native replay authority"
        );
    }

    #[test]
    fn replay_identity_binds_original_training_view_dimensions_and_targets() {
        for mutation in [
            "identity[0] = [*identity[0][:2], 'train', *identity[0][3:]]",
            "identity[0] = [identity[0][0], 'different-resource', *identity[0][2:]]",
            "identity[3] = 6; identity[5] = [[1, 1, 7]]",
            "identity[4] = 3",
            "identity[5] = [[2, 2, 6]]",
        ] {
            let change = format!("identity = list(row[1])\n{mutation}\nrow[1] = tuple(identity)");
            let rows = replay_transport(&change, 1 << 20, 1024).unwrap();
            assert!(
                ReplayRow::parse(&rows.sequence().unwrap()[0]).is_err(),
                "{mutation}"
            );
        }
    }

    #[test]
    fn replay_identity_cannot_detach_from_the_record_projection() {
        let rows = replay_transport(
            r#"
episode = json.loads(row[2])
episode['training_view']['mask_policy_hash'] = '9'*64
episode.pop('content_sha256')
canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'))
digest = hashlib.sha256(canonical(episode).encode()).hexdigest()
episode['content_sha256'] = digest
row[1] = (*row[1][:2], digest, *row[1][3:])
row[2] = canonical(episode)
"#,
            1 << 20,
            1024,
        )
        .unwrap();
        let error = ReplayRow::parse(&rows.sequence().unwrap()[0]).unwrap_err();
        assert!(error
            .to_string()
            .contains("training identity differs from its original record projection"));
    }

    #[test]
    fn replay_learning_eligibility_excludes_sealed_evaluation_contexts() {
        for usage in ["eval", "test", "unknown"] {
            let (values, _) = task_inputs(&format!(
                "replay = (replay_fixture('{usage}'),); inputs = (*inputs[:3], replay, *inputs[4:])"));
            let error = match TaskAuthority::parse(&values) {
                Ok(_) => panic!("accepted inadmissible replay usage {usage}"),
                Err(error) => error,
            };
            assert!(error
                .to_string()
                .contains("replay training identity names another example or a sealed partition"));
        }
    }

    #[test]
    fn full_replay_transport_retains_original_episode_materials_and_evidence() {
        let rows = replay_transport("", 1 << 20, 1024).unwrap();
        let row = ReplayRow::parse(&rows.sequence().unwrap()[0]).unwrap();
        assert_eq!(row.evidence.as_ref(), b"carrier-evidence");
        assert_eq!(
            row.identity.fields(6).unwrap()[0].fields(9).unwrap()[6],
            ColdValue::Integer("1267650600228229401496703205376".into())
        );
        assert!(row.record_line.contains("\\ud83d\\ude00"));
        assert_eq!(row.record.len(), 8);
        let ReplayBasis::Episode { execution } = &row.basis else {
            panic!("episode basis expected")
        };
        assert_eq!(
            execution["materials"].as_array().unwrap().len(),
            row.materials.len()
        );
        assert!(row
            .materials
            .iter()
            .any(|item| item.reconstruction.is_some() && item.bytes.is_none()));
        assert!(row
            .materials
            .iter()
            .filter(|item| item.kind != "training-view")
            .filter_map(|item| item.bytes.as_ref())
            .all(|bytes| bytes.starts_with(b"carrier:")));
    }

    #[test]
    fn full_replay_transport_rejects_missing_detached_reordered_and_altered_content() {
        for change in [
            "row = row[1]",
            "row[2] = row[2].replace('retain original', 'changed original')",
            "row[2] = '{\\\"content_sha256\\\":\\\"duplicate\\\",' + row[2][1:]",
            "row[3] = row[3][1:]",
            "row[3] = tuple(reversed(row[3]))",
            "row[3] = (*row[3], row[3][0])",
            "ref, payload = row[3][0]; row[3] = ((dict(ref, extra=1), payload), *row[3][1:])",
            "ref, payload = row[3][0]; row[3] = ((dict(ref, bytes_sha256='not-a-digest'), payload), *row[3][1:])",
            "ref, payload = row[3][0]; row[3] = ((dict(ref, kind='unknown-kind'), payload), *row[3][1:])",
            "items = list(row[3]); index = next(i for i,(ref,_) in enumerate(items) if ref['reconstruction']); ref = items[index][0]; items[index] = (dict(ref, reconstruction=dict(operation='forward', inputs=['f'*64])), None); row[3] = tuple(items)",
            "items = list(row[3]); index = next(i for i,(ref,_) in enumerate(items) if ref['reconstruction']); ref = items[index][0]; items[index] = (dict(ref, reconstruction=dict(operation='forward', inputs=[])), None); row[3] = tuple(items)",
            "items = list(row[3]); index = next(i for i,(ref,_) in enumerate(items) if ref['reconstruction']); ref = items[index][0]; items[index] = (dict(ref, reconstruction=dict(ref['reconstruction'], extra=1)), None); row[3] = tuple(items)",
            "ref, payload = row[3][0]; row[3] = ((ref, b'changed'), *row[3][1:])",
            "ref, payload = row[3][0]; row[3] = ((ref, None), *row[3][1:])",
            "ref, payload = row[3][0]; row[3] = ((ref, bytearray(payload)), *row[3][1:])",
            "row[4] = bytearray(row[4])",
            "row[4] = 'evidence'",
            "row[4] = b''",
            "context = list(row[1][0]); context[1] = 'another-identity'; row[1] = (context, *row[1][1:])",
            "row[1] = (row[1][0], '2'*64, *row[1][2:])",
            "row[1] = (*row[1][:2], '3'*64, *row[1][3:])",
        ] {
            let result = replay_transport(change, 1 << 20, 1024)
                .and_then(|rows| ReplayRow::parse(&rows.sequence()?[0]));
            assert!(result.is_err(), "accepted altered carrier: {change}");
        }
    }

    #[test]
    fn replay_transport_rejects_custom_container_and_payload_handlers() {
        let dictionary = replay_transport(
            "class Custom(dict):\n def __getitem__(self, key): raise AssertionError('custom callback ran')\nref, payload = row[3][0]; row[3] = ((Custom(ref), payload), *row[3][1:])",
            1 << 20, 1024,
        ).unwrap_err();
        assert!(dictionary
            .to_string()
            .contains("exact builtin dictionaries"));
        let bytes = replay_transport(
            "row[4] = type('CustomBytes', (bytes,), {})(row[4])",
            1 << 20,
            1024,
        )
        .unwrap_err();
        assert!(bytes.to_string().contains("exact builtin bytes"));
    }

    #[test]
    fn empty_replay_requires_no_material_or_evidence_bytes() {
        Python::initialize();
        Python::attach(|py| {
            let rows = PyTuple::empty(py);
            assert!(read_replay_rows(rows.as_any(), &mut 1, 1, 1, 1)
                .unwrap()
                .sequence()
                .unwrap()
                .is_empty());
            for caps in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
                assert!(read_replay_rows(rows.as_any(), &mut 1, caps.0, caps.1, caps.2).is_err());
            }
        });
    }

    #[test]
    fn replay_payload_limits_match_per_item_and_unique_total_caps() {
        Python::initialize();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            py.run(
                &CString::new(format!(
                    "{TASK_INPUT}\nrows = (replay_fixture(), replay_fixture())"
                ))
                .unwrap(),
                Some(&globals),
                None,
            )
            .unwrap();
            let rows = globals.get_item("rows").unwrap().unwrap();
            let read = |material, total, evidence| {
                read_replay_rows(&rows, &mut (16 * 1024 * 1024), material, total, evidence)
            };
            let parsed = read(1 << 20, 1 << 20, 1024).unwrap();
            let row = ReplayRow::parse(&parsed.sequence().unwrap()[0]).unwrap();
            let material_bytes = row
                .materials
                .iter()
                .filter_map(|item| item.bytes.as_ref())
                .map(|bytes| bytes.len())
                .sum::<usize>();
            let maximum = row
                .materials
                .iter()
                .filter_map(|item| item.bytes.as_ref())
                .map(|bytes| bytes.len())
                .max()
                .unwrap();
            // Both episodes reuse the same blobs; evidence is bounded per row.
            read(maximum, material_bytes, row.evidence.len()).unwrap();
            assert!(read(maximum - 1, material_bytes, row.evidence.len()).is_err());
            assert!(read(maximum, material_bytes - 1, row.evidence.len()).is_err());
            assert!(read(maximum, material_bytes, row.evidence.len() - 1).is_err());
        });
    }

    #[test]
    fn replay_caps_bound_each_material_and_evidence_without_double_counting_shared_blobs() {
        let rows = replay_transport("", 1 << 20, 1024).unwrap();
        let row = ReplayRow::parse(&rows.sequence().unwrap()[0]).unwrap();
        let maximum = row
            .materials
            .iter()
            .filter_map(|item| item.bytes.as_ref())
            .map(|bytes| bytes.len())
            .max()
            .unwrap();
        replay_transport("", maximum, row.evidence.len())
            .expect("per-material cap is not a whole-row sum");
    }

    #[test]
    fn replay_shared_material_payloads_retain_one_owned_allocation() {
        Python::initialize();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            py.run(
                &CString::new(format!(
                    "{TASK_INPUT}\nrows = (replay_fixture(), replay_fixture())"
                ))
                .unwrap(),
                Some(&globals),
                None,
            )
            .unwrap();
            let rows = read_replay_rows(
                &globals.get_item("rows").unwrap().unwrap(),
                &mut (16 * 1024 * 1024),
                1 << 20,
                1 << 20,
                1024,
            )
            .unwrap();
            let first = ReplayRow::parse(&rows.sequence().unwrap()[0]).unwrap();
            let second = ReplayRow::parse(&rows.sequence().unwrap()[1]).unwrap();
            assert!(
                std::sync::Arc::ptr_eq(
                    first.materials[0].bytes.as_ref().unwrap(),
                    second.materials[0].bytes.as_ref().unwrap()
                ),
                "one SHA-addressed material was copied once per episode"
            );
        });
    }

    #[test]
    fn replay_python_view_round_trips_and_isolates_mutable_carrier_metadata() {
        Python::initialize();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            py.run(
                &CString::new(format!(
                    "{TASK_INPUT}\nsource = list(replay[0]); source[1] = list(source[1])"
                ))
                .unwrap(),
                Some(&globals),
                None,
            )
            .unwrap();
            let source = globals.get_item("source").unwrap().unwrap();
            let input = PyTuple::new(py, [&source]).unwrap();
            let retained = read_replay_rows(
                input.as_any(),
                &mut (16 * 1024 * 1024),
                1 << 20,
                1 << 20,
                1024,
            )
            .unwrap();
            // This is the complete data carrier, not fabricated native evidence.
            // All caller-owned lists and reference dictionaries can now change.
            py.run(c"source[1][0][6] = -1\nsource[2] = 'changed episode'\nsource[3][0][0]['identity'] = 'changed identity'\nnext(ref for ref, _ in source[3] if ref['reconstruction'])['reconstruction']['inputs'].clear()\nsource[4] = b'changed evidence'",
                Some(&globals), None).unwrap();
            let row = ReplayRow::parse(&retained.sequence().unwrap()[0]).unwrap();
            assert_eq!(
                row.identity.fields(6).unwrap()[0].fields(9).unwrap()[6],
                ColdValue::Integer("1267650600228229401496703205376".into())
            );
            assert_eq!(row.evidence.as_ref(), b"carrier-evidence");
            assert!(row.record_line.contains("-0.0"));
            assert!(row.record_line.contains("\\ud83d\\ude00"));

            let exported = row.python_view(py).unwrap();
            let output = PyTuple::new(py, [exported.bind(py)]).unwrap();
            let round_trip = read_replay_rows(
                output.as_any(),
                &mut (16 * 1024 * 1024),
                1 << 20,
                1 << 20,
                1024,
            )
            .unwrap();
            assert_eq!(round_trip, retained);
            ReplayRow::parse(&round_trip.sequence().unwrap()[0]).unwrap();

            globals.set_item("exported", exported).unwrap();
            py.run(c"assert type(exported[1][0][6]) is int and exported[1][0][6] == 2**100\nassert any(ref['reconstruction'] is not None and payload is None for ref, payload in exported[3])\nexported[3][0][0]['kind'] = 'changed kind'\nnext(ref for ref, _ in exported[3] if ref['reconstruction'])['reconstruction']['inputs'].append('changed input')",
                Some(&globals), None).unwrap();
            let fresh = row.python_view(py).unwrap();
            let output = PyTuple::new(py, [fresh.bind(py)]).unwrap();
            let after_mutation = read_replay_rows(
                output.as_any(),
                &mut (16 * 1024 * 1024),
                1 << 20,
                1 << 20,
                1024,
            )
            .unwrap();
            assert_eq!(after_mutation, retained);
            ReplayRow::parse(&after_mutation.sequence().unwrap()[0]).unwrap();
        });
    }

    #[test]
    fn import_cleanup_preserves_primary_exception_and_refuses_cleanup_failure() {
        use pyo3::exceptions::{PyRuntimeError, PyValueError};

        Python::initialize();
        Python::attach(|py| {
            let primary = PyValueError::new_err("original replay failure");
            let primary_object = primary.value(py).clone();
            let cleanup = PyRuntimeError::new_err("original owner cleanup failure");
            let cleanup_object = cleanup.value(py).clone();
            let error =
                super::finish_with_cleanup::<()>(py, Err(primary), Err(cleanup)).unwrap_err();
            assert!(error.value(py).is(&primary_object));
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(error.cause(py).unwrap().value(py).is(&cleanup_object));

            let cleanup = PyRuntimeError::new_err("cleanup prevents issuance");
            let cleanup_object = cleanup.value(py).clone();
            let error = super::finish_with_cleanup(py, Ok(17), Err(cleanup)).unwrap_err();
            assert!(error.value(py).is(&cleanup_object));
            assert_eq!(super::finish_with_cleanup(py, Ok(17), Ok(())).unwrap(), 17);
        });
    }

    #[test]
    fn import_cleanup_retains_existing_causes_and_each_later_cleanup_error() {
        use pyo3::exceptions::{PyRuntimeError, PyValueError};

        Python::initialize();
        Python::attach(|py| {
            let primary = PyValueError::new_err("original replay failure");
            let original_cause = PyRuntimeError::new_err("original callback cause");
            let successor_cleanup = PyRuntimeError::new_err("successor reader cleanup failure");
            let successor_cause = PyRuntimeError::new_err("successor completion failure");
            let predecessor_cleanup = PyRuntimeError::new_err("predecessor reader cleanup failure");
            let expected = [
                primary.value(py).clone(),
                original_cause.value(py).clone(),
                successor_cleanup.value(py).clone(),
                successor_cause.value(py).clone(),
                predecessor_cleanup.value(py).clone(),
            ];
            primary.set_cause(py, Some(original_cause));
            successor_cleanup.set_cause(py, Some(successor_cause));

            let comparison =
                super::finish_with_cleanup::<()>(py, Err(primary), Err(successor_cleanup));
            let error =
                super::finish_with_cleanup(py, comparison, Err(predecessor_cleanup)).unwrap_err();
            let mut next = Some(error);
            for expected_object in expected {
                let current = next
                    .take()
                    .expect("each original exception remains reachable");
                assert!(current.value(py).is(&expected_object));
                next = current.cause(py);
            }
            assert!(next.is_none(), "cleanup must not create a cause cycle");
        });
    }

    #[test]
    fn import_cleanup_deduplicates_overlapping_and_cyclic_exception_chains() {
        use pyo3::exceptions::{PyRuntimeError, PyValueError};

        Python::initialize();
        Python::attach(|py| {
            let primary = PyValueError::new_err("original replay failure");
            let cause = PyRuntimeError::new_err("existing cause");
            let cleanup = PyRuntimeError::new_err("cleanup failure");
            let expected = [
                primary.value(py).clone(),
                cause.value(py).clone(),
                cleanup.value(py).clone(),
            ];
            primary.set_cause(py, Some(cause.clone_ref(py)));
            cause.set_cause(py, Some(primary.clone_ref(py)));
            cleanup.set_cause(py, Some(cause));

            let error =
                super::finish_with_cleanup::<()>(py, Err(primary), Err(cleanup)).unwrap_err();
            let repeated = error.cause(py).unwrap();
            let error =
                super::finish_with_cleanup::<()>(py, Err(error), Err(repeated)).unwrap_err();
            let mut next = Some(error);
            for expected_object in expected {
                let current = next
                    .take()
                    .expect("each distinct exception remains reachable");
                assert!(current.value(py).is(&expected_object));
                next = current.cause(py);
            }
            assert!(next.is_none(), "overlapping causes must not create a cycle");
        });
    }

    #[test]
    fn replay_nested_json_structure_refuses_duplicate_keys_and_noncompact_containers() {
        for replacement in [
            "line = row[2].replace('\\\"task_text\\\":', '\\\"task_text\\\":\\\"shadow\\\",\\\"task_text\\\":', 1)",
            "line = row[2].replace('\\\"intent\\\":{', '\\\"intent\\\":{ ', 1)",
            "line = row[2].replace('\\\"leakage_keys\\\":[', '\\\"leakage_keys\\\":[ ', 1)",
            "line = row[2].replace('\\\"constraints\\\":[],\\\"filter_identity\\\":null', '\\\"filter_identity\\\":null,\\\"constraints\\\":[]', 1)",
        ] {
            let change = format!("{replacement}\nold = row[1][2]; body = line.replace('\\\"content_sha256\\\":\\\"'+old+'\\\",', '', 1); digest = hashlib.sha256(body.encode()).hexdigest(); row[2] = line.replace(old, digest, 1); row[1] = (*row[1][:2], digest, *row[1][3:])");
            let result = replay_transport(&change, 1 << 20, 1024).and_then(|rows| ReplayRow::parse(&rows.sequence()?[0]));
            assert!(result.is_err(), "accepted noncanonical JSON structure: {replacement}");
        }
    }

    #[test]
    fn replay_json_key_order_preserves_lone_surrogates_and_non_bmp_code_points() {
        super::validate_replay_json_structure(
            r#"{"a":0,"\ud800":1,"\ue000":2,"\ud800\udc00":3}"#,
            0,
        )
        .unwrap();
        for raw in [r#"{"a":0,"\u0061":1}"#, r#"{"\ud800\udc00":3,"\ue000":2}"#] {
            assert!(
                super::validate_replay_json_structure(raw, 0).is_err(),
                "{raw}"
            );
        }
    }

    #[test]
    fn replay_reconstruction_json_refuses_duplicate_native_fields() {
        let result = replay_transport(
            "old = row[1][2]; line = row[2].replace('\\\"operation\\\":', '\\\"operation\\\":\\\"ignored\\\",\\\"operation\\\":', 1); body = line.replace('\\\"content_sha256\\\":\\\"'+old+'\\\",', '', 1); digest = hashlib.sha256(body.encode()).hexdigest(); row[2] = line.replace(old, digest, 1); row[1] = (*row[1][:2], digest, *row[1][3:])",
            1 << 20, 1024,
        ).and_then(|rows| ReplayRow::parse(&rows.sequence()?[0]));
        assert!(
            result.is_err(),
            "duplicate reconstruction operation was silently overwritten"
        );
    }

    #[test]
    fn replay_episode_hash_preserves_uninterpreted_json_lexemes() {
        let rows = replay_transport(
            "episode = json.loads(row[2]); episode['intent']['task_text'] += '\\ud800'; episode['intent']['budgets']['fuel'] = 2**200; episode['intent']['budgets']['\\ud800'] = 1; episode['intent']['budgets']['\\ue000'] = 2; episode['intent']['budgets']['\\U00010000'] = 3; episode['action_trace']['total_return'] = 1e-09\nepisode.pop('content_sha256'); canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':')); digest = hashlib.sha256(canonical(episode).encode()).hexdigest(); episode['content_sha256'] = digest\nrow[1] = canonical(episode); row[1] = (*row[1][:2], digest, *row[1][3:])",
            1 << 20, 1024,
        ).unwrap();
        let row = ReplayRow::parse(&rows.sequence().unwrap()[0]).unwrap();
        assert!(row.record_line.contains("\\ud800"));
        assert!(row.record_line.contains("1e-09"));
        assert!(row
            .record_line
            .contains("1606938044258990275541962092341162602522202993782792835301376"));
    }

    fn observed_source_authority(change: &str) -> PyResult<(TaskAuthority, AuthoritySnapshot)> {
        Python::initialize();
        Python::attach(|py| {
            let globals = pyo3::types::PyDict::new(py);
            let source = r#"
import hashlib, json
tokens = (40, 41, 42)
field = 'observed π😀\nfield'
tokenizer, revision = 'tokenizer', 'revision'
digest = lambda value: hashlib.sha256(json.dumps(value, separators=(',', ':'), sort_keys=True).encode()).hexdigest()
observation = [field, digest(list(tokens)), digest(dict(template='observed-source-field:v1', field=field, tokenizer_hash=tokenizer, tokenizer_revision=revision)), len(tokens)]
entry = ['example', 'request-local-replay', '4'*64, '5'*64, '6'*64,
         '8'*64, ['leakage-key'], envelope,
         ['live', ['request-local-replay'], 'request-local'], observation]
sources = [('source', '7'*64, tokenizer, revision, entry, tokens)]
mapping = [('source', 1, 7)]
"#;
            py.run(
                &CString::new(format!("{TASK_INPUT}\n{source}\n{change}")).unwrap(),
                Some(&globals),
                None,
            )?;
            let mut budget = 16 * 1024 * 1024;
            let inputs = object_sequence(&globals.get_item("inputs")?.unwrap(), &mut budget)?;
            let inputs = inputs
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    if index == 3 {
                        read_replay_rows(value, &mut budget, 1 << 20, 1 << 20, 1024)
                    } else {
                        ColdValue::read(value, &mut budget, 0)
                    }
                })
                .collect::<PyResult<Vec<_>>>()?;
            let mut authority = TaskAuthority::parse(&inputs)?;
            let mut read =
                |name| ColdValue::read(&globals.get_item(name)?.unwrap(), &mut budget, 0);
            let snapshot = AuthoritySnapshot::parse(&read("snapshot")?)?;
            authority.bind_initial_sources(&read("sources")?, &read("mapping")?)?;
            Ok((authority, snapshot))
        })
    }

    #[test]
    fn observed_source_decontamination_follows_the_actual_provenance_branch() {
        observed_source_authority("entry[5] = None").unwrap();
        observed_source_authority("entry[5] = '8'*64").unwrap();
        assert!(observed_source_authority("entry[5] = 'not-a-digest'").is_err());
        assert!(observed_source_authority("entry[5] = None\nentry[7] = ['corpus', 'corpus-source', 'row', 'revision', 'retrieved', 'license', 'obligations', ['backbone'], ['training'], 'restrictions', 'retention']\nentry[1] = 'train'\nentry[8] = ['corpus', ['train', 'dev'], 'corpus']").is_err());
    }

    #[test]
    fn observed_source_preserves_real_payload_transform_origin_and_task_authority() {
        let (authority, snapshot) = observed_source_authority("").unwrap();
        assert_eq!(authority.initial_sources[0].tokens, [40, 41, 42]);
        assert_eq!(authority.source_mapping[0].source, "source");
        assert_eq!(authority.source_mapping[0].offset, 1);
        assert_eq!(authority.source_mapping[0].logical_position, 7);
        authority.check_use("inference", &snapshot, true).unwrap();
        let mut changed = authority.clone();
        changed.inference.clear();
        assert!(changed.check_use("inference", &snapshot, true).is_err());
        let mut expired = snapshot.clone();
        expired.observed_at.micros = 11_000_000;
        assert!(authority.check_use("inference", &expired, true).is_err());
        assert_ne!(
            authority.canonical,
            TaskAuthority::parse(&task_inputs("").0).unwrap().canonical
        );
    }

    fn fresh_source_authority(change: &str) -> PyResult<(TaskAuthority, AuthoritySnapshot)> {
        // Fresh observations have no replay target/MASK projection. Keep the
        // actual source, derived receipt, feedback and observer dependency roots.
        observed_source_authority(&format!(
            "nodes = (nodes[0], ('receipt', 'derived', ('source',), (), (), None, ('statement', 1)), nodes[4], nodes[5])\n\
             ids = tuple(node[0] for node in nodes)\n\
             inputs = (*inputs[:3], (), nodes, inputs[5], (grant('publication'),), (grant('inference'),), (grant('training')+('current-request-fast-adaptation',),))\n\
             {change}",
        ))
    }

    #[test]
    fn fresh_training_uses_admitted_sources_without_inventing_replay() {
        let (authority, snapshot) = fresh_source_authority("").unwrap();
        assert!(authority.replay.is_empty());
        assert!(!authority.initial_sources.is_empty());
        assert!(!authority.source_mapping.is_empty());
        for before in [true, false] {
            authority.check_use("training", &snapshot, before).unwrap();
            authority.check_use("inference", &snapshot, before).unwrap();
            let replay = TaskAuthority::parse(&task_inputs("").0).unwrap();
            assert!(replay.initial_sources.is_empty());
            replay.check_use("training", &snapshot, before).unwrap();
        }

        let mut state = TaskUseState {
            phase: TaskUsePhase::Imported,
            snapshot,
        };
        let scope = state.begin_build().unwrap();
        state.finish_build(&scope).unwrap();
        let (_, before) = task_inputs("snapshot = (1, *snapshot[1:])");
        state
            .admit_built_segment(&scope, &authority, "training".into(), before)
            .unwrap();
        let (_, after) =
            task_inputs("snapshot = (2, ('1970-01-01T00:00:06Z', 6_000_000), *snapshot[2:])");
        // This is the production authority state machine, not a native FINAL
        // lease or permission to deliver an external effect.
        state.revalidate_activation(&authority, after).unwrap();
    }

    #[test]
    fn fresh_training_requires_both_admitted_sources_and_mapping() {
        for change in ["sources = []; mapping = []", "mapping = []"] {
            let (authority, snapshot) = fresh_source_authority(change).unwrap();
            assert!(authority.sources_allow_learning);
            for before in [true, false] {
                authority.check_use("inference", &snapshot, before).unwrap();
                assert!(authority.check_use("training", &snapshot, before).is_err());
            }
        }
        assert!(fresh_source_authority("sources = []")
            .err()
            .unwrap()
            .to_string()
            .contains("unknown source"));
    }

    #[test]
    fn fresh_training_preserves_independent_rights_and_source_partitions() {
        for (change, expected) in [
            ("envelope[8] = []; entry[1] = None; entry[8] = ['live', [], 'request-local']",
                "source admission excludes learning"),
            ("entry[1] = 'eval'; entry[8][1] = ['eval']", "source admission excludes learning"),
            ("inputs = (*inputs[:6], (), *inputs[7:])", "full dependency closure"),
            ("inputs = (*inputs[:8], ())", "full dependency closure"),
            ("inputs = (*inputs[:8], ((*inputs[8][0][:4], ('source',), *inputs[8][0][5:]),))",
                "full dependency closure"),
            ("inputs = (*inputs[:8], ((*inputs[8][0][:2], 'deny', *inputs[8][0][3:]),))",
                "denied, expired, or revoked"),
            ("inputs = (*inputs[:8], ((*inputs[8][0][:5], ('1970-01-01T00:00:01Z', 1_000_000), *inputs[8][0][6:]),))",
                "denied, expired, or revoked"),
            ("snapshot = (*snapshot[:4], ('training-revoke',))", "denied, expired, or revoked"),
            ("snapshot = (*snapshot[:4], ('publication-revoke',))", "denied, expired, or revoked"),
            ("inputs = (*inputs[:8], ((*inputs[8][0][:7], 'durable-slow-consolidation'),))",
                "immutable split or replay scope"),
        ] {
            let (authority, snapshot) = fresh_source_authority(change).unwrap();
            for before in [true, false] {
                let error = authority.check_use("training", &snapshot, before).unwrap_err();
                assert!(error.to_string().contains(expected), "{change}: {error}");
            }
        }
    }

    #[test]
    fn fresh_training_revalidates_freshness_expiry_and_revocation_before_activation() {
        let (authority, snapshot) = fresh_source_authority("").unwrap();
        let (_, before) = task_inputs("snapshot = (1, *snapshot[1:])");
        for change in [
            "snapshot = (1, *snapshot[1:])",
            "snapshot = (2, ('1970-01-01T00:00:08Z', 8_000_000), *snapshot[2:])",
            "snapshot = (2, ('1970-01-01T00:00:10Z', 10_000_000), *snapshot[2:])",
            "snapshot = (2, *snapshot[1:4], ('training-revoke',))",
            "snapshot = (2, *snapshot[1:4], ('publication-revoke',))",
            "snapshot = (2, *snapshot[1:3], ((envelope, snapshot[2], ('live-revoke',)),), ())",
        ] {
            let mut state = TaskUseState {
                phase: TaskUsePhase::Imported,
                snapshot: snapshot.clone(),
            };
            state
                .admit_segment(&authority, "training".into(), before.clone())
                .unwrap();
            let (_, after) = task_inputs(change);
            assert!(
                state.revalidate_activation(&authority, after).is_err(),
                "{change}"
            );
            assert!(matches!(state.phase, TaskUsePhase::Refused));
            assert!(state.require_public_use().is_err());
        }
    }

    #[test]
    fn observed_source_rejects_payload_mapping_origin_and_target_ancestry_substitution() {
        for change in [
            "sources[0] = (*sources[0][:5], (42, 41, 40))",
            "sources[0] = (*sources[0][:2], 'other-tokenizer', *sources[0][3:])",
            "sources[0] = ('foreign-task-source', *sources[0][1:])",
            "mapping = [('source', 3, 7)]",
            "mapping = [('source', 0, 7), ('source', 1, 7)]",
            "entry[7] = [*envelope[:9], 'another-request', *envelope[10:]]",
            "entry[9][3] = 4",
            "entry[9][0] = 'different-field'",
            "sources[0] = ('target', *sources[0][1:]); mapping = [('target', 0, 7)]",
            "nodes += (('hidden-target', 'derived', ('source',), ('target',), (), None, None),)\ninputs = (*inputs[:4], nodes, *inputs[5:])\nsources[0] = ('hidden-target', *sources[0][1:]); mapping = [('hidden-target', 0, 7)]",
            "sources.append(sources[0])",
        ] {
            assert!(observed_source_authority(change).is_err(), "{change}");
        }
    }

    #[test]
    fn observed_inference_only_live_source_does_not_acquire_learning_rights() {
        let (authority, snapshot) = observed_source_authority(
            "envelope[8] = []\nentry[1] = None\nentry[8] = ['live', [], 'request-local']",
        )
        .unwrap();
        authority.check_use("inference", &snapshot, true).unwrap();
        assert!(!authority.sources_allow_learning);
        assert!(authority.check_use("training", &snapshot, true).is_err());
        assert!(observed_source_authority("envelope[8] = []").is_err());
    }

    #[test]
    fn observed_corpus_source_retains_the_canonical_admitted_entry() {
        let (authority, snapshot) = observed_source_authority(
            "entry[7] = ['corpus', 'corpus-source', 'row', 'revision', 'retrieved', 'license', 'obligations', ['backbone'], ['training'], 'restrictions', 'retention']\nentry[1] = 'train'\nentry[8] = ['corpus', ['train', 'dev'], 'corpus']",
        ).unwrap();
        authority.check_use("inference", &snapshot, true).unwrap();
        assert!(authority.sources_allow_learning);
        assert!(authority.initial_sources[0]
            .origin
            .windows(b"corpus-source".len())
            .any(|part| part == b"corpus-source"));
    }

    #[test]
    fn observed_request_local_source_does_not_inherit_durable_replay_permission() {
        let (authority, snapshot) = observed_source_authority(
            "envelope[8] = ['current-request-fast-adaptation', 'durable-slow-consolidation']\nenvelope[13] = envelope[14] = True\nreplay = (replay_fixture('train'),)\ninputs = (*inputs[:3], replay, *inputs[4:8], (grant('training')+('durable-slow-consolidation',),))",
        ).unwrap();
        assert!(authority.check_use("training", &snapshot, true).is_err());
    }

    #[test]
    fn observed_corpus_only_task_does_not_fabricate_a_live_envelope() {
        let (authority, snapshot) = observed_source_authority(
            "entry[7] = ['corpus', 'corpus-source', 'row', 'revision', 'retrieved', 'license', 'obligations', ['backbone'], ['training'], 'restrictions', 'retention']\nentry[1] = 'train'\nentry[8] = ['corpus', ['train', 'dev'], 'corpus']\nnodes = tuple((node[0], node[1], node[2], node[3], (), *node[5:]) for node in nodes)\ninputs = (*inputs[:2], (), inputs[3], nodes, *inputs[5:])\nsnapshot = (*snapshot[:3], (), snapshot[4])",
        ).unwrap();
        assert!(authority.live.is_empty());
        authority.check_use("inference", &snapshot, true).unwrap();
        let mut expired = snapshot.clone();
        expired.observed_at = snapshot.segment_end.clone();
        assert!(authority.check_use("inference", &expired, true).is_err());
        let mut extended = snapshot.clone();
        extended.revision += 1;
        extended.segment_end.micros += 1;
        assert!(extended.newer_than(&snapshot).is_err());
    }

    #[test]
    fn live_snapshot_cannot_substitute_a_different_global_segment_boundary() {
        let (inputs, snapshot) = task_inputs(
            "snapshot = (*snapshot[:2], ('1970-01-01T00:00:06Z', 6_000_000), *snapshot[3:])",
        );
        let authority = TaskAuthority::parse(&inputs).unwrap();
        assert!(authority.check_use("inference", &snapshot, true).is_err());
    }

    #[test]
    fn live_data_without_learning_purposes_preserves_independent_use_grants() {
        let (inputs, snapshot) = task_inputs(
            "envelope, retention = inputs[2][0]\n\
             envelope = (*envelope[:8], (), *envelope[9:])\n\
             inputs = (*inputs[:2], ((envelope, retention),), *inputs[3:])\n\
             snapshot = (*snapshot[:3], ((envelope, *snapshot[3][0][1:]),), *snapshot[4:])",
        );
        let authority = TaskAuthority::parse(&inputs).unwrap();
        authority.check_use("inference", &snapshot, true).unwrap();
        assert!(authority
            .check_use("training", &snapshot, true)
            .unwrap_err()
            .to_string()
            .contains("training grant exceeds the live data-use learning purpose"));

        let mut missing_inference = TaskAuthority::parse(&inputs).unwrap();
        missing_inference.inference.clear();
        assert!(missing_inference
            .check_use("inference", &snapshot, true)
            .unwrap_err()
            .to_string()
            .contains("required independent grant does not cover the full dependency closure"));

        let mut missing_publication = authority;
        missing_publication.publication.clear();
        assert!(missing_publication
            .check_use("inference", &snapshot, true)
            .unwrap_err()
            .to_string()
            .contains("required independent grant does not cover the full dependency closure"));
    }

    #[test]
    fn live_data_rejects_unrecognized_learning_purposes() {
        let (inputs, _) = task_inputs(
            "envelope, retention = inputs[2][0]\n\
             envelope = (*envelope[:8], ('inference',), *envelope[9:])\n\
             inputs = (*inputs[:2], ((envelope, retention),), *inputs[3:])",
        );
        assert!(TaskAuthority::parse(&inputs)
            .err()
            .unwrap()
            .to_string()
            .contains("live envelope has an invalid learning purpose"));
    }

    #[test]
    fn live_data_use_rejects_snapshot_learning_purpose_mismatch() {
        let (inputs, snapshot) = task_inputs(
            "envelope, retention = inputs[2][0]\n\
             envelope = (*envelope[:8], (), *envelope[9:])\n\
             inputs = (*inputs[:2], ((envelope, retention),), *inputs[3:])",
        );
        let authority = TaskAuthority::parse(&inputs).unwrap();
        assert!(authority
            .check_use("inference", &snapshot, true)
            .unwrap_err()
            .to_string()
            .contains("snapshot envelope differs from the imported live provenance"));
    }

    #[test]
    fn feedback_schema_depends_on_layout_not_statement_values() {
        let first = FeedbackSchema::new([&[0, 1], &[2], &[3]]).unwrap();
        let other = FeedbackSchema::new([&[255, 3], &[4], &[5]]).unwrap();
        assert_eq!(first.statement_bytes, 2);
        assert_eq!(first.feature_width, 22);
        assert_eq!(first.identity, other.identity);
        assert_ne!(
            first.identity,
            FeedbackSchema::new([&[0], &[1], &[2]]).unwrap().identity
        );
        assert!(FeedbackSchema::new([&[], &[1], &[2]]).is_err());
    }

    #[test]
    fn feedback_schema_seals_flat_bit_and_presence_order() {
        use sha2::{Digest, Sha256};

        let schema = FeedbackSchema::new([&[0, 1], &[2], &[3]]).unwrap();
        let mut hash = Sha256::new();
        hash.update(b"xlog.feedback.flat-bit-presence.v1\0");
        hash.update(2u64.to_le_bytes());
        hash.update(b"byte[i]:bits[0..8],presence;pro,1,contra,1;invalid:all-zero");
        let expected: [u8; 32] = hash.finalize().into();
        assert_eq!(schema.identity, expected);
    }

    #[test]
    fn parent_text_transport_preserves_original_ring_slots_and_requires_typed_flags() {
        Python::initialize();
        Python::attach(|py| {
            let rows = py
                .eval(
                    c"[(7, 2, 1, 1, True, False, False, 4), (9, 1, 2, 0, True, False, False, 5)]",
                    None,
                    None,
                )
                .unwrap();
            let mut budget = 4096;
            let slots = parse_text_slots(&ColdValue::read(&rows, &mut budget, 0).unwrap()).unwrap();
            assert_eq!(slots[0].logical_position, 2);
            assert_eq!(slots[1].logical_position, 1);
            assert_eq!(slots[0].provenance_record, 4);
            assert_eq!(slots[1].kind, 2);
            let rows = py
                .eval(c"[(7, 2, 1, 1, 1, False, False, 4)]", None, None)
                .unwrap();
            assert!(parse_text_slots(&ColdValue::read(&rows, &mut budget, 0).unwrap()).is_err());
        });
    }

    #[test]
    fn dependency_projection_preserves_data_control_and_native_record_edges() {
        Python::initialize();
        Python::attach(|py| {
            let dependencies = vec![super::Dependency {
                identity: "query-input".into(),
                kind: "mask".into(),
                data_parents: vec!["source-b".into(), "source-a".into()],
                control_parents: vec!["choice".into()],
                live_envelopes: vec!["source-grant".into()],
                target: Some((3, 7)),
                native_record: Some(("support".into(), 5)),
            }];
            let lineage = super::dependency_lineage_object(py, &dependencies).unwrap();
            let locals = pyo3::types::PyDict::new(py);
            locals.set_item("lineage", lineage).unwrap();
            py.run(
                &CString::new(
                    r#"
assert type(lineage) is tuple
assert lineage == ((
    'query-input', 'mask', ('source-b', 'source-a'), ('choice',),
    ('source-grant',), (3, 7), ('support', 5),
),)
assert type(lineage[0]) is tuple
"#,
                )
                .unwrap(),
                None,
                Some(&locals),
            )
            .unwrap();
        });
    }

    #[test]
    fn descriptor_projection_preserves_native_meaning_and_typed_null() {
        use xlog_core::{RelId, ScalarType};
        use xlog_cuda::SemanticActionDescriptor;

        Python::initialize();
        Python::attach(|py| {
            let project = |descriptor: &SemanticActionDescriptor| {
                super::descriptor_meaning_object(py, descriptor).unwrap()
            };
            let target = project(&SemanticActionDescriptor::Target {
                predicate: RelId(19),
            });
            assert_eq!(
                target.extract::<(String, u32)>(py).unwrap(),
                ("target".into(), 19)
            );

            let bits = vec![0, 0, 0, 128];
            let operand = project(&SemanticActionDescriptor::Operand {
                scalar_type: ScalarType::F32,
                sort_label: "measurement".into(),
                encoded_value: bits.clone(),
            });
            let (kind, scalar, sort, bytes) = operand
                .extract::<(String, u8, String, Vec<u8>)>(py)
                .unwrap();
            assert_eq!(
                (kind.as_str(), scalar, sort.as_str(), bytes),
                ("operand", ScalarType::F32.to_code(), "measurement", bits)
            );

            let empty = project(&SemanticActionDescriptor::QualifierBundle { records: vec![] });
            assert!(!empty.is_none(py));
            assert_eq!(
                empty.extract::<(String, Vec<u32>)>(py).unwrap(),
                ("qualifier_bundle".into(), vec![])
            );
            let qualified = project(&SemanticActionDescriptor::QualifierBundle {
                records: vec![7, 2, 7],
            });
            assert_eq!(
                qualified.extract::<(String, Vec<u32>)>(py).unwrap(),
                ("qualifier_bundle".into(), vec![7, 2, 7])
            );
            let support = project(&SemanticActionDescriptor::SupportEvent { source_record: 5 });
            assert_eq!(
                support.extract::<(String, u32)>(py).unwrap(),
                ("support_event".into(), 5)
            );
        });
    }

    #[test]
    fn published_parent_cannot_be_constructed_or_transition_mode_coerced() {
        run_python(
            r#"
try:
    SemanticPublishedParent()
except TypeError:
    pass
else:
    raise AssertionError('a parent requires an actual native acquire')
"#,
        );
        for mode in ["proposal", "recompute", "drain"] {
            assert!(super::transition_kind(&ColdValue::Text(mode.into())).is_ok());
        }
        for mode in [
            ColdValue::Bool(true),
            ColdValue::Integer("1".into()),
            ColdValue::Text("training".into()),
        ] {
            assert!(super::transition_kind(&mode).is_err());
        }
    }

    #[test]
    fn content_witness_requires_native_issuance_and_exposes_only_verification() {
        run_python(
            r#"
assert 'SemanticTensorContentWitness' in globals(), 'native content witness type is not exported'
assert hasattr(SemanticPublishedParent, 'guard_content')
assert hasattr(SemanticTransitionController, 'capture_tensor_content')
import inspect
assert tuple(inspect.signature(SemanticTransitionController.bind_model_content).parameters) == (
    'self', 'task_use', 'parent', 'tensors', 'consumer_stream', 'bank')
model_content = inspect.signature(SemanticTransitionController.bind_model_content)
assert model_content.parameters['bank'].kind is inspect.Parameter.KEYWORD_ONLY
assert model_content.parameters['bank'].default is None
continuation = inspect.signature(SemanticTransitionController.bind_continuation)
assert tuple(continuation.parameters) == (
    'self', 'task_use', 'parent', 'text_rows', 'text_row_count', 'selected_text',
    'active_rows', 'active_row_count', 'numerical_admissibility', 'tensors',
    'producer_witness', 'consumer_stream', 'bank')
for name in tuple(continuation.parameters)[2:-1]:
    parameter = continuation.parameters[name]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
assert continuation.parameters['bank'].kind is inspect.Parameter.KEYWORD_ONLY
assert continuation.parameters['bank'].default is None
assert hasattr(SemanticTensorContentWitness, 'verify')
for name in ('digest', 'pointer', 'status', 'identity', 'tensors', 'parent'):
    assert not hasattr(SemanticTensorContentWitness, name), name
for construct in (lambda: SemanticTensorContentWitness(),
                  lambda: object.__new__(SemanticTensorContentWitness)):
    try:
        construct()
    except TypeError:
        pass
    else:
        raise AssertionError('caller constructed a witness without native capture')
try:
    SemanticTensorContentWitness.verify(object(), tensors=(), consumer_stream=1)
except TypeError:
    pass
else:
    raise AssertionError('verification accepted a caller-built witness')
"#,
        );
    }

    #[test]
    fn published_range_keys_require_the_actual_native_parent() {
        run_python(
            r#"
assert hasattr(SemanticPublishedParent, 'range_keys'), 'accepted range roster is not exported'
try:
    SemanticPublishedParent.range_keys(object())
except TypeError:
    pass
else:
    raise AssertionError('a caller-built parent supplied the accepted range roster')
"#,
        );
    }

    #[test]
    fn controller_abort_requires_the_actual_native_controller() {
        run_python(
            r#"
assert hasattr(SemanticTransitionController, 'abort'), 'terminal session abort is not exported'
try:
    SemanticTransitionController.abort(object())
except TypeError:
    pass
else:
    raise AssertionError('a caller-built controller invoked native abort')
"#,
        );
    }

    #[cfg(feature = "semantic-policy")]
    #[test]
    fn prepared_policy_original_owners_survive_the_single_invocation_handoff() {
        Python::initialize();
        Python::attach(|py| {
            let output = pyo3::types::PyDict::new(py).into_any().unbind();
            let inputs = std::array::from_fn(|_| pyo3::types::PyDict::new(py).into_any().unbind());
            let mut retained = super::PreparedPolicyInputs {
                model_output: output.clone_ref(py),
                inputs,
                invocation_issued: false,
            };
            let pointers = retained.inputs.each_ref().map(|input| input.as_ptr());
            let (issued_output, issued_inputs) = retained.issue_originals(py).unwrap();
            assert_eq!(issued_output.as_ptr(), output.as_ptr());
            assert_eq!(
                issued_inputs.each_ref().map(|input| input.as_ptr()),
                pointers
            );
            assert_eq!(retained.model_output.as_ptr(), output.as_ptr());
            assert_eq!(
                retained.inputs.each_ref().map(|input| input.as_ptr()),
                pointers
            );
            assert!(retained.issue_originals(py).is_err());
        });
    }

    #[cfg(feature = "semantic-policy")]
    #[test]
    fn prepared_policy_binding_requires_original_step_and_content_witness() {
        run_python(
            r#"
try:
    SemanticTransitionController.bind_prepared_policy(object(), object(), step=object(), bank=0, binding=(1, bytes(32)), model_output=object(), text_logits=object(), product_support=object(), parameters=object(), component_baselines=object(), producer_witness=object(), consumer_stream=1)
except TypeError:
    pass
else:
    raise AssertionError('unissued Python objects bound an original prepared policy')
"#,
        );
    }

    #[cfg(feature = "semantic-policy")]
    #[test]
    fn policy_invocation_cannot_be_constructed_without_native_execution() {
        run_python(
            r#"
import inspect
backward = inspect.signature(SemanticPolicyInvocation.backward)
assert tuple(backward.parameters) == ('self', 'score_cotangents', 'consumer_stream')
assert backward.parameters['consumer_stream'].kind is inspect.Parameter.KEYWORD_ONLY
assert backward.parameters['consumer_stream'].default is inspect.Parameter.empty
for method in ('finish_inference', 'finish_refusal'):
    finish = inspect.signature(getattr(SemanticPolicyInvocation, method))
    assert tuple(finish.parameters) == ('self', 'consumer_stream')
    assert finish.parameters['consumer_stream'].kind is inspect.Parameter.KEYWORD_ONLY
    assert finish.parameters['consumer_stream'].default is inspect.Parameter.empty
assert inspect.isgetsetdescriptor(SemanticPolicyInvocation.refusal)
assert SemanticPolicyInvocation.refusal.__name__ == 'refusal'
try:
    SemanticPolicyInvocation()
except TypeError:
    pass
else:
    raise AssertionError('caller constructed an invocation without native execution')
"#,
        );
    }

    #[test]
    fn canonical_authority_bytes_preserve_types_boundaries_and_large_integers() {
        let (values, snapshot) = task_inputs("");
        let authority = TaskAuthority::parse(&values).unwrap();
        let exact = ColdValue::Sequence(values.clone()).canonical_bytes();
        assert_eq!(authority.canonical, exact);
        assert!(exact
            .windows(31)
            .any(|part| part == b"1267650600228229401496703205376"));
        assert!(!snapshot.canonical.is_empty());
        assert_ne!(
            ColdValue::Bool(false).canonical_bytes(),
            ColdValue::Integer("0".into()).canonical_bytes()
        );
        assert_ne!(
            ColdValue::None.canonical_bytes(),
            ColdValue::Sequence(vec![]).canonical_bytes()
        );
        assert_ne!(
            ColdValue::Text("12".into()).canonical_bytes(),
            ColdValue::Integer("12".into()).canonical_bytes()
        );
        let left = ColdValue::Sequence(vec![
            ColdValue::Text("a".into()),
            ColdValue::Text("bc".into()),
        ]);
        let right = ColdValue::Sequence(vec![
            ColdValue::Text("ab".into()),
            ColdValue::Text("c".into()),
        ]);
        assert_ne!(left.canonical_bytes(), right.canonical_bytes());
        let mut changed = values;
        let ColdValue::Sequence(nodes) = &mut changed[4] else {
            panic!("dependency sequence")
        };
        let ColdValue::Sequence(node) = &mut nodes[4] else {
            panic!("dependency record")
        };
        node[3] = ColdValue::Sequence(vec![ColdValue::Text("source".into())]);
        assert_ne!(
            authority.canonical,
            TaskAuthority::parse(&changed).unwrap().canonical
        );
    }

    #[test]
    fn task_authority_rejects_identity_only_replay_rows() {
        assert!(
            replay_transport("row = row[1]", 1 << 20, 1024).is_err(),
            "identity-only rows omit the episode, material closure and native evidence"
        );
    }

    #[test]
    fn task_authority_preserves_canonical_values_and_checks_independent_grants() {
        let (values, snapshot) = task_inputs("");
        let authority = TaskAuthority::parse(&values).unwrap();
        authority
            .bind_native_reads(&observation_roots([0, 1, 1], vec![]), &[])
            .unwrap();
        let view = authority.replay[0].identity.fields(6).unwrap()[0]
            .fields(9)
            .unwrap();
        assert_eq!(
            view[6],
            ColdValue::Integer("1267650600228229401496703205376".into())
        );
        assert_eq!(
            authority.live[0].canonical.fields(16).unwrap()[3]
                .text()
                .unwrap(),
            "original acquisition spelling"
        );
        authority.check_use("inference", &snapshot, true).unwrap();
        authority.check_use("training", &snapshot, true).unwrap();

        let (values, snapshot) = task_inputs("inputs = inputs[:7] + ((),) + inputs[8:]");
        let authority = TaskAuthority::parse(&values).unwrap();
        assert!(authority.check_use("inference", &snapshot, true).is_err());
        authority.check_use("training", &snapshot, true).unwrap();

        let (values, snapshot) = task_inputs("inputs = inputs[:6] + ((),) + inputs[7:]");
        let authority = TaskAuthority::parse(&values).unwrap();
        assert!(authority.check_use("inference", &snapshot, true).is_err());
        assert!(authority.check_use("training", &snapshot, true).is_err());
    }

    #[test]
    fn one_training_use_cannot_union_incompatible_learning_purposes() {
        let (values, snapshot) = task_inputs(
            "envelope[8]=['current-request-fast-adaptation','durable-slow-consolidation']; envelope[13]=True; envelope[14]=True\ninputs=inputs[:8]+((grant('fast')+('current-request-fast-adaptation',), grant('slow')+('durable-slow-consolidation',)),)",
        );
        let authority = TaskAuthority::parse(&values).unwrap();
        authority.check_use("inference", &snapshot, true).unwrap();
        assert!(authority.check_use("training", &snapshot, true).is_err());
    }

    #[test]
    fn cold_metadata_decoder_round_trips_archived_snapshot_and_large_identity_integers() {
        let (values, snapshot) = task_inputs("");
        let restored = ColdValue::from_canonical_bytes(&snapshot.canonical).unwrap();
        let restored_snapshot = AuthoritySnapshot::parse(&restored).unwrap();
        assert_eq!(restored_snapshot.canonical, snapshot.canonical);
        assert_eq!(
            restored_snapshot.segment_end.micros,
            snapshot.segment_end.micros
        );
        let identity = &values[3].sequence().unwrap()[0].fields(4).unwrap()[0];
        assert_eq!(
            &ColdValue::from_canonical_bytes(&identity.canonical_bytes()).unwrap(),
            identity
        );
        let primitives = ColdValue::Sequence(vec![
            ColdValue::None,
            ColdValue::Bool(true),
            ColdValue::Bool(false),
            ColdValue::Integer("0".into()),
            ColdValue::Integer("-123456789012345678901234567890".into()),
            ColdValue::Text(String::new()),
            ColdValue::Text("original π😀\ntext".into()),
            ColdValue::Sequence(Vec::new()),
        ]);
        assert_eq!(
            ColdValue::from_canonical_bytes(&primitives.canonical_bytes()).unwrap(),
            primitives
        );
    }

    #[test]
    fn cold_metadata_decoder_refuses_truncation_trailing_bytes_and_unrecoverable_tags() {
        let (_, snapshot) = task_inputs("");
        for end in 0..snapshot.canonical.len() {
            assert!(
                ColdValue::from_canonical_bytes(&snapshot.canonical[..end]).is_err(),
                "truncation {end}"
            );
        }
        let mut trailing = snapshot.canonical.clone();
        trailing.push(0);
        assert!(ColdValue::from_canonical_bytes(&trailing).is_err());
        for body in [
            vec![1, 2],
            vec![5, 0],
            vec![255],
            vec![3, 1, 0, 0, 0, 0, 0, 0, 0, 255],
        ] {
            let mut bytes = b"xlog.cold.value.v1\0".to_vec();
            bytes.extend(body);
            assert!(ColdValue::from_canonical_bytes(&bytes).is_err());
        }
        let payload_hash =
            ColdValue::Bytes(std::sync::Arc::from(&b"retained payload"[..])).canonical_bytes();
        assert!(ColdValue::from_canonical_bytes(&payload_hash).is_err());
        let mut wrong_domain = snapshot.canonical;
        wrong_domain[0] = b'y';
        assert!(ColdValue::from_canonical_bytes(&wrong_domain).is_err());
    }

    #[test]
    fn cold_metadata_decoder_requires_canonical_decimal_integers_and_checked_bounds() {
        for integer in [
            "", "00", "01", "-0", "-01", "+1", "1.0", "1e9", " 1", "1 ", "--1", "1_0", "٠",
        ] {
            let bytes = ColdValue::Integer(integer.into()).canonical_bytes();
            assert!(
                ColdValue::from_canonical_bytes(&bytes).is_err(),
                "{integer:?}"
            );
        }
        for tag in [2, 3, 4] {
            let mut bytes = b"xlog.cold.value.v1\0".to_vec();
            bytes.push(tag);
            bytes.extend_from_slice(&u64::MAX.to_le_bytes());
            assert!(ColdValue::from_canonical_bytes(&bytes).is_err());
        }
        let mut bounded = ColdValue::None;
        for _ in 0..32 {
            bounded = ColdValue::Sequence(vec![bounded]);
        }
        ColdValue::from_canonical_bytes(&bounded.canonical_bytes()).unwrap();
        assert!(ColdValue::from_canonical_bytes(
            &ColdValue::Sequence(vec![bounded]).canonical_bytes()
        )
        .is_err());
        let mut over_budget = b"xlog.cold.value.v1\0".to_vec();
        over_budget.push(3);
        over_budget.extend_from_slice(&(16u64 * 1024 * 1024).to_le_bytes());
        assert!(ColdValue::from_canonical_bytes(&over_budget).is_err());
    }

    #[test]
    fn cold_input_budget_counts_nodes_and_utf8_bytes() {
        Python::initialize();
        Python::attach(|py| {
            let value = py
                .eval(&CString::new("('abc',)").unwrap(), None, None)
                .unwrap();
            assert!(ColdValue::read(&value, &mut 4, 0).is_err());
            let mut exact_budget = 5;
            ColdValue::read(&value, &mut exact_budget, 0).unwrap();
            assert_eq!(exact_budget, 0);
            let value = py.eval(&CString::new("'😀'").unwrap(), None, None).unwrap();
            assert!(ColdValue::read(&value, &mut 4, 0).is_err());
            ColdValue::read(&value, &mut 5, 0).unwrap();
        });
    }

    #[test]
    fn task_feedback_checks_transitive_data_and_control_dependencies() {
        for change in [
            "nodes = nodes[:3] + (('receipt','derived',('mask',),('target',),(),None,('statement',1)),) + nodes[4:]",
            "nodes = nodes[:3] + (('receipt','derived',('missing',),(),(),None,('statement',1)),) + nodes[4:]",
            "nodes = nodes[:3] + (('receipt','derived',('feedback',),(),(),None,('statement',1)),) + nodes[4:]",
            "nodes = nodes[:1] + nodes[2:]",
            "nodes = nodes[:1] + (('mask','mask',('source',),(),(),(0,2),None),) + nodes[2:]",
        ] {
            let (values, _) = task_inputs(&format!("{change}\ninputs = inputs[:4]+(nodes,)+inputs[5:]"));
            assert!(TaskAuthority::parse(&values).is_err(), "accepted invalid closure: {change}");
        }
    }

    #[test]
    fn native_observer_roots_cannot_be_omitted_from_feedback_authority() {
        let (values, _) = task_inputs("inputs=inputs[:5]+((),)+inputs[6:]");
        let authority = TaskAuthority::parse(&values).unwrap();
        authority
            .bind_native_reads(&observation_roots([0, 1, 1], vec![]), &[])
            .unwrap();
        assert!(authority
            .bind_native_reads(&observation_roots([0, 2, 2], vec![]), &[])
            .is_err());
        assert!(authority
            .bind_native_reads(&observation_roots([0, 1, 1], vec![]), &[0])
            .is_err());
        let (values, _) = task_inputs("nodes=nodes[:5]+(('observer','derived',('source',),('target',),(),None,('observer',0)),); inputs=inputs[:4]+(nodes,(),)+inputs[6:]");
        assert!(TaskAuthority::parse(&values).is_err());
        let (values, _) = task_inputs("nodes=nodes[:5]+(('observer','derived',('source',),(),(),None,None),); inputs=inputs[:4]+(nodes,)+inputs[5:]");
        assert!(TaskAuthority::parse(&values)
            .unwrap()
            .bind_native_reads(&observation_roots([0, 1, 1], vec![]), &[])
            .is_err());
    }

    fn observation_roots(
        query_records: [u32; 3],
        contributors: Vec<(u32, Option<u32>, u32)>,
    ) -> xlog_cuda::SemanticTaskObservationRoots {
        // This authority-unit input grants nothing. The native material tests
        // separately verify how the owner produces these exact coordinates.
        xlog_cuda::SemanticTaskObservationRoots {
            root_digest: xlog_cuda::Identity256::default(),
            root_extents: [0; 3],
            query_records,
            contributors,
        }
    }

    #[test]
    fn native_observation_roots_cover_preexisting_alias_contributors_without_mutation_rights() {
        let extra = "nodes += (('original','derived',('source',),(),(),None,('statement',7)), ('pro','derived',('original',),(),(),None,('support',0)), ('contra','derived',('original',),(),(),None,('support',1))); inputs=inputs[:4]+(nodes,)+inputs[5:]";
        let (values, _) = task_inputs(extra);
        let authority = TaskAuthority::parse(&values).unwrap();
        let observations = observation_roots([0, 1, 1], vec![(0, Some(7), 0), (0, Some(7), 1)]);
        authority.bind_native_reads(&observations, &[]).unwrap();
        assert!(authority.bind_native_reads(&observations, &[2]).is_err());
        let mut missing = observations.clone();
        missing.contributors.pop();
        assert!(authority.bind_native_reads(&missing, &[]).is_err());
        let (values, _) = task_inputs("");
        let incomplete = TaskAuthority::parse(&values).unwrap();
        assert!(incomplete.bind_native_reads(&observations, &[]).is_err());
        let mut derived = observations.clone();
        derived.contributors[0].1 = None;
        assert!(authority.bind_native_reads(&derived, &[]).is_err());
        let mut absent_query = observations.clone();
        absent_query.contributors[0].0 = 2;
        assert!(authority.bind_native_reads(&absent_query, &[]).is_err());
        let (values, _) = task_inputs(&format!("{extra}\nnodes=nodes[:-1]+(('contra','derived',('original',),('target',),(),None,('support',1)),); inputs=inputs[:4]+(nodes,)+inputs[5:]"));
        assert!(TaskAuthority::parse(&values).is_err());
    }

    #[test]
    fn task_use_phase_does_not_treat_admission_as_publication() {
        let (values, snapshot) = task_inputs("");
        let authority = TaskAuthority::parse(&values).unwrap();
        let mut state = TaskUseState {
            phase: TaskUsePhase::Imported,
            snapshot,
        };
        let (_, before) = task_inputs("snapshot=(1,snapshot[1],snapshot[2],snapshot[3],())");
        assert!(state
            .revalidate_activation(&authority, before.clone())
            .is_err());
        state
            .admit_segment(&authority, "inference".into(), before.clone())
            .unwrap();
        let mut stale = TaskUseState {
            phase: state.phase.clone(),
            snapshot: state.snapshot.clone(),
        };
        assert!(stale
            .admit_segment(&authority, "inference".into(), before)
            .is_err());
        let (_, after) = task_inputs(
            "snapshot=(2,('1970-01-01T00:00:06Z',6_000_000),snapshot[2],snapshot[3],())",
        );
        // Authority alone can pass. The public controller separately requires
        // the actual native FINAL lease and terminal intent, not this state.
        state.revalidate_activation(&authority, after).unwrap();
        let (_, revoked) = task_inputs(
            "snapshot=(3,('1970-01-01T00:00:07Z',7_000_000),snapshot[2],snapshot[3],('publication-revoke',))",
        );
        assert!(state.revalidate_activation(&authority, revoked).is_err());
        assert!(matches!(state.phase, TaskUsePhase::Refused));
    }

    #[test]
    fn cold_import_selection_keeps_original_ordinal_and_complete_identity() {
        let (values, _) = task_inputs("");
        let authority = TaskAuthority::parse(&values).unwrap();
        let row = authority.replay[0].clone();
        let rows = vec![row.clone(), row];
        assert_eq!(select_replay_row(&rows, &ColdValue::None).unwrap(), None);
        let selection = |index: &str, identity| {
            ColdValue::Sequence(vec![ColdValue::Integer(index.into()), identity])
        };
        assert_eq!(
            select_replay_row(&rows, &selection("1", rows[1].identity.clone())).unwrap(),
            Some(1)
        );
        assert!(select_replay_row(&rows, &selection("2", rows[0].identity.clone())).is_err());
        assert!(select_replay_row(&rows, &selection("-1", rows[0].identity.clone())).is_err());
        assert!(select_replay_row(&rows, &ColdValue::Integer("0".into())).is_err());
        let mut identity = rows[0].identity.sequence().unwrap().to_vec();
        identity[2] = ColdValue::Text("0".repeat(64));
        assert!(select_replay_row(&rows, &selection("0", ColdValue::Sequence(identity))).is_err());
    }

    #[test]
    fn cold_import_reads_never_admit_public_execution() {
        let (values, snapshot) = task_inputs("");
        let authority = TaskAuthority::parse(&values).unwrap();
        let mut state = TaskUseState {
            phase: TaskUsePhase::Importing {
                operation: "inference".into(),
                reads: false,
            },
            snapshot,
        };
        assert!(state.require_read().is_err());
        assert!(state.require_public_use().is_err());
        assert!(state.content_handoff_binding().is_err());
        state.set_import_reads(true).unwrap();
        assert!(state.require_read().is_ok());
        assert!(state.require_public_use().is_err());
        assert!(state.content_handoff_binding().is_ok());
        let (_, newer) = task_inputs("snapshot=(1,snapshot[1],snapshot[2],snapshot[3],())");
        assert!(state
            .admit_segment(&authority, "inference".into(), newer.clone())
            .is_err());
        assert!(state.revalidate_activation(&authority, newer).is_err());
        state.set_import_reads(false).unwrap();
        assert!(state.require_read().is_err());
        assert!(state.content_handoff_binding().is_err());
        state.phase = TaskUsePhase::Refused;
        assert!(state.set_import_reads(true).is_err());
        assert!(state.require_read().is_err());
        assert!(state.require_public_use().is_err());
    }

    #[test]
    fn task_issuance_never_resurrects_when_archived_native_coordinates_repeat() {
        use std::sync::{atomic::AtomicU64, Arc};
        let counter = Arc::new(AtomicU64::new(0));
        let original = super::TaskIssuance::issue(Arc::clone(&counter)).unwrap();
        original.require_current().unwrap();
        let restored = super::TaskIssuance::issue(Arc::clone(&counter)).unwrap();
        assert!(original.require_current().is_err());
        restored.require_current().unwrap();
        assert!(!original.same_as(&restored));
        assert!(restored.same_as(&restored.clone()));
        let foreign = super::TaskIssuance::issue(Arc::new(AtomicU64::new(1))).unwrap();
        assert_eq!(restored.generation, foreign.generation);
        assert!(!restored.same_as(&foreign));
        // Issuance is independent of restored native identity and epoch: no
        // archived field participates in either currentness or owner equality.
        let next = super::TaskIssuance::issue(counter).unwrap();
        assert!(original.require_current().is_err());
        assert!(restored.require_current().is_err());
        next.require_current().unwrap();
    }

    #[test]
    fn task_issuance_is_atomic_and_never_wraps() {
        use std::sync::{
            atomic::{AtomicU64, Ordering},
            Arc,
        };
        let counter = Arc::new(AtomicU64::new(0));
        let handles = (0..16)
            .map(|_| {
                let counter = Arc::clone(&counter);
                std::thread::spawn(move || super::TaskIssuance::issue(counter).unwrap())
            })
            .collect::<Vec<_>>();
        let mut issued = handles
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        issued.sort_by_key(|token| token.generation);
        assert_eq!(
            issued
                .iter()
                .map(|token| token.generation)
                .collect::<Vec<_>>(),
            (1..=16).collect::<Vec<_>>()
        );
        assert!(issued[..15]
            .iter()
            .all(|token| token.require_current().is_err()));
        issued[15].require_current().unwrap();
        let exhausted = Arc::new(AtomicU64::new(u64::MAX - 1));
        assert_eq!(
            super::TaskIssuance::issue(Arc::clone(&exhausted))
                .unwrap()
                .generation,
            u64::MAX
        );
        assert!(super::TaskIssuance::issue(Arc::clone(&exhausted)).is_err());
        assert_eq!(exhausted.load(Ordering::Acquire), u64::MAX);
    }

    #[test]
    fn shared_import_state_admits_only_its_controlled_reader_phase() {
        let (_, snapshot) = task_inputs("");
        let mut state = TaskUseState {
            phase: TaskUsePhase::Imported,
            snapshot,
        };
        state.require_read_during_import(false).unwrap();
        assert!(state.require_read_during_import(true).is_err());
        state.phase = TaskUsePhase::Segment("inference".into());
        state.require_read_during_import(false).unwrap();
        assert!(state.require_read_during_import(true).is_err());
        state.phase = TaskUsePhase::Importing {
            operation: "inference".into(),
            reads: false,
        };
        assert!(state.require_read_during_import(true).is_err());
        state.set_import_reads(true).unwrap();
        state.require_read_during_import(true).unwrap();
        assert!(state.require_read_during_import(false).is_err());
        state.phase = TaskUsePhase::Refused;
        assert!(state.require_read_during_import(false).is_err());
        assert!(state.require_read_during_import(true).is_err());
    }

    #[test]
    fn producer_release_runs_finalizer_after_unlocking_original_roster() {
        use std::sync::{Arc, Mutex};

        Python::initialize();
        Python::attach(|py| {
            let producers = Arc::new(Mutex::new(Vec::<Py<PyAny>>::new()));
            let original_roster = Arc::clone(&producers);
            let creator = std::thread::current().id();
            let check_retirement =
                pyo3::types::PyCFunction::new_closure(py, None, None, move |_args, _kwargs| {
                    let unlocked_and_empty = original_roster
                        .try_lock()
                        .map(|roster| roster.is_empty())
                        .unwrap_or(false);
                    Ok::<_, PyErr>((std::thread::current().id() == creator, unlocked_and_empty))
                })
                .unwrap();
            let globals = PyDict::new(py);
            globals
                .set_item("check_retirement", check_retirement)
                .unwrap();
            py.run(c"import weakref\nfinalized = []\nclass Producer:\n    def __del__(self):\n        finalized.append(check_retirement())\nproducer = Producer()\nreference = weakref.ref(producer)", Some(&globals), None).unwrap();
            producers
                .lock()
                .unwrap()
                .push(globals.get_item("producer").unwrap().unwrap().unbind());
            globals.del_item("producer").unwrap();
            let reference = globals.get_item("reference").unwrap().unwrap();
            assert!(
                !reference.call0().unwrap().is_none(),
                "the original roster must retain its actual Python producer"
            );

            super::release_python_producers(&producers);

            assert!(producers.lock().unwrap().is_empty());
            assert_eq!(
                globals
                    .get_item("finalized")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<(bool, bool)>>()
                    .unwrap(),
                vec![(true, true)],
                "the original finalizer must reenter its unlocked roster on the creator thread"
            );
            assert!(
                reference.call0().unwrap().is_none(),
                "producer retirement leaked the original Python owner"
            );
        });
    }

    #[cfg(feature = "semantic-policy")]
    #[test]
    fn retained_semantic_classes_allow_worker_access_and_drop_under_runtime_custody() {
        use pyo3::impl_::pyclass::{PyClassImpl, PyClassThreadChecker};
        fn worker_payload<T: PyClassImpl + Send + Sync>() -> bool
        where
            T::ThreadChecker: Send + 'static,
        {
            let checker = <T::ThreadChecker as PyClassThreadChecker<T>>::new();
            std::thread::spawn(move || {
                Python::attach(|py| {
                    <T::ThreadChecker as PyClassThreadChecker<T>>::check(&checker)
                        && <T::ThreadChecker as PyClassThreadChecker<T>>::can_drop(&checker, py)
                })
            })
            .join()
            .unwrap()
        }
        Python::initialize();
        let classes = [
            (
                "Session",
                worker_payload::<super::PySemanticTransitionSession>(),
            ),
            (
                "Controller",
                worker_payload::<super::PySemanticTransitionController>(),
            ),
            (
                "TaskUse",
                worker_payload::<super::PySemanticTransitionTaskUse>(),
            ),
            (
                "Parent",
                worker_payload::<super::PySemanticPublishedParent>(),
            ),
            (
                "Witness",
                worker_payload::<super::PySemanticTensorContentWitness>(),
            ),
            (
                "PolicyInvocation",
                worker_payload::<super::PySemanticPolicyInvocation>(),
            ),
        ];
        assert!(
            classes.iter().all(|(_, transferable)| *transferable),
            "{classes:?}"
        );
    }

    #[test]
    fn creator_thread_gate_refuses_worker_before_conversion_or_callback() {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };
        let creator = std::thread::current().id();
        super::require_creator_thread(creator).unwrap();
        let called = Arc::new(AtomicBool::new(false));
        let callback = Arc::clone(&called);
        let refused = std::thread::spawn(move || {
            super::require_creator_thread(creator)
                .map(|()| {
                    callback.store(true, Ordering::Release);
                })
                .is_err()
        })
        .join()
        .unwrap();
        assert!(refused);
        assert!(!called.load(Ordering::Acquire));
    }

    #[test]
    fn witness_consumer_stream_requires_an_explicit_producer_handoff_stream() {
        Python::initialize();
        Python::attach(|py| {
            for value in ["1", "3", "18446744073709551615"] {
                let value = py.eval(&CString::new(value).unwrap(), None, None).unwrap();
                super::parse_witness_consumer_stream(&value, &mut 128).unwrap();
            }
            for value in ["0", "2", "True", "-1", "'1'", "18446744073709551616"] {
                let value = py.eval(&CString::new(value).unwrap(), None, None).unwrap();
                assert!(super::parse_witness_consumer_stream(&value, &mut 128).is_err());
            }
        });
    }

    #[test]
    fn cold_import_finish_requires_a_closed_original_import_and_refuses_reuse() {
        let (values, original) = task_inputs("");
        let authority = TaskAuthority::parse(&values).unwrap();
        let (_, newer) = task_inputs("snapshot=(1,snapshot[1],snapshot[2],snapshot[3],())");
        for phase in [
            TaskUsePhase::Importing {
                operation: "inference".into(),
                reads: true,
            },
            TaskUsePhase::Imported,
            TaskUsePhase::Segment("inference".into()),
            TaskUsePhase::Refused,
        ] {
            let mut state = TaskUseState {
                phase,
                snapshot: original.clone(),
            };
            assert!(state.finish_import(&authority, newer.clone()).is_err());
            assert!(matches!(state.phase, TaskUsePhase::Refused));
            assert_eq!(state.snapshot.canonical, original.canonical);
            assert!(state.require_read().is_err());
            assert!(state.require_public_use().is_err());
            assert!(state.finish_import(&authority, newer.clone()).is_err());
            assert!(matches!(state.phase, TaskUsePhase::Refused));
        }
    }

    #[test]
    fn cold_import_finish_refreshes_rights_for_only_its_original_operation() {
        let (values, original) = task_inputs("");
        let authority = TaskAuthority::parse(&values).unwrap();
        for (operation, unrelated_revocation) in [
            ("inference", "training-revoke"),
            ("training", "inference-revoke"),
        ] {
            let (_, newer) = task_inputs(&format!(
                "snapshot=(1,('1970-01-01T00:00:06Z',6_000_000),snapshot[2],snapshot[3],('{unrelated_revocation}',))"
            ));
            // Import completion rechecks current rights, not a new future
            // segment. The original preregistered end remains unchanged.
            authority.check_use(operation, &newer, false).unwrap();
            assert!(authority.check_use(operation, &newer, true).is_err());
            let mut state = TaskUseState {
                phase: TaskUsePhase::Importing {
                    operation: operation.into(),
                    reads: false,
                },
                snapshot: original.clone(),
            };
            state.finish_import(&authority, newer.clone()).unwrap();
            assert!(matches!(state.phase, TaskUsePhase::Imported));
            assert_eq!(state.snapshot.canonical, newer.canonical);
            assert_eq!(state.snapshot.segment_end, original.segment_end);
            state.require_read().unwrap();
            state.require_public_use().unwrap();
            assert!(state.content_handoff_binding().is_err());
            assert!(state.finish_import(&authority, newer).is_err());
            assert!(matches!(state.phase, TaskUsePhase::Refused));
        }
    }

    #[test]
    fn cold_import_finish_refuses_stale_changed_expired_or_revoked_authority() {
        let (values, original) = task_inputs("");
        let authority = TaskAuthority::parse(&values).unwrap();
        for change in [
            "snapshot=(0,('1970-01-01T00:00:02Z',2_000_000),snapshot[2],snapshot[3],())",
            "snapshot=(1,('1970-01-01T00:00:00Z',0),snapshot[2],snapshot[3],())",
            "snapshot=(1,snapshot[1],('1970-01-01T00:00:04Z',4_000_000),snapshot[3],())",
            "snapshot=(1,snapshot[1],snapshot[2],((envelope,('1970-01-01T00:00:04Z',4_000_000),()),),())",
            "snapshot=(1,snapshot[1],snapshot[2],snapshot[3],('publication-revoke',))",
            "snapshot=(1,snapshot[1],snapshot[2],snapshot[3],('inference-revoke',))",
            "snapshot=(1,snapshot[1],snapshot[2],((envelope,snapshot[3][0][1],('live-revoke',)),),())",
            "snapshot=(1,('1970-01-01T00:00:09Z',9_000_000),snapshot[2],snapshot[3],())",
            "envelope[3]='different acquisition spelling'; snapshot=(1,*snapshot[1:])",
        ] {
            let (_, newer) = task_inputs(change);
            let mut state = TaskUseState {
                phase: TaskUsePhase::Importing { operation: "inference".into(), reads: false },
                snapshot: original.clone(),
            };
            assert!(state.finish_import(&authority, newer).is_err(), "{change}");
            assert!(matches!(state.phase, TaskUsePhase::Refused), "{change}");
            assert_eq!(state.snapshot.canonical, original.canonical, "{change}");
            assert!(state.require_read().is_err());
            assert!(state.require_public_use().is_err());
        }
    }

    #[test]
    fn controller_execution_binding_requires_the_exact_closed_phase() {
        use super::PySemanticTransitionController;
        let (_, snapshot) = task_inputs("");
        let canonical = snapshot.canonical.clone();
        let mut state = TaskUseState {
            phase: TaskUsePhase::Segment("training".into()),
            snapshot,
        };
        let binding = |state: &TaskUseState, importing| {
            PySemanticTransitionController::execution_binding(state, importing)
        };
        assert_eq!(
            binding(&state, false).unwrap(),
            ("training".into(), canonical.clone())
        );
        assert!(binding(&state, true).is_err());
        state.phase = TaskUsePhase::Importing {
            operation: "inference".into(),
            reads: false,
        };
        assert_eq!(
            binding(&state, true).unwrap(),
            ("inference".into(), canonical)
        );
        assert!(binding(&state, false).is_err());
        state.set_import_reads(true).unwrap();
        assert!(binding(&state, true).is_err());
        assert!(binding(&state, false).is_err());
        for phase in [TaskUsePhase::Imported, TaskUsePhase::Refused] {
            state.phase = phase;
            assert!(binding(&state, true).is_err());
            assert!(binding(&state, false).is_err());
        }
    }

    #[test]
    fn tensor_content_handoff_detects_authority_refresh_and_refusal() {
        let (values, snapshot) = task_inputs("");
        let authority = TaskAuthority::parse(&values).unwrap();
        let mut state = TaskUseState {
            phase: TaskUsePhase::Imported,
            snapshot,
        };
        assert!(state.content_handoff_binding().is_err());
        let (_, first) = task_inputs("snapshot=(1,snapshot[1],snapshot[2],snapshot[3],())");
        state
            .admit_segment(&authority, "inference".into(), first)
            .unwrap();
        let captured = state.content_handoff_binding().unwrap();
        assert_eq!(captured, state.content_handoff_binding().unwrap());
        let (_, refreshed) = task_inputs(
            "snapshot=(2,('1970-01-01T00:00:02Z',2_000_000),snapshot[2],snapshot[3],())",
        );
        state
            .admit_segment(&authority, "inference".into(), refreshed)
            .unwrap();
        assert_ne!(captured, state.content_handoff_binding().unwrap());
        let (_, revoked) = task_inputs(
            "snapshot=(3,('1970-01-01T00:00:03Z',3_000_000),snapshot[2],snapshot[3],('inference-revoke',))",
        );
        assert!(state
            .admit_segment(&authority, "inference".into(), revoked)
            .is_err());
        assert!(state.content_handoff_binding().is_err());
    }

    #[test]
    fn each_transition_rechecks_rights_without_changing_the_original_operation() {
        let (values, snapshot) = task_inputs("");
        let authority = TaskAuthority::parse(&values).unwrap();
        let mut state = TaskUseState {
            phase: TaskUsePhase::Imported,
            snapshot,
        };
        let (_, first) = task_inputs("snapshot=(1,snapshot[1],snapshot[2],snapshot[3],())");
        state
            .admit_segment(&authority, "inference".into(), first)
            .unwrap();
        let (_, next) = task_inputs(
            "snapshot=(2,('1970-01-01T00:00:02Z',2_000_000),snapshot[2],snapshot[3],())",
        );
        state
            .admit_segment(&authority, "inference".into(), next)
            .unwrap();
        let (_, changed) = task_inputs(
            "snapshot=(3,('1970-01-01T00:00:03Z',3_000_000),snapshot[2],snapshot[3],())",
        );
        assert!(state
            .admit_segment(&authority, "training".into(), changed)
            .is_err());
        let (_, expired) = task_inputs(
            "snapshot=(4,('1970-01-01T00:00:06Z',6_000_000),snapshot[2],snapshot[3],())",
        );
        assert!(state
            .admit_segment(&authority, "inference".into(), expired)
            .is_err());
        assert!(matches!(state.phase, TaskUsePhase::Refused));
    }

    #[test]
    fn task_authority_refuses_foreign_scope_incomplete_grants_and_bad_target_slots() {
        for change in [
            "inputs = inputs[:1]+(('foreign',)+scope[1:],)+inputs[2:]",
            "replay = ((replay[0][0], replay[0][1][:5]+(((1,1,6),),), *replay[0][2:]),); inputs = inputs[:3]+(replay,)+inputs[4:]",
            "g = grant('inference'); inputs = inputs[:7]+((g[:3]+('foreign-task',)+g[4:],),)+inputs[8:]",
        ] {
            let (values, _) = task_inputs(change);
            assert!(TaskAuthority::parse(&values).is_err(), "accepted mismatched identity: {change}");
        }
        let (values, snapshot) = task_inputs(
            "g=grant('inference'); inputs=inputs[:7]+((g[:4]+(('source',),)+g[5:],),)+inputs[8:]",
        );
        assert!(TaskAuthority::parse(&values)
            .unwrap()
            .check_use("inference", &snapshot, true)
            .is_err());
    }

    #[test]
    fn task_authority_rechecks_expiry_revocation_and_snapshot_freshness() {
        let (values, original) = task_inputs("");
        let authority = TaskAuthority::parse(&values).unwrap();
        assert!(original.newer_than(&original).is_err());
        for change in [
            "snapshot=(1,('1970-01-01T00:00:02Z',2_000_000),snapshot[2],snapshot[3],('inference-revoke',))",
            "snapshot=(1,('1970-01-01T00:00:02Z',2_000_000),snapshot[2],((envelope,snapshot[3][0][1],('live-revoke',)),),())",
            "snapshot=(1,('1970-01-01T00:00:09Z',9_000_000),snapshot[2],snapshot[3],())",
        ] {
            let (_, snapshot) = task_inputs(change);
            snapshot.newer_than(&original).unwrap();
            assert!(authority.check_use("inference", &snapshot, false).is_err());
        }
        let (_, changed_end) = task_inputs(
            "snapshot=(1,snapshot[1],snapshot[2],((envelope,('1970-01-01T00:00:04Z',4_000_000),()),),())",
        );
        assert!(changed_end.newer_than(&original).is_err());
        let (_, changed_envelope) = task_inputs("envelope[3]='different acquisition spelling'");
        assert!(authority.check_snapshot(&changed_envelope).is_err());
        let (_, at_deadline) = task_inputs(
            "snapshot=(1,snapshot[1],snapshot[2],((envelope,('1970-01-01T00:00:10Z',10_000_000),()),),())",
        );
        assert!(authority.check_snapshot(&at_deadline).is_err());
    }

    #[test]
    fn cold_task_transport_does_not_invoke_custom_conversion_handlers() {
        Python::initialize();
        Python::attach(|py| {
            let globals = pyo3::types::PyDict::new(py);
            py.run(
                &CString::new(
                    r#"
class ForeignInt(int):
    def __str__(self): raise AssertionError('int conversion executed')
class ForeignTuple(tuple):
    def __iter__(self): raise AssertionError('iteration executed')
class ForeignString(str):
    def __str__(self): raise AssertionError('string conversion executed')
values = (ForeignInt(3), ForeignTuple((1,)), ForeignString('value'), object())
"#,
                )
                .unwrap(),
                Some(&globals),
                None,
            )
            .unwrap();
            let values = globals.get_item("values").unwrap().unwrap();
            for value in values.cast::<pyo3::types::PyTuple>().unwrap() {
                let error = ColdValue::read(&value, &mut (16 * 1024 * 1024), 0).unwrap_err();
                assert!(error.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            }
        });
    }

    #[test]
    fn python_semantic_session_rejects_invalid_typed_inputs_before_cuda() {
        run_python(
            r#"
Session = SemanticTransitionSession
base = dict(predicates=[], records=[], supports=[], capacities=(3, 8, 8, 8),
            admission_limits=(64, 256, 128, 4096), device_ordinal=0,
            memory_bytes=512 * 1024 * 1024)
cases = [
    dict(predicates=[(1, 'statement', [('value', 255, 'value')], [0])]),
    dict(predicates=[(1, 'unknown', [], [])]),
    dict(records=[(1, [(0, True)], [])]),
    dict(records=[(1, [(4, 1.5)], [])]),
    dict(records=[(1, [(6, 1)], [])]),
    dict(records=[(1, [(0, -1)], [])]),
    dict(supports=[(0, 'unknown', 1, 2, 3, 4)]),
    dict(capacities=(0, 8, 8, 8)),
]
for change in cases:
    try:
        Session(**(base | change))
    except (ValueError, TypeError, OverflowError):
        pass
    else:
        raise AssertionError(f'invalid typed input was accepted: {change!r}')
"#,
        );
    }

    #[test]
    fn python_semantic_controller_requires_the_real_session_owner() {
        run_python(
            r#"
assert 'SemanticTransitionController' in globals(), 'native controller is missing'
assert 'SemanticTransitionTaskUse' in globals(), 'native task-use handle is missing'
for invalid in (None, object(), {}, 1):
    try:
        SemanticTransitionController(invalid)
    except TypeError:
        pass
    else:
        raise AssertionError('controller accepted a fabricated Session')
try:
    SemanticTransitionTaskUse()
except TypeError:
    pass
else:
    raise AssertionError('consumer minted its own task use')
"#,
        );
    }

    #[test]
    #[ignore = "requires an explicitly authorized remote CUDA run"]
    fn python_controller_binds_real_task_and_invalidates_older_uses() {
        run_python(&format!(
            "{TASK_INPUT}\n{}",
            r#"
session = SemanticTransitionSession(
    predicates=[(1, 'statement', [('left', 0, 'bit'), ('right', 0, 'bit')], [0,1])],
    records=[(1, [(0, 1), (0, 1)], []), (1, [(0, 0), (0, 1)], [])], supports=[],
    capacities=(3, 8, 8, 8), admission_limits=(64, 256, 128, 4096),
    device_ordinal=0, memory_bytes=512*1024*1024)
controller = SemanticTransitionController(session)
arguments = dict(task_ref=inputs[0], task_scope=inputs[1],
    statement_records=(0,1), allowed_support_records=(),
    task_program_source='pred rainfall(u32). rainfall(8). ?- rainfall(8). ?- rainfall(2).',
    task_query_ordinals=(0,1), task_scoring=(14,7,1,45,15,1), admissible_truth_masks=(7,7),
    live_authorities=inputs[2], replay_rows=(), max_material_bytes=1<<20, max_total_material_bytes=1<<20, max_evidence_bytes=1024, dependencies=inputs[4],
    feedback_roots=inputs[5], publication_grants=inputs[6],
    inference_grants=inputs[7], training_grants=inputs[8],
    initial_sources=(), source_mapping=(), snapshot=snapshot,
    replay_selection=None, replay_operation=None, restore_invocation=None,
    pack_policy=None, finish_invocation=None, refresh_snapshot=None)
first = controller.import_task(**arguments)
assert first.binding() == session.binding()
assert first.policy_layout() == session.policy_layout()
second = controller.import_task(**arguments)
try:
    first.binding()
except ValueError:
    pass
else:
    raise AssertionError('same-task reimport retained stale use authority')
other_controller = SemanticTransitionController(session)
third = other_controller.import_task(**arguments)
try:
    second.binding()
except ValueError:
    pass
else:
    raise AssertionError('cross-controller reimport retained stale use authority')
try:
    other_controller.acquire(third)
except RuntimeError:
    pass
else:
    raise AssertionError('cold task import was treated as an initialized native parent')
"#,
        ));
    }

    #[test]
    fn python_admission_preserves_exact_scalar_bits_and_schema_keys() {
        Python::initialize();
        Python::attach(|py| {
            let input = py
                .eval(
                    &CString::new(
                        r#"(
                            [(17, 'statement', [('left', 0, 'entity'), ('right', 0, 'entity')], [1, 0])],
                            [(17, [(0, 4294967295), (1, 18446744073709551615),
                                   (2, -2147483648), (3, -9223372036854775808),
                                   (4, 2147483648), (5, 9223372036854775808),
                                   (6, True), (7, 23)], [9, 8])],
                            [(3, 'contra', 4, 5, 6, 7)]
                        )"#,
                    )
                    .unwrap(),
                    None,
                    None,
                )
                .unwrap();
            let (predicates, records, supports) = input
                .extract::<(Vec<PredicateInput>, Vec<RecordInput>, Vec<SupportInput>)>()
                .unwrap();
            // This CPU boundary check exercises conversion only. The native
            // admission owns schema arity, role and reference validation.
            let parsed = parse_admission(py, predicates, records, supports).unwrap();
            assert_eq!(parsed.predicates[0].role, SemanticRecordRole::Statement);
            assert_eq!(parsed.predicates[0].schema.key_columns, [1, 0]);
            assert_eq!(
                parsed.predicates[0].schema.sort_labels(),
                ["entity", "entity"]
            );
            assert_eq!(parsed.records[0].predicate.0, 17);
            assert_eq!(parsed.records[0].qualifiers, [9, 8]);
            assert_eq!(
                parsed.records[0].arguments,
                [
                    SemanticArgument::U32(u32::MAX),
                    SemanticArgument::U64(u64::MAX),
                    SemanticArgument::I32(i32::MIN),
                    SemanticArgument::I64(i64::MIN),
                    SemanticArgument::F32Bits((-0.0f32).to_bits()),
                    SemanticArgument::F64Bits((-0.0f64).to_bits()),
                    SemanticArgument::Bool(true),
                    SemanticArgument::Symbol(23),
                ]
            );
            let support = &parsed.supports[0];
            assert_eq!(support.polarity, SemanticPolarity::Contra);
            assert_eq!(
                (
                    support.statement,
                    support.provenance,
                    support.source,
                    support.context,
                    support.scope
                ),
                (3, 4, 5, 6, 7)
            );
        });
    }

    #[test]
    #[ignore = "requires an explicitly authorized remote CUDA run"]
    fn python_semantic_session_layout_retains_its_admitted_owner() {
        run_python(
            r#"
base = dict(supports=[], capacities=(3, 8, 8, 8),
            admission_limits=(64, 256, 128, 4096), device_ordinal=0,
            memory_bytes=512 * 1024 * 1024)
predicates = [(1, 'statement', [('value', 0, 'entity')], [0])]
records = [(1, [(0, 7)], [])]
first = SemanticTransitionSession(predicates=predicates, records=records, **base)
first_layout = first.policy_layout()
predicates.append((2, 'statement', [('value', 0, 'entity')], [0]))
records.append((2, [(0, 11)], []))
second = SemanticTransitionSession(predicates=predicates, records=records, **base)
second_layout = second.policy_layout()
assert first.policy_layout() == first_layout
assert first.binding() == first_layout[0]
assert second.binding() == second_layout[0]
assert first_layout[0] != second_layout[0]
assert isinstance(first_layout, tuple)
assert isinstance(first_layout[0][1], bytes) and len(first_layout[0][1]) == 32
first_fields, second_fields = first_layout[4], second_layout[4]
assert isinstance(first_fields, tuple) and len(first_fields) == 18
assert first_fields[0][1] is None
assert all(field[1] == 0 for field in first_fields[1:])
assert any(a[0] != b[0] for a, b in zip(first_fields, second_fields))
for layout in (first_layout, second_layout):
    cursor = layout[3][1]
    for cardinality, null_category, embeddings, biases in layout[4]:
        assert isinstance(embeddings, tuple) and isinstance(biases, tuple)
        assert embeddings[0] == cursor
        assert embeddings[1] - embeddings[0] == 128 * (cardinality - (null_category is not None))
        assert biases[0] == embeddings[1]
        cursor = biases[1]
    assert cursor == layout[5]
del second, predicates, records
assert first.binding() == first_layout[0]
assert first.policy_layout() == first_layout
"#,
        );
    }
}
