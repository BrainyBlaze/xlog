//! Immutable cold material and its original, pre-reserved producer allocation.

use super::*;
use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use std::ffi::{c_int, c_void};

pub(super) enum CheckpointExportOrigin {
    Published(xlog_cuda::SemanticPublishedStateObservationBinding),
    #[cfg(feature = "semantic-policy")]
    Completed(SemanticPreparedStep),
}

/// The original export survives unresolved reads, callbacks and fallible Python
/// handoffs. Its cursors describe host writes, never native completion rights.
pub(super) struct OriginalCheckpointExport {
    pub(super) origin: CheckpointExportOrigin,
    pub(super) serializer: Py<PyAny>,
    pub(super) streams: Vec<u64>,
    pub(super) model_capacity: usize,
    pub(super) model_working_bytes: u64,
    pub(super) native_capacity: usize,
    pub(super) prefill_extent: usize,
    pub(super) snapshot_canonical: Option<Vec<u8>>,
    pub(super) saved_task_binding: Option<TaskCheckpointBinding>,
    pub(super) saved_phase: Option<CheckpointTaskPhase>,
    pub(super) saved_prior_snapshot: Option<Vec<u8>>,
    pub(super) saved_proposal_expense: Option<ProposalExpense>,
    #[cfg(feature = "semantic-policy")]
    pub(super) saved_completed_binding: Option<(Option<String>, Vec<u8>)>,
    #[cfg(feature = "semantic-policy")]
    pub(super) private_model_context_save:
        Option<xlog_cuda::SemanticPrivateModelContextCheckpointSave>,
    #[cfg(feature = "semantic-policy")]
    pub(super) saved_model_binding: Option<(
        xlog_cuda::SemanticPublishedIdentity,
        Option<xlog_cuda::SemanticPublishedIdentity>,
        u64,
        Identity256,
        Identity256,
    )>,
    pub(super) observation: Option<xlog_cuda::SemanticPublishedStateObservation>,
    pub(super) prefill_observation: Option<xlog_cuda::SemanticCheckpointInitialPrefillObservation>,
    pub(super) native: Option<xlog_cuda::SemanticReplayLoadedMaterial>,
    pub(super) initial_prefill: Option<xlog_cuda::SemanticReplayLoadedMaterial>,
    pub(super) model_context: Option<xlog_cuda::SemanticReplayLoadedMaterial>,
    pub(super) model_context_capacity: usize,
    pub(super) model_context_request_bound: bool,
    #[cfg(feature = "semantic-policy")]
    pub(super) model_context_custody:
        Option<Box<model_context_checkpoint::CheckpointModelContextCustody>>,
    pub(super) task: Option<EncodedTaskCheckpoint>,
    pub(super) config: Option<Vec<u8>>,
    pub(super) output: Option<xlog_cuda::SemanticReplayMaterialOutput>,
    pub(super) writer_scratch: Option<Arc<OriginalMaterialScratch>>,
    pub(super) writer: Option<Py<PySemanticMaterialWriter>>,
    pub(super) model: Option<Py<PySemanticMaterialBytes>>,
    pub(super) callback_entered: bool,
    pub(super) callback_error: Option<PyErr>,
    pub(super) phase_seal: Option<Vec<u8>>,
    pub(super) frame_cursor: usize,
    pub(super) checkpoint: Option<Py<PySemanticMaterialBytes>>,
    pub(super) metadata: Vec<xlog_cuda::SemanticReplayMaterialReservation>,
    _owner_metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

impl OriginalCheckpointExport {
    pub(super) fn new(
        _py: Python<'_>,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
        origin: CheckpointExportOrigin,
        serializer: &Bound<'_, PyAny>,
        streams: &[u64],
        model_capacity: usize,
        model_working_bytes: u64,
    ) -> PyResult<Arc<Mutex<Self>>> {
        let extent = std::mem::size_of::<Mutex<Self>>()
            .checked_add(2 * std::mem::size_of::<usize>())
            .and_then(|size| {
                size.checked_add(streams.len().checked_mul(std::mem::size_of::<u64>())?)
            })
            .and_then(|size| {
                size.checked_add(
                    3 * std::mem::size_of::<xlog_cuda::SemanticReplayMaterialReservation>(),
                )
            })
            .ok_or_else(|| invalid("checkpoint export metadata extent overflow"))?;
        let metadata = reserve_material_metadata(budget, extent)?;
        let mut original_streams = Vec::new();
        original_streams
            .try_reserve_exact(streams.len())
            .map_err(|_| invalid("checkpoint export stream allocation failed"))?;
        original_streams.extend_from_slice(streams);
        let mut receipts = Vec::new();
        receipts
            .try_reserve_exact(3)
            .map_err(|_| invalid("checkpoint metadata receipt allocation failed"))?;
        Ok(Arc::new(Mutex::new(Self {
            origin,
            serializer: serializer.clone().unbind(),
            streams: original_streams,
            model_capacity,
            model_working_bytes,
            native_capacity: 0,
            prefill_extent: 0,
            snapshot_canonical: None,
            saved_task_binding: None,
            saved_phase: None,
            saved_prior_snapshot: None,
            saved_proposal_expense: None,
            #[cfg(feature = "semantic-policy")]
            saved_completed_binding: None,
            #[cfg(feature = "semantic-policy")]
            private_model_context_save: None,
            #[cfg(feature = "semantic-policy")]
            saved_model_binding: None,
            observation: None,
            prefill_observation: None,
            native: None,
            initial_prefill: None,
            model_context: None,
            model_context_capacity: 0,
            model_context_request_bound: false,
            #[cfg(feature = "semantic-policy")]
            model_context_custody: None,
            task: None,
            config: None,
            output: None,
            writer_scratch: None,
            writer: None,
            model: None,
            callback_entered: false,
            callback_error: None,
            phase_seal: None,
            frame_cursor: 0,
            checkpoint: None,
            metadata: receipts,
            _owner_metadata: metadata,
        })))
    }

    pub(super) fn require_request(
        &self,
        py: Python<'_>,
        serializer: &Bound<'_, PyAny>,
        streams: &[u64],
        capacity: usize,
        working_bytes: u64,
    ) -> PyResult<()> {
        if !self.serializer.bind(py).is(serializer)
            || self.streams != streams
            || self.model_capacity != capacity
            || self.model_working_bytes != working_bytes
        {
            return Err(invalid(
                "checkpoint continuation must retain its original serializer and bounds",
            ));
        }
        Ok(())
    }
}

struct OriginalMaterialBytes {
    payload: xlog_cuda::SemanticReplayLoadedMaterial,
    budget: xlog_cuda::SemanticReplayMaterialBudget,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

pub(crate) fn python_instance_extent<T: pyo3::PyTypeInfo>(py: Python<'_>) -> PyResult<usize> {
    let class = T::type_object_raw(py);
    // SAFETY: PyTypeInfo supplies the live original type object while attached
    // to Python. Its fixed instance size includes the entire Rust pyclass.
    let extent = usize::try_from(unsafe { (*class).tp_basicsize })
        .map_err(|_| invalid("material class has an invalid instance extent"))?;
    // A garbage-collected instance also has CPython's two-word GC header.
    let gc = unsafe { ffi::PyType_HasFeature(class, ffi::Py_TPFLAGS_HAVE_GC) } != 0;
    let extent = extent
        .checked_add(if gc {
            2 * std::mem::size_of::<usize>()
        } else {
            0
        })
        .ok_or_else(|| invalid("material instance metadata extent overflow"))?;
    Ok(extent)
}

pub(crate) fn python_integer_extent(py: Python<'_>) -> PyResult<usize> {
    python_instance_extent::<pyo3::types::PyInt>(py)?
        .checked_add(std::mem::size_of::<u64>() * std::mem::size_of::<u32>())
        .ok_or_else(|| invalid("original integer metadata extent overflow"))
}

pub(super) fn reserve_python_material<T: pyo3::PyTypeInfo>(
    py: Python<'_>,
    budget: &xlog_cuda::SemanticReplayMaterialBudget,
) -> PyResult<xlog_cuda::SemanticReplayMaterialReservation> {
    reserve_material_metadata(budget, python_instance_extent::<T>(py)?)
}

pub(super) fn reserve_material_metadata(
    budget: &xlog_cuda::SemanticReplayMaterialBudget,
    extent: usize,
) -> PyResult<xlog_cuda::SemanticReplayMaterialReservation> {
    budget
        .reserve_auxiliary(
            u64::try_from(extent).map_err(|_| invalid("material metadata exceeds u64"))?,
        )
        .map_err(xlog_err)
}

#[cfg(feature = "semantic-policy")]
struct OriginalColdCheckpointWriter {
    original_work: xlog_cuda::SemanticColdModelWork,
    original_region: Option<xlog_cuda::SemanticColdModelWorkRegion>,
    ordinal: usize,
    capacity: usize,
    working_bytes: u64,
    resident_read: Py<super::resident_model_memory::PySemanticResidentModelRead>,
    codec_staging_bytes: usize,
    readback: xlog_cuda::SemanticPublicationReadbackLoan,
}

#[cfg(feature = "semantic-policy")]
impl OriginalColdCheckpointWriter {
    pub(super) fn require_bounds(&self, capacity: usize, working_bytes: u64) -> PyResult<()> {
        if self.capacity != capacity || self.working_bytes != working_bytes {
            return Err(invalid(
                "checkpoint writer port changed its original material bounds",
            ));
        }
        Ok(())
    }

    pub(super) fn bounds(&self) -> (usize, u64) {
        (self.capacity, self.working_bytes)
    }

    fn require_callback(
        &self,
        py: Python<'_>,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
        work: &learning_phase::PySemanticColdModelWork,
    ) -> PyResult<learning_phase::cold_model_work::ColdModelWorkCustody> {
        work.check(py)?;
        let custody = work.custody(py)?;
        if custody.region.as_ref().map_or(0, |region| region.ordinal()) != self.ordinal {
            return Err(invalid(
                "checkpoint writer entered another original report region",
            ));
        }
        let reader = custody.reader.borrow(py);
        let session = reader.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        owner
            .require_cold_model_work_handoff(
                &self.original_work,
                &custody.inner,
                self.original_region.as_ref(),
                custody.region.as_ref(),
            )
            .map_err(xlog_err)?;
        if !budget.same_pool(
            &reader
                .task_use
                .borrow(py)
                .checkpoint
                .material_sources
                .native_budget()?,
        ) {
            return Err(invalid(
                "checkpoint writer port changed its original byte pool",
            ));
        }
        drop(owner);
        drop(session);
        drop(reader);
        Ok(custody)
    }
}

fn reserve_material_owner(
    budget: &xlog_cuda::SemanticReplayMaterialBudget,
) -> PyResult<xlog_cuda::SemanticReplayMaterialReservation> {
    let extent = std::mem::size_of::<OriginalMaterialBytes>()
        .checked_add(2 * std::mem::size_of::<usize>())
        .ok_or_else(|| invalid("material owner metadata extent overflow"))?;
    reserve_material_metadata(budget, extent)
}

#[pyclass(name = "SemanticMaterialBytes", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticMaterialBytes {
    original: Arc<OriginalMaterialBytes>,
    range: std::ops::Range<usize>,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

impl PySemanticMaterialBytes {
    pub(super) fn from_native(
        py: Python<'_>,
        payload: xlog_cuda::SemanticReplayLoadedMaterial,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
    ) -> PyResult<Self> {
        payload.require_budget(budget).map_err(xlog_err)?;
        let owner_metadata = reserve_material_owner(budget)?;
        let view_metadata = reserve_python_material::<Self>(py, budget)?;
        let extent = payload.as_bytes().len();
        Ok(Self {
            original: Arc::new(OriginalMaterialBytes {
                payload,
                budget: budget.clone(),
                _metadata: owner_metadata,
            }),
            range: 0..extent,
            _metadata: view_metadata,
        })
    }

    pub(super) fn as_bytes(&self) -> &[u8] {
        &self.original.payload.as_bytes()[self.range.clone()]
    }

    pub(super) fn material_budget(&self) -> &xlog_cuda::SemanticReplayMaterialBudget {
        &self.original.budget
    }

    pub(super) fn original_loaded(
        &self,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
    ) -> PyResult<xlog_cuda::SemanticReplayLoadedMaterial> {
        self.original
            .payload
            .require_budget(budget)
            .map_err(xlog_err)?;
        self.original
            .payload
            .slice_payload(self.range.clone())
            .map_err(xlog_err)
    }
}

#[pymethods]
impl PySemanticMaterialBytes {
    fn __len__(&self) -> usize {
        self.as_bytes().len()
    }

    fn reserve_read_workspace(
        &self,
        py: Python<'_>,
        allocation_bytes: u64,
    ) -> PyResult<Py<material_owner::PySemanticMaterialReadWorkspace>> {
        material_owner::prepare_material_read_workspace(
            py,
            self.original_loaded(&self.original.budget)?,
            &self.original.budget,
            allocation_bytes,
        )
    }

    fn slice(&self, py: Python<'_>, begin: usize, end: usize) -> PyResult<Py<Self>> {
        if begin > end || end > self.range.len() {
            return Err(invalid("material slice exceeds its original extent"));
        }
        let metadata = reserve_python_material::<Self>(py, &self.original.budget)?;
        Py::new(
            py,
            Self {
                original: Arc::clone(&self.original),
                range: (self.range.start + begin)..(self.range.start + end),
                _metadata: metadata,
            },
        )
    }

    fn range_equals(
        &self,
        py: Python<'_>,
        begin: usize,
        end: usize,
        source: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        if begin > end || end > self.range.len() {
            return Err(invalid("material comparison exceeds its original extent"));
        }
        with_readonly_material_buffer(py, source, |source| {
            Ok(self.as_bytes()[begin..end] == *source)
        })
    }

    // SAFETY: the immutable native allocation cannot be resized or mutated.
    // FillInfo retains the exporter in view.obj; every memoryview and slice
    // therefore keeps the same original allocation and byte receipt alive.
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("material buffer view is null"));
        }
        let material = slf.borrow();
        let bytes = material.as_bytes();
        let extent = isize::try_from(bytes.len())
            .map_err(|_| PyBufferError::new_err("material buffer exceeds Python extent"))?;
        let status = unsafe {
            ffi::PyBuffer_FillInfo(
                view,
                slf.as_ptr(),
                bytes.as_ptr().cast::<c_void>().cast_mut(),
                extent,
                1,
                flags,
            )
        };
        if status != 0 {
            return Err(PyErr::fetch(slf.py()));
        }
        Ok(())
    }
}

struct OriginalMaterialWrite {
    output: Option<xlog_cuda::SemanticReplayMaterialOutput>,
    comparison_target: Option<xlog_cuda::SemanticReplayLoadedMaterial>,
    checkpoint_geometry: Option<(usize, usize, usize)>,
    #[cfg(feature = "semantic-policy")]
    activation_identity: Option<xlog_cuda::SemanticColdNativeWorkActivationIdentity>,
    owner_metadata: Option<xlog_cuda::SemanticReplayMaterialReservation>,
    sealed: Option<Arc<OriginalMaterialBytes>>,
    handoff: Option<Py<PySemanticMaterialBytes>>,
    frame_owner_metadata: Option<xlog_cuda::SemanticReplayMaterialReservation>,
    frame_sealed: Option<Arc<OriginalMaterialBytes>>,
    frame_handoff: Option<Py<PySemanticMaterialBytes>>,
}

pub(super) enum CheckpointWriterOrigin {
    Published {
        parent: Py<PySemanticPublishedParent>,
        binding: xlog_cuda::SemanticPublishedStateObservationBinding,
        #[cfg(feature = "semantic-policy")]
        work: Option<Py<learning_phase::cold_model_work::PySemanticColdModelWork>>,
    },
    #[cfg(feature = "semantic-policy")]
    Completed { step: Py<PySemanticPreparedStep> },
}

impl CheckpointWriterOrigin {
    fn with_authority<T>(
        &self,
        py: Python<'_>,
        original: &Mutex<OriginalMaterialWrite>,
        operation: impl FnOnce() -> PyResult<T>,
    ) -> PyResult<T> {
        match self {
            Self::Published {
                parent,
                binding,
                #[cfg(feature = "semantic-policy")]
                work,
            } => {
                let parent_owner = parent.borrow(py);
                let session = parent_owner.session.borrow(py);
                session.require_creator()?;
                let owner = session.owner()?;
                parent_owner
                    .task_use
                    .borrow(py)
                    .require_original_identity(&owner)?;
                let lease = parent_owner.lease()?;
                owner
                    .require_published_state_observation_binding(&lease, binding)
                    .map_err(xlog_err)?;
                #[cfg(feature = "semantic-policy")]
                let current_work = session
                    .active_cold_model_work
                    .lock()
                    .map_err(|_| invalid("checkpoint writer lost its actual callback scope"))?
                    .as_ref()
                    .map(|work| work.clone_ref(py));
                #[cfg(feature = "semantic-policy")]
                if work
                    .as_ref()
                    .zip(current_work.as_ref())
                    .is_some_and(|(expected, current)| expected.as_ptr() != current.as_ptr())
                {
                    return Err(invalid(
                        "checkpoint writer changed its original callback report",
                    ));
                }
                #[cfg(feature = "semantic-policy")]
                if let Some(work) = work.as_ref().or(current_work.as_ref()) {
                    let work = work.borrow(py);
                    let custody = work.custody(py)?;
                    if custody.parent()?.as_ptr() != parent.as_ptr() {
                        return Err(invalid(
                            "checkpoint writer changed its original callback parent",
                        ));
                    }
                    let reader = custody.reader.borrow(py);
                    let source = reader.session.borrow(py);
                    source.require_creator()?;
                    if reader.session.as_ptr() == parent_owner.session.as_ptr() {
                        let identity = owner
                            .require_cold_model_work_activation(
                                &custody.inner,
                                custody.region.as_ref(),
                                &lease,
                            )
                            .map_err(xlog_err)?;
                        Self::retain_activation(original, identity)?;
                        return operation();
                    }
                    let source_owner = source.owner()?;
                    let permit = source_owner
                        .cold_model_work_activation_permit(&custody.inner, custody.region.as_ref())
                        .map_err(xlog_err)?;
                    let identity = permit
                        .identity_for_callback(&owner, &lease)
                        .map_err(xlog_err)?;
                    Self::retain_activation(original, identity)?;
                    return operation();
                }
                owner
                    .require_ordinary_checkpoint_material_output(&lease)
                    .map_err(xlog_err)?;
                operation()
            }
            #[cfg(feature = "semantic-policy")]
            Self::Completed { step } => {
                let step = step.borrow(py);
                let session = step.session.borrow(py);
                session.require_creator()?;
                let owner = session.owner()?;
                step.task_use.borrow(py).require_original_identity(&owner)?;
                owner
                    .require_prepared_checkpoint_material_output(&step.inner)
                    .map_err(xlog_err)?;
                operation()
            }
        }
    }

    #[cfg(feature = "semantic-policy")]
    fn retain_activation(
        original: &Mutex<OriginalMaterialWrite>,
        identity: xlog_cuda::SemanticColdNativeWorkActivationIdentity,
    ) -> PyResult<()> {
        let mut original = original
            .lock()
            .map_err(|_| invalid("material writer custody is poisoned"))?;
        if original
            .activation_identity
            .as_ref()
            .is_some_and(|held| !held.same_context(&identity))
        {
            return Err(invalid(
                "checkpoint writer changed its original Recording context",
            ));
        }
        if original.activation_identity.is_none() {
            original.activation_identity = Some(identity);
        }
        Ok(())
    }
}

pub(super) struct OriginalMaterialScratch {
    native: xlog_cuda::SemanticReplayMaterialScratch,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

pub(super) fn prepare_material_scratch(
    budget: &xlog_cuda::SemanticReplayMaterialBudget,
    working_bytes: u64,
) -> PyResult<Arc<OriginalMaterialScratch>> {
    let native = budget
        .prepare_material_scratch(working_bytes)
        .map_err(xlog_err)?;
    retain_material_scratch(budget, native)
}

fn retain_material_scratch(
    budget: &xlog_cuda::SemanticReplayMaterialBudget,
    native: xlog_cuda::SemanticReplayMaterialScratch,
) -> PyResult<Arc<OriginalMaterialScratch>> {
    let metadata = reserve_material_metadata(
        budget,
        std::mem::size_of::<OriginalMaterialScratch>() + 2 * std::mem::size_of::<usize>(),
    )?;
    Ok(Arc::new(OriginalMaterialScratch {
        native,
        _metadata: metadata,
    }))
}

#[pyclass(name = "SemanticMaterialWriter", module = "pyxlog._native", unsendable)]
pub(crate) struct PySemanticMaterialWriter {
    original: Mutex<OriginalMaterialWrite>,
    origin: Mutex<Option<CheckpointWriterOrigin>>,
    scratch: Mutex<Option<Arc<OriginalMaterialScratch>>>,
    working_bytes: u64,
    budget: xlog_cuda::SemanticReplayMaterialBudget,
    #[cfg(feature = "semantic-policy")]
    cold_origin: Option<OriginalColdCheckpointWriter>,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

impl PySemanticMaterialWriter {
    fn reserve_metadata(
        py: Python<'_>,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
    ) -> PyResult<xlog_cuda::SemanticReplayMaterialReservation> {
        let extent = python_instance_extent::<Self>(py)?;
        reserve_material_metadata(budget, extent)
    }
    pub(super) fn prepare_checkpoint_port(
        py: Python<'_>,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
        origin: CheckpointWriterOrigin,
        prefix_capacity: usize,
        model_capacity: usize,
        suffix_extent: usize,
        working_bytes: u64,
    ) -> PyResult<Py<Self>> {
        let owner_metadata = reserve_material_owner(budget)?;
        let frame_owner_metadata = reserve_material_owner(budget)?;
        let metadata = Self::reserve_metadata(py, budget)?;
        Py::new(
            py,
            Self {
                original: Mutex::new(OriginalMaterialWrite {
                    output: None,
                    comparison_target: None,
                    checkpoint_geometry: Some((prefix_capacity, model_capacity, suffix_extent)),
                    #[cfg(feature = "semantic-policy")]
                    activation_identity: None,
                    owner_metadata: Some(owner_metadata),
                    sealed: None,
                    handoff: None,
                    frame_owner_metadata: Some(frame_owner_metadata),
                    frame_sealed: None,
                    frame_handoff: None,
                }),
                origin: Mutex::new(Some(origin)),
                scratch: Mutex::new(None),
                working_bytes,
                budget: budget.clone(),
                #[cfg(feature = "semantic-policy")]
                cold_origin: None,
                _metadata: metadata,
            },
        )
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn prepare_original_cold_port(
        py: Python<'_>,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
        work: xlog_cuda::SemanticColdModelWork,
        region: Option<xlog_cuda::SemanticColdModelWorkRegion>,
        ordinal: usize,
        capacity: usize,
        working_bytes: u64,
        owner: &SemanticTransitionSession,
        resident_read: Py<super::resident_model_memory::PySemanticResidentModelRead>,
        codec_staging_bytes: usize,
    ) -> PyResult<Py<Self>> {
        if capacity > isize::MAX as usize || working_bytes > isize::MAX as u64 {
            return Err(invalid("checkpoint writer bounds exceed this host"));
        }
        let owner_metadata = reserve_material_owner(budget)?;
        let frame_owner_metadata = reserve_material_owner(budget)?;
        let metadata = Self::reserve_metadata(py, budget)?;
        let readback = resident_read
            .borrow(py)
            .inner()
            .prepare_checkpoint_readback(owner, &work, ordinal, working_bytes, codec_staging_bytes)
            .map_err(xlog_err)?;
        let scratch = retain_material_scratch(budget, readback.material_scratch())?;
        Py::new(
            py,
            Self {
                original: Mutex::new(OriginalMaterialWrite {
                    output: None,
                    comparison_target: None,
                    checkpoint_geometry: None,
                    activation_identity: None,
                    owner_metadata: Some(owner_metadata),
                    sealed: None,
                    handoff: None,
                    frame_owner_metadata: Some(frame_owner_metadata),
                    frame_sealed: None,
                    frame_handoff: None,
                }),
                origin: Mutex::new(None),
                scratch: Mutex::new(Some(scratch)),
                working_bytes,
                budget: budget.clone(),
                cold_origin: Some(OriginalColdCheckpointWriter {
                    original_work: work,
                    original_region: region,
                    ordinal,
                    capacity,
                    working_bytes,
                    resident_read,
                    codec_staging_bytes,
                    readback,
                }),
                _metadata: metadata,
            },
        )
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn original_cold_bounds(&self) -> PyResult<(usize, u64)> {
        self.cold_origin
            .as_ref()
            .map(OriginalColdCheckpointWriter::bounds)
            .ok_or_else(|| invalid("checkpoint writer has no original cold report binding"))
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn require_original_readback(
        &self,
        py: Python<'_>,
        read: &Py<super::resident_model_memory::PySemanticResidentModelRead>,
        extent: usize,
    ) -> PyResult<()> {
        let original = self
            .cold_origin
            .as_ref()
            .ok_or_else(|| invalid("checkpoint writer has no original cold readback"))?;
        if original.resident_read.as_ptr() != read.as_ptr()
            || original.codec_staging_bytes != extent
        {
            return Err(invalid(
                "checkpoint writer changed its original resident read or codec extent",
            ));
        }
        let _ = py;
        Ok(())
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn original_readback_loan(
        &self,
    ) -> Option<xlog_cuda::SemanticPublicationReadbackLoan> {
        self.cold_origin
            .as_ref()
            .map(|original| original.readback.clone())
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn require_cold_bounds(&self, capacity: usize, working_bytes: u64) -> PyResult<()> {
        self.cold_origin
            .as_ref()
            .ok_or_else(|| invalid("checkpoint writer has no original cold report binding"))?
            .require_bounds(capacity, working_bytes)
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn require_original_cold_port(
        &self,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
        owner: &SemanticTransitionSession,
        work: &xlog_cuda::SemanticColdModelWork,
        region: Option<&xlog_cuda::SemanticColdModelWorkRegion>,
        ordinal: usize,
        capacity: usize,
        working_bytes: u64,
    ) -> PyResult<()> {
        let original = self
            .cold_origin
            .as_ref()
            .ok_or_else(|| invalid("checkpoint writer lost its original report metadata"))?;
        if !self.budget.same_pool(budget) || original.ordinal != ordinal {
            return Err(invalid(
                "checkpoint writer changed its original report ordinal or pool",
            ));
        }
        original.require_bounds(capacity, working_bytes)?;
        owner
            .require_cold_model_work_handoff(
                &original.original_work,
                work,
                original.original_region.as_ref(),
                region,
            )
            .map_err(xlog_err)
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn bind_original_cold_callback(
        &self,
        py: Python<'_>,
        work_handle: &Py<learning_phase::PySemanticColdModelWork>,
    ) -> PyResult<()> {
        let port = self
            .cold_origin
            .as_ref()
            .ok_or_else(|| invalid("checkpoint callback requires its original cold writer"))?;
        let custody = port.require_callback(py, &self.budget, &work_handle.borrow(py))?;
        let origin = self
            .origin
            .lock()
            .map_err(|_| invalid("material writer origin is poisoned"))?;
        let Some(CheckpointWriterOrigin::Published { parent, work, .. }) = origin.as_ref() else {
            return Err(invalid(
                "cold checkpoint writer lacks its actual publication origin",
            ));
        };
        if parent.as_ptr() != custody.parent()?.as_ptr()
            || work
                .as_ref()
                .is_some_and(|original| original.as_ptr() != work_handle.as_ptr())
        {
            return Err(invalid(
                "checkpoint writer changed its original callback parent or report",
            ));
        }
        Ok(())
    }

    pub(super) fn bind_checkpoint_origin(
        &self,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
        mut supplied: CheckpointWriterOrigin,
        prefix: usize,
        capacity: usize,
        suffix: usize,
        working_bytes: u64,
    ) -> PyResult<()> {
        if !self.budget.same_pool(budget) || self.working_bytes != working_bytes {
            return Err(invalid(
                "checkpoint writer changed its original pool or scratch bound",
            ));
        }
        #[cfg(feature = "semantic-policy")]
        self.require_cold_bounds(capacity, working_bytes)?;
        #[cfg(feature = "semantic-policy")]
        if let CheckpointWriterOrigin::Published { work, .. } = &mut supplied {
            // The report retains this writer. Fresh callback lookup authenticates
            // the report without making a writer-to-report ownership cycle.
            *work = None;
        }
        let mut origin = self
            .origin
            .lock()
            .map_err(|_| invalid("material writer origin is poisoned"))?;
        let mut original = self
            .original
            .lock()
            .map_err(|_| invalid("material writer custody is poisoned"))?;
        let geometry = (prefix, capacity, suffix);
        if origin.is_some() || original.checkpoint_geometry.is_some() {
            return Err(invalid(
                "checkpoint writer cannot bind a second original publication",
            ));
        }
        if original.output.is_some() || original.sealed.is_some() {
            return Err(invalid(
                "checkpoint writer cannot bind after original output entry",
            ));
        }
        original.checkpoint_geometry = Some(geometry);
        *origin = Some(supplied);
        Ok(())
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn bind_comparison_origin(
        &self,
        py: Python<'_>,
        parent: &Py<PySemanticPublishedParent>,
        expected: xlog_cuda::SemanticReplayLoadedMaterial,
    ) -> PyResult<()> {
        expected.require_budget(&self.budget).map_err(xlog_err)?;
        let (capacity, _) = self.original_cold_bounds()?;
        if expected.as_bytes().len() > capacity {
            return Err(invalid(
                "saved model exceeds its original prepared material capacity",
            ));
        }
        let published = parent.borrow(py);
        let session = published.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        published
            .task_use
            .borrow(py)
            .require_original_identity(&owner)?;
        let lease = published.lease()?;
        let binding = owner
            .published_state_observation_binding(&*lease)
            .map_err(xlog_err)?;
        let mut origin = self
            .origin
            .lock()
            .map_err(|_| invalid("material writer origin is poisoned"))?;
        let mut original = self
            .original
            .lock()
            .map_err(|_| invalid("material writer custody is poisoned"))?;
        if let Some(target) = &original.comparison_target {
            if target.as_bytes().as_ptr() != expected.as_bytes().as_ptr()
                || target.as_bytes().len() != expected.as_bytes().len()
                || !matches!(origin.as_ref(), Some(CheckpointWriterOrigin::Published { parent: actual, .. }) if actual.as_ptr() == parent.as_ptr())
            {
                return Err(invalid(
                    "model comparison changed its original retained range or parent",
                ));
            }
            return Ok(());
        }
        if origin.is_some()
            || original.output.is_some()
            || original.checkpoint_geometry.is_some()
            || original.sealed.is_some()
        {
            return Err(invalid(
                "model comparison cannot replace another original writer operation",
            ));
        }
        original.comparison_target = Some(expected);
        *origin = Some(CheckpointWriterOrigin::Published {
            parent: parent.clone_ref(py),
            binding,
            work: None,
        });
        Ok(())
    }

    fn with_authority<T>(
        &self,
        py: Python<'_>,
        operation: impl FnOnce() -> PyResult<T>,
    ) -> PyResult<T> {
        let origin = self
            .origin
            .lock()
            .map_err(|_| invalid("material writer origin is poisoned"))?;
        if let Some(origin) = origin.as_ref() {
            #[cfg(feature = "semantic-policy")]
            if let Some(port) = &self.cold_origin {
                let CheckpointWriterOrigin::Published { parent, .. } = origin else {
                    return Err(invalid(
                        "cold checkpoint writer changed its original producer kind",
                    ));
                };
                let parent = parent.borrow(py);
                let session = parent.session.borrow(py);
                let current = session
                    .active_cold_model_work
                    .lock()
                    .map_err(|_| invalid("checkpoint writer lost its original callback scope"))?
                    .as_ref()
                    .map(|work| work.clone_ref(py))
                    .ok_or_else(|| {
                        invalid("cold checkpoint writer requires its original active callback")
                    })?;
                port.require_callback(py, &self.budget, &current.borrow(py))?;
            }
            origin.with_authority(py, &self.original, operation)
        } else {
            #[cfg(feature = "semantic-policy")]
            if self.cold_origin.is_some() {
                return Err(invalid(
                    "checkpoint writer's original callback has not activated",
                ));
            }
            operation()
        }
    }

    fn activate_original_output(&self) -> PyResult<()> {
        let mut original = self
            .original
            .lock()
            .map_err(|_| invalid("material writer custody is poisoned"))?;
        if original.output.is_some() {
            if let Some(scratch) = self
                .scratch
                .lock()
                .map_err(|_| invalid("material scratch custody is poisoned"))?
                .as_ref()
            {
                scratch.native.enter_codec_workspace().map_err(xlog_err)?;
            }
            return Ok(());
        }
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| invalid("material scratch custody is poisoned"))?;
        if scratch.is_none() {
            *scratch = Some(prepare_material_scratch(&self.budget, self.working_bytes)?);
        }
        original.output = Some(if let Some(expected) = &original.comparison_target {
            self.budget
                .prepare_material_comparison(expected)
                .map_err(xlog_err)?
        } else {
            let (prefix, model, suffix) = original
                .checkpoint_geometry
                .ok_or_else(|| invalid("material writer lost its original output geometry"))?;
            self.budget
                .prepare_framed_material_output(prefix, model, suffix)
                .map_err(xlog_err)?
        });
        scratch
            .as_ref()
            .expect("original material scratch")
            .native
            .enter_codec_workspace()
            .map_err(xlog_err)?;
        Ok(())
    }
    pub(super) fn prepare(
        py: Python<'_>,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
        capacity: usize,
        working_bytes: u64,
    ) -> PyResult<Py<Self>> {
        Self::prepare_original(py, budget, working_bytes, || {
            budget.prepare_material_output(capacity).map_err(xlog_err)
        })
    }

    pub(super) fn prepare_checkpoint(
        py: Python<'_>,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
        model_begin: usize,
        model_capacity: usize,
        suffix_extent: usize,
        working_bytes: u64,
    ) -> PyResult<Py<Self>> {
        Self::prepare_original(py, budget, working_bytes, || {
            budget
                .prepare_framed_material_output(model_begin, model_capacity, suffix_extent)
                .map_err(xlog_err)
        })
    }

    fn prepare_original(
        py: Python<'_>,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
        working_bytes: u64,
        prepare: impl FnOnce() -> PyResult<xlog_cuda::SemanticReplayMaterialOutput>,
    ) -> PyResult<Py<Self>> {
        let scratch = prepare_material_scratch(budget, working_bytes)?;
        Self::prepare_with_scratch(py, budget, scratch, prepare)
    }

    fn prepare_with_scratch(
        py: Python<'_>,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
        scratch: Arc<OriginalMaterialScratch>,
        prepare: impl FnOnce() -> PyResult<xlog_cuda::SemanticReplayMaterialOutput>,
    ) -> PyResult<Py<Self>> {
        let owner_metadata = reserve_material_owner(budget)?;
        let frame_owner_metadata = reserve_material_owner(budget)?;
        let metadata = Self::reserve_metadata(py, budget)?;
        let output = prepare()?;
        Py::new(
            py,
            Self {
                original: Mutex::new(OriginalMaterialWrite {
                    output: Some(output),
                    comparison_target: None,
                    checkpoint_geometry: None,
                    #[cfg(feature = "semantic-policy")]
                    activation_identity: None,
                    owner_metadata: Some(owner_metadata),
                    sealed: None,
                    handoff: None,
                    frame_owner_metadata: Some(frame_owner_metadata),
                    frame_sealed: None,
                    frame_handoff: None,
                }),
                origin: Mutex::new(None),
                scratch: Mutex::new(Some(scratch)),
                working_bytes: 0,
                budget: budget.clone(),
                #[cfg(feature = "semantic-policy")]
                cold_origin: None,
                _metadata: metadata,
            },
        )
    }

    pub(super) fn from_original_output(
        py: Python<'_>,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
        output: &xlog_cuda::SemanticReplayMaterialOutput,
        working_bytes: u64,
    ) -> PyResult<Py<Self>> {
        output.require_budget(budget).map_err(xlog_err)?;
        Self::prepare_original(py, budget, working_bytes, || Ok(output.clone()))
    }

    pub(super) fn from_reserved_original_output(
        py: Python<'_>,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
        output: &xlog_cuda::SemanticReplayMaterialOutput,
        scratch: Arc<OriginalMaterialScratch>,
    ) -> PyResult<Py<Self>> {
        output.require_budget(budget).map_err(xlog_err)?;
        Self::prepare_with_scratch(py, budget, scratch, || Ok(output.clone()))
    }

    pub(super) fn original_output(&self) -> PyResult<xlog_cuda::SemanticReplayMaterialOutput> {
        Ok(self
            .original
            .lock()
            .map_err(|_| invalid("material writer custody is poisoned"))?
            .output
            .as_ref()
            .ok_or_else(|| invalid("checkpoint writer requires its original admitted activation"))?
            .clone())
    }

    pub(super) fn write_frame(&self, offset: usize, bytes: &[u8]) -> PyResult<()> {
        self.original_output()?
            .write_frame(offset, bytes)
            .map_err(xlog_err)
    }

    pub(super) fn seal_frame(&self, py: Python<'_>) -> PyResult<Py<PySemanticMaterialBytes>> {
        let view_metadata = reserve_python_material::<PySemanticMaterialBytes>(py, &self.budget)?;
        let sealed = {
            let mut original = self
                .original
                .lock()
                .map_err(|_| invalid("material writer custody is poisoned"))?;
            if let Some(handoff) = &original.frame_handoff {
                return Ok(handoff.clone_ref(py));
            }
            if original.frame_sealed.is_none() {
                let payload = original
                    .output
                    .as_ref()
                    .ok_or_else(|| invalid("checkpoint writer has no original output"))?
                    .seal_frame()
                    .map_err(xlog_err)?;
                let metadata = original.frame_owner_metadata.take().ok_or_else(|| {
                    invalid("checkpoint writer lost its original frame metadata receipt")
                })?;
                original.frame_sealed = Some(Arc::new(OriginalMaterialBytes {
                    payload,
                    budget: self.budget.clone(),
                    _metadata: metadata,
                }));
            }
            Arc::clone(
                original
                    .frame_sealed
                    .as_ref()
                    .expect("retained original sealed frame"),
            )
        };
        let extent = sealed.payload.as_bytes().len();
        let result = Py::new(
            py,
            PySemanticMaterialBytes {
                original: sealed,
                range: 0..extent,
                _metadata: view_metadata,
            },
        )?;
        self.original
            .lock()
            .map_err(|_| invalid("material writer custody is poisoned"))?
            .frame_handoff = Some(result.clone_ref(py));
        Ok(result)
    }

    pub(super) fn model_extent(&self) -> PyResult<usize> {
        self.original_output()?.model_extent().map_err(xlog_err)
    }

    pub(super) fn complete_producer(&self) -> PyResult<()> {
        if self
            .original
            .lock()
            .map_err(|_| invalid("material writer custody is poisoned"))?
            .sealed
            .is_none()
        {
            return Err(invalid(
                "material producer has not sealed its original output",
            ));
        }
        let scratch = self
            .scratch
            .lock()
            .map_err(|_| invalid("material scratch custody is poisoned"))?
            .take();
        drop(scratch);
        let retired = self
            .origin
            .lock()
            .map_err(|_| invalid("material writer origin is poisoned"))?
            .take();
        drop(retired);
        Ok(())
    }

    pub(super) fn require_output(&self, output: &PySemanticMaterialBytes) -> PyResult<()> {
        let original = self
            .original
            .lock()
            .map_err(|_| invalid("material writer custody is poisoned"))?;
        if original.sealed.as_ref().is_none_or(|sealed| {
            !Arc::ptr_eq(sealed, &output.original)
                || output.range != (0..sealed.payload.as_bytes().len())
        }) {
            return Err(invalid(
                "material producer returned a different original output",
            ));
        }
        Ok(())
    }

    pub(super) fn require_sealed_material(
        &self,
        py: Python<'_>,
        payload: &Bound<'_, PyAny>,
    ) -> PyResult<xlog_cuda::SemanticReplayLoadedMaterial> {
        if !payload.is_exact_instance_of::<PySemanticMaterialBytes>() {
            return Err(invalid(
                "material producer must return its original sealed carrier",
            ));
        }
        let original = self
            .original
            .lock()
            .map_err(|_| invalid("material writer custody is poisoned"))?;
        if original
            .handoff
            .as_ref()
            .is_none_or(|handoff| !handoff.bind(py).as_any().is(payload))
        {
            return Err(invalid(
                "material producer returned a different sealed object",
            ));
        }
        drop(original);
        let material = payload.extract::<PyRef<'_, PySemanticMaterialBytes>>()?;
        self.require_output(&material)?;
        material.original_loaded(&self.budget)
    }
}

struct OriginalBorrowedBuffer<'a>(&'a mut ffi::Py_buffer);

impl Drop for OriginalBorrowedBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: successful GetBuffer initialized this original stable stack
        // view; this guard is its sole release owner.
        unsafe { ffi::PyBuffer_Release(self.0) };
    }
}

fn with_readonly_material_buffer<T>(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    read: impl FnOnce(&[u8]) -> PyResult<T>,
) -> PyResult<T> {
    let mut storage = std::mem::MaybeUninit::<ffi::Py_buffer>::zeroed();
    // SAFETY: storage remains at this stack address until ReleaseBuffer.
    // GetBuffer acquires the original exporter; no borrowed data escapes.
    if unsafe {
        ffi::PyObject_GetBuffer(source.as_ptr(), storage.as_mut_ptr(), ffi::PyBUF_CONTIG_RO)
    } != 0
    {
        return Err(PyErr::fetch(py));
    }
    // SAFETY: GetBuffer succeeded and fully initialized this view.
    let buffer = OriginalBorrowedBuffer(unsafe { storage.assume_init_mut() });
    if buffer.0.readonly == 0 || buffer.0.itemsize != 1 || buffer.0.ndim != 1 || buffer.0.len < 0 {
        return Err(PyBufferError::new_err(
            "material access requires a read-only contiguous byte buffer",
        ));
    }
    let length =
        usize::try_from(buffer.0.len).map_err(|_| invalid("material buffer exceeds this host"))?;
    let source = if length == 0 {
        &[]
    } else {
        if buffer.0.buf.is_null() {
            return Err(PyBufferError::new_err(
                "material access has no original buffer",
            ));
        }
        // SAFETY: CONTIG_RO supplies this initialized source span while the
        // original exporter remains acquired by the sole release guard.
        unsafe { std::slice::from_raw_parts(buffer.0.buf.cast::<u8>(), length) }
    };
    let result = read(source);
    drop(buffer);
    result
}

#[pymethods]
impl PySemanticMaterialWriter {
    #[cfg(feature = "semantic-policy")]
    #[getter]
    fn region_ordinal(&self) -> PyResult<usize> {
        self.cold_origin
            .as_ref()
            .map(|origin| origin.ordinal)
            .ok_or_else(|| invalid("material writer has no original cold region"))
    }

    #[cfg(feature = "semantic-policy")]
    #[getter]
    fn model_material_capacity(&self) -> PyResult<usize> {
        self.original_cold_bounds().map(|bounds| bounds.0)
    }

    #[cfg(feature = "semantic-policy")]
    #[getter]
    fn model_material_working_bytes(&self) -> PyResult<u64> {
        self.original_cold_bounds().map(|bounds| bounds.1)
    }

    fn declare_spans(
        &self,
        py: Python<'_>,
        extent: usize,
        spans: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.with_authority(py, || self.activate_original_output())?;
        if !spans.is_exact_instance_of::<PyTuple>() {
            return Err(invalid("material spans require an exact immutable tuple"));
        }
        let declared = spans.cast::<PyTuple>()?;
        let metadata_extent = declared
            .len()
            .checked_mul(std::mem::size_of::<(usize, usize)>())
            .and_then(|size| size.checked_add(std::mem::size_of::<Vec<(usize, usize)>>()))
            .ok_or_else(|| invalid("material span metadata extent overflow"))?;
        let _metadata = reserve_material_metadata(&self.budget, metadata_extent)?;
        let mut roster = Vec::new();
        roster
            .try_reserve_exact(declared.len())
            .map_err(|_| invalid("material span allocation failed"))?;
        for entry in declared.iter() {
            if !entry.is_exact_instance_of::<PyTuple>() {
                return Err(invalid("material span requires an exact immutable pair"));
            }
            let entry = entry.cast::<PyTuple>()?;
            if entry.len() != 2 {
                return Err(invalid("material span requires begin and end"));
            }
            let begin = entry.get_item(0)?;
            let end = entry.get_item(1)?;
            if !begin.is_exact_instance_of::<pyo3::types::PyInt>()
                || !end.is_exact_instance_of::<pyo3::types::PyInt>()
            {
                return Err(invalid(
                    "material span offsets require exact nonnegative integers",
                ));
            }
            roster.push((begin.extract::<usize>()?, end.extract::<usize>()?));
        }
        self.with_authority(py, || {
            self.original_output()?
                .declare_spans(extent, &roster)
                .map_err(xlog_err)
        })
    }

    fn write(&self, py: Python<'_>, offset: usize, source: &Bound<'_, PyAny>) -> PyResult<()> {
        with_readonly_material_buffer(py, source, |source| {
            self.with_authority(py, || {
                self.original_output()?
                    .write_model(offset, source)
                    .map_err(xlog_err)
            })
        })
    }

    fn seal(&self, py: Python<'_>) -> PyResult<Py<PySemanticMaterialBytes>> {
        let view_metadata = reserve_python_material::<PySemanticMaterialBytes>(py, &self.budget)?;
        let sealed = self.with_authority(py, || {
            let mut original = self
                .original
                .lock()
                .map_err(|_| invalid("material writer custody is poisoned"))?;
            if original.handoff.is_some() {
                return Err(invalid(
                    "material writer has already sealed its original output",
                ));
            }
            if original.sealed.is_none() {
                let payload = original
                    .output
                    .as_ref()
                    .ok_or_else(|| invalid("material writer has no original output"))?
                    .seal_model()
                    .map_err(xlog_err)?;
                let metadata = original.owner_metadata.take().ok_or_else(|| {
                    invalid("material writer lost its original owner metadata receipt")
                })?;
                original.sealed = Some(Arc::new(OriginalMaterialBytes {
                    payload,
                    budget: self.budget.clone(),
                    _metadata: metadata,
                }));
            }
            Ok(Arc::clone(
                original
                    .sealed
                    .as_ref()
                    .expect("retained original sealed output"),
            ))
        })?;
        let extent = sealed.payload.as_bytes().len();
        let result = Py::new(
            py,
            PySemanticMaterialBytes {
                original: sealed,
                range: 0..extent,
                _metadata: view_metadata,
            },
        )?;
        self.original
            .lock()
            .map_err(|_| invalid("material writer custody is poisoned"))?
            .handoff = Some(result.clone_ref(py));
        Ok(result)
    }
}
