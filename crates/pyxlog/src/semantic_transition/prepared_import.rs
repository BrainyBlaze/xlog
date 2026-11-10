use super::*;

#[cfg(feature = "semantic-policy")]
use super::resident_model_memory::PySemanticPreparedResidentModelMemory;
#[cfg(feature = "semantic-policy")]
use xlog_cuda::{
    SemanticPreparedSourceContent, SemanticPreparedSourceContentPorts, SemanticPublishedIdentity,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum ImportPreparationStage {
    MetadataEntering,
    Prepared,
    ExecutionEntering,
    Complete,
    CancellationEntering,
    Cancelled,
}

#[cfg(feature = "semantic-policy")]
#[pyclass(
    name = "SemanticPreparedSourceContent",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticPreparedSourceContent {
    session: Py<PySemanticTransitionSession>,
    task_use: Py<PySemanticTransitionTaskUse>,
    parent: Py<PySemanticPublishedParent>,
    pub(super) inner: SemanticPreparedSourceContent,
    consumer_stream: u64,
    ports: Mutex<Option<OriginalSourcePortCustody>>,
    port_projection: Arc<Mutex<Option<OriginalSourcePortProjection>>>,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
    _handoff_metadata: Arc<xlog_cuda::SemanticReplayMaterialReservation>,
    _port_metadata: Arc<xlog_cuda::SemanticReplayMaterialReservation>,
    pub(super) content_geometry:
        Py<super::source_content_geometry::PySemanticPreparedSourceContentGeometry>,
}

#[cfg(feature = "semantic-policy")]
struct OriginalSourcePortCustody {
    model_context: Py<PyTuple>,
    prefix: Py<PyTuple>,
    terminal_state: Py<PyAny>,
    terminal_state_allocation: Py<PyNativeTensorAllocation>,
    feedback: Py<PyTuple>,
    caches: Vec<((u64, u64), Py<PyAny>, Py<PyTuple>)>,
}

#[cfg(feature = "semantic-policy")]
struct OriginalSourceTensorExport {
    tensor: Option<DlpackManagedTensor>,
    wrapped: bool,
    capsule: Option<Py<PyAny>>,
}

#[cfg(feature = "semantic-policy")]
impl OriginalSourceTensorExport {
    fn new(tensor: DlpackManagedTensor) -> Self {
        Self {
            tensor: Some(tensor),
            wrapped: false,
            capsule: None,
        }
    }

    fn project(
        &mut self,
        py: Python<'_>,
        session: &Py<PySemanticTransitionSession>,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
    ) -> PyResult<Py<PyAny>> {
        if let Some(capsule) = &self.capsule {
            return Ok(capsule.clone_ref(py));
        }
        if !self.wrapped {
            if self
                .tensor
                .as_ref()
                .is_none_or(|tensor| tensor.as_ptr().is_null())
            {
                return Err(invalid("original SOURCE export lost its native tensor"));
            }
            let extent = std::mem::size_of::<PublishedExportContext>()
                .checked_add(std::mem::size_of::<xlog_cuda::dlpack::DLManagedTensor>())
                .ok_or_else(|| invalid("SOURCE export metadata extent overflow"))?;
            let metadata = budget
                .reserve_auxiliary(
                    u64::try_from(extent)
                        .map_err(|_| invalid("SOURCE export metadata exceeds u64"))?,
                )
                .map_err(xlog_err)?;
            let owner_thread = session.borrow(py).owner_thread;
            self.tensor = Some(retain_export_owner(
                self.tensor.take().expect("retained original SOURCE tensor"),
                PublishedExportOwner::Source {
                    _session: session.clone_ref(py),
                    _metadata: metadata,
                },
                owner_thread,
            )?);
            self.wrapped = true;
        }
        let capsule = crate::dlpack_capsule_from_retained_tensor(py, &mut self.tensor)?;
        self.capsule = Some(capsule.clone_ref(py));
        Ok(capsule)
    }
}

#[cfg(feature = "semantic-policy")]
struct OriginalSourceModelContext {
    parent: SemanticPublishedIdentity,
    model_geometry_digest: Identity256,
    semantic_root: (u64, u64, u64, Identity256, [u64; 3]),
    tensors: [OriginalSourceTensorExport; 4],
    generations: (u64, u64, u64, u64, u64),
    topology_identity: Identity256,
    table_identity: Identity256,
    capacities: (u64, u64, u64),
}

#[cfg(feature = "semantic-policy")]
struct OriginalSourceCacheProjection {
    key: (u64, u64),
    provenance: DeviceAllocationProvenance,
    tensor: OriginalSourceTensorExport,
    backing: OriginalSourceTensorExport,
    allocation: Option<Py<PyNativeTensorAllocation>>,
    allocation_pair: Option<Py<PyTuple>>,
}

#[cfg(feature = "semantic-policy")]
struct OriginalSourcePortProjection {
    context: OriginalSourceModelContext,
    prefix_identity: SemanticPublishedIdentity,
    prefix: OriginalSourceTensorExport,
    terminal: OriginalSourceTensorExport,
    terminal_provenance: DeviceAllocationProvenance,
    terminal_allocation: Option<Py<PyNativeTensorAllocation>>,
    feedback_identity: SemanticPublishedIdentity,
    feedback_schema: FeedbackSchema,
    feedback: [OriginalSourceTensorExport; 6],
    caches: Vec<OriginalSourceCacheProjection>,
    _metadata: Arc<xlog_cuda::SemanticReplayMaterialReservation>,
}

#[cfg(feature = "semantic-policy")]
struct OriginalSourceContentHandoff {
    inner: SemanticPreparedSourceContent,
    content_geometry: Py<super::source_content_geometry::PySemanticPreparedSourceContentGeometry>,
    resident_memory: Py<PySemanticPreparedResidentModelMemory>,
    consumer_stream: u64,
    native_ports: Option<SemanticPreparedSourceContentPorts>,
    projection: Arc<Mutex<Option<OriginalSourcePortProjection>>>,
    metadata: Arc<xlog_cuda::SemanticReplayMaterialReservation>,
}

#[cfg(feature = "semantic-policy")]
impl OriginalSourcePortProjection {
    fn metadata_extent(
        py: Python<'_>,
        ports: &SemanticPreparedSourceContentPorts,
    ) -> PyResult<u64> {
        let count = ports.caches.len();
        let mut extent = count
            .checked_mul(
                std::mem::size_of::<OriginalSourceCacheProjection>()
                    + std::mem::size_of::<((u64, u64), Py<PyAny>, Py<PyTuple>)>(),
            )
            .ok_or_else(|| invalid("SOURCE cache projection extent overflow"))?;
        extent = extent
            .checked_add(std::mem::size_of::<
                xlog_cuda::SemanticReplayMaterialReservation,
            >())
            .and_then(|value| value.checked_add(2 * std::mem::size_of::<usize>()))
            .ok_or_else(|| invalid("SOURCE projection receipt extent overflow"))?;
        // Python capsules can remain as used_dltensor metadata after the native
        // consumer retires. Their receipt belongs to the original recipe's
        // projection lifetime, independently of the native export deleter.
        let capsules = count
            .checked_mul(2)
            .and_then(|count| count.checked_add(12))
            .ok_or_else(|| invalid("SOURCE capsule projection extent overflow"))?;
        extent = extent
            .checked_add(
                capsules
                    .checked_mul(material_bytes::python_instance_extent::<
                        pyo3::types::PyCapsule,
                    >(py)?)
                    .ok_or_else(|| invalid("SOURCE capsule metadata extent overflow"))?,
            )
            .ok_or_else(|| invalid("SOURCE capsule metadata extent overflow"))?;
        for len in [
            8,
            3,
            5,
            3,
            2,
            2,
            5,
            2,
            3,
            2,
            3,
            5,
            3,
            6,
            xlog_cuda::SEMANTIC_FEEDBACK_SUPPORT_FIELDS.len(),
        ] {
            extent = extent
                .checked_add(python_metadata_tuple_extent(py, len)?)
                .ok_or_else(|| invalid("SOURCE tuple projection extent overflow"))?;
        }
        let per_cache = python_metadata_tuple_extent(py, 2)?
            + material_bytes::python_instance_extent::<PyNativeTensorAllocation>(py)?;
        extent = extent
            .checked_add(world::value_schema_projection_extent(
                py,
                ports.feedback.schema.statement_bytes,
            )?)
            .and_then(|value| value.checked_add(count.checked_mul(per_cache)?))
            .and_then(|value| {
                value.checked_add(
                    10 * (material_bytes::python_instance_extent::<PyBytes>(py).ok()? + 32),
                )
            })
            .and_then(|value| {
                value.checked_add(
                    36 * (material_bytes::python_instance_extent::<PyInt>(py).ok()? + 32),
                )
            })
            .and_then(|value| {
                value.checked_add(
                    material_bytes::python_instance_extent::<PyNativeTensorAllocation>(py).ok()?,
                )
            })
            .ok_or_else(|| invalid("SOURCE projection metadata extent overflow"))?;
        for field in xlog_cuda::SEMANTIC_FEEDBACK_SUPPORT_FIELDS {
            extent = extent
                .checked_add(material_bytes::python_instance_extent::<PyString>(py)? + field.len())
                .ok_or_else(|| invalid("SOURCE support field metadata extent overflow"))?;
        }
        u64::try_from(extent).map_err(|_| invalid("SOURCE projection metadata exceeds u64"))
    }

    fn from_native(
        ports: SemanticPreparedSourceContentPorts,
        metadata: xlog_cuda::SemanticReplayMaterialReservation,
    ) -> Self {
        let context = ports.model_context;
        let feedback_identity = ports.feedback.parent;
        let feedback_schema = ports.feedback.schema.clone();
        let feedback = ports
            .feedback
            .into_dlpack()
            .map(OriginalSourceTensorExport::new);
        let mut caches = Vec::with_capacity(ports.caches.len());
        for cache in ports.caches {
            caches.push(OriginalSourceCacheProjection {
                key: (cache.layout.role, cache.layout.index),
                provenance: cache.allocation,
                tensor: OriginalSourceTensorExport::new(cache.tensor),
                backing: OriginalSourceTensorExport::new(cache.allocation_tensor),
                allocation: None,
                allocation_pair: None,
            });
        }
        Self {
            context: OriginalSourceModelContext {
                parent: context.parent,
                model_geometry_digest: context.model_geometry_digest,
                semantic_root: context.semantic_root,
                tensors: [
                    context.prefix.0,
                    context.prefix.1,
                    context.ring_head,
                    context.source,
                ]
                .map(OriginalSourceTensorExport::new),
                generations: context.generations,
                topology_identity: context.topology_identity,
                table_identity: context.table_identity,
                capacities: context.capacities,
            },
            prefix_identity: ports.prefix.0,
            prefix: OriginalSourceTensorExport::new(ports.prefix.1),
            terminal: OriginalSourceTensorExport::new(ports.terminal_state),
            terminal_provenance: ports.terminal_state_allocation,
            terminal_allocation: None,
            feedback_identity,
            feedback_schema,
            feedback,
            caches,
            _metadata: Arc::new(metadata),
        }
    }

    fn project(
        &mut self,
        py: Python<'_>,
        session: &Py<PySemanticTransitionSession>,
        task_epoch: u64,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
    ) -> PyResult<OriginalSourcePortCustody> {
        let identity = |identity: SemanticPublishedIdentity| {
            (
                PyBytes::new(py, identity.instance.as_bytes()),
                identity.word,
                PyBytes::new(py, identity.logical_digest.as_bytes()),
            )
        };
        let context = &mut self.context;
        let root = context.semantic_root;
        let model_context = (
            identity(context.parent),
            (
                root.0,
                root.1,
                root.2,
                PyBytes::new(py, root.3.as_bytes()),
                (root.4[0], root.4[1], root.4[2]),
            ),
            (
                context.tensors[0].project(py, session, budget)?,
                context.tensors[1].project(py, session, budget)?,
            ),
            (
                context.tensors[2].project(py, session, budget)?,
                context.tensors[3].project(py, session, budget)?,
            ),
            context.generations,
            (
                PyBytes::new(py, context.topology_identity.as_bytes()),
                PyBytes::new(py, context.table_identity.as_bytes()),
            ),
            context.capacities,
            PyBytes::new(py, context.model_geometry_digest.as_bytes()),
        )
            .into_pyobject(py)?
            .unbind();
        let prefix = (
            identity(self.prefix_identity),
            self.prefix.project(py, session, budget)?,
        )
            .into_pyobject(py)?
            .unbind();
        let terminal_state = self.terminal.project(py, session, budget)?;
        if self.terminal_allocation.is_none() {
            self.terminal_allocation = Some(Py::new(
                py,
                PyNativeTensorAllocation {
                    provenance: self.terminal_provenance.clone(),
                    policy_layout: None,
                },
            )?);
        }
        let feedback = (
            identity(self.feedback_identity),
            feedback_schema_object(py, &self.feedback_schema)?,
            self.feedback[0].project(py, session, budget)?,
            self.feedback[1].project(py, session, budget)?,
            (
                task_epoch,
                PyTuple::new(
                    py,
                    xlog_cuda::SEMANTIC_FEEDBACK_SUPPORT_FIELDS.iter().copied(),
                )?,
                self.feedback[2].project(py, session, budget)?,
                self.feedback[3].project(py, session, budget)?,
                self.feedback[4].project(py, session, budget)?,
                self.feedback[5].project(py, session, budget)?,
            ),
        )
            .into_pyobject(py)?
            .unbind();
        let mut caches = Vec::with_capacity(self.caches.len());
        for cache in &mut self.caches {
            let tensor = cache.tensor.project(py, session, budget)?;
            if cache.allocation.is_none() {
                cache.allocation = Some(Py::new(
                    py,
                    PyNativeTensorAllocation {
                        provenance: cache.provenance.clone(),
                        policy_layout: None,
                    },
                )?);
            }
            if cache.allocation_pair.is_none() {
                cache.allocation_pair = Some(
                    (
                        cache
                            .allocation
                            .as_ref()
                            .expect("original allocation projection")
                            .clone_ref(py),
                        cache.backing.project(py, session, budget)?,
                    )
                        .into_pyobject(py)?
                        .unbind(),
                );
            }
            caches.push((
                cache.key,
                tensor,
                cache
                    .allocation_pair
                    .as_ref()
                    .expect("original allocation pair")
                    .clone_ref(py),
            ));
        }
        Ok(OriginalSourcePortCustody {
            model_context,
            prefix,
            terminal_state,
            terminal_state_allocation: self
                .terminal_allocation
                .as_ref()
                .expect("original terminal allocation")
                .clone_ref(py),
            feedback,
            caches,
        })
    }
}

#[cfg(feature = "semantic-policy")]
impl PySemanticPreparedSourceContent {
    fn activate_original(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let stream = ColdValue::read(consumer_stream, &mut 128, 0)?.unsigned()?;
        if stream != self.consumer_stream {
            return Err(invalid(
                "SOURCE activation changed its original consumer stream",
            ));
        }
        let parent = self.parent.borrow(py);
        let session = self.session.borrow(py);
        session.require_creator()?;
        let task = self.task_use.borrow(py);
        let original = task.already_issued_cold_model_work(py)?;
        let custody = original.borrow(py).custody(py)?;
        if custody.parent()?.as_ptr() != self.parent.as_ptr() {
            return Err(invalid(
                "SOURCE activation changed its original callback parent",
            ));
        }
        let reader = custody.published_reader()?.borrow(py);
        let source = reader.session.borrow(py);
        source.require_creator()?;
        let mut owner = session.owner()?;
        task.require_original_identity(&owner)?;
        let lease = parent.lease()?;
        if reader.session.as_ptr() == self.session.as_ptr() {
            owner
                .activate_prepared_source_content(
                    &lease,
                    &self.inner,
                    stream,
                    &custody.inner,
                    custody.region.as_ref(),
                )
                .map_err(xlog_err)
        } else {
            let source_owner = source.owner()?;
            let permit = source_owner
                .cold_model_work_activation_permit(&custody.inner, custody.region.as_ref())
                .map_err(xlog_err)?;
            owner
                .activate_prepared_source_content_with_permit(&lease, &self.inner, stream, &permit)
                .map_err(xlog_err)
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "original SOURCE identities and sole port custody remain distinct"
    )]
    fn from_native(
        py: Python<'_>,
        session: Py<PySemanticTransitionSession>,
        task_use: Py<PySemanticTransitionTaskUse>,
        parent: Py<PySemanticPublishedParent>,
        inner: SemanticPreparedSourceContent,
        consumer_stream: u64,
        port_projection: Arc<Mutex<Option<OriginalSourcePortProjection>>>,
        metadata: xlog_cuda::SemanticReplayMaterialReservation,
        handoff_metadata: Arc<xlog_cuda::SemanticReplayMaterialReservation>,
        port_metadata: Arc<xlog_cuda::SemanticReplayMaterialReservation>,
        content_geometry: Py<
            super::source_content_geometry::PySemanticPreparedSourceContentGeometry,
        >,
    ) -> Self {
        let _ = py;
        Self {
            session,
            task_use,
            parent,
            inner,
            consumer_stream,
            ports: Mutex::new(None),
            port_projection,
            _metadata: metadata,
            _handoff_metadata: handoff_metadata,
            _port_metadata: port_metadata,
            content_geometry,
        }
    }

    fn finish_port_projection(&self, py: Python<'_>) -> PyResult<()> {
        self.require_original(py)?;
        let mut ports = self
            .ports
            .lock()
            .map_err(|_| invalid("SOURCE port custody is poisoned"))?;
        if ports.is_some() {
            return Ok(());
        }
        let budget = self.material_budget(py)?;
        let mut projection = self
            .port_projection
            .lock()
            .map_err(|_| invalid("SOURCE port projection custody is poisoned"))?;
        let projection = projection
            .as_mut()
            .ok_or_else(|| invalid("original SOURCE port projection has retired"))?;
        *ports = Some(projection.project(
            py,
            &self.session,
            self.task_use.borrow(py).task_epoch,
            &budget,
        )?);
        Ok(())
    }

    pub(super) fn material_budget(
        &self,
        py: Python<'_>,
    ) -> PyResult<xlog_cuda::SemanticReplayMaterialBudget> {
        self.task_use
            .borrow(py)
            .checkpoint
            .material_sources
            .native_budget()
    }

    pub(super) fn require_original(&self, py: Python<'_>) -> PyResult<()> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        self.task_use.borrow(py).require_original_identity(&owner)?;
        let parent = self.parent.borrow(py);
        if parent.task_use.as_ptr() != self.task_use.as_ptr()
            || parent.session.as_ptr() != self.session.as_ptr()
        {
            return Err(invalid("SOURCE recipe changed its original task or parent"));
        }
        let lease = parent.lease()?;
        let result = owner
            .require_prepared_source_content(&lease, &self.inner)
            .map_err(xlog_err);
        result
    }

    fn retirement_receipt(
        &self,
        py: Python<'_>,
    ) -> PyResult<xlog_cuda::SemanticSourceContentRetirement> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        self.task_use.borrow(py).require_original_identity(&owner)?;
        let parent = self.parent.borrow(py);
        let lease = parent.lease()?;
        let result = owner
            .source_content_retirement(&lease, &self.inner)
            .map_err(xlog_err);
        result
    }

    fn retain_non_submission(
        &self,
        py: Python<'_>,
        target: &Py<PySemanticTransitionSession>,
        step: &SemanticPreparedStep,
        proof: &xlog_cuda::SemanticPreparedSegmentNonSubmission,
    ) -> PyResult<()> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        self.task_use.borrow(py).require_original_identity(&owner)?;
        let parent = self.parent.borrow(py);
        let lease = parent.lease()?;
        if target.as_ptr() == self.session.as_ptr() {
            owner
                .retain_source_content_non_submission(&lease, &self.inner, &owner, step, proof)
                .map_err(xlog_err)
        } else {
            let target = target.borrow(py);
            let target_owner = target.owner()?;
            owner
                .retain_source_content_non_submission(
                    &lease,
                    &self.inner,
                    &target_owner,
                    step,
                    proof,
                )
                .map_err(xlog_err)
        }
    }

    fn retire_ports(&self) {
        let ports = self
            .ports
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        let projection = self
            .port_projection
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        // Consumed tensor aliases retain their original native deleter. Only
        // cached capsules and allocation references leave this recipe here.
        drop(ports);
        drop(projection);
    }
}

#[cfg(feature = "semantic-policy")]
#[pymethods]
impl PySemanticPreparedSourceContent {
    #[pyo3(signature = (*, consumer_stream))]
    fn activate(&self, py: Python<'_>, consumer_stream: &Bound<'_, PyAny>) -> PyResult<()> {
        self.activate_original(py, consumer_stream)
    }

    fn model_context(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_original(py)?;
        let ports = self
            .ports
            .lock()
            .map_err(|_| invalid("SOURCE port custody is poisoned"))?;
        Ok(ports
            .as_ref()
            .ok_or_else(|| invalid("original SOURCE port custody has retired"))?
            .model_context
            .clone_ref(py))
    }

    fn prefix(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_original(py)?;
        let ports = self
            .ports
            .lock()
            .map_err(|_| invalid("SOURCE port custody is poisoned"))?;
        Ok(ports
            .as_ref()
            .ok_or_else(|| invalid("original SOURCE port custody has retired"))?
            .prefix
            .clone_ref(py))
    }

    fn terminal_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.require_original(py)?;
        let ports = self
            .ports
            .lock()
            .map_err(|_| invalid("SOURCE port custody is poisoned"))?;
        Ok(ports
            .as_ref()
            .ok_or_else(|| invalid("original SOURCE port custody has retired"))?
            .terminal_state
            .clone_ref(py))
    }

    fn terminal_state_allocation(&self, py: Python<'_>) -> PyResult<Py<PyNativeTensorAllocation>> {
        self.require_original(py)?;
        let ports = self
            .ports
            .lock()
            .map_err(|_| invalid("SOURCE port custody is poisoned"))?;
        Ok(ports
            .as_ref()
            .ok_or_else(|| invalid("original SOURCE port custody has retired"))?
            .terminal_state_allocation
            .clone_ref(py))
    }

    fn feedback(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_original(py)?;
        let ports = self
            .ports
            .lock()
            .map_err(|_| invalid("SOURCE port custody is poisoned"))?;
        Ok(ports
            .as_ref()
            .ok_or_else(|| invalid("original SOURCE port custody has retired"))?
            .feedback
            .clone_ref(py))
    }

    fn tensor(
        &self,
        py: Python<'_>,
        role: &Bound<'_, PyAny>,
        index: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.require_original(py)?;
        let key = (
            ColdValue::read(role, &mut 128, 0)?.unsigned()?,
            ColdValue::read(index, &mut 128, 0)?.unsigned()?,
        );
        let ports = self
            .ports
            .lock()
            .map_err(|_| invalid("SOURCE port custody is poisoned"))?;
        let (_, tensor, _) = ports
            .as_ref()
            .ok_or_else(|| invalid("original SOURCE port custody has retired"))?
            .caches
            .iter()
            .find(|(original, _, _)| *original == key)
            .ok_or_else(|| invalid("SOURCE cache port is outside its original roster"))?;
        Ok(tensor.clone_ref(py))
    }

    fn tensor_allocation(
        &self,
        py: Python<'_>,
        role: &Bound<'_, PyAny>,
        index: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        self.require_original(py)?;
        let key = (
            ColdValue::read(role, &mut 128, 0)?.unsigned()?,
            ColdValue::read(index, &mut 128, 0)?.unsigned()?,
        );
        let ports = self
            .ports
            .lock()
            .map_err(|_| invalid("SOURCE port custody is poisoned"))?;
        let (_, _, allocation) = ports
            .as_ref()
            .ok_or_else(|| invalid("original SOURCE port custody has retired"))?
            .caches
            .iter()
            .find(|(original, _, _)| *original == key)
            .ok_or_else(|| invalid("SOURCE cache allocation is outside its original roster"))?;
        Ok(allocation.clone_ref(py))
    }
}

/// Retained allocation provenance and Python producer of an original SOURCE port.
pub(super) struct OriginalImportTensorPort {
    _native_allocation: Option<DeviceAllocationProvenance>,
    _producer: Py<PyAny>,
}

impl OriginalImportTensorPort {
    pub(super) fn retain(
        py: Python<'_>,
        input: &SemanticTensorInput,
        producer: &Py<PyAny>,
    ) -> Self {
        Self::retain_geometry(
            py,
            &(
                input.layout,
                input.logical_begin,
                input.logical_end,
                input.native_allocation.clone(),
            ),
            producer,
        )
    }

    pub(super) fn retain_geometry(
        py: Python<'_>,
        geometry: &(
            SemanticTensorLayout,
            u64,
            u64,
            Option<DeviceAllocationProvenance>,
        ),
        producer: &Py<PyAny>,
    ) -> Self {
        Self {
            _native_allocation: geometry.3.clone(),
            _producer: producer.clone_ref(py),
        }
    }
}

/// The original SOURCE import suspended before its numerical invocation.
#[pyclass(name = "SemanticPreparedTaskImport", module = "pyxlog._native", frozen)]
pub(crate) struct PySemanticPreparedTaskImport {
    pub(super) session: Py<PySemanticTransitionSession>,
    pub(super) task_use: Py<PySemanticTransitionTaskUse>,
    pub(super) parent: Py<PySemanticPublishedParent>,
    controller_identity: Arc<()>,
    pub(super) ordinal: usize,
    pub(super) material: Arc<xlog_cuda::SemanticReplayMaterial>,
    original_decisions: OriginalReplayDecisions,
    pub(super) replay_row: xlog_cuda::SemanticReplayTrainingViewRow,
    expected: (String, Vec<u8>),
    prepare_invocation: Py<PyAny>,
    restore_invocation: Py<PyAny>,
    pack_policy: Py<PyAny>,
    finish_invocation: Py<PyAny>,
    refresh_snapshot: Py<PyAny>,
    stage: Mutex<ImportPreparationStage>,
    member: Mutex<Option<Py<PySemanticRetainedReplayMember>>>,
    completed: Mutex<Option<Py<PyAny>>>,
    cancelled_final_use: Mutex<Option<ReplayImportFinalization>>,
    source_ports: Mutex<Option<Vec<OriginalImportTensorPort>>>,
    source_owner: Mutex<Option<Py<PySemanticTensorOwner>>>,
    #[cfg(feature = "semantic-policy")]
    source_content: Mutex<Option<Py<PySemanticPreparedSourceContent>>>,
    #[cfg(feature = "semantic-policy")]
    source_content_handoff: Mutex<Option<OriginalSourceContentHandoff>>,
    pub(super) source_target: Arc<Mutex<RetainedSourceTarget>>,
}

impl PySemanticPreparedTaskImport {
    pub(super) fn prepare(
        controller: &PySemanticTransitionController,
        py: Python<'_>,
        task_use: &Py<PySemanticTransitionTaskUse>,
        ordinal: usize,
        material: NativeReplayBinding,
        prepare_invocation: &Bound<'_, PyAny>,
        restore_invocation: &Bound<'_, PyAny>,
        pack_policy: &Bound<'_, PyAny>,
        finish_invocation: &Bound<'_, PyAny>,
        refresh_snapshot: &Bound<'_, PyAny>,
        import: &mut ColdImportGuard<'_>,
    ) -> PyResult<Py<Self>> {
        let issued = task_use.borrow(py);
        let replay_row = controller.original_replay_import_row(py, &issued, ordinal, import)?;
        let (parent, expected) = controller.acquire_replay_import_parent(
            py,
            task_use,
            SemanticTransitionKind::Proposal,
            import,
        )?;
        let prepared = Py::new(
            py,
            Self {
                session: controller.session.clone_ref(py),
                task_use: task_use.clone_ref(py),
                parent,
                controller_identity: Arc::clone(&controller.identity),
                ordinal,
                material: Arc::new(material.material),
                original_decisions: material.original_decisions,
                replay_row: replay_row.training_view_row()?,
                expected,
                prepare_invocation: prepare_invocation.clone().unbind(),
                restore_invocation: restore_invocation.clone().unbind(),
                pack_policy: pack_policy.clone().unbind(),
                finish_invocation: finish_invocation.clone().unbind(),
                refresh_snapshot: refresh_snapshot.clone().unbind(),
                stage: Mutex::new(ImportPreparationStage::MetadataEntering),
                member: Mutex::new(None),
                completed: Mutex::new(None),
                cancelled_final_use: Mutex::new(Some(ReplayImportFinalization::default())),
                source_ports: Mutex::new(None),
                source_owner: Mutex::new(None),
                #[cfg(feature = "semantic-policy")]
                source_content: Mutex::new(None),
                #[cfg(feature = "semantic-policy")]
                source_content_handoff: Mutex::new(None),
                source_target: Arc::new(Mutex::new(RetainedSourceTarget::new())),
            },
        )?;
        {
            let mut retained = import
                .session
                .prepared_task_import
                .lock()
                .map_err(|_| invalid("prepared task import custody is poisoned"))?;
            if retained.is_some() {
                return Err(invalid("prepared import cannot replace an original owner"));
            }
            *retained = Some(prepared.clone_ref(py));
        }
        if let Some(child) = import.session.private_replay_custody()? {
            child.retain_prepared_import(py, &prepared)?;
        }
        // Retain the original issuance and reader even if the callback or its
        // Python return is lost. Preparation is not a completed import.
        import.retained_preparation = true;
        import.deferred = true;
        let original = prepared.borrow(py);
        let acquired = original.parent.borrow(py);
        // Allocate the original future domain owners only after recovery custody
        // is installed, before metadata preparation freezes their real geometry.
        import
            .session
            .owner()?
            .prepare_policy_numerical_domain_storage(&*acquired.lease()?)
            .map_err(xlog_err)?;
        {
            let _reads = ImportReadScope::enter(&issued)?;
            if !original
                .prepare_invocation
                .bind(py)
                .call1((task_use.clone_ref(py), original.parent.clone_ref(py)))?
                .is_none()
            {
                return Err(invalid(
                    "import metadata preparation must retain original ports and return None",
                ));
            }
        }
        controller.check_import_callback(py, &issued, &acquired, import, &original.expected)?;
        *original
            .stage
            .lock()
            .map_err(|_| invalid("prepared import stage is poisoned"))? =
            ImportPreparationStage::Prepared;
        drop(acquired);
        drop(original);
        Ok(prepared)
    }

    fn require_controller(
        &self,
        py: Python<'_>,
        controller: &PySemanticTransitionController,
    ) -> PyResult<()> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        if controller.session.as_ptr() != self.session.as_ptr()
            || !Arc::ptr_eq(&controller.identity, &self.controller_identity)
            || self.parent.borrow(py).task_use.as_ptr() != self.task_use.as_ptr()
        {
            return Err(invalid(
                "prepared import requires its original controller, task and SOURCE reader",
            ));
        }
        Ok(())
    }

    pub(super) fn require_original(
        &self,
        py: Python<'_>,
        controller: &PySemanticTransitionController,
    ) -> PyResult<()> {
        self.require_controller(py, controller)?;
        let session = self.session.borrow(py);
        if !session.importing.load(Ordering::Acquire) {
            return Err(invalid(
                "unexecuted preparation requires its original active SOURCE import",
            ));
        }
        let owner = session.owner()?;
        self.task_use.borrow(py).require_original_identity(&owner)?;
        Ok(())
    }

    pub(super) fn require_completed(&self) -> PyResult<()> {
        if *self
            .stage
            .lock()
            .map_err(|_| invalid("prepared import stage is poisoned"))?
            != ImportPreparationStage::Complete
        {
            return Err(invalid(
                "future historical consumer requires its genuine completed original import",
            ));
        }
        Ok(())
    }

    pub(super) fn require_ready(&self, py: Python<'_>) -> PyResult<()> {
        self.require_metadata(py)?;
        if *self
            .stage
            .lock()
            .map_err(|_| invalid("prepared import stage is poisoned"))?
            != ImportPreparationStage::Prepared
        {
            return Err(invalid(
                "original metadata preparation has not genuinely returned",
            ));
        }
        Ok(())
    }

    pub(super) fn require_cancellation_preparation(&self, py: Python<'_>) -> PyResult<()> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        if !matches!(
            *self
                .stage
                .lock()
                .map_err(|_| invalid("prepared import stage is poisoned"))?,
            ImportPreparationStage::Prepared | ImportPreparationStage::CancellationEntering
        ) || self
            .member
            .lock()
            .map_err(|_| invalid("prepared import member is poisoned"))?
            .is_some()
            || self.parent.borrow(py).task_use.as_ptr() != self.task_use.as_ptr()
            || self.parent.borrow(py).session.as_ptr() != self.session.as_ptr()
            || !Arc::ptr_eq(
                &self.task_use.borrow(py).controller,
                &self.controller_identity,
            )
            || session
                .prepared_task_import
                .lock()
                .map_err(|_| invalid("prepared import custody is poisoned"))?
                .as_ref()
                .is_none_or(|original| !std::ptr::eq(self, &*original.borrow(py)))
        {
            return Err(invalid(
                "unexecuted cancellation requires its original genuinely prepared SOURCE import",
            ));
        }
        Ok(())
    }

    pub(super) fn require_cancelled_ack(&self, py: Python<'_>) -> PyResult<()> {
        self.session.borrow(py).require_creator()?;
        let progress = self
            .cancelled_final_use
            .try_lock()
            .map_err(|_| invalid("original unexecuted cancellation is active or poisoned"))?;
        if *self
            .stage
            .lock()
            .map_err(|_| invalid("prepared import stage is poisoned"))?
            != ImportPreparationStage::Cancelled
            || progress.as_ref().is_none_or(|progress| {
                !progress.predecessor_released
                    || progress.streams.is_none()
                    || progress.native_pending
            })
            || self
                .member
                .lock()
                .map_err(|_| invalid("prepared import member is poisoned"))?
                .is_some()
            || self.parent.borrow(py).task_use.as_ptr() != self.task_use.as_ptr()
            || self.parent.borrow(py).session.as_ptr() != self.session.as_ptr()
            || !Arc::ptr_eq(
                &self.task_use.borrow(py).controller,
                &self.controller_identity,
            )
        {
            return Err(invalid(
                "unexecuted cancellation has no positive original reader-release acknowledgment",
            ));
        }
        Ok(())
    }

    pub(super) fn finish_original_cancellation(&self, py: Python<'_>) -> PyResult<()> {
        if self.require_cancelled_ack(py).is_ok() {
            return Ok(());
        }
        self.require_cancellation_preparation(py)?;
        let session = self.session.borrow(py);
        let child = session.private_replay_custody()?.ok_or_else(|| {
            invalid("unexecuted cancellation lost its original instruction child")
        })?;
        let cancellation = child.cancelled_prepared_import_proof(py, self)?;
        if self
            .source_target
            .lock()
            .map_err(|_| invalid("prepared SOURCE target custody is poisoned"))?
            .completion
            .is_some()
        {
            return Err(invalid(
                "unexecuted cancellation cannot replace actual target completion",
            ));
        }
        let mut progress = self
            .cancelled_final_use
            .try_lock()
            .map_err(|_| invalid("original unexecuted cancellation is active or poisoned"))?
            .take()
            .ok_or_else(|| invalid("original unexecuted cancellation is already active"))?;
        *self
            .stage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            ImportPreparationStage::CancellationEntering;
        let issued = self.task_use.borrow(py);
        let parent = self.parent.borrow(py);
        let mut import = ColdImportGuard {
            session: &session,
            issuance: issued.issuance.clone(),
            original_decisions: Some(Arc::clone(&self.original_decisions)),
            committed: false,
            deferred: false,
            retained_final_use: true,
            retained_preparation: true,
            native_pending: std::cell::Cell::new(false),
        };
        let result = (|| -> PyResult<()> {
            let streams = PySemanticTransitionController::finish_original_import_invocation(
                py,
                &self.task_use,
                &self.parent,
                &issued,
                self.finish_invocation.bind(py),
                &mut progress,
            )?;
            // Target final use and the one original cleanup have returned.
            // The retired owner must leave before native reader retirement;
            // real producer backings and pending release custody remain held.
            #[cfg(feature = "semantic-policy")]
            if let Some(recipe) = self
                .source_content
                .lock()
                .map_err(|_| invalid("original SOURCE recipe custody is poisoned"))?
                .as_ref()
                .map(|recipe| recipe.clone_ref(py))
            {
                recipe.borrow(py).retain_non_submission(
                    py,
                    &cancellation.0,
                    &cancellation.1,
                    &cancellation.2,
                )?;
            }
            self.retire_source_owner(py)?;
            if !progress.predecessor_released {
                let mut owner = session.owner()?;
                issued.require_original_identity(&owner)?;
                let mut lease = parent.lease()?;
                let result = owner.release(&mut lease, &streams);
                let pending = result.as_ref().is_err_and(|error| {
                    matches!(error, xlog_cuda::SemanticTransitionError::Runtime { .. })
                }) && owner
                    .published_reader_release_pending(&lease)
                    .map_err(xlog_err)?;
                drop(lease);
                drop(owner);
                PySemanticTransitionController::finish_import_native_stage(
                    py,
                    &import,
                    result,
                    pending,
                    &mut progress.native_pending,
                )?;
                parent.release_continuation_producers();
                progress.predecessor_released = true;
            }
            issued.state()?.phase = TaskUsePhase::Refused;
            session.importing.store(false, Ordering::Release);
            Ok(())
        })();
        *self
            .cancelled_final_use
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(progress);
        if result.is_ok() {
            import.committed = true;
            let retired_target = {
                let mut group = self
                    .source_target
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                (group.session.take(), group.target.take())
            };
            self.retire_source_port_custody();
            *self
                .stage
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) =
                ImportPreparationStage::Cancelled;
            drop(retired_target);
        }
        result
    }

    pub(super) fn require_future_roster(&self, py: Python<'_>) -> PyResult<()> {
        #[cfg(feature = "semantic-policy")]
        let source_content = self
            .source_content
            .lock()
            .map_err(|_| invalid("SOURCE recipe custody is poisoned"))?
            .as_ref()
            .map(|source| source.clone_ref(py))
            .ok_or_else(|| {
                invalid(
                    "historical replay preparation requires its original structural SOURCE recipe",
                )
            })?;
        #[cfg(not(feature = "semantic-policy"))]
        if self
            .source_ports
            .lock()
            .map_err(|_| invalid("SOURCE model ports are poisoned"))?
            .is_none()
            || self
                .source_owner
                .lock()
                .map_err(|_| invalid("SOURCE model owner is poisoned"))?
                .is_none()
        {
            return Err(invalid(
                "historical replay preparation requires its complete original SOURCE model ports and owner",
            ));
        }
        {
            let target = self
                .source_target
                .lock()
                .map_err(|_| invalid("prepared SOURCE target custody is poisoned"))?;
            if target.session.is_none() || target.target.is_none() || target.completion.is_some() {
                return Err(invalid(
                    "historical SOURCE preparation lost its original uncompleted target",
                ));
            }
        }
        #[cfg(feature = "semantic-policy")]
        {
            let source = source_content.borrow(py);
            if source.session.as_ptr() != self.session.as_ptr()
                || source.task_use.as_ptr() != self.task_use.as_ptr()
                || source.parent.as_ptr() != self.parent.as_ptr()
            {
                return Err(invalid("SOURCE preparation changed its original recipe"));
            }
            let session = self.session.borrow(py);
            let owner = session.owner()?;
            self.task_use.borrow(py).require_original_identity(&owner)?;
            let parent = self.parent.borrow(py);
            let lease = parent.lease()?;
            owner
                .seal_prepared_source_content(&lease, &source.inner)
                .map_err(xlog_err)?;
        }
        Ok(())
    }

    pub(super) fn expected_binding(&self) -> &(String, Vec<u8>) {
        &self.expected
    }

    pub(super) fn owns_member(
        &self,
        py: Python<'_>,
        member: &PySemanticRetainedReplayMember,
    ) -> bool {
        self.member
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .is_some_and(|original| std::ptr::eq(member, &*original.borrow(py)))
    }

    pub(super) fn require_source_port_owner(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
    ) -> PyResult<()> {
        if !matches!(
            *self
                .stage
                .lock()
                .map_err(|_| invalid("prepared import stage is poisoned"))?,
            ImportPreparationStage::MetadataEntering | ImportPreparationStage::ExecutionEntering
        ) || !std::ptr::eq(task_use, &*self.task_use.borrow(py))
            || !std::ptr::eq(parent, &*self.parent.borrow(py))
            || parent.session.as_ptr() != self.session.as_ptr()
        {
            return Err(invalid(
                "SOURCE model ports changed their original import owner or stage",
            ));
        }
        Ok(())
    }

    pub(super) fn retain_source_ports(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
        ports: Vec<OriginalImportTensorPort>,
    ) -> PyResult<()> {
        self.require_source_port_owner(py, task_use, parent)?;
        let mut original = self
            .source_ports
            .lock()
            .map_err(|_| invalid("SOURCE model ports are poisoned"))?;
        if original.is_some() {
            return Err(invalid("original SOURCE model ports were already retained"));
        }
        *original = Some(ports);
        Ok(())
    }

    pub(super) fn retain_source_owner(
        &self,
        py: Python<'_>,
        owner: Py<PySemanticTensorOwner>,
    ) -> PyResult<()> {
        let original = owner.borrow(py);
        if original.session.as_ptr() != self.session.as_ptr()
            || !matches!(&original.parent, ContentStepOwner::Published(parent) if parent.as_ptr() == self.parent.as_ptr())
            || !original.model_binding
            || !matches!(
                *self
                    .stage
                    .lock()
                    .map_err(|_| invalid("prepared import stage is poisoned"))?,
                ImportPreparationStage::ExecutionEntering
            )
            || self
                .source_ports
                .lock()
                .map_err(|_| invalid("SOURCE model ports are poisoned"))?
                .is_none()
        {
            return Err(invalid(
                "SOURCE model owner changed its original authenticated owner",
            ));
        }
        drop(original);
        let mut retained = self
            .source_owner
            .lock()
            .map_err(|_| invalid("SOURCE model owner is poisoned"))?;
        if retained.is_some() {
            return Err(invalid("original SOURCE model owner was already retained"));
        }
        *retained = Some(owner);
        Ok(())
    }

    pub(super) fn retire_source_owner(&self, py: Python<'_>) -> PyResult<()> {
        #[cfg(feature = "semantic-policy")]
        let recipe = self
            .source_content
            .lock()
            .map_err(|_| invalid("original SOURCE recipe custody is poisoned"))?
            .as_ref()
            .map(|recipe| recipe.clone_ref(py));
        #[cfg(feature = "semantic-policy")]
        let receipt = recipe
            .as_ref()
            .map(|recipe| recipe.borrow(py).retirement_receipt(py))
            .transpose()?;
        let owner = self
            .source_owner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        drop(owner);
        #[cfg(feature = "semantic-policy")]
        {
            if let Some(recipe) = recipe {
                recipe.borrow(py).retire_ports();
            }
            // No Session or port-custody lock is held while native/foreign
            // allocation owners and DLPack deleters leave the original recipe.
            drop(receipt);
        }
        #[cfg(not(feature = "semantic-policy"))]
        let _ = py;
        Ok(())
    }

    /// Called only after original Runtime cleanup and native reader releases.
    /// Executed imports additionally require final use, full comparison and
    /// TaskUse completion; unexecuted cancellation grants no import rights.
    pub(super) fn retire_source_port_custody(&self) {
        let ports = self
            .source_ports
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        let owner = self
            .source_owner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        // All retained owners leave on the creator thread without a Session,
        // group-use, final-use or roster mutex held. Cached results remain.
        drop(ports);
        drop(owner);
    }

    fn require_metadata(&self, py: Python<'_>) -> PyResult<()> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        let stage = *self
            .stage
            .lock()
            .map_err(|_| invalid("prepared import stage is poisoned"))?;
        if !matches!(
            stage,
            ImportPreparationStage::MetadataEntering | ImportPreparationStage::Prepared
        ) {
            return Err(invalid(
                "SOURCE preparation metadata belongs to the original unexecuted import",
            ));
        }
        let owner = session.owner()?;
        self.task_use.borrow(py).require_original_identity(&owner)?;
        owner
            .published_identity(&*self.parent.borrow(py).lease()?)
            .map_err(xlog_err)?;
        Ok(())
    }
}

#[pymethods]
impl PySemanticPreparedTaskImport {
    /// Retain the same original native recipe and all genuine structural ports.
    #[cfg(feature = "semantic-policy")]
    #[pyo3(signature = (*, consumer_stream, resident_memory, content_geometry))]
    fn prepare_source_content(
        &self,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
        resident_memory: Py<PySemanticPreparedResidentModelMemory>,
        content_geometry: Py<
            super::source_content_geometry::PySemanticPreparedSourceContentGeometry,
        >,
    ) -> PyResult<Py<PySemanticPreparedSourceContent>> {
        self.require_metadata(py)?;
        let geometry = content_geometry.borrow(py);
        geometry.require_original(py)?;
        if *self
            .stage
            .lock()
            .map_err(|_| invalid("prepared import stage is poisoned"))?
            != ImportPreparationStage::MetadataEntering
        {
            return Err(invalid(
                "SOURCE recipe belongs to its original metadata preparation",
            ));
        }
        let stream = ColdValue::read(consumer_stream, &mut 128, 0)?.unsigned()?;
        let resident = resident_memory.borrow(py);
        let (resident_parent, resident_inner, resident_stream) = resident.original_binding();
        if resident_parent.as_ptr() != self.parent.as_ptr() || resident_stream != stream {
            return Err(invalid(
                "SOURCE recipe changed its original resident parent or stream",
            ));
        }
        let parent = self.parent.borrow(py);
        let session = self.session.borrow(py);
        session.require_creator()?;
        geometry.require_child_feedback_baseline(py, &session)?;
        let child = session
            .private_replay_custody()?
            .ok_or_else(|| invalid("SOURCE recipe lost its original instruction child"))?;
        let (target_session, target_step) = child.original_source_target(py, self)?;
        if target_session.as_ptr() == self.session.as_ptr() {
            return Err(invalid(
                "SOURCE structural preparation requires its original distinct target owner",
            ));
        }
        let target = target_session.borrow(py);
        target.require_creator()?;
        {
            let mut original = self
                .source_target
                .lock()
                .map_err(|_| invalid("prepared SOURCE target custody is poisoned"))?;
            if original
                .session
                .as_ref()
                .is_some_and(|session| session.as_ptr() != target_session.as_ptr())
                || original
                    .target
                    .as_ref()
                    .is_some_and(|step| !step.same_handle(&target_step))
                || original.completion.is_some()
            {
                return Err(invalid("SOURCE preparation changed its original target"));
            }
            if original.session.is_none() {
                original.session = Some(target_session.clone_ref(py));
                original.target = Some(target_step.clone());
            }
        }
        // Authenticate and drain before any handoff or native owner lock. A
        // producer's final Python decref may reenter this original import.
        drop(target.owner()?);
        drop(session.owner()?);
        let target_owner = target.serialized_owner()?;
        let mut owner = session.serialized_owner()?;
        let mut handoff = self
            .source_content_handoff
            .try_lock()
            .map_err(|_| invalid("original SOURCE port handoff is already active or poisoned"))?;
        let cached = self
            .source_content
            .lock()
            .map_err(|_| invalid("SOURCE recipe custody is poisoned"))?
            .as_ref()
            .map(|original| original.clone_ref(py));
        if cached.is_some() && handoff.is_none() {
            return Err(invalid("original SOURCE recipe was already issued"));
        }
        if let Some(original) = handoff.as_ref() {
            if original.consumer_stream != stream
                || original.content_geometry.as_ptr() != content_geometry.as_ptr()
                || original.resident_memory.as_ptr() != resident_memory.as_ptr()
            {
                return Err(invalid(
                    "SOURCE port handoff changed its original geometry, resident owner or stream",
                ));
            }
        }
        self.task_use.borrow(py).require_original_identity(&owner)?;
        let lease = parent.lease()?;
        let budget = self
            .task_use
            .borrow(py)
            .checkpoint
            .material_sources
            .native_budget()?;
        if handoff.is_none() {
            let extent = std::mem::size_of::<OriginalSourceContentHandoff>()
                .checked_add(std::mem::size_of::<
                    Mutex<Option<OriginalSourcePortProjection>>,
                >())
                .and_then(|extent| extent.checked_add(4 * std::mem::size_of::<usize>()))
                .and_then(|extent| {
                    extent.checked_add(std::mem::size_of::<
                        xlog_cuda::SemanticReplayMaterialReservation,
                    >())
                })
                .ok_or_else(|| invalid("SOURCE handoff metadata extent overflow"))?;
            let metadata = budget
                .reserve_auxiliary(
                    u64::try_from(extent)
                        .map_err(|_| invalid("SOURCE handoff metadata exceeds u64"))?,
                )
                .map_err(xlog_err)?;
            let inner = owner
                .prepare_source_content(
                    &lease,
                    resident_inner,
                    &self.material,
                    u64::try_from(self.ordinal)
                        .map_err(|_| invalid("SOURCE ordinal exceeds native range"))?,
                    &self.replay_row,
                    stream,
                    &target_owner,
                    &target_step,
                    geometry.native(),
                )
                .map_err(xlog_err)?;
            // The original native recipe is retained before its first fallible
            // alias or Python handoff. A retry never enters the factory again.
            *handoff = Some(OriginalSourceContentHandoff {
                inner,
                content_geometry: content_geometry.clone_ref(py),
                resident_memory: resident_memory.clone_ref(py),
                consumer_stream: stream,
                native_ports: None,
                projection: Arc::new(Mutex::new(None)),
                metadata: Arc::new(metadata),
            });
        }
        let original = handoff.as_mut().expect("retained original SOURCE handoff");
        owner
            .require_prepared_source_content(&lease, &original.inner)
            .map_err(xlog_err)?;
        let mut projection = original
            .projection
            .lock()
            .map_err(|_| invalid("SOURCE port projection custody is poisoned"))?;
        if projection.is_none() {
            if original.native_ports.is_none() {
                original.native_ports = Some(
                    owner
                        .prepared_source_content_ports(&lease, &original.inner)
                        .map_err(xlog_err)?,
                );
            }
            let ports = original
                .native_ports
                .as_ref()
                .expect("retained original native SOURCE ports");
            for (index, cache) in ports.caches.iter().enumerate() {
                if ports.caches[..index].iter().any(|previous| {
                    previous.layout.role == cache.layout.role
                        && previous.layout.index == cache.layout.index
                }) {
                    return Err(invalid("original SOURCE cache roster repeats a port"));
                }
            }
            let metadata = budget
                .reserve_auxiliary(OriginalSourcePortProjection::metadata_extent(py, ports)?)
                .map_err(xlog_err)?;
            *projection = Some(OriginalSourcePortProjection::from_native(
                original
                    .native_ports
                    .take()
                    .expect("retained original native SOURCE ports"),
                metadata,
            ));
        }
        let port_metadata = Arc::clone(
            &projection
                .as_ref()
                .expect("retained original SOURCE projection")
                ._metadata,
        );
        drop(projection);
        drop(lease);
        drop(owner);
        drop(target_owner);
        drop(target);
        drop(session);
        drop(parent);
        drop(resident);
        drop(geometry);
        let prepared = if let Some(prepared) = cached {
            prepared
        } else {
            let extent =
                material_bytes::python_instance_extent::<PySemanticPreparedSourceContent>(py)?;
            let metadata = budget
                .reserve_auxiliary(
                    u64::try_from(extent)
                        .map_err(|_| invalid("SOURCE recipe Python metadata exceeds u64"))?,
                )
                .map_err(xlog_err)?;
            let prepared = Py::new(
                py,
                PySemanticPreparedSourceContent::from_native(
                    py,
                    self.session.clone_ref(py),
                    self.task_use.clone_ref(py),
                    self.parent.clone_ref(py),
                    original.inner.clone(),
                    stream,
                    Arc::clone(&original.projection),
                    metadata,
                    Arc::clone(&original.metadata),
                    port_metadata,
                    content_geometry,
                ),
            )?;
            *self
                .source_content
                .lock()
                .map_err(|_| invalid("SOURCE recipe custody is poisoned"))? =
                Some(prepared.clone_ref(py));
            prepared
        };
        drop(handoff);
        prepared.borrow(py).finish_port_projection(py)?;
        let mut handoff = self
            .source_content_handoff
            .try_lock()
            .map_err(|_| invalid("original SOURCE port handoff is already active or poisoned"))?;
        let retired_handoff = handoff.take();
        drop(handoff);
        drop(retired_handoff);
        Ok(prepared)
    }

    #[getter]
    fn task_use(&self, py: Python<'_>) -> PyResult<Py<PySemanticTransitionTaskUse>> {
        self.session.borrow(py).require_creator()?;
        Ok(self.task_use.clone_ref(py))
    }

    #[getter]
    fn parent(&self, py: Python<'_>) -> PyResult<Py<PySemanticPublishedParent>> {
        self.session.borrow(py).require_creator()?;
        Ok(self.parent.clone_ref(py))
    }

    /// Acknowledge the original reader-only cancellation performed by native retirement.
    fn finish_cancelled_private_segment(&self, py: Python<'_>) -> PyResult<()> {
        self.require_cancelled_ack(py)
    }

    fn source_model_contract_metadata(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_metadata(py)?;
        let (bytes, layout) = self
            .material
            .source_model_contract_metadata()
            .map_err(xlog_err)?;
        Ok((
            PyBytes::new(py, bytes),
            (
                layout.schema_begin,
                layout.schema_bytes,
                layout.schema_digest_offset,
                layout.generation_offset,
                layout.identity_offset,
            ),
        )
            .into_pyobject(py)?
            .unbind())
    }

    fn source_model_memory_geometry(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_metadata(py)?;
        let (allocations, storages, views, layouts) = self.material.source_model_memory_geometry();
        let layouts = layouts
            .into_iter()
            .map(|layout| ((layout.role, layout.index), layout))
            .collect::<BTreeMap<_, _>>();
        let storages = PyTuple::new(
            py,
            storages
                .into_iter()
                .map(|row| (row.allocation, row.byte_offset, row.span_bytes)),
        )?;
        let views = views
            .into_iter()
            .map(|view| -> PyResult<_> {
                let layout = layouts.get(&(view.role, view.index)).ok_or_else(|| {
                    invalid("SOURCE model metadata lost its original view layout")
                })?;
                Ok((
                    view.role,
                    view.index,
                    view.storage,
                    view.byte_offset,
                    (
                        layout.element_bytes,
                        layout.scalar_type,
                        layout.rank,
                        layout.logical_axis,
                        PyTuple::new(py, layout.dimensions)?,
                        PyTuple::new(py, layout.strides_bytes)?,
                    ),
                )
                    .into_pyobject(py)?
                    .into_any())
            })
            .collect::<PyResult<Vec<_>>>()?;
        let views = PyTuple::new(py, views)?;
        Ok((PyTuple::new(py, allocations)?, storages, views)
            .into_pyobject(py)?
            .unbind())
    }
}

#[pymethods]
impl PySemanticTransitionController {
    fn execute_task_import(
        &self,
        py: Python<'_>,
        prepared_import: Py<PySemanticPreparedTaskImport>,
    ) -> PyResult<Py<PyAny>> {
        let prepared = prepared_import.borrow(py);
        prepared.require_controller(py, self)?;
        if let Some(result) = prepared
            .completed
            .lock()
            .map_err(|_| invalid("prepared import result is poisoned"))?
            .as_ref()
        {
            return Ok(result.clone_ref(py));
        }
        prepared.require_original(py, self)?;
        let session = self.session.borrow(py);
        if session
            .prepared_task_import
            .lock()
            .map_err(|_| invalid("prepared import custody is poisoned"))?
            .as_ref()
            .is_none_or(|original| original.as_ptr() != prepared_import.as_ptr())
        {
            return Err(invalid(
                "task import execution cannot substitute its retained preparation",
            ));
        }
        let child = session.private_replay_custody()?;
        if let Some(child) = &child {
            child.begin_original_import_execution(py)?;
        }
        {
            let mut stage = prepared
                .stage
                .lock()
                .map_err(|_| invalid("prepared import stage is poisoned"))?;
            if *stage != ImportPreparationStage::Prepared {
                return Err(invalid(
                    "unknown import preparation or execution cannot repeat its original callbacks",
                ));
            }
            *stage = ImportPreparationStage::ExecutionEntering;
        }
        let issued = prepared.task_use.borrow(py);
        let mut import = ColdImportGuard {
            session: &session,
            issuance: issued.issuance.clone(),
            original_decisions: Some(Arc::clone(&prepared.original_decisions)),
            committed: false,
            deferred: true,
            retained_final_use: false,
            retained_preparation: true,
            native_pending: std::cell::Cell::new(false),
        };
        self.check_import_callback(
            py,
            &issued,
            &prepared.parent.borrow(py),
            &import,
            &prepared.expected,
        )?;
        let member = self
            .replay_import(
                py,
                &prepared.task_use,
                prepared.ordinal,
                Arc::clone(&prepared.material),
                Arc::clone(&prepared.original_decisions),
                SemanticTransitionKind::Proposal,
                prepared.restore_invocation.bind(py),
                prepared.pack_policy.bind(py),
                prepared.finish_invocation.bind(py),
                prepared.refresh_snapshot.bind(py),
                &import,
                Some(&prepared),
            )?
            .ok_or_else(|| invalid("prepared retained import lost its genuine completed member"))?;
        *prepared
            .member
            .lock()
            .map_err(|_| invalid("prepared import member is poisoned"))? =
            Some(member.clone_ref(py));
        if let Some(child) = &child {
            child.retain_member(py, &member)?;
        }
        let result = (prepared.task_use.clone_ref(py), member)
            .into_pyobject(py)?
            .unbind()
            .into_any();
        *prepared
            .stage
            .lock()
            .map_err(|_| invalid("prepared import stage is poisoned"))? =
            ImportPreparationStage::Complete;
        if let Some(child) = &child {
            child.finish_original_import_execution(py)?;
        }
        *prepared
            .completed
            .lock()
            .map_err(|_| invalid("prepared import result is poisoned"))? =
            Some(result.clone_ref(py));
        import.committed = true;
        Ok(result)
    }
}

#[pymethods]
impl PySemanticTransitionSession {
    #[getter]
    fn prepared_task_import(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<Py<PySemanticPreparedTaskImport>>> {
        self.require_creator()?;
        Ok(self
            .prepared_task_import
            .lock()
            .map_err(|_| invalid("prepared import custody is poisoned"))?
            .as_ref()
            .map(|owner| owner.clone_ref(py)))
    }
}

#[pymethods]
impl PySemanticPublishedParent {
    fn source_model_contract_metadata(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        self.task_use.borrow(py).require_original_identity(&owner)?;
        let (bytes, layout) = owner
            .published_model_contract_metadata(&*self.lease()?)
            .map_err(xlog_err)?;
        Ok((
            PyBytes::new(py, &bytes),
            (
                layout.schema_begin,
                layout.schema_bytes,
                layout.schema_digest_offset,
                layout.generation_offset,
                layout.identity_offset,
            ),
        )
            .into_pyobject(py)?
            .unbind())
    }

    fn source_model_memory_geometry(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let prepared = self.original_prepared_import(py)?;
        let result = prepared.borrow(py).source_model_memory_geometry(py);
        result
    }
}

impl PySemanticTransitionController {
    pub(super) fn original_model_import_owner(
        &self,
        py: Python<'_>,
        task_use: &PySemanticTransitionTaskUse,
        parent: &PySemanticPublishedParent,
    ) -> PyResult<Option<Py<PySemanticPreparedTaskImport>>> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        let prepared = session
            .prepared_task_import
            .lock()
            .map_err(|_| invalid("prepared import custody is poisoned"))?
            .as_ref()
            .map(|owner| owner.clone_ref(py));
        if let Some(prepared) = &prepared {
            let original = prepared.borrow(py);
            original.require_controller(py, self)?;
            original.require_source_port_owner(py, task_use, parent)?;
        }
        Ok(prepared)
    }
}

#[pymethods]
impl PySemanticPublishedParent {
    pub(super) fn original_prepared_import(
        &self,
        py: Python<'_>,
    ) -> PyResult<Py<PySemanticPreparedTaskImport>> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        let prepared = session
            .prepared_task_import
            .lock()
            .map_err(|_| invalid("prepared import custody is poisoned"))?
            .as_ref()
            .map(|owner| owner.clone_ref(py))
            .ok_or_else(|| invalid("SOURCE metadata requires its original prepared import"))?;
        let original = prepared.borrow(py);
        if !std::ptr::eq(self, &*original.parent.borrow(py))
            || self.task_use.as_ptr() != original.task_use.as_ptr()
        {
            return Err(invalid(
                "SOURCE metadata cannot substitute another import reader",
            ));
        }
        drop(original);
        Ok(prepared)
    }
}
