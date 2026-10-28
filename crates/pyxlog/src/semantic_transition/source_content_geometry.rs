//! Authenticated task metadata shared by early geometry and original child imports.

use super::source_content_occurrences::exact_tuple;
use super::*;

pub(super) struct SourceTaskProjection {
    tuple: Py<PyTuple>,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

pub(super) struct SourceTaskGeometry {
    task: Arc<xlog_cuda::SemanticTaskEvaluationSpec>,
    metadata: Arc<dyn Send + Sync>,
}

struct GeometryTuple {
    value: Py<PyTuple>,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

pub(super) struct OriginalSourceGeometry {
    handles: Arc<[xlog_cuda::SemanticPreparedSourceContentGeometry]>,
    sources: Vec<(u64, Arc<VerifiedCheckpointSource>)>,
    projections: Vec<Py<PySemanticPreparedSourceContentGeometry>>,
    projection: Option<Py<PyTuple>>,
    projection_metadata: Option<xlog_cuda::SemanticReplayMaterialReservation>,
    tuple_metadata: Option<xlog_cuda::SemanticReplayMaterialReservation>,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

#[pymethods]
impl PySemanticTransitionController {
    /// Lower the original admitted roster before any model factory or tensor export.
    #[pyo3(signature = (task_use, *, instruction_admission))]
    fn prepare_source_content_geometry(
        &self,
        py: Python<'_>,
        task_use: Py<PySemanticTransitionTaskUse>,
        instruction_admission: Py<PySemanticSegmentInstructionAdmission>,
    ) -> PyResult<Py<PyTuple>> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        let task = task_use.borrow(py);
        self.require_issued(&task)?;
        let instruction = Arc::clone(&instruction_admission.borrow(py).inner);
        instruction.require_owner(py)?;
        if task.session.as_ptr() != self.session.as_ptr()
            || instruction.session.as_ptr() != self.session.as_ptr()
            || instruction.task_use.as_ptr() != task_use.as_ptr()
        {
            return Err(invalid(
                "source geometry changed its original admitted task",
            ));
        }
        let work = {
            let preparation = instruction.preparation()?;
            if preparation.build_entered
                || preparation.callback_failed
                || preparation.result.is_some()
            {
                return Err(invalid(
                    "source geometry must precede original preparation effects",
                ));
            }
            preparation
                .work
                .as_ref()
                .ok_or_else(|| {
                    invalid("source geometry requires its original admitted work registrar")
                })?
                .clone_ref(py)
        };
        work.borrow(py).check(py)?;
        let custody = work.borrow(py).custody(py)?;
        let budget = task.checkpoint.material_sources.native_budget()?;
        let mut preparation = instruction.preparation()?;
        if preparation.source_geometry.is_none() {
            let parent = instruction.parent.borrow(py);
            let lease = parent.lease()?;
            let mut native = session.owner()?;
            task.require_identity(&native)?;
            let count = native
                .source_content_geometry_member_ordinals(&lease, &instruction.inner)
                .map_err(xlog_err)?
                .count();
            let metadata = budget
                .reserve_auxiliary(
                    u64::try_from(multiply(
                        count,
                        std::mem::size_of::<(u64, Arc<VerifiedCheckpointSource>)>(),
                    )?)
                    .map_err(|_| invalid("original source geometry roster exceeds u64"))?,
                )
                .map_err(xlog_err)?;
            let _seed_metadata = budget
                .reserve_auxiliary(
                    u64::try_from(multiply(
                        count,
                        std::mem::size_of::<(u64, xlog_cuda::SemanticSourceContentGeometrySeed)>(),
                    )?)
                    .map_err(|_| invalid("original source seed roster exceeds u64"))?,
                )
                .map_err(xlog_err)?;
            let mut seeds = Vec::with_capacity(count);
            let mut sources = Vec::with_capacity(count);
            // Borrow the original checkpoint roster. Cloning its ColdValue
            // projection would recreate metadata and lose the original receipt.
            let roster = task
                .checkpoint
                .original_training_roster
                .lock()
                .map_err(|_| invalid("original training roster owner mutex is poisoned"))?;
            if roster.handoff.is_some() {
                return Err(invalid(
                    "source geometry cannot interrupt original roster materialization",
                ));
            }
            let rows = roster.projection.fields(2)?[0].sequence()?;
            let verified = task
                .checkpoint
                .checkpoint_sources
                .lock()
                .map_err(|_| invalid("checkpoint source owner mutex is poisoned"))?;
            for ordinal in native
                .source_content_geometry_member_ordinals(&lease, &instruction.inner)
                .map_err(xlog_err)?
            {
                let row = rows
                    .get(usize::try_from(ordinal).map_err(|_| {
                        invalid("original source ordinal exceeds native address space")
                    })?)
                    .ok_or_else(|| {
                        invalid("source geometry row is absent from its original roster")
                    })?
                    .fields(5)?;
                if row[0].text()? != "episode" {
                    return Err(invalid(
                        "source geometry requires its original native episode",
                    ));
                }
                let evidence = match &row[4] {
                    ColdValue::ReplayBytes(payload) => {
                        if !Arc::ptr_eq(&payload.source()?, &task.checkpoint.material_sources) {
                            return Err(invalid(
                                "source geometry evidence changed its original material registry",
                            ));
                        }
                        payload.bytes()
                    }
                    _ => {
                        return Err(invalid(
                            "source geometry requires its original guarded evidence",
                        ))
                    }
                };
                let key = native_replay_cache_key(row[2].text()?, evidence);
                let key = std::str::from_utf8(&key).expect("canonical original replay cache key");
                let binding = task
                    .checkpoint
                    .material_sources
                    .cached_native_binding(key)?
                    .ok_or_else(|| {
                        invalid("source geometry lost its already-decoded original replay owner")
                    })?;
                let digest = binding.checkpoint_source.ok_or_else(|| {
                    invalid("source geometry lost its original verified checkpoint join")
                })?;
                let source = verified.verified.get(&digest).ok_or_else(|| {
                    invalid("source geometry lost its original verified checkpoint owner")
                })?;
                if !Arc::ptr_eq(
                    &source.seed.material_sources,
                    &task.checkpoint.material_sources,
                ) {
                    return Err(invalid(
                        "source geometry changed its original material registry",
                    ));
                }
                seeds.push((
                    ordinal,
                    source_content_geometry_seed(source, binding.material)?,
                ));
                sources.push((ordinal, Arc::clone(source)));
            }
            let handles = native
                .prepare_source_content_geometry(&lease, &instruction.inner, &custody.inner, seeds)
                .map_err(xlog_err)?;
            preparation.source_geometry = Some(OriginalSourceGeometry {
                handles,
                sources,
                projections: Vec::new(),
                projection: None,
                projection_metadata: None,
                tuple_metadata: None,
                _metadata: metadata,
            });
        }
        let original = preparation
            .source_geometry
            .as_mut()
            .expect("original geometry roster retained");
        if let Some(value) = &original.projection {
            return Ok(value.clone_ref(py));
        }
        if original.projections.capacity() < original.handles.len() {
            // Native handles are already retained before this fallible Python
            // handoff. The same partial projection resumes, without reissuing.
            let metadata = budget
                .reserve_auxiliary(
                    u64::try_from(multiply(
                        original.handles.len(),
                        std::mem::size_of::<Py<PySemanticPreparedSourceContentGeometry>>(),
                    )?)
                    .map_err(|_| invalid("source geometry projection roster exceeds u64"))?,
                )
                .map_err(xlog_err)?;
            original.projections.reserve_exact(original.handles.len());
            // Retain this receipt in the original owner rather than retiring it
            // while its actual Python-reference vector remains live.
            original.projection_metadata = Some(metadata);
        }
        while original.projections.len() < original.handles.len() {
            let handle = &original.handles[original.projections.len()];
            let source = original
                .sources
                .iter()
                .find(|(ordinal, _)| *ordinal == handle.member_ordinal())
                .map(|(_, source)| Arc::clone(source))
                .ok_or_else(|| {
                    invalid("source geometry projection lost its exact original member")
                })?;
            let metadata = material_bytes::reserve_python_material::<
                PySemanticPreparedSourceContentGeometry,
            >(py, &budget)?;
            original.projections.push(Py::new(
                py,
                PySemanticPreparedSourceContentGeometry {
                    inner: handle.clone(),
                    source,
                    instruction: Arc::downgrade(&instruction),
                    projections: Mutex::new(GeometryProjections::default()),
                    _metadata: metadata,
                },
            )?);
        }
        let metadata = budget
            .reserve_auxiliary(
                u64::try_from(python_metadata_tuple_extent(
                    py,
                    original.projections.len(),
                )?)
                .map_err(|_| invalid("source geometry immutable roster exceeds u64"))?,
            )
            .map_err(xlog_err)?;
        let value = PyTuple::new(
            py,
            original.projections.iter().map(|value| value.clone_ref(py)),
        )?
        .unbind();
        original.projection = Some(value.clone_ref(py));
        original.tuple_metadata = Some(metadata);
        Ok(value)
    }
}

#[derive(Default)]
struct GeometryProjections {
    model: Option<GeometryTuple>,
    inputs: [Option<GeometryTuple>; 5],
    feedback: [Option<GeometryTuple>; 5],
    policy: [Option<GeometryTuple>; 5],
    uniform: [Option<GeometryTuple>; 5],
    task_metadata: Option<GeometryTuple>,
    input_generation: Option<Py<PySemanticModelInputGenerationPort>>,
    context: Option<GeometryTuple>,
    contract: Option<Py<material_bytes::PySemanticMaterialBytes>>,
    content_overlays: [Option<GeometryTuple>; 2],
    content_overlay_backings: [Option<ContentOverlayBacking>; 2],
}

struct ContentOverlayBacking {
    _producer: Py<PyAny>,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

fn content_overlay_kind(value: &Bound<'_, PyAny>) -> PyResult<(SemanticTransitionKind, usize)> {
    match ColdValue::read(value, &mut 128, 0)?.text()? {
        "recompute" => Ok((SemanticTransitionKind::Recompute, 0)),
        "proposal" => Ok((SemanticTransitionKind::Proposal, 1)),
        _ => Err(invalid(
            "content overlay prepared_kind must be recompute or proposal",
        )),
    }
}

/// Geometry identity is an original native allocation port, not a device pointer.
#[pyclass(
    name = "SemanticSourceGeometryAllocation",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticSourceGeometryAllocation {
    pub(super) inner: xlog_cuda::SemanticSourceGeometryAllocation,
    _owner: SourceGeometryAllocationOwner,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

#[derive(Clone)]
enum SourceGeometryAllocationOwner {
    Source(xlog_cuda::SemanticPreparedSourceContentGeometry),
    InputGeneration(xlog_cuda::SemanticModelInputGenerationPort),
    PrefillRestore(xlog_cuda::SemanticModelContextPrefillRestoreDeclaration),
}

pub(in crate::semantic_transition) fn prefill_restore_allocation(
    py: Python<'_>,
    inner: xlog_cuda::SemanticSourceGeometryAllocation,
    declaration: &xlog_cuda::SemanticModelContextPrefillRestoreDeclaration,
) -> PyResult<Py<PySemanticSourceGeometryAllocation>> {
    let metadata = material_bytes::reserve_python_material::<PySemanticSourceGeometryAllocation>(
        py,
        declaration.material_budget(),
    )?;
    Py::new(
        py,
        PySemanticSourceGeometryAllocation {
            inner,
            _owner: SourceGeometryAllocationOwner::PrefillRestore(declaration.clone()),
            _metadata: metadata,
        },
    )
}

pub(in crate::semantic_transition) fn prefill_restore_model_memory(
    py: Python<'_>,
    declaration: &xlog_cuda::SemanticModelContextPrefillRestoreDeclaration,
) -> PyResult<(Py<PyTuple>, xlog_cuda::SemanticReplayMaterialReservation)> {
    let original = model_memory_projection(
        py,
        declaration.material_budget(),
        SourceGeometryAllocationOwner::PrefillRestore(declaration.clone()),
        declaration.model_allocation_layouts().count(),
        declaration.model_allocation_layouts(),
        declaration.model_storage_layouts(),
        declaration.model_view_layouts().count(),
        declaration.model_view_layouts(),
    )?;
    Ok((original.value, original._metadata))
}

#[pyclass(
    name = "SemanticModelInputGenerationPort",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticModelInputGenerationPort {
    inner: xlog_cuda::SemanticModelInputGenerationPort,
    instruction: std::sync::Weak<SegmentInstructionCustody>,
    model: Mutex<Option<GeometryTuple>>,
    indices: Py<PyTuple>,
    _index_metadata: xlog_cuda::SemanticReplayMaterialReservation,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

impl PySemanticModelInputGenerationPort {
    fn require_original(&self, py: Python<'_>) -> PyResult<()> {
        self.instruction
            .upgrade()
            .ok_or_else(|| invalid("model input generation lost its original instruction"))?
            .require_owner(py)
    }

    fn project(
        py: Python<'_>,
        inner: xlog_cuda::SemanticModelInputGenerationPort,
        instruction: std::sync::Weak<SegmentInstructionCustody>,
    ) -> PyResult<Py<Self>> {
        let metadata =
            material_bytes::reserve_python_material::<Self>(py, &inner.material_budget())?;
        let index_metadata = inner
            .material_budget()
            .reserve_auxiliary(
                u64::try_from(add(
                    python_metadata_tuple_extent(py, 2)?,
                    multiply(2, integer_extent(py)?)?,
                )?)
                .map_err(|_| invalid("model generation selector metadata exceeds u64"))?,
            )
            .map_err(xlog_err)?;
        let indices = (inner.step_index(), inner.preceding_update())
            .into_pyobject(py)?
            .unbind();
        Py::new(
            py,
            Self {
                inner,
                instruction,
                model: Mutex::new(None),
                indices,
                _index_metadata: index_metadata,
                _metadata: metadata,
            },
        )
    }
}

#[pymethods]
impl PySemanticModelInputGenerationPort {
    fn same_port(&self, other: &Self) -> bool {
        self.inner.same_handle(&other.inner)
    }

    fn step_index(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.require_original(py)?;
        Ok(self.indices.bind(py).get_item(0)?.unbind())
    }

    fn preceding_update(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.require_original(py)?;
        Ok(self.indices.bind(py).get_item(1)?.unbind())
    }

    fn model_memory_layouts(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_original(py)?;
        let mut cache = self
            .model
            .try_lock()
            .map_err(|_| invalid("original model input geometry is already being projected"))?;
        if cache.is_none() {
            *cache = Some(model_memory_projection(
                py,
                &self.inner.material_budget(),
                SourceGeometryAllocationOwner::InputGeneration(self.inner.clone()),
                self.inner.model_allocation_layouts().count(),
                self.inner.model_allocation_layouts(),
                self.inner.model_storage_layouts(),
                self.inner.model_view_layouts().count(),
                self.inner.model_view_layouts(),
            )?);
        }
        Ok(cache
            .as_ref()
            .expect("original model input geometry")
            .value
            .clone_ref(py))
    }

    fn allocation_matches(
        &self,
        py: Python<'_>,
        declared: &PySemanticSourceGeometryAllocation,
        actual: &PyNativeTensorAllocation,
    ) -> PyResult<bool> {
        self.require_original(py)?;
        self.inner
            .allocation_matches(&declared.inner, &actual.provenance)
            .map_err(xlog_err)
    }
}

#[pymethods]
impl PySemanticTransitionController {
    #[pyo3(signature = (task_use, *, instruction_admission, step_index))]
    fn model_input_generation_port(
        &self,
        py: Python<'_>,
        task_use: Py<PySemanticTransitionTaskUse>,
        instruction_admission: Py<PySemanticSegmentInstructionAdmission>,
        step_index: usize,
    ) -> PyResult<Py<PySemanticModelInputGenerationPort>> {
        let session = self.session.borrow(py);
        session.require_creator()?;
        let task = task_use.borrow(py);
        self.require_issued(&task)?;
        let instruction = Arc::clone(&instruction_admission.borrow(py).inner);
        instruction.require_owner(py)?;
        if instruction.session.as_ptr() != self.session.as_ptr()
            || instruction.task_use.as_ptr() != task_use.as_ptr()
        {
            return Err(invalid(
                "model input generation changed its original admitted task",
            ));
        }
        let parent = instruction.parent.borrow(py);
        let lease = parent.lease()?;
        let owner = session.owner()?;
        task.require_identity(&owner)?;
        let inner = owner
            .model_input_generation_port(&lease, &instruction.inner, step_index)
            .map_err(xlog_err)?;
        drop(owner);
        PySemanticModelInputGenerationPort::project(py, inner, Arc::downgrade(&instruction))
    }
}

#[pymethods]
impl PySemanticSourceGeometryAllocation {
    fn span_bytes(&self) -> u64 {
        self.inner.span_bytes()
    }

    fn same_allocation(&self, other: &Self) -> bool {
        self.inner.same_handle(&other.inner)
    }
}

#[pyclass(
    name = "SemanticPreparedSourceContentGeometry",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticPreparedSourceContentGeometry {
    inner: xlog_cuda::SemanticPreparedSourceContentGeometry,
    source: Arc<VerifiedCheckpointSource>,
    instruction: std::sync::Weak<SegmentInstructionCustody>,
    projections: Mutex<GeometryProjections>,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

impl PySemanticPreparedSourceContentGeometry {
    pub(super) fn require_original(&self, py: Python<'_>) -> PyResult<()> {
        self.instruction
            .upgrade()
            .ok_or_else(|| invalid("source geometry lost its original instruction owner"))?
            .require_owner(py)
    }

    pub(super) fn require_child_feedback_baseline(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
    ) -> PyResult<()> {
        self.require_original(py)?;
        session.require_creator()?;
        if session.learning_preparing.load(Ordering::Acquire)
            || session
                .learning_transition
                .try_lock()
                .map_err(|_| invalid("original child feedback phase custody is already active"))?
                .is_some()
        {
            return Err(invalid(
                "source geometry feedback baseline differs from its original fresh child",
            ));
        }
        Ok(())
    }

    fn reserve(&self, extent: usize) -> PyResult<xlog_cuda::SemanticReplayMaterialReservation> {
        self.inner
            .material_budget()
            .reserve_auxiliary(
                u64::try_from(extent)
                    .map_err(|_| invalid("source geometry metadata exceeds u64"))?,
            )
            .map_err(xlog_err)
    }

    fn projection_section(
        &self,
        prepared_kind: Option<&Bound<'_, PyAny>>,
        prepared_bank: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Option<(SemanticTransitionKind, usize)>, usize)> {
        let section =
            source_content_occurrences::prepared_section_arguments(prepared_kind, prepared_bank)?;
        let index =
            match (self.inner.kind(), section) {
                (xlog_cuda::SemanticSourceContentGeometryKind::Historical, None) => 4,
                (
                    xlog_cuda::SemanticSourceContentGeometryKind::Current
                    | xlog_cuda::SemanticSourceContentGeometryKind::CurrentDemonstration,
                    Some((SemanticTransitionKind::Recompute, bank)),
                ) => bank,
                (
                    xlog_cuda::SemanticSourceContentGeometryKind::Current
                    | xlog_cuda::SemanticSourceContentGeometryKind::CurrentDemonstration,
                    Some((SemanticTransitionKind::Proposal, bank)),
                ) => 2 + bank,
                _ => return Err(invalid(
                    "source geometry requires its original historical or prepared child section",
                )),
            };
        Ok((section, index))
    }

    fn allocation(
        &self,
        py: Python<'_>,
        inner: xlog_cuda::SemanticSourceGeometryAllocation,
    ) -> PyResult<Py<PySemanticSourceGeometryAllocation>> {
        let metadata = self.reserve(material_bytes::python_instance_extent::<
            PySemanticSourceGeometryAllocation,
        >(py)?)?;
        Py::new(
            py,
            PySemanticSourceGeometryAllocation {
                inner,
                _owner: SourceGeometryAllocationOwner::Source(self.inner.clone()),
                _metadata: metadata,
            },
        )
    }

    fn tensor_projection(
        &self,
        py: Python<'_>,
        tensors: impl ExactSizeIterator<
            Item = Result<
                xlog_cuda::SemanticSourceGeometryTensor,
                xlog_cuda::SemanticTransitionError,
            >,
        >,
    ) -> PyResult<GeometryTuple> {
        let count = tensors.len();
        let extent = add(
            python_metadata_tuple_extent(py, count)?,
            multiply(
                count,
                add(
                    tensor_projection_extent(py)?,
                    std::mem::size_of::<Py<PyTuple>>(),
                )?,
            )?,
        )?;
        let metadata = self.reserve(extent)?;
        let mut rows = Vec::with_capacity(count);
        for tensor in tensors {
            let tensor = tensor.map_err(xlog_err)?;
            let layout = tensor_metadata_value(
                py,
                (tensor.layout, tensor.logical_begin, tensor.logical_end),
            )?;
            rows.push(
                (
                    layout.bind(py).get_item(0)?,
                    tensor.logical_begin,
                    tensor.logical_end,
                    self.allocation(py, tensor.allocation)?,
                    tensor.byte_offset,
                )
                    .into_pyobject(py)?
                    .unbind(),
            );
        }
        Ok(GeometryTuple {
            value: PyTuple::new(py, rows)?.unbind(),
            _metadata: metadata,
        })
    }

    fn task_metadata_field(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        self.require_original(py)?;
        let mut cache = self
            .projections
            .try_lock()
            .map_err(|_| invalid("original source task metadata is already being projected"))?;
        if cache.task_metadata.is_none() {
            let law = self.inner.task_objective_law().map_err(xlog_err)?;
            let layout = self.inner.policy_layout();
            let binding = self.inner.policy_binding();
            let (spans, cells) = self.inner.policy_support_layout();
            let schema = self.inner.feedback_schema();
            let fields = usize::try_from(schema.statement_bytes)
                .map_err(|_| invalid("original feedback field roster exceeds address space"))?
                .checked_add(2)
                .ok_or_else(|| invalid("original feedback field roster overflow"))?;
            let mut extent = python_metadata_tuple_extent(py, 10)?;
            // These are the reached canonical serializers' actual shapes. The
            // original raw task projection reserves its own metadata separately.
            for count in [4, 2, law.coefficients.len(), 2, 6, 2, 2, 2, 2, 3, 2, 7, 3] {
                extent = add(extent, python_metadata_tuple_extent(py, count)?)?;
            }
            extent = add(
                extent,
                python_metadata_tuple_extent(py, layout.fields.len())?,
            )?;
            extent = add(
                extent,
                multiply(
                    layout.fields.len(),
                    add(
                        python_metadata_tuple_extent(py, 4)?,
                        multiply(2, python_metadata_tuple_extent(py, 2)?)?,
                    )?,
                )?,
            )?;
            extent = add(
                extent,
                material_bytes::python_instance_extent::<pyo3::types::PyList>(py)?,
            )?;
            extent = add(
                extent,
                multiply(
                    spans.len(),
                    add(
                        std::mem::size_of::<*mut pyo3::ffi::PyObject>(),
                        python_metadata_tuple_extent(py, 2)?,
                    )?,
                )?,
            )?;
            extent = add(extent, python_metadata_tuple_extent(py, fields)?)?;
            extent = add(
                extent,
                multiply(
                    fields,
                    add(
                        python_metadata_tuple_extent(py, 4)?,
                        python_metadata_text_extent(py, "statement")?,
                    )?,
                )?,
            )?;
            let integers = add(
                27 + law.coefficients.len(),
                add(
                    multiply(layout.fields.len(), 6)?,
                    add(multiply(spans.len(), 2)?, multiply(fields, 2)?)?,
                )?,
            )?;
            extent = add(extent, multiply(integers, integer_extent(py)?)?)?;
            let identities = if self.inner.task_spec().training_domain.is_some() {
                9
            } else {
                8
            };
            extent = add(
                extent,
                multiply(
                    identities,
                    add(material_bytes::python_instance_extent::<PyBytes>(py)?, 33)?,
                )?,
            )?;
            if self.inner.task_spec().training_domain.is_some() {
                extent = add(extent, python_metadata_tuple_extent(py, 2)?)?;
            }
            let metadata = self.reserve(extent)?;
            let _scratch = self.reserve(add(
                multiply(spans.len(), std::mem::size_of::<(usize, usize)>())?,
                add(
                    multiply(fields, std::mem::size_of::<(&str, u64, u8, bool)>())?,
                    multiply(
                        add(add(layout.fields.len(), fields)?, law.coefficients.len())?,
                        std::mem::size_of::<Py<PyAny>>(),
                    )?,
                )?,
            )?)?;
            let domain = if let Some(domain) = &self.inner.task_spec().training_domain {
                let original = source_task_projection(py, &self.source)?;
                let raw = original
                    .bind(py)
                    .get_item(1)?
                    .cast_into::<PyTuple>()?
                    .get_item(2)?;
                (PyBytes::new(py, domain.identity().as_bytes()), raw)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind()
            } else {
                py.None()
            };
            let objective = task_objective_law_object(
                py,
                law,
                xlog_cuda::Identity256::from_bytes(self.source.binding.scoring_law),
            )?;
            let policy = policy_layout_object(py, layout, self.inner.model_catalogue_binding())?;
            let support = policy_support_layout_object(py, binding, spans.iter().cloned(), cells)?;
            let feedback = feedback_schema_object(py, schema)?;
            cache.task_metadata = Some(GeometryTuple {
                value: (
                    domain,
                    objective,
                    policy,
                    support,
                    feedback,
                    self.inner.prefix_capacity(),
                    self.inner.feedback_capacity(),
                    PyBytes::new(py, self.inner.schema_generation().as_bytes()),
                    self.inner.structural_capacities(),
                    PyBytes::new(py, self.inner.position_table_identity().as_bytes()),
                )
                    .into_pyobject(py)?
                    .unbind(),
                _metadata: metadata,
            });
        }
        Ok(cache
            .task_metadata
            .as_ref()
            .expect("original source task metadata")
            .value
            .bind(py)
            .get_item(index)?
            .unbind())
    }

    pub(super) fn native(&self) -> &xlog_cuda::SemanticPreparedSourceContentGeometry {
        &self.inner
    }
}

fn tensor_projection_extent(py: Python<'_>) -> PyResult<usize> {
    // The canonical helper builds layout8, dimensions4, strides4 and its
    // temporary metadata3. The public geometry adds exactly one tensor5 row.
    let mut extent = python_metadata_tuple_extent(py, 8)?;
    extent = add(extent, multiply(2, python_metadata_tuple_extent(py, 4)?)?)?;
    extent = add(extent, python_metadata_tuple_extent(py, 3)?)?;
    extent = add(extent, python_metadata_tuple_extent(py, 5)?)?;
    add(extent, multiply(19, integer_extent(py)?)?)
}

#[expect(
    clippy::too_many_arguments,
    reason = "one canonical model geometry serializer consumes the original allocation, storage and view rosters"
)]
fn model_memory_projection(
    py: Python<'_>,
    budget: &xlog_cuda::SemanticReplayMaterialBudget,
    owner: SourceGeometryAllocationOwner,
    allocation_count: usize,
    allocations: impl Iterator<Item = (xlog_cuda::SemanticSourceGeometryAllocation, u64)>,
    storages: &[xlog_cuda::SemanticModelStorage],
    view_count: usize,
    views: impl Iterator<Item = (SemanticTensorLayout, u64, u64)>,
) -> PyResult<GeometryTuple> {
    let mut extent = python_metadata_tuple_extent(py, 3)?;
    for count in [allocation_count, storages.len(), view_count] {
        extent = add(extent, python_metadata_tuple_extent(py, count)?)?;
        extent = add(extent, multiply(count, std::mem::size_of::<Py<PyTuple>>())?)?;
    }
    extent = add(
        extent,
        multiply(
            allocation_count,
            add(python_metadata_tuple_extent(py, 2)?, integer_extent(py)?)?,
        )?,
    )?;
    extent = add(
        extent,
        multiply(
            storages.len(),
            add(
                python_metadata_tuple_extent(py, 3)?,
                multiply(3, integer_extent(py)?)?,
            )?,
        )?,
    )?;
    extent = add(extent, multiply(view_count, tensor_projection_extent(py)?)?)?;
    let metadata = budget
        .reserve_auxiliary(
            u64::try_from(extent)
                .map_err(|_| invalid("original model geometry metadata exceeds u64"))?,
        )
        .map_err(xlog_err)?;
    let mut allocation_rows = Vec::with_capacity(allocation_count);
    for (inner, span) in allocations {
        let allocation_metadata = material_bytes::reserve_python_material::<
            PySemanticSourceGeometryAllocation,
        >(py, budget)?;
        let allocation = Py::new(
            py,
            PySemanticSourceGeometryAllocation {
                inner,
                _owner: owner.clone(),
                _metadata: allocation_metadata,
            },
        )?;
        allocation_rows.push((allocation, span).into_pyobject(py)?.unbind());
    }
    let mut storage_rows = Vec::with_capacity(storages.len());
    for storage in storages {
        storage_rows.push(
            (storage.allocation, storage.byte_offset, storage.span_bytes)
                .into_pyobject(py)?
                .unbind(),
        );
    }
    let mut view_rows = Vec::with_capacity(view_count);
    for (layout, storage, offset) in views {
        let layout = tensor_metadata_value(py, (layout, 0, 0))?;
        view_rows.push(
            (layout.bind(py).get_item(0)?, storage, offset)
                .into_pyobject(py)?
                .unbind(),
        );
    }
    Ok(GeometryTuple {
        value: (
            PyTuple::new(py, allocation_rows)?,
            PyTuple::new(py, storage_rows)?,
            PyTuple::new(py, view_rows)?,
        )
            .into_pyobject(py)?
            .unbind(),
        _metadata: metadata,
    })
}

#[pymethods]
impl PySemanticPreparedSourceContentGeometry {
    #[pyo3(signature = (other, *, prepared_kind, allocation_layouts))]
    fn content_overlay(
        &self,
        py: Python<'_>,
        other: &Self,
        prepared_kind: &Bound<'_, PyAny>,
        allocation_layouts: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        self.require_original(py)?;
        other.require_original(py)?;
        let (kind, index) = content_overlay_kind(prepared_kind)?;
        let mut remaining = 1usize << 24;
        let alternatives = exact_tuple(allocation_layouts, Some(4), &mut remaining)?;
        let mut count = 0usize;
        for alternative in alternatives.iter() {
            count = add(
                count,
                exact_tuple(&alternative, None, &mut remaining)?.len(),
            )?;
        }
        let input_metadata =
            self.reserve(multiply(count, std::mem::size_of::<(u64, u64, u64)>())?)?;
        let mut layouts: [Vec<(u64, u64, u64)>; 4] = std::array::from_fn(|_| Vec::new());
        for (index, alternative) in alternatives.iter().enumerate() {
            let rows = exact_tuple(&alternative, None, &mut remaining)?;
            layouts[index].reserve_exact(rows.len());
            for row in rows.iter() {
                let row = exact_tuple(&row, Some(3), &mut remaining)?;
                layouts[index].push((
                    ColdValue::read(&row.get_item(0)?, &mut remaining, 0)?.unsigned()?,
                    ColdValue::read(&row.get_item(1)?, &mut remaining, 0)?.unsigned()?,
                    ColdValue::read(&row.get_item(2)?, &mut remaining, 0)?.unsigned()?,
                ));
            }
        }
        let mut cache = self
            .projections
            .try_lock()
            .map_err(|_| invalid("original content overlay is already being projected"))?;
        let (allocation, placements) = self
            .inner
            .content_overlay(&other.inner, kind, layouts)
            .map_err(xlog_err)?;
        drop(input_metadata);
        if let Some(original) = &cache.content_overlays[index] {
            return Ok(original.value.clone_ref(py));
        }
        let mut extent = add(
            python_metadata_tuple_extent(py, 2)?,
            python_metadata_tuple_extent(py, 4)?,
        )?;
        for rows in &placements {
            extent = add(extent, python_metadata_tuple_extent(py, rows.len())?)?;
            extent = add(
                extent,
                multiply(
                    rows.len(),
                    add(
                        python_metadata_tuple_extent(py, 3)?,
                        add(
                            multiply(3, integer_extent(py)?)?,
                            std::mem::size_of::<Py<PyTuple>>(),
                        )?,
                    )?,
                )?,
            )?;
        }
        extent = add(extent, multiply(4, std::mem::size_of::<Py<PyTuple>>())?)?;
        let metadata = self.reserve(extent)?;
        let allocation = self.allocation(py, allocation)?;
        let mut alternatives = Vec::with_capacity(4);
        for rows in placements {
            let rows = rows
                .into_iter()
                .map(|row| row.into_pyobject(py).map(|row| row.unbind()))
                .collect::<PyResult<Vec<_>>>()?;
            alternatives.push(PyTuple::new(py, rows)?.unbind());
        }
        let value = (allocation, PyTuple::new(py, alternatives)?)
            .into_pyobject(py)?
            .unbind();
        cache.content_overlays[index] = Some(GeometryTuple {
            value: value.clone_ref(py),
            _metadata: metadata,
        });
        Ok(value)
    }

    #[pyo3(signature = (other, *, prepared_kind, backing, consumer_stream))]
    fn bind_content_overlay(
        &self,
        py: Python<'_>,
        other: &Self,
        prepared_kind: &Bound<'_, PyAny>,
        backing: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.require_original(py)?;
        other.require_original(py)?;
        let (kind, index) = content_overlay_kind(prepared_kind)?;
        let stream = ColdValue::read(consumer_stream, &mut 128, 0)?.unsigned()?;
        let producer_stream = i64::try_from(stream)
            .map_err(|_| invalid("content overlay stream exceeds DLPack address space"))?;
        if matches!(stream, 0 | 2) {
            return Err(invalid(
                "content overlay requires an explicit supported CUDA consumer stream",
            ));
        }
        let instruction = self
            .instruction
            .upgrade()
            .ok_or_else(|| invalid("content overlay lost its original instruction owner"))?;
        if other
            .instruction
            .upgrade()
            .is_none_or(|other| !Arc::ptr_eq(&instruction, &other))
        {
            return Err(invalid(
                "content overlay banks belong to different original instructions",
            ));
        }
        let device = instruction.session.borrow(py).device_ordinal;
        let check = || {
            self.require_original(py)?;
            other.require_original(py)?;
            let session = instruction.session.borrow(py);
            session.require_creator()?;
            let owner = session.owner()?;
            self.inner
                .require_content_overlay_backing_stream(&owner, &instruction.inner, stream)
                .map_err(xlog_err)
        };
        // Authenticate the original native import stream before asking the
        // producer to enqueue readiness or transfer any managed owner.
        check()?;
        let metadata = self.reserve(std::mem::size_of::<ContentOverlayBacking>())?;
        let mut cache = self
            .projections
            .try_lock()
            .map_err(|_| invalid("original content overlay backing is already being bound"))?;
        if cache.content_overlay_backings[index].is_some() {
            return Err(invalid(
                "content overlay retains one original backing handoff",
            ));
        }
        // Keep the actual producer object before every producer callback.
        // A failed or unknown native handoff does not discard that owner.
        cache.content_overlay_backings[index] = Some(ContentOverlayBacking {
            _producer: backing.clone().unbind(),
            _metadata: metadata,
        });
        // A producer can reenter an existing geometry getter, whose cache lock
        // is blocking. No cache mutex may survive across that Python callback.
        drop(cache);
        validate_producer_device_guarded(backing, device, &check)?;
        let mut handoff = TensorHandoff(vec![crate::dlpack_from_py_for_stream_guarded(
            backing,
            producer_stream,
            &check,
        )?]);
        check()?;
        let session = instruction.session.borrow(py);
        session.require_creator()?;
        let owner = session.owner()?;
        let backing = handoff
            .0
            .pop()
            .ok_or_else(|| invalid("content overlay lost its real managed backing"))?;
        self.inner
            .bind_content_overlay(
                &other.inner,
                &owner,
                &instruction.inner,
                kind,
                backing,
                stream,
            )
            .map_err(xlog_err)
    }

    fn kind(&self) -> &'static str {
        match self.inner.kind() {
            xlog_cuda::SemanticSourceContentGeometryKind::Historical => "historical",
            xlog_cuda::SemanticSourceContentGeometryKind::Current => "current",
            xlog_cuda::SemanticSourceContentGeometryKind::CurrentDemonstration => "demonstration",
        }
    }
    fn step_index(&self) -> usize {
        self.inner.step_index()
    }
    fn member_ordinal(&self) -> u64 {
        self.inner.member_ordinal()
    }
    fn bank(&self) -> Option<usize> {
        self.inner.bank()
    }
    fn restore_material_mode(&self) -> PyResult<&'static str> {
        self.inner.restore_material_mode().map_err(xlog_err)
    }
    fn source_task(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_original(py)?;
        source_task_projection(py, &self.source)
    }
    /// Metadata of the original future child; never a replacement TaskUse.
    fn task_training_domain(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.task_metadata_field(py, 0)
    }
    fn task_objective_law(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.task_metadata_field(py, 1)
    }
    /// Both canonical factories issue a fresh child without a learning phase.
    /// The original late SOURCE/CURRENT binding checks this same baseline.
    fn learning_feedback_intervention(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.require_original(py)?;
        Ok(py.None())
    }
    fn policy_layout(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.task_metadata_field(py, 2)
    }
    fn policy_support_layout(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.task_metadata_field(py, 3)
    }
    fn feedback_schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.task_metadata_field(py, 4)
    }
    fn prefix_capacity(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.task_metadata_field(py, 5)
    }
    fn feedback_capacity(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.task_metadata_field(py, 6)
    }
    fn schema_generation(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.task_metadata_field(py, 7)
    }
    fn structural_capacities(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.task_metadata_field(py, 8)
    }
    fn position_table_identity(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.task_metadata_field(py, 9)
    }
    fn model_input_generation_port(
        &self,
        py: Python<'_>,
    ) -> PyResult<Py<PySemanticModelInputGenerationPort>> {
        self.require_original(py)?;
        let mut cache = self
            .projections
            .try_lock()
            .map_err(|_| invalid("original source geometry is already being projected"))?;
        if cache.input_generation.is_none() {
            cache.input_generation = Some(PySemanticModelInputGenerationPort::project(
                py,
                self.inner.model_input_generation_port().map_err(xlog_err)?,
                self.instruction.clone(),
            )?);
        }
        Ok(cache
            .input_generation
            .as_ref()
            .expect("original CURRENT input generation")
            .clone_ref(py))
    }
    fn model_contract_material(
        &self,
        py: Python<'_>,
    ) -> PyResult<Py<material_bytes::PySemanticMaterialBytes>> {
        self.require_original(py)?;
        let mut cache = self
            .projections
            .lock()
            .map_err(|_| invalid("source geometry projection custody is poisoned"))?;
        if cache.contract.is_none() {
            cache.contract = Some(Py::new(
                py,
                material_bytes::PySemanticMaterialBytes::from_native(
                    py,
                    self.inner.model_contract_material().map_err(xlog_err)?,
                    &self.inner.material_budget(),
                )?,
            )?);
        }
        Ok(cache
            .contract
            .as_ref()
            .expect("original guarded contract")
            .clone_ref(py))
    }
    fn model_restore_context(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_original(py)?;
        let mut cache = self
            .projections
            .lock()
            .map_err(|_| invalid("source geometry projection custody is poisoned"))?;
        if let Some(original) = &cache.context {
            return Ok(original.value.clone_ref(py));
        }
        // CURRENT has no numerical context before its genuine Actor exists.
        let (parent, generation, resident, geometry) =
            self.inner.model_restore_context().map_err(xlog_err)?;
        let mut extent = python_metadata_tuple_extent(py, 4)?;
        extent = add(extent, multiply(2, python_metadata_tuple_extent(py, 3)?)?)?;
        extent = add(
            extent,
            multiply(
                3,
                add(material_bytes::python_instance_extent::<PyBytes>(py)?, 33)?,
            )?,
        )?;
        extent = add(extent, multiply(5, integer_extent(py)?)?)?;
        let metadata = self.reserve(extent)?;
        let value = (
            completed_publication_identity(py, parent)?,
            generation,
            resident,
            PyBytes::new(py, geometry.as_bytes()),
        )
            .into_pyobject(py)?
            .unbind();
        cache.context = Some(GeometryTuple {
            value: value.clone_ref(py),
            _metadata: metadata,
        });
        Ok(value)
    }
    fn model_memory_layouts(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_original(py)?;
        let mut cache = self
            .projections
            .lock()
            .map_err(|_| invalid("source geometry projection custody is poisoned"))?;
        if let Some(original) = &cache.model {
            return Ok(original.value.clone_ref(py));
        }
        cache.model = Some(model_memory_projection(
            py,
            &self.inner.material_budget(),
            SourceGeometryAllocationOwner::Source(self.inner.clone()),
            self.inner.model_allocation_layouts().count(),
            self.inner.model_allocation_layouts(),
            self.inner.model_storage_layouts(),
            self.inner.model_view_layouts().count(),
            self.inner.model_view_layouts(),
        )?);
        Ok(cache
            .model
            .as_ref()
            .expect("original model geometry")
            .value
            .clone_ref(py))
    }
    #[pyo3(signature = (*, prepared_kind=None, prepared_bank=None))]
    fn input_tensor_layouts(
        &self,
        py: Python<'_>,
        prepared_kind: Option<&Bound<'_, PyAny>>,
        prepared_bank: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyTuple>> {
        self.require_original(py)?;
        let (section, index) = self.projection_section(prepared_kind, prepared_bank)?;
        let mut cache = self
            .projections
            .lock()
            .map_err(|_| invalid("source geometry projection custody is poisoned"))?;
        if cache.inputs[index].is_none() {
            // The original native iterator is allocation-free, so count and
            // construction use the same immutable producer geometry.
            let count = self
                .inner
                .input_tensor_layouts(section)
                .map_err(xlog_err)?
                .count();
            let metadata = self.reserve(add(
                python_metadata_tuple_extent(py, count)?,
                multiply(
                    count,
                    add(
                        tensor_projection_extent(py)?,
                        std::mem::size_of::<Py<PyTuple>>(),
                    )?,
                )?,
            )?)?;
            let mut rows = Vec::with_capacity(count);
            for tensor in self.inner.input_tensor_layouts(section).map_err(xlog_err)? {
                let tensor = tensor.map_err(xlog_err)?;
                let layout = tensor_metadata_value(
                    py,
                    (tensor.layout, tensor.logical_begin, tensor.logical_end),
                )?;
                rows.push(
                    (
                        layout.bind(py).get_item(0)?,
                        tensor.logical_begin,
                        tensor.logical_end,
                        self.allocation(py, tensor.allocation)?,
                        tensor.byte_offset,
                    )
                        .into_pyobject(py)?
                        .unbind(),
                );
            }
            cache.inputs[index] = Some(GeometryTuple {
                value: PyTuple::new(py, rows)?.unbind(),
                _metadata: metadata,
            });
        }
        Ok(cache.inputs[index]
            .as_ref()
            .expect("original input geometry")
            .value
            .clone_ref(py))
    }
    #[pyo3(signature = (*, prepared_kind=None, prepared_bank=None))]
    fn feedback_tensor_layouts(
        &self,
        py: Python<'_>,
        prepared_kind: Option<&Bound<'_, PyAny>>,
        prepared_bank: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyTuple>> {
        self.require_original(py)?;
        let (section, index) = self.projection_section(prepared_kind, prepared_bank)?;
        let mut cache = self
            .projections
            .lock()
            .map_err(|_| invalid("source geometry projection custody is poisoned"))?;
        if cache.feedback[index].is_none() {
            let tensors = self
                .inner
                .feedback_tensor_layouts(section)
                .map_err(xlog_err)?;
            cache.feedback[index] = Some(self.tensor_projection(py, tensors.into_iter().map(Ok))?);
        }
        Ok(cache.feedback[index]
            .as_ref()
            .expect("original feedback geometry")
            .value
            .clone_ref(py))
    }
    #[pyo3(signature = (*, prepared_kind=None, prepared_bank=None))]
    fn policy_tensor_layouts(
        &self,
        py: Python<'_>,
        prepared_kind: Option<&Bound<'_, PyAny>>,
        prepared_bank: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyTuple>> {
        self.require_original(py)?;
        let (section, index) = self.projection_section(prepared_kind, prepared_bank)?;
        let mut cache = self
            .projections
            .lock()
            .map_err(|_| invalid("source geometry projection custody is poisoned"))?;
        if cache.policy[index].is_none() {
            let layouts = self
                .inner
                .policy_tensor_layouts(section)
                .map_err(xlog_err)?;
            // External outputs have typed layouts, not native allocation leaves.
            // The original producer Capture supplies their physical alias proof.
            let count = layouts.len();
            let layout_extent = add(
                add(
                    python_metadata_tuple_extent(py, 8)?,
                    multiply(2, python_metadata_tuple_extent(py, 4)?)?,
                )?,
                add(
                    python_metadata_tuple_extent(py, 3)?,
                    multiply(16, integer_extent(py)?)?,
                )?,
            )?;
            let metadata = self.reserve(add(
                python_metadata_tuple_extent(py, count)?,
                multiply(count, add(layout_extent, std::mem::size_of::<Py<PyAny>>())?)?,
            )?)?;
            let mut rows = Vec::with_capacity(count);
            for layout in layouts {
                rows.push(
                    tensor_metadata_value(py, (layout, 0, 0))?
                        .bind(py)
                        .get_item(0)?
                        .unbind(),
                );
            }
            cache.policy[index] = Some(GeometryTuple {
                value: PyTuple::new(py, rows)?.unbind(),
                _metadata: metadata,
            });
        }
        Ok(cache.policy[index]
            .as_ref()
            .expect("original policy geometry")
            .value
            .clone_ref(py))
    }
    #[pyo3(signature = (*, prepared_kind=None, prepared_bank=None))]
    fn policy_numerical_domain_layouts(
        &self,
        py: Python<'_>,
        prepared_kind: Option<&Bound<'_, PyAny>>,
        prepared_bank: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyTuple>> {
        self.require_original(py)?;
        let (section, index) = self.projection_section(prepared_kind, prepared_bank)?;
        let mut cache = self
            .projections
            .try_lock()
            .map_err(|_| invalid("original uniform geometry is already being projected"))?;
        if cache.uniform[index].is_none() {
            let tensors = self
                .inner
                .policy_numerical_domain_layouts(section)
                .map_err(xlog_err)?;
            cache.uniform[index] = Some(self.tensor_projection(py, tensors.into_iter().map(Ok))?);
        }
        Ok(cache.uniform[index]
            .as_ref()
            .expect("original uniform geometry")
            .value
            .clone_ref(py))
    }
    fn continuation_capacity(&self) -> PyResult<u64> {
        self.inner.continuation_capacity().map_err(xlog_err)
    }
    fn continuation_blocks(&self) -> PyResult<Option<usize>> {
        self.inner.continuation_blocks().map_err(xlog_err)
    }
}

pub(super) fn add(left: usize, right: usize) -> PyResult<usize> {
    left.checked_add(right)
        .ok_or_else(|| invalid("original source metadata extent overflow"))
}

fn multiply(count: usize, bytes: usize) -> PyResult<usize> {
    count
        .checked_mul(bytes)
        .ok_or_else(|| invalid("original source metadata extent overflow"))
}

pub(super) fn integer_extent(py: Python<'_>) -> PyResult<usize> {
    material_bytes::python_integer_extent(py)
}

fn projection_extent(py: Python<'_>, value: &ColdValue) -> PyResult<usize> {
    match value {
        ColdValue::Bytes(bytes) => add(
            material_bytes::python_instance_extent::<PyBytes>(py)?,
            add(bytes.len(), 1)?,
        ),
        // The guarded view reserves its own actual Python/Loaded metadata before
        // construction. It aliases the original payload, never copies it.
        ColdValue::ReplayBytes(_) => Ok(0),
        ColdValue::Stored(item) => {
            let mut bytes = add(
                python_metadata_tuple_extent(py, 4)?,
                python_metadata_tuple_extent(py, 2)?,
            )?;
            bytes = add(bytes, python_metadata_text_extent(py, "stored")?)?;
            bytes = add(bytes, python_metadata_text_extent(py, &item.digest)?)?;
            add(bytes, multiply(3, integer_extent(py)?)?)
        }
        ColdValue::Sequence(items) => {
            let mut bytes = python_metadata_tuple_extent(py, items.len())?;
            for item in items {
                bytes = add(bytes, projection_extent(py, item)?)?;
            }
            Ok(bytes)
        }
        _ => value.python_metadata_projection_allocation_bound(py),
    }
}

fn projection_scratch_extent(value: &ColdValue) -> PyResult<usize> {
    let ColdValue::Sequence(items) = value else {
        return Ok(0);
    };
    let mut extent = multiply(items.len(), std::mem::size_of::<Py<PyAny>>())?;
    for item in items {
        extent = add(extent, projection_scratch_extent(item)?)?;
    }
    Ok(extent)
}

fn project_value(py: Python<'_>, value: &ColdValue) -> PyResult<Py<PyAny>> {
    match value {
        ColdValue::ReplayBytes(payload) => {
            let source = payload.source()?;
            let material = material_bytes::PySemanticMaterialBytes::from_native(
                py,
                payload.original_loaded()?,
                &source.native_budget()?,
            )?;
            Ok(Py::new(py, material)?.into_any())
        }
        ColdValue::Sequence(items) => project_sequence(py, items).map(|value| value.into_any()),
        _ => value.python_value(py),
    }
}

fn project_sequence(py: Python<'_>, items: &[ColdValue]) -> PyResult<Py<PyTuple>> {
    let mut values = Vec::with_capacity(items.len());
    for item in items {
        values.push(project_value(py, item)?);
    }
    Ok(PyTuple::new(py, values)?.unbind())
}

pub(super) fn source_task_projection(
    py: Python<'_>,
    source: &VerifiedCheckpointSource,
) -> PyResult<Py<PyTuple>> {
    let mut cached = source
        .projection
        .lock()
        .map_err(|_| invalid("original source projection custody is poisoned"))?;
    if let Some(cached) = cached.as_ref() {
        return Ok(cached.tuple.clone_ref(py));
    }
    let cold = source
        .cold
        .as_ref()
        .ok_or_else(|| invalid("original source lost its canonical cold task"))?;
    let mut extent = add(
        python_metadata_tuple_extent(py, 2)?,
        add(
            python_metadata_tuple_extent(py, 5)?,
            python_metadata_tuple_extent(py, 6)?,
        )?,
    )?;
    let mut scratch_extent = 0;
    for text in [&cold.initial_theory, &cold.input_facts] {
        extent = add(extent, python_metadata_text_extent(py, text)?)?;
    }
    if let Some(text) = &cold.observer_source {
        extent = add(extent, python_metadata_text_extent(py, text)?)?;
    }
    extent = add(
        extent,
        python_metadata_tuple_extent(py, cold.statements.len())?,
    )?;
    for (statement, _) in &cold.statements {
        extent = add(extent, python_metadata_tuple_extent(py, 2)?)?;
        extent = add(extent, python_metadata_text_extent(py, statement)?)?;
        extent = add(extent, integer_extent(py)?)?;
    }
    extent = add(
        extent,
        python_metadata_tuple_extent(py, cold.query_records.len())?,
    )?;
    extent = add(
        extent,
        multiply(cold.query_records.len(), integer_extent(py)?)?,
    )?;
    for values in [&source.seed.authority, &source.seed.evaluation] {
        extent = add(extent, python_metadata_tuple_extent(py, values.len())?)?;
        scratch_extent = add(
            scratch_extent,
            multiply(values.len(), std::mem::size_of::<Py<PyAny>>())?,
        )?;
        for value in values {
            extent = add(extent, projection_extent(py, value)?)?;
            scratch_extent = add(scratch_extent, projection_scratch_extent(value)?)?;
        }
    }
    for value in [
        &source.seed.training_domain,
        &source.seed.initial_sources,
        &source.seed.source_mapping,
        &source.seed.replay_capacity,
    ] {
        extent = add(extent, projection_extent(py, value)?)?;
        scratch_extent = add(scratch_extent, projection_scratch_extent(value)?)?;
    }
    let budget = source.seed.material_sources.native_budget()?;
    let _scratch = budget
        .reserve_auxiliary(
            u64::try_from(scratch_extent)
                .map_err(|_| invalid("original source projection scratch exceeds u64"))?,
        )
        .map_err(xlog_err)?;
    let metadata = budget
        .reserve_auxiliary(
            u64::try_from(extent).map_err(|_| invalid("original source projection exceeds u64"))?,
        )
        .map_err(xlog_err)?;
    let cold = (
        cold.initial_theory.as_str(),
        cold.input_facts.as_str(),
        cold.observer_source.as_deref(),
        PyTuple::new(py, &cold.statements)?,
        PyTuple::new(py, &cold.query_records)?,
    )
        .into_pyobject(py)?
        .unbind()
        .into_any();
    let fields = PyTuple::new(
        py,
        [
            project_sequence(py, &source.seed.authority)?.into_any(),
            project_sequence(py, &source.seed.evaluation)?.into_any(),
            project_value(py, &source.seed.training_domain)?,
            project_value(py, &source.seed.initial_sources)?,
            project_value(py, &source.seed.source_mapping)?,
            project_value(py, &source.seed.replay_capacity)?,
        ],
    )?
    .unbind()
    .into_any();
    let tuple = PyTuple::new(py, [cold, fields])?.unbind();
    *cached = Some(SourceTaskProjection {
        tuple: tuple.clone_ref(py),
        _metadata: metadata,
    });
    Ok(tuple)
}

pub(super) fn source_content_geometry_seed(
    source: &Arc<VerifiedCheckpointSource>,
    material: xlog_cuda::SemanticReplayMaterial,
) -> PyResult<xlog_cuda::SemanticSourceContentGeometrySeed> {
    let needs_compiler = source
        .task_geometry
        .lock()
        .map_err(|_| invalid("original source task geometry custody is poisoned"))?
        .is_none();
    if needs_compiler {
        // Replacement Python construction and previous-error retirement may
        // execute GC callbacks. Complete both before the geometry cache lock.
        if let Err(blocking) =
            Python::attach(|py| source.seed.material_sources.ensure_compiler_diagnostic(py))
        {
            // A reentrant callback may have completed this exact cached task.
            // Reuse that known result; a separate blocking report is not a veto.
            let ready = source
                .task_geometry
                .lock()
                .map_err(|_| invalid("original source task geometry custody is poisoned"))?
                .is_some();
            if !ready {
                return Err(blocking);
            }
            drop(blocking);
        }
    }
    let mut cached = source
        .task_geometry
        .lock()
        .map_err(|_| invalid("original source task geometry custody is poisoned"))?;
    let cold = source
        .cold
        .as_ref()
        .ok_or_else(|| invalid("original source lost its canonical executable admission"))?;
    if cached.is_none() {
        let priorities = source
            .seed
            .authority
            .get(9)
            .ok_or_else(|| invalid("original source has no task priority roster"))?;
        let levels = priorities.sequence()?;
        // The cache value is inline in the already-paid verified source.
        // Only the actual transported priority buffers allocate here.
        let mut extent = multiply(
            levels.len(),
            std::mem::size_of::<xlog_cuda::SemanticTaskPriorityLevel>(),
        )?;
        for level in levels {
            extent = add(
                extent,
                multiply(
                    level.sequence()?.len(),
                    std::mem::size_of::<xlog_cuda::SemanticTaskPriorityGoal>(),
                )?,
            )?;
        }
        let budget = source.seed.material_sources.native_budget()?;
        let metadata: Arc<dyn Send + Sync> = budget
            .reserve_compiler_allocation(
                u64::try_from(extent).map_err(|_| invalid("original task geometry exceeds u64"))?,
            )
            .map_err(xlog_err)?;
        let observer = source
            .seed
            .evaluation
            .get(2)
            .ok_or_else(|| invalid("original source task evaluation is incomplete"))?;
        let observer_matches = match (&cold.observer_source, observer) {
            (Some(original), ColdValue::Text(actual)) => original == actual,
            (None, ColdValue::None) => true,
            _ => false,
        };
        if !observer_matches {
            return Err(invalid(
                "original source observer differs from its canonical admission",
            ));
        }
        let mut task = task_evaluation_spec_with_allocation_owner(
            &source.seed.evaluation,
            task_priority_levels(priorities)?,
            Some(cold.program_owner_with_library(source.proof.evaluation_rule_library())),
            &source.seed.training_domain,
            Some(&cold.initial_source),
            &source.seed.material_sources,
        )?;
        task.evaluation_intervention = source
            .proof
            .evaluation_fork_origin()
            .map(|origin| origin.intervention());
        *cached = Some(SourceTaskGeometry {
            task: Arc::new(task),
            metadata,
        });
    }
    let cached = cached.as_ref().expect("original task geometry retained");
    let config = source.config;
    let (proof, admission) = material.predecessor_geometry().map_err(xlog_err)?;
    Ok(xlog_cuda::SemanticSourceContentGeometrySeed {
        material,
        proof,
        admission,
        capacities: SemanticHypergraphCapacities::try_new(
            config.capacities.0,
            config.capacities.1,
            config.capacities.2,
            config.capacities.3,
        )
        .map_err(val_err)?,
        admission_limits: SemanticAdmissionLimits {
            max_records: config.admission_limits.0,
            max_terms: config.admission_limits.1,
            max_references: config.admission_limits.2,
            max_utf8_bytes: config.admission_limits.3,
        },
        program: Some(Arc::clone(&cold.program)),
        task: Arc::clone(&cached.task),
        metadata: Arc::clone(&cached.metadata),
    })
}
