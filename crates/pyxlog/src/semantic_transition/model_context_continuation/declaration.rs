//! Authenticated restoration geometry used to declare the original cold plan.

use super::*;

/// Saved geometry grants metadata access only. Native restoration must bind
/// the actual allocation owners before this declaration can match a proof.
#[pyclass(
    name = "SemanticModelContextPrefillRestoreDeclaration",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticModelContextPrefillRestoreDeclaration {
    inner: xlog_cuda::SemanticModelContextPrefillRestoreDeclaration,
    continuation: xlog_cuda::SemanticModelContextContinuation,
    source_session: Py<PySemanticTransitionSession>,
    source_controller: Py<PySemanticTransitionController>,
    source_task: Py<PySemanticTransitionTaskUse>,
    source_parent: Py<PySemanticPublishedParent>,
    _source_model: Py<PyAny>,
    target_task: Py<PySemanticTransitionTaskUse>,
    source_layout: Py<PyTuple>,
    source_mapping: Py<PyTuple>,
    completion_lineage: Py<PyTuple>,
    tensor_metadata: Py<PyTuple>,
    model_tensor_metadata: Py<PyTuple>,
    model_memory_layouts: Py<PyTuple>,
    _model_memory_metadata: xlog_cuda::SemanticReplayMaterialReservation,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

impl PySemanticModelContextPrefillRestoreDeclaration {
    pub(in crate::semantic_transition) fn issue(
        py: Python<'_>,
        checkpoint: Py<PySemanticTransitionRestoredCheckpoint>,
        continuation: xlog_cuda::SemanticModelContextContinuation,
        target_task: Py<PySemanticTransitionTaskUse>,
        inner: xlog_cuda::SemanticModelContextPrefillRestoreDeclaration,
    ) -> PyResult<Py<Self>> {
        checkpoint.borrow(py).session.borrow(py).require_creator()?;
        let budget = inner.material_budget();
        if !budget.same_pool(
            &checkpoint
                .borrow(py)
                .task_use
                .borrow(py)
                .checkpoint
                .material_sources
                .native_budget()?,
        ) {
            return Err(invalid(
                "prefill restoration declaration changed its original material pool",
            ));
        }
        {
            let original = checkpoint.borrow(py);
            let source = original.session.borrow(py);
            let task = original.task_use.borrow(py);
            original.controller.borrow(py).require_read_issued(&task)?;
            original.parent.borrow(py).require_task(py, &task)?;
            let target = target_task.borrow(py);
            let target_session = target.session.borrow(py);
            target_session.require_creator()?;
            let source_owner = source.owner()?;
            let declared = target_session
                .owner()?
                .model_context_checkpoint_attachment_restore_declaration(
                    &continuation,
                    &source_owner,
                    &*original.parent.borrow(py).lease()?,
                )
                .map_err(xlog_err)?;
            if !inner.same_handle(&declared) {
                return Err(invalid(
                    "prefill declaration changed its original native restore owner",
                ));
            }
        }
        let count = add(inner.tensor_count(), inner.model_tensor_count())?;
        let extent = add(
            projection_extent(py, &inner, count)?,
            inner.source_layout_clone_extent().map_err(xlog_err)?,
        )?;
        let metadata = material_bytes::reserve_material_metadata(budget, extent)?;
        let outputs = inner.tensor_metadata().map_err(xlog_err)?;
        let model = inner.model_tensor_metadata().map_err(xlog_err)?;
        if outputs.len() != inner.tensor_count() || model.len() != inner.model_tensor_count() {
            return Err(invalid(
                "prefill declaration changed its authenticated tensor geometry roster",
            ));
        }
        let source_layout = initial_source_parts(py, inner.source_layout().clone())?;
        let source_mapping = PyTuple::new(
            py,
            inner
                .source_mapping()
                .iter()
                .map(|row| (row.source.as_str(), row.offset, row.logical_position)),
        )?
        .unbind();
        let (source, source_digest, target_digest, task, epoch, generation) =
            inner.completion_lineage();
        let completion_lineage = (
            (
                PyBytes::new(py, source.instance.as_bytes()),
                source.word,
                PyBytes::new(py, source.logical_digest.as_bytes()),
                PyBytes::new(py, source.state_digest.as_bytes()),
            ),
            PyBytes::new(py, source_digest.as_bytes()),
            PyBytes::new(py, target_digest.as_bytes()),
            PyBytes::new(py, task.as_bytes()),
            epoch,
            generation,
        )
            .into_pyobject(py)?
            .unbind();
        let tensor_metadata = project_tensor_rows(py, &inner, outputs)?;
        let model_tensor_metadata = project_tensor_rows(py, &inner, model)?;
        let (model_memory_layouts, model_memory_metadata) =
            source_content_geometry::prefill_restore_model_memory(py, &inner)?;
        let source = checkpoint.borrow(py);
        let projected = Self {
            inner,
            continuation,
            source_session: source.session.clone_ref(py),
            source_controller: source.controller.clone_ref(py),
            source_task: source.task_use.clone_ref(py),
            source_parent: source.parent.clone_ref(py),
            _source_model: source.model.clone_ref(py),
            target_task,
            source_layout,
            source_mapping,
            completion_lineage,
            tensor_metadata,
            model_tensor_metadata,
            model_memory_layouts,
            _model_memory_metadata: model_memory_metadata,
            _metadata: metadata,
        };
        drop(source);
        projected.require_current(py)?;
        Py::new(py, projected)
    }

    fn require_current(&self, py: Python<'_>) -> PyResult<()> {
        let source = self.source_session.borrow(py);
        source.require_creator()?;
        let controller = self.source_controller.borrow(py);
        let task = self.source_task.borrow(py);
        if controller.session.as_ptr() != self.source_session.as_ptr()
            || task.session.as_ptr() != self.source_session.as_ptr()
            || self.source_parent.borrow(py).task_use.as_ptr() != self.source_task.as_ptr()
        {
            return Err(invalid(
                "prefill declaration changed its genuine source custody",
            ));
        }
        controller.require_read_issued(&task)?;
        let parent = self.source_parent.borrow(py);
        parent.require_task(py, &task)?;
        let target_task = self.target_task.borrow(py);
        let target = target_task.session.borrow(py);
        target.require_creator()?;
        let source_owner = source.owner()?;
        let declaration = target
            .owner()?
            .model_context_checkpoint_attachment_restore_declaration(
                &self.continuation,
                &source_owner,
                &*parent.lease()?,
            )
            .map_err(xlog_err)?;
        if !self.inner.same_handle(&declaration) {
            return Err(invalid(
                "prefill declaration belongs to another original native restore",
            ));
        }
        Ok(())
    }

    pub(in crate::semantic_transition) fn require_bound(
        &self,
        proof: &xlog_cuda::SemanticRestoredModelContextPrefill,
    ) -> PyResult<()> {
        self.inner.require_bound(proof).map_err(xlog_err)
    }

    pub(in crate::semantic_transition) fn bind_geometry(
        &self,
        py: Python<'_>,
        geometry: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.require_current(py)?;
        let mut transport = 16 * 1024 * 1024;
        let rows = source_content_occurrences::exact_tuple(geometry, Some(5), &mut transport)?;
        let mut extent = rows
            .len()
            .checked_mul(std::mem::size_of::<RestoreFactoryGeometryGroup>())
            .ok_or_else(|| invalid("restore factory geometry extent overflow"))?;
        for row in rows.iter() {
            let row = source_content_occurrences::exact_tuple(&row, Some(3), &mut transport)?;
            for (index, bytes) in [
                (
                    1,
                    std::mem::size_of::<xlog_cuda::SemanticSourceTensorDescriptor>(),
                ),
                (
                    2,
                    std::mem::size_of::<
                        xlog_cuda::SemanticSourceAllocationDescriptor<
                            xlog_cuda::SemanticSourceContentAllocation,
                        >,
                    >(),
                ),
            ] {
                let count = source_content_occurrences::exact_tuple(
                    &row.get_item(index)?,
                    None,
                    &mut transport,
                )?
                .len();
                extent = add(
                    extent,
                    count
                        .checked_mul(bytes)
                        .ok_or_else(|| invalid("restore factory geometry extent overflow"))?,
                )?;
            }
        }
        let _metadata =
            material_bytes::reserve_material_metadata(self.inner.material_budget(), extent)?;
        let mut transport = 16 * 1024 * 1024;
        let rows = source_content_occurrences::exact_tuple(geometry, Some(5), &mut transport)?;
        let mut groups = Vec::with_capacity(rows.len());
        for row in rows.iter() {
            let row = source_content_occurrences::exact_tuple(&row, Some(3), &mut transport)?;
            let verify = row.get_item(0)?;
            if !verify.is_exact_instance_of::<PyBool>() || !verify.extract::<bool>()? {
                return Err(invalid(
                    "restore factory geometry requires five exact verification groups",
                ));
            }
            groups.push((
                verify.extract::<bool>()?,
                source_content_occurrences::parse_tensor_descriptors(
                    &row.get_item(1)?,
                    &mut transport,
                )?,
                source_content_occurrences::parse_allocation_descriptors(
                    &row.get_item(2)?,
                    &mut transport,
                )?,
            ));
        }
        self.inner.bind_geometry(&groups).map_err(xlog_err)
    }
}

#[pymethods]
impl PySemanticModelContextPrefillRestoreDeclaration {
    #[getter]
    fn source_layout(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_current(py)?;
        Ok(self.source_layout.clone_ref(py))
    }

    #[getter]
    fn source_mapping(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_current(py)?;
        Ok(self.source_mapping.clone_ref(py))
    }

    #[getter]
    fn completion_lineage(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_current(py)?;
        Ok(self.completion_lineage.clone_ref(py))
    }

    #[getter]
    fn tensor_metadata(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_current(py)?;
        Ok(self.tensor_metadata.clone_ref(py))
    }

    #[getter]
    fn model_tensor_metadata(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_current(py)?;
        Ok(self.model_tensor_metadata.clone_ref(py))
    }

    fn model_memory_layouts(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_current(py)?;
        Ok(self.model_memory_layouts.clone_ref(py))
    }

    fn allocation_matches(
        &self,
        py: Python<'_>,
        declared: &source_content_geometry::PySemanticSourceGeometryAllocation,
        actual: &PyNativeTensorAllocation,
    ) -> PyResult<bool> {
        self.require_current(py)?;
        self.inner
            .allocation_matches(&declared.inner, &actual.provenance)
            .map_err(xlog_err)
    }
}

type TensorGeometryRow = (
    SemanticTensorLayout,
    u64,
    u64,
    xlog_cuda::SemanticSourceGeometryAllocation,
    u64,
);

type RestoreFactoryGeometryGroup = (
    bool,
    Vec<xlog_cuda::SemanticSourceTensorDescriptor>,
    Vec<xlog_cuda::SemanticSourceAllocationDescriptor<xlog_cuda::SemanticSourceContentAllocation>>,
);

fn project_tensor_rows(
    py: Python<'_>,
    declaration: &xlog_cuda::SemanticModelContextPrefillRestoreDeclaration,
    rows: Vec<TensorGeometryRow>,
) -> PyResult<Py<PyTuple>> {
    PyTuple::new(
        py,
        rows.into_iter()
            .map(|(layout, begin, end, allocation, offset)| {
                Ok((
                    tensor_layout_value(py, layout)?,
                    begin,
                    end,
                    source_content_geometry::prefill_restore_allocation(
                        py,
                        allocation,
                        declaration,
                    )?,
                    offset,
                )
                    .into_pyobject(py)?
                    .unbind())
            })
            .collect::<PyResult<Vec<Py<PyTuple>>>>()?,
    )
    .map(Bound::unbind)
}

fn projection_extent(
    py: Python<'_>,
    declaration: &xlog_cuda::SemanticModelContextPrefillRestoreDeclaration,
    count: usize,
) -> PyResult<usize> {
    let multiply = |count: usize, bytes: usize| {
        count
            .checked_mul(bytes)
            .ok_or_else(|| invalid("prefill declaration projection extent overflow"))
    };
    let integer = material_bytes::python_integer_extent(py)?;
    let layout = declaration.source_layout();
    let mut extent = material_bytes::python_instance_extent::<
        PySemanticModelContextPrefillRestoreDeclaration,
    >(py)?;
    extent = add(extent, python_metadata_tuple_extent(py, 4)?)?;
    for count in [layout.source.len(), layout.prefix.len()] {
        extent = add(extent, python_metadata_tuple_extent(py, count)?)?;
        extent = add(
            extent,
            multiply(
                count,
                add(python_metadata_tuple_extent(py, 8)?, multiply(8, integer)?)?,
            )?,
        )?;
    }
    extent = add(
        extent,
        python_metadata_tuple_extent(py, layout.prefill_tokens.len())?,
    )?;
    extent = add(
        extent,
        multiply(add(layout.prefill_tokens.len(), 1)?, integer)?,
    )?;
    let mapping = declaration.source_mapping();
    extent = add(extent, python_metadata_tuple_extent(py, mapping.len())?)?;
    for row in mapping {
        extent = add(extent, python_metadata_tuple_extent(py, 3)?)?;
        extent = add(extent, python_metadata_text_extent(py, &row.source)?)?;
        extent = add(extent, multiply(2, integer)?)?;
    }
    let row_extent = add(
        add(
            python_metadata_tuple_extent(py, 8)?,
            multiply(2, python_metadata_tuple_extent(py, 4)?)?,
        )?,
        add(python_metadata_tuple_extent(py, 5)?, multiply(17, integer)?)?,
    )?;
    extent = add(extent, multiply(count, row_extent)?)?;
    extent = add(
        extent,
        python_metadata_tuple_extent(py, declaration.tensor_count())?,
    )?;
    extent = add(
        extent,
        python_metadata_tuple_extent(py, declaration.model_tensor_count())?,
    )?;
    extent = add(
        extent,
        multiply(
            count,
            add(
                std::mem::size_of::<TensorGeometryRow>(),
                std::mem::size_of::<Py<PyTuple>>(),
            )?,
        )?,
    )?;
    extent = add(extent, python_metadata_tuple_extent(py, 6)?)?;
    extent = add(extent, python_metadata_tuple_extent(py, 4)?)?;
    extent = add(
        extent,
        multiply(
            6,
            add(material_bytes::python_instance_extent::<PyBytes>(py)?, 33)?,
        )?,
    )?;
    add(extent, multiply(3, integer)?)
}
