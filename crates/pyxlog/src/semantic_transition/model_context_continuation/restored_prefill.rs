//! Python custody for a native authenticated, restored completed prefill.

use super::*;

/// Shared with the genuine stage and witnesses without retaining either one.
/// The native proof, current model and continuation survive every projection.
pub(in crate::semantic_transition) struct RestoredPrefillPythonCustody {
    continuation: xlog_cuda::SemanticModelContextContinuation,
    controller: Py<PySemanticTransitionController>,
    task: Py<PySemanticTransitionTaskUse>,
    source_controller: Py<PySemanticTransitionController>,
    source_task: Py<PySemanticTransitionTaskUse>,
    source_parent: Py<PySemanticPublishedParent>,
    proof: xlog_cuda::SemanticRestoredModelContextPrefill,
    model_owner: Py<PyAny>,
    model_tensors: Py<PyAny>,
    model_allocations: Py<PyAny>,
    _metadata: Arc<xlog_cuda::SemanticReplayMaterialReservation>,
}

impl RestoredPrefillPythonCustody {
    pub(in crate::semantic_transition) fn require_current(&self, py: Python<'_>) -> PyResult<()> {
        let controller = self.controller.borrow(py);
        let task = self.task.borrow(py);
        if controller.session.as_ptr() != task.session.as_ptr() {
            return Err(invalid(
                "restored prefill changed its original controller or target task",
            ));
        }
        controller.require_read_issued(&task)?;
        let session = task.session.borrow(py);
        session.require_creator()?;
        session
            .owner()?
            .require_restored_model_context_prefill(&self.continuation, &self.proof)
            .map_err(xlog_err)?;
        Ok(())
    }

    pub(in crate::semantic_transition) fn content_error(
        &self,
        py: Python<'_>,
        error: xlog_cuda::SemanticTransitionError,
    ) -> PyErr {
        let original = xlog_err(error);
        let pending = (|| -> PyResult<bool> {
            self.require_current(py)?;
            let task = self.task.borrow(py);
            let session = task.session.borrow(py);
            let result = session
                .owner()?
                .restored_model_context_prefill_consumer_commands_pending(
                    &self.continuation,
                    &self.proof,
                )
                .map_err(xlog_err);
            result
        })();
        match pending {
            Ok(true) => {
                let pending = SemanticModelContextContinuationPending::new_err(
                    "original restored prefill content completion is pending; resolve its retained consumer commands",
                );
                pending.set_cause(py, Some(original));
                pending
            }
            Ok(false) => original,
            Err(error) => {
                error.set_cause(py, Some(original));
                error
            }
        }
    }
}

#[pyclass(
    name = "SemanticRestoredModelContextPrefill",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticRestoredModelContextPrefill {
    custody: Arc<RestoredPrefillPythonCustody>,
    stage: Py<PySemanticInitialPrefillStage>,
    receipt: Py<PySemanticInitialPrefillReceipt>,
    model_content_witness: Py<PySemanticInitialPrefillContentWitness>,
    boundary_witness: Py<PySemanticInitialPrefillContentWitness>,
    source_layout: Py<PyTuple>,
    source_mapping: Py<PyTuple>,
    tensor_layouts: Py<PyTuple>,
    tensor_metadata: Py<PyTuple>,
    native_capacity_tensor_metadata: Py<PyTuple>,
    completion_lineage: Py<PyTuple>,
    restored_source_parent: Py<PyTuple>,
}

impl PySemanticRestoredModelContextPrefill {
    pub(super) fn require_current(&self, py: Python<'_>) -> PyResult<()> {
        self.custody.require_current(py)
    }
    /// Called only by the canonical restore-return after native known completion.
    /// No forward, producer callback, new seal or publication is performed here.
    pub(in crate::semantic_transition) fn issue(
        py: Python<'_>,
        context: &PySemanticModelContextContinuation,
        proof: xlog_cuda::SemanticRestoredModelContextPrefill,
        model_owner: Py<PyAny>,
        model_tensors: Py<PyAny>,
        model_allocations: Py<PyAny>,
    ) -> PyResult<Py<Self>> {
        let task = context.target_task.borrow(py);
        let session = task.session.borrow(py);
        context.require_target(py, &context.target_controller.borrow(py), &task)?;
        if context.model_owner.as_ptr() != model_owner.as_ptr()
            || context.state()?.prefill.is_some()
        {
            return Err(invalid(
                "restored prefill must project its original model and native stage once",
            ));
        }
        session
            .owner()?
            .require_restored_model_context_prefill(&context.inner, &proof)
            .map_err(xlog_err)?;
        let budget = proof.material_budget();
        if !budget.same_pool(&task.checkpoint.material_sources.native_budget()?) {
            return Err(invalid(
                "restored prefill projection changed its original material pool",
            ));
        }
        let layout_count = proof.capacity_tensor_count();
        let extent = add(
            projection_extent(
                py,
                proof.source_layout(),
                proof.source_mapping(),
                layout_count,
            )?,
            proof.source_layout_clone_extent().map_err(xlog_err)?,
        )?;
        let metadata = Arc::new(material_bytes::reserve_material_metadata(budget, extent)?);
        let (
            lease,
            scope,
            layout,
            model_witness,
            output_witness,
            layouts,
            capacity_metadata,
            output_metadata,
        ) = {
            let owner = session.owner()?;
            let (lease, scope, layout, model_witness, output_witness) = owner
                .project_restored_model_context_prefill_boundary(&context.inner, &proof)
                .map_err(xlog_err)?;
            let layouts = owner
                .restored_model_context_prefill_tensor_layouts(&context.inner, &proof)
                .map_err(xlog_err)?;
            let mut capacity_metadata = Vec::with_capacity(layouts.len());
            let mut output_metadata = Vec::with_capacity(layouts.len());
            for layout in &layouts {
                let role = SemanticStateRole::from_code(layout.role)
                    .ok_or_else(|| invalid("restored prefill has an unknown native tensor role"))?;
                if matches!(layout.role, 4..=13) {
                    capacity_metadata.push(
                        owner
                            .restored_model_context_prefill_native_capacity_tensor_metadata(
                                &context.inner,
                                &proof,
                                role,
                                layout.index,
                            )
                            .map_err(xlog_err)?,
                    );
                    output_metadata.push(
                        owner
                            .restored_model_context_prefill_tensor_metadata(
                                &context.inner,
                                &proof,
                                role,
                                layout.index,
                            )
                            .map_err(xlog_err)?,
                    );
                }
            }
            (
                lease,
                scope,
                layout,
                model_witness,
                output_witness,
                layouts,
                capacity_metadata,
                output_metadata,
            )
        };
        if layouts.len() != layout_count {
            return Err(invalid(
                "restored prefill native layout roster changed during projection",
            ));
        }
        let prefix_len = layout.prefill_tokens.len();
        let source_layout = initial_source_parts(py, layout)?;
        let source_mapping = PyTuple::new(
            py,
            proof
                .source_mapping()
                .iter()
                .map(|row| (row.source.as_str(), row.offset, row.logical_position)),
        )?
        .unbind();
        let native_capacity_tensor_metadata = PyTuple::new(
            py,
            capacity_metadata
                .into_iter()
                .map(|row| tensor_metadata_value(py, row))
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .unbind();
        let tensor_layouts = PyTuple::new(
            py,
            layouts
                .into_iter()
                .map(|layout| tensor_layout_value(py, layout))
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .unbind();
        let tensor_metadata = PyTuple::new(
            py,
            output_metadata
                .into_iter()
                .map(|row| tensor_metadata_value(py, row))
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .unbind();
        let (
            source_parent,
            source_digest,
            target_digest,
            task_identity,
            task_epoch,
            model_generation,
        ) = proof.completion_lineage();
        let completion_lineage = (
            (
                PyBytes::new(py, source_parent.instance.as_bytes()),
                source_parent.word,
                PyBytes::new(py, source_parent.logical_digest.as_bytes()),
                PyBytes::new(py, source_parent.state_digest.as_bytes()),
            ),
            PyBytes::new(py, source_digest.as_bytes()),
            PyBytes::new(py, target_digest.as_bytes()),
            PyBytes::new(py, task_identity.as_bytes()),
            task_epoch,
            model_generation,
        )
            .into_pyobject(py)?
            .unbind();
        let current_source = proof.restored_source_parent();
        let restored_source_parent = (
            PyBytes::new(py, current_source.instance.as_bytes()),
            current_source.word,
            PyBytes::new(py, current_source.logical_digest.as_bytes()),
            PyBytes::new(py, current_source.state_digest.as_bytes()),
        )
            .into_pyobject(py)?
            .unbind();
        let custody = Arc::new(RestoredPrefillPythonCustody {
            continuation: context.inner.clone(),
            controller: context.target_controller.clone_ref(py),
            task: context.target_task.clone_ref(py),
            source_controller: context.source_controller.clone_ref(py),
            source_task: context.source.task().clone_ref(py),
            source_parent: context.source.parent().clone_ref(py),
            proof,
            model_owner,
            model_tensors,
            model_allocations,
            _metadata: metadata,
        });
        let stage = Py::new(
            py,
            PySemanticInitialPrefillStage {
                session: task.session.clone_ref(py),
                task_use: context.target_task.clone_ref(py),
                scope: Arc::clone(&scope),
                lease: lease.clone(),
                layout: source_layout.clone_ref(py),
                prefix_len,
                model_producers: Mutex::new(vec![
                    custody.model_tensors.clone_ref(py),
                    custody.model_allocations.clone_ref(py),
                ]),
                input_producer: Mutex::new(None),
                output_producers: Mutex::new(Vec::new()),
                restored_custody: Some(Arc::clone(&custody)),
            },
        )?;
        let receipt = Py::new(
            py,
            PySemanticInitialPrefillReceipt {
                stage: stage.clone_ref(py),
                consumed: AtomicBool::new(true),
            },
        )?;
        let model_content_witness = Py::new(
            py,
            PySemanticInitialPrefillContentWitness {
                inner: model_witness,
                _inputs: custody.model_tensors.clone_ref(py),
                _producers: vec![custody.model_allocations.clone_ref(py)],
                stage: stage.clone_ref(py),
            },
        )?;
        let boundary_witness = Py::new(
            py,
            PySemanticInitialPrefillContentWitness {
                inner: output_witness,
                // The genuine stage retains every native restored output owner.
                _inputs: stage.clone_ref(py).into_any(),
                _producers: Vec::new(),
                stage: stage.clone_ref(py),
            },
        )?;
        let projected = Py::new(
            py,
            Self {
                custody,
                stage: stage.clone_ref(py),
                receipt,
                model_content_witness,
                boundary_witness,
                source_layout: source_layout.clone_ref(py),
                source_mapping,
                tensor_layouts,
                tensor_metadata,
                native_capacity_tensor_metadata,
                completion_lineage,
                restored_source_parent,
            },
        )?;
        let mut state = context.state()?;
        state.prefill_lease = Some((lease, scope));
        state.prefill_layout = Some((source_layout, prefix_len));
        state.prefill = Some(stage);
        drop(state);
        Ok(projected)
    }

    fn export(
        slf: Py<Self>,
        py: Python<'_>,
        role: SemanticStateRole,
        index: u64,
        stream: u64,
        export: impl FnOnce(
            &mut xlog_cuda::SemanticTransitionSession,
            &xlog_cuda::SemanticModelContextContinuation,
            &xlog_cuda::SemanticRestoredModelContextPrefill,
            SemanticStateRole,
            u64,
            u64,
        ) -> Result<DlpackManagedTensor, xlog_cuda::SemanticTransitionError>,
    ) -> PyResult<Py<PyAny>> {
        let original = slf.borrow(py);
        original.require_current(py)?;
        let metadata = original.export_metadata(py, false)?;
        let task = original.custody.task.borrow(py);
        let session = task.session.borrow(py);
        let tensor = {
            let mut owner = session.owner()?;
            owner
                .require_restored_model_context_prefill(
                    &original.custody.continuation,
                    &original.custody.proof,
                )
                .map_err(xlog_err)?;
            export(
                &mut owner,
                &original.custody.continuation,
                &original.custody.proof,
                role,
                index,
                stream,
            )
            .map_err(xlog_err)?
        };
        let owner_thread = session.owner_thread;
        drop(session);
        drop(task);
        drop(original);
        crate::dlpack_capsule_from_tensor(
            py,
            retain_export_owner(
                tensor,
                PublishedExportOwner::RestoredPrefill {
                    _prefill: slf,
                    _metadata: metadata,
                },
                owner_thread,
            )?,
        )
    }

    fn export_metadata(
        &self,
        py: Python<'_>,
        allocation: bool,
    ) -> PyResult<xlog_cuda::SemanticReplayMaterialReservation> {
        let mut extent = add(
            std::mem::size_of::<PublishedExportContext>(),
            std::mem::size_of::<xlog_cuda::dlpack::DLManagedTensor>(),
        )?;
        extent = add(
            extent,
            material_bytes::python_instance_extent::<pyo3::types::PyCapsule>(py)?,
        )?;
        if allocation {
            extent = add(
                extent,
                material_bytes::python_instance_extent::<PyNativeTensorAllocation>(py)?,
            )?;
            extent = add(extent, python_metadata_tuple_extent(py, 2)?)?;
        }
        material_bytes::reserve_material_metadata(self.custody.proof.material_budget(), extent)
    }
}

#[pymethods]
impl PySemanticRestoredModelContextPrefill {
    #[getter]
    fn stage(&self, py: Python<'_>) -> PyResult<Py<PySemanticInitialPrefillStage>> {
        self.require_current(py)?;
        Ok(self.stage.clone_ref(py))
    }

    #[getter]
    fn receipt(&self, py: Python<'_>) -> PyResult<Py<PySemanticInitialPrefillReceipt>> {
        self.require_current(py)?;
        Ok(self.receipt.clone_ref(py))
    }

    #[getter]
    fn model_content_witness(
        &self,
        py: Python<'_>,
    ) -> PyResult<Py<PySemanticInitialPrefillContentWitness>> {
        self.require_current(py)?;
        Ok(self.model_content_witness.clone_ref(py))
    }

    #[getter]
    fn boundary_witness(
        &self,
        py: Python<'_>,
    ) -> PyResult<Py<PySemanticInitialPrefillContentWitness>> {
        self.require_current(py)?;
        Ok(self.boundary_witness.clone_ref(py))
    }

    #[getter]
    fn model_owner(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.require_current(py)?;
        Ok(self.custody.model_owner.clone_ref(py))
    }

    #[getter]
    fn target_task_use(&self, py: Python<'_>) -> PyResult<Py<PySemanticTransitionTaskUse>> {
        self.require_current(py)?;
        Ok(self.custody.task.clone_ref(py))
    }

    #[getter]
    fn model_tensors(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.require_current(py)?;
        Ok(self.custody.model_tensors.clone_ref(py))
    }

    #[getter]
    fn model_allocations(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.require_current(py)?;
        Ok(self.custody.model_allocations.clone_ref(py))
    }

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
    /// Complete eight-field layouts from the restored native tensor table.
    fn tensor_layouts(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_current(py)?;
        Ok(self.tensor_layouts.clone_ref(py))
    }

    #[getter]
    /// Cache rows `(layout, begin, end)` in the original prefill geometry.
    /// These match the restored boundary witness, including compact attention.
    fn tensor_metadata(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_current(py)?;
        Ok(self.tensor_metadata.clone_ref(py))
    }

    #[getter]
    /// Cache rows in native capacity geometry, distinct from boundary geometry.
    fn native_capacity_tensor_metadata(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_current(py)?;
        Ok(self.native_capacity_tensor_metadata.clone_ref(py))
    }

    #[getter]
    /// Saved source identity, both native checkpoint digests and prefill task basis.
    fn completion_lineage(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_current(py)?;
        Ok(self.completion_lineage.clone_ref(py))
    }

    #[getter]
    /// The actual freshly restored source identity, separate from saved lineage.
    fn restored_source_parent(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.require_current(py)?;
        Ok(self.restored_source_parent.clone_ref(py))
    }

    /// Join only the original already issued content consumer commands.
    fn resolve_pending_content(&self, py: Python<'_>) -> PyResult<()> {
        self.require_current(py)?;
        let source_task = self.custody.source_task.borrow(py);
        let source_controller = self.custody.source_controller.borrow(py);
        source_controller.require_read_issued(&source_task)?;
        let source = source_task.session.borrow(py);
        source.require_creator()?;
        let active = source_task.already_issued_cold_model_work(py)?;
        let work = active.borrow(py).custody(py)?;
        if work.parent()?.as_ptr() != self.custody.source_parent.as_ptr() {
            return Err(invalid(
                "restored content completion changed its original source callback parent",
            ));
        }
        let target_task = self.custody.task.borrow(py);
        let target = target_task.session.borrow(py);
        target.require_creator()?;
        let source_owner = source.owner()?;
        source_task.require_current(&source_owner)?;
        let mut target_owner = target.owner()?;
        source_owner
            .require_model_context_material_restore_callback_work(
                &target_owner,
                &self.custody.continuation,
                &*self.custody.source_parent.borrow(py).lease()?,
                &work.inner,
                work.region.as_ref().ok_or_else(|| {
                    invalid("restored content completion lost its original source report region")
                })?,
            )
            .map_err(xlog_err)?;
        let result = target_owner.resolve_restored_model_context_prefill_consumer_commands(
            &self.custody.continuation,
            &self.custody.proof,
        );
        drop(target_owner);
        drop(source_owner);
        result.map_err(|error| self.custody.content_error(py, error))
    }

    fn require_basis(
        &self,
        py: Python<'_>,
        selected_model_owner: Py<PyAny>,
        target_task_use: Py<PySemanticTransitionTaskUse>,
        target_parent_or_step: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.require_current(py)?;
        if selected_model_owner.as_ptr() != self.custody.model_owner.as_ptr()
            || target_task_use.as_ptr() != self.custody.task.as_ptr()
        {
            return Err(invalid(
                "restored prefill basis changed its original model or target task",
            ));
        }
        let task = self.custody.task.borrow(py);
        let session = task.session.borrow(py);
        if target_parent_or_step.is_exact_instance_of::<PySemanticPreparedStep>() {
            let step = target_parent_or_step.extract::<PyRef<'_, PySemanticPreparedStep>>()?;
            step.require_task(py, &task)?;
            session
                .owner()?
                .require_model_context_prepared_model_basis(&self.custody.continuation, &step.inner)
                .map_err(xlog_err)
        } else if target_parent_or_step.is_exact_instance_of::<PySemanticPublishedParent>() {
            let parent = target_parent_or_step.extract::<PyRef<'_, PySemanticPublishedParent>>()?;
            parent.require_task(py, &task)?;
            let result = session
                .owner()?
                .require_model_context_initial_prefill_basis(
                    &self.custody.continuation,
                    &self.stage.borrow(py).lease,
                    &*parent.lease()?,
                )
                .map_err(xlog_err);
            result
        } else {
            Err(invalid(
                "restored prefill basis requires its genuine frozen step or acquired target parent",
            ))
        }
    }

    #[pyo3(signature = (role, index, *, consumer_stream))]
    /// Export seal-verified restored outputs in the original prefill geometry.
    /// The capsule retains their native proof, current model and continuation.
    fn persistent_tensor_view(
        slf: Py<Self>,
        py: Python<'_>,
        role: &Bound<'_, PyAny>,
        index: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        slf.borrow(py).require_current(py)?;
        let (role, index, stream) = export_arguments(role, index, consumer_stream)?;
        Self::export(
            slf,
            py,
            role,
            index,
            stream,
            xlog_cuda::SemanticTransitionSession::restored_model_context_prefill_tensor,
        )
    }

    #[pyo3(signature = (role, index, *, consumer_stream))]
    /// Export the actual native cache allocation with its capacity geometry.
    /// Only the logical interval in native_capacity_tensor_metadata is valid.
    fn native_capacity_tensor_view(
        slf: Py<Self>,
        py: Python<'_>,
        role: &Bound<'_, PyAny>,
        index: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        slf.borrow(py).require_current(py)?;
        let (role, index, stream) = export_arguments(role, index, consumer_stream)?;
        Self::export(slf, py, role, index, stream, xlog_cuda::SemanticTransitionSession::restored_model_context_prefill_native_capacity_tensor)
    }

    #[pyo3(signature = (role, index, *, consumer_stream))]
    /// Original output allocation provenance and its complete U8 backing owner.
    fn persistent_tensor_allocation(
        slf: Py<Self>,
        py: Python<'_>,
        role: &Bound<'_, PyAny>,
        index: &Bound<'_, PyAny>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        let original = slf.borrow(py);
        original.require_current(py)?;
        let (role, index, stream) = export_arguments(role, index, consumer_stream)?;
        let metadata = original.export_metadata(py, true)?;
        let task = original.custody.task.borrow(py);
        let session = task.session.borrow(py);
        let (provenance, tensor) = session
            .owner()?
            .restored_model_context_prefill_tensor_allocation(
                &original.custody.continuation,
                &original.custody.proof,
                role,
                index,
                stream,
            )
            .map_err(xlog_err)?;
        let owner_thread = session.owner_thread;
        drop(session);
        drop(task);
        drop(original);
        let allocation = Py::new(
            py,
            PyNativeTensorAllocation {
                provenance,
                policy_layout: None,
            },
        )?;
        let tensor = retain_export_owner(
            tensor,
            PublishedExportOwner::RestoredPrefill {
                _prefill: slf,
                _metadata: metadata,
            },
            owner_thread,
        )?;
        Ok((allocation, crate::dlpack_capsule_from_tensor(py, tensor)?)
            .into_pyobject(py)?
            .unbind())
    }

    #[pyo3(signature = (*, consumer_stream))]
    /// The actual native tensor table record, retaining its original owner.
    fn tensor_table_rows(
        slf: Py<Self>,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        slf.borrow(py).require_current(py)?;
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        Self::export(
            slf,
            py,
            SemanticStateRole::TensorLayout,
            0,
            stream,
            xlog_cuda::SemanticTransitionSession::restored_model_context_prefill_record,
        )
    }

    #[pyo3(signature = (*, consumer_stream))]
    /// Actual restored token provenance records; no authority is synthesized.
    fn provenance_records(
        slf: Py<Self>,
        py: Python<'_>,
        consumer_stream: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        slf.borrow(py).require_current(py)?;
        let stream = parse_witness_consumer_stream(consumer_stream, &mut 128)?;
        Self::export(
            slf,
            py,
            SemanticStateRole::TokenProvenanceRecords,
            0,
            stream,
            xlog_cuda::SemanticTransitionSession::restored_model_context_prefill_record,
        )
    }
}

fn export_arguments(
    role: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    stream: &Bound<'_, PyAny>,
) -> PyResult<(SemanticStateRole, u64, u64)> {
    let mut budget = 128;
    let role = SemanticStateRole::from_code(ColdValue::read(role, &mut budget, 0)?.unsigned()?)
        .ok_or_else(|| invalid("unknown native restored prefill tensor role"))?;
    let index = ColdValue::read(index, &mut budget, 0)?.unsigned()?;
    let stream = parse_witness_consumer_stream(stream, &mut budget)?;
    Ok((role, index, stream))
}

fn projection_extent(
    py: Python<'_>,
    layout: &SemanticInitialSourceLayout,
    mapping: &[SemanticSourceMapping],
    layout_count: usize,
) -> PyResult<usize> {
    let multiply = |count: usize, width: usize| {
        count
            .checked_mul(width)
            .ok_or_else(|| invalid("restored prefill projection extent overflow"))
    };
    let integer = material_bytes::python_integer_extent(py)?;
    let mut extent = add(
        material_bytes::python_instance_extent::<PySemanticRestoredModelContextPrefill>(py)?,
        std::mem::size_of::<RestoredPrefillPythonCustody>(),
    )?;
    extent = add(extent, 4 * std::mem::size_of::<usize>())?;
    extent = add(
        extent,
        std::mem::size_of::<xlog_cuda::SemanticReplayMaterialReservation>(),
    )?;
    extent = add(
        extent,
        material_bytes::python_instance_extent::<PySemanticInitialPrefillStage>(py)?,
    )?;
    extent = add(
        extent,
        material_bytes::python_instance_extent::<PySemanticInitialPrefillReceipt>(py)?,
    )?;
    extent = add(
        extent,
        multiply(
            2,
            material_bytes::python_instance_extent::<PySemanticInitialPrefillContentWitness>(py)?,
        )?,
    )?;
    extent = add(extent, multiply(3, std::mem::size_of::<Py<PyAny>>())?)?;
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
    extent = add(extent, python_metadata_tuple_extent(py, mapping.len())?)?;
    for row in mapping {
        extent = add(extent, python_metadata_tuple_extent(py, 3)?)?;
        extent = add(extent, python_metadata_text_extent(py, &row.source)?)?;
        extent = add(extent, multiply(2, integer)?)?;
    }
    let tensor_row = add(
        add(
            python_metadata_tuple_extent(py, 8)?,
            multiply(2, python_metadata_tuple_extent(py, 4)?)?,
        )?,
        add(python_metadata_tuple_extent(py, 3)?, multiply(16, integer)?)?,
    )?;
    extent = add(extent, multiply(multiply(3, layout_count)?, tensor_row)?)?;
    extent = add(
        extent,
        multiply(3, python_metadata_tuple_extent(py, layout_count)?)?,
    )?;
    extent = add(
        extent,
        multiply(
            layout_count,
            add(
                std::mem::size_of::<SemanticTensorLayout>(),
                2 * std::mem::size_of::<(SemanticTensorLayout, u64, u64)>(),
            )?,
        )?,
    )?;
    extent = add(
        extent,
        multiply(
            multiply(3, layout_count)?,
            std::mem::size_of::<Py<PyTuple>>(),
        )?,
    )?;
    extent = add(extent, python_metadata_tuple_extent(py, 6)?)?;
    extent = add(extent, multiply(2, python_metadata_tuple_extent(py, 4)?)?)?;
    extent = add(
        extent,
        multiply(
            9,
            add(material_bytes::python_instance_extent::<PyBytes>(py)?, 33)?,
        )?,
    )?;
    add(extent, multiply(4, integer)?)
}
