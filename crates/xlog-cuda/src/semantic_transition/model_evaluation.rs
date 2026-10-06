//! Read-only model invocations over one original native-selected cohort.

use super::*;

/// Original selection and its immutable device content, independent of the
/// source Runtime's lifetime. Final evaluations borrow this same allocation.
pub struct SemanticEvaluationCohort {
    selected: SemanticSelectedTrainingView,
    content: TensorContentBuffers,
    domain: Identity256,
    task: SemanticTaskContentIdentity,
    device: usize,
    provider: Arc<CudaKernelProvider>,
}

/// Session-issued invocation handle, not an Update or a publication capability.
#[derive(Clone)]
pub struct SemanticModelEvaluation {
    issuer: Arc<()>,
    token: u64,
    cohort: Arc<SemanticEvaluationCohort>,
}

impl SemanticModelEvaluation {
    pub fn cohort(&self) -> Arc<SemanticEvaluationCohort> {
        Arc::clone(&self.cohort)
    }
}

/// Cold observation of the original FP32 outputs and canonical work tally.
/// Memory is retained allocation capacity, not an allocator-wide time sample.
#[derive(Clone, Copy, Debug)]
pub struct SemanticModelEvaluationResult {
    pub status: u64,
    pub loss_bits: [u32; 6],
    pub model_work: u64,
    pub operation_count: u64,
    pub work_bound: u64,
    pub retained_allocation_bytes: u64,
    pub model_calls: u64,
}

pub(super) struct EvaluationStorage {
    cohort: Arc<SemanticEvaluationCohort>,
    work: PreparedModelWork,
    report: TrackedCudaSlice<u64>,
    source_material: Identity256,
    stream: u64,
    submitted: bool,
    result: Option<SemanticModelEvaluationResult>,
}

impl SemanticTransitionSession {
    /// Select only for the source observation. Supplying its issued cohort
    /// reuses all seventeen ports, never final-model RNG or a caller ordinal.
    pub fn prepare_model_evaluation(
        &mut self,
        lease: &SemanticPublishedLease,
        cohort: Option<Arc<SemanticEvaluationCohort>>,
        event_capacity: usize,
    ) -> Result<SemanticModelEvaluation, SemanticTransitionError> {
        self.checked_reader(lease)?;
        if self.steps[&lease.token].evaluation.is_some() || self.prepared_segment.is_some() {
            return Err(publication_input_error(
                "evaluation requires an unused acquired parent outside a prepared segment",
            ));
        }
        let arena = self
            .training_views
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let domain = arena.training_domain_identity();
        let task = self
            .task
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .0
            .content_identity();
        let device = self.provider.device().ordinal();
        let source_bytes = self.published_state_material(lease)?;
        PublicationMaterial::decode(&source_bytes)?.require_current_model_caches()?;
        let source_material = Identity256::from_bytes(Sha256::digest(&source_bytes).into());
        let cohort = if let Some(original) = cohort {
            if original.domain != domain || original.task != task || original.device != device {
                return Err(publication_input_error("evaluation cohort differs from the original admitted task, domain or CUDA device"));
            }
            original
        } else {
            let selected = self.select_training_view(lease)?;
            let cells = Arc::new(allocate_publication(&self.provider, 17 * 4)?);
            let mut tensors = Vec::with_capacity(17);
            let mut digests = Vec::with_capacity(17);
            for (index, port) in SemanticTrainingViewPort::ALL.into_iter().enumerate() {
                let (view, shape, strides, dtype) = selected.port(port)?;
                let mut layout = SemanticTensorLayout {
                    role: 0,
                    index: index as u64,
                    element_bytes: u64::from(dtype.1 / 8),
                    scalar_type: match dtype {
                        (1, 64) => 3,
                        (0, 64) => 7,
                        (2, 32) => 6,
                        _ => return Err(SemanticTransitionError::ObservationMismatch),
                    },
                    rank: shape.len() as u64,
                    logical_axis: u64::MAX,
                    dimensions: [0; 4],
                    strides_bytes: [0; 4],
                };
                for (axis, (extent, stride)) in shape.iter().zip(&strides).enumerate() {
                    layout.dimensions[axis] = *extent as u64;
                    layout.strides_bytes[axis] = *stride as u64 * layout.element_bytes;
                }
                tensors.push(PreparedSemanticTensor {
                    layout,
                    logical_begin: 0,
                    logical_end: 0,
                    data: *view.device_ptr(),
                    native_allocation: view.allocation_provenance(),
                    source: Some(view),
                    _empty_owner: None,
                });
                digests.push(CapturedTensorDigest::Tensor {
                    cells: Arc::clone(&cells),
                    offset: index * 4,
                    producer_sealed: false,
                });
            }
            let content = TensorContentBuffers {
                tensors,
                seals: TensorContentSeals::Captured(digests),
                verification_inputs: Vec::new(),
            };
            let seal = self
                .provider
                .device()
                .inner()
                .get_func(
                    "xlog_semantic_transition",
                    "semantic_tensor_content_witness",
                )
                .ok_or_else(|| {
                    runtime_error("kernel lookup", "tensor content witness unavailable")
                })?;
            content.enqueue(&self.domain, &mut self.poisoned, &seal, false)?;
            Arc::new(SemanticEvaluationCohort {
                selected,
                content,
                domain,
                task,
                device,
                provider: Arc::clone(&self.provider),
            })
        };
        let bytes = event_capacity
            .checked_mul(size_of::<ModelWorkEvent>() + 3 * size_of::<u64>())
            .and_then(|bytes| bytes.checked_add(12 * size_of::<u64>()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let mut reservation = self
            .provider
            .memory()
            .reserve_bytes(bytes as u64)
            .map_err(|error| runtime_error("evaluation work reservation", error))?;
        let work = PreparedModelWork::allocate(&self.provider, &mut reservation, event_capacity)?;
        let report = reservation
            .alloc(12)
            .map_err(|error| runtime_error("evaluation report allocation", error))?;
        let handle = SemanticModelEvaluation {
            issuer: Arc::clone(&self.publication_issuer),
            token: lease.token,
            cohort: Arc::clone(&cohort),
        };
        self.steps
            .get_mut(&lease.token)
            .expect("checked acquired step")
            .evaluation = Some(EvaluationStorage {
            cohort,
            work,
            report,
            source_material,
            stream: self.stream.cu_stream() as u64,
            submitted: false,
            result: None,
        });
        self.guard_evaluation_cohort(&handle)?;
        let work = &self.steps[&lease.token]
            .evaluation
            .as_ref()
            .expect("issued evaluation")
            .work;
        work.reset_slots(&self.domain, &mut self.poisoned, 0, event_capacity)?;
        Ok(handle)
    }

    fn evaluation(
        &self,
        handle: &SemanticModelEvaluation,
    ) -> Result<&EvaluationStorage, SemanticTransitionError> {
        self.ensure_quiescent()?;
        let evaluation = self
            .steps
            .get(&handle.token)
            .and_then(|step| step.evaluation.as_ref())
            .ok_or_else(|| {
                publication_input_error("evaluation has no retained original invocation")
            })?;
        if !Arc::ptr_eq(&handle.issuer, &self.publication_issuer)
            || !Arc::ptr_eq(&handle.cohort, &evaluation.cohort)
        {
            return Err(publication_input_error(
                "evaluation belongs to another Session or original cohort",
            ));
        }
        Ok(evaluation)
    }

    fn guard_evaluation_cohort(
        &mut self,
        handle: &SemanticModelEvaluation,
    ) -> Result<(), SemanticTransitionError> {
        let cohort = Arc::clone(&self.evaluation(handle)?.cohort);
        let seal = self
            .provider
            .device()
            .inner()
            .get_func(
                "xlog_semantic_transition",
                "semantic_tensor_content_witness",
            )
            .ok_or_else(|| runtime_error("kernel lookup", "tensor content witness unavailable"))?;
        cohort
            .content
            .enqueue(&self.domain, &mut self.poisoned, &seal, true)
    }

    pub fn evaluation_stream(
        &self,
        handle: &SemanticModelEvaluation,
    ) -> Result<u64, SemanticTransitionError> {
        Ok(self.evaluation(handle)?.stream)
    }

    pub fn evaluation_training_view_port(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticModelEvaluation,
        port: SemanticTrainingViewPort,
        stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.checked_evaluation_parent(lease, handle)?;
        let (view, shape, strides, dtype) = self.evaluation(handle)?.cohort.selected.port(port)?;
        self.export_step_view(lease, view, shape, strides, dtype, stream)
    }

    fn checked_evaluation_parent(
        &self,
        lease: &SemanticPublishedLease,
        handle: &SemanticModelEvaluation,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_reader(lease)?;
        if lease.token != handle.token || self.evaluation(handle)?.result.is_some() {
            return Err(publication_input_error(
                "evaluation is closed or belongs to another acquired parent",
            ));
        }
        Ok(())
    }

    pub fn evaluation_model_work_buffer(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticModelEvaluation,
        stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.checked_evaluation_parent(lease, handle)?;
        let evaluation = self.evaluation(handle)?;
        if stream != evaluation.stream
            || evaluation.work.evaluation_recording
            || evaluation.submitted
        {
            return Err(publication_input_error(
                "evaluation work scratch requires its cold original stream",
            ));
        }
        // SAFETY: the entire padding-free bank was initialized by the cold reset.
        let view = unsafe { evaluation.work.actual.view().cast::<u8>() }
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let capacity = (evaluation.work.actual.len() / 3) as i64;
        self.export_step_view(lease, view, vec![capacity, 3], vec![3, 1], (1, 64), stream)
    }

    pub fn begin_model_evaluation(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticModelEvaluation,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_evaluation_parent(lease, handle)?;
        let evaluation = self.evaluation(handle)?;
        if evaluation.submitted
            || evaluation.work.evaluation_recording
            || evaluation.work.recording.frozen_bound().is_some()
        {
            return Err(publication_input_error(
                "evaluation recording begins exactly once",
            ));
        }
        self.guard_evaluation_cohort(handle)?;
        self.steps
            .get_mut(&handle.token)
            .expect("checked invocation")
            .evaluation
            .as_mut()
            .expect("checked evaluation")
            .work
            .evaluation_recording = true;
        Ok(())
    }

    pub fn record_evaluation_model_work(
        &mut self,
        handle: &SemanticModelEvaluation,
        kind: ModelWorkKind,
        dimensions: &[u64],
        device_produced: bool,
    ) -> Result<usize, SemanticTransitionError> {
        self.evaluation(handle)?;
        let evaluation = self
            .steps
            .get_mut(&handle.token)
            .expect("checked invocation")
            .evaluation
            .as_mut()
            .expect("checked evaluation");
        let work = &mut evaluation.work;
        let slot = work.next_slot().map_err(publication_input_error)?;
        let event = if device_produced {
            let actual = work
                .actual
                .device_ptr_value()
                .checked_add((slot * 3 * size_of::<u64>()) as u64)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            ModelWorkEvent::device_operation(kind, dimensions, actual)
        } else {
            ModelWorkEvent::operation(kind, dimensions)
        }
        .map_err(publication_input_error)?;
        work.record_event(event).map_err(publication_input_error)?;
        if device_produced {
            work.reset_slots(&self.domain, &mut self.poisoned, slot, 1)?;
        }
        Ok(slot)
    }

    /// Same original work registrar and stream as the actual read-only model.
    pub fn record_evaluation_model_invocation(
        &mut self,
        handle: &SemanticModelEvaluation,
    ) -> Result<(), SemanticTransitionError> {
        self.evaluation(handle)?;
        let result = self
            .steps
            .get_mut(&handle.token)
            .expect("checked invocation")
            .evaluation
            .as_mut()
            .expect("checked evaluation")
            .work
            .record_invocation(&self.domain, &mut self.poisoned);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    /// Freeze the same registrar, validate its genuine output witness, then
    /// observe only after stream completion and exact full-publication recheck.
    /// A failed completion retains every buffer in the Session and quarantines it.
    pub fn finish_model_evaluation(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticModelEvaluation,
        witness: &SemanticTensorContentWitness,
        streams: &[u64],
    ) -> Result<SemanticModelEvaluationResult, SemanticTransitionError> {
        self.checked_evaluation_parent(lease, handle)?;
        self.checked_content_witness(lease, witness)?;
        let output = &self.steps[&lease.token].content[witness.index];
        if output.tensors.len() != 3 {
            return Err(publication_input_error("evaluation output witness requires six FP32 losses, one Bool admissibility and the original UInt8 slab"));
        }
        let losses = output.tensors[0].clone();
        let admissible = output.tensors[1].clone();
        let slab = &output.tensors[2];
        let vector = |tensor: &PreparedSemanticTensor, scalar, bytes, extent| {
            tensor.layout.role == 0
                && tensor.layout.scalar_type == scalar
                && tensor.layout.element_bytes == bytes
                && tensor.layout.rank == 1
                && tensor.layout.dimensions == [extent, 0, 0, 0]
                && tensor.layout.strides_bytes == [bytes, 0, 0, 0]
                && tensor.layout.logical_axis == u64::MAX
                && tensor.logical_begin == 0
                && tensor.logical_end == 0
        };
        if losses.layout.index != 0
            || admissible.layout.index != 1
            || slab.layout.index != 2
            || !vector(&losses, 6, 4, 6)
            || !vector(&admissible, 8, 1, 1)
            || !vector(slab, 1, 1, slab.layout.dimensions[0])
            || slab.layout.dimensions[0] == 0
        {
            return Err(publication_input_error(
                "evaluation outputs changed their original complete typed geometry",
            ));
        }
        let external_bytes = accounted_tensor_allocations(&[&losses, &admissible], slab)?;
        let slab_bytes = tensor_layout_bytes(&slab.layout)?;
        for allocation in &self
            .publication
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .allocations
        {
            if step_input_overlap(
                slab.data,
                slab_bytes,
                allocation.device_ptr_value(),
                allocation.len(),
            )? {
                return Err(publication_input_error(
                    "evaluation slab aliases original publication storage",
                ));
            }
        }
        let stream = self.evaluation(handle)?.stream;
        self.enqueue_tensor_content(lease, witness, stream, true)?;
        self.guard_evaluation_cohort(handle)?;
        let cohort_provider = &self.evaluation(handle)?.cohort.provider;
        let retained_source_bytes = if Arc::ptr_eq(cohort_provider, &self.provider) {
            0
        } else {
            cohort_provider.memory().allocated_bytes()
        };
        let peak = self
            .provider
            .memory()
            .allocated_bytes()
            .checked_add(retained_source_bytes)
            .and_then(|bytes| bytes.checked_add(external_bytes))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let kernel = self
            .provider
            .device()
            .inner()
            .get_func(
                "xlog_semantic_transition",
                "semantic_model_evaluation_result",
            )
            .ok_or_else(|| runtime_error("kernel lookup", "model evaluation result unavailable"))?;
        let evaluation = self
            .steps
            .get_mut(&handle.token)
            .expect("checked invocation")
            .evaluation
            .as_mut()
            .expect("checked evaluation");
        if !evaluation.work.evaluation_recording || evaluation.submitted {
            return Err(publication_input_error(
                "evaluation result requires its original open recording",
            ));
        }
        evaluation
            .work
            .recording
            .require_model_invocations()
            .map_err(publication_input_error)?;
        evaluation
            .work
            .recording
            .freeze()
            .map_err(publication_input_error)?;
        evaluation.work.evaluation_recording = false;
        let work = &evaluation.work;
        let mut destination = work.device.view().slice(..work.recording.events().len());
        self.provider
            .htod_launch_metadata_sync_copy_into(work.recording.events(), &mut destination)
            .map_err(|error| runtime_error("evaluation work metadata upload", error))?;
        let input = work.descriptor();
        let selected = evaluation.cohort.selected.selection();
        let mut recorder = self.domain.new_strict_recorder();
        work.record_reads(&mut recorder);
        recorder.read(&selected);
        recorder.read(
            losses
                .source
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?,
        );
        recorder.read(
            admissible
                .source
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?,
        );
        recorder.write(&evaluation.report);
        evaluation.submitted = true;
        let arguments = (
            input.events,
            input.count,
            input.bound,
            *selected.device_ptr(),
            losses.data,
            admissible.data,
            peak,
            evaluation.report.device_ptr_value(),
        );
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |enqueue| {
            // SAFETY: all typed inputs and the complete native result are retained
            // by this invocation; the kernel writes exactly twelve U64 words.
            unsafe {
                kernel.clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    arguments,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })?;
        self.complete_step_consumers(lease, streams)?;
        let report = self.steps[&handle.token]
            .evaluation
            .as_ref()
            .expect("retained evaluation")
            .report
            .view();
        let words = self.publication_read(report)?;
        let material =
            Identity256::from_bytes(Sha256::digest(self.published_state_material(lease)?).into());
        let evaluation = self
            .steps
            .get_mut(&handle.token)
            .expect("retained invocation")
            .evaluation
            .as_mut()
            .expect("retained evaluation");
        if material != evaluation.source_material {
            self.poisoned = true;
            return Err(publication_input_error(
                "read-only evaluation changed the full original publication",
            ));
        }
        let result = SemanticModelEvaluationResult {
            status: words[0],
            loss_bits: std::array::from_fn(|index| words[index + 1] as u32),
            model_work: words[7],
            operation_count: words[8],
            work_bound: words[9],
            retained_allocation_bytes: words[10],
            model_calls: words[11],
        };
        evaluation.result = Some(result);
        if !matches!(result.status, 0 | 1) {
            return Err(publication_input_error(
                "incomplete evaluation expenditure retains its original invocation; no completed numerical result is available",
            ));
        }
        Ok(result)
    }

    /// Cancellation joins actual consumers before allowing the original parent
    /// to be reused. Unknown completion never clears the retained invocation.
    pub fn cancel_model_evaluation(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticModelEvaluation,
        streams: &[u64],
    ) -> Result<(), SemanticTransitionError> {
        self.checked_evaluation_parent(lease, handle)?;
        if self
            .evaluation(handle)?
            .result
            .is_some_and(|result| !matches!(result.status, 0 | 1))
        {
            return Err(publication_input_error(
                "incomplete evaluation expenditure retains its original invocation; cancellation cannot reopen admission",
            ));
        }
        self.complete_step_consumers(lease, streams)?;
        self.guard_evaluation_cohort(handle)?;
        let material =
            Identity256::from_bytes(Sha256::digest(self.published_state_material(lease)?).into());
        if material != self.evaluation(handle)?.source_material {
            self.poisoned = true;
            return Err(publication_input_error(
                "cancelled evaluation changed the original publication",
            ));
        }
        self.steps
            .get_mut(&handle.token)
            .expect("completed invocation")
            .evaluation = None;
        Ok(())
    }

    pub(super) fn require_closed_evaluations(&self) -> Result<(), SemanticTransitionError> {
        if self.steps.values().any(|step| {
            step.evaluation.as_ref().is_some_and(|evaluation| {
                evaluation
                    .result
                    .is_none_or(|result| !matches!(result.status, 0 | 1))
            })
        }) {
            return Err(publication_input_error(
                "state-changing work or release cannot interrupt an original read-only evaluation",
            ));
        }
        Ok(())
    }
}
