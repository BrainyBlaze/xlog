//! Original read-only Program instruction custody and durable observations.
use super::*;
use serde_json::{json, Value};

#[derive(Clone)]
pub struct SemanticEvaluationInstructionAdmission {
    pub(super) inner: SemanticSegmentInstructionAdmission,
}

#[derive(Clone, Copy)]
pub(super) enum EvaluationInstructionOutcome {
    Observed(SemanticModelEvaluationResult),
    Cancelled {
        model_work: u64,
        model_calls: u64,
        capture: SemanticCancelledEvaluationCapture,
    },
}

#[derive(Clone, Copy)]
pub(super) struct EvaluationInstructionObservation {
    pub(super) outcome: EvaluationInstructionOutcome,
    pub(super) backing_peak: Option<u64>,
}

fn read_word(reader: &mut SemanticMaterialReader<'_>) -> Result<u64, SemanticTransitionError> {
    reader.u64().map_err(SemanticTransitionError::Semantic)
}

impl EvaluationInstructionObservation {
    pub(super) fn write(&self, bytes: &mut Vec<u8>) {
        match &self.outcome {
            EvaluationInstructionOutcome::Observed(result) => {
                material_u64(bytes, 0);
                material_u64(bytes, result.status);
                for bits in result.loss_bits {
                    material_u64(bytes, u64::from(bits));
                }
                for value in [
                    result.model_work,
                    result.operation_count,
                    result.work_bound,
                    result.retained_allocation_bytes,
                    result.model_calls,
                ] {
                    material_u64(bytes, value);
                }
            }
            EvaluationInstructionOutcome::Cancelled {
                model_work,
                model_calls,
                capture,
            } => {
                material_u64(bytes, 1);
                material_u64(bytes, *model_work);
                material_u64(bytes, *model_calls);
                material_u64(
                    bytes,
                    match capture {
                        SemanticCancelledEvaluationCapture::NotEntered => 0,
                        SemanticCancelledEvaluationCapture::EnteredAborted => 1,
                        SemanticCancelledEvaluationCapture::RecordedAndCancelled => 2,
                    },
                );
            }
        }
        material_u64(bytes, u64::from(self.backing_peak.is_some()));
        if let Some(peak) = self.backing_peak {
            material_u64(bytes, peak);
        }
    }

    pub(super) fn allocation_bytes(&self) -> u64 {
        let outcome = match self.outcome {
            EvaluationInstructionOutcome::Observed(_) => 1 + 12,
            EvaluationInstructionOutcome::Cancelled { .. } => 1 + 3,
        };
        (outcome + 1 + u64::from(self.backing_peak.is_some())) * 8
    }

    pub(super) fn read(
        reader: &mut SemanticMaterialReader<'_>,
    ) -> Result<Self, SemanticTransitionError> {
        let outcome = match read_word(reader)? {
            0 => {
                let status = read_word(reader)?;
                let mut loss_bits = [0; 6];
                for bits in &mut loss_bits {
                    *bits = u32::try_from(read_word(reader)?)
                        .map_err(|_| SemanticTransitionError::ObservationMismatch)?;
                }
                let result = SemanticModelEvaluationResult {
                    status,
                    loss_bits,
                    model_work: read_word(reader)?,
                    operation_count: read_word(reader)?,
                    work_bound: read_word(reader)?,
                    retained_allocation_bytes: read_word(reader)?,
                    model_calls: read_word(reader)?,
                };
                if result.status > 1
                    || result.model_work > result.work_bound
                    || result.model_calls > 1
                    || result.status == 0
                        && (result.model_calls != 1
                            || result
                                .loss_bits
                                .iter()
                                .any(|bits| !f32::from_bits(*bits).is_finite()))
                {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                EvaluationInstructionOutcome::Observed(result)
            }
            1 => {
                let model_work = read_word(reader)?;
                let model_calls = read_word(reader)?;
                let capture = match read_word(reader)? {
                    0 => SemanticCancelledEvaluationCapture::NotEntered,
                    1 => SemanticCancelledEvaluationCapture::EnteredAborted,
                    2 => SemanticCancelledEvaluationCapture::RecordedAndCancelled,
                    _ => return Err(SemanticTransitionError::ObservationMismatch),
                };
                if model_calls > 1 {
                    return Err(SemanticTransitionError::ObservationMismatch);
                }
                EvaluationInstructionOutcome::Cancelled {
                    model_work,
                    model_calls,
                    capture,
                }
            }
            _ => return Err(SemanticTransitionError::ObservationMismatch),
        };
        let backing_peak = match read_word(reader)? {
            0 => None,
            1 => Some(read_word(reader)?),
            _ => return Err(SemanticTransitionError::ObservationMismatch),
        };
        Ok(Self {
            outcome,
            backing_peak,
        })
    }

    pub(super) fn resource_usage(
        &self,
        cold: SemanticColdModelWorkResult,
    ) -> Result<[u64; 3], SemanticTransitionError> {
        let (work, calls) = match self.outcome {
            EvaluationInstructionOutcome::Observed(result) => {
                (result.model_work, result.model_calls)
            }
            EvaluationInstructionOutcome::Cancelled {
                model_work,
                model_calls,
                ..
            } => (model_work, model_calls),
        };
        Ok([
            work.checked_add(cold.model_work)
                .and_then(|work| work.checked_add(cold.native_work))
                .ok_or(SemanticTransitionError::GenerationExhausted)?,
            self.backing_peak.ok_or_else(|| {
                publication_input_error(
                    "evaluation expense awaits its original backing observation",
                )
            })?,
            calls
                .checked_add(cold.model_calls)
                .ok_or(SemanticTransitionError::GenerationExhausted)?,
        ])
    }

    pub(super) fn refusal(
        &self,
        cold: SemanticColdModelWorkResult,
        budget: [u64; 3],
    ) -> Result<Option<&'static str>, SemanticTransitionError> {
        if matches!(self.outcome, EvaluationInstructionOutcome::Cancelled { .. }) {
            return Ok(Some("cancelled-evaluation"));
        }
        if matches!(self.outcome, EvaluationInstructionOutcome::Observed(result) if result.status != 0)
        {
            return Ok(Some("native-evaluation-refused"));
        }
        let actual = self.resource_usage(cold)?;
        for (index, reason) in [
            "evaluation-work-budget-exceeded",
            "evaluation-memory-budget-exceeded",
            "evaluation-model-calls-budget-exceeded",
        ]
        .into_iter()
        .enumerate()
        {
            if actual[index] > budget[index] {
                return Ok(Some(reason));
            }
        }
        Ok(None)
    }

    fn result_projection(&self) -> Value {
        match &self.outcome {
            EvaluationInstructionOutcome::Observed(result) => json!({
                "status": result.status, "loss_bits": result.loss_bits,
                "model_work": result.model_work, "model_calls": result.model_calls,
                "operation_count": result.operation_count, "work_bound": result.work_bound,
                "retained_allocation_bytes": result.retained_allocation_bytes,
            }),
            EvaluationInstructionOutcome::Cancelled {
                model_work,
                model_calls,
                capture,
            } => json!({
                "cancelled": true, "model_work": model_work, "model_calls": model_calls,
                "capture_disposition": match capture {
                    SemanticCancelledEvaluationCapture::NotEntered => "not-entered",
                    SemanticCancelledEvaluationCapture::EnteredAborted => "entered-aborted",
                    SemanticCancelledEvaluationCapture::RecordedAndCancelled => "recorded-and-cancelled",
                },
            }),
        }
    }
}

impl SemanticTransitionSession {
    pub fn prepare_evaluation_instruction_cohort(
        &mut self,
        admission: &SemanticEvaluationInstructionAdmission,
        parent: &SemanticPublishedLease,
        work: &SemanticColdModelWork,
    ) -> Result<Arc<SemanticEvaluationCohort>, SemanticTransitionError> {
        self.require_evaluation_instruction_cold_work(admission, parent, work)?;
        let (coordinates, _) = self.evaluation_instruction_info(admission)?;
        let native = self.cold_native_work(parent.token)?.ok_or_else(|| {
            publication_input_error("evaluation cohort lost its original cold tally")
        })?;
        self.prepare_original_family_cohorts(parent, Some(coordinates), native)?;
        self.prepared_family_cohort(coordinates)
    }

    pub fn bind_evaluation_instruction(
        &mut self,
        admission: &SemanticEvaluationInstructionAdmission,
        parent: &SemanticPublishedLease,
        evaluation: &SemanticModelEvaluation,
    ) -> Result<(), SemanticTransitionError> {
        self.require_segment_instruction_admission(&admission.inner, parent)?;
        let (coordinates, _) = self.evaluation_instruction_info(admission)?;
        self.require_model_evaluation_invocation(parent, evaluation)?;
        if !Arc::ptr_eq(
            &evaluation.cohort(),
            &self.prepared_family_cohort(coordinates)?,
        ) {
            return Err(publication_input_error(
                "evaluation changed its original frozen cohort",
            ));
        }
        let claim = self.instruction_admission_mut(&admission.inner)?;
        if let Some(original) = &claim.evaluation_invocation {
            return if original.same_invocation(evaluation) {
                Ok(())
            } else {
                Err(SemanticTransitionError::ObservationMismatch)
            };
        }
        if claim.state != actor_refresh_program::InstructionState::Admitted {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        claim.evaluation_invocation = Some(evaluation.clone());
        claim.state = actor_refresh_program::InstructionState::Bound;
        Ok(())
    }

    pub fn retire_evaluation_instruction_observation(
        &mut self,
        admission: &SemanticEvaluationInstructionAdmission,
        parent: &SemanticPublishedLease,
        evaluation: &SemanticModelEvaluation,
        completed: &SemanticCompletedModelEvaluation,
    ) -> Result<(), SemanticTransitionError> {
        self.require_segment_instruction_admission(&admission.inner, parent)?;
        self.evaluation_instruction_info(admission)?;
        let claim = self.instruction_admission(&admission.inner)?;
        if claim
            .evaluation_invocation
            .as_ref()
            .is_none_or(|original| !original.same_invocation(evaluation))
            || !completed.belongs_to_original(evaluation)
            || completed.parent() != parent.identity
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if claim.evaluation_result.is_some() {
            return Ok(());
        }
        self.retire_completed_evaluation_storage(parent)?;
        self.instruction_admission_mut(&admission.inner)?
            .evaluation_result = Some(EvaluationInstructionObservation {
            outcome: EvaluationInstructionOutcome::Observed(completed.result()),
            backing_peak: None,
        });
        Ok(())
    }

    pub fn retire_evaluation_instruction_cancellation(
        &mut self,
        admission: &SemanticEvaluationInstructionAdmission,
        parent: &SemanticPublishedLease,
        evaluation: &SemanticModelEvaluation,
        cancelled: &SemanticCancelledModelEvaluation,
    ) -> Result<(), SemanticTransitionError> {
        self.require_segment_instruction_admission(&admission.inner, parent)?;
        self.evaluation_instruction_info(admission)?;
        let claim = self.instruction_admission(&admission.inner)?;
        if claim
            .evaluation_invocation
            .as_ref()
            .is_none_or(|original| !original.same_invocation(evaluation))
            || !cancelled.belongs_to(self, evaluation)
            || cancelled.parent() != parent.identity
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if claim.evaluation_result.is_some() {
            return Ok(());
        }
        let step = self
            .steps
            .get(&parent.token)
            .ok_or(SemanticTransitionError::NotBound)?;
        if step.evaluation.is_some() || step.evaluation_preparation.is_some() {
            return Err(publication_input_error(
                "cancelled evaluation retains its original storage",
            ));
        }
        self.instruction_admission_mut(&admission.inner)?
            .evaluation_result = Some(EvaluationInstructionObservation {
            outcome: EvaluationInstructionOutcome::Cancelled {
                model_work: cancelled.model_work(),
                model_calls: cancelled.model_calls(),
                capture: cancelled.capture_disposition(),
            },
            backing_peak: None,
        });
        Ok(())
    }

    pub fn evaluation_instruction_observation(
        &self,
        admission: &SemanticEvaluationInstructionAdmission,
    ) -> Result<Option<Value>, SemanticTransitionError> {
        let (coordinates, budget) = self.evaluation_instruction_info(admission)?;
        let claim = self.instruction_admission(&admission.inner)?;
        if !self.evaluation_instruction_resources_retired(admission)? {
            return Ok(None);
        }
        let observation = claim
            .evaluation_result
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let cold = claim.cold_result.ok_or(SemanticTransitionError::NotBound)?;
        let frame = self
            .original_family_cohort(coordinates)?
            .ok_or(SemanticTransitionError::NotBound)?;
        Ok(Some(json!({
            "program_ordinal": self.admitted_segment_first_program_ordinal(&admission.inner)?,
            "result": observation.result_projection(),
            "resource_usage": observation.resource_usage(cold)?,
            "refusal": observation.refusal(cold, budget)?,
            "source_cohort": frame.projection()?,
            "resources_retired": true,
        })))
    }

    pub fn evaluation_instruction_cold_result(
        &self,
        admission: &SemanticEvaluationInstructionAdmission,
    ) -> Result<Option<SemanticColdModelWorkResult>, SemanticTransitionError> {
        self.evaluation_instruction_info(admission)?;
        Ok(self.instruction_admission(&admission.inner)?.cold_result)
    }
}

impl SemanticTransitionError {
    /// Classify only the original native completion uncertainty.
    pub fn native_completion_pending(&self) -> bool {
        cold_model_work::cold_completion_is_pending(self)
    }
}
