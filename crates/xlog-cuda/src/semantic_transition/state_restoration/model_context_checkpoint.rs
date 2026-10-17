//! Checkpoint authority retained by the original complete restoration owner.

use super::*;

struct CheckpointContextFacts {
    issuer: Arc<()>,
    parent: SemanticPublishedIdentity,
    checkpoint: Identity256,
    context: SemanticFrozenModelContext,
    task: Identity256,
    epoch: u64,
    program: Arc<()>,
    program_identity: Identity256,
    history: Vec<SemanticLearningPhaseRecord>,
    _metadata: SemanticReplayMaterialReservation,
}

struct PrivateCheckpointOrigin {
    source: Arc<CheckpointContextFacts>,
    admission: SemanticLearningPhaseAdmission,
    public_lineage: Option<Arc<CheckpointContextFacts>>,
    _metadata: SemanticReplayMaterialReservation,
}

/// Original admitted learning owner shared only by its authenticated children.
#[derive(Clone)]
pub struct SemanticPrivateModelContextCheckpointOrigin {
    owner: Arc<PrivateCheckpointOrigin>,
}

impl SemanticPrivateModelContextCheckpointOrigin {
    pub fn requires_public_checkpoint_promotion(&self) -> bool {
        self.owner.public_lineage.is_some()
    }
}

/// One selected private save, issued after the binding consumes its active entry.
/// This capability is never exposed as a Python constructor or phase flag.
#[derive(Clone)]
pub struct SemanticPrivateModelContextCheckpointSave {
    origin: SemanticPrivateModelContextCheckpointOrigin,
    issuer: Arc<()>,
    restoration: Arc<()>,
    parent: SemanticPublishedIdentity,
}

pub(super) struct ModelContextCheckpointBoundary {
    facts: Arc<CheckpointContextFacts>,
    private: Option<SemanticPrivateModelContextCheckpointOrigin>,
    public_lineage: Option<Arc<CheckpointContextFacts>>,
    delivered: Option<(Identity256, SemanticPublishedIdentity)>,
    selected: Option<(Identity256, SemanticPublishedIdentity)>,
}

impl CheckpointContextFacts {
    fn capture(
        session: &SemanticTransitionSession,
        parent: &SemanticPublishedLease,
        checkpoint: Identity256,
    ) -> Result<Arc<Self>, SemanticTransitionError> {
        session.checked_reader(parent)?;
        let context = session
            .frozen_model_context()?
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let (task, _) = session
            .task
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let (program, program_identity) = session
            .actor_refresh_program
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .checkpoint_owner_binding();
        let bytes = size_of::<Self>()
            .checked_add(2 * size_of::<usize>())
            .and_then(|bytes| {
                bytes.checked_add(
                    session
                        .learning_phases
                        .len()
                        .checked_mul(size_of::<SemanticLearningPhaseRecord>())?,
                )
            })
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let metadata = session
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .reserve_auxiliary(bytes)?;
        let mut history = Vec::new();
        history
            .try_reserve_exact(session.learning_phases.len())
            .map_err(|_| {
                publication_input_error("checkpoint provenance history allocation failed")
            })?;
        history.extend_from_slice(&session.learning_phases);
        Ok(Arc::new(Self {
            issuer: Arc::clone(&session.publication_issuer),
            parent: parent.identity,
            checkpoint,
            context,
            task: task.identity(),
            epoch: session.task_epoch,
            program,
            program_identity,
            history,
            _metadata: metadata,
        }))
    }

    fn require_content(
        &self,
        session: &SemanticTransitionSession,
    ) -> Result<(), SemanticTransitionError> {
        let context = session
            .frozen_model_context()?
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let (task, _) = session
            .task
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        let (_, program) = session
            .actor_refresh_program
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .checkpoint_owner_binding();
        if !Arc::ptr_eq(&self.context.catalogue, &context.catalogue)
            || self.context.member != context.member
            || self.task != task.identity()
            || self.epoch != session.task_epoch
            || self.program_identity != program
            || !session.learning_phases.starts_with(&self.history)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }

    fn require_current(
        &self,
        session: &SemanticTransitionSession,
        parent: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        self.require_content(session)?;
        let (program, _) = session
            .actor_refresh_program
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .checkpoint_owner_binding();
        if !Arc::ptr_eq(&self.issuer, &session.publication_issuer)
            || !Arc::ptr_eq(&self.program, &program)
            || self.parent.instance != parent.identity.instance
            || session
                .publication
                .as_ref()
                .is_none_or(|storage| storage.instance != self.parent.instance)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(())
    }
}

impl SemanticTransitionSession {
    fn completed_context_checkpoint_restore(
        &self,
    ) -> Result<&Arc<OriginalStateMaterialRestore>, SemanticTransitionError> {
        let restoration = self
            .state_material_restore
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if !Arc::ptr_eq(&restoration.issuer, &self.publication_issuer)
            || restoration.pending.load(Ordering::Acquire)
            || !restoration.initialized.load(Ordering::Acquire)
            || self
                .publication
                .as_ref()
                .is_none_or(|storage| storage.instance != restoration.instance)
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let original = restoration
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if original.stage != StateRestoreStage::Completed
            || original.poisoned
            || original.retain_current_models
            || original.result.is_none()
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        drop(original);
        Ok(restoration)
    }

    /// Called by the trusted public importer only after model, authority and full
    /// checkpoint custody verification has completed. Membership alone cannot seal.
    pub fn seal_restored_model_context_checkpoint(
        &mut self,
        parent: &SemanticPublishedLease,
        checkpoint: Identity256,
    ) -> Result<(), SemanticTransitionError> {
        if self
            .frozen_model_context()?
            .is_none_or(|context| context.member != 1)
        {
            return Ok(());
        }
        self.checked_step(parent)?;
        self.checked_reader(parent)?;
        self.actor_refresh_program
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .require_completed_checkpoint_prefix()?;
        let restoration = Arc::clone(self.completed_context_checkpoint_restore()?);
        let mut original = restoration
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if original.transitioned
            || original.result != Some(parent.identity)
            || checkpoint == Identity256::default()
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if let Some(boundary) = &original.checkpoint_boundary {
            return if boundary.private.is_none()
                && boundary.delivered == Some((checkpoint, parent.identity))
            {
                boundary.facts.require_current(self, parent)
            } else {
                Err(SemanticTransitionError::ObservationMismatch)
            };
        }
        let facts = CheckpointContextFacts::capture(self, parent, checkpoint)?;
        original.checkpoint_boundary = Some(ModelContextCheckpointBoundary {
            facts: Arc::clone(&facts),
            private: None,
            public_lineage: Some(facts),
            delivered: Some((checkpoint, parent.identity)),
            selected: None,
        });
        Ok(())
    }

    /// Freeze the actual signed learning origin while its source reader is live.
    pub fn prepare_private_model_context_checkpoint_origin(
        &self,
        parent: &SemanticPublishedLease,
        native: &SemanticReplayStateMaterialProof,
        checkpoint: Identity256,
        admission: SemanticLearningPhaseAdmission,
    ) -> Result<Option<SemanticPrivateModelContextCheckpointOrigin>, SemanticTransitionError> {
        let Some(context) = self.frozen_model_context()? else {
            return Ok(None);
        };
        self.checked_original_consumer_step(parent)?;
        self.checked_reader(parent)?;
        native.require_frozen_model_context(Some(&context))?;
        self.require_checkpoint_source_digest(
            parent,
            native.original_identity(),
            native.source_digest(),
        )?;
        let budget = self
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        admission.require_budget(budget)?;
        let source = CheckpointContextFacts::capture(self, parent, checkpoint)?;
        let public_lineage = if context.member == 1 {
            self.require_model_context_checkpoint_attachment_presence(parent, false, None, None)?;
            if self.model_context_continuation.is_some() {
                Some(Arc::clone(&source))
            } else {
                Some(Arc::clone(
                    self.completed_context_checkpoint_restore()?
                        .original
                        .lock()
                        .map_err(|_| SemanticTransitionError::Poisoned)?
                        .checkpoint_boundary
                        .as_ref()
                        .and_then(|boundary| boundary.public_lineage.as_ref())
                        .ok_or(SemanticTransitionError::ObservationMismatch)?,
                ))
            }
        } else {
            None
        };
        let metadata = budget.reserve_auxiliary(
            u64::try_from(size_of::<PrivateCheckpointOrigin>() + 2 * size_of::<usize>())
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )?;
        Ok(Some(SemanticPrivateModelContextCheckpointOrigin {
            owner: Arc::new(PrivateCheckpointOrigin {
                source,
                admission,
                public_lineage,
                _metadata: metadata,
            }),
        }))
    }

    /// Bind a completed authentic private restore to its original learning owner.
    pub fn retain_private_model_context_checkpoint_origin(
        &mut self,
        parent: &SemanticPublishedLease,
        checkpoint: Identity256,
        origin: &SemanticPrivateModelContextCheckpointOrigin,
    ) -> Result<(), SemanticTransitionError> {
        origin.owner.source.require_content(self)?;
        origin.owner.admission.require_budget(
            self.replay_material_budget
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?,
        )?;
        self.checked_step(parent)?;
        self.checked_reader(parent)?;
        let restoration = Arc::clone(self.completed_context_checkpoint_restore()?);
        let mut original = restoration
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if original.result != Some(parent.identity) || checkpoint == Identity256::default() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if let Some(boundary) = &original.checkpoint_boundary {
            return if boundary
                .private
                .as_ref()
                .is_some_and(|prior| Arc::ptr_eq(&prior.owner, &origin.owner))
                && boundary.facts.checkpoint == checkpoint
            {
                boundary.facts.require_current(self, parent)
            } else {
                Err(SemanticTransitionError::ObservationMismatch)
            };
        }
        original.checkpoint_boundary = Some(ModelContextCheckpointBoundary {
            facts: CheckpointContextFacts::capture(self, parent, checkpoint)?,
            private: Some(origin.clone()),
            public_lineage: None,
            delivered: None,
            selected: None,
        });
        Ok(())
    }

    /// Mint only after the binding has consumed its exact active private save.
    pub fn private_model_context_checkpoint_save(
        &self,
        parent: &SemanticPublishedLease,
        origin: &SemanticPrivateModelContextCheckpointOrigin,
    ) -> Result<SemanticPrivateModelContextCheckpointSave, SemanticTransitionError> {
        let restoration = self.completed_context_checkpoint_restore()?;
        let original = restoration
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let boundary = original
            .checkpoint_boundary
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        boundary.facts.require_current(self, parent)?;
        if boundary.delivered.is_some()
            || boundary
                .private
                .as_ref()
                .is_none_or(|private| !Arc::ptr_eq(&private.owner, &origin.owner))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        Ok(SemanticPrivateModelContextCheckpointSave {
            origin: origin.clone(),
            issuer: Arc::clone(&self.publication_issuer),
            restoration: Arc::clone(&restoration.issuance),
            parent: parent.identity,
        })
    }

    pub(in crate::semantic_transition) fn require_restored_model_context_checkpoint_boundary(
        &self,
        parent: &SemanticPublishedLease,
        private: Option<&SemanticPrivateModelContextCheckpointSave>,
    ) -> Result<(), SemanticTransitionError> {
        let restoration = self.completed_context_checkpoint_restore()?;
        let original = restoration
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let boundary = original
            .checkpoint_boundary
            .as_ref()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        boundary.facts.require_current(self, parent)?;
        if let Some(private) = private {
            if !Arc::ptr_eq(&private.issuer, &self.publication_issuer)
                || !Arc::ptr_eq(&private.restoration, &restoration.issuance)
                || private.parent != parent.identity
                || boundary.delivered.is_some()
                || boundary
                    .private
                    .as_ref()
                    .is_none_or(|origin| !Arc::ptr_eq(&origin.owner, &private.origin.owner))
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
        } else {
            let lineage = boundary
                .public_lineage
                .as_ref()
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            lineage.require_content(self)?;
            if lineage.context.member != 1 || boundary.delivered.is_none() {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
            self.actor_refresh_program
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?
                .require_completed_checkpoint_prefix()?;
        }
        Ok(())
    }

    /// Retain the exact selected snapshot before its delivery accounting closes.
    /// Selection remains private until durable activation consumes this same owner.
    pub fn prepare_private_model_context_checkpoint_delivery(
        &mut self,
        parent: &SemanticPublishedLease,
        origin: &SemanticPrivateModelContextCheckpointOrigin,
        checkpoint: Identity256,
        native: &SemanticReplayStateMaterialProof,
    ) -> Result<(), SemanticTransitionError> {
        let Some(lineage) = &origin.owner.public_lineage else {
            return Ok(());
        };
        self.checked_step(parent)?;
        self.checked_reader(parent)?;
        self.require_checkpoint_source_digest(
            parent,
            native.original_identity(),
            native.source_digest(),
        )?;
        self.actor_refresh_program
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .require_completed_checkpoint_prefix()?;
        lineage.require_content(self)?;
        let restoration = self.completed_context_checkpoint_restore()?;
        let mut original = restoration
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let boundary = original
            .checkpoint_boundary
            .as_mut()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        boundary.facts.require_current(self, parent)?;
        if boundary
            .private
            .as_ref()
            .is_none_or(|private| !Arc::ptr_eq(&private.owner, &origin.owner))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        if let Some(selected) = boundary.selected {
            return if selected == (checkpoint, parent.identity) {
                Ok(())
            } else {
                Err(SemanticTransitionError::ObservationMismatch)
            };
        }
        if checkpoint == Identity256::default() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        boundary.selected = Some((checkpoint, parent.identity));
        Ok(())
    }

    /// Promote only after the trusted binding verifies durable delivery and the
    /// identical selected learning owners. No allocation or model work is added.
    pub fn promote_private_model_context_checkpoint(
        &mut self,
        parent: &SemanticPublishedLease,
        origin: &SemanticPrivateModelContextCheckpointOrigin,
    ) -> Result<(), SemanticTransitionError> {
        // Initial SOURCE rules remain authoritative; private work never fabricates
        // a dormant continuation or turns a SOURCE snapshot into public authority.
        let Some(lineage) = &origin.owner.public_lineage else {
            return Ok(());
        };
        self.checked_step(parent)?;
        self.checked_reader(parent)?;
        lineage.require_content(self)?;
        let restoration = self.completed_context_checkpoint_restore()?;
        let mut original = restoration
            .original
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let boundary = original
            .checkpoint_boundary
            .as_mut()
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        boundary.facts.require_current(self, parent)?;
        if boundary
            .private
            .as_ref()
            .is_none_or(|private| !Arc::ptr_eq(&private.owner, &origin.owner))
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let selected = boundary
            .selected
            .filter(|(_, selected)| *selected == parent.identity)
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        if let Some(delivered) = boundary.delivered {
            return if delivered == selected {
                Ok(())
            } else {
                Err(SemanticTransitionError::ObservationMismatch)
            };
        }
        boundary.public_lineage = Some(Arc::clone(lineage));
        boundary.delivered = Some(selected);
        Ok(())
    }
}
