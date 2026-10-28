//! Frozen rule libraries and invocation-bound restrictions of the same action grammar.
use super::*;

pub(super) const RULE_CONTROL_MAGIC: u64 = 0x584c4f4752435452;
pub(super) const RULE_ACTIVITY_MAGIC: u64 = 0x584c4f4752414354;
pub(super) const RULE_LIBRARY_COUNT: usize = 32;

pub(super) struct FrozenRuleLibraryFrame {
    pub(super) declaration_begin: usize,
    pub(super) declaration_bytes: usize,
    pub(super) rule_word_begin: usize,
    pub(super) digest: Identity256,
    pub(super) binding: Identity256,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum SemanticEvaluationIntervention {
    NoEdit = 0,
    FixedRuleActivation = 1,
    Structural = 3,
}

impl SemanticEvaluationIntervention {
    pub fn from_code(code: u64) -> Result<Self, SemanticTransitionError> {
        match code {
            0 => Ok(Self::NoEdit),
            1 => Ok(Self::FixedRuleActivation),
            3 => Ok(Self::Structural),
            _ => Err(publication_input_error(
                "unknown evaluation rule intervention",
            )),
        }
    }
}

/// Exact canonical declaration and its typed lowering. Neither is executable
/// evidence until an original invocation selects its rule control.
pub struct SemanticEvaluationRuleLibrary {
    declaration: PublicationMaterialBytes,
    digest: Identity256,
    binding: Identity256,
    rules: Vec<crate::SemanticProgramRule>,
    choices: Vec<[u32; 18]>,
    _reservation: Arc<SemanticReplayMaterialReservation>,
}

impl std::fmt::Debug for SemanticEvaluationRuleLibrary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemanticEvaluationRuleLibrary")
            .field("digest", &self.digest)
            .field("binding", &self.binding)
            .field("rules", &self.rules.len())
            .finish()
    }
}

impl SemanticEvaluationRuleLibrary {
    fn validate_grammar(rule: crate::SemanticProgramRule) -> Result<(), SemanticTransitionError> {
        if rule.body_count != 2
            || rule.body[2] != crate::SemanticProgramAtom::default()
            || [&rule.head, &rule.body[0], &rule.body[1]]
                .iter()
                .any(|atom| {
                    atom.arity != 2
                        || atom
                            .terms
                            .iter()
                            .any(|term| term.kind != 0 || term.value >= 4)
                })
        {
            return Err(publication_input_error(
                "fixed rule library changes the frozen binary variable grammar",
            ));
        }
        let bound = rule.body[..2]
            .iter()
            .flat_map(|atom| atom.terms)
            .fold(0u8, |bound, term| bound | (1 << term.value));
        if rule
            .head
            .terms
            .iter()
            .any(|term| bound & (1 << term.value) == 0)
        {
            return Err(publication_input_error(
                "fixed rule library contains an unbound head variable",
            ));
        }
        Ok(())
    }

    pub(super) fn from_codebook_frame(
        bytes: &[u8],
        budget: &SemanticReplayMaterialBudget,
    ) -> Result<Option<Arc<Self>>, SemanticTransitionError> {
        if !bytes.len().is_multiple_of(8) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let word =
            |index: usize| u64::from_le_bytes(bytes[index * 8..index * 8 + 8].try_into().unwrap());
        let Some(frame) = validate_frozen_codebook_frame(bytes.len() / 8, word)? else {
            return Ok(None);
        };
        let temporary = budget.reserve_auxiliary(
            (RULE_LIBRARY_COUNT * size_of::<crate::SemanticProgramRule>()) as u64,
        )?;
        let declaration = budget
            .reserve_owned(
                frame.declaration_bytes as u64,
                frame.declaration_bytes as u64,
            )?
            .retain_owned_payload(
                bytes[frame.declaration_begin..frame.declaration_begin + frame.declaration_bytes]
                    .to_vec(),
            )?
            .bytes;
        let mut rules = Vec::with_capacity(RULE_LIBRARY_COUNT);
        for rule in 0..RULE_LIBRARY_COUNT {
            let values: [u32; 25] =
                std::array::from_fn(|index| word(frame.rule_word_begin + rule * 25 + index) as u32);
            let atom = |offset: usize| crate::SemanticProgramAtom {
                predicate: values[offset],
                arity: values[offset + 1],
                terms: [
                    crate::SemanticProgramTerm {
                        kind: values[offset + 2],
                        value: values[offset + 3],
                    },
                    crate::SemanticProgramTerm {
                        kind: values[offset + 4],
                        value: values[offset + 5],
                    },
                ],
            };
            rules.push(crate::SemanticProgramRule {
                body_count: values[0],
                head: atom(1),
                body: [atom(7), atom(13), atom(19)],
            });
        }
        let library = Self::retain(declaration, frame.digest, rules, None, budget)?;
        if library.binding != frame.binding {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        drop(temporary);
        Ok(Some(library))
    }
    pub fn digest(&self) -> Identity256 {
        self.digest
    }
    pub fn binding(&self) -> Identity256 {
        self.binding
    }
    pub fn declaration(&self) -> &[u8] {
        &self.declaration
    }
    pub fn rules(&self) -> &[crate::SemanticProgramRule] {
        &self.rules
    }

    fn category(
        codebooks: &ActionCodebooks,
        field: usize,
        index: usize,
    ) -> Result<u32, SemanticTransitionError> {
        let table = if field == 2 { 0 } else { 1 };
        let begin = codebooks.words[29 + 2 * table];
        let end = codebooks.words[30 + 2 * table];
        begin
            .checked_add(index as u64)
            .filter(|value| *value < end)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or(SemanticTransitionError::ObservationMismatch)
    }

    fn rule_choices(
        codebooks: &ActionCodebooks,
        rule: crate::SemanticProgramRule,
    ) -> Result<[u32; 18], SemanticTransitionError> {
        if rule.body_count != 2 || rule.body[2] != crate::SemanticProgramAtom::default() {
            return Err(publication_input_error(
                "fixed rule library requires canonical binary two-body rules",
            ));
        }
        let mut choices = [0; 18];
        choices[0] = 1;
        choices[1] = 2;
        let mut bound = 0u8;
        for (index, atom) in [&rule.head, &rule.body[0], &rule.body[1]]
            .into_iter()
            .enumerate()
        {
            if atom.arity != 2
                || atom
                    .terms
                    .iter()
                    .any(|term| term.kind != 0 || term.value >= 4)
            {
                return Err(publication_input_error(
                    "fixed rule library changes the frozen binary variable grammar",
                ));
            }
            let entries = if index == 0 {
                &codebooks.targets
            } else {
                &codebooks.operands
            };
            let field = 2 + index;
            let entry = entries.iter().position(|entry| matches!(entry, SemanticActionDescriptor::RulePredicate { predicate, arity } if *predicate == atom.predicate && *arity == 2)).ok_or(SemanticTransitionError::ObservationMismatch)?;
            choices[field] = Self::category(codebooks, field, entry)?;
            for (argument, term) in atom.terms.iter().enumerate() {
                let field = 5 + 2 * index + argument;
                let entry = codebooks.operands.iter().position(|entry| matches!(entry, SemanticActionDescriptor::RuleVariable { index } if *index == term.value)).ok_or(SemanticTransitionError::ObservationMismatch)?;
                choices[field] = Self::category(codebooks, field, entry)?;
                if index != 0 {
                    bound |= 1 << term.value;
                }
            }
        }
        if rule
            .head
            .terms
            .iter()
            .any(|term| bound & (1 << term.value) == 0)
        {
            return Err(publication_input_error(
                "fixed rule library contains an unbound head variable",
            ));
        }
        Ok(choices)
    }

    fn retain(
        declaration: PublicationMaterialBytes,
        digest: Identity256,
        rules: Vec<crate::SemanticProgramRule>,
        codebooks: Option<&ActionCodebooks>,
        budget: &SemanticReplayMaterialBudget,
    ) -> Result<Arc<Self>, SemanticTransitionError> {
        if rules.len() != RULE_LIBRARY_COUNT {
            return Err(publication_input_error(
                "fixed rule library requires exactly 32 ordered rules",
            ));
        }
        let extent = rules
            .capacity()
            .checked_mul(size_of::<crate::SemanticProgramRule>())
            .and_then(|bytes| {
                bytes.checked_add(
                    RULE_LIBRARY_COUNT * size_of::<[u32; 18]>()
                        + size_of::<Self>()
                        + size_of::<SemanticReplayMaterialReservation>()
                        + 4 * size_of::<usize>(),
                )
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = Arc::new(budget.reserve_auxiliary(extent as u64)?);
        let mut choices = Vec::with_capacity(RULE_LIBRARY_COUNT);
        let mut binding = Sha256::new();
        binding.update(b"xlog.semantic.fixed-rule-library.v1\0");
        binding.update(digest.as_bytes());
        for (ordinal, &rule) in rules.iter().enumerate() {
            Self::validate_grammar(rule)?;
            let words = semantic_program_rule_words(rule);
            if rules[..ordinal].iter().any(|other| *other == rule) {
                return Err(publication_input_error(
                    "fixed rule library repeats an ordered rule",
                ));
            }
            for word in words {
                binding.update(word.to_le_bytes());
            }
            if let Some(codebooks) = codebooks {
                choices.push(Self::rule_choices(codebooks, rule)?);
            }
        }
        Ok(Arc::new(Self {
            declaration,
            digest,
            binding: Identity256::from_bytes(binding.finalize().into()),
            rules,
            choices,
            _reservation: reservation,
        }))
    }

    pub(super) fn for_codebooks(
        &self,
        codebooks: &ActionCodebooks,
        budget: &SemanticReplayMaterialBudget,
    ) -> Result<Arc<Self>, SemanticTransitionError> {
        let reservation = budget.reserve_auxiliary(
            (self.rules.len() * size_of::<crate::SemanticProgramRule>()) as u64,
        )?;
        let rules = self.rules.clone();
        let result = Self::retain(
            self.declaration.clone(),
            self.digest,
            rules,
            Some(codebooks),
            budget,
        );
        drop(reservation);
        result
    }

    pub(super) fn rebase(
        &self,
        offset: u32,
        codebooks: &ActionCodebooks,
        budget: &SemanticReplayMaterialBudget,
    ) -> Result<Arc<Self>, SemanticTransitionError> {
        let reservation = budget.reserve_auxiliary(
            (self.rules.len() * size_of::<crate::SemanticProgramRule>()) as u64,
        )?;
        let mut rules = self.rules.clone();
        for rule in &mut rules {
            for atom in std::iter::once(&mut rule.head).chain(rule.body[..2].iter_mut()) {
                atom.predicate = atom
                    .predicate
                    .checked_add(offset)
                    .ok_or(SemanticTransitionError::GenerationExhausted)?;
            }
        }
        let result = Self::retain(
            self.declaration.clone(),
            self.digest,
            rules,
            Some(codebooks),
            budget,
        );
        drop(reservation);
        result
    }

    pub(super) fn word_count(&self) -> usize {
        8 + self.rules.len() * (25 + 18)
    }
    pub(super) fn visit_control_words(
        &self,
        mode: SemanticEvaluationIntervention,
        mut visit: impl FnMut(u64),
    ) {
        for word in [RULE_CONTROL_MAGIC, 1, mode as u64, self.rules.len() as u64] {
            visit(word);
        }
        for word in identity_words(*self.binding.as_bytes()) {
            visit(word);
        }
        for &rule in &self.rules {
            for word in semantic_program_rule_words(rule) {
                visit(word);
            }
        }
        for choices in &self.choices {
            for &choice in choices {
                visit(u64::from(choice));
            }
        }
    }
}

pub struct SemanticEvaluationInterventionPreparation {
    pub(super) issuer: Arc<()>,
    pub(super) original: Identity256,
    pub(super) identity: Identity256,
    pub(super) mode: SemanticEvaluationIntervention,
    pub(super) words: Vec<u64>,
    pub(super) active: AtomicBool,
    _reservation: SemanticReplayMaterialReservation,
}

impl SemanticEvaluationInterventionPreparation {
    pub fn task_identity(&self) -> Identity256 {
        self.identity
    }
    pub fn intervention(&self) -> SemanticEvaluationIntervention {
        self.mode
    }
}

impl SemanticTransitionSession {
    pub fn activate_evaluation_intervention(
        &mut self,
        preparation: &SemanticEvaluationForkPreparation,
    ) -> Result<(), SemanticTransitionError> {
        self.ensure_rebindable()?;
        if !Arc::ptr_eq(&preparation.issuer, &self.publication_issuer) {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let mode = preparation.origin.intervention();
        if let Some(control) = &preparation.intervention {
            let device = &self
                .task
                .as_ref()
                .ok_or(SemanticTransitionError::NotBound)?
                .1;
            if !Arc::ptr_eq(&control.issuer, &self.publication_issuer)
                || control.original != preparation.original_task
                || control.identity != preparation.target_task
                || Some(control.mode) != mode
                || control.words.len() != device.len()
                || control.words.get(2..6)
                    != Some(identity_words(*control.identity.as_bytes()).as_slice())
            {
                return Err(SemanticTransitionError::ObservationMismatch);
            }
        } else if mode.is_some() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        #[cfg(feature = "semantic-policy")]
        let mut staged = preparation
            .program_assignments
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let binding = &self
            .task
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .0;
        if preparation.active.load(Ordering::Acquire)
            && binding.identity() == preparation.target_task
            && binding.spec.evaluation_intervention == mode
        {
            return Ok(());
        }
        if preparation.active.load(Ordering::Acquire)
            || preparation
                .intervention
                .as_ref()
                .is_some_and(|control| control.active.load(Ordering::Acquire))
            || binding.identity() != preparation.original_task
            || binding.spec.evaluation_intervention.is_some()
            || binding.identity_with_intervention(mode) != preparation.target_task
            || binding
                .spec
                .rule_library
                .as_ref()
                .map(|library| library.binding())
                != preparation.origin.rule_library_binding()
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        #[cfg(feature = "semantic-policy")]
        if let Some(source) = staged.as_ref() {
            source.require_target(self)?;
        }
        self.task
            .as_mut()
            .expect("validated original task")
            .0
            .spec
            .evaluation_intervention = mode;
        #[cfg(feature = "semantic-policy")]
        if let Some(source) = staged.take() {
            source.install(self);
        }
        if let Some(control) = &preparation.intervention {
            control.active.store(true, Ordering::Release);
        }
        preparation.active.store(true, Ordering::Release);
        Ok(())
    }

    pub fn bind_evaluation_rule_library(
        &mut self,
        declaration: SemanticReplayLoadedMaterial,
        expected_digest: Identity256,
        rules: Vec<crate::SemanticProgramRule>,
    ) -> Result<Arc<SemanticEvaluationRuleLibrary>, SemanticTransitionError> {
        self.ensure_rebindable()?;
        if self.cold_content.is_some()
            || self.task.is_some()
            || self.publication.is_some()
            || self.codebooks.model_catalogue.is_some()
            || self.rule_library.is_some()
        {
            return Err(publication_input_error(
                "rule library must bind before the original cold observation and model catalogue",
            ));
        }
        let budget = self
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?;
        declaration.require_budget(budget)?;
        let bytes = declaration.as_bytes();
        if std::str::from_utf8(bytes).is_err()
            || bytes.last() != Some(&b'\n')
            || bytes
                .split(|byte| *byte == b'\n')
                .filter(|line| !line.is_empty())
                .count()
                != RULE_LIBRARY_COUNT
            || Identity256::from_bytes(Sha256::digest(bytes).into()) != expected_digest
        {
            return Err(publication_input_error(
                "rule library declaration changes its exact ordered canonical bytes",
            ));
        }
        let library = SemanticEvaluationRuleLibrary::retain(
            declaration.bytes.copy_owned()?,
            expected_digest,
            rules,
            Some(&self.codebooks),
            budget,
        )?;
        self.rule_library = Some(Arc::clone(&library));
        Ok(library)
    }

    pub fn evaluation_rule_library(&self) -> Option<Arc<SemanticEvaluationRuleLibrary>> {
        self.rule_library.clone().or_else(|| {
            self.codebooks
                .model_catalogue
                .as_ref()
                .and_then(|(catalogue, member)| catalogue.member_rule_library(*member))
        })
    }

    pub fn evaluation_intervention(&self) -> Option<(SemanticEvaluationIntervention, Identity256)> {
        self.task.as_ref().and_then(|(binding, _)| {
            binding.spec.evaluation_intervention.map(|mode| {
                (
                    mode,
                    binding
                        .spec
                        .rule_library
                        .as_ref()
                        .map_or(Identity256::default(), |library| library.digest()),
                )
            })
        })
    }

    pub fn prepare_evaluation_intervention(
        &self,
        mode: SemanticEvaluationIntervention,
    ) -> Result<Arc<SemanticEvaluationInterventionPreparation>, SemanticTransitionError> {
        self.ensure_rebindable()?;
        if self.publication.is_some()
            || self.initial_prefill.is_some()
            || self.captured.is_some()
            || self.evaluation_fork.is_some()
            || self
                .codebooks
                .model_catalogue
                .as_ref()
                .is_none_or(|(_, member)| *member < 2)
        {
            return Err(publication_input_error(
                "evaluation intervention requires its original newly imported frozen member",
            ));
        }
        let binding = &self
            .task
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .0;
        if binding.spec.evaluation_intervention.is_some() || binding.spec.rule_library.is_none() {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let bytes = binding
            .words_allocation_bytes()?
            .checked_add(
                (size_of::<SemanticEvaluationInterventionPreparation>() + 2 * size_of::<usize>())
                    as u64,
            )
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = self
            .replay_material_budget
            .as_ref()
            .ok_or(SemanticTransitionError::NotBound)?
            .reserve_auxiliary(bytes)?;
        Ok(Arc::new(SemanticEvaluationInterventionPreparation {
            issuer: Arc::clone(&self.publication_issuer),
            original: binding.identity(),
            identity: binding.identity_with_intervention(Some(mode)),
            mode,
            words: binding.words_with_intervention(self.graph.transition_arena()[1], Some(mode)),
            active: AtomicBool::new(false),
            _reservation: reservation,
        }))
    }
}
