//! Original common-world ownership and the rule-cell state selected with a publication.
//!
//! This module owns storage and categorical cell identities. It does not choose a
//! probabilistic interpretation of a coefficient or manufacture event realizations.

use super::*;
use xlog_ir::semantic_world::{SemanticInitialRuleCells, SemanticWorldDeclaration};

const SOURCE_HEADER_WORDS: usize = 13;
const STATE_HEADER_WORDS: usize = 16;
const CELL_WORDS: usize = 6;

// Source header: format, world ID[4], seed ID[4], context count, original
// declaration byte extent, cell capacity, original cell count.
// State header: format, world ID[4], seed ID[4], context count, ledger generation,
// cell count/capacity, next birth low/high, qualified context. Context-local next
// rule indices follow it, then (context, rule, birth low/high, generation, code).

/// One immutable full-roster source retained by every qualified context. There is
/// no mutable current root, counter, or publication selector in this owner.
#[derive(Debug)]
pub struct SemanticWorldSource {
    declaration: Arc<SemanticWorldDeclaration>,
    programs: Arc<[Arc<crate::SemanticProgramAdmission>]>,
    material: Arc<[u8]>,
    identity: Identity256,
    seed_identity: Identity256,
    original_cells: usize,
    cell_capacity: usize,
}

/// A restriction of the original owner, rather than another semantic namespace.
#[derive(Clone, Debug)]
pub struct SemanticWorldContext {
    source: Arc<SemanticWorldSource>,
    ordinal: u32,
}

fn invalid_world(detail: &str) -> SemanticTransitionError {
    publication_input_error(detail)
}

fn word(bytes: &[u8], index: usize) -> u64 {
    u64::from_le_bytes(
        bytes[index * 8..index * 8 + 8]
            .try_into()
            .expect("checked world extent"),
    )
}

fn push_word(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

impl SemanticWorldSource {
    /// Retains the complete canonical declaration and its actual compiler outputs.
    /// The content identity is retained from that declaration's canonical producer.
    pub fn new(
        declaration: Arc<SemanticWorldDeclaration>,
        programs: Vec<Arc<crate::SemanticProgramAdmission>>,
        material: Arc<[u8]>,
        identity: Identity256,
        seed_identity: Identity256,
    ) -> Result<Arc<Self>, SemanticTransitionError> {
        declaration
            .validate()
            .map_err(|error| invalid_world(&error.to_string()))?;
        if programs.len() != declaration.contexts.len() {
            return Err(invalid_world(
                "common world compiler and context rosters differ",
            ));
        }
        let mut original_cells = 0usize;
        let mut mutation_capacity = None;
        for (context, program) in declaration.contexts.iter().zip(&programs) {
            program
                .validate()
                .map_err(|error| invalid_world(&error.to_string()))?;
            context
                .bind_program(
                    program.initial_facts.len(),
                    &program
                        .initial_rules
                        .iter()
                        .map(|rule| rule.body_count as usize)
                        .collect::<Vec<_>>(),
                    program.queries.len(),
                )
                .map_err(|error| invalid_world(&error.to_string()))?;
            let capacity = usize::try_from(program.image.mutation_capacity)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
            if mutation_capacity.is_some_and(|original| original != capacity) {
                return Err(invalid_world(
                    "common world images have different cumulative mutation ceilings",
                ));
            }
            mutation_capacity = Some(capacity);
            original_cells = original_cells
                .checked_add(program.initial_rules.len())
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
        }
        let cell_capacity = original_cells
            .checked_add(
                mutation_capacity
                    .ok_or_else(|| invalid_world("common world has no original context"))?,
            )
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        STATE_HEADER_WORDS
            .checked_add(programs.len())
            .and_then(|words| words.checked_add(CELL_WORDS.checked_mul(cell_capacity)?))
            .and_then(|words| words.checked_mul(8))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        Ok(Arc::new(Self {
            declaration,
            programs: programs.into(),
            material,
            identity,
            seed_identity,
            original_cells,
            cell_capacity,
        }))
    }

    /// The actual shared owner and its new compiler-owner array. Declaration,
    /// program contents, and source material retain their original allocations.
    pub fn allocation_bytes(&self) -> Result<usize, SemanticTransitionError> {
        size_of::<Self>()
            .checked_add(4 * size_of::<usize>())
            .and_then(|bytes| {
                bytes.checked_add(
                    self.programs
                        .len()
                        .checked_mul(size_of::<Arc<crate::SemanticProgramAdmission>>())?,
                )
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)
    }

    /// Creates only a qualified view of this same immutable owner.
    pub fn context(
        self: &Arc<Self>,
        ordinal: u32,
    ) -> Result<SemanticWorldContext, SemanticTransitionError> {
        if ordinal as usize >= self.declaration.contexts.len() {
            return Err(invalid_world(
                "semantic context is outside the original common roster",
            ));
        }
        Ok(SemanticWorldContext {
            source: Arc::clone(self),
            ordinal,
        })
    }

    fn source_bytes(&self) -> Vec<u8> {
        // The full immutable declaration remains in its original run owner.
        // Every checkpoint binds that explicit owner by its complete content ID;
        // it never instantiates or copies another declaration from saved state.
        let length = SOURCE_HEADER_WORDS * 8;
        let mut bytes = Vec::with_capacity(length);
        push_word(&mut bytes, 1);
        bytes.extend_from_slice(self.identity.as_bytes());
        bytes.extend_from_slice(self.seed_identity.as_bytes());
        push_word(&mut bytes, self.declaration.contexts.len() as u64);
        push_word(&mut bytes, self.material.len() as u64);
        push_word(&mut bytes, self.cell_capacity as u64);
        push_word(&mut bytes, self.original_cells as u64);
        bytes
    }

    fn initial_state(&self, context: u32) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(
            (STATE_HEADER_WORDS + self.programs.len() + CELL_WORDS * self.original_cells) * 8,
        );
        push_word(&mut bytes, 1);
        bytes.extend_from_slice(self.identity.as_bytes());
        bytes.extend_from_slice(self.seed_identity.as_bytes());
        push_word(&mut bytes, self.declaration.contexts.len() as u64);
        push_word(&mut bytes, 1); // Selected rule-cell ledger generation.
        push_word(&mut bytes, self.original_cells as u64);
        push_word(&mut bytes, self.cell_capacity as u64);
        push_word(&mut bytes, self.original_cells as u64 + 1); // Never-reused next birth, low word.
        push_word(&mut bytes, 0); // High word of the same u128 birth identity.
        push_word(&mut bytes, u64::from(context)); // This branch's qualified view.
        for program in self.programs.iter() {
            push_word(&mut bytes, program.initial_rules.len() as u64);
        }
        let mut birth = 1u64;
        for (context, (declaration, program)) in self
            .declaration
            .contexts
            .iter()
            .zip(self.programs.iter())
            .enumerate()
        {
            for rule in 0..program.initial_rules.len() {
                let code = match &declaration.rule_cells {
                    SemanticInitialRuleCells::Uniform(code) => *code,
                    SemanticInitialRuleCells::Codes(codes) => codes[rule],
                };
                for value in [context as u64, rule as u64, birth, 0, 1, u64::from(code)] {
                    push_word(&mut bytes, value);
                }
                birth += 1;
            }
        }
        bytes
    }

    fn initial_records(&self, context: u32) -> [SemanticStateRecord; 2] {
        let source = self.source_bytes();
        let state = self.initial_state(context);
        [
            SemanticStateRecord {
                role: SemanticStateRole::SemanticWorldSource,
                index: 0,
                capacity_bytes: source.len(),
                bytes: source,
            },
            SemanticStateRecord {
                role: SemanticStateRole::SemanticWorldState,
                index: 0,
                capacity_bytes: (STATE_HEADER_WORDS
                    + self.programs.len()
                    + CELL_WORDS * self.cell_capacity)
                    * 8,
                bytes: state,
            },
        ]
    }

    fn validate_state(
        &self,
        bytes: &[u8],
        capacity: usize,
        budget: &SemanticReplayMaterialBudget,
    ) -> Result<(), SemanticTransitionError> {
        let mismatch = || {
            invalid_world("selected common world rule-cell state differs from its original namespace or bounds")
        };
        if bytes.len() < (STATE_HEADER_WORDS + self.programs.len()) * 8
            || bytes.len() % 8 != 0
            || capacity
                != (STATE_HEADER_WORDS + self.programs.len() + CELL_WORDS * self.cell_capacity) * 8
            || word(bytes, 0) != 1
            || &bytes[8..40] != self.identity.as_bytes()
            || &bytes[40..72] != self.seed_identity.as_bytes()
            || word(bytes, 9) != self.declaration.contexts.len() as u64
            || word(bytes, 10) == 0
            || word(bytes, 12) != self.cell_capacity as u64
            || word(bytes, 15) >= self.declaration.contexts.len() as u64
        {
            return Err(mismatch());
        }
        let count = usize::try_from(word(bytes, 11)).map_err(|_| mismatch())?;
        if count > self.cell_capacity
            || count < self.original_cells
            || bytes.len() != (STATE_HEADER_WORDS + self.programs.len() + CELL_WORDS * count) * 8
        {
            return Err(mismatch());
        }
        let next_birth = (u128::from(word(bytes, 14)) << 64) | u128::from(word(bytes, 13));
        if next_birth == 0 {
            return Err(mismatch());
        }
        let mut target_count = 0u64;
        for (context, program) in self.programs.iter().enumerate() {
            let next = word(bytes, STATE_HEADER_WORDS + context);
            if next < program.initial_rules.len() as u64
                || next
                    > program.initial_rules.len() as u64 + program.image.mutation_capacity as u64
            {
                return Err(mismatch());
            }
            target_count = target_count.checked_add(next).ok_or_else(mismatch)?;
        }
        if target_count != count as u64 {
            return Err(mismatch());
        }
        let _births_owner = budget.reserve_ordered_index::<u128, ()>(count)?;
        let _targets_owner = budget.reserve_ordered_index::<(usize, u64), ()>(count)?;
        let mut births = BTreeSet::new();
        let mut targets = BTreeSet::new();
        let mut original = 0usize;
        for index in 0..count {
            let begin = STATE_HEADER_WORDS + self.programs.len() + CELL_WORDS * index;
            let context = usize::try_from(word(bytes, begin)).map_err(|_| mismatch())?;
            let program = self.programs.get(context).ok_or_else(mismatch)?;
            let rule = word(bytes, begin + 1);
            let birth =
                (u128::from(word(bytes, begin + 3)) << 64) | u128::from(word(bytes, begin + 2));
            if birth == 0
                || birth >= next_birth
                || word(bytes, begin + 4) == 0
                || !births.insert(birth)
                || !targets.insert((context, rule))
                || !(1..=32769).contains(&word(bytes, begin + 5))
                || rule >= word(bytes, STATE_HEADER_WORDS + context)
            {
                return Err(mismatch());
            }
            if rule < program.initial_rules.len() as u64 {
                original += 1;
            } else if rule
                >= program.initial_rules.len() as u64 + program.image.mutation_capacity as u64
            {
                return Err(mismatch());
            }
        }
        if original != self.original_cells {
            return Err(mismatch());
        }
        Ok(())
    }
}

impl SemanticWorldContext {
    pub(super) fn validate_queries(&self, count: usize) -> Result<(), SemanticTransitionError> {
        if self.source.declaration.contexts[self.ordinal as usize]
            .queries
            .len()
            != count
        {
            return Err(invalid_world(
                "semantic value query roster differs from its original common context",
            ));
        }
        Ok(())
    }

    pub(super) fn initial_records(&self) -> [SemanticStateRecord; 2] {
        self.source.initial_records(self.ordinal)
    }

    fn validate_source<B: PublicationByteStorage>(
        &self,
        ranges: &[PublicationMaterialRangeStorage<B>],
    ) -> Result<(), SemanticTransitionError> {
        let saved = ranges
            .iter()
            .find(|range| {
                range.range.role == SemanticStateRole::SemanticWorldSource as u64
                    && range.range.index == 0
            })
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let bytes = saved.bytes.as_ref();
        let source = &self.source;
        if bytes.len() != SOURCE_HEADER_WORDS * 8
            || saved.capacity != bytes.len()
            || word(bytes, 0) != 1
            || &bytes[8..40] != source.identity.as_bytes()
            || &bytes[40..72] != source.seed_identity.as_bytes()
            || word(bytes, 9) != source.declaration.contexts.len() as u64
            || word(bytes, 10) != source.material.len() as u64
            || word(bytes, 11) != source.cell_capacity as u64
            || word(bytes, 12) != source.original_cells as u64
        {
            return Err(invalid_world(
                "checkpoint changes the complete original semantic world or seed namespace",
            ));
        }
        Ok(())
    }

    fn validate_program_cells<B: PublicationByteStorage>(
        &self,
        ranges: &[PublicationMaterialRangeStorage<B>],
    ) -> Result<(), SemanticTransitionError> {
        let state = ranges
            .iter()
            .find(|range| {
                range.range.role == SemanticStateRole::SemanticWorldState as u64
                    && range.range.index == 0
            })
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let ground = ranges
            .iter()
            .find(|range| {
                range.range.role == SemanticStateRole::TaskGround as u64 && range.range.index == 0
            })
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let layout = task_ground::TaskGroundLayout::from_material(ground.bytes.as_ref())?;
        let frame = layout.frame(0)?;
        let selected_rules = word(&ground.bytes[frame], 4);
        if word(state.bytes.as_ref(), 11) - self.source.original_cells as u64 != selected_rules {
            return Err(invalid_world(
                "selected rule cells differ from the actual published program image",
            ));
        }
        Ok(())
    }

    /// Checks actual archived selected bytes. Never replaces them with cold state.
    pub(super) fn validate_material<B: PublicationByteStorage>(
        &self,
        ranges: &[PublicationMaterialRangeStorage<B>],
        budget: &SemanticReplayMaterialBudget,
    ) -> Result<(), SemanticTransitionError> {
        self.validate_source(ranges)?;
        let state = ranges
            .iter()
            .find(|range| {
                range.range.role == SemanticStateRole::SemanticWorldState as u64
                    && range.range.index == 0
            })
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        self.source
            .validate_state(state.bytes.as_ref(), state.capacity, budget)?;
        if word(state.bytes.as_ref(), 15) != u64::from(self.ordinal) {
            return Err(invalid_world(
                "checkpoint changes its original qualified semantic context",
            ));
        }
        self.validate_program_cells(ranges)
    }

    /// A private evaluation branch inherits the full selected cell ledger and
    /// changes only its qualified view. The ordinary publication codec and CAS
    /// retain the derived bytes together with that branch's task ground.
    pub(super) fn derive_context<B: PublicationByteStorage>(
        &self,
        ranges: &mut [PublicationMaterialRangeStorage<B>],
        budget: &SemanticReplayMaterialBudget,
    ) -> Result<(), SemanticTransitionError> {
        self.validate_source(ranges)?;
        let selected = ranges
            .iter()
            .find(|range| {
                range.range.role == SemanticStateRole::SemanticWorldState as u64
                    && range.range.index == 0
            })
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        self.source
            .validate_state(selected.bytes.as_ref(), selected.capacity, budget)?;
        self.validate_program_cells(ranges)?;
        let state = ranges
            .iter_mut()
            .find(|range| {
                range.range.role == SemanticStateRole::SemanticWorldState as u64
                    && range.range.index == 0
            })
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let selected = state.bytes.writable_bytes()?;
        selected[15 * 8..16 * 8].copy_from_slice(&u64::from(self.ordinal).to_le_bytes());
        Ok(())
    }
}
