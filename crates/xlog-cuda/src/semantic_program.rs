//! Arity-aware positive program data shared by cold source lowering and the resident evaluator.

use crate::DeviceRepr;

/// Native transition storage admits at most this many input facts per task.
pub(crate) const RESIDENT_PROGRAM_FACT_CAPACITY: usize = 4096;
/// The two sampled edits reserve two entries in this resident rule bank.
pub(crate) const RESIDENT_PROGRAM_RULE_CAPACITY: usize = 256;

/// One typed tuple. Predicate indices come from the admitted declaration order.
/// Unary tuples have a canonical zero in the inactive second argument.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SemanticProgramFact {
    pub predicate: u32,
    pub arity: u32,
    pub first: u32,
    pub second: u32,
}

// SAFETY: the C layout has four u32 fields and no references or padding.
unsafe impl DeviceRepr for SemanticProgramFact {}

/// An active term is either a variable (kind zero) or a u32 constant (kind one).
/// Equal variable indices retain repeated-variable equalities across all atoms.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SemanticProgramTerm {
    pub kind: u32,
    pub value: u32,
}

// SAFETY: the C layout has two u32 fields and no references or padding.
unsafe impl DeviceRepr for SemanticProgramTerm {}

/// One unary or binary atom; inactive argument storage is canonical zero.
/// An unused body atom has every field zero, including its arity.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SemanticProgramAtom {
    pub predicate: u32,
    pub arity: u32,
    pub terms: [SemanticProgramTerm; 2],
}

// SAFETY: the C layout has six u32 fields and no references or padding.
unsafe impl DeviceRepr for SemanticProgramAtom {}

impl SemanticProgramAtom {
    fn valid_for(&self, predicate_arities: &[u32]) -> bool {
        (1..=2).contains(&self.arity)
            && predicate_arities.get(self.predicate as usize) == Some(&self.arity)
            && self.terms[..self.arity as usize]
                .iter()
                .all(|term| term.kind == 1 || (term.kind == 0 && term.value < 6))
            && self.terms[self.arity as usize..]
                .iter()
                .all(|term| *term == SemanticProgramTerm::default())
    }
}

/// One range-restricted positive rule with one to three authored body atoms.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SemanticProgramRule {
    pub head: SemanticProgramAtom,
    pub body: [SemanticProgramAtom; 3],
    pub body_count: u32,
}

// SAFETY: the C layout has twenty-five u32 fields and no references or padding.
unsafe impl DeviceRepr for SemanticProgramRule {}

impl SemanticProgramRule {
    /// Canonical device words, independent of host struct layout.
    pub fn words(&self) -> [u64; 25] {
        let mut words = [0; 25];
        words[0] = u64::from(self.body_count);
        for (index, atom) in std::iter::once(&self.head).chain(&self.body).enumerate() {
            let offset = 1 + 6 * index;
            words[offset..offset + 6].copy_from_slice(&[
                u64::from(atom.predicate),
                u64::from(atom.arity),
                u64::from(atom.terms[0].kind),
                u64::from(atom.terms[0].value),
                u64::from(atom.terms[1].kind),
                u64::from(atom.terms[1].value),
            ]);
        }
        words
    }
}

/// Exact initial program and input facts prepared before a resident task starts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticProgramAdmission {
    pub predicate_count: u32,
    pub predicate_arities: Vec<u32>,
    pub initial_facts: Vec<SemanticProgramFact>,
    pub initial_rules: Vec<SemanticProgramRule>,
    pub queries: Vec<SemanticProgramFact>,
    /// Ordered original Statement records selected for the query axis.
    pub query_records: Vec<u32>,
    /// Original admitted statement-record indices and their compiled tuples.
    /// This roster is independent of the selected query axis.
    pub observation_facts: Vec<(u32, SemanticProgramFact)>,
}

impl SemanticProgramAdmission {
    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        let valid_fact = |fact: &SemanticProgramFact| {
            (1..=2).contains(&fact.arity)
                && self.predicate_arities.get(fact.predicate as usize) == Some(&fact.arity)
                && (fact.arity == 2 || fact.second == 0)
        };
        if self.predicate_count == 0
            || self.predicate_arities.len() != self.predicate_count as usize
            || self
                .predicate_arities
                .iter()
                .any(|arity| !(1..=2).contains(arity))
            || self.queries.is_empty()
            || self.queries.len() > u32::MAX as usize
            || self.query_records.len() != self.queries.len()
            || self.initial_facts.len() > u32::MAX as usize
            || self.initial_rules.len() > u32::MAX as usize
            || self
                .queries
                .iter()
                .chain(&self.initial_facts)
                .any(|fact| !valid_fact(fact))
            || self
                .observation_facts
                .iter()
                .any(|(_, fact)| !valid_fact(fact))
            || self
                .observation_facts
                .iter()
                .enumerate()
                .any(|(index, (record, _))| {
                    self.observation_facts[..index]
                        .iter()
                        .any(|(before, _)| before == record)
                })
            || self
                .query_records
                .iter()
                .zip(&self.queries)
                .any(|(record, fact)| {
                    !self
                        .observation_facts
                        .iter()
                        .any(|(original, tuple)| original == record && tuple == fact)
                })
        {
            return Err("positive program has an invalid predicate, arity, or fact count");
        }
        if self.initial_facts.len() > RESIDENT_PROGRAM_FACT_CAPACITY {
            return Err("positive program exceeds resident fact capacity");
        }
        if self.initial_rules.len() > RESIDENT_PROGRAM_RULE_CAPACITY - 2 {
            return Err("positive program leaves no room for two rule edits");
        }
        for rule in &self.initial_rules {
            if !(1..=3).contains(&rule.body_count)
                || !rule.head.valid_for(&self.predicate_arities)
                || rule.body[..rule.body_count as usize]
                    .iter()
                    .any(|atom| !atom.valid_for(&self.predicate_arities))
                || rule.body[rule.body_count as usize..]
                    .iter()
                    .any(|atom| *atom != SemanticProgramAtom::default())
            {
                return Err("positive program has an invalid rule atom or body");
            }
            let mut bound = [false; 6];
            for atom in &rule.body[..rule.body_count as usize] {
                for term in &atom.terms[..atom.arity as usize] {
                    if term.kind == 0 {
                        bound[term.value as usize] = true;
                    }
                }
            }
            if rule.head.terms[..rule.head.arity as usize]
                .iter()
                .any(|term| term.kind == 0 && !bound[term.value as usize])
            {
                return Err("positive rule head contains an unbound variable");
            }
        }
        Ok(())
    }

    /// Canonical device-bank words, independent of host struct padding.
    pub(crate) fn words(&self) -> Vec<u64> {
        let mut words = Vec::new();
        words.extend([
            2,
            u64::from(self.predicate_count),
            self.initial_facts.len() as u64,
            self.initial_rules.len() as u64,
            self.queries.len() as u64,
        ]);
        words.extend(self.predicate_arities.iter().copied().map(u64::from));
        for fact in self.queries.iter().chain(&self.initial_facts) {
            words.extend([
                u64::from(fact.predicate),
                u64::from(fact.arity),
                u64::from(fact.first),
                u64::from(fact.second),
            ]);
        }
        for rule in &self.initial_rules {
            words.extend(rule.words());
        }
        words
    }
}

const _: () = {
    assert!(std::mem::size_of::<SemanticProgramFact>() == 16);
    assert!(std::mem::size_of::<SemanticProgramTerm>() == 8);
    assert!(std::mem::size_of::<SemanticProgramAtom>() == 24);
    assert!(std::mem::size_of::<SemanticProgramRule>() == 100);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program_bank_words_match_the_native_decoder_layout() {
        let program = SemanticProgramAdmission {
            predicate_count: 2,
            initial_facts: vec![SemanticProgramFact {
                predicate: 0,
                first: 1,
                second: 3,
            }],
            initial_rules: vec![SemanticProgramRule {
                head_predicate: 1,
                left_predicate: 0,
                right_predicate: 0,
                body_count: 1,
                head_variables: [0, 1],
                left_variables: [0, 1],
                right_variables: [0, 0],
            }],
            queries: [
                SemanticProgramFact {
                    predicate: 1,
                    first: 1,
                    second: 3,
                },
                SemanticProgramFact {
                    predicate: 1,
                    first: 2,
                    second: 3,
                },
                SemanticProgramFact {
                    predicate: 1,
                    first: 3,
                    second: 4,
                },
            ],
        };
        assert_eq!(
            program.words(),
            [2, 1, 1, 1, 1, 3, 1, 2, 3, 1, 3, 4, 0, 1, 3, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0]
        );
        assert_eq!(program.validate(), Ok(()));
        let mut unsafe_rule = program.clone();
        unsafe_rule.initial_rules[0].head_variables = [2, 2];
        assert_eq!(
            unsafe_rule.validate(),
            Err("binary rule head contains an unbound variable")
        );
        let mut too_many_facts = program.clone();
        too_many_facts.initial_facts = vec![program.initial_facts[0]; 4097];
        assert_eq!(
            too_many_facts.validate(),
            Err("binary program exceeds resident fact capacity")
        );
        let mut too_many_rules = program.clone();
        too_many_rules.initial_rules = vec![program.initial_rules[0]; 255];
        assert_eq!(
            too_many_rules.validate(),
            Err("binary program leaves no room for two rule edits")
        );
    }
}
