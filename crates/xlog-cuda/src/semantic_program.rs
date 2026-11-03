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
    pub fn evaluation_context_allocation_bytes_from_optional_source(
        source: Option<&Self>,
        target: &Self,
    ) -> Result<usize, &'static str> {
        // The target-only case has overlapping old/new query and observation
        // vectors. Count two actual target carriers; create no source program.
        Self::evaluation_context_allocation_bytes(source.unwrap_or(target), target)
    }

    /// Lower an evaluation task next to the actual selected training world.
    /// A non-positive source world remains a hypergraph; no empty positive
    /// program is invented for it.
    pub fn evaluation_context_from_optional_source(
        source: Option<&Self>,
        target: &Self,
        target_record_offset: u32,
    ) -> Result<Self, &'static str> {
        if let Some(source) = source {
            return Self::evaluation_context(source, target, target_record_offset);
        }
        target.validate()?;
        let mut result = target.clone();
        for record in &mut result.query_records {
            *record = record
                .checked_add(target_record_offset)
                .ok_or("evaluation statement context exceeds the native domain")?;
        }
        for (record, _) in &mut result.observation_facts {
            *record = record
                .checked_add(target_record_offset)
                .ok_or("evaluation statement context exceeds the native domain")?;
        }
        result.validate()?;
        Ok(result)
    }

    /// Simultaneous output vectors and ordered constant-map allocations used by
    /// evaluation_context. The caller reserves this from its original pool.
    pub fn evaluation_context_allocation_bytes(
        source: &Self,
        target: &Self,
    ) -> Result<usize, &'static str> {
        let overflow = || "evaluation context allocation exceeds the native domain";
        let mut bytes = std::mem::size_of::<Self>();
        for program in [source, target] {
            for (count, width) in [
                (program.predicate_arities.len(), std::mem::size_of::<u32>()),
                (
                    program.initial_facts.len() + program.queries.len(),
                    std::mem::size_of::<SemanticProgramFact>(),
                ),
                (
                    program.initial_rules.len(),
                    std::mem::size_of::<SemanticProgramRule>(),
                ),
                (program.query_records.len(), std::mem::size_of::<u32>()),
                (
                    program.observation_facts.len(),
                    std::mem::size_of::<(u32, SemanticProgramFact)>(),
                ),
            ] {
                bytes = count
                    .checked_mul(width)
                    .and_then(|extent| bytes.checked_add(extent.checked_mul(2)?))
                    .ok_or_else(overflow)?;
            }
            let constants = program
                .initial_facts
                .len()
                .checked_add(program.queries.len())
                .and_then(|count| count.checked_add(program.observation_facts.len()))
                .and_then(|count| count.checked_mul(2))
                .and_then(|count| count.checked_add(program.initial_rules.len().checked_mul(8)?))
                .ok_or_else(overflow)?;
            // BTree nodes, map entries, minimum leaves and the two context sets
            // overlap. Every actual constant occurrence encloses a distinct node.
            bytes = constants
                .checked_add(4)
                .and_then(|count| count.checked_mul(512))
                .and_then(|extent| bytes.checked_add(extent))
                .ok_or_else(overflow)?;
        }
        Ok(bytes)
    }

    /// Preserve the complete source program and append an independent target
    /// context before the model catalogue is frozen. Only the target's own
    /// predicates and constants are lowered into disjoint native identities.
    /// Learned source rules therefore retain every original word.
    pub fn evaluation_context(
        source: &Self,
        target: &Self,
        target_record_offset: u32,
    ) -> Result<Self, &'static str> {
        source.validate()?;
        target.validate()?;
        let predicate_count = source
            .predicate_count
            .checked_add(target.predicate_count)
            .ok_or("evaluation predicate context exceeds the native domain")?;
        let mut source_constants = std::collections::BTreeSet::new();
        let mut target_constants = std::collections::BTreeSet::new();
        for (program, constants) in [
            (source, &mut source_constants),
            (target, &mut target_constants),
        ] {
            for fact in program
                .initial_facts
                .iter()
                .chain(&program.queries)
                .chain(program.observation_facts.iter().map(|(_, fact)| fact))
            {
                constants.insert(fact.first);
                if fact.arity == 2 {
                    constants.insert(fact.second);
                }
            }
            for rule in &program.initial_rules {
                for atom in
                    std::iter::once(&rule.head).chain(&rule.body[..rule.body_count as usize])
                {
                    for term in &atom.terms[..atom.arity as usize] {
                        if term.kind == 1 {
                            constants.insert(term.value);
                        }
                    }
                }
            }
        }
        let mut mapping = std::collections::BTreeMap::new();
        let mut next = 0u32;
        for original in target_constants {
            while source_constants.contains(&next) {
                next = next
                    .checked_add(1)
                    .ok_or("evaluation constant context exceeds the native domain")?;
            }
            mapping.insert(original, next);
            source_constants.insert(next);
        }
        let fact = |mut fact: SemanticProgramFact| -> Result<SemanticProgramFact, &'static str> {
            fact.predicate = fact
                .predicate
                .checked_add(source.predicate_count)
                .ok_or("evaluation predicate context exceeds the native domain")?;
            fact.first = *mapping
                .get(&fact.first)
                .ok_or("evaluation constant was not frozen")?;
            if fact.arity == 2 {
                fact.second = *mapping
                    .get(&fact.second)
                    .ok_or("evaluation constant was not frozen")?;
            }
            Ok(fact)
        };
        let mut result = Self {
            predicate_count,
            predicate_arities: source.predicate_arities.clone(),
            initial_facts: source.initial_facts.clone(),
            initial_rules: source.initial_rules.clone(),
            queries: target
                .queries
                .iter()
                .copied()
                .map(&fact)
                .collect::<Result<_, _>>()?,
            query_records: target
                .query_records
                .iter()
                .map(|record| {
                    record
                        .checked_add(target_record_offset)
                        .ok_or("evaluation statement context exceeds the native domain")
                })
                .collect::<Result<_, _>>()?,
            observation_facts: source.observation_facts.clone(),
        };
        result
            .predicate_arities
            .extend_from_slice(&target.predicate_arities);
        result.initial_facts.extend(
            target
                .initial_facts
                .iter()
                .copied()
                .map(&fact)
                .collect::<Result<Vec<_>, _>>()?,
        );
        for original in &target.initial_rules {
            let mut rule = *original;
            for atom in
                std::iter::once(&mut rule.head).chain(&mut rule.body[..rule.body_count as usize])
            {
                atom.predicate = atom
                    .predicate
                    .checked_add(source.predicate_count)
                    .ok_or("evaluation predicate context exceeds the native domain")?;
                for term in &mut atom.terms[..atom.arity as usize] {
                    if term.kind == 1 {
                        term.value = *mapping
                            .get(&term.value)
                            .ok_or("evaluation constant was not frozen")?;
                    }
                }
            }
            result.initial_rules.push(rule);
        }
        for (record, tuple) in &target.observation_facts {
            result.observation_facts.push((
                record
                    .checked_add(target_record_offset)
                    .ok_or("evaluation statement context exceeds the native domain")?,
                fact(*tuple)?,
            ));
        }
        result.validate()?;
        Ok(result)
    }

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

    pub(crate) fn word_count(&self) -> Option<usize> {
        self.queries
            .len()
            .checked_add(self.initial_facts.len())?
            .checked_mul(4)?
            .checked_add(self.predicate_arities.len())?
            .checked_add(
                self.initial_rules
                    .len()
                    .checked_mul(SemanticProgramRule::default().words().len())?,
            )?
            .checked_add(5)
    }

    /// Canonical device-bank words, independent of host struct padding.
    pub(crate) fn visit_words(&self, mut visit: impl FnMut(u64)) {
        for word in [
            2,
            u64::from(self.predicate_count),
            self.initial_facts.len() as u64,
            self.initial_rules.len() as u64,
            self.queries.len() as u64,
        ] {
            visit(word);
        }
        for &arity in &self.predicate_arities {
            visit(u64::from(arity));
        }
        for fact in self.queries.iter().chain(&self.initial_facts) {
            for word in [
                u64::from(fact.predicate),
                u64::from(fact.arity),
                u64::from(fact.first),
                u64::from(fact.second),
            ] {
                visit(word);
            }
        }
        for rule in &self.initial_rules {
            for word in rule.words() {
                visit(word);
            }
        }
    }

    pub(crate) fn words(&self) -> Vec<u64> {
        let mut words = Vec::with_capacity(
            self.word_count()
                .expect("validated program has a finite canonical encoding"),
        );
        self.visit_words(|word| words.push(word));
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
