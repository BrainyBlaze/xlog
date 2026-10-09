//! Binary positive program data shared by cold source lowering and the resident evaluator.

use crate::DeviceRepr;

/// Native transition storage admits at most this many input facts per task.
pub(crate) const RESIDENT_PROGRAM_FACT_CAPACITY: usize = 4096;
/// The two sampled edits reserve two entries in this resident rule bank.
pub(crate) const RESIDENT_PROGRAM_RULE_CAPACITY: usize = 256;

/// One typed binary tuple. Predicate indices come from the admitted declaration order.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticProgramFact {
    pub predicate: u32,
    pub first: u32,
    pub second: u32,
}

// SAFETY: the C layout has three u32 fields and no references or padding.
unsafe impl DeviceRepr for SemanticProgramFact {}

/// One range-restricted positive rule with one or two binary body atoms.
/// Equal variable indices encode repeated-variable equalities.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticProgramRule {
    pub head_predicate: u32,
    pub left_predicate: u32,
    pub right_predicate: u32,
    pub body_count: u32,
    pub head_variables: [u32; 2],
    pub left_variables: [u32; 2],
    pub right_variables: [u32; 2],
}

// SAFETY: the C layout has ten u32 fields and no references or padding.
unsafe impl DeviceRepr for SemanticProgramRule {}

/// Exact initial program and input facts prepared before a resident task starts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticProgramAdmission {
    pub predicate_count: u32,
    pub initial_facts: Vec<SemanticProgramFact>,
    pub initial_rules: Vec<SemanticProgramRule>,
    pub queries: Vec<SemanticProgramFact>,
    /// Original admitted statement-record indices and their compiled tuples.
    /// This roster is independent of the selected query axis.
    pub observation_facts: Vec<(u32, SemanticProgramFact)>,
}

impl SemanticProgramAdmission {
    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        if self.predicate_count == 0
            || self.queries.is_empty()
            || self.queries.len() > u32::MAX as usize
            || self.initial_facts.len() > u32::MAX as usize
            || self.initial_rules.len() > u32::MAX as usize
            || self
                .queries
                .iter()
                .chain(&self.initial_facts)
                .any(|fact| fact.predicate >= self.predicate_count)
            || self.observation_facts.iter().any(|(_, fact)| fact.predicate >= self.predicate_count)
            || self.observation_facts.iter().enumerate().any(|(index, (record, _))| {
                self.observation_facts[..index].iter().any(|(before, _)| before == record)
            })
        {
            return Err("binary program has an invalid predicate or fact count");
        }
        if self.initial_facts.len() > RESIDENT_PROGRAM_FACT_CAPACITY {
            return Err("binary program exceeds resident fact capacity");
        }
        if self.initial_rules.len() > RESIDENT_PROGRAM_RULE_CAPACITY - 2 {
            return Err("binary program leaves no room for two rule edits");
        }
        for rule in &self.initial_rules {
            if !(1..=2).contains(&rule.body_count)
                || rule.head_predicate >= self.predicate_count
                || rule.left_predicate >= self.predicate_count
                || (rule.body_count == 2 && rule.right_predicate >= self.predicate_count)
                || (rule.body_count == 1
                    && (rule.right_predicate != 0 || rule.right_variables != [0, 0]))
            {
                return Err("binary program has an invalid rule predicate or body");
            }
            let mut bound = [false; 4];
            for &variable in rule.left_variables.iter().chain(
                (rule.body_count == 2)
                    .then_some(&rule.right_variables)
                    .into_iter()
                    .flatten(),
            ) {
                let slot = bound
                    .get_mut(variable as usize)
                    .ok_or("binary rule variable exceeds its four-slot domain")?;
                *slot = true;
            }
            if rule
                .head_variables
                .iter()
                .any(|&variable| !bound.get(variable as usize).copied().unwrap_or(false))
            {
                return Err("binary rule head contains an unbound variable");
            }
        }
        Ok(())
    }

    /// Canonical device-bank words, independent of host struct padding.
    pub(crate) fn words(&self) -> Vec<u64> {
        let mut words = Vec::new();
        words.extend([
            u64::from(self.predicate_count),
            self.initial_facts.len() as u64,
            self.initial_rules.len() as u64,
            self.queries.len() as u64,
        ]);
        for fact in self.queries.iter().chain(&self.initial_facts) {
            words.extend([
                u64::from(fact.predicate),
                u64::from(fact.first),
                u64::from(fact.second),
            ]);
        }
        for rule in &self.initial_rules {
            words.extend([
                u64::from(rule.head_predicate),
                u64::from(rule.left_predicate),
                u64::from(rule.right_predicate),
                u64::from(rule.body_count),
                u64::from(rule.head_variables[0]),
                u64::from(rule.head_variables[1]),
                u64::from(rule.left_variables[0]),
                u64::from(rule.left_variables[1]),
                u64::from(rule.right_variables[0]),
                u64::from(rule.right_variables[1]),
            ]);
        }
        words
    }
}

const _: () = {
    assert!(std::mem::size_of::<SemanticProgramFact>() == 12);
    assert!(std::mem::size_of::<SemanticProgramRule>() == 40);
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
