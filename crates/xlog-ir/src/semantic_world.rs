//! Authored shared Boolean events, measure laws, and qualified context declarations.
//!
//! A declaration describes immutable input. Mutable event roots and allocation state
//! belong to the publication bank that selects a semantic version.

use std::collections::BTreeSet;
use xlog_core::{Result, XlogError};

/// Authored Boolean expression. Indices refer to earlier declaration nodes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticEventNode {
    /// Empty event.
    False,
    /// Entire sample space.
    True,
    /// Signed, nonzero variable in the common sample space.
    Literal(i32),
    /// Boolean intersection.
    And(Vec<u32>),
    /// Boolean union; disjointness is established by the native representation.
    Or(Vec<u32>),
    /// Complement relative to the same sample space.
    Not(u32),
}

/// Exact rational input to the common measure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticMeasureFactor {
    /// Unit mass on the unique assignment of an empty variable domain.
    Unit,
    /// A base Boolean variable with an explicitly authored Bernoulli law.
    Bernoulli {
        variable: u32,
        numerator: u64,
        denominator: u64,
    },
    /// A nonnegative potential applied when an authored event holds.
    EventWeight {
        event: u32,
        numerator: u64,
        denominator: u64,
    },
}

/// Cold allocation bounds. Device counters remain versioned publication state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticMeasureCapacities {
    /// Circuit node slots.
    pub nodes: u32,
    /// Circuit and authored event edges.
    pub edges: u32,
    /// Reduced-node unique-table slots.
    pub unique_slots: u32,
    /// Boolean APPLY memo entries.
    pub apply_entries: u32,
    /// Reverse derivation dependencies.
    pub dependency_edges: u32,
    /// Affected-node worklist entries.
    pub frontier_entries: u32,
    /// Particle slots reserved by the declaration.
    pub particles: u32,
    /// Largest variable identifier and variable storage capacity.
    pub variables: u32,
}

/// Meaning of a qualified context or query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticValueQueryKind {
    /// Truth under a complete assignment.
    Assignment,
    /// Joint support mass under a conditioned common measure.
    Measure,
}

/// Independent positive and negative support channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticSupportPolarity {
    /// Positive support.
    Pro,
    /// Negative support.
    Contra,
}

/// Authored view on a declared world.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticContextView {
    /// Declared world namespace; no task-name inference is permitted.
    pub world_ordinal: u32,
    /// Assignment or measure semantics.
    pub kind: SemanticValueQueryKind,
    /// Event restricting the common measure.
    pub condition_event: u32,
    /// Ordered complete assignment, or empty for a measure view.
    pub assignment: Vec<(u32, bool)>,
}

/// Initial signed event binding for one compiled source fact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticFactEventBinding {
    /// Ordinal in the canonical compiled fact roster.
    pub fact: u32,
    /// Positive event root in the declaration.
    pub pro_event: u32,
    /// Negative event root in the declaration.
    pub contra_event: u32,
}

/// Signed channels and gate for one compiled source rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticRuleEventBinding {
    /// Ordinal in the canonical compiled rule roster.
    pub rule: u32,
    /// Head channel.
    pub head: SemanticSupportPolarity,
    /// Channel read by every ordered body atom.
    pub body: Vec<SemanticSupportPolarity>,
    /// Authored gate event, distinct from the mutable rule coefficient.
    pub gate_event: u32,
}

/// Explicit lowering law for authored source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticSignedProgram {
    /// Canonical positive source and actual authored evidence-leaf polarities.
    NativeProgram,
    /// Complete signed fact and rule bindings, checked against the native compiler.
    SignedProgram {
        /// Full initial fact roster.
        facts: Vec<SemanticFactEventBinding>,
        /// Full initial rule roster.
        rules: Vec<SemanticRuleEventBinding>,
    },
}

/// Explicit initial categorical rule-coefficient codes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticInitialRuleCells {
    /// One authored absolute code for every original rule.
    Uniform(u32),
    /// One authored absolute code per original rule, in compiler order.
    Codes(Vec<u32>),
}

/// Requested proof obligation, independent of support truth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticValueProofKind {
    /// Exact Boolean structure with explicit numerical availability.
    Structural,
    /// Requires an actual numerical certificate.
    Certified,
}

/// One original query and its requested semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticQueryView {
    /// Original query ordinal.
    pub query: u32,
    /// Assignment truth or joint measure view.
    pub kind: SemanticValueQueryKind,
    /// Structural or certified obligation.
    pub proof: SemanticValueProofKind,
}

/// Immutable semantics of one authored local source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticContextDeclaration {
    /// Qualified world restriction.
    pub view: SemanticContextView,
    /// Native or fully signed source law.
    pub program: SemanticSignedProgram,
    /// Initial authored rule coefficients.
    pub rule_cells: SemanticInitialRuleCells,
    /// Full ordered query roster.
    pub queries: Vec<SemanticQueryView>,
}

/// Common semantic declaration. Content identities are computed by its canonical
/// material owner; this structure contains no process address or CURRENT selector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticWorldDeclaration {
    /// Full ordered context roster.
    pub contexts: Vec<SemanticContextDeclaration>,
    /// Shared Boolean input expressions.
    pub event_nodes: Vec<SemanticEventNode>,
    /// One common measure, including all authored potentials.
    pub measure_factors: Vec<SemanticMeasureFactor>,
    /// Cold allocation bounds.
    pub capacities: SemanticMeasureCapacities,
}

fn invalid(detail: &str) -> XlogError {
    XlogError::Compilation(detail.to_owned())
}

impl SemanticWorldDeclaration {
    /// Checks topology, complete variable laws, view bindings, and cold bounds.
    /// Compiler fact/rule ordinals are checked later against the actual lowering.
    pub fn validate(&self) -> Result<()> {
        let cap = self.capacities;
        let node_count = u32::try_from(self.event_nodes.len())
            .map_err(|_| invalid("semantic event count exceeds u32"))?;
        if self.contexts.is_empty()
            || self.contexts.len() > u32::MAX as usize
            || node_count < 2
            || cap.nodes < node_count
            || !cap.unique_slots.is_power_of_two()
            || cap.unique_slots <= cap.nodes
            || cap.apply_entries == 0
            || cap.frontier_entries == 0
            || cap.dependency_edges < self.contexts.len() as u32
        {
            return Err(invalid(
                "semantic declaration has incomplete or insufficient bounded geometry",
            ));
        }
        if self.event_nodes[0] != SemanticEventNode::False
            || self.event_nodes[1] != SemanticEventNode::True
        {
            return Err(invalid("semantic event nodes begin with false and true"));
        }
        let mut variable_ids = BTreeSet::new();
        let mut edges = 0u64;
        for (ordinal, node) in self.event_nodes.iter().enumerate() {
            let earlier = |node: u32| -> Result<()> {
                if node as usize >= ordinal {
                    Err(invalid(
                        "semantic event topology must refer only to earlier nodes",
                    ))
                } else {
                    Ok(())
                }
            };
            match node {
                SemanticEventNode::False | SemanticEventNode::True => {}
                SemanticEventNode::Literal(literal) => {
                    if *literal == 0
                        || *literal == i32::MIN
                        || literal.unsigned_abs() > cap.variables
                    {
                        return Err(invalid(
                            "semantic literal is outside the declared variable domain",
                        ));
                    }
                    variable_ids.insert(literal.unsigned_abs());
                }
                SemanticEventNode::And(children) | SemanticEventNode::Or(children) => {
                    if children.is_empty() {
                        return Err(invalid(
                            "authored semantic intersections and unions are nonempty",
                        ));
                    }
                    for &child in children {
                        earlier(child)?;
                    }
                    edges = edges
                        .checked_add(children.len() as u64)
                        .ok_or_else(|| invalid("semantic event edge extent overflow"))?;
                }
                SemanticEventNode::Not(child) => {
                    earlier(*child)?;
                    edges += 1;
                }
            }
        }
        if edges > u64::from(cap.edges) {
            return Err(invalid("semantic event edges exceed the declared capacity"));
        }
        let mut laws = BTreeSet::new();
        let singleton = self.measure_factors.as_slice() == [SemanticMeasureFactor::Unit];
        for factor in &self.measure_factors {
            match factor {
                SemanticMeasureFactor::Unit => {
                    if !singleton || !variable_ids.is_empty() {
                        return Err(invalid("unit measure requires the unique empty assignment"));
                    }
                }
                SemanticMeasureFactor::Bernoulli {
                    variable,
                    numerator,
                    denominator,
                } => {
                    if *variable == 0
                        || *variable > cap.variables
                        || *denominator == 0
                        || numerator > denominator
                        || !laws.insert(*variable)
                    {
                        return Err(invalid(
                            "a base variable requires one valid authored Bernoulli law",
                        ));
                    }
                }
                SemanticMeasureFactor::EventWeight {
                    event, denominator, ..
                } => {
                    if *event >= node_count || *denominator == 0 {
                        return Err(invalid(
                            "semantic event potential has an invalid root or denominator",
                        ));
                    }
                }
            }
        }
        let base_variables = laws.iter().next_back().copied().unwrap_or(0);
        if (!singleton
            && (self.measure_factors.is_empty() || laws.len() != base_variables as usize))
            || variable_ids.iter().any(|var| !laws.contains(var))
        {
            return Err(invalid(
                "common measure must define the complete contiguous variable domain",
            ));
        }
        let mut worlds = BTreeSet::new();
        for context in &self.contexts {
            worlds.insert(context.view.world_ordinal);
            if context.view.condition_event >= node_count {
                return Err(invalid(
                    "semantic context condition is outside the common event DAG",
                ));
            }
            match context.view.kind {
                SemanticValueQueryKind::Assignment => {
                    if context.view.assignment.len() != base_variables as usize
                        || context
                            .view
                            .assignment
                            .iter()
                            .enumerate()
                            .any(|(index, (var, _))| *var as usize != index + 1)
                    {
                        return Err(invalid(
                            "assignment context requires every variable in canonical order",
                        ));
                    }
                }
                SemanticValueQueryKind::Measure if !context.view.assignment.is_empty() => {
                    return Err(invalid("measure context does not carry an assignment"));
                }
                SemanticValueQueryKind::Measure => {}
            }
            if context.queries.is_empty()
                || context.queries.iter().enumerate().any(|(index, query)| {
                    query.query as usize != index
                        || (query.kind == SemanticValueQueryKind::Assignment
                            && context.view.kind != SemanticValueQueryKind::Assignment)
                })
            {
                return Err(invalid(
                    "semantic query views require the full ordered original roster",
                ));
            }
            match &context.rule_cells {
                SemanticInitialRuleCells::Uniform(code) => validate_rule_code(*code)?,
                SemanticInitialRuleCells::Codes(codes) => {
                    for &code in codes {
                        validate_rule_code(code)?;
                    }
                }
            }
            if let SemanticSignedProgram::SignedProgram { facts, rules } = &context.program {
                if facts.iter().enumerate().any(|(index, fact)| {
                    fact.fact as usize != index
                        || fact.pro_event >= node_count
                        || fact.contra_event >= node_count
                }) || rules.iter().enumerate().any(|(index, rule)| {
                    rule.rule as usize != index || rule.gate_event >= node_count
                }) {
                    return Err(invalid("signed source bindings require ordered complete native ordinals and event roots"));
                }
            }
        }
        if worlds
            .iter()
            .enumerate()
            .any(|(index, world)| *world as usize != index)
        {
            return Err(invalid(
                "qualified world ordinals are contiguous in the authored declaration",
            ));
        }
        Ok(())
    }
}

/// Validates an authored coefficient code: NULL is transport-only, and code
/// `1 + m` denotes `m / 32768` for `0 <= m <= 32768`.
pub fn validate_rule_code(code: u32) -> Result<()> {
    if !(1..=32769).contains(&code) {
        return Err(invalid(
            "semantic rule coefficient requires a non-NULL absolute code",
        ));
    }
    Ok(())
}

impl SemanticContextDeclaration {
    /// Verifies complete binding against the canonical compiler's actual geometry.
    pub fn bind_program(
        &self,
        facts: usize,
        rule_body_arities: &[usize],
        queries: usize,
    ) -> Result<()> {
        if self.queries.len() != queries {
            return Err(invalid(
                "semantic query views differ from the compiled query roster",
            ));
        }
        if let SemanticInitialRuleCells::Codes(codes) = &self.rule_cells {
            if codes.len() != rule_body_arities.len() {
                return Err(invalid(
                    "initial rule coefficient roster differs from the compiled source",
                ));
            }
        }
        if let SemanticSignedProgram::SignedProgram {
            facts: bindings,
            rules,
        } = &self.program
        {
            if bindings.len() != facts
                || rules.len() != rule_body_arities.len()
                || rules
                    .iter()
                    .zip(rule_body_arities)
                    .any(|(rule, arity)| rule.body.len() != *arity)
            {
                return Err(invalid(
                    "signed bindings differ from the complete compiled source geometry",
                ));
            }
        }
        Ok(())
    }

    /// Expands only explicitly authored initial coefficient codes.
    pub fn initial_rule_codes(&self, rules: usize) -> Result<Vec<u32>> {
        match &self.rule_cells {
            SemanticInitialRuleCells::Uniform(code) => Ok(vec![*code; rules]),
            SemanticInitialRuleCells::Codes(codes) if codes.len() == rules => Ok(codes.clone()),
            _ => Err(invalid(
                "initial coefficient roster differs from compiled rules",
            )),
        }
    }
}
