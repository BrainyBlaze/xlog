//! One device-readable encoding of the original immutable joint-world declaration.

use super::*;
use xlog_ir::semantic_world::{
    SemanticEventNode, SemanticMeasureFactor, SemanticSignedProgram, SemanticSupportPolarity,
    SemanticValueProofKind, SemanticValueQueryKind,
};

pub(super) const HEADER_WORDS: usize = 32;
const CONTEXT_WORDS: usize = 12;
const EVENT_WORDS: usize = 4;
const FACTOR_WORDS: usize = 4;

fn kind(kind: SemanticValueQueryKind) -> u64 {
    match kind {
        SemanticValueQueryKind::Assignment => 0,
        SemanticValueQueryKind::Measure => 1,
    }
}

fn reserve(words: &mut Vec<u64>, count: usize) -> Result<usize, SemanticTransitionError> {
    let begin = words.len();
    let end = begin
        .checked_add(count)
        .filter(|end| *end <= usize::MAX / 8)
        .ok_or(SemanticTransitionError::GenerationExhausted)?;
    words
        .try_reserve_exact(count)
        .map_err(|_| invalid_world("immutable semantic world source exceeds available memory"))?;
    words.resize(end, 0);
    Ok(begin)
}

fn rows(count: usize, width: usize) -> Result<usize, SemanticTransitionError> {
    count
        .checked_mul(width)
        .ok_or(SemanticTransitionError::GenerationExhausted)
}

pub(super) fn encode(
    declaration: &SemanticWorldDeclaration,
    programs: &[Arc<crate::SemanticProgramAdmission>],
    material_bytes: usize,
    identity: Identity256,
    seed_identity: Identity256,
    cell_capacity: usize,
    original_cells: usize,
) -> Result<Arc<[u8]>, SemanticTransitionError> {
    let mut words = Vec::new();
    reserve(&mut words, HEADER_WORDS)?;
    let contexts = reserve(&mut words, rows(declaration.contexts.len(), CONTEXT_WORDS)?)?;
    let events = reserve(
        &mut words,
        rows(declaration.event_nodes.len(), EVENT_WORDS)?,
    )?;
    let factors = reserve(
        &mut words,
        rows(declaration.measure_factors.len(), FACTOR_WORDS)?,
    )?;
    let payload = words.len();
    let mut base_variables = 0u64;
    let mut selectors = 0u64;
    for (index, node) in declaration.event_nodes.iter().enumerate() {
        let row = events + index * EVENT_WORDS;
        let record = match node {
            SemanticEventNode::False => [0, 0, 0, 0],
            SemanticEventNode::True => [1, 0, 0, 0],
            SemanticEventNode::Literal(value) => [2, i64::from(*value) as u64, 0, 0],
            SemanticEventNode::And(children) | SemanticEventNode::Or(children) => {
                let begin = reserve(&mut words, children.len())?;
                for (offset, child) in children.iter().enumerate() {
                    words[begin + offset] = u64::from(*child);
                }
                [
                    if matches!(node, SemanticEventNode::And(_)) {
                        3
                    } else {
                        4
                    },
                    begin as u64,
                    children.len() as u64,
                    0,
                ]
            }
            SemanticEventNode::Not(child) => [5, u64::from(*child), 0, 0],
        };
        words[row..row + EVENT_WORDS].copy_from_slice(&record);
    }
    for (index, factor) in declaration.measure_factors.iter().enumerate() {
        let record = match factor {
            SemanticMeasureFactor::Unit => [0, 0, 1, 1],
            SemanticMeasureFactor::Bernoulli {
                variable,
                numerator,
                denominator,
            } => {
                base_variables = base_variables.max(u64::from(*variable));
                [1, u64::from(*variable), *numerator, *denominator]
            }
            SemanticMeasureFactor::EventWeight {
                event,
                numerator,
                denominator,
            } => {
                selectors += 1;
                [2, u64::from(*event), *numerator, *denominator]
            }
        };
        let row = factors + index * FACTOR_WORDS;
        words[row..row + FACTOR_WORDS].copy_from_slice(&record);
    }
    for (index, (context, program)) in declaration.contexts.iter().zip(programs).enumerate() {
        let fact_begin = reserve(&mut words, rows(program.initial_facts.len(), 2)?)?;
        let rule_begin = reserve(&mut words, rows(program.initial_rules.len(), 3)?)?;
        let query_begin = reserve(&mut words, rows(context.queries.len(), 2)?)?;
        let assignment_begin = reserve(&mut words, context.view.assignment.len())?;
        let signed = match &context.program {
            SemanticSignedProgram::NativeProgram => {
                for fact in 0..program.initial_facts.len() {
                    words[fact_begin + fact * 2] = 1;
                }
                for rule in 0..program.initial_rules.len() {
                    words[rule_begin + rule * 3] = 1;
                }
                0
            }
            SemanticSignedProgram::SignedProgram { facts, rules } => {
                for (index, fact) in facts.iter().enumerate() {
                    let row = fact_begin + index * 2;
                    words[row] = u64::from(fact.pro_event);
                    words[row + 1] = u64::from(fact.contra_event);
                }
                for (index, rule) in rules.iter().enumerate() {
                    let row = rule_begin + index * 3;
                    words[row] = u64::from(rule.gate_event);
                    words[row + 1] = u64::from(rule.head == SemanticSupportPolarity::Contra);
                    words[row + 2] =
                        rule.body
                            .iter()
                            .enumerate()
                            .fold(0, |bits, (body, polarity)| {
                                bits | (u64::from(*polarity == SemanticSupportPolarity::Contra)
                                    << body)
                            });
                }
                1
            }
        };
        for (index, query) in context.queries.iter().enumerate() {
            let row = query_begin + index * 2;
            words[row] = kind(query.kind);
            words[row + 1] = u64::from(query.proof == SemanticValueProofKind::Certified);
        }
        for (index, (_, value)) in context.view.assignment.iter().enumerate() {
            words[assignment_begin + index] = u64::from(*value);
        }
        let row = contexts + index * CONTEXT_WORDS;
        words[row..row + CONTEXT_WORDS].copy_from_slice(&[
            u64::from(context.view.world_ordinal),
            kind(context.view.kind),
            u64::from(context.view.condition_event),
            program.initial_facts.len() as u64,
            program.initial_rules.len() as u64,
            context.queries.len() as u64,
            context.view.assignment.len() as u64,
            fact_begin as u64,
            rule_begin as u64,
            query_begin as u64,
            assignment_begin as u64,
            signed,
        ]);
    }
    let cap = declaration.capacities;
    words[0] = 2;
    for (index, chunk) in identity.as_bytes().chunks_exact(8).enumerate() {
        words[1 + index] = u64::from_le_bytes(chunk.try_into().expect("identity word"));
    }
    for (index, chunk) in seed_identity.as_bytes().chunks_exact(8).enumerate() {
        words[5 + index] = u64::from_le_bytes(chunk.try_into().expect("seed identity word"));
    }
    let word_count = words.len() as u64;
    words[9..HEADER_WORDS].copy_from_slice(&[
        declaration.contexts.len() as u64,
        material_bytes as u64,
        cell_capacity as u64,
        original_cells as u64,
        declaration.event_nodes.len() as u64,
        declaration.measure_factors.len() as u64,
        contexts as u64,
        events as u64,
        factors as u64,
        payload as u64,
        word_count,
        u64::from(cap.nodes),
        u64::from(cap.edges),
        u64::from(cap.unique_slots),
        u64::from(cap.apply_entries),
        u64::from(cap.dependency_edges),
        u64::from(cap.frontier_entries),
        u64::from(cap.particles),
        u64::from(cap.variables),
        base_variables,
        selectors,
        EVENT_WORDS as u64,
        CONTEXT_WORDS as u64,
    ]);
    Ok(words
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect::<Vec<_>>()
        .into())
}
