//! Structural completion over the same typed program used for actual choices.

use std::cmp::Reverse;
use std::sync::Mutex;

use super::super::ACQUIRED_SOURCE_BYTES;
use super::*;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
struct ValueKey {
    shapes: [(usize, usize, bool); 2],
    dependencies: TensorOutputDependencies,
    numeric_arguments: bool,
}

impl ValueKey {
    /// At T=1 every callable argument and output has one row. Search consumers
    /// preserve rows or require one row, and never use Row/Stack, so multirow
    /// values cannot reach an output. Physical Linear weights remain available.
    fn has_single_row_at_one_token(&self) -> bool {
        self.shapes[0].0 == 1
    }

    fn covers(&self, other: &Self) -> bool {
        self.shapes == other.shapes
            && (!self.numeric_arguments || other.numeric_arguments)
            && (!other.dependencies.tokens || self.dependencies.tokens)
            && self
                .dependencies
                .arguments
                .is_superset(&other.dependencies.arguments)
            && self
                .dependencies
                .definitions
                .is_superset(&other.dependencies.definitions)
            && self
                .dependencies
                .parameters
                .is_superset(&other.dependencies.parameters)
    }
}

#[derive(serde::Serialize)]
struct StateKey {
    remaining: usize,
    values: Vec<ValueKey>,
}

/// For one environment/frontier and a legal row allowance, a completed result
/// can fill further slots with the same canonical Tanh used by Search::visit.
/// Thus a positive proof covers larger allowances, and a negative proof covers
/// smaller ones. These bounds retain answers, never a prospective source graph.
#[derive(Default)]
struct CompletionBounds {
    possible_from: Option<usize>,
    impossible_through: Option<usize>,
}

impl CompletionBounds {
    fn get(&self, remaining: usize) -> Option<bool> {
        if self.possible_from.is_some_and(|rows| remaining >= rows) {
            Some(true)
        } else if self
            .impossible_through
            .is_some_and(|rows| remaining <= rows)
        {
            Some(false)
        } else {
            None
        }
    }

    fn remember(&mut self, remaining: usize, possible: bool) {
        if possible {
            self.possible_from = Some(
                self.possible_from
                    .map_or(remaining, |rows| rows.min(remaining)),
            );
        } else {
            self.impossible_through = Some(
                self.impossible_through
                    .map_or(remaining, |rows| rows.max(remaining)),
            );
        }
    }
}

#[derive(Default)]
struct CompletionProofs {
    environment: Vec<u8>,
    empty_plans: CompletionBounds,
    prefixes: BTreeMap<Vec<u8>, bool>,
    prefix_bytes: usize,
}

impl CompletionProofs {
    fn bind(&mut self, environment: &[u8]) {
        if self.environment != environment {
            *self = Self {
                environment: environment.to_vec(),
                ..Self::default()
            };
        }
    }
}

/// Structural answers only; no model state or authored source. One exact
/// environment retains empty-plan bounds and a source-byte allowance of
/// serialized external-query keys. Intermediate frontier bounds are reset for
/// each mask; prefix eviction changes only recomputation.
#[derive(Default)]
pub(crate) struct CompletionCache {
    entries: Mutex<BTreeMap<Vec<u8>, BTreeMap<Vec<ValueKey>, CompletionBounds>>>,
    proofs: Mutex<CompletionProofs>,
}

impl CompletionCache {
    pub(crate) fn clear_search(&self) {
        self.entries.lock().unwrap().clear();
    }

    fn empty_plan(&self, environment: &[u8], output: usize) -> Option<bool> {
        let proofs = self.proofs.lock().unwrap();
        if proofs.environment != environment {
            return None;
        }
        proofs.empty_plans.get(output + 1)
    }

    fn remember_empty_plan(&self, environment: &[u8], output: usize, result: bool) {
        let mut proofs = self.proofs.lock().unwrap();
        proofs.bind(environment);
        proofs.empty_plans.remember(output + 1, result);
    }

    fn prefix(&self, environment: &[u8], key: &[u8]) -> Option<bool> {
        let proofs = self.proofs.lock().unwrap();
        if proofs.environment != environment {
            return None;
        }
        proofs.prefixes.get(key).copied()
    }

    fn remember_prefix(&self, environment: &[u8], key: Vec<u8>, result: bool) {
        let mut proofs = self.proofs.lock().unwrap();
        proofs.bind(environment);
        if let Some(answer) = proofs.prefixes.get_mut(&key) {
            *answer = result;
            return;
        }
        if key.len() > ACQUIRED_SOURCE_BYTES {
            return;
        }
        if proofs.prefix_bytes + key.len() > ACQUIRED_SOURCE_BYTES {
            proofs.prefixes.clear();
            proofs.prefix_bytes = 0;
        }
        proofs.prefix_bytes += key.len();
        proofs.prefixes.insert(key, result);
    }
}

fn value_key(program: &Program, id: usize, numeric_arguments: bool) -> ValueKey {
    ValueKey {
        shapes: std::array::from_fn(|length| {
            let shape = program.shapes[length][id];
            (shape.rows, shape.columns, shape.float32)
        }),
        dependencies: program.dependencies.as_ref().unwrap().values[id].clone(),
        numeric_arguments,
    }
}

/// Unselected operands need one representative per equal complete value key.
/// Explicit indices and decimal-prefix choices retain their original identities.
pub(super) fn operand_representatives(program: &Program) -> Vec<usize> {
    let mut seen = BTreeSet::new();
    program
        .body
        .value_argument_dependencies()
        .into_iter()
        .enumerate()
        .filter_map(|(id, arguments)| seen.insert(value_key(program, id, arguments)).then_some(id))
        .collect()
}

fn frontier(program: &Program) -> Vec<(usize, ValueKey)> {
    let arguments = program.body.value_argument_dependencies();
    let mut values: Vec<(usize, ValueKey)> = Vec::new();
    for (id, &arguments) in arguments.iter().enumerate() {
        let key = value_key(program, id, arguments);
        if !key.has_single_row_at_one_token()
            || values.iter().any(|(_, available)| available.covers(&key))
        {
            continue;
        }
        values.retain(|(_, available)| !key.covers(available));
        values.push((id, key));
    }
    values
}

/// Enumerate actual shape-valid operand vectors, deduplicating equal output
/// contributions. Unused formals still carry the canonical causal read flag.
pub(super) fn call_arguments(
    program: &Program,
    name: &str,
    signature: &TensorSignature,
    prefix: &[usize],
    available: &[usize],
) -> Vec<Vec<usize>> {
    let [Ok(first), Ok(last)] = [
        signature.arguments(1),
        signature.arguments(program.body.max_context),
    ] else {
        return Vec::new();
    };
    if prefix.len() > first.len() {
        return Vec::new();
    }
    let facts = program.dependencies.as_ref().unwrap();
    let reads = program.body.value_argument_dependencies();
    let mut partial = BTreeMap::from([((TensorOutputDependencies::default(), false), Vec::new())]);
    for slot in 0..first.len() {
        let used = facts.callees[name].arguments.contains(&slot);
        let mut choices = prefix
            .get(slot)
            .map_or_else(|| available.to_vec(), |&id| vec![id]);
        choices.retain(|&id| {
            program.shapes[0].get(id) == Some(&first[slot])
                && program.shapes[1].get(id) == Some(&last[slot])
        });
        if !used && prefix.get(slot).is_none() {
            choices.sort_by_key(|&id| reads[id]);
            choices.truncate(1);
        }
        let mut next = BTreeMap::new();
        for ((dependencies, numeric), arguments) in partial {
            for &id in &choices {
                let mut dependencies = dependencies.clone();
                if used {
                    dependencies.merge(&facts.values[id]);
                }
                let mut arguments = arguments.clone();
                arguments.push(id);
                next.entry((dependencies, numeric || reads[id]))
                    .or_insert(arguments);
            }
        }
        partial = next;
    }
    partial.into_values().collect()
}

pub(super) fn possible(
    source: &Program,
    body: &TensorProgram,
    output: usize,
    signatures: &Signatures,
    cache: &CompletionCache,
) -> bool {
    let facts = source.dependencies.as_ref().unwrap();
    let required = source.entry_requirements.as_ref().unwrap();
    let environment = serde_json::to_vec(&(
        &body.signature,
        &body.parameters,
        body.vocabulary,
        body.max_context,
        source.column_limit,
        &source.raw_tokens,
        signatures,
        &facts.callees,
        &required.calls,
        &required.parameters,
    ))
    .unwrap();
    // Parsing a cumulative prefix revisits the completed operation count even
    // after later operations have arrived. Its empty-body feasibility proof is
    // unchanged by those bytes; reuse it only under the full original inputs.
    let empty_plan = body.operations.is_empty();
    if empty_plan {
        if let Some(result) = cache.empty_plan(&environment, output) {
            return result;
        }
    }
    let mut program = source.clone();
    program.body = body.clone();
    program.body.output = output;
    program.planned_output = Some(output);
    program.body.operations.truncate(facts.values.len());
    // Search transitions use push's canonical inference and the existing
    // necessary bound, without recursively invoking this completion search.
    program.dependencies.as_mut().unwrap().completion_cache = None;
    if body.operations.len() > facts.values.len()
        && program
            .push(body.operations.last().unwrap().clone(), signatures)
            .is_err()
    {
        return false;
    }
    let search = Search {
        environment,
        signatures,
        cache,
    };
    let result = search.visit(program, !empty_plan);
    if empty_plan {
        cache.remember_empty_plan(&search.environment, output, result);
    }
    result
}

struct Search<'a> {
    environment: Vec<u8>,
    signatures: &'a Signatures,
    cache: &'a CompletionCache,
}

impl Search<'_> {
    fn producers(&self, mut program: Program) -> Option<Program> {
        let required = program.entry_requirements.as_ref().unwrap().clone();
        let mut available = TensorOutputDependencies::default();
        for value in &program.dependencies.as_ref().unwrap().values {
            available.merge(value);
        }
        // A direct vector read has no operands and commutes with every future
        // row. Do not branch on its scheduling when no callee can supply it.
        let weights = program
            .body
            .parameters
            .iter()
            .enumerate()
            .filter_map(|(weight, parameter)| {
                (parameter.shape.len() == 1
                    && required.parameters.contains(&parameter.name)
                    && !available.parameters.contains(&parameter.name)
                    && !program
                        .dependencies
                        .as_ref()
                        .unwrap()
                        .callees
                        .values()
                        .any(|callee| callee.parameters.contains(&parameter.name)))
                .then_some(weight)
            })
            .collect::<Vec<_>>();
        for weight in weights {
            if available
                .parameters
                .contains(&program.body.parameters[weight].name)
            {
                continue;
            }
            program
                .push(Op::Parameter { weight }, self.signatures)
                .ok()?;
            available.merge(
                program
                    .dependencies
                    .as_ref()
                    .unwrap()
                    .values
                    .last()
                    .unwrap(),
            );
        }
        let roots = required
            .calls
            .iter()
            .filter(|name| {
                !available.definitions.contains(*name)
                    && !required.calls.iter().any(|parent| {
                        parent != *name
                            && program.dependencies.as_ref().unwrap().callees[parent]
                                .definitions
                                .contains(*name)
                    })
            })
            .cloned()
            .collect::<Vec<_>>();
        for name in roots {
            let signature = &self.signatures[&name];
            if !program.dependencies.as_ref().unwrap().callees[&name]
                .arguments
                .is_empty()
                || matches!(signature, TensorSignature::Pure { input_widths, .. } if input_widths.len() == 2)
            {
                continue;
            }
            let ids = (0..program.body.operations.len()).collect::<Vec<_>>();
            let reads = program.body.value_argument_dependencies();
            // An output-independent argument with no numeric reads is already
            // optimal for the causal boundary. Such a root call also commutes;
            // scan-capable cells remain choices because Scan can join state.
            if let Some(arguments) = call_arguments(&program, &name, signature, &[], &ids)
                .into_iter()
                .find(|arguments| arguments.iter().all(|&id| !reads[id]))
            {
                program
                    .push(
                        Op::Call {
                            function: name,
                            arguments,
                        },
                        self.signatures,
                    )
                    .ok()?;
            }
        }
        Some(program)
    }

    fn visit(&self, program: Program, retain_prefix: bool) -> bool {
        let Some(mut program) = self.producers(program) else {
            return false;
        };
        let output = program.planned_output.unwrap();
        let remaining = output + 1 - program.body.operations.len();
        if remaining == 0 {
            return program.outputs().contains(&output);
        }
        let values = frontier(&program);
        if values.iter().any(|(id, key)| {
            program.satisfies_output(&key.dependencies)
                && (0..2).all(|length| program.matches_output(program.shapes[length][*id], length))
        }) {
            // A canonical Tanh can fill every remaining slot while preserving
            // the structural dependencies and shape. No such row is emitted.
            return true;
        }
        let mut keys = values
            .iter()
            .map(|(_, key)| key.clone())
            .collect::<Vec<_>>();
        keys.sort();
        let key = StateKey {
            remaining,
            values: keys,
        };
        // Serialize the complete existing memo key only for an external typed
        // query. Recursive expansion keeps the mask-local cache and does not
        // consume the persistent prefix allowance.
        let prefix_key = retain_prefix.then(|| serde_json::to_vec(&key).unwrap());
        if let Some(key) = prefix_key.as_deref() {
            if let Some(result) = self.cache.prefix(&self.environment, key) {
                return result;
            }
        }
        let cached = self
            .cache
            .entries
            .lock()
            .unwrap()
            .get(&self.environment)
            .and_then(|entries| entries.get(&key.values))
            .and_then(|bounds| bounds.get(key.remaining));
        if let Some(result) = cached {
            if let Some(key) = prefix_key {
                self.cache.remember_prefix(&self.environment, key, result);
            }
            return result;
        }
        let count = program.body.operations.len();
        let mut next = BTreeMap::new();
        for operation in self.operations(&program, &values) {
            if program.push(operation, self.signatures).is_ok() {
                let id = program.body.operations.len() - 1;
                let reads = program.body.value_argument_dependencies();
                let value = value_key(&program, id, reads[id]);
                if value.has_single_row_at_one_token()
                    && !values.iter().any(|(_, available)| available.covers(&value))
                {
                    next.entry(value)
                        .or_insert_with(|| program.body.operations[id].clone());
                }
            }
            // Push can fail after appending the operation but before appending
            // its facts. Restore every stack before the next trial, then keep
            // only the representative operation rather than a cloned prefix.
            program.body.operations.truncate(count);
            for shapes in &mut program.shapes {
                shapes.truncate(count);
            }
            program
                .dependencies
                .as_mut()
                .unwrap()
                .values
                .truncate(count);
        }
        let required = program.entry_requirements.as_ref().unwrap();
        let mut next = next.into_iter().collect::<Vec<_>>();
        next.sort_by_key(|(value, _)| {
            Reverse((
                value
                    .dependencies
                    .definitions
                    .intersection(&required.calls)
                    .count()
                    + value
                        .dependencies
                        .parameters
                        .intersection(&required.parameters)
                        .count()
                    + usize::from(value.dependencies.has_input()),
                value.shapes
                    == std::array::from_fn(|length| {
                        let shape = program
                            .body
                            .signature
                            .output(
                                [1, program.body.max_context][length],
                                program.body.vocabulary,
                            )
                            .unwrap();
                        (shape.rows, shape.columns, shape.float32)
                    }),
                !value.numeric_arguments,
            ))
        });
        let result = next.into_iter().any(|(_, operation)| {
            let mut candidate = program.clone();
            candidate.push(operation, self.signatures).is_ok() && self.visit(candidate, false)
        });
        self.cache
            .entries
            .lock()
            .unwrap()
            .entry(self.environment.clone())
            .or_default()
            .entry(key.values)
            .or_default()
            .remember(key.remaining, result);
        if let Some(key) = prefix_key {
            self.cache.remember_prefix(&self.environment, key, result);
        }
        result
    }

    fn operations(&self, program: &Program, values: &[(usize, ValueKey)]) -> Vec<Op> {
        let widths = program.completion_widths(self.signatures);
        let mut operations = Vec::new();
        for &columns in &widths {
            operations.push(Op::Zeros { columns });
            if columns <= program.raw_tokens.len() {
                operations.push(Op::TokenIndicators {
                    token_ids: program.raw_tokens[..columns].to_vec(),
                });
            }
        }
        for argument in 0..program.body.signature.arguments(1).unwrap().len() {
            operations.push(Op::Argument { argument });
        }
        for weight in 0..program.body.parameters.len() {
            operations.push(Op::Parameter { weight });
        }
        for (input, value) in values {
            operations.push(Op::Broadcast { input: *input });
            for &columns in &widths {
                if value.shapes[0].1 != columns {
                    operations.push(Op::GatherColumns {
                        input: *input,
                        indices: vec![0; columns],
                    });
                }
            }
            for (weight, parameter) in program.body.parameters.iter().enumerate() {
                if parameter.shape.len() == 2 {
                    operations.push(Op::Linear {
                        input: *input,
                        weight,
                    });
                }
            }
        }
        // All acquired values are FP32. Select with a repeated operand has
        // the same shape/dependency transition as a binary elementwise join.
        for (index, (left, shape)) in values.iter().enumerate() {
            for (right_index, (right, other)) in values.iter().enumerate().skip(index + 1) {
                if shape.shapes != other.shapes {
                    continue;
                }
                operations.push(Op::Select {
                    condition: *left,
                    positive: *left,
                    negative: *right,
                });
                for (third, other) in values.iter().skip(right_index + 1) {
                    if shape.shapes == other.shapes {
                        operations.push(Op::Select {
                            condition: *left,
                            positive: *right,
                            negative: *third,
                        });
                    }
                }
            }
        }
        let available = values.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        for (name, signature) in self.signatures {
            for arguments in call_arguments(program, name, signature, &[], &available) {
                operations.push(Op::Call {
                    function: name.clone(),
                    arguments,
                });
            }
            if matches!(signature, TensorSignature::Pure { input_widths, .. } if input_widths.len() == 2)
            {
                for &input in &available {
                    for &initial in &available {
                        operations.push(Op::Scan {
                            input,
                            initial,
                            function: name.clone(),
                            max_steps: program.body.max_context,
                        });
                    }
                }
            }
        }
        operations
    }
}
