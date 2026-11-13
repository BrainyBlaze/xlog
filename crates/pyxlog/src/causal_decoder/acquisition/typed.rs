//! Prospective operand choices use the executable compiler's shape inference.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;

mod completion;
pub(super) use completion::CompletionCache;

use serde_json::{Map, Value};
use xlog_core::XlogError;
use xlog_ir::tensor::{
    TensorOperation as Op, TensorOutputDependencies, TensorProgram, TensorShapeConstraint,
    TensorSignature, TensorValueFacts, TensorValueIdentity, TensorValueShape,
};

pub(super) type Signatures = BTreeMap<String, TensorSignature>;

#[derive(Clone)]
pub(super) enum Operand {
    Value(String),
    Argument(usize),
}

pub(super) struct Ancestor {
    pub(super) fields: Map<String, Value>,
    pub(super) operand: Operand,
}

#[derive(Clone, Copy)]
pub(super) struct PartialConstraints<'a> {
    pub(super) ancestors: &'a [Ancestor],
    pub(super) fields: &'a BTreeMap<String, Vec<String>>,
}

impl PartialConstraints<'_> {
    pub(super) fn expected(
        self,
        program: &Program,
        length: usize,
    ) -> Result<TensorShapeConstraint, XlogError> {
        if program.scalar_resource_domains {
            return Ok(self.expected_pair(program)?[length]);
        }
        let tokens = [1, program.body.max_context][length];
        let mut expected = program
            .body
            .signature
            .output(tokens, program.body.vocabulary)?
            .into();
        for ancestor in self.ancestors {
            expected = program.partial_shape(
                &ancestor.fields,
                Some(&ancestor.operand),
                expected,
                length,
                self.fields,
            )?;
        }
        Ok(expected)
    }

    pub(super) fn permits(self, program: &Program, fields: &Map<String, Value>) -> bool {
        if program.partial_dependency_domains
            && !program.dependency_continuation_possible(
                program.partial_dependencies(fields, None, self.fields),
                self.ancestors,
                self.fields,
            )
        {
            return false;
        }
        if program.scalar_resource_domains {
            return program.partial_feature_possible(fields, self.ancestors)
                && self
                    .expected_pair(program)
                    .and_then(|expected| {
                        program.partial_shapes(fields, None, expected, self.fields)
                    })
                    .is_ok();
        }
        program.partial_feature_possible(fields, self.ancestors)
            && (0..2).all(|length| {
                self.expected(program, length)
                    .and_then(|expected| {
                        program.partial_shape(fields, None, expected, length, self.fields)
                    })
                    .is_ok()
            })
    }

    fn allows(self, program: &Program, id: usize) -> bool {
        if program.partial_dependency_domains
            && !program.dependency_continuation_possible(
                program
                    .dependencies
                    .as_ref()
                    .and_then(|facts| facts.values.get(id))
                    .cloned(),
                self.ancestors,
                self.fields,
            )
        {
            return false;
        }
        if program.scalar_resource_domains {
            return program.feature_continuation_possible(
                program
                    .dependencies
                    .as_ref()
                    .is_some_and(|facts| facts.predicates[id]),
                self.ancestors,
            ) && self.expected_pair(program).is_ok_and(|expected| {
                expected.into_iter().enumerate().all(|(length, mut shape)| {
                    shape.intersect(program.shapes[length][id].into()).is_ok()
                })
            });
        }
        program.feature_continuation_possible(
            program
                .dependencies
                .as_ref()
                .is_some_and(|facts| facts.predicates[id]),
            self.ancestors,
        ) && (0..2).all(|length| {
            self.expected(program, length).is_ok_and(|mut expected| {
                expected
                    .intersect(program.shapes[length][id].into())
                    .is_ok()
            })
        })
    }

    fn expected_pair(self, program: &Program) -> Result<[TensorShapeConstraint; 2], XlogError> {
        let mut expected = [TensorShapeConstraint::default(); 2];
        for (length, tokens) in [1, program.body.max_context].into_iter().enumerate() {
            expected[length] = program
                .body
                .signature
                .output(tokens, program.body.vocabulary)?
                .into();
        }
        for ancestor in self.ancestors {
            expected = program.partial_shapes(
                &ancestor.fields,
                Some(&ancestor.operand),
                expected,
                self.fields,
            )?;
        }
        Ok(expected)
    }
}

/// Only source-known siblings are carried here. An unknown sibling ends the
/// chain; reaching the root instead requires the current definition's output.
#[derive(Clone, Copy)]
pub(super) struct Continuation<'a> {
    pub(super) ancestors: &'a [Ancestor],
    pub(super) root: bool,
    pub(super) partial: Option<PartialConstraints<'a>>,
    pub(super) expected: Option<[TensorShapeConstraint; 2]>,
    pub(super) operation_count: Option<usize>,
}

impl Continuation<'_> {
    fn allows(self, program: &Program, signatures: &Signatures) -> bool {
        let id = program.body.operations.len() - 1;
        self.allows_value(program, id, signatures)
    }

    fn allows_value(self, program: &Program, id: usize, signatures: &Signatures) -> bool {
        if self.expected.is_some_and(|expected| {
            expected.into_iter().enumerate().any(|(length, mut shape)| {
                shape.intersect(program.shapes[length][id].into()).is_err()
            })
        }) {
            return false;
        }
        if self
            .partial
            .is_some_and(|partial| !partial.allows(program, id))
        {
            return false;
        }
        let Some((ancestor, rest)) = self.ancestors.split_first() else {
            return !self.root
                || (self
                    .operation_count
                    .is_none_or(|count| program.body.operations.len() == count)
                    && program.outputs().contains(&id));
        };
        let mut fields = ancestor.fields.clone();
        match &ancestor.operand {
            Operand::Value(name) => {
                fields.insert(name.clone(), Value::from(id));
            }
            Operand::Argument(slot) => {
                let arguments = fields
                    .get_mut("arguments")
                    .and_then(Value::as_array_mut)
                    .unwrap();
                debug_assert_eq!(arguments.len(), *slot);
                arguments.push(Value::from(id));
            }
        }
        program.possible(
            &fields,
            None,
            signatures,
            Some(Self {
                ancestors: rest,
                root: self.root,
                // The partial path belongs to the original current candidate,
                // not each subsequent concrete ancestor or generic successor.
                partial: None,
                expected: None,
                operation_count: self.operation_count,
            }),
        )
    }
}

#[derive(Clone)]
pub(super) struct EntryRequirements {
    pub(super) calls: BTreeSet<String>,
    pub(super) parameters: BTreeSet<String>,
}

#[derive(Clone)]
struct OutputDependencies {
    callees: BTreeMap<String, TensorOutputDependencies>,
    values: Vec<TensorOutputDependencies>,
    predicates: Vec<bool>,
    reserve_frontier: bool,
    completion_cache: Option<Arc<CompletionCache>>,
    value_identities: bool,
    predicate_values: bool,
    feature_output: bool,
}

impl EntryRequirements {
    pub(super) fn producers(
        &self,
        callees: &BTreeMap<String, TensorOutputDependencies>,
    ) -> Option<Self> {
        let mut producers = self.clone();
        for name in &self.calls {
            let callee = callees.get(name)?;
            for dependency in &callee.definitions {
                if dependency != name {
                    producers.calls.remove(dependency);
                }
            }
            producers
                .parameters
                .retain(|name| !callee.parameters.contains(name));
        }
        Some(producers)
    }
}

impl OutputDependencies {
    fn operation(
        &self,
        body: &TensorProgram,
        operation: &Op,
    ) -> Result<TensorValueFacts, XlogError> {
        self.operation_with(body, operation, &|id| {
            self.values
                .get(id)
                .cloned()
                .map(|dependencies| TensorValueFacts {
                    dependencies,
                    zero_or_nan: self.predicates[id],
                })
                .ok_or_else(|| XlogError::Compilation("dependency value is not preceding".into()))
        })
    }

    fn operation_with(
        &self,
        body: &TensorProgram,
        operation: &Op,
        value: &dyn Fn(usize) -> Result<TensorValueFacts, XlogError>,
    ) -> Result<TensorValueFacts, XlogError> {
        let mut resolve = |name: &str| {
            self.callees.get(name).cloned().ok_or_else(|| {
                XlogError::Compilation("callee output dependencies are absent".into())
            })
        };
        if self.predicate_values {
            return body.operation_predicate_dependencies(
                operation,
                self.values.len(),
                value,
                &mut resolve,
            );
        }
        let dependencies = if self.value_identities {
            body.operation_value_dependencies(
                operation,
                self.values.len(),
                &|id| Ok(value(id)?.dependencies),
                &mut resolve,
            )
        } else {
            body.operation_output_dependencies(
                operation,
                &|id| Ok(value(id)?.dependencies),
                &mut resolve,
            )
        }?;
        Ok(TensorValueFacts {
            dependencies,
            zero_or_nan: false,
        })
    }
}

/// Disjoint required facts that no available value or intrinsic callee output
/// joins yet. Transitive overlap is deliberately optimistic: this is a row
/// lower bound, not a synthesized completion or a shape-reachability proof.
struct DependencyFrontier {
    definitions: BTreeMap<String, usize>,
    parameters: BTreeMap<String, usize>,
    parents: Vec<usize>,
    groups: usize,
}

impl DependencyFrontier {
    fn new(required: &EntryRequirements) -> Self {
        // Group zero represents either numeric input or raw tokens.
        let definitions = required
            .calls
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), index + 1))
            .collect();
        let parameters = required
            .parameters
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), index + 1 + required.calls.len()))
            .collect();
        let groups = 1 + required.calls.len() + required.parameters.len();
        Self {
            definitions,
            parameters,
            parents: (0..groups).collect(),
            groups,
        }
    }

    fn contains(&self, dependencies: &TensorOutputDependencies) -> bool {
        dependencies.tokens
            || dependencies
                .definitions
                .iter()
                .any(|name| self.definitions.contains_key(name))
            || dependencies
                .parameters
                .iter()
                .any(|name| self.parameters.contains_key(name))
    }

    fn root(&self, mut index: usize) -> usize {
        while self.parents[index] != index {
            index = self.parents[index];
        }
        index
    }

    fn connect(&mut self, dependencies: &TensorOutputDependencies, numeric_input: bool) {
        let indices = dependencies
            .definitions
            .iter()
            .filter_map(|name| self.definitions.get(name).copied())
            .chain(
                dependencies
                    .parameters
                    .iter()
                    .filter_map(|name| self.parameters.get(name).copied()),
            )
            .chain(
                (dependencies.tokens || (numeric_input && dependencies.has_input())).then_some(0),
            )
            .collect::<Vec<_>>();
        let Some(&first) = indices.first() else {
            return;
        };
        let first = self.root(first);
        for index in indices.into_iter().skip(1) {
            let other = self.root(index);
            if first != other {
                self.parents[other] = first;
                self.groups -= 1;
            }
        }
    }
}

#[derive(Clone)]
pub(super) struct Program {
    pub(super) body: TensorProgram,
    pub(super) planned_output: Option<usize>,
    pub(super) scalar_resource_domains: bool,
    pub(super) partial_dependency_domains: bool,
    /// Bound future work without synthesizing a completion graph.
    pub(super) necessary_continuations: bool,
    /// Maximum rows available to an open body, without selecting its output.
    pub(super) row_allowance: Option<usize>,
    pub(super) entry_requirements: Option<EntryRequirements>,
    shapes: [Vec<TensorValueShape>; 2],
    raw_tokens: Vec<u32>,
    column_limit: usize,
    dependencies: Option<OutputDependencies>,
}

impl Program {
    /// An upper bound on output-reaching identities, not a completion graph.
    /// Unknown value slots remain unbounded. Scalar tails cannot introduce
    /// identities, except the actual physical-weight and callee domains.
    /// Joining all operands is optimistic: canonical value/predicate transfer
    /// may discard dependencies, but cannot create another physical alias or call.
    fn partial_dependencies(
        &self,
        fields: &Map<String, Value>,
        active: Option<(&Operand, &TensorOutputDependencies)>,
        catalogue: &BTreeMap<String, Vec<String>>,
    ) -> Option<TensorOutputDependencies> {
        let facts = self.dependencies.as_ref()?;
        let required = catalogue.get(fields.get("operation")?.as_str()?)?;
        let value = |value: &Value| facts.values.get(value.as_u64()? as usize);
        let mut result = TensorOutputDependencies::default();
        for field in required {
            if super::is_value_reference(field) {
                let dependency = match active {
                    Some((Operand::Value(name), dependency)) if name == field => dependency,
                    _ => value(fields.get(field)?)?,
                };
                result.merge(dependency);
            } else if field == "inputs" {
                // Unlike Call's expression array, this scalar array enters the
                // selected fields only after its closing delimiter was read.
                for input in fields.get(field)?.as_array()? {
                    result.merge(value(input)?);
                }
            } else if field == "arguments" {
                // The canonical callee summary owns which arguments can reach
                // its output. A missing used argument is still unrestricted;
                // unused future arguments cannot add output dependencies.
                let function = fields.get("function")?.as_str()?;
                let callee = facts.callees.get(function)?;
                for slot in &callee.arguments {
                    let dependency = match active {
                        Some((Operand::Argument(index), dependency)) if index == slot => dependency,
                        _ => value(fields.get(field)?.as_array()?.get(*slot)?)?,
                    };
                    result.merge(dependency);
                }
            } else if field == "weight" {
                let selected = fields.get(field).and_then(Value::as_u64);
                for (index, parameter) in self.body.parameters.iter().enumerate() {
                    if selected.is_none_or(|selected| selected == index as u64) {
                        result.parameters.insert(parameter.name.clone());
                    }
                }
            } else if field == "function" {
                let selected = fields.get(field).and_then(Value::as_str);
                for (name, callee) in &facts.callees {
                    if selected.is_none_or(|selected| selected == name) {
                        result.merge(callee);
                        result.definitions.insert(name.clone());
                    }
                }
            }
        }
        Some(result)
    }

    fn dependency_continuation_possible(
        &self,
        mut dependencies: Option<TensorOutputDependencies>,
        ancestors: &[Ancestor],
        catalogue: &BTreeMap<String, Vec<String>>,
    ) -> bool {
        let Some(required) = &self.entry_requirements else {
            return true;
        };
        for parent in ancestors.iter().rev() {
            let Some(current) = dependencies else {
                return true;
            };
            dependencies = self.partial_dependencies(
                &parent.fields,
                Some((&parent.operand, &current)),
                catalogue,
            );
        }
        // This bound concerns only required identities. Numerical input,
        // returned-value identity and feature predicates retain their owners.
        dependencies.is_none_or(|possible| {
            required.calls.is_subset(&possible.definitions)
                && required.parameters.is_subset(&possible.parameters)
        })
    }

    fn partial_predicate(
        &self,
        fields: &Map<String, Value>,
        active: Option<(&Operand, bool)>,
    ) -> bool {
        let Some(facts) = &self.dependencies else {
            return false;
        };
        let known = |value: &Value| {
            value
                .as_u64()
                .and_then(|id| facts.predicates.get(id as usize))
                .copied()
                .unwrap_or(false)
        };
        let operand = |name: &str| {
            if let Some((Operand::Value(field), proof)) = active {
                if field == name {
                    return proof;
                }
            }
            if name == "returned" {
                return fields
                    .get("function")
                    .and_then(Value::as_str)
                    .and_then(|name| facts.callees.get(name))
                    .is_some_and(|callee| match callee.value_identity {
                        Some(TensorValueIdentity::Zero) => true,
                        Some(TensorValueIdentity::Argument(slot)) => {
                            if let Some((Operand::Argument(current), proof)) = active {
                                if *current == slot {
                                    return proof;
                                }
                            }
                            fields
                                .get("arguments")
                                .and_then(Value::as_array)
                                .and_then(|values| values.get(slot))
                                .is_some_and(known)
                        }
                        _ => false,
                    });
            }
            fields.get(name).is_some_and(known)
        };
        let inputs = fields
            .get("inputs")
            .and_then(Value::as_array)
            .map(|values| values.iter().map(known).collect::<Vec<_>>());
        TensorValueFacts::prove_zero_or_nan(
            fields["operation"].as_str().unwrap(),
            &operand,
            inputs.as_deref(),
        )
    }

    fn feature_continuation_possible(&self, mut proof: bool, ancestors: &[Ancestor]) -> bool {
        if !self
            .dependencies
            .as_ref()
            .is_some_and(|facts| facts.feature_output)
        {
            return true;
        }
        // These are the complete source-selected ancestors, even when concrete
        // shape/operation continuation stops at an unknown sibling or scalar.
        for parent in ancestors.iter().rev() {
            proof = self.partial_predicate(&parent.fields, Some((&parent.operand, proof)));
        }
        !proof
    }

    fn partial_feature_possible(
        &self,
        fields: &Map<String, Value>,
        ancestors: &[Ancestor],
    ) -> bool {
        if !self
            .dependencies
            .as_ref()
            .is_some_and(|facts| facts.feature_output)
        {
            return true;
        }
        self.feature_continuation_possible(self.partial_predicate(fields, None), ancestors)
    }

    pub(super) fn entry_feature_possible(&self, id: usize) -> bool {
        self.dependencies.as_ref().is_none_or(|facts| {
            !facts.feature_output || facts.predicates.get(id).is_some_and(|proof| !proof)
        })
    }

    fn partial_shape(
        &self,
        fields: &Map<String, Value>,
        active: Option<&Operand>,
        mut output: TensorShapeConstraint,
        length: usize,
        catalogue: &BTreeMap<String, Vec<String>>,
    ) -> Result<TensorShapeConstraint, XlogError> {
        let tag = fields["operation"].as_str().unwrap();
        let names = catalogue[tag]
            .iter()
            .filter(|field| super::is_value_reference(field))
            .collect::<Vec<_>>();
        let mut inputs = names
            .iter()
            .map(|name| {
                fields.get(*name).and_then(Value::as_u64).map_or(
                    Ok(TensorShapeConstraint::default()),
                    |id| {
                        self.shapes[length]
                            .get(id as usize)
                            .copied()
                            .map(TensorShapeConstraint::from)
                            .ok_or_else(|| {
                                XlogError::Compilation("partial operand is not preceding".into())
                            })
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let related = Op::constrain_shapes(
            tag,
            &mut inputs,
            &mut output,
            [1, self.body.max_context][length],
            fields
                .get("indices")
                .and_then(Value::as_array)
                .map(Vec::len),
        )?;
        // An unsupported relation leaves the active child free. No scalar,
        // callee, shape or preceding value is supplied for that future slot.
        match active {
            Some(Operand::Value(name)) if related => Ok(inputs[names
                .iter()
                .position(|field| field.as_str() == name.as_str())
                .unwrap()]),
            Some(_) => Ok(TensorShapeConstraint::default()),
            None => Ok(output),
        }
    }

    fn partial_shapes(
        &self,
        fields: &Map<String, Value>,
        active: Option<&Operand>,
        output: [TensorShapeConstraint; 2],
        catalogue: &BTreeMap<String, Vec<String>>,
    ) -> Result<[TensorShapeConstraint; 2], XlogError> {
        if fields["operation"].as_str() != Some("linear") {
            return Ok([
                self.partial_shape(fields, active, output[0], 0, catalogue)?,
                self.partial_shape(fields, active, output[1], 1, catalogue)?,
            ]);
        }
        let mut input = [TensorShapeConstraint::default(); 2];
        if let Some(id) = fields.get("input").and_then(Value::as_u64) {
            for (length, shape) in input.iter_mut().enumerate() {
                *shape = self.shapes[length]
                    .get(id as usize)
                    .copied()
                    .ok_or_else(|| {
                        XlogError::Compilation("partial operand is not preceding".into())
                    })?
                    .into();
            }
        }
        let selected = fields
            .get("weight")
            .map(|value| {
                value.as_u64().ok_or_else(|| {
                    XlogError::Compilation("linear weight must be a parameter index".into())
                })
            })
            .transpose()?;
        let mut admitted: Option<([TensorShapeConstraint; 2], [TensorShapeConstraint; 2])> = None;
        for (index, parameter) in self.body.parameters.iter().enumerate() {
            if selected.is_some_and(|selected| selected != index as u64) {
                continue;
            }
            let mut candidate_input = input;
            let mut candidate_output = output;
            // One physical index must be admitted at both lengths. These are
            // actual closed resources, not prospective activation choices.
            if (0..2).any(|length| {
                Op::constrain_linear_shapes(
                    parameter,
                    &mut candidate_input[length],
                    &mut candidate_output[length],
                )
                .is_err()
            }) {
                continue;
            }
            admitted = Some(
                admitted.map_or((candidate_input, candidate_output), |previous| {
                    (
                        std::array::from_fn(|length| {
                            previous.0[length].common(candidate_input[length])
                        }),
                        std::array::from_fn(|length| {
                            previous.1[length].common(candidate_output[length])
                        }),
                    )
                }),
            );
        }
        let Some((admitted_input, admitted_output)) = admitted else {
            return Err(XlogError::Compilation(
                "linear has no compatible declared matrix weight".into(),
            ));
        };
        Ok(match active {
            Some(Operand::Value(name)) if name == "input" => admitted_input,
            Some(_) => [TensorShapeConstraint::default(); 2],
            None => admitted_output,
        })
    }

    /// Bound retained prefix state without allocating a serialized copy. The
    /// shared completion cache is not copied into an operation checkpoint.
    pub(super) fn fits_prefix_cache(&self, limit: usize) -> bool {
        struct Extent(usize);
        impl std::io::Write for Extent {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0 = self
                    .0
                    .checked_sub(bytes.len())
                    .ok_or(std::io::ErrorKind::WriteZero)?;
                Ok(bytes.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let shapes = self.shapes.each_ref().map(|shapes| {
            shapes
                .iter()
                .map(|shape| (shape.rows, shape.columns, shape.float32))
                .collect::<Vec<_>>()
        });
        serde_json::to_writer(
            Extent(limit),
            &(
                &self.body,
                self.planned_output,
                self.row_allowance,
                self.entry_requirements
                    .as_ref()
                    .map(|required| (&required.calls, &required.parameters)),
                shapes,
                &self.raw_tokens,
                self.column_limit,
                self.dependencies
                    .as_ref()
                    .map(|facts| (&facts.callees, &facts.values, facts.reserve_frontier)),
            ),
        )
        .is_ok()
    }

    pub(super) fn new(
        signature: TensorSignature,
        vocabulary: usize,
        max_context: usize,
        raw_tokens: Vec<u32>,
        column_limit: usize,
    ) -> Self {
        Self {
            body: TensorProgram {
                signature,
                parameters: Vec::new(),
                operations: Vec::new(),
                output: 0,
                vocabulary,
                max_context,
            },
            planned_output: None,
            scalar_resource_domains: false,
            partial_dependency_domains: false,
            necessary_continuations: false,
            row_allowance: None,
            entry_requirements: None,
            shapes: [Vec::new(), Vec::new()],
            raw_tokens,
            column_limit,
            dependencies: None,
        }
    }

    pub(super) fn set_output_dependencies(
        &mut self,
        callees: BTreeMap<String, TensorOutputDependencies>,
        reserve_frontier: bool,
        completion_cache: Option<Arc<CompletionCache>>,
        value_identities: bool,
        predicate_values: bool,
        feature_output: bool,
    ) {
        self.dependencies = Some(OutputDependencies {
            callees,
            values: Vec::new(),
            predicates: Vec::new(),
            reserve_frontier,
            completion_cache,
            value_identities,
            predicate_values,
            feature_output,
        });
    }

    pub(super) fn output_dependencies(&self) -> Result<TensorOutputDependencies, String> {
        self.dependencies
            .as_ref()
            .and_then(|dependencies| dependencies.values.get(self.body.output))
            .cloned()
            .map(TensorOutputDependencies::into_callee_summary)
            .ok_or_else(|| "numerical output dependencies are absent".to_owned())
    }

    fn infer(
        &self,
        operation: &Op,
        signatures: &Signatures,
        length: usize,
    ) -> Result<TensorValueShape, XlogError> {
        let tokens = [1, self.body.max_context][length];
        self.body.operation_shape_with(
            operation,
            tokens,
            &|id| {
                self.shapes[length].get(id).copied().ok_or_else(|| {
                    XlogError::Compilation("numerical value is not preceding".into())
                })
            },
            &|name| {
                signatures
                    .get(name)
                    .cloned()
                    .ok_or_else(|| XlogError::Compilation("callee must precede its caller".into()))
            },
        )
    }

    pub(super) fn push(&mut self, operation: Op, signatures: &Signatures) -> Result<(), String> {
        self.push_with_dependencies(operation, signatures, None)
    }

    fn push_with_dependencies(
        &mut self,
        operation: Op,
        signatures: &Signatures,
        prospective: Option<&TensorValueFacts>,
    ) -> Result<(), String> {
        if self
            .row_allowance
            .is_some_and(|rows| self.body.operations.len() >= rows)
        {
            return Err("operations exceed the available row allowance".into());
        }
        if self
            .continuation_output()
            .is_some_and(|output| self.body.operations.len() > output)
        {
            return Err("operations exceed the planned output slot".into());
        }
        let first = self
            .infer(&operation, signatures, 0)
            .map_err(|error| error.to_string())?;
        let last = self
            .infer(&operation, signatures, 1)
            .map_err(|error| error.to_string())?;
        if self.is_output_slot()
            && (!self.matches_output(first, 0) || !self.matches_output(last, 1))
        {
            return Err("planned output does not match the signature".into());
        }
        let dependencies = self
            .dependencies
            .as_ref()
            .map(|facts| {
                prospective
                    .cloned()
                    .map_or_else(|| facts.operation(&self.body, &operation), Ok)
            })
            .transpose()
            .map_err(|error| error.to_string())?;
        self.body.operations.push(operation);
        self.body
            .validate_acquired_body()
            .map_err(|error| error.to_string())?;
        if self.continuation_output().is_some_and(|output| {
            !self.completion_possible(&self.body, output, signatures, dependencies.as_ref())
        }) {
            return Err("remaining entry operations cannot cover required calls and reads".into());
        }
        self.shapes[0].push(first);
        self.shapes[1].push(last);
        if let Some(facts) = &mut self.dependencies {
            let dependencies = dependencies.unwrap();
            facts.predicates.push(dependencies.zero_or_nan);
            facts.values.push(dependencies.dependencies);
        }
        Ok(())
    }

    /// Push can fail after appending an operation but before its facts. Every
    /// prospective caller restores all stacks, including that failure path.
    fn with_trial(
        &mut self,
        operation: Op,
        signatures: &Signatures,
        accept: impl FnOnce(&Self) -> bool,
    ) -> bool {
        self.with_trial_dependencies(operation, signatures, None, accept)
    }

    fn with_trial_dependencies(
        &mut self,
        operation: Op,
        signatures: &Signatures,
        prospective: Option<&TensorValueFacts>,
        accept: impl FnOnce(&Self) -> bool,
    ) -> bool {
        let count = self.body.operations.len();
        let accepted = self
            .push_with_dependencies(operation, signatures, prospective)
            .is_ok()
            && accept(self);
        self.body.operations.truncate(count);
        for shapes in &mut self.shapes {
            shapes.truncate(count);
        }
        if let Some(facts) = &mut self.dependencies {
            facts.values.truncate(count);
            facts.predicates.truncate(count);
        }
        accepted
    }

    pub(super) fn is_output_slot(&self) -> bool {
        self.continuation_output() == Some(self.body.operations.len())
    }

    fn continuation_output(&self) -> Option<usize> {
        self.planned_output
            .or_else(|| self.row_allowance.and_then(|rows| rows.checked_sub(1)))
    }

    pub(super) fn output_columns(&self) -> Option<usize> {
        self.is_output_slot().then(|| {
            self.body
                .signature
                .output(1, self.body.vocabulary)
                .unwrap()
                .columns
        })
    }

    fn matches_output(&self, shape: TensorValueShape, length: usize) -> bool {
        self.body
            .signature
            .output([1, self.body.max_context][length], self.body.vocabulary)
            .ok()
            == Some(shape)
    }

    pub(super) fn completion_widths(&self, signatures: &Signatures) -> BTreeSet<usize> {
        let mut widths = BTreeSet::from([1]);
        for signature in std::iter::once(&self.body.signature).chain(signatures.values()) {
            if let Ok(shape) = signature.output(1, self.body.vocabulary) {
                widths.insert(shape.columns);
            }
            if let Ok(arguments) = signature.arguments(1) {
                widths.extend(arguments.into_iter().map(|shape| shape.columns));
            }
        }
        for parameter in &self.body.parameters {
            widths.extend(parameter.shape.iter().copied());
        }
        widths.extend(self.shapes[0].iter().map(|shape| shape.columns));
        widths
    }

    pub(super) fn column_range(&self) -> Range<usize> {
        self.output_columns()
            .map_or(1..self.column_limit + 1, |columns| columns..columns + 1)
    }

    pub(super) fn input_columns(&self, input: usize) -> Option<usize> {
        Some(
            self.shapes[0]
                .get(input)?
                .columns
                .min(self.shapes[1].get(input)?.columns),
        )
    }

    pub(super) fn searches_completion(&self) -> bool {
        self.dependencies
            .as_ref()
            .is_some_and(|facts| facts.completion_cache.is_some())
    }

    pub(super) fn outputs(&self) -> Vec<usize> {
        (0..self.body.operations.len())
            .filter(|&id| {
                self.dependencies.as_ref().is_none_or(|facts| {
                    self.satisfies_output(&facts.values[id], facts.predicates[id])
                }) && [1, self.body.max_context]
                    .into_iter()
                    .enumerate()
                    .all(|(length, tokens)| {
                        self.body
                            .signature
                            .output(tokens, self.body.vocabulary)
                            .ok()
                            == Some(self.shapes[length][id])
                    })
            })
            .collect()
    }

    pub(super) fn required_output(&self) -> Result<[[usize; 2]; 2], String> {
        let shape = |tokens| {
            self.body
                .signature
                .output(tokens, self.body.vocabulary)
                .map(|shape| [shape.rows, shape.columns])
                .map_err(|error| error.to_string())
        };
        Ok([shape(1)?, shape(self.body.max_context)?])
    }

    pub(super) fn last_value(&self) -> Option<[[usize; 2]; 2]> {
        let first = self.shapes[0].last()?;
        let last = self.shapes[1].last()?;
        Some([[first.rows, first.columns], [last.rows, last.columns]])
    }

    fn compatible_with_dependencies(
        &self,
        operation: Op,
        signatures: &Signatures,
        dependencies: Option<&TensorValueFacts>,
    ) -> bool {
        if self
            .row_allowance
            .is_some_and(|rows| self.body.operations.len() >= rows)
        {
            return false;
        }
        if !(0..2).all(|length| {
            self.infer(&operation, signatures, length)
                .is_ok_and(|shape| !self.is_output_slot() || self.matches_output(shape, length))
        }) {
            return false;
        }
        // The same causal-dependency check used by the complete compiler applies
        // after shape inference. No prospective graph is ever emitted.
        let mut body = self.body.clone();
        body.operations.push(operation);
        body.validate_acquired_body().is_ok()
            && self.continuation_output().is_none_or(|output| {
                self.completion_possible(&body, output, signatures, dependencies)
            })
    }

    pub(super) fn permits_plan(&self, output: usize, signatures: &Signatures) -> bool {
        self.completion_possible(&self.body, output, signatures, None)
    }

    pub(super) fn entry_parameters_available(&self) -> bool {
        self.missing_entry_parameters()
            .is_some_and(|missing| missing.is_empty())
    }

    pub(super) fn missing_entry_parameters(&self) -> Option<BTreeSet<String>> {
        let Some(required) = &self.entry_requirements else {
            return Some(BTreeSet::new());
        };
        let mut missing = self
            .dependencies
            .as_ref()
            .and_then(|facts| required.producers(&facts.callees))?
            .parameters;
        missing.retain(|name| {
            !self
                .body
                .parameters
                .iter()
                .any(|parameter| &parameter.name == name)
        });
        Some(missing)
    }

    fn completion_possible(
        &self,
        body: &TensorProgram,
        output: usize,
        signatures: &Signatures,
        dependencies: Option<&TensorValueFacts>,
    ) -> bool {
        if let Some(facts) = &self.dependencies {
            return self.dependency_completion_possible(
                body,
                output,
                signatures,
                facts,
                dependencies,
            );
        }
        let Some(required) = &self.entry_requirements else {
            return true;
        };
        let Some(remaining) = (output + 1).checked_sub(body.operations.len()) else {
            return false;
        };
        let mut calls = required.calls.clone();
        let mut parameters = required.parameters.clone();
        for operation in &body.operations {
            if let Some(name) = operation.callee() {
                calls.remove(name);
            }
            if let Some(weight) = operation.parameter() {
                let Some(parameter) = body.parameters.get(weight) else {
                    return false;
                };
                parameters.remove(&parameter.name);
            }
        }
        let needed = calls.len() + parameters.len();
        if remaining < needed
            || parameters.iter().any(|name| {
                !body
                    .parameters
                    .iter()
                    .any(|parameter| &parameter.name == name)
            })
        {
            return false;
        }
        if needed == 0 || remaining > needed {
            return true;
        }
        // Each outstanding root call or direct physical read needs a distinct
        // row. With no spare row, at least one must also yield the entry result.
        // Allow hypothetical operands here: this is a necessary lower bound,
        // not a claim that those operands already exist or can still be built.
        calls
            .iter()
            .any(|name| self.call_can_finish(name, signatures))
            || body
                .parameters
                .iter()
                .enumerate()
                .any(|(weight, parameter)| {
                    parameters.contains(&parameter.name)
                        && self.parameter_can_finish(weight, signatures)
                })
    }

    fn satisfies_output(&self, dependencies: &TensorOutputDependencies, zero_or_nan: bool) -> bool {
        !self
            .dependencies
            .as_ref()
            .is_some_and(|facts| facts.feature_output && zero_or_nan)
            && self.entry_requirements.as_ref().is_none_or(|required| {
                dependencies.has_input()
                    && required.calls.is_subset(&dependencies.definitions)
                    && required.parameters.is_subset(&dependencies.parameters)
            })
    }

    fn dependency_completion_possible(
        &self,
        body: &TensorProgram,
        output: usize,
        signatures: &Signatures,
        facts: &OutputDependencies,
        prospective: Option<&TensorValueFacts>,
    ) -> bool {
        let Some(remaining) = (output + 1).checked_sub(body.operations.len()) else {
            return false;
        };
        let computed;
        let next = if body.operations.len() > facts.values.len() {
            if body.operations.len() != facts.values.len() + 1 {
                return false;
            }
            match prospective {
                Some(dependencies) => Some(dependencies),
                None => {
                    let Ok(dependencies) = facts.operation(body, body.operations.last().unwrap())
                    else {
                        return false;
                    };
                    computed = dependencies;
                    Some(&computed)
                }
            }
        } else {
            None
        };
        let values = || {
            facts
                .values
                .iter()
                .chain(next.map(|facts| &facts.dependencies))
        };
        if remaining == 0 {
            let proof = facts
                .predicates
                .get(output)
                .copied()
                .or_else(|| next.map(|facts| facts.zero_or_nan))
                .unwrap_or(false);
            return values()
                .nth(output)
                .is_some_and(|dependencies| self.satisfies_output(dependencies, proof));
        }
        let Some(required) = &self.entry_requirements else {
            return true;
        };
        let mut available = TensorOutputDependencies::default();
        for dependencies in values() {
            available.merge(dependencies);
        }

        // Roots are computed from returned values, not from calls serialized
        // elsewhere in helper bodies. Each missing root needs its own call row.
        let Some(EntryRequirements {
            calls: mut roots,
            mut parameters,
        }) = required.producers(&facts.callees)
        else {
            return false;
        };
        roots.retain(|name| !available.definitions.contains(name));
        parameters.retain(|name| !available.parameters.contains(name));
        if parameters.iter().any(|name| {
            !body
                .parameters
                .iter()
                .any(|parameter| &parameter.name == name)
        }) {
            return false;
        }
        // An unseen numeric/token input needs a separate producer unless a
        // required call can itself inspect raw tokens. Existing values are an
        // optimistic union: joining them may still cost additional rows.
        let input_row =
            !available.has_input() && !roots.iter().any(|name| facts.callees[name].tokens);
        let needed = roots.len() + parameters.len() + usize::from(input_row);
        if remaining < needed {
            return false;
        }
        if facts.reserve_frontier {
            let mut frontier = DependencyFrontier::new(required);
            for dependencies in values() {
                frontier.connect(dependencies, true);
            }
            // A helper's own output-reaching facts are one producer group.
            // Formal arguments remain separate until an actual call uses them.
            for dependencies in facts.callees.values() {
                frontier.connect(dependencies, false);
            }
            let call_capacity = |name: &str| {
                let dependencies = &facts.callees[name];
                let intrinsic = usize::from(frontier.contains(dependencies));
                let call = (dependencies.arguments.len() + intrinsic).saturating_sub(1);
                let scan = match signatures.get(name) {
                    Some(TensorSignature::Pure { input_widths, .. }) if input_widths.len() == 2 => {
                        // Scan always carries initial; item is output-used only
                        // when the canonical cell facts name numeric formal 1.
                        (1 + usize::from(dependencies.arguments.contains(&1)) + intrinsic)
                            .saturating_sub(1)
                    }
                    _ => 0,
                };
                call.max(scan)
            };
            // Select joins at most three existing groups. A call or scan may
            // additionally join its intrinsic group and output-used operands.
            let capacity = facts
                .callees
                .keys()
                .map(|name| call_capacity(name))
                .max()
                .unwrap_or(0)
                .max(2);
            let mandatory_capacity =
                roots.iter().map(|name| call_capacity(name)).sum::<usize>()
                    + parameters
                        .iter()
                        .map(|name| {
                            // The acquired grammar reads a physical parameter with
                            // Parameter or Linear; only a matrix can join an input.
                            usize::from(body.parameters.iter().any(|parameter| {
                                &parameter.name == name && parameter.shape.len() == 2
                            }))
                        })
                        .sum::<usize>()
                    + if input_row {
                        facts
                            .callees
                            .iter()
                            .filter(|(_, dependencies)| dependencies.tokens)
                            .map(|(name, _)| call_capacity(name))
                            .max()
                            .unwrap_or(0)
                    } else {
                        0
                    };
            let joins = frontier
                .groups
                .saturating_sub(1)
                .saturating_sub(mandatory_capacity)
                .div_ceil(capacity);
            if remaining < needed + joins {
                return false;
            }
        }
        if !self.necessary_continuations {
            if let Some(cache) = &facts.completion_cache {
                return completion::possible(self, body, output, signatures, cache, prospective);
            }
        }
        if needed == 0 || remaining > needed {
            return true;
        }
        // With no spare row, the result must be one of the missing producers.
        // This remains a necessary shape bound; it never invents an operand.
        roots
            .iter()
            .any(|name| self.call_can_finish(name, signatures))
            || body
                .parameters
                .iter()
                .enumerate()
                .any(|(weight, parameter)| {
                    parameters.contains(&parameter.name)
                        && self.parameter_can_finish(weight, signatures)
                })
            || (input_row && self.input_can_finish(signatures))
    }

    fn input_can_finish(&self, signatures: &Signatures) -> bool {
        (0..self.body.signature.arguments(1).unwrap().len()).any(|argument| {
            (0..2).all(|length| {
                self.infer(&Op::Argument { argument }, signatures, length)
                    .is_ok_and(|shape| self.matches_output(shape, length))
            })
        }) || {
            let columns = self
                .body
                .signature
                .output(1, self.body.vocabulary)
                .unwrap()
                .columns;
            let token_ids: Vec<_> = self.raw_tokens.iter().copied().take(columns).collect();
            (0..2).all(|length| {
                self.infer(
                    &Op::TokenIndicators {
                        token_ids: token_ids.clone(),
                    },
                    signatures,
                    length,
                )
                .is_ok_and(|shape| self.matches_output(shape, length))
            })
        } || self.dependencies.as_ref().is_some_and(|facts| {
            facts.callees.iter().any(|(name, dependencies)| {
                dependencies.tokens && self.call_can_finish(name, signatures)
            })
        })
    }

    fn parameter_can_finish(&self, weight: usize, signatures: &Signatures) -> bool {
        (0..2).all(|length| {
            self.infer(&Op::Parameter { weight }, signatures, length)
                .is_ok_and(|shape| self.matches_output(shape, length))
        }) || [1, self.body.max_context]
            .into_iter()
            .enumerate()
            .all(|(length, tokens)| {
                let parameter = &self.body.parameters[weight];
                let [_, inner] = parameter.shape.as_slice() else {
                    return false;
                };
                let Ok(mut input) = self.body.signature.output(tokens, self.body.vocabulary) else {
                    return false;
                };
                input.columns = *inner;
                self.body
                    .operation_shape_with(
                        &Op::Linear { input: 0, weight },
                        tokens,
                        &|_| Ok(input),
                        &|name| {
                            signatures.get(name).cloned().ok_or_else(|| {
                                XlogError::Compilation("callee must precede its caller".into())
                            })
                        },
                    )
                    .is_ok_and(|shape| self.matches_output(shape, length))
            })
    }

    fn call_can_finish(&self, name: &str, signatures: &Signatures) -> bool {
        let Some(signature) = signatures.get(name) else {
            return false;
        };
        if [1, self.body.max_context]
            .into_iter()
            .enumerate()
            .all(|(length, tokens)| {
                signature
                    .output(tokens, self.body.vocabulary)
                    .is_ok_and(|shape| self.matches_output(shape, length))
            })
        {
            return true;
        }
        if !matches!(signature, TensorSignature::Pure { input_widths, .. } if input_widths.len() == 2)
        {
            return false;
        }
        [1, self.body.max_context]
            .into_iter()
            .enumerate()
            .all(|(length, tokens)| {
                let Ok(mut arguments) = signature.arguments(tokens) else {
                    return false;
                };
                arguments[1].rows = tokens;
                self.body
                    .operation_shape_with(
                        &Op::Scan {
                            input: 1,
                            initial: 0,
                            function: name.into(),
                            max_steps: self.body.max_context,
                        },
                        tokens,
                        &|id| Ok(arguments[id]),
                        &|_| Ok(signature.clone()),
                    )
                    .is_ok_and(|shape| self.matches_output(shape, length))
            })
    }

    fn references(&self, fields: &Map<String, Value>, name: &str, extent: usize) -> Vec<usize> {
        match fields.get(name).and_then(Value::as_u64) {
            Some(id) => vec![id as usize],
            None => (0..extent).collect(),
        }
    }

    fn possible_call(
        &self,
        function: &str,
        signature: &TensorSignature,
        mut arguments: Vec<usize>,
        check: &mut impl FnMut(Op, Option<TensorValueFacts>) -> bool,
    ) -> bool {
        let [Ok(first), Ok(last)] = [
            signature.arguments(1),
            signature.arguments(self.body.max_context),
        ] else {
            return false;
        };
        let count = self.body.operations.len();
        if self.searches_completion() {
            return completion::call_arguments(
                self,
                function,
                signature,
                &arguments,
                &(0..count).collect::<Vec<_>>(),
            )
            .into_iter()
            .any(|arguments| {
                check(
                    Op::Call {
                        function: function.into(),
                        arguments,
                    },
                    None,
                )
            });
        }
        let mut abstract_arguments = arguments.clone();
        let mut abstract_values = BTreeMap::new();
        for slot in arguments.len()..first.len() {
            let choices = (0..count)
                .filter(|&id| self.shapes[0][id] == first[slot] && self.shapes[1][id] == last[slot])
                .collect::<Vec<_>>();
            let Some(&id) = choices.first() else {
                return false;
            };
            arguments.push(id);
            if let Some(facts) = &self.dependencies {
                // Unchosen operands are an optimistic union of shape-valid
                // preceding values, not an authored replacement argument.
                // Once the array is complete, every argument is exact.
                let mut dependencies = TensorOutputDependencies::default();
                let zero_or_nan = choices.iter().all(|&id| facts.predicates[id]);
                let identity = facts.values[id].value_identity.clone();
                let same_identity = choices
                    .iter()
                    .all(|&id| facts.values[id].value_identity == identity);
                for id in choices {
                    dependencies.merge(&facts.values[id]);
                }
                if facts.value_identities {
                    dependencies.value_identity = Some(if same_identity {
                        identity.unwrap_or(TensorValueIdentity::Unknown)
                    } else {
                        TensorValueIdentity::Unknown
                    });
                }
                let id = count + slot;
                abstract_arguments.push(id);
                abstract_values.insert(
                    id,
                    TensorValueFacts {
                        dependencies,
                        zero_or_nan,
                    },
                );
            }
        }
        let operation = Op::Call {
            function: function.into(),
            arguments,
        };
        let Some(facts) = self
            .dependencies
            .as_ref()
            .filter(|_| !abstract_values.is_empty())
        else {
            return check(operation, None);
        };
        let dependencies = facts.operation_with(
            &self.body,
            &Op::Call {
                function: function.into(),
                arguments: abstract_arguments,
            },
            &|id| {
                abstract_values
                    .get(&id)
                    .cloned()
                    .or_else(|| {
                        facts
                            .values
                            .get(id)
                            .cloned()
                            .map(|dependencies| TensorValueFacts {
                                dependencies,
                                zero_or_nan: facts.predicates[id],
                            })
                    })
                    .ok_or_else(|| XlogError::Compilation("dependency value is absent".into()))
            },
        );
        dependencies.is_ok_and(|dependencies| check(operation, Some(dependencies)))
    }

    pub(super) fn number_possible(
        &self,
        fields: &Map<String, Value>,
        name: &str,
        value: usize,
        signatures: &Signatures,
        continuation: Option<Continuation<'_>>,
    ) -> bool {
        let mut candidate = fields.clone();
        candidate.insert(name.into(), Value::from(value));
        self.possible(&candidate, None, signatures, continuation)
    }

    pub(super) fn array_range(
        &self,
        fields: &Map<String, Value>,
        name: &str,
        prefix: &[usize],
    ) -> Range<usize> {
        if name == "indices" {
            if self
                .output_columns()
                .is_some_and(|columns| prefix.len() >= columns)
            {
                return 0..0;
            }
            let id = fields["input"].as_u64().unwrap() as usize;
            return 0..self
                .shapes
                .iter()
                .map(|shapes| shapes[id].columns)
                .min()
                .unwrap();
        }
        0..self.body.operations.len()
    }

    /// Numeric literal arrays have one coordinate per output column. Reuse the
    /// same partial ancestor relations at both lengths; unknown widths stay free.
    pub(super) fn array_minimum(
        &self,
        fields: &Map<String, Value>,
        name: &str,
        continuation: Option<Continuation<'_>>,
    ) -> Result<usize, XlogError> {
        if !matches!(name, "indices" | "token_ids") {
            return Ok(0);
        }
        let mut minimum = self.output_columns().unwrap_or(1);
        let mut fields = fields.clone();
        // A prefix length is not the selected final width of an open array.
        fields.remove(name);
        for length in 0..2 {
            let mut expected = TensorShapeConstraint::default();
            if let Some(partial) = continuation.and_then(|value| value.partial) {
                expected = self.partial_shape(
                    &fields,
                    None,
                    partial.expected(self, length)?,
                    length,
                    partial.fields,
                )?;
            }
            if let Some(constraints) = continuation.and_then(|value| value.expected) {
                expected.intersect(constraints[length])?;
            }
            if let Some(columns) = expected.columns {
                minimum = minimum.max(columns);
            }
        }
        Ok(minimum)
    }

    pub(super) fn array_number_possible(
        &self,
        fields: &Map<String, Value>,
        name: &str,
        prefix: &[usize],
        value: usize,
        signatures: &Signatures,
        continuation: Option<Continuation<'_>>,
    ) -> bool {
        if name == "indices" && !self.searches_completion() && continuation.is_none() {
            // The contiguous domain is exact; the complete array still uses
            // canonical gather inference in array_can_end.
            return true;
        }
        let mut candidate = fields.clone();
        let mut values = prefix.to_vec();
        values.push(value);
        candidate.insert(name.into(), serde_json::json!(values));
        self.possible(&candidate, Some(name), signatures, continuation)
    }

    pub(super) fn array_can_end(
        &self,
        fields: &Map<String, Value>,
        name: &str,
        prefix: &[usize],
        signatures: &Signatures,
        continuation: Option<Continuation<'_>>,
    ) -> bool {
        let mut candidate = fields.clone();
        candidate.insert(name.into(), serde_json::json!(prefix));
        self.possible(&candidate, None, signatures, continuation)
    }

    /// Necessary leaf feasibility uses the same scalar domains and canonical
    /// push/rollback, with no output slot or generic successor search.
    pub(super) fn leaf_possible(
        &self,
        catalogue: &BTreeMap<String, Vec<String>>,
        signatures: &Signatures,
        expected: [TensorShapeConstraint; 2],
    ) -> bool {
        let mut program = self.clone();
        program.planned_output = None;
        program.row_allowance = None;
        catalogue.iter().any(|(name, required)| {
            if required
                .iter()
                .any(|field| super::is_value_reference(field))
            {
                return false;
            }
            let mut fields = Map::from_iter([("operation".into(), Value::from(name.as_str()))]);
            if required.iter().any(|field| field == "arguments") {
                // A closed empty argument domain admits only real zero-arity
                // callees. It supplies neither an SSA operand nor source bytes.
                fields.insert("arguments".into(), Value::Array(Vec::new()));
            }
            program.possible(
                &fields,
                None,
                signatures,
                Some(Continuation {
                    ancestors: &[],
                    root: false,
                    partial: None,
                    expected: Some(expected),
                    operation_count: None,
                }),
            )
        })
    }

    /// Check exactly the existing source-selected value through the same
    /// canonical ancestors, without appending a leaf or a generic successor.
    pub(super) fn reference_possible(
        &self,
        id: usize,
        signatures: &Signatures,
        continuation: Continuation<'_>,
    ) -> bool {
        if id >= self.body.operations.len() {
            return false;
        }
        let mut program = self.clone();
        program.planned_output = None;
        program.row_allowance = None;
        continuation.allows_value(&program, id, signatures)
    }

    pub(super) fn source_call_arity(
        &self,
        signature: &TensorSignature,
        arguments: &[Value],
        minimum: usize,
        expected: [TensorShapeConstraint; 2],
    ) -> Option<usize> {
        let count = signature.arguments(1).ok()?.len();
        if count < minimum {
            return None;
        }
        (0..2)
            .all(|length| {
                signature
                    .call_output(
                        &self.body.signature,
                        [1, self.body.max_context][length],
                        self.body.vocabulary,
                        (0..count).map(|slot| {
                            arguments.get(slot).map_or(Ok(None), |value| {
                                value
                                    .as_u64()
                                    .and_then(|id| self.shapes[length].get(id as usize))
                                    .copied()
                                    .map(Some)
                                    .ok_or_else(|| {
                                        XlogError::Compilation(
                                            "call operand is not preceding".into(),
                                        )
                                    })
                            })
                        }),
                    )
                    .is_ok_and(|shape| {
                        let mut required = expected[length];
                        required.intersect(shape.into()).is_ok()
                    })
            })
            .then_some(count)
    }

    pub(super) fn possible(
        &self,
        fields: &Map<String, Value>,
        open_array: Option<&str>,
        signatures: &Signatures,
        continuation: Option<Continuation<'_>>,
    ) -> bool {
        let Some(tag) = fields.get("operation").and_then(Value::as_str) else {
            return false;
        };
        let count = self.body.operations.len();
        let widths = || {
            let mut widths = self.completion_widths(signatures);
            if let Some(expected) = continuation.and_then(|continuation| continuation.expected) {
                widths.extend(expected.into_iter().filter_map(|shape| shape.columns));
            }
            widths
        };
        let refs = |name| {
            if self.searches_completion() && fields.get(name).and_then(Value::as_u64).is_none() {
                completion::operand_representatives(self)
            } else {
                self.references(fields, name, count)
            }
        };
        let scalar = |name: &str, first: usize| {
            fields
                .get(name)
                .and_then(Value::as_u64)
                .map_or(first, |value| value as usize)
        };
        let array = |name: &str| {
            fields.get(name).and_then(Value::as_array).map(|values| {
                values
                    .iter()
                    .map(|value| value.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>()
            })
        };
        let mut trial = continuation.map(|_| {
            let mut trial = self.clone();
            // Open ancestors already count in the source reader. This trial
            // checks their local types, not an additional flat-row completion.
            // With no output slot it cannot consult or poison generic caches.
            trial.planned_output = None;
            trial.row_allowance = None;
            trial
        });
        let mut check = |operation: Op, dependencies: Option<TensorValueFacts>| {
            if let Some(continuation) = continuation {
                let prospective = dependencies.as_ref().filter(|_| {
                    self.dependencies
                        .as_ref()
                        .is_some_and(|facts| facts.value_identities)
                });
                if !trial.as_mut().unwrap().with_trial_dependencies(
                    operation.clone(),
                    signatures,
                    prospective,
                    |program| continuation.allows(program, signatures),
                ) {
                    return false;
                }
            }
            self.compatible_with_dependencies(operation, signatures, dependencies.as_ref())
        };
        match tag {
            "zeros" => {
                if self.searches_completion() && !fields.contains_key("columns") {
                    widths()
                        .into_iter()
                        .any(|columns| check(Op::Zeros { columns }, None))
                } else {
                    check(
                        Op::Zeros {
                            columns: scalar("columns", self.output_columns().unwrap_or(1)),
                        },
                        None,
                    )
                }
            }
            "argument" => self
                .references(
                    fields,
                    "argument",
                    self.body.signature.arguments(1).unwrap().len(),
                )
                .into_iter()
                .any(|argument| check(Op::Argument { argument }, None)),
            "parameter" => self
                .references(fields, "weight", self.body.parameters.len())
                .into_iter()
                .any(|weight| check(Op::Parameter { weight }, None)),
            "token_indicators" => {
                let mut token_ids = array("token_ids").unwrap_or_default();
                if self.searches_completion()
                    && (!fields.contains_key("token_ids") || open_array == Some("token_ids"))
                {
                    let mut widths = widths();
                    widths.insert(token_ids.len().max(1));
                    return widths
                        .into_iter()
                        .filter(|&columns| columns >= token_ids.len())
                        .any(|columns| {
                            let mut candidate = token_ids.clone();
                            for &id in &self.raw_tokens {
                                if candidate.len() >= columns {
                                    break;
                                }
                                if !candidate.contains(&(id as usize)) {
                                    candidate.push(id as usize);
                                }
                            }
                            candidate.len() == columns
                                && check(
                                    Op::TokenIndicators {
                                        token_ids: candidate
                                            .into_iter()
                                            .map(|id| id as u32)
                                            .collect(),
                                    },
                                    None,
                                )
                        });
                }
                if !fields.contains_key("token_ids") || open_array == Some("token_ids") {
                    let columns = self.output_columns().unwrap_or(1);
                    // Prospective distinct IDs prove that a continuation exists;
                    // only the model's actual array is ever added to the body.
                    for &id in &self.raw_tokens {
                        if token_ids.len() >= columns {
                            break;
                        }
                        if !token_ids.contains(&(id as usize)) {
                            token_ids.push(id as usize);
                        }
                    }
                }
                check(
                    Op::TokenIndicators {
                        token_ids: token_ids.into_iter().map(|id| id as u32).collect(),
                    },
                    None,
                )
            }
            "broadcast" => refs("input")
                .into_iter()
                .any(|input| check(Op::Broadcast { input }, None)),
            "tanh" => refs("input")
                .into_iter()
                .any(|input| check(Op::Tanh { input }, None)),
            "linear" => refs("input").into_iter().any(|input| {
                self.references(fields, "weight", self.body.parameters.len())
                    .into_iter()
                    .any(|weight| check(Op::Linear { input, weight }, None))
            }),
            "matmul" => refs("left").into_iter().any(|left| {
                refs("right")
                    .into_iter()
                    .any(|right| check(Op::MatMul { left, right }, None))
            }),
            "add" | "subtract" | "multiply" | "divide" => refs("left").into_iter().any(|left| {
                refs("right").into_iter().any(|right| {
                    check(
                        match tag {
                            "add" => Op::Add { left, right },
                            "subtract" => Op::Subtract { left, right },
                            "multiply" => Op::Multiply { left, right },
                            _ => Op::Divide { left, right },
                        },
                        None,
                    )
                })
            }),
            "select" => refs("condition").into_iter().any(|condition| {
                refs("positive").into_iter().any(|positive| {
                    refs("negative").into_iter().any(|negative| {
                        check(
                            Op::Select {
                                condition,
                                positive,
                                negative,
                            },
                            None,
                        )
                    })
                })
            }),
            "gather_columns" => refs("input").into_iter().any(|input| {
                let mut indices = array("indices").unwrap_or_else(|| vec![0]);
                if self.searches_completion()
                    && (!fields.contains_key("indices") || open_array == Some("indices"))
                {
                    let mut widths = widths();
                    widths.insert(indices.len().max(1));
                    return widths
                        .into_iter()
                        .filter(|&columns| columns >= indices.len())
                        .any(|columns| {
                            let mut candidate = indices.clone();
                            candidate.resize(columns, 0);
                            check(
                                Op::GatherColumns {
                                    input,
                                    indices: candidate,
                                },
                                None,
                            )
                        });
                }
                if let Some(columns) = self.output_columns() {
                    if !fields.contains_key("indices") || open_array == Some("indices") {
                        if indices.len() > columns {
                            return false;
                        }
                        indices.resize(columns, 0);
                    }
                } else if indices.is_empty() && open_array == Some("indices") {
                    indices.push(0);
                }
                check(Op::GatherColumns { input, indices }, None)
            }),
            "call" => {
                let selected = fields.get("function").and_then(Value::as_str);
                signatures
                    .iter()
                    .filter(|(name, _)| selected.is_none_or(|selected| selected == name.as_str()))
                    .any(|(function, signature)| {
                        let arguments = array("arguments").unwrap_or_default();
                        let expected = [
                            signature.arguments(1),
                            signature.arguments(self.body.max_context),
                        ];
                        let [Ok(first), Ok(last)] = expected else {
                            return false;
                        };
                        if arguments.len() > first.len()
                            || (fields.contains_key("arguments")
                                && open_array != Some("arguments")
                                && arguments.len() != first.len())
                        {
                            return false;
                        }
                        if self.dependencies.is_some() {
                            return self.possible_call(function, signature, arguments, &mut check);
                        }
                        let mut arguments = arguments;
                        for slot in arguments.len()..first.len() {
                            let Some(id) = (0..count).find(|&id| {
                                self.shapes[0][id] == first[slot]
                                    && self.shapes[1][id] == last[slot]
                            }) else {
                                return false;
                            };
                            arguments.push(id);
                        }
                        check(
                            Op::Call {
                                function: function.clone(),
                                arguments,
                            },
                            None,
                        )
                    })
            }
            "scan" => {
                let selected = fields.get("function").and_then(Value::as_str);
                signatures
                    .keys()
                    .filter(|name| selected.is_none_or(|selected| selected == name.as_str()))
                    .any(|function| {
                        refs("input").into_iter().any(|input| {
                            refs("initial").into_iter().any(|initial| {
                                check(
                                    Op::Scan {
                                        input,
                                        initial,
                                        function: function.clone(),
                                        max_steps: scalar("max_steps", self.body.max_context),
                                    },
                                    None,
                                )
                            })
                        })
                    })
            }
            _ => false,
        }
    }
}
