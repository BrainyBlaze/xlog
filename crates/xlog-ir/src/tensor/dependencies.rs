//! Possible value dependencies of a numerical result, without executing it.

use std::collections::{BTreeMap, BTreeSet};

use super::*;

/// Output-reaching identities and inputs in the canonical numerical ABI.
/// Arguments address numeric arguments; causal tokens are a separate input.
/// The original contract records structural dependencies. Explicit value-aware
/// analysis additionally carries sound identities, never general nonconstancy.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorOutputDependencies {
    /// Used numeric formals in `TensorSignature::arguments` order.
    pub arguments: BTreeSet<usize>,
    /// Whether raw causal token values may affect the result.
    pub tokens: bool,
    /// Registry identities reached by the returned value.
    pub definitions: BTreeSet<String>,
    /// Physical parameter identities reached by the returned value.
    pub parameters: BTreeSet<String>,
    /// Absent in the original structural contract. Local operation identities
    /// are scoped to one body and become unknown in exported callee summaries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_identity: Option<TensorValueIdentity>,
}

/// Proven returned-value identities, without evaluating parameters or rewriting IR.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TensorValueIdentity {
    /// No identity or constant has been established.
    Unknown,
    /// Every element is the positive FP32 zero produced by `Zeros` or copying it.
    Zero,
    /// The unchanged numeric formal in the current signature.
    Argument(usize),
    /// An actual preceding value in this body, never a callee-local identifier.
    Operation(usize),
}

/// Query-local predicate proof alongside the unchanged exported dependency record.
/// This is not a value identity: distinct signed zeros and NaNs remain distinct.
/// The proof is deliberately absent from callee JSON and saved numerical IR.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct TensorValueFacts {
    /// The existing output-reaching identities and inputs, unchanged on arithmetic.
    pub dependencies: TensorOutputDependencies,
    /// Every element belongs to {-0, +0, NaN}, so the CUDA predicate >0 is false.
    pub zero_or_nan: bool,
}

impl TensorValueFacts {
    /// Canonical proof transfer for complete values and partial source slots.
    /// An unknown slot supplies false (no proof), never an invented value.
    /// Scalar widths and Gather indices cannot introduce a numeric dependency.
    pub fn prove_zero_or_nan(
        operation: &str,
        operand: &dyn Fn(&str) -> bool,
        inputs: Option<&[bool]>,
    ) -> bool {
        match operation {
            "zeros" => true,
            "multiply" | "matmul" => operand("left") || operand("right"),
            "add" => operand("left") && operand("right"),
            "tanh" | "broadcast" | "gather_columns" | "row" => operand("input"),
            "stack" => inputs.is_some_and(|inputs| !inputs.is_empty() && inputs.iter().all(|x| *x)),
            "select" => {
                if operand("condition") {
                    operand("negative")
                } else {
                    operand("positive") && operand("negative")
                }
            }
            // The caller owns the existing Zero/returned-formal callee summary.
            "call" => operand("returned"),
            _ => false,
        }
    }
}

impl TensorOutputDependencies {
    /// Join possible dependencies, preserving physical parameter aliases.
    pub fn merge(&mut self, other: &Self) {
        self.arguments.extend(other.arguments.iter().copied());
        self.tokens |= other.tokens;
        self.definitions.extend(other.definitions.iter().cloned());
        self.parameters.extend(other.parameters.iter().cloned());
        if self.value_identity != other.value_identity {
            self.value_identity = None;
        }
    }

    /// Preserve proofs meaningful to a caller, hiding body-local SSA identities.
    pub fn into_callee_summary(mut self) -> Self {
        if matches!(self.value_identity, Some(TensorValueIdentity::Operation(_))) {
            self.value_identity = Some(TensorValueIdentity::Unknown);
        }
        self
    }

    /// Whether the result may depend on a numeric argument or raw tokens.
    pub fn has_input(&self) -> bool {
        self.tokens || !self.arguments.is_empty()
    }
}

impl TensorProgram {
    /// Propagate one operation through exact callee-output facts. Both complete
    /// admission and prospective prefix choices use this same analysis.
    pub fn operation_output_dependencies(
        &self,
        operation: &TensorOperation,
        value: &dyn Fn(usize) -> Result<TensorOutputDependencies>,
        resolve: &mut dyn FnMut(&str) -> Result<TensorOutputDependencies>,
    ) -> Result<TensorOutputDependencies> {
        self.operation_dependencies(operation, value, resolve, None)
    }

    /// The same dependency owner with explicit local identities and known zeros.
    /// `output` is the canonical index of this operation, not a guessed future ID.
    pub fn operation_value_dependencies(
        &self,
        operation: &TensorOperation,
        output: usize,
        value: &dyn Fn(usize) -> Result<TensorOutputDependencies>,
        resolve: &mut dyn FnMut(&str) -> Result<TensorOutputDependencies>,
    ) -> Result<TensorOutputDependencies> {
        self.operation_dependencies(operation, value, resolve, Some(output))
    }

    /// Explicit predicate-aware mode. Historical structural/value entrypoints
    /// keep their original transfer; arithmetic dependencies are never erased.
    pub fn operation_predicate_dependencies(
        &self,
        operation: &TensorOperation,
        output: usize,
        value: &dyn Fn(usize) -> Result<TensorValueFacts>,
        resolve: &mut dyn FnMut(&str) -> Result<TensorOutputDependencies>,
    ) -> Result<TensorValueFacts> {
        let mut callees = BTreeMap::<String, TensorOutputDependencies>::new();
        let mut callee = |name: &str| {
            if let Some(facts) = callees.get(name) {
                return Ok(facts.clone());
            }
            let facts = resolve(name)?;
            callees.insert(name.to_owned(), facts.clone());
            Ok(facts)
        };
        let mut dependencies = self.operation_value_dependencies(
            operation,
            output,
            &|id| Ok(value(id)?.dependencies),
            &mut callee,
        )?;
        let mut zero_or_nan = dependencies.value_identity == Some(TensorValueIdentity::Zero);
        zero_or_nan |= match operation {
            TensorOperation::Multiply { left, right }
            | TensorOperation::MatMul { left, right }
            | TensorOperation::Add { left, right } => {
                let left = value(*left)?.zero_or_nan;
                let right = value(*right)?.zero_or_nan;
                let tag = match operation {
                    TensorOperation::Multiply { .. } => "multiply",
                    TensorOperation::MatMul { .. } => "matmul",
                    _ => "add",
                };
                TensorValueFacts::prove_zero_or_nan(
                    tag,
                    &|name| if name == "left" { left } else { right },
                    None,
                )
            }
            TensorOperation::Tanh { input }
            | TensorOperation::Broadcast { input }
            | TensorOperation::GatherColumns { input, .. }
            | TensorOperation::Row { input, .. } => {
                let tag = match operation {
                    TensorOperation::Tanh { .. } => "tanh",
                    TensorOperation::Broadcast { .. } => "broadcast",
                    TensorOperation::GatherColumns { .. } => "gather_columns",
                    _ => "row",
                };
                let input = value(*input)?.zero_or_nan;
                TensorValueFacts::prove_zero_or_nan(tag, &|_| input, None)
            }
            TensorOperation::Stack { inputs } => {
                let inputs = inputs
                    .iter()
                    .map(|id| Ok(value(*id)?.zero_or_nan))
                    .collect::<Result<Vec<_>>>()?;
                TensorValueFacts::prove_zero_or_nan("stack", &|_| false, Some(&inputs))
            }
            TensorOperation::Select {
                condition,
                positive,
                negative,
            } => {
                let negative = value(*negative)?;
                let condition = value(*condition)?.zero_or_nan;
                let positive = value(*positive)?.zero_or_nan;
                if condition {
                    // Preserve the executable strict comparison, including NaN.
                    // The arithmetic that produced the condition is not rewritten.
                    dependencies = negative.dependencies;
                }
                TensorValueFacts::prove_zero_or_nan(
                    "select",
                    &|name| match name {
                        "condition" => condition,
                        "positive" => positive,
                        _ => negative.zero_or_nan,
                    },
                    None,
                )
            }
            TensorOperation::Call {
                function,
                arguments,
            } => {
                let returned = match callee(function)?.value_identity {
                    Some(TensorValueIdentity::Zero) => true,
                    Some(TensorValueIdentity::Argument(argument)) => {
                        let input = arguments
                            .get(argument)
                            .ok_or_else(|| invalid("callee predicate argument is absent"))?;
                        value(*input)?.zero_or_nan
                    }
                    // Opaque callees do not export this private predicate proof.
                    _ => false,
                };
                TensorValueFacts::prove_zero_or_nan("call", &|_| returned, None)
            }
            // In particular, do not infer facts from parameter contents, a lack
            // of dependencies, division, or a scan cell without its initial span.
            _ => false,
        };
        Ok(TensorValueFacts {
            dependencies,
            zero_or_nan,
        })
    }

    fn operation_dependencies(
        &self,
        operation: &TensorOperation,
        value: &dyn Fn(usize) -> Result<TensorOutputDependencies>,
        resolve: &mut dyn FnMut(&str) -> Result<TensorOutputDependencies>,
        output: Option<usize>,
    ) -> Result<TensorOutputDependencies> {
        let mut result = TensorOutputDependencies::default();
        let mut identity = output.map(TensorValueIdentity::Operation);
        let same_value = |left: &TensorOutputDependencies, right: &TensorOutputDependencies| {
            left.value_identity.as_ref().is_some_and(|identity| {
                *identity != TensorValueIdentity::Unknown
                    && right.value_identity.as_ref() == Some(identity)
            })
        };
        match operation {
            TensorOperation::Argument { argument } => {
                if *argument >= self.signature.arguments(1)?.len() {
                    return Err(invalid("unknown numerical dependency argument"));
                }
                result.arguments.insert(*argument);
                if output.is_some() {
                    identity = Some(TensorValueIdentity::Argument(*argument));
                }
            }
            TensorOperation::Zeros { .. } if output.is_some() => {
                identity = Some(TensorValueIdentity::Zero);
            }
            TensorOperation::Select {
                condition,
                positive,
                negative,
            } if output.is_some() => {
                let condition = value(*condition)?;
                let positive_value = value(*positive)?;
                let negative_value = value(*negative)?;
                if condition.value_identity == Some(TensorValueIdentity::Zero) {
                    // The executable predicate is condition > 0, including +0.
                    result = negative_value;
                    identity = result.value_identity.clone();
                } else if positive == negative || same_value(&positive_value, &negative_value) {
                    identity = positive_value.value_identity.clone();
                    result.merge(&positive_value);
                    result.merge(&negative_value);
                } else {
                    result.merge(&condition);
                    result.merge(&positive_value);
                    result.merge(&negative_value);
                }
            }
            TensorOperation::Finite { input } if output.is_some() => {
                result = value(*input)?;
                // On successful execution the finiteness check and scale by
                // one preserve this finite FP32 value, including signed zero.
                identity = result.value_identity.clone();
            }
            TensorOperation::Broadcast { input }
            | TensorOperation::GatherColumns { input, .. }
            | TensorOperation::Row { input, .. }
                if output.is_some() =>
            {
                result = value(*input)?;
                if result.value_identity == Some(TensorValueIdentity::Zero) {
                    identity = Some(TensorValueIdentity::Zero);
                }
            }
            TensorOperation::Stack { inputs } if output.is_some() => {
                let mut zero = true;
                for input in inputs {
                    let input = value(*input)?;
                    zero &= input.value_identity == Some(TensorValueIdentity::Zero);
                    result.merge(&input);
                }
                if zero && !inputs.is_empty() {
                    identity = Some(TensorValueIdentity::Zero);
                }
            }
            TensorOperation::TokenIndicators { .. } | TensorOperation::Embedding { .. } => {
                result.tokens = true;
            }
            TensorOperation::Call {
                function,
                arguments,
            } => {
                let callee = resolve(function)?;
                if output.is_some() {
                    match callee.value_identity {
                        Some(TensorValueIdentity::Zero) => {
                            identity = Some(TensorValueIdentity::Zero);
                        }
                        Some(TensorValueIdentity::Argument(argument)) => {
                            let input = arguments.get(argument).ok_or_else(|| {
                                invalid("callee value identity argument is absent")
                            })?;
                            identity = value(*input)?.value_identity;
                        }
                        _ => {}
                    }
                }
                result.tokens = callee.tokens;
                result.definitions = callee.definitions;
                result.definitions.insert(function.clone());
                result.parameters = callee.parameters;
                for argument in callee.arguments {
                    let input = arguments
                        .get(argument)
                        .ok_or_else(|| invalid("callee dependency argument is absent"))?;
                    result.merge(&value(*input)?);
                }
            }
            TensorOperation::Scan {
                function,
                input,
                initial,
                ..
            } => {
                let cell = resolve(function)?;
                if cell.tokens || cell.arguments.iter().any(|&argument| argument > 1) {
                    return Err(invalid("scan dependency requires a pure state/item cell"));
                }
                result.definitions = cell.definitions;
                result.definitions.insert(function.clone());
                result.parameters = cell.parameters;
                // Before the raw span, and for an empty span, the observable
                // result is the initial state even if the cell ignores state.
                result.merge(&value(*initial)?);
                if cell.arguments.contains(&1) {
                    result.merge(&value(*input)?);
                }
            }
            _ => {
                for input in operation.inputs() {
                    result.merge(&value(input)?);
                }
            }
        }
        if let Some(weight) = operation.parameter() {
            let parameter = self
                .parameters
                .get(weight)
                .ok_or_else(|| invalid("unknown numerical dependency parameter"))?;
            result.parameters.insert(parameter.name.clone());
        }
        if output.is_some() {
            result.value_identity = Some(identity.unwrap_or(TensorValueIdentity::Unknown));
        }
        Ok(result)
    }

    /// Dependencies of every preceding value, in canonical operation order.
    pub fn value_output_dependencies(
        &self,
        resolve: &mut dyn FnMut(&str) -> Result<TensorOutputDependencies>,
    ) -> Result<Vec<TensorOutputDependencies>> {
        self.value_dependencies(resolve, false)
    }

    fn value_dependencies(
        &self,
        resolve: &mut dyn FnMut(&str) -> Result<TensorOutputDependencies>,
        identities: bool,
    ) -> Result<Vec<TensorOutputDependencies>> {
        let mut values = Vec::<TensorOutputDependencies>::with_capacity(self.operations.len());
        for operation in &self.operations {
            let dependencies = self.operation_dependencies(
                operation,
                &|id| {
                    values
                        .get(id)
                        .cloned()
                        .ok_or_else(|| invalid("dependency value is not preceding"))
                },
                resolve,
                identities.then_some(values.len()),
            )?;
            values.push(dependencies);
        }
        Ok(values)
    }

    /// Only dependencies reaching the declared observable result survive.
    pub fn output_dependencies(
        &self,
        resolve: &mut dyn FnMut(&str) -> Result<TensorOutputDependencies>,
    ) -> Result<TensorOutputDependencies> {
        self.value_output_dependencies(resolve)?
            .get(self.output)
            .cloned()
            .ok_or_else(|| invalid("numerical dependency output is absent"))
    }

    /// Export returned-value facts without leaking a body-local SSA identity.
    /// Unknown arithmetic, parameters and nonzero values stay unknown.
    pub fn output_value_dependencies(
        &self,
        resolve: &mut dyn FnMut(&str) -> Result<TensorOutputDependencies>,
    ) -> Result<TensorOutputDependencies> {
        self.value_dependencies(resolve, true)?
            .get(self.output)
            .cloned()
            .map(TensorOutputDependencies::into_callee_summary)
            .ok_or_else(|| invalid("numerical dependency output is absent"))
    }

    /// The same operation transfer used by prospective predicate-aware choices.
    /// Only the existing five-field dependency record crosses a callee boundary.
    pub fn output_predicate_dependencies(
        &self,
        resolve: &mut dyn FnMut(&str) -> Result<TensorOutputDependencies>,
    ) -> Result<TensorOutputDependencies> {
        Ok(self
            .output_predicate_facts(resolve)?
            .dependencies
            .into_callee_summary())
    }

    /// Retain the local returned-value proof for a new entry's admission gate.
    /// Exported callee records still use `output_predicate_dependencies`.
    pub fn output_predicate_facts(
        &self,
        resolve: &mut dyn FnMut(&str) -> Result<TensorOutputDependencies>,
    ) -> Result<TensorValueFacts> {
        let mut values = Vec::<TensorValueFacts>::with_capacity(self.operations.len());
        for operation in &self.operations {
            let facts = self.operation_predicate_dependencies(
                operation,
                values.len(),
                &|id| {
                    values
                        .get(id)
                        .cloned()
                        .ok_or_else(|| invalid("predicate value is not preceding"))
                },
                resolve,
            )?;
            values.push(facts);
        }
        values
            .get(self.output)
            .cloned()
            .ok_or_else(|| invalid("numerical dependency output is absent"))
    }
}
