//! Possible value dependencies of a numerical result, without executing it.

use std::collections::BTreeSet;

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
}
