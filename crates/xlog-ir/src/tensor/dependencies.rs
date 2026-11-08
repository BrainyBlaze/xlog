//! Possible value dependencies of a numerical result, without executing it.

use std::collections::BTreeSet;

use super::*;

/// Output-reaching identities and inputs in the canonical numerical ABI.
/// Arguments address numeric arguments; causal tokens are a separate input.
/// These are structural dependencies, not a proof of semantic nonconstancy.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
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
}

impl TensorOutputDependencies {
    /// Join possible dependencies, preserving physical parameter aliases.
    pub fn merge(&mut self, other: &Self) {
        self.arguments.extend(other.arguments.iter().copied());
        self.tokens |= other.tokens;
        self.definitions.extend(other.definitions.iter().cloned());
        self.parameters.extend(other.parameters.iter().cloned());
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
        let mut result = TensorOutputDependencies::default();
        match operation {
            TensorOperation::Argument { argument } => {
                if *argument >= self.signature.arguments(1)?.len() {
                    return Err(invalid("unknown numerical dependency argument"));
                }
                result.arguments.insert(*argument);
            }
            TensorOperation::TokenIndicators { .. } | TensorOperation::Embedding { .. } => {
                result.tokens = true;
            }
            TensorOperation::Call {
                function,
                arguments,
            } => {
                let callee = resolve(function)?;
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
        Ok(result)
    }

    /// Dependencies of every preceding value, in canonical operation order.
    pub fn value_output_dependencies(
        &self,
        resolve: &mut dyn FnMut(&str) -> Result<TensorOutputDependencies>,
    ) -> Result<Vec<TensorOutputDependencies>> {
        let mut values = Vec::<TensorOutputDependencies>::with_capacity(self.operations.len());
        for operation in &self.operations {
            let dependencies = self.operation_output_dependencies(
                operation,
                &|id| {
                    values
                        .get(id)
                        .cloned()
                        .ok_or_else(|| invalid("dependency value is not preceding"))
                },
                resolve,
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
}
