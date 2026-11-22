//! Resolve numerical calls from the same authored registry used by every caller.

use super::*;
use xlog_core::{Result as NativeResult, XlogError};
use xlog_ir::tensor::{TensorOperation as Op, TensorSignature};

fn invalid(message: &str) -> XlogError {
    XlogError::Compilation(message.into())
}

struct NumericalExpansion<'a> {
    tokens: usize,
    position: usize,
    history_span: Option<(usize, usize)>,
    disabled_definition: Option<&'a str>,
    disabled_calls: usize,
    stack: Vec<String>,
}

impl FunctionRegistry {
    pub(super) fn validate_numerical_program(&self, program: &TensorProgram) -> NativeResult<()> {
        for tokens in [1, program.max_context] {
            program.shapes_resolved(tokens, 0, &|name| {
                let callee = self.numerical(name)?;
                if callee.vocabulary != program.vocabulary
                    || callee.max_context != program.max_context
                {
                    return Err(invalid(
                        "called function has a different token/context contract",
                    ));
                }
                Ok(callee.signature.clone())
            })?;
        }
        Ok(())
    }

    /// Derive an execution graph for this complete prefix. Definitions remain the
    /// sole authored state; call-local references resolve to the entry's physical IDs.
    pub fn compile_numerical(
        &self,
        name: &str,
        tokens: usize,
        position: usize,
        history_span: Option<(usize, usize)>,
    ) -> NativeResult<TensorProgram> {
        self.compile_numerical_with_intervention(name, tokens, position, history_span, None)
            .map(|(program, _)| program)
    }

    /// Compile this prefix with an optional numerical callee replaced by its
    /// typed zero result. Return the number of calls replaced in this expansion.
    pub fn compile_numerical_with_intervention(
        &self,
        name: &str,
        tokens: usize,
        position: usize,
        history_span: Option<(usize, usize)>,
        disabled_definition: Option<&str>,
    ) -> NativeResult<(TensorProgram, usize)> {
        self.validate()
            .map_err(|error| invalid(&error.to_string()))?;
        if history_span.is_some_and(|(start, end)| start > end || end > tokens) {
            return Err(invalid("raw history span is outside the causal input"));
        }
        let entry = self.numerical(name)?;
        if !entry.signature.is_entrypoint() {
            return Err(invalid("execution requires a numerical entrypoint"));
        }
        if let Some(disabled) = disabled_definition {
            if self.numerical(disabled)?.signature.is_entrypoint() {
                return Err(invalid("only a numerical callee can be disabled"));
            }
        }
        let mut program = entry.clone();
        program.operations.clear();
        let mut expansion = NumericalExpansion {
            tokens,
            position,
            history_span,
            disabled_definition,
            disabled_calls: 0,
            stack: Vec::new(),
        };
        let arguments = (0..entry.signature.arguments(tokens)?.len())
            .map(|argument| emit(&mut program, Op::Argument { argument }))
            .collect::<NativeResult<Vec<_>>>()?;
        program.output = self.expand_numerical(name, &arguments, &mut program, &mut expansion)?;
        if matches!(program.signature, TensorSignature::Observations { .. })
            && program
                .operations
                .iter()
                .any(|op| matches!(op, Op::TokenIndicators { .. } | Op::Embedding { .. }))
        {
            return Err(invalid("a numerical observation cannot read token IDs"));
        }
        program.shapes(tokens, position)?;
        Ok((program, expansion.disabled_calls))
    }

    fn expand_numerical(
        &self,
        name: &str,
        arguments: &[usize],
        target: &mut TensorProgram,
        expansion: &mut NumericalExpansion<'_>,
    ) -> NativeResult<usize> {
        let tokens = expansion.tokens;
        if expansion.stack.iter().any(|ancestor| ancestor == name) {
            return Err(invalid("numerical calls must form a DAG"));
        }
        let body = self.numerical(name)?;
        body.shapes_resolved(tokens, expansion.position, &|callee| {
            Ok(self.numerical(callee)?.signature.clone())
        })?;
        if arguments.len() != body.signature.arguments(tokens)?.len() {
            return Err(invalid("numerical call arity differs"));
        }
        if expansion.disabled_definition == Some(name) {
            let shape = body.signature.output(tokens, body.vocabulary)?;
            let zero = emit(
                target,
                Op::Zeros {
                    columns: shape.columns,
                },
            )?;
            expansion.disabled_calls += 1;
            return if shape.rows == 1 {
                Ok(zero)
            } else {
                emit(target, Op::Broadcast { input: zero })
            };
        }
        let parameters = body.parameters.iter().map(|parameter| {
            target.parameters.iter().position(|physical| physical == parameter)
                .ok_or_else(|| invalid("called definition's parameter identity is absent or has a different type"))
        }).collect::<NativeResult<Vec<_>>>()?;
        expansion.stack.push(name.into());
        let mut values = Vec::with_capacity(body.operations.len());
        for operation in &body.operations {
            let id = match operation {
                Op::Argument { argument } => arguments[*argument],
                Op::Call {
                    function,
                    arguments,
                } => {
                    let actual = arguments.iter().map(|&id| values[id]).collect::<Vec<_>>();
                    self.expand_numerical(function, &actual, target, expansion)?
                }
                Op::Scan {
                    input,
                    initial,
                    function,
                    max_steps,
                } => {
                    let mut state = values[*initial];
                    let mut states = Vec::with_capacity(tokens);
                    let (start, end) = expansion.history_span.ok_or_else(|| {
                        invalid("causal scan requires an explicit raw history span")
                    })?;
                    if end - start > *max_steps {
                        return Err(invalid("complete raw history exceeds scan bound"));
                    }
                    for row in 0..tokens {
                        if row < start || row >= end {
                            states.push(state);
                            continue;
                        }
                        let current = emit(
                            target,
                            Op::Row {
                                input: values[*input],
                                row,
                            },
                        )?;
                        state =
                            self.expand_numerical(function, &[state, current], target, expansion)?;
                        states.push(state);
                    }
                    emit(target, Op::Stack { inputs: states })?
                }
                _ => emit(target, operation.remap(&values, &parameters))?,
            };
            values.push(id);
        }
        expansion.stack.pop();
        let output = values[body.output];
        if matches!(body.signature, TensorSignature::Causal { .. }) {
            emit(target, Op::Finite { input: output })
        } else {
            Ok(output)
        }
    }
}

fn emit(program: &mut TensorProgram, operation: Op) -> NativeResult<usize> {
    // Bound compilation storage as well as execution storage. No partial graph
    // is published when an expansion would exceed this admitted extent.
    if program.operations.len() >= 1_048_576 {
        return Err(invalid(
            "expanded numerical program exceeds its operation budget",
        ));
    }
    let id = program.operations.len();
    program.operations.push(operation);
    Ok(id)
}
