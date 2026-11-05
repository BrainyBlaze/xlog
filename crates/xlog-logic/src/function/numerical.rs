//! Resolve numerical calls from the same authored registry used by every caller.

use super::*;
use xlog_core::{Result as NativeResult, XlogError};
use xlog_ir::tensor::{TensorOperation as Op, TensorSignature};

fn invalid(message: &str) -> XlogError {
    XlogError::Compilation(message.into())
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
        self.validate()
            .map_err(|error| invalid(&error.to_string()))?;
        if history_span.is_some_and(|(start, end)| start > end || end > tokens) {
            return Err(invalid("raw history span is outside the causal input"));
        }
        let entry = self.numerical(name)?;
        if entry.signature != TensorSignature::Decoder {
            return Err(invalid("execution entry must be a decoder"));
        }
        let mut program = entry.clone();
        program.operations.clear();
        let mut stack = Vec::new();
        program.output = self.expand_numerical(
            name,
            &[],
            tokens,
            position,
            history_span,
            &mut program,
            &mut stack,
        )?;
        program.shapes(tokens, position)?;
        Ok(program)
    }

    fn expand_numerical(
        &self,
        name: &str,
        arguments: &[usize],
        tokens: usize,
        position: usize,
        history_span: Option<(usize, usize)>,
        target: &mut TensorProgram,
        stack: &mut Vec<String>,
    ) -> NativeResult<usize> {
        if stack.iter().any(|ancestor| ancestor == name) {
            return Err(invalid("numerical calls must form a DAG"));
        }
        let body = self.numerical(name)?;
        body.shapes_resolved(tokens, position, &|callee| {
            Ok(self.numerical(callee)?.signature.clone())
        })?;
        if arguments.len() != body.signature.arguments(tokens)?.len() {
            return Err(invalid("numerical call arity differs"));
        }
        let parameters = body.parameters.iter().map(|parameter| {
            target.parameters.iter().position(|physical| physical == parameter)
                .ok_or_else(|| invalid("called definition's parameter identity is absent or has a different type"))
        }).collect::<NativeResult<Vec<_>>>()?;
        stack.push(name.into());
        let mut values = Vec::with_capacity(body.operations.len());
        for operation in &body.operations {
            let id = match operation {
                Op::Argument { argument } => arguments[*argument],
                Op::Call {
                    function,
                    arguments,
                } => {
                    let actual = arguments.iter().map(|&id| values[id]).collect::<Vec<_>>();
                    self.expand_numerical(
                        function,
                        &actual,
                        tokens,
                        position,
                        history_span,
                        target,
                        stack,
                    )?
                }
                Op::Scan {
                    input,
                    initial,
                    function,
                    max_steps,
                } => {
                    let mut state = values[*initial];
                    let mut states = Vec::with_capacity(tokens);
                    let (start, end) = history_span.ok_or_else(|| {
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
                        state = self.expand_numerical(
                            function,
                            &[state, current],
                            tokens,
                            position,
                            history_span,
                            target,
                            stack,
                        )?;
                        states.push(state);
                    }
                    emit(target, Op::Stack { inputs: states })?
                }
                _ => emit(target, operation.remap(&values, &parameters))?,
            };
            values.push(id);
        }
        stack.pop();
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
