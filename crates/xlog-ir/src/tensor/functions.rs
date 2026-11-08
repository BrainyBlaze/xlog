//! Typed numerical calls and the primitives used by acquired definitions.

use super::*;

/// Calling convention of a numerical definition. Pure cells cannot read tokens.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TensorSignature {
    /// Original tokens and prediction position produce a full vocabulary row.
    Decoder,
    /// One hidden row per available position produces one feature row per position.
    Causal {
        /// Coordinates in each hidden argument row.
        input_width: usize,
        /// Coordinates in each result row.
        output_width: usize,
    },
    /// Numeric single-row arguments only, suitable for a causal scan cell.
    Pure {
        /// Coordinates of each single-row numeric argument.
        input_widths: Vec<usize>,
        /// Coordinates in each result row.
        output_width: usize,
    },
}

fn fp32(rows: usize, columns: usize) -> Result<TensorValueShape> {
    if rows == 0
        || columns == 0
        || rows
            .checked_mul(columns)
            .and_then(|n| n.checked_mul(4))
            .is_none()
    {
        return Err(invalid("invalid numerical function extent"));
    }
    Ok(TensorValueShape {
        rows,
        columns,
        float32: true,
    })
}

impl TensorSignature {
    /// Actual numeric argument extents, excluding the explicit causal token boundary.
    pub fn arguments(&self, tokens: usize) -> Result<Vec<TensorValueShape>> {
        match self {
            Self::Decoder => Ok(vec![]),
            Self::Causal { input_width, .. } => Ok(vec![fp32(tokens, *input_width)?]),
            Self::Pure { input_widths, .. } => {
                input_widths.iter().map(|&width| fp32(1, width)).collect()
            }
        }
    }

    /// Actual return extent at this causal input length.
    pub fn output(&self, tokens: usize, vocabulary: usize) -> Result<TensorValueShape> {
        match self {
            Self::Decoder => fp32(1, vocabulary),
            Self::Causal { output_width, .. } => fp32(tokens, *output_width),
            Self::Pure { output_width, .. } => fp32(1, *output_width),
        }
    }
}

impl TensorOperation {
    pub(super) fn function_inputs(&self) -> Option<Vec<usize>> {
        use TensorOperation::*;
        Some(match self {
            Zeros { .. } | Argument { .. } | Parameter { .. } | TokenIndicators { .. } => vec![],
            Broadcast { input }
            | Tanh { input }
            | GatherColumns { input, .. }
            | Bfloat16 { input }
            | Finite { input }
            | Row { input, .. } => vec![*input],
            Multiply { left, right } | Subtract { left, right } | Divide { left, right } => {
                vec![*left, *right]
            }
            Select {
                condition,
                positive,
                negative,
            } => vec![*condition, *positive, *negative],
            Call { arguments, .. } => arguments.clone(),
            Scan { input, initial, .. } => vec![*input, *initial],
            Stack { inputs } => inputs.clone(),
            _ => return None,
        })
    }

    /// Called definition, for the canonical dependency graph.
    pub fn callee(&self) -> Option<&str> {
        match self {
            Self::Call { function, .. } | Self::Scan { function, .. } => Some(function),
            _ => None,
        }
    }

    /// Physical parameter read, independent of the consuming operation.
    pub fn parameter(&self) -> Option<usize> {
        match *self {
            Self::Parameter { weight }
            | Self::Embedding { weight }
            | Self::Linear { weight, .. }
            | Self::RmsNorm { weight, .. } => Some(weight),
            _ => None,
        }
    }

    /// Remap call-local references while preserving shared parameter identities.
    pub fn remap(&self, values: &[usize], parameters: &[usize]) -> Self {
        use TensorOperation::*;
        let mut result = self.clone();
        match &mut result {
            Zeros { .. } | Argument { .. } | TokenIndicators { .. } => {}
            Parameter { weight } | Embedding { weight } => *weight = parameters[*weight],
            Linear { input, weight } | RmsNorm { input, weight, .. } => {
                *input = values[*input];
                *weight = parameters[*weight];
            }
            Rotary { input, .. }
            | PredictionRow { input }
            | Float32 { input }
            | Broadcast { input }
            | Tanh { input }
            | GatherColumns { input, .. }
            | Bfloat16 { input }
            | Finite { input }
            | Row { input, .. } => *input = values[*input],
            Attention {
                query, key, value, ..
            } => {
                *query = values[*query];
                *key = values[*key];
                *value = values[*value];
            }
            Add { left, right }
            | Multiply { left, right }
            | Subtract { left, right }
            | Divide { left, right } => {
                *left = values[*left];
                *right = values[*right];
            }
            SiluGate { gate, value } => {
                *gate = values[*gate];
                *value = values[*value];
            }
            Select {
                condition,
                positive,
                negative,
            } => {
                *condition = values[*condition];
                *positive = values[*positive];
                *negative = values[*negative];
            }
            Call { arguments, .. } | Stack { inputs: arguments } => {
                for id in arguments {
                    *id = values[*id];
                }
            }
            Scan { input, initial, .. } => {
                *input = values[*input];
                *initial = values[*initial];
            }
        }
        result
    }
}

impl TensorProgram {
    /// Numeric-argument reads of every value, including unused call operands.
    /// This causal boundary is deliberately distinct from output dependencies.
    pub fn value_argument_dependencies(&self) -> Vec<bool> {
        let mut arguments = Vec::with_capacity(self.operations.len());
        for operation in &self.operations {
            arguments.push(
                matches!(operation, TensorOperation::Argument { .. })
                    || operation
                        .inputs()
                        .iter()
                        .any(|&id| arguments.get(id).copied().unwrap_or(true)),
            );
        }
        arguments
    }

    /// Enforce the executable acquisition grammar and its causal information boundary.
    pub fn validate_acquired_body(&self) -> Result<()> {
        use TensorOperation::*;
        if matches!(self.signature, TensorSignature::Decoder) {
            return Ok(());
        }
        let arguments = self.value_argument_dependencies();
        for (index, operation) in self.operations.iter().enumerate() {
            if !matches!(
                operation,
                Zeros { .. }
                    | Argument { .. }
                    | Parameter { .. }
                    | TokenIndicators { .. }
                    | Broadcast { .. }
                    | Tanh { .. }
                    | Multiply { .. }
                    | Subtract { .. }
                    | Divide { .. }
                    | GatherColumns { .. }
                    | Select { .. }
                    | Call { .. }
                    | Scan { .. }
                    | Linear { .. }
                    | Add { .. }
            ) {
                return Err(invalid("operation is outside the acquired FP32 grammar"));
            }
            if matches!(operation, Scan { .. }) && arguments[index] {
                return Err(invalid(
                    "causal scans read raw-prefix features, not other positions' hidden states",
                ));
            }
        }
        if self.parameters.iter().any(|parameter| !parameter.float32) {
            return Err(invalid(
                "acquired numerical functions require FP32 parameters",
            ));
        }
        Ok(())
    }

    pub(super) fn function_shape(
        &self,
        operation: &TensorOperation,
        input: &dyn Fn(usize) -> Result<TensorValueShape>,
        tokens: usize,
        resolve: &dyn Fn(&str) -> Result<TensorSignature>,
    ) -> Result<Option<TensorValueShape>> {
        use TensorOperation::*;
        let float = |id| {
            let shape = input(id)?;
            if !shape.float32 {
                return Err(invalid("FP32 operation received BF16"));
            }
            Ok(shape)
        };
        if matches!(self.signature, TensorSignature::Pure { .. })
            && matches!(
                operation,
                TokenIndicators { .. }
                    | Broadcast { .. }
                    | Embedding { .. }
                    | PredictionRow { .. }
                    | Attention { .. }
                    | Rotary { .. }
                    | Scan { .. }
            )
        {
            return Err(invalid(
                "pure numerical cells cannot inspect a causal token sequence",
            ));
        }
        let shape = match operation {
            Zeros { columns } => fp32(1, *columns)?,
            Argument { argument } => *self
                .signature
                .arguments(tokens)?
                .get(*argument)
                .ok_or_else(|| invalid("numerical argument is absent"))?,
            Parameter { weight } => {
                let parameter = self
                    .parameters
                    .get(*weight)
                    .ok_or_else(|| invalid("unknown numerical parameter"))?;
                if !parameter.float32 {
                    return Err(invalid("acquired parameter reads require FP32"));
                }
                match parameter.shape.as_slice() {
                    &[columns] => fp32(1, columns)?,
                    &[rows, columns] => fp32(rows, columns)?,
                    _ => return Err(invalid("numerical parameter must be a vector or matrix")),
                }
            }
            TokenIndicators { token_ids } => {
                if token_ids.iter().any(|&id| id as usize >= self.vocabulary) {
                    return Err(invalid(
                        "indicator token is outside the original vocabulary",
                    ));
                }
                fp32(tokens, token_ids.len())?
            }
            Broadcast { input: id } => {
                let source = float(*id)?;
                if source.rows != 1 {
                    return Err(invalid("broadcast requires one row"));
                }
                fp32(tokens, source.columns)?
            }
            Tanh { input: id } | Finite { input: id } => float(*id)?,
            Bfloat16 { input: id } => TensorValueShape {
                float32: false,
                ..float(*id)?
            },
            Add { left, right }
            | Multiply { left, right }
            | Subtract { left, right }
            | Divide { left, right } => {
                let lhs = input(*left)?;
                if lhs != input(*right)? || (!matches!(operation, Add { .. }) && !lhs.float32) {
                    return Err(invalid(
                        "elementwise numerical input types or extents differ",
                    ));
                }
                lhs
            }
            GatherColumns { input: id, indices } => {
                let source = float(*id)?;
                if indices.iter().any(|&index| index >= source.columns) {
                    return Err(invalid("gather index out of bounds"));
                }
                fp32(source.rows, indices.len())?
            }
            Select {
                condition,
                positive,
                negative,
            } => {
                let shape = float(*positive)?;
                if shape != float(*negative)? || shape != float(*condition)? {
                    return Err(invalid("mask inputs require identical FP32 extents"));
                }
                shape
            }
            Call {
                function,
                arguments,
            } => {
                let signature = resolve(function)?;
                if matches!(signature, TensorSignature::Decoder)
                    || (matches!(self.signature, TensorSignature::Pure { .. })
                        && !matches!(signature, TensorSignature::Pure { .. }))
                    || arguments
                        .iter()
                        .map(|&id| input(id))
                        .collect::<Result<Vec<_>>>()?
                        != signature.arguments(tokens)?
                {
                    return Err(invalid(
                        "numerical call arguments or causal boundary differ",
                    ));
                }
                signature.output(tokens, self.vocabulary)?
            }
            Scan {
                input: id,
                initial,
                function,
                max_steps,
            } => {
                let source = float(*id)?;
                let state = float(*initial)?;
                if source.rows != tokens
                    || state.rows != 1
                    || *max_steps == 0
                    || *max_steps > self.max_context
                    || resolve(function)?
                        != (TensorSignature::Pure {
                            input_widths: vec![state.columns, source.columns],
                            output_width: state.columns,
                        })
                {
                    return Err(invalid(
                        "scan cell signature, initial state or complete-prefix bound differs",
                    ));
                }
                fp32(tokens, state.columns)?
            }
            Row { input: id, row } => {
                let source = float(*id)?;
                if *row >= source.rows {
                    return Err(invalid("scan row out of bounds"));
                }
                fp32(1, source.columns)?
            }
            Stack { inputs } => {
                let first = float(*inputs.first().ok_or_else(|| invalid("empty scan states"))?)?;
                if first.rows != 1 || inputs.iter().any(|&id| float(id).ok() != Some(first)) {
                    return Err(invalid("scan states have inconsistent extents"));
                }
                fp32(inputs.len(), first.columns)?
            }
            _ => return Ok(None),
        };
        Ok(Some(shape))
    }
}
