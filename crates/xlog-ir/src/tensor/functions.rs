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

fn related_shape<const N: usize>(
    tag: &str,
    inputs: [TensorValueShape; N],
    tokens: usize,
    gather_columns: Option<usize>,
) -> Result<TensorValueShape> {
    let mut inputs = inputs.map(TensorShapeConstraint::from);
    let mut output = TensorShapeConstraint::default();
    TensorOperation::constrain_shapes(tag, &mut inputs, &mut output, tokens, gather_columns)?;
    output
        .concrete()
        .ok_or_else(|| invalid("numerical output shape is unresolved"))
}

impl TensorSignature {
    /// State and item widths of an available pure, state-preserving scan cell.
    /// This is the same ABI domain used by concrete Scan inference; it supplies
    /// no activation, initial state or source operation.
    pub fn scan_widths(&self) -> Option<[usize; 2]> {
        match self {
            Self::Pure {
                input_widths,
                output_width,
            } if input_widths.len() == 2 && input_widths[0] == *output_width => {
                Some([input_widths[0], input_widths[1]])
            }
            _ => None,
        }
    }

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

    /// Check a numerical call's ABI, leaving only unauthored arguments unknown.
    /// Concrete inference supplies every argument; source-prefix admission may
    /// omit shapes without inventing values or relaxing the causal boundary.
    pub fn call_output(
        &self,
        caller: &Self,
        tokens: usize,
        vocabulary: usize,
        arguments: impl IntoIterator<Item = Result<Option<TensorValueShape>>>,
    ) -> Result<TensorValueShape> {
        if matches!(self, Self::Decoder)
            || (matches!(caller, Self::Pure { .. }) && !matches!(self, Self::Pure { .. }))
        {
            return Err(invalid(
                "numerical call arguments or causal boundary differ",
            ));
        }
        let arguments = arguments.into_iter().collect::<Result<Vec<_>>>()?;
        let expected = self.arguments(tokens)?;
        if arguments.len() != expected.len()
            || arguments
                .iter()
                .zip(expected)
                .any(|(actual, expected)| actual.is_some_and(|actual| actual != expected))
        {
            return Err(invalid(
                "numerical call arguments or causal boundary differ",
            ));
        }
        self.output(tokens, vocabulary)
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
            Multiply { left, right }
            | MatMul { left, right }
            | Subtract { left, right }
            | Divide { left, right } => {
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
            | MatMul { left, right }
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
                    | MatMul { .. }
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
                let shape = related_shape("broadcast", [source], tokens, None)
                    .map_err(|_| invalid("broadcast requires one row"))?;
                fp32(shape.rows, shape.columns)?
            }
            Tanh { input: id } | Finite { input: id } => {
                related_shape("tanh", [float(*id)?], tokens, None)?
            }
            Bfloat16 { input: id } => TensorValueShape {
                float32: false,
                ..float(*id)?
            },
            MatMul { left, right } => {
                let shape = related_shape("matmul", [float(*left)?, float(*right)?], tokens, None)?;
                fp32(shape.rows, shape.columns)?
            }
            Add { left, right }
            | Multiply { left, right }
            | Subtract { left, right }
            | Divide { left, right } => related_shape(
                if matches!(operation, Add { .. }) {
                    "add"
                } else {
                    "multiply"
                },
                [input(*left)?, input(*right)?],
                tokens,
                None,
            )
            .map_err(|_| invalid("elementwise numerical input types or extents differ"))?,
            GatherColumns { input: id, indices } => {
                let source = float(*id)?;
                if indices.iter().any(|&index| index >= source.columns) {
                    return Err(invalid("gather index out of bounds"));
                }
                let shape = related_shape("gather_columns", [source], tokens, Some(indices.len()))?;
                fp32(shape.rows, shape.columns)?
            }
            Select {
                condition,
                positive,
                negative,
            } => {
                let positive = float(*positive)?;
                let negative = float(*negative)?;
                let mut inputs = [
                    TensorShapeConstraint::default(),
                    positive.into(),
                    negative.into(),
                ];
                let mut output = TensorShapeConstraint::default();
                // Preserve concrete operand evaluation order and its early
                // refusal, using the same partial relation for both steps.
                TensorOperation::constrain_shapes("select", &mut inputs, &mut output, tokens, None)
                    .map_err(|_| invalid("mask inputs require identical FP32 extents"))?;
                inputs[0] = float(*condition)?.into();
                TensorOperation::constrain_shapes("select", &mut inputs, &mut output, tokens, None)
                    .map_err(|_| invalid("mask inputs require identical FP32 extents"))?;
                output
                    .concrete()
                    .ok_or_else(|| invalid("numerical output shape is unresolved"))?
            }
            Call {
                function,
                arguments,
            } => {
                let signature = resolve(function)?;
                signature.call_output(
                    &self.signature,
                    tokens,
                    self.vocabulary,
                    arguments.iter().map(|&id| input(id).map(Some)),
                )?
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
                    || resolve(function)?.scan_widths() != Some([state.columns, source.columns])
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
