//! Typed, acyclic numerical programs with shared parameter identities.

use serde::{Deserialize, Serialize};
use xlog_core::{Result, XlogError};

// Internally tagged enums buffer their fields through Serde's Content. When the
// application's serde_json enables arbitrary_precision, a fractional JSON number
// arrives through Number's numeric carrier rather than deserialize_f32. Decode
// that carrier at the owning field without changing the serialized decimal or
// rounding it through f64 before restoring the actual FP32 coefficient.
fn deserialize_f32<'de, D>(deserializer: D) -> std::result::Result<f32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let number = serde_json::Number::deserialize(deserializer)?;
    let value = number
        .to_string()
        .parse::<f32>()
        .map_err(serde::de::Error::custom)?;
    if !value.is_finite() {
        return Err(serde::de::Error::custom(
            "numerical operation coefficient must be finite FP32",
        ));
    }
    Ok(value)
}

/// A shared parameter allocation. Multiple operations may reference one index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorParameter {
    /// Checkpoint tensor name, independent of the consuming operation.
    pub name: String,
    /// Original checkpoint dimensions; parameters use BF16 storage.
    pub shape: Vec<usize>,
}

impl TensorParameter {
    /// Checked byte extent of this parameter's BF16 storage.
    pub fn bytes(&self) -> Result<usize> {
        if self.shape.is_empty() {
            return Err(invalid("numerical parameter must have a nonempty shape"));
        }
        self.shape.iter().try_fold(2usize, |bytes, &dimension| {
            if dimension == 0 {
                return Err(invalid("zero-sized numerical parameter"));
            }
            bytes
                .checked_mul(dimension)
                .ok_or_else(|| invalid("numerical parameter extent overflow"))
        })
    }
}

/// Numerical operation. Value references address preceding operations.
/// All intermediate results are BF16 except the explicit Float32 conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum TensorOperation {
    /// Gather one embedding row for each input token.
    Embedding {
        /// Shared vocabulary-by-hidden parameter index.
        weight: usize,
    },
    /// Row-major input multiplied by a shared row-major weight transpose.
    Linear {
        /// Preceding activation value.
        input: usize,
        /// Shared output-by-input parameter index.
        weight: usize,
    },
    /// RMS normalization independently on each weight-width group.
    RmsNorm {
        /// Preceding activation value.
        input: usize,
        /// Shared normalization scale vector index.
        weight: usize,
        /// Positive constant added to the mean square.
        #[serde(deserialize_with = "deserialize_f32")]
        epsilon: f32,
    },
    /// Rotate each attention head using absolute causal token positions.
    Rotary {
        /// Preceding query or key value in token/head/coordinate order.
        input: usize,
        /// Number of heads in this value.
        heads: usize,
        /// Even number of coordinates per head.
        head_dim: usize,
        /// Positive rotary frequency base.
        #[serde(deserialize_with = "deserialize_f32")]
        theta: f32,
    },
    /// Causal grouped-query attention with FP32 softmax accumulation.
    Attention {
        /// Preceding query value.
        query: usize,
        /// Preceding key value.
        key: usize,
        /// Preceding value projection.
        value: usize,
        /// Number of query heads.
        heads: usize,
        /// Number of shared key/value heads, dividing the query head count.
        kv_heads: usize,
        /// Coordinates per head.
        head_dim: usize,
    },
    /// BF16 residual addition with identical input shapes.
    Add {
        /// First preceding activation.
        left: usize,
        /// Second preceding activation of identical shape.
        right: usize,
    },
    /// BF16 SiLU of the gate, followed by BF16 elementwise multiplication.
    SiluGate {
        /// Preceding activation to which SiLU is applied.
        gate: usize,
        /// Preceding activation multiplied by the activated gate.
        value: usize,
    },
    /// Select the caller's prediction position, retaining its exact hidden row.
    PredictionRow {
        /// Complete preceding causal sequence.
        input: usize,
    },
    /// Convert an already rounded BF16 result to FP32.
    Float32 {
        /// Preceding BF16 value.
        input: usize,
    },
}

impl TensorOperation {
    /// Value dependencies in operand order, including repeated references.
    pub fn inputs(&self) -> Vec<usize> {
        match *self {
            Self::Embedding { .. } => vec![],
            Self::Linear { input, .. }
            | Self::RmsNorm { input, .. }
            | Self::Rotary { input, .. }
            | Self::PredictionRow { input }
            | Self::Float32 { input } => vec![input],
            Self::Attention {
                query, key, value, ..
            } => vec![query, key, value],
            Self::Add { left, right } => vec![left, right],
            Self::SiluGate { gate, value } => vec![gate, value],
        }
    }
}

/// Runtime shape of one numerical value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorValueShape {
    /// Number of contiguous rows.
    pub rows: usize,
    /// Number of elements per row.
    pub columns: usize,
    /// Whether storage is FP32 rather than BF16.
    pub float32: bool,
}

/// One numerical function, including its parameter identity table.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorProgram {
    /// Parameters in canonical checkpoint order. Aliases reuse an index.
    pub parameters: Vec<TensorParameter>,
    /// Topologically ordered numerical operations.
    pub operations: Vec<TensorOperation>,
    /// Index of the externally observable result.
    pub output: usize,
    /// Admitted original vocabulary cardinality.
    pub vocabulary: usize,
    /// Maximum complete causal input length.
    pub max_context: usize,
}

impl TensorProgram {
    /// Check all references, matrix shapes and numerical attributes before execution.
    pub fn shapes(
        &self,
        tokens: usize,
        prediction_position: usize,
    ) -> Result<Vec<TensorValueShape>> {
        if tokens == 0 || tokens > self.max_context || prediction_position >= tokens {
            return Err(invalid(
                "causal input length or prediction position is out of bounds",
            ));
        }
        if self.output >= self.operations.len() || self.vocabulary == 0 {
            return Err(invalid(
                "numerical program has no valid output or vocabulary",
            ));
        }
        for parameter in &self.parameters {
            parameter.bytes()?;
        }
        let mut shapes: Vec<TensorValueShape> = Vec::with_capacity(self.operations.len());
        for operation in &self.operations {
            let input = |id: usize| -> Result<TensorValueShape> {
                let shape = *shapes
                    .get(id)
                    .ok_or_else(|| invalid("numerical value is not a preceding definition"))?;
                if shape.float32 {
                    return Err(invalid("BF16 operation received an FP32 value"));
                }
                Ok(shape)
            };
            let parameter = |id: usize| -> Result<&[usize]> {
                self.parameters
                    .get(id)
                    .map(|value| value.shape.as_slice())
                    .ok_or_else(|| invalid("unknown numerical parameter"))
            };
            let shape = match *operation {
                TensorOperation::Embedding { weight } => {
                    let &[vocabulary, columns] = parameter(weight)? else {
                        return Err(invalid("embedding weight must be a matrix"));
                    };
                    if vocabulary != self.vocabulary {
                        return Err(invalid(
                            "embedding vocabulary differs from program vocabulary",
                        ));
                    }
                    TensorValueShape {
                        rows: tokens,
                        columns,
                        float32: false,
                    }
                }
                TensorOperation::Linear { input: id, weight } => {
                    let source = input(id)?;
                    let &[columns, inner] = parameter(weight)? else {
                        return Err(invalid("linear weight must be a matrix"));
                    };
                    if inner != source.columns {
                        return Err(invalid("linear input and parameter dimensions differ"));
                    }
                    TensorValueShape { columns, ..source }
                }
                TensorOperation::RmsNorm {
                    input: id,
                    weight,
                    epsilon,
                } => {
                    let source = input(id)?;
                    let &[width] = parameter(weight)? else {
                        return Err(invalid("normalization weight must be a vector"));
                    };
                    if source.columns % width != 0 || !epsilon.is_finite() || epsilon <= 0.0 {
                        return Err(invalid("invalid RMS normalization width or epsilon"));
                    }
                    source
                }
                TensorOperation::Rotary {
                    input: id,
                    heads,
                    head_dim,
                    theta,
                } => {
                    let source = input(id)?;
                    if head_dim == 0
                        || head_dim % 2 != 0
                        || heads.checked_mul(head_dim) != Some(source.columns)
                        || source.rows != tokens
                        || !theta.is_finite()
                        || theta <= 0.0
                    {
                        return Err(invalid(
                            "invalid rotary embedding geometry or frequency base",
                        ));
                    }
                    source
                }
                TensorOperation::Attention {
                    query,
                    key,
                    value,
                    heads,
                    kv_heads,
                    head_dim,
                } => {
                    let q = input(query)?;
                    let k = input(key)?;
                    let v = input(value)?;
                    if kv_heads == 0
                        || heads == 0
                        || heads % kv_heads != 0
                        || head_dim == 0
                        || heads.checked_mul(head_dim) != Some(q.columns)
                        || kv_heads.checked_mul(head_dim) != Some(k.columns)
                        || q.rows != tokens
                        || k.rows != tokens
                        || k != v
                    {
                        return Err(invalid("invalid causal grouped-query attention geometry"));
                    }
                    q
                }
                TensorOperation::Add { left, right }
                | TensorOperation::SiluGate {
                    gate: left,
                    value: right,
                } => {
                    let lhs = input(left)?;
                    if lhs != input(right)? {
                        return Err(invalid("elementwise numerical input shapes differ"));
                    }
                    lhs
                }
                TensorOperation::PredictionRow { input: id } => {
                    let source = input(id)?;
                    if source.rows != tokens {
                        return Err(invalid(
                            "prediction row must select from the complete causal sequence",
                        ));
                    }
                    TensorValueShape { rows: 1, ..source }
                }
                TensorOperation::Float32 { input: id } => TensorValueShape {
                    float32: true,
                    ..input(id)?
                },
            };
            shape
                .rows
                .checked_mul(shape.columns)
                .and_then(|n| n.checked_mul(if shape.float32 { 4 } else { 2 }))
                .ok_or_else(|| invalid("numerical value extent overflow"))?;
            shapes.push(shape);
        }
        if shapes[self.output]
            != (TensorValueShape {
                rows: 1,
                columns: self.vocabulary,
                float32: true,
            })
        {
            return Err(invalid(
                "causal program must return one full FP32 vocabulary row",
            ));
        }
        Ok(shapes)
    }
}

fn invalid(message: &str) -> XlogError {
    XlogError::Compilation(message.to_owned())
}
