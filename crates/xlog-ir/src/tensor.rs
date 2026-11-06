//! Typed, acyclic numerical programs with shared parameter identities.

use serde::{Deserialize, Serialize};
use xlog_core::{Result, XlogError};

mod functions;
pub use functions::TensorSignature;

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
    /// Physical dimensions, shared by every call referencing this identity.
    pub shape: Vec<usize>,
    /// FP32 acquired parameters, or BF16 imported parameters when false.
    pub float32: bool,
}

impl TensorParameter {
    /// Checked byte extent in this physical parameter's declared storage type.
    pub fn bytes(&self) -> Result<usize> {
        if self.shape.is_empty() {
            return Err(invalid("numerical parameter must have a nonempty shape"));
        }
        self.shape.iter().try_fold(
            if self.float32 { 4usize } else { 2usize },
            |bytes, &dimension| {
                if dimension == 0 {
                    return Err(invalid("zero-sized numerical parameter"));
                }
                bytes
                    .checked_mul(dimension)
                    .ok_or_else(|| invalid("numerical parameter extent overflow"))
            },
        )
    }
}

/// Numerical operation. Value references address preceding operations.
/// Imported decoder operations use BF16; acquired operations use explicit FP32.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum TensorOperation {
    /// A fixed zero row, including a scan's nontrainable initial state.
    Zeros {
        /// Number of FP32 coordinates.
        columns: usize,
    },
    /// A numerical argument in the definition's declared signature.
    Argument {
        /// Zero-based numeric argument index.
        argument: usize,
    },
    /// Read a shared FP32 parameter without copying its physical identity.
    Parameter {
        /// Shared parameter-table index.
        weight: usize,
    },
    /// Indicators of original token IDs, independently at each causal position.
    TokenIndicators {
        /// Original vocabulary IDs, in output-coordinate order.
        token_ids: Vec<u32>,
    },
    /// Broadcast a single FP32 row to the available causal positions.
    Broadcast {
        /// Preceding numeric value.
        input: usize,
    },
    /// FP32 hyperbolic tangent.
    Tanh {
        /// Preceding numeric value.
        input: usize,
    },
    /// FP32 elementwise product.
    Multiply {
        /// First preceding FP32 value.
        left: usize,
        /// Second FP32 value with the same extent.
        right: usize,
    },
    /// FP32 elementwise subtraction.
    Subtract {
        /// Minuend value.
        left: usize,
        /// Subtrahend with the same extent.
        right: usize,
    },
    /// FP32 elementwise division; nonfinite results reject an update.
    Divide {
        /// Numerator value.
        left: usize,
        /// Denominator with the same extent.
        right: usize,
    },
    /// Gather coordinates in the declared order, permitting repeated indices.
    GatherColumns {
        /// Preceding FP32 rows.
        input: usize,
        /// Discrete source coordinates, including repeated indices.
        indices: Vec<usize>,
    },
    /// Discrete mask: positive where condition > 0, negative otherwise.
    Select {
        /// Discrete sign mask with the same shape as both branches.
        condition: usize,
        /// Numeric branch selected where the condition is positive.
        positive: usize,
        /// Numeric branch selected elsewhere.
        negative: usize,
    },
    /// Round FP32 to BF16. Its reverse rule is the declared straight-through surrogate.
    Bfloat16 {
        /// Preceding numeric value.
        input: usize,
    },
    /// Validate a called causal function's FP32 result before its projection.
    Finite {
        /// Preceding feature rows, retaining the identity reverse rule.
        input: usize,
    },
    /// Invoke an immutable definition in the same function registry.
    Call {
        /// Canonical registry name.
        function: String,
        /// Preceding values in numeric argument order.
        arguments: Vec<usize>,
    },
    /// Inclusive causal scan; the pure cell takes (previous state, current input).
    /// No prefix truncation is permitted when the bound is exceeded.
    Scan {
        /// Sequence of raw-prefix feature rows.
        input: usize,
        /// Single FP32 row preceding the first raw token.
        initial: usize,
        /// Pure cell definition in the common registry.
        function: String,
        /// Maximum number of raw tokens, with no truncation.
        max_steps: usize,
    },
    /// Select a statically known row during bounded scan lowering.
    Row {
        /// Preceding FP32 matrix.
        input: usize,
        /// Row selected by scan lowering.
        row: usize,
    },
    /// Assemble scan states in causal order during lowering.
    Stack {
        /// Single-row states in causal position order.
        inputs: Vec<usize>,
    },
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
        if let Some(inputs) = self.function_inputs() {
            return inputs;
        }
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
            _ => unreachable!("function operation handled above"),
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorProgram {
    /// Public numerical calling convention of this canonical definition.
    pub signature: TensorSignature,
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
        self.shapes_resolved(tokens, prediction_position, &|_| {
            Err(invalid("unresolved numerical call"))
        })
    }

    /// Validate with signatures resolved from the authoritative function registry.
    pub fn shapes_resolved(
        &self,
        tokens: usize,
        prediction_position: usize,
        resolve: &dyn Fn(&str) -> Result<TensorSignature>,
    ) -> Result<Vec<TensorValueShape>> {
        self.validate_acquired_body()?;
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
        let mut parameter_names = std::collections::HashSet::new();
        for parameter in &self.parameters {
            parameter.bytes()?;
            if parameter.name.is_empty() || !parameter_names.insert(&parameter.name) {
                return Err(invalid(
                    "parameter identities must be nonempty and unique; aliases reuse an index",
                ));
            }
        }
        let mut shapes: Vec<TensorValueShape> = Vec::with_capacity(self.operations.len());
        for operation in &self.operations {
            let shape = self.operation_shape_with(
                operation,
                tokens,
                &|id| {
                    shapes
                        .get(id)
                        .copied()
                        .ok_or_else(|| invalid("numerical value is not preceding"))
                },
                resolve,
            )?;
            shapes.push(shape);
        }
        if shapes[self.output] != self.signature.output(tokens, self.vocabulary)? {
            return Err(invalid(
                "numerical program output differs from its declared signature",
            ));
        }
        Ok(shapes)
    }

    /// Infer one operation using the compiler's canonical shape rules.
    /// Both complete compilation and prefix operand selection resolve real
    /// preceding values and known callee signatures through this boundary.
    pub fn operation_shape_with(
        &self,
        operation: &TensorOperation,
        tokens: usize,
        value: &dyn Fn(usize) -> Result<TensorValueShape>,
        resolve: &dyn Fn(&str) -> Result<TensorSignature>,
    ) -> Result<TensorValueShape> {
        if let Some(shape) = self.function_shape(operation, value, tokens, resolve)? {
            return Ok(shape);
        }
        let input = |id: usize| -> Result<TensorValueShape> {
            let shape = value(id)?;
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
                if self
                    .parameters
                    .get(weight)
                    .is_some_and(|parameter| parameter.float32)
                {
                    return Err(invalid("decoder embeddings require BF16 parameters"));
                }
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
                let &[columns, inner] = parameter(weight)? else {
                    return Err(invalid("linear weight must be a matrix"));
                };
                let source = value(id)?;
                if source.float32
                    != self
                        .parameters
                        .get(weight)
                        .ok_or_else(|| invalid("unknown parameter"))?
                        .float32
                {
                    return Err(invalid("linear parameter and activation types differ"));
                }
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
                if self
                    .parameters
                    .get(weight)
                    .is_some_and(|parameter| parameter.float32)
                {
                    return Err(invalid("decoder normalization requires BF16 parameters"));
                }
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
            TensorOperation::SiluGate {
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
            _ => unreachable!("function operation handled above"),
        };
        shape
            .rows
            .checked_mul(shape.columns)
            .and_then(|n| n.checked_mul(if shape.float32 { 4 } else { 2 }))
            .ok_or_else(|| invalid("numerical value extent overflow"))?;
        Ok(shape)
    }
}

fn invalid(message: &str) -> XlogError {
    XlogError::Compilation(message.to_owned())
}
