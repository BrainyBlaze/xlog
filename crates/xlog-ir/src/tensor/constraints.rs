//! Necessary shape relations shared by concrete and partial numerical inference.

use super::*;

/// Each absent component is unconstrained, rather than a representative tensor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TensorShapeConstraint {
    /// Known row count at the current context length.
    pub rows: Option<usize>,
    /// Known number of coordinates in each row.
    pub columns: Option<usize>,
    /// Known storage precision.
    pub float32: Option<bool>,
}

impl From<TensorValueShape> for TensorShapeConstraint {
    fn from(shape: TensorValueShape) -> Self {
        Self {
            rows: Some(shape.rows),
            columns: Some(shape.columns),
            float32: Some(shape.float32),
        }
    }
}

impl TensorShapeConstraint {
    /// Facts shared by alternative admitted domains. A differing or unknown
    /// component stays unknown; no representative shape is selected.
    pub fn common(self, other: Self) -> Self {
        Self {
            rows: self.rows.filter(|value| other.rows == Some(*value)),
            columns: self.columns.filter(|value| other.columns == Some(*value)),
            float32: self.float32.filter(|value| other.float32 == Some(*value)),
        }
    }

    /// Intersect known facts, refusing only contradictory selected components.
    pub fn intersect(&mut self, other: Self) -> Result<()> {
        fn component<T: Copy + Eq>(value: &mut Option<T>, other: Option<T>) -> Result<()> {
            if value.is_some() && other.is_some() && *value != other {
                return Err(invalid("numerical shape constraints conflict"));
            }
            if other.is_some() {
                *value = other;
            }
            Ok(())
        }
        component(&mut self.rows, other.rows)?;
        component(&mut self.columns, other.columns)?;
        component(&mut self.float32, other.float32)
    }

    pub(super) fn concrete(self) -> Option<TensorValueShape> {
        Some(TensorValueShape {
            rows: self.rows?,
            columns: self.columns?,
            float32: self.float32?,
        })
    }
}

impl TensorOperation {
    /// Intersect the known physical Linear weight with possibly unknown
    /// activation/result slots. No representative activation or SSA value is
    /// created; full inference uses the same physical matrix domain.
    pub fn constrain_linear_shapes(
        parameter: &TensorParameter,
        input: &mut TensorShapeConstraint,
        output: &mut TensorShapeConstraint,
    ) -> Result<()> {
        let [columns, inner] = parameter.linear_dimensions()?;
        if input
            .float32
            .is_some_and(|float32| float32 != parameter.float32)
        {
            return Err(invalid("linear parameter and activation types differ"));
        }
        if input.columns.is_some_and(|columns| columns != inner) {
            return Err(invalid("linear input and parameter dimensions differ"));
        }
        input.intersect(TensorShapeConstraint {
            columns: Some(inner),
            float32: Some(parameter.float32),
            ..Default::default()
        })?;
        output.intersect(TensorShapeConstraint {
            rows: input.rows,
            columns: Some(columns),
            float32: Some(parameter.float32),
        })?;
        input.intersect(TensorShapeConstraint {
            rows: output.rows,
            ..Default::default()
        })
    }

    /// Propagate canonical equal-extent, broadcast and gather relations through
    /// unknown operand slots in wire operand order. Unknown gather columns stay
    /// free. Other operations return false and leave all slots unchanged; their
    /// concrete inference and scalar/callee domains remain authoritative.
    pub fn constrain_shapes(
        tag: &str,
        inputs: &mut [TensorShapeConstraint],
        output: &mut TensorShapeConstraint,
        tokens: usize,
        gather_columns: Option<usize>,
    ) -> Result<bool> {
        let (arity, equal, fp32) = match tag {
            "tanh" | "finite" => (1, true, true),
            "add" => (2, true, false),
            "multiply" | "subtract" | "divide" => (2, true, true),
            "select" => (3, true, true),
            "broadcast" | "gather_columns" => (1, false, true),
            _ => return Ok(false),
        };
        if inputs.len() != arity {
            return Err(invalid("numerical shape operand arity differs"));
        }
        let precision = TensorShapeConstraint {
            float32: fp32.then_some(true),
            ..Default::default()
        };
        output.intersect(precision)?;
        if equal {
            let mut shared = *output;
            for input in inputs.iter() {
                shared.intersect(*input)?;
            }
            *output = shared;
            inputs.fill(shared);
        } else {
            let input = &mut inputs[0];
            input.intersect(precision)?;
            if tag == "broadcast" {
                input.intersect(TensorShapeConstraint {
                    rows: Some(1),
                    ..Default::default()
                })?;
                output.intersect(TensorShapeConstraint {
                    rows: Some(tokens),
                    columns: input.columns,
                    ..Default::default()
                })?;
                input.intersect(TensorShapeConstraint {
                    columns: output.columns,
                    ..Default::default()
                })?;
            } else {
                output.intersect(TensorShapeConstraint {
                    rows: input.rows,
                    columns: gather_columns,
                    ..Default::default()
                })?;
                input.intersect(TensorShapeConstraint {
                    rows: output.rows,
                    ..Default::default()
                })?;
            }
        }
        Ok(true)
    }
}
