//! Typed encoding of ground logic terms into relation-column bytes.

use std::fmt;

use xlog_core::{symbol, ScalarType};

use crate::Term;

/// A structured failure produced while encoding a ground term for a scalar column.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum GroundTermEncodingError {
    /// An integer cannot be represented by the requested scalar type.
    IntegerOutOfRange {
        /// Requested scalar type.
        expected: ScalarType,
        /// Integer value that could not be represented.
        value: i64,
    },
    /// A boolean integer was neither zero nor one.
    InvalidBooleanInteger {
        /// Integer supplied for the boolean column.
        value: i64,
    },
    /// A boolean symbol was neither `true` nor `false`.
    InvalidBooleanSymbol {
        /// Resolved symbol text supplied for the boolean column.
        symbol: String,
    },
    /// A fact contained a named variable instead of a ground term.
    Variable {
        /// Variable name found in the fact.
        name: String,
    },
    /// A fact contained an anonymous wildcard instead of a ground term.
    Anonymous,
    /// A fact contained an aggregate expression instead of a ground term.
    Aggregate,
    /// The term form is not supported by the requested scalar type.
    TypeMismatch {
        /// Requested scalar type.
        expected: ScalarType,
        /// Term that did not match the requested scalar type.
        actual: Term,
    },
}

impl fmt::Display for GroundTermEncodingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IntegerOutOfRange { expected, value } => {
                let scalar_name = match expected {
                    ScalarType::U32 => "u32",
                    ScalarType::U64 => "u64",
                    ScalarType::I32 => "i32",
                    ScalarType::I64 => "i64",
                    ScalarType::F32 => "f32",
                    ScalarType::F64 => "f64",
                    ScalarType::Bool => "bool",
                    ScalarType::Symbol => "symbol",
                };
                write!(formatter, "{scalar_name} out of range: {value}")
            }
            Self::InvalidBooleanInteger { value } => {
                write!(formatter, "bool expects 0/1, got {value}")
            }
            Self::InvalidBooleanSymbol { symbol } => write!(
                formatter,
                "Expected boolean symbol 'true' or 'false', got '{symbol}'"
            ),
            Self::Variable { name } => write!(formatter, "Fact cannot contain variable {name}"),
            Self::Anonymous => formatter.write_str("Fact cannot contain anonymous wildcard '_'"),
            Self::Aggregate => formatter.write_str("Fact cannot contain aggregate"),
            Self::TypeMismatch { expected, actual } => write!(
                formatter,
                "Type mismatch in fact: expected {expected:?}, got {actual:?}"
            ),
        }
    }
}

impl std::error::Error for GroundTermEncodingError {}

/// Append the little-endian physical representation of a typed ground term.
///
/// Existing bytes in `output` are preserved. If encoding fails, `output` is
/// left unchanged and the returned structured error identifies the rejected
/// value or term form.
pub fn append_ground_term_bytes(
    output: &mut Vec<u8>,
    term: &Term,
    scalar_type: ScalarType,
) -> Result<(), GroundTermEncodingError> {
    match (scalar_type, term) {
        (ScalarType::U32, Term::Integer(value)) => {
            let encoded =
                u32::try_from(*value).map_err(|_| GroundTermEncodingError::IntegerOutOfRange {
                    expected: ScalarType::U32,
                    value: *value,
                })?;
            output.extend_from_slice(&encoded.to_le_bytes());
        }
        (ScalarType::U64, Term::Integer(value)) => {
            let encoded =
                u64::try_from(*value).map_err(|_| GroundTermEncodingError::IntegerOutOfRange {
                    expected: ScalarType::U64,
                    value: *value,
                })?;
            output.extend_from_slice(&encoded.to_le_bytes());
        }
        (ScalarType::I32, Term::Integer(value)) => {
            let encoded =
                i32::try_from(*value).map_err(|_| GroundTermEncodingError::IntegerOutOfRange {
                    expected: ScalarType::I32,
                    value: *value,
                })?;
            output.extend_from_slice(&encoded.to_le_bytes());
        }
        (ScalarType::I64, Term::Integer(value)) => {
            output.extend_from_slice(&value.to_le_bytes());
        }
        (ScalarType::F32, Term::Float(value)) => {
            output.extend_from_slice(&(*value as f32).to_le_bytes());
        }
        (ScalarType::F64, Term::Float(value)) => {
            output.extend_from_slice(&value.to_le_bytes());
        }
        (ScalarType::F32, Term::Integer(value)) => {
            output.extend_from_slice(&(*value as f32).to_le_bytes());
        }
        (ScalarType::F64, Term::Integer(value)) => {
            output.extend_from_slice(&(*value as f64).to_le_bytes());
        }
        (ScalarType::Bool, Term::Integer(value)) => match *value {
            0 => output.push(0),
            1 => output.push(1),
            value => {
                return Err(GroundTermEncodingError::InvalidBooleanInteger { value });
            }
        },
        (ScalarType::Bool, Term::Symbol(id)) => {
            let value = symbol::resolve(*id);
            match value.as_str() {
                "false" => output.push(0),
                "true" => output.push(1),
                _ => {
                    return Err(GroundTermEncodingError::InvalidBooleanSymbol { symbol: value });
                }
            }
        }
        (ScalarType::Symbol, Term::String(value)) => {
            output.extend_from_slice(&symbol::intern(value).to_le_bytes());
        }
        (ScalarType::Symbol, Term::Symbol(id)) => {
            output.extend_from_slice(&id.to_le_bytes());
        }
        (_, Term::Variable(name)) => {
            return Err(GroundTermEncodingError::Variable { name: name.clone() });
        }
        (_, Term::Anonymous) => return Err(GroundTermEncodingError::Anonymous),
        (_, Term::Aggregate(_)) => return Err(GroundTermEncodingError::Aggregate),
        (expected, actual) => {
            return Err(GroundTermEncodingError::TypeMismatch {
                expected,
                actual: actual.clone(),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{append_ground_term_bytes, GroundTermEncodingError};
    use crate::ast::{AggExpr, AggOp};
    use crate::Term;
    use xlog_core::{symbol, ScalarType};

    #[test]
    fn encodes_supported_scalar_terms() {
        let true_symbol = symbol::intern("true");
        let false_symbol = symbol::intern("false");
        let existing_symbol = symbol::intern("existing");
        let string_symbol = symbol::intern("from-string");
        let cases = vec![
            (
                "u32 integer",
                ScalarType::U32,
                Term::Integer(42),
                42_u32.to_le_bytes().to_vec(),
            ),
            (
                "u64 integer",
                ScalarType::U64,
                Term::Integer(43),
                43_u64.to_le_bytes().to_vec(),
            ),
            (
                "i32 integer",
                ScalarType::I32,
                Term::Integer(-44),
                (-44_i32).to_le_bytes().to_vec(),
            ),
            (
                "i64 integer",
                ScalarType::I64,
                Term::Integer(-45),
                (-45_i64).to_le_bytes().to_vec(),
            ),
            (
                "f32 float",
                ScalarType::F32,
                Term::Float(1.5),
                1.5_f32.to_le_bytes().to_vec(),
            ),
            (
                "f64 float",
                ScalarType::F64,
                Term::Float(2.25),
                2.25_f64.to_le_bytes().to_vec(),
            ),
            (
                "f32 integer",
                ScalarType::F32,
                Term::Integer(46),
                46_f32.to_le_bytes().to_vec(),
            ),
            (
                "f64 integer",
                ScalarType::F64,
                Term::Integer(47),
                47_f64.to_le_bytes().to_vec(),
            ),
            ("false integer", ScalarType::Bool, Term::Integer(0), vec![0]),
            ("true integer", ScalarType::Bool, Term::Integer(1), vec![1]),
            (
                "true symbol",
                ScalarType::Bool,
                Term::Symbol(true_symbol),
                vec![1],
            ),
            (
                "false symbol",
                ScalarType::Bool,
                Term::Symbol(false_symbol),
                vec![0],
            ),
            (
                "string symbol",
                ScalarType::Symbol,
                Term::String("from-string".to_string()),
                string_symbol.to_le_bytes().to_vec(),
            ),
            (
                "interned symbol",
                ScalarType::Symbol,
                Term::Symbol(existing_symbol),
                existing_symbol.to_le_bytes().to_vec(),
            ),
        ];

        for (label, scalar_type, term, expected) in cases {
            let mut actual = vec![0xA5];
            append_ground_term_bytes(&mut actual, &term, scalar_type)
                .unwrap_or_else(|error| panic!("{label} should encode successfully, got {error}"));
            let mut expected_with_prefix = vec![0xA5];
            expected_with_prefix.extend_from_slice(&expected);
            assert_eq!(actual, expected_with_prefix, "{label}");
        }
    }

    #[test]
    fn rejects_out_of_range_and_non_ground_terms() {
        let invalid_bool = symbol::intern("not-a-boolean");
        let cases = vec![
            (
                "negative u32",
                ScalarType::U32,
                Term::Integer(-1),
                GroundTermEncodingError::IntegerOutOfRange {
                    expected: ScalarType::U32,
                    value: -1,
                },
                "u32 out of range: -1",
            ),
            (
                "large u32",
                ScalarType::U32,
                Term::Integer(i64::from(u32::MAX) + 1),
                GroundTermEncodingError::IntegerOutOfRange {
                    expected: ScalarType::U32,
                    value: i64::from(u32::MAX) + 1,
                },
                "u32 out of range: 4294967296",
            ),
            (
                "negative u64",
                ScalarType::U64,
                Term::Integer(-1),
                GroundTermEncodingError::IntegerOutOfRange {
                    expected: ScalarType::U64,
                    value: -1,
                },
                "u64 out of range: -1",
            ),
            (
                "small i32",
                ScalarType::I32,
                Term::Integer(i64::from(i32::MIN) - 1),
                GroundTermEncodingError::IntegerOutOfRange {
                    expected: ScalarType::I32,
                    value: i64::from(i32::MIN) - 1,
                },
                "i32 out of range: -2147483649",
            ),
            (
                "large i32",
                ScalarType::I32,
                Term::Integer(i64::from(i32::MAX) + 1),
                GroundTermEncodingError::IntegerOutOfRange {
                    expected: ScalarType::I32,
                    value: i64::from(i32::MAX) + 1,
                },
                "i32 out of range: 2147483648",
            ),
            (
                "invalid integer boolean",
                ScalarType::Bool,
                Term::Integer(2),
                GroundTermEncodingError::InvalidBooleanInteger { value: 2 },
                "bool expects 0/1, got 2",
            ),
            (
                "invalid boolean symbol",
                ScalarType::Bool,
                Term::Symbol(invalid_bool),
                GroundTermEncodingError::InvalidBooleanSymbol {
                    symbol: "not-a-boolean".to_string(),
                },
                "Expected boolean symbol 'true' or 'false', got 'not-a-boolean'",
            ),
            (
                "variable",
                ScalarType::U32,
                Term::Variable("X".to_string()),
                GroundTermEncodingError::Variable {
                    name: "X".to_string(),
                },
                "Fact cannot contain variable X",
            ),
            (
                "anonymous",
                ScalarType::U32,
                Term::Anonymous,
                GroundTermEncodingError::Anonymous,
                "Fact cannot contain anonymous wildcard '_'",
            ),
            (
                "aggregate",
                ScalarType::U64,
                Term::Aggregate(AggExpr {
                    op: AggOp::Count,
                    variable: "X".to_string(),
                }),
                GroundTermEncodingError::Aggregate,
                "Fact cannot contain aggregate",
            ),
            (
                "float mismatch",
                ScalarType::U32,
                Term::Float(1.5),
                GroundTermEncodingError::TypeMismatch {
                    expected: ScalarType::U32,
                    actual: Term::Float(1.5),
                },
                "Type mismatch in fact: expected U32, got Float(1.5)",
            ),
            (
                "string mismatch",
                ScalarType::U64,
                Term::String("text".to_string()),
                GroundTermEncodingError::TypeMismatch {
                    expected: ScalarType::U64,
                    actual: Term::String("text".to_string()),
                },
                "Type mismatch in fact: expected U64, got String(\"text\")",
            ),
            (
                "list mismatch",
                ScalarType::U64,
                Term::List(vec![]),
                GroundTermEncodingError::TypeMismatch {
                    expected: ScalarType::U64,
                    actual: Term::List(vec![]),
                },
                "Type mismatch in fact: expected U64, got List([])",
            ),
            (
                "cons mismatch",
                ScalarType::U64,
                Term::Cons {
                    head: Box::new(Term::Integer(1)),
                    tail: Box::new(Term::List(vec![])),
                },
                GroundTermEncodingError::TypeMismatch {
                    expected: ScalarType::U64,
                    actual: Term::Cons {
                        head: Box::new(Term::Integer(1)),
                        tail: Box::new(Term::List(vec![])),
                    },
                },
                "Type mismatch in fact: expected U64, got Cons { head: Integer(1), tail: List([]) }",
            ),
            (
                "compound mismatch",
                ScalarType::U64,
                Term::Compound {
                    functor: "pair".to_string(),
                    args: vec![Term::Integer(1), Term::Integer(2)],
                },
                GroundTermEncodingError::TypeMismatch {
                    expected: ScalarType::U64,
                    actual: Term::Compound {
                        functor: "pair".to_string(),
                        args: vec![Term::Integer(1), Term::Integer(2)],
                    },
                },
                "Type mismatch in fact: expected U64, got Compound { functor: \"pair\", args: [Integer(1), Integer(2)] }",
            ),
            (
                "predicate reference mismatch",
                ScalarType::U64,
                Term::PredRef("target".to_string()),
                GroundTermEncodingError::TypeMismatch {
                    expected: ScalarType::U64,
                    actual: Term::PredRef("target".to_string()),
                },
                "Type mismatch in fact: expected U64, got PredRef(\"target\")",
            ),
        ];

        for (label, scalar_type, term, expected_error, expected_message) in cases {
            let mut output = vec![0x5A];
            let error = match append_ground_term_bytes(&mut output, &term, scalar_type) {
                Ok(()) => panic!("{label} unexpectedly encoded"),
                Err(error) => error,
            };
            assert_eq!(error, expected_error, "{label}");
            assert_eq!(error.to_string(), expected_message, "{label}");
            assert_eq!(output, vec![0x5A], "{label} mutated output on failure");
        }
    }
}
