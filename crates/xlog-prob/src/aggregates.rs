use xlog_core::{f64_total_order_key_from_bits, Result, XlogError};
use xlog_logic::ast::AggOp;

use crate::provenance::Value;

#[derive(Debug, Clone)]
pub(crate) enum AggState {
    Count(u64),
    SumI128(i128),
    SumF64(f64),
    Min(Option<Value>),
    Max(Option<Value>),
    LogSumExp { max: f64, sumexp: f64, init: bool },
}

/// Canonical, totally-ordered key over an [`AggState`] for deduplicating
/// dynamic-programming states in factorized aggregate-outcome folding.
/// Floats are keyed by their exact bit pattern.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum AggStateKey {
    Count(u64),
    SumI128(i128),
    SumF64(u64),
    Min(Option<Value>),
    Max(Option<Value>),
    LogSumExp {
        max_bits: u64,
        sumexp_bits: u64,
        init: bool,
    },
}

impl AggState {
    pub(crate) fn new(op: AggOp) -> Self {
        match op {
            AggOp::Count => AggState::Count(0),
            AggOp::Sum => AggState::SumI128(0),
            AggOp::Min => AggState::Min(None),
            AggOp::Max => AggState::Max(None),
            AggOp::LogSumExp => AggState::LogSumExp {
                max: f64::NEG_INFINITY,
                sumexp: 0.0,
                init: false,
            },
        }
    }

    pub(crate) fn update(&mut self, op: AggOp, v: &Value) -> Result<()> {
        match op {
            AggOp::Count => match self {
                AggState::Count(c) => {
                    *c = c.saturating_add(1);
                    Ok(())
                }
                _ => Err(internal_state_error()),
            },
            AggOp::Sum => match self {
                AggState::SumI128(acc) => match v {
                    Value::I64(i) => {
                        *acc += *i as i128;
                        Ok(())
                    }
                    Value::F64(bits) => {
                        let f = f64::from_bits(*bits);
                        let acc_f = *acc as f64;
                        *self = AggState::SumF64(acc_f + f);
                        Ok(())
                    }
                    _ => Err(XlogError::Compilation(
                        "sum() aggregate requires numeric input".to_string(),
                    )),
                },
                AggState::SumF64(acc) => match v {
                    Value::I64(i) => {
                        *acc += *i as f64;
                        Ok(())
                    }
                    Value::F64(bits) => {
                        *acc += f64::from_bits(*bits);
                        Ok(())
                    }
                    _ => Err(XlogError::Compilation(
                        "sum() aggregate requires numeric input".to_string(),
                    )),
                },
                _ => Err(internal_state_error()),
            },
            AggOp::Min => match self {
                AggState::Min(current) => {
                    match current {
                        None => *current = Some(v.clone()),
                        Some(c) => {
                            if value_le(v, c)? {
                                *current = Some(v.clone());
                            }
                        }
                    }
                    Ok(())
                }
                _ => Err(internal_state_error()),
            },
            AggOp::Max => match self {
                AggState::Max(current) => {
                    match current {
                        None => *current = Some(v.clone()),
                        Some(c) => {
                            if value_le(c, v)? {
                                *current = Some(v.clone());
                            }
                        }
                    }
                    Ok(())
                }
                _ => Err(internal_state_error()),
            },
            AggOp::LogSumExp => match self {
                AggState::LogSumExp { max, sumexp, init } => {
                    let x = match v {
                        Value::I64(i) => *i as f64,
                        Value::F64(bits) => f64::from_bits(*bits),
                        _ => {
                            return Err(XlogError::Compilation(
                                "logsumexp() aggregate requires numeric input".to_string(),
                            ))
                        }
                    };
                    if x.is_nan() {
                        return Err(XlogError::Compilation(
                            "logsumexp() aggregate encountered NaN".to_string(),
                        ));
                    }
                    if !*init {
                        *max = x;
                        *sumexp = 1.0;
                        *init = true;
                        return Ok(());
                    }
                    if x > *max {
                        *sumexp = *sumexp * (*max - x).exp() + 1.0;
                        *max = x;
                    } else {
                        *sumexp += (x - *max).exp();
                    }
                    Ok(())
                }
                _ => Err(internal_state_error()),
            },
        }
    }

    pub(crate) fn dp_key(&self) -> AggStateKey {
        match self {
            AggState::Count(c) => AggStateKey::Count(*c),
            AggState::SumI128(acc) => AggStateKey::SumI128(*acc),
            AggState::SumF64(acc) => AggStateKey::SumF64(acc.to_bits()),
            AggState::Min(v) => AggStateKey::Min(v.clone()),
            AggState::Max(v) => AggStateKey::Max(v.clone()),
            AggState::LogSumExp { max, sumexp, init } => AggStateKey::LogSumExp {
                max_bits: max.to_bits(),
                sumexp_bits: sumexp.to_bits(),
                init: *init,
            },
        }
    }

    pub(crate) fn finish(&self, op: AggOp) -> Result<Value> {
        match (op, self) {
            (AggOp::Count, AggState::Count(c)) => {
                let v: i64 = (*c)
                    .try_into()
                    .map_err(|_| XlogError::Compilation("count() overflowed i64".to_string()))?;
                Ok(Value::I64(v))
            }
            (AggOp::Sum, AggState::SumI128(acc)) => {
                let v: i64 = (*acc)
                    .try_into()
                    .map_err(|_| XlogError::Compilation("sum() overflowed i64".to_string()))?;
                Ok(Value::I64(v))
            }
            (AggOp::Sum, AggState::SumF64(v)) => Ok(Value::F64(v.to_bits())),
            (AggOp::Min, AggState::Min(v)) => v.clone().ok_or_else(|| {
                XlogError::Compilation("min() aggregate produced no value".to_string())
            }),
            (AggOp::Max, AggState::Max(v)) => v.clone().ok_or_else(|| {
                XlogError::Compilation("max() aggregate produced no value".to_string())
            }),
            (AggOp::LogSumExp, AggState::LogSumExp { max, sumexp, init }) => {
                if !*init {
                    return Ok(Value::F64(f64::NEG_INFINITY.to_bits()));
                }
                Ok(Value::F64((max + sumexp.ln()).to_bits()))
            }
            _ => Err(internal_state_error()),
        }
    }
}

fn value_le(a: &Value, b: &Value) -> Result<bool> {
    match (a, b) {
        (Value::I64(x), Value::I64(y)) => Ok(x <= y),
        (Value::F64(x), Value::F64(y)) => {
            Ok(f64_total_order_key_from_bits(*x) <= f64_total_order_key_from_bits(*y))
        }
        (Value::Symbol(x), Value::Symbol(y)) => Ok(x <= y),
        (Value::String(x), Value::String(y)) => Ok(x <= y),
        _ => Err(XlogError::Compilation(
            "min/max aggregate requires consistent comparable types".to_string(),
        )),
    }
}

fn internal_state_error() -> XlogError {
    XlogError::Compilation("Internal aggregate state mismatch".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn selected_f64_bits(op: AggOp, inputs: &[u64]) -> u64 {
        let mut state = AggState::new(op);
        for &bits in inputs {
            state
                .update(op, &Value::F64(bits))
                .expect("aggregate update");
        }
        let Value::F64(bits) = state.finish(op).expect("aggregate result") else {
            panic!("expected f64 aggregate result");
        };
        bits
    }

    #[test]
    fn floating_min_max_are_total_and_input_order_independent() {
        let pairs = [
            ((-0.0_f64).to_bits(), 0.0_f64.to_bits()),
            (0xfff8_0000_0000_0042, 0x7ff8_0000_0000_0024),
            (0x7ff8_0000_0000_0001, 0x7ff8_0000_0000_0002),
        ];

        for (left, right) in pairs {
            let expected_min =
                if f64_total_order_key_from_bits(left) <= f64_total_order_key_from_bits(right) {
                    left
                } else {
                    right
                };
            let expected_max =
                if f64_total_order_key_from_bits(left) >= f64_total_order_key_from_bits(right) {
                    left
                } else {
                    right
                };
            for inputs in [[left, right], [right, left]] {
                assert_eq!(selected_f64_bits(AggOp::Min, &inputs), expected_min);
                assert_eq!(selected_f64_bits(AggOp::Max, &inputs), expected_max);
            }
        }
    }
}
