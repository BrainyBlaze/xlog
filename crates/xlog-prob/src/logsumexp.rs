use xlog_core::{Result, XlogError};

fn nan_error(context: &str) -> XlogError {
    XlogError::Compilation(format!("{context} encountered NaN"))
}

/// Validates one true/false log-weight pair for value-only circuit evaluation.
///
/// Both polarities are part of the weight input contract even when the circuit consumes only one
/// of them. Positive and negative infinity remain valid for value-only evaluation.
pub(crate) fn validate_circuit_log_weight_pair(weights: (f64, f64)) -> Result<(f64, f64)> {
    if weights.0.is_nan() || weights.1.is_nan() {
        return Err(nan_error("circuit log weights"));
    }
    Ok(weights)
}

/// Rejects a NaN produced while evaluating a circuit node or read back from a GPU root.
pub(crate) fn validate_circuit_value(value: f64) -> Result<f64> {
    if value.is_nan() {
        return Err(nan_error("circuit evaluation"));
    }
    Ok(value)
}

/// Validates gradient arrays returned across a device-to-host boundary.
///
/// Gradient outputs must be finite even when value-only circuit evaluation legitimately returns
/// infinity. True-polarity gradients are scanned before false-polarity gradients, and each array
/// is scanned in ascending variable order, so the reported error is deterministic.
pub(crate) fn validate_circuit_gradient_values(
    grad_true: &[f64],
    grad_false: &[f64],
) -> Result<()> {
    for (polarity, gradients) in [("true", grad_true), ("false", grad_false)] {
        if let Some((index, _)) = gradients
            .iter()
            .enumerate()
            .find(|(_, gradient)| !gradient.is_finite())
        {
            return Err(XlogError::Compilation(format!(
                "circuit gradient {polarity}[{index}] is non-finite"
            )));
        }
    }
    Ok(())
}

/// Computes circuit log-sum-exp with a stable max shift for finite inputs.
///
/// Empty inputs and inputs containing only negative infinity return negative infinity.
/// Positive infinity returns positive infinity when no NaN is present. The initial full
/// scan gives NaN explicit precedence over infinity and returns a typed
/// [`XlogError::Compilation`] error whenever any input is NaN.
pub(crate) fn circuit_logsumexp(values: &[f64]) -> Result<f64> {
    let mut max = f64::NEG_INFINITY;
    for &value in values {
        if value.is_nan() {
            return Err(nan_error("circuit log-sum-exp"));
        }
        if value > max {
            max = value;
        }
    }
    if max.is_infinite() {
        return Ok(max);
    }

    let mut sum = 0.0;
    for &value in values {
        sum += (value - max).exp();
    }
    Ok(max + sum.ln())
}

#[cfg(test)]
mod tests {
    use super::{circuit_logsumexp, validate_circuit_gradient_values};
    use xlog_core::XlogError;

    #[test]
    fn circuit_logsumexp_edge_contract() {
        let valid_cases: &[(&str, &[f64], f64)] = &[
            ("empty", &[], f64::NEG_INFINITY),
            (
                "all negative infinity",
                &[f64::NEG_INFINITY, f64::NEG_INFINITY],
                f64::NEG_INFINITY,
            ),
            (
                "positive infinity",
                &[f64::NEG_INFINITY, 3.0, f64::INFINITY],
                f64::INFINITY,
            ),
            (
                "large positive finite",
                &[1000.0, 999.0],
                1000.3132616875182,
            ),
            (
                "large negative finite",
                &[-1000.0, -1001.0],
                -999.6867383124818,
            ),
        ];

        for &(name, values, expected) in valid_cases {
            let actual = circuit_logsumexp(values)
                .unwrap_or_else(|error| panic!("{name}: expected a value, got {error}"));
            if expected.is_infinite() {
                assert_eq!(actual, expected, "{name}");
            } else {
                assert!((actual - expected).abs() < 1e-12, "{name}: {actual}");
            }
        }

        for values in [
            &[f64::NAN][..],
            &[0.0, f64::NAN][..],
            &[f64::INFINITY, f64::NAN][..],
        ] {
            let error = circuit_logsumexp(values).expect_err("NaN must be rejected");
            assert!(
                matches!(error, XlogError::Compilation(ref message) if message.contains("NaN")),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn gradient_value_validation_is_finite_and_deterministic() {
        validate_circuit_gradient_values(&[0.0, 1.0], &[2.0]).unwrap();

        let true_error =
            validate_circuit_gradient_values(&[0.0, f64::INFINITY], &[f64::NAN]).unwrap_err();
        assert!(
            matches!(true_error, XlogError::Compilation(ref message) if message.contains("true[1]")),
            "unexpected error: {true_error}"
        );

        let false_error =
            validate_circuit_gradient_values(&[0.0], &[f64::NEG_INFINITY]).unwrap_err();
        assert!(
            matches!(false_error, XlogError::Compilation(ref message) if message.contains("false[0]")),
            "unexpected error: {false_error}"
        );
    }
}
