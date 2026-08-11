use xlog_core::{Result, XlogError};

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
            return Err(XlogError::Compilation(
                "circuit log-sum-exp encountered NaN".to_string(),
            ));
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
    use super::circuit_logsumexp;
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
}
