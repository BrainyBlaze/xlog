use xlog_cuda::CudaDevice;
use xlog_prob::exact::ExactDdnnfProgram;

fn has_cuda_device() -> bool {
    // cudarc::driver::CudaDevice::count() may panic in restricted containers. Attempt real init instead.
    CudaDevice::new(0).is_ok()
}

/// Verify that gradients through negation have the correct sign and magnitude.
///
/// For `dry() :- not rain()`, we have `P(dry) = 1 - P(rain)`.
/// Therefore `dP(dry)/dP(rain) = -1`.
///
/// This test verifies that when we query `dry()`, the gradient with respect
/// to the `rain` leaf shows that increasing P(rain) decreases P(dry).
///
/// The gradient computation in exact inference computes:
///   grad[Q] = grad[log Z(E∧Q)] - grad[log Z(E)]
///
/// For no evidence, Z(E) = 1 (tautology), so the gradients simplify.
/// The observed gradient pattern for negation is:
///   grad_true[rain] = -p_rain (negative: rain=true excludes dry worlds)
///   grad_false[rain] = p_rain (positive: rain=false enables dry worlds)
///
/// These gradients sum to 0, reflecting that the total probability is conserved.
/// The negative gradient for grad_true confirms that increasing p_rain decreases P(dry).
#[test]
fn test_exact_negation_gradient_direction() {
    if !has_cuda_device() {
        eprintln!("Skipping test: no CUDA device available");
        return;
    }

    let p_rain = 0.5_f64;
    let source = format!(
        r#"
{p_rain}::rain().
dry() :- not rain().
query(dry()).
"#
    );

    let compiled = ExactDdnnfProgram::compile_source(&source).unwrap();
    let result = compiled.evaluate_gpu_with_grads().unwrap();

    assert_eq!(result.query_grads.len(), 1);
    let grads = &result.query_grads[0];

    // Verify P(dry) = 1 - p_rain
    let expected_prob = 1.0 - p_rain;
    assert!(
        (grads.prob - expected_prob).abs() < 1e-6,
        "P(dry) should be {}, got {}",
        expected_prob,
        grads.prob
    );

    // Verify gradients exist
    assert!(
        grads.grad_true.len() > 1,
        "Expected gradient vector to have entries for leaf variables"
    );

    // The rain leaf gets variable index 1
    let rain_grad_true = grads.grad_true[1];
    let rain_grad_false = grads.grad_false[1];

    // Key verification: gradient for rain=true should be NEGATIVE
    // This confirms that increasing P(rain) decreases P(dry)
    assert!(
        rain_grad_true < 0.0,
        "grad_true[rain] should be negative for negation, got {}",
        rain_grad_true
    );

    // Gradient for rain=false should be POSITIVE
    // This confirms that increasing P(not rain) increases P(dry)
    assert!(
        rain_grad_false > 0.0,
        "grad_false[rain] should be positive for negation, got {}",
        rain_grad_false
    );

    // The gradients sum to 0 (probability conservation)
    let grad_sum = rain_grad_true + rain_grad_false;
    assert!(
        grad_sum.abs() < 1e-6,
        "Gradients should sum to 0, got {}",
        grad_sum
    );

    // Verify the actual values match the expected pattern
    // For negation without evidence: grad_true = -p, grad_false = p
    // (This is because the circuit encodes the negation through auxiliary variables)
    assert!(
        (rain_grad_true - (-p_rain)).abs() < 1e-6,
        "grad_true[rain] should be {}, got {}",
        -p_rain,
        rain_grad_true
    );
    assert!(
        (rain_grad_false - p_rain).abs() < 1e-6,
        "grad_false[rain] should be {}, got {}",
        p_rain,
        rain_grad_false
    );
}

/// Verify gradients for negation with a different probability value.
/// This confirms the gradient formula works for non-symmetric probabilities.
#[test]
fn test_exact_negation_gradient_asymmetric() {
    if !has_cuda_device() {
        eprintln!("Skipping test: no CUDA device available");
        return;
    }

    let p_rain = 0.3_f64;
    let source = format!(
        r#"
{p_rain}::rain().
dry() :- not rain().
query(dry()).
"#
    );

    let compiled = ExactDdnnfProgram::compile_source(&source).unwrap();
    let result = compiled.evaluate_gpu_with_grads().unwrap();

    assert_eq!(result.query_grads.len(), 1);
    let grads = &result.query_grads[0];

    // Verify P(dry) = 1 - p_rain = 0.7
    let expected_prob = 1.0 - p_rain;
    assert!(
        (grads.prob - expected_prob).abs() < 1e-6,
        "P(dry) should be {}, got {}",
        expected_prob,
        grads.prob
    );

    let rain_grad_true = grads.grad_true[1];
    let rain_grad_false = grads.grad_false[1];

    // Verify the gradient pattern: grad_true = -p, grad_false = p
    assert!(
        (rain_grad_true - (-p_rain)).abs() < 1e-6,
        "grad_true[rain] should be {}, got {}",
        -p_rain,
        rain_grad_true
    );
    assert!(
        (rain_grad_false - p_rain).abs() < 1e-6,
        "grad_false[rain] should be {}, got {}",
        p_rain,
        rain_grad_false
    );

    // Verify the sign relationship (key test for negation)
    assert!(rain_grad_true < 0.0, "grad_true[rain] should be negative");
    assert!(rain_grad_false > 0.0, "grad_false[rain] should be positive");
}

#[test]
fn test_exact_ddnnf_gpu_with_grads_matches_expected_for_or_evidence() {
    if !has_cuda_device() {
        eprintln!("Skipping test: no CUDA device available");
        return;
    }

    let source = r#"
0.7::a().
0.2::b().
c() :- a().
c() :- b().
evidence(c(), true).
query(a()).
query(b()).
"#;

    let compiled = ExactDdnnfProgram::compile_source(source).unwrap();
    let result = compiled.evaluate_gpu_with_grads().unwrap();

    let a = 0.7_f64;
    let b = 0.2_f64;
    let z_e = a + b - a * b;

    // log P(a|c) = log Z(E∧a) - log Z(E)
    // Under world semantics: Z(E)=P(a ∨ b), Z(E∧a)=P(a).
    // With smooth WMC gradients, ∂logZ/∂log w_x gives posterior marginal of that literal.
    let expected_a_true = 1.0 - a / z_e;
    let expected_a_false = -((1.0 - a) * b) / z_e;
    let expected_b_true = b - b / z_e;
    let expected_b_false = (1.0 - b) - a * (1.0 - b) / z_e;

    let g = result
        .query_grads
        .iter()
        .find(|q| q.atom.predicate == "a")
        .expect("missing gradients for query a()");

    assert!((g.grad_true[1] - expected_a_true).abs() < 1e-9);
    assert!((g.grad_false[1] - expected_a_false).abs() < 1e-9);
    assert!((g.grad_true[2] - expected_b_true).abs() < 1e-9);
    assert!((g.grad_false[2] - expected_b_false).abs() < 1e-9);
}
