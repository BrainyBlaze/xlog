use xlog_prob::exact::ExactDdnnfProgram;

fn has_cuda_device() -> bool {
    cudarc::driver::CudaDevice::count().unwrap_or(0) > 0
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

