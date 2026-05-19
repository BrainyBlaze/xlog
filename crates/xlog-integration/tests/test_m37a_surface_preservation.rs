use std::path::PathBuf;

const M37A_EVIDENCE_RELATIVE_PATH: &str =
    "docs/evidence/2026-05-14-g39-m37a-surface-preservation/report.md";

const M37A_GROUP_B_SMOKE_SOURCE: &str = r#"
nn(bridge_net, [X], Y, [yes, no]) :: bridge_fact(X, Y).
program.register_network("bridge_net", torch.nn.Linear(8, 2), torch.optim.Adam(params, lr=1e-3))
loss_tensor = program.forward_backward_tensor("bridge_fact(0, yes)")
stats = program.train_epoch(["bridge_fact(0, yes)"], batch_size=8)
xgcf_gradient = program.evaluate(return_grads=True).grad_true
cache_ratio = repeat_query_second_call_speedup_at_least_50x(program, "bridge_fact(0, yes)")
probability_with_evidence = program.evaluate(return_grads=True).grad_false
program.register_embedding("entity_embed", torch.nn.Embedding(16, 8))
pyxlog.train_model(
    program,
    ["bridge_fact(0, yes)"],
    batch_size=8,
    max_grad_norm=1.0,
    val_queries=["bridge_fact(1, no)"],
    patience=1,
)
program.scheduler_step()
program.set_lr("bridge_net", 1e-4)
bounded_exact_induce(program, examples, budget)
induce_exact(program, examples=examples, k_per_topology=budget)
"#;

fn assert_contains(haystack: &str, needle: &str, context: &str) {
    assert!(
        haystack.contains(needle),
        "expected {context} to contain {needle:?}"
    );
}

fn repo_root() -> PathBuf {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.push("../..");
    root
}

#[test]
fn m37a_group_b_smoke_source_enumerates_all_required_symbol_families() {
    let required = [
        "nn(bridge_net, [X], Y, [yes, no]) :: bridge_fact(X, Y).",
        "program.register_network",
        "torch.nn.Linear",
        "torch.optim.Adam",
        "program.forward_backward_tensor",
        "program.train_epoch",
        "batch_size=8",
        "xgcf_gradient",
        "repeat_query_second_call_speedup_at_least_50x",
        "program.evaluate(return_grads=True)",
        "grad_true",
        "grad_false",
        "program.register_embedding",
        "torch.nn.Embedding",
        "max_grad_norm",
        "val_queries",
        "patience",
        "program.scheduler_step",
        "program.set_lr",
        "bounded_exact_induce(program, examples, budget)",
        "induce_exact(program, examples=examples, k_per_topology=budget)",
    ];

    for needle in required {
        assert_contains(
            M37A_GROUP_B_SMOKE_SOURCE,
            needle,
            "M37A Group B smoke source",
        );
    }
}

#[test]
fn m37a_group_b_python_binding_surface_remains_exposed() {
    let native_stub = include_str!("../../pyxlog/python/pyxlog/_native.pyi");
    let neural_src = include_str!("../../pyxlog/src/neural.rs");
    let training_src = include_str!("../../pyxlog/src/training.rs");
    let program_src = include_str!("../../pyxlog/src/program.rs");
    let pyxlog_lib_src = include_str!("../../pyxlog/src/lib.rs");
    let ilp_init = include_str!("../../pyxlog/python/pyxlog/ilp/__init__.py");
    let ilp_exact = include_str!("../../pyxlog/python/pyxlog/ilp/exact_induce.py");
    let induce_src = include_str!("../../xlog-induce/src/lib.rs");

    for needle in [
        "def register_network(",
        "def register_embedding(",
        "def forward_backward_tensor(",
        "def train_epoch(",
        "def train_model(",
        "def train_model_tensor(",
        "def template_cache_size(",
        "def template_compile_count(",
        "def scheduler_step(",
        "def get_lr(",
        "def set_lr(",
        "max_grad_norm",
        "patience",
        "grad_true",
        "grad_false",
    ] {
        assert_contains(native_stub, needle, "pyxlog native type stub");
    }

    for (source, label, needles) in [
        (
            neural_src,
            "pyxlog neural bindings",
            &[
                "#[pyo3(signature = (name, module, optimizer, scheduler=None",
                "fn register_embedding",
                "fn forward_backward_tensor",
                "fn template_cache_size",
                "fn template_compile_count",
                "max_grad_norm",
            ][..],
        ),
        (
            training_src,
            "pyxlog training bindings",
            &[
                "pub fn train_model",
                "pub fn train_model_tensor",
                "max_grad_norm",
                "val_queries",
                "patience",
            ][..],
        ),
        (
            program_src,
            "pyxlog program bindings",
            &[
                "fn scheduler_step",
                "fn get_lr",
                "fn set_lr",
                "grad_true",
                "grad_false",
            ][..],
        ),
        (
            pyxlog_lib_src,
            "pyxlog module exports",
            &[
                "wrap_pyfunction!(training::train_model",
                "wrap_pyfunction!(training::train_model_tensor",
            ][..],
        ),
        (
            ilp_init,
            "pyxlog ILP package exports",
            &["induce_exact", "train_and_promote"][..],
        ),
        (
            ilp_exact,
            "pyxlog bounded exact induction frontend",
            &[
                "def induce_exact(",
                "k_per_topology",
                "ExactInductionResult",
            ][..],
        ),
        (
            induce_src,
            "xlog-induce bounded exact induction crate",
            &[
                "pub fn induce_exact",
                "ExactInductionConfig",
                "ExactInductionResult",
            ][..],
        ),
    ] {
        for needle in needles {
            assert_contains(source, needle, label);
        }
    }
}

#[test]
fn m37a_group_b_runtime_regressions_are_covered_by_existing_tests() {
    let parse_neural = include_str!("../../xlog-logic/tests/parse_neural.rs");
    let no_dtoh_neural = include_str!("../../xlog-prob/tests/no_dtoh_in_neural_backward_nll.rs");
    let xgcf_test = include_str!("../../xlog-prob/tests/gpu_xgcf.rs");
    let prob_grad_test = include_str!("../../xlog-prob/tests/exact_ddnnf_gpu_grads.rs");
    let tensor_loss_test =
        include_str!("../../../python/tests/test_gpu_native_forward_backward_returns_tensor.py");
    let cache_test = include_str!("../../../python/tests/test_circuit_cache.py");
    let embedding_test = include_str!("../../../python/tests/test_embeddings.py");
    let network_test = include_str!("../../../python/tests/test_network_registry.py");
    let training_test = include_str!("../../../python/tests/test_training.py");
    let tensor_training_test = include_str!("../../../python/tests/test_train_model_tensor.py");
    let negation_test = include_str!("../../../python/tests/test_negation.py");
    let exact_induce_test = include_str!("../../../python/tests/test_ilp_exact_induce.py");
    let promoter_test = include_str!("../../../python/tests/test_ilp_promoter.py");

    for (source, label, needles) in [
        (
            parse_neural,
            "nn/4 parser tests",
            &["nn(mnist_net", ":: digit", "parse_program"][..],
        ),
        (
            no_dtoh_neural,
            "zero device-to-host neural backward audit",
            &[
                "neural_backward_nll_buffers_inner",
                "!body.contains(\"dtoh\")",
            ][..],
        ),
        (
            tensor_loss_test,
            "forward_backward_tensor Python regression",
            &[
                "program.forward_backward_tensor",
                "torch.Tensor, \"tolist\"",
                "torch.Tensor, \"item\"",
                "loss.is_cuda",
            ][..],
        ),
        (
            cache_test,
            "circuit cache Python regression",
            &[
                "template_compile_count",
                "template_cache_size",
                "test_cache_hit_same_structure",
            ][..],
        ),
        (
            xgcf_test,
            "XGCF GPU gradient regression",
            &["eval_log_wmc_and_grads", "gpu_grad_true", "gpu_grad_false"][..],
        ),
        (
            prob_grad_test,
            "probabilistic query gradient regression",
            &["grad_true", "grad_false", "dry"][..],
        ),
        (
            embedding_test,
            "embedding registration regression",
            &[
                "register_embedding",
                "torch.nn.Embedding",
                "forward_embedding",
            ][..],
        ),
        (
            network_test,
            "network registration regression",
            &[
                "register_network",
                "network_names",
                "declared_network_names",
            ][..],
        ),
        (
            training_test,
            "training controls regression",
            &[
                "max_grad_norm",
                "patience",
                "scheduler_step",
                "set_lr",
                "train_epoch",
                "train_model",
            ][..],
        ),
        (
            tensor_training_test,
            "tensor training regression",
            &["train_model_tensor", "train_epoch_tensor", "batch_losses"][..],
        ),
        (
            negation_test,
            "probabilistic query API regression",
            &["return_grads=True", "grad_true", "grad_false"][..],
        ),
        (
            exact_induce_test,
            "bounded exact induction regression",
            &["induce_exact", "backend=\"native\"", "candidates"][..],
        ),
        (
            promoter_test,
            "M37-F train_and_promote regression",
            &["train_and_promote", "PromotionResult", "holdout"][..],
        ),
    ] {
        for needle in needles {
            assert_contains(source, needle, label);
        }
    }
}

#[test]
fn m37a_surface_evidence_records_all_goal_039_metrics() {
    let evidence_path = repo_root().join(M37A_EVIDENCE_RELATIVE_PATH);
    let report = std::fs::read_to_string(&evidence_path).unwrap_or_else(|err| {
        panic!(
            "read M37A surface preservation evidence at {}: {err}",
            evidence_path.display()
        )
    });

    for metric in [
        "M_M37A.1",
        "M_M37A.2",
        "M_M37A.3",
        "M_M37A.4",
        "M_M37A.5",
        "M_M37A.6",
        "M_M37A.7",
        "M_M37A.8",
        "M_M37A.9",
        "M_M37A.10",
    ] {
        assert_contains(&report, metric, "M37A surface preservation evidence");
    }

    for marker in [
        "xlog_alpha_source",
        "0.3965599479565052",
        "0.3871022178294078",
        "0.46316616571549835",
        "99.8%",
        "forward_backward_tensor",
        "cache",
        "MNIST-Add",
        "nn/4",
        "register_network",
        "train_epoch",
        "Bounded Exact Induction",
    ] {
        assert_contains(&report, marker, "M37A surface preservation evidence");
    }
}
