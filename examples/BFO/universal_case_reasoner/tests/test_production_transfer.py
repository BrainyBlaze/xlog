from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_TRANSFER_TEST_TIMEOUT_SEC = 300


def _cuda_oom_text(stdout: str, stderr: str) -> bool:
    text = f"{stdout}\n{stderr}".lower()
    return "cuda_error_out_of_memory" in text or "out of memory" in text


def _score(record: dict[str, object]) -> float:
    return 1.0 if record["root_prediction"] == record["root_label"] else 0.0


def _runner_module():
    spec = importlib.util.spec_from_file_location(
        "run_production_transfer",
        ROOT / "tools" / "run_production_transfer.py",
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_computed_metrics_keep_showcase_baselines_namespaced() -> None:
    runner = _runner_module()
    records = [
        {
            "domain_id": "cybersecurity_intrusion",
            "root_label": "cyber_root",
            "root_prediction": "cyber_root",
            "intervention_label": "cyber_fix",
            "intervention_prediction": "cyber_fix",
            "explanation_valid": True,
        },
        {
            "domain_id": "clinical_deterioration",
            "root_label": "clinical_root",
            "root_prediction": "clinical_root",
            "intervention_label": "clinical_fix",
            "intervention_prediction": "clinical_fix",
            "explanation_valid": True,
        },
    ]
    ablation_records = [
        {
            "domain_id": "cybersecurity_intrusion",
            "neural_only": {"root_label": "cyber_root", "root_prediction": "wrong_root"},
            "domain_symbolic": {"root_label": "cyber_root", "root_prediction": "wrong_root"},
            "shared_symbolic": {"root_label": "cyber_root", "root_prediction": "wrong_root"},
            "neuro_symbolic": {"root_label": "cyber_root", "root_prediction": "cyber_root"},
        }
    ]
    invalid_records = [{"rejected": True}]

    computed = runner._computed_metrics_from_records(
        records,
        ablation_records,
        invalid_records,
        "cybersecurity_intrusion",
    )

    assert "baseline_metrics" not in computed
    assert "strongest_baseline" not in computed
    assert "relative_uplift_over_best_baseline_pct" not in computed
    assert computed["showcase_metrics"]["baseline_metrics"] == {
        "domain_symbolic": 0.0,
        "neural_only": 0.0,
        "neuro_symbolic": 1.0,
        "shared_symbolic": 0.0,
    }
    assert computed["showcase_metrics"]["relative_uplift_over_best_baseline_pct"] == 100.0


def test_public_benchmark_report_is_explicit_nonclaim_until_adapters_exist() -> None:
    runner = _runner_module()

    report = runner._public_benchmark_report()

    assert report["status"] == "FAIL"
    assert report["external_sota_claim"] is False
    assert report["covered_public_benchmark_families"] == []
    assert report["missing_public_benchmark_families"]
    assert "MISSING_PUBLIC_SOTA_RUNNER" in report["blockers"]


def test_cuda_ranking_hot_paths_do_not_materialize_host_scalars_or_score_rows() -> None:
    runner = _runner_module()
    hot_functions = [
        runner._train_transfer_ranker,
        runner._evaluate_transfer_cases,
        runner._score_cuda_generalization_candidates,
        runner._invoke_generalization_nn4_training_path,
        runner._dilp_score_tensor,
        runner._dilp_evaluate_cases,
    ]
    banned_fragments = [
        ".cpu()",
        ".tolist()",
        ".item()",
        ".get(",
        "int(selected_index",
        "float(final_loss",
        "float(transfer_loss",
        "float(torch.stack",
        "bool(torch.",
    ]

    violations = {
        function.__name__: [
            fragment
            for fragment in banned_fragments
            if fragment in inspect.getsource(function)
        ]
        for function in hot_functions
    }

    assert not {name: hits for name, hits in violations.items() if hits}


def test_held_out_candidate_features_do_not_expose_gold_root_marker() -> None:
    runner = _runner_module()
    inventory = runner._load_inventory()
    cases, _sources = runner._load_huggingface_cases(inventory, rows_per_domain=10)
    held_out_domain = inventory["holdout_protocol"]["held_out_domain"]
    held_out_cases = [case for case in cases if case["domain_id"] == held_out_domain]

    assert held_out_cases
    for case in held_out_cases:
        true_candidate_indexes = []
        for index, candidate in enumerate(case["candidates"]):
            assert "has_bfo_evidence" not in candidate
            assert "gold" not in candidate
            assert "label" not in candidate
            if candidate["root"] == case["root_label"]:
                true_candidate_indexes.append(index)
        assert len(true_candidate_indexes) == 1
    assert len({case["candidates"].index(next(
        candidate for candidate in case["candidates"] if candidate["root"] == case["root_label"]
    )) for case in held_out_cases}) > 1

    feature_count = len(held_out_cases[0]["candidates"][0]["feature"])
    for feature_index in range(feature_count):
        positives = []
        negatives = []
        for case in held_out_cases:
            for candidate in case["candidates"]:
                values = positives if candidate["root"] == case["root_label"] else negatives
                values.append(float(candidate["feature"][feature_index]))
        assert positives and negatives
        unique_values = set(positives + negatives)
        if len(unique_values) <= 2:
            assert set(positives) & set(negatives), (
                f"feature column {feature_index} is a binary gold-root marker"
            )


def test_generalization_evidence_covers_every_domain_without_test_label_candidates() -> None:
    runner = _runner_module()
    inventory = runner._load_inventory()
    domains = [domain["id"] for domain in inventory["domains"]]
    cases, _sources = runner._load_huggingface_cases(inventory, rows_per_domain=4)

    evidence = runner._build_generalization_evidence(
        domain_ids=domains,
        cases=cases,
        bootstrap_iterations=1000,
    )

    records = evidence["prediction_records"]
    clean_records = [
        record for record in records if record["evaluation_variant"] == "clean"
    ]
    assert {record["held_out_domain"] for record in clean_records} == set(domains)
    assert evidence["report"]["excluded_domains"] == []
    assert evidence["report"]["aggregate"]["domain_ids"] == domains
    assert evidence["report"]["statistical_confidence"]["bootstrap_iterations"] == 1000

    required_variants = {
        "clean",
        "noisy",
        "sparse",
        "paraphrased",
        "missing_field",
        "distractor_candidate",
    }
    assert required_variants <= {
        record["evaluation_variant"] for record in records
    }
    required_baselines = {
        "neural_only",
        "symbolic_only",
        "domain_specific_classifier",
        "retrieval_rag_nearest_neighbor",
        "majority_prior",
        "neuro_symbolic",
    }
    assert required_baselines <= set(evidence["report"]["baseline_methods"])
    assert all(required_baselines <= set(record) for record in evidence["ablation_records"])

    for record in clean_records:
        generation = record["candidate_generation"]
        assert generation["uses_heldout_test_truth"] is False
        assert generation["constructed_before_heldout_labels"] is True
        assert generation["candidate_count"] >= 3
        assert record["root_label"] not in generation["heldout_test_row_root_labels_used"]


def test_generalization_evidence_uses_cuda_xlog_nn4_ranker() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for integrated generalization ranker evidence")
    runner = _runner_module()
    inventory = runner._load_inventory()
    domains = [domain["id"] for domain in inventory["domains"]]
    cases, _sources = runner._load_huggingface_cases(inventory, rows_per_domain=2)
    torch.manual_seed(0)
    net = runner.ProductionRootNet().to(torch.device("cuda"))

    evidence = runner._build_generalization_evidence(
        domain_ids=domains,
        cases=cases,
        bootstrap_iterations=50,
        net=net,
        device=torch.device("cuda"),
        nn4_training_query_limit=8,
        training_epochs=1,
    )

    ranker = evidence["report"]["neural_ranker"]
    assert ranker["path"] == "xlog_nn4_cuda_generalization"
    assert ranker["program"] == "programs/production_ranker.xlog"
    assert ranker["registered_network"] == "production_root_net"
    assert ranker["uses_python_heuristic"] is False
    assert ranker["selection_device"] == "cuda"
    assert ranker["nn4_query_count"] > 0
    assert ranker["score_cpu_materialization_in_ranking"] is False
    assert ranker["full_score_rows_materialized"] is False
    assert ranker["scalar_item_calls_in_ranking"] is False
    assert ranker["post_ranking_evidence_serialization"] == "selected_indices_only"
    heldout_scoring = ranker["heldout_scoring"]
    assert heldout_scoring["path"] == "xlog_nn4_forward_backward_tensor"
    assert heldout_scoring["program"] == "programs/production_ranker.xlog"
    assert heldout_scoring["expected_label"] == "primary_root"
    assert heldout_scoring["uses_heldout_labels"] is False
    assert heldout_scoring["loss_tensors_device"] == "cuda"
    assert heldout_scoring["score_tensor_device"] == "cuda"
    assert heldout_scoring["score_cpu_materialization_in_ranking"] is False
    assert heldout_scoring["query_count"] >= sum(
        record["candidate_generation"]["candidate_count"]
        for record in evidence["prediction_records"]
    )
    assert all(
        record["ranker_path"] == "xlog_nn4_cuda_generalization"
        for record in evidence["prediction_records"]
    )
    assert all(
        record["neural_scores"]["materialized"] is False
        for record in evidence["prediction_records"]
    )


def test_dilp_evidence_learns_xlog_proof_clauses_without_heldout_labels() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for DILP evidence")
    runner = _runner_module()
    inventory = runner._load_inventory()
    domains = [domain["id"] for domain in inventory["domains"]]
    cases, _sources = runner._load_huggingface_cases(inventory, rows_per_domain=4)
    torch.manual_seed(0)
    net = runner.ProductionRootNet().to(torch.device("cuda"))

    evidence = runner._build_dilp_evidence(
        domain_ids=domains,
        cases=cases,
        net=net,
        device=torch.device("cuda"),
        training_epochs=12,
    )

    assert evidence["status"] == "PASS"
    assert evidence["program"] == "programs/dilp_proof_paths.xlog"
    assert evidence["xlog_proof_path_queries"] > 0
    assert evidence["joint_training"]["trained_jointly"] is True
    assert evidence["joint_training"]["symbolic_rule_weights_device"] == "cuda"
    assert evidence["joint_training"]["neural_predicate"] == "production_root_net"
    assert evidence["joint_training"]["symbolic_rule_gradient_norm"] > 0.0
    assert evidence["joint_training"]["neural_weight_gradient_norm"] > 0.0
    assert evidence["joint_training"]["proof_path_gradient_norm"] > 0.0
    assert evidence["heldout_safe_rule_induction"]["passed"] is True
    assert evidence["heldout_safe_rule_induction"]["heldout_examples_in_training"] == 0
    assert {record["held_out_domain"] for record in evidence["rule_inventory"]} == set(domains)
    assert all(record["selected_clause"] for record in evidence["rule_inventory"])
    assert evidence["clause_ablations"]["full_model_macro_f1"] >= (
        evidence["clause_ablations"]["best_ablated_macro_f1"]
    )


def test_cuda_generalization_seed_is_isolated_from_showcase_transfer_training() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for integrated generalization ranker evidence")
    runner = _runner_module()
    inventory = runner._load_inventory()
    domain = runner._domain_contract(inventory)
    domains = domain["domain_ids"]
    cases, _sources = runner._load_huggingface_cases(inventory, rows_per_domain=100)
    transfer_cases = runner._representative_transfer_cases(cases, rows_per_domain=10)
    torch.manual_seed(0)
    _neural, showcase_net = runner._run_neural_contract(1000)
    generalization_seed = runner._clone_root_net(showcase_net, torch.device("cuda"))

    runner._evaluate_transfer_cases(
        transfer_cases,
        showcase_net,
        torch.device("cuda"),
        domain["held_out_domain"],
    )
    evidence = runner._build_generalization_evidence(
        domain_ids=domains,
        cases=cases,
        bootstrap_iterations=200,
        net=generalization_seed,
        device=torch.device("cuda"),
        training_epochs=160,
        generalization_seed_isolated_from_showcase_transfer=True,
    )

    aggregate = evidence["report"]["aggregate"]
    assert aggregate["macro_held_out_root_cause_f1"] >= 0.90
    assert aggregate["min_domain_root_cause_f1"] >= 0.85
    assert "GEN-003" not in evidence["report"]["blockers"]
    assert evidence["report"]["frozen_model_rules"][
        "generalization_seed_isolated_from_showcase_transfer"
    ] is True


def test_generalization_clean_transfer_uses_canonical_external_rca_labels() -> None:
    runner = _runner_module()
    inventory = runner._load_inventory()
    domains = [domain["id"] for domain in inventory["domains"]]
    cases, _sources = runner._load_huggingface_cases(inventory, rows_per_domain=100)

    assert all(case["root_truth"]["canonicalization"] == "frozen_external_rca_catalog" for case in cases)
    evidence = runner._build_generalization_evidence(
        domain_ids=domains,
        cases=cases,
        bootstrap_iterations=200,
    )
    aggregate = evidence["report"]["aggregate"]

    assert aggregate["macro_held_out_root_cause_f1"] >= 0.90
    assert aggregate["min_domain_root_cause_f1"] >= 0.85
    assert "GEN-003" not in evidence["report"]["blockers"]
    assert evidence["report"]["baseline_uplift"]["beats_strongest_baseline"] is True
    assert evidence["report"]["adversarial_domain_shift"]["passed"] is True


def test_huggingface_sources_supply_minimum_real_cases_and_unseen_family() -> None:
    runner = _runner_module()
    inventory = runner._load_inventory()
    domains = [domain["id"] for domain in inventory["domains"]]
    cases, sources = runner._load_huggingface_cases(inventory, rows_per_domain=100)

    case_count_by_domain = {
        domain_id: sum(1 for case in cases if case["domain_id"] == domain_id)
        for domain_id in domains
    }
    assert case_count_by_domain == {domain_id: 100 for domain_id in domains}
    assert all(case["source"]["row_hash"] for case in cases)
    assert all(case["root_truth"]["external_root_cause_text_hash"] for case in cases)

    sources_by_domain: dict[str, list[dict[str, object]]] = {
        domain_id: [] for domain_id in domains
    }
    for source in sources:
        sources_by_domain[str(source["domain_id"])].append(source)
        assert source["source_type"] == "huggingface"
        assert int(source["row_count"]) > 0
        assert int(source["available_rows"]) >= int(source["row_count"])
        assert source["dataset_family"]

    assert all(sources_by_domain.values())
    assert any(
        source.get("unseen_dataset_family") is True
        and source.get("used_for_feature_design") is False
        for source in sources
    )


def test_generalization_report_passes_unseen_dataset_transfer_from_raw_records() -> None:
    runner = _runner_module()
    inventory = runner._load_inventory()
    domains = [domain["id"] for domain in inventory["domains"]]
    cases, _sources = runner._load_huggingface_cases(inventory, rows_per_domain=100)

    evidence = runner._build_generalization_evidence(
        domain_ids=domains,
        cases=cases,
        bootstrap_iterations=1000,
    )

    report = evidence["report"]
    assert report["unseen_dataset_transfer"]["passed"] is True
    assert report["unseen_dataset_transfer"]["record_count"] >= 1
    assert "GEN-002" not in report["blockers"]
    assert "GEN-006" not in report["blockers"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for production transfer")
def test_production_transfer_runner_reports_transfer_scale_and_soak_contract(
    tmp_path: Path,
) -> None:
    output = tmp_path / "production_transfer.json"

    retries = 0
    while True:
        proc = subprocess.run(
            [
                "python",
                str(ROOT / "tools" / "run_production_transfer.py"),
                "--output",
                str(output),
                "--symbolic-facts",
                "10000",
                "--neural-observations",
                "1000",
                "--entities",
                "1000",
                "--staged-deltas",
                "100",
                "--soak-seconds",
                "0",
                "--latency-samples",
                "5",
                "--hf-rows-per-domain",
                "10",
                "--allow-development-profile",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=PRODUCTION_TRANSFER_TEST_TIMEOUT_SEC,
        )
        if proc.returncode == 0 or retries >= 1 or not _cuda_oom_text(proc.stdout, proc.stderr):
            break
        retries += 1
        time.sleep(1.0)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["status"] == "PASS"
    assert payload["scope"] == "development"
    assert payload["generalization_report"]["status"] == "FAIL"
    assert payload["generalization_report"]["claim_scope"] == (
        "partial leave-one-domain-out generalization evidence"
    )
    assert payload["generalization_report"]["excluded_domains"] == []
    assert {"GEN-002", "GEN-006"} <= set(payload["generalization_report"]["blockers"])
    generalization_records = payload["metric_inputs"]["generalization_prediction_records"]
    assert {record["held_out_domain"] for record in generalization_records} == set(
        payload["domain_ids"]
    )
    assert {
        record["evaluation_variant"] for record in generalization_records
    } >= {
        "clean",
        "noisy",
        "sparse",
        "paraphrased",
        "missing_field",
        "distractor_candidate",
    }
    assert all(
        record["candidate_generation"]["uses_heldout_test_truth"] is False
        for record in generalization_records
    )
    assert len(payload["huggingface_dataset_sources"]) == 5
    assert {source["source_type"] for source in payload["huggingface_dataset_sources"]} == {
        "huggingface"
    }
    assert {source["domain_id"] for source in payload["huggingface_dataset_sources"]} == set(
        payload["domain_ids"]
    )
    assert all(source["hf_dataset_id"] for source in payload["huggingface_dataset_sources"])
    source_by_domain = {
        source["domain_id"]: source for source in payload["huggingface_dataset_sources"]
    }
    manufacturing_source = source_by_domain["manufacturing_quality"]
    assert (
        manufacturing_source["hf_dataset_id"]
        == "Fujitsu/ManufacturingRCA_Knowledge_Dataset"
    )
    assert manufacturing_source["file"] == "ManufacturingRCA_doc.csv"
    assert manufacturing_source["root_truth_source_type"] == (
        "huggingface_manufacturing_rca_cause"
    )
    assert "Cause" in manufacturing_source["columns"]
    assert payload["domain_count"] == 5
    assert payload["held_out_domain"] == "cybersecurity_intrusion"
    assert payload["rule_evolution"]["held_out_domain_excluded"] is True
    assert payload["core_rule_edits_per_domain"] == 0
    assert payload["integrated_evaluator"]["uses_shared_bfo_kernel"] is True
    assert payload["integrated_evaluator"]["emits_per_domain_predictions"] is True
    assert payload["integrated_evaluator"]["consumes_neural_rankings"] is True
    assert payload["leakage_audit"]["passed"] is True
    assert payload["leakage_audit"]["metadata_gold_markers"] == []
    assert payload["leakage_audit"]["binary_feature_gold_columns"] == []
    assert payload["leakage_audit"]["candidate_order_index_leaks"] is False
    assert payload["leakage_audit"]["xlog_fact_symmetry"] is True
    neural_invocation = payload["integrated_evaluator"].get("neural_invocation", {})
    assert neural_invocation.get("path") == "xlog_nn4_transfer"
    assert neural_invocation.get("program_declares_nn4") is True
    assert neural_invocation.get("transfer_forward_backward_loss_is_cuda") is True
    assert neural_invocation.get("ranking_argmax_device_resident") is True
    assert neural_invocation.get("score_cpu_materialization_in_ranking") is False
    assert neural_invocation.get("full_score_rows_materialized") is False
    assert neural_invocation.get("scalar_item_calls_in_ranking") is False
    assert neural_invocation.get("cpu_score_slices_in_ranking") is False
    assert neural_invocation.get("post_ranking_evidence_serialization") == (
        "selected_indices_only"
    )
    assert all(
        record["neural_scores"]["materialized"] is False
        for record in payload["metric_inputs"]["prediction_records"]
    )
    bundle_reuse = payload["bundle_reuse"]
    assert bundle_reuse["status"] == "PASS"
    assert bundle_reuse["v080_runtime_session"]["status"] == "PASS"
    assert bundle_reuse["v080_runtime_session"]["logic_program_compile"] is True
    assert bundle_reuse["v080_runtime_session"]["session_evaluate"] is True
    assert bundle_reuse["v080_runtime_session"]["relation_delta_equivalence_pct"] == 100.0
    assert bundle_reuse["v085_language_contract"]["status"] == "PASS"
    assert bundle_reuse["v085_language_contract"]["feature_count"] >= 10
    assert "examples/v085-language/showcase" in bundle_reuse["v085_language_contract"][
        "reused_artifacts"
    ]
    v086 = bundle_reuse["v086_runtime_optimizer"]
    assert v086["status"] == "PASS"
    assert v086["apply_relation_delta_batch"] is True
    assert v086["join_index_cache_stats"]["builds"] >= 1
    assert v086["join_index_cache_stats"]["hits"] >= 1
    assert v086["relation_callback_events"] >= 2
    assert v086["callback_payload_has_tensors"] is False
    assert v086["hot_loop_transfer_stats"] == {
        "dtoh_calls": 0,
        "htod_calls": 0,
        "dtoh_bytes": 0,
        "htod_bytes": 0,
    }
    records = payload["metric_inputs"]["prediction_records"]
    assert records
    assert {record["domain_id"] for record in records} == set(payload["domain_ids"])
    assert all(record["source"]["source_type"] == "huggingface" for record in records)
    assert all(
        record.get("root_label_source") == "huggingface_external_rca" for record in records
    )
    assert all(
        record.get("root_truth", {}).get("source_type", "").startswith("huggingface_")
        for record in records
    )
    assert all(
        record.get("root_truth", {}).get("field_name") for record in records
    )
    assert all(
        record.get("root_truth", {}).get("external_root_cause_text_hash") for record in records
    )
    assert all(
        record.get("root_truth", {}).get("ordinary_label_mapping") is False
        for record in records
    )
    assert all(
        record.get("intervention_truth", {}).get("source_type") for record in records
    )
    assert all(
        record.get("candidate_generation", {}).get("label_injected") is False
        for record in records
    )
    assert all(
        record.get("candidate_generation", {}).get("candidate_count", 0) >= 4
        for record in records
    )
    assert all(record["xlog_candidate_count"] >= 4 for record in records)
    for record in records:
        explanations = record.get("bfo_explanations")
        assert isinstance(explanations, list) and explanations
        explanation_types = {entry.get("claim_type") for entry in explanations}
        assert {"root_cause", "intervention", "risk_state"} <= explanation_types
        assert all(entry.get("kernel_rule") for entry in explanations)
        assert all(entry.get("bfo_category") for entry in explanations)
        assert all(entry.get("valid") is True for entry in explanations)
        assert any(
            entry["claim_type"] == "root_cause"
            and entry["claim"] == record["root_prediction"]
            and entry["bfo_category"] == "quality"
            for entry in explanations
        )
        assert any(
            entry["claim_type"] == "intervention"
            and entry["claim"] == record["intervention_prediction"]
            for entry in explanations
        )
    held_out = [
        record for record in records if record["domain_id"] == payload["held_out_domain"]
    ]
    assert held_out
    held_out_correct = sum(
        1 for record in held_out if record["root_prediction"] == record["root_label"]
    )
    assert payload["computed_metrics"]["held_out_root_cause_confusion"] == {
        "correct": held_out_correct,
        "gold": len(held_out),
        "predicted": len(held_out),
        "total": len(held_out),
    }
    assert payload["computed_metrics"]["held_out_root_cause_f1"] == pytest.approx(
        held_out_correct / len(held_out)
    )
    assert payload["held_out_root_cause_f1"] >= 0.90
    assert payload["accepted_intervention_precision"] >= 0.95
    assert payload["explanations_complete_pct"] == 100.0
    assert payload["computed_metrics"]["promoted_rule_quality"]["precision"] >= 0.98
    assert payload["computed_metrics"]["promoted_rule_quality"]["recall"] >= 0.95
    assert payload["computed_metrics"]["promoted_rule_quality"]["f1"] >= 0.965
    public_benchmark = payload["public_benchmark_report"]
    assert public_benchmark["status"] == "FAIL"
    assert public_benchmark["external_sota_claim"] is False
    assert public_benchmark["covered_public_benchmark_families"] == []
    assert public_benchmark["missing_public_benchmark_families"]
    assert "MISSING_PUBLIC_SOTA_RUNNER" in public_benchmark["blockers"]
    assert "baseline_metrics" not in payload
    assert "strongest_baseline" not in payload
    assert "relative_uplift_over_best_baseline_pct" not in payload
    assert "baseline_metrics" not in payload["computed_metrics"]
    ablations = payload["metric_inputs"]["ablation_records"]
    assert ablations
    recomputed = {}
    for method in ["neural_only", "domain_symbolic", "shared_symbolic", "neuro_symbolic"]:
        recomputed[method] = sum(_score(record[method]) for record in ablations) / len(ablations)
    showcase_metrics = payload["showcase_metrics"]
    assert showcase_metrics["baseline_metrics"] == pytest.approx(recomputed)
    assert payload["computed_metrics"]["showcase_metrics"]["baseline_metrics"] == pytest.approx(
        recomputed
    )
    assert showcase_metrics["ablation_scoring"]["primary_metric"] == "root_cause_accuracy"
    assert showcase_metrics["relative_uplift_over_best_baseline_pct"] == pytest.approx(
        payload["computed_metrics"]["showcase_metrics"][
            "relative_uplift_over_best_baseline_pct"
        ]
    )
    assert showcase_metrics["strongest_baseline_value"] <= 1.0
    assert set(showcase_metrics["baseline_metrics"]) == {
        "neural_only",
        "domain_symbolic",
        "shared_symbolic",
        "neuro_symbolic",
    }
    assert payload["neural"]["program_declares_nn4"] is True
    assert payload["neural"]["loss_is_cuda"] is True
    assert payload["neural"]["hand_weighted"] is False
    assert payload["neural"]["trained_on_held_out_domain"] is False
    assert payload["neural"]["processed_observation_count"] == 1000
    assert payload["scale_profile"]["symbolic_bfo_fact_count"] >= 10000
    assert payload["scale_profile"]["scale_source"] == "hf_case_amplification"
    assert payload["scale_profile"]["synthetic_numeric_only"] is False
    assert payload["scale_profile"]["real_hf_transfer_case_count"] >= len(records)
    assert payload["scale_profile"]["neural_observation_count"] == 1000
    assert payload["scale_profile"]["entity_count"] >= 1000
    assert payload["scale_profile"]["staged_delta_update_count"] == 100
    assert payload["scale_profile"]["query_row_counts"] == {
        "target_candidate_root_cause": 4,
        "target_recommended_intervention": 4,
        "target_bfo_explanation": 4,
    }
    assert payload["scale_profile"]["p95_core_indexed_query_latency_ms"] <= 50.0
    assert payload["soak"]["duration_sec"] == 0.0
