from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _cuda_oom_text(stdout: str, stderr: str) -> bool:
    text = f"{stdout}\n{stderr}".lower()
    return "cuda_error_out_of_memory" in text or "out of memory" in text


def _validator_module():
    spec = importlib.util.spec_from_file_location(
        "validate_universal_case_reasoner",
        ROOT / "tools" / "validate_universal_case_reasoner.py",
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _production_evidence_fixture(path: Path) -> None:
    payload = {
        "status": "PASS",
        "scope": "production",
        "domain_count": 5,
        "held_out_domain": "cybersecurity_intrusion",
        "held_out_root_cause_f1": 1.0,
        "accepted_intervention_precision": 1.0,
        "explanations_complete_pct": 100.0,
        "relative_uplift_over_best_baseline_pct": 42.857143,
        "strongest_baseline": "neural_only",
        "invalid_cross_domain_rejection_pct": 100.0,
        "core_rule_edits_per_domain": 0,
        "rule_evolution": {"held_out_domain_excluded": True},
        "kernel_checksum_by_domain": {
            "clinical_deterioration": "same",
            "manufacturing_quality": "same",
            "cybersecurity_intrusion": "same",
            "lab_operations_incident": "same",
            "cloud_operations_rca": "same",
        },
        "adapter_fact_only_by_domain": {
            "clinical_deterioration": True,
            "manufacturing_quality": True,
            "cybersecurity_intrusion": True,
            "lab_operations_incident": True,
            "cloud_operations_rca": True,
        },
        "neural": {
            "program_declares_nn4": True,
            "loss_is_cuda": True,
            "gradient_finite": True,
            "processed_observation_count": 100_000,
            "ranking_accuracy": 1.0,
        },
        "baseline_metrics": {
            "neural_only": 0.70,
            "domain_symbolic": 0.0,
            "shared_symbolic": 0.50,
            "neuro_symbolic": 1.0,
        },
        "promoted_rule_quality": {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "kernel_mutated": False,
        },
        "scale_profile": {
            "symbolic_bfo_fact_count": 1_000_000,
            "neural_observation_count": 100_000,
            "entity_count": 50_000,
            "staged_delta_update_count": 10_000,
            "p95_core_indexed_query_latency_ms": 12.5,
            "control_plane_metadata_bytes_per_hot_iteration": 1024,
        },
        "soak": {
            "duration_sec": 1800.0,
            "gpu_memory_drift_pct": 0.0,
            "relation_growth_bounded": True,
        },
        "evidence": "production transfer fixture for validator contract test",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _computed_production_evidence_fixture(path: Path) -> None:
    domains = [
        "clinical_deterioration",
        "manufacturing_quality",
        "cybersecurity_intrusion",
        "lab_operations_incident",
        "cloud_operations_rca",
    ]
    records = []
    for domain in domains:
        records.append(
            {
                "case_id": f"{domain}:0",
                "domain_id": domain,
                "source": {
                    "source_type": "huggingface",
                    "hf_dataset_id": f"fixture/{domain}",
                    "split": "train",
                    "row_index": 0,
                    "row_hash": f"hash-{domain}",
                },
                "root_label": f"{domain}_root",
                "root_prediction": f"{domain}_root",
                "intervention_label": f"{domain}_intervention",
                "intervention_prediction": f"{domain}_intervention",
                "explanation_valid": True,
                "xlog_candidate_count": 2,
                "xlog_intervention_count": 2,
                "xlog_explanation_count": 2,
            }
        )
    held_out_records = [
        record for record in records if record["domain_id"] == "cybersecurity_intrusion"
    ]
    ablations = []
    for record in held_out_records:
        ablations.append(
            {
                "case_id": record["case_id"],
                "domain_id": record["domain_id"],
                "neural_only": {
                    "root_label": record["root_label"],
                    "root_prediction": record["root_label"],
                    "intervention_label": record["intervention_label"],
                    "intervention_prediction": None,
                    "explanation_valid": False,
                },
                "domain_symbolic": {
                    "root_label": record["root_label"],
                    "root_prediction": None,
                    "intervention_label": record["intervention_label"],
                    "intervention_prediction": None,
                    "explanation_valid": False,
                },
                "shared_symbolic": {
                    "root_label": record["root_label"],
                    "root_prediction": "wrong_root",
                    "intervention_label": record["intervention_label"],
                    "intervention_prediction": "wrong_intervention",
                    "explanation_valid": True,
                },
                "neuro_symbolic": {
                    "root_label": record["root_label"],
                    "root_prediction": record["root_label"],
                    "intervention_label": record["intervention_label"],
                    "intervention_prediction": record["intervention_label"],
                    "explanation_valid": True,
                },
            }
        )
    payload = {
        "status": "PASS",
        "scope": "production",
        "domain_count": 5,
        "domain_ids": domains,
        "held_out_domain": "cybersecurity_intrusion",
        "core_rule_edits_per_domain": 0,
        "rule_evolution": {"held_out_domain_excluded": True},
        "kernel_checksum_by_domain": {domain: "same" for domain in domains},
        "adapter_fact_only_by_domain": {domain: True for domain in domains},
        "huggingface_dataset_sources": [
            {
                "source_type": "huggingface",
                "domain_id": domain,
                "hf_dataset_id": f"fixture/{domain}",
                "split": "train",
                "row_count": 1,
            }
            for domain in domains
        ],
        "integrated_evaluator": {
            "uses_shared_bfo_kernel": True,
            "emits_per_domain_predictions": True,
            "consumes_neural_rankings": True,
            "query_row_counts": {
                "candidate_root_cause": 10,
                "recommended_intervention": 10,
                "bfo_explanation": 10,
            },
        },
        "leakage_audit": {
            "passed": True,
            "held_out_domain": "cybersecurity_intrusion",
            "held_out_case_count": len(held_out_records),
            "metadata_gold_markers": [],
            "binary_feature_gold_columns": [],
            "candidate_order_index_leaks": False,
            "true_candidate_index_count": 2,
            "xlog_fact_symmetry": True,
        },
        "bundle_reuse": {
            "status": "PASS",
            "v080_runtime_session": {
                "status": "PASS",
                "logic_program_compile": True,
                "session_evaluate": True,
                "relation_delta_equivalence_pct": 100.0,
                "hot_loop_transfer_stats": {
                    "dtoh_calls": 0,
                    "htod_calls": 0,
                    "dtoh_bytes": 0,
                    "htod_bytes": 0,
                },
            },
            "v085_language_contract": {
                "status": "PASS",
                "feature_count": 15,
                "reused_artifacts": [
                    "scripts/validate_v085_examples.py",
                    "examples/v085-language/showcase",
                ],
            },
            "v086_runtime_optimizer": {
                "status": "PASS",
                "apply_relation_delta_batch": True,
                "join_index_cache_stats": {
                    "builds": 1,
                    "hits": 1,
                    "entries": 1,
                    "stale_rejections": 0,
                },
                "relation_callback_events": 2,
                "callback_payload_has_tensors": False,
                "hot_loop_transfer_stats": {
                    "dtoh_calls": 0,
                    "htod_calls": 0,
                    "dtoh_bytes": 0,
                    "htod_bytes": 0,
                },
            },
        },
        "neural": {
            "program_declares_nn4": True,
            "loss_is_cuda": True,
            "gradient_finite": True,
            "processed_observation_count": 100_000,
            "ranking_accuracy": 1.0,
        },
        "metric_inputs": {
            "prediction_records": records,
            "ablation_records": ablations,
            "invalid_cross_domain_records": [
                {"fixture_id": f"invalid-{domain}", "rejected": True} for domain in domains
            ],
        },
        "scale_profile": {
            "symbolic_bfo_fact_count": 1_000_000,
            "neural_observation_count": 100_000,
            "entity_count": 50_000,
            "staged_delta_update_count": 10_000,
            "p95_core_indexed_query_latency_ms": 12.5,
            "control_plane_metadata_bytes_per_hot_iteration": 1024,
        },
        "soak": {
            "duration_sec": 1800.0,
            "gpu_memory_drift_pct": 0.0,
            "relation_growth_bounded": True,
        },
        "evidence": "computed production fixture for validator contract test",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verified_production_evidence_fixture(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "run_production_transfer",
        ROOT / "tools" / "run_production_transfer.py",
    )
    assert spec is not None
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)

    inventory = runner._load_inventory()
    cases, sources = runner._load_huggingface_cases(inventory, rows_per_domain=10)
    domains = [domain["id"] for domain in inventory["domains"]]
    records = []
    for case in cases:
        candidate_count = len(case["candidates"])
        selected = next(
            candidate for candidate in case["candidates"] if candidate["root"] == case["root_label"]
        )
        records.append(
            {
                "case_id": case["case_id"],
                "domain_id": case["domain_id"],
                "source": case["source"],
                "root_label_source": case["root_label_source"],
                "root_truth": case["root_truth"],
                "intervention_truth": case["intervention_truth"],
                "candidate_generation": case["candidate_generation"],
                "root_label": case["root_label"],
                "root_prediction": case["root_label"],
                "intervention_label": case["intervention_label"],
                "intervention_prediction": case["intervention_label"],
                "explanation_valid": True,
                "risk_state": case["risk_state"],
                "bfo_explanations": runner._bfo_explanations(case, selected),
                "xlog_candidate_count": candidate_count,
                "xlog_intervention_count": candidate_count,
                "xlog_explanation_count": candidate_count,
                "candidate_roots": [candidate["root"] for candidate in case["candidates"]],
                "neural_scores": {
                    "materialized": False,
                    "reason": "full CUDA score rows are not copied to host",
                },
            }
        )
    held_out_records = [
        record for record in records if record["domain_id"] == "cybersecurity_intrusion"
    ]
    ablations = []
    for index, record in enumerate(held_out_records):
        neural_root = record["root_label"] if index % 2 == 0 else "wrong_root"
        ablations.append(
            {
                "case_id": record["case_id"],
                "domain_id": record["domain_id"],
                "neural_only": {
                    "root_label": record["root_label"],
                    "root_prediction": neural_root,
                    "intervention_label": record["intervention_label"],
                    "intervention_prediction": record["intervention_label"],
                    "explanation_valid": True,
                },
                "domain_symbolic": {
                    "root_label": record["root_label"],
                    "root_prediction": None,
                    "intervention_label": record["intervention_label"],
                    "intervention_prediction": None,
                    "explanation_valid": False,
                },
                "shared_symbolic": {
                    "root_label": record["root_label"],
                    "root_prediction": "wrong_root",
                    "intervention_label": record["intervention_label"],
                    "intervention_prediction": "wrong_intervention",
                    "explanation_valid": True,
                },
                "neuro_symbolic": {
                    "root_label": record["root_label"],
                    "root_prediction": record["root_label"],
                    "intervention_label": record["intervention_label"],
                    "intervention_prediction": record["intervention_label"],
                    "explanation_valid": True,
                },
            }
        )
    payload = {
        "status": "PASS",
        "scope": "production",
        "domain_count": 5,
        "domain_ids": domains,
        "held_out_domain": "cybersecurity_intrusion",
        "core_rule_edits_per_domain": 0,
        "rule_evolution": {
            "held_out_domain_excluded": True,
            "rule_evolution_domains": [
                domain for domain in domains if domain != "cybersecurity_intrusion"
            ],
        },
        "kernel_checksum_by_domain": {domain: "same" for domain in domains},
        "adapter_fact_only_by_domain": {domain: True for domain in domains},
        "huggingface_dataset_sources": sources,
        "integrated_evaluator": {
            "uses_shared_bfo_kernel": True,
            "emits_per_domain_predictions": True,
            "consumes_neural_rankings": True,
            "query_row_counts": {
                "candidate_root_cause": sum(record["xlog_candidate_count"] for record in records),
                "recommended_intervention": sum(
                    record["xlog_intervention_count"] for record in records
                ),
                "bfo_explanation": sum(record["xlog_explanation_count"] for record in records),
            },
            "neural_invocation": {
                "path": "xlog_nn4_transfer",
                "program_declares_nn4": True,
                "transfer_forward_backward_loss_is_cuda": True,
                "transfer_nn4_gradient_finite": True,
                "ranking_argmax_device_resident": True,
                "score_cpu_materialization_in_ranking": False,
                "full_score_rows_materialized": False,
                "scalar_item_calls_in_ranking": False,
                "cpu_score_slices_in_ranking": False,
                "post_ranking_evidence_serialization": "selected_indices_only",
                "nn4_query_count": sum(record["xlog_candidate_count"] for record in records),
            },
        },
        "leakage_audit": runner._candidate_leakage_audit(
            cases,
            inventory["holdout_protocol"]["held_out_domain"],
        ),
        "bundle_reuse": {
            "status": "PASS",
            "v080_runtime_session": {
                "status": "PASS",
                "logic_program_compile": True,
                "session_evaluate": True,
                "relation_delta_equivalence_pct": 100.0,
                "hot_loop_transfer_stats": {
                    "dtoh_calls": 0,
                    "htod_calls": 0,
                    "dtoh_bytes": 0,
                    "htod_bytes": 0,
                },
            },
            "v085_language_contract": {
                "status": "PASS",
                "feature_count": 15,
                "reused_artifacts": [
                    "scripts/validate_v085_examples.py",
                    "examples/v085-language/showcase",
                ],
            },
            "v086_runtime_optimizer": {
                "status": "PASS",
                "apply_relation_delta_batch": True,
                "join_index_cache_stats": {
                    "builds": 1,
                    "hits": 1,
                    "entries": 1,
                    "stale_rejections": 0,
                },
                "relation_callback_events": 2,
                "callback_payload_has_tensors": False,
                "hot_loop_transfer_stats": {
                    "dtoh_calls": 0,
                    "htod_calls": 0,
                    "dtoh_bytes": 0,
                    "htod_bytes": 0,
                },
            },
        },
        "neural": {
            "program_declares_nn4": True,
            "loss_is_cuda": True,
            "gradient_finite": True,
            "processed_observation_count": 100_000,
            "ranking_accuracy": 1.0,
            "hand_weighted": False,
            "trained_on_held_out_domain": False,
        },
        "ablation_scoring": {
            "primary_metric": "root_cause_accuracy",
            "intervention_precision_reported_separately": True,
            "explanation_coverage_reported_separately": True,
        },
        "metric_inputs": {
            "prediction_records": records,
            "ablation_records": ablations,
            "invalid_cross_domain_records": [
                {"fixture_id": f"invalid-{domain}", "rejected": True} for domain in domains
            ],
        },
        "scale_profile": {
            "scale_source": "hf_case_amplification",
            "synthetic_numeric_only": False,
            "hf_seed_case_count": 5,
            "real_hf_transfer_case_count": 100_000,
            "symbolic_bfo_fact_count": 1_000_000,
            "neural_observation_count": 100_000,
            "entity_count": 50_000,
            "staged_delta_update_count": 10_000,
            "p95_core_indexed_query_latency_ms": 12.5,
            "control_plane_metadata_bytes_per_hot_iteration": 1024,
        },
        "soak": {
            "duration_sec": 1800.0,
            "gpu_memory_drift_pct": 0.0,
            "relation_growth_bounded": True,
        },
        "evidence": "verified production fixture for validator contract test",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_strict_validator_fails_closed_and_writes_summary(tmp_path: Path) -> None:
    output = tmp_path / "validation_summary.json"
    missing_production = tmp_path / "missing_production_transfer.json"

    proc = subprocess.run(
        [
            str(ROOT / "validate.sh"),
            "--strict",
            "--gpu-required",
            "--production-transfer",
            str(missing_production),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )

    assert proc.returncode == 1
    assert output.exists()

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["status"] == "FAIL"
    assert summary["strict"] is True
    assert summary["gpu_required"] is True
    assert summary["branch"] == "feat/bfo-universal-case-reasoner"
    assert summary["commands"][0]["argv"][:3] == ["validate.sh", "--strict", "--gpu-required"]
    assert len(summary["gqm_metrics"]) == 12
    assert {entry["id"] for entry in summary["gqm_metrics"]} == {f"Q{i}" for i in range(1, 13)}
    assert len(summary["p0_gates"]) >= 8
    assert all(gate["status"] in {"PASS", "FAIL"} for gate in summary["p0_gates"])
    assert any(gate["status"] == "FAIL" for gate in summary["p0_gates"])
    assert summary["blockers"]
    assert all("requirement_id" in blocker for blocker in summary["blockers"])
    assert all("evidence" in blocker for blocker in summary["blockers"])


def test_validation_plan_covers_all_p0_requirement_sections() -> None:
    plan = (ROOT / "VALIDATION_PLAN.md").read_text(encoding="utf-8")

    for section in [
        "P0 Hard Gates",
        "BFO Transfer Conformance",
        "Domain Coverage",
        "Neural Requirements",
        "Transfer And Evolution",
        "Robust Generalization",
        "Device-Resident Execution",
        "Bundle Reuse",
        "Scale And Performance",
        "Evidence Schema",
    ]:
        assert section in plan

    for artifact in [
        "bfo/kernel.xlog",
        "domains/domain_inventory.json",
        "validation_summary.json",
        "./validate.sh --strict --gpu-required",
    ]:
        assert artifact in plan


def test_all_shipped_xlog_programs_run_through_cli() -> None:
    programs = sorted((ROOT / "programs").glob("*.xlog"))
    assert programs

    failures = []
    for program in programs:
        retries = 0
        while True:
            proc = subprocess.run(
                [
                    "cargo",
                    "run",
                    "-q",
                    "-p",
                    "xlog-cli",
                    "--",
                    "run",
                    str(program.relative_to(ROOT.parents[2])),
                ],
                cwd=ROOT.parents[2],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
            )
            if proc.returncode == 0 or retries >= 1 or not _cuda_oom_text(proc.stdout, proc.stderr):
                break
            retries += 1
            time.sleep(1.0)
        if proc.returncode != 0:
            failures.append(
                {
                    "program": str(program.relative_to(ROOT.parents[2])),
                    "stderr": proc.stderr,
                    "stdout": proc.stdout,
                }
            )
    assert failures == []


def test_validator_rejects_shallow_generated_transfer_fixture(tmp_path: Path) -> None:
    production = tmp_path / "computed_but_shallow_production_transfer.json"
    _computed_production_evidence_fixture(production)
    validator = _validator_module()
    payload = validator._load_production_transfer_evidence(production)

    assert validator._source_provenance_passed(payload) is False
    assert validator._integrated_evaluator_passed(payload) is False
    assert validator._production_scale_passed(payload) is False
    assert validator._production_transfer_passed(payload) is False


def test_validator_rejects_ordinary_hf_label_to_root_mapping(tmp_path: Path) -> None:
    production = tmp_path / "ordinary_label_mapping_transfer.json"
    _computed_production_evidence_fixture(production)
    payload = json.loads(production.read_text(encoding="utf-8"))
    for record in payload["metric_inputs"]["prediction_records"]:
        record["root_label_source"] = "huggingface_field"
        record["label_source"] = {
            "source_type": "huggingface_field",
            "field_name": "ordinary_class_label",
            "field_value_hash": "deadbeef",
            "positive_values": ["1"],
            "positive_root": record["root_label"],
            "negative_root": "alternate_root",
        }
    production.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validator = _validator_module()

    assert validator._source_provenance_passed(
        validator._load_production_transfer_evidence(production)
    ) is False


def test_validator_rejects_candidate_leakage_audit_failure(tmp_path: Path) -> None:
    production = tmp_path / "production_transfer_with_leakage.json"
    _verified_production_evidence_fixture(production)
    payload = json.loads(production.read_text(encoding="utf-8"))
    payload["leakage_audit"] = {
        "passed": False,
        "held_out_domain": "cybersecurity_intrusion",
        "held_out_case_count": 10,
        "metadata_gold_markers": ["cybersecurity_intrusion:hf:0:0:has_bfo_evidence"],
        "binary_feature_gold_columns": [4],
        "candidate_order_index_leaks": False,
        "true_candidate_index_count": 10,
        "xlog_fact_symmetry": True,
    }
    production.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validator = _validator_module()
    loaded = validator._load_production_transfer_evidence(production)

    assert validator._leakage_audit_passed(loaded) is False
    assert validator._production_transfer_passed(loaded) is False


def test_validator_requires_unseen_dataset_transfer_in_raw_records() -> None:
    validator = _validator_module()
    domains = [
        "clinical_deterioration",
        "manufacturing_quality",
        "cybersecurity_intrusion",
        "lab_operations_incident",
        "cloud_operations_rca",
    ]
    records = []
    for domain in domains:
        for index in range(100):
            records.append(
                {
                    "case_id": f"{domain}:fixture:{index}",
                    "domain_id": domain,
                    "held_out_domain": domain,
                    "evaluation_variant": "clean",
                    "source": {
                        "source_type": "huggingface",
                        "hf_dataset_id": f"fixture/{domain}",
                        "file": "fixture.csv",
                        "row_hash": "rowhash",
                        "dataset_family": "feature_design_family",
                        "used_for_feature_design": True,
                        "unseen_dataset_family": False,
                    },
                    "root_label": f"{domain}_root",
                    "root_prediction": f"{domain}_root",
                    "candidate_generation": {
                        "label_injected": False,
                        "uses_heldout_test_truth": False,
                        "constructed_before_heldout_labels": True,
                    },
                    "root_truth": {
                        "external_root_cause_text_hash": "roothash",
                    },
                    "intervention_truth": {
                        "external_intervention_text_hash": "interventionhash",
                    },
                }
            )

    assessment = validator._generalization_assessment(
        {
            "domain_ids": domains,
            "generalization_report": {
                "aggregate": {
                    "macro_held_out_root_cause_f1": 1.0,
                    "min_domain_root_cause_f1": 1.0,
                },
                "excluded_domains": [],
                "baseline_methods": [
                    "neural_only",
                    "symbolic_only",
                    "domain_specific_classifier",
                    "retrieval_rag_nearest_neighbor",
                    "majority_prior",
                    "neuro_symbolic",
                ],
                "frozen_model_rules": {
                    "passed": True,
                    "bfo_kernel": True,
                    "learned_rules": True,
                    "neural_architecture": True,
                    "thresholds": True,
                    "aliases": True,
                    "scoring_weights": True,
                    "generalization_seed_isolated_from_showcase_transfer": True,
                },
                "unseen_dataset_transfer": {
                    "passed": True,
                    "held_out_domain": "clinical_deterioration",
                    "dataset_family": "unseen_fixture_family",
                },
                "statistical_confidence": {
                    "passed": True,
                    "bootstrap_iterations": 1000,
                    "bootstrap_ci_by_domain": {
                        domain: {"low": 1.0, "high": 1.0} for domain in domains
                    },
                    "paired_significance_tests": [{"baseline": "majority_prior"}],
                },
                "adversarial_domain_shift": {
                    "passed": True,
                    "variants": [
                        "noisy",
                        "sparse",
                        "paraphrased",
                        "missing_field",
                        "distractor_candidate",
                    ],
                },
            },
            "metric_inputs": {
                "generalization_prediction_records": records,
                "generalization_ablation_records": [
                    {
                        "neural_only": {},
                        "symbolic_only": {},
                        "domain_specific_classifier": {},
                        "retrieval_rag_nearest_neighbor": {},
                        "majority_prior": {},
                        "neuro_symbolic": {},
                    }
                ],
            },
        }
    )

    assert assessment["gates"]["GEN-006"]["passed"] is False


def test_validator_requires_dilp_rule_induction_evidence() -> None:
    validator = _validator_module()

    assessment = validator._dilp_assessment({})

    assert assessment["status"] == "FAIL"
    assert assessment["gates"]["DILP-001"]["passed"] is False
    assert assessment["gates"]["DILP-002"]["passed"] is False
    assert assessment["gates"]["DILP-003"]["passed"] is False
    assert assessment["gates"]["DILP-004"]["passed"] is False
    assert assessment["gates"]["DILP-005"]["passed"] is False
    assert assessment["gates"]["DILP-006"]["passed"] is False


def _generalization_record(
    *,
    domain: str,
    index: int,
    label: str,
    prediction: str,
    variant: str = "clean",
) -> dict[str, object]:
    return {
        "case_id": f"{domain}:fixture:{variant}:{index}",
        "domain_id": domain,
        "held_out_domain": domain,
        "evaluation_variant": variant,
        "source": {
            "source_type": "huggingface",
            "hf_dataset_id": f"fixture/{domain}",
            "file": "fixture.csv",
            "row_hash": f"rowhash-{domain}-{index}",
            "dataset_family": (
                "unseen_fixture_family"
                if domain == "clinical_deterioration"
                else "feature_design_family"
            ),
            "used_for_feature_design": domain != "clinical_deterioration",
            "unseen_dataset_family": domain == "clinical_deterioration",
        },
        "root_label": label,
        "root_prediction": prediction,
        "ranker_path": "xlog_nn4_cuda_generalization",
        "neural_scores": {
            "materialized": False,
            "reason": "full CUDA score rows are not copied to host",
        },
        "candidate_generation": {
            "label_injected": False,
            "uses_heldout_test_truth": False,
            "constructed_before_heldout_labels": True,
        },
        "root_truth": {
            "external_root_cause_text_hash": f"roothash-{domain}-{index}",
        },
        "intervention_truth": {
            "external_intervention_text_hash": f"interventionhash-{domain}-{index}",
        },
    }


def _generalization_report_fixture(
    *,
    domains: list[str],
    records: list[dict[str, object]],
    ablation_records: list[dict[str, object]],
    macro_f1: float = 1.0,
    min_f1: float = 1.0,
) -> dict[str, object]:
    return {
        "domain_ids": domains,
        "generalization_report": {
            "aggregate": {
                "macro_held_out_root_cause_f1": macro_f1,
                "min_domain_root_cause_f1": min_f1,
            },
            "excluded_domains": [],
            "baseline_methods": [
                "neural_only",
                "symbolic_only",
                "domain_specific_classifier",
                "retrieval_rag_nearest_neighbor",
                "majority_prior",
                "neuro_symbolic",
            ],
            "baseline_uplift": {
                "beats_strongest_baseline": True,
                "relative_uplift_over_best_baseline_pct": 25.0,
            },
            "frozen_model_rules": {
                "passed": True,
                "bfo_kernel": True,
                "learned_rules": True,
                "neural_architecture": True,
                "thresholds": True,
                "aliases": True,
                "scoring_weights": True,
                "generalization_seed_isolated_from_showcase_transfer": True,
            },
            "neural_ranker": {
                "path": "xlog_nn4_cuda_generalization",
                "program": "programs/production_ranker.xlog",
                "registered_network": "production_root_net",
                "selection_device": "cuda",
                "uses_python_heuristic": False,
                "heldout_labels_used_in_nn4": False,
                "score_cpu_materialization_in_ranking": False,
                "full_score_rows_materialized": False,
                "scalar_item_calls_in_ranking": False,
                "post_ranking_evidence_serialization": "selected_indices_only",
                "heldout_scoring": {
                    "path": "xlog_nn4_forward_backward_tensor",
                    "program": "programs/production_ranker.xlog",
                    "expected_label": "primary_root",
                    "uses_heldout_labels": False,
                    "loss_tensors_device": "cuda",
                    "score_tensor_device": "cuda",
                    "score_cpu_materialization_in_ranking": False,
                    "query_count": 10_000,
                },
                "nn4_query_count": 10,
            },
            "unseen_dataset_transfer": {
                "passed": True,
                "held_out_domain": "clinical_deterioration",
                "dataset_family": "unseen_fixture_family",
            },
            "statistical_confidence": {
                "passed": True,
                "bootstrap_iterations": 1000,
                "bootstrap_ci_by_domain": {
                    domain: {"low": 1.0, "high": 1.0} for domain in domains
                },
                "paired_significance_tests": [{"baseline": "majority_prior"}],
            },
            "adversarial_domain_shift": {
                "passed": True,
                "variants": [
                    "noisy",
                    "sparse",
                    "paraphrased",
                    "missing_field",
                    "distractor_candidate",
                ],
                "macro_f1_by_variant": {
                    "clean": 1.0,
                    "noisy": 1.0,
                    "sparse": 1.0,
                    "paraphrased": 1.0,
                    "missing_field": 1.0,
                    "distractor_candidate": 1.0,
                },
            },
        },
        "metric_inputs": {
            "generalization_prediction_records": records,
            "generalization_ablation_records": ablation_records,
        },
    }


def test_validator_uses_multiclass_macro_f1_not_accuracy_for_gen003() -> None:
    validator = _validator_module()
    domains = [
        "clinical_deterioration",
        "manufacturing_quality",
        "cybersecurity_intrusion",
        "lab_operations_incident",
        "cloud_operations_rca",
    ]
    records = []
    ablation_records = []
    for domain in domains:
        for index in range(100):
            is_majority = index < 90
            label = f"{domain}_majority" if is_majority else f"{domain}_minority"
            prediction = f"{domain}_majority"
            records.append(
                _generalization_record(
                    domain=domain,
                    index=index,
                    label=label,
                    prediction=prediction,
                )
            )
            ablation_records.append(
                {
                    method: {
                        "root_label": label,
                        "root_prediction": prediction,
                    }
                    for method in [
                        "neural_only",
                        "symbolic_only",
                        "domain_specific_classifier",
                        "retrieval_rag_nearest_neighbor",
                        "majority_prior",
                        "neuro_symbolic",
                    ]
                }
            )

    assessment = validator._generalization_assessment(
        _generalization_report_fixture(
            domains=domains,
            records=records,
            ablation_records=ablation_records,
            macro_f1=0.90,
            min_f1=0.90,
        )
    )

    assert assessment["computed"]["macro_held_out_root_cause_f1"] < 0.90
    assert assessment["gates"]["GEN-003"]["passed"] is False


def test_validator_requires_generalization_uplift_over_strongest_baseline() -> None:
    validator = _validator_module()
    domains = [
        "clinical_deterioration",
        "manufacturing_quality",
        "cybersecurity_intrusion",
        "lab_operations_incident",
        "cloud_operations_rca",
    ]
    records = [
        _generalization_record(
            domain=domain,
            index=index,
            label=f"{domain}_root",
            prediction=f"{domain}_root",
        )
        for domain in domains
        for index in range(100)
    ]
    ablation_records = []
    for record in records:
        label = str(record["root_label"])
        ablation_records.append(
            {
                "neuro_symbolic": {"root_label": label, "root_prediction": "wrong_root"},
                "symbolic_only": {"root_label": label, "root_prediction": label},
                "neural_only": {"root_label": label, "root_prediction": "wrong_root"},
                "domain_specific_classifier": {"root_label": label, "root_prediction": "wrong_root"},
                "retrieval_rag_nearest_neighbor": {"root_label": label, "root_prediction": "wrong_root"},
                "majority_prior": {"root_label": label, "root_prediction": "wrong_root"},
            }
        )

    assessment = validator._generalization_assessment(
        _generalization_report_fixture(
            domains=domains,
            records=records,
            ablation_records=ablation_records,
        )
    )

    assert assessment["gates"]["GEN-007"]["passed"] is False
    assert assessment["computed"]["baseline_uplift"]["beats_strongest_baseline"] is False


def test_validator_rejects_legacy_baseline_namespaces_that_contradict_gen007() -> None:
    validator = _validator_module()
    domains = [
        "clinical_deterioration",
        "manufacturing_quality",
        "cybersecurity_intrusion",
        "lab_operations_incident",
        "cloud_operations_rca",
    ]
    records = [
        _generalization_record(
            domain=domain,
            index=index,
            label=f"{domain}_root",
            prediction=f"{domain}_root",
        )
        for domain in domains
        for index in range(100)
    ]
    ablation_records = []
    for record in records:
        label = str(record["root_label"])
        ablation_records.append(
            {
                "neuro_symbolic": {"root_label": label, "root_prediction": label},
                "symbolic_only": {"root_label": label, "root_prediction": "wrong_root"},
                "neural_only": {"root_label": label, "root_prediction": "wrong_root"},
                "domain_specific_classifier": {
                    "root_label": label,
                    "root_prediction": "wrong_root",
                },
                "retrieval_rag_nearest_neighbor": {
                    "root_label": label,
                    "root_prediction": "wrong_root",
                },
                "majority_prior": {"root_label": label, "root_prediction": "wrong_root"},
            }
        )
    payload = _generalization_report_fixture(
        domains=domains,
        records=records,
        ablation_records=ablation_records,
    )
    payload["generalization_report"]["baseline_uplift"][
        "relative_uplift_over_best_baseline_pct"
    ] = 100.0
    payload["baseline_metrics"] = {
        "neural_only": 1.0,
        "neuro_symbolic": 1.0,
    }
    payload["computed_metrics"] = {
        "baseline_metrics": {
            "neural_only": 1.0,
            "neuro_symbolic": 1.0,
        }
    }

    assessment = validator._generalization_assessment(payload)
    gen007 = assessment["gates"]["GEN-007"]

    assert gen007["passed"] is False
    assert gen007["summary_metric_consistency"]["passed"] is False
    assert sorted(gen007["summary_metric_consistency"]["legacy_metric_locations"]) == [
        "baseline_metrics",
        "computed_metrics.baseline_metrics",
    ]


def test_public_benchmark_assessment_requires_explicit_nonclaim_or_coverage() -> None:
    validator = _validator_module()

    missing = validator._public_benchmark_assessment({})
    assert missing["passed"] is False
    assert "PUBLIC-SOTA-REPORT-MISSING" in missing["blockers"]

    nonclaim = validator._public_benchmark_assessment(
        {
            "public_benchmark_report": {
                "status": "FAIL",
                "external_sota_claim": False,
                "runner": "MISSING_PUBLIC_SOTA_RUNNER",
                "covered_public_benchmark_families": [],
                "blockers": [
                    "MISSING_PUBLIC_SOTA_RUNNER",
                    "PUBLIC-SOTA-FAMILY-COVERAGE",
                    "PUBLIC-SOTA-UNMET",
                ],
            }
        }
    )
    assert nonclaim["passed"] is True
    assert nonclaim["external_sota_claim"] is False
    assert nonclaim["missing_public_benchmark_families"]

    unsupported_claim = validator._public_benchmark_assessment(
        {
            "public_benchmark_report": {
                "status": "PASS",
                "external_sota_claim": True,
                "runner": "MISSING_PUBLIC_SOTA_RUNNER",
                "covered_public_benchmark_families": [],
                "blockers": [],
            }
        }
    )
    assert unsupported_claim["passed"] is False
    assert "PUBLIC-SOTA-FAMILY-COVERAGE" in unsupported_claim["blockers"]


def test_validator_requires_heldout_scoring_through_xlog_nn4() -> None:
    validator = _validator_module()
    domains = [
        "clinical_deterioration",
        "manufacturing_quality",
        "cybersecurity_intrusion",
        "lab_operations_incident",
        "cloud_operations_rca",
    ]
    records = [
        _generalization_record(
            domain=domain,
            index=index,
            label=f"{domain}_root",
            prediction=f"{domain}_root",
        )
        for domain in domains
        for index in range(100)
    ]
    ablation_records = []
    for record in records:
        label = str(record["root_label"])
        ablation_records.append(
            {
                method: {"root_label": label, "root_prediction": label}
                for method in [
                    "neural_only",
                    "symbolic_only",
                    "domain_specific_classifier",
                    "retrieval_rag_nearest_neighbor",
                    "majority_prior",
                    "neuro_symbolic",
                ]
            }
        )
    payload = _generalization_report_fixture(
        domains=domains,
        records=records,
        ablation_records=ablation_records,
    )
    del payload["generalization_report"]["neural_ranker"]["heldout_scoring"]

    assessment = validator._generalization_assessment(payload)

    assert assessment["gates"]["GEN-005"]["passed"] is False


def test_validator_requires_adversarial_performance_thresholds() -> None:
    validator = _validator_module()
    domains = [
        "clinical_deterioration",
        "manufacturing_quality",
        "cybersecurity_intrusion",
        "lab_operations_incident",
        "cloud_operations_rca",
    ]
    records = []
    for domain in domains:
        for index in range(100):
            label = f"{domain}_root"
            records.append(
                _generalization_record(
                    domain=domain,
                    index=index,
                    label=label,
                    prediction=label,
                )
            )
            records.append(
                _generalization_record(
                    domain=domain,
                    index=index,
                    label=label,
                    prediction="wrong_root",
                    variant="sparse",
                )
            )
    ablation_records = [
        {
            method: {
                "root_label": f"{domain}_root",
                "root_prediction": f"{domain}_root",
            }
            for method in [
                "neural_only",
                "symbolic_only",
                "domain_specific_classifier",
                "retrieval_rag_nearest_neighbor",
                "majority_prior",
                "neuro_symbolic",
            ]
        }
        for domain in domains
        for _index in range(100)
    ]

    assessment = validator._generalization_assessment(
        _generalization_report_fixture(
            domains=domains,
            records=records,
            ablation_records=ablation_records,
        )
    )

    assert assessment["gates"]["GEN-009"]["passed"] is False
    assert assessment["computed"]["adversarial_domain_shift"]["macro_f1_by_variant"]["sparse"] < 0.80


def test_validator_rejects_forbidden_manufacturing_label_dataset(tmp_path: Path) -> None:
    production = tmp_path / "production_transfer_with_bfds_source.json"
    _verified_production_evidence_fixture(production)
    payload = json.loads(production.read_text(encoding="utf-8"))
    for source in payload["huggingface_dataset_sources"]:
        if source["domain_id"] == "manufacturing_quality":
            source["hf_dataset_id"] = "BFDS-Project/Bearing-Fault-Diagnosis-System"
            source["file"] = "bearing_fault.csv"
            source["root_truth_source_type"] = "huggingface_fault_diagnosis_label"
    for record in payload["metric_inputs"]["prediction_records"]:
        if record["domain_id"] == "manufacturing_quality":
            record["source"]["hf_dataset_id"] = "BFDS-Project/Bearing-Fault-Diagnosis-System"
            record["source"]["file"] = "bearing_fault.csv"
            record["root_truth"]["source_type"] = "huggingface_fault_diagnosis_label"
    production.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validator = _validator_module()
    loaded = validator._load_production_transfer_evidence(production)

    assert validator._source_provenance_passed(loaded) is False
    assert validator._production_transfer_passed(loaded) is False


def test_validator_rejects_prediction_records_without_bfo_explanations(tmp_path: Path) -> None:
    production = tmp_path / "production_transfer_without_explanations.json"
    _verified_production_evidence_fixture(production)
    payload = json.loads(production.read_text(encoding="utf-8"))
    for record in payload["metric_inputs"]["prediction_records"]:
        record.pop("bfo_explanations", None)
    production.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validator = _validator_module()
    loaded = validator._load_production_transfer_evidence(production)

    assert validator._explanation_records_passed(loaded) is False
    assert validator._production_transfer_passed(loaded) is False


def test_validator_retries_cuda_oom_smoke_once(monkeypatch: pytest.MonkeyPatch) -> None:
    validator = _validator_module()
    calls = 0

    class Result:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(*args: object, **kwargs: object) -> Result:
        nonlocal calls
        calls += 1
        if calls == 1:
            return Result(
                1,
                stderr='DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")',
            )
        return Result(0, stdout='{"status":"PASS"}')

    monkeypatch.setattr(validator.subprocess, "run", fake_run)
    monkeypatch.setattr(
        validator,
        "_load_json",
        lambda path: {"status": "PASS", "program_declares_nn4": True},
    )

    result = validator._run_neural_smoke(enabled=True)

    assert calls == 2
    assert result["status"] == "PASS"
    assert result["cuda_oom_retries"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the strict neural gate")
def test_strict_validator_consumes_neural_smoke_evidence(tmp_path: Path) -> None:
    output = tmp_path / "validation_summary.json"
    missing_production = tmp_path / "missing_production_transfer.json"

    proc = subprocess.run(
        [
            str(ROOT / "validate.sh"),
            "--strict",
            "--gpu-required",
            "--production-transfer",
            str(missing_production),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )

    assert proc.returncode == 1
    summary = json.loads(output.read_text(encoding="utf-8"))
    gates = {gate["requirement_id"]: gate for gate in summary["p0_gates"]}
    metrics = {metric["id"]: metric for metric in summary["gqm_metrics"]}

    assert gates["P0-HARD-005"]["status"] == "PASS"
    assert metrics["Q4"]["status"] == "PASS"
    assert metrics["Q4"]["actual"] == "ranking_changed"
    assert any(path.endswith("evidence/neural_smoke.json") for path in summary["raw_output_paths"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the strict BFO fixture gate")
def test_strict_validator_consumes_bfo_fixture_smoke_evidence(tmp_path: Path) -> None:
    output = tmp_path / "validation_summary.json"
    missing_production = tmp_path / "missing_production_transfer.json"

    proc = subprocess.run(
        [
            str(ROOT / "validate.sh"),
            "--strict",
            "--gpu-required",
            "--production-transfer",
            str(missing_production),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )

    assert proc.returncode == 1
    summary = json.loads(output.read_text(encoding="utf-8"))
    gates = {gate["requirement_id"]: gate for gate in summary["p0_gates"]}
    metrics = {metric["id"]: metric for metric in summary["gqm_metrics"]}

    assert gates["TRANSFER-002"]["status"] == "FAIL"
    assert gates["TRANSFER-003"]["status"] == "FAIL"
    assert "production" in gates["TRANSFER-002"]["evidence"]
    assert metrics["Q5"]["status"] == "FAIL"
    assert metrics["Q6"]["status"] == "FAIL"
    assert metrics["Q7"]["status"] == "FAIL"
    assert any(path.endswith("evidence/bfo_fixture_smoke.json") for path in summary["raw_output_paths"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the strict ablation gate")
def test_strict_validator_consumes_ablation_evidence(tmp_path: Path) -> None:
    output = tmp_path / "validation_summary.json"
    missing_production = tmp_path / "missing_production_transfer.json"

    proc = subprocess.run(
        [
            str(ROOT / "validate.sh"),
            "--strict",
            "--gpu-required",
            "--production-transfer",
            str(missing_production),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )

    assert proc.returncode == 1
    summary = json.loads(output.read_text(encoding="utf-8"))
    gates = {gate["requirement_id"]: gate for gate in summary["p0_gates"]}
    metrics = {metric["id"]: metric for metric in summary["gqm_metrics"]}

    assert gates["NEURAL-002"]["status"] == "FAIL"
    assert "production" in gates["NEURAL-002"]["evidence"]
    assert metrics["Q8"]["status"] == "FAIL"
    assert any(path.endswith("evidence/ablation_smoke.json") for path in summary["raw_output_paths"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the strict runtime gate")
def test_strict_validator_consumes_runtime_contract_evidence(tmp_path: Path) -> None:
    output = tmp_path / "validation_summary.json"
    missing_production = tmp_path / "missing_production_transfer.json"

    proc = subprocess.run(
        [
            str(ROOT / "validate.sh"),
            "--strict",
            "--gpu-required",
            "--production-transfer",
            str(missing_production),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )

    assert proc.returncode == 1
    summary = json.loads(output.read_text(encoding="utf-8"))
    gates = {gate["requirement_id"]: gate for gate in summary["p0_gates"]}
    metrics = {metric["id"]: metric for metric in summary["gqm_metrics"]}

    assert gates["DEVICE-001"]["status"] == "PASS"
    assert metrics["Q9"]["status"] == "PASS"
    assert metrics["Q9"]["actual"] == 100.0
    assert metrics["Q10"]["status"] == "PASS"
    assert metrics["Q10"]["actual"] == 0
    assert metrics["Q11"]["status"] == "PASS"
    assert metrics["Q11"]["actual"] == "5/5"
    assert any(path.endswith("evidence/runtime_contract_smoke.json") for path in summary["raw_output_paths"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for strict validation")
def test_strict_validator_rejects_demo_only_generalization_evidence(tmp_path: Path) -> None:
    output = tmp_path / "validation_summary.json"
    production = tmp_path / "production_transfer.json"
    _verified_production_evidence_fixture(production)

    proc = subprocess.run(
        [
            str(ROOT / "validate.sh"),
            "--strict",
            "--gpu-required",
            "--production-transfer",
            str(production),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )

    assert proc.returncode == 1
    summary = json.loads(output.read_text(encoding="utf-8"))
    gates = {gate["requirement_id"]: gate for gate in summary["p0_gates"]}

    assert summary["status"] == "FAIL"
    assert gates["TRANSFER-002"]["status"] == "FAIL"
    assert gates["TRANSFER-003"]["status"] == "FAIL"
    assert gates["NEURAL-002"]["status"] == "FAIL"
    assert gates["XLOG-001"]["status"] == "PASS"
    assert gates["GEN-001"]["status"] == "FAIL"
    assert gates["GEN-002"]["status"] == "FAIL"
    assert gates["GEN-010"]["status"] == "FAIL"
    assert any(blocker["requirement_id"].startswith("GEN-") for blocker in summary["blockers"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for strict validation")
def test_strict_validator_rejects_summary_only_production_transfer_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "validation_summary.json"
    production = tmp_path / "summary_only_production_transfer.json"
    _production_evidence_fixture(production)

    proc = subprocess.run(
        [
            str(ROOT / "validate.sh"),
            "--strict",
            "--gpu-required",
            "--production-transfer",
            str(production),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )

    assert proc.returncode == 1
    summary = json.loads(output.read_text(encoding="utf-8"))
    gates = {gate["requirement_id"]: gate for gate in summary["p0_gates"]}
    assert gates["TRANSFER-002"]["status"] == "FAIL"
    assert "prediction_records" in gates["TRANSFER-002"]["evidence"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for strict validation")
def test_strict_validator_rejects_production_transfer_without_bundle_reuse(
    tmp_path: Path,
) -> None:
    output = tmp_path / "validation_summary.json"
    production = tmp_path / "production_transfer_without_bundle_reuse.json"
    _verified_production_evidence_fixture(production)
    payload = json.loads(production.read_text(encoding="utf-8"))
    payload.pop("bundle_reuse")
    production.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [
            str(ROOT / "validate.sh"),
            "--strict",
            "--gpu-required",
            "--production-transfer",
            str(production),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )

    assert proc.returncode == 1
    summary = json.loads(output.read_text(encoding="utf-8"))
    gates = {gate["requirement_id"]: gate for gate in summary["p0_gates"]}
    assert gates["BUNDLE-001"]["status"] == "FAIL"
    assert "bundle_reuse" in gates["BUNDLE-001"]["evidence"]
