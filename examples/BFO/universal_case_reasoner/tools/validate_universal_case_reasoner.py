#!/usr/bin/env python3
"""Strict validator for the BFO universal case reasoner example."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "validation_summary.json"
EXPECTED_HF_SOURCE_CONTRACTS = {
    "clinical_deterioration": [
        {
            "hf_dataset_id": "RootCauseAnalytics/Healthcare-Library-Sample",
            "file": "ground_truth.jsonl",
            "root_truth_source_type": "huggingface_ground_truth_diagnosis",
        },
        {
            "hf_dataset_id": "sajjadhadi/disease-diagnosis-dataset",
            "file": "data/test-00000-of-00001.parquet",
            "root_truth_source_type": "huggingface_external_diagnosis",
        },
    ],
    "manufacturing_quality": [
        {
            "hf_dataset_id": "Fujitsu/ManufacturingRCA_Knowledge_Dataset",
            "file": "ManufacturingRCA_doc.csv",
            "root_truth_source_type": "huggingface_manufacturing_rca_cause",
        },
    ],
    "cybersecurity_intrusion": [
        {
            "hf_dataset_id": "Perfectyash/human-style-cyber-incident-judgment",
            "file": "human_style_cyber_incident_judgment.csv",
            "root_truth_source_type": "huggingface_human_incident_reasoning",
        },
        {
            "hf_dataset_id": "savaniDhruv/Cybersecurity_Attack_Dataset",
            "file": "Attack_Dataset.csv",
            "root_truth_source_type": "huggingface_attack_vulnerability",
        },
    ],
    "lab_operations_incident": [
        {
            "hf_dataset_id": "LHRS-UM-FERI/MENTHOS-dataset-rootcause",
            "file": "root-cause-train.csv",
            "root_truth_source_type": "huggingface_root_cause_marked_log",
        },
    ],
    "cloud_operations_rca": [
        {
            "hf_dataset_id": "heetha/RCA",
            "file": "rca_data.csv",
            "root_truth_source_type": "huggingface_rca_json_root_cause",
        },
    ],
}
FORBIDDEN_LABEL_MAPPING_DATASETS = {
    "BFDS-Project/Bearing-Fault-Diagnosis-System",
}
GENERALIZATION_REQUIREMENTS = {
    "leave_one_domain_out_coverage": "Leave-one-domain-out evaluation covers every production domain.",
    "minimum_heldout_domain_size": "Every held-out domain has at least 100 real Hugging Face cases with row and field hashes.",
    "macro_transfer_quality": "Macro held-out root-cause F1 is >= 0.90 and every domain F1 is >= 0.85.",
    "heldout_candidate_independence": "Held-out candidate spaces are not constructed from held-out test RCA/root/intervention labels.",
    "frozen_model_and_ranker": "BFO rules, learned rules, neural architecture, thresholds, aliases, and scoring weights are frozen before held-out evaluation.",
    "unseen_dataset_transfer": "At least one held-out evaluation uses an unseen dataset family.",
    "strong_baseline_uplift": "Strong baselines cover neural-only, symbolic-only, domain-specific classifier, retrieval/RAG nearest-neighbor, majority/prior, and neuro-symbolic methods.",
    "statistical_confidence": "Bootstrap confidence intervals and paired significance tests are reported.",
    "adversarial_domain_shift": "Adversarial domain-shift variants are evaluated.",
    "aggregate_generalization_recompute": "Validator recomputes aggregate generalization metrics from raw records with no excluded domains.",
}
REQUIRED_GENERALIZATION_BASELINES = {
    "neural_only",
    "symbolic_only",
    "domain_specific_classifier",
    "retrieval_rag_nearest_neighbor",
    "majority_prior",
    "neuro_symbolic",
}
REQUIRED_ADVERSARIAL_VARIANTS = {
    "noisy",
    "sparse",
    "paraphrased",
    "missing_field",
    "distractor_candidate",
}
REQUIRED_PUBLIC_BENCHMARK_FAMILIES = {
    "aiops_rca",
    "clinical_diagnosis",
    "cross_domain_ontology_shift",
    "cybersecurity_intrusion",
    "manufacturing_equipment_fault",
    "phm_fault",
    "root_cause_aiops",
}
GENERALIZATION_THRESHOLDS = {
    "macro_f1": 0.90,
    "min_domain_f1": 0.85,
    "baseline_uplift_pct": 15.0,
    "adversarial_macro_f1": 0.80,
}
DIFFERENTIABLE_INDUCTIVE_LOGIC_REQUIREMENTS = {
    "xlog_proof_paths": "XLOG proof-path clauses are executed and selected for rule induction.",
    "joint_training": "Neural predicates and symbolic rule weights are trained jointly on CUDA.",
    "rule_inventory": "Learned rule inventories cover every leave-one-domain-out fold.",
    "clause_ablations": "Clause ablations are reported and full differentiable ILP macro F1 is >= 0.90 while matching or beating every learned-clause ablation.",
    "proof_gradients": "Proof-level gradients are present and device-resident.",
    "heldout_safe_induction": "Rule induction is held-out safe and uses no held-out labels during training.",
}


def _expected_hf_source_contracts(domain: str) -> list[dict[str, str]]:
    expected = EXPECTED_HF_SOURCE_CONTRACTS.get(domain)
    if expected is None:
        return []
    if isinstance(expected, list):
        return expected
    return [expected]


def _hf_source_matches_contract(source: dict[str, Any], contract: dict[str, str]) -> bool:
    return (
        source.get("hf_dataset_id") == contract["hf_dataset_id"]
        and source.get("file") == contract["file"]
        and source.get("root_truth_source_type") == contract["root_truth_source_type"]
    )


@dataclass(frozen=True)
class Gate:
    requirement_id: str
    description: str
    status: str
    evidence: str

    def to_json(self) -> dict[str, str]:
        return {
            "requirement_id": self.requirement_id,
            "description": self.description,
            "status": self.status,
            "evidence": self.evidence,
        }


def _run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _field_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalized_label_value(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip().lower()


def _load_hf_rows(dataset_id: str, filename: str) -> dict[int, dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename=filename)
    )
    if filename.endswith(".jsonl"):
        rows: dict[int, dict[str, Any]] = {}
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if line.strip():
                    rows[index] = json.loads(line)
        return rows
    if filename.endswith(".csv"):
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            return {index: dict(row) for index, row in enumerate(csv.DictReader(handle))}
    if filename.endswith(".parquet"):
        import pyarrow.parquet as parquet

        return {
            index: dict(row)
            for index, row in enumerate(parquet.read_table(path).to_pylist())
        }
    raise ValueError(f"unsupported Hugging Face evidence file: {filename}")


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _import_status(module: str) -> tuple[bool, str, Any | None]:
    try:
        loaded = importlib.import_module(module)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None
    version = getattr(loaded, "__version__", "unknown")
    return True, str(version), loaded


def _runtime_details() -> dict[str, Any]:
    torch_ok, torch_version, torch_mod = _import_status("torch")
    cuda_available = False
    cuda_device = None
    cuda_version = None
    if torch_ok and torch_mod is not None:
        try:
            cuda_available = bool(torch_mod.cuda.is_available())
            cuda_version = getattr(torch_mod.version, "cuda", None)
            if cuda_available:
                cuda_device = torch_mod.cuda.get_device_name(0)
        except Exception as exc:
            cuda_device = f"unavailable: {type(exc).__name__}: {exc}"

    had_cubin_dir = "XLOG_CUBIN_DIR" in os.environ
    previous_cubin_dir = os.environ.get("XLOG_CUBIN_DIR")
    pyxlog_ok, pyxlog_version, _ = _import_status("pyxlog")
    if had_cubin_dir and previous_cubin_dir is not None:
        os.environ["XLOG_CUBIN_DIR"] = previous_cubin_dir
    else:
        os.environ.pop("XLOG_CUBIN_DIR", None)

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": {
            "available": torch_ok,
            "version": torch_version if torch_ok else None,
            "error": None if torch_ok else torch_version,
            "cuda_available": cuda_available,
            "cuda_version": cuda_version,
            "cuda_device": cuda_device,
        },
        "pyxlog": {
            "available": pyxlog_ok,
            "version": pyxlog_version if pyxlog_ok else None,
            "error": None if pyxlog_ok else pyxlog_version,
        },
    }


def _kernel_facts(kernel_path: Path) -> dict[str, Any]:
    if not kernel_path.exists():
        return {
            "exists": False,
            "bfo_category_count": 0,
            "bfo_relation_family_count": 0,
            "root_cause_rule_count": 0,
            "intervention_rule_count": 0,
            "checksum": None,
        }
    source = kernel_path.read_text(encoding="utf-8")
    return {
        "exists": True,
        "bfo_category_count": source.count("bfo_category("),
        "bfo_relation_family_count": source.count("bfo_relation_family("),
        "root_cause_rule_count": source.count("candidate_root_cause("),
        "intervention_rule_count": source.count("recommended_intervention("),
        "checksum": _file_sha256(kernel_path),
    }


def _domain_facts(inventory_path: Path) -> dict[str, Any]:
    payload = _load_json(inventory_path)
    if not isinstance(payload, dict):
        return {
            "exists": inventory_path.exists(),
            "domain_count": 0,
            "domain_names": [],
            "all_classes_mapped": False,
            "max_adapter_core_rule_ratio": None,
            "holdout_domain": None,
            "domains_with_required_fixtures": 0,
        }

    domains = payload.get("domains", [])
    if not isinstance(domains, list):
        domains = []
    required_fixture_keys = {
        "root_cause",
        "failure_chain",
        "risk_state",
        "intervention",
        "explanation",
    }
    all_classes_mapped = True
    ratios: list[float] = []
    domains_with_required_fixtures = 0
    names: list[str] = []
    for domain in domains:
        if not isinstance(domain, dict):
            all_classes_mapped = False
            continue
        names.append(str(domain.get("id", "")))
        classes = domain.get("classes", [])
        if not classes:
            all_classes_mapped = False
        for item in classes:
            if not isinstance(item, dict) or not item.get("bfo_category"):
                all_classes_mapped = False
        ratio = domain.get("adapter_core_rule_ratio")
        if isinstance(ratio, (int, float)):
            ratios.append(float(ratio))
        fixtures = domain.get("fixtures", {})
        if isinstance(fixtures, dict) and required_fixture_keys.issubset(fixtures):
            domains_with_required_fixtures += 1
    holdout = payload.get("holdout_protocol", {}).get("held_out_domain")
    return {
        "exists": inventory_path.exists(),
        "domain_count": len(domains),
        "domain_names": names,
        "all_classes_mapped": all_classes_mapped,
        "max_adapter_core_rule_ratio": max(ratios) if ratios else None,
        "holdout_domain": holdout,
        "domains_with_required_fixtures": domains_with_required_fixtures,
    }


def _decode_subprocess_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _cuda_oom_text(stdout: str, stderr: str) -> bool:
    text = f"{stdout}\n{stderr}".lower()
    return "cuda_error_out_of_memory" in text or "out of memory" in text


def _run_smoke_json(
    *,
    cmd: list[str],
    output: Path,
    timeout: int,
    failure_evidence: str,
    max_cuda_oom_retries: int = 3,
) -> dict[str, Any]:
    start = time.perf_counter()
    cuda_oom_retries = 0
    last_stdout = ""
    last_stderr = ""
    while True:
        try:
            output.unlink()
        except FileNotFoundError:
            pass
        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            last_stdout = proc.stdout
            last_stderr = proc.stderr
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            last_stdout = _decode_subprocess_text(exc.stdout or exc.output)
            last_stderr = _decode_subprocess_text(exc.stderr)
            returncode = 124

        duration = round(time.perf_counter() - start, 6)
        payload = _load_json(output)
        if returncode == 0 and isinstance(payload, dict):
            payload["duration_sec"] = duration
            payload["stdout"] = last_stdout
            payload["stderr"] = last_stderr
            payload["output"] = str(output)
            payload["cuda_oom_retries"] = cuda_oom_retries
            return payload

        if (
            returncode != 0
            and cuda_oom_retries < max_cuda_oom_retries
            and _cuda_oom_text(last_stdout, last_stderr)
        ):
            cuda_oom_retries += 1
            time.sleep(1.0 + cuda_oom_retries)
            continue

        return {
            "status": "FAIL",
            "output": str(output),
            "duration_sec": duration,
            "stdout": last_stdout,
            "stderr": last_stderr,
            "cuda_oom_retries": cuda_oom_retries,
            "evidence": failure_evidence,
        }


def _run_neural_smoke(enabled: bool) -> dict[str, Any]:
    output = ROOT / "evidence" / "neural_smoke.json"
    if not enabled:
        return {
            "status": "SKIP",
            "output": str(output),
            "evidence": "strict GPU validation was not requested",
        }
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "run_neural_smoke.py"),
        "--output",
        str(output),
    ]
    return _run_smoke_json(
        cmd=cmd,
        output=output,
        timeout=45,
        failure_evidence="neural smoke did not produce PASS evidence",
    )


def _run_bfo_fixture_smoke(enabled: bool) -> dict[str, Any]:
    output = ROOT / "evidence" / "bfo_fixture_smoke.json"
    if not enabled:
        return {
            "status": "SKIP",
            "output": str(output),
            "evidence": "strict GPU validation was not requested",
        }
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "run_bfo_fixture_smoke.py"),
        "--output",
        str(output),
    ]
    return _run_smoke_json(
        cmd=cmd,
        output=output,
        timeout=60,
        failure_evidence="BFO fixture smoke did not produce PASS evidence",
    )


def _run_ablation_smoke(enabled: bool) -> dict[str, Any]:
    output = ROOT / "evidence" / "ablation_smoke.json"
    if not enabled:
        return {
            "status": "SKIP",
            "output": str(output),
            "evidence": "strict GPU validation was not requested",
        }
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "run_ablation_smoke.py"),
        "--output",
        str(output),
    ]
    return _run_smoke_json(
        cmd=cmd,
        output=output,
        timeout=60,
        failure_evidence="Ablation smoke did not produce PASS evidence",
    )


def _run_runtime_contract_smoke(enabled: bool) -> dict[str, Any]:
    output = ROOT / "evidence" / "runtime_contract_smoke.json"
    if not enabled:
        return {
            "status": "SKIP",
            "output": str(output),
            "evidence": "strict GPU validation was not requested",
        }
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "run_runtime_contract_smoke.py"),
        "--output",
        str(output),
    ]
    return _run_smoke_json(
        cmd=cmd,
        output=output,
        timeout=80,
        failure_evidence="Runtime contract smoke did not produce PASS evidence",
    )


def _load_production_transfer_evidence(output: Path) -> dict[str, Any]:
    payload = _load_json(output)
    if not isinstance(payload, dict):
        return {
            "status": "MISSING",
            "scope": "production",
            "output": str(output),
            "evidence": (
                "Missing production transfer evidence: required true "
                "domain-agnostic transfer over >=5 domains with one held out "
                "during rule evolution."
            ),
        }
    payload["output"] = str(output)
    if isinstance(payload.get("computed_metrics"), dict):
        payload["source_computed_metrics"] = payload["computed_metrics"]
    payload["computed_metrics"] = _compute_production_metrics(payload)
    return payload


def _score_ablation(record: dict[str, Any]) -> float:
    return 1.0 if record.get("root_prediction") == record.get("root_label") else 0.0


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _macro_f1_from_pairs(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    labels = sorted({gold for gold, _pred in pairs} | {pred for _gold, pred in pairs})
    scores: list[float] = []
    for label in labels:
        true_positive = sum(1 for gold, pred in pairs if gold == label and pred == label)
        false_positive = sum(1 for gold, pred in pairs if gold != label and pred == label)
        false_negative = sum(1 for gold, pred in pairs if gold == label and pred != label)
        if true_positive == false_positive == false_negative == 0:
            continue
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        scores.append(
            _safe_ratio(2.0 * precision * recall, precision + recall)
            if precision + recall
            else 0.0
        )
    return sum(scores) / float(len(scores)) if scores else 0.0


def _macro_f1_from_records(records: list[dict[str, Any]]) -> float:
    return _macro_f1_from_pairs(
        [
            (str(record.get("root_label")), str(record.get("root_prediction")))
            for record in records
        ]
    )


def _compute_production_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metric_inputs = payload.get("metric_inputs")
    if not isinstance(metric_inputs, dict):
        return {
            "valid": False,
            "failure": "missing metric_inputs.prediction_records",
        }

    records = metric_inputs.get("prediction_records")
    ablation_records = metric_inputs.get("ablation_records")
    invalid_records = metric_inputs.get("invalid_cross_domain_records")
    if not isinstance(records, list) or not records:
        return {
            "valid": False,
            "failure": "missing metric_inputs.prediction_records",
        }
    if not isinstance(ablation_records, list) or not ablation_records:
        return {
            "valid": False,
            "failure": "missing metric_inputs.ablation_records",
        }
    if not isinstance(invalid_records, list) or not invalid_records:
        return {
            "valid": False,
            "failure": "missing metric_inputs.invalid_cross_domain_records",
        }

    held_out = payload.get("held_out_domain")
    held_out_records = [record for record in records if record.get("domain_id") == held_out]
    non_held_out_records = [record for record in records if record.get("domain_id") != held_out]
    if not held_out_records:
        return {
            "valid": False,
            "failure": "no held-out prediction_records for held_out_domain",
        }
    if not non_held_out_records:
        return {
            "valid": False,
            "failure": "no non-held-out prediction_records for promoted-rule quality",
        }

    root_correct = sum(
        1
        for record in held_out_records
        if record.get("root_prediction") == record.get("root_label")
    )
    intervention_correct = sum(
        1
        for record in held_out_records
        if record.get("intervention_prediction") == record.get("intervention_label")
    )
    explanation_complete = sum(
        1 for record in held_out_records if record.get("explanation_valid") is True
    )
    promoted_correct = sum(
        1
        for record in non_held_out_records
        if record.get("root_prediction") == record.get("root_label")
    )
    invalid_rejected = sum(1 for record in invalid_records if record.get("rejected") is True)

    required_methods = {
        "neural_only",
        "domain_symbolic",
        "shared_symbolic",
        "neuro_symbolic",
    }
    baseline_metrics: dict[str, float] = {}
    for method in sorted(required_methods):
        method_records = [
            record.get(method) for record in ablation_records if isinstance(record.get(method), dict)
        ]
        if len(method_records) != len(ablation_records):
            return {
                "valid": False,
                "failure": f"missing ablation method {method}",
            }
        baseline_metrics[method] = sum(_score_ablation(record) for record in method_records) / len(
            method_records
        )

    non_neuro = {
        key: value for key, value in baseline_metrics.items() if key != "neuro_symbolic"
    }
    strongest_baseline = max(non_neuro, key=non_neuro.__getitem__)
    strongest_value = non_neuro[strongest_baseline]
    neuro_value = baseline_metrics["neuro_symbolic"]
    uplift = ((neuro_value - strongest_value) / strongest_value * 100.0) if strongest_value else (
        100.0 if neuro_value > 0.0 else 0.0
    )

    showcase_metrics = {
        "baseline_metrics": baseline_metrics,
        "ablation_scoring": {
            "primary_metric": "root_cause_accuracy",
            "intervention_precision_reported_separately": True,
            "explanation_coverage_reported_separately": True,
        },
        "strongest_baseline": strongest_baseline,
        "strongest_baseline_value": strongest_value,
        "relative_uplift_over_best_baseline_pct": uplift,
    }

    return {
        "valid": True,
        "held_out_root_cause_f1": _safe_ratio(root_correct, len(held_out_records)),
        "held_out_root_cause_confusion": {
            "correct": root_correct,
            "gold": len(held_out_records),
            "predicted": len(held_out_records),
            "total": len(held_out_records),
        },
        "accepted_intervention_precision": _safe_ratio(
            intervention_correct, len(held_out_records)
        ),
        "intervention_confusion": {
            "correct": intervention_correct,
            "predicted": len(held_out_records),
            "total": len(held_out_records),
        },
        "explanations_complete_pct": _safe_ratio(
            explanation_complete, len(held_out_records)
        )
        * 100.0,
        "invalid_cross_domain_rejection_pct": _safe_ratio(
            invalid_rejected, len(invalid_records)
        )
        * 100.0,
        "promoted_rule_quality": {
            "precision": _safe_ratio(promoted_correct, len(non_held_out_records)),
            "recall": _safe_ratio(promoted_correct, len(non_held_out_records)),
            "f1": _safe_ratio(promoted_correct, len(non_held_out_records)),
            "kernel_mutated": False,
        },
        "showcase_metrics": showcase_metrics,
    }


def _generalization_records(
    production_transfer: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    metric_inputs = production_transfer.get("metric_inputs")
    if not isinstance(metric_inputs, dict):
        return [], "missing metric_inputs"

    records = metric_inputs.get("generalization_prediction_records")
    if isinstance(records, list) and records:
        return [record for record in records if isinstance(record, dict)], (
            "metric_inputs.generalization_prediction_records"
        )

    demo_records = metric_inputs.get("prediction_records")
    held_out = production_transfer.get("held_out_domain")
    if not isinstance(demo_records, list) or not held_out:
        return [], "missing metric_inputs.generalization_prediction_records"

    fallback_records: list[dict[str, Any]] = []
    for record in demo_records:
        if not isinstance(record, dict) or record.get("domain_id") != held_out:
            continue
        fallback = dict(record)
        fallback.setdefault("held_out_domain", held_out)
        fallback.setdefault("evaluation_variant", "clean")
        fallback_records.append(fallback)
    return fallback_records, (
        "demo fallback from metric_inputs.prediction_records for the single held-out domain"
    )


def _generalization_baseline_methods(production_transfer: dict[str, Any]) -> set[str]:
    report = production_transfer.get("generalization_report")
    if isinstance(report, dict):
        methods = report.get("baseline_methods")
        if isinstance(methods, list):
            return {str(method) for method in methods}

    metric_inputs = production_transfer.get("metric_inputs")
    if not isinstance(metric_inputs, dict):
        return set()
    records = metric_inputs.get("generalization_ablation_records")
    if not isinstance(records, list) or not records:
        records = metric_inputs.get("ablation_records")
    if not isinstance(records, list):
        return set()

    metadata_keys = {"case_id", "domain_id", "held_out_domain", "evaluation_variant"}
    methods: set[str] = set()
    for record in records:
        if isinstance(record, dict):
            methods.update(
                key
                for key, value in record.items()
                if key not in metadata_keys and isinstance(value, dict)
            )
    return methods


def _record_has_hf_hashes(record: dict[str, Any]) -> bool:
    source = record.get("source")
    root_truth = record.get("root_truth")
    if not isinstance(source, dict) or not isinstance(root_truth, dict):
        return False
    return (
        source.get("source_type") == "huggingface"
        and bool(source.get("row_hash"))
        and bool(root_truth.get("field_value_hash"))
        and bool(root_truth.get("external_root_cause_text_hash"))
    )


def _candidate_generation_independent(record: dict[str, Any]) -> bool:
    generation = record.get("candidate_generation")
    if not isinstance(generation, dict):
        return False
    source_text = " ".join(
        str(generation.get(key, ""))
        for key in ["mode", "source", "candidate_source_scope"]
    ).lower()
    forbidden_source_markers = [
        "test_truth",
        "heldout_truth",
        "held_out_truth",
        "external_truth",
        "external_rca_candidate_space",
    ]
    return (
        generation.get("uses_heldout_test_truth") is False
        and generation.get("constructed_before_heldout_labels") is True
        and not any(marker in source_text for marker in forbidden_source_markers)
    )


def _reported_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _legacy_baseline_namespace_assessment(
    production_transfer: dict[str, Any],
    canonical_baseline_macro_f1: dict[str, float],
) -> dict[str, Any]:
    legacy_metric_locations: list[str] = []
    legacy_metric_values: dict[str, Any] = {}
    conflict_details: dict[str, dict[str, dict[str, Any]]] = {}
    for location, candidate in [
        ("baseline_metrics", production_transfer.get("baseline_metrics")),
        (
            "computed_metrics.baseline_metrics",
            (production_transfer.get("computed_metrics") or {}).get("baseline_metrics")
            if isinstance(production_transfer.get("computed_metrics"), dict)
            else None,
        ),
        (
            "source_computed_metrics.baseline_metrics",
            (production_transfer.get("source_computed_metrics") or {}).get("baseline_metrics")
            if isinstance(production_transfer.get("source_computed_metrics"), dict)
            else None,
        ),
    ]:
        if not isinstance(candidate, dict):
            continue
        legacy_metric_locations.append(location)
        legacy_metric_values[location] = candidate
        location_conflicts: dict[str, dict[str, Any]] = {}
        for method, raw_value in candidate.items():
            reported_value = _reported_float(raw_value)
            canonical_value = canonical_baseline_macro_f1.get(str(method))
            if (
                reported_value is None
                or canonical_value is None
                or abs(reported_value - canonical_value) > 1e-9
            ):
                location_conflicts[str(method)] = {
                    "legacy_value": raw_value,
                    "canonical_generalization_value": canonical_value,
                }
        if location_conflicts:
            conflict_details[location] = location_conflicts
    return {
        "passed": not legacy_metric_locations,
        "canonical_metric_source": "generalization_report.baseline_uplift",
        "canonical_baseline_macro_f1": canonical_baseline_macro_f1,
        "legacy_metric_locations": legacy_metric_locations,
        "legacy_metric_values": legacy_metric_values,
        "conflicts": conflict_details,
        "required_action": (
            "keep showcase ablation metrics under showcase_metrics and leave "
            "generalization_report.baseline_uplift as the only baseline-uplift source"
        ),
    }


def _public_benchmark_assessment(production_transfer: dict[str, Any]) -> dict[str, Any]:
    report = production_transfer.get("public_benchmark_report")
    if not isinstance(report, dict):
        return {
            "passed": False,
            "status": "MISSING",
            "external_state_of_the_art_claim": None,
            "covered_public_benchmark_families": [],
            "required_public_benchmark_families": sorted(REQUIRED_PUBLIC_BENCHMARK_FAMILIES),
            "missing_public_benchmark_families": sorted(REQUIRED_PUBLIC_BENCHMARK_FAMILIES),
            "blockers": ["PUBLIC_STATE_OF_THE_ART_REPORT_MISSING"],
            "claim_boundary": "public benchmark report is required for honest claim scope",
        }

    covered = {
        str(family)
        for family in report.get("covered_public_benchmark_families") or []
        if str(family)
    }
    missing_families = sorted(REQUIRED_PUBLIC_BENCHMARK_FAMILIES - covered)
    blockers = [str(blocker) for blocker in report.get("blockers") or [] if str(blocker)]
    status = str(report.get("status", "")).upper()
    external_state_of_the_art_claim = report.get("external_state_of_the_art_claim")
    runner = str(report.get("runner") or "")

    if external_state_of_the_art_claim is True:
        if status != "PASS":
            blockers.append("PUBLIC_STATE_OF_THE_ART_STATUS_NOT_PASS")
        if runner == "MISSING_PUBLIC_STATE_OF_THE_ART_RUNNER" or not runner:
            blockers.append("MISSING_PUBLIC_STATE_OF_THE_ART_RUNNER")
        if missing_families:
            blockers.append("PUBLIC_STATE_OF_THE_ART_FAMILY_COVERAGE")
        passed = not blockers and not missing_families
        claim_boundary = "external state-of-the-art performance claimed"
    elif external_state_of_the_art_claim is False:
        if status not in {"FAIL", "BLOCKED", "NOT_CLAIMED"}:
            blockers.append("PUBLIC_STATE_OF_THE_ART_NONCLAIM_STATUS_MISSING")
        if not blockers:
            blockers.append("PUBLIC_STATE_OF_THE_ART_NONCLAIM_BLOCKERS_MISSING")
        passed = status in {"FAIL", "BLOCKED", "NOT_CLAIMED"} and bool(blockers)
        claim_boundary = "external state-of-the-art performance not claimed"
    else:
        blockers.append("PUBLIC_STATE_OF_THE_ART_CLAIM_BOUNDARY_MISSING")
        passed = False
        claim_boundary = "external state-of-the-art claim boundary missing"

    blockers = sorted(set(blockers))
    return {
        "passed": passed,
        "status": status or "UNKNOWN",
        "external_state_of_the_art_claim": external_state_of_the_art_claim,
        "runner": runner,
        "covered_public_benchmark_families": sorted(covered),
        "required_public_benchmark_families": sorted(REQUIRED_PUBLIC_BENCHMARK_FAMILIES),
        "missing_public_benchmark_families": missing_families,
        "blockers": blockers,
        "claim_boundary": claim_boundary,
        "protocol_hashes": report.get("protocol_hashes") or {},
        "baseline_citations": report.get("baseline_citations") or {},
    }


def _generalization_assessment(production_transfer: dict[str, Any]) -> dict[str, Any]:
    domains = {
        str(domain)
        for domain in production_transfer.get("domain_ids") or []
        if str(domain)
    }
    if not domains and production_transfer.get("domain_count"):
        domains = {
            source.get("domain_id")
            for source in production_transfer.get("huggingface_dataset_sources") or []
            if isinstance(source, dict) and source.get("domain_id")
        }
        domains = {str(domain) for domain in domains}

    records, record_source = _generalization_records(production_transfer)
    clean_records = [
        record
        for record in records
        if str(record.get("evaluation_variant", "clean")) == "clean"
    ]
    records_by_holdout: dict[str, list[dict[str, Any]]] = {}
    for record in clean_records:
        held_out = record.get("held_out_domain") or record.get("domain_id")
        if held_out:
            records_by_holdout.setdefault(str(held_out), []).append(record)

    f1_by_domain: dict[str, float] = {}
    case_count_by_domain: dict[str, int] = {}
    hash_coverage_by_domain: dict[str, bool] = {}
    for domain in sorted(records_by_holdout):
        domain_records = records_by_holdout[domain]
        case_count_by_domain[domain] = len(domain_records)
        f1_by_domain[domain] = _macro_f1_from_records(domain_records)
        hash_coverage_by_domain[domain] = all(
            _record_has_hf_hashes(record) for record in domain_records
        )

    evaluated_domains = set(records_by_holdout)
    missing_domains = sorted(domains - evaluated_domains)
    macro_f1 = _macro_f1_from_records(clean_records)
    min_domain_f1 = min(f1_by_domain.values()) if f1_by_domain else 0.0

    report = production_transfer.get("generalization_report")
    report = report if isinstance(report, dict) else {}
    aggregate = report.get("aggregate") if isinstance(report.get("aggregate"), dict) else {}
    reported_macro = _reported_float(aggregate.get("macro_held_out_root_cause_f1"))
    reported_min = _reported_float(aggregate.get("min_domain_root_cause_f1"))
    report_matches = (
        reported_macro is not None
        and reported_min is not None
        and abs(reported_macro - macro_f1) <= 1e-9
        and abs(reported_min - min_domain_f1) <= 1e-9
    )

    frozen = report.get("frozen_model_rules")
    frozen = frozen if isinstance(frozen, dict) else {}
    frozen_keys = [
        "bfo_kernel",
        "learned_rules",
        "neural_architecture",
        "thresholds",
        "aliases",
        "scoring_weights",
        "generalization_seed_isolated_from_showcase_transfer",
    ]
    frozen_passed = frozen.get("passed") is True and all(
        frozen.get(key) is True for key in frozen_keys
    )
    ranker = report.get("neural_ranker")
    ranker = ranker if isinstance(ranker, dict) else {}
    heldout_scoring = ranker.get("heldout_scoring")
    heldout_scoring = heldout_scoring if isinstance(heldout_scoring, dict) else {}
    expected_generalization_candidate_count = sum(
        int((record.get("candidate_generation") or {}).get("candidate_count", 0))
        for record in records
        if isinstance(record, dict)
    )
    ranker_passed = (
        ranker.get("path") == "xlog_nn4_cuda_generalization"
        and ranker.get("program") == "programs/production_ranker.xlog"
        and ranker.get("registered_network") == "production_root_net"
        and ranker.get("selection_device") == "cuda"
        and ranker.get("uses_python_heuristic") is False
        and ranker.get("heldout_labels_used_in_nn4") is False
        and ranker.get("score_cpu_materialization_in_ranking") is False
        and ranker.get("full_score_rows_materialized") is False
        and ranker.get("scalar_item_calls_in_ranking") is False
        and ranker.get("post_ranking_evidence_serialization") == "selected_indices_only"
        and heldout_scoring.get("path") == "xlog_nn4_forward_backward_tensor"
        and heldout_scoring.get("program") == "programs/production_ranker.xlog"
        and heldout_scoring.get("expected_label") == "primary_root"
        and heldout_scoring.get("uses_heldout_labels") is False
        and heldout_scoring.get("loss_tensors_device") == "cuda"
        and heldout_scoring.get("score_tensor_device") == "cuda"
        and heldout_scoring.get("score_cpu_materialization_in_ranking") is False
        and int(heldout_scoring.get("query_count", 0))
        >= expected_generalization_candidate_count
        and int(ranker.get("nn4_query_count", 0)) > 0
        and all(
            record.get("ranker_path") == "xlog_nn4_cuda_generalization"
            for record in records
        )
        and all(
            isinstance(record.get("neural_scores"), dict)
            and record["neural_scores"].get("materialized") is False
            for record in records
        )
    )

    unseen = report.get("unseen_dataset_transfer")
    unseen = unseen if isinstance(unseen, dict) else {}
    unseen_records = [
        record
        for record in clean_records
        if isinstance(record.get("source"), dict)
        and record["source"].get("unseen_dataset_family") is True
        and record["source"].get("used_for_feature_design") is False
        and record["source"].get("dataset_family") == unseen.get("dataset_family")
        and record.get("held_out_domain") == unseen.get("held_out_domain")
    ]
    unseen_passed = (
        unseen.get("passed") is True
        and bool(unseen.get("held_out_domain"))
        and bool(unseen.get("dataset_family"))
        and bool(unseen_records)
    )

    confidence = report.get("statistical_confidence")
    confidence = confidence if isinstance(confidence, dict) else {}
    ci_by_domain = confidence.get("bootstrap_ci_by_domain")
    ci_by_domain = ci_by_domain if isinstance(ci_by_domain, dict) else {}
    paired_tests = confidence.get("paired_significance_tests")
    paired_tests = paired_tests if isinstance(paired_tests, list) else []
    confidence_passed = (
        confidence.get("passed") is True
        and int(confidence.get("bootstrap_iterations", 0)) >= 1000
        and domains <= {str(domain) for domain in ci_by_domain}
        and bool(paired_tests)
    )

    adversarial = report.get("adversarial_domain_shift")
    adversarial = adversarial if isinstance(adversarial, dict) else {}
    reported_variants = {
        str(variant)
        for variant in adversarial.get("variants", [])
        if str(variant)
    }
    record_variants = {
        str(record.get("evaluation_variant"))
        for record in records
        if record.get("evaluation_variant")
    }
    adversarial_variants = reported_variants | record_variants
    variant_macro_f1 = {
        variant: _macro_f1_from_records(
            [
                record
                for record in records
                if str(record.get("evaluation_variant", "clean")) == variant
            ]
        )
        for variant in adversarial_variants
    }
    adversarial_passed = (
        adversarial.get("passed") is True
        and REQUIRED_ADVERSARIAL_VARIANTS <= adversarial_variants
        and all(
            variant_macro_f1.get(variant, 0.0)
            >= GENERALIZATION_THRESHOLDS["adversarial_macro_f1"]
            for variant in REQUIRED_ADVERSARIAL_VARIANTS
        )
    )

    baseline_methods = _generalization_baseline_methods(production_transfer)
    metric_inputs = production_transfer.get("metric_inputs")
    metric_inputs = metric_inputs if isinstance(metric_inputs, dict) else {}
    ablation_records = metric_inputs.get("generalization_ablation_records")
    ablation_records = ablation_records if isinstance(ablation_records, list) else []
    baseline_macro_f1: dict[str, float] = {}
    for method in REQUIRED_GENERALIZATION_BASELINES:
        method_records = [
            record.get(method)
            for record in ablation_records
            if isinstance(record, dict) and isinstance(record.get(method), dict)
        ]
        baseline_macro_f1[method] = _macro_f1_from_records(method_records)
    non_neuro_baselines = {
        method: value
        for method, value in baseline_macro_f1.items()
        if method != "neuro_symbolic"
    }
    strongest_baseline = (
        max(non_neuro_baselines, key=non_neuro_baselines.__getitem__)
        if non_neuro_baselines
        else ""
    )
    strongest_baseline_value = non_neuro_baselines.get(strongest_baseline, 0.0)
    neuro_symbolic_value = baseline_macro_f1.get("neuro_symbolic", 0.0)
    baseline_uplift_pct = (
        (neuro_symbolic_value - strongest_baseline_value)
        / strongest_baseline_value
        * 100.0
        if strongest_baseline_value
        else (100.0 if neuro_symbolic_value > 0.0 else 0.0)
    )
    baseline_uplift = {
        "baseline_macro_f1": baseline_macro_f1,
        "strongest_baseline": strongest_baseline,
        "strongest_baseline_macro_f1": strongest_baseline_value,
        "neuro_symbolic_macro_f1": neuro_symbolic_value,
        "relative_uplift_over_best_baseline_pct": baseline_uplift_pct,
        "beats_strongest_baseline": (
            neuro_symbolic_value > strongest_baseline_value
            and baseline_uplift_pct >= GENERALIZATION_THRESHOLDS["baseline_uplift_pct"]
        ),
    }
    reported_uplift = report.get("baseline_uplift")
    reported_uplift = reported_uplift if isinstance(reported_uplift, dict) else {}
    summary_metric_consistency = _legacy_baseline_namespace_assessment(
        production_transfer,
        baseline_macro_f1,
    )
    baseline_uplift_passed = (
        REQUIRED_GENERALIZATION_BASELINES <= baseline_methods
        and baseline_uplift["beats_strongest_baseline"] is True
        and reported_uplift.get("beats_strongest_baseline") is True
        and summary_metric_consistency["passed"] is True
        and abs(
            _safe_ratio(
                float(reported_uplift.get("relative_uplift_over_best_baseline_pct", 0.0))
                - baseline_uplift_pct,
                1.0,
            )
        )
        <= 1e-6
    )
    excluded_raw = report.get("excluded_domains", [])
    excluded_domains = (
        sorted(str(domain) for domain in excluded_raw)
        if isinstance(excluded_raw, list)
        else []
    )
    min_case_requirement_passed = bool(domains) and all(
        case_count_by_domain.get(domain, 0) >= 100
        and hash_coverage_by_domain.get(domain) is True
        for domain in domains
    )
    candidate_independence_passed = bool(clean_records) and all(
        _candidate_generation_independent(record) for record in clean_records
    )
    uses_generalization_records = (
        record_source == "metric_inputs.generalization_prediction_records"
    )
    aggregate_generalization_recompute_passed = (
        uses_generalization_records
        and bool(domains)
        and not missing_domains
        and not excluded_domains
        and report_matches
    )

    gates = {
        "leave_one_domain_out_coverage": {
            "passed": bool(domains) and not missing_domains and bool(evaluated_domains),
            "evaluated_domains": sorted(evaluated_domains),
            "required_domains": sorted(domains),
            "missing_domains": missing_domains,
        },
        "minimum_heldout_domain_size": {
            "passed": min_case_requirement_passed,
            "case_count_by_domain": case_count_by_domain,
            "hash_coverage_by_domain": hash_coverage_by_domain,
            "minimum_required_per_domain": 100,
        },
        "macro_transfer_quality": {
            "passed": bool(domains)
            and not missing_domains
            and macro_f1 >= GENERALIZATION_THRESHOLDS["macro_f1"]
            and min_domain_f1 >= GENERALIZATION_THRESHOLDS["min_domain_f1"],
            "macro_held_out_root_cause_f1": macro_f1,
            "min_domain_root_cause_f1": min_domain_f1,
            "f1_by_domain": f1_by_domain,
            "metric": "standard_multiclass_macro_f1",
        },
        "heldout_candidate_independence": {
            "passed": candidate_independence_passed,
            "record_source": record_source,
            "required_candidate_generation_flags": {
                "uses_heldout_test_truth": False,
                "constructed_before_heldout_labels": True,
            },
        },
        "frozen_model_and_ranker": {
            "passed": frozen_passed and ranker_passed,
            "frozen_model_rules": frozen,
            "neural_ranker": ranker,
            "required_frozen_keys": frozen_keys,
        },
        "unseen_dataset_transfer": {
            "passed": unseen_passed,
            "unseen_dataset_transfer": unseen,
            "raw_unseen_record_count": len(unseen_records),
        },
        "strong_baseline_uplift": {
            "passed": baseline_uplift_passed,
            "baseline_methods": sorted(baseline_methods),
            "required_baselines": sorted(REQUIRED_GENERALIZATION_BASELINES),
            "missing_baselines": sorted(REQUIRED_GENERALIZATION_BASELINES - baseline_methods),
            "baseline_uplift": baseline_uplift,
            "reported_baseline_uplift": reported_uplift,
            "summary_metric_consistency": summary_metric_consistency,
            "minimum_relative_uplift_pct": GENERALIZATION_THRESHOLDS[
                "baseline_uplift_pct"
            ],
        },
        "statistical_confidence": {
            "passed": confidence_passed,
            "statistical_confidence": confidence,
            "required_domains": sorted(domains),
        },
        "adversarial_domain_shift": {
            "passed": adversarial_passed,
            "variants": sorted(adversarial_variants),
            "required_variants": sorted(REQUIRED_ADVERSARIAL_VARIANTS),
            "missing_variants": sorted(REQUIRED_ADVERSARIAL_VARIANTS - adversarial_variants),
            "macro_f1_by_variant": variant_macro_f1,
            "minimum_macro_f1": GENERALIZATION_THRESHOLDS["adversarial_macro_f1"],
        },
        "aggregate_generalization_recompute": {
            "passed": aggregate_generalization_recompute_passed,
            "record_source": record_source,
            "uses_generalization_prediction_records": uses_generalization_records,
            "missing_domains": missing_domains,
            "excluded_domains": excluded_domains,
            "reported_metrics_match_recomputed": report_matches,
            "recomputed_macro_held_out_root_cause_f1": macro_f1,
            "recomputed_min_domain_root_cause_f1": min_domain_f1,
            "reported_macro_held_out_root_cause_f1": reported_macro,
            "reported_min_domain_root_cause_f1": reported_min,
        },
    }
    return {
        "status": "PASS" if all(gate["passed"] for gate in gates.values()) else "FAIL",
        "record_source": record_source,
        "clean_record_count": len(clean_records),
        "evaluated_domains": sorted(evaluated_domains),
        "required_domains": sorted(domains),
        "computed": {
            "case_count_by_domain": case_count_by_domain,
            "macro_held_out_root_cause_f1": macro_f1,
            "min_domain_root_cause_f1": min_domain_f1,
            "f1_by_domain": f1_by_domain,
            "baseline_uplift": baseline_uplift,
            "adversarial_domain_shift": {
                "macro_f1_by_variant": variant_macro_f1,
                "minimum_macro_f1": GENERALIZATION_THRESHOLDS["adversarial_macro_f1"],
            },
        },
        "gates": gates,
    }


def _differentiable_inductive_logic_assessment(production_transfer: dict[str, Any]) -> dict[str, Any]:
    report = production_transfer.get("dilp_report")
    report = report if isinstance(report, dict) else {}
    metric_inputs = production_transfer.get("metric_inputs")
    metric_inputs = metric_inputs if isinstance(metric_inputs, dict) else {}
    records = metric_inputs.get("dilp_prediction_records")
    records = records if isinstance(records, list) else []
    domains = {
        str(domain)
        for domain in production_transfer.get("domain_ids") or []
        if str(domain)
    }
    inventory = report.get("rule_inventory")
    inventory = inventory if isinstance(inventory, list) else []
    inventory_domains = {
        str(entry.get("held_out_domain"))
        for entry in inventory
        if isinstance(entry, dict) and entry.get("held_out_domain")
    }
    joint = report.get("joint_training")
    joint = joint if isinstance(joint, dict) else {}
    ablations = report.get("clause_ablations")
    ablations = ablations if isinstance(ablations, dict) else {}
    heldout_safe = report.get("heldout_safe_rule_induction")
    heldout_safe = heldout_safe if isinstance(heldout_safe, dict) else {}
    recomputed_full_f1 = _macro_f1_from_records(records)
    reported_full_f1 = _reported_float(ablations.get("full_model_macro_f1"))
    reported_best_ablation = _reported_float(ablations.get("best_ablated_macro_f1"))
    without_clause = ablations.get("without_clause_f1")
    without_clause = without_clause if isinstance(without_clause, dict) else {}
    prediction_domains = {
        str(record.get("held_out_domain") or record.get("domain_id"))
        for record in records
        if isinstance(record, dict) and (record.get("held_out_domain") or record.get("domain_id"))
    }
    proof_path_passed = (
        report.get("path") == "xlog_cuda_dilp_rule_induction"
        and report.get("program") == "programs/dilp_proof_paths.xlog"
        and int(report.get("xlog_proof_path_queries", 0)) > 0
        and report.get("xlog_proof_tensors_cuda") is True
        and bool(records)
        and all(
            isinstance(record, dict)
            and record.get("ranker_path") == "xlog_cuda_dilp_rule_induction"
            and bool(record.get("selected_clause"))
            for record in records
        )
    )
    joint_passed = (
        joint.get("trained_jointly") is True
        and joint.get("neural_predicate") == "production_root_net"
        and joint.get("neural_program") == "programs/production_ranker.xlog"
        and int(joint.get("nn4_query_count", 0)) > 0
        and joint.get("symbolic_rule_weights_device") == "cuda"
        and float(joint.get("symbolic_rule_gradient_norm", 0.0)) > 0.0
        and float(joint.get("neural_weight_gradient_norm", 0.0)) > 0.0
        and float(joint.get("proof_path_gradient_norm", 0.0)) > 0.0
        and joint.get("loss_decreased") is True
        and joint.get("score_cpu_materialization_in_training") is False
        and joint.get("scalar_item_calls_in_training") is False
    )
    inventory_passed = (
        bool(domains)
        and domains <= inventory_domains
        and len(inventory) >= len(domains)
        and all(
            isinstance(entry, dict)
            and entry.get("program") == "programs/dilp_proof_paths.xlog"
            and bool(entry.get("selected_clause"))
            and entry.get("trained_on_held_out_domain") is False
            and int(entry.get("heldout_label_count_used", -1)) == 0
            and isinstance(entry.get("clause_weights"), dict)
            and bool(entry.get("clause_weights"))
            for entry in inventory
        )
    )
    ablation_passed = (
        bool(records)
        and isinstance(without_clause, dict)
        and len(without_clause) >= 3
        and reported_full_f1 is not None
        and reported_full_f1 >= GENERALIZATION_THRESHOLDS["macro_f1"]
        and abs(recomputed_full_f1 - reported_full_f1) <= 1e-9
        and reported_best_ablation is not None
        and reported_full_f1 >= reported_best_ablation
        and ablations.get("full_model_beats_or_matches_best_ablation") is True
    )
    proof_gradient_passed = (
        joint.get("proof_path_tensor_device") == "cuda"
        and float(joint.get("proof_path_gradient_norm", 0.0)) > 0.0
    )
    heldout_safe_passed = (
        heldout_safe.get("passed") is True
        and bool(domains)
        and domains <= prediction_domains
        and int(heldout_safe.get("fold_count", 0)) >= len(domains)
        and int(heldout_safe.get("heldout_examples_in_training", -1)) == 0
        and heldout_safe.get("trained_on_held_out_domain") is False
        and heldout_safe.get("candidate_spaces_use_heldout_test_truth") is False
        and heldout_safe.get("rules_frozen_before_heldout_scoring") is True
        and all(
            _candidate_generation_independent(record)
            for record in records
            if isinstance(record, dict)
        )
    )
    gates = {
        "xlog_proof_paths": {
            "passed": proof_path_passed,
            "program": report.get("program"),
            "xlog_proof_path_queries": report.get("xlog_proof_path_queries"),
            "xlog_proof_tensors_cuda": report.get("xlog_proof_tensors_cuda"),
            "prediction_record_count": len(records),
        },
        "joint_training": {
            "passed": joint_passed,
            "joint_training": joint,
        },
        "rule_inventory": {
            "passed": inventory_passed,
            "required_domains": sorted(domains),
            "inventory_domains": sorted(inventory_domains),
            "inventory_count": len(inventory),
        },
        "clause_ablations": {
            "passed": ablation_passed,
            "recomputed_full_model_macro_f1": recomputed_full_f1,
            "reported_full_model_macro_f1": reported_full_f1,
            "reported_best_ablated_macro_f1": reported_best_ablation,
            "minimum_full_model_macro_f1": GENERALIZATION_THRESHOLDS["macro_f1"],
            "without_clause_f1": without_clause,
        },
        "proof_gradients": {
            "passed": proof_gradient_passed,
            "proof_path_tensor_device": joint.get("proof_path_tensor_device"),
            "proof_path_gradient_norm": joint.get("proof_path_gradient_norm"),
        },
        "heldout_safe_induction": {
            "passed": heldout_safe_passed,
            "heldout_safe_rule_induction": heldout_safe,
            "prediction_domains": sorted(prediction_domains),
            "required_domains": sorted(domains),
        },
    }
    return {
        "status": "PASS" if all(gate["passed"] for gate in gates.values()) else "FAIL",
        "gates": gates,
        "record_count": len(records),
        "rule_inventory_count": len(inventory),
    }


def _source_provenance_passed(production_transfer: dict[str, Any]) -> bool:
    sources = production_transfer.get("huggingface_dataset_sources")
    domains = set(production_transfer.get("domain_ids") or [])
    if not isinstance(sources, list) or len(sources) < 5 or len(domains) < 5:
        return False
    source_domains = {source.get("domain_id") for source in sources if isinstance(source, dict)}
    if source_domains != domains:
        return False
    if not all(source.get("source_type") == "huggingface" for source in sources):
        return False
    if not all(source.get("hf_dataset_id") for source in sources):
        return False
    if not all(int(source.get("row_count", 0)) > 0 for source in sources):
        return False
    for source in sources:
        if not isinstance(source, dict):
            return False
        domain = str(source.get("domain_id"))
        contracts = _expected_hf_source_contracts(domain)
        if not contracts:
            return False
        if source.get("hf_dataset_id") in FORBIDDEN_LABEL_MAPPING_DATASETS:
            return False
        if not any(_hf_source_matches_contract(source, contract) for contract in contracts):
            return False

    metric_inputs = production_transfer.get("metric_inputs")
    if not isinstance(metric_inputs, dict):
        return False
    prediction_records = metric_inputs.get("prediction_records")
    if not isinstance(prediction_records, list) or not prediction_records:
        return False
    records = [record for record in prediction_records if isinstance(record, dict)]
    generalization_records = metric_inputs.get("generalization_prediction_records")
    if isinstance(generalization_records, list):
        records.extend(record for record in generalization_records if isinstance(record, dict))
    if not records:
        return False

    source_by_key = {
        (
            str(source.get("domain_id")),
            str(source.get("hf_dataset_id")),
            str(source.get("file")),
        ): source
        for source in sources
        if isinstance(source, dict)
    }
    required_rows: dict[tuple[str, str], set[int]] = {}
    for record in records:
        if not isinstance(record, dict):
            return False
        domain = str(record.get("domain_id"))
        source = record.get("source")
        root_truth = record.get("root_truth")
        intervention_truth = record.get("intervention_truth")
        candidate_generation = record.get("candidate_generation")
        if not isinstance(source, dict) or not isinstance(root_truth, dict):
            return False
        if not isinstance(intervention_truth, dict) or not isinstance(candidate_generation, dict):
            return False
        if record.get("root_label_source") != "huggingface_external_root_cause_analysis":
            return False
        if not str(root_truth.get("source_type", "")).startswith("huggingface_"):
            return False
        contracts = _expected_hf_source_contracts(domain)
        if not any(
            source.get("hf_dataset_id") == contract["hf_dataset_id"]
            and source.get("file") == contract["file"]
            and root_truth.get("source_type") == contract["root_truth_source_type"]
            for contract in contracts
        ):
            return False
        if root_truth.get("ordinary_label_mapping") is not False:
            return False
        if not root_truth.get("external_root_cause_text_hash"):
            return False
        if candidate_generation.get("label_injected") is not False:
            return False
        if int(candidate_generation.get("candidate_count", 0)) < 4:
            return False
        domain_source = source_by_key.get(
            (domain, str(source.get("hf_dataset_id")), str(source.get("file")))
        )
        if not isinstance(domain_source, dict):
            return False
        if source.get("source_type") != "huggingface":
            return False
        if source.get("hf_dataset_id") in FORBIDDEN_LABEL_MAPPING_DATASETS:
            return False
        if source.get("hf_dataset_id") != domain_source.get("hf_dataset_id"):
            return False
        if source.get("split") != domain_source.get("split"):
            return False
        if source.get("file") != domain_source.get("file"):
            return False
        try:
            row_index = int(source.get("row_index"))
        except (TypeError, ValueError):
            return False
        if row_index < 0:
            return False
        required_rows.setdefault(
            (str(source["hf_dataset_id"]), str(source["file"])),
            set(),
        ).add(row_index)

    try:
        loaded_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
        for (dataset_id, filename), indexes in required_rows.items():
            rows_by_index = _load_hf_rows(dataset_id, filename)
            for index in indexes:
                if index in rows_by_index:
                    loaded_rows[(dataset_id, filename, index)] = rows_by_index[index]
    except Exception:
        return False

    for record in records:
        source = record["source"]
        root_truth = record["root_truth"]
        intervention_truth = record["intervention_truth"]
        dataset_id = str(source["hf_dataset_id"])
        filename = str(source["file"])
        row_index = int(source["row_index"])
        row = loaded_rows.get((dataset_id, filename, row_index))
        if row is None:
            return False
        if source.get("row_hash") != _row_hash(row):
            return False
        field_name = root_truth.get("field_name")
        if not isinstance(field_name, str) or field_name not in row:
            return False
        raw_value = row[field_name]
        if root_truth.get("field_value_hash") != _field_hash(raw_value):
            return False
        root_text = root_truth.get("external_root_cause_text")
        if not isinstance(root_text, str) or not root_text.strip():
            return False
        if root_truth.get("external_root_cause_text_hash") != _field_hash(root_text):
            return False
        if record.get("root_label") != root_truth.get("root_label"):
            return False
        if record.get("intervention_label") != intervention_truth.get("intervention_label"):
            return False
        intervention_field = intervention_truth.get("field_name")
        if intervention_field:
            if not isinstance(intervention_field, str) or intervention_field not in row:
                return False
            if intervention_truth.get("field_value_hash") != _field_hash(row[intervention_field]):
                return False
        intervention_text = intervention_truth.get("external_intervention_text")
        if not isinstance(intervention_text, str) or not intervention_text.strip():
            return False
        if intervention_truth.get("external_intervention_text_hash") != _field_hash(
            intervention_text
        ):
            return False
    return True


def _integrated_evaluator_passed(production_transfer: dict[str, Any]) -> bool:
    evaluator = production_transfer.get("integrated_evaluator") or {}
    counts = evaluator.get("query_row_counts") or {}
    neural_invocation = evaluator.get("neural_invocation") or {}
    metric_inputs = production_transfer.get("metric_inputs") or {}
    records = metric_inputs.get("prediction_records") or []
    expected_candidate_count = sum(
        int(record.get("xlog_candidate_count", 0))
        for record in records
        if isinstance(record, dict)
    )
    return (
        evaluator.get("uses_shared_bfo_kernel") is True
        and evaluator.get("emits_per_domain_predictions") is True
        and evaluator.get("consumes_neural_rankings") is True
        and int(counts.get("candidate_root_cause", 0)) > 0
        and int(counts.get("recommended_intervention", 0)) > 0
        and int(counts.get("bfo_explanation", 0)) > 0
        and neural_invocation.get("path") == "xlog_nn4_transfer"
        and neural_invocation.get("program_declares_nn4") is True
        and neural_invocation.get("transfer_forward_backward_loss_is_cuda") is True
        and neural_invocation.get("transfer_nn4_gradient_finite") is True
        and neural_invocation.get("ranking_argmax_device_resident") is True
        and neural_invocation.get("score_cpu_materialization_in_ranking") is False
        and neural_invocation.get("full_score_rows_materialized") is False
        and neural_invocation.get("scalar_item_calls_in_ranking") is False
        and neural_invocation.get("cpu_score_slices_in_ranking") is False
        and neural_invocation.get("post_ranking_evidence_serialization")
        == "selected_indices_only"
        and int(neural_invocation.get("nn4_query_count", 0)) >= expected_candidate_count
        and all(
            isinstance(record.get("neural_scores"), dict)
            and record["neural_scores"].get("materialized") is False
            for record in records
            if isinstance(record, dict)
        )
    )


def _explanation_records_passed(production_transfer: dict[str, Any]) -> bool:
    metric_inputs = production_transfer.get("metric_inputs") or {}
    records = metric_inputs.get("prediction_records") or []
    if not isinstance(records, list) or not records:
        return False
    required_types = {"root_cause", "intervention", "risk_state"}
    allowed_rules = {
        "candidate_root_cause/2",
        "recommended_intervention/2",
        "risk_state/2",
    }
    allowed_categories = {
        "quality",
        "process",
        "disposition",
        "role",
        "material_entity",
    }
    for record in records:
        if not isinstance(record, dict):
            return False
        explanations = record.get("bfo_explanations")
        if not isinstance(explanations, list) or not explanations:
            return False
        by_type: dict[str, dict[str, Any]] = {}
        for explanation in explanations:
            if not isinstance(explanation, dict):
                return False
            claim_type = str(explanation.get("claim_type", ""))
            by_type[claim_type] = explanation
            if explanation.get("valid") is not True:
                return False
            if not explanation.get("claim"):
                return False
            if explanation.get("case_id") != record.get("case_id"):
                return False
            if explanation.get("kernel_rule") not in allowed_rules:
                return False
            if explanation.get("bfo_category") not in allowed_categories:
                return False
            if not explanation.get("bfo_relation_family"):
                return False
            facts = explanation.get("supporting_facts")
            if not isinstance(facts, list) or not facts:
                return False
        if not required_types <= set(by_type):
            return False
        if by_type["root_cause"].get("claim") != record.get("root_prediction"):
            return False
        if by_type["root_cause"].get("bfo_category") != "quality":
            return False
        if by_type["intervention"].get("claim") != record.get("intervention_prediction"):
            return False
        if by_type["risk_state"].get("claim") != record.get("risk_state"):
            return False
    return True


def _leakage_audit_passed(production_transfer: dict[str, Any]) -> bool:
    audit = production_transfer.get("leakage_audit")
    if not isinstance(audit, dict):
        return False
    return (
        audit.get("passed") is True
        and int(audit.get("held_out_case_count", 0)) > 0
        and audit.get("metadata_gold_markers") == []
        and audit.get("binary_feature_gold_columns") == []
        and audit.get("candidate_order_index_leaks") is False
        and int(audit.get("true_candidate_index_count", 0)) > 1
        and audit.get("xlog_fact_symmetry") is True
    )


def _bundle_reuse_passed(production_transfer: dict[str, Any]) -> bool:
    bundle = production_transfer.get("bundle_reuse")
    if not isinstance(bundle, dict) or bundle.get("status") != "PASS":
        return False

    runtime_session = bundle.get("runtime_session_reuse") or {}
    language_contract = bundle.get("language_contract_reuse") or {}
    runtime_optimizer = bundle.get("runtime_optimizer_reuse") or {}
    runtime_session_transfer = runtime_session.get("hot_loop_transfer_stats") or {}
    runtime_optimizer_transfer = runtime_optimizer.get("hot_loop_transfer_stats") or {}
    cache_stats = runtime_optimizer.get("join_index_cache_stats") or {}
    reused_language_contract = set(language_contract.get("reused_artifacts") or [])

    return (
        runtime_session.get("status") == "PASS"
        and runtime_session.get("logic_program_compile") is True
        and runtime_session.get("session_evaluate") is True
        and float(runtime_session.get("relation_delta_equivalence_pct", 0.0)) >= 100.0
        and all(int(runtime_session_transfer.get(key, -1)) == 0 for key in ["dtoh_calls", "htod_calls", "dtoh_bytes", "htod_bytes"])
        and language_contract.get("status") == "PASS"
        and int(language_contract.get("feature_count", 0)) >= 10
        and "language completeness showcase" in reused_language_contract
        and runtime_optimizer.get("status") == "PASS"
        and runtime_optimizer.get("apply_relation_delta_batch") is True
        and int(cache_stats.get("builds", 0)) >= 1
        and int(cache_stats.get("hits", 0)) >= 1
        and int(runtime_optimizer.get("relation_callback_events", 0)) >= 2
        and runtime_optimizer.get("callback_payload_has_tensors") is False
        and all(int(runtime_optimizer_transfer.get(key, -1)) == 0 for key in ["dtoh_calls", "htod_calls", "dtoh_bytes", "htod_bytes"])
    )


def _production_transfer_passed(production_transfer: dict[str, Any]) -> bool:
    checksums = production_transfer.get("kernel_checksum_by_domain") or {}
    adapter_fact_only = production_transfer.get("adapter_fact_only_by_domain") or {}
    rule_evolution = production_transfer.get("rule_evolution") or {}
    neural = production_transfer.get("neural") or {}
    computed = production_transfer.get("computed_metrics") or {}
    generalization = _generalization_assessment(production_transfer)
    differentiable_inductive_logic = _differentiable_inductive_logic_assessment(production_transfer)
    gen_gates = generalization.get("gates") or {}
    gen_uplift = (
        (gen_gates.get("strong_baseline_uplift") or {})
        .get("baseline_uplift", {})
    )
    return (
        computed.get("valid") is True
        and production_transfer.get("status") == "PASS"
        and production_transfer.get("scope") == "production"
        and _source_provenance_passed(production_transfer)
        and _leakage_audit_passed(production_transfer)
        and _integrated_evaluator_passed(production_transfer)
        and _explanation_records_passed(production_transfer)
        and _bundle_reuse_passed(production_transfer)
        and differentiable_inductive_logic.get("status") == "PASS"
        and int(production_transfer.get("domain_count", 0)) >= 5
        and production_transfer.get("held_out_domain")
        and production_transfer.get("core_rule_edits_per_domain") == 0
        and rule_evolution.get("held_out_domain_excluded") is True
        and len(checksums) >= 5
        and len(set(checksums.values())) == 1
        and len(adapter_fact_only) >= 5
        and all(value is True for value in adapter_fact_only.values())
        and neural.get("program_declares_nn4") is True
        and neural.get("loss_is_cuda") is True
        and neural.get("gradient_finite") is True
        and neural.get("hand_weighted") is False
        and neural.get("trained_on_held_out_domain") is False
        and int(neural.get("processed_observation_count", 0)) >= 100_000
        and float(neural.get("ranking_accuracy", 0.0)) >= 0.999
        and float(computed.get("held_out_root_cause_f1", 0.0)) >= 0.90
        and float(computed.get("accepted_intervention_precision", 0.0)) >= 0.95
        and float(computed.get("explanations_complete_pct", 0.0)) >= 100.0
        and bool((gen_gates.get("strong_baseline_uplift") or {}).get("passed"))
        and gen_uplift.get("beats_strongest_baseline") is True
        and float(gen_uplift.get("relative_uplift_over_best_baseline_pct", 0.0)) >= 15.0
    )


def _production_scale_passed(production_transfer: dict[str, Any]) -> bool:
    profile = production_transfer.get("scale_profile") or {}
    return (
        profile.get("scale_source") == "hf_case_amplification"
        and profile.get("synthetic_numeric_only") is False
        and int(profile.get("hf_seed_case_count", 0)) > 0
        and int(profile.get("real_hf_transfer_case_count", 0))
        >= int(profile.get("neural_observation_count", 0))
        and int(profile.get("symbolic_bfo_fact_count", 0)) >= 1_000_000
        and int(profile.get("neural_observation_count", 0)) >= 100_000
        and int(profile.get("entity_count", 0)) >= 50_000
        and int(profile.get("staged_delta_update_count", 0)) >= 10_000
        and float(profile.get("p95_core_indexed_query_latency_ms", float("inf"))) <= 50.0
    )


def _production_soak_passed(production_transfer: dict[str, Any]) -> bool:
    soak = production_transfer.get("soak") or {}
    return (
        float(soak.get("duration_sec", 0.0)) >= 1800.0
        and float(soak.get("gpu_memory_drift_pct", float("inf"))) <= 2.0
        and soak.get("relation_growth_bounded") is True
    )


def _promoted_rule_quality_passed(production_transfer: dict[str, Any]) -> bool:
    quality = (production_transfer.get("computed_metrics") or {}).get("promoted_rule_quality") or {}
    return (
        float(quality.get("precision", 0.0)) >= 0.98
        and float(quality.get("recall", 0.0)) >= 0.95
        and float(quality.get("f1", 0.0)) >= 0.965
        and quality.get("kernel_mutated") is False
    )


def _invalid_cross_domain_passed(production_transfer: dict[str, Any]) -> bool:
    computed = production_transfer.get("computed_metrics") or {}
    return (
        computed.get("valid") is True
        and float(computed.get("invalid_cross_domain_rejection_pct", 0.0)) >= 100.0
    )


def _control_plane_passed(production_transfer: dict[str, Any]) -> bool:
    profile = production_transfer.get("scale_profile") or {}
    return int(profile.get("control_plane_metadata_bytes_per_hot_iteration", 999_999)) <= 4096


def _run_shipped_xlog_programs(strict: bool) -> dict[str, Any]:
    programs = sorted((ROOT / "programs").glob("*.xlog"))
    if not strict:
        return {
            "status": "SKIP",
            "program_count": len(programs),
            "programs": [str(path.relative_to(ROOT)) for path in programs],
            "reason": "strict validation disabled",
        }
    repo_root = ROOT.parents[2]
    results: list[dict[str, Any]] = []
    for program in programs:
        relative = program.relative_to(repo_root)
        cuda_oom_retries = 0
        while True:
            try:
                proc = subprocess.run(
                    [
                        "cargo",
                        "run",
                        "-q",
                        "-p",
                        "xlog-cli",
                        "--",
                        "run",
                        "--memory-mb",
                        "128",
                        str(relative),
                    ],
                    cwd=repo_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=120,
                )
            except Exception as exc:
                results.append(
                    {
                        "program": str(relative),
                        "status": "FAIL",
                        "returncode": None,
                        "stderr": f"{type(exc).__name__}: {exc}",
                        "stdout": "",
                        "cuda_oom_retries": cuda_oom_retries,
                    }
                )
                break
            if (
                proc.returncode != 0
                and cuda_oom_retries < 3
                and _cuda_oom_text(proc.stdout, proc.stderr)
            ):
                cuda_oom_retries += 1
                time.sleep(1.0 + cuda_oom_retries)
                continue
            results.append(
                {
                    "program": str(relative),
                    "status": "PASS" if proc.returncode == 0 else "FAIL",
                    "returncode": proc.returncode,
                    "stderr": proc.stderr[-4000:],
                    "stdout": proc.stdout[-4000:],
                    "cuda_oom_retries": cuda_oom_retries,
                }
            )
            break
    return {
        "status": "PASS" if results and all(result["status"] == "PASS" for result in results) else "FAIL",
        "program_count": len(programs),
        "programs": results,
    }


def _gate(status: bool, requirement_id: str, description: str, evidence: str) -> Gate:
    return Gate(requirement_id, description, "PASS" if status else "FAIL", evidence)


def _build_gates(
    *,
    strict: bool,
    gpu_required: bool,
    runtime: dict[str, Any],
    kernel: dict[str, Any],
    domains: dict[str, Any],
    neural: dict[str, Any],
    bfo_fixture: dict[str, Any],
    ablation: dict[str, Any],
    runtime_contract: dict[str, Any],
    production_transfer: dict[str, Any],
    shipped_xlog_programs: dict[str, Any],
    generalization: dict[str, Any],
) -> list[Gate]:
    neural_passed = (
        neural.get("status") == "PASS"
        and neural.get("program_declares_nn4") is True
        and neural.get("loss_is_cuda") is True
        and neural.get("gradient_finite") is True
        and neural.get("ranking_changed") is True
    )
    production_transfer_passed = _production_transfer_passed(production_transfer)
    production_scale_passed = _production_scale_passed(production_transfer)
    production_soak_passed = _production_soak_passed(production_transfer)
    promoted_rule_quality_passed = _promoted_rule_quality_passed(production_transfer)
    invalid_cross_domain_passed = _invalid_cross_domain_passed(production_transfer)
    control_plane_passed = _control_plane_passed(production_transfer)
    bundle_reuse_passed = _bundle_reuse_passed(production_transfer)
    computed = production_transfer.get("computed_metrics") or {}
    transfer_stats = runtime_contract.get("hot_loop_transfer_stats") or {}
    runtime_contract_passed = runtime_contract.get("status") == "PASS"
    gen_gates = generalization.get("gates") or {}
    differentiable_inductive_logic = _differentiable_inductive_logic_assessment(production_transfer)
    differentiable_inductive_logic_gates = differentiable_inductive_logic.get("gates") or {}
    public_benchmark = _public_benchmark_assessment(production_transfer)
    showcase_metrics = computed.get("showcase_metrics") or {}
    zero_hot_loop_transfers = runtime_contract_passed and all(
        int(transfer_stats.get(key, -1)) == 0
        for key in ["dtoh_calls", "htod_calls", "dtoh_bytes", "htod_bytes"]
    )
    gates = [
        _gate(
            strict and gpu_required,
            "authoritative_validation_command",
            "`./validate.sh --strict --gpu-required` is the authoritative gate.",
            f"strict={strict}, gpu_required={gpu_required}",
        ),
        _gate(
            (not gpu_required) or bool(runtime["torch"]["cuda_available"]),
            "cuda_required",
            "CUDA is mandatory when `--gpu-required` is set.",
            json.dumps(runtime["torch"], sort_keys=True),
        ),
        _gate(
            bool(runtime["torch"]["available"]),
            "pytorch_available",
            "PyTorch is mandatory.",
            json.dumps(runtime["torch"], sort_keys=True),
        ),
        _gate(
            bool(runtime["pyxlog"]["available"]),
            "pyxlog_available",
            "`pyxlog` is mandatory.",
            json.dumps(runtime["pyxlog"], sort_keys=True),
        ),
        _gate(
            neural_passed,
            "real_cuda_neural_bridge",
            "A real CUDA PyTorch model must be registered and invoked through XLOG `nn/4`.",
            json.dumps(
                {
                    "status": neural.get("status"),
                    "program_declares_nn4": neural.get("program_declares_nn4"),
                    "loss_is_cuda": neural.get("loss_is_cuda"),
                    "gradient_finite": neural.get("gradient_finite"),
                    "ranking_changed": neural.get("ranking_changed"),
                    "output": neural.get("output"),
                    "evidence": neural.get("evidence"),
                },
                sort_keys=True,
            ),
        ),
        _gate(
            shipped_xlog_programs.get("status") == "PASS",
            "shipped_xlog_programs_run",
            "Every shipped XLOG program in `programs/` runs through `xlog-cli run`.",
            json.dumps(shipped_xlog_programs, sort_keys=True),
        ),
        _gate(
            bool(kernel["exists"])
            and kernel["bfo_category_count"] >= 12
            and kernel["bfo_relation_family_count"] >= 8,
            "bfo_kernel_coverage",
            "BFO kernel declares at least 12 upper categories and 8 relation families.",
            json.dumps(kernel, sort_keys=True),
        ),
        _gate(
            bool(domains["exists"])
            and domains["domain_count"] >= 5
            and bool(domains["all_classes_mapped"])
            and domains["domains_with_required_fixtures"] >= 5,
            "domain_inventory_coverage",
            "At least five domains exist and every domain class maps to BFO with required fixture families.",
            json.dumps(domains, sort_keys=True),
        ),
        _gate(
            domains["max_adapter_core_rule_ratio"] is not None
            and domains["max_adapter_core_rule_ratio"] <= 0.25,
            "thin_domain_adapters",
            "Adapter/core rule ratio is <= 0.25 per domain.",
            json.dumps({"max_adapter_core_rule_ratio": domains["max_adapter_core_rule_ratio"]}, sort_keys=True),
        ),
        _gate(
            invalid_cross_domain_passed,
            "invalid_cross_domain_rejection",
            "100% of invalid cross-domain fixtures are rejected or inconsistent.",
            json.dumps(
                {
                    "invalid_cross_domain_rejection_pct": production_transfer.get(
                        "invalid_cross_domain_rejection_pct"
                    ),
                    "computed_invalid_cross_domain_rejection_pct": computed.get(
                        "invalid_cross_domain_rejection_pct"
                    ),
                    "output": production_transfer.get("output"),
                    "evidence": production_transfer.get("evidence"),
                    "computed_failure": computed.get("failure"),
                },
                sort_keys=True,
            ),
        ),
        _gate(
            bool(domains["holdout_domain"]),
            "heldout_domain_exists",
            "At least one domain is held out during rule evolution.",
            json.dumps({"holdout_domain": domains["holdout_domain"]}, sort_keys=True),
        ),
        _gate(
            production_transfer_passed,
            "heldout_root_cause_quality",
            "Held-out root-cause F1 is >= 0.90.",
            json.dumps(
                {
                    "scope": production_transfer.get("scope"),
                    "held_out_domain": production_transfer.get("held_out_domain"),
                    "held_out_root_cause_f1": computed.get("held_out_root_cause_f1"),
                    "held_out_root_cause_confusion": computed.get(
                        "held_out_root_cause_confusion"
                    ),
                    "output": production_transfer.get("output"),
                    "evidence": production_transfer.get("evidence"),
                    "computed_failure": computed.get("failure"),
                    "smoke_only_fixture": bfo_fixture.get("output"),
                },
                sort_keys=True,
            ),
        ),
        _gate(
            production_transfer_passed,
            "accepted_intervention_precision",
            "Accepted intervention precision is >= 0.95.",
            json.dumps(
                {
                    "scope": production_transfer.get("scope"),
                    "accepted_intervention_precision": computed.get(
                        "accepted_intervention_precision"
                    ),
                    "intervention_confusion": computed.get("intervention_confusion"),
                    "output": production_transfer.get("output"),
                    "evidence": production_transfer.get("evidence"),
                    "computed_failure": computed.get("failure"),
                    "smoke_only_fixture": bfo_fixture.get("output"),
                },
                sort_keys=True,
            ),
        ),
        _gate(
            production_transfer_passed,
            "neuro_symbolic_baseline_uplift",
            "Neuro-symbolic uplift is >= 15% over the strongest baseline.",
            json.dumps(
                {
                    "scope": production_transfer.get("scope"),
                    "status": production_transfer.get("status"),
                    "showcase_baseline_metrics": showcase_metrics.get("baseline_metrics"),
                    "showcase_relative_uplift_over_best_baseline_pct": showcase_metrics.get(
                        "relative_uplift_over_best_baseline_pct"
                    ),
                    "showcase_strongest_baseline": showcase_metrics.get("strongest_baseline"),
                    "generalization_baseline_uplift": (
                        (gen_gates.get("strong_baseline_uplift") or {}).get("baseline_uplift")
                    ),
                    "output": production_transfer.get("output"),
                    "evidence": production_transfer.get("evidence"),
                    "computed_failure": computed.get("failure"),
                    "smoke_only_ablation": ablation.get("output"),
                },
                sort_keys=True,
            ),
        ),
        *[
            _gate(
                bool((gen_gates.get(requirement_id) or {}).get("passed")),
                requirement_id,
                description,
                json.dumps(gen_gates.get(requirement_id) or {}, sort_keys=True),
            )
            for requirement_id, description in GENERALIZATION_REQUIREMENTS.items()
        ],
        *[
            _gate(
                bool((differentiable_inductive_logic_gates.get(requirement_id) or {}).get("passed")),
                requirement_id,
                description,
                json.dumps(differentiable_inductive_logic_gates.get(requirement_id) or {}, sort_keys=True),
            )
            for requirement_id, description in DIFFERENTIABLE_INDUCTIVE_LOGIC_REQUIREMENTS.items()
        ],
        _gate(
            public_benchmark["passed"],
            "public_benchmark_claim_coverage",
            "Public benchmark state is explicit; external state-of-the-art claims require required-family coverage.",
            json.dumps(public_benchmark, sort_keys=True),
        ),
        _gate(
            bundle_reuse_passed,
            "merged_runtime_bundle_reuse",
            "The production transfer runner reuses the merged runtime, language-contract, and optimizer bundle through executable probes.",
            json.dumps(
                {
                    "bundle_reuse": production_transfer.get("bundle_reuse"),
                    "output": production_transfer.get("output"),
                    "evidence": production_transfer.get("evidence"),
                },
                sort_keys=True,
            ),
        ),
        _gate(
            promoted_rule_quality_passed,
            "promoted_rule_quality",
            "Promoted rule quality on non-held-out domains meets precision/recall/F1 thresholds without mutating the BFO kernel.",
            json.dumps(
                {
                    "promoted_rule_quality": computed.get("promoted_rule_quality"),
                    "output": production_transfer.get("output"),
                    "evidence": production_transfer.get("evidence"),
                    "computed_failure": computed.get("failure"),
                },
                sort_keys=True,
            ),
        ),
        _gate(
            zero_hot_loop_transfers,
            "device_resident_hot_loop",
            "Hot-loop data-plane device-to-host and host-to-device transfers after initial load are 0.",
            json.dumps(
                {
                    "hot_loop_transfer_stats": transfer_stats,
                    "output": runtime_contract.get("output"),
                    "evidence": runtime_contract.get("evidence"),
                },
                sort_keys=True,
            ),
        ),
        _gate(
            control_plane_passed,
            "control_plane_metadata_budget",
            "Control-plane metadata per hot iteration is <= 4096 bytes.",
            json.dumps(
                {
                    "control_plane_metadata_bytes_per_hot_iteration": (
                        production_transfer.get("scale_profile") or {}
                    ).get("control_plane_metadata_bytes_per_hot_iteration"),
                    "output": production_transfer.get("output"),
                },
                sort_keys=True,
            ),
        ),
        _gate(
            production_scale_passed,
            "production_scale_latency",
            "Production scale and p95 latency requirements are met.",
            json.dumps(
                {
                    "scale_profile": production_transfer.get("scale_profile"),
                    "output": production_transfer.get("output"),
                    "evidence": production_transfer.get("evidence"),
                },
                sort_keys=True,
            ),
        ),
        _gate(
            production_soak_passed,
            "production_soak_stability",
            "Soak run is >=30 minutes with GPU memory drift <=2%.",
            json.dumps(
                {
                    "soak": production_transfer.get("soak"),
                    "output": production_transfer.get("output"),
                    "evidence": production_transfer.get("evidence"),
                },
                sort_keys=True,
            ),
        ),
    ]
    return gates


def _metric(
    qid: str,
    question: str,
    metric: str,
    required: str,
    actual: Any,
    status: str,
    evidence: str,
) -> dict[str, Any]:
    return {
        "id": qid,
        "question": question,
        "metric": metric,
        "required": required,
        "actual": actual,
        "status": status,
        "evidence": evidence,
    }


def _gqm_metrics(
    *,
    kernel: dict[str, Any],
    domains: dict[str, Any],
    neural: dict[str, Any],
    bfo_fixture: dict[str, Any],
    ablation: dict[str, Any],
    runtime_contract: dict[str, Any],
    production_transfer: dict[str, Any],
) -> list[dict[str, Any]]:
    max_ratio = domains["max_adapter_core_rule_ratio"]
    neural_q4_passed = (
        neural.get("status") == "PASS"
        and neural.get("program_declares_nn4") is True
        and neural.get("loss_is_cuda") is True
        and neural.get("ranking_changed") is True
    )
    production_transfer_passed = _production_transfer_passed(production_transfer)
    production_scale_passed = _production_scale_passed(production_transfer)
    computed = production_transfer.get("computed_metrics") or {}
    generalization = _generalization_assessment(production_transfer)
    gen_uplift = (
        ((generalization.get("gates") or {}).get("strong_baseline_uplift") or {}).get(
            "baseline_uplift",
            {},
        )
    )
    runtime_contract_passed = runtime_contract.get("status") == "PASS"
    transfer_stats = runtime_contract.get("hot_loop_transfer_stats") or {}
    zero_hot_loop_transfer_count = sum(int(transfer_stats.get(key, -1)) for key in transfer_stats)
    determinism = runtime_contract.get("determinism") or {}
    return [
        _metric("Q1", "Is the BFO core unchanged across domains?", "Core rule edits per domain", "0", 0 if kernel["exists"] else None, "PASS" if kernel["exists"] else "FAIL", "single shared kernel checksum"),
        _metric("Q2", "Are enough domains represented?", "Domain adapters", ">= 5", domains["domain_count"], "PASS" if domains["domain_count"] >= 5 else "FAIL", "domains/domain_inventory.json"),
        _metric("Q3", "Are adapters thin?", "Adapter/core rule ratio", "<= 0.25 per domain", max_ratio, "PASS" if max_ratio is not None and max_ratio <= 0.25 else "FAIL", "domains/domain_inventory.json"),
        _metric("Q4", "Is neural evidence real and causally used?", "CUDA `nn/4` model affects rankings", "Yes", "ranking_changed" if neural_q4_passed else None, "PASS" if neural_q4_passed else "FAIL", str(neural.get("output") or "neural runner not implemented")),
        _metric("Q5", "Does root-cause inference transfer?", "Held-out domain root-cause F1", ">= 0.90", computed.get("held_out_root_cause_f1") if production_transfer_passed else None, "PASS" if production_transfer_passed and float(computed.get("held_out_root_cause_f1", 0.0)) >= 0.90 else "FAIL", str(production_transfer.get("output") or "production held-out transfer evidence missing")),
        _metric("Q6", "Are interventions useful?", "Accepted intervention precision", ">= 0.95", computed.get("accepted_intervention_precision") if production_transfer_passed else None, "PASS" if production_transfer_passed and float(computed.get("accepted_intervention_precision", 0.0)) >= 0.95 else "FAIL", str(production_transfer.get("output") or "production intervention evidence missing")),
        _metric("Q7", "Are explanations complete?", "Top-level claims with BFO explanation", "100%", computed.get("explanations_complete_pct") if production_transfer_passed else None, "PASS" if production_transfer_passed and float(computed.get("explanations_complete_pct", 0.0)) >= 100.0 else "FAIL", str(production_transfer.get("output") or "production explanation evidence missing")),
        _metric("Q8", "Does neuro-symbolic beat baselines?", "Generalization relative uplift over best baseline", ">= 15%", gen_uplift.get("relative_uplift_over_best_baseline_pct") if production_transfer_passed else None, "PASS" if production_transfer_passed and gen_uplift.get("beats_strongest_baseline") is True and float(gen_uplift.get("relative_uplift_over_best_baseline_pct", 0.0)) >= 15.0 else "FAIL", str(production_transfer.get("output") or "production ablations missing")),
        _metric("Q9", "Is online adaptation exact?", "Delta output equals full recompute", "100%", runtime_contract.get("delta_output_equals_full_recompute_pct") if runtime_contract_passed else None, "PASS" if runtime_contract_passed and float(runtime_contract.get("delta_output_equals_full_recompute_pct", 0.0)) >= 100.0 else "FAIL", str(runtime_contract.get("output") or "delta validation not implemented")),
        _metric("Q10", "Is the hot path device-resident?", "Device-to-host and host-to-device transfers after initial load", "0", 0 if runtime_contract_passed and zero_hot_loop_transfer_count == 0 else None, "PASS" if runtime_contract_passed and zero_hot_loop_transfer_count == 0 else "FAIL", str(runtime_contract.get("output") or "transfer counters not captured")),
        _metric("Q11", "Is the result deterministic?", "Fixed-seed byte-identical runs", "5/5", f"{determinism.get('matching_runs')}/{determinism.get('runs')}" if runtime_contract_passed else None, "PASS" if runtime_contract_passed and determinism.get("byte_identical") is True and determinism.get("matching_runs") == 5 and determinism.get("runs") == 5 else "FAIL", str(runtime_contract.get("output") or "five-run determinism not implemented")),
        _metric("Q12", "Is performance production-grade?", "p95 core query latency", "<= 50 ms", (production_transfer.get("scale_profile") or {}).get("p95_core_indexed_query_latency_ms") if production_scale_passed else None, "PASS" if production_scale_passed else "FAIL", str(production_transfer.get("output") or "production profile missing")),
    ]


def _summary(args: argparse.Namespace, elapsed_sec: float) -> dict[str, Any]:
    runtime = _runtime_details()
    kernel = _kernel_facts(ROOT / "bfo" / "kernel.xlog")
    domains = _domain_facts(ROOT / "domains" / "domain_inventory.json")
    neural = _run_neural_smoke(args.strict and args.gpu_required)
    bfo_fixture = _run_bfo_fixture_smoke(args.strict and args.gpu_required)
    ablation = _run_ablation_smoke(args.strict and args.gpu_required)
    runtime_contract = _run_runtime_contract_smoke(args.strict and args.gpu_required)
    shipped_xlog_programs = _run_shipped_xlog_programs(args.strict)
    production_transfer = _load_production_transfer_evidence(args.production_transfer)
    generalization = _generalization_assessment(production_transfer)
    differentiable_inductive_logic = _differentiable_inductive_logic_assessment(production_transfer)
    public_benchmark = _public_benchmark_assessment(production_transfer)
    gates = _build_gates(
        strict=args.strict,
        gpu_required=args.gpu_required,
        runtime=runtime,
        kernel=kernel,
        domains=domains,
        neural=neural,
        bfo_fixture=bfo_fixture,
        ablation=ablation,
        runtime_contract=runtime_contract,
        production_transfer=production_transfer,
        shipped_xlog_programs=shipped_xlog_programs,
        generalization=generalization,
    )
    gate_payload = [gate.to_json() for gate in gates]
    blockers = [gate.to_json() for gate in gates if gate.status == "FAIL"]
    metrics = _gqm_metrics(
        kernel=kernel,
        domains=domains,
        neural=neural,
        bfo_fixture=bfo_fixture,
        ablation=ablation,
        runtime_contract=runtime_contract,
        production_transfer=production_transfer,
    )
    metric_failures = [
        {
            "requirement_id": metric["id"],
            "description": metric["question"],
            "status": metric["status"],
            "evidence": metric["evidence"],
        }
        for metric in metrics
        if metric["status"] == "FAIL"
    ]
    status = "PASS" if not blockers and not metric_failures else "FAIL"
    argv = ["validate.sh", *sys.argv[1:]]
    return {
        "schema_version": 1,
        "example": "BFO Universal Case Reasoner",
        "status": status,
        "strict": bool(args.strict),
        "gpu_required": bool(args.gpu_required),
        "branch": _run_git(["branch", "--show-current"]),
        "git_sha": _run_git(["rev-parse", "HEAD"]),
        "commands": [
            {
                "argv": argv,
                "cwd": str(ROOT),
                "duration_sec": round(elapsed_sec, 6),
            }
        ],
        "runtime": runtime,
        "artifacts": {
            "kernel": kernel,
            "domains": domains,
            "neural_smoke": neural,
            "bfo_fixture_smoke": bfo_fixture,
            "ablation_smoke": ablation,
            "runtime_contract_smoke": runtime_contract,
            "shipped_xlog_programs": shipped_xlog_programs,
            "production_transfer": production_transfer,
            "generalization_assessment": generalization,
            "differentiable_inductive_logic_assessment": differentiable_inductive_logic,
            "public_benchmark_assessment": public_benchmark,
            "validation_plan": str(ROOT / "VALIDATION_PLAN.md"),
        },
        "gqm_metrics": metrics,
        "production_blocking_gates": gate_payload,
        "raw_output_paths": [
            path
            for path in [
                neural.get("output"),
                bfo_fixture.get("output"),
                ablation.get("output"),
                runtime_contract.get("output"),
                production_transfer.get("output"),
            ]
            if path
            and (
                (path == neural.get("output") and neural.get("status") == "PASS")
                or (path == bfo_fixture.get("output") and bfo_fixture.get("status") == "PASS")
                or (path == ablation.get("output") and ablation.get("status") == "PASS")
                or (
                    path == runtime_contract.get("output")
                    and runtime_contract.get("status") == "PASS"
                )
                or (
                    path == production_transfer.get("output")
                    and production_transfer.get("status") == "PASS"
                )
            )
        ],
        "explanation_paths": [production_transfer["output"]]
        if production_transfer.get("status") == "PASS"
        else [],
        "blockers": blockers + metric_failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Enable strict production gate behavior.")
    parser.add_argument("--gpu-required", action="store_true", help="Treat missing CUDA as a production-blocking failure.")
    parser.add_argument(
        "--production-transfer",
        type=Path,
        default=ROOT / "evidence" / "production_transfer.json",
        help="Production transfer/profile/soak evidence JSON path.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Summary JSON output path.")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    summary = _summary(args, 0.0)
    summary["commands"][0]["duration_sec"] = round(time.perf_counter() - start, 6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for blocker in summary["blockers"]:
        print(f"BLOCKER {blocker['requirement_id']}: {blocker['evidence']}")
    print(json.dumps({"status": summary["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
