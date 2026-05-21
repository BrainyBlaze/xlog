from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _cuda_oom_text(stdout: str, stderr: str) -> bool:
    text = f"{stdout}\n{stderr}".lower()
    return "cuda_error_out_of_memory" in text or "out of memory" in text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the BFO fixture smoke")
def test_bfo_fixture_smoke_runs_shared_kernel_across_five_domains(tmp_path: Path) -> None:
    output = tmp_path / "bfo_fixture_smoke.json"

    retries = 0
    while True:
        proc = subprocess.run(
            [
                "python",
                str(ROOT / "tools" / "run_bfo_fixture_smoke.py"),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=45,
        )
        if proc.returncode == 0 or retries >= 1 or not _cuda_oom_text(proc.stdout, proc.stderr):
            break
        retries += 1
        time.sleep(1.0)

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["status"] == "PASS"
    assert payload["domain_count"] == 5
    assert payload["held_out_domain"] == "cybersecurity_intrusion"
    assert payload["core_rule_edits_per_domain"] == 0
    assert payload["held_out_root_cause_f1"] == 1.0
    assert payload["accepted_intervention_precision"] == 1.0
    assert payload["explanations_complete_pct"] == 100.0
    assert payload["query_row_counts"]["candidate_root_cause"] == 5
    assert payload["query_row_counts"]["recommended_intervention"] == 5
    assert payload["query_row_counts"]["bfo_explanation"] == 5
    assert payload["query_tensors_cuda"] is True
