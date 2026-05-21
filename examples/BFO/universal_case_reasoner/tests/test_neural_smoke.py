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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the strict neural smoke")
def test_neural_smoke_invokes_real_cuda_model_through_xlog_nn4(tmp_path: Path) -> None:
    output = tmp_path / "neural_smoke.json"

    retries = 0
    while True:
        proc = subprocess.run(
            [
                "python",
                str(ROOT / "tools" / "run_neural_smoke.py"),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0 or retries >= 1 or not _cuda_oom_text(proc.stdout, proc.stderr):
            break
        retries += 1
        time.sleep(1.0)

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["status"] == "PASS"
    assert payload["program_declares_nn4"] is True
    assert payload["loss_is_cuda"] is True
    assert payload["gradient_finite"] is True
    assert payload["initial_top_label"] == "high"
    assert payload["flipped_top_label"] == "low"
    assert payload["ranking_changed"] is True
