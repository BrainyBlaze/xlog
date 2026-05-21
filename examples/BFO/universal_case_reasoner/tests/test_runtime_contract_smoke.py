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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for runtime contract smoke")
def test_runtime_contract_smoke_proves_delta_device_and_determinism(tmp_path: Path) -> None:
    output = tmp_path / "runtime_contract_smoke.json"

    retries = 0
    while True:
        proc = subprocess.run(
            [
                "python",
                str(ROOT / "tools" / "run_runtime_contract_smoke.py"),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        if proc.returncode == 0 or retries >= 1 or not _cuda_oom_text(proc.stdout, proc.stderr):
            break
        retries += 1
        time.sleep(1.0)

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["status"] == "PASS"
    assert payload["delta_output_equals_full_recompute_pct"] == 100.0
    assert payload["determinism"]["byte_identical"] is True
    assert payload["determinism"]["runs"] == 5
    assert payload["determinism"]["matching_runs"] == 5
    assert payload["hot_loop_transfer_stats"]["dtoh_calls"] == 0
    assert payload["hot_loop_transfer_stats"]["htod_calls"] == 0
    assert payload["hot_loop_transfer_stats"]["dtoh_bytes"] == 0
    assert payload["hot_loop_transfer_stats"]["htod_bytes"] == 0
