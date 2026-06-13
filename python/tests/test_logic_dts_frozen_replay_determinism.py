"""external consumer frozen-bundle replay regression for pyxlog LogicProgram sessions."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()


FROZEN_ROOT = Path(
    os.environ.get(
        "XLOG_DTS_FROZEN_ROOT",
        "/home/dev/projects/dts-dlm/out/m34-strata/2026-04-26-m34-sweep",
    )
)
FALLBACK_BUNDLE = Path(__file__).parent / "fixtures" / "dts_m34_frozen_inputs.json"

REPLAY_CODE = r"""
import hashlib
import json
import sys

import numpy as np
import pyxlog
import torch

bundle = json.load(open(sys.argv[1]))
program = pyxlog.LogicProgram.compile(bundle["compile_source"], device=0, memory_mb=512)
session = program.session()

for upload in bundle["relation_uploads"]:
    cols = [
        torch.tensor(col, dtype=torch.int64, device="cuda").contiguous()
        for col in upload["columns"]
    ]
    session.put_relation(upload["name"], cols)

result = session.evaluate()
fingerprints = []
for query in result.queries:
    cols = [
        torch.from_dlpack(t).cpu().numpy().astype(np.int64, copy=False)
        for t in query.tensors
    ]
    if cols:
        rows = np.stack(cols, axis=1)
        sort_index = np.lexsort(tuple(rows[:, i] for i in reversed(range(rows.shape[1]))))
        sorted_rows = rows[sort_index]
    else:
        rows = np.zeros((query.num_rows, 0), dtype=np.int64)
        sorted_rows = rows
    fingerprints.append(
        (
            int(query.num_rows),
            hashlib.sha256(rows.tobytes()).hexdigest(),
            hashlib.sha256(sorted_rows.tobytes()).hexdigest(),
        )
    )

print(repr(fingerprints))
"""


@pytest.mark.parametrize("call_id", [28, 29, 30, 31])
def test_dts_frozen_logic_replay_is_deterministic_across_subprocesses(call_id: int):
    bundle = FROZEN_ROOT / f"stratum_call_{call_id}" / "frozen_inputs.json"
    if not bundle.exists():
        bundle = FALLBACK_BUNDLE
    if not bundle.exists():
        pytest.skip(f"external consumer frozen replay bundle is not available: {bundle}")

    expected = None
    for replay_idx in range(20):
        completed = subprocess.run(
            [sys.executable, "-c", REPLAY_CODE, str(bundle)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr
        current = ast.literal_eval(completed.stdout.strip())
        if expected is None:
            expected = current
        assert current == expected, (
            f"stratum_call_{call_id} replay {replay_idx} diverged:\n"
            f"expected={expected}\n"
            f"actual={current}"
        )
