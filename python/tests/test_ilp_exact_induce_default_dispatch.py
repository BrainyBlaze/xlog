"""Default exact-induction dispatch policy.

The public ``induce_exact`` entry point must default to the native backend so
unqualified production callers do not accidentally enter the host-orchestrated
Python reference scorer.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load_exact_induce_module():
    repo = Path(__file__).resolve().parents[2]
    module_path = repo / "crates/pyxlog/python/pyxlog/ilp/exact_induce.py"
    spec = importlib.util.spec_from_file_location(
        "pyxlog_ilp_exact_induce_under_test",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _NativeOnlyProgram:
    def __init__(self):
        self.calls = []

    def induce_exact_native(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "candidates": [],
            "total_scored": 0,
            "candidate_count": 0,
            "positive_count": 0,
            "negative_count": 0,
        }

    def ilp_schema_size(self):
        raise AssertionError("default induce_exact must not use the Python scorer")


def test_induce_exact_default_dispatches_native_backend():
    exact_induce = _load_exact_induce_module()
    prog = _NativeOnlyProgram()
    positive_arg0 = torch.empty(0, dtype=torch.int64)
    positive_arg1 = torch.empty(0, dtype=torch.int64)

    result = exact_induce.induce_exact(
        prog,
        head_relation="target",
        candidate_relations=["edge"],
        positive_arg0=positive_arg0,
        positive_arg1=positive_arg1,
    )

    assert len(prog.calls) == 1
    call = prog.calls[0]
    assert call["head_relation"] == "target"
    assert call["candidate_relations"] == ["edge"]
    assert call["positive_arg0"] is positive_arg0
    assert call["positive_arg1"] is positive_arg1
    assert call["negative_arg0"] is None
    assert call["negative_arg1"] is None
    assert call["k_per_topology"] == 2
    assert call["deterministic"] is True
    assert result.total_scored == 0


def test_induce_exact_python_reference_requires_explicit_opt_in():
    exact_induce = _load_exact_induce_module()
    prog = _NativeOnlyProgram()
    positive_arg0 = torch.empty(0, dtype=torch.int64)
    positive_arg1 = torch.empty(0, dtype=torch.int64)

    with pytest.raises(Exception, match="XLOG_ALLOW_PYTHON_ILP_REFERENCE"):
        exact_induce.induce_exact(
            prog,
            head_relation="target",
            candidate_relations=["edge"],
            positive_arg0=positive_arg0,
            positive_arg1=positive_arg1,
            backend="python",
        )
