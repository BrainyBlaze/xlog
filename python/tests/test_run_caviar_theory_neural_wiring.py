"""Unit tests for `run_caviar_theory._induce_neural_theory_for_target`'s
``min_fit`` parameter -- CPU-only, via injected FAKE ``kfold_select``/
``train_engine_mode`` closures.

WHY FAKES WORK HERE. `_induce_neural_theory_for_target` takes `torch`,
`kfold_select`, `train_engine_mode`, and every other engine dependency as
PLAIN PARAMETERS (dependency injection, not a module-level import) --
exactly so its own control-flow wiring (does it forward `min_fit` to
`kfold_select` correctly? does it skip `train_engine_mode` entirely on an
honest abstain?) is testable without a CUDA device or a compiled `pyxlog`
program at all. The REAL `kfold_select`/`train_engine_mode` (DLPack over
`device='cuda'` tensors) are pod-verified, out of scope here -- see
`run_caviar_cv.py`'s own module docstring for that split.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "caviar_woled"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import run_caviar_theory  # noqa: E402


class _FakeSelection:
    """Mirrors `relational_search.BodySelection`'s shape enough for
    `theory_loop.induce_theory` (it only ever reads `.rule`, and `.margin`
    via `getattr`)."""

    def __init__(self, rule):
        self.rule = rule
        self.margin = 0.1


class _FakeNetModule(torch.nn.Module):
    """A deterministic [N, 2] stand-in network: every row scores 0.9 on
    label 1 -- never actually reached by these tests (every fake
    `kfold_select` below abstains immediately), kept only so `make_network`
    has something real to hand back if the wiring ever DID reach it."""

    def forward(self, x):
        n = x.shape[0]
        return torch.tensor([[0.1, 0.9]] * n, dtype=torch.float32)


def _base_args():
    return SimpleNamespace(k=2, seed=7, steps=5, tie_tolerance=None, max_clauses=2, min_new_covered=1)


def _never_called_train_engine_mode(*a, **kw):
    raise AssertionError(
        "train_engine_mode must never be called here: every fake "
        "kfold_select in this file abstains (selection.rule is None) on "
        "its very first call, so theory_loop.induce_theory stops before "
        "any clause is ever committed."
    )


def test_min_fit_defaults_to_0_75_forwarded_to_kfold_select():
    captured = {}

    def fake_kfold_select(prog_factory, mask_name, facts, is_positive, make_network,
                           features, *, neural_relations, folds, seed, steps, topology,
                           tie_tolerance, min_fit=0.75):
        captured["min_fit"] = min_fit
        return _FakeSelection(rule=None)

    theory, nets = run_caviar_theory._induce_neural_theory_for_target(
        torch, fake_kfold_select, _never_called_train_engine_mode, prog=None,
        make_network=lambda: _FakeNetModule(), features_train=torch.zeros(4, 2),
        neural_relations={"close_nn": object()}, activity_sets_train={"both_active": {(0, 1)}},
        args=_base_args(), facts=[(0, 1), (1, 1), (2, 1), (3, 1)],
        target_labels=[True, False, True, False], wall={}, wall_key="theory_loop_init",
    )
    assert captured["min_fit"] == 0.75
    assert theory["clauses"] == []
    assert theory["stop_reason"] == "select_once abstained"
    assert nets == {}


def test_explicit_min_fit_is_forwarded_to_kfold_select_instead_of_the_default():
    captured = {}

    def fake_kfold_select(prog_factory, mask_name, facts, is_positive, make_network,
                           features, *, neural_relations, folds, seed, steps, topology,
                           tie_tolerance, min_fit=0.75):
        captured["min_fit"] = min_fit
        return _FakeSelection(rule=None)

    run_caviar_theory._induce_neural_theory_for_target(
        torch, fake_kfold_select, _never_called_train_engine_mode, prog=None,
        make_network=lambda: _FakeNetModule(), features_train=torch.zeros(4, 2),
        neural_relations={"close_nn": object()}, activity_sets_train={"both_active": {(0, 1)}},
        args=_base_args(), facts=[(0, 1), (1, 1), (2, 1), (3, 1)],
        target_labels=[True, False, True, False], wall={}, wall_key="theory_loop_init",
        min_fit=0.321,
    )
    assert captured["min_fit"] == 0.321


def test_min_fit_keyword_is_the_only_new_argument_kfold_select_sees():
    # A caller that predates `min_fit` (a fake matching the OLD call
    # signature, no min_fit parameter at all) would TypeError if
    # `_induce_neural_theory_for_target` started passing min_fit
    # unconditionally in a way that broke every existing caller -- this
    # pins that every OTHER keyword forwarded to kfold_select is untouched
    # by the min_fit addition.
    captured = {}

    def fake_kfold_select(prog_factory, mask_name, facts, is_positive, make_network,
                           features, *, neural_relations, folds, seed, steps, topology,
                           tie_tolerance, min_fit=0.75):
        captured.update(
            folds=folds, seed=seed, steps=steps, topology=topology,
            tie_tolerance=tie_tolerance, neural_relations=neural_relations,
        )
        return _FakeSelection(rule=None)

    args = SimpleNamespace(k=3, seed=11, steps=17, tie_tolerance=0.02, max_clauses=1, min_new_covered=1)
    run_caviar_theory._induce_neural_theory_for_target(
        torch, fake_kfold_select, _never_called_train_engine_mode, prog="PROG",
        make_network=lambda: _FakeNetModule(), features_train=torch.zeros(3, 2),
        neural_relations={"close_nn": "SPEC"}, activity_sets_train={"both_active": {(0, 1)}},
        args=args, facts=[(0, 1), (1, 1), (2, 1)], target_labels=[True, False, True],
        wall={}, wall_key="theory_loop_term", min_fit=0.5,
    )
    assert captured["folds"] == 3
    assert captured["seed"] == 11
    assert captured["steps"] == 17
    assert captured["topology"] == "star"
    assert captured["tie_tolerance"] == 0.02
    assert captured["neural_relations"] == {"close_nn": "SPEC"}
