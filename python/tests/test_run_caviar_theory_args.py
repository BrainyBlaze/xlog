"""Unit tests for `run_caviar_theory.py`'s `parse_args` -- CPU, no CUDA, no
pkl, no engine. `parse_args` (and everything it touches) never imports
torch/pyxlog at module level (see the module's own docstring), so this file
imports `run_caviar_theory` directly and checks argument defaults/choices
only; it does not, and cannot, exercise the CUDA-only induction paths.

The one behavior these tests exist to guard: adding `--protocol` must leave
every OTHER argument's default untouched, and must itself default to
`"direct"` -- the flag that keeps the CLI's existing behavior byte-identical
when omitted.
"""
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "caviar_woled"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import run_caviar_theory  # noqa: E402

REQUIRED = ["--mode", "relational", "--pkl", "x.pkl", "--steps", "5", "--out", "o.json"]


def test_run_caviar_theory_module_does_not_bind_torch_or_pyxlog_at_import_time():
    # `import run_caviar_theory` above already happened without CUDA/torch
    # being required; this additionally checks the module never bound
    # either name at module scope (both stay function-local, imported lazily
    # past `parse_args` -- see the module docstring's CUDA-ONLY paragraph).
    assert not hasattr(run_caviar_theory, "torch")
    assert not hasattr(run_caviar_theory, "pyxlog")


def test_protocol_defaults_to_direct():
    args = run_caviar_theory.parse_args(REQUIRED)
    assert args.protocol == "direct"


def test_protocol_accepts_ec():
    args = run_caviar_theory.parse_args(REQUIRED + ["--protocol", "ec"])
    assert args.protocol == "ec"


def test_protocol_rejects_an_unknown_value():
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(REQUIRED + ["--protocol", "bogus"])


def test_omitting_protocol_leaves_every_other_default_unchanged():
    # Pinned against the pre-`--protocol` CLI's own documented defaults.
    args = run_caviar_theory.parse_args(REQUIRED)
    assert args.mode == "relational"
    assert args.fold == "fold1"
    assert args.k == 4
    assert args.seed == 7
    assert args.hidden == 16
    assert args.max_clauses == 4


def test_explicit_protocol_direct_parses_identically_to_omitting_it():
    omitted = run_caviar_theory.parse_args(REQUIRED)
    explicit = run_caviar_theory.parse_args(REQUIRED + ["--protocol", "direct"])
    assert vars(omitted) == vars(explicit)


# ---------------------------------------------------------------------------
# `--data`/`--test-json`/`--min-new-covered` -- added alongside the
# continuous-dataset loader (caviar_continuous.py). The one behavior these
# guard: `--data` must default to `"pkl"` (byte-identical CLI behavior when
# omitted) and every OTHER argument's default (including the ones the
# `--protocol` tests above already pin) must stay unchanged.
# ---------------------------------------------------------------------------


def test_data_defaults_to_pkl():
    args = run_caviar_theory.parse_args(REQUIRED)
    assert args.data == "pkl"
    assert args.test_json is None


def test_min_new_covered_defaults_to_ten():
    args = run_caviar_theory.parse_args(REQUIRED)
    assert args.min_new_covered == 10


def test_min_new_covered_is_overridable():
    args = run_caviar_theory.parse_args(REQUIRED + ["--min-new-covered", "3"])
    assert args.min_new_covered == 3


def test_data_rejects_an_unknown_value():
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(REQUIRED + ["--data", "bogus"])


def test_data_continuous_requires_test_json():
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(REQUIRED + ["--data", "continuous"])


def test_data_continuous_with_test_json_parses():
    args = run_caviar_theory.parse_args(
        REQUIRED + ["--data", "continuous", "--test-json", "caviar-test.json"]
    )
    assert args.data == "continuous"
    assert args.test_json == "caviar-test.json"


def test_omitting_data_leaves_every_other_default_unchanged_including_new_flags():
    omitted = run_caviar_theory.parse_args(REQUIRED)
    explicit = run_caviar_theory.parse_args(REQUIRED + ["--data", "pkl"])
    assert vars(omitted) == vars(explicit)


def test_tie_tolerance_default_is_none_and_explicit_value_parses():
    assert run_caviar_theory.parse_args(REQUIRED).tie_tolerance is None
    args = run_caviar_theory.parse_args(REQUIRED + ["--tie-tolerance", "0.005"])
    assert args.tie_tolerance == 0.005


# ---------------------------------------------------------------------------
# EC don't-care wiring helpers -- both are plain Python (no torch/pyxlog/
# CUDA), so they are directly CPU-testable here, unlike the CUDA-only
# induction paths that use them (`_run_relational_ec`/`_run_neural_ec`).
# ---------------------------------------------------------------------------


def test_exclude_dontcare_drops_flagged_rows():
    facts = [(0, 1), (1, 1), (2, 1), (3, 1)]
    labels = [False, True, False, True]
    dontcare = [True, False, True, False]
    kept_facts, kept_labels = run_caviar_theory._exclude_dontcare(facts, labels, dontcare)
    assert kept_facts == [(1, 1), (3, 1)]
    assert kept_labels == [True, True]


def test_exclude_dontcare_with_none_mask_is_a_no_op():
    facts = [(0, 1), (1, 1)]
    labels = [False, True]
    kept_facts, kept_labels = run_caviar_theory._exclude_dontcare(facts, labels, None)
    assert kept_facts == facts
    assert kept_labels == labels


def test_exclude_dontcare_all_dontcare_yields_empty_lists():
    facts = [(0, 1), (1, 1)]
    labels = [True, False]
    kept_facts, kept_labels = run_caviar_theory._exclude_dontcare(facts, labels, [True, True])
    assert kept_facts == []
    assert kept_labels == []


def test_ec_relations_with_transitions_merges_only_when_present():
    train = {"relations": {"close": [(0, 1)]}, "transition_relations": {"any_became_active": [(0, 1)]}}
    test = {"relations": {"close": [(1, 1)]}, "transition_relations": {"any_became_active": []}}
    train_rel, test_rel = run_caviar_theory._ec_relations_with_transitions(train, test)
    assert train_rel == {"close": [(0, 1)], "any_became_active": [(0, 1)]}
    assert test_rel == {"close": [(1, 1)], "any_became_active": []}


def test_ec_relations_with_transitions_pkl_data_is_unaugmented():
    # --data pkl's converted dicts have no "transition_relations" key at
    # all -- the merge must be a no-op, returning the SAME relations dicts
    # unchanged, so a direct-protocol-shaped vocabulary never gains anything.
    train = {"relations": {"close": [(0, 1)]}}
    test = {"relations": {"close": [(1, 1)]}}
    train_rel, test_rel = run_caviar_theory._ec_relations_with_transitions(train, test)
    assert train_rel is train["relations"]
    assert test_rel is test["relations"]


def test_filtered_relation_names_never_sees_transition_relations():
    # `_filtered_relation_names` is the DIRECT-protocol (relational mode)
    # vocabulary builder; it must only ever be handed a "relations" dict,
    # never a merged one -- pinning that guarantee here, at the function
    # that would silently perturb --protocol direct if it ever were.
    converted = {
        "relations": {"close": [(0, 1)], "coords_missing": [(1, 1)]},
        "transition_relations": {"any_became_active": [(0, 1)]},
    }
    assert run_caviar_theory._filtered_relation_names(converted) == ["close"]
