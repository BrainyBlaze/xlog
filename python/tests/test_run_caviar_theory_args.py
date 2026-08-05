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


# ---------------------------------------------------------------------------
# --max-body-literals -- the 3-literal relational_search.py upgrade. The
# behaviors these guard: the flag defaults to 2 (byte-identical to every
# behavior that predates it), accepts only 2 or 3, and REFUSES (typed
# argparse error, not a silent no-op) any combination with '3' other than
# '--mode relational --protocol ec'.
# ---------------------------------------------------------------------------


def test_max_body_literals_defaults_to_2():
    args = run_caviar_theory.parse_args(REQUIRED)
    assert args.max_body_literals == 2


def test_max_body_literals_accepts_3_with_relational_ec():
    args = run_caviar_theory.parse_args(
        REQUIRED + ["--protocol", "ec", "--max-body-literals", "3"]
    )
    assert args.max_body_literals == 3


def test_max_body_literals_rejects_values_other_than_2_or_3():
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(REQUIRED + ["--max-body-literals", "4"])
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(REQUIRED + ["--max-body-literals", "1"])


def test_max_body_literals_3_refuses_neural_mode():
    neural_required = ["--mode", "neural", "--pkl", "x.pkl", "--steps", "5", "--out", "o.json"]
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(
            neural_required + ["--protocol", "ec", "--max-body-literals", "3"]
        )


def test_max_body_literals_3_refuses_direct_protocol():
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(
            REQUIRED + ["--protocol", "direct", "--max-body-literals", "3"]
        )
    # Omitting --protocol also defaults to "direct" -- must refuse the same way.
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(REQUIRED + ["--max-body-literals", "3"])


def test_omitting_max_body_literals_leaves_every_other_default_unchanged():
    omitted = run_caviar_theory.parse_args(REQUIRED)
    explicit = run_caviar_theory.parse_args(REQUIRED + ["--max-body-literals", "2"])
    assert vars(omitted) == vars(explicit)


# ---------------------------------------------------------------------------
# --holdout-score -- the recall-aware F1 holdout metric for the
# '--max-body-literals 3' relational_search.py search. The behaviors these
# guard: the flag defaults to 'accuracy' (byte-identical to every behavior
# that predates it), accepts only 'accuracy'/'f1', and is refused (typed
# argparse error) with anything other than '--max-body-literals 3'.
# ---------------------------------------------------------------------------


def test_holdout_score_defaults_to_accuracy():
    args = run_caviar_theory.parse_args(REQUIRED)
    assert args.holdout_score == "accuracy"


def test_holdout_score_accepts_f1_with_max_body_literals_3():
    args = run_caviar_theory.parse_args(
        REQUIRED + ["--protocol", "ec", "--max-body-literals", "3", "--holdout-score", "f1"]
    )
    assert args.holdout_score == "f1"


def test_holdout_score_rejects_an_unknown_value():
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(REQUIRED + ["--holdout-score", "bogus"])


def test_holdout_score_f1_refused_without_max_body_literals_3():
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(REQUIRED + ["--holdout-score", "f1"])
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(
            REQUIRED + ["--protocol", "ec", "--holdout-score", "f1"]
        )


def test_omitting_holdout_score_leaves_every_other_default_unchanged():
    omitted = run_caviar_theory.parse_args(REQUIRED)
    explicit = run_caviar_theory.parse_args(REQUIRED + ["--holdout-score", "accuracy"])
    assert vars(omitted) == vars(explicit)


# ---------------------------------------------------------------------------
# --ec-fit-mode/--min-fit/--null-permutations/--null-quantile/--null-perm-seed
# -- the pre-registered permutation-null fit-gate threshold for the
# '--max-body-literals 3' relational_search.py search. The behaviors these
# guard: '--ec-fit-mode' defaults to 'fixed' and '--min-fit' defaults to
# None (byte-identical to every behavior that predates these flags), both
# are scoped to '--max-body-literals 3' ONLY, '--min-fit' is further scoped
# to '--ec-fit-mode fixed' ONLY, and the three '--null-*' knobs are scoped
# to '--ec-fit-mode permutation-null' ONLY.
# ---------------------------------------------------------------------------


def test_ec_fit_mode_defaults_to_fixed_and_min_fit_defaults_to_none():
    args = run_caviar_theory.parse_args(REQUIRED)
    assert args.ec_fit_mode == "fixed"
    assert args.min_fit is None


def test_null_knobs_default_values():
    args = run_caviar_theory.parse_args(REQUIRED)
    assert args.null_permutations == 1000
    assert args.null_quantile == 0.95
    assert args.null_perm_seed == 7


def test_ec_fit_mode_accepts_permutation_null_with_max_body_literals_3():
    # --holdout-score f1 is REQUIRED alongside permutation-null: the derived
    # threshold is an F1-axis quantile, so the holdout scores it gates must
    # be F1 too (see the refusal test below for the mismatch case).
    args = run_caviar_theory.parse_args(
        REQUIRED + ["--protocol", "ec", "--max-body-literals", "3",
                    "--holdout-score", "f1", "--ec-fit-mode", "permutation-null"]
    )
    assert args.ec_fit_mode == "permutation-null"
    assert args.holdout_score == "f1"


def test_ec_fit_mode_permutation_null_refused_with_accuracy_holdout():
    # The permutation-null threshold is a quantile of PER-FOLD F1 samples;
    # gated against accuracy-axis holdout scores (plateau ~0.9996 on a
    # rare-positive holdout, vs an F1-quantile gate of ~0.04-0.17) it
    # decides nothing while the JSON still records
    # ec_fit_mode: permutation-null. Both the default holdout score and an
    # explicit --holdout-score accuracy must be refused.
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(
            REQUIRED + ["--protocol", "ec", "--max-body-literals", "3",
                        "--ec-fit-mode", "permutation-null"]
        )
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(
            REQUIRED + ["--protocol", "ec", "--max-body-literals", "3",
                        "--ec-fit-mode", "permutation-null",
                        "--holdout-score", "accuracy"]
        )


def test_ec_fit_mode_rejects_an_unknown_value():
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(REQUIRED + ["--ec-fit-mode", "bogus"])


def test_ec_fit_mode_permutation_null_refused_without_max_body_literals_3():
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(REQUIRED + ["--ec-fit-mode", "permutation-null"])
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(
            REQUIRED + ["--protocol", "ec", "--ec-fit-mode", "permutation-null"]
        )


def test_min_fit_refused_without_max_body_literals_3():
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(REQUIRED + ["--min-fit", "0.8"])
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(
            REQUIRED + ["--protocol", "ec", "--min-fit", "0.8"]
        )


def test_min_fit_accepts_explicit_value_with_max_body_literals_3_fixed():
    args = run_caviar_theory.parse_args(
        REQUIRED + ["--protocol", "ec", "--max-body-literals", "3", "--min-fit", "0.8"]
    )
    assert args.min_fit == 0.8
    assert args.ec_fit_mode == "fixed"


def test_min_fit_refused_with_permutation_null():
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(
            REQUIRED + [
                "--protocol", "ec", "--max-body-literals", "3",
                "--ec-fit-mode", "permutation-null", "--min-fit", "0.8",
            ]
        )


def test_null_knobs_refused_without_permutation_null_mode():
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(
            REQUIRED + ["--protocol", "ec", "--max-body-literals", "3",
                        "--null-permutations", "500"]
        )
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(
            REQUIRED + ["--protocol", "ec", "--max-body-literals", "3",
                        "--null-quantile", "0.9"]
        )
    with pytest.raises(SystemExit):
        run_caviar_theory.parse_args(
            REQUIRED + ["--protocol", "ec", "--max-body-literals", "3",
                        "--null-perm-seed", "1"]
        )


def test_null_knobs_accepted_with_permutation_null_mode():
    args = run_caviar_theory.parse_args(
        REQUIRED + [
            "--protocol", "ec", "--max-body-literals", "3",
            "--holdout-score", "f1", "--ec-fit-mode", "permutation-null",
            "--null-permutations", "500", "--null-quantile", "0.9",
            "--null-perm-seed", "1",
        ]
    )
    assert args.null_permutations == 500
    assert args.null_quantile == 0.9
    assert args.null_perm_seed == 1


def test_omitting_ec_fit_mode_knobs_leaves_every_other_default_unchanged():
    omitted = run_caviar_theory.parse_args(REQUIRED)
    explicit = run_caviar_theory.parse_args(
        REQUIRED + [
            "--ec-fit-mode", "fixed", "--null-permutations", "1000",
            "--null-quantile", "0.95", "--null-perm-seed", "7",
        ]
    )
    assert vars(omitted) == vars(explicit)


def test_prepare_out_path_probe_does_not_destroy_an_existing_out_file(tmp_path):
    # The fail-fast writability probe must never replace an existing results
    # file: a run that then fails early would otherwise have destroyed the
    # previous run's output.
    out = tmp_path / "RESULT.json"
    out.write_text("ORIGINAL RESULT BYTES")
    returned = run_caviar_theory._prepare_out_path(str(out))
    assert returned == out
    assert out.read_text() == "ORIGINAL RESULT BYTES"
    # The probe file itself must not be left behind either.
    assert list(tmp_path.iterdir()) == [out]


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
