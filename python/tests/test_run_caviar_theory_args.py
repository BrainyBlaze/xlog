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
