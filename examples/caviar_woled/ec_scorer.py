"""Inertia reconstruction and frame-level scoring for the Event-Calculus
(initiatedAt/terminatedAt) evaluation protocol -- pure Python, no torch, no
engine import at all, so this is CPU-testable with hand-built prediction
sequences the same way `scorer.py` and `theory_loop.py` are.

WHAT THIS ADDS ON TOP OF `scorer.py`. `scorer.prf1`/`theory_predictions`
score a DIRECT holdsAt-style prediction against a direct holdsAt-style gold
label, one comparison per pair-time. This module instead starts from a pair
of EVENT predictions -- ``is_init``/``is_term``, `caviar_convert.
derive_ec_targets`'s own target shape -- and reconstructs the holdsAt
sequence those events imply by the classic Event-Calculus INERTIA rule: once
initiated, a fluent keeps holding until something terminates it. `frame_f1`
then reuses `scorer.prf1` unchanged to score the reconstructed sequence
against the same per-timestep gold used everywhere else in this example
(``convert_split``'s ``is_positive``).

SIMULTANEOUS INIT+TERM RULE. By construction, `derive_ec_targets`'s OWN
``is_init``/``is_term`` are mutually exclusive at every pair-time (one
requires ``label[t] == target``, the other requires ``label[t] != target``).
A caller's PREDICTED ``is_init``/``is_term`` (the whole point of running two
separate theories, see `run_caviar_theory.py`) carries no such guarantee --
two independently induced theories can easily fire on the very same step.
`reconstruct_holds` resolves that case with a deterministic, documented
rule: at each step, a term prediction clears the fluent FIRST, and an init
prediction on that SAME step then re-sets it -- so a step where both fire is
read as HOLDING, not as terminating. This matches the standard reading of
``initiatedAt``: the fluent is asserted to hold starting AT the step it is
initiated, and the whole point of frame-level scoring here is to compare a
same-step holdsAt reconstruction against a same-step gold label (`is_target`
at that exact ``t``, not the next one) -- so an initiation at ``t`` must win
over a termination the model separately (and, on this reading, wrongly)
proposed for that identical step, rather than have the two cancel into "not
holding" and silently throw away a positive that a real held-out clause
predicted. It is equivalent to the interval-closure reading `holdsAt(t) =
exists t' <= t with init(t') and no t'' in (t', t] with term(t'')`: taking
``t' = t`` when both fire at the same step makes the exclusion window
``(t, t]`` empty, so init wins there too -- the state-machine order below is
just the left-to-right way of computing that same closure.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from scorer import prf1  # noqa: E402


def reconstruct_holds(
    init_pred: list[bool], term_pred: list[bool], num_windows: int, T: int
) -> list[bool]:
    """Reconstruct a holdsAt sequence from per-pair-time init/term
    predictions by the classic Event-Calculus inertia closure, one
    independent state machine PER WINDOW (state always resets to
    ``False`` at each window's own ``t == 0`` -- a window never carries
    state in from whatever preceded it, since nothing outside the window
    was observed).

    ``init_pred``/``term_pred`` must both have exactly ``num_windows * T``
    entries, in the same ``pt = window_index * T + t`` indexing
    `caviar_convert.derive_ec_targets` uses; a length mismatch is refused
    (``ValueError``) rather than silently truncated or zero-padded.

    Per window, per step, in order: a term prediction clears the state
    FIRST; an init prediction at that SAME step then sets it -- see the
    module docstring's "SIMULTANEOUS INIT+TERM RULE" for why this order,
    and not the reverse, is the documented choice. Returns one ``bool`` per
    pair-time, aligned with the inputs.
    """
    expected = num_windows * T
    if len(init_pred) != expected:
        raise ValueError(
            f"reconstruct_holds: init_pred has {len(init_pred)} entries, "
            f"expected num_windows * T = {num_windows} * {T} = {expected}."
        )
    if len(term_pred) != expected:
        raise ValueError(
            f"reconstruct_holds: term_pred has {len(term_pred)} entries, "
            f"expected num_windows * T = {num_windows} * {T} = {expected}."
        )

    holds: list[bool] = []
    for w in range(num_windows):
        state = False
        for t in range(T):
            pt = w * T + t
            if term_pred[pt]:
                state = False
            if init_pred[pt]:
                state = True
            holds.append(state)
    return holds


def frame_f1(holds_pred: list[bool], holds_gold: list[bool]) -> dict:
    """Precision/recall/F1 of a reconstructed holdsAt sequence against the
    per-timestep gold label (`convert_split`'s ``is_positive`` -- the
    complex label equal to the target, e.g. "meeting"), one comparison per
    pair-time. A thin, honestly-named wrapper around `scorer.prf1` (same
    return shape, same length-mismatch refusal): this function adds no
    scoring logic of its own, only the Event-Calculus-specific name at the
    call site in `run_caviar_theory.py`.
    """
    return prf1(holds_pred, holds_gold)
