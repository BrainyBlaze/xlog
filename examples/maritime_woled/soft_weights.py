"""Deterministic noisy-OR clause-weight trainer for the pre-registered
maritime soft-credit column (`docs/experiments/maritime/PREREG_SOFT.md`,
section (b)).

SEMANTICS. One logit weight per body; the prediction is

    score(pt) = 1 - PROD_c (1 - sigmoid(w_c) * cover_c(pt))

— the same noisy-OR reading the engine's relational real-credit columns
use (`pyxlog.ilp.neural_credit.credit_nll`: credit accumulates
multiplicatively over the clauses that cover a fact, trained end to end
by NLL/BCE on the labels). Training is full-batch BCE under Adam with
the pre-registered constants (steps=300, lr=0.05, seed=7), CPU, weights
initialized to -2.0 (every clause "off" — the sparse start the
pre-registration fixes).

DETERMINISM. `train_soft_weights` calls
`torch.use_deterministic_algorithms(True)` — a PROCESS-GLOBAL switch it
never restores (the same scope caveat `neural_credit.train_engine_mode`
documents: co-resident code that needs nondeterministic kernels will
start failing) — and `torch.manual_seed(seed)` right before the
optimizer loop, so two calls with identical arguments return
bitwise-equal weights regardless of ambient RNG state. (The computation
itself is seed-free — full batch, no dropout, fixed init — but the
seeding pins the contract, not a fixture.)

INCREMENTAL TRAINING (`PREREG_ONLINE.md`). The Adam state lives in
`SoftState(weights, m, v, t)`; `partial_fit` performs EXACTLY ONE Adam
step on one mini-batch (per-batch-mean BCE) and returns a new state —
the online column threads one state through the chronological windows.
`train_soft_weights` is now the composition `init_soft_state` + `steps`
x `partial_fit` over the full batch, pinned bitwise-identical to the
pre-refactor trainer by test
(`test_maritime_online.test_train_soft_weights_bitwise_matches_pre_refactor_snapshot`).

This module owns ONLY the weights and the scoring; it knows nothing
about folds, gates or corpora (the CV runner owns those)."""

from __future__ import annotations

from typing import NamedTuple

SOFT_STEPS = 300
SOFT_LR = 0.05
SOFT_SEED = 7
INIT_LOGIT = -2.0
_CLAMP_EPS = 1e-7

# torch.optim.Adam's own defaults (see the comment in partial_fit).
_BETA1, _BETA2, _ADAM_EPS = 0.9, 0.999, 1e-8


class SoftState(NamedTuple):
    """One logit + Adam moment per body, plus the step counter `t`
    (the number of partial_fit steps already taken — Adam's
    bias-correction exponent). Immutable: partial_fit returns a NEW
    state and never mutates its input."""

    weights: "torch.Tensor"  # [C] logits
    m: "torch.Tensor"        # [C] first moment
    v: "torch.Tensor"        # [C] second moment
    t: int


def init_soft_state(n_bodies: int) -> SoftState:
    """The pre-registered start: every logit at INIT_LOGIT (-2.0, every
    clause "off"), zero moments, step counter 0."""
    import torch

    if n_bodies < 1:
        raise ValueError(f"n_bodies must be >= 1 (got {n_bodies!r}).")
    weights = torch.full((n_bodies,), INIT_LOGIT)
    return SoftState(weights, torch.zeros_like(weights), torch.zeros_like(weights), 0)


def partial_fit(state: SoftState, cover, y, *, lr: float = SOFT_LR) -> SoftState:
    """One Adam step on one mini-batch: BCE (mean over THIS batch's rows)
    of the noisy-OR scores against `y`, gradients into a new state.
    `cover` is `[C, B]` bool, `y` `[B]` bool, `B >= 1`; `C` must match the
    state. Deterministic and RNG-free — two calls with equal arguments
    return bitwise-equal states."""
    import torch

    if cover.ndim != 2 or y.ndim != 1 or cover.shape[1] != y.shape[0]:
        raise ValueError(
            f"cover {tuple(cover.shape)} and y {tuple(y.shape)} must be "
            "[C, B] and [B] over the same B pt rows."
        )
    if cover.shape[0] != state.weights.shape[0]:
        raise ValueError(
            f"cover has {cover.shape[0]} bodies but the state carries "
            f"{state.weights.shape[0]}."
        )
    if y.shape[0] == 0:
        raise ValueError("an empty mini-batch has no gradient (B must be >= 1).")

    torch.use_deterministic_algorithms(True)

    target = y.to(torch.get_default_dtype())
    weights = state.weights.detach().clone().requires_grad_(True)
    scores = soft_scores(cover, weights).clamp(_CLAMP_EPS, 1.0 - _CLAMP_EPS)
    loss = torch.nn.functional.binary_cross_entropy(scores, target)
    loss.backward()
    g = weights.grad

    # The Adam update rule, written out by hand with torch.optim.Adam's own
    # defaults (beta1=0.9, beta2=0.999, eps=1e-8, no weight decay; eps added
    # AFTER the bias-corrected sqrt, as in the reference single-tensor
    # implementation). NOT torch.optim.Adam itself: constructing any
    # torch.optim optimizer imports torch._dynamo, which hard-requires
    # sympy — absent from the pinned CPU test environment this experiment
    # runs on. Same arithmetic, no optimizer object.
    t = state.t + 1
    m = _BETA1 * state.m + (1.0 - _BETA1) * g
    v = _BETA2 * state.v + (1.0 - _BETA2) * g * g
    m_hat = m / (1.0 - _BETA1 ** t)
    v_hat = v / (1.0 - _BETA2 ** t)
    with torch.no_grad():
        weights -= lr * m_hat / (v_hat.sqrt() + _ADAM_EPS)
    return SoftState(weights.detach(), m, v, t)


def soft_scores(cover, weights):
    """`1 - prod_c (1 - sigmoid(w_c) * cover_c)` per pt: float tensor
    `[N]` from a bool cover `[C, N]` and logit weights `[C]`. Pure — no
    clamping here (a pt covered by nothing scores exactly 0.0); the
    trainer clamps only inside its BCE."""
    import torch

    active = torch.sigmoid(weights).unsqueeze(1) * cover.to(torch.get_default_dtype())
    return 1.0 - torch.prod(1.0 - active, dim=0)


def train_soft_weights(cover, y, *, steps: int = SOFT_STEPS, lr: float = SOFT_LR,
                       seed: int = SOFT_SEED):
    """Train one logit per body (rows of `cover` `[C, N]`) against bool
    labels `y` `[N]` by full-batch BCE over the noisy-OR scores; returns
    the trained logits `[C]` (detached). Scores are clamped to
    `[1e-7, 1 - 1e-7]` inside the loss only, so an uncovered pt or a
    fully-confident clause cannot produce log(0)."""
    import torch

    if cover.ndim != 2 or y.ndim != 1 or cover.shape[1] != y.shape[0]:
        raise ValueError(
            f"cover {tuple(cover.shape)} and y {tuple(y.shape)} must be "
            "[C, N] and [N] over the same N pt rows."
        )

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)

    state = init_soft_state(cover.shape[0])
    for _ in range(steps):
        state = partial_fit(state, cover, y, lr=lr)
    return state.weights
