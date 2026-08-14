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

This module owns ONLY the weights and the scoring; it knows nothing
about folds, gates or corpora (the CV runner owns those)."""

from __future__ import annotations

SOFT_STEPS = 300
SOFT_LR = 0.05
SOFT_SEED = 7
INIT_LOGIT = -2.0
_CLAMP_EPS = 1e-7


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

    target = y.to(torch.get_default_dtype())
    weights = torch.full((cover.shape[0],), INIT_LOGIT, requires_grad=True)

    # The Adam update rule, written out by hand with torch.optim.Adam's own
    # defaults (beta1=0.9, beta2=0.999, eps=1e-8, no weight decay; eps added
    # AFTER the bias-corrected sqrt, as in the reference single-tensor
    # implementation). NOT torch.optim.Adam itself: constructing any
    # torch.optim optimizer imports torch._dynamo, which hard-requires
    # sympy — absent from the pinned CPU test environment this experiment
    # runs on. Same arithmetic, no optimizer object.
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = torch.zeros_like(weights)
    v = torch.zeros_like(weights)
    for step in range(1, steps + 1):
        if weights.grad is not None:
            weights.grad = None
        scores = soft_scores(cover, weights).clamp(_CLAMP_EPS, 1.0 - _CLAMP_EPS)
        loss = torch.nn.functional.binary_cross_entropy(scores, target)
        loss.backward()
        g = weights.grad
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)
        with torch.no_grad():
            weights -= lr * m_hat / (v_hat.sqrt() + eps)
    return weights.detach()
