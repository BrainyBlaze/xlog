#!/usr/bin/env python3
"""XLOG dILP Showcase: Progressive Knowledge Graph Discovery.

Demonstrates every feature of XLOG's Tensorized Differentiable ILP engine
through 4 sequential stages, each discovering a logical rule from data.

Stage 1: Graph Reachability   -- basic training loop + per-fact credit
Stage 2: Family Grandparent   -- distractor relations + negative examples + missed-positive penalty
Stage 3: Workplace Colleague  -- different join pattern (shared-endpoint) + head projection
Stage 4: Arithmetic plus2     -- temperature annealing + rule commit + post-commit validation

Usage:
    .venv/bin/python python/examples/ilp_showcase.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# -- Shared Helpers -----------------------------------------------------------

def build_budget_aware_mask(
    W: torch.Tensor, tau: float, budget: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ST-Gumbel-Softmax with global top-k hard selection.

    Per-(i,j) argmax produces up to N^2 entries, which can exceed the
    executor's max_active_rules=32 (lower.rs:524, provider.rs:10616).
    Global top-k keeps the Python hard mask in sync with what the GPU
    executor actually processes -- no silent truncation mismatch.

    Returns (M_hard, M_soft) both shaped (N, N, N).
    """
    M_soft = F.gumbel_softmax(W, tau=tau, hard=False, dim=-1)

    flat = M_soft.view(-1)
    k = min(budget, flat.numel())
    _, topk_idx = flat.topk(k)

    M_hard_flat = torch.zeros_like(flat)
    M_hard_flat[topk_idx] = 1.0
    M_hard = M_hard_flat.view_as(M_soft)

    return M_hard, M_soft


def compute_loss(
    prog,
    M_soft: torch.Tensor,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    rel_names: list[str],
    n: int,
) -> torch.Tensor:
    """Per-fact surrogate loss with missed-positive penalty (RD-21).

    Positive examples: if derived, use per-fact credit from
    tagged_entries_containing_fact; if not derived, apply a differentiable
    missed-positive penalty that pushes M_soft toward the target k slice.

    Negative examples: penalize rules that derive them.
    """
    loss = torch.tensor(0.0, device="cuda")

    for rel_name, values in positives:
        contributing = prog.tagged_entries_containing_fact(rel_name, values)
        if contributing:
            credit = sum(M_soft[i, j, k] for (i, j, k) in contributing)
            loss = loss + (-torch.log(credit.clamp(min=1e-8)))
        else:
            k_idx = rel_names.index(rel_name)
            loss = loss + (-M_soft[:, :, k_idx].sum() / (n * n))

    for rel_name, values in negatives:
        contributing = prog.tagged_entries_containing_fact(rel_name, values)
        if contributing:
            credit = sum(M_soft[i, j, k] for (i, j, k) in contributing)
            loss = loss + (-torch.log((1.0 - credit).clamp(min=1e-8)))

    return loss


def decode_argmax(W: torch.Tensor, rel_names: list[str]) -> tuple[str, str, str]:
    """Decode the argmax (i,j,k) of W into relation name triple."""
    with torch.no_grad():
        flat = W.view(-1)
        idx = flat.argmax().item()
        n = W.shape[0]
        i = idx // (n * n)
        j = (idx % (n * n)) // n
        k = idx % n
    return (rel_names[i], rel_names[j], rel_names[k])


def format_rule(left: str, right: str, head: str, template: str) -> str:
    """Format a discovered (left, right, head) triple as a Datalog rule string."""
    return template.replace("{H}", head).replace("{L}", left).replace("{R}", right)


def log_step(step: int, loss: torch.Tensor, tau: float,
             rule: tuple[str, str, str], stable: int,
             M_soft: torch.Tensor):
    """Print one training step's metrics."""
    disc = M_soft.max(dim=-1)[0].mean().item()
    loss_val = loss.item()
    left, right, head = rule
    print(f"  Step {step:3d} | tau={tau:.3f} | loss={loss_val:7.3f} "
          f"| rule: {left}+{right}->{head} | stable={stable} | disc={disc:.2f}")


# -- Training Loop ------------------------------------------------------------

@dataclass
class StageConfig:
    """Configuration for one dILP learning stage."""
    name: str
    source: str
    mask_name: str
    positives: list[tuple[str, list[int]]]
    negatives: list[tuple[str, list[int]]]
    target_rule: str
    rule_template: str
    max_steps: int = 100
    tau_start: float = 2.0
    tau_end: float = 0.1
    lr: float = 0.1
    budget: int = 32
    commit: bool = False
    retries: int = 3
    stable_threshold: int = 5


def train_stage(config: StageConfig) -> tuple[str, bool, int]:
    """Train one dILP stage. Returns (discovered_rule_str, converged, steps)."""
    import pyxlog

    for attempt in range(config.retries):
        if attempt > 0:
            print(f"  Retry {attempt}/{config.retries} (new random init)")

        prog = pyxlog.IlpProgramFactory.compile(
            config.source, device=0, memory_mb=512,
        )
        n = prog.ilp_schema_size()
        rel_names = prog.ilp_relation_names()

        W = torch.randn((n, n, n), requires_grad=True, device="cuda")
        optimizer = torch.optim.Adam([W], lr=config.lr)
        prev_argmax = None
        stable_count = 0

        for step in range(config.max_steps):
            frac = step / max(config.max_steps - 1, 1)
            tau = config.tau_start + (config.tau_end - config.tau_start) * frac
            optimizer.zero_grad()

            M_hard, M_soft = build_budget_aware_mask(W, tau, config.budget)

            prog.set_rule_mask(
                config.mask_name,
                M_hard.detach().contiguous().view(-1),
                M_soft.detach().contiguous().view(-1),
                n,
            )
            prog.evaluate()

            loss = compute_loss(
                prog, M_soft,
                config.positives, config.negatives,
                rel_names, n,
            )

            if loss.requires_grad:
                loss.backward()
                optimizer.step()

            argmax = decode_argmax(W, rel_names)
            if argmax == prev_argmax:
                stable_count += 1
            else:
                stable_count = 0
            prev_argmax = argmax

            log_step(step, loss, tau, argmax, stable_count, M_soft)

            if stable_count >= config.stable_threshold:
                all_derived = all(
                    prog.fact_exists(rel, vals)
                    for rel, vals in config.positives
                )
                if all_derived:
                    # Validate that the argmax rule ALONE derives all positives.
                    left, right, head = argmax
                    i_idx = rel_names.index(left)
                    j_idx = rel_names.index(right)
                    k_idx = rel_names.index(head)
                    M_check = torch.zeros((n, n, n), device="cuda")
                    M_check[i_idx, j_idx, k_idx] = 1.0
                    flat_check = M_check.contiguous().view(-1)
                    prog.set_rule_mask(
                        config.mask_name, flat_check, flat_check, n,
                    )
                    prog.evaluate()
                    argmax_ok = all(
                        prog.fact_exists(rel, vals)
                        for rel, vals in config.positives
                    )
                    if argmax_ok:
                        rule_str = format_rule(left, right, head, config.rule_template)
                        return rule_str, True, step + 1
                    # Argmax alone doesn't work; keep training.
                    stable_count = 0

    left, right, head = prev_argmax if prev_argmax else ("?", "?", "?")
    rule_str = format_rule(left, right, head, config.rule_template)
    return rule_str, False, config.max_steps


def commit_and_validate(
    config: StageConfig, rule_str: str,
) -> bool:
    """Commit a discovered rule and validate post-commit derivations."""
    import pyxlog

    print(f"  Committing: {rule_str}")
    prog = pyxlog.IlpProgramFactory.compile(
        config.source, device=0, memory_mb=512,
    )
    prog.commit_induced_rule(rule_str)

    all_ok = True
    for rel_name, values in config.positives:
        exists = prog.fact_exists(rel_name, values)
        status = "OK" if exists else "MISSING"
        print(f"    {rel_name}{tuple(values)} = {status}")
        if not exists:
            all_ok = False

    for rel_name, values in config.negatives:
        exists = prog.fact_exists(rel_name, values)
        status = "OK (absent)" if not exists else "UNEXPECTED"
        print(f"    {rel_name}{tuple(values)} = {status}")
        if exists:
            all_ok = False

    return all_ok


def run_stage(config: StageConfig) -> tuple[str, bool, int, str]:
    """Run one stage with header/footer logging. Returns (rule, converged, steps, tag)."""
    print(f"\n{'='*60}")
    print(f"  Stage: {config.name}")
    print(f"{'='*60}")

    rule_str, converged, steps = train_stage(config)

    tag = ""
    if converged:
        print(f"  >>> Converged in {steps} steps: {rule_str}")
        if config.commit:
            print(f"\n  --- Post-Convergence: Rule Commit ---")
            commit_ok = commit_and_validate(config, rule_str)
            if commit_ok:
                print(f"  >>> Commit validated: all facts correct")
                tag = "committed"
            else:
                print(f"  >>> Commit validation FAILED")
                converged = False
    else:
        print(f"  >>> DID NOT CONVERGE after {config.max_steps} steps.")
        print(f"      Best guess: {rule_str}")

    return rule_str, converged, steps, tag


# -- Preflight ----------------------------------------------------------------

def preflight_check():
    """Fail fast with actionable environment diagnostics."""
    if not torch.cuda.is_available():
        print("FATAL: torch.cuda.is_available() == False")
        print("  - Check: nvidia-smi")
        print("  - Check: LD_LIBRARY_PATH includes /usr/lib/wsl/lib (WSL2)")
        print(f"  - Current LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '(unset)')}")
        sys.exit(1)

    try:
        import pyxlog  # noqa: F401
    except ImportError as e:
        print(f"FATAL: Cannot import pyxlog: {e}")
        print("  - Run: cd /home/dev/projects/xlog && .venv/bin/maturin develop --release")
        sys.exit(1)

    try:
        import pyxlog
        prog = pyxlog.IlpProgramFactory.compile(
            "edge(1,2). learnable(W) :: r(X,Y) :- b1(X,Z), b2(Z,Y).",
            device=0, memory_mb=256,
        )
        n = prog.ilp_schema_size()
        zeros = torch.zeros((n, n, n), device="cuda")
        flat = zeros.contiguous().view(-1)
        prog.set_rule_mask("W", flat, flat, n)
        prog.evaluate()
    except Exception as e:
        print(f"FATAL: ILP smoke test failed: {e}")
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '(unset)')}")
        sys.exit(1)

    print(f"Preflight OK: GPU={torch.cuda.get_device_name(0)}, "
          f"CUDA={torch.version.cuda}, N_smoke={n}")


# -- Stage Definitions --------------------------------------------------------

STAGE_1 = StageConfig(
    name="Graph Reachability",
    source="""
        edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
        learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """,
    mask_name="W_reach",
    positives=[
        ("reach", [1, 3]),
        ("reach", [2, 4]),
        ("reach", [3, 5]),
        ("reach", [4, 6]),
    ],
    negatives=[],
    target_rule="reach(X, Y) :- edge(X, Z), edge(Z, Y).",
    rule_template="{H}(X, Y) :- {L}(X, Z), {R}(Z, Y).",
    max_steps=100,
    tau_start=1.0,
    tau_end=0.1,
    lr=0.1,
)

STAGE_2 = StageConfig(
    name="Family Grandparent",
    source="""
        parent(1, 2). parent(2, 3). parent(2, 4). parent(3, 5). parent(4, 6).
        gender(1, 0). gender(2, 1). gender(3, 1). gender(4, 0).
        sibling(2, 7). sibling(7, 2).
        learnable(W_gp) :: grandparent(X, Y) :- bL(X, Z), bR(Z, Y).
    """,
    mask_name="W_gp",
    positives=[
        ("grandparent", [1, 3]),
        ("grandparent", [1, 4]),
        ("grandparent", [2, 5]),
        ("grandparent", [2, 6]),
    ],
    negatives=[
        ("grandparent", [1, 2]),
        ("grandparent", [3, 1]),
    ],
    target_rule="grandparent(X, Y) :- parent(X, Z), parent(Z, Y).",
    rule_template="{H}(X, Y) :- {L}(X, Z), {R}(Z, Y).",
    max_steps=120,
    tau_start=2.0,
    tau_end=0.05,
    lr=0.15,
    retries=5,
)

STAGE_3 = StageConfig(
    name="Workplace Colleague",
    source="""
        worksAt(1, 101). worksAt(7, 101). worksAt(2, 102). worksAt(4, 102).
        livesIn(1, 201). livesIn(2, 201). livesIn(3, 202). livesIn(4, 202).
        learnable(W_col) :: colleague(X, Y) :- bL(X, Z), bR(Y, Z).
    """,
    mask_name="W_col",
    positives=[
        ("colleague", [1, 7]),
        ("colleague", [7, 1]),
        ("colleague", [2, 4]),
        ("colleague", [4, 2]),
    ],
    negatives=[
        ("colleague", [1, 2]),
        ("colleague", [3, 4]),
    ],
    target_rule="colleague(X, Y) :- worksAt(X, Z), worksAt(Y, Z).",
    rule_template="{H}(X, Y) :- {L}(X, Z), {R}(Y, Z).",
    max_steps=150,
    tau_start=2.0,
    tau_end=0.05,
    lr=0.15,
    retries=7,
)

STAGE_4 = StageConfig(
    name="Arithmetic plus2",
    source="""
        succ(0, 1). succ(1, 2). succ(2, 3). succ(3, 4). succ(4, 5).
        pred(1, 0). pred(2, 1). pred(3, 2). pred(4, 3). pred(5, 4).
        learnable(W_p2) :: plus2(X, Y) :- bL(X, Z), bR(Z, Y).
    """,
    mask_name="W_p2",
    positives=[
        ("plus2", [0, 2]),
        ("plus2", [1, 3]),
        ("plus2", [2, 4]),
        ("plus2", [3, 5]),
    ],
    negatives=[
        ("plus2", [0, 1]),
        ("plus2", [5, 0]),
    ],
    target_rule="plus2(X, Y) :- succ(X, Z), succ(Z, Y).",
    rule_template="{H}(X, Y) :- {L}(X, Z), {R}(Z, Y).",
    max_steps=150,
    tau_start=2.5,
    tau_end=0.05,
    lr=0.15,
    commit=True,  # THIS STAGE COMMITS THE DISCOVERED RULE
    retries=5,
)



# -- Main ---------------------------------------------------------------------

STAGES = [STAGE_1, STAGE_2, STAGE_3, STAGE_4]

if __name__ == "__main__":
    preflight_check()

    results: list[tuple[str, str, str, bool, int, str]] = []

    for i, stage in enumerate(STAGES, 1):
        rule, ok, steps, tag = run_stage(stage)
        results.append((f"Stage {i}", stage.name, rule, ok, steps, tag))

    print(f"\n{'='*60}")
    print("  XLOG dILP Showcase -- Summary")
    print(f"{'='*60}")
    for label, name, rule, ok, steps, tag in results:
        suffix = f" ({tag})" if tag else ""
        status = f"OK {steps} steps{suffix}" if ok else "FAILED"
        print(f"  {label}: {rule:<50s} {status}")
    print(f"{'='*60}")

    if not all(ok for _, _, _, ok, _, _ in results):
        print("\nSome stages did not converge. See output above.")
        sys.exit(1)
    else:
        print("\nAll stages converged and validated successfully.")
