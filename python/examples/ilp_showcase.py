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
                    left, right, head = argmax
                    rule_str = format_rule(left, right, head, config.rule_template)
                    return rule_str, True, step + 1

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


# -- Main ---------------------------------------------------------------------

if __name__ == "__main__":
    preflight_check()
    print("\n(Stages not yet implemented)")
