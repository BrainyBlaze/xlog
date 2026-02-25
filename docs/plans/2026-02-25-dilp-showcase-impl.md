# dILP Showcase Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a single Python showcase script (`python/examples/ilp_showcase.py`) that demonstrates every feature of XLOG's Tensorized Differentiable ILP through 4 progressive stages — each must converge to the correct rule.

**Architecture:** 4 sequential stages, each an independent `IlpProgramFactory.compile()` with one learnable rule. Shared helper functions handle budget-aware mask construction, surrogate loss, training loop, and convergence detection. Stage 4 commits the discovered rule and validates post-commit.

**Tech Stack:** Python 3, PyTorch (CUDA), pyxlog (via maturin), torch.nn.functional (Gumbel-Softmax)

**Design doc:** `docs/plans/2026-02-25-dilp-showcase-design.md`

**Test environment:** `.venv/bin/python` (pyxlog installed via maturin develop)

---

### Task 1: Scaffold script with preflight check

**Files:**
- Create: `python/examples/ilp_showcase.py`

**Step 1: Create the directory and skeleton script**

```python
#!/usr/bin/env python3
"""XLOG dILP Showcase: Progressive Knowledge Graph Discovery.

Demonstrates every feature of XLOG's Tensorized Differentiable ILP engine
through 4 sequential stages, each discovering a logical rule from data.

Stage 1: Graph Reachability   — basic training loop + per-fact credit
Stage 2: Family Grandparent   — distractor relations + negative examples + missed-positive penalty
Stage 3: Workplace Colleague  — different join pattern (shared-endpoint) + head projection
Stage 4: Arithmetic plus2     — temperature annealing + rule commit + post-commit validation

Usage:
    .venv/bin/python python/examples/ilp_showcase.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def preflight_check():
    """Fail fast with actionable environment diagnostics.

    Checks: torch CUDA, pyxlog import, and a tiny ILP compile/evaluate smoke
    test using cudarc. Prints specific remediation steps on failure.
    """
    # 1. CUDA runtime
    if not torch.cuda.is_available():
        print("FATAL: torch.cuda.is_available() == False")
        print("  - Check: nvidia-smi")
        print("  - Check: LD_LIBRARY_PATH includes /usr/lib/wsl/lib (WSL2)")
        print(f"  - Current LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '(unset)')}")
        sys.exit(1)

    # 2. pyxlog import
    try:
        import pyxlog  # noqa: F401
    except ImportError as e:
        print(f"FATAL: Cannot import pyxlog: {e}")
        print("  - Run: cd /home/dev/projects/xlog && .venv/bin/maturin develop --release")
        sys.exit(1)

    # 3. ILP compile/evaluate smoke check (exercises cudarc + GPU kernels)
    try:
        import pyxlog
        prog = pyxlog.IlpProgramFactory.compile(
            "edge(1,2). learnable(W) :: r(X,Y) :- b1(X,Z), b2(Z,Y).",
            device=0, memory_mb=256,
        )
        n = prog.ilp_schema_size()
        W = torch.zeros((n, n, n), device="cuda")
        flat = W.contiguous().view(-1)
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


if __name__ == "__main__":
    preflight_check()
    print("\n(Stages not yet implemented)")
```

**Step 2: Create the `python/examples/` directory and run the preflight**

Run:
```bash
mkdir -p /home/dev/projects/xlog/python/examples
```

Then run the script:
```bash
.venv/bin/python python/examples/ilp_showcase.py
```

Expected: `Preflight OK: GPU=..., CUDA=..., N_smoke=...` followed by `(Stages not yet implemented)`.

If preflight fails, fix the environment issue before proceeding.

**Step 3: Commit**

```bash
git add python/examples/ilp_showcase.py
git commit -m "feat(ilp): scaffold dILP showcase with cudarc preflight"
```

---

### Task 2: Add shared helpers — mask, loss, decode, logging

**Files:**
- Modify: `python/examples/ilp_showcase.py`

**Step 1: Add the budget-aware mask builder**

Insert after the imports, before `preflight_check()`:

```python
# ── Shared Helpers ───────────────────────────────────────────────────────────

def build_budget_aware_mask(
    W: torch.Tensor, tau: float, budget: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ST-Gumbel-Softmax with global top-k hard selection.

    Per-(i,j) argmax produces up to N^2 entries, which can exceed the
    executor's max_active_rules=32 (lower.rs:524, provider.rs:10616).
    Global top-k keeps the Python hard mask in sync with what the GPU
    executor actually processes — no silent truncation mismatch.

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
            # RD-21: Missed-positive penalty — encourage M_soft mass on target k
            k_idx = rel_names.index(rel_name)
            loss = loss + (-M_soft[:, :, k_idx].sum() / (n * n))

    for rel_name, values in negatives:
        contributing = prog.tagged_entries_containing_fact(rel_name, values)
        if contributing:
            credit = sum(M_soft[i, j, k] for (i, j, k) in contributing)
            loss = loss + (-torch.log((1.0 - credit).clamp(min=1e-8)))

    return loss


def decode_argmax(W: torch.Tensor, rel_names: list[str]) -> tuple[str, str, str]:
    """Decode the argmax (i,j,k) of W into relation name triple.

    Returns (left_rel, right_rel, head_rel) — the discovered rule expressed
    as which relations fill the body1, body2, head slots.
    """
    with torch.no_grad():
        flat = W.view(-1)
        idx = flat.argmax().item()
        n = W.shape[0]
        i = idx // (n * n)
        j = (idx % (n * n)) // n
        k = idx % n
    return (rel_names[i], rel_names[j], rel_names[k])


def format_rule(left: str, right: str, head: str, template: str) -> str:
    """Format a discovered (left, right, head) triple as a Datalog rule string.

    template is e.g. "reach(X, Y) :- {L}(X, Z), {R}(Z, Y)." for chain join
    or "colleague(X, Y) :- {L}(X, Z), {R}(Y, Z)." for shared-endpoint join.
    """
    return template.replace("{H}", head).replace("{L}", left).replace("{R}", right)


def log_step(step: int, loss: torch.Tensor, tau: float,
             rule: tuple[str, str, str], stable: int, n: int,
             M_soft: torch.Tensor):
    """Print one training step's metrics."""
    disc = M_soft.max(dim=-1)[0].mean().item()
    loss_val = loss.item() if loss.requires_grad else loss.item()
    left, right, head = rule
    print(f"  Step {step:3d} | tau={tau:.3f} | loss={loss_val:7.3f} "
          f"| rule: {left}+{right}->{head} | stable={stable} | disc={disc:.2f}")
```

**Step 2: Run the script to verify no syntax errors**

```bash
.venv/bin/python python/examples/ilp_showcase.py
```

Expected: Same preflight output, no import errors or syntax errors.

**Step 3: Commit**

```bash
git add python/examples/ilp_showcase.py
git commit -m "feat(ilp): add shared helpers — mask, loss, decode, logging"
```

---

### Task 3: Add StageConfig and train_stage loop

**Files:**
- Modify: `python/examples/ilp_showcase.py`

**Step 1: Add the StageConfig dataclass and train_stage function**

Insert after the helper functions, before `preflight_check()`:

```python
# ── Training Loop ────────────────────────────────────────────────────────────

@dataclass
class StageConfig:
    """Configuration for one dILP learning stage."""
    name: str                                   # Display name
    source: str                                 # xlog source with learnable rule
    mask_name: str                              # Mask name in learnable() decl
    positives: list[tuple[str, list[int]]]      # Positive examples
    negatives: list[tuple[str, list[int]]]       # Negative examples
    target_rule: str                            # Expected rule (for verification)
    rule_template: str                          # Format template with {L},{R},{H}
    max_steps: int = 100                        # Training budget
    tau_start: float = 2.0                      # Initial temperature
    tau_end: float = 0.1                        # Final temperature
    lr: float = 0.1                             # Adam learning rate
    budget: int = 32                            # max_active_rules budget
    commit: bool = False                        # Whether to commit after convergence
    retries: int = 3                            # Max random restarts
    stable_threshold: int = 5                   # Steps of stable argmax for convergence


def train_stage(config: StageConfig) -> tuple[str, bool, int]:
    """Train one dILP stage. Returns (discovered_rule_str, converged, steps).

    Implements the full training loop:
    1. Compile source with IlpProgramFactory
    2. Initialize W ~ N(0,1) on CUDA
    3. Budget-aware ST-Gumbel-Softmax → set_rule_mask → evaluate
    4. Surrogate loss (credit + missed-positive + negative penalties)
    5. Convergence detection: stable argmax + all positives derived
    6. Retry with fresh W on non-convergence
    """
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

            log_step(step, loss, tau, argmax, stable_count, n, M_soft)

            # Convergence: stable argmax + all positives derived
            if stable_count >= config.stable_threshold:
                all_derived = all(
                    prog.fact_exists(rel, vals)
                    for rel, vals in config.positives
                )
                if all_derived:
                    left, right, head = argmax
                    rule_str = format_rule(left, right, head, config.rule_template)
                    return rule_str, True, step + 1

        # This attempt didn't converge — try again

    # All retries exhausted
    left, right, head = prev_argmax if prev_argmax else ("?", "?", "?")
    rule_str = format_rule(left, right, head, config.rule_template)
    return rule_str, False, config.max_steps
```

**Step 2: Run the script to verify no syntax errors**

```bash
.venv/bin/python python/examples/ilp_showcase.py
```

Expected: Preflight OK, `(Stages not yet implemented)`. No errors.

**Step 3: Commit**

```bash
git add python/examples/ilp_showcase.py
git commit -m "feat(ilp): add StageConfig dataclass and train_stage loop"
```

---

### Task 4: Implement Stage 1 — Graph Reachability

**Files:**
- Modify: `python/examples/ilp_showcase.py`

**Step 1: Add Stage 1 config and wire into `__main__`**

Replace the `if __name__ == "__main__"` block with:

```python
# ── Stage Definitions ────────────────────────────────────────────────────────

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
    max_steps=50,
    tau_start=1.0,
    tau_end=0.3,
    lr=0.1,
)


def run_stage(config: StageConfig) -> tuple[str, bool, int]:
    """Run one stage with header/footer logging."""
    print(f"\n{'='*60}")
    print(f"  Stage: {config.name}")
    print(f"{'='*60}")

    rule_str, converged, steps = train_stage(config)

    if converged:
        print(f"  >>> Converged in {steps} steps: {rule_str}")
    else:
        print(f"  >>> DID NOT CONVERGE after {config.max_steps} steps.")
        print(f"      Best guess: {rule_str}")

    return rule_str, converged, steps


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    preflight_check()

    results = []

    rule, ok, steps = run_stage(STAGE_1)
    results.append(("Stage 1", config.name if (config := STAGE_1) else "", rule, ok, steps))

    # Summary
    print(f"\n{'='*60}")
    print("  dILP Showcase — Summary")
    print(f"{'='*60}")
    for label, name, rule, ok, steps in results:
        status = f"OK {steps} steps" if ok else "FAILED"
        print(f"  {label}: {rule:<50s} {status}")
    print(f"{'='*60}")

    if not all(ok for _, _, _, ok, _ in results):
        print("\nSome stages did not converge. See output above.")
        sys.exit(1)
```

**Step 2: Run Stage 1**

```bash
.venv/bin/python python/examples/ilp_showcase.py
```

Expected: Stage 1 converges within 50 steps. Output like:
```
  Step  0 | tau=1.000 | loss=... | rule: ...+...->...
  ...
  >>> Converged in N steps: reach(X, Y) :- edge(X, Z), edge(Z, Y).
```

If it does NOT converge in 50 steps with 3 retries: increase `max_steps` to 100 or lower `tau_end` to 0.1. This is hyperparameter tuning — the design says retry up to 3 times.

**Step 3: Commit**

```bash
git add python/examples/ilp_showcase.py
git commit -m "feat(ilp): Stage 1 — graph reachability (warmup)"
```

---

### Task 5: Implement Stage 2 — Family Grandparent

**Files:**
- Modify: `python/examples/ilp_showcase.py`

**Step 1: Add Stage 2 config**

Add after `STAGE_1`:

```python
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
        ("grandparent", [1, 2]),  # parent, not grandparent
        ("grandparent", [3, 1]),  # inverse direction
    ],
    target_rule="grandparent(X, Y) :- parent(X, Z), parent(Z, Y).",
    rule_template="{H}(X, Y) :- {L}(X, Z), {R}(Z, Y).",
    max_steps=100,
    tau_start=2.0,
    tau_end=0.1,
    lr=0.1,
)
```

**Step 2: Wire Stage 2 into `__main__`**

After Stage 1's `run_stage`, add:

```python
    rule, ok, steps = run_stage(STAGE_2)
    results.append(("Stage 2", STAGE_2.name, rule, ok, steps))
```

**Step 3: Run both stages**

```bash
.venv/bin/python python/examples/ilp_showcase.py
```

Expected: Stage 1 converges quickly. Stage 2 converges within 100 steps — this is the hard one with distractors (gender, sibling) and negative examples. The missed-positive penalty should be visible in early steps (when loss has a negative term from `M_soft[:,:,k].sum()`).

**Step 4: Commit**

```bash
git add python/examples/ilp_showcase.py
git commit -m "feat(ilp): Stage 2 — family grandparent (negatives + distractors)"
```

---

### Task 6: Implement Stage 3 — Workplace Colleague (different join pattern)

**Files:**
- Modify: `python/examples/ilp_showcase.py`

**Step 1: Add Stage 3 config**

Add after `STAGE_2`:

```python
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
        ("colleague", [1, 2]),  # different companies
        ("colleague", [3, 4]),  # different companies
    ],
    target_rule="colleague(X, Y) :- worksAt(X, Z), worksAt(Y, Z).",
    # Shared-endpoint join: bR(Y, Z) not bR(Z, Y) — head_projection is [0, 2]
    rule_template="{H}(X, Y) :- {L}(X, Z), {R}(Y, Z).",
    max_steps=100,
    tau_start=2.0,
    tau_end=0.1,
    lr=0.1,
)
```

**Note on reflexive pairs:** `worksAt(X,Z) join worksAt(Y,Z)` naturally derives `colleague(X,X)`. The positive examples only include X != Y pairs. Self-pairs are semantically valid but excluded from our evaluation. The `fact_exists` check only looks at our positive list (all X != Y), so self-pairs don't affect convergence detection.

**Step 2: Wire Stage 3 into `__main__`**

After Stage 2's `run_stage`, add:

```python
    rule, ok, steps = run_stage(STAGE_3)
    results.append(("Stage 3", STAGE_3.name, rule, ok, steps))
```

**Step 3: Run all 3 stages**

```bash
.venv/bin/python python/examples/ilp_showcase.py
```

Expected: Stage 3 converges. The discovered rule uses the shared-endpoint join pattern: `colleague(X, Y) :- worksAt(X, Z), worksAt(Y, Z).` The key verification here is that the system handles `bR(Y, Z)` (left[1]=right[1]) correctly, not just chain joins.

**Step 4: Commit**

```bash
git add python/examples/ilp_showcase.py
git commit -m "feat(ilp): Stage 3 — workplace colleague (shared-endpoint join)"
```

---

### Task 7: Implement Stage 4 — Arithmetic plus2 with rule commit

**Files:**
- Modify: `python/examples/ilp_showcase.py`

**Step 1: Add Stage 4 config**

Add after `STAGE_3`:

```python
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
        ("plus2", [0, 1]),  # succ, not plus2
        ("plus2", [5, 0]),  # impossible
    ],
    target_rule="plus2(X, Y) :- succ(X, Z), succ(Z, Y).",
    rule_template="{H}(X, Y) :- {L}(X, Z), {R}(Z, Y).",
    max_steps=100,
    tau_start=2.0,
    tau_end=0.05,  # Extra-cold for maximum discreteness
    lr=0.1,
    commit=True,  # This stage commits the discovered rule
)
```

**Step 2: Add commit_and_validate function**

Add after the `train_stage` function:

```python
def commit_and_validate(
    config: StageConfig, rule_str: str,
) -> bool:
    """Commit a discovered rule and validate post-commit derivations.

    1. Recompiles the base source (without learnable) + committed rule
    2. Evaluates the recompiled program
    3. Checks fact_exists for all positive examples
    4. Checks fact_exists returns False for all negative examples
    """
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
```

**Step 3: Wire Stage 4 with commit into `__main__`**

After Stage 3's `run_stage`, add:

```python
    rule, ok, steps = run_stage(STAGE_4)
    if ok and STAGE_4.commit:
        print(f"\n  --- Post-Convergence: Rule Commit ---")
        commit_ok = commit_and_validate(STAGE_4, rule)
        if commit_ok:
            print(f"  >>> Commit validated: all facts correct")
        else:
            print(f"  >>> Commit validation FAILED")
            ok = False
        results.append(("Stage 4", STAGE_4.name, rule, ok, steps, "committed"))
    else:
        results.append(("Stage 4", STAGE_4.name, rule, ok, steps, ""))
```

Also update the summary to handle the optional commit tag. Replace the summary block with:

```python
    # Summary
    print(f"\n{'='*60}")
    print("  XLOG dILP Showcase -- All Stages Complete")
    print(f"{'='*60}")
    for entry in results:
        label, name, rule, ok, steps = entry[0], entry[1], entry[2], entry[3], entry[4]
        tag = f" ({entry[5]})" if len(entry) > 5 and entry[5] else ""
        status = f"OK {steps} steps{tag}" if ok else "FAILED"
        print(f"  {label}: {rule:<50s} {status}")
    print(f"{'='*60}")

    if not all(entry[3] for entry in results):
        print("\nSome stages did not converge. See output above.")
        sys.exit(1)
    else:
        print("\nAll stages converged and validated successfully.")
```

**Step 4: Run all 4 stages**

```bash
.venv/bin/python python/examples/ilp_showcase.py
```

Expected: All 4 stages converge. Stage 4 additionally shows the commit validation:
```
  --- Post-Convergence: Rule Commit ---
  Committing: plus2(X, Y) :- succ(X, Z), succ(Z, Y).
    plus2(0, 2) = OK
    plus2(1, 3) = OK
    plus2(2, 4) = OK
    plus2(3, 5) = OK
    plus2(0, 1) = OK (absent)
    plus2(5, 0) = OK (absent)
  >>> Commit validated: all facts correct
```

**Step 5: Commit**

```bash
git add python/examples/ilp_showcase.py
git commit -m "feat(ilp): Stage 4 — arithmetic plus2 with rule commit + validation"
```

---

### Task 8: Add pytest integration test

**Files:**
- Create: `python/tests/test_ilp_showcase.py`

**Step 1: Write a pytest wrapper that runs the showcase as a subprocess**

The showcase is designed as a standalone script, but we want CI to catch regressions. A test that runs it as a subprocess and checks exit code is the simplest integration point.

```python
"""Integration test: run the dILP showcase end-to-end.

This test runs the showcase script as a subprocess and verifies
all 4 stages converge. It's an integration smoke test, not a unit test.
"""

import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required for ILP showcase", allow_module_level=True)


def test_ilp_showcase_all_stages_converge():
    """Run the showcase script and verify exit code 0 (all stages converged)."""
    result = subprocess.run(
        [sys.executable, "python/examples/ilp_showcase.py"],
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout for 4 stages
        cwd="/home/dev/projects/xlog",
    )

    # Print output for debugging on failure
    if result.returncode != 0:
        print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

    assert result.returncode == 0, (
        f"Showcase exited with code {result.returncode}.\n"
        f"Last output: {result.stdout[-500:]}"
    )
    assert "All stages converged" in result.stdout, (
        f"Expected convergence message not found.\n"
        f"Last output: {result.stdout[-500:]}"
    )
```

**Step 2: Run the test**

```bash
.venv/bin/python -m pytest python/tests/test_ilp_showcase.py -v --timeout=300
```

Expected: `PASSED` — the showcase runs to completion with all stages converging.

**Step 3: Commit**

```bash
git add python/tests/test_ilp_showcase.py
git commit -m "test(ilp): add integration test for dILP showcase (all 4 stages)"
```

---

### Task 9: Hyperparameter tuning and convergence hardening

This task is conditional — only needed if any stage fails to converge reliably.

**Files:**
- Modify: `python/examples/ilp_showcase.py`

**Step 1: Run the showcase 5 times to check reliability**

```bash
for i in 1 2 3 4 5; do echo "=== Run $i ==="; .venv/bin/python python/examples/ilp_showcase.py; echo "Exit: $?"; done
```

**Step 2: If any run fails, apply these fixes in order:**

1. **Increase retries to 5** for the failing stage
2. **Lower tau_end** (more aggressive annealing): 0.1 → 0.05
3. **Increase max_steps**: 100 → 200
4. **Lower lr**: 0.1 → 0.05 (more stable but slower)
5. **Add more positive examples** if the domain allows it

**Step 3: If all 5 runs pass, commit any tuning changes**

```bash
git add python/examples/ilp_showcase.py
git commit -m "fix(ilp): tune hyperparameters for reliable convergence"
```

---

### Task 10: Final validation and cleanup

**Files:**
- Verify: `python/examples/ilp_showcase.py`
- Verify: `python/tests/test_ilp_showcase.py`

**Step 1: Run the full Rust test suite to ensure no regressions**

```bash
cargo test --workspace --all-targets --exclude pyxlog --release
```

Expected: All tests pass.

**Step 2: Run all Python ILP tests**

```bash
.venv/bin/python -m pytest python/tests/test_ilp.py python/tests/test_ilp_showcase.py -v
```

Expected: All tests pass (9 existing + 1 new showcase test).

**Step 3: Run the showcase one final time with output visible**

```bash
.venv/bin/python python/examples/ilp_showcase.py
```

Expected: Clean output showing all 4 stages converging with the final summary.

**Step 4: Commit any final cleanup**

```bash
git add -A
git commit -m "feat(ilp): complete dILP showcase — 4-stage progressive knowledge graph discovery"
```
