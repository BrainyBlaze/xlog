# Progressive Knowledge Graph Discovery — dILP Showcase Design

**Status:** Approved
**Date:** 2026-02-25
**Depends on:** Tensorized ILP (merged to main, PR #1)

---

## 1. Goal

A single Python script (`python/examples/ilp_showcase.py`) that demonstrates
**every feature** of XLOG's Tensorized Differentiable ILP engine through 4
sequential stages. Each stage is an independent compilation with one learnable
rule, progressively introducing new capabilities. Every stage must actually
converge to the correct target rule.

---

## 2. Hard Invariants

1. **One learnable rule per compilation.** `extract_tmj_meta` returns only the
   first TMJ profile (`pyxlog/src/lib.rs:3707`), and the executor's
   `ilp_last_result` is a single overwrite slot (`executor.rs:183`).
   Multi-learnable-per-compilation would mis-attribute credit.

2. **Budget-aware hard mask (k ≤ 32).** `max_active_rules` is fixed at 32 in
   lowering (`lower.rs:524`), and the GPU kernel truncates by M_soft priority
   (`provider.rs:10616`). Per-(i,j) argmax produces up to N² entries
   (N=7 → 49 > 32). Python must pre-select global top-k in the hard mask to
   stay in sync with what the executor actually processes.

3. **cudarc preflight.** A tiny ILP compile → evaluate smoke check at startup.
   Fail fast with actionable environment diagnostics (LD_LIBRARY_PATH, CUDA
   driver version, GPU memory) instead of relying solely on
   `torch.cuda.is_available()`.

4. **Reflexive colleague handling.** Stage 3's `worksAt(X,Z) ⋈ worksAt(Y,Z)`
   naturally derives self-pairs `colleague(X,X)`. The demo treats this as
   semantically correct (X is trivially their own colleague) but excludes
   self-pairs from **evaluation metrics** (precision/recall computed on X ≠ Y
   pairs only). The rule itself is not modified.

---

## 3. Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     ilp_showcase.py                             │
├────────────────────────────────────────────────────────────────┤
│ preflight_check()          → fail-fast env diagnostics         │
│ train_stage(config)        → shared training loop              │
│ compute_loss(...)          → per-fact credit + penalties        │
│ build_budget_aware_mask()  → global top-k hard selection        │
│ decode_rule(...)           → (i,j,k) → relation name string    │
│ commit_and_validate(...)   → commit_induced_rule + fact_exists  │
├────────────────────────────────────────────────────────────────┤
│ Stage 1: Graph Reachability     (N≈4,  warmup)                 │
│ Stage 2: Family Grandparent     (N≈7,  credit + penalties)     │
│ Stage 3: Workplace Colleague    (N≈6,  join diversity)         │
│ Stage 4: Arithmetic plus2       (N≈5,  commit + certification) │
└────────────────────────────────────────────────────────────────┘
```

Each stage:
1. Compiles an independent xlog source with one `learnable(W) ::` rule
2. Runs a training loop with budget-aware ST-Gumbel-Softmax
3. Logs per-step metrics: loss, τ, argmax rule, discreteness, derived facts
4. Converges to the correct rule within its step budget
5. Reports the discovered rule decoded as human-readable Datalog

---

## 4. Budget-Aware Hard Mask Construction

The per-(i,j) argmax approach from the RFC produces up to N² active rules,
which exceeds the executor's max_active_rules=32 for N ≥ 6. Instead, use
**global top-k selection**:

```python
def build_budget_aware_mask(W, tau, budget=32):
    """ST-Gumbel-Softmax with global top-k hard selection.

    Instead of per-(i,j) argmax (up to N² entries), select the top-k
    entries globally by M_soft value. This keeps the hard mask in sync
    with what the executor actually processes after GPU-side truncation.
    """
    M_soft = F.gumbel_softmax(W, tau=tau, hard=False, dim=-1)

    flat = M_soft.view(-1)
    k = min(budget, flat.numel())
    _, topk_idx = flat.topk(k)

    M_hard_flat = torch.zeros_like(flat)
    M_hard_flat[topk_idx] = 1.0
    M_hard = M_hard_flat.view_as(M_soft)

    return M_hard, M_soft
```

This guarantees ≤ 32 non-zero entries in M_hard. The executor sees exactly the
rules Python selected — no silent truncation mismatch.

Gradient flow is unchanged: loss is computed using M_soft at the top-k
positions. M_hard is detached before DLPack export (RD-16).

---

## 5. Stage Definitions

### Stage 1: Graph Reachability (Warmup)

**Source:**
```xlog
edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
```

**Target rule:** `reach(X,Y) :- edge(X,Z), edge(Z,Y).`

**Join pattern:** Chain — left[1] = right[0]

**Schema:** N ≈ 4 (edge, reach, bL, bR). Search space: 64.

**Positive examples:** `reach(1,3), reach(2,4), reach(3,5), reach(4,6)`

**Negative examples:** None (warmup stage, positive-only).

**Budget:** max_steps=50, τ: 1.0 → 0.3, lr=0.1

**Features exercised:**
- `learnable(W) ::` syntax
- 3D Gumbel-Softmax + ST estimator
- Budget-aware hard mask (k=32, though N²=16 < 32 here)
- Per-fact surrogate credit (`tagged_entries_containing_fact`)
- Full training loop with Adam

---

### Stage 2: Family Grandparent (Credit & Penalties)

**Source:**
```xlog
parent(1, 2). parent(2, 3). parent(2, 4). parent(3, 5). parent(4, 6).
gender(1, 0). gender(2, 1). gender(3, 1). gender(4, 0).
sibling(2, 7). sibling(7, 2).
learnable(W_gp) :: grandparent(X, Y) :- bL(X, Z), bR(Z, Y).
```

**Target rule:** `grandparent(X,Y) :- parent(X,Z), parent(Z,Y).`

**Join pattern:** Chain — left[1] = right[0]

**Schema:** N ≈ 7 (parent, gender, sibling, grandparent, bL, bR + possible
extra from parser). Search space: ~343. N²=49 > 32, so top-k budget
enforcement is **active and load-bearing** in this stage.

**Positive examples:** `grandparent(1,3), grandparent(1,4), grandparent(2,5), grandparent(2,6)`

**Negative examples:** `grandparent(1,2), grandparent(3,1)`
- (1,2) is parent, not grandparent — prevents the optimizer from picking
  `parent ⋈ identity`
- (3,1) is inverse grandparent — prevents picking the wrong direction

**Budget:** max_steps=100, τ: 2.0 → 0.1, lr=0.1

**Features exercised (new):**
- Distractor relations (gender, sibling) — optimizer must ignore noise
- Negative examples — prevents over-generalization
- Missed-positive penalty (RD-21) — early steps likely won't derive positives

---

### Stage 3: Workplace Colleague (Join Diversity)

**Source:**
```xlog
worksAt(1, 101). worksAt(7, 101). worksAt(2, 102). worksAt(4, 102).
livesIn(1, 201). livesIn(2, 201). livesIn(3, 202). livesIn(4, 202).
learnable(W_col) :: colleague(X, Y) :- bL(X, Z), bR(Y, Z).
```

**Target rule:** `colleague(X,Y) :- worksAt(X,Z), worksAt(Y,Z).`

**Join pattern:** Shared-endpoint — left[1] = right[1] (NOT chain!)
- `bL(X, Z)` → col 0=X, col 1=Z
- `bR(Y, Z)` → col 0=Y, col 1=Z
- Z is the shared variable at positions left[1] and right[1]

**Head projection:** [0, 2] — X from left col 0, Y from right col 0 (which
is col 2 in joined result since left_arity=2).

**Schema:** N ≈ 6 (worksAt, livesIn, colleague, bL, bR + extras).

**Positive examples:** `colleague(1,7), colleague(7,1), colleague(2,4), colleague(4,2)`

**Negative examples:** `colleague(1,2), colleague(3,4)`
- (1,2) — Alice and Bob are at different companies
- (3,4) — Charlie and Diana are at different companies

**Reflexive behavior:** `colleague(1,1), colleague(2,2)`, etc. are naturally
derived. These are semantically valid (X is their own colleague) but excluded
from evaluation metrics. Only X ≠ Y pairs count for precision/recall.

**Budget:** max_steps=100, τ: 2.0 → 0.1, lr=0.1

**Features exercised (new):**
- Different join pattern (shared-endpoint vs. chain)
- Different head projection ([0,2] vs. [0,3])
- Distractor: `livesIn` has same shape as `worksAt` but wrong semantics

---

### Stage 4: Arithmetic plus2 (Commit & Certification)

**Source:**
```xlog
succ(0, 1). succ(1, 2). succ(2, 3). succ(3, 4). succ(4, 5).
pred(1, 0). pred(2, 1). pred(3, 2). pred(4, 3). pred(5, 4).
learnable(W_p2) :: plus2(X, Y) :- bL(X, Z), bR(Z, Y).
```

**Target rule:** `plus2(X,Y) :- succ(X,Z), succ(Z,Y).`

**Join pattern:** Chain — left[1] = right[0]

**Schema:** N ≈ 5 (succ, pred, plus2, bL, bR). Distractor: `pred`
(reverse of succ — if selected as body, would produce different derivations).

**Positive examples:** `plus2(0,2), plus2(1,3), plus2(2,4), plus2(3,5)`

**Negative examples:** `plus2(0,1), plus2(5,0)`
- (0,1) is succ, not plus2
- (5,0) is impossible — prevents wrap-around

**Budget:** max_steps=100, τ: 2.0 → 0.05 (extra-cold final τ for maximum
discreteness), lr=0.1

**Post-convergence actions:**
1. Decode argmax (i,j,k) → "plus2(X,Y) :- succ(X,Z), succ(Z,Y)."
2. Call `prog.commit_induced_rule("plus2(X,Y) :- succ(X,Z), succ(Z,Y).")`
3. Call `prog.evaluate()` on the recompiled program
4. Validate: `prog.fact_exists("plus2", [0, 2])` returns True for all positives
5. Validate: `prog.fact_exists("plus2", [0, 1])` returns False for negatives

**Features exercised (new):**
- Temperature annealing visualization (discreteness metric over τ schedule)
- Rule commit (`commit_induced_rule`)
- Post-commit validation (`fact_exists`)
- Convergence certification (rule matches target exactly)

---

## 6. Cudarc Preflight

```python
def preflight_check():
    """Fail fast with actionable diagnostics."""
    import sys, os

    # 1. CUDA runtime
    if not torch.cuda.is_available():
        print("FATAL: torch.cuda.is_available() == False")
        print("  Check: nvidia-smi, CUDA driver, LD_LIBRARY_PATH")
        sys.exit(1)

    # 2. cudarc / pyxlog import
    try:
        import pyxlog
    except ImportError as e:
        print(f"FATAL: Cannot import pyxlog: {e}")
        print("  Check: maturin develop --release in xlog root")
        sys.exit(1)

    # 3. ILP compile/evaluate smoke
    try:
        prog = pyxlog.IlpProgramFactory.compile(
            "edge(1,2). learnable(W) :: r(X,Y) :- b1(X,Z), b2(Z,Y).",
            device=0, memory_mb=256,
        )
        n = prog.ilp_schema_size()
        W = torch.zeros((n, n, n), device='cuda')
        flat = W.contiguous().view(-1)
        prog.set_rule_mask("W", flat, flat, n)
        prog.evaluate()
    except Exception as e:
        print(f"FATAL: ILP smoke test failed: {e}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '(unset)')}")
        sys.exit(1)

    print(f"Preflight OK: GPU={torch.cuda.get_device_name(0)}, "
          f"CUDA={torch.version.cuda}, N_smoke={n}")
```

---

## 7. Training Loop Structure

```python
@dataclass
class StageConfig:
    name: str               # "Graph Reachability"
    source: str             # xlog source code
    mask_name: str          # "W_reach"
    positives: list         # [("reach", [1, 3]), ...]
    negatives: list         # [("reach", [6, 1]), ...]
    target_rule: str        # "reach(X,Y) :- edge(X,Z), edge(Z,Y)."
    max_steps: int          # 50-100
    tau_start: float        # 2.0
    tau_end: float          # 0.1
    lr: float               # 0.1
    budget: int             # 32
    commit: bool            # True for Stage 4 only

def train_stage(config: StageConfig) -> tuple[str, bool]:
    """Train one stage. Returns (discovered_rule, converged)."""
    prog = pyxlog.IlpProgramFactory.compile(
        config.source, device=0, memory_mb=512
    )
    n = prog.ilp_schema_size()
    rel_names = prog.ilp_relation_names()

    W = torch.randn((n, n, n), requires_grad=True, device='cuda')
    optimizer = torch.optim.Adam([W], lr=config.lr)
    prev_argmax = None
    stable_count = 0

    for step in range(config.max_steps):
        tau = config.tau_start + (config.tau_end - config.tau_start) * step / config.max_steps
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
            prog, M_soft, config.positives, config.negatives, rel_names, n
        )

        if loss.requires_grad:
            loss.backward()
            optimizer.step()

        # Decode current best rule
        argmax = decode_argmax(W, rel_names)
        if argmax == prev_argmax:
            stable_count += 1
        else:
            stable_count = 0
        prev_argmax = argmax

        # Log
        log_step(step, loss, tau, argmax, stable_count)

        # Convergence: stable argmax for 5 steps + all positives derived
        if stable_count >= 5:
            all_derived = all(
                prog.fact_exists(rel, vals)
                for rel, vals in config.positives
            )
            if all_derived:
                return argmax, True

    return prev_argmax, False
```

---

## 8. Loss Function

```python
def compute_loss(prog, M_soft, positives, negatives, rel_names, n):
    loss = torch.tensor(0.0, device='cuda')

    # Positive examples: credit or missed-positive penalty
    for rel_name, values in positives:
        contributing = prog.tagged_entries_containing_fact(rel_name, values)
        if contributing:
            credit = sum(M_soft[i, j, k] for (i, j, k) in contributing)
            loss = loss + (-torch.log(credit.clamp(min=1e-8)))
        else:
            # RD-21: Missed-positive penalty
            k_idx = rel_names.index(rel_name)
            loss = loss + (-M_soft[:, :, k_idx].sum() / (n * n))

    # Negative examples: penalize rules that derive them
    for rel_name, values in negatives:
        contributing = prog.tagged_entries_containing_fact(rel_name, values)
        if contributing:
            credit = sum(M_soft[i, j, k] for (i, j, k) in contributing)
            loss = loss + (-torch.log((1.0 - credit).clamp(min=1e-8)))

    return loss
```

---

## 9. Success Criteria

| Stage | Convergence Budget | Pass Criterion |
|-------|--------------------|----------------|
| 1 | 50 steps, 3 retries | Correct rule, all 4 positives derived |
| 2 | 100 steps, 3 retries | Correct rule, all 4 positives derived, 0 negatives |
| 3 | 100 steps, 3 retries | Correct rule, all 4 positives (X≠Y) derived |
| 4 | 100 steps, 3 retries | Correct rule, commit succeeds, post-commit fact_exists passes |

**Retry strategy:** On non-convergence, reinitialize W and retry (up to 3
times per stage). Different random init can find the basin.

---

## 10. Output Format

Per-step logging:
```
[Stage 1: Graph Reachability]
  Step  0 | τ=1.000 | loss=2.341 | rule: bR⋈bL→edge    | stable=0 | disc=0.42
  Step  1 | τ=0.986 | loss=1.823 | rule: edge⋈bR→reach  | stable=0 | disc=0.45
  ...
  Step 14 | τ=0.600 | loss=0.012 | rule: edge⋈edge→reach | stable=5 | disc=0.97
  ✓ Converged: reach(X,Y) :- edge(X,Z), edge(Z,Y).
  Verified: reach(1,3)=✓ reach(2,4)=✓ reach(3,5)=✓ reach(4,6)=✓
```

Final summary:
```
═══════════════════════════════════════════════════════
  XLOG dILP Showcase — All 4 Stages Complete
═══════════════════════════════════════════════════════
  Stage 1: reach(X,Y)       :- edge(X,Z), edge(Z,Y).       ✓ 14 steps
  Stage 2: grandparent(X,Y) :- parent(X,Z), parent(Z,Y).   ✓ 37 steps
  Stage 3: colleague(X,Y)   :- worksAt(X,Z), worksAt(Y,Z). ✓ 52 steps
  Stage 4: plus2(X,Y)       :- succ(X,Z), succ(Z,Y).       ✓ 28 steps (committed)
═══════════════════════════════════════════════════════
```

---

## 11. File Layout

```
python/examples/ilp_showcase.py    # The single showcase script
```

No other files needed. All xlog source is inline in the script.

---

## 12. Non-Goals

- Multi-learnable-per-compilation (requires per-mask metadata refactor, v0.4.0-beta)
- Mutual recursion (even/odd) — requires two linked learnable rules
- Visualization plots (matplotlib) — text logging suffices
- Benchmark timing — this is a correctness showcase, not a perf benchmark
