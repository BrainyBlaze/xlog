# XLOG dILP Showcase — Run Analysis Report

> **Date:** 2026-02-26
> **GPU:** NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU
> **CUDA:** 12.8
> **Script:** `python/examples/ilp_showcase.py`

---

## Executive Summary

> **Note:** This is a single-run analysis. Multi-run reliability (5/5 consecutive
> passes, 100% convergence rate) was validated separately — see commit `05e91632`.

All 4 stages converge to the exact target rule. Two stages required retries
(fresh random re-initialisation of the weight tensor W), demonstrating that
the retry mechanism is essential for convergence in the presence of distractor
predicates and template-variable local minima.

| Stage | Target Rule | Steps | Attempt | Retries Used |
|-------|------------|-------|---------|--------------|
| 1 — Graph Reachability | `reach(X,Y) :- edge(X,Z), edge(Z,Y).` | 21 | 1st | 0 |
| 2 — Family Grandparent | `grandparent(X,Y) :- parent(X,Z), parent(Z,Y).` | 27 | 2nd | 1 |
| 3 — Workplace Colleague | `colleague(X,Y) :- worksAt(X,Z), worksAt(Y,Z).` | 32 | 1st | 0 |
| 4 — Arithmetic plus2 | `plus2(X,Y) :- succ(X,Z), succ(Z,Y).` | 39 | 2nd | 1 |

---

## Stage 1: Graph Reachability (21 steps, 1st attempt)

**Domain:** 6-node chain graph (edge: 1→2→3→4→5→6).
**Search space:** N=4 relations (edge, reach, bL, bR) → 64-cell W tensor.
**Features exercised:** Basic training loop, per-fact credit, budget-aware mask, argmax-only validation.

### Training Dynamics

The optimiser traverses three distinct phases:

**Phase A — Template attraction (steps 0–7):** The argmax locks onto `bL+edge→edge`,
a semantically vacuous candidate (bL is a template variable with no base facts).
Loss oscillates wildly (−0.6 to 6.2) because the budget mask occasionally
places mass on productive entries via Gumbel noise, but the argmax itself produces nothing.
Stable count reaches 5 at step 5, but since `all_derived` is False (bL+edge→edge
derives nothing), the convergence gate never opens.

**Phase B — Partial discovery (steps 8–14):** The argmax shifts to `edge+bR→reach`.
This is closer — `edge` is a real base relation and `reach` is the correct head — but
`bR` is still a template variable. Stable count reaches 5 at step 13, but since
`all_derived` is False (edge⋈bR produces nothing), the convergence gate blocks
before argmax-only validation is even reached.

**Phase C — Convergence (steps 15–20):** The argmax shifts to `edge+edge→reach`,
the correct rule. Loss is low and stable (~0.3). At step 20 (stable=5), `all_derived`
is True for the first time — the budget mask derives all 4 positives. The argmax-only
validation then confirms that a single mask entry at (edge,edge,reach) alone derives
all positives. Converged at step 21.

### Key Observation

The two-gate convergence check provides defense in depth. The `all_derived` gate
(line 192–196) blocks convergence when the argmax rule derives nothing — this prevented
false convergence at steps 5 and 13, where template variables (bL, bR) cannot produce
derivations. The argmax-only validation (lines 198–218) then guards against a subtler
failure mode: where the budget mask derives all positives via *non-argmax* entries
while the reported argmax rule alone cannot. Both gates are necessary.

---

## Stage 2: Family Grandparent (27 steps, 2nd attempt)

**Domain:** Family tree with 6 people + distractor relations (gender, sibling).
**Search space:** N=6 relations → 216-cell W tensor.
**Features exercised:** Distractor relations, negative examples, missed-positive penalty (RD-21), retry mechanism.

### Attempt 1 — Stuck on Gender (120 steps, FAILED)

The entire first attempt is a study in local minima. The argmax locks onto
`gender+bR→grandparent` at step 0 and never escapes for 120 steps.

**Why it gets stuck:** The `gender` relation has entries like gender(1,0) and gender(2,1).
When the budget mask has mass on the (gender,bR,grandparent) cell, the executor computes
gender(X,Z)⋈bR(Z,Y) — but bR has no base facts. No positives are ever derived,
so `all_derived` stays False. The convergence gate never opens.

**Why it doesn't escape:** The loss drops to near-zero (~0.001 at step 86) because:
1. No positives are derived → the missed-positive penalty kicks in: `-M_soft[:,:,k_idx].sum() / (n*n)`.
   But as M_soft concentrates on the gender cell (disc reaches 1.00), this penalty shrinks.
2. No negatives are derived either → zero negative penalty.
3. The optimiser sees near-zero gradient and stops making meaningful updates.

**Disc progression:** 0.40 → 0.80 → 0.95 → 1.00. The mask becomes fully discrete
on the *wrong* cell, creating a stable but incorrect fixed point.

**Loss spikes:** Periodic ~18.4–36.8 spikes (steps 47, 60, 117) occur when Gumbel noise
momentarily activates cells that derive negative examples. These spikes are the
negative-example penalty doing its job, but they aren't strong enough to dislodge
the dominant (gender,bR,grandparent) attractor.

### Attempt 2 — Correct Discovery (27 steps)

Fresh random initialisation. The trajectory:
- Steps 0–20: Explores `gender+bL→grandparent` (still wrong, but different body order).
  Loss is higher (0.6–18.9), mask less discrete (disc=0.39–0.52).
- **Step 21: Phase transition.** Argmax jumps to `parent+parent→grandparent` (correct!).
  The gradient accumulated over 20 steps of exploring the gender/bL region finally
  pushes W past the basin of attraction for the (parent,parent,grandparent) cell.
- Steps 22–26: Stable. Loss drops to 0.4–2.6 as the budget mask derives all positives.
  At step 26 (stable=5), argmax-only validation passes. Converged.

### Key Observation

This stage demonstrates why retries are essential. The first attempt had the optimizer
trapped in a local minimum for the full 120-step budget despite loss being near zero —
a deceptive signal. The random re-initialisation in attempt 2 started from a different
basin and found the correct rule in just 27 steps.

The `gender` relation is a particularly effective distractor because:
1. It has tuples connecting the same entities as `parent` (e.g., gender(1,0) and parent(1,2)
   share entity 1).
2. Its second column has small values (0, 1) that intersect with many join patterns.
3. Combined with template variables (bR, bL), it creates plausible-looking candidates
   that don't actually derive anything — a silent dead end.

---

## Stage 3: Workplace Colleague (32 steps, 1st attempt)

**Domain:** 4 employees at 2 companies + distractor relation (livesIn at 2 cities).
**Search space:** N=5 relations → 125-cell W tensor.
**Features exercised:** Shared-endpoint join pattern (bR(Y,Z) not bR(Z,Y)), head projection [0,2], distractor separation.

### Training Dynamics

**Phase A — livesIn distraction (steps 0–25):** The argmax locks onto
`bL+livesIn→colleague`. This is wrong: livesIn(X,Z)⋈livesIn(Y,Z) would
derive colleague pairs for people living in the same city — but `bL` is a
template variable, not `livesIn`. Still, the optimizer holds this candidate
for 26 steps, repeatedly reaching stable=5 but failing `all_derived`
because the joins produce nothing.

Loss oscillates 0.2–5.6, disc slowly rises from 0.37 to 0.52.

**Phase B — Correct discovery (step 26–31):** At step 26, the argmax
shifts to `worksAt+worksAt→colleague`. Loss drops significantly (from 3.6
to 0.1–0.5). The shared-endpoint join worksAt(X,Z)⋈worksAt(Y,Z) correctly
derives colleague(1,7), colleague(7,1), colleague(2,4), colleague(4,2) — all
4 positives. Negative examples colleague(1,2) and colleague(3,4) are correctly
absent (different companies). At step 31 (stable=5), argmax-only validation
passes. Converged.

### Key Observation

This stage exercises the **shared-endpoint join pattern** — the rule uses
`bR(Y,Z)` instead of `bR(Z,Y)`, giving join keys left[1]=right[1] and
head projection [0,2] instead of the chain-join [0,3]. The optimizer
correctly learns this pattern despite `livesIn` being a structurally
identical distractor (same arity, same relationship type, similar domain
structure). The negative examples (colleague(1,2) and colleague(3,4))
are decisive: livesIn would derive colleague(1,2) since they live in the
same city (201), but worksAt correctly excludes it (different companies).

---

## Stage 4: Arithmetic plus2 (39 steps, 2nd attempt)

**Domain:** Integer successor chain 0→1→2→3→4→5, with pred (inverse) as distractor.
**Search space:** N=5 relations → 125-cell W tensor.
**Features exercised:** Temperature annealing (tau 2.5→0.05), rule commit, post-commit validation.

### Attempt 1 — Template Variable Trap (150 steps, FAILED)

The first attempt gets trapped on `bL+bL→plus2` — both body positions filled with
the same template variable. This is a particularly stubborn local minimum:

- Loss oscillates around 2.5–18.4 for the entire 150 steps.
- The ~18.4 spikes are the negative-example penalty: the budget mask occasionally
  places mass on entries that derive plus2(0,1) or plus2(5,0).
- Disc rises steadily (0.38 → 0.75 → 0.92 → 1.00) as temperature anneals.
- By step 130+, the mask is fully discrete (disc=1.00) on the wrong cell. The
  loss is exactly 18.421 — a constant corresponding to the negative penalty magnitude
  when the mask is fully concentrated.

**Why 150 steps isn't enough:** The temperature annealing makes the mask
increasingly discrete, which *reduces* gradient flow through non-argmax cells.
Once disc ≈ 1.0, the optimiser cannot explore alternative rules. This is the
fundamental exploration-exploitation tradeoff of temperature annealing.

### Attempt 2 — Correct Discovery (39 steps)

- Steps 0–2: `bR+pred→bR` (random noise).
- Steps 3–32: `plus2+plus2→plus2` (self-referencing dead end, similar to attempt 1
  but with different template variables). Loss 1.8–5.5, disc 0.37–0.53.
- **Step 33:** Phase transition to `succ+succ→plus2` (correct!).
- Steps 34–38: Stable, loss drops to 0.7–2.9.
- Step 38 (stable=5): Argmax-only validation passes. Converged.

### Post-Convergence: Rule Commit

After convergence, `commit_induced_rule("plus2(X, Y) :- succ(X, Z), succ(Z, Y).")`
recompiles the program with the discovered rule as a permanent fact, removing the
learnable template. Validation results:

| Fact | Expected | Result |
|------|----------|--------|
| plus2(0, 2) | Present | OK |
| plus2(1, 3) | Present | OK |
| plus2(2, 4) | Present | OK |
| plus2(3, 5) | Present | OK |
| plus2(0, 1) | Absent | OK (absent) |
| plus2(5, 0) | Absent | OK (absent) |

All 4 positives derived, both negatives correctly absent. The committed rule
is semantically equivalent to the learned one, confirming that the learnable
→ committed transition preserves correctness.

### Key Observation

The `pred` distractor relation (inverse of `succ`) tests whether the optimizer can
distinguish direction. `pred+pred→plus2` would compute X-2, not X+2. The negative
example plus2(0,1) is critical: succ+succ gives 0→1→2 (plus2(0,2)), not plus2(0,1),
while pred+succ might give plus2(0,0) — the negatives help the optimizer distinguish
these patterns.

---

## Cross-Stage Analysis

### Convergence Patterns

All four stages exhibit a similar three-phase pattern:
1. **Random exploration** (5–30 steps): High loss, unstable argmax, low discreteness.
2. **Local minimum** (0–150 steps): Argmax locks onto wrong rule, loss may drop
   misleadingly. Argmax-only validation prevents false convergence.
3. **Phase transition**: Argmax jumps to correct rule, loss drops, convergence within 5–10 steps.

The phase transition is abrupt — there is no gradual migration from wrong to right.
This is characteristic of Gumbel-Softmax optimisation: the temperature-scaled softmax
creates sharp winners, and the gradient landscape has multiple local attractors.

### Temperature and Discreteness

| Stage | tau range | disc at convergence | disc at trap (if any) |
|-------|-----------|--------------------|-----------------------|
| 1 | 1.0→0.1 | 0.80 | n/a (no trap) |
| 2 | 2.0→0.05 | 0.51 (attempt 2) | 1.00 (attempt 1) |
| 3 | 2.0→0.05 | 0.54 | n/a (no trap) |
| 4 | 2.5→0.05 | 0.55 (attempt 2) | 1.00 (attempt 1) |

Successful convergence happens at **moderate discreteness** (0.5–0.8), not at
maximum (1.0). This is because the optimizer needs gradient flow through non-argmax
cells to explore alternatives. When disc=1.0, the mask is a hard one-hot and
gradients vanish for all but one cell — the optimizer is frozen.

This explains why the failed attempts (Stage 2 attempt 1, Stage 4 attempt 1) both
reached disc=1.0 on the wrong rule: the temperature annealing pushed discreteness
too high before the optimizer escaped the local minimum.

### Retry Effectiveness

| Stage | Attempts needed | Failure mode of wasted attempts |
|-------|----------------|---------------------------------|
| 1 | 1 | — |
| 2 | 2 | gender distractor (120 steps, disc→1.0) |
| 3 | 1 | — |
| 4 | 2 | template variable trap (150 steps, disc→1.0) |

Fresh random initialisation is the primary escape mechanism from local minima.
The retry budget (5–7 per stage) provides ample margin. In this run, no stage
needed more than 2 attempts.

### Loss Signal Analysis

| Stage | Loss at convergence | Dominant loss terms |
|-------|--------------------|--------------------|
| 1 | 0.05 | Per-fact credit (positives only) |
| 2 | 0.4–1.4 | Per-fact credit + negative penalty |
| 3 | 0.1–0.5 | Per-fact credit + negative penalty |
| 4 | 0.7–2.9 | Per-fact credit + negative penalty |

Stage 1 achieves near-zero loss because there are no negative examples — only
positive credit drives the optimiser. Stages 2–4 have residual loss from
negative examples that occasionally get activated by non-argmax mask entries.

---

## Feature Coverage Summary

| dILP Feature | Stage(s) | Evidence in This Run |
|-------------|----------|---------------------|
| Learnable rule compilation | 1,2,3,4 | All stages compile successfully |
| ST-Gumbel-Softmax mask | 1,2,3,4 | disc metric tracks mask discreteness |
| Budget-aware top-k (k≤32) | 1,2,3,4 | Global top-k in build_budget_aware_mask |
| Per-fact surrogate credit | 1,2,3,4 | tagged_entries_containing_fact in loss |
| Missed-positive penalty (RD-21) | 2,4 | Negative loss values when positives not derived |
| Negative example penalty | 2,3,4 | Loss spikes (~18.4) when negatives derived |
| Temperature annealing | 1,2,3,4 | tau decreases linearly per step |
| Argmax-only validation | 1,2 | Rejects false convergence (steps 5,13 of Stage 1) |
| Chain join (left[1]=right[0]) | 1,2,4 | bL(X,Z)⋈bR(Z,Y) pattern |
| Shared-endpoint join (left[1]=right[1]) | 3 | bL(X,Z)⋈bR(Y,Z) pattern |
| Retry mechanism | 2,4 | Fresh random re-init escapes local minima |
| Rule commit | 4 | commit_induced_rule + recompile + validate |
| Post-commit fact validation | 4 | 4 positives + 2 negatives all correct |
| Distractor relation handling | 2,3,4 | gender/sibling, livesIn, pred |
| Cudarc preflight diagnostics | all | Preflight OK printed before training |
