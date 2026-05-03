# v0.6.5 Closure Board

**Status:** v0.6.5 is NOT releasable. 21 open items required for tag.
**Source of truth:** ROADMAP.md §1203–1268 + internal commitments
created during slices 4–5.
**Last updated:** 2026-05-04.

## Process Rules (durable)

These rules govern all work toward v0.6.5 closure. They override
any contrary slice-internal decision.

1. **No item is marked DONE without explicit user approval in
   the thread.** The implementation agent does NOT self-mark
   DONE. A slice closes work; the user reviews the slice; if
   the user explicitly states "mark X DONE" (or equivalent),
   the agent updates the board in a follow-up commit. Status
   states permitted: `OPEN`, `IN-PROGRESS`, `BLOCKED`, `DONE`.
   No `deferred`, no `future-slice`, no `v0.6.6` state.
2. **No code change without a board item ID.** Every commit
   message references the board item it advances. Refactors
   that don't advance an item don't ship until the board is
   empty (or the user authorizes an explicit exception).
3. **Slice plans open with the board IDs they close.** Plan
   header lists "this slice moves items X, Y, Z from OPEN to
   IN-PROGRESS / DONE." If a plan doesn't move an item to
   DONE, the plan states why and the user confirms before the
   plan body is written.
4. **End-of-slice closure update is mandatory.** Slice
   commit-of-evidence proposes a board update (OPEN → DONE
   for each closed item) but does NOT mark DONE itself. The
   user reviews and approves; a separate commit applies the
   approved DONE markings.
5. **No `v0.6.6` reference in any new file, comment, plan,
   evidence README, or commit message until this board has
   zero OPEN items.** Existing `v0.6.6` references in shipped
   slices stay (rewriting history is its own risk); new ones
   are forbidden.
6. **Push and tag gated on this board.** `v0.6.5` tag fires
   only after every board item is DONE AND the user explicitly
   says "tag v0.6.5." No silent push, no silent tag.
7. **`BLOCKED` is a real state.** An item is `BLOCKED` when
   another board item must complete first. The blocker IDs
   are listed in the item's row. `BLOCKED` items do not move
   to `IN-PROGRESS` until every blocker is `DONE`.

## Status Tally

| State | Count |
|-------|-------|
| DONE | 0 (none in this closure cycle yet) |
| IN-PROGRESS | 1 (W1.1) |
| BLOCKED | 2 (W2.5, W2.6) |
| OPEN | 18 |
| **Total** | **21** |

The 4 ROADMAP items already DONE from slices 1, 2, 4 are
recorded in ROADMAP.md and the slice evidence READMEs; they're
not on this board because the board tracks closure work, not
prior shipped work.

## Open Items

### Wave 1 — Roadmap process

| ID | Status | Required deliverable | Acceptance gate |
|----|--------|----------------------|-----------------|
| W1.1 | IN-PROGRESS | This closure board exists, committed, with all 21 items enumerated. Process rules locked. | This file committed on `main`; ROADMAP.md points to it. User approves tally + amendments; agent then commits referencing W1.1. |

### Wave 2 — Foundation closure

| ID | Source | Status | Blocked by | Required deliverable | Acceptance gate |
|----|--------|--------|------------|----------------------|-----------------|
| W2.1 | ROADMAP item #2 | OPEN | — | Variable-ordering cost model for WCOJ. Currently dispatch/admission only. Decide which slot becomes the lookup-key vs iteration-key based on stats. | Cert that two semantically-equivalent WCOJ rules with different statistical input shapes pick different slot orderings; row-set agreement preserved. |
| W2.2 | ROADMAP item #3 | OPEN | — | Populate `xlog-logic::optimizer::selectivity_pass` with real selectivity-driven join reordering. Currently no-op. | Cert: **same rule** compiled twice with **two different stats snapshots** produces **two different chosen join orders** AND identical row sets. (Deterministic canonicalization that ignores stats does NOT pass this gate.) |
| W2.3 | ROADMAP item #6 | OPEN | — | Statistics integration into recursive SCC evaluation. Per-iteration cardinality update from delta sizes; cost model sees current iteration stats, not seed-only. | Cert: recursive triangle's cost-model decisions evolve across iterations as deltas grow; each iteration's `binary_est` reflects the iteration's actual delta, not the seed. |
| W2.4 | Internal | OPEN | — | `record_join_result` feedback wired from successful WCOJ dispatch back into `xlog-stats::StatsManager`. Tighten future cardinality estimates with observed selectivity. | Cert: same recursive program run twice (warm StatsManager) makes a different — better — dispatch decision than first run. |
| W2.5 | Internal | BLOCKED | W2.1, W2.2, W2.3, W2.4, W3.2, W4.1, W5.1, W5.2 | Default-flip `RuntimeConfig::wcoj_cost_model` from `SkewClassifier` to `Cardinality`. **Blocked until foundation + kernel + runtime + cert evidence is in hand.** Default cannot flip without proof from W5.1/W5.2 benchmarks that the cardinality model is at least at parity on representative workloads. | New default ships; slice 4 stable-triangle counter still 1 (cardinality + missing-stats safety floor delegates correctly); explicit env opt-out (`XLOG_WCOJ_COST_MODEL=skew`) restores legacy behavior; bench evidence from W5.2 documents the parity / improvement. |
| W2.6 | ROADMAP item #16 | BLOCKED | W2.1, W2.4 | Selectivity + heat statistics feed into the variable ordering from W2.1. Closes the loop between observed stats and slot ordering. **Scheduled last within Wave 2** — depends on the variable-ordering structure (W2.1) and the feedback wire (W2.4). | Cert: hot relation gets preferred lookup-key slot; cold extensional relation gets iteration-key slot; row-set agreement preserved. |

### Wave 3 — Kernel closure

| ID | Source | Status | Blocked by | Required deliverable | Acceptance gate |
|----|--------|--------|------------|----------------------|-----------------|
| W3.1 | ROADMAP item #7 | OPEN | — | Sorted relation accessors beyond the triangle layout helper. Generic `wcoj_layout_sort_*` for any 2+arity slot. | xlog-cuda cert: empty / already-sorted / unsorted+duplicated all round-trip via the new accessors at u32, u64, Symbol widths. |
| W3.2 | ROADMAP item #9 | OPEN | — | General-arity WCOJ kernel template covering k = 5 AND k = 6 from a single template (3 + 4 are slice 1 / slice 2). | xlog-cuda cert: 5-clique fixture matches CPU oracle AND 6-clique fixture matches CPU oracle; **k=6 cert MUST pass without adding any new `.cu` source for k=6** (template instantiation only). Counter increments on dispatch; binary-join fallback row-set parity at both k. |
| W3.3 | ROADMAP item #10 | OPEN | — | Histogram-guided block scheduling / heavy-row offload. | Bench: super-hub fixture's heavy-row case shows **≥ 2.0× speedup** vs. uniform block dispatch on the canonical fixture (`tests/...adaptive_dispatch::superhub_fixture`); deterministic output preserved (`download_triples` row-for-row equal to baseline run); no regression on uniform fixture (within ±5%). |
| W3.4 | ROADMAP item #11 | OPEN | — | Kernel fusion where benchmarks show materialization overhead dominates. Targeted candidates: layout + count, or count + materialize. | Bench: fused kernel shows **≥ 1.3× speedup** vs. 2-kernel sequence on a fixture where materialize is the long pole; deterministic. No regression on a small fixture where fusion penalty exceeds savings (must auto-disable below threshold). |
| W3.5 | ROADMAP item #12 | OPEN | — | Shared-memory optimization for small relations in WCOJ. Threshold below which the kernel reads sorted slot from `__shared__`. | Bench: sub-1K-row WCOJ slot benefits **≥ 1.5×** vs. global-memory path; output deterministic (row-for-row equal); no regression on relations above the threshold. |
| W3.6 | ROADMAP item #13 | OPEN | — | Warp-level primitives for small-relation optimization. Cooperative warp `__shfl_*` lookups. | Bench: warp-coop path beats global path **by ≥ 1.3×** below the W3.5 threshold; output deterministic; no regression above. |

### Wave 4 — Runtime closure

| ID | Source | Status | Blocked by | Required deliverable | Acceptance gate |
|----|--------|--------|------------|----------------------|-----------------|
| W4.1 | Internal + ROADMAP item #4 finishing | OPEN | — | Multi-recursive WCOJ (≥ 2 in-SCC body Scans). Slice 4 gated this out; close it with per-rule variant union + dedup safe under WCOJ output. | Cert: multi-recursive triangle + multi-recursive 4-cycle dispatch on each iteration's variant; row-set parity vs. binary-join reference. Promoter gate `count <= 1` is removed; replaced with `count <= rule.recursive_arity` or equivalent. |
| W4.2 | ROADMAP item #14 | OPEN | — | Nested-loop join operator for small relations. Adaptive selection: when both sides are below a threshold, nested-loop is cheaper than hash. | Cert: small × small fixture picks nested-loop; large × small picks hash; row-set agreement. |
| W4.3 | ROADMAP item #15 | OPEN | — | General sort-merge join operator for pre-sorted binary relations. Triangle-layout helper is a special case; this is the generic path. | Cert: pre-sorted binary join skips the sort step, matches reference output. |

### Wave 5 — Certification closure

| ID | Source | Status | Blocked by | Required deliverable | Acceptance gate |
|----|--------|--------|------------|----------------------|-----------------|
| W5.1 | ROADMAP item #17 | OPEN | — | GPU Same Generation cert (currently oracle-only); skewed multiway GPU cert; deep-recursive WCOJ cert. | Three new test files in `xlog-integration/tests/`; each asserts row-set parity vs. CPU oracle and dispatch counter > 0. |
| W5.2 | ROADMAP item #18 | OPEN | — | Skewed multi-way GPU benchmark suite beyond triangle. Covers 4-cycle, 5-clique (W3.2), and a pivot-heavy multi-way pattern. | Bench harness committed; evidence file with crossover thresholds vs. binary-join. |
| W5.3 | ROADMAP item #19 | OPEN | — | Single test harness for WCOJ + binary-join + recursive determinism. Today the parts exist separately. | One test file that runs all three execution modes on the same fixture and asserts deterministic equality across runs. |
| W5.4 | ROADMAP item #20 | OPEN | — | Downstream widened-frontier stress replay clean gate. Replay harness must be built — none exists today. | Replay harness committed; clean run logged in evidence; integration cert gates on replay determinism. |

### Wave 6 — Documentation closure

| ID | Source | Status | Blocked by | Required deliverable | Acceptance gate |
|----|--------|--------|------------|----------------------|-----------------|
| W6.1 | ROADMAP item #21 | OPEN | — | Dedicated WCOJ architecture guide. Covers RIR, promoter, dispatch, cost model, recursive integration. | New file `docs/wcoj-architecture-guide.md`; cross-linked from ROADMAP. |
| W6.2 | ROADMAP item #22 | OPEN | — | User-facing WCOJ eligibility / fallback / performance tuning guide. NOT just code-doc rehash. | New file `docs/wcoj-user-guide.md`; covers env vars, config builders, when to opt into cardinality, threshold tuning, troubleshooting. |

### Wave 7 — Release

| ID | Status | Blocked by | Required deliverable | Acceptance gate |
|----|--------|------------|----------------------|-----------------|
| W7.1 | OPEN | every other board item | Full workspace gate + CUDA cert + real_world replay; tag `v0.6.5` only after every other board item is DONE AND user explicitly authorizes the tag. | Board reaches 0 OPEN; user says "tag v0.6.5"; tag pushed. |

## Completed

(empty — populated as items move OPEN → DONE per process rule #1)

## Provenance

- ROADMAP.md §1203–1268: 22 v0.6.5 items, 4 already DONE in
  shipped slices 1/2/4, 18 open here as W2–W6 entries.
- Internal commitments tracked in slice plans/evidence:
  - Slice 4 plan §"Open Questions": multi-recursive deferral.
  - Slice 5 plan §"Out-of-Slice (Deferred)": `record_join_result`
    feedback; default flip.
  - Slice 4 evidence §"Risks / Out-of-Slice": stats integration
    perf concern.
- All three internal commitments now W2.3 / W2.4 / W2.5 / W4.1
  on this board.
