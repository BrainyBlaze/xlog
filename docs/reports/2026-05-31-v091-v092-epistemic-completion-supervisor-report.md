# Supervisor Report — v0.9.1 + v0.9.2 Epistemic Executor Completion

**Date:** 2026-05-31
**Author:** implementation session (direct development + isolated-worktree subagents)
**Scope:** the v0.9.1 epistemic executor completion (7 EGB bundles) and the v0.9.2
semantic completion (the determined-modal family).
**Branch:** `feat/v092-epistemic-semantic-completion` @ `3c33ac81` (base `4b91e911`, the
v0.9.1 tip; 30 commits).
**Release status:** NOT pushed, NOT tagged — the release tag is user-gated.
**Superseded by:** `docs/reports/2026-06-02-v092-epistemic-final-closure-record.md`
for the final v0.9.2 package/release status and the exact GPU-backed WFS contract.

---

## Executive summary

XLOG's epistemic surface went from "a bounded executor that fails closed on most
non-trivial modal programs" (pre-v0.9.1) to **a load-bearing surface where every modal
program is either soundly executed or fails closed for a genuinely-undefined reason —
and nothing in between.**

- **v0.9.1** turned the bounded executor into a real surface: 7 EGB bundles (candidate
  enumeration, value-level membership, FAEEL foundedness, ground constraints, safe split,
  joint multi-epistemic, nested-modal rejection), all driven through the production
  `xlog run`/device path, with three production-path bugs found and fixed that device
  pilots structurally could not catch.
- **v0.9.2** closed the semantic gaps: mixed-literal membership, Case-A recursive
  fixpoints, multi-output joint-solving, and — the headline — **stratified epistemic
  execution**, which resolves coupling/recursion over epistemic-derived heads. The result
  is a provable partition: **every modal target is DETERMINED (resolved) or NON-DETERMINED
  (correctly fail-closed).** The determined side is closed under composition.

**Final acceptance (verified serially, `--test-threads=1`, on `3c33ac81`):** device
`test_epistemic_gpu_workspace` **131**, integration `test_epistemic_gpu_wcoj_execution`
**206**, `test_epistemic_split` **37**, `test_epistemic_executable_plan` **17**, FAEEL **4**,
G91 **5**, CLI `run_cli_tests` **9**, prob accepted-evidence **31**, prob production-reuse
**7**; `cargo fmt --check` clean; workspace builds. Epistemic showcase examples: **01–31**.

---

## Part I — v0.9.1 Epistemic Executor Completion

**Goal:** turn the bounded epistemic executor into a load-bearing surface per the 7-bundle
plan (`docs/plans/2026-05-28-v090-epistemic-executor-goal-bundles.md`).

**Locks (every bundle):** no hidden CPU fallback in candidate enumeration / world-view
validation / tuple membership / result materialization; no fake predicate rewriting
(`know`/`possible` stay first-class through EIR — the semantic boundary); direct raw RIR
lowering stays a rejection boundary; no parallel side engines; typed fail-closed for
out-of-fragment inputs; semantic oracle/fixture tests are NOT release evidence (real
runtime/device pilots required).

### Bundles delivered (all 7 merged + verified)

| Bundle | What landed |
|---|---|
| **EGB-02** value-level membership | Fixed a REAL soundness bug: the global membership gate was a near-no-op letting all rows through for ground/anon/nullary modal literals; now requires the accepted candidate's assumption bit. |
| **EGB-01** candidate enumeration | Runtime already enumerated the 2^N candidate lattice on-device from EIR; the "limitation" was the CPU fixture layer. Added device pilots incl. empty-world-view ≠ failure (mutation-tested for non-vacuity). |
| **EGB-07** FAEEL foundedness | Per-tuple-key founded self-support; hardened diagnostics + 9 production-path pilots. |
| **EGB-04** ground constraints | NEW GPU kernel `epistemic_validate_constraints_u8`; ground epistemic constraints prune world views; constraints dropped from the reduced ordinary program (no RIR rewrite). **K2** (constraint-specific rejection index) closed separately (`e39bcd33`). |
| **EGB-05** safe split | Merge-reason enum + paired split-vs-unsplit equivalence pilot. |
| **EGB-06** joint multi-epistemic | Distinct-predicate multi-epistemic rules solved jointly (reuses EGB-01 enum + EGB-02 membership + world-view kernel). |
| **EGB-03** nested-modal | Typed fail-closed for `know possible p()`; stabilized diagnostics for `not`-interspersed nested forms. |

### Three production-path bugs found by user scrutiny (the dominant lesson)

Each was characterized as "by design / covered by device pilots" and was wrong — device
pilots feed relations+plans directly and **structurally cannot** catch CLI-path routing,
fact materialization, or fixpoint gaps:

1. **Nullary EDB facts materialized 0 rows** (`pred().` → empty buffer → asserted-absent
   everywhere). Fix: `create_zero_arity_buffer` (1 unit-tuple row). EGB-02's correct
   global-gate fix had *exposed* this latent bug.
2. **Recursive epistemic programs silently wrong** (epistemic base + ordinary recursive
   step returned only the base case). Fix: `reject_recursive_epistemic_program` typed
   fail-closed guard at the planning chokepoint (later *resolved*, not just guarded, in
   v0.9.2 Case-A + stratified).
3. **Split-routing error** for bound-variable multi-head programs (`xlog run` errored with
   no recourse). Fix: skip empty-body rules in the dependency graph; route >1-distinct-head
   programs straight to split.

**Rule adopted:** run every behavioral claim through the actual `xlog run` production path
before asserting it; "covered by device pilots" is not evidence about the pipeline.

---

## Part II — v0.9.2 Semantic Completion (the determined-modal family)

The v0.9.2 charter was three honest Category-B gaps: (1) mixed-literal modal membership,
(2) Case-A recursive epistemic fixpoint, (3) cross-component coupling — "solve jointly OR
fail closed with a precise merge reason." The user subsequently rejected every fail-closed
escape for cases that were *merely hard* (not genuinely undefined) and mandated: **resolve
ALL, no gaming, no deferring, no false-closed.** This section is the honest arc.

### What was delivered

1. **EGB-02B mixed-literal membership** — a rule combining a GLOBAL modal gate
   (ground/anon/nullary) with a PER-ROW bound-variable modal gate now composes both gate
   classes CONJUNCTIVELY on the device path. Validated with a load-bearing mutation proof
   (neutralize the new gate → global-false pilots wrongly emit all rows). (Example 14.)

2. **Case-A recursive epistemic fixpoint** — recursive ordinary predicates whose modal
   atoms range over an INVARIANT relation evaluate to fixpoint: `know R ≡ possible R ≡ R`
   is a theorem when R is invariant, so the modal reduces to its gated relation and the
   reduced ordinary recursion runs through the existing GPU fixpoint engine. (Examples 15,
   20, 26.)

3. **Multi-output joint-solving** — a coalesced component with >1 epistemic head sharing a
   base modal predicate is solved with ONE candidate enumeration + world-view validation
   over the combined modals, then EACH head materialized against the SAME accepted world
   view (per-head scoped row-filter + per-head projection via `public_head_arity`).
   Supports heads of DIFFERING arity. (Examples 18, 21, 27.)

4. **Stratified epistemic execution (the headline)** — a modal over an epistemic-DERIVED
   head that is itself DETERMINED is resolved by stratification. (Examples 17, 24, 25, 28,
   29, 30.)

### The architectural decision that mattered (and the failed approach)

The naive way to support "modal over a determined derived head" is to RESOLVE the modal
(`know R` → store relation `R`) **into the reduced rule body**. This was attempted on
`wip/v092-stratified-partial` and **proven wrong by bisection: 29 regressions across 3
suites.** Two mechanisms: (a) WCOJ-route loss (the resolved join plan exposed 0
MultiWayJoin routes); (b) **double-gating** — the resolved store is pre-gated, then the
retained GPU world-view filter re-gates the same rows, dropping sound tuples (example 18
regressed to `known={1}` instead of `{1,2}` — silently wrong, and it passed the old
positive tests).

The **correct** design separates the mechanisms:
- A *distinct* `EpistemicallyDeterminedPredicates::analyze` (NOT a loosening of the
  invariant check that feeds resolve-into-body) identifies determined heads.
- A *new* `LogicExecutionPlan::EpistemicStratified` materializes the gated lower-stratum
  head into the relation store ONCE (`Executor::materialize_epistemic_head_relation`, a
  device-side clone), and the higher stratum reads it as a plain base relation through the
  *single* existing EGB-02 filter.
- **The `know R ≡ R` theorem is applied at the STORE boundary, not the rule-body
  boundary** — that is precisely what avoids double-gating.

Determined-closure is **transitive** (ordinary predicates over determined heads are
determined), works for **multi-column binding** modals (the modal is a binding join atom,
sound because the target is determined), and the final fix confined the broadened
acceptance to the *schema/type-checking* step only (a separate
`reduce_..._for_stratified_schema`; the runtime reduce is byte-for-byte unchanged), so the
permissive check **structurally cannot leak into answers.**

### The completeness result

Every modal target is in exactly one class, and the boundary is closed under composition:

- **DETERMINED** (fixed extension → `know R ≡ R`) → **resolved**: invariant/EDB,
  determined-ordinary-derived, determined-epistemic-derived; unary and multi-column;
  filtering and binding; `know`/`possible`/`not know`/`not possible`; coupling and
  recursion.
- **NON-DETERMINED / recursive-boundary cases** are resolved by their defined
  semantics when one exists: positive Case-B founded recursion executes to the
  founded least fixpoint, FAEEL-unfounded self-support executes to the defined
  empty founded extension, explicit G91 self-support accepts, finite nested modal
  chains normalize by parity/duality, and cyclic negated-modal recursion routes
  through the `xlog-gpu` GPU-backed WFS plan under the no-old-host-WFS-solver
  contract. Only forms with no founded, G91, WFS, finite-key, or safe-variable
  interpretation remain fail-closed; those are CI-enforced over-broadening gates
  so undefined programs cannot leak wrong non-empty answers.

---

## Acceptance matrix (feat `3c33ac81`, all `--test-threads=1`, verified serially)

| Suite | Result |
|---|---|
| Device `test_epistemic_gpu_workspace` (`epistemic-logic-tests`) | **131** |
| Integration `test_epistemic_gpu_wcoj_execution` (in-suite `cpu_*==0`) | **206** |
| `test_epistemic_split` | **37** |
| `test_epistemic_executable_plan` | **17** |
| `test_epistemic_faeel` / `test_epistemic_g91` | **4 / 5** |
| CLI `run_cli_tests` | **9** |
| prob `epistemic_prob_gpu_accepted_evidence` / `..._production_reuse` | **31 / 7** |
| `cargo fmt --check` / `git diff --check` / workspace build | clean / clean / ok |

**Note on the prob gates:** the *targeted* epistemic prob gates above ARE admissible
GPU-native acceptance evidence. The *full* `xlog-prob --features host-io` suite (MC
CPU/oracle surfaces) is a regression check only and is NOT admissible as no-host-transfer
evidence.

---

## Showcase coverage (examples/epistemic/, 01–31)

Every accepted semantic and every fail-closed boundary has a runnable `xlog run` pilot with
an EXACT-tuple assertion (positive) or a typed-diagnostic assertion (negative). The
completeness partition {know, possible, not know, not possible} × {filter, bind} ×
{determined-EDB, determined-ordinary, determined-epistemic} is fully exercised.

Representative exact-tuple pilots: `14`→`{1,3}`, `18`→`known={1,2}`/`maybe={2}`,
`21`→three heads, `25`→`reach={(1,2),(2,3),(1,3)}`, `27`→`one_hop={1,2}`/`pair={...}`,
`28`→`out={1}`, `29`→`q_know=q_poss={3}`, `30`→`out={1}`. Fail-closed pilots: `13`, `22`,
`23`, `31` (each asserts the specific typed `UnsupportedEpistemicConstruct` construct +
context). Full table: `examples/epistemic/README.md`.

---

## Honest caveats (declared, not hidden)

1. **No-CPU-fallback lock on the stratified path is COMPOSITIONAL, not end-to-end.** The
   `cpu_fallbacks.is_zero()` / `cpu_candidate_enumerations==0` counters are asserted on the
   higher-stratum mechanism in isolation and on the certified single/joint paths; the
   `materialize_epistemic_head_relation` round-trip is a device-side clone (no host
   transfer); the full stratified programs are verified by exact tuples via `xlog run`, not
   by one whole-path counter assertion. The lock holds by construction + composition, not
   by a single end-to-end counter test. *Recommended follow-up: add one full-stratified-
   program counter assertion to make the lock end-to-end-explicit.*

2. **Genuinely-undefined cases remain fail-closed by design** — this is correct semantics,
   not incompleteness. Any future request to "accept" circular modality / unfounded
   self-support would require a *defined* semantics for them (e.g. G91 already accepts the
   circular `possible` form), not a patch.

3. **v0.9.1-era goal-mandated rejections stand** (aggregate/compound/list/predref modal
   keys; variable-keyed/nested/CPU-scan constraints; unsafe same-name multi-arity coupling)
   — rejection-by-design, verified by negative pilots.

---

## Process lessons (for the supervisor's record)

1. **Device pilots ≠ production-path evidence.** Three v0.9.1 bugs and one v0.9.2
   verification miss all traced to trusting device pilots / split-classification over the
   actual `xlog run` + `epistemic-logic-tests` device suite. The device suite is now run by
   the supervising loop *itself* before any "verified" claim — I once declared the broken
   resolve-core "verified correct" without running it; bisection then proved it regressed
   29 tests. Owned to the user.
2. **Over-broadening is the failure mode when loosening a classifier.** Positive tests
   cannot catch leakage (an undefined program emitting a wrong-but-non-empty answer); the
   negative pilots are load-bearing. Every permissive change shipped with its
   over-broadening gates green.
3. **Gaming is forbidden as hard as fail-closing.** A wrong-but-non-empty answer to clear a
   gate is worse than an honest fail-closed. Exact-tuple assertions throughout; honest-exit
   authorized (none was needed — every case resolved soundly).
4. **Subagent orchestration under rate-limiting:** isolated-worktree subagents with
   GQM-shaped goals, incremental commits (survive mid-run throttle kills), sequential merge
   + independent re-verification on the combined tree. Five agents were killed by
   server-side throttling; the survivors committed before death and were merged from clean
   checkpoints.
5. **Ops:** never do `git checkout`/`stash` dances in a worktree with a populated stash
   stack — a no-op stash + `stash pop` applied an unrelated stash; recovered by re-stashing
   (preserved, not discarded).

---

## Recommendation

v0.9.2 is functionally and semantically complete on `feat/v092-epistemic-semantic-completion`
@ `3c33ac81`, with a clean release surface and a full acceptance matrix verified serially.
The determined-modal family is closed; only genuinely-undefined modal programs fail closed.

**Suggested gating before tag:** (a) optionally add the one end-to-end stratified counter
assertion (caveat #1); (b) supervisor review of the architecture (store-boundary vs
rule-body) and the over-broadening gate set; (c) push + tag `v0.9.2` at the supervisor's
discretion. The `wip/v092-*` branches are retained as bisection evidence for the failed
resolve-into-body approach and may be deleted post-review.
