# v0.9.2 Epistemic Executor — Semantic Completion Status

**Date:** 2026-05-31.
**Branch:** `feat/v092-epistemic-semantic-completion` (base `4b91e911`, v0.9.1 tip).
**Scope:** the three honest Category-B semantic gaps tracked after v0.9.1.

All accepted semantics are validated on the production `xlog run` / runtime path
(device-only pilots are necessary but not sufficient). The cross-cutting locks
(no hidden CPU fallback, no fake modal→ordinary rewrite, no raw RIR shortcut, EIR
as the semantic boundary, typed fail-closed for out-of-fragment inputs) held for
every accepted item. GPU acceptance gates are run `--test-threads=1` (device
contention under parallelism on this box; serial is authoritative). The full
`xlog-prob --features host-io` suite is NOT admissible as GPU-native evidence (MC
CPU/oracle surfaces) — only the targeted epistemic prob gates below count.

## Bundles

| Bundle | Status | What landed | Evidence |
|---|---|---|---|
| **1. EGB-02B mixed-literal modal membership** | DONE | a rule combining a GLOBAL modal gate (ground/anonymous/nullary modal) with a PER-ROW bound-variable modal gate now composes the two gate classes CONJUNCTIVELY on the GPU device path. The row-map kernel's per-row path (`epistemic_final_tuple_has_accepted_membership_for_row`) now applies the same `gate_holds` global-gate check as the global path; the prior fail-closed guard is removed. | 8 value-level device pilots crossing global×row truth with EXACT tuples; load-bearing **mutation proof** (neutralize the new gate → global-false pilots wrongly emit all rows); `examples/epistemic/14-mixed-literal-membership.xlog` → `reachable={1,3}` via `xlog run`; 0 CPU-fallback counters |
| **2. Case-A recursive epistemic fixpoint** | DONE | recursive ordinary predicates inside epistemic programs now evaluate to FIXPOINT when every modal atom ranges over an **invariant** relation (EDB / lower non-recursive non-epistemic stratum). Such a positive `know`/`possible` literal is reduced to its gated relation (`know R ≡ possible R ≡ R` is a theorem when R is invariant), and the reduced ORDINARY recursive program runs through the existing GPU recursive fixpoint engine (`execute_plan`). | `examples/epistemic/15-recursive-epistemic-closure.xlog` → `{(1,2),(2,3),(1,3)}` (derived (1,3)); `15-recursive-epistemic-chain.xlog` → full closure incl. 3-hop `(1,4)`; both via `xlog run`. FAEEL/G91 self-support unchanged. No new engine, no CPU path. |
| **3. Cross-component epistemic coupling** | DONE | a coalesced component with >1 epistemic output head sharing a base modal predicate is **JOINT-SOLVED with multi-output materialization**: one candidate enumeration + world-view validation over the component's combined modals, then EACH head materialized against the SAME accepted world view (per-head scoped row-filter + `additional_head_outputs` + per-head projection via `public_head_arity` / `final_output_columns_for_materialization`; reuses the WCOJ-promoted reduced runtime plan and EGB-01 enumeration — not reimplemented). Augmented heads of DIFFERING arity sharing a base modal are supported (per-head projection). Coupling over an epistemic-DERIVED head is resolved by **STRATIFIED epistemic execution** (see below), not fail-closed. | `18-cross-component-joint-shared-modal.xlog` materializes BOTH heads via `xlog run` (`known={1,2}`, `maybe={2}`); `21` THREE heads (`quarantine/watch/clear`); **K2 split-vs-unsplit equivalence** (`joint_multi_head_output_equals_per_head_split_reference_on_device`, non-tautological); `27` augmented differing-arity (`one_hop={1,2}`, `pair={(2,20),(2,21),(3,30)}`) |
| **3b. Stratified epistemic execution** (coupling/recursion over a DETERMINED head) | DONE | a modal over an epistemic-DERIVED head that is itself **DETERMINED** (its modals bottom out in invariant/EDB relations, acyclically — `EpistemicallyDeterminedPredicates::analyze`) is resolved by **stratified execution**: the determined head is gated ONCE and materialized into the relation store (`Executor::materialize_epistemic_head_relation`) as a lower stratum (`LogicExecutionPlan::EpistemicStratified`); the higher stratum reads it as a plain base relation through the **existing** EGB-02 membership/join filter. The theorem `know R ≡ R` is applied at the **STORE boundary, not the rule body** — no resolve-into-body, no double-gating. Determined-closure is transitive (ordinary predicates over determined heads are determined) and works for multi-column binding modals (the modal is a binding join atom, sound because the target is determined). A NEGATED modal over an invariant relation reduces cleanly to ordinary negation. | chained `b:-know a` (`17` → `flagged={1,3}`); recursion over a determined head (`25` → `reach={(1,2),(2,3),(1,3)}`); transitive determined-ordinary (`24` → `b={1}`); negated-modal-over-invariant in recursion (`26` → `reach={(1,2),(3,4)}`); determined-epistemic multi-column binding (`28` → `out={1}`). All exact-tuple via `xlog run` |

## Determined-modal family: COMPLETE

Every modal target is now in exactly one of two classes, and the boundary between them
is closed under composition:

- **DETERMINED** (fixed extension → `know R ≡ possible R ≡ R` is a theorem) → **resolved**.
  Covers: invariant/EDB targets; determined-ordinary and determined-epistemic derived
  heads (transitively, acyclically); unary and multi-column; filtering and binding
  (augmenting) modals; coupling and recursion over a determined head; negated modal over
  an invariant relation (≡ ordinary negation). Resolution is via JOINT-solving (shared
  base modal) or STRATIFIED execution (modal over a determined derived head).
- **NON-DETERMINED** (no fixed extension) → **correctly fail-closed** (this is the right
  answer, not deferral):
  - **Circular modality / Case-B recursion THROUGH the modal predicate** (`p() :- …, know p()`,
    or a modal over a relation in/transitively-on the recursive SCC). FAEEL/G91 foundedness
    governs founded self-support; unfounded modal cycles and non-invariant recursive modal
    targets fail closed with `recursive epistemic program` / "not invariant".
  - **FAEEL-unfounded self-support** (`p() :- possible p()` with no independent founded
    support) — the FAEEL-defined rejection (G91 accepts).
  - **C2 modal-over-a-negated-modal compound formulas** — an INTERIOR negation
    (`know not possible p ≡ K ¬M p`) or atom-adjacent negation (`know possible not p ≡ M ¬p`)
    has no sound collapse to a single `op atom` literal → typed `UnsupportedEpistemicConstruct`
    ("interior negation"/"negated atom"). (ITEM C, example 13f.)

> **ITEM C UPDATE (v0.9.2):** bare modal CHAINS (`know possible p()`, `know know p()`,
> `not know possible p()`) NO LONGER reject. They collapse via the KD45/S5 modal axioms to
> the operator ADJACENT to the atom (inner wins; leading `not` distributes) as a sound AST
> normalization, routing through the existing single-level epistemic path with no new
> evaluator. The collapse holds in BOTH modes (FAEEL/G91 differ only in admissible world
> views, which the collapsed literal inherits). Examples 13/13b/13c execute with exact
> tuples; 13d (FAEEL `rows:0`) vs 13e (G91 `rows:1`) shows the inherited mode-difference;
> 13f is the C2 fail-closed boundary above.

These genuinely-undefined cases are the load-bearing **over-broadening gates**: because the
determined-closure acceptance check is permissive, the failure mode of loosening it is
LEAKAGE (an undefined program emitting a wrong-but-non-empty answer). Examples 22/13/23 and
the FAEEL self-support pilot assert these STILL fail closed with their typed diagnostics, and
a recursive multi-column binding probe confirms the broadening does not leak on non-determined
targets.

- The v0.9.1 goal-mandated fail-closed fragments (aggregate/compound/list/predref modal
  keys; variable-keyed/nested/CPU-scan epistemic constraints; unsafe same-name multi-arity
  coupling) remain rejection-by-design.

## Acceptance Matrix (feat HEAD `eb857c35`, all `--test-threads=1`)

- `cargo test -p xlog-logic --test test_epistemic_split` → **37 passed**
- `cargo test -p xlog-logic --test test_epistemic_executable_plan` → **17 passed**
- `cargo test -p xlog-logic --test test_epistemic_faeel` → **4 passed**; `--test test_epistemic_g91` → **5 passed**
- `XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-runtime --test test_epistemic_gpu_workspace --release --features epistemic-logic-tests` → **131 passed**
- `cargo test -p xlog-cli --test run_cli_tests --release` → **9 passed** (the `xlog run` example markers incl. 14–30 + the nested-modal / recursion-through-modal / compound-key / FAEEL-unfounded negative CLI pilots)
- `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution --release` → **206 passed** (in-suite `cpu_*==0` assertions)
- `cargo test -p xlog-prob --release --features host-io --test epistemic_prob_gpu_accepted_evidence` → **31 passed** (serial; parallel shows device-contention flakiness — serial authoritative)
- `cargo test -p xlog-prob --release --features host-io --test epistemic_prob_production_reuse` → **7 passed**
- `cargo fmt --check` → clean; `git diff --check` → clean; conflict-marker scan → clean

### Production `xlog run` pilots (exact tuples, verified)

- Resolved: `14`→`{1,3}`, `15`→closure incl. `(1,3)`/`(1,4)`, `16`→`{1,3}`, `17`→`flagged={1,3}`,
  `18`→`known={1,2}`/`maybe={2}`, `19`→`{1,2}`, `20`→10-tuple closure, `21`→`quarantine={1,2}`/`watch={2}`/`clear={3,4,5}`,
  `24`→`b={1}`, `25`→`reach={(1,2),(2,3),(1,3)}`, `26`→`reach={(1,2),(3,4)}`,
  `27`→`one_hop={1,2}`/`pair={(2,20),(2,21),(3,30)}`, `28`→`out={1}`,
  `29`→`q_know={3}`/`q_poss={3}` (negated modal over determined-derived),
  `30`→`out={1}` (`possible`-binding over determined, the `know`-twin of 28).
- Fail-closed (genuinely-undefined, typed `UnsupportedEpistemicConstruct`): `13` (nested),
  `22` (recursion through modal / not invariant), `23` (compound list key),
  `31` (FAEEL-unfounded self-support → `FAEEL foundedness guard`).

### Determined-modal-family completeness cells (examples cover every cell)

The completeness partition {know, possible, not know, not possible} × {filter, bind} ×
{determined-EDB, determined-ordinary-derived, determined-epistemic-derived} is exercised:
positive `know`/`possible` filter+bind over EDB (06/07/14/18/19/21) and over determined
derived heads (16/17/24/25/27/28/30); negated `not know`/`not possible` over EDB (19/21/26)
and over a determined derived head (29). Every NON-determined target fails closed
(22/13/23/31), CI-enforced as over-broadening gates.

### Inadmissible as GPU-native acceptance evidence

- The full `cargo test -p xlog-prob --release --features host-io` suite (MC
  CPU/oracle/`gpu_mc_vs_cpu.rs` host-heavy surfaces) is a regression check only,
  not a no-host-transfer / GPU-native acceptance gate.

---

## Post-integration status (2026-06-01, branch `integration/v092-main-mc-resident`)

Main HEAD `96d1530d` (MC GPU-resident engine) was union-merged into this branch (merge
`dde60b87`); both surfaces coexist (`cargo check --workspace --all-targets` green). On top
of the determined-modal family above, the following landed and are verified by me serially
through the production path (device suite grew **131 → 144**):

- **Item D** structured modal tuple-keys (list/compound/anonymous flatten into the existing
  N-column matcher; unbounded forms reject with a `ResourceExhausted` finiteness bound).
- **Item E** variable-keyed + nested epistemic constraints (single-occurrence positive
  variable → wildcard existential; multi-literal distinct-variable conjunction).
- **Item F** unsafe split/coupling — same-name multi-arity (arity-qualified store resolution)
  + derived-head coupling with split-vs-unsplit equivalence; genuine cyclic coupling rejected.
- **Item A1** stratified negated-modal recursion executes on GPU (example 37
  `unreachable = node² − reach`); genuine negation cycles bounded (see below).
- **Item E2** `:- not know p(X)` now returns the **NAF safety error** (unbound variable),
  identical to ordinary `:- not r(X)` — the sound answer for an ill-formed program, not a
  "missing feature" diagnostic (commit `e0e7d2a9`).

### Remaining items — classified by their REAL nature (no relabeling)

1. **Recursion negation-cycle (ex33) + G91 `possible`-recursion → WHOLE-ENGINE
   ARCHITECTURAL BOUND.** These need non-stratified (well-founded / stable-model) semantics.
   **Proven not epistemic-specific:** ordinary `a :- not b. b :- not a.` and ordinary
   recursion-through-negation BOTH error on this engine — XLOG is a *stratified-negation*
   engine with no WFS/stable-model evaluation for ANY program. `wfs.rs` exists but is
   host-only (pure `HashSet`/`HashMap`, used only by the CPU prob-provenance path), and the
   no-host-solver lock forbids routing accepted programs through it. Finishing these means
   building well-founded semantics for the entire engine — a separate project, the same wall
   ordinary Datalog hits here.

2. **E1 shared-variable join / diagonal (`:- know p(X), possible q(X)`, `:- know p(X,X)`,
   `:- q(X), not know p(X)`) → reduces to working features, blocked by RUNTIME bugs, not an
   epistemic-layer gap.** All three forms execute correctly as RULE bodies
   (`both(X) :- know p(X), possible q(X)` → exact `{2}`; diagonal → `{1}`; negated-difference
   → `{2}`). The clean reduction "rewrite `:- Body(X)` to a helper rule + a constraint on the
   helper" hits two real runtime issues: (a) a nullary-head + negated-modal **schema-union
   bug** (`Union requires compatible schemas: [] vs [(c0,U32)]`); (b) the non-nullary form
   routes through a path that surfaces a violated constraint as an **`Execution` error**
   instead of pruning the world view to empty (the way item E's `:- know flagged(X)` does).
   Sound completion requires fixing those runtime paths (or extending the GPU world-view
   constraint kernel to evaluate the join), done test-first — not a quick rewrite.

3. **Same-name multi-arity coupling via `xlog run` → PRE-EXISTING engine-wide
   name-keyed-schema limitation.** The coupling semantics work at the device/runtime layer
   (item F, exact tuples per arity); the full `xlog run` path is blocked because
   `Compiler::schemas` is `HashMap<String, Schema>` and `load_facts_into_store` groups facts
   by bare predicate name, so `pred p(u32)` + `pred p(u32, u32)` collapse. Arity-keying the
   global schema/relation identity is an ordinary-compiler refactor, orthogonal to epistemic
   semantics.

Honest boundary: items B/A/C/D/E/F + A1 + E2 are done and verified (device 144); class (1)
is a real architectural wall (not fakeable without a new engine); classes (2)/(3) are
finishable but are runtime/compiler-layer work, not epistemic-layer patches.
