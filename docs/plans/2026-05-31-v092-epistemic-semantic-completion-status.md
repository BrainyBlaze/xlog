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
- **NON-DETERMINED / boundary cases**:
  - **Cyclic negated-modal recursion** (example 33) reduces to a non-monotone ordinary SCC.
    The sound semantics is WFS, and the final closure routes it through the
    `xlog-gpu` GPU-backed WFS plan without the old `xlog_prob` host-WFS solver.
    This is not a device-resident/no-host-interaction WFS residency claim; host
    orchestration and metadata convergence reads may still occur. Fresh
    focused/full gates passed under this exact contract.
  - **FAEEL-unfounded self-support** (`p() :- possible p()` with no independent founded
    support) executes to the FAEEL-defined empty founded extension (`rows: 0`); the G91
    companion accepts (`rows: 1`).
  - **C2 nested/interior negation** now executes for the finite chain cases. Example 13f proves
    `know not possible p()` dualizes to `not possible p()` for a determined present target; the
    13f companion cells cover target `{present,absent}` x mode `{FAEEL,G91}`; and examples 13g-13v cover all 64 two-operator negation cells over `{know,possible}^2`,
    leading/interior/atom-adjacent negation, and present/absent atoms under FAEEL; 13w*
    replays those cells under explicit G91.

> **ITEM C UPDATE (v0.9.2):** bare modal CHAINS (`know possible p()`, `know know p()`,
> `not know possible p()`) NO LONGER reject. They collapse via the KD45/S5 modal axioms to
> the operator ADJACENT to the atom (inner wins; leading `not` distributes) as a sound AST
> normalization, routing through the existing single-level epistemic path with no new
> evaluator. The collapse holds in BOTH modes (FAEEL/G91 differ only in admissible world
> views, which the collapsed literal inherits). Examples 13/13b/13c execute with exact
> tuples; 13d (FAEEL `rows:0`) vs 13e (G91 `rows:1`) shows the inherited mode-difference;
> 13f executes the interior-negation duality mini-matrix; 13g-13v exhaust the 64 two-operator
> negation cells under FAEEL, and 13w* replays them under explicit G91.

The load-bearing **over-broadening gates** remain important: because the determined-closure
acceptance check is permissive, the failure mode of loosening it is LEAKAGE (an undefined
program emitting a wrong-but-non-empty answer). Current boundary examples include unsafe
unbound negated modal variables and unbounded structured modal tuple keys; accepted examples
13/13f/13g-13v/13w*/22/31/33 prove defined semantics rather than rejection.

- The v0.9.1 goal-mandated fail-closed fragments (aggregate/compound/list/predref modal
  keys; variable-keyed/nested/CPU-scan epistemic constraints; unsafe same-name multi-arity
  coupling) remain rejection-by-design.

## Historical Acceptance Matrix (feat HEAD `eb857c35`, all `--test-threads=1`)

This matrix is a historical snapshot. It must be rerun on the current tree after the host-WFS
removal, C2 rename, and expanded example matrices before it can support a merge claim.

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
- Fail-closed boundaries in the current source are not this older list. Current notable
  boundaries include unsafe unbound negated modal variables and unbounded modal tuple keys
  such as 23b. Examples 13, 13f, 13g-13v, 13w*, 22, 31, 33, and 43 are defined semantics cases,
  not generic unsupported-construct evidence.

### Determined-modal-family completeness cells (examples cover every cell)

The completeness partition {know, possible, not know, not possible} × {filter, bind} ×
{determined-EDB, determined-ordinary-derived, determined-epistemic-derived} is exercised:
positive `know`/`possible` filter+bind over EDB (06/07/14/18/19/21) and over determined
derived heads (16/17/24/25/27/28/30); negated `not know`/`not possible` over EDB (19/21/26)
and over a determined derived head (29). NON-determined targets are now classified by their
real semantics rather than a generic rejection bucket: founded-empty FAEEL (31), G91
self-support (32), stratified ordinary negation (37), GPU-backed WFS (33), or typed
unfounded/unsafe boundaries (13/22/23).

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

### Closure items — classified by their REAL nature (no relabeling)

1. **Recursion negation-cycle (ex33) → CLOSED under the exact GPU-backed WFS contract.**
   The high-level GPU compiler now detects the non-monotone reduced SCC and routes it through
   a GPU-backed alternating-fixpoint WFS plan. `wfs.rs` remains host-only (`HashSet`/`HashMap`,
   CPU prob-provenance path) and is still not an accepted production fallback; `xlog-gpu` no
   longer declares an `xlog-prob` dependency. This is not a
   device-resident/no-host-interaction WFS residency claim: host orchestration remains, and
   convergence may use metadata row-count reads. The focused example surface now includes the
   canonical ex33 fixture, the `33a*` operator/mode matrix covering `{FAEEL,G91} ×
   `{not know,not possible}`, the `33c*` seed-state matrix adding seed `{present,absent}`,
   `33b` for WFS plus ordinary EDB negation, the `33d*` matrix covering the same
   mode/operator/seed product with ordinary EDB negation inside the reduced SCC, and `33e*`
   load-bearing cells where `not banned(2)` flips the seed-founded reach tuple on/off.
   The `xlog-gpu` logic-runner now also has CUDA-independent guards for the manifest/source
   boundary (`xlog-gpu` must not depend on `xlog-prob`, and `src/logic.rs` must not reintroduce
   the old host-WFS solver tokens) and the WFS compile/plan-kind matrix
   (`wfs_gpu_recursive` / `epistemic_wfs_gpu`) over mode, negated-modal operator, seed-state, and
   ordinary-EDB-negation axes; a zero-iteration-bound clamp in WFS plan JSON; deterministic single-,
   multi-predicate, and ordinary-EDB-negation `wfs_fixed_relations` maps in WFS plan JSON; explicit
   `wfs_convergence_predicates` and `wfs_gpu_passes` in WFS plan JSON; explicit
   `host_wfs_fallback_allowed:false`; the same full-axis GPU runtime matrix; and a fixed-relation
   name-collision regression for user-owned `__wfs_*` predicates. Fresh focused and full gates
   have passed under this exact no-old-host-WFS-solver contract; use
   `docs/plans/2026-06-02-v092-epistemic-closure-validation-checklist.md` as the gate ledger.

2. **E1 shared-variable join / diagonal / negated-difference (`:- know p(X), possible q(X)`,
   `:- know p(X,X)`, `:- q(X), not know p(X)`) → DONE (commits `f449bc43`, `bee3a0de`,
   `268dd590`).** Resolved by a sound PROGRAM-level desugaring at the `normalize_program`
   pre-pass: `:- L1, …, Ln.` ⟶ `__epi_join_N(Vars) :- ord(L1), …, ord(Ln).` + `:- know
   __epi_join_N(Vars).`, where `ord` ordinary-izes each modal literal (`know/possible r → r`,
   `not know/possible r → not r`). For a base/EDB or purely-ordinary-derived target
   `know r == possible r == r`, so the ordinary join is exactly the forbidden binding set; the
   single-occurrence `:- know __epi_join_N(Vars)` routes through the EXISTING variable-keyed
   world-view constraint path (prune-to-empty) — **no new kernel**. The earlier "runtime bug"
   diagnosis was an artifact of the WRONG desugar shape (helper rule + an *ordinary* constraint,
   or a nullary helper) which routed through the violated-constraint **error** path / a
   nullary schema-union; ordinary-izing into the helper and keeping a *single-occurrence modal*
   constraint avoids both. Guarded to non-modal-derived targets (where the equivalence holds);
   a modal-derived target falls through to the core compiler's existing shared-variable
   rejection (still covered by a device test feeding the un-normalized program directly).
   Examples 38/39 (diagonal), 40 (join, `p∩q`), 41 (negated-difference, `q\p`); CLI regression
   asserts the load-bearing prune. Gates: device 144, integration 206, split 44,
   run_cli_tests 12.

3. **Same-name multi-arity coupling via `xlog run` → DONE and freshly gated.**
   The production path now includes the base disambiguation example plus exhaustive
   `42a*` and `42b*` matrix fixtures covering single-literal arity cells and every cross-arity
   conjunction. Do not list this as an undone semantic boundary.

Honest boundary: current source has closed C2 interior-negation, same-name multi-arity, FAEEL
empty-extension self-support, G91 possible recursion, stratified negated recursion, GPU-backed
WFS for ex33/33a/33c/33b/33d/33e cyclic negated-modal recursion, and the shared-variable constraint joins.
The WFS claim is limited to the no-old-host-WFS-solver contract above, not a
device-resident/no-host-interaction residency contract. Fresh focused/full gates
have passed for this release surface; the current checklist is
`docs/plans/2026-06-02-v092-epistemic-closure-validation-checklist.md`.
