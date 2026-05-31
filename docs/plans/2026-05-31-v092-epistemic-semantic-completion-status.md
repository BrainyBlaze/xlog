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
| **3. Cross-component epistemic coupling** | DONE (scope below) | a coalesced dependency component carrying >1 epistemic output head (cross-component modal coupling) previously failed with a MISLEADING "independent outputs, use split" message; it now fails closed with a PRECISE typed `cross-component epistemic coupling` diagnostic naming the coupled heads and the `EpistemicComponentMergeReason`. Single-epistemic-output-head coupling (an ordinary rule consuming an epistemic head) is accepted and split-vs-unsplit equivalent. Shared extensional inputs do not force false coalescing. | `examples/epistemic/16-cross-component-coupling.xlog` → `trusted={1,3}`; `17-…-rejected.xlog` → typed diagnostic naming `["flagged","trusted"]` + `DerivedPredicate(trusted)`; device split-vs-unsplit equivalence pilot; recomposition `[0,1]` exactly once |

## Remaining typed fail-closed (NOT done — honestly tracked)

- **Multi-epistemic-output-head cross-component joint solving.** When ≥2 epistemic
  output heads are coalesced into one component, there is no single-output unsplit
  reference to recompose toward (the monolithic path also rejects multi-output), so
  joint materialization is impossible without a multi-output GPU kernel — a separate
  bundle. This fails closed with the precise diagnostic above; it is NOT joint-solved.
- **Case-B recursion** (recursion THROUGH the modal predicate, e.g. `p() :- …, know p()`)
  remains governed by FAEEL/G91 foundedness; unfounded modal cycles fail closed.
- **Negated modal literals in a recursive program**, and **modals over non-invariant
  (derived/recursive/epistemic) relations** in a recursive program, fail closed.
- The v0.9.1 goal-mandated fail-closed fragments (nested modal semantics; aggregate/
  compound/list/predref modal keys; variable-keyed/nested/CPU-scan epistemic
  constraints; unsafe same-name multi-arity coupling) remain rejection-by-design.

## Acceptance Matrix (feat HEAD `d37dd512`, all `--test-threads=1`)

- `cargo test -p xlog-logic --test test_epistemic_split` → **30 passed**
- `cargo test -p xlog-logic --test test_epistemic_executable_plan` → **17 passed**
- `XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-runtime --test test_epistemic_gpu_workspace --release --features epistemic-logic-tests` → **126 passed**
- `cargo test -p xlog-cli --test run_cli_tests --release` → **4 passed** (16 `xlog run` example markers + the nested-modal + cross-component negative CLI pilots)
- `cargo test -p xlog-integration --release` → all binaries pass; `test_epistemic_gpu_wcoj_execution` → **206 passed**
- `cargo test -p xlog-prob --release --features host-io --test epistemic_prob_gpu_accepted_evidence` → **31 passed** (serial; parallel shows device-contention flakiness — serial authoritative)
- `cargo test -p xlog-prob --release --features host-io --test epistemic_prob_production_reuse` → **7 passed**
- `cargo fmt --check` → clean; `git diff --check` → clean; conflict-marker scan → clean

### Inadmissible as GPU-native acceptance evidence

- The full `cargo test -p xlog-prob --release --features host-io` suite (MC
  CPU/oracle/`gpu_mc_vs_cpu.rs` host-heavy surfaces) is a regression check only,
  not a no-host-transfer / GPU-native acceptance gate.
