# Supervisor Report — v0.9.2 FULL Epistemic Semantic Completion

**Date:** 2026-06-01
**Branch:** `integration/v092-main-mc-resident` @ `ba34152e` (NOT pushed, NOT tagged)
**Base:** v0.9.1 tip + union-merge of main `96d1530d` (MC GPU-resident engine), merge `dde60b87`
**Scope:** the full-semantic-completion mandate — make every formerly-rejected epistemic
construct EITHER execute with exact semantics OR fail closed only for a genuine
formal/architectural reason; no diagnostic-only closure, no gaming.

---

## Executive summary

Every epistemic construct that can be resolved at the **epistemic layer** now executes with
exact, verified semantics. The constructs that remain rejected have each been **root-caused
to a core-engine or representation limitation outside the epistemic layer**, with evidence —
not relabeled rejections.

**Acceptance (verified serially, `--test-threads=1`):** device `test_epistemic_gpu_workspace`
**144**, integration `test_epistemic_gpu_wcoj_execution` **206**, CLI `run_cli_tests` **12**,
`test_epistemic_split` **44**, FAEEL/foundedness/G91 **4/7/5**, `logic_runner` **8**;
`cargo check --workspace --all-targets`, `cargo fmt --check` clean. Epistemic examples 01–41.

---

## Delivered (executable, exact semantics, verified)

| Item | Result |
|---|---|
| **B** FAEEL founded-extension | `p :- possible p` → exact empty founded extension (`rows:0`), G91 → `rows:1`. Not a rejection. |
| **A** Case-B recursive fixpoint | positive `know`/`possible` recursion → founded least fixpoint via the semi-naive engine. |
| **A1** stratified negated-modal recursion | `not know R` over a strictly-lower stratum → ordinary anti-join (ex 37); genuine cycle bounded (below). |
| **C** nested modal operators | KD45/S5 chain collapse (`know possible p ≡ possible p`); per-mode sound. |
| **D** structured modal keys | list/compound/anonymous keys flatten into the N-column matcher; unbounded → `ResourceExhausted`. |
| **E** variable-keyed constraints | single-occurrence existential + multi-literal distinct-variable conjunction. |
| **E1** shared-variable constraint joins | join / diagonal / negated-difference via program-level desugaring (ex 38–41). |
| **E2** unsafe negated-variable-keyed | `:- not know p(X)` → NAF safety error (ill-formed, matches ordinary Datalog). |
| **F** unsafe split / coupling | same-name multi-arity (arity-qualified resolution) + derived-head stratification w/ equivalence. |

### Method (the through-line)

Each item was resolved by **finding a sound reduction that reuses existing machinery**,
verified end-to-end through the production `xlog run` path with exact tuples and a
load-bearing mutation/flip, guarded for soundness, and merged only with all gates green:

- **E1** (this session's headline): a shared-variable epistemic constraint `:- L1,…,Ln`
  desugars at the `normalize_program` pre-pass to `__epi_join_N(Vars) :- ord(L1),…,ord(Ln).`
  + `:- know __epi_join_N(Vars).`, where `ord` ordinary-izes each modal literal
  (`know/possible r → r`, `not know/possible r → not r`). For a base/EDB or
  ordinary-derived target `know r ≡ possible r ≡ r`, so the ordinary join is exactly the
  forbidden binding set; the single-occurrence `:- know __epi_join_N(Vars)` routes through
  item E's existing variable-keyed **prune-to-empty** path — **no new kernel**. Guarded to
  non-modal-derived targets; a modal-derived target falls through to the core compiler's
  existing rejection (which a device test still covers by feeding the un-normalized program
  directly). The earlier "runtime bug" diagnosis was an artifact of the *wrong* desugar shape
  (helper-rule + an *ordinary* constraint → violated-constraint error path; or a nullary
  helper → schema-union bug); ordinary-izing into the helper while keeping a *single-occurrence
  modal* constraint avoids both.

---

## Current candidate status — validation pending

The branch must not claim full semantic completion until fresh gates pass. Current source has
removed the accepted host-side WFS route and now attempts cyclic negated-modal recursion through
a GPU-native alternating-fixpoint WFS plan instead of CPU-backed execution.

1. **Recursion negation-cycle (ex 33) — GPU-native WFS candidate implemented, gates pending.**
   The reduced ordinary program contains a non-monotone SCC, so the high-level GPU compiler now
   builds a GPU-native WFS alternating-fixpoint plan. `wfs.rs` remains host-only (pure
   `HashSet`/`HashMap`, used by the CPU prob-provenance path); the no-host-solver lock still
   forbids routing accepted epistemic programs through it, and `xlog-gpu` no longer declares an
   `xlog-prob` dependency. Required evidence is a fresh focused
   ex33/`33a*`/`33c*`/33b/`33d*`/`33e*` CLI matrix pass, the
   CUDA-independent `xlog-gpu` manifest/source guard, WFS compile/plan-kind matrix over mode,
   negated-modal operator, seed-state, and ordinary-EDB-negation axes, and zero-iteration
   clamp check with deterministic single-, multi-predicate, and ordinary-EDB-negation
   `wfs_fixed_relations` JSON plus explicit `wfs_convergence_predicates` and `wfs_gpu_passes`,
   `host_wfs_fallback_allowed:false`, the same full-axis GPU runtime WFS matrix plus fixed-name
   collision regression, and the broader release gates. The exact blocker-to-gate checklist is
   `docs/plans/2026-06-02-v092-epistemic-closure-validation-checklist.md`.

2. **Same-name multi-arity via `xlog run` — done in current source, but requires rerun gates.**
   The full production path now has committed `42a*` and `42b*` matrix fixtures, plus a CLI
   test that covers arity `{1,2}` × modal form `{K,M,not K,not M}` × tuple state and every
   cross-arity conjunction cell. This item must not be listed as undone, but the current edit
   set still needs focused and full gate reruns before any merge claim.

3. **C2 interior-negation / finite nested negation matrix — done in current source, gates pending.**
   The fixture is now named `13f-nested-modal-interior-negation.xlog`, declares and asserts
   `p()`, and proves the stated duality case through `xlog run`: present `p()` makes
   `not possible p()` false, so `q` is empty with exit 0. Its companion cells cover target
   `{present,absent}` x mode `{FAEEL,G91}`. The broader 13g-13v matrix covers all
   64 two-operator negation cells over `{know,possible}^2`, leading/interior/atom-adjacent
   negation, and present/absent atoms under FAEEL; 13w* replays those cells under explicit G91.
   Fresh focused/full gates are still required before using it as merge evidence; see
   `docs/plans/2026-06-02-v092-epistemic-closure-validation-checklist.md`.

---

## Process notes

- **Soundness discipline tightened:** cyclic WFS is no longer accepted through host grounding or
  host WFS; the candidate source routes it through GPU-native WFS and still needs fresh gates.
- **Accepted examples expanded:** same-name multi-arity, single-modal truth tables, and WFS
  cyclic-negated mode/operator cases now have finite fixture matrices rather than one-off pilots.
- **Verified floor protected:** an incomplete EIR-layer attempt at the diagonal (accepted at
  planning but un-materialized at execution) was reverted rather than shipped, then redone
  correctly at the Program/`normalize_program` layer.
- **Honest accounting over surface completion:** current edits require fresh gate evidence. Do
  not reuse the older green SHA as proof for this tree.

## Recommendation

**HOLD_FOR_FIXES.** If v0.9.2 requires accepted cyclic negated-modal recursion, the remaining
missing deliverable is GPU-native WFS (or explicit host-WFS authorization). If that case is
scoped as a boundary for this release, rerun the focused and full gates on the current tree and
record the new evidence before any merge/tag decision.
