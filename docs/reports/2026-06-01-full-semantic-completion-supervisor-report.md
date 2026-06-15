# Supervisor Report — Full Epistemic Semantic Completion

**Date:** 2026-06-01
**Final closure update:** 2026-06-02, epistemic release-closure branch @ `f231278b` (NOT pushed, NOT tagged)
**Superseded pre-closure branch:** epistemic and Monte Carlo resident integration branch @ `ba34152e`
**Base:** executor-completion tip + union-merge of main `96d1530d` (MC GPU-resident engine), merge `dde60b87`; final branch has local `main` `975ab780` as an ancestor.
**Final closure record:** final June 2 epistemic closure record
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
`test_epistemic_split` **44**, FAEEL/foundedness/Gelfond 1991 compatibility **4/7/5**, `logic_runner` **8**;
`cargo check --workspace --all-targets`, `cargo fmt --check` clean. Epistemic examples 01–41.

---

## Delivered (executable, exact semantics, verified)

| Item | Result |
|---|---|
| FAEEL founded-extension | `p :- possible p` → exact empty founded extension (`rows:0`), Gelfond 1991 compatibility → `rows:1`. Not a rejection. |
| Positive recursive fixpoint | Positive `know`/`possible` recursion → founded least fixpoint via the semi-naive engine. |
| Stratified negated-modal recursion | `not know R` over a strictly-lower stratum → ordinary anti-join (ex 37); genuine cycle bounded (below). |
| Nested modal operators | Standard modal-chain collapse (`know possible p ≡ possible p`); per-mode sound. |
| Structured modal keys | List/compound/anonymous keys flatten into the N-column matcher; unbounded → `ResourceExhausted`. |
| Variable-keyed constraints | Single-occurrence existential + multi-literal distinct-variable conjunction. |
| Shared-variable constraint joins | Join / diagonal / negated-difference via program-level desugaring (ex 38–41). |
| Unsafe negated-variable-keyed constraints | `:- not know p(X)` → NAF safety error (ill-formed, matches ordinary Datalog). |
| Unsafe split / coupling | Same-name multi-arity (arity-qualified resolution) + derived-head stratification with equivalence. |

### Method (the through-line)

Each item was resolved by **finding a sound reduction that reuses existing machinery**,
verified end-to-end through the production `xlog run` path with exact tuples and a
load-bearing mutation/flip, guarded for soundness, and merged only with all gates green:

- **Shared-variable constraint joins** (this session's headline): a shared-variable epistemic constraint
  desugars at the `normalize_program` pre-pass to `__epi_join_N(Vars) :- ordinary(ModalA),…,ordinary(ModalZ).`
  + `:- know __epi_join_N(Vars).`, where `ord` ordinary-izes each modal literal
  (`know/possible r → r`, `not know/possible r → not r`). For a base/EDB or
  ordinary-derived target `know r ≡ possible r ≡ r`, so the ordinary join is exactly the
  forbidden binding set; the single-occurrence `:- know __epi_join_N(Vars)` routes through
  the existing variable-keyed **prune-to-empty** path — **no new kernel**. Guarded to
  non-modal-derived targets; a modal-derived target falls through to the core compiler's
  existing rejection (which a device test still covers by feeding the un-normalized program
  directly). The earlier "runtime bug" diagnosis was an artifact of the *wrong* desugar shape
  (helper-rule + an *ordinary* constraint → violated-constraint error path; or a nullary
  helper → schema-union bug); ordinary-izing into the helper while keeping a *single-occurrence
  modal* constraint avoids both.

---

## Current candidate status — fresh gates passed under the exact WFS contract

The branch may claim semantic closure only under the contract actually
implemented and verified: accepted cyclic negated-modal recursion no longer
routes through the old `xlog_prob` host WFS solver, `xlog-gpu` no longer depends
on `xlog-prob`, and the high-level executor builds the `xlog-gpu`
GPU-backed alternating-fixpoint WFS plan. This is **not** a
device-resident/no-host-interaction WFS residency claim; the path still has host
orchestration, and convergence may use metadata row-count reads.

1. **Recursion negation-cycle (ex 33) — GPU-backed WFS plan verified under the
   no-old-host-WFS-solver contract.** The reduced ordinary program contains a
   non-monotone SCC, so the high-level GPU compiler builds a GPU-backed WFS
   alternating-fixpoint plan. `wfs.rs` remains host-only (pure
   `HashSet`/`HashMap`, used by the CPU prob-provenance path); the
   no-host-solver lock forbids routing accepted epistemic programs through it,
   and `xlog-gpu` no longer declares an `xlog-prob` dependency. Fresh focused
   evidence covers the focused recursion-negation fixture matrix, the
   CUDA-independent `xlog-gpu` manifest/source guard, WFS compile/plan-kind
   matrix over mode, negated-modal operator, seed-state, and
   ordinary-EDB-negation axes, the zero-iteration clamp, deterministic single-,
   multi-predicate, and ordinary-EDB-negation `wfs_fixed_relations` JSON,
   explicit `wfs_convergence_predicates`, `wfs_gpu_passes`, and
   `host_wfs_fallback_allowed:false`, the same full-axis GPU runtime WFS matrix,
   and a fixed-name collision regression.

2. **Same-name multi-arity via `xlog run` — done and freshly gated.**
   The full production path now has committed same-name multi-arity matrix fixtures, plus a CLI
   test that covers arity `{1,2}` × modal form `{K,M,not K,not M}` × tuple state and every
   cross-arity conjunction cell. This item must not be listed as undone.

3. **Interior-negation / finite nested negation matrix — done and freshly gated.**
   The fixture is now named `13f-nested-modal-interior-negation.xlog`, declares and asserts
   `p()`, and proves the stated duality case through `xlog run`: present `p()` makes
   `not possible p()` false, so `q` is empty with exit 0. Its companion cells cover target
   `{present,absent}` x mode `{FAEEL,Gelfond 1991 compatibility}`. The broader matrix covers all
   64 two-operator negation cells over `{know,possible}^2`, leading/interior/atom-adjacent
   negation, and present/absent atoms under FAEEL; the compatibility matrix replays those cells under explicit Gelfond 1991 compatibility.
   Fresh focused/full gates passed before using it as merge evidence; see
   the June 2 epistemic closure validation checklist.

---

## Process notes

- **Soundness discipline tightened:** cyclic WFS is no longer accepted through host grounding or
  host WFS; the current source routes it through the `xlog-gpu` GPU-backed WFS
  plan and the fresh gates have passed under that exact contract.
- **Accepted examples expanded:** same-name multi-arity, single-modal truth tables, and WFS
  cyclic-negated mode/operator cases now have finite fixture matrices rather than one-off pilots.
- **Verified floor protected:** an incomplete EIR-layer attempt at the diagonal (accepted at
  planning but un-materialized at execution) was reverted rather than shipped, then redone
  correctly at the Program/`normalize_program` layer.
- **Honest accounting over surface completion:** the release claim must remain
  scoped to the verified contract. Do not market this as device-resident or
  no-host-interaction WFS unless a separate residency implementation and gate
  set lands.

## Recommendation

**Merge candidate under the exact WFS contract above.** If the branch is marketed
as "no old host-WFS solver for accepted cyclic negated-modal recursion," the
implementation and gates satisfy that claim. If it instead requires a
device-resident/no-host-interaction WFS residency contract, this branch is still
held for fixes for that stronger requirement.
