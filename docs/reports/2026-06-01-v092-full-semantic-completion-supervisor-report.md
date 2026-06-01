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

## Honestly undone — root-caused (core-engine / representation, NOT epistemic-layer)

These are rejected, and the rejection is justified by a real limitation *below* the epistemic
layer. None is fakeable without the named larger effort. Each has a precise recipe.

1. **Recursion negation-cycle (ex 33) + G91 `possible`-recursion — WHOLE-ENGINE BOUND.**
   These need non-stratified (well-founded / stable-model) semantics. **Proven not
   epistemic-specific:** ordinary `a :- not b. b :- not a.` and ordinary
   recursion-through-negation BOTH error on this engine — XLOG is a stratified-negation
   engine. `wfs.rs` exists but is host-only (pure `HashSet`/`HashMap`, used only by the CPU
   prob-provenance path); the no-host-solver lock forbids routing accepted programs through it.
   *Recipe:* build a GPU (or admitted-host) well-founded/stable-model evaluator for the whole
   engine — a separate project.

2. **Same-name multi-arity coupling via `xlog run` — PERVASIVE name-keyed-schema refactor.**
   The coupling *semantics* already work at the device/runtime layer (item F, exact tuples per
   arity). The full `xlog run` path is blocked because predicate identity is name-keyed across
   the ordinary compiler — `Compiler::schemas: HashMap<String, Schema>`, `rel_ids()` (name→RelId),
   `cardinality_hints`, `fact_counts`, and the lowerer — so `pred p(u32)` + `pred p(u32,u32)`
   collapse, and `load_facts_into_store` fails the arity check. *Recipe:* arity-key the global
   schema / relation-identity model (an ordinary-compiler refactor used by every program; high
   cross-engine regression risk).

3. **Modal-over-compound-formula, C2 interior-negation (`know not possible p`, ex 13f) —
   REPRESENTATION EXTENSION.** Modal duality gives a reduction in principle
   (`know not possible p ≡ know not p ≡ not p` for determined targets), but the intermediate
   form `K¬p` (know-of-a-negated-atom) has no EIR representation and the final step needs the
   determined-stage analysis — so it is not a clean parser-level collapse like item C.
   *Recipe:* add a `K¬p` EIR node (or a determined-stage duality reduction) + grammar/pilots.

---

## Process notes

- **Soundness discipline held:** every accepted construct verified through the production path
  with exact tuples + a load-bearing flip; no construct closed by relabeling a rejection.
  Where the diagonal desugaring could be unsound (modal-derived target), a guard was added and
  the unsound case verified to fall through to a coherent rejection.
- **Verified floor protected:** an incomplete EIR-layer attempt at the diagonal (accepted at
  planning but un-materialized at execution) was reverted rather than shipped, then redone
  correctly at the Program/`normalize_program` layer.
- **Honest accounting over surface completion:** the three residuals were investigated to root
  cause and left undone rather than faked or risk-broken with a depth-exhausted core-engine
  refactor.

## Recommendation

The epistemic-layer semantic completion is done and verified. **Merge/tag is a supervisor
decision.** The three residuals are real, scoped follow-ups (one whole-engine WFS effort, one
ordinary-compiler schema refactor, one EIR representation extension) — each warrants a
dedicated, fresh-context effort with full gate discipline, not a tail-end attempt. The branch
is internally consistent (docs reconciled) and gate-green at `ba34152e`.
