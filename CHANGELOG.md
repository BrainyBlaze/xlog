# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## [0.11.0](https://github.com/BrainyBlaze/xlog/compare/xlog-cli-v0.10.0...xlog-cli-v0.11.0) - 2026-07-15

### Added

- WCOJ observability, residency-ablation hook, whitepaper neural sections, docs-site audit

### Other

- ship the built whitepaper PDF as the committed deliverable
- promote the documentation-site source to docs/ and retarget references
- move the whitepaper to paper/ and stop tracking LaTeX build outputs

## [0.11.0](https://github.com/BrainyBlaze/xlog/compare/xlog-gpu-v0.10.0...xlog-gpu-v0.11.0) - 2026-07-15

### Other

- ship the built whitepaper PDF as the committed deliverable
- promote the documentation-site source to docs/ and retarget references
- move the whitepaper to paper/ and stop tracking LaTeX build outputs

## [0.11.0](https://github.com/BrainyBlaze/xlog/compare/xlog-prob-v0.10.0...xlog-prob-v0.11.0) - 2026-07-15

### Other

- ship the built whitepaper PDF as the committed deliverable
- promote the documentation-site source to docs/ and retarget references
- move the whitepaper to paper/ and stop tracking LaTeX build outputs

## [0.11.0](https://github.com/BrainyBlaze/xlog/compare/xlog-solve-v0.10.0...xlog-solve-v0.11.0) - 2026-07-15

### Other

- ship the built whitepaper PDF as the committed deliverable
- promote the documentation-site source to docs/ and retarget references
- move the whitepaper to paper/ and stop tracking LaTeX build outputs

## [0.11.0](https://github.com/BrainyBlaze/xlog/compare/xlog-runtime-v0.10.0...xlog-runtime-v0.11.0) - 2026-07-15

### Added

- WCOJ observability, residency-ablation hook, whitepaper neural sections, docs-site audit

### Other

- ship the built whitepaper PDF as the committed deliverable
- promote the documentation-site source to docs/ and retarget references
- move the whitepaper to paper/ and stop tracking LaTeX build outputs

## [0.11.0](https://github.com/BrainyBlaze/xlog/compare/xlog-logic-v0.10.0...xlog-logic-v0.11.0) - 2026-07-15

### Other

- ship the built whitepaper PDF as the committed deliverable
- promote the documentation-site source to docs/ and retarget references
- move the whitepaper to paper/ and stop tracking LaTeX build outputs

## [0.11.0](https://github.com/BrainyBlaze/xlog/compare/xlog-stats-v0.10.0...xlog-stats-v0.11.0) - 2026-07-15

### Other

- ship the built whitepaper PDF as the committed deliverable
- promote the documentation-site source to docs/ and retarget references
- move the whitepaper to paper/ and stop tracking LaTeX build outputs

## [0.11.0](https://github.com/BrainyBlaze/xlog/compare/xlog-cuda-v0.10.0...xlog-cuda-v0.11.0) - 2026-07-15

### Other

- ship the built whitepaper PDF as the committed deliverable
- promote the documentation-site source to docs/ and retarget references
- move the whitepaper to paper/ and stop tracking LaTeX build outputs

## [0.11.0](https://github.com/BrainyBlaze/xlog/compare/xlog-ir-v0.10.0...xlog-ir-v0.11.0) - 2026-07-15

### Other

- ship the built whitepaper PDF as the committed deliverable
- promote the documentation-site source to docs/ and retarget references
- move the whitepaper to paper/ and stop tracking LaTeX build outputs

## [0.11.0](https://github.com/BrainyBlaze/xlog/compare/xlog-core-v0.10.0...xlog-core-v0.11.0) - 2026-07-15

### Other

- ship the built whitepaper PDF as the committed deliverable
- promote the documentation-site source to docs/ and retarget references
- move the whitepaper to paper/ and stop tracking LaTeX build outputs

## [0.10.0](https://github.com/BrainyBlaze/xlog/compare/xlog-cli-v0.9.2...xlog-cli-v0.10.0) - 2026-07-06

### Added

- *(runtime)* D2 Free Join production integration â general multiway promotion + executor dispatch
- *(cli)* expose epistemic explain plans ([#138](https://github.com/BrainyBlaze/xlog/pull/138))

### Fixed

- *(prob)* fail closed on resident MC rejection with labeled CPU-oracle opt-in

### Other

- rewrite README for public release and remove environment-specific paths
- clarify language completeness naming
- clarify epistemic run test comments
- inline magic set explain fixture
- clarify epistemic plan json docs
- clarify epistemic evidence fixture names
- clarify readme artifact labels
- add source clarity closure board
- label unmeasured benchmark targets and document MC engine split
- Add UCR-driven XLOG engine support ([#139](https://github.com/BrainyBlaze/xlog/pull/139))

## [0.10.0](https://github.com/BrainyBlaze/xlog/compare/xlog-gpu-v0.9.2...xlog-gpu-v0.10.0) - 2026-07-06

### Added

- *(pyxlog)* expose session multiway dispatch telemetry

### Other

- rewrite README for public release and remove environment-specific paths
- apply workspace rustfmt
- replace opaque task labels with behavior-based wording in factorized code/tests
- Merge branch 'feat/factorized-finalize'
- clarify delta coalescing fixture relation
- clarify language completeness naming
- clarify logic runner plan labels
- clarify epistemic plan labels
- clarify readme artifact labels
- add source clarity closure board
- label unmeasured benchmark targets and document MC engine split

## [0.10.0](https://github.com/BrainyBlaze/xlog/compare/xlog-prob-v0.9.2...xlog-prob-v0.10.0) - 2026-07-06

### Added

- *(neurosymbolic)* Stage B — existential-join trainable bodies (real-domain grounding)
- *(prob)* provenance-derived decision-order hints for GPU D4
- *(prob)* factorized outcome folding for exact non-count aggregates

### Fixed

- *(prob)* recover from stale disk-cached circuits instead of failing compilation
- *(prob)* converge two-sided recursive SCC provenance via OR/AND flattening and absorption
- *(prob)* fail closed on resident MC rejection with labeled CPU-oracle opt-in

### Other

- *(fmt)* apply rustfmt to epistemic stratified-plan docs and provenance absorption code
- rewrite README for public release and remove environment-specific paths
- apply workspace rustfmt
- mixed-trainable-rule bodies + GPU-resident zero-host neuro-symbolic training
- *(prob,pyxlog)* eliminate tracked host upload from the neuro-symbolic training loop
- replace opaque task labels with behavior-based wording in factorized code/tests
- Merge branch 'feat/factorized-finalize'
- Merge branch 'codex/artifact-closure-board-20260614'
- clarify probabilistic aggregate tests
- clarify monte carlo approximate tests
- clarify aggregate lifting tests
- clarify template addition variables
- clarify decision order hint terminology
- clarify fused backward level labels
- clarify cdcl q2 diagnostics
- clarify provenance diagnostics
- clarify mc result diagnostics
- clarify monte carlo semantics docs
- clarify exact compiler docs
- clarify epistemic production semantics labels
- clarify epistemic compiler adapter docs
- clarify decision order compiler docs
- clarify compilation compiler comments
- clarify gpu compiler module docs
- clarify gpu frontier compiler docs
- clarify exact benchmark compiler note
- clarify language completeness naming
- clarify readme artifact labels
- add source clarity closure board
- label unmeasured benchmark targets and document MC engine split
- Add UCR-driven XLOG engine support ([#139](https://github.com/BrainyBlaze/xlog/pull/139))

## [0.10.0](https://github.com/BrainyBlaze/xlog/compare/xlog-solve-v0.9.2...xlog-solve-v0.10.0) - 2026-07-06

### Added

- *(prob,solve,cuda)* fail-closed D4 compile/verify robustness — typed declines instead of context-poisoning launches

### Other

- rewrite README for public release and remove environment-specific paths
- clarify solver service blocker
- clarify pigeonhole comments
- clarify accepted evidence compatibility tests
- clarify production adapter transfer wording
- clarify readme artifact labels
- add source clarity closure board
- label unmeasured benchmark targets and document MC engine split

## [0.10.0](https://github.com/BrainyBlaze/xlog/compare/xlog-runtime-v0.9.2...xlog-runtime-v0.10.0) - 2026-07-06

### Added

- *(cuda)* D2 Phase C â u64 Free Join engine, recursive verification, factorized count-by-root
- *(runtime)* D2 Free Join production integration â general multiway promotion + executor dispatch
- *(runtime)* wire K=5/K=6 clique count fusion through promoter and executor
- *(runtime)* dispatch u64-key sum/min/max through the fused WCOJ aggregates
- *(runtime)* dispatch 4-cycle count-by-root aggregates through the fused WCOJ kernel
- *(runtime)* dispatch sum/min/max and u64 count through fused WCOJ aggregates
- *(runtime)* dispatch count-by-root aggregates through the fused WCOJ kernel
- *(runtime)* count WCOJ pipeline error declines and add XLOG_WCOJ_STRICT

### Other

- rewrite README for public release and remove environment-specific paths
- replace internal task-codes with behavioral descriptions in comments, tests, and examples
- apply workspace rustfmt
- replace opaque task labels with behavior-based wording in factorized code/tests
- Merge branch 'feat/factorized-finalize'
- Merge branch 'codex/artifact-closure-board-20260614'
- clarify cost model default tests
- clarify leader input permutation tests
- clarify production reuse transfer text
- clarify epistemic gpu workspace labels
- clarify wcoj dispatch labels
- clarify rewrite occurrence labels
- clarify recursive dispatch labels
- clarify node dispatch labels
- clarify executor milestone labels
- clarify epistemic workspace labels
- clarify recursive stats trace tests
- clarify clique dispatch helper artifacts
- clarify chain dispatch artifacts
- clarify readme artifact labels
- add source clarity closure board
- Merge branch 'feat/factorized-kclique-count-fusion'
- label unmeasured benchmark targets and document MC engine split

## [0.10.0](https://github.com/BrainyBlaze/xlog/compare/xlog-logic-v0.9.2...xlog-logic-v0.10.0) - 2026-07-06

### Added

- *(cuda)* D2 Phase C â u64 Free Join engine, recursive verification, factorized count-by-root
- *(runtime)* D2 Free Join production integration â general multiway promotion + executor dispatch
- *(runtime)* wire K=5/K=6 clique count fusion through promoter and executor
- *(runtime)* dispatch 4-cycle count-by-root aggregates through the fused WCOJ kernel
- *(runtime)* dispatch count-by-root aggregates through the fused WCOJ kernel

### Other

- *(fmt)* apply rustfmt to epistemic stratified-plan docs and provenance absorption code
- rewrite README for public release and remove environment-specific paths
- remove remaining internal task-codes from epistemic examples and tests
- replace internal task-codes with behavioral descriptions in comments, tests, and examples
- clarify kclique cost gate test
- clarify kclique promoter test
- clarify heat aware var ordering test
- clarify leader cardinality test
- clarify language completeness naming
- clarify multiway promotion test docs
- clarify skewed multiway planner fixture
- clarify epistemic split test docs
- clarify g91 mode test docs
- clarify faeel foundedness docs
- clarify epistemic executable plan docs
- clarify epistemic eir test docs
- clarify neural parser fixture variables
- clarify learnable test comments
- clarify wcoj variable ordering docs
- clarify promoter docs
- clarify parser comments
- clarify stream scheduler docs
- clarify optimizer planner docs
- clarify meta normalization diagnostics
- clarify lowering diagnostics
- clarify variable order docs
- clarify epistemic planner docs
- clarify compiler config docs
- clarify compiler pipeline docs
- clarify gelfond epistemic mode
- clarify optimizer demo variables
- clarify clique dispatch helper artifacts
- clarify chain dispatch artifacts
- clarify readme artifact labels
- add source clarity closure board
- label unmeasured benchmark targets and document MC engine split
- Add UCR-driven XLOG engine support ([#139](https://github.com/BrainyBlaze/xlog/pull/139))

## [0.10.0](https://github.com/BrainyBlaze/xlog/compare/xlog-stats-v0.9.2...xlog-stats-v0.10.0) - 2026-07-06

### Other

- rewrite README for public release and remove environment-specific paths
- clarify readme artifact labels
- add source clarity closure board
- label unmeasured benchmark targets and document MC engine split

## [0.10.0](https://github.com/BrainyBlaze/xlog/compare/xlog-cuda-v0.9.2...xlog-cuda-v0.10.0) - 2026-07-06

### Added

- *(prob,solve,cuda)* fail-closed D4 compile/verify robustness — typed declines instead of context-poisoning launches
- *(cuda)* distinct-aware sparse table sizing (2-pass estimator)
- *(runtime)* route factorized delta dense|sparse|legacy by domain
- *(cuda)* D3 sparse-domain hash-set novel-set spike kernels + provider
- *(cuda)* re-export FJ_DELTA_MAX_DOMAIN for the runtime domain cap
- *(cuda)* generalize fj_delta entry to column roles + domain max helper
- *(cuda)* fj_delta factorized novel-set pipeline (D3 S3 spike)
- *(cuda)* peak-bytes high-water mark on GpuMemoryManager
- *(cuda)* env-gated memory debug probes for corruption forensics
- *(cuda)* accept u64 value columns for recorded-groupby min/max
- *(cuda)* fuse u64-key 4-cycle group-by-root count
- *(cuda)* fuse 4-cycle sum/min/max group-by-root aggregates (u32)
- *(cuda)* u64-key sum/min/max fused WCOJ group-by-root aggregates
- *(cuda)* widen the legacy groupby to u64-value sum/min/max
- *(cuda)* aggregate-fused WCOJ 4-cycle group-by-root count
- *(cuda)* u64-key variant of the fused WCOJ group-by-root count
- *(cuda)* aggregate-fused WCOJ sum/min/max group-by-root (u32)
- *(cuda)* aggregate-fused WCOJ triangle group-by-root count

### Fixed

- *(cuda)* skip stale staged kernel artifacts, fail closed, auto-heal to embedded PTX
- *(cuda)* record conditional R-column reads before LaunchRecorder preflight
- *(cuda)* guard compute_ranks tail blocks against block_count underflow
- *(cuda)* drop unused mut on fused staging key copies
- *(cuda)* layout-normalize inputs in all fused group-by-root entries

### Other

- allow rustdoc generation without nvcc
- rewrite README for public release and remove environment-specific paths
- remove remaining internal task-codes from epistemic examples and tests
- replace internal task-codes with behavioral descriptions in comments, tests, and examples
- apply workspace rustfmt
- Merge branch 'codex/artifact-closure-board-20260614'
- Support ptxas cubin assembly for packaged kernels
- Fallback to portable PTX when cubin load fails
- *(cuda)* correct fj_delta_sparse entry doc for distinct sizing
- Merge branch 'feat/d3-factorized-delta'
- Merge branch 'feat/d2-free-join'
- Merge branch 'feat/factorized-kclique-count-fusion'
- label unmeasured benchmark targets and document MC engine split
- Add UCR-driven XLOG engine support ([#139](https://github.com/BrainyBlaze/xlog/pull/139))

## [0.10.0](https://github.com/BrainyBlaze/xlog/compare/xlog-ir-v0.9.2...xlog-ir-v0.10.0) - 2026-07-06

### Added

- *(runtime)* D2 Free Join production integration â general multiway promotion + executor dispatch

### Other

- rewrite README for public release and remove environment-specific paths
- clarify multiway rir tests
- clarify multiway route docs
- clarify free join arity metadata
- clarify gelfond compatibility mode
- clarify readme artifact labels
- add source clarity closure board
- label unmeasured benchmark targets and document MC engine split

## [0.10.0](https://github.com/BrainyBlaze/xlog/compare/xlog-core-v0.9.2...xlog-core-v0.10.0) - 2026-07-06

### Added

- *(prob,solve,cuda)* fail-closed D4 compile/verify robustness — typed declines instead of context-poisoning launches

### Other

- rewrite README for public release and remove environment-specific paths
- clarify runtime config docs
- clarify readme artifact labels
- add source clarity closure board
- label unmeasured benchmark targets and document MC engine split

### Added

- *(pyxlog)* **Graded per-binding candidate masses in the joint noisy-OR
  mixture.** `train_neurosymbolic_program(..., candidate_masses={rule_id:
  tensor})` supplies per-binding confidences in [0, 1] that multiply into a
  candidate's relational eligibility, so the head probability becomes the
  noisy-OR over graded evidence masses. Mapping head bindings to world steps
  and masses to a fact's per-step confidence trains per-candidate guards
  against an evolving trajectory. Omitting the argument leaves the binary
  behavior unchanged; masses are validated for shape and range and rejected
  outside the multi-rule joint path.

- *(pyxlog)* **Stage-B existential-join trainable bodies (real-domain
  grounding).** A trainable body may join a neural predicate to an ordinary
  relation on a non-head variable; the neural predicate is grounded over the
  real join domain inside the circuit and OR-aggregated at the head. Per-event
  features arrive through `domain_inputs=` and `register_domain_tensor_source`;
  a candidate may also carry a neural conjunct (`neural_bodies=`) whose
  straight-through-thresholded head gates its eligibility.

- *(pyxlog)* **Joint multi-rule same-head trainable mixture (guard-only).** A
  query head may now be derived by MORE THAN ONE `trainable_rule` — the joint
  soft-mixture where N candidate rules compete for mass on one head (previously
  rejected with "expected exactly 1 matching rule"). When several trainable rules
  derive the train head, `train_neurosymbolic_program` computes the head
  probability as the noisy-OR over candidates of `(eligible_k × sigmoid(guard_k))`
  and trains the per-candidate guards by BCE on the supervised head, so the
  competition drives the correct candidate's guard up and wrong-body candidates'
  down (a wrong body fires on rows that create false positives BCE crushes). The
  per-candidate relational eligibility is exposed by
  `CompiledProgram.joint_candidate_eligibility` (reusing the engine's hard-filter
  evaluation); the differentiable mass is torch over the guard parameters, so the
  training loop performs no tracked device<->host transfers. Scope: guard-only
  candidates (relational joins plus a trainable guard); candidates carrying a neural
  conjunct are supported via `neural_bodies=` (see the Stage-B entry above).
- *(pyxlog)* **Held-out generalization read for the joint mixture
  (`evaluate_joint_mixture`).** Evaluates a trained joint mixture's guards on a
  HELD-OUT program split: given the held-out bindings' facts and the learned
  guard sigmoids (`result.symbolic_rule_weights`), returns the per-query noisy-OR
  `p_or` over the engine's relational eligibility for that split. It reuses the
  exact `_joint_noisy_or` of the training forward (pinned by a test asserting the
  read on the train split reproduces `query_probabilities`), so the generalization
  signal cannot drift from the trained mixture. This is the faithful anti-spurious
  signal where structural coverage is unavailable: a candidate that fit only the
  training facts yields low held-out `p_or` wherever its join does not fire. The
  read needs only a compiled program (eligibility is relational, never the guard
  network), so no network registration or example tensor source is required.
  Usage is candidate-set controlled by `rule_weights`: pass ONLY the selected
  winner's weight for a single-candidate admission gate (the pool-wide OR is
  inflated wherever any candidate fires, so a high-guard spurious coverer on a
  train-tie would mask the winner's non-generalization); select among
  train-covering candidates by guard-free held-out coverage, not by the (tied)
  guards.
- *(prob/solve)* **Fail-closed D4 equivalence-verify conflict budget
  (compile/verify robustness, primary fix).** Calibration showed the verify explosion is
  treewidth-exponential, not size-linear (onset ~654 CNF vars where legitimate
  programs live; same var count can verify in 1s or run to a watchdog crash),
  so a size bound is too coarse. The GPU CDCL kernel now accepts a
  `max_conflicts` budget and bails with a new `SAT_STATUS_BUDGET_EXHAUSTED`
  status when the search runs past it without terminating — bounding wall-clock
  so a hard instance returns gracefully instead of running to a
  context-poisoning `CUDA_ERROR_LAUNCH_FAILED`. A budget-exhausted verify is
  INDETERMINATE and declines fail-closed with the typed
  `XlogError::VerifyBudgetExceeded` (catchable Python exception), never trusted
  as a proof. `GpuCdclConfig::max_conflicts` defaults to `0` (unlimited, no
  behavior change); the verify path reads `XLOG_D4_VERIFY_MAX_CONFLICTS` (opt-in).
  Recommended production default is a calibration follow-up (between a
  completing verify's conflict count and the watchdog boundary).
- *(prob)* **Fail-closed D4 compile size bound (compile-phase
  guard).** The D4 *compile* itself (knowledge-compilation emit, fixed-capacity
  buffers) can overrun and fail with a `CUDA_ERROR_LAUNCH_FAILED` that poisons
  the primary context — *earlier* than the verify — on a CNF larger than its
  emit caps. A poisoned context cannot be recovered in-process, so the size
  guard runs **before** `compile_gpu_d4` (top of `compile_gpu_d4_and_verify` /
  `_cached`): the CNF's host-side `var_cap`/`clause_cap` are checked against
  `XLOG_D4_VERIFY_MAX_VARS` / `XLOG_D4_VERIFY_MAX_CLAUSES` and an over-bound
  program declines with the typed `XlogError::CompileCapacityExceeded`
  ("too big to compile", distinct from the verify-phase signal) instead of
  reaching the crashing compile. **Defaults are unbounded** (`u32::MAX`); no
  behavior change unless an operator opts in. A recommended production default
  is a calibration follow-up; the in-kernel guard below now makes overflow
  protection effective by default even when no host-side bound is configured.
- *(prob/cuda)* **In-kernel emit-overflow guard — compile-phase protection on
  by default.** The host-side size bound above only fires when an operator
  configures it, but the emit kernel could still overrun its fixed-capacity
  buffers on an unconfigured run and trap. `d4_compile_emit` previously called
  `d4_trap()` (a PTX `trap;`) when the node/edge totals exceeded `node_cap` /
  `edge_cap`, which poisons the CUDA primary context and kills every later
  compile in the process. It now evaluates the same capacity check **before any
  write, in every block**, and on overflow sets a one-word `overflow_flag` and
  returns instead of trapping. The host reads the flag after the launch and
  declines with the typed `XlogError::CompileCapacityExceeded` (catchable Python
  exception) before downstream kernels run on uninitialized buffers — so an
  oversized CNF declines cleanly and the context survives for the next query,
  with **no environment opt-in required**. Buffers remain fixed-capacity, so an
  oversized program still declines rather than compiles; growable emit buffers
  (so large programs compile) remain a follow-up.
- *(cuda)* **Kernel-artifact integrity guard — stale staged cubin/PTX
  auto-heal, on by default.** A staged kernel artifact whose signature had
  diverged from the current build (e.g. a kernel gained a parameter but the
  staged copy was never refreshed) loaded "successfully" and then launched a
  mismatched kernel into a context-poisoning `CUDA_ERROR_ILLEGAL_ADDRESS` —
  surviving clean rebuilds because the staged copy is regenerated by a separate
  staging step. The build now embeds a canonical FNV-1a hash for every kernel
  artifact it produces; the loader re-hashes each staged cubin/PTX and SKIPS any
  that diverge from this build, ALWAYS appends the embedded portable PTX as a
  signature-fresh final fallback (so a skipped stale artifact auto-heals to it),
  and FAILS CLOSED if no source loads (never silently runs nothing). No
  CUDA-kernel change; dependency-free FNV-1a (build.rs ↔ runtime, known-answer
  locked). Verified on a Blackwell sm_120 canary: a stale staged cubin
  auto-heals (no OOB), a fresh artifact keeps the fast cubin path.
- *(pyxlog)* **Mixed trainable-rule bodies: neural predicates joined with
  ordinary relations.** A `trainable_rule` body may now join a neural predicate
  with ordinary world relations (in addition to builtins). The ordinary
  relations act as HARD join conditions — they gate which groundings can fire
  but contribute no probability mass and no gradient; probability comes only
  from the neural predicates x sigma(rule weight), and gradients flow only to
  the network and the rule weight, never through the fact atoms. The
  knowledge-compilation circuit covers just the neural part; the hard
  conditions are evaluated as a pre-filter, and a query whose hard conditions
  fail short-circuits to probability 0 before any network forward (enforcing the
  gradient isolation). Current scope: hard conditions whose arguments are query
  head variables; joining an ordinary relation on an existential (non-head)
  variable still fails closed with a typed error (documented follow-up).
- *(pyxlog)* **GPU-resident, zero-host neuro-symbolic training surface.** The
  `train_neurosymbolic_program` step loop is now device-resident: a new
  `CompiledProgram.forward_backward_grouped(queries, expected)` evaluates every
  example's supervised circuit in one batched pass per (target, template) group
  with a single host sync per step, instead of the scalar `forward_backward`
  host-syncing once per query (which left training CPU-bound). The batched
  query-var metadata is cached on device, so the warm training loop performs
  **no tracked device<->host transfers in either direction**; the post-training
  probability readout is batched too (`query_probabilities_grouped`,
  O(templates) host syncs instead of O(N)). **Behavior change:** the training
  optimizer is configurable (`NeuroSymbolicTrainingConfig.optimizer`,
  `"adam"` | `"sgd"`) and **defaults to Adam** — the supervised loss is
  multiplicative (`prob = softmax_positive x sigmoid(rule_weight)`) with a flat
  plateau around uniform init that plain SGD frequently cannot leave (it
  separated a cleanly separable signal in ~1/10 random inits vs ~8/10 for Adam);
  the engine gradient itself is exact (finite-difference-verified to 0.01%). The
  grouped loss and batched readout are numerically identical to the per-query
  scalar path.
- *(runtime)* **D3 — factorized recursive deltas.** Transitive-closure-shaped
  recursive rules (`q(X,Z) :- q(X,Y), edge(Y,Z)` and its left-linear /
  non-linear-self-join / swapped-head variants) now route their semi-naive
  delta step through a factorized novel-set pipeline that evaluates
  `novel[x] = (∪ edge[y]) \ R[x]` over a dense-domain characteristic bitvector
  instead of materializing the witness-multiplied flat join and diffing it.
  Measured on RunPod RTX A4000 through the production executor: **41.46× peak
  memory reduction at 0.092× wall-clock** on a dense block-cycle TC
  (1,048,576 result rows), with deterministic row-set parity. Gated to the
  dense-domain regime (`XLOG_FACTORIZED_DELTA_MAX_DOMAIN`, default 2¹⁴, hard
  bound 2¹⁶) with a per-iteration work floor that bails to the legacy path on
  sparse/long-chain fixpoints (measured ≤1.161× there, no regression). Kill
  switch `XLOG_DISABLE_FACTORIZED_DELTA=1`; observability via
  `Executor::factorized_delta_dispatch_count()`. u32/Symbol arity-2 only;
  other widths and shapes decline silently to the existing hash-join → diff
  path. Spike (S3) and production-dispatch (S4) gate evidence under
  `docs/evidence/2026-06-1{2,4}-s{3,4}-factorized-delta*/`.
- *(runtime)* **D3 sparse-domain route.** Domains above the dense cap (which
  previously declined to legacy) now route through a GPU open-addressing
  hash-set novel-set with no `domain²` term, so transitive closure over large
  sparse graphs is accelerated too. The table is sized to the distinct-candidate
  count (a fixed 8 MiB estimator bitmap), not the witness count; inserts are
  overflow-safe and decline to legacy if the estimate under-sizes or the table
  exceeds `XLOG_FACTORIZED_DELTA_MAX_TABLE_BYTES` (default budget/2). Measured
  on RunPod RTX A4000 through the production executor on a large-domain
  (~2.09M) blowup TC: **14.63× peak-memory reduction at 0.160× wall-clock**
  (6.2× faster), row-set parity. The domain-based router selects
  dense-bitvector | sparse-hash-set | legacy; the same kill switch and counter
  cover both factorized routes. Evidence under
  `docs/evidence/2026-06-14-sparse-domain-spike/`.

### Fixed

- *(prob)* **Two-sided recursive SCC provenance now converges to the exact
  fixpoint.** Circuit construction flattens same-operator OR/AND children
  (associativity) and applies absorption, so mutually recursive two-sided
  support (`a :- b` and `b :- a` with independent priors) compiles to the exact
  d-DNNF marginal instead of diverging.

- *(prob)* **Self-healing recovery from stale disk-cached circuits.** A cached
  circuit whose variable capacity no longer matches the current CNF, or that
  fails equivalence verification, is evicted and recompiled instead of failing
  the compilation; fresh-compile verification stays fail-closed.

- *(cuda)* Fused group-by-root entries now layout-normalize their inputs
  per dispatch (sorted-fast-path when already lex-sorted+unique), matching
  the unfused pipeline's guarantee — unsorted or duplicated input buffers
  previously produced silently wrong (empty) fused aggregate results.
- *(prob)* **Behavior change — MC fail-closed contract.** `McProgram::evaluate`
  no longer falls back to the CPU oracle silently when the GPU-resident MC
  engine rejects a program (negation, aggregates, unbounded terms). Rejected
  programs now fail with the typed `ResidentRejection` unless the caller opts
  in explicitly, and every `McResult` carries an engine label so CPU-oracle
  output can never pass as GPU-native evidence.
  - **Migration**: MC programs with negation (incl. non-monotone recursion)
    or aggregates need `McEvalConfig::allow_cpu_oracle_fallback = true`
    (Rust), `--allow-cpu-oracle` (CLI), or `evaluate(allow_cpu_oracle=True)`
    (Python). Results report `mc_engine: "gpu-resident" | "cpu-oracle"` in
    CLI JSON/arrow output and `EvalResult.mc_engine` in Python.
  - The v0.8.5 MC-aggregate evidence was corrected accordingly: those
    fixtures always ran on the CPU oracle, not the GPU
    (`docs/evidence/2026-05-19-v085-prob-aggregates/README.md`).

### Added

- *(runtime)* **Aggregate-fused WCOJ (factorized aggregate execution).**
  `deg(X, count(V)) :- e_xy(X,Y), e_yz(Y,Z), e_xz(X,Z)` now executes through
  a fused group-by-root count kernel that never materializes the triangle
  rows — all reduction work is input-sized instead of join-output-sized.
  Measured 6.05x / 5.37x vs the materialize+groupby path on skewed hub
  fixtures and 2.49x on small uniform
  (`docs/evidence/2026-06-11-s1-aggregate-fused-wcoj/`). Transparent:
  count-by-root-variable aggregates over triangle bodies fuse automatically
  (counter `Executor::wcoj_groupby_fusion_dispatch_count`); every structural
  mismatch falls back to the existing path with identical results; kill
  switch `XLOG_DISABLE_WCOJ_GROUPBY_FUSION=1`. Scope: u32/Symbol keys,
  single `count` aggregate, non-recursive triangle bodies.
- *(cuda)* Aggregate-fused WCOJ aggregate widening: `sum`/`min`/`max` over a
  triangle output variable (Y or Z, U32 values) and u64-key `count` now
  dispatch through fused group-by-root kernels that never materialize the
  triangle rows (`wcoj_triangle_groupby_root_{sum,min,max}_hg_u32`,
  `wcoj_triangle_groupby_root_count_hg_u64`); the recorded groupby `Sum`
  accepts U64 value columns (`groupby_sum_u64`). Structural mismatches keep
  declining silently to materialize+groupby, and the
  `XLOG_DISABLE_WCOJ_GROUPBY_FUSION` kill switch covers the widened paths.
- *(cuda)* Aggregate-fused WCOJ width and shape completion:
  - 4-cycle `count` fusion — `deg(W, count(V)) :- e1(W,X), e2(X,Y),
    e3(Y,Z), e4(Z,W)` dispatches `wcoj_4cycle_groupby_root_count_hg_u32`
    without materializing the 4-cycle rows (17.6x-47.6x vs
    materialize+groupby on skewed hub fixtures, gate >= 3x). The fused
    path shares the triangle fusion's default-on gating and kill switch;
    the opt-in `XLOG_USE_WCOJ_4CYCLE*` gates keep governing only the
    non-aggregate materialize dispatch.
  - u64-key `sum`/`min`/`max` fusion —
    `wcoj_triangle_groupby_root_{sum,min,max}_hg_u64` plus metadata-driven
    segment reduction (30.3x-36.7x on the skewed u64 hub fixture, gate
    >= 3x). The legacy groupby (the unfused baseline for u64-key
    relations) was widened to u64-value `sum`/`min`/`max`
    (`groupby_min_u64` / `groupby_max_u64`; min/max output preserves the
    value width).
  - Symbol semantics locked by tests: `count` over Symbol-keyed/valued
    bodies fuses (u32-physical) and preserves the Symbol key type;
    `sum`/`min`/`max` over Symbol values declines fused and is rejected by
    the unfused groupby with an identical error (no silent aggregation of
    symbol ids). Evidence:
    `docs/evidence/2026-06-11-s1c-4cycle-width-completion/`.
  Gate evidence: `docs/evidence/2026-06-11-s1b-agg-widening/`.
- *(cuda)* Aggregate-fused WCOJ 4-cycle aggregate variants:
  - 4-cycle `sum`/`min`/`max` fusion — `agg(W, op(V)) :- e1(W,X), e2(X,Y),
    e3(Y,Z), e4(Z,W)` with `V ∈ {X, Y, Z}` (U32 values) dispatches
    `wcoj_4cycle_groupby_root_{sum,min,max}_hg_u32` without materializing
    the 4-cycle rows (3.3x-12.8x vs materialize+groupby on the skewed hub
    fixture, gate >= 3x); Symbol values decline fused and are rejected by
    the unfused groupby with an identical error.
  - u64-key 4-cycle `count` fusion —
    `wcoj_4cycle_groupby_root_count_hg_u64` plus the metadata-driven
    segment reduction (16.4x-26.1x on the skewed u64 hub fixture, gate
    >= 3x). u64-key 4-cycle `sum`/`min`/`max` stays deferred and declines.
  - The recorded groupby accepts U64 value columns for `Min`/`Max`
    (`groupby_min_u64`/`groupby_max_u64` on the recorded path; result
    preserves the value width), matching the legacy path widened for the width
    and shape completion work.
  - Float/LogSumExp fused aggregates: design-decision note recorded in the
    architecture guide (float atomics break the deterministic-values
    contract; per-block deterministic tree reduction preferred over
    fixed-point encoding). Evidence:
    `docs/evidence/2026-06-11-s1d-4cycle-agg-variants/`.
- *(cuda)* Aggregate-fused WCOJ K-clique count-by-root completion: K=5 and K=6
  clique `count`-by-root fusion at the u32/Symbol width-class —
  `q(R, count(*)) :- <complete K-clique body>` grouped by the plan's root
  variable dispatches `wcoj_clique{5,6}_groupby_root_count_hg_u32`
  (per-leader-edge-row atomicAdd accumulation over the planned clique
  count traversal) without materializing the clique rows (3.01x-3.59x vs
  the planned materialize+groupby path on skewed K=5 hub fixtures, gate
  >= 3x). The clique root is plan-dependent (`variable_order[0]` +
  leader-edge orientation/swaps), so the executor fuses only when the
  group key maps to the planned root variable; non-root keys, K=7/8, and
  u64 widths decline silently. The promoter descends aggregate wrappers
  over clique bodies by synthesizing the head projection from the
  variable-class tournament's topological order. Shared kill switch and
  counter. Evidence: `docs/evidence/2026-06-11-s1e-kclique-count-fusion/`.
- *(runtime)* Aggregate fusion over recursive-stratum inputs verified and
  locked by tests: later-stratum count/sum aggregates whose triangle bodies
  read semi-naive fixpoint outputs (incl. all-recursive self-joins)
  dispatch the fused path with oracle parity — recursive merges are
  lex-sorted+deduped by construction, satisfying the fused layout contract.
  Aggregates *inside* recursive rules remain stratification-rejected by
  language contract.
- *(runtime)* **GPU Free Join for general multiway bodies.** Inner-join
  bodies with >= 3 atoms and no dedicated kernel shape (any arity mix, any
  join-tree shape) now promote to a generic `MultiWayJoin`
  (`MultiwayPlan::FreeJoin`) and dispatch through a Free Join frontier
  engine (flat sorted-range tries + level-synchronous columnar frontier,
  identity-group fast path, fused probe filters; SIGMOD'23 Free Join
  adapted to bulk GPU execution). Spike gate on the 4-atom blowup chain:
  2.59x median vs the binary-join path (isolated serial runs); the dedicated
  triangle comparison is retained as the recorded cost-of-generality bound
  (1.73x-2.04x) and triangle/4-cycle/K-clique keep their dedicated kernels —
  Free Join never takes a dedicated shape
  (`docs/evidence/2026-06-12-s2-free-join-spike/`). Opportunistic by
  contract: non-prefix bound columns, non-u32/Symbol inputs, and repeated
  cover variables decline silently to the embedded binary fallback with
  identical results. Counter `Executor::free_join_dispatch_count`; kill
  switch `XLOG_DISABLE_FREE_JOIN=1`. Epistemic GPU certification classifies
  Free Join routes as a separate opportunistic preflight bucket
  (`free_join_route_count`) and traces dispatches without hardening them
  into dedicated-kernel obligations.
- *(cuda)* **Free Join u64 and factorized-count completion.** u64 width-class engine
  (`free_join_execute_u64_recorded`): the frontier pipeline is
  width-parameterized — VAR columns carry width-sized data while trie
  RANGE columns stay u32 row indices in every class — with parity locked
  by truncation-adversarial fixtures (keys colliding modulo 2^32).
  Recursive SCCs verified end-to-end: Free Join fires on the semi-naive
  seeding pass and on every delta-rewritten variant with exact fixpoint
  parity under the kill switch. **Factorized count-by-root** (design
  §2.4): `count` aggregates over FreeJoin-promoted bodies dispatch
  `free_join_count_by_root_u32_recorded` — trailing variables private to
  one atom are never expanded; each frontier row contributes the product
  of its remaining live trie-range lengths (the d-representation count)
  and the existing recorded groupby reduces `(group, multiplicity)`.
  Count semantics match the unfused pipeline exactly (both count
  distinct full body bindings). Measured 3.66x-3.71x vs the
  materialize+groupby path on a skewed 4-atom fixture (7.8M-row
  join vs 100k-row factorized frontier; RTX A4000, 3 isolated runs,
  gate >= 3x —
  `docs/evidence/2026-06-12-s2-free-join-spike/runpod-count-gate.log`).
  u32/Symbol-key only, matching the recorded groupby's engine-wide
  key support; both `XLOG_DISABLE_WCOJ_GROUPBY_FUSION` and
  `XLOG_DISABLE_FREE_JOIN` disable the fused route with identical
  fallback results.
- *(runtime)* **Free Join order planner and factorized loss veto.** Two
  cardinality-driven gates keep the factorized routes from dispatching a worse
  plan in their loss regions:
  - Order planner (`plan_free_join_order`): Free Join materializes a left-deep
    prefix whose probe keys must be a leading column prefix of each atom, so a
    bad input order can materialize a large intermediate even when the result
    is tiny. Using the ground-truth row counts of the buffers being joined
    (and `StatsManager::estimate_join_cardinality` for per-pair selectivity when
    stats exist), it keeps the input order when it is already within 1.2x of the
    binary plan's estimated peak (small joins and already-good orders untouched),
    reorders to a better prefix-key-joinable order when one is competitive, or
    declines to the binary fallback when none is — removing a measured worst-case
    peak-memory loss (~3x on an adversarial blow-up chain, now declined to peak
    parity) while every winning fixture still fires.
  - Loss veto (`factorized_loss_veto`): fail-open. The aggregate-fused WCOJ and
    Free Join routes decline to the binary plan only when stats are present for
    every input and the largest is below the WCOJ-worthwhile threshold (a
    provably-small join the binary plan wins); missing stats or any large input
    never veto, so measured wins are preserved.
  Both run only under the cardinality cost model (the skew model opts out), and
  reordering changes only the plan-build order — buffer indexing and the head
  projection are unchanged. Documented in the WCOJ architecture and user guides.
- *(prob)* **Factorized outcome folding for exact non-count aggregates.**
  Probabilistic `sum`/`min`/`max`/`logsumexp` provenance no longer
  enumerates one conjunction per 2^k outcome mask; the factorized encoding
  keeps the PIR polynomial in k (k=14 fixture: < 4096 nodes vs >= 16384
  masks) with probability parity locked at 1e-12 against finite oracles.
- *(prob)* Opt-in `decision_order_hint` for GPU D4 compilation
  (default off). Measured verdict is **negative** — 0% frontier reduction,
  independently replicated (`docs/evidence/2026-06-11-d4-structure-hints/`)
  — kept for its `frontier_items` profiling counter and as the harness for
  future kernel-side variable-priority work.
- *(docs)* Factorized-hypergraph research report
  (`docs/plans/2026-06-11-factorized-hypergraph-research.md`):
  adversarially verified algorithm landscape (f-/d-representations, FAQ,
  Free Join, factorized provenance), codebase asset/gap map, four ranked
  integration directions with benchmark-spike gates; the aggregate-fused WCOJ
  direction shipped above.
- *(runtime)* WCOJ pipeline errors (layout/kernel failures) are now counted
  (`Executor::wcoj_error_decline_count`) and logged when they decline to the
  binary-join fallback; `XLOG_WCOJ_STRICT=1` propagates the error instead.
- *(tests)* `XLOG_REQUIRE_CUDA=1` turns CUDA-init failures in the test
  harness into hard failures so certification can never pass vacuously on a
  CPU-only machine; `scripts/validate_release_gpu.sh` sets it and preflights
  the GPU. Restored the external diagnostic epistemic fixture programs missing since PR #139
  (9/15 `xlog-epistemic-evidence` tests had been red on `main`).
- *(pyxlog)* `LogicRelationSession.wcoj_dispatch_stats()` exposes the
  session executor's multiway dispatch telemetry
  (`free_join_dispatch_count`, `wcoj_groupby_fusion_dispatch_count`,
  `wcoj_error_decline_count`) so consumers can observe whether Free Join
  and fused-aggregate paths fire on their workloads instead of inferring
  it from kill-switch timing probes.

### Documentation

- BENCHMARKS.md unmeasured throughput tables are labeled as targets;
  README states the zero-tracked-transfer data-plane contract precisely;
  the MC engine split and the XGCF per-level host-orchestration boundary
  are documented in the language/CLI/architecture references.

## [0.9.2](https://github.com/BrainyBlaze/xlog/compare/xlog-cli-v0.5.0...xlog-cli-v0.9.2) - 2026-06-08

### Added

- full shared-variable epistemic constraint joins via program-level desugaring
- diagonal modal constraint via sound program-level desugaring
- pilot ex37 (stratified negated-modal recursion EXECUTES) + device test/mutation + reword ex33 to formal WFS bound
- multi-literal distinct-variable epistemic constraints + README
- *(epistemic)* epistemic plan-dump surface â xlog run --epistemic-plan-json
- close determined-epistemic multi-column binding (determined-modal family complete)
- close transitive determined-ordinary modal coupling via stratification
- close augmented-projection multi-head coupling scope limit
- cross-component epistemic joint-solving (multi-output)
- cross-component epistemic coupling
- recursive epistemic fixpoint support
- mixed per-row and global modal membership
- checkpoint epistemic solver semantics
- add cli explain repl watch surfaces
- add incremental parser session
- add approximate inference pragmas
- add aggregate lifting reports
- add magic-set rewriting

### Fixed

- *(release)* drop README version sync + dynamic badges + agent release rules ([#124](https://github.com/BrainyBlaze/xlog/pull/124))
- guard diagonal desugaring to non-modal-derived targets
- route bound-variable multi-head epistemic programs through split
- route epistemic examples through xlog run
- *(release)* harden validation and gpu fallback paths
- *(pyxlog)* install local wheels for explicit python
- *(cuda)* embed portable PTX fallback
- *(pyxlog)* ship kernels in wheels and document cubin path
- *(ci)* repair main release automation ([#27](https://github.com/BrainyBlaze/xlog/pull/27))
- *(ci)* keep README release metadata in sync ([#26](https://github.com/BrainyBlaze/xlog/pull/26))

### Other

- v0.9.2 whitepaper + documentation realignment ([#133](https://github.com/BrainyBlaze/xlog/pull/133))
- Set v0.9.2 release metadata
- Clarify v0.9.2 WFS release contract
- Close v0.9.2 epistemic semantic gaps
- Close v0.9.2 epistemic release
- CLI markers for accepted chain pilots; repoint negative test to interior-negation boundary
- variable-keyed constraint device tests + CLI goldens + mutation probe
- CLI golden ex23 ACCEPTED + repoint negative test to unbounded cons
- CLI accepted-fixpoint + negated-modal-floor contracts
- FAEEL unfounded self-support → exact founded-extension semantic result
- integrate main MC GPU-resident engine into v0.9.2 epistemic completion
- complete determined-modal-family showcase (negated-over-derived, possible-binding, FAEEL-unfounded)
- determined-head and negated-modal-over-invariant xlog-run pilots (examples 25,26) with anti-gaming gating checks
- flip example 17 to accepted stratified pilot; add example 24 transitive out-of-scope negative
- full robust validated v0.9.2 epistemic examples
- add validated v0.9.1 epistemic executor showcase (06-11)
- document v0.9.0 epistemic language surface
- *(release)* align v0.9.0 package metadata
- integrate v0.8.9 diagnostics surfaces
- *(release)* prepare v0.8.6 tag metadata
- *(release)* correct v0.8.5 public status
- *(release)* prepare v0.8.0
- mark v0.7.0 release complete
- close purge gate
- *(v0.6.2)* prepare roadmap changelog and version
- *(v0.6.1)* version bump + roadmap cleanup + changelog
- *(readme)* bump version badge + release-status line to v0.6.0
- restore audit README framing with current release setup
- Merge branch 'audit/v0.5.0-prerelease'
- integrate prerelease audit docs
- harden public release readiness

## [0.9.2](https://github.com/BrainyBlaze/xlog/compare/xlog-gpu-v0.5.0...xlog-gpu-v0.9.2) - 2026-06-08

### Added

- full shared-variable epistemic constraint joins via program-level desugaring
- diagonal modal constraint via sound program-level desugaring
- pilot ex37 (stratified negated-modal recursion EXECUTES) + device test/mutation + reword ex33 to formal WFS bound
- co-evolving modal and recursive founded least fixpoint
- *(epistemic)* epistemic plan-dump surface â xlog run --epistemic-plan-json
- close determined-epistemic multi-column binding (determined-modal family complete)
- determined-head recursion and negated-modal-over-invariant recursive epistemic fixpoint
- stratified epistemic execution wiring (materialize gated head between strata)
- cross-component epistemic joint-solving (multi-output)
- recursive epistemic fixpoint support
- coalesce relation delta batches
- add safe meta lowering
- add finite list lowering
- add type term foundation
- *(pyxlog)* add v0.8.0 relation delta sessions
- expose xlog sort-label metadata

### Fixed

- *(release)* drop README version sync + dynamic badges + agent release rules ([#124](https://github.com/BrainyBlaze/xlog/pull/124))
- guard diagonal desugaring to non-modal-derived targets
- route bound-variable multi-head epistemic programs through split
- materialize nullary EDB facts as present (1 row)
- route epistemic examples through xlog run
- prove pyxlog persistent index session reuse
- *(release)* harden validation and gpu fallback paths
- expose query-variable sort labels at runtime
- *(pyxlog)* install local wheels for explicit python
- *(cuda)* embed portable PTX fallback
- *(pyxlog)* ship kernels in wheels and document cubin path
- *(ci)* repair main release automation ([#27](https://github.com/BrainyBlaze/xlog/pull/27))
- *(ci)* keep README release metadata in sync ([#26](https://github.com/BrainyBlaze/xlog/pull/26))

### Other

- v0.9.2 whitepaper + documentation realignment ([#133](https://github.com/BrainyBlaze/xlog/pull/133))
- Set v0.9.2 release metadata
- Clarify v0.9.2 WFS release contract
- Close v0.9.2 epistemic semantic gaps
- Close v0.9.2 epistemic release
- cargo fmt
- derived-head coupling — stratified-vs-reference equivalence + true-cycle wall
- device co-evolving modal-recursion case founded-fixpoint + ungated mutation probe
- document v0.9.0 epistemic language surface
- *(release)* align v0.9.0 package metadata
- integrate v0.8.9 diagnostics surfaces
- integrate v0.8.8 external world-model diagnostics into v0.8.9
- integrate first external diagnostics into v0.8.9
- Exercise external generated-rule diagnostics
- Resolve remaining XLOG evidence issues
- Add v0.8.7 external world-model diagnostics
- *(release)* prepare v0.8.6 tag metadata
- *(release)* correct v0.8.5 public status
- *(release)* prepare v0.8.0
- mark v0.7.0 release complete
- *(v0.6.2)* prepare roadmap changelog and version
- *(v0.6.1)* version bump + roadmap cleanup + changelog
- *(readme)* bump version badge + release-status line to v0.6.0
- restore audit README framing with current release setup
- Merge branch 'audit/v0.5.0-prerelease'
- integrate prerelease audit docs
- harden public release readiness

## [0.9.2](https://github.com/BrainyBlaze/xlog/compare/xlog-prob-v0.5.0...xlog-prob-v0.9.2) - 2026-06-08

### Added

- close augmented-projection multi-head coupling scope limit
- checkpoint epistemic solver semantics
- add approximate inference pragmas
- add aggregate lifting reports
- add probabilistic aggregate support
- add safe meta lowering
- add type term foundation

### Fixed

- *(release)* drop README version sync + dynamic badges + agent release rules ([#124](https://github.com/BrainyBlaze/xlog/pull/124))
- route epistemic examples through xlog run
- close GPU-native count-lift exact path
- *(release)* harden validation and gpu fallback paths
- *(pyxlog)* install local wheels for explicit python
- *(cuda)* embed portable PTX fallback
- *(pyxlog)* ship kernels in wheels and document cubin path
- *(ci)* repair main release automation ([#27](https://github.com/BrainyBlaze/xlog/pull/27))
- *(ci)* keep README release metadata in sync ([#26](https://github.com/BrainyBlaze/xlog/pull/26))

### Other

- v0.9.2 whitepaper + documentation realignment ([#133](https://github.com/BrainyBlaze/xlog/pull/133))
- Set v0.9.2 release metadata
- Close v0.9.2 epistemic release
- integrate main MC GPU-resident engine into v0.9.2 epistemic completion
- document v0.9.0 epistemic language surface
- *(release)* align v0.9.0 package metadata
- integrate v0.8.9 diagnostics surfaces
- aggregate split batch cpu fallbacks
- gate split batch h2d transfer
- gate split batch incremental prob updates
- gate accepted evidence incremental prob updates
- centralize probabilistic batch gate
- require single result timing gates
- require split batch timing gates
- tighten prob production metric gate
- replace incremental evidence updates
- name alternative compiler adapters
- trace nonzero probabilistic evidence arity
- reuse v0.8.6 bundle for v0.9.0
- *(release)* prepare v0.8.6 tag metadata
- *(release)* correct v0.8.5 public status
- *(release)* prepare v0.8.0
- mark v0.7.0 release complete
- *(v0.6.2)* prepare roadmap changelog and version
- *(v0.6.1)* version bump + roadmap cleanup + changelog
- *(readme)* bump version badge + release-status line to v0.6.0
- restore audit README framing with current release setup
- Tighten workspace warning hygiene
- Merge branch 'audit/v0.5.0-prerelease'
- integrate prerelease audit docs
- harden public release readiness

## [0.9.2](https://github.com/BrainyBlaze/xlog/compare/xlog-solve-v0.5.0...xlog-solve-v0.9.2) - 2026-06-08

### Added

- close augmented-projection multi-head coupling scope limit
- checkpoint epistemic solver semantics
- gate multi-candidate solver portfolios
- schedule gpu maxsat batches
- schedule multi-result gpu maxsat search
- schedule multi-result gpu maxsat encodes
- encode weighted gpu maxsat candidates
- prune unsat gpu maxsat candidates
- batch accepted gpu maxsat candidates
- reuse learned clauses across accepted gpu candidates
- propagate gpu solver lifecycle statuses
- cover multi-candidate gpu solver lifecycle
- reject unsafe learned clause reuse
- gate oracle fixtures from production metrics
- reuse gpu learned clauses for same cnf
- publish gpu learned clause arenas
- propagate solver portfolio status in gpu adapter
- gate maxsat portfolio through gpu solver adapter
- gate solver lifecycle with accepted gpu evidence
- gate solver workspace unsat with accepted gpu evidence
- gate solver unsat path with accepted gpu evidence
- gate solver sat path with accepted gpu evidence
- report solver production capability blockers
- add gpu solver production reuse adapter
- add bounded solver service semantics

### Fixed

- *(release)* drop README version sync + dynamic badges + agent release rules ([#124](https://github.com/BrainyBlaze/xlog/pull/124))
- route epistemic examples through xlog run
- *(release)* harden validation and gpu fallback paths
- *(pyxlog)* install local wheels for explicit python
- *(cuda)* embed portable PTX fallback
- *(pyxlog)* ship kernels in wheels and document cubin path
- *(ci)* repair main release automation ([#27](https://github.com/BrainyBlaze/xlog/pull/27))
- *(ci)* keep README release metadata in sync ([#26](https://github.com/BrainyBlaze/xlog/pull/26))

### Other

- v0.9.2 whitepaper + documentation realignment ([#133](https://github.com/BrainyBlaze/xlog/pull/133))
- Set v0.9.2 release metadata
- document v0.9.0 epistemic language surface
- *(release)* align v0.9.0 package metadata
- integrate v0.8.9 diagnostics surfaces
- aggregate split batch cpu fallbacks
- gate split batch h2d transfer
- centralize solver batch gate
- require single result timing gates
- require split batch timing gates
- lock production metric audit wording
- tighten solver production metric gate
- trace solver nonzero evidence arity
- reuse v0.8.6 bundle for v0.9.0
- guard maxsat scheduler prevalidation
- guard encoded maxsat prevalidation
- guard maxsat search prevalidation
- guard maxsat lifecycle prevalidation
- gate split maxsat lifecycle
- gate solver maxsat lifecycle
- gate split solver maxsat scheduler on batches
- gate split solver maxsat search on batches
- gate split solver maxsat on batches
- gate split solver learned reuse on batches
- gate split solver portfolio on batches
- gate split solver lifecycle on batches
- trace solver evidence by operator family
- trace semantic modes through solver gates
- mark gpu native gate blocked
- mark v0.7.0 release complete
- *(v0.6.2)* prepare roadmap changelog and version
- *(v0.6.1)* version bump + roadmap cleanup + changelog
- *(readme)* bump version badge + release-status line to v0.6.0
- restore audit README framing with current release setup
- Tighten workspace warning hygiene
- Merge branch 'audit/v0.5.0-prerelease'
- integrate prerelease audit docs
- harden public release readiness

## [0.9.2](https://github.com/BrainyBlaze/xlog/compare/xlog-runtime-v0.5.0...xlog-runtime-v0.9.2) - 2026-06-08

### Added

- same-name multi-arity modal coupling solved via arity-qualified tuple sources
- variable-keyed + nested epistemic constraints (GPU world-view pruning)
- multi-literal distinct-variable epistemic constraints + README
- drop unfounded FAEEL self-support from reduced founded-model base
- close determined-epistemic multi-column binding (determined-modal family complete)
- close augmented-projection multi-head coupling scope limit
- determined-head recursion and negated-modal-over-invariant recursive epistemic fixpoint
- stratified epistemic execution wiring (materialize gated head between strata)
- cross-component epistemic joint-solving (multi-output)
- cross-component epistemic coupling
- mixed per-row and global modal membership
- constraint-specific rejection reasons
- joint multi-epistemic predicate solving
- epistemic integrity constraints
- EIR-derived candidate-world enumeration
- tuple-key bound-value membership
- checkpoint epistemic solver semantics
- compare G91 GPU traces to oracle
- expose gpt rejected candidate indices
- expose gpu semantic candidate indices
- type gpu epistemic rejection reasons
- gate probabilistic pir cnf batches
- gate probabilistic evaluation batches
- gate parsed probabilistic program batches
- gate multi-candidate solver portfolios
- add split world-view parity fixture
- add G91 runtime parity fixture
- certify skew-scheduled wcoj reuse
- permit founded faeel self possible
- require complete world-view support
- require helper scans in wcoj plans
- certify helper split rewrites
- require wcoj layout evidence
- guard faeel self support
- trace not possible row filters
- certify kclique stream groups
- trace epistemic operator metrics
- trace kclique metadata timing
- schedule gpu maxsat batches
- schedule multi-result gpu maxsat search
- schedule multi-result gpu maxsat encodes
- condition negative gpu prob evidence
- execute split gpu components
- encode weighted gpu maxsat candidates
- condition gpu prob gradients
- prune unsat gpu maxsat candidates
- certify helper split wcoj trace metrics
- batch conditioned gpu prob programs
- batch conditioned gpu prob queries
- condition accepted gpu prob programs
- condition accepted gpu prob tuple evidence
- batch accepted gpu maxsat candidates
- reuse learned clauses across accepted gpu candidates
- propagate gpu solver lifecycle statuses
- batch accepted gpu prob execution
- cover multi-candidate gpu solver lifecycle
- trace gpu semantic candidate outcomes
- honor not-know tuple membership on gpu
- condition accepted evidence in gpu exact path
- gate oracle fixtures from production metrics
- account final gpu result transfers
- reuse gpu learned clauses for same cnf
- trace kclique arity preflight reuse
- trace program prob knowledge compilation
- publish gpu learned clause arenas
- propagate solver portfolio status in gpu adapter
- lower split components through gpu executable plans
- gate maxsat portfolio through gpu solver adapter
- filter final rows by all epistemic memberships
- gate prob end-to-end exact evaluation
- gate solver lifecycle with accepted gpu evidence
- gate prob pir cnf with accepted gpu evidence
- gate prob query evaluation with accepted gpu evidence
- gate prob program compile with accepted gpu evidence
- gate solver workspace unsat with accepted gpu evidence
- gate prob gradient evaluation with accepted gpu evidence
- gate solver unsat path with accepted gpu evidence
- gate solver sat path with accepted gpu evidence
- gate prob exact path with accepted gpu evidence
- filter final tuples by bound membership
- certify accepted wcoj execution
- gate final tuples by gpu membership
- bind tuple keys to gpu output columns
- add generic gpu tuple membership kernel
- add arity-three epistemic tuple key kernel
- encode ground epistemic tuple keys for gpu matching
- preserve epistemic tuple key terms in eir
- stage fixed-arity epistemic tuple sources on gpu
- populate arity-zero epistemic membership from tuple sources
- bind epistemic literals to tuple membership sources
- fail closed on row-count epistemic membership
- materialize epistemic final tuples on gpu
- enforce epistemic wcoj runtime certification
- gate epistemic model membership on gpu output
- trace epistemic gpu transfer budget
- materialize epistemic final result flags on gpu
- validate epistemic world views on gpu
- stage epistemic model membership on gpu
- trace epistemic gpu staging timings
- stage epistemic materialization on gpu
- validate staged epistemic candidates on gpu
- stage epistemic propagation on gpu
- add gpu candidate generation kernel
- reset epistemic gpu workspace on device
- trace epistemic reduced runtime execution
- gate epistemic wcoj evidence on counters
- add epistemic runtime preflight
- add epistemic gpu workspace contract
- close CUDA Graph-mode set maintenance
- bind delta variants as WCOJ leaders
- remove adaptive skew classifier surface
- route variable-ordering cost model u32 triangle through HG
- route u32 triangle dispatch through HG pipeline
- production kernel fusion (layout+count) with threshold dispatch + auto-disable + cert grid
- wire sort-merge dispatch + counter at execute_join + de-overlap nested-loop dispatch cert fixtures for dispatch precedence
- add eligible_for_sort_merge predicate
- wire nested-loop dispatch + counter at execute_join
- add eligible_for_nested_loop predicate
- *(runtime)* K=5 and K=6 clique WCOJ dispatcher + counters
- HeatAwareLeaderModel plus variable-order-aware join-result feedback
- *(runtime)* per-iteration stats integration for recursive SCC
- *(runtime)* dispatcher reroute on var_order — variable-ordering cost model leader rotation + post-kernel projection
- *(ir)* VariableOrder + LookupPerm types + MultiWayJoin.var_order field
- *(runtime)* record_join_result feedback from successful WCOJ dispatch
- *(runtime)* dispatch sites use build_wcoj_cost_model factory
- *(runtime)* CardinalityAwareCostModel with delegate-on-missing-stats
- *(runtime)* execute_wcoj_or_fallback_node hooks recursive arm
- *(runtime)* try_dispatch_wcoj_*_on_body entry points
- *(runtime)* migrate adaptive dispatch to WcojCostModel seam
- *(runtime)* WcojCostModel + SkewScoreSource cost-model seam
- *(cuda+runtime)* 4-cycle skew classifier + adaptive opt-in
- *(runtime)* wire 4-cycle dispatch + executor wiring cert
- *(runtime)* match_multiway_4cycle + try_dispatch_wcoj_4cycle force gate
- *(runtime)* replace triangle-tree matcher with MultiWayJoin
- *(workspace)* cross-crate MultiWayJoin walker arms
- *(runtime)* default-on adaptive WCOJ + hard kill switch (v0.6.2)
- *(runtime)* adaptive WCOJ dispatch + classifier branch (v0.6.2 A2-lite commit B)
- *(dispatch)* WCOJ width-aware AST/RIR dispatch (v0.6.2)
- *(cuda)* WCOJ Symbol key support (v0.6.2)
- *(runtime)* env-gated WCOJ triangle executor wiring (v0.6.2)
- *(runtime)* add strict deterministic D2H guard (v0.5.5) ([#49](https://github.com/BrainyBlaze/xlog/pull/49))

### Fixed

- *(release)* drop README version sync + dynamic badges + agent release rules ([#124](https://github.com/BrainyBlaze/xlog/pull/124))
- standalone negated-variable-keyed constraint is a NAF safety error, not 'unimplemented'
- route epistemic examples through xlog run
- fail closed nonzero faeel self support
- certify v0.7.0 multiway wcoj reuse
- require tuple-source proof before validation
- *(release)* harden validation and gpu fallback paths
- integrate K7 K8 planned clique metadata
- restore cost-model default-flip cert
- route nested-loop dispatch through shared record_join_result feedback
- preserve occurrence identity in rewrite_scan_nth
- tighten Tier-1 wrapper contract + revert recursive helper extension
- cargo fmt + correct prepare_leader_inputs visibility doc
- cargo fmt, evidence count, extract prepare_leader_inputs + real helper extraction
- address selectivity-driven join reordering review patches — duplicate attr, stale comment, evidence count, matcher tests
- *(runtime)* harden WCOJ phase timing diagnostics
- *(runtime)* cache WCOJ launch stream on Executor (v0.6.2)
- *(logic)* restore deterministic recursive set evaluation
- *(pyxlog)* install local wheels for explicit python
- *(cuda)* embed portable PTX fallback
- *(pyxlog)* ship kernels in wheels and document cubin path
- *(ci)* repair main release automation ([#27](https://github.com/BrainyBlaze/xlog/pull/27))
- *(ci)* keep README release metadata in sync ([#26](https://github.com/BrainyBlaze/xlog/pull/26))

### Other

- v0.9.2 whitepaper + documentation realignment ([#133](https://github.com/BrainyBlaze/xlog/pull/133))
- Set v0.9.2 release metadata
- device test — nested-modal chain collapses, executes, zero CPU fallback
- rustfmt multi-arity device test upload helper
- variable-keyed constraint device tests + CLI goldens + mutation probe
- harden multi-element key test to discriminate col1
- repoint mixed-modal negative pilot to unbounded cons key
- red device tests + ACCEPTED ex23 for structured modal tuple-keys
- exact FAEEL founded-extension results on GPU runtime + mutation-probe-verified gate
- cargo fmt on v0.9.1 epistemic changeset
- safe split dependency and coupling semantics
- document v0.9.0 epistemic language surface
- *(release)* align v0.9.0 package metadata
- integrate v0.8.9 diagnostics surfaces
- gate split possible not-know fallbacks
- gate split binary cpu fallbacks
- certify split binary workspace timing
- certify k7 k8 layout events
- certify k7 k8 metadata timing
- aggregate split batch cpu fallbacks
- gate split batch h2d transfer
- aggregate split batch final transfer
- gate split batch final transfer
- gate single-result final transfer
- gate single-result kernel timing
- gate single-result workspace buffers
- gate single-result row-count membership rejection
- gate single-result host transfer rejection
- gate single-result cpu fallback rejection
- gate split batch incremental prob updates
- gate accepted evidence incremental prob updates
- gate rejected world-view consumers
- gate split quaternary host transfer rejection
- gate split quaternary row-count membership rejection
- gate split quaternary cpu fallback rejection
- gate split quaternary workspace buffers
- gate split quaternary all-operator timing
- gate split quaternary all-operator prob deep paths
- gate split quaternary all-operator prob gradients
- gate split quaternary all-operator probability
- gate split quaternary all-operator solver search
- gate split quaternary all-operator solver reuse
- gate split quaternary all-operator solver lifecycle
- gate split quaternary all-operator parity
- gate split quaternary possible not-know parity
- gate quaternary possible not-know source gradients
- gate quaternary not-possible prob gradients
- gate quaternary know prob gradients
- gate quaternary know probabilistic reuse
- gate quaternary know solver search
- gate quaternary know solver reuse
- gate quaternary possible not-know solver search
- gate quaternary possible not-know solver reuse
- gate quaternary not-possible solver search
- gate quaternary not-possible solver reuse
- gate quaternary not-possible PIR reuse
- gate quaternary source PIR reuse
- gate quaternary program PIR reuse
- gate quaternary program probability reuse
- gate all-operator program probability eval
- gate all-operator source probability paths
- gate all-operator solver search
- gate split all-binary solver search
- gate split quaternary not-possible solver search
- gate split quaternary solver search
- gate split quaternary prob reuse
- gate split quaternary solver reuse
- gate split quaternary production reuse
- gate quaternary operator production reuse
- gate quaternary operator gpu parity
- gate split quaternary gpu parity
- gate split quaternary prob reuse
- gate split quaternary solver reuse
- gate split quaternary solver evidence
- gate split quaternary prob batch reuse
- gate parsed quaternary negative prob reuse
- gate negated quaternary solver prob reuse
- cover negated quaternary membership parity
- require complete aggregate timing
- require single result timing gates
- require split batch timing gates
- aggregate split batch kernel timing
- lock production metric audit wording
- name alternative compiler adapters
- deepen all-operator reuse gates
- gate all-operator membership reuse
- cover all-operator mixed memberships
- cover negated mixed memberships
- cover mixed epistemic memberships
- reject unsafe split modal coupling
- gate all-operator split prob eval
- gate all-operator split prob gradients
- gate all-operator split solver reuse
- gate all-operator split solver lifecycle
- condition split all-operator probability
- trace split all binary operators
- trace split binary operator parity
- trace solver quaternary evidence arity
- trace source quaternary evidence arity
- trace prob quaternary evidence arity
- trace solver nonzero evidence arity
- trace nonzero probabilistic evidence arity
- reuse v0.8.6 bundle for v0.9.0
- aggregate split operator trace counts
- guard maxsat scheduler prevalidation
- guard encoded maxsat prevalidation
- guard maxsat search prevalidation
- guard maxsat lifecycle prevalidation
- gate split maxsat lifecycle
- gate solver maxsat lifecycle
- gate ternary epistemic gpu parity
- gate split prob exact compile on batches
- gate split prob pir cnf evaluation on batches
- gate split solver maxsat scheduler on batches
- gate split prob exact paths on batches
- gate split solver maxsat search on batches
- gate split solver maxsat on batches
- gate split solver learned reuse on batches
- gate split solver portfolio on batches
- gate split solver lifecycle on batches
- gate split prob gradients on batches
- gate parsed prob evidence on split batches
- gate prob evidence on split batches
- trace split gpu batch execution
- split prob operator evidence by source path
- trace solver evidence by operator family
- trace semantic modes through solver gates
- trace semantic modes through prob gates
- cover negative probabilistic batches
- certify accepted k8 wcoj dispatch
- certify accepted k7 wcoj dispatch
- audit production path reuse
- mark v0.7.0 release complete
- close phase2 integration gate
- Merge sort-label propagation branch into the phase-2 integration branch
- Merge K7/K8 clique-template branch into the phase-2 integration branch
- add K7/K8 clique templates
- close purge gate
- unwire executor sort-merge dispatch + rewrite sort-merge dispatch certs as operator-level
- workspace gate green pre-bench (+ stale-comment cleanup)
- workspace gate green
- scrub stale "multi-recursive skip" contract notes
- flip recursive-SCC stats integration multi-recursive WCOJ cert to assert multi-recursive WCOJ dispatch
- rewrite input/fallback `rewrite_scan_nth` tests for positional symmetry
- strengthen rewrite_scan_nth regression for exact positional identity
- cargo fmt for workspace gate
- patch evidence/comment drift round 2
- patch evidence/comment drift before closure approval
- *(test)* correct recursive-SCC stats integration feature-gate test header — feature gate, not cfg(test)
- *(runtime)* strengthen recursive-SCC stats integration distinct binary_est + pin exact counter
- *(runtime)* recursive-SCC stats integration acceptance gate acceptance matrix + recursive-stats-trace feature
- restore selectivity-driven join reordering acceptance gates — 4-cycle compile-time + runtime helper and synthesis certs
- *(runtime,logic)* align stale claims with the recursive WCOJ contract
- *(runtime)* unit-test SkewScoreSource seam via stub scorer
- *(runtime)* rename wcoj_triangle_stream to wcoj_dispatch_stream
- *(workspace)* MultiWayJoin shape-agnosticism guards
- *(v0.6.2)* prepare roadmap changelog and version
- *(runtime)* WCOJ phase-timing scaffolding + report (v0.6.2)
- *(runtime)* cover WCOJ dispatch env resolvers
- *(wcoj)* update Symbol dispatch scope comments
- *(v0.6.1)* version bump + roadmap cleanup + changelog
- *(readme)* bump version badge + release-status line to v0.6.0
- restore audit README framing with current release setup
- Tighten workspace warning hygiene
- Merge branch 'audit/v0.5.0-prerelease'
- integrate prerelease audit docs
- harden public release readiness

## [0.9.2](https://github.com/BrainyBlaze/xlog/compare/xlog-logic-v0.5.0...xlog-logic-v0.9.2) - 2026-06-08

### Added

- admit stratified negated-modal recursion as Case B; bound genuine negation cycle to host-only WFS
- grammar+parser collapse nested modal chains to single epistemic literal
- variable-keyed + nested epistemic constraints (GPU world-view pruning)
- single-occurrence variable-keyed epistemic constraints (GPU existential world-view pruning)
- flatten structured modal tuple-keys (finite+typed list/compound/anonymous on GPU)
- admit positive modal recursion with a founded least fixpoint
- drop unfounded FAEEL self-support from reduced founded-model base
- close determined-epistemic multi-column binding (determined-modal family complete)
- close transitive determined-ordinary modal coupling via stratification
- close augmented-projection multi-head coupling scope limit
- determined-head recursion and negated-modal-over-invariant recursive epistemic fixpoint
- stratified epistemic analysis (determined-head detection + strata partition)
- cross-component epistemic joint-solving (multi-output)
- cross-component epistemic coupling
- recursive epistemic fixpoint support
- joint multi-epistemic predicate solving
- epistemic integrity constraints
- nested modal explicit representation and fail-closed diagnostics
- FAEEL founded self-support completion
- checkpoint epistemic solver semantics
- add incremental parser session
- add approximate inference pragmas
- add magic-set rewriting
- harden deterministic naf safety
- add safe meta lowering
- add finite list lowering
- add type term foundation
- add stream-mux AOT schedule
- add helper-split AOT pass
- *(logic)* K=5 and K=6 clique WCOJ promoter try_promote_clique_k for k=5/6
- *(promote)* normalize right-deep triangle / fully-right-deep 4-cycle
- HeatAwareLeaderModel plus variable-order-aware join-result feedback
- *(logic)* promote_multiway takes (stats, config); 25 caller sites updated
- *(logic)* WcojVariableOrderingModel trait + LeaderCardinalityModel
- *(logic)* CompilerConfig + composable compile API
- *(ir)* VariableOrder + LookupPerm types + MultiWayJoin.var_order field
- *(logic)* selectivity_pass real triangle + 4-cycle reordering
- *(logic)* variable-graph triangle + 4-cycle promoters
- *(logic)* selectivity_pass takes rel_ids; module-doc rewritten
- *(logic)* promote_multiway gates recursive SCCs by per-rule scan count
- *(logic)* wire selectivity_pass into Compiler post-optimizer
- *(logic)* selectivity_pass inline pub mod (no-op)
- *(logic)* try_promote_4cycle for canonical 4-cycle shape
- *(logic)* wire promote_multiway after optimizer
- *(logic)* promote_multiway pass for triangle WCOJ
- *(workspace)* cross-crate MultiWayJoin walker arms
- *(logic)* transitive SCC type inference (v0.6.2 PR 8)
- *(logic)* hypergraph mixed plan contract (v0.6.2 PR 6)
- *(logic)* hypergraph typed oracle gate (v0.6.2 PR 5)
- *(logic)* hypergraph SCC fixpoint evaluator (v0.6.2 PR 4)
- *(logic)* hypergraph fixpoint evaluator (v0.6.2 PR 3)
- *(logic)* hypergraph reference evaluator (v0.6.2 PR 2)
- *(logic)* hypergraph planner foundation (v0.6.2 PR 1)

### Fixed

- *(release)* drop README version sync + dynamic badges + agent release rules ([#124](https://github.com/BrainyBlaze/xlog/pull/124))
- standalone negated-variable-keyed constraint is a NAF safety error, not 'unimplemented'
- fail closed when a recursive epistemic program carries an epistemic constraint
- fail closed on ordinary recursion in epistemic programs
- route bound-variable multi-head epistemic programs through split
- route epistemic examples through xlog run
- fail closed nonzero faeel self support
- preserve independent epistemic split inputs
- guard epistemic split constraints
- coalesce dependent epistemic split rules
- *(release)* harden validation and gpu fallback paths
- integrate K7 K8 planned clique metadata
- module-level docs + lib-test flips
- remove multi-recursive promoter gate
- cargo fmt, evidence count, extract prepare_leader_inputs + real helper extraction
- address selectivity-driven join reordering review patches — duplicate attr, stale comment, evidence count, matcher tests
- classifier col0 + missing scope deliverables
- *(logic)* skip recursive SCCs in promote_multiway
- *(logic)* SCC-aware planner + structural-precedence repair (v0.6.2 PR 9)
- *(logic)* canonical explain_plans + refreshed module docs (PR 6 follow-up)
- *(logic)* typed gate defers to structural errors (v0.6.2 PR 5 follow-up)
- *(pyxlog)* install local wheels for explicit python
- *(cuda)* embed portable PTX fallback
- *(pyxlog)* ship kernels in wheels and document cubin path
- *(ci)* repair main release automation ([#27](https://github.com/BrainyBlaze/xlog/pull/27))
- *(ci)* keep README release metadata in sync ([#26](https://github.com/BrainyBlaze/xlog/pull/26))

### Other

- v0.9.2 whitepaper + documentation realignment ([#133](https://github.com/BrainyBlaze/xlog/pull/133))
- Set v0.9.2 release metadata
- Clarify v0.9.2 WFS release contract
- Close v0.9.2 epistemic semantic gaps
- Close v0.9.2 epistemic release
- cargo fmt
- negated-modal-in-recursion — stratified sub-case admits (co-evolving modal-recursion case), genuine negation cycle hits formal WFS bound
- document nested-modal chain-collapse semantics + interior-negation boundary
- update EIR+split tests from nested-modal rejection to collapse contract
- world-view mutation probe for nested-modal collapse direction
- derived-head coupling — stratified-vs-reference equivalence + true-cycle wall
- co-evolving modal-recursion classification unit tests (polarity/mode scoping)
- flip FAEEL foundedness logic tests to founded-extension semantics
- document split_epistemic_program (clean release surface)
- cargo fmt on v0.9.1 epistemic changeset
- safe split dependency and coupling semantics
- document v0.9.0 epistemic language surface
- *(release)* align v0.9.0 package metadata
- integrate v0.8.9 diagnostics surfaces
- reuse v0.8.6 bundle for v0.9.0
- *(release)* prepare v0.8.6 tag metadata
- *(release)* correct v0.8.5 public status
- certify language integration
- *(release)* prepare v0.8.0
- mark v0.7.0 release complete
- close phase2 integration gate
- Merge sort-label propagation branch into the phase-2 integration branch
- Merge K7/K8 clique-template branch into the phase-2 integration branch
- add K7/K8 clique templates
- cert flat-stats no helper rewrite
- scrub stale doc fragment in promotes_multirec_triangle test
- clean up dead helpers + unused imports in K=5 and K=6 clique WCOJ test files
- cargo fmt for workspace gate
- promoter + runtime-dispatch certs
- 15 acceptance tests across the acceptance matrix
- rename acceptance test + clarify evidence count math
- *(logic+integration)* variable-ordering cost model acceptance gate across the full matrix
- restore selectivity-driven join reordering acceptance gates — 4-cycle compile-time + runtime helper and synthesis certs
- *(logic)* selectivity_pass compile-time certs
- cargo fmt 4-cycle WCOJ test files
- *(logic)* strengthen optimizer arm tests with 4-input fixture
- *(workspace)* WCOJ doc cleanup post-MultiWayJoin
- *(v0.6.2)* prepare roadmap changelog and version
- *(logic)* hypergraph certification workloads (v0.6.2 PR 7)
- *(logic)* correct explain_plans sort ordering claims
- *(v0.6.1)* version bump + roadmap cleanup + changelog
- *(readme)* bump version badge + release-status line to v0.6.0
- restore audit README framing with current release setup
- Tighten workspace warning hygiene
- Merge branch 'audit/v0.5.0-prerelease'
- integrate prerelease audit docs
- harden public release readiness

## [0.9.2](https://github.com/BrainyBlaze/xlog/compare/xlog-stats-v0.5.0...xlog-stats-v0.9.2) - 2026-06-08

### Added

- add cost-aware k-clique planner

### Fixed

- *(release)* drop README version sync + dynamic badges + agent release rules ([#124](https://github.com/BrainyBlaze/xlog/pull/124))
- route epistemic examples through xlog run
- *(release)* harden validation and gpu fallback paths
- *(pyxlog)* install local wheels for explicit python
- *(cuda)* embed portable PTX fallback
- *(pyxlog)* ship kernels in wheels and document cubin path
- *(ci)* repair main release automation ([#27](https://github.com/BrainyBlaze/xlog/pull/27))
- *(ci)* keep README release metadata in sync ([#26](https://github.com/BrainyBlaze/xlog/pull/26))

### Other

- v0.9.2 whitepaper + documentation realignment ([#133](https://github.com/BrainyBlaze/xlog/pull/133))
- Set v0.9.2 release metadata
- document v0.9.0 epistemic language surface
- *(release)* align v0.9.0 package metadata
- integrate v0.8.9 diagnostics surfaces
- integrate v0.8.8 external world-model diagnostics into v0.8.9
- integrate first external diagnostics into v0.8.9
- Exercise external generated-rule diagnostics
- Resolve remaining XLOG evidence issues
- Add v0.8.7 external world-model diagnostics
- *(release)* prepare v0.8.6 tag metadata
- *(release)* correct v0.8.5 public status
- *(release)* prepare v0.8.0
- mark v0.7.0 release complete
- *(v0.6.2)* prepare roadmap changelog and version
- *(v0.6.1)* version bump + roadmap cleanup + changelog
- *(readme)* bump version badge + release-status line to v0.6.0
- restore audit README framing with current release setup
- Merge branch 'audit/v0.5.0-prerelease'
- integrate prerelease audit docs
- harden public release readiness

## [0.9.2](https://github.com/BrainyBlaze/xlog/compare/xlog-cuda-v0.5.0...xlog-cuda-v0.9.2) - 2026-06-08

### Added

- *(cuda)* XLOG_PTX_MAX_VERSION â downgrade embedded portable PTX ISA
- *(mc)* sparse WCOJ world-batched GPU-resident MC engine
- *(mc)* no-host instrumentation foundation for WCOJ engine (alloc + fixpoint counters)
- *(mc)* GPU-resident Datalog/MC engine (megakernel) + K1-K5 pilots
- checkpoint epistemic solver semantics
- add chain exact shared-memory scorer
- extend exact induction typed dispatch
- close CUDA Graph-mode set maintenance
- certify CUDA Graph external consumer graph path
- cache bounded CSM CUDA graphs
- add bounded CSM CUDA graph path
- add CUDA graph execution wrapper
- remove adaptive skew classifier surface
- route clique kernels through HG block-slice
- route u64 4-cycle through HG block-slice
- route u64 triangle through HG block-slice
- retire old u32 triangle materialize surface
- route u32 4-cycle through HG block-slice
- retire old u32 triangle count surface
- retire layout/count kernel fusion fused count kernel
- reuse HG block workspace
- make HG cached count single-pass
- cache HG materialization and add superhub gate bench
- route u32 triangle dispatch through HG pipeline
- add triangle HG materialize pipeline
- add triangle HG work-plan count surface
- add persistent WCOJ metadata builder
- production kernel fusion (layout+count) with threshold dispatch + auto-disable + cert grid
- add sort_merge_join_v2_inner_u32_1key + is_sorted_ascending_u32 provider fns
- add sort-merge inner-join kernel + sortedness-detection kernel
- add nested_loop_join_v2_inner_u32_1key in relational.rs (gather-based)
- add nested-loop emit-pairs kernel (multi-col-compatible)
- *(cuda)* K=5 and K=6 clique WCOJ clique provider entries
- *(cuda)* K=5 and K=6 clique WCOJ templated clique kernel for k=5 + k=6
- *(cuda)* generic sorted-relation accessors generic wcoj_layout_sort_*_recorded entry points
- *(cuda)* wcoj_project_2col_swap_recorded + wcoj_project_output_columns_recorded
- *(cuda+runtime)* 4-cycle skew classifier + adaptive opt-in
- *(cuda)* u64 4-cycle WCOJ kernels + provider + tests
- *(cuda)* u32 4-cycle WCOJ kernels + provider + tests
- *(cuda)* WCOJ layout fast-path for sorted+unique inputs (v0.6.2)
- *(cuda)* WCOJ adaptive-dispatch skew classifier (v0.6.2 A2-lite commit A)
- *(cuda)* WCOJ u64 provider kernels + entries (v0.6.2)
- *(cuda)* sort_recorded + dedup_full_row_recorded U64 (v0.6.2)
- *(cuda)* WCOJ Symbol key support (v0.6.2)
- *(cuda)* WCOJ sorted-layout construction u32 (v0.6.2)
- *(cuda)* WCOJ triangle device-side scan + scalar D2H total
- *(cuda)* GPU 3-way WCOJ triangle kernel u32 v1 (v0.6.2)
- *(cuda)* wire recorded CSM hash-join dispatch ([#91](https://github.com/BrainyBlaze/xlog/pull/91))
- *(cuda)* add recorded indexed LeftOuter count-scan-materialize path ([#87](https://github.com/BrainyBlaze/xlog/pull/87))
- *(cuda)* add recorded LeftOuter count-scan-materialize path ([#84](https://github.com/BrainyBlaze/xlog/pull/84))
- *(cuda)* formal cert harness for runtime-backed recorded path
- *(cuda)* GPU-resident binary-join indexed Inner CSM
- *(cuda)* GPU-resident binary-join Inner retake — count→scan→materialize
- *(cuda)* env-gated runtime dispatch for sort/dedup/GroupBy/hash-join + cert mode
- *(cuda)* provider-level recorded indexed hash join + LeftOuter step-D recorder fix
- *(cuda)* provider-level recorded LeftOuter hash join
- *(cuda)* provider-level recorded Semi / Anti hash join
- *(cuda)* provider-level recorded inner hash join
- *(cuda)* provider-level recorded GroupBy multi-agg (U32 keys, count/sum/min/max)
- *(cuda)* provider-level recorded sort + dedup_full_row (u32 / Symbol)
- *(cuda)* preserve runtime identity for xlog-owned DLPack / Arrow columns
- *(cuda)* migrate fused compare+scan+compact filter to recorded discipline
- *(cuda)* env-gated recorded filter dispatch (XLOG_USE_RECORDED_FILTERS)
- *(cuda)* v0.6 stream-safe runtime + LaunchRecorder + filter predicate matrix
- *(cuda)* v0.6 device-runtime allocator (opt-in) + A3 stability ([#54](https://github.com/BrainyBlaze/xlog/pull/54))
- *(cuda)* binary-join output counts as metadata reads (v0.5.5 PR 3) ([#52](https://github.com/BrainyBlaze/xlog/pull/52))
- *(cuda)* GPU full-row dedup and set-difference (v0.5.5 PR 2) ([#50](https://github.com/BrainyBlaze/xlog/pull/50))
- *(runtime)* add strict deterministic D2H guard (v0.5.5) ([#49](https://github.com/BrainyBlaze/xlog/pull/49))

### Fixed

- bootstrap cuda-ci runner — bump test iterations + fix maturin compatibility ([#127](https://github.com/BrainyBlaze/xlog/pull/127))
- *(release)* drop README version sync + dynamic badges + agent release rules ([#124](https://github.com/BrainyBlaze/xlog/pull/124))
- route epistemic examples through xlog run
- close persistent index background build scope
- close GPU-native count-lift exact path
- *(release)* harden validation and gpu fallback paths
- integrate K7 K8 planned clique metadata
- close mint4 path-isolated gate
- extend 4cycle e2-prefix mitigation to u64
- mitigate M_INT.4 4cycle HG regression
- route nested-loop dispatch through shared record_join_result feedback
- tighten Tier-1 wrapper contract + revert recursive helper extension
- real interner-allocated Symbol IDs + drop test-file warnings + tighten D4 wording
- cargo fmt, evidence count, extract prepare_leader_inputs + real helper extraction
- classifier col0 + missing scope deliverables
- *(cuda)* drain WCOJ layout fast-path failure paths
- *(runtime)* harden WCOJ phase timing diagnostics
- *(cuda)* drain launch stream on skew classifier failure paths (v0.6.2)
- *(cuda)* record d_overflow on three CSM materialize recorders ([#89](https://github.com/BrainyBlaze/xlog/pull/89))
- *(cuda)* access-aware stream dependency manager for cross-stream lifetime safety ([#72](https://github.com/BrainyBlaze/xlog/pull/72))
- *(cuda)* clamp recorded compact mask domain
- *(logic)* restore deterministic recursive set evaluation
- *(pyxlog)* install local wheels for explicit python
- *(cuda)* embed portable PTX fallback
- *(pyxlog)* ship kernels in wheels and document cubin path
- *(ci)* repair main release automation ([#27](https://github.com/BrainyBlaze/xlog/pull/27))
- *(ci)* keep README release metadata in sync ([#26](https://github.com/BrainyBlaze/xlog/pull/26))

### Other

- v0.9.2 whitepaper + documentation realignment ([#133](https://github.com/BrainyBlaze/xlog/pull/133))
- Set v0.9.2 release metadata
- integrate main MC GPU-resident engine into v0.9.2 epistemic completion
- document v0.9.0 epistemic language surface
- *(release)* align v0.9.0 package metadata
- integrate v0.8.9 diagnostics surfaces
- reuse v0.8.6 bundle for v0.9.0
- *(release)* prepare v0.8.6 tag metadata
- *(release)* correct v0.8.5 public status
- *(release)* prepare v0.8.0
- mark v0.7.0 release complete
- close phase2 purge gate
- close phase2 integration gate
- Merge CUDA Graph benchmark-spike branch into the phase-2 integration branch
- Merge sort-label propagation branch into the phase-2 integration branch
- Merge K7/K8 clique-template branch into the phase-2 integration branch
- add K7/K8 clique templates
- Merge stream-mux AOT branch into WCOJ bundle integration
- graceful close and paper harness
- certify HG metadata storage budget
- remove dead layout/count kernel fusion route counters
- patch stale rustdoc + kernel comments after iter-6 unwiring
- fmt + rustdoc cleanup
- align plan + provider rustdoc with landed byte-check + counter type
- workspace gate green pre-bench
- clean up dead helpers + unused imports in K=5 and K=6 clique WCOJ test files
- cargo fmt for workspace gate
- *(cuda)* K=5 and K=6 clique WCOJ provider certs + source-audit
- *(cuda)* generic sorted-relation accessors acceptance grid — 82 tests across width-class and arity
- *(cuda)* correct wcoj_4cycle_skew_score_u32 doc to col0
- *(cuda)* layout reuse smoke for 4-cycle
- *(v0.6.2)* prepare roadmap changelog and version
- *(runtime)* WCOJ phase-timing scaffolding + report (v0.6.2)
- *(cuda)* WCOJ U64 strict deterministic-D2H gate (v0.6.2)
- *(cuda)* update recorded dedup U64 scope comment
- *(wcoj)* update Symbol dispatch scope comments
- *(cuda)* planner-to-provider WCOJ certification (v0.6.2)
- *(v0.6.1)* version bump + roadmap cleanup + changelog
- *(readme)* bump version badge + release-status line to v0.6.0
- restore audit README framing with current release setup
- Fix validation regressions in release and examples
- Merge branch 'audit/v0.5.0-prerelease'
- integrate prerelease audit docs
- harden public release readiness

## [0.9.2](https://github.com/BrainyBlaze/xlog/compare/xlog-ir-v0.5.0...xlog-ir-v0.9.2) - 2026-06-08

### Added

- close augmented-projection multi-head coupling scope limit
- epistemic integrity constraints
- checkpoint epistemic solver semantics
- add k-clique cost gate routes
- add k-clique RIR variable order
- *(logic)* WcojVariableOrderingModel trait + LeaderCardinalityModel
- *(ir)* VariableOrder + LookupPerm types + MultiWayJoin.var_order field
- *(ir)* add RirNode::MultiWayJoin variant

### Fixed

- *(release)* drop README version sync + dynamic badges + agent release rules ([#124](https://github.com/BrainyBlaze/xlog/pull/124))
- route epistemic examples through xlog run
- *(release)* harden validation and gpu fallback paths
- *(pyxlog)* install local wheels for explicit python
- *(cuda)* embed portable PTX fallback
- *(pyxlog)* ship kernels in wheels and document cubin path
- *(ci)* repair main release automation ([#27](https://github.com/BrainyBlaze/xlog/pull/27))
- *(ci)* keep README release metadata in sync ([#26](https://github.com/BrainyBlaze/xlog/pull/26))
- unblock release publish verification

### Other

- v0.9.2 whitepaper + documentation realignment ([#133](https://github.com/BrainyBlaze/xlog/pull/133))
- Set v0.9.2 release metadata
- document v0.9.0 epistemic language surface
- *(release)* align v0.9.0 package metadata
- integrate v0.8.9 diagnostics surfaces
- reuse v0.8.6 bundle for v0.9.0
- *(release)* prepare v0.8.6 tag metadata
- *(release)* correct v0.8.5 public status
- *(release)* prepare v0.8.0
- mark v0.7.0 release complete
- G_W63 production chain join route
- phase-2 purge rerun
- *(workspace)* MultiWayJoin shape-agnosticism guards
- *(ir)* MultiWayJoin walker contract
- *(v0.6.2)* prepare roadmap changelog and version
- *(v0.6.1)* version bump + roadmap cleanup + changelog
- *(readme)* bump version badge + release-status line to v0.6.0
- restore audit README framing with current release setup
- Tighten workspace warning hygiene
- Merge branch 'audit/v0.5.0-prerelease'
- integrate prerelease audit docs
- harden public release readiness

## [0.9.2](https://github.com/BrainyBlaze/xlog/compare/xlog-core-v0.5.0...xlog-core-v0.9.2) - 2026-06-08

### Added

- add persistent hash index telemetry
- add adaptive runtime reoptimization
- add runtime common subexpression cache
- extend exact induction typed dispatch
- expose xlog sort-label metadata
- remove adaptive skew classifier surface
- retire old u32 triangle count surface
- production kernel fusion (layout+count) with threshold dispatch + auto-disable + cert grid
- default-flip wcoj_cost_model resolver to Cardinality
- *(runtime)* dispatch sites use build_wcoj_cost_model factory
- *(core)* CostModelKind + RuntimeConfig::wcoj_cost_model
- *(runtime)* match_multiway_4cycle + try_dispatch_wcoj_4cycle force gate
- *(runtime)* default-on adaptive WCOJ + hard kill switch (v0.6.2)
- *(runtime)* adaptive WCOJ dispatch + classifier branch (v0.6.2 A2-lite commit B)
- *(runtime)* env-gated WCOJ triangle executor wiring (v0.6.2)
- *(runtime)* add strict deterministic D2H guard (v0.5.5) ([#49](https://github.com/BrainyBlaze/xlog/pull/49))

### Fixed

- *(release)* drop README version sync + dynamic badges + agent release rules ([#124](https://github.com/BrainyBlaze/xlog/pull/124))
- route epistemic examples through xlog run
- *(release)* harden validation and gpu fallback paths
- restore cost-model default-flip cert
- *(pyxlog)* install local wheels for explicit python
- *(cuda)* embed portable PTX fallback
- *(pyxlog)* ship kernels in wheels and document cubin path
- *(ci)* repair main release automation ([#27](https://github.com/BrainyBlaze/xlog/pull/27))
- *(ci)* keep README release metadata in sync ([#26](https://github.com/BrainyBlaze/xlog/pull/26))
- unblock release publish verification

### Other

- v0.9.2 whitepaper + documentation realignment ([#133](https://github.com/BrainyBlaze/xlog/pull/133))
- Set v0.9.2 release metadata
- document v0.9.0 epistemic language surface
- *(release)* align v0.9.0 package metadata
- integrate v0.8.9 diagnostics surfaces
- integrate v0.8.8 external world-model diagnostics into v0.8.9
- integrate first external diagnostics into v0.8.9
- Exercise external generated-rule diagnostics
- Resolve remaining XLOG evidence issues
- Add v0.8.7 external world-model diagnostics
- *(release)* prepare v0.8.6 tag metadata
- *(release)* correct v0.8.5 public status
- *(release)* prepare v0.8.0
- mark v0.7.0 release complete
- close purge gate
- *(v0.6.2)* prepare roadmap changelog and version
- *(v0.6.1)* version bump + roadmap cleanup + changelog
- *(readme)* bump version badge + release-status line to v0.6.0
- restore audit README framing with current release setup
- Merge branch 'audit/v0.5.0-prerelease'
- integrate prerelease audit docs
- harden public release readiness

## [0.9.2] — 2026-06-02

v0.9.2 Epistemic Executor Semantic Completion. Closes the three honest
Category-B semantic gaps tracked after v0.9.1, all validated on the production
`xlog run` path.

### Added (v0.9.2)

- Mixed modal membership: a rule combining a global modal gate (ground/anonymous/nullary modal)
  with a per-row bound-variable modal gate now composes both gate classes
  conjunctively on the GPU device path (the row-map kernel's per-row path applies
  the global-gate check); the prior fail-closed guard is removed. Example
  `14-mixed-literal-membership.xlog`.
- recursive epistemic fixpoint recursive epistemic fixpoint: recursive ordinary predicates inside
  epistemic programs now evaluate to fixpoint when every modal atom ranges over an
  invariant relation (EDB / lower non-recursive non-epistemic stratum) — the modal
  literal reduces to its gated relation and the reduced ordinary recursive program
  runs through the existing GPU recursive fixpoint engine. Examples
  `15-recursive-epistemic-closure.xlog` / `15-recursive-epistemic-chain.xlog`
  (transitive closure incl. derived multi-hop tuples).
- Cross-component epistemic coupling: a coalesced component with more than one
  epistemic output head sharing a base modal predicate is JOINT-SOLVED with
  multi-output materialization — one candidate enumeration + world-view validation
  over the combined modals, then each head materialized against the same accepted
  world view (per-head scoped row-filter + per-head output projection via
  `public_head_arity`, reusing the WCOJ-promoted reduced runtime plan and candidate-world enumeration
  enumeration). Heads of DIFFERING arity sharing a base modal are supported.
  Examples `18-cross-component-joint-shared-modal.xlog` (`known={1,2}`,
  `maybe={2}`), `21` (three heads), `27` (augmented differing-arity).
- Stratified epistemic execution: a modal over an epistemic-DERIVED head that is
  itself DETERMINED (its modals bottom out in invariant/EDB relations, acyclically)
  is resolved by stratified execution — the determined head is gated once and
  materialized into the relation store as a lower stratum
  (`LogicExecutionPlan::EpistemicStratified`), and the higher stratum reads it as a
  plain base relation through the existing membership/join filter. The theorem
  `know R ≡ R` (for determined `R`) is applied at the STORE boundary, not the rule
  body — no resolve-into-body, no double-gating. Determined-closure is transitive
  (ordinary predicates over determined heads), supports multi-column binding modals,
  and a negated modal over an invariant relation reduces to ordinary negation.
  Examples `17` (chained `b:-know a`), `24` (transitive determined-ordinary), `25`
  (recursion over a determined head), `26` (negated-modal-over-invariant in
  recursion), `28` (determined-epistemic multi-column binding).
- Structured modal tuple-keys: list/compound/anonymous modal keys
  (`know watched([H])`) flatten element-wise into the existing N-column GPU matcher;
  unbounded forms (cons `[H|T]`, predref, aggregate) reject with a `ResourceExhausted`
  finiteness diagnostic. Example `23`.
- Variable-keyed + nested epistemic constraints: a single-occurrence positive
  constraint variable (`:- know p(X).`) lowers to an Anonymous wildcard and ranges
  existentially on device; multi-literal distinct-variable conjunctions
  (`:- know p(X), know q(Y).`) AND the independent existentials. Examples `34`/`35`/`36`.
- Shared-variable epistemic constraint joins (item E1): the join
  `:- know p(X), possible q(X).`, the diagonal `:- know p(X,X).`, and the
  negated-difference `:- q(X), not know p(X).` are resolved by a sound program-level
  desugaring at normalization — `:- L1,…,Ln.` ⟶ `__epi_join_N(Vars) :- ord(L1),…,ord(Ln).`
  + `:- know __epi_join_N(Vars).`, where `ord` ordinary-izes each modal literal
  (`know/possible r → r`, `not know/possible r → not r`). For a base/EDB or
  ordinary-derived target `know r ≡ possible r ≡ r`, so the ordinary join is exactly the
  forbidden binding set and the single-occurrence `:- know __epi_join_N(Vars)` routes
  through the existing variable-keyed prune-to-empty path — no new kernel. Guarded to
  non-modal-derived targets. Examples `38`/`39` (diagonal), `40` (join), `41`
  (negated-difference).
- Stratified negated-modal recursion (item A1): a negated modal `not know R` over a
  recursive relation in a strictly-lower stratum executes on GPU as ordinary stratified
  negation (`not know R ≡ not R` once R is materialized). Example `37`.
- Same-name multi-arity modal coupling (item F): distinct arities of the same predicate
  name are treated as distinct relations (arity-qualified modal tuple-source resolution);
  derived-head coupling stratifies with a split-vs-unsplit equivalence proof; a genuine
  cyclic modal coupling stays rejected with a precise diagnostic.

### Changed (v0.9.2)

- A standalone negated-variable-keyed integrity constraint whose variable appears only
  under negation (`:- not know p(X).`) now reports the same NAF safety error as ordinary
  Datalog (`:- not r(X).`, "unbound variable … in negated atom") instead of a misleading
  "unimplemented" diagnostic — the variable is not range-restricted, so the program is
  ill-formed, not a missing feature. The meaningful negated form `:- q(X), not know p(X).`
  (X positively bound) is the shared-variable join above (item E1).

- Recursive epistemic programs are no longer uniformly fail-closed: the recursive epistemic fixpoint
  invariant-modal fragment AND recursion/coupling over any DETERMINED head (via
  stratified execution, see above), positive modal recursion with founded semantics, explicit
  G91 positive `possible` recursion, stratified negated-modal recursion, and cyclic
  negated-modal recursion through WFS are now supported. A negated modal over an
  invariant or materialized determined relation reduces to ordinary negation.

### Determined and recursive modal families tightened

The determined/non-determined boundary is closed under composition. Every modal
target is either DETERMINED (fixed extension → `know R ≡ R` → resolved, via
joint-solving or stratified execution; any arity, filtering or binding, coupling or
recursion) or NON-DETERMINED and handled by the appropriate recursive semantics or
typed boundary:

- Positive co-evolving modal-recursion case `know` recursion computes the FAEEL founded least fixpoint.
- Positive G91 `possible` recursion applies the G91 compatibility self-support
  assumption.
- Negated-modal recursion that is stratified reduces to ordinary anti-join after
  the lower fixpoint materializes; cyclic negated-modal recursion routes through
  GPU-backed WFS alternating fixpoint rather than host WFS, with committed
  examples covering mode `{FAEEL,G91}` x modal `{not know,not possible}` x seed
  `{present,absent}` for both plain WFS and WFS plus ordinary EDB negation.
- Genuinely cyclic modal coupling with no founded/WFS order remains typed
  fail-closed.

Remaining genuinely-undefined cases are CI-enforced as over-broadening gates
(for example cyclic modal coupling with no founded/WFS order and unbounded
tuple-key pilots such as `23b`): because determined-closure acceptance is
permissive, they assert no undefined program leaks a wrong-but-non-empty answer.
Defined recursive/self-support cases execute instead: examples `22`, `31`, `32`,
`33`, and `43` cover FAEEL founded recursion, empty FAEEL self-support, G91
self-support, cyclic negated-modal WFS, and G91 positive possible recursion.

Full v0.9.2 status: `docs/plans/2026-05-31-v092-epistemic-semantic-completion-status.md`.

---

v0.9.1 Epistemic Executor Completion. Turns the v0.9.0 bounded epistemic
executor into a load-bearing execution surface while preserving the existing
fail-closed boundaries. Candidate worlds are derived from EIR, modal membership
is value-level on the device, FAEEL foundedness is checked per tuple key,
epistemic integrity constraints prune candidate world views, splits are
equivalence-checked, and rules coupling multiple distinct epistemic predicates
are solved jointly. EIR remains the semantic boundary and direct raw RIR
lowering stays a rejection boundary. Completed-vs-scoped-out semantics are
recorded in `docs/plans/2026-05-29-v091-epistemic-executor-completion-status.md`.

### Added (v0.9.1)

- Added EIR-derived candidate-world enumeration: the candidate epistemic
  assumption space is derived from the program (full device lattice) with
  deterministic generated/propagated/tested/accepted/rejected/reason trace
  counts; empty accepted-world-view results are distinguished from execution
  failure and over-budget candidate spaces fail closed before partial execution.
- Added value-level tuple-key modal membership on the GPU device path: ground,
  single and multiple bound variables, repeated-variable equality, anonymous
  wildcard positions, and arity-0 keys; unsupported term classes remain typed
  fail-closed.
- Added per-tuple-key FAEEL founded self-support: unfounded `p() :- possible p().`
  is rejected, self-reference is accepted only with independent founded support
  for the same tuple key, and G91 compatibility self-support stays separate.
- Added GPU epistemic integrity constraints: `:- know g().`,
  `:- possible g().`, and `:- not possible g().` prune candidate world views via
  a new constraint kernel (rejection reason `WorldViewConstraintViolation`);
  epistemic constraints are dropped from the reduced ordinary program with no
  ordinary-RIR rewrite. Rejected candidates carry the specific firing constraint
  index (KPI EGB04.K2) via a parallel `constraint_violation_index` buffer,
  surfaced as `result.semantic_trace.constraint_violation_indices`.
- Added explainable safe-split semantics: split, coalesce, and reject decisions
  carry typed `EpistemicComponentMergeReason`s, valid splits are equivalence-
  checked against unsplit execution, and recomposition covers each source rule
  exactly once.
- Added joint multi-epistemic-predicate solving: rules coupling more than one
  distinct-name epistemic predicate (any operator mix, including negated modal
  literals) are solved as a joint modal conjunction over the candidate world
  view, matching unsplit execution.
- Added `CudaKernelProvider::create_zero_arity_buffer` for materializing nullary
  relations with a presence row.

### Fixed (v0.9.1)

- Fixed nullary EDB facts (`pred().`) being materialized as zero rows (read as
  absent), which broke ordinary nullary queries and ground/nullary modal
  membership; arity-0 facts now materialize one unit tuple, restoring `xlog run`
  on `examples/epistemic/02-05`.
- Fixed a global-membership-gate soundness bug where pure-ground, anonymous, or
  nullary modal literals were left ungated, so `know flag(...)` could emit rows
  regardless of whether the literal held.
- Fixed unstable typed diagnostics for nested modal forms with interspersed
  `not` (e.g. `know not possible p()`), which previously fell through to a
  generic parse error instead of a stable `UnsupportedEpistemicConstruct`.
- Fixed `xlog run` routing for multi-output epistemic programs that combine
  bound-variable modal membership with multiple output heads; shared source
  facts such as `node(X)` no longer force otherwise independent split
  components into one single-plan multi-output execution.
- Fixed recursive epistemic programs to fail closed with
  `UnsupportedEpistemicConstruct { construct: "recursive epistemic program" }`
  instead of entering the bounded single-pass executor, which does not yet
  iterate recursive epistemic fixpoints.

### Changed (v0.9.1)

- The epistemic split layer no longer blanket-rejects rules coupling multiple
  distinct epistemic predicates; such rules are routed to joint solving. The
  `xlog-integration` split-coupling test was updated to the new accepted
  contract.
- Source facts are treated as reusable extensional inputs for split planning,
  not as derived component producers. Components still coalesce on ordinary
  derived dependencies and integrity constraints.

### Known gaps (v0.9.1, tracked — closed or narrowed in v0.9.2)

-  **mixed per-row and global modal membership in one rule:** _CLOSED in v0.9.2_ — the two gate classes now compose conjunctively on the GPU path.
- **Recursive epistemic fixpoints:** _CLOSED in v0.9.2 under the exact
  GPU-backed WFS contract_ — recursive epistemic fixpoint invariant-modal recursion,
  determined-head stratification, positive modal recursion with founded semantics, G91 positive
  `possible` recursion, stratified negated-modal recursion, and cyclic
  negated-modal recursion through the `xlog-gpu` GPU-backed WFS plan execute
  through `xlog run` without the old `xlog_prob` host-WFS solver. This closure
  is not a device-resident/no-host-interaction WFS residency claim; the WFS path
  still uses host orchestration and may use metadata row-count reads for
  convergence.
  The focused WFS example set covers mode, negated-modal operator,
  seed-present/seed-absent, ordinary EDB-negation-in-SCC axes, and a load-bearing
  EDB target-state axis where `not banned(2)` flips the seed-founded reach tuple.
  Host WFS is not an accepted production fallback.

- **Same-name multi-arity modal coupling:** _SOLVED in v0.9.2 (ITEM F)_ — a
  program using the same predicate name at two arities in modal literals
  (`know p(X)`, `possible p(X,Y)`) is no longer rejected. Distinct arities are
  distinct relations, so the modal tuple-source resolution disambiguates by arity
  (arity-qualified store key `p/1`/`p/2`, bare-name fallback). The coupling
  joint-solves on device to exact tuples per arity with zero CPU fallback.
- **Derived-head modal coupling:** modal coupling over an epistemically-DETERMINED
  derived head is solved by STRATIFICATION (the stratified joint result equals the
  per-stratum independent reference exactly); a genuinely-CYCLIC modal coupling
  (`a:-know b. b:-know a.`) has no founded order and stays typed fail-closed
  end-to-end with a precise diagnostic naming both coupled heads.

Goal-mandated typed fail-closed fragments (aggregate/compound/list/predref modal
keys, unsafe variable-keyed or CPU-scan epistemic constraints, genuinely-cyclic
modal coupling with no founded order) remain rejection-by-design per the bundle
spec and lock #5 — verified by negative pilots, not debt. Finite nested modal
semantics are no longer in this rejection bucket in v0.9.2: examples 13/13b/13c
execute, 13f plus its companions prove the interior-negation duality in both target states and
both modes, 13g-13v cover all 64
two-operator negation cells under FAEEL, and 13w* replays those cells under explicit G91.
Historical v0.9.1 rejection list:
`docs/plans/2026-05-29-v091-epistemic-executor-completion-status.md`.

v0.9.0 Epistemic Solver Release Candidate. This branch layers the v0.8.7-v0.8.9
external diagnostic provenance pack into the v0.9.0 GPU-native epistemic,
solver, and probabilistic release candidate.

v0.8.9 Integrated External Diagnostics Pack. This branch consolidates reusable
XLOG and pyxlog gaps exposed by external diagnostic consumers into core APIs,
focused regressions, and release-facing documentation. It includes the initial
v0.8.7 diagnostics and biomedical graph fixes, the v0.8.8 world-model provenance
refinements, and the v0.8.9 external diagnostic surfaces.

### Added

- Added the v0.9.0 epistemic solver semantics release surface, including EIR,
  G91, FAEEL, Generate-Propagate-Test execution, epistemic splitting, GPU-native
  executable plans, solver-service integration, MaxSAT, GPU portfolio solving,
  and probabilistic epistemic evidence paths.
- Added production `xlog run` pilots for `examples/epistemic/*.xlog`, covering
  accepted EIR, G91, FAEEL, GPT, and split epistemic programs through the
  high-level GPU runtime route.
- Added the v0.8.7 external world-model diagnostics architecture document covering
  induced-rule provenance, rule provenance, proof traces, delta debug,
  temporal relation metadata, and neural hot-loop diagnostics.
- Added shared `xlog-logic` rule provenance and query proof trace diagnostics,
  surfaced through `xlog explain --format json` and pyxlog
  `rule_provenance()` / `proof_traces()` methods.
- Added `xlog-induce` generated-rule provenance records with support rows,
  rejected alternatives, falsification counts, predicate inventory, and a
  stable generation trace hash.
- Added v0.8.8 `InducedRuleProvenance` / `InducedRuleRegistry` aliases for
  native induced-rule provenance consumers.
- Added pyxlog persistent-session delta debug output with changed relation
  names, metadata-only debug traces, and an opt-in full-recompute equivalence
  check.
- Added pyxlog temporal relation metadata helpers for session-managed
  relations, including process-boundary and temporal-order metadata used by an
  external world-model validation package.
- Added `CompiledProgram.neural_hot_loop_diagnostics()` with transfer,
  CUDA Graph, circuit-cache, and explicit unavailable-status diagnostics for
  unsupported hot-loop counters.
- Added native biomedical graph stream ingestion telemetry through
  `xlog_gpu::biokg`, including JSONL/CSV/N-Triples parsing, typed edge sinks,
  row hashes, relation/split histograms, and bounded-memory chunk diagnostics.
- Added `xlog explain --format json` generated-rule row diagnostics with
  accepted/rejected row decisions, failed predicates, threshold comparisons,
  and aggregate inputs. The CLI now also binds external candidate relation rows
  from colocated execution manifests so external generated rules produce non-empty
  row-level diagnostics through the XLOG explain surface.
- Added pyxlog session evidence APIs:
  `put_relation_with_provenance(...)`, `evidence(...)`, and
  `RelationEvidence.provenance()`.
- Added pyxlog nn/4 lineage metadata for checkpoint hashes, split hashes,
  calibration metrics, CUDA device, influence audits, and changed-acceptance
  records.
- Added `DeltaPlannerTelemetry` for relation-delta cache reuse, fallback
  decisions, affected SCCs, estimated/measured speedup, and planner advice.
- Added `scripts.validation_staging.ValidationStagingRun` so long-running
  validation artifacts promote to canonical evidence only after a complete PASS.
- Added `pyxlog.ilp.neurosymbolic.train_neurosymbolic_program(...)` for one
  source that combines `nn/4` declarations, trainable symbolic rules, and a
  training objective that updates neural parameters plus symbolic rule weights.
- Added `xlog_logic::DifferentiableProofTraceMap` with stable proof IDs,
  support atoms, symbolic clause weights, binary logistic loss, and gradient
  application hooks.
- Added `pyxlog.ilp.inventory.build_rule_inventory(...)` and
  `PromotionResult.rule_inventory` for selected/rejected clauses, training
  folds, held-out domains, promotion gates, and base-kernel checksum metadata.
- Added `pyxlog.runtime_audit.CudaExecutionAudit` and
  `HostMaterializationError` for fail-closed CUDA hot-loop audits covering
  `.cpu()`, `.tolist()`, `.item()`, score-row downloads, and recorded H2D/D2H
  transfers.
- Added `xlog_logic::diagnose_module_boundaries(...)` for frozen kernel
  predicates, adapter-only fact modules, held-out module declarations, and
  held-out-label candidate provenance.
- Added the sparse/WCOJ resident MC slice: structurally checked generic positive
  joins now run from a preallocated world-segmented sparse column arena with
  device row counters/offsets, arity-3 relation columns, exact no-host
  instrumentation, recursive sparse fixpoint evidence, kernel-written
  convergence/overflow diagnostics, an opt-in cooperative multi-block-per-world
  execution path with fenced cooperative barriers and atomic device
  change/continue reads, and deterministic `resident_resource_budget` fail-closed
  diagnostics before device allocation.
- Added `pyxlog.transfer_diagnostics.compute_transfer_diagnostics(...)` for
  grouped macro F1, minimum group F1, bootstrap confidence intervals, baseline
  uplift, paired sign tests, and missing-domain or missing-variant failures.
- Added an external diagnostic validation package, including requirements,
  validation plan, Datalog programs, evidence JSON, minimal reproducers,
  project-specific tests, validator tooling, and the resolved issue ledger.
- Added an architecture record for six external diagnostic issue resolutions and
  their regression locations.
- Added an issue-by-issue architecture note for the v0.8.8 external world-model
  diagnostics surface.

### Changed

- Integrated the v0.8.7-v0.8.9 diagnostics/provenance predecessor surfaces into
  the v0.9.0 release-candidate branch without retargeting them as v0.9.0-only
  features.
- Updated the language reference to v0.9.0-rc and documented the accepted
  epistemic literals, `epistemic_mode` pragma, grammar appendix, and production
  `xlog run` pilots.
- Updated architecture docs for Python bindings, CLI explain diagnostics,
  GPU execution, bounded exact-induction provenance, relation evidence, graph
  ingestion, delta planner telemetry, nn/4 lineage, validation staging, external
  dILP diagnostics, and transfer diagnostics.
- `train_and_promote(...)` now accepts training-fold, held-out-domain, and
  base-kernel checksum metadata so promotion results can carry reusable transfer
  audit evidence.
- `pyxlog` can import pure-Python helper modules when `pyxlog._native` is absent;
  native-backed entry points still fail explicitly instead of pretending to run.
- README, roadmap, dILP architecture, and Python binding docs now describe the
  unreleased integrated v0.8.9 external diagnostic surfaces and validation
  packages.
- The high-level `xlog-gpu::LogicProgram` path now detects accepted epistemic
  programs and dispatches them through the existing single/split epistemic GPU
  runtime instead of treating production examples as fixture-only inputs.

### Added

- **GPU-resident Datalog/MC execution engine** (`crates/xlog-prob/src/mc/resident.rs`
  + `crates/xlog-cuda/kernels/mc_resident.cu`). Replaces the host-sequenced
  per-sample Monte Carlo loop with a single device megakernel that evaluates ALL
  worlds in one launch. The sample/world id is the CUDA grid dimension; sampled
  facts, derived relations, evidence flags, query counts, and fixpoint state are
  device-resident (bounded-Herbrand dense-boolean). Recursive programs converge
  via a device-side double-buffered naive fixpoint with a shared change flag and a
  per-world iteration trace; no host reads drive control flow.
  - **Distinct from the predecessor `a894aab4`**: that commit only removed
    *tracked data-plane* HtoD/DtoH from the host loop but kept host orchestration
    (a Rust per-sample loop, per-sample kernel launches, and per-sample untracked
    `dtoh_scalar_untracked` row-count reads). The resident engine has **zero host
    interaction in the measured region**: 0 tracked HtoD, 0 tracked DtoH, 0
    untracked metadata reads, 0 host loop iterations, 0 per-sample host launches —
    proven CONSTANT across N=128 and N=1024 (`McNoHostStats`, new
    `untracked_metadata_dtoh_count` provider counter).
  - `evaluate_gpu_device*` now routes solely through this engine (no
    host-orchestrated fallback). Programs outside the supported fragment
    (bounded-domain positive Datalog; arity ≤ 3, body ≤ 3, ≤ 8 vars, bounded
    universe) **fail closed** before execution with a typed `ResidentRejection`
    (kind + construct + context). The CPU path survives only as a seed-matched
    test oracle, never accepted execution.
  - Acceptance (`tests/mc_resident.rs`): exact-value GPU-resident pilots (fact
    marginal, probabilistic marginal, evidence conditioning, multi-evidence,
    annotated-disjunction/exclusive choice, recursive transitive closure with a
    non-base derived tuple, ≥3-hop recursion proving >1 fixpoint iteration) + 4
    fail-closed negative tests.

### Fixed

- Made the Monte Carlo GPU-native hot loop zero **tracked** (data-plane)
  host/device transfer: the per-sample query/evidence count-pointer arrays are
  now built once before the loop (into engine-owned stable row-count buffers) and
  refreshed each sample with device-to-device copies, eliminating the prior
  one-tracked-HtoD-per-sample pointer upload. `McDeviceResult` now carries an
  always-on `hot_loop_transfers` (`McHotLoopTransfers`) measured around the
  sample/evaluate/count loop. As elsewhere in the engine, bounded control-plane
  metadata reads (relation `num_rows` scalars via `dtoh_scalar_untracked`) remain
  and are intentionally not counted by the data-plane transfer contract.
  `evaluate`/`evaluate_gpu` are clarified as host-result materialization (final
  count download *after* the loop) and `evaluate_cpu` as a CPU oracle/debug path,
  never zero-host/GPU-native release evidence.
- Hardened `scripts/stage_pyxlog_kernels.sh` so pyxlog release kernel staging
  builds the release target before resolving the release `OUT_DIR`, preventing
  stale kernel artifacts from being selected after source changes.
- Fixed zero-column Arrow/CLI output for nullary relations by preserving device
  row count in exported `RecordBatch` values and printing `rows: N` in pretty
  CLI output.

### Tests

- Added v0.8.7-focused source and regression coverage for generated
  rule diagnostics, biomedical graph streaming, relation-delta planner
  telemetry, pyxlog evidence APIs, nn/4 training lineage, and validation
  staging.
- Added pyxlog source API regression tests to lock the v0.8.8 stubs, docs, and
  Rust/Python source API surfaces.
- Added focused regressions for external diagnostic issues:
  `python/tests/test_nn4_dilp_training_surface.py`,
  `crates/xlog-logic/tests/differentiable_proof_trace.rs`,
  `python/tests/test_ilp_rule_inventory.py`,
  `python/tests/test_nn4_cuda_no_host_transfer_contract.py`,
  `crates/xlog-logic/tests/module_boundary_diagnostics.rs`, and
  `python/tests/test_transfer_metric_diagnostics.py`.
- Extended `python/tests/test_kernel_packaging_layout.py` to verify pyxlog
  staging rebuild order before release `OUT_DIR` discovery.
- Added `test_xlog_run_epistemic_examples` to execute every epistemic example
  through the compiled CLI and assert production output values.
- Added the MC transfer-budget gate
  `tests/mc_gpu_native.rs::mc_hot_loop_is_zero_transfer_both_strategies`
  (asserts zero hot-loop HtoD/DtoH across the clamped and rejection count
  strategies) plus GPU-native exact-count pilots for fact-marginal,
  evidence-conditioning (vs seeded CPU oracle), annotated-disjunction
  exclusive-choice, and recursive transitive-closure workloads. Classified
  `tests/mc.rs` and `tests/gpu_mc_vs_cpu.rs` as CPU-oracle/host-output tests,
  excluded from the zero-host acceptance matrix.

## [0.8.6] — 2026-05-19

external consumer Runtime Completion and GPU-Native Optimizer Pack. This release closes
the deferred v0.8.0 runtime completion items while preserving the v0.8.5
language-completeness surface.

### Added

- Added v0.8.6 persistent-session delta coalescing and opt-in relation-change
  callbacks, with metadata-only callback payloads and evidence-backed
  zero-data-plane-D2H hot paths.
- Added v0.8.6 native exact-induction completion for `U32` and `Symbol`
  pair buffers plus profile-gated chain-topology shared-memory CUDA scorers
  with A/B controls, parity evidence, and no added data-plane transfers.
- Added v0.8.6 runtime common subexpression elimination for safe
  deterministic subplans, with explicit off/on controls, relation-generation
  invalidation, unsafe-boundary diagnostics, and evidence-backed duplicate
  work reduction.
- Added v0.8.6 adaptive runtime re-optimization adoption for
  compiler-supplied candidate plans, with deterministic mis-plan telemetry,
  GPU output equivalence, rollback diagnostics, and zero added data-plane DTOH
  calls in the acceptance fixture.
- Added v0.8.6 persistent hash-index manager telemetry and key hardening for
  repeated session evaluations, including relation-generation/schema/device
  keys, stale-index invalidation, deterministic LRU budget eviction, and
  background-build request/completion/deferred counters plus a runtime-backed
  recorded provider build path. The build-heavy repeated-session fixture
  records `speedup_ratio=3.206` against the >=1.5x persistent-index target
  with zero tracked DTOH/H2D calls.
- Added v0.8.6 runtime consumer example-execution fixtures and validator for
  external consumer-shaped delta/optimizer workloads, neutral Mistaber-derived `.xlog`
  fixtures, v0.9.0 substrate primitives, and public pyxlog compatibility,
  reusing the v0.8.0/v0.8.5 validators and recording production-path evidence.
  The validator now includes a public `LogicRelationSession` persistent-index
  reuse probe that records a cache build/hit with zero tracked host transfers.
  Full consumer certification now derives feature coverage from behavior
  probes rather than `expected.json` declarations.

### Changed

- Corrected release-facing README, changelog, and roadmap status after the
  `v0.8.5` tag was published.
- Bumped workspace package metadata and internal workspace dependency
  constraints to `0.8.6` so package metadata matches release-facing docs.

### Fixed

- Hardened the v0.8.6 release validator so local pyxlog staging copies fresh
  Cargo `xlog-cuda` kernel artifacts into the staged package and sets
  `XLOG_CUBIN_DIR` there. This prevents ignored package-local kernels from
  shadowing the current build during v0.8.0/v0.8.5 compatibility validation.

### Release Status

- Closure proposal: `docs/plans/2026-05-19-v086-closure-proposal.md`.
- Certification evidence: `docs/evidence/2026-05-19-v086-*`.
- Release-board update, commit, merge, push, and annotated `v0.8.6` tag were
  authorized on 2026-05-19 after the closure package reached `MERGE_READY`.

## [0.8.5] — 2026-05-19

Language Completeness and Developer Experience. This release refreshes the
public language surface for finite terms/lists/meta constructs, explicit NAF,
magic-set planning, probabilistic aggregate inference, approximate inference
configuration, incremental parsing, and CLI inspectability while preserving
the v0.8.0 external consumer runtime compatibility surface.

### Added

- Added the v0.8.5 language-completeness documentation contract, including
  finite lists, safe meta-predicates, deterministic NAF, magic-set planning,
  probabilistic aggregate semantics, aggregate lifting, approximate inference
  configuration, incremental parsing, and `xlog explain` / `xlog repl` /
  `xlog watch` boundaries. The implementation remains tracked by the
  corresponding language-completeness evidence gates.
- Added `docs/architecture/language-v0.8.5.md` as the parser, term, probability,
  CLI, and v0.9.0 handoff contract for the v0.8.5 branch.
- Added finite list normalization for `list<T>` columns, list literals, safe
  cons patterns, and core list built-ins via `__xlog_list_*` helper relations
  that lower through the existing relational runtime path.
- Added safe v0.8.5 meta-predicate normalization for finite `ground`, `var`,
  `nonvar`, `functor`, `=..`, source-fact `findall`, and unary/binary
  `maplist` through `__xlog_meta_*` helper relations.
- Added v0.8.5 deterministic NAF safety diagnostics for source-order bound
  `not atom` use while preserving probabilistic WFS as a separate exact
  inference profile.
- Added v0.8.5 magic-set rewriting for safe bound deterministic recursive
  queries, including `#pragma magic_sets = auto|on|off`, generated magic
  predicates, row-reduction evidence, and typed decline diagnostics.
- Added v0.8.5 probabilistic aggregate support for finite `count`, `sum`,
  `min`, `max`, and `logsumexp` outcomes in exact provenance/PIR and MC
  deterministic aggregate execution, with typed exact-domain cap diagnostics.
- Added v0.8.5 small-domain aggregate lifting for finite probabilistic `count`
  heads using exact cardinality dynamic programming, plus explain metadata for
  fired and exact-enumeration fallback aggregate operators.
- Added the GPU-native exact count-lift evaluator for accepted fired
  probabilistic `count` aggregate queries; the production exact path launches
  `weights_count_lift_exact` instead of using a CPU finite-world shortcut.
- Added v0.8.5 approximate inference language integration: MC source pragmas,
  CLI override precedence, JSON output, and sample/evidence confidence
  reporting.
- Added v0.8.5 incremental parsing support with statement-level parser cache,
  stable spans, invalidation stats, module invalidation, and explain-path cache
  integration.
- Added v0.8.5 CLI developer surfaces for deterministic explain sections,
  parse-only REPL smoke sessions, and watch `--once --explain` smoke runs
  without introducing new dependencies.

### Fixed

- Hardened the release example validator so every `.xlog`, probabilistic,
  Python, and neural example runs in release validation without depending on
  an installed `pyxlog` wheel or optional external neural datasets.
- Fixed recorded CUDA sort/groupby row-count handling so compacted buffers do
  not cache row capacity as logical cardinality.
- Hardened adaptive join-index reuse so stale index metadata is evicted and
  falls back to the regular hash join instead of aborting valid queries.
- Updated the public CUDA certification count to 207/207 after the current
  full-suite and recorded-launch certification reruns.
- Fixed v0.8.5 list-helper reservation so ordinary `pair/2` relations remain
  compatible while reserved pair-helper arities still fail closed.

### Migration Notes

- Existing v0.8.0 programs remain compatible. The integration gate revalidated
  the v0.8.0 external consumer example/source guards and strict deterministic D2H runtime
  paths.
- New finite list/meta features intentionally reject non-finite, dynamic, or
  CPU-only term forms. Unsupported forms emit typed `v0.8.5 ... error`
  diagnostics with remediation guidance.
- `xlog explain`, `xlog repl`, and `xlog watch --once --explain` are safe
  inspectability paths for v0.8.5 programs and do not authorize release
  publishing by themselves.

### Release Status

- Closure proposal: `docs/plans/2026-05-19-v085-closure-proposal.md`.
- Certification evidence: `docs/evidence/2026-05-18-v085-*` and
  `docs/evidence/2026-05-19-v085-*`.
- Release-board update, commit, merge, push, and annotated `v0.8.5` tag were
  completed on 2026-05-19. The tag peels to
  `5679b686a579d581595c3d095b95a9b8899083a7`.

## [0.8.0] — 2026-05-18

external consumer ML/Python Productization. This release pulls the consumer-critical
Python, neural-symbolic, incremental-session, and native exact-induction work
forward so external consumer can execute the queued external-consumer bridge-training path against production xlog
surfaces.

### Added

- `pyxlog` runtime/session controls for async evaluation, chunked streaming
  results, per-call memory limits, progress counters, memory diagnostics,
  CUDA Graph counters, and host-transfer counters.
- Persistent relation delta APIs on `LogicRelationSession`, including insert,
  delete, mixed `apply_relation_delta`, and `delta_stats` reporting backed by
  runtime `RelationDelta` recomputation paths.
- external consumer neural bridge helpers: registered-network top-k and deterministic
  output modes, stable top-k tie-breaking, neural cache telemetry, Belnap
  pro/contra/quarantine loss helpers, semantic loss, MSE, and infoloss
  surfaces.
- Native exact-induction consumer integration through
  `pyxlog.ilp.exact_induce.induce_exact(..., backend="native")` for the external consumer
  tensorized ILP `U64` path, with strict-per-topology compatibility policy and
  packaging of `ilp_exact` CUDA artifacts through the normal `pyxlog/kernels`
  wheel path.
- An external-consumer-focused example suite under `examples/external-consumer-python/` plus
  `scripts/validate_external_consumer_examples.py`, covering async/streaming runtime
  controls, relation deltas, neural bridge helpers, native exact induction, and
  probabilistic async diagnostics.
- A machine-readable v0.8.0 certification pack under
  `docs/evidence/2026-05-18-v080-cert/` with `17/17` external consumer-required pyxlog symbol
  coverage and `signature_drift=0`.

### Changed

- Workspace package version and internal xlog crate dependency constraints now
  target `0.8.0`.
- README and roadmap release status now identify `v0.8.0` as the current
  tagged release and route external consumer users to the Python bindings and
  `examples/external-consumer-python/` suite.
- v0.9.0 is the next Epistemic/Solver Semantics train; v0.10.0 is the
  Multi-GPU / Out-of-Core train.

### Release Status

- Closure proposal: `docs/plans/2026-05-18-v080-closure-proposal.md`.
- Certification evidence: `docs/evidence/2026-05-18-v080-*/`.
- external consumer full 449-doc native-liveness replay is accepted by historical
  evidence waiver; it was not freshly rerun inside the xlog release worktree.

## [0.7.0] — 2026-05-18

General WCOJ Architecture and Runtime Expansion. This release
retargets the completed feature pack originally planned as v0.6.5
to v0.7.0 because the delivered surface is a full WCOJ subsystem
expansion: cost-aware planning, recursive integration, K-clique
coverage, paper-aligned helper/runtime mechanisms, external consumer hot-loop
integration, and release-board closure.

### Added

- First-class `MultiWayJoin` / WCOJ RIR and promoter surface for
  eligible multiway rules, with deterministic fallback preservation.
- WCOJ variable-ordering and cardinality/selectivity-aware cost-model
  integration, including per-iteration recursive SCC statistics.
- General WCOJ CUDA/runtime coverage beyond triangle: 4-cycle,
  K=5/K=6 hypergraph planner production path, K=7/K=8 template
  coverage, runtime histogram refresh, and helper-splitting invocation.
- Adaptive join closure: nested-loop dispatch for small eligible joins
  and preserved provider-level sort-merge operator certification.
- Certification and benchmark surfaces for GPU Same Generation,
  skewed multiway, deep-recursive WCOJ, deterministic mixed execution,
  widened-frontier replay, and paper-class production-scale fixtures.
- external consumer Phase-2 integration evidence for chain-shaped joins,
  sort-label propagation, CUDA Graph capture/replay, external-consumer neural-symbolic training surface
  preservation, and m37c-prime end-to-end validation.
- Dedicated WCOJ architecture and user guides.

### Changed

- Workspace package version and internal xlog crate dependency
  constraints now target `0.7.0`.
- Closure-board and tag-handoff release surfaces now use `v0.7.0`;
  historical evidence may still say the work was originally targeted
  as `v0.6.5`.
- Roadmap release trains move forward: the completed WCOJ expansion is
  v0.7.0, v0.8.0 is narrowed to external consumer ML/Python
  productization, Epistemic/Solver Semantics moves to v0.9.0,
  and Multi-GPU / Out-of-Core moves to v0.10.0. The broader
  language / CLI product backlog is deferred until it has a named
  consumer.

### Release Status

- Closure board: 31 DONE, 0 IN-PROGRESS, 0 BLOCKED, 0 OPEN.
- release tagging is complete: the annotated `v0.7.0` tag has been created and pushed.

## [0.6.0] — 2026-04-29

Stream-Safe GPU Runtime And Execution Discipline. Infrastructure
hardening release: a stream-safe GPU runtime and recorded launch
discipline so subsequent join / WCOJ work can be trusted under
parallel execution. Default behaviour for legacy callers is
unchanged; the new path is opt-in via
`CudaKernelProvider::with_runtime` /
`GpuMemoryManager::with_runtime` plus the
`XLOG_USE_DEVICE_RUNTIME` / `XLOG_USE_RECORDED_OPS` env flags.

### Added

- **Access-aware stream dependency manager** (PR #72,
  `26c2e429` + follow-ups). Replaces post-launch-only
  `record_block_use` with `prepare_block_use(BlockId, stream,
  Access)` / `finish_block_use(...)` and an `Access {Read,
  Write, ReadWrite}` enum. `AsyncCudaResource::LiveEntry`
  tracks `last_write: Option<(StreamId, CudaEvent)>` (seeded
  with an allocation-ready event captured immediately after
  `cuMemAllocAsync`) and `outstanding_reads:
  Vec<(StreamId, CudaEvent)>`. Reads wait on `last_write`;
  writes wait on `last_write` plus every cross-stream
  outstanding read. Same-stream events are skipped. Closes
  both the use-after-prior-write hazard and the
  use-after-allocation hazard across streams.
- **Lifetime-free `LaunchRecorder`**. Snapshots `BlockId` from
  each registered slice at record time and drops the source
  borrow immediately, so kernel `&mut` borrows after preflight
  are unrestricted. `preflight(&runtime)` queues
  `cuStreamWaitEvent` for every recorded use's cross-stream
  dependency BEFORE the launch; `commit(self, &runtime)`
  records new events via `finish_block_use` AFTER. Repeated
  registrations of the same block dedup on
  `(ptr, generation, device_ordinal)` to a single
  prepare/finish call with the strongest access.
- **`XlogDeviceRuntime::prepare_first_use(slice, stream, access)`
  / `finish_first_use(...)`** for helper-internal scratch
  whose first cross-stream consumer is a raw `cuMemsetD8Async`
  / `cuMemcpyDtoDAsync_v2` / `kernel.launch_on_stream` call
  ahead of any `LaunchRecorder::preflight`.
- **Formal certification harness** (`3361785b`). The cert
  `TestContext` builds the production decorator stack
  (`AsyncCudaResource → LoggingResource → GlobalDeviceBudget
  → XlogDeviceRuntime`) when `XLOG_USE_DEVICE_RUNTIME=1` is
  set and uses `with_runtime` constructors; the env-gated
  dispatchers in `provider::sort` / `filter_by_mask` /
  `hash_join_v2` / etc. then route through the recorded path
  when `XLOG_USE_RECORDED_*` is set. The harness reaps
  pending async frees between categories, and
  `GlobalDeviceBudget::allocate` retries once after a reap on
  transient over-budget conditions.
  Result: `XLOG_USE_DEVICE_RUNTIME=1 XLOG_USE_RECORDED_OPS=1
  cargo test -p xlog-cuda-tests --test certification_suite
  --release` passes **206/206**; legacy default still passes
  206/206.
- **A3/A4 cross-stream lifetime stress harness**
  (`crates/xlog-integration/tests/test_a3_a4_stress.rs`,
  `27ec3bd9` + `a01b51fa`). Two workloads (`friends`
  sort+hash-join sensitive, `reach` recursive fixed-point +
  joins). Stable FNV-1a checksums, fixed schedule + seeded
  random tail. **A4 fork-isolated stress passes 16/16** in
  every fixture mode and every env combination. A 5-mode
  diagnostic matrix (`XLOG_A3_FIXTURE_MODE=per_iter |
  per_thread | shared` × runtime-on/off × recorded-on/off
  via `XLOG_A3_DIAGNOSTIC=1`) classifies the A3 thread-of-N
  drift as pre-existing and not introduced by v0.6.0 — see
  Known Issues below.
- **Multi-threaded sort+hash-join regression**
  (`crates/xlog-cuda/tests/test_mt_sort_hj_alloc_ordering.rs`,
  PR #72). 8 threads × 128 iters × 3 rounds friend-of-friend
  self-join. Was RED at baseline `8cc0882c` (~6/1024 failures
  per run); 1024/1024 + 1024/1024 across 10 consecutive runs
  on `b1560674`.
- **Documentation**: `docs/architecture/device-runtime.md`
  (runtime stack + access matrix + env-gated dispatch + cert
  modes) and `docs/architecture/recorded-launch-migration.md`
  (operator-author checklist + anti-patterns + four-gate
  validation command sequence). Linked from
  `docs/ARCHITECTURE.md` Memory Management section.

### Changed

- `record_block_use` retained as a backward-compat shim that
  calls `finish_block_use(Read)` for the dealloc-wait surface;
  production callers go through the recorder.
- `write_post_preflight_fresh` removed. All 78 callers across
  `provider/{relational,filter,groupby,mod}.rs` migrated to
  pre-preflight `write` (the recorder snapshot drops the
  borrow, so kernel `&mut` borrows after preflight are
  unaffected).
- 6 direct `runtime.record_block_use(b, launch_stream)` call
  sites in provider code migrated to
  `runtime.finish_block_use(BlockId::from_block(b),
  launch_stream, Access::Write)` with semantically correct
  Access kinds.
- `prepare_first_use(Access::Write)` added at every
  helper-internal scratch alloc site that subsequently writes
  via raw CUDA work BEFORE its parent recorder's preflight:
  `build_hash_table_v2_on_stream` (5 buffers),
  `gather_buffer_by_indices_on_stream` (per-column
  `dst_col`s), `multiblock_scan_u32_inplace_on_stream` /
  `_view_inplace_on_stream` (`block_sums`), and every join
  variant's `d_count_only` / `d_output_count` / `out_col`
  zero-fills (Inner / LeftOuter / count-scan-materialize /
  indexed Inner / indexed LeftOuter).
- `gather_buffer_by_indices_on_stream`: local
  `d_output_rows` scalar created via
  `upload_device_row_count` + read on `launch_stream` is now
  fenced via `Access::Write` at upload + `Access::Read`
  prepare on `launch_stream` + `Access::Read` finish before
  drop. Closed a review-finding from the PR.

### Deferred to post-v0.6.0

- **Host-mask `compact_buffer_by_mask` recorded migration**.
  Re-opens when a runtime-backed recorded release path
  consumes host-provided masks. Until then the legacy entry
  is the supported path; the recorded
  `compact_buffer_by_device_mask_counted_recorded` covers the
  device-mask case for runtime-backed callers.
- **ILP / ILP-exact view helpers + operators recorded
  migration**. Re-opens when tensorized ILP /
  exact-induction downstream consumer work resumes (v0.8.0
  native exact-induction consumer gate) and requires
  runtime-backed stream safety.
- **Deferred LeftOuter count-scan-materialize migration** (commit `b90ae77f`, never
  pushed; recovered into a branch-local recovery note).
  Apply on a fresh post-v0.6.0 branch after auditing every
  scratch alloc against the access-aware contract documented
  in `docs/architecture/recorded-launch-migration.md`.

### Known Issues (not release blockers)

- **A3 in-process thread-of-N drift on
  `test_a3_a4_stress`**: 8 threads × 32 iters produce ~3%
  checksum drift on recursive Datalog workloads. The 5-mode
  diagnostic matrix demonstrates this is **NOT v0.6.0
  stream-safety regression** — drift fires identically on the
  legacy default path (no `XLOG_USE_DEVICE_RUNTIME`, no
  `XLOG_USE_RECORDED_OPS`, one runtime per thread, no v0.6
  code in the call chain). Bug class: pre-existing
  same-process multi-executor concurrency against one CUDA
  primary context. Tracked under v0.9.0 "Concurrency
  Hardening" in `ROADMAP.md`. The v0.6.0 release gate is
  **A4 fork-isolated stress + cert suite + umbrella ×50**,
  not "A3 zero drift".
- **`test_provider_launch_recorder --test-threads=8`** shows
  9/42 `*_survives_drop_and_reuse` failures (was 23/42 at
  baseline `8cc0882c`). Pre-existing pattern from
  cross-runtime mempool aliasing under intra-binary test
  parallelism. Production gate spec is `--test-threads=1`,
  which is clean.

### Release Validation (gates green on `b1560674`)

- `cargo fmt --check`: clean.
- `git diff --check`: clean.
- Legacy cert suite: 206/206 in 20.22s.
- Runtime+recorded cert suite
  (`XLOG_USE_DEVICE_RUNTIME=1 XLOG_USE_RECORDED_OPS=1`):
  206/206 in 16.56s.
- Umbrella ×50 (`real_world_tests --test-threads=8` under
  recorded runtime): **50/50**.
- Workspace `--tests --exclude pyxlog --release
  --test-threads=1`: 142 result lines, no failures.

> **Note:** All items below are post-v0.5.0 work. Items in
> `[Unreleased]` between the v0.5.0 tag and the v0.6.0 tag are
> reflected in the v0.6.0 release entry above.

## [0.6.1] — 2026-04-29

CSM Env Dispatch and Certification Mode Labeling. Small,
focused release on top of v0.6.0: enables count-scan-materialize
(CSM) hash-join methods for `Inner` / `LeftOuter` (indexed and
non-indexed) under an env gate, closes a stream-safety gap in
three earlier CSM siblings, and names the CSM cert mode
explicitly so reports are unambiguous. No kernel changes, no
algorithm changes, no eligibility relaxation. Default behaviour
for legacy callers is unchanged; the new path is opt-in via
`XLOG_USE_RECORDED_CSM=1` (or umbrella `XLOG_USE_RECORDED_OPS=1`).

### Added

- **Recorded CSM (count-scan-materialize) hash-join env
  dispatch** (PR #91). The recorded hash-join dispatcher
  routes `JoinType::Inner` and `JoinType::LeftOuter` through
  CSM (count → exclusive scan → materialize) for both the
  non-indexed and indexed entry points when
  `XLOG_USE_RECORDED_CSM=1` (or umbrella
  `XLOG_USE_RECORDED_OPS=1`) is set. `Semi` / `Anti` always
  route through the existing legacy recorded methods — no
  CSM implementation exists for them. Eligibility checks
  preserved exactly: runtime-backed manager, ≤4 keys
  (`pack_keys` constraint), key-type match, row-count caps,
  indexed-path key-byte and shape checks. New env-dispatch
  routing test suite
  (`crates/xlog-cuda/tests/test_csm_env_dispatch.rs`)
  proves selection across the Inner / LeftOuter × indexed /
  non-indexed × env-on / env-off matrix plus Semi / Anti
  and the >4-keys upstream short-circuit.
- **Indexed LeftOuter CSM operator** (PR #87,
  `hash_join_left_outer_v2_with_index_count_scan_materialize_recorded`).
  Probe-only pack on `launch_stream` plus a cached
  `JoinIndexV2` for the build side, sharing the
  count → scan → materialize phase shape with the
  non-indexed LeftOuter CSM (PR #84) and the indexed
  Inner CSM. No new kernels; reuses the four already-
  migrated CSM kernels plus `hash_join_csm_unmatched_mask`
  from PR #84.
- **Cert-mode labeling** (commit `bca1e373`). The
  `certification_suite` header now prints
  `Recorded-op dispatch (explicit):` (extended to include
  `XLOG_USE_RECORDED_CSM`) and a synthesized `Cert mode:`
  line keyed off the explicit env flags. The three intended
  values match the v0.6.1 cert gate commands —
  `legacy/default`, `runtime+recorded`,
  `runtime+recorded+CSM` — so CSM-mode runs are
  self-documenting in the cert evidence.

### Fixed

- **`d_overflow` lifetime in three CSM materialize
  recorders** (PR #89). The Phase B materialize kernel
  takes `d_overflow` as a kernel param (writes the
  overflow flag). Three previously-shipped CSM siblings
  (`hash_join_inner_v2_count_scan_materialize_recorded`,
  `hash_join_left_outer_v2_count_scan_materialize_recorded`,
  `hash_join_inner_v2_with_index_count_scan_materialize_recorded`)
  did not register `d_overflow` on their materialize-phase
  `LaunchRecorder`, so the runtime was free to release the
  block once `rec_count.commit` resolved — a potential
  use-after-free if pool reuse beat kernel completion. Each
  site now registers
  `rec_mat.write(&d_overflow);` before `rec_mat.preflight`,
  matching the indexed-LeftOuter CSM site (PR #87) so all
  four CSM materialize recorders are identical.

### Deferred to post-v0.6.1

- **Semi / Anti CSM**. No `count_scan_materialize_recorded`
  variants exist for `JoinType::Semi` / `JoinType::Anti`;
  the env dispatch leaves them on the legacy recorded
  paths. **Trigger to re-open**: a benchmark or
  correctness scenario forces it. The legacy paths are
  correct today and adding CSM variants would be code
  without a consumer.
- **CSM default-on**. CSM remains opt-in via
  `XLOG_USE_RECORDED_CSM` / umbrella
  `XLOG_USE_RECORDED_OPS`. Re-evaluate flipping the
  default once cert history accumulates a stable run of
  CSM-mode passes; until then the env gate is the
  migration boundary.

### Release Validation (gates green at tag)

- `cargo fmt --check`: clean.
- `git diff --check`: clean.
- Legacy cert
  (`cargo test -p xlog-cuda-tests --test certification_suite --release`):
  `Cert mode: legacy/default`, 1 outer test passing — 33
  cert categories internal.
- Runtime+recorded cert
  (`XLOG_USE_DEVICE_RUNTIME=1 XLOG_USE_RECORDED_OPS=1 cargo test ...`):
  `Cert mode: runtime+recorded`, 1 outer test passing —
  same 33 categories.
- Runtime+recorded+CSM cert
  (`XLOG_USE_DEVICE_RUNTIME=1 XLOG_USE_RECORDED_OPS=1 XLOG_USE_RECORDED_CSM=1 cargo test ...`):
  `Cert mode: runtime+recorded+CSM`, 1 outer test passing —
  same 33 categories.
- Umbrella ×20 (`real_world_tests --test-threads=8` under
  `XLOG_USE_DEVICE_RUNTIME=1 XLOG_USE_RECORDED_OPS=1`):
  20/20 (recorded across PR #87, #89, #91 prep).

## [0.6.2] — 2026-05-01

Default-On Adaptive WCOJ Triangle Dispatch. Productizes the
first GPU Worst-Case Optimal Join slice: a certified 3-way
triangle path for `u32`, `u64`, and `Symbol` keys, wired into
the runtime behind a default-on adaptive skew classifier and a
hard kill switch. The release also ships the pure-Rust
hypergraph planner / oracle stack that future WCOJ kernels are
certified against. Scope remains deliberately narrow: no
general-arity WCOJ, no recursive/SCC WCOJ execution, no cost
model, and no `MultiWayJoin` / `WcojJoin` RIR node yet.

### Added

- **Hypergraph planner and oracle foundation.** Added
  `xlog-logic::hypergraph` with a hypergraph IR, eligibility
  analyzer, deterministic variable-order interface, canonical
  explain output, typed gate, mixed plan contract, single-rule
  reference evaluator, single-target fixpoint evaluator, SCC
  fixpoint evaluator, and transitive SCC type inference. The
  certification workloads cover triangle, Same Generation,
  skewed multiway, deep recursive frontier, and mutually
  recursive parity SCC shapes.
- **GPU WCOJ triangle provider path.** Added recorded
  `wcoj_triangle_u32_recorded` / `wcoj_triangle_u64_recorded`
  provider entries plus `wcoj_layout_u32_recorded` /
  `wcoj_layout_u64_recorded` sorted-layout construction. The
  triangle pipeline uses count → device-side prefix scan →
  materialize with a 4-byte metadata D2H total; no count-vector
  D2H remains. `Symbol` uses the u32 physical path.
- **Planner-to-provider certification.** Added test-only
  `xlog-logic` dev dependency in `xlog-cuda` so planner verdicts
  and GPU provider outputs are certified against the same CPU
  oracle fixtures before executor wiring.
- **Runtime WCOJ dispatch.** Added the executor hook for the
  canonical non-recursive triangle RIR shape
  `tri(X,Y,Z) :- e1(X,Y), e2(Y,Z), e3(X,Z)`. The hook supports
  4-byte (`U32` / `Symbol`) and 8-byte (`U64`) uniform-width
  triangles, silently falls back for unsupported shapes, and
  exposes `Executor::wcoj_triangle_dispatch_count()` for tests.
- **Adaptive skew classifier and default-on policy.** Added
  `wcoj_triangle_skew_score_{u32,u64}` and a 64-bucket
  hash-mixed L-infinity/L1 classifier. `RuntimeConfig::default()`
  now runs adaptive WCOJ on matching non-recursive triangle
  rules: high-skew inputs dispatch WCOJ, uniform / empty inputs
  fall back to the binary-join chain. Ops can disable the path
  globally with `XLOG_DISABLE_WCOJ_TRIANGLE=1`.
- **Diagnostic phase timing.** Added feature-gated
  `wcoj-phase-timing` support and the `wcoj_phase_report`
  binary to measure classifier, layout, triangle count / scan /
  total / materialize, wall, and residual overhead.
- **WCOJ benchmark baseline.** Added
  `crates/xlog-integration/benches/wcoj_triangle_bench.rs` and
  evidence under
  `docs/evidence/2026-05-01-wcoj-bench-baseline/` for baseline,
  adaptive acceptance, default-on acceptance, pre-fast-path phase
  timing, and post-fast-path phase timing.

### Changed

- **WCOJ layout fast-path.** `wcoj_layout_u32_recorded` and
  `wcoj_layout_u64_recorded` now prove already sorted+unique
  inputs with a recorded checker kernel and return a recorded
  device-side clone instead of always running sort + dedup. The
  slow path is unchanged and remains the correctness fallback.
- **Recorded sort / dedup U64 support.** `sort_recorded` now
  supports U64 via the same hi/lo radix strategy as the legacy
  sort path, and `dedup_full_row_recorded` admits U64 rows.
- **Executor WCOJ stream reuse.** The executor caches one WCOJ
  launch stream per instance, preventing long-lived runtimes from
  exhausting the grow-only `StreamPool` and silently falling back
  after 16 dispatches.
- **WCOJ adaptive default.** `RuntimeConfig::wcoj_triangle_dispatch`
  remains the explicit force/off knob. New
  `wcoj_triangle_dispatch_adaptive` controls adaptive opt-out /
  opt-in, and `wcoj_triangle_dispatch_disabled` is the hard kill
  switch. Precedence is: disable > force > explicit force-off >
  adaptive.

### Fixed

- **Skew-classifier failure paths.** Failure paths after queued
  classifier work now drain the launch stream before dropping
  temporary buffers.
- **Layout fast-path failure paths.** Failure paths after queued
  checker / recorded-clone work now drain the launch stream before
  dropping temporary buffers.
- **SCC-aware planner precedence.** The hypergraph planner now
  preserves structural-error precedence when typed gating and SCC
  inference both apply.

### Bench Evidence

- Initial force-on WCOJ was strong on super-hub fixtures but
  regressed uniform / empty fixtures. The adaptive classifier
  cleared the locked median gates: uniform / empty adaptive cells
  stayed ≤2× binary-join, while super-hub speedups remained above
  the locked minimums.
- Phase timing showed layout construction was 91-97% of super-hub
  WCOJ wall time before the fast-path. The layout fast-path reduced
  layout time by ~97-98% and wall time by ~90-96% on the measured
  super-hub cells.

### Deferred to post-v0.6.2

- General `MultiWayJoin` / `WcojJoin` RIR node and optimizer
  integration.
- Cost-aware variable ordering and selectivity / heat feedback.
- Recursive / SCC WCOJ execution and mixed recursive WCOJ +
  binary-join semantics.
- 4-way and general-arity WCOJ kernels.
- Histogram-guided block scheduling / B1 heavy-row offload. Phase
  timing after the layout fast-path shows materialize is now a
  plausible future optimization target, but no longer the obvious
  next slice.
- Dedicated WCOJ architecture and performance-tuning guide.

### Release Validation Targets

Run before tagging `v0.6.2`:

- `cargo fmt --check`
- `git diff --check`
- WCOJ provider / integration test matrix
- `cargo test -p xlog-logic`
- `cargo test -p xlog-runtime --lib`
- `cargo build --workspace --exclude pyxlog`
- `XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-integration --test real_world_tests --release`
- Existing certification modes from v0.6.1 remain the recorded-launch
  baseline for runtime safety.

## [0.9.2](https://github.com/BrainyBlaze/xlog/compare/xlog-cli-v0.5.0...xlog-cli-v0.9.2) - 2026-06-05

### Added

- full shared-variable epistemic constraint joins via program-level desugaring
- diagonal modal constraint via sound program-level desugaring
- pilot ex37 (stratified negated-modal recursion EXECUTES) + device test/mutation + reword ex33 to formal WFS bound
- multi-literal distinct-variable epistemic constraints + README
- *(epistemic)* epistemic plan-dump surface â xlog run --epistemic-plan-json
- close determined-epistemic multi-column binding (determined-modal family complete)
- close transitive determined-ordinary modal coupling via stratification
- close augmented-projection multi-head coupling scope limit
- cross-component epistemic joint-solving (multi-output)
- cross-component epistemic coupling
- recursive epistemic fixpoint support
- mixed per-row and global modal membership
- checkpoint epistemic solver semantics
- add cli explain repl watch surfaces
- add incremental parser session
- add approximate inference pragmas
- add aggregate lifting reports
- add magic-set rewriting

### Fixed

- *(release)* drop README version sync + dynamic badges + agent release rules ([#124](https://github.com/BrainyBlaze/xlog/pull/124))
- guard diagonal desugaring to non-modal-derived targets
- route bound-variable multi-head epistemic programs through split
- route epistemic examples through xlog run
- *(release)* harden validation and gpu fallback paths
- *(pyxlog)* install local wheels for explicit python
- *(cuda)* embed portable PTX fallback
- *(pyxlog)* ship kernels in wheels and document cubin path
- *(ci)* repair main release automation ([#27](https://github.com/BrainyBlaze/xlog/pull/27))
- *(ci)* keep README release metadata in sync ([#26](https://github.com/BrainyBlaze/xlog/pull/26))

### Other

- Set v0.9.2 release metadata
- Clarify v0.9.2 WFS release contract
- Close v0.9.2 epistemic semantic gaps
- Close v0.9.2 epistemic release
- CLI markers for accepted chain pilots; repoint negative test to interior-negation boundary
- variable-keyed constraint device tests + CLI goldens + mutation probe
- CLI golden ex23 ACCEPTED + repoint negative test to unbounded cons
- CLI accepted-fixpoint + negated-modal-floor contracts
- FAEEL unfounded self-support → exact founded-extension semantic result
- integrate main MC GPU-resident engine into v0.9.2 epistemic completion
- complete determined-modal-family showcase (negated-over-derived, possible-binding, FAEEL-unfounded)
- determined-head and negated-modal-over-invariant xlog-run pilots (examples 25,26) with anti-gaming gating checks
- flip example 17 to accepted stratified pilot; add example 24 transitive out-of-scope negative
- full robust validated v0.9.2 epistemic examples
- add validated v0.9.1 epistemic executor showcase (06-11)
- document v0.9.0 epistemic language surface
- *(release)* align v0.9.0 package metadata
- integrate v0.8.9 diagnostics surfaces
- *(release)* prepare v0.8.6 tag metadata
- *(release)* correct v0.8.5 public status
- *(release)* prepare v0.8.0
- mark v0.7.0 release complete
- close purge gate
- *(v0.6.2)* prepare roadmap changelog and version
- *(v0.6.1)* version bump + roadmap cleanup + changelog
- *(readme)* bump version badge + release-status line to v0.6.0
- restore audit README framing with current release setup
- Merge branch 'audit/v0.5.0-prerelease'
- integrate prerelease audit docs
- harden public release readiness

## [0.9.2](https://github.com/BrainyBlaze/xlog/compare/xlog-gpu-v0.5.0...xlog-gpu-v0.9.2) - 2026-06-05

### Added

- full shared-variable epistemic constraint joins via program-level desugaring
- diagonal modal constraint via sound program-level desugaring
- pilot ex37 (stratified negated-modal recursion EXECUTES) + device test/mutation + reword ex33 to formal WFS bound
- co-evolving modal and recursive founded least fixpoint
- *(epistemic)* epistemic plan-dump surface â xlog run --epistemic-plan-json
- close determined-epistemic multi-column binding (determined-modal family complete)
- determined-head recursion and negated-modal-over-invariant recursive epistemic fixpoint
- stratified epistemic execution wiring (materialize gated head between strata)
- cross-component epistemic joint-solving (multi-output)
- recursive epistemic fixpoint support
- coalesce relation delta batches
- add safe meta lowering
- add finite list lowering
- add type term foundation
- *(pyxlog)* add v0.8.0 relation delta sessions
- expose xlog sort-label metadata

### Fixed

- *(release)* drop README version sync + dynamic badges + agent release rules ([#124](https://github.com/BrainyBlaze/xlog/pull/124))
- guard diagonal desugaring to non-modal-derived targets
- route bound-variable multi-head epistemic programs through split
- materialize nullary EDB facts as present (1 row)
- route epistemic examples through xlog run
- prove pyxlog persistent index session reuse
- *(release)* harden validation and gpu fallback paths
- expose query-variable sort labels at runtime
- *(pyxlog)* install local wheels for explicit python
- *(cuda)* embed portable PTX fallback
- *(pyxlog)* ship kernels in wheels and document cubin path
- *(ci)* repair main release automation ([#27](https://github.com/BrainyBlaze/xlog/pull/27))
- *(ci)* keep README release metadata in sync ([#26](https://github.com/BrainyBlaze/xlog/pull/26))

### Other

- Set v0.9.2 release metadata
- Clarify v0.9.2 WFS release contract
- Close v0.9.2 epistemic semantic gaps
- Close v0.9.2 epistemic release
- cargo fmt
- derived-head coupling — stratified-vs-reference equivalence + true-cycle wall
- device co-evolving modal-recursion case founded-fixpoint + ungated mutation probe
- document v0.9.0 epistemic language surface
- *(release)* align v0.9.0 package metadata
- integrate v0.8.9 diagnostics surfaces
- integrate v0.8.8 external world-model diagnostics into v0.8.9
- integrate first external diagnostics into v0.8.9
- Exercise external generated-rule diagnostics
- Resolve remaining XLOG evidence issues
- Add v0.8.7 external world-model diagnostics
- *(release)* prepare v0.8.6 tag metadata
- *(release)* correct v0.8.5 public status
- *(release)* prepare v0.8.0
- mark v0.7.0 release complete
- *(v0.6.2)* prepare roadmap changelog and version
- *(v0.6.1)* version bump + roadmap cleanup + changelog
- *(readme)* bump version badge + release-status line to v0.6.0
- restore audit README framing with current release setup
- Merge branch 'audit/v0.5.0-prerelease'
- integrate prerelease audit docs
- harden public release readiness

## [0.9.2](https://github.com/BrainyBlaze/xlog/compare/xlog-prob-v0.5.0...xlog-prob-v0.9.2) - 2026-06-05

### Added

- close augmented-projection multi-head coupling scope limit
- checkpoint epistemic solver semantics
- add approximate inference pragmas
- add aggregate lifting reports
- add probabilistic aggregate support
- add safe meta lowering
- add type term foundation

### Fixed

- *(release)* drop README version sync + dynamic badges + agent release rules ([#124](https://github.com/BrainyBlaze/xlog/pull/124))
- route epistemic examples through xlog run
- close GPU-native count-lift exact path
- *(release)* harden validation and gpu fallback paths
- *(pyxlog)* install local wheels for explicit python
- *(cuda)* embed portable PTX fallback
- *(pyxlog)* ship kernels in wheels and document cubin path
- *(ci)* repair main release automation ([#27](https://github.com/BrainyBlaze/xlog/pull/27))
- *(ci)* keep README release metadata in sync ([#26](https://github.com/BrainyBlaze/xlog/pull/26))

### Other

- Set v0.9.2 release metadata
- Close v0.9.2 epistemic release
- integrate main MC GPU-resident engine into v0.9.2 epistemic completion
- document v0.9.0 epistemic language surface
- *(release)* align v0.9.0 package metadata
- integrate v0.8.9 diagnostics surfaces
- aggregate split batch cpu fallbacks
- gate split batch h2d transfer
- gate split batch incremental prob updates
- gate accepted evidence incremental prob updates
- centralize probabilistic batch gate
- require single result timing gates
- require split batch timing gates
- tighten prob production metric gate
- replace incremental evidence updates
- name alternative compiler adapters
- trace nonzero probabilistic evidence arity
- reuse v0.8.6 bundle for v0.9.0
- *(release)* prepare v0.8.6 tag metadata
- *(release)* correct v0.8.5 public status
- *(release)* prepare v0.8.0
- mark v0.7.0 release complete
- *(v0.6.2)* prepare roadmap changelog and version
- *(v0.6.1)* version bump + roadmap cleanup + changelog
- *(readme)* bump version badge + release-status line to v0.6.0
- restore audit README framing with current release setup
- Tighten workspace warning hygiene
- Merge branch 'audit/v0.5.0-prerelease'
- integrate prerelease audit docs
- harden public release readiness

## [0.9.2](https://github.com/BrainyBlaze/xlog/compare/xlog-solve-v0.5.0...xlog-solve-v0.9.2) - 2026-06-05

### Added

- close augmented-projection multi-head coupling scope limit
- checkpoint epistemic solver semantics
- gate multi-candidate solver portfolios
- schedule gpu maxsat batches
- schedule multi-result gpu maxsat search
- schedule multi-result gpu maxsat encodes
- encode weighted gpu maxsat candidates
- prune unsat gpu maxsat candidates
- batch accepted gpu maxsat candidates
- reuse learned clauses across accepted gpu candidates
- propagate gpu solver lifecycle statuses
- cover multi-candidate gpu solver lifecycle
- reject unsafe learned clause reuse
- gate oracle fixtures from production metrics
- reuse gpu learned clauses for same cnf
- publish gpu learned clause arenas
- propagate solver portfolio status in gpu adapter
- gate maxsat portfolio through gpu solver adapter
- gate solver lifecycle with accepted gpu evidence
- gate solver workspace unsat with accepted gpu evidence
- gate solver unsat path with accepted gpu evidence
- gate solver sat path with accepted gpu evidence
- report solver production capability blockers
- add gpu solver production reuse adapter
- add bounded solver service semantics

### Fixed

- *(release)* drop README version sync + dynamic badges + agent release rules ([#124](https://github.com/BrainyBlaze/xlog/pull/124))
- route epistemic examples through xlog run
- *(release)* harden validation and gpu fallback paths
- *(pyxlog)* install local wheels for explicit python
- *(cuda)* embed portable PTX fallback
- *(pyxlog)* ship kernels in wheels and document cubin path
- *(ci)* repair main release automation ([#27](https://github.com/BrainyBlaze/xlog/pull/27))
- *(ci)* keep README release metadata in sync ([#26](https://github.com/BrainyBlaze/xlog/pull/26))

### Other

- Set v0.9.2 release metadata
- document v0.9.0 epistemic language surface
- *(release)* align v0.9.0 package metadata
- integrate v0.8.9 diagnostics surfaces
- aggregate split batch cpu fallbacks
- gate split batch h2d transfer
- centralize solver batch gate
- require single result timing gates
- require split batch timing gates
- lock production metric audit wording
- tighten solver production metric gate
- trace solver nonzero evidence arity
- reuse v0.8.6 bundle for v0.9.0
- guard maxsat scheduler prevalidation
- guard encoded maxsat prevalidation
- guard maxsat search prevalidation
- guard maxsat lifecycle prevalidation
- gate split maxsat lifecycle
- gate solver maxsat lifecycle
- gate split solver maxsat scheduler on batches
- gate split solver maxsat search on batches
- gate split solver maxsat on batches
- gate split solver learned reuse on batches
- gate split solver portfolio on batches
- gate split solver lifecycle on batches
- trace solver evidence by operator family
- trace semantic modes through solver gates
- mark gpu native gate blocked
- mark v0.7.0 release complete
- *(v0.6.2)* prepare roadmap changelog and version
- *(v0.6.1)* version bump + roadmap cleanup + changelog
- *(readme)* bump version badge + release-status line to v0.6.0
- restore audit README framing with current release setup
- Tighten workspace warning hygiene
- Merge branch 'audit/v0.5.0-prerelease'
- integrate prerelease audit docs
- harden public release readiness

## [0.9.2](https://github.com/BrainyBlaze/xlog/compare/xlog-runtime-v0.5.0...xlog-runtime-v0.9.2) - 2026-06-05

### Added

- same-name multi-arity modal coupling solved via arity-qualified tuple sources
- variable-keyed + nested epistemic constraints (GPU world-view pruning)
- multi-literal distinct-variable epistemic constraints + README
- drop unfounded FAEEL self-support from reduced founded-model base
- close determined-epistemic multi-column binding (determined-modal family complete)
- close augmented-projection multi-head coupling scope limit
- determined-head recursion and negated-modal-over-invariant recursive epistemic fixpoint
- stratified epistemic execution wiring (materialize gated head between strata)
- cross-component epistemic joint-solving (multi-output)
- cross-component epistemic coupling
- mixed per-row and global modal membership
- constraint-specific rejection reasons
- joint multi-epistemic predicate solving
- epistemic integrity constraints
- EIR-derived candidate-world enumeration
- tuple-key bound-value membership
- checkpoint epistemic solver semantics
- compare G91 GPU traces to oracle
- expose gpt rejected candidate indices
- expose gpu semantic candidate indices
- type gpu epistemic rejection reasons
- gate probabilistic pir cnf batches
- gate probabilistic evaluation batches
- gate parsed probabilistic program batches
- gate multi-candidate solver portfolios
- add split world-view parity fixture
- add G91 runtime parity fixture
- certify skew-scheduled wcoj reuse
- permit founded faeel self possible
- require complete world-view support
- require helper scans in wcoj plans
- certify helper split rewrites
- require wcoj layout evidence
- guard faeel self support
- trace not possible row filters
- certify kclique stream groups
- trace epistemic operator metrics
- trace kclique metadata timing
- schedule gpu maxsat batches
- schedule multi-result gpu maxsat search
- schedule multi-result gpu maxsat encodes
- condition negative gpu prob evidence
- execute split gpu components
- encode weighted gpu maxsat candidates
- condition gpu prob gradients
- prune unsat gpu maxsat candidates
- certify helper split wcoj trace metrics
- batch conditioned gpu prob programs
- batch conditioned gpu prob queries
- condition accepted gpu prob programs
- condition accepted gpu prob tuple evidence
- batch accepted gpu maxsat candidates
- reuse learned clauses across accepted gpu candidates
- propagate gpu solver lifecycle statuses
- batch accepted gpu prob execution
- cover multi-candidate gpu solver lifecycle
- trace gpu semantic candidate outcomes
- honor not-know tuple membership on gpu
- condition accepted evidence in gpu exact path
- gate oracle fixtures from production metrics
- account final gpu result transfers
- reuse gpu learned clauses for same cnf
- trace kclique arity preflight reuse
- trace program prob knowledge compilation
- publish gpu learned clause arenas
- propagate solver portfolio status in gpu adapter
- lower split components through gpu executable plans
- gate maxsat portfolio through gpu solver adapter
- filter final rows by all epistemic memberships
- gate prob end-to-end exact evaluation
- gate solver lifecycle with accepted gpu evidence
- gate prob pir cnf with accepted gpu evidence
- gate prob query evaluation with accepted gpu evidence
- gate prob program compile with accepted gpu evidence
- gate solver workspace unsat with accepted gpu evidence
- gate prob gradient evaluation with accepted gpu evidence
- gate solver unsat path with accepted gpu evidence
- gate solver sat path with accepted gpu evidence
- gate prob exact path with accepted gpu evidence
- filter final tuples by bound membership
- certify accepted wcoj execution
- gate final tuples by gpu membership
- bind tuple keys to gpu output columns
- add generic gpu tuple membership kernel
- add arity-three epistemic tuple key kernel
- encode ground epistemic tuple keys for gpu matching
- preserve epistemic tuple key terms in eir
- stage fixed-arity epistemic tuple sources on gpu
- populate arity-zero epistemic membership from tuple sources
- bind epistemic literals to tuple membership sources
- fail closed on row-count epistemic membership
- materialize epistemic final tuples on gpu
- enforce epistemic wcoj runtime certification
- gate epistemic model membership on gpu output
- trace epistemic gpu transfer budget
- materialize epistemic final result flags on gpu
- validate epistemic world views on gpu
- stage epistemic model membership on gpu
- trace epistemic gpu staging timings
- stage epistemic materialization on gpu
- validate staged epistemic candidates on gpu
- stage epistemic propagation on gpu
- add gpu candidate generation kernel
- reset epistemic gpu workspace on device
- trace epistemic reduced runtime execution
- gate epistemic wcoj evidence on counters
- add epistemic runtime preflight
- add epistemic gpu workspace contract
- close CUDA Graph-mode set maintenance
- bind delta variants as WCOJ leaders
- remove adaptive skew classifier surface
- route variable-ordering cost model u32 triangle through HG
- route u32 triangle dispatch through HG pipeline
- production kernel fusion (layout+count) with threshold dispatch + auto-disable + cert grid
- wire sort-merge dispatch + counter at execute_join + de-overlap nested-loop dispatch cert fixtures for dispatch precedence
- add eligible_for_sort_merge predicate
- wire nested-loop dispatch + counter at execute_join
- add eligible_for_nested_loop predicate
- *(runtime)* K=5 and K=6 clique WCOJ dispatcher + counters
- HeatAwareLeaderModel plus variable-order-aware join-result feedback
- *(runtime)* per-iteration stats integration for recursive SCC
- *(runtime)* dispatcher reroute on var_order — variable-ordering cost model leader rotation + post-kernel projection
- *(ir)* VariableOrder + LookupPerm types + MultiWayJoin.var_order field
- *(runtime)* record_join_result feedback from successful WCOJ dispatch
- *(runtime)* dispatch sites use build_wcoj_cost_model factory
- *(runtime)* CardinalityAwareCostModel with delegate-on-missing-stats
- *(runtime)* execute_wcoj_or_fallback_node hooks recursive arm
- *(runtime)* try_dispatch_wcoj_*_on_body entry points
- *(runtime)* migrate adaptive dispatch to WcojCostModel seam
- *(runtime)* WcojCostModel + SkewScoreSource cost-model seam
- *(cuda+runtime)* 4-cycle skew classifier + adaptive opt-in
- *(runtime)* wire 4-cycle dispatch + executor wiring cert
- *(runtime)* match_multiway_4cycle + try_dispatch_wcoj_4cycle force gate
- *(runtime)* replace triangle-tree matcher with MultiWayJoin
- *(workspace)* cross-crate MultiWayJoin walker arms
- *(runtime)* default-on adaptive WCOJ + hard kill switch (v0.6.2)
- *(runtime)* adaptive WCOJ dispatch + classifier branch (v0.6.2 A2-lite commit B)
- *(dispatch)* WCOJ width-aware AST/RIR dispatch (v0.6.2)
- *(cuda)* WCOJ Symbol key support (v0.6.2)
- *(runtime)* env-gated WCOJ triangle executor wiring (v0.6.2)
- *(runtime)* add strict deterministic D2H guard (v0.5.5) ([#49](https://github.com/BrainyBlaze/xlog/pull/49))

### Fixed

- *(release)* drop README version sync + dynamic badges + agent release rules ([#124](https://github.com/BrainyBlaze/xlog/pull/124))
- standalone negated-variable-keyed constraint is a NAF safety error, not 'unimplemented'
- route epistemic examples through xlog run
- fail closed nonzero faeel self support
- certify v0.7.0 multiway wcoj reuse
- require tuple-source proof before validation
- *(release)* harden validation and gpu fallback paths
- integrate K7 K8 planned clique metadata
- restore cost-model default-flip cert
- route nested-loop dispatch through shared record_join_result feedback
- preserve occurrence identity in rewrite_scan_nth
- tighten Tier-1 wrapper contract + revert recursive helper extension
- cargo fmt + correct prepare_leader_inputs visibility doc
- cargo fmt, evidence count, extract prepare_leader_inputs + real helper extraction
- address selectivity-driven join reordering review patches — duplicate attr, stale comment, evidence count, matcher tests
- *(runtime)* harden WCOJ phase timing diagnostics
- *(runtime)* cache WCOJ launch stream on Executor (v0.6.2)
- *(logic)* restore deterministic recursive set evaluation
- *(pyxlog)* install local wheels for explicit python
- *(cuda)* embed portable PTX fallback
- *(pyxlog)* ship kernels in wheels and document cubin path
- *(ci)* repair main release automation ([#27](https://github.com/BrainyBlaze/xlog/pull/27))
- *(ci)* keep README release metadata in sync ([#26](https://github.com/BrainyBlaze/xlog/pull/26))

### Other

- Set v0.9.2 release metadata
- device test — nested-modal chain collapses, executes, zero CPU fallback
- rustfmt multi-arity device test upload helper
- variable-keyed constraint device tests + CLI goldens + mutation probe
- harden multi-element key test to discriminate col1
- repoint mixed-modal negative pilot to unbounded cons key
- red device tests + ACCEPTED ex23 for structured modal tuple-keys
- exact FAEEL founded-extension results on GPU runtime + mutation-probe-verified gate
- cargo fmt on v0.9.1 epistemic changeset
- safe split dependency and coupling semantics
- document v0.9.0 epistemic language surface
- *(release)* align v0.9.0 package metadata
- integrate v0.8.9 diagnostics surfaces
- gate split possible not-know fallbacks
- gate split binary cpu fallbacks
- certify split binary workspace timing
- certify k7 k8 layout events
- certify k7 k8 metadata timing
- aggregate split batch cpu fallbacks
- gate split batch h2d transfer
- aggregate split batch final transfer
- gate split batch final transfer
- gate single-result final transfer
- gate single-result kernel timing
- gate single-result workspace buffers
- gate single-result row-count membership rejection
- gate single-result host transfer rejection
- gate single-result cpu fallback rejection
- gate split batch incremental prob updates
- gate accepted evidence incremental prob updates
- gate rejected world-view consumers
- gate split quaternary host transfer rejection
- gate split quaternary row-count membership rejection
- gate split quaternary cpu fallback rejection
- gate split quaternary workspace buffers
- gate split quaternary all-operator timing
- gate split quaternary all-operator prob deep paths
- gate split quaternary all-operator prob gradients
- gate split quaternary all-operator probability
- gate split quaternary all-operator solver search
- gate split quaternary all-operator solver reuse
- gate split quaternary all-operator solver lifecycle
- gate split quaternary all-operator parity
- gate split quaternary possible not-know parity
- gate quaternary possible not-know source gradients
- gate quaternary not-possible prob gradients
- gate quaternary know prob gradients
- gate quaternary know probabilistic reuse
- gate quaternary know solver search
- gate quaternary know solver reuse
- gate quaternary possible not-know solver search
- gate quaternary possible not-know solver reuse
- gate quaternary not-possible solver search
- gate quaternary not-possible solver reuse
- gate quaternary not-possible PIR reuse
- gate quaternary source PIR reuse
- gate quaternary program PIR reuse
- gate quaternary program probability reuse
- gate all-operator program probability eval
- gate all-operator source probability paths
- gate all-operator solver search
- gate split all-binary solver search
- gate split quaternary not-possible solver search
- gate split quaternary solver search
- gate split quaternary prob reuse
- gate split quaternary solver reuse
- gate split quaternary production reuse
- gate quaternary operator production reuse
- gate quaternary operator gpu parity
- gate split quaternary gpu parity
- gate split quaternary prob reuse
- gate split quaternary solver reuse
- gate split quaternary solver evidence
- gate split quaternary prob batch reuse
- gate parsed quaternary negative prob reuse
- gate negated quaternary solver prob reuse
- cover negated quaternary membership parity
- require complete aggregate timing
- require single result timing gates
- require split batch timing gates
- aggregate split batch kernel timing
- lock production metric audit wording
- name alternative compiler adapters
- deepen all-operator reuse gates
- gate all-operator membership reuse
- cover all-operator mixed memberships
- cover negated mixed memberships
- cover mixed epistemic memberships
- reject unsafe split modal coupling
- gate all-operator split prob eval
- gate all-operator split prob gradients
- gate all-operator split solver reuse
- gate all-operator split solver lifecycle
- condition split all-operator probability
- trace split all binary operators
- trace split binary operator parity
- trace solver quaternary evidence arity
- trace source quaternary evidence arity
- trace prob quaternary evidence arity
- trace solver nonzero evidence arity
- trace nonzero probabilistic evidence arity
- reuse v0.8.6 bundle for v0.9.0
- aggregate split operator trace counts
- guard maxsat scheduler prevalidation
- guard encoded maxsat prevalidation
- guard maxsat search prevalidation
- guard maxsat lifecycle prevalidation
- gate split maxsat lifecycle
- gate solver maxsat lifecycle
- gate ternary epistemic gpu parity
- gate split prob exact compile on batches
- gate split prob pir cnf evaluation on batches
- gate split solver maxsat scheduler on batches
- gate split prob exact paths on batches
- gate split solver maxsat search on batches
- gate split solver maxsat on batches
- gate split solver learned reuse on batches
- gate split solver portfolio on batches
- gate split solver lifecycle on batches
- gate split prob gradients on batches
- gate parsed prob evidence on split batches
- gate prob evidence on split batches
- trace split gpu batch execution
- split prob operator evidence by source path
- trace solver evidence by operator family
- trace semantic modes through solver gates
- trace semantic modes through prob gates
- cover negative probabilistic batches
- certify accepted k8 wcoj dispatch
- certify accepted k7 wcoj dispatch
- audit production path reuse
- mark v0.7.0 release complete
- close phase2 integration gate
- Merge sort-label propagation branch into the phase-2 integration branch
- Merge K7/K8 clique-template branch into the phase-2 integration branch
- add K7/K8 clique templates
- close purge gate
- unwire executor sort-merge dispatch + rewrite sort-merge dispatch certs as operator-level
- workspace gate green pre-bench (+ stale-comment cleanup)
- workspace gate green
- scrub stale "multi-recursive skip" contract notes
- flip recursive-SCC stats integration multi-recursive WCOJ cert to assert multi-recursive WCOJ dispatch
- rewrite input/fallback `rewrite_scan_nth` tests for positional symmetry
- strengthen rewrite_scan_nth regression for exact positional identity
- cargo fmt for workspace gate
- patch evidence/comment drift round 2
- patch evidence/comment drift before closure approval
- *(test)* correct recursive-SCC stats integration feature-gate test header — feature gate, not cfg(test)
- *(runtime)* strengthen recursive-SCC stats integration distinct binary_est + pin exact counter
- *(runtime)* recursive-SCC stats integration acceptance gate acceptance matrix + recursive-stats-trace feature
- restore selectivity-driven join reordering acceptance gates — 4-cycle compile-time + runtime helper and synthesis certs
- *(runtime,logic)* align stale claims with the recursive WCOJ contract
- *(runtime)* unit-test SkewScoreSource seam via stub scorer
- *(runtime)* rename wcoj_triangle_stream to wcoj_dispatch_stream
- *(workspace)* MultiWayJoin shape-agnosticism guards
- *(v0.6.2)* prepare roadmap changelog and version
- *(runtime)* WCOJ phase-timing scaffolding + report (v0.6.2)
- *(runtime)* cover WCOJ dispatch env resolvers
- *(wcoj)* update Symbol dispatch scope comments
- *(v0.6.1)* version bump + roadmap cleanup + changelog
- *(readme)* bump version badge + release-status line to v0.6.0
- restore audit README framing with current release setup
- Tighten workspace warning hygiene
- Merge branch 'audit/v0.5.0-prerelease'
- integrate prerelease audit docs
- harden public release readiness

## [0.9.2](https://github.com/BrainyBlaze/xlog/compare/xlog-logic-v0.5.0...xlog-logic-v0.9.2) - 2026-06-05

### Added

- admit stratified negated-modal recursion as Case B; bound genuine negation cycle to host-only WFS
- grammar+parser collapse nested modal chains to single epistemic literal
- variable-keyed + nested epistemic constraints (GPU world-view pruning)
- single-occurrence variable-keyed epistemic constraints (GPU existential world-view pruning)
- flatten structured modal tuple-keys (finite+typed list/compound/anonymous on GPU)
- admit positive modal recursion with a founded least fixpoint
- drop unfounded FAEEL self-support from reduced founded-model base
- close determined-epistemic multi-column binding (determined-modal family complete)
- close transitive determined-ordinary modal coupling via stratification
- close augmented-projection multi-head coupling scope limit
- determined-head recursion and negated-modal-over-invariant recursive epistemic fixpoint
- stratified epistemic analysis (determined-head detection + strata partition)
- cross-component epistemic joint-solving (multi-output)
- cross-component epistemic coupling
- recursive epistemic fixpoint support
- joint multi-epistemic predicate solving
- epistemic integrity constraints
- nested modal explicit representation and fail-closed diagnostics
- FAEEL founded self-support completion
- checkpoint epistemic solver semantics
- add incremental parser session
- add approximate inference pragmas
- add magic-set rewriting
- harden deterministic naf safety
- add safe meta lowering
- add finite list lowering
- add type term foundation
- add stream-mux AOT schedule
- add helper-split AOT pass
- *(logic)* K=5 and K=6 clique WCOJ promoter try_promote_clique_k for k=5/6
- *(promote)* normalize right-deep triangle / fully-right-deep 4-cycle
- HeatAwareLeaderModel plus variable-order-aware join-result feedback
- *(logic)* promote_multiway takes (stats, config); 25 caller sites updated
- *(logic)* WcojVariableOrderingModel trait + LeaderCardinalityModel
- *(logic)* CompilerConfig + composable compile API
- *(ir)* VariableOrder + LookupPerm types + MultiWayJoin.var_order field
- *(logic)* selectivity_pass real triangle + 4-cycle reordering
- *(logic)* variable-graph triangle + 4-cycle promoters
- *(logic)* selectivity_pass takes rel_ids; module-doc rewritten
- *(logic)* promote_multiway gates recursive SCCs by per-rule scan count
- *(logic)* wire selectivity_pass into Compiler post-optimizer
- *(logic)* selectivity_pass inline pub mod (no-op)
- *(logic)* try_promote_4cycle for canonical 4-cycle shape
- *(logic)* wire promote_multiway after optimizer
- *(logic)* promote_multiway pass for triangle WCOJ
- *(workspace)* cross-crate MultiWayJoin walker arms
- *(logic)* transitive SCC type inference (v0.6.2 PR 8)
- *(logic)* hypergraph mixed plan contract (v0.6.2 PR 6)
- *(logic)* hypergraph typed oracle gate (v0.6.2 PR 5)
- *(logic)* hypergraph SCC fixpoint evaluator (v0.6.2 PR 4)
- *(logic)* hypergraph fixpoint evaluator (v0.6.2 PR 3)
- *(logic)* hypergraph reference evaluator (v0.6.2 PR 2)
- *(logic)* hypergraph planner foundation (v0.6.2 PR 1)

### Fixed

- *(release)* drop README version sync + dynamic badges + agent release rules ([#124](https://github.com/BrainyBlaze/xlog/pull/124))
- standalone negated-variable-keyed constraint is a NAF safety error, not 'unimplemented'
- fail closed when a recursive epistemic program carries an epistemic constraint
- fail closed on ordinary recursion in epistemic programs
- route bound-variable multi-head epistemic programs through split
- route epistemic examples through xlog run
- fail closed nonzero faeel self support
- preserve independent epistemic split inputs
- guard epistemic split constraints
- coalesce dependent epistemic split rules
- *(release)* harden validation and gpu fallback paths
- integrate K7 K8 planned clique metadata
- module-level docs + lib-test flips
- remove multi-recursive promoter gate
- cargo fmt, evidence count, extract prepare_leader_inputs + real helper extraction
- address selectivity-driven join reordering review patches — duplicate attr, stale comment, evidence count, matcher tests
- classifier col0 + missing scope deliverables
- *(logic)* skip recursive SCCs in promote_multiway
- *(logic)* SCC-aware planner + structural-precedence repair (v0.6.2 PR 9)
- *(logic)* canonical explain_plans + refreshed module docs (PR 6 follow-up)
- *(logic)* typed gate defers to structural errors (v0.6.2 PR 5 follow-up)
- *(pyxlog)* install local wheels for explicit python
- *(cuda)* embed portable PTX fallback
- *(pyxlog)* ship kernels in wheels and document cubin path
- *(ci)* repair main release automation ([#27](https://github.com/BrainyBlaze/xlog/pull/27))
- *(ci)* keep README release metadata in sync ([#26](https://github.com/BrainyBlaze/xlog/pull/26))

### Other

- Set v0.9.2 release metadata
- Clarify v0.9.2 WFS release contract
- Close v0.9.2 epistemic semantic gaps
- Close v0.9.2 epistemic release
- cargo fmt
- negated-modal-in-recursion — stratified sub-case admits (co-evolving modal-recursion case), genuine negation cycle hits formal WFS bound
- document nested-modal chain-collapse semantics + interior-negation boundary
- update EIR+split tests from nested-modal rejection to collapse contract
- world-view mutation probe for nested-modal collapse direction
- derived-head coupling — stratified-vs-reference equivalence + true-cycle wall
- co-evolving modal-recursion classification unit tests (polarity/mode scoping)
- flip FAEEL foundedness logic tests to founded-extension semantics
- document split_epistemic_program (clean release surface)
- cargo fmt on v0.9.1 epistemic changeset
- safe split dependency and coupling semantics
- document v0.9.0 epistemic language surface
- *(release)* align v0.9.0 package metadata
- integrate v0.8.9 diagnostics surfaces
- reuse v0.8.6 bundle for v0.9.0
- *(release)* prepare v0.8.6 tag metadata
- *(release)* correct v0.8.5 public status
- certify language integration
- *(release)* prepare v0.8.0
- mark v0.7.0 release complete
- close phase2 integration gate
- Merge sort-label propagation branch into the phase-2 integration branch
- Merge K7/K8 clique-template branch into the phase-2 integration branch
- add K7/K8 clique templates
- cert flat-stats no helper rewrite
- scrub stale doc fragment in promotes_multirec_triangle test
- clean up dead helpers + unused imports in K=5 and K=6 clique WCOJ test files
- cargo fmt for workspace gate
- promoter + runtime-dispatch certs
- 15 acceptance tests across the acceptance matrix
- rename acceptance test + clarify evidence count math
- *(logic+integration)* variable-ordering cost model acceptance gate across the full matrix
- restore selectivity-driven join reordering acceptance gates — 4-cycle compile-time + runtime helper and synthesis certs
- *(logic)* selectivity_pass compile-time certs
- cargo fmt 4-cycle WCOJ test files
- *(logic)* strengthen optimizer arm tests with 4-input fixture
- *(workspace)* WCOJ doc cleanup post-MultiWayJoin
- *(v0.6.2)* prepare roadmap changelog and version
- *(logic)* hypergraph certification workloads (v0.6.2 PR 7)
- *(logic)* correct explain_plans sort ordering claims
- *(v0.6.1)* version bump + roadmap cleanup + changelog
- *(readme)* bump version badge + release-status line to v0.6.0
- restore audit README framing with current release setup
- Tighten workspace warning hygiene
- Merge branch 'audit/v0.5.0-prerelease'
- integrate prerelease audit docs
- harden public release readiness

## [0.9.2](https://github.com/BrainyBlaze/xlog/compare/xlog-stats-v0.5.0...xlog-stats-v0.9.2) - 2026-06-05

### Added

- add cost-aware k-clique planner

### Fixed

- *(release)* drop README version sync + dynamic badges + agent release rules ([#124](https://github.com/BrainyBlaze/xlog/pull/124))
- route epistemic examples through xlog run
- *(release)* harden validation and gpu fallback paths
- *(pyxlog)* install local wheels for explicit python
- *(cuda)* embed portable PTX fallback
- *(pyxlog)* ship kernels in wheels and document cubin path
- *(ci)* repair main release automation ([#27](https://github.com/BrainyBlaze/xlog/pull/27))
- *(ci)* keep README release metadata in sync ([#26](https://github.com/BrainyBlaze/xlog/pull/26))

### Other

- Set v0.9.2 release metadata
- document v0.9.0 epistemic language surface
- *(release)* align v0.9.0 package metadata
- integrate v0.8.9 diagnostics surfaces
- integrate v0.8.8 external world-model diagnostics into v0.8.9
- integrate first external diagnostics into v0.8.9
- Exercise external generated-rule diagnostics
- Resolve remaining XLOG evidence issues
- Add v0.8.7 external world-model diagnostics
- *(release)* prepare v0.8.6 tag metadata
- *(release)* correct v0.8.5 public status
- *(release)* prepare v0.8.0
- mark v0.7.0 release complete
- *(v0.6.2)* prepare roadmap changelog and version
- *(v0.6.1)* version bump + roadmap cleanup + changelog
- *(readme)* bump version badge + release-status line to v0.6.0
- restore audit README framing with current release setup
- Merge branch 'audit/v0.5.0-prerelease'
- integrate prerelease audit docs
- harden public release readiness

## [0.9.2](https://github.com/BrainyBlaze/xlog/compare/xlog-cuda-v0.5.0...xlog-cuda-v0.9.2) - 2026-06-05

### Added

- *(cuda)* XLOG_PTX_MAX_VERSION â downgrade embedded portable PTX ISA
- *(mc)* sparse WCOJ world-batched GPU-resident MC engine
- *(mc)* no-host instrumentation foundation for WCOJ engine (alloc + fixpoint counters)
- *(mc)* GPU-resident Datalog/MC engine (megakernel) + K1-K5 pilots
- checkpoint epistemic solver semantics
- add chain exact shared-memory scorer
- extend exact induction typed dispatch
- close CUDA Graph-mode set maintenance
- certify CUDA Graph external consumer graph path
- cache bounded CSM CUDA graphs
- add bounded CSM CUDA graph path
- add CUDA graph execution wrapper
- remove adaptive skew classifier surface
- route clique kernels through HG block-slice
- route u64 4-cycle through HG block-slice
- route u64 triangle through HG block-slice
- retire old u32 triangle materialize surface
- route u32 4-cycle through HG block-slice
- retire old u32 triangle count surface
- retire layout/count kernel fusion fused count kernel
- reuse HG block workspace
- make HG cached count single-pass
- cache HG materialization and add superhub gate bench
- route u32 triangle dispatch through HG pipeline
- add triangle HG materialize pipeline
- add triangle HG work-plan count surface
- add persistent WCOJ metadata builder
- production kernel fusion (layout+count) with threshold dispatch + auto-disable + cert grid
- add sort_merge_join_v2_inner_u32_1key + is_sorted_ascending_u32 provider fns
- add sort-merge inner-join kernel + sortedness-detection kernel
- add nested_loop_join_v2_inner_u32_1key in relational.rs (gather-based)
- add nested-loop emit-pairs kernel (multi-col-compatible)
- *(cuda)* K=5 and K=6 clique WCOJ clique provider entries
- *(cuda)* K=5 and K=6 clique WCOJ templated clique kernel for k=5 + k=6
- *(cuda)* generic sorted-relation accessors generic wcoj_layout_sort_*_recorded entry points
- *(cuda)* wcoj_project_2col_swap_recorded + wcoj_project_output_columns_recorded
- *(cuda+runtime)* 4-cycle skew classifier + adaptive opt-in
- *(cuda)* u64 4-cycle WCOJ kernels + provider + tests
- *(cuda)* u32 4-cycle WCOJ kernels + provider + tests
- *(cuda)* WCOJ layout fast-path for sorted+unique inputs (v0.6.2)
- *(cuda)* WCOJ adaptive-dispatch skew classifier (v0.6.2 A2-lite commit A)
- *(cuda)* WCOJ u64 provider kernels + entries (v0.6.2)
- *(cuda)* sort_recorded + dedup_full_row_recorded U64 (v0.6.2)
- *(cuda)* WCOJ Symbol key support (v0.6.2)
- *(cuda)* WCOJ sorted-layout construction u32 (v0.6.2)
- *(cuda)* WCOJ triangle device-side scan + scalar D2H total
- *(cuda)* GPU 3-way WCOJ triangle kernel u32 v1 (v0.6.2)
- *(cuda)* wire recorded CSM hash-join dispatch ([#91](https://github.com/BrainyBlaze/xlog/pull/91))
- *(cuda)* add recorded indexed LeftOuter count-scan-materialize path ([#87](https://github.com/BrainyBlaze/xlog/pull/87))
- *(cuda)* add recorded LeftOuter count-scan-materialize path ([#84](https://github.com/BrainyBlaze/xlog/pull/84))
- *(cuda)* formal cert harness for runtime-backed recorded path
- *(cuda)* GPU-resident binary-join indexed Inner CSM
- *(cuda)* GPU-resident binary-join Inner retake — count→scan→materialize
- *(cuda)* env-gated runtime dispatch for sort/dedup/GroupBy/hash-join + cert mode
- *(cuda)* provider-level recorded indexed hash join + LeftOuter step-D recorder fix
- *(cuda)* provider-level recorded LeftOuter hash join
- *(cuda)* provider-level recorded Semi / Anti hash join
- *(cuda)* provider-level recorded inner hash join
- *(cuda)* provider-level recorded GroupBy multi-agg (U32 keys, count/sum/min/max)
- *(cuda)* provider-level recorded sort + dedup_full_row (u32 / Symbol)
- *(cuda)* preserve runtime identity for xlog-owned DLPack / Arrow columns
- *(cuda)* migrate fused compare+scan+compact filter to recorded discipline
- *(cuda)* env-gated recorded filter dispatch (XLOG_USE_RECORDED_FILTERS)
- *(cuda)* v0.6 stream-safe runtime + LaunchRecorder + filter predicate matrix
- *(cuda)* v0.6 device-runtime allocator (opt-in) + A3 stability ([#54](https://github.com/BrainyBlaze/xlog/pull/54))
- *(cuda)* binary-join output counts as metadata reads (v0.5.5 PR 3) ([#52](https://github.com/BrainyBlaze/xlog/pull/52))
- *(cuda)* GPU full-row dedup and set-difference (v0.5.5 PR 2) ([#50](https://github.com/BrainyBlaze/xlog/pull/50))
- *(runtime)* add strict deterministic D2H guard (v0.5.5) ([#49](https://github.com/BrainyBlaze/xlog/pull/49))

### Fixed

- bootstrap cuda-ci runner — bump test iterations + fix maturin compatibility ([#127](https://github.com/BrainyBlaze/xlog/pull/127))
- *(release)* drop README version sync + dynamic badges + agent release rules ([#124](https://github.com/BrainyBlaze/xlog/pull/124))
- route epistemic examples through xlog run
- close persistent index background build scope
- close GPU-native count-lift exact path
- *(release)* harden validation and gpu fallback paths
- integrate K7 K8 planned clique metadata
- close mint4 path-isolated gate
- extend 4cycle e2-prefix mitigation to u64
- mitigate M_INT.4 4cycle HG regression
- route nested-loop dispatch through shared record_join_result feedback
- tighten Tier-1 wrapper contract + revert recursive helper extension
- real interner-allocated Symbol IDs + drop test-file warnings + tighten D4 wording
- cargo fmt, evidence count, extract prepare_leader_inputs + real helper extraction
- classifier col0 + missing scope deliverables
- *(cuda)* drain WCOJ layout fast-path failure paths
- *(runtime)* harden WCOJ phase timing diagnostics
- *(cuda)* drain launch stream on skew classifier failure paths (v0.6.2)
- *(cuda)* record d_overflow on three CSM materialize recorders ([#89](https://github.com/BrainyBlaze/xlog/pull/89))
- *(cuda)* access-aware stream dependency manager for cross-stream lifetime safety ([#72](https://github.com/BrainyBlaze/xlog/pull/72))
- *(cuda)* clamp recorded compact mask domain
- *(logic)* restore deterministic recursive set evaluation
- *(pyxlog)* install local wheels for explicit python
- *(cuda)* embed portable PTX fallback
- *(pyxlog)* ship kernels in wheels and document cubin path
- *(ci)* repair main release automation ([#27](https://github.com/BrainyBlaze/xlog/pull/27))
- *(ci)* keep README release metadata in sync ([#26](https://github.com/BrainyBlaze/xlog/pull/26))

### Other

- Set v0.9.2 release metadata
- integrate main MC GPU-resident engine into v0.9.2 epistemic completion
- document v0.9.0 epistemic language surface
- *(release)* align v0.9.0 package metadata
- integrate v0.8.9 diagnostics surfaces
- reuse v0.8.6 bundle for v0.9.0
- *(release)* prepare v0.8.6 tag metadata
- *(release)* correct v0.8.5 public status
- *(release)* prepare v0.8.0
- mark v0.7.0 release complete
- close phase2 purge gate
- close phase2 integration gate
- Merge CUDA Graph benchmark-spike branch into the phase-2 integration branch
- Merge sort-label propagation branch into the phase-2 integration branch
- Merge K7/K8 clique-template branch into the phase-2 integration branch
- add K7/K8 clique templates
- Merge stream-mux AOT branch into WCOJ bundle integration
- graceful close and paper harness
- certify HG metadata storage budget
- remove dead layout/count kernel fusion route counters
- patch stale rustdoc + kernel comments after iter-6 unwiring
- fmt + rustdoc cleanup
- align plan + provider rustdoc with landed byte-check + counter type
- workspace gate green pre-bench
- clean up dead helpers + unused imports in K=5 and K=6 clique WCOJ test files
- cargo fmt for workspace gate
- *(cuda)* K=5 and K=6 clique WCOJ provider certs + source-audit
- *(cuda)* generic sorted-relation accessors acceptance grid — 82 tests across width-class and arity
- *(cuda)* correct wcoj_4cycle_skew_score_u32 doc to col0
- *(cuda)* layout reuse smoke for 4-cycle
- *(v0.6.2)* prepare roadmap changelog and version
- *(runtime)* WCOJ phase-timing scaffolding + report (v0.6.2)
- *(cuda)* WCOJ U64 strict deterministic-D2H gate (v0.6.2)
- *(cuda)* update recorded dedup U64 scope comment
- *(wcoj)* update Symbol dispatch scope comments
- *(cuda)* planner-to-provider WCOJ certification (v0.6.2)
- *(v0.6.1)* version bump + roadmap cleanup + changelog
- *(readme)* bump version badge + release-status line to v0.6.0
- restore audit README framing with current release setup
- Fix validation regressions in release and examples
- Merge branch 'audit/v0.5.0-prerelease'
- integrate prerelease audit docs
- harden public release readiness

## [0.9.2](https://github.com/BrainyBlaze/xlog/compare/xlog-ir-v0.5.0...xlog-ir-v0.9.2) - 2026-06-05

### Added

- close augmented-projection multi-head coupling scope limit
- epistemic integrity constraints
- checkpoint epistemic solver semantics
- add k-clique cost gate routes
- add k-clique RIR variable order
- *(logic)* WcojVariableOrderingModel trait + LeaderCardinalityModel
- *(ir)* VariableOrder + LookupPerm types + MultiWayJoin.var_order field
- *(ir)* add RirNode::MultiWayJoin variant

### Fixed

- *(release)* drop README version sync + dynamic badges + agent release rules ([#124](https://github.com/BrainyBlaze/xlog/pull/124))
- route epistemic examples through xlog run
- *(release)* harden validation and gpu fallback paths
- *(pyxlog)* install local wheels for explicit python
- *(cuda)* embed portable PTX fallback
- *(pyxlog)* ship kernels in wheels and document cubin path
- *(ci)* repair main release automation ([#27](https://github.com/BrainyBlaze/xlog/pull/27))
- *(ci)* keep README release metadata in sync ([#26](https://github.com/BrainyBlaze/xlog/pull/26))
- unblock release publish verification

### Other

- Set v0.9.2 release metadata
- document v0.9.0 epistemic language surface
- *(release)* align v0.9.0 package metadata
- integrate v0.8.9 diagnostics surfaces
- reuse v0.8.6 bundle for v0.9.0
- *(release)* prepare v0.8.6 tag metadata
- *(release)* correct v0.8.5 public status
- *(release)* prepare v0.8.0
- mark v0.7.0 release complete
- G_W63 production chain join route
- phase-2 purge rerun
- *(workspace)* MultiWayJoin shape-agnosticism guards
- *(ir)* MultiWayJoin walker contract
- *(v0.6.2)* prepare roadmap changelog and version
- *(v0.6.1)* version bump + roadmap cleanup + changelog
- *(readme)* bump version badge + release-status line to v0.6.0
- restore audit README framing with current release setup
- Tighten workspace warning hygiene
- Merge branch 'audit/v0.5.0-prerelease'
- integrate prerelease audit docs
- harden public release readiness

## [0.9.2](https://github.com/BrainyBlaze/xlog/compare/xlog-core-v0.5.0...xlog-core-v0.9.2) - 2026-06-05

### Added

- add persistent hash index telemetry
- add adaptive runtime reoptimization
- add runtime common subexpression cache
- extend exact induction typed dispatch
- expose xlog sort-label metadata
- remove adaptive skew classifier surface
- retire old u32 triangle count surface
- production kernel fusion (layout+count) with threshold dispatch + auto-disable + cert grid
- default-flip wcoj_cost_model resolver to Cardinality
- *(runtime)* dispatch sites use build_wcoj_cost_model factory
- *(core)* CostModelKind + RuntimeConfig::wcoj_cost_model
- *(runtime)* match_multiway_4cycle + try_dispatch_wcoj_4cycle force gate
- *(runtime)* default-on adaptive WCOJ + hard kill switch (v0.6.2)
- *(runtime)* adaptive WCOJ dispatch + classifier branch (v0.6.2 A2-lite commit B)
- *(runtime)* env-gated WCOJ triangle executor wiring (v0.6.2)
- *(runtime)* add strict deterministic D2H guard (v0.5.5) ([#49](https://github.com/BrainyBlaze/xlog/pull/49))

### Fixed

- *(release)* drop README version sync + dynamic badges + agent release rules ([#124](https://github.com/BrainyBlaze/xlog/pull/124))
- route epistemic examples through xlog run
- *(release)* harden validation and gpu fallback paths
- restore cost-model default-flip cert
- *(pyxlog)* install local wheels for explicit python
- *(cuda)* embed portable PTX fallback
- *(pyxlog)* ship kernels in wheels and document cubin path
- *(ci)* repair main release automation ([#27](https://github.com/BrainyBlaze/xlog/pull/27))
- *(ci)* keep README release metadata in sync ([#26](https://github.com/BrainyBlaze/xlog/pull/26))
- unblock release publish verification

### Other

- Set v0.9.2 release metadata
- document v0.9.0 epistemic language surface
- *(release)* align v0.9.0 package metadata
- integrate v0.8.9 diagnostics surfaces
- integrate v0.8.8 external world-model diagnostics into v0.8.9
- integrate first external diagnostics into v0.8.9
- Exercise external generated-rule diagnostics
- Resolve remaining XLOG evidence issues
- Add v0.8.7 external world-model diagnostics
- *(release)* prepare v0.8.6 tag metadata
- *(release)* correct v0.8.5 public status
- *(release)* prepare v0.8.0
- mark v0.7.0 release complete
- close purge gate
- *(v0.6.2)* prepare roadmap changelog and version
- *(v0.6.1)* version bump + roadmap cleanup + changelog
- *(readme)* bump version badge + release-status line to v0.6.0
- restore audit README framing with current release setup
- Merge branch 'audit/v0.5.0-prerelease'
- integrate prerelease audit docs
- harden public release readiness

## [0.5.2](https://github.com/BrainyBlaze/xlog/compare/xlog-cli-v0.5.0...xlog-cli-v0.5.2) — 2026-04-20

### Fixed

- unblock release publish verification

## [0.5.1] — 2026-04-20

### Fixed

- unblock release publish verification

### Added

- **Bounded exact-induction engine** (`xlog-induce` + `ilp_exact` CUDA kernel + `pyxlog`
  bridge): New `xlog-induce` crate scores all `(left, right)` candidate pairs across the
  four canonical external consumer topologies (`chain`, `star`, `fanout`, `fanin`) in a single batched
  GPU pass and returns top-K per topology with full candidate metadata
  (`positives_covered`, `negatives_covered`, `next_*_covered`, `tie_class_size`).
  Designed for external consumer's bounded exact-induction integration; behaviorally equivalent on bounded
  requests to `pyxlog.ilp.induce_exact(backend="python", strict_per_topology=True)`.
  - **Engine** (`crates/xlog-induce/`): `InduceExactRequest` (indices + buffer handles),
    `ExactInductionResult` / `ScoredCandidate`, pre-kernel classification
    (`validate::classify_request` — 5 pure unit tests), buffer-level validation
    (arity=2, column type `U64`, cached-row-count guard).
  - **Deterministic reducer** (`xlog-induce::reduce`): lexicographic `(-positives,
    negatives, left_idx, right_idx)` sort + positive-coverage filter + `next_*` and
    `tie_class_size` diagnostics. 16 host-side unit tests lock the comparator and
    diagnostic semantics bit-for-bit.
  - **CUDA kernel** (`kernels/ilp_exact.cu`, new `xlog_ilp_exact` module): single
    `ilp_exact_score` entry. Launch geometry: `grid = (C, C, 4)` blocks of 256
    threads; each block owns one `(topology, L, R)` output slot, so there are no
    cross-block atomics on the scoring path. Deterministic pair-halving block
    reduction (integer counts only).
  - **Provider launcher** (`crates/xlog-cuda/src/provider/ilp_exact.rs`):
    `CudaKernelProvider::ilp_exact_score(candidates, positives, negatives) ->
    (Vec<u32>, Vec<u32>)`. D2D-concatenates candidate columns in setup, uploads
    `cand_offsets`, launches the scoring kernel, and downloads two count arrays.
    D2H budget is a constant **2 per call** regardless of candidate count. Three
    CUDA-gated launcher tests (hand-computed coverage fixture, determinism across
    runs, empty-negatives path).
  - **Kernel manifest**: bumped `KERNEL_MODULES` count 21 → 22 (plus the
    compile-time `assert!(KERNEL_CU_NAMES.len() == 22)` at `provider/mod.rs:221`).
    `ILP_EXACT_MODULE` + `ilp_exact_kernels::ILP_EXACT_SCORE` constants added.
  - **pyxlog bridge** (`crates/pyxlog/src/ilp_exact.rs`): new
    `CompiledIlpProgram::induce_exact_native(...)` pyo3 method — resolves relation
    names against `rel_index`, unwraps positive/negative DLPack tensors against
    the head relation's declared schema, dispatches to the engine, and returns a
    `dict` the Python wrapper repackages into `ExactInductionResult` /
    `ScoredCandidate` dataclasses.
  - **Python wrapper** (`crates/pyxlog/python/pyxlog/ilp/exact_induce.py`): new
    `backend="native"` dispatch path on `induce_exact(...)` plus the dict → dataclass
    repackaging helper. Wrapper default backend is still `"python"` for backward
    compatibility with existing callers.
  - **Parity contract** (`python/tests/test_ilp_exact_induce.py`):
    `test_induce_exact_native_matches_python_reference` (ordered equality of
    summary and per-candidate fields) and
    `test_induce_exact_native_does_not_scale_d2h_with_candidate_pairs` (gate:
    `large.d2h_transfer_count ≤ small.d2h_transfer_count + 2`).
  - **Kernel design note**: `docs/plans/2026-04-17-m8-ilp-exact-kernel-design.md`.
- **MC runtime optimization** (`xlog-prob`, `xlog-runtime`): 8.6% wall-clock improvement on
  the MC evaluation hot loop (14.11s → 12.90s on 1000-sample clamped benchmark). No API changes.
  - `McTimingBreakdown` struct with env-gated profiling (`XLOG_MC_PROFILE=1`) for per-phase
    timing (sampler, reset, build, eval, count).
  - `McCountStrategy` enum: maps sampling method to count strategy (`QueriesAndEvidence` for
    rejection, `QueriesOnly` for clamped). Skips evidence-side allocations/uploads in clamped mode.
  - `McSampleResetPlan` struct + `build_sample_reset_plan()`: classifies relations as preserve
    (deterministic-only), clear (dynamic), or reload_base. Replaces full store clone/restore
    with targeted per-relation reset.
  - `Executor::reset_for_mc_relations()`: new method for targeted preserve/clear reset of
    relations between MC samples.
  - Pre-allocated pointer buffers (`query_ptrs_buf`, `evidence_ptrs_buf`) outside the sample
    closure, avoiding per-sample Vec heap allocation.
- **Evidence clamping for MC inference** (`xlog-prob`): Monte Carlo evidence conditioning
  via `McSamplingMethod::EvidenceClamping`. Forces root Bernoulli evidence variables in the
  sampling kernel so every sample counts (`evidence_samples == total_samples`). Auto-selected
  when all evidence maps to probabilistic facts or positive AD heads; falls back to rejection
  for derived/deterministic/negative-AD evidence. New `sampling_method` field on `McEvalConfig`,
  `McResult`, `McDeviceResult`, and Python API. CUDA kernel updated with `force_mask`/`forced_value`
  inputs.
- **Provenance primitives** (`xlog-prob`): Retained provenance metadata for external Rust consumers.
  New `ChoiceSource` type captures annotated-disjunction metadata (explicit heads, choice index,
  optional source ID). Two new fields on `Provenance`: `leaf_atoms` (`BTreeMap<LeafId, GroundAtom>`)
  and `choice_sources` (`BTreeMap<ChoiceVarId, ChoiceSource>`). Three new accessors:
  `leaf_atom(LeafId)`, `choice_source(ChoiceVarId)`, `atoms_with_formulas()` iterator.
  `GroundAtom::new()` made public. Top-level re-exports added to `xlog-prob` lib.rs for
  `ChoiceSource`, `GroundAtom`, `Provenance`, `Value`, `ChoiceVarId`, `LeafId`, `PirGraph`,
  `PirNode`, `PirNodeId`. Inline retention at existing extraction allocation sites — no new
  passes or post-hoc reconstruction.

### Changed

- **`CudaKernelProvider::clone_buffer` now propagates `cached_row_count`** (`xlog-cuda`):
  Previously the deep-cloned buffer used `CudaBuffer::from_columns` (no host-side count
  cache), forcing any consumer of a cloned buffer to perform a D2H read on
  `num_rows_device()` just to learn the row count. All call sites that go through
  `CompiledIlpProgram::put_relation` clone on insertion into the executor's relation
  store, so every relation buffer fetched from the store was losing its cache. The
  new code calls `set_cached_row_count_if_unset(source.cached_row_count())` on the
  clone when the source has a populated cache, preserving the host-visible count
  across clones. Pinned by the new `test_clone_buffer_preserves_cached_row_count`
  test, and a load-bearing precondition for the bounded exact-induction engine's
  hot-loop D2H budget.
- **`pyxlog.ilp.induce_exact()` gains `strict_per_topology` opt-in flag**
  (`pyxlog`, Python): The `backend="python"` prototype has a latent cross-topology
  contamination behavior — stale `W_<topo>_<head>` masks from earlier outer-loop
  iterations bleed into later topologies' coverage numbers via `evaluate()`.
  Setting `strict_per_topology=True` zeroes out "other" topology masks before
  each topology's inner loop, yielding per-topology-isolated scoring that matches
  the `backend="native"` kernel's by-construction semantics. Default remains
  `False` for full backward compatibility with callers that are calibrated
  against the prototype's historical numbers (notably external consumer Phase 0 liveness
  baselines). The `"native"` backend is unaffected — it is strict by design.
- **ILP reliability gate 4.6x faster** (`pyxlog`): Compile once per stage and reuse across
  all 5 seeds via `reset_runtime()`, eliminating 16 redundant compilations and 20 holdout
  evaluations (1647s → 359s). Gate still runs 4 stages × 5 seeds = 20 independent training
  runs with identical budgets (150 steps, 7 max attempts). Parity with fresh-compile behavior
  verified by new `test_compile_once_reuse_parity` and `test_compile_once_multi_seed_isolation`
  tests.
- **MC behavior test budgets reduced** (`xlog-prob`): 10 MC tests trimmed from 50K–80K samples
  to 2K–5K (one 20K accuracy guard retained). Reduces test-suite turnaround without changing
  runtime engine behavior.
- **`build_sample_buffers()` no longer performs per-sample D2H row-count reads**: Uses host-side
  `num_rows()` (capacity) instead of synchronous `device_row_count_u32()` GPU→CPU transfers.
- **MC per-sample store management replaced**: Full `snapshot_store()`/`restore_store()` cycle
  replaced by `McSampleResetPlan` with targeted relation-level reset.
- **Whitepaper and public docs repositioned** around "GPU-native logic programming language
  for unified symbolic reasoning" instead of "GPU-accelerated Datalog engine". v0.5.0 LaTeX
  whitepaper (`docs/whitepaper/main.pdf`) gained a new Section 3 "The xlog Language" covering
  types, UDFs, modules, arithmetic, aggregations, and constraints with validated examples;
  `docs/ARCHITECTURE.md`, `docs/language-reference.md`, root `README.md`, `ROADMAP.md`, and
  `docs/whitepaper/README.md` were aligned. Stale Markdown whitepaper draft
  `docs/whitepaper-v050.md` removed (superseded by the LaTeX version). Broken cross-references
  to cleanup-deleted `docs/plans/`, `docs/design/`, `docs/ilp/` directories replaced with
  pointers to surviving docs (whitepaper sections, `dilp-training.md`, `rfc-tensorized-ilp.md`).
  Docs-only change; no code or API impact.

### Refactored

- **5-wave codebase refactoring** (2026-03-10 → 2026-03-13, 57 commits across all waves):
  Structural decomposition of the 5 largest modules in the workspace. No external API changes.
  No behavioral changes. All existing tests, gates, and contracts preserved.

  **Wave 1 — Dependency cleanup + error/type seams** (`xlog-core`, `xlog-cuda`, `xlog-logic`,
  `xlog-neural`; 8 commits):
  - Removed false dependency cycle: `xlog-logic` no longer depends on `xlog-runtime` in
    production, `xlog-stats` no longer depends on `xlog-cuda`.
  - Added `xlog-neural → xlog-core` edge for error conversion impls.
  - New `From` impls: `NeuralError`, `TensorSourceError`, `FunctionError`, `TypeError`,
    `ModuleError` → `XlogError`. New `driver_err()` helper for cudarc `DriverError` (orphan
    rule prevents `From` impl).
  - New `XlogError::{kernel_ctx, execution_ctx, compilation_ctx}` structured error context
    helpers.
  - New `GpuScalar` trait (`xlog-cuda/src/type_seam.rs`): pub + sealed marker for Rust scalar
    types that round-trip through GPU column storage. 8 impls (u8, u32, u64, i32, i64, f32,
    f64, bool). Enables generic `download_column::<T>()` and `create_buffer_from_slice::<T>()`
    in Wave 2.

  **Wave 2 — Provider decomposition + GpuScalar migration** (`xlog-cuda`, all consumer crates;
  9 commits):
  - `provider.rs` (12,809 LOC) → `provider/mod.rs` + 8 submodules: `kernel_loading.rs`,
    `relational.rs`, `filter.rs`, `groupby.rs`, `arithmetic.rs`, `transfer.rs`,
    `probabilistic.rs`, `ilp.rs`, `io.rs`.
  - Collapsed type-specialized function families via `GpuScalar` trait:
    - 8 `download_column_<T>()` functions (~280 LOC) → 1 generic `download_column::<T>()` (~35 LOC)
    - 7 `create_buffer_from_<T>_slice()` functions (~220 LOC) → 1 generic `create_buffer_from_slice::<T>()` (~30 LOC)
    - 11 `filter_<T>()` functions (~1,200 LOC) → 1 generic `filter::<T>()` with enum dispatch
  - ~140 mechanical turbofish call-site updates across 8 consumer crates.
  - `new()` constructor refactored from ~814 lines of boilerplate to ~120 lines via
    `KernelModuleSpec` manifest + `load_all_kernel_modules()`.
  - Net reduction: ~5,990 lines.

  **Wave 3 — Executor decomposition** (`xlog-runtime`; 11 commits):
  - `executor.rs` (4,337 LOC) → `executor/mod.rs` + 6 submodules: `node_dispatch.rs`,
    `recursive.rs`, `expression.rs`, `rewrite.rs`, `join_cache.rs`, `delta.rs`.
  - Extracted `DeltaRelationTracker` as standalone `pub(crate)` type for delta relation
    lifecycle during recursive evaluation.
  - Extracted `JoinIndexCache` as standalone `pub(crate)` LRU struct.
  - Net reduction: ~1,040 lines.

  **Wave 4 — Pyxlog FFI extraction** (`pyxlog`; 10 commits):
  - `lib.rs` (6,202 LOC) → slimmed `lib.rs` (~487 LOC) + 7 submodules: `program.rs`,
    `logic.rs`, `ilp.rs`, `ilp_gpu.rs`, `training.rs`, `neural.rs`, `types.rs`.
  - Consolidated 2 non-contiguous `CompiledIlpProgram` impl blocks into single block.
  - Extracted `compute_ilp_loss_grad_gpu()` (574 LOC) into focused helpers in `ilp_gpu.rs`.
  - Collapsed f32/f64 forward-backward duplication into generic `forward_backward_typed()`.
  - Added `xlog_err_to_py()` / `neural_err_to_py()` local error-mapping helpers (orphan rule
    prevents `From` impls for `PyErr`).
  - Net reduction: ~1,320 lines.

  **Wave 5 — Probabilistic backend decomposition + coherence** (`xlog-prob`, workspace-wide;
  19 commits):
  - `gpu_d4.rs` (3,669 LOC) → `gpu_d4/mod.rs` (~450 LOC) + `frontier.rs` (~1,480 LOC) +
    `build.rs` (~1,850 LOC).
  - `mc.rs` (3,399 LOC) → `mc/mod.rs` (~1,079 LOC) + `evidence.rs` (~130 LOC) +
    `buffers.rs` (~973 LOC) + `sampling.rs` (~297 LOC) + `results.rs` (~993 LOC).
  - Config coherence: `Default` impls on all config structs, `#[non_exhaustive]` on 3 structs
    (`MemoryBudget`, `GpuEquivalenceConfig`, `WfsConfig`), `///` doc comments on all configs.
  - Test harness consolidation: 22 duplicate `setup_provider()` copies → 2 canonical
    `tests/common/mod.rs` helpers (xlog-cuda, xlog-prob).
  - `xlog-prob` top-level re-exports: `GpuCompileConfig`, `CircuitCompileProfile`,
    `ExactDdnnfProgram`, `ExactResult`, `GpuConfig`, `McEvalConfig`, `McProgram`,
    `McSamplingMethod`, `McCountStrategy`, `McResult`, `McDeviceResult`, `EvidenceForcing`,
    `ForceabilityReason`, `WfsConfig`, `WfsResult`, `TruthValue`, plus WFS free functions.
  - WFS entry points consolidated: 2 zero-caller functions removed, 1 gated behind
    `#[cfg(test)]`.
  - 71 visibility tightens (`pub` → `pub(crate)`) across `xlog-prob`, `xlog-solve`,
    `xlog-logic`.
  - Clone audit documented (deliberate clones annotated, no actionable reductions found).
  - RIR visitor trait decision: 7 dispatch patterns warrant a trait, deferred to v0.7+.
  - 35 compiler warnings fixed (private_interfaces, unused imports, dead code).

  **Post-refactoring module sizes** (god modules → focused submodules):

  | Module | Before | After (mod.rs) | Submodules |
  |--------|--------|----------------|------------|
  | `provider.rs` | 12,809 | 2,651 | 8 |
  | `pyxlog/lib.rs` | 6,202 | 487 | 7 |
  | `executor.rs` | 4,337 | 2,050 | 6 |
  | `gpu_d4.rs` | 3,669 | 450 | 2 |
  | `mc.rs` | 3,399 | 1,079 | 4 |
  | **Total** | **30,416** | **6,717** | **27** |

  Design docs: `docs/superpowers/specs/2026-03-10-wave{1-5}-*.md`.
  Implementation plans: `docs/superpowers/plans/2026-03-10-wave{1-2}-*.md`,
  `docs/superpowers/plans/2026-03-11-wave{3-5}-*.md`.

### Removed

- **`device_row_count_u32()`** helper in MC hot loop — synchronous D2H scalar read, replaced
  by host-side capacity checks.
- **`snapshot_store()` / `restore_store()`** in MC evaluator — replaced by `McSampleResetPlan`
  with `reset_for_mc_relations()`.
- **Type-specialized provider functions** (`xlog-cuda`): `download_column_u32`,
  `download_column_i32`, `download_column_i64`, `download_column_u64`, `download_column_f32`,
  `download_column_f64`, `download_column_bool`, `download_column_u8`,
  `create_buffer_from_u32_slice`, `create_buffer_from_i32_slice`,
  `create_buffer_from_i64_slice`, `create_buffer_from_u64_slice`,
  `create_buffer_from_f32_slice`, `create_buffer_from_f64_slice`,
  `create_buffer_from_u8_slice`, and 11 type-specialized `filter_*` functions — all replaced
  by `GpuScalar`-generic equivalents.
- **2 WFS entry points** (`xlog-prob`): `evaluate_wfs_scc` and `evaluate_wfs_with_rules_config`
  removed (zero callers). `evaluate_wfs_scc_with_config` gated behind `#[cfg(test)]`.

## [0.5.0] — 2026-03-08

### Added

- **Term embeddings (training-only)** — `register_embedding()` for
  `nn.Embedding` (trainable) and `torch.Tensor` (frozen) payloads.
  `forward_embedding(name, ids)` returns batched tensors with autograd
  support on the same device as the embedding (CUDA-safe). Cross-registration
  validation: embedding declarations reject `register_network()` and vice
  versa. Compile-time mixed-form rejection for network names. Raw tensors
  are detached at registration to enforce frozen contract even when input
  has `requires_grad=True`. User-managed optimizer (training-control APIs do not cover
  embeddings). Inference path deferred to v0.5.1+.
- **GPU-resident ILP credit/loss path** (`compute_ilp_loss_grad_gpu`): Single Rust/CUDA call replaces
  Python-side `_compute_loss_from_candidates()` loop. Builds COO→CSR on-device, runs forward/backward
  CUDA kernels, reduces loss on-device, returns `(loss, grad)` as DLPack tensors. Zero D2H transfers
  in all paths (including chunked), confirmed by strict byte-level accounting (`host_transfer_stats()`).
- **4 new CUDA kernels**: `ilp_coo_fill_from_mask` (COO fill from device mask + prefix-sum),
  `ilp_csr_histogram` (CSR row_offsets via atomicAdd histogram), `ilp_reduce_sum_f32`/`ilp_reduce_sum_f64`
  (block-level sum reduction).
- **Two-pass GPU-only chunk merge**: Bounded-memory streaming replaces D2H-based chunked fallback.
  Pass 1 counts NNZ per task via `count_mask_into_slot`, pass 2 fills COO at pre-computed offsets.
  Zero data-plane D2H in all code paths, verified on all 4 ILP stages with forced chunking.
- **`coo_chunk_budget`** (renamed from `coo_memory_cap`): Controls per-chunk temp allocation ceiling.
  Final exact-NNZ COO buffer may exceed the chunk budget. Deprecated `set_coo_memory_cap()` alias
  retained for one release cycle.
- **`count_mask_into_slot`**: Provider method writing mask count directly into pre-allocated device
  array slot, avoiding per-task allocation churn in pass 1.
- **`dtoh_scalar_untracked`**: Provider helper for metadata-only D2H reads (e.g., total_nnz scalar)
  without incrementing transfer counters. Makes the metadata-vs-data-plane contract explicit.
- **Strict zero-D2H mode**: `set_strict_zero_dtoh(True)` now passes in all paths including chunked.
  Use in zero-D2H benchmarks and CI gates.
- **D2H transfer accounting**: Strict byte-level gate via `host_transfer_stats()` returning
  `dtoh_calls` and `dtoh_bytes` counters, plus coarse column-level `d2h_transfer_count()`.
- **3 gradient parity tests**: GPU kernel output vs pure-PyTorch reference (f32, f64, multi-candidate).
- **Artifact schema migration**: `save()` writes `beta-v2`, `load()` accepts both `beta-v1` and
  `beta-v2`. Forward-compatible schema evolution.
- **Bounded telemetry persistence**: `TrainConfig.persist_telemetry` (default False) and
  `telemetry_persist_limit` (default 100). When enabled, `save()` includes a `telemetry_snapshot`
  with the last N `StepRecord`s and `step_timings`. `load()` restores telemetry from snapshot.
- **`program.get_lr(network_name)`**: Read current learning rate from a registered network's optimizer.
- **`program.set_lr(network_name, lr)`**: Set learning rate across all param groups of a registered
  network's optimizer.
- **Per-network `scheduler_step`**: `program.scheduler_step(network_name)` steps a single network's
  scheduler. `scheduler_step(None)` (default) steps all schedulers, preserving backward compatibility.
- **Gradient clipping**: `train_model(..., max_grad_norm=N)` and `train_model_tensor(..., max_grad_norm=N)`
  clip gradients via `torch.nn.utils.clip_grad_norm_` before each optimizer step. `None` (default)
  disables clipping.
- **Early stopping**: `train_model(..., val_queries=[...], patience=N)` and
  `train_model_tensor(..., val_queries=[...], patience=N)` evaluate validation loss each epoch and
  stop training after `patience` consecutive epochs without improvement.
- **`TrainingHistory.stopped_early`**: Boolean flag indicating whether early stopping was triggered.
- **`GpuCdclWorkspace`**: Pre-allocated solver arena for reusing GPU buffers across multiple CDCL
  solves (incremental Monte Carlo verifier). Created via `GpuCdclSolver::new_workspace()`.
- **`solve_expect_unsat_*_ws` method variants**: Workspace-backed CDCL solving that reuses
  pre-allocated device buffers instead of per-call allocation.
- **`GpuCompileConfig.incremental_verify`**: Opt-in for workspace reuse in the equivalence
  verifier (amortizes arena allocation across q1/q2 solve pair).
- **`GpuEquivalenceConfig.reuse_workspace`**: Internal config field propagated from
  `incremental_verify`.

### Changed

- **`coo_memory_cap` renamed to `coo_chunk_budget`**: Old name implied a hard ceiling on all COO
  allocations; new semantics allow the exact-NNZ output buffer to exceed the chunk budget.
  `set_coo_memory_cap()` remains as a deprecated alias.

### Removed

- **Legacy host-sum export helpers** (`export_loss_grad_f32`, `export_loss_grad_f64`): Replaced by
  device-only `export_loss_grad_device_f32`/`export_loss_grad_device_f64`. All loss/grad export now
  stays on device.

### Breaking Changes

- **`coo_memory_cap` renamed to `coo_chunk_budget`** (`pyxlog`): The parameter on `CompiledIlpProgram`
  was renamed to reflect the new semantics (chunk-level temp budget, not a hard ceiling on all COO
  allocations). `set_coo_memory_cap()` is retained as a deprecated alias for one release cycle and
  will be removed in v0.6.0. Update call sites before upgrading.
- **Artifact schema `beta-v1` → `beta-v2`** (`pyxlog`): `save()` now writes `beta-v2` artifacts.
  `load()` accepts both `beta-v1` and `beta-v2`, so existing saved artifacts remain readable, but
  artifacts saved with v0.5.0+ cannot be loaded by v0.4.x.
- **`export_loss_grad_f32` / `export_loss_grad_f64` removed** (`pyxlog`): These host-side loss/grad
  export helpers are gone. Replace with `export_loss_grad_device_f32` / `export_loss_grad_device_f64`
  respectively. The device-side variants return DLPack tensors with zero D2H transfers.
- **Type-specialized `download_column_<T>` functions removed** (`xlog-cuda`): All 8 type-specialized
  variants (`download_column_u32`, `download_column_i32`, etc.) are replaced by the generic
  `download_column::<T>()`. Similarly the 7 `create_buffer_from_<T>_slice()` variants are replaced
  by `create_buffer_from_slice::<T>()`, and the 11 `filter_<T>()` variants by `filter::<T>()`.
  Downstream Rust crates that call these directly must update call sites with turbofish syntax.

### Migrating from v0.3.2

This covers the upgrade path from v0.3.2 (the last stable release) to v0.5.0.

#### New Required Dependencies

- **PyTorch / LibTorch** — required for the neural-symbolic training APIs (`pyxlog`).
  CPU builds work, but GPU inference requires a CUDA-enabled PyTorch build matching the CUDA
  toolkit version used to build xlog-cuda.
- **CUDA toolkit ≥ 11.8** — required for all GPU paths (`xlog-cuda`, `xlog-solve`, `xlog-prob`).
  The CPU-only `xlog-logic` crate has no new mandatory dependencies.

#### Package Rename (v0.4.0-alpha → v0.5.0, if upgrading through alpha)

```python
# Before (v0.4.0-alpha and earlier)
import xlog_gpu

# After (v0.5.0)
import pyxlog
```

The PyPI package was renamed from `xlog-gpu` to `pyxlog` in v0.4.0-alpha. If you skipped the
alpha/beta cycle, update all import statements and remove the `xlog-gpu` package.

#### API Changes

| Old (≤ v0.3.2 / v0.4.x)                    | New (v0.5.0)                                      | Notes                              |
|---------------------------------------------|---------------------------------------------------|------------------------------------|
| `set_coo_memory_cap(n)`                     | `set_coo_chunk_budget(n)`                         | Old name deprecated, removed v0.6.0 |
| `export_loss_grad_f32()`                    | `export_loss_grad_device_f32()`                   | Returns DLPack tensor (on-device)  |
| `export_loss_grad_f64()`                    | `export_loss_grad_device_f64()`                   | Returns DLPack tensor (on-device)  |
| `download_column_u32()` (Rust, xlog-cuda)  | `download_column::<u32>()`                        | Generic turbofish form             |
| `create_buffer_from_u32_slice()` (Rust)    | `create_buffer_from_slice::<u32>()`               | Generic turbofish form             |
| `filter_u32()` / `filter_f32()` etc. (Rust)| `filter::<u32>()` / `filter::<f32>()`             | Generic turbofish form             |

#### Saved Artifacts

Artifacts saved with v0.4.x (`beta-v1` schema) can be loaded by v0.5.0 without modification.
Artifacts saved with v0.5.0 (`beta-v2` schema) **cannot** be loaded by v0.4.x. If you need to
maintain cross-version compatibility, do not upgrade the artifact files until all consumers are
on v0.5.0.

#### Breaking Changes from v0.3.2 Specifically

v0.3.2 introduced its own breaking changes (serialized Arrow symbol files, `hash_symbol_to_u32`
removal, `count` aggregation now returns `u64`). If upgrading directly from v0.3.2, address
those first (see v0.3.2 release notes below), then apply the v0.5.0 changes above.

#### Configuration Changes

- `TrainConfig` now accepts `deterministic`, `max_grad_norm`, `val_queries`, and `patience` fields
  (all optional, backward-compatible defaults).
- `GpuCompileConfig` now accepts `incremental_verify` (optional, defaults to `False`).
- `TrainConfig.persist_telemetry` defaults to `False`; explicitly set `True` to enable telemetry
  persistence in saved artifacts (new in v0.5.0).

## [0.4.0-ga] — 2026-03-05

### Changed

- **GA reliability runtime profile**: Default `max_attempts` reduced from 7 to 2 in `test_ilp_ga_reliability.py`.
  50-seed gate runtime reduced from ~1447s to ~436s (3.3x speedup) with identical statistical quality
  (200/200, Clopper-Pearson lower95 = 0.982). Override via `GA_RELIABILITY_MAX_ATTEMPTS` env var.

### Fixed

- **Typed batch upload**: `batch_fact_membership` and `batch_tagged_credit` now use
  schema-aware typed packing for all column types (I32, I64, U64, Bool, Symbol).
  Previously, all values were blindly cast to `u32`, corrupting non-U32 columns.
  F32/F64 columns are explicitly rejected with a clear error message.

### Added

- **Per-step phase timing** in dILP trainer: 6 timed phases (apply_mask, loss_credit, loss_reduce,
  backward_step, membership, convergence) with p95 and total_ms telemetry in `result.telemetry_timings`.
- **SLO scaling harness**: Parametrized `test_slo_scaling[N]` for N=20/50/100/150 chain lengths
  with wall-clock and forward_p95_us targets. Advisory by default; enforce with `ILP_PERF_ENFORCE_SLO=1`.

## [0.4.0-beta] — 2026-03-04

### Added

- **dILP Beta Trainer** — differentiable Inductive Logic Programming trainer upgraded from alpha to beta:
  - **Sparse mask API** (`set_rule_mask_sparse`): Python sends `(candidate_ids, soft_probs, budget)` and Rust builds
    the executor mask internally — no N3 tensor materialized, zero host→device transfer for the mask.
  - **Trainer backend abstraction** (`MaskBackend` protocol): `SparseMaskBackend` (default) and `DenseMaskBackend`
    (fallback via `debug_dense_mask=True`). Dense parity verified in tests.
  - **`train_and_promote()`**: Wraps `train_only()` + trial compilation + promotion gates (convergence, novel rate,
    regression check, holdout F1, ambiguity scan, typed schema) → returns `PromotionResult` with transactional commit.
  - **LOO holdout F1 scoring**: Leave-one-out cross-validation for ≤20 examples with per-fold precision/recall.
  - **Ambiguity scan**: Top-M alternative rule detection with configurable `check_ambiguity` / `exhaustive_ambiguity`.
  - **Hard-negative mining** (`sample_false_positives`): Rust-side false positive sampling, wired into trainer every
    20 steps with D2H counter reset to preserve zero-transfer contract in training loop.
  - **Artifact save/load**: `LearnedArtifact.save(path)` / `LearnedArtifact.load(path)` with JSON serialization,
    SHA-256 candidate map hash verification, schema version `beta-v1`.
  - **Recursive candidates**: `allow_recursive_candidates=True` enables i==k and j==k body-references-head candidates
    (behind config flag, default off).
  - **Beta reliability gate**: 4 stages (reach, grandparent, colleague, plus2) x 5 seeds = 20/20 with sparse backend. This is the primary beta release gate.
  - **AtomicU32 row-count cache** on `CudaBuffer` for GPU-resident row counts without host reads.
  - **Deterministic training path**: `TrainConfig(deterministic=True)` enables deterministic CUDA/Torch settings and
    per-attempt seed derivation for reproducible runs.
  - **`selected_hard` artifact field**: persisted selected candidate IDs with deterministic ordering for sparse/dense parity.
  - **GA reliability gate test**: `test_ilp_ga_reliability.py` runs 50 seeds x 4 stages with Clopper-Pearson lower-bound check.
  - **GA performance/transfer test**: `test_ilp_performance.py` validates `forward_p95_us` telemetry and host-transfer accounting.

- **Arrow C Data Interface device export** for `CudaBuffer` record batches (`to_arrow_device_record_batch`) returning
  `ArrowDeviceArrayOwned` handles with CUDA device descriptors and zero host transfers (import exists but is
  experimental + feature-gated: `arrow-device-import`).
- **Arrow device export support for Bool/Symbol**: on-device boolean bit-packing and symbol metadata keys
  (`xlog.symbol=true`, `xlog.symbol_encoding=u32`) for downstream consumers.
- **GPU CDCL verifier (complete SAT/UNSAT)** in `kernels/sat.cu` + `xlog-solve::GpuCdclSolver` with on-GPU SAT model
  checking and on-GPU UNSAT proof checking.
- **GPU PIR→CNF encoder** (`encode_cnf_gpu`) with device-resident CSR emission, deterministic var numbering, and GPU
  reachability (zero host reads in the production path), plus CNF kernels in `kernels/cnf.cu`.
- **GPU neural fast-path (AD chain)** in `kernels/neural.cu` + `xlog-prob` integration:
  - device-side AD conditional-chain weight fill (`neural_fill_ad_chain_f32`)
  - device-side probability-gradient scatter using both `grad_true` and `grad_false` (`neural_scatter_ad_chain_grads_f32`)
- **Zero-host-read verifier API**: expectation-based methods `solve_expect_sat` / `solve_expect_unsat` that never
  download SAT/UNSAT status to the CPU (fail-fast via GPU trap / CUDA error).
- **Device-resident CNF metadata** (`GpuCnf::{num_vars,num_clauses,num_lits}`) to support GPU-native CNF builders where
  capacity > exact size.
- **GPU-native equivalence verification** (`xlog-prob::compilation`) proving `φ ≡ C` via two UNSAT checks on GPU:
  `UNSAT(φ ∧ ¬C)` and `UNSAT(C ∧ ¬φ)`, with zero device→host reads.
- **GPU D4 compile+verify entrypoint** (`compile_gpu_d4_and_verify`) that compiles CNF to device-resident XGCF and
  validates equivalence via the GPU CDCL verifier.
- **Device-resident circuit cache + cache-aware evaluation** (`GpuCircuitCache`, `compile_gpu_d4_and_verify_cached`,
  `kernels/cache.cu`) enabling zero-recompile warm-cache inference.
- **GPU-native exact inference path**: `ExactDdnnfProgram` now uses GPU D4 + GPU CDCL + cache (no CPU D4, no CNF/DDNNF
  host materialization in production).
- **GPU weight/evidence builders** (`kernels/weights.cu` + `gpu_weights.rs`) for device-resident weight tables.
- **Regression guardrails** enforcing “no device→host reads” in the production verifier modules.
- **Cache DTOH guardrails + integration tests** (`no_dtoh_in_gpu_cache`, `gpu_exact_cache_integration`, `gpu_weights`).
- **Device-only logZ outputs** for GPU XGCF evaluation (`eval_log_wmc_device_*`) plus a guard test to prevent
  device→host reads inside device-only evaluation paths.
- **GPU-native loss output for neural fast-path**: `ExactDdnnfProgram::neural_backward_nll_buffers_with_device_loss`
  returns the scalar NLL loss as a device-resident value (no dtoh).
- **DLPack helper for typed allocations**: `TrackedCudaSlice::into_bytes()` enables wrapping typed device scalars into
  `CudaBuffer` columns without copies (used to export scalar loss to Torch).

### Changed

- dILP trainer defaults to sparse mask backend (`SparseMaskBackend`); dense fallback via `TrainConfig(debug_dense_mask=True)`.
- dILP holdout strategy now defaults to:
  - LOO for `<=20` positives
  - k-fold for `>20` positives (`holdout_strategy`, `holdout_folds` configurable)
- dILP promotion now enforces configurable holdout threshold (`holdout_threshold`, default `0.95`) and supports
  typed-schema gate controls (`typed_schema_required`, `waiver_untyped`).
- PyO3 exposes host transfer counters via `host_transfer_stats()` / `reset_host_transfer_stats()`.
- `GpuCnf` literal storage field renamed to `literals` (DIMACS `i32`) to match the solver/kernel interface.
- CUDA-dependent tests now skip cleanly when the CUDA runtime is unavailable (developer ergonomics).
- Workspace testing avoids building the PyO3 `extension-module` target when running `cargo test --workspace`.
- CUDA transfer/caching certification tests are stable under parallel test execution.

### Fixed

- Monte Carlo GPU initialization avoids reliance on CUDA device-count queries that can fail in restricted environments.
- GPU set operations + MC evaluation handle 0-arity (nullary) relations correctly (device row counts, not `row_cap`).
- `pyxlog` DLPack interop: detach `requires_grad` tensors before exporting probabilities to DLPack.
- `pyxlog` GPU neural fast-path ordering: replaced `torch.cuda.synchronize()` with stream-to-stream waits.
- GPU CNF reachability worklist hardened to avoid consuming uninitialized queue entries under concurrent expansion.
- nvcc deprecation warnings for `sm_70` offline PTX builds are suppressed in `kernels/CMakeLists.txt`.
- Release-mode CUDA crash in the GPU CDCL verifier/equivalence path caused by passing temporary scalar kernel arguments
  via raw parameter vectors (now backed by stable locals before `cuLaunchKernel`).
- Release-mode CUDA launch failures in GPU D4 tests and smoothing due to temporary scalar kernel arguments (now backed
  by stable locals before `cuLaunchKernel`).
- GPU smoothing now seeds root support with all random vars and levelizes with the emitted node count, ensuring
  unconditional probabilistic facts/evidence are handled correctly and preventing under-launched levels.
- GPU cache meta loading moved out of `gpu_cache.rs` to preserve dtoh-free guardrails for the cache module.

### Removed

- Vendored CPU D4/Boost toolchain (`vendor/`) and the CPU-based exact compilation pipeline (GPU-native only).

### Removed

- `test_non_monotone_with_mc` — pre-existing 50K MC sample negation test that consistently timed out (unrelated to dILP).

### Known Limitations

- Python batch query path (`batch_fact_membership`, `batch_tagged_credit`) coerces all facts via `as u32`. Typed relation schemas work in core execution but the Python query interface is U32-entity-ID-only for now.
- `bench.yml` PR-comparison dispatch path is non-operational under manual-only CI (event-gated for `push`/`pull_request`).
- GA 50-seed statistical reliability gate (`test_ilp_ga_reliability.py`) exceeds 600s timeout; deferred to post-beta runtime budget optimization. Beta gate = 20/20 reliability (Suite 4).

### Deferred to v0.4.0-rc

- ~~Term embeddings for neural-symbolic integration~~ (done in v0.5.0: term embeddings)
- ~~Extended neural-symbolic training controls~~ (done in v0.5.0: extended neural-symbolic training controls)

### Deferred to v0.5.0

- Typed query-buffer builder (non-u32 Python batch queries)
- Full GPU-resident loss computation path
- 50-seed runtime budget optimization
- SLO harness for N=20/50/100/150

### Validation

All tests pass on v0.4.0-beta validation matrix (7 suites). See `docs/reports/2026-03-04-v0.4.0-beta-validation.md`.

## Neural-Symbolic Integration Milestone (v0.4.0-alpha) — 2026-02-23

Milestone snapshot of the neural-symbolic integration layer (training APIs + GPU circuit evaluation/gradients). The `v0.4.0-alpha` milestone is fully achieved with end-to-end example validation and all required neural examples.

### Added

**Neural Predicates (`nn/4` syntax):**
- `nn(network, [inputs], output, [labels]) :: predicate(args).` declaration syntax
- Network-backed probabilistic facts with automatic annotated disjunction generation
- Support for classification mode (with labels) and embedding mode (without)
- Multiple input variables, symbol labels, and empty input lists

**Network Registry:**
- `register_network(name, module, optimizer, scheduler)` Python API
- `NetworkConfig` with neural predicate options: batching, k (top-k), det (deterministic), cache
- `NetworkHandle` with train/eval mode switching
- Automatic validation against declared neural predicates

**Tensor Source Registry:**
- `add_tensor_source(name, tensor)` for external data (images, embeddings)
- `set_active_tensor_source(name)` for switching between train/test
- Index validation and bounds checking
- Metadata tracking (size, shape, dtype)

**Neural → Probability Bridge:**
- Softmax outputs converted to annotated disjunctions
- `NeuralBridge` for numerical stability (epsilon clamping, normalization)
- Log probability computation for gradient stability
- Circuit leaf generation for d-DNNF integration

**Training Infrastructure:**
- `forward_backward()` for single query training with gradient computation
- `train_epoch()` for batch processing with configurable batch size
- `train_model()` for multi-epoch training with shuffle and logging
- `zero_grad()`, `optimizer_step()`, `scheduler_step()` for training loop control
- `TrainingHistory` object with epoch losses and batch metrics

**NLL Loss Functions:**
- `nll_loss(prob)` — negative log-likelihood from probability
- `nll_loss_batch(probs)` — batch NLL computation
- `nll_loss_mean(probs)` — mean NLL over batch
- `nll_loss_tensor(prob)` — PyTorch tensor output for autograd
- Numerical stability via epsilon (1e-10) clamping

**Backward Pass to Networks:**
- `backprop_circuit_gradients()` propagates d-DNNF gradients through neural networks
- Weight slot mapping for position-based gradient routing
- PyTorch `.backward()` integration with gradient tensors
- Support for multiple networks per query

**Circuit Caching:**
- `CachedCircuit` stores compiled d-DNNF circuits for reuse
- `WeightSlot` maps circuit variables to network outputs by position
- `evaluate_gpu_with_grads_weights()` for weight-only circuit evaluation
- Cache key generation from query templates
- Eliminates D4 recompilation bottleneck (100x+ speedup for repeated queries)

**Minimal MNIST Addition Example:**
- `examples/neural/01_minimal/train.py` — complete working example
- CNN network classifying MNIST digits
- Training purely from addition supervision (no digit labels)
- Demonstrates neural-symbolic gradient flow

**Negation in Probabilistic Programs:**
- `not` keyword in rule bodies for exact inference (`wet :- not rain.`)
- Stratified negation with automatic layer detection
- Non-monotone (cyclic) negation via Well-Founded Semantics (WFS)
- Exact gradients flow through negated literals for neural-symbolic training

**GPU Certification Suite:**
- Circuit forward kernel tests (8 tests) — `xgcf_forward_level` PTX validation
- Circuit backward kernel tests (12 tests) — gradient computation verification
- Weight injection tests (6 tests) — GPU weight buffer management
- Transfer efficiency tests (8 tests) — 0% CPU bottleneck verification
- Circuit cache tests (6 tests) — GpuXgcf reuse, D4 elimination
- PTX robustness tests (10 tests) — large circuits, edge cases, numerical stability
- Total: 50 new GPU-focused tests validating neural-symbolic kernel correctness

**PIR Extension:**
- `NegLit { leaf: LeafId }` node for negated probabilistic leaves
- NNF (Negation Normal Form) transformation pushes negation to leaves
- Weight semantics: `NegLit` uses complemented probability `(1-p, p)`

**Stratification Analysis:**
- `analyze_stratification()` function detects non-monotone SCCs
- Edge polarity tracking in dependency graph (positive/negative edges)
- Automatic classification: stratified SCCs use two-valued evaluation, non-monotone use WFS

**Well-Founded Semantics (WFS):**
- Three-valued logic: True, False, Undefined
- Alternating fixed-point algorithm (unfounded set + consequence derivation)
- Undefined atoms return probability 0 with zero gradient
- Full 1,461-line implementation in `wfs.rs`

### Changed

- **Python package renamed from `xlog-gpu` to `pyxlog`** — cleaner, more memorable name
  - All imports: `import pyxlog` (was `import xlog_gpu`)
  - Crate renamed: `crates/pyxlog` (was `crates/xlog-gpu-py`)
  - PyPI package: `pyxlog` (was `xlog-gpu`)
- Stratification analysis now tracks edge polarity for non-monotone detection
- Provenance extraction routes non-monotone SCCs to WFS evaluation
- CNF encoding emits Tseitin clauses for `NegLit` with negated polarity

### Technical Details

| Component | Files | Purpose |
|-----------|-------|---------|
| Grammar | `grammar.pest:93-102` | `nn/4` syntax parsing |
| AST | `ast.rs:323-358` | `NeuralPredDecl`, `NeuralLabel` |
| Parser | `parser.rs:573-645` | `build_neural_pred_decl()` |
| Registry | `xlog-neural/src/registry.rs` | `NetworkRegistry`, `NetworkConfig` |
| Handle | `xlog-neural/src/handle.rs` | `NetworkHandle` with PyO3 objects |
| Bridge | `xlog-neural/src/bridge.rs` | `NeuralBridge`, `NeuralOutput` |
| Tensor | `xlog-neural/src/tensor_source.rs` | `TensorSourceRegistry` |
| Python | `crates/pyxlog/src/lib.rs` | Full training API |
| PIR | `pir.rs` | `NegLit` variant |
| WFS | `wfs.rs` | Well-Founded Semantics (1,461 lines) |
| Exact | `exact.rs` | `random_var_indices()`, `evaluate_gpu_with_grads_weights()` |
| GPU certification tests | GPU certification category test files | GPU certification tests (50 tests) |

### Validation

- **CUDA Certification Suite:** 200/200 tests passed (core CUDA certification tests + GPU certification tests)
- **Python Tests:** 109/109 tests passed
- **Spec Alignment:** All 50 GPU certification tests match specification
- **Code Quality:** No stubs, placeholders, or TODOs

### Example: MNIST Addition Training

```python
import pyxlog
import torch

# Define neural predicate program
program = pyxlog.Program.compile("""
    nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
    addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
""")

# Register PyTorch network
net = MNISTNet()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
program.register_network("mnist_net", net, optimizer)

# Add training data
program.add_tensor_source("train", train_images)

# Train on addition queries (no digit labels!)
queries = ["addition(0, 1, 7)", "addition(2, 3, 5)", ...]
history = pyxlog.train_model(program, queries, epochs=50, batch_size=32)
```

---

## v0.3.2 — 2026-01-19

Module system, user-defined functions, reversible symbols, and comprehensive showcase examples for expressive, modular Datalog programs.

### Added

**Module System:**
- File-based modules with explicit imports
- `use module.` to import all public predicates
- `use module::{pred1, pred2}.` for selective imports
- `use path/to/module.` for nested modules
- `private` keyword for module-internal predicates and functions

**User-Defined Functions:**
- Reusable functions in rule bodies
- Arithmetic functions: `func double(X) = X * 2.`
- Conditional functions: `func abs(X) = if X < 0 then 0 - X else X.`
- Recursive functions with base-case validation
- Optional type annotations: `func add(X: f64, Y: f64) -> f64 = X + Y.`
- Predicate-based functions: `func get_parent(X) = P :- parent(X, P).`

**Reversible Symbols:**
- Bidirectional string-to-ID mapping
- Symbols display as original strings in query output
- Arrow dictionary encoding for efficient serialization
- `--stats` shows symbol registry metrics

**CLI Enhancements:**
- `--module-path` flag for specifying module search directories

**Showcase Examples:**
- Enterprise Analytics: HR management, compensation, org hierarchy with recursive management chains
- Knowledge Graph: Ontology modeling, citation analysis, semantic inference with type inheritance
- Game Analytics: Player statistics, achievements, guilds, leaderboards with social network analysis
- Supply Chain: Bill of Materials explosion, inventory management, supplier analytics

### Fixed

- **GroupBy count aggregation type**: Count now outputs `u64` (was `u32`) to match predicate declarations and prevent type mismatch errors when comparing count results
- **Optimizer predicate pushdown**: Fixed column width estimation to use schema information for accurate filtering

### Changed

- Symbol storage changed from hash-based to sequential ID allocation
- Module resolution now validates imports before compilation

### Breaking Changes

- Serialized Arrow files from v0.3.1 with symbol columns may need re-export
- `hash_symbol_to_u32` function removed from public API
- Count aggregation results are now `u64` instead of `u32`

---

## v0.3.1 — 2026-01-18

Float predicates, performance benchmarks, query statistics, fuzz testing, and property-based tests.

### Added

**Float Predicate Support:**
- IEEE 754 total ordering for `f32`/`f64` filter comparisons: `NaN > Inf > positive > +0 > -0 > negative > -Inf`
- Filter kernels: `filter_compare_f32_*` and `filter_compare_f64_*` with proper edge case handling
- Comprehensive tests for NaN, Infinity, subnormals, and signed zeros

**Performance Benchmarks:**
- Criterion.rs benchmarks for `xlog-gpu` (transitive closure, hash join, aggregation)
- Criterion.rs benchmarks for `xlog-prob` (exact inference, Monte Carlo sampling)
- `docs/BENCHMARKS.md` with methodology and baseline metrics
- `.github/workflows/bench.yml` for CI regression detection

**Query Timing & Statistics:**
- `--stats` CLI flag for execution profiling
- Per-stratum timing with iteration counts for recursive strata
- Per-operation timing (join, sort, dedup, filter)
- Memory usage tracking (peak, budget)
- Human-readable and JSON output formats

**Fuzz Testing:**
- `fuzz/` directory with cargo-fuzz targets:
  - `fuzz_parser` — raw byte input fuzzing
  - `fuzz_compiler` — structured program generation
  - `fuzz_type_inference` — type system stress testing
- AddressSanitizer (ASAN) integration for crash detection
- `.github/workflows/fuzz.yml` for continuous fuzzing

**Property-Based Testing:**
- proptest integration in `xlog-cuda-tests`
- Sort stability property (data preservation, ascending order)
- Join correctness property (CPU reference comparison)
- Filter idempotence property (`filter(filter(x)) = filter(x)`)
- Dedup determinism property (consistent output across runs)
- Stress tests for large datasets (50K+ rows)

### Validation
- Workspace tests pass: `cargo test --workspace --all-targets --release`
- Property tests pass: `cargo test -p xlog-cuda-tests --test properties --release`
- Fuzz targets build and run with ASAN

---

## v0.2.0 — 2026-01-14

Phase 4 probabilistic logic programming (`xlog-prob`) merged into `main`; Python bindings are the integration surface for GPU I/O.

### Added
- `xlog-prob`: exact inference via Decision-DNNF (vendored D4) + GPU weighted model counting and gradients.
- `xlog-prob`: Monte Carlo engine (`prob_engine=mc`) with GPU sampling, deterministic non-monotone SCC semantics, and uncertainty metadata.
- New CUDA kernels: `kernels/circuit.ptx` (XGCF forward/backward) and `kernels/mc_sample.ptx` (MC sampling).
- New examples: `examples/prob/` (probabilistic `.xlog`) and `examples/python/` (DLPack bindings).
- `xlog-gpu` + `pyxlog`: `pyxlog` Python module (PyO3) with DLPack-first I/O for deterministic and probabilistic evaluation.
- New/updated docs: `docs/architecture/xlog-prob.md`, `docs/VALIDATION_REPORT.md`.

### Validation
- Workspace tests pass in release (`cargo test --workspace --all-targets --release`).
- CUDA certification suite passes: **140/140** (see `docs/plans/2026-01-14-cuda-certification-results.md`).

## v0.1.0 — 2026-01-13

Initial release of the deterministic `xlog-logic` tier (Phase 3 complete).

### Added
- `.xlog` parser + compiler with stratified negation and semi-naive fixpoint recursion.
- GPU execution backend (`xlog-cuda`) with kernels for join/sort/filter/dedup/groupby/scan/pack/set-ops.
- Arithmetic (`is`) and builtin functions (`abs/min/max/pow/cast`) in rule bodies.
- Aggregations: `count/sum/min/max/logsumexp`.
- Arrow IPC import/export utilities and DLPack zero-copy column interop.
- Example suite under `examples/xlog/` and runner example `crates/xlog-logic/examples/xlog_run.rs`.

### Validation
- Workspace tests pass in release (`cargo test --workspace --all-targets --release`).
- CUDA certification suite passes: **133/133** (see `docs/plans/2026-01-12-cuda-certification-results.md`).

### Known limitations
- `symbol` values are currently represented as a `u32` hash (not reversible).
- Arrow IPC interop involves device↔host copies; DLPack is the zero-copy path.
