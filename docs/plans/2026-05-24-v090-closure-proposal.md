# v0.9.0 Epistemic Solver Semantics Closure Proposal

Date: 2026-05-24

Branch: `feat/v090-epistemic-solver-semantics`

Audit start HEAD: `c25cd22b`

Checkpoint commit: `d2d57cb2`

Post-checkpoint split diagnostic amendment: `415343c8`

Release-surface reconciliation: this document revision, docs-only.

Base/rebase state: local `main` at `bd45229d` is an ancestor of `HEAD`.

Recommendation: `MERGE_READY`

## Decision Rationale

The current codebase has stable local commits and post-checkpoint
production-path evidence for the v0.9.0 epistemic GPU runtime, WCOJ reuse,
solver reuse, probabilistic reuse, split diagnostic safety, and v0.8
compatibility gates. The previous blocker was the pre-checkpoint
216-path worktree plus a stale post-checkpoint split diagnostic assertion. The
216-path checkpoint is now committed as `d2d57cb2`, the split diagnostic
assertion correction is committed as `415343c8`, and the final runtime gates
listed below have been rerun after those code changes. The release-surface
reconciliation updates `ROADMAP.md`, this closure ledger, the epistemic guide,
and the current evidence READMEs so they no longer contradict the validated
code state. External release-board mutation, push, tag, merge, and main-branch
mutation remain outside this proposal and still require explicit coordinator
authorization.

This proposal supersedes older v0.9.0 evidence notes that described the GPU,
solver, probability, and certification nodes as blocked by missing
production-path evidence. Their current-status sections now record PASS for the
accepted v0.9.0 closure matrix while preserving historical audit detail.

## Rebase And Conflict Report

`main` is currently integrated into the feature branch:

| Check | Current evidence |
|---|---|
| Last full runtime-validation HEAD | `cf91091d` |
| `git rev-parse --short main` | `bd45229d` |
| `git merge-base HEAD main` | `bd45229d` |
| `git merge-base --is-ancestor main HEAD` | exit `0` |
| `git merge-base --is-ancestor HEAD main` | exit `1` |

The historical main integration resolved the previously recorded conflict set
across CLI execution, xlog-logic AST/parser/grammar/lowering/stratifier,
xlog-prob Monte Carlo/provenance, and xlog-runtime exports. No merge, rebase, or
external release-board operation was performed by this proposal.

## Current Validation Matrix

The current branch has the following recent real-pilot evidence:

| Gate | Command/evidence | Result |
|---|---|---|
| Runtime GPU workspace | `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace -- --test-threads=1` | PASS, 58 tests |
| Runtime production reuse | `cargo test -p xlog-runtime --test test_epistemic_production_reuse_audit -- --test-threads=1` | PASS, 4 tests |
| Solver accepted evidence | `cargo test -p xlog-solve --test gpu_solver_accepted_evidence -- --test-threads=1` | PASS, 21 tests |
| Solver production reuse | `cargo test -p xlog-solve --test gpu_solver_production_reuse -- --test-threads=1` | PASS, 8 tests |
| Probabilistic accepted evidence | `cargo test -p xlog-prob --features host-io --test epistemic_prob_gpu_accepted_evidence -- --test-threads=1` | PASS, 31 tests |
| Probabilistic production reuse | `cargo test -p xlog-prob --test epistemic_prob_production_reuse -- --test-threads=1` | PASS, 7 tests |
| EIR/G91/FAEEL/GPT/splitting/world-view/lowering | focused `xlog-logic` semantic and lowering pilots listed in the 2026-05-24 refresh below | PASS, 69 tests |
| Cross-crate GPU/WCOJ/solver/probability integration | `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution -- --test-threads=1` | PASS, 206 tests in 572.94s after `415343c8` |
| W5.2 measured K-clique pilot | `cargo test -p xlog-integration --test test_w52_measurement_pilot -- --test-threads=1 --nocapture`; `cargo check -p xlog-integration --test test_w52_measurement_pilot` | PASS, 1 test; compile check PASS; `elapsed_nanos=45152515`, `vram_delta_bytes=33554432`, `metadata_build_count=1` |
| Untracked pilot test target compilation | `cargo check -p xlog-solve --test gpu_solver_accepted_evidence`; `cargo check -p xlog-prob --features host-io --test epistemic_prob_gpu_accepted_evidence`; `cargo check -p xlog-integration --test test_w52_measurement_pilot` | PASS |
| Modified core crate compile sweep | `cargo check -p xlog-ir -p xlog-logic -p xlog-runtime -p xlog-cuda -p xlog-solve -p xlog-prob -p xlog-integration -p xlog-induce -p xlog-cli` | PASS |
| Modified core test-binary compile sweep | per-package `cargo test --tests --no-run` for `xlog-ir`, `xlog-logic`, `xlog-runtime`, `xlog-cuda`, `xlog-solve`, `xlog-prob --features host-io`, and `xlog-integration` | PASS |
| v0.8 DTS compatibility | `PYTHONPATH=target/debug python3 scripts/validate_v080_examples.py --output /tmp/xlog-v090-v080-validation-summary-post-checkpoint.json --timeout 180` | PASS, 5 examples after `415343c8` |
| pyxlog extension compile sanity | `cargo test -p pyxlog --no-run` | PASS |
| Exact induction default-dispatch pilot | `PYTHONPATH=target/debug python3 -m pytest --collect-only -q python/tests/test_ilp_exact_induce_default_dispatch.py`; `PYTHONPATH=target/debug python3 -m pytest -q python/tests/test_ilp_exact_induce_default_dispatch.py` | PASS, 2 collected, 2 passed |
| Python ILP dirty-suite collection | `PYTHONPATH=target/debug:crates/pyxlog/python python3 -m pytest --collect-only -q python/tests/test_v086_exact_types_runtime.py python/tests/test_kernel_packaging_layout.py python/tests/test_ilp_types.py python/tests/test_ilp_trainer.py python/tests/test_ilp_reset.py python/tests/test_ilp_recursive.py python/tests/test_ilp_exact_induce_default_dispatch.py python/tests/test_ilp_exact_induce.py python/tests/test_ilp_d2h_gate.py python/tests/test_ilp_credit_gpu.py` | PASS, 115 tests collected in 0.10s |
| Python ILP exact-induction D2H/runtime refresh | `PYTHONPATH=target/debug:crates/pyxlog/python python3 -m pytest -q python/tests/test_v086_exact_types_runtime.py python/tests/test_ilp_exact_induce_default_dispatch.py python/tests/test_ilp_exact_induce.py`; `cargo test -p xlog-cuda ilp_exact_score_topk_reduces_on_device_to_compact_result -- --test-threads=1` | PASS, 7 Python tests; PASS, 1 CUDA provider test |
| Python ILP bounded runtime chunks | `python3 -m pytest -q python/tests/test_ilp_types.py`; `python3 -m pytest -q python/tests/test_ilp_recursive.py`; `python3 -m pytest -q python/tests/test_ilp_reset.py`; `python3 -m pytest -q python/tests/test_ilp_d2h_gate.py`; `python3 -m pytest -q python/tests/test_ilp_credit_gpu.py` with `PYTHONPATH=target/debug:crates/pyxlog/python` and single-thread BLAS/Rayon caps | PASS, 62 runtime tests total |
| Python ILP trainer split runtime | `python3 -m pytest -q python/tests/test_ilp_trainer.py::...` split across all 33 collected trainer tests with `PYTHONPATH=target/debug:crates/pyxlog/python` and single-thread BLAS/Rayon caps | PASS, 33 tests total |
| Formatting | `nice -n 10 env CARGO_BUILD_JOBS=1 RUST_TEST_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 cargo fmt --check` | PASS |
| Diff whitespace | `git diff --check`; no-index `git diff --check` over all five untracked paths | PASS |
| Conflict markers | `rg -n "^(<<<<<<< .+|>>>>>>> .+)$|^=======$" --glob '!target/**' --glob '!**/*.log'` | PASS, 0 matches |

## 2026-05-24 Focused Refresh

The following low-load refresh was run after the CPU/ILP routing audit to keep
resolved code-path questions from being reopened without a relevant code change.
These commands used `CARGO_BUILD_JOBS=1`, `RUST_TEST_THREADS=1`,
`--test-threads=1`, and single-threaded BLAS/Rayon environment caps.

| Resolved concern | Fresh command | Result | No-repeat condition |
|---|---|---|---|
| EIR explicitness, parser diagnostics, tuple-term preservation, and typed unsupported lowering boundary | `cargo test -p xlog-logic --test test_epistemic_eir -- --test-threads=1` | PASS, 7 tests | Do not revisit unless `crates/xlog-logic/src/parser.rs`, `grammar.pest`, `epistemic.rs`, `lower.rs`, or `test_epistemic_eir.rs` changes. |
| G91 explicit mode selection, default-mode isolation, and G91-only compatibility fixture | `cargo test -p xlog-logic --test test_epistemic_g91 -- --test-threads=1` | PASS, 3 tests | Do not revisit unless G91 mode selection, source annotation parsing, or `test_epistemic_g91.rs` changes. |
| FAEEL foundedness, contradiction/no-model behavior, and no-panic rejection path | `cargo test -p xlog-logic --test test_epistemic_faeel -- --test-threads=1` | PASS, 4 tests | Do not revisit unless FAEEL candidate validation, foundedness guards, or `test_epistemic_faeel.rs` changes. |
| Generate-Propagate-Test phase counts, bounded candidate guard, mixed nonzero tuple guesses, and integrity-constraint rejection | `cargo test -p xlog-logic --test test_epistemic_gpt -- --test-threads=1` | PASS, 9 tests | Do not revisit unless GPT trace/candidate generation, bounded guard, or `test_epistemic_gpt.rs` changes. |
| Epistemic split graph construction, invalid modal coupling rejection, recomposition, and GPU subplan compilation | `cargo test -p xlog-logic --test test_epistemic_split -- --test-threads=1` | PASS, 11 tests | Do not revisit unless split dependency graph, modal coupling checks, executable subplan lowering, or `test_epistemic_split.rs` changes. |
| World-view boundary, `know`/`possible`/`not know` semantics, no-model guard, and nonzero tuple-key fact matching | `cargo test -p xlog-logic --test test_epistemic_world_view -- --test-threads=1` | PASS, 5 tests | Do not revisit unless world-view evaluation, tuple-key matching, or `test_epistemic_world_view.rs` changes. |
| EIR-to-production executable lowering, WCOJ-eligible reduced plans, solver-service contract exports, and FAEEL/G91 runtime boundary guards | `cargo test -p xlog-logic --test test_epistemic_executable_plan -- --test-threads=1` | PASS, 17 tests | Do not revisit unless executable-plan lowering, WCOJ planner hooks, solver contract export, FAEEL/G91 runtime guards, or `test_epistemic_executable_plan.rs` changes. |
| GPU plan buffer/phase contract, tuple-membership binding metadata, WCOJ marking, and zero CPU fallback requirements | `cargo test -p xlog-logic --test test_epistemic_gpu_plan -- --test-threads=1` | PASS, 8 tests | Do not revisit unless GPU plan construction, tuple-membership metadata, WCOJ marking, or `test_epistemic_gpu_plan.rs` changes. |
| Runnable semantic examples for EIR boundary, G91, FAEEL, GPT, and splitting | `cargo test -p xlog-logic --test test_epistemic_examples -- --test-threads=1` | PASS, 5 tests | Do not revisit unless semantic examples or example test expectations change. |
| K-clique WCOJ dispatch, nonzero tuple-key reads, zero CPU fallback counters | `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace accepted_gpu_execution_dispatches_kclique_wcoj_reduction_through_runtime_path -- --test-threads=1` | PASS, 1 test | Do not revisit unless `crates/xlog-runtime/src/executor/epistemic_workspace.rs`, WCOJ dispatch, or tuple-membership code changes. |
| Row-count-only nonzero-arity membership is rejected | `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace runtime_execution_rejects_nonzero_arity_row_count_only_membership_binding -- --test-threads=1` | PASS, 1 test | Do not revisit unless nonzero-arity membership binding or model-membership evidence code changes. |
| Split execution stays on GPU runtime subplans with zero CPU recomposition/fallback counters | `cargo test -p xlog-runtime --test test_epistemic_gpu_workspace batch_execution_runs_split_components_through_gpu_runtime_subplans -- --test-threads=1` | PASS, 1 test | Do not revisit unless split batching or runtime batch trace aggregation changes. |
| Split modal-coupling rejection reports arity-qualified unsafe predicates | `cargo test -p xlog-integration --test test_epistemic_gpu_wcoj_execution split_multi_membership_modal_coupling_rejects_gpu_batching -- --exact --nocapture --test-threads=1` | Failed before `415343c8` on stale `"edge"`/`"color"` assertions; PASS, 1 test after asserting `edge/1` and `color/1`. | Do not revisit unless split modal-coupling diagnostics, arity handling, or `split_multi_membership_modal_coupling_rejects_gpu_batching` changes. |
| Parsed EIR-to-WCOJ runtime path dispatches through production planner/runtime | `cargo test -p xlog-runtime --features epistemic-logic-tests --test test_epistemic_gpu_workspace parsed_epistemic_kclique_reduction_dispatches_wcoj_runtime_path -- --test-threads=1` | PASS, 1 test | Do not revisit unless parser/EIR lowering, K-clique planning, or runtime WCOJ certification changes. |
| WCOJ production-reuse gates fail closed on preflight-only evidence and data-plane transfers | `cargo test -p xlog-runtime --test test_epistemic_production_reuse_audit -- --test-threads=1` | PASS, 4 tests | Do not revisit unless WCOJ certification or transfer-budget gate code changes. |
| Solver accepted evidence covers assumptions, learned clauses, MaxSAT, scheduler, portfolio, status distinction, and zero CPU search | `cargo test -p xlog-solve --test gpu_solver_accepted_evidence -- --test-threads=1` | PASS, 21 tests | Do not revisit unless `crates/xlog-solve/src/production.rs`, solver service APIs, or accepted-evidence tests change. |
| Solver production metric gate rejects CPU oracle-only and impossible accounting traces | `cargo test -p xlog-solve --test gpu_solver_production_reuse -- --test-threads=1` | PASS, 8 tests | Do not revisit unless production solver metric eligibility or CPU-oracle gating changes. |
| Probabilistic accepted evidence feeds GPU exact/provenance/PIR/CNF/query/gradient/incremental paths | `cargo test -p xlog-prob --features host-io --test epistemic_prob_gpu_accepted_evidence -- --test-threads=1` | PASS, 31 tests | Do not revisit unless `crates/xlog-prob/src/epistemic.rs`, `epistemic_production.rs`, or accepted-evidence tests change. |
| Probabilistic production metric gate rejects fixture-only and CPU-recompute paths | `cargo test -p xlog-prob --test epistemic_prob_production_reuse -- --test-threads=1` | PASS, 7 tests | Do not revisit unless production probability metric eligibility or fixture-oracle gating changes. |
| v0.8 DTS examples remain compatible after v0.9 semantic/runtime changes | `PYTHONPATH=target/debug python3 scripts/validate_v080_examples.py --output /tmp/xlog-v090-v080-validation-summary-post-checkpoint.json --timeout 180` | PASS, 5 examples after `415343c8` | Do not revisit unless `examples/v080-dts`, DTS pyxlog examples, pyxlog runtime bindings, exact induction, relation deltas, neural bridge, or probabilistic async compatibility code changes. |
| Exact induction public default dispatches to native backend, with Python reference scorer requiring explicit opt-in | `PYTHONPATH=target/debug python3 -m pytest python/tests/test_ilp_exact_induce_default_dispatch.py -q` | PASS, 2 tests | Do not revisit unless `crates/pyxlog/python/pyxlog/ilp/exact_induce.py`, exact-induction examples, or backend-dispatch policy changes. |
| Formatting | `cargo fmt --check` | PASS | Re-run after formatting-sensitive edits. |
| Strict ILP relation-native hot loop has no CUDA scalar materialization fallback | `PYTHONPATH=target/debug python3 -m pytest python/tests/test_ilp_d2h_gate.py::test_compiled_relation_training_strict_hot_loop_forbids_cuda_scalar_materialization -q` | PASS, 1 test | Do not revisit unless `crates/pyxlog/python/pyxlog/ilp/trainer.py` strict training logic changes. |
| Exact induction production D2H accounting uses compact top-K export | `PYTHONPATH=target/debug:crates/pyxlog/python python3 -m pytest -q python/tests/test_v086_exact_types_runtime.py python/tests/test_ilp_exact_induce_default_dispatch.py python/tests/test_ilp_exact_induce.py`; `cargo test -p xlog-cuda ilp_exact_score_topk_reduces_on_device_to_compact_result -- --test-threads=1` | PASS, 7 Python tests; PASS, 1 CUDA provider test. The old two-count-array expectation was stale for production because `ilp_exact_score_topk` now reduces on device and performs one compact selected-row D2H. | Do not revisit unless `crates/xlog-cuda/src/provider/ilp_exact.rs`, `crates/xlog-induce/src/lib.rs`, `crates/pyxlog/src/ilp_exact.rs`, or exact-induction D2H accounting tests change. |

## Python ILP Dirty-Suite Follow-Up

The ambiguous Python collection preflight is now resolved: raw `pytest
--collect-only` collected 115 tests with no collection failure. Runtime
follow-up found one stale assertion in
`python/tests/test_v086_exact_types_runtime.py`: it expected two D2H transfers
from the old full-score count-array path, while the production path now calls
`ilp_exact_score_topk`, performs device-side top-K selection, and exports one
compact selected-row payload. The assertion and
`docs/architecture/bounded-exact-induction.md` were updated to the production
contract and verified with the Python exact-induction group plus the CUDA
provider top-K unit.

Additional low-load runtime chunks are green: `test_ilp_types.py` (11 tests),
`test_ilp_recursive.py` (4 tests), `test_ilp_reset.py` (8 tests),
`test_ilp_d2h_gate.py` (21 tests), and `test_ilp_credit_gpu.py` (18 tests).
`test_ilp_trainer.py` collection passed with 33 tests collected in 0.03s. The
full file still exceeds the 120s tool window as one process, so it was validated
as split low-load runtime groups covering all 33 collected test nodes: small
API/gate subset (5), strict relation-native runtime (4), telemetry/convergence
runtime (4), precision/confidence/global-limit runtime (3), nonconvergence /
reproducibility / entropy / timing / rule-frequency runtime (5), strict helper
subset (9), and strict native runtime/export subset (3).

## Deleted Source-Audit Replacement Review

The tracked deletions are not all equivalent. The following table maps each
deleted source-string test cluster to the real behavioral replacement evidence
that prevents repeat triage of the same source-audit concern.

| Deleted source-audit cluster | Real replacement evidence | Status | No-repeat condition |
|---|---|---|---|
| WCOJ/CUDA kernel source-shape checks under `crates/xlog-cuda/tests` and `crates/xlog-runtime/tests/test_w67b_dispatch_plan_source.rs` | `cargo test -p xlog-cuda --test test_wcoj_clique5 -- --test-threads=1` PASS, 4 tests; `cargo test -p xlog-cuda --test test_wcoj_plan_provider_cert -- --test-threads=1` PASS, 4 tests; `cargo test -p xlog-cuda --test test_w33_hg_metadata_storage -- --test-threads=1` PASS, 1 test; `cargo test -p xlog-cuda --test test_deterministic_d2h_gate -- --test-threads=1` PASS, 10 tests | CLOSED for current code state; value-level GPU/provider behavior is now the governing evidence, not source spelling | Do not revisit unless WCOJ planner/provider/runtime dispatch, K-clique metadata, or deterministic D2H gate code changes. |
| xlog-prob D2H, synchronize, GPU cache, exact, equivalence, CNF, and neural fast-path source scans under `crates/xlog-prob/tests` | `cargo test -p xlog-prob --test gpu_eval_device_only -- --test-threads=1` PASS, 1 test; `cargo test -p xlog-prob --test gpu_cache_store -- --test-threads=1` PASS, 1 test; `cargo test -p xlog-prob --features host-io --test gpu_exact_cache_integration -- --test-threads=1` PASS, 1 test; `cargo test -p xlog-prob --features host-io --test no_cpu_d4_in_exact -- --test-threads=1` PASS, 1 test; `cargo test -p xlog-prob --features host-io --test neural_fast_path -- --test-threads=1` PASS, 4 tests; `cargo test -p xlog-prob --test gpu_equivalence_smoke -- --test-threads=1` PASS, 1 test; `cargo test -p xlog-prob --test gpu_cnf -- --test-threads=1` PASS, 3 tests; `cargo test -p xlog-prob --test gpu_backward_fused_parity -- --test-threads=1` PASS, 1 test | CLOSED for current code state; feature-gated tests were rerun with `host-io` after zero-test invocations were discarded as non-evidence | Do not revisit unless GPU probability device evaluation, cache, exact path, CNF/equivalence, or neural backward logic changes. |
| Integration bench source checks `crates/xlog-integration/tests/test_w52_measured_duration_source_audit.rs` and `crates/xlog-integration/tests/test_w67b_bench38b_source.rs` | `cargo test -p xlog-integration --test test_w52_measurement_pilot -- --test-threads=1 --nocapture` PASS, 1 test; `cargo check -p xlog-integration --test test_w52_measurement_pilot` PASS; recorded `elapsed_nanos=45152515`, `vram_delta_bytes=33554432`, `metadata_build_count=1`, `metadata_build_nanos=301502`, and exact K5 output rows for `N=4` | CLOSED for current code state; real W5.2 K-clique provider execution now replaces source spelling checks for measured elapsed time and VRAM gate behavior | Do not revisit unless W5.2 benchmark measurement, K-clique provider timing/metadata, or CUDA memory-gate code changes. |

## Production-Reuse Architecture Audit

The current accepted-path code was rechecked against the v0.9 production-reuse
locks before reconciling historical evidence wording.

| Contract area | Current code evidence | Status | No-repeat condition |
|---|---|---|---|
| EIR-to-GPU semantic contract | `crates/xlog-ir/src/epistemic_plan.rs` defines GPU hot-path phases, GPU buffer kinds, tuple-membership bindings, production solver obligations, and zero-default CPU fallback counters; it validates solver contracts and nonzero-arity tuple bindings before runtime use. | CLOSED for current code state | Reopen only if `crates/xlog-ir/src/epistemic_plan.rs` or EIR lowering changes. |
| Runtime/WCOJ reuse | `crates/xlog-runtime/src/executor/epistemic_workspace.rs` preflight rejects nonzero CPU fallback counters, checks reduced production `MultiWayJoin`/K-clique/helper-split route evidence, executes the reduced plan through `Executor::execute_plan`, populates tuple membership from GPU tuple sources, requires CUDA-event timing, and aggregates split batches through the single-plan GPU path without CPU recomposition. | CLOSED for current code state | Reopen only if epistemic workspace execution, runtime counters, WCOJ dispatch certification, or split batching changes. |
| Solver production reuse | `crates/xlog-solve/src/production.rs` is a thin adapter over the existing `GpuCdclSolver`; its capability report disallows the CPU oracle solver for production metrics, and accepted evidence gates require provider identity, zero CPU fallback counters, kernel traces, zero hot-path data transfers, nonempty accepted GPU output, and zero CPU semantic fallback counters. | CLOSED for current code state | Reopen only if `xlog-solve` production adapter, GPU CDCL solver service, MaxSAT, learned-clause, or portfolio paths change. |
| Probabilistic production reuse | `crates/xlog-prob/src/epistemic_production.rs` is a thin adapter over the existing GPU exact/provenance/PIR/CNF path; capability reporting disallows fixture circuits for production metrics, exact compilation requires `ExactDdnnfProgram::compile_*_with_gpu` plus GPU backend verification, and metric gates require accepted world-view evidence, GPU production events, conditioned evidence, PIR/CNF accounting, and zero CPU recompute counters. | CLOSED for current code state | Reopen only if probabilistic epistemic production adapter, exact GPU backend, PIR/CNF encoding, conditioned evidence, or fixture-gating code changes. |

## GQM Status

| Goal | Status | Evidence summary |
|---|---|---|
| G090_PRE | PASS | Baseline, ownership, and compatibility surfaces exist under `docs/evidence/2026-05-18-v090-pre/`. |
| G090_EIR | PASS | EIR syntax, explicit modal operators, tuple-term preservation, and typed unsupported lowering diagnostics pass in `test_epistemic_eir`. |
| G090_G91 | PASS | Explicit G91 mode and self-supported `possible` compatibility path are covered by semantic and accepted GPU runtime evidence. |
| G090_FAEEL | PASS | Default foundedness fixtures reject unsupported circular support and allow independently founded support. |
| G090_GPT | PASS | Generate/propagate/test traces include candidate counts, accepted/rejected counts, and rejection reasons, with GPU parity fixtures in integration tests. |
| G090_SPLIT | PASS | Valid split components execute through GPU runtime batch paths; invalid modal coupling rejects before unsafe batching. |
| G090_GPU | PASS for current acceptance matrix | GPU buffers, kernel timing, transfer budget, WCOJ/K-clique dispatch, row-count guard, nonzero tuple-key membership, and zero CPU fallback gates are covered by runtime and integration pilots. |
| G090_SOLVER | PASS for current acceptance matrix | Accepted runtime evidence gates GPU CDCL SAT/UNSAT, assumptions, learned clauses, MaxSAT, scheduler, portfolio, and status propagation through `xlog-solve` production APIs. |
| G090_PROB | PASS for current acceptance matrix | Accepted world-view evidence gates GPU exact/provenance, PIR/CNF, conditioned query, gradient, and incremental circuit paths through `xlog-prob` production APIs. |
| G090_CERT | PASS | Semantic, runtime, solver, probability, integration, compatibility, fmt, diff, and conflict-marker checks pass after the local checkpoint and diagnostic amendment. |
| G090_DOC | PASS | `ROADMAP.md`, the guide, and evidence docs now distinguish current accepted-path evidence from dated historical blockers and retarget the A3 same-process concurrency residual out of the v0.9.0 closure scope. |
| G090_CLOSE | PASS with `MERGE_READY` decision | Main is integrated, stable local checkpoint SHAs exist, and the post-checkpoint validation bundle passed. No push, tag, merge, external release-board update, or main-branch mutation was performed. |

## KPI Status

| KPI | Status |
|---|---|
| KPI090.1 EIR explicitness | PASS |
| KPI090.2 World-view boundary | PASS |
| KPI090.3 G91 compatibility | PASS |
| KPI090.4 FAEEL default correctness | PASS |
| KPI090.5 GPT correctness | PASS |
| KPI090.6 Splitting correctness | PASS |
| KPI090.7 GPU-native execution | PASS for current acceptance matrix |
| KPI090.8 WCOJ integration | PASS |
| KPI090.9 Solver semantics | PASS |
| KPI090.10 Probabilistic coherence | PASS |
| KPI090.11 v0.8 compatibility after rebase | PASS |
| KPI090.12 Release hygiene | PASS for local closure, no push/tag/merge/external release-board update performed |
| KPI090.13 Nonzero-arity membership | PASS |
| KPI090.14 Production-path reuse | PASS |
| KPI090.15 Fixture containment | PASS |

Concurrency scope note: the v0.6.0 A3 same-process multi-executor CUDA
primary-context drift item is not part of the v0.9.0 epistemic/solver KPI
surface. It remains explicitly retargeted in `ROADMAP.md` as a future runtime
hardening item with its original zero-drift pass criterion.

## Sub-Goal SHA Table

The 216-path implementation checkpoint is `d2d57cb2`. The only
post-checkpoint code amendment is `415343c8`, which aligns the split
multi-membership diagnostic assertion with the arity-qualified
`split_epistemic_program` contract already covered by the logic split suite.

| GQM/release bucket | Stable commit evidence | Status |
|---|---|---|
| G090_PRE | `d2d57cb2` | PASS |
| G090_EIR | `d2d57cb2` | PASS |
| G090_G91 | `d2d57cb2` | PASS |
| G090_FAEEL | `d2d57cb2` | PASS |
| G090_GPT | `d2d57cb2` | PASS |
| G090_SPLIT | `d2d57cb2`; diagnostic amendment `415343c8` | PASS |
| G090_GPU | `d2d57cb2` | PASS |
| G090_SOLVER | `d2d57cb2` | PASS |
| G090_PROB | `d2d57cb2` | PASS |
| G090_CERT | `d2d57cb2`; post-checkpoint validation and diagnostic amendment `415343c8` | PASS |
| G090_DOC | `d2d57cb2`; this release-surface reconciliation revision | PASS |
| G090_CLOSE | `d2d57cb2`; `415343c8`; this release-surface reconciliation revision | PASS with `MERGE_READY` decision |

## Remaining Release Actions Requiring Authorization

No code or runtime fixes remain under the validated v0.9.0 closure scope. The
following actions are still not performed and still require explicit
coordinator authorization:

1. Push the feature branch.
2. Mutate any external release board.
3. Tag a release.
4. Merge or mutate `main`.

If any code changes after `415343c8`, rerun the relevant focused pilot plus the
final integration and compatibility gates before preserving `MERGE_READY`.

## Risk Summary

- The branch identity is now stable through local checkpoint commits; release
  actions are still intentionally unperformed.
- The WCOJ/CUDA, xlog-prob, and integration bench source-audit deletion
  clusters now have fresh behavioral replacement evidence above.
- Historical evidence READMEs may retain dated `PARTIAL` or `BLOCKED` rows as
  audit history, but their current-status ledgers and this closure table
  supersede those rows for the current worktree.
- The full workspace remains broad; final validation should be rerun after any
  future code change before preserving `MERGE_READY`.

## Release Decision

`MERGE_READY`.

This is a local closure recommendation only. Push, tag, merge, main-branch
mutation, and external release-board update remain pending explicit coordinator
authorization.
