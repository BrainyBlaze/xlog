# v0.9.0 Production Reuse Source Audit

Date: 2026-05-18

Goal node: `G090_CERT - Certification And Regression Gates`

Metric: `M090_CERT.13 no parallel engines`

Branch: `feat/v090-epistemic-solver-semantics`

## Scope

This artifact is machine-checkable source-audit evidence for the v0.9.0
requirement that accepted epistemic execution reuses existing production paths.
It is not a closure claim. The current branch still has runtime, solver, and
probability blockers before `G090_CLOSE`.

The audit follows the same closure-evidence discipline used by the prior G38,
G38-B, and G39 chains: production path reuse must be proven by concrete source
markers, counters, and explicit blocker status, while board, merge, push, and
tag gates remain separate user-authorized actions.

## Prior Evidence Reuse

| Prior chain | Reused rule | v0.9.0 application |
|---|---|---|
| v0.7.0 WCOJ expansion | General WCOJ evidence must include the first-class `RirNode::MultiWayJoin` and deterministic 4-cycle runtime dispatch path, not only later K-clique planner metadata. | The v0.9.0 audit requires accepted epistemic 4-cycle execution to advance `wcoj_4cycle_dispatch_count`, records `certified_multiway_reductions`, and rejects required non-hash `MultiWayJoin` reductions without WCOJ dispatch deltas. |
| G38 completion audit | Production reuse claims must name the single production launch or dispatch path, as in the cached HG u32 triangle audit. | The v0.9.0 runtime audit names the existing RIR runtime dispatch, `MultiwayPlan::WcojWithPlan`, and `CudaBuffer` relation columns instead of accepting a parallel epistemic WCOJ path. |
| G38-B closure proposal and integration audit | Hypergraph planner and K-clique work is production evidence only when source/routing audits prove the production path is used, including runtime-histogram, skew scheduling, and helper-split reuse after Authorization 5. | The v0.9.0 audit requires WCOJ/K-clique/helper-split metadata from the existing runtime plan, requires production helper relation rules and WCOJ helper input scans before helper-split specs can certify reuse, ties accepted K5/K6 dispatch to certified stream-group/skew-scheduled-helper/histogram counters, and keeps broader helper/skew runtime coverage as a blocker until launch counters prove it. |
| G39 completion audit and K7/K8 evidence | K7/K8 template claims must reuse the shared provider/template path and keep board, tag, and handoff gates user-controlled. | The v0.9.0 audit reuses the G39 K7/K8 planner/preflight surface and production dispatch counters, but this artifact updates certification evidence only; it does not mark `G090_CERT` or `G090_CLOSE` complete and does not imply a board or release action. |

## Accepted-Path Audit

| Area | Reused production path | Audit evidence | Remaining blocker |
|---|---|---|---|
| Runtime/WCOJ lowering | Existing logic compiler and runtime RIR execution | `compile_epistemic_gpu_execution` rejects default FAEEL unsupported self-supported `possible` rules before reduced runtime compilation, allows zero-arity self-`possible` only when the same predicate has independent ordinary founded support, rejects nonzero-arity self-`possible` without tuple-level foundedness proof, preserves explicit G91 compatibility self-support, then lowers accepted reduced ordinary programs through `compile_program_with_stats_snapshot`; `compile_epistemic_gpu_split_execution` lowers valid split components through the same executable path; runtime calls `self.execute_plan(&executable.reduced_runtime_plan)`, records `EpistemicGpuRuntimePreflight`, certifies accepted v0.7.0 4-cycle plus K5, K6, K7, and K8 WCOJ dispatch from production counter deltas, records post-hot-path final-result transfer accounting through `EpistemicGpuFinalResultTransferTrace`, records device-derived GPT counts and candidate indices through `EpistemicGpuSemanticTrace`, decodes device rejection codes through `EpistemicGpuRejectionReason`, and keeps explicit G91 self-supported `possible`, independently founded zero-arity FAEEL self-`possible`, bounded GPT-oracle trace parity, v0.7.0 4-cycle `MultiWayJoin` dispatch, K5/K6 G38-B skew-scheduled helper/histogram, and K7/K8 G39 preflight coverage through production runtime surfaces, including stream-group scheduling metadata. Split execution uses `execute_epistemic_gpu_execution_batch` and the aggregate `execute_epistemic_gpu_execution_batch_with_trace` wrapper, which delegate each component executable to the existing single-plan GPU runtime path instead of introducing a split-only engine. `EpistemicGpuBatchExecutionTrace` records component count, GPU runtime component executions, zero CPU recomposition steps, zero CPU candidate/world-view fallbacks, zero per-candidate host round trips, and aggregate `know`/`possible`/`not know`/`not possible` operator counts across the accepted split fixtures, including the all-binary-operator split batch, the split quaternary `know fact4/4` plus `not possible fact4/4` batch, and the split quaternary `possible fact4/4` plus `not know fact4/4` batch, while the possible-vs-not-known fixture executes both split components over the same absent tuple source. | Broader accepted WCOJ runtime dispatch and semantic parity coverage are still missing. |
| WCOJ layout and scheduling | Existing WCOJ, K-clique, planned-hash, stream-group, sorted-layout, helper-split, skew-scheduling, and runtime-histogram route metadata | `summarize_runtime_routes` inspects general `MultiWayJoin` routes plus `MultiwayPlan::WcojWithPlan`, counts `multiway_reduction_count`, `planned_hash_route_count`, `kclique_wcoj_max_arity`, `kclique_wcoj_edge_permutation_count`, distinct `kclique_stream_group_count`, `kclique_skew_scheduled_plan_count`, `helper_split_spec_count`, sorted-layout requirements, and runtime WCOJ dispatch deltas; `EpistemicGpuRuntimePreflight` also counts compiler-created `__w37_helper_*` helper relation rules and WCOJ helper input scans, and fails closed with `epistemic GPU helper-split certification` when helper-split specs exist without matching production helper rewrites or when helper scans occur outside the WCOJ plan. `EpistemicGpuRuntimeWcojCertification::Certified` carries `certified_multiway_reductions`, `certified_edge_permutation_slots`, `certified_stream_groups`, `certified_skew_scheduled_plans`, `certified_sorted_layout_requirements`, `certified_helper_split_specs`, `certified_helper_relation_rules`, `certified_helper_relation_scans`, observed layout sort or fast-path events, observed metadata builds, and observed metadata-build nanoseconds only after required production WCOJ dispatch counters advance. `MissingRequiredWcojDispatch` fails closed when a required non-hash `MultiWayJoin` or K-clique plan lacks matching dispatch evidence; `MissingRequiredWcojLayout` fails closed when sorted-layout obligations exist but no layout sort or layout fast-path event is observed. The accepted v0.7.0 fixture asserts `wcoj_4cycle_dispatch_count >= 1`; the accepted K5 fixture asserts `10` edge slots, `1` stream group, `1` skew-scheduled helper plan, `1` sorted-layout requirement, `1` helper-split spec, `1` helper relation rule, and `1` WCOJ helper input scan inside the certified dispatch trace; accepted K6 asserts `15` edge slots, nonzero stream-group/skew-scheduled/sorted-layout/helper-split/helper-relation counts, `wcoj_clique6_dispatch_count >= 1`, `kclique_metadata_build_count >= 1`, and `kclique_metadata_build_nanos >= 1`; accepted K7 and K8 fixtures supply production `wcoj_clique7_dispatch_count` and `wcoj_clique8_dispatch_count` evidence with edge-permutation and stream-group metadata, and `epistemic_k7_k8_reductions_reuse_g39_kclique_planner_preflight_surface` proves K7/K8 epistemic reductions carry full production edge-permutation and stream-group metadata through preflight. | Broader helper/skew runtime coverage remains missing beyond the accepted v0.7.0 4-cycle and K5/K6 certified helper/histogram traces. |
| Tuple membership | Existing relation layouts and device buffers | Membership staging reads existing `CudaBuffer` relation columns via `source_relation.column(...)`, compares against reduced output columns via `output.column(bound_col_index)`, and carries the certified binding polarity; preflight records explicit `know_operator_count`, `possible_operator_count`, `not_know_operator_count`, and `not_possible_operator_count`; world-view validation reads the GPU candidate-assumption matrix plus model-membership bytes and rejects candidates missing any required epistemic membership before final-row filtering; final row filtering flattens per-membership row-count pointer/key-offset/polarity metadata over the same relation and reduced-output device buffers and records `row_filter_count` plus `negated_row_filter_count`; accepted unary, possible, not possible, binary, ternary specialized-arity, quaternary all-operator generic-arity, all-`know` multi-membership, mixed `know`/`possible` multi-membership, negated `not know`/`not possible` multi-membership, missing-required-membership, `not know`, split possible-vs-not-known, split all-binary-operator, split quaternary `know`/`not possible fact4/4`, and split quaternary `possible`/`not know fact4/4` fixtures filter final rows by tuple key on device. | Broader semantic parity fixtures for nonzero-arity accepted world views are still missing. |
| Solver | Existing `xlog-solve` GPU CNF/CDCL production path | `GpuSolverProductionAdapter` wraps `GpuCdclSolver::new`, validates accepted `EpistemicGpuExecutionResult` evidence through `solve_expect_sat_with_gpu_execution_result`, `solve_expect_unsat_with_gpu_execution_result`, `solve_expect_unsat_with_branch_limit_ws_with_gpu_execution_result`, `solve_assumption_lifecycle_with_gpu_execution_result`, `solve_multi_candidate_assumption_lifecycle_with_gpu_execution_results`, `solve_maxsat_lifecycle_with_gpu_execution_result`, `solve_multi_candidate_maxsat_lifecycle_with_gpu_execution_results`, `solve_assumption_lifecycle_with_gpu_batch_execution_result`, `solve_maxsat_lifecycle_with_gpu_batch_execution_result`, `solve_unsat_and_publish_learned_clause_arena_with_gpu_execution_result`, `solve_unsat_then_reuse_learned_clauses_with_gpu_execution_result`, `solve_multi_candidate_learned_clause_reuse_with_gpu_execution_results`, `solve_learned_clause_reuse_with_gpu_batch_execution_result`, `solve_weighted_maxsat_candidates_with_gpu_execution_result`, `solve_multi_candidate_weighted_maxsat_with_gpu_execution_results`, `solve_weighted_maxsat_candidates_with_gpu_batch_execution_result`, `solve_weighted_maxsat_search_with_gpu_execution_result`, `solve_multi_candidate_weighted_maxsat_search_with_gpu_execution_results`, `solve_weighted_maxsat_search_with_gpu_batch_execution_result`, `solve_weighted_maxsat_encoded_search_with_gpu_execution_result`, `solve_multi_candidate_weighted_maxsat_encoded_search_with_gpu_execution_results`, `solve_weighted_maxsat_encoded_search_with_gpu_batch_execution_result`, `solve_maxsat_schedule_with_gpu_execution_results`, `solve_maxsat_schedule_with_gpu_batch_execution_result`, `solve_portfolio_with_gpu_execution_result`, `solve_multi_candidate_portfolio_with_gpu_execution_results`, and `solve_portfolio_with_gpu_batch_execution_result`; `accepted_ternary_gpu_execution_result_records_solver_nonzero_arity_evidence_trace`, `accepted_quaternary_gpu_execution_result_records_solver_nonzero_arity_evidence_trace`, and `accepted_quaternary_not_possible_solver_nonzero_arity_evidence_trace` prove accepted ternary, quaternary, and negated quaternary tuple-key evidence flows into SAT solving through this adapter while recording `accepted_nonzero_arity_gpu_candidate_evidence_consumed` and `accepted_gpu_candidate_tuple_key_column_reads_consumed`. `require_maxsat_lifecycle_inputs` rejects empty lifecycle steps or empty MaxSAT candidate sets before lifecycle or MaxSAT trace mutation, `require_weighted_maxsat_candidates` rejects empty candidate sets before MaxSAT trace mutation, `require_weighted_maxsat_search_candidates` rejects empty or all-UNSAT MaxSAT search candidate sets before accepted-evidence accounting or solver trace mutation, `require_weighted_maxsat_search_selections` rejects empty or all-UNSAT encoded MaxSAT selections before accepted-evidence accounting or GPU-CNF encode counters advance, `require_weighted_maxsat_encoding_inputs` rejects invalid encoded weighted MaxSAT inputs before accepted-evidence accounting or GPU-CNF encode counters advance, and `require_maxsat_schedule_jobs` rejects invalid scheduled jobs before accepted-batch evidence, scheduler, encode, or solver counters advance. It also validates accepted split-batch solver evidence through `GpuSolverProductionBatchExecutionEvidence` and `EpistemicGpuBatchExecutionResult`, requiring aggregate `EpistemicGpuBatchExecutionTrace` zero CPU recomposition, zero CPU candidate/world-view fallbacks, zero tracked D2H, and zero per-candidate host round trips before delegating each component result to the existing multi-candidate GPU CDCL lifecycle, combined lifecycle-plus-MaxSAT, learned-clause reuse, MaxSAT, MaxSAT search pruning, weighted MaxSAT encoding/search, generalized MaxSAT scheduler, or portfolio paths, including split-batch quaternary `know fact4/4` plus `not possible fact4/4` lifecycle, learned-clause reuse, and MaxSAT fixtures that record two nonzero-arity evidence consumptions, eight tuple-key column reads, and zero CPU search or learned-clause transfers. Those gates call existing SAT/UNSAT APIs, including reusable workspace-backed UNSAT, bounded lifecycle steps and combined lifecycle-plus-MaxSAT reports with `candidate_evidence_records`, `gpu_assumption_pushes`, `gpu_assumption_retractions`, `gpu_lifecycle_workspace_reuses`, `accepted_gpu_batch_candidate_evidence_consumed`, `accepted_gpu_batch_candidate_component_evidence_consumed`, `gpu_lifecycle_unknown_status_steps`, `gpu_lifecycle_timeout_status_steps`, `accepted_g91_gpu_candidate_evidence_consumed`, `accepted_faeel_gpu_candidate_evidence_consumed`, `accepted_know_gpu_candidate_evidence_consumed`, `accepted_possible_gpu_candidate_evidence_consumed`, `accepted_not_possible_gpu_candidate_evidence_consumed`, `accepted_not_know_gpu_candidate_evidence_consumed`, `accepted_nonzero_arity_gpu_candidate_evidence_consumed`, `accepted_gpu_candidate_tuple_key_column_reads_consumed`, `gpu_maxsat_candidate_solves`, and `gpu_maxsat_optima`, learned-clause arena publication plus same-device-CNF import/reuse with `gpu_learned_clause_*` counters, two-record and accepted split-batch learned-clause reuse with `candidate_evidence_records`, distinct-CNF import rejection with `gpu_learned_clause_reuse_rejections`, bounded single-, two-record, and accepted split-batch MaxSAT candidate solves with `candidate_evidence_records`, `gpu_maxsat_candidate_solves`, `gpu_maxsat_optima`, single-result, two-record, and accepted split-batch MaxSAT search pruning with `gpu_maxsat_unsat_candidate_prunes`, single-result, two-record, and accepted split-batch weighted soft-clause selection encoding through `GpuCnf::from_host` with `gpu_maxsat_candidate_encodes` and `gpu_cdcl_candidate_encodes`, heterogeneous and accepted split-batch MaxSAT scheduler jobs with `gpu_maxsat_scheduler_*` counters, bounded single-result, two-record, and accepted split-batch SAT/MaxSAT portfolio jobs with `gpu_portfolio_*` counters, and UNKNOWN/TIMEOUT status propagation counters. `GpuSolverProductionTrace::require_production_metric_eligibility` rejects traces without accepted GPU candidate evidence, status-only UNKNOWN/TIMEOUT traces, traces without an existing GPU CDCL/MaxSAT/scheduler/portfolio production-path counter, or traces with CPU assignment, MaxSAT, or learned-clause transfer counters. | Broader solver release coverage remains missing beyond the combined lifecycle, empty-MaxSAT lifecycle rejection, all-UNSAT MaxSAT search rejection, invalid encoded MaxSAT rejection, invalid encoded scheduler rejection, split-batch combined lifecycle, bounded scheduler, split-batch scheduler, portfolio, split-batch lifecycle, split-batch quaternary lifecycle, split-batch quaternary learned-clause/MaxSAT, split-batch learned-clause reuse, split-batch MaxSAT, split-batch MaxSAT search pruning, split-batch weighted MaxSAT encoding, nonzero-arity SAT evidence tracing, and split-batch portfolio fixtures. |
| Probability | Existing `xlog-prob` GPU exact/provenance/PIR/CNF paths | `EpistemicProbProductionAdapter` calls `compile_source_with_gpu_execution_result`, `compile_source_for_gpu_execution_results`, `compile_source_for_gpu_batch_execution_result`, `compile_program_with_gpu_execution_result`, `compile_program_for_gpu_execution_results`, `compile_program_for_gpu_batch_execution_result`, `compile_and_evaluate_source_with_gpu_execution_result`, `compile_and_evaluate_source_for_gpu_execution_results`, `compile_and_evaluate_source_for_gpu_batch_execution_result`, `compile_and_evaluate_program_with_gpu_execution_result`, `compile_and_evaluate_program_for_gpu_execution_results`, `compile_and_evaluate_program_for_gpu_batch_execution_result`, `compile_and_evaluate_conditioned_source_with_gpu_execution_result`, `compile_and_evaluate_conditioned_source_for_gpu_execution_results`, `compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result`, `compile_and_evaluate_conditioned_program_with_gpu_execution_result`, `compile_and_evaluate_conditioned_program_for_gpu_execution_results`, `compile_and_evaluate_conditioned_program_for_gpu_batch_execution_result`, `compile_and_evaluate_conditioned_source_with_grads_with_gpu_execution_result`, `compile_and_evaluate_conditioned_source_with_grads_for_gpu_execution_results`, `compile_and_evaluate_conditioned_source_with_grads_for_gpu_batch_execution_result`, `compile_and_evaluate_conditioned_program_with_grads_with_gpu_execution_result`, `compile_and_evaluate_conditioned_program_with_grads_for_gpu_execution_results`, `compile_and_evaluate_conditioned_program_with_grads_for_gpu_batch_execution_result`, `encode_source_pir_cnf_with_gpu_execution_result`, `encode_program_pir_cnf_with_gpu_execution_result`, `encode_source_pir_cnf_for_gpu_execution_results`, `encode_program_pir_cnf_for_gpu_execution_results`, `encode_source_pir_cnf_for_gpu_batch_execution_result`, `encode_program_pir_cnf_for_gpu_batch_execution_result`, `evaluate_with_gpu_execution_result`, `evaluate_for_gpu_execution_results`, `evaluate_for_gpu_batch_execution_result`, `evaluate_gpu_with_grads_with_gpu_execution_result`, `evaluate_gpu_with_grads_for_gpu_execution_results`, and `evaluate_gpu_with_grads_for_gpu_batch_execution_result`, validates accepted `EpistemicGpuExecutionResult` evidence through `AcceptedWorldViewEvidence::from_gpu_execution_result` and `EpistemicProbGpuExecutionEvidence`, validates accepted split-batch evidence through `EpistemicProbGpuBatchExecutionEvidence` and the aggregate `EpistemicGpuBatchExecutionTrace`, then calls `ExactDdnnfProgram::compile_source_with_gpu`, `ExactDdnnfProgram::compile_from_program`, `GpuPirGraph::from_host`, `GpuPirRoots::from_host`, `encode_cnf_gpu`, `ExactDdnnfProgram::evaluate`, and `evaluate_gpu_with_grads`, with `gpu_pir_graph_uploads`, `gpu_source_pir_graph_uploads`, `gpu_program_pir_graph_uploads`, `gpu_cnf_encodes`, `gpu_source_cnf_encodes`, `gpu_program_cnf_encodes`, aggregate/source/program knowledge-compilation counters, accepted-assumption/evidence-fact counters, `accepted_gpu_batch_evidence_consumed`, `accepted_gpu_batch_component_evidence_consumed`, `gpu_conditioned_nonzero_arity_evidence_facts`, `gpu_source_conditioned_nonzero_arity_evidence_facts`, `gpu_program_conditioned_nonzero_arity_evidence_facts`, `gpu_conditioned_max_evidence_arity`, `gpu_source_conditioned_max_evidence_arity`, `gpu_program_conditioned_max_evidence_arity`, `gpu_conditioned_negative_evidence_facts`, `gpu_source_conditioned_evidence_facts`, `gpu_program_conditioned_evidence_facts`, `gpu_source_conditioned_negative_evidence_facts`, `gpu_program_conditioned_negative_evidence_facts`, source/program-specific operator-conditioned counters, `gpu_source_conditioned_gradient_evaluations`, `gpu_program_conditioned_gradient_evaluations`, `gpu_conditioned_know_evidence_facts`, `gpu_conditioned_possible_evidence_facts`, `gpu_conditioned_not_known_evidence_facts`, `gpu_conditioned_not_possible_evidence_facts`, accepted G91/default FAEEL mode-specific evidence counters, source/program-specific exact-query counters, source/program-specific conditioned-gradient counters, source/program-specific PIR/CNF counters, source/parsed-program/split-batch quaternary max-arity evidence coverage, and zero CPU-only probability recomputation counters. Source and parsed-program batch adapters validate all accepted world-view boundaries before per-record exact compile and compile/evaluate with different accepted evidence per result; split-batch source/program exact compile, compile-evaluate, conditioned source/program query/gradient, all-binary-operator conditioned source/program query and gradient, split-batch quaternary parsed-program `not possible` query conditioning, PIR/CNF, query, and gradient evaluation adapters validate one accepted batch before reusing existing GPU exact/provenance APIs per accepted evidence record; conditioned source and parsed-program evaluation appends parsed `Evidence` AST entries for zero-arity and concrete nonzero-arity tuple assumptions, preserving ternary arity-three source evidence, quaternary arity-four source evidence, negated quaternary arity-four source evidence, quaternary arity-four parsed-program evidence, negated quaternary arity-four parsed-program evidence, split-batch quaternary arity-four parsed-program evidence, true `know`, true `possible`, false `possible`/`not possible`, and false `know`/`not know` assumptions as exact evidence before calling the existing exact path; positive and negative source batches plus conditioned source/program query and gradient batch adapters reuse the same accepted boundary. `EpistemicProbProductionCapabilities` disallows fixture circuits for production metrics, and `EpistemicProbProductionTrace::require_production_metric_eligibility` rejects traces without accepted world-view evidence, conditioned evidence facts alone, traces without an aggregate or source/program-specific GPU exact/provenance/PIR/CNF/knowledge-compilation path counter, or traces with CPU/fixture recomputation counters. | Broader probabilistic semantic coverage remains incomplete. |

Solver audit note: the all-binary-operator split-batch solver fixtures prove the lifecycle, learned-clause reuse, and MaxSAT paths consume four split components and advance the accepted `know`/`possible`/`not possible`/`not know` solver evidence counters without CPU search or learned-clause transfers. The source audit now also requires `accepted_split_all_binary_operator_batch_gates_solver_search_scheduler_and_portfolio_paths`, proving the same four-component batch reaches MaxSAT search-pruning, weighted MaxSAT encoding/scheduler, and status-aware portfolio paths with all four accepted operator-family counters, four nonzero-arity evidence consumptions, eight tuple-key column reads, four UNSAT prunes, eight encoded candidates, twenty-four scheduled GPU CDCL candidate solves, four SAT jobs, four MaxSAT jobs, and zero CPU search. The split-batch quaternary solver fixtures prove the same lifecycle, learned-clause reuse, and MaxSAT gates consume one `know fact4/4` component and one `not possible fact4/4` component while recording nonzero-arity tuple-key evidence at the batch boundary, including eight tuple-key column reads for learned-clause reuse and MaxSAT. The source audit now also requires `accepted_split_quaternary_not_possible_batch_gates_solver_search_scheduler_and_portfolio_paths`, proving the same accepted batch reaches MaxSAT search-pruning, weighted MaxSAT encoding/scheduler, and status-aware portfolio paths with one accepted `know` counter, one accepted `not possible` counter, two UNSAT prunes, four encoded candidates, twelve scheduled GPU CDCL candidate solves, two SAT jobs, two MaxSAT jobs, eight tuple-key column reads, and zero CPU search. The split-batch solver evidence gate also rejects stale batch evidence when aggregate CUDA-event timing is absent or any component phase is untimed, and the single-result solver evidence gate rejects stale accepted evidence when candidate-generation CUDA-event timing is absent.
The source audit also requires
`accepted_quaternary_possible_and_not_know_results_gate_solver_and_probabilistic_paths`,
which proves single-result quaternary `possible fact4/4` and
`not know fact4/4` accepted GPU results reach the existing solver SAT gate, record one
accepted `possible` counter, one accepted `not know` counter, two nonzero-arity
evidence consumptions, eight tuple-key column reads, and zero CPU search.
The audit also requires
`accepted_split_quaternary_possible_and_not_know_batch_gates_solver_and_probabilistic_paths`,
which proves split-batch quaternary `possible fact4/4` and
`not know fact4/4` accepted GPU components reach the same GPU CDCL lifecycle
adapter with accepted batch/component evidence, two nonzero-arity evidence
consumptions, eight tuple-key column reads, balanced lifecycle pushes and
retractions, workspace reuse, and zero CPU search. The source audit now also
requires
`accepted_split_quaternary_possible_and_not_know_batch_gates_solver_reuse_and_maxsat_paths`,
which proves the same accepted batch reaches the existing learned-clause reuse
and bounded MaxSAT candidate paths with two arena publications/imports/reused
solves, four GPU CDCL candidate solves, two MaxSAT optima, one accepted
`possible` counter, one accepted `not know` counter, and zero CPU search or
learned-clause transfers.
The source audit now also requires
`accepted_split_quaternary_possible_and_not_know_batch_gates_solver_search_scheduler_and_portfolio_paths`,
which proves the same accepted batch reaches the existing MaxSAT search-pruning,
weighted MaxSAT encoding/scheduler, and status-aware portfolio paths with two
UNSAT prunes, four encoded candidates, twelve scheduled GPU CDCL candidate
solves, two SAT jobs, two MaxSAT jobs, one accepted `possible` counter, one
accepted `not know` counter, eight tuple-key column reads, and zero CPU search.
All accepted solver split-batch entrypoints now call the single
`accepted_solver_results_from_gpu_batch_execution_evidence` validator/accounting
helper, and `production_solver_batch_paths_use_single_gpu_batch_gate`
source-audits that no second manual batch guard or accepted-batch trace
accounting block can drift from the central timing and zero-fallback checks.
The solver production metric gate now rejects lifecycle UNKNOWN/TIMEOUT status
counters by themselves; a trace also needs a GPU CDCL, MaxSAT, scheduler, or
portfolio production-path counter before it can be eligible for production
solver metrics.

Probability audit note: the all-binary-operator split-batch probability fixtures now prove conditioned source/program query and gradient evidence plus all-binary split-batch source/program PIR-CNF plus already-compiled exact query/gradient evaluation through the existing GPU exact/provenance/PIR/CNF paths. The split-batch quaternary fixtures also prove one `know fact4/4` component and one `not possible fact4/4` component condition parsed-program exact queries, source/program gradients, source/program PIR-CNF, and already-compiled exact query/gradient evaluation through the same accepted batch evidence gate while recording arity-four source/program evidence counters, negative-evidence counters, and zero CPU recomputation. The split-batch probability evidence gates also reject stale batch evidence when aggregate CUDA-event timing is absent or any component phase is untimed, and the single-result accepted-world-view gate rejects stale evidence when candidate-generation CUDA-event timing is absent.
The same quaternary possible/not-know source-audit marker proves the
probability adapter consumes two accepted single-result GPU evidences through
`compile_and_evaluate_conditioned_source_for_gpu_execution_results`, records
source-conditioned arity-four evidence, one negative fact, `possible` and
`not know` operator-family counters, two GPU exact query evaluations, and zero CPU
probability recomputation.
The split-batch quaternary possible/not-know marker proves the probability
adapter consumes the same two accepted components through
`compile_and_evaluate_conditioned_source_for_gpu_batch_execution_result`,
records accepted batch/component evidence, source-conditioned arity-four
evidence, one negative fact, source-conditioned `possible` and `not know`
counters, two GPU exact query evaluations, and zero CPU probability
recomputation.
The split-batch quaternary possible/not-know deep marker proves the same
accepted batch also routes through conditioned source/program gradients,
source/program PIR-CNF, and already-compiled exact query/gradient adapters while
recording arity-four source/program evidence counters, batch/component
evidence, and zero CPU probability recomputation.
All accepted probability split-batch entrypoints now call the single
`accepted_world_views_from_gpu_batch_execution_evidence` validator, and
`production_prob_batch_paths_use_single_gpu_batch_gate` source-audits that no
second manual batch guard can drift from the central timing and zero-fallback
checks.
The source audit also requires the design-only `KnowledgeCompilerAdapter`
contracts for c2d and miniC2D, keeping alternative compiler adapters explicit
without treating them as accepted production dispatch paths.
The production metric gate now rejects conditioned evidence facts alone; a
trace also needs an aggregate or source/program-specific GPU
exact/provenance/PIR/CNF/knowledge-compilation path counter before it can be
eligible for production probability metrics.

Split guard audit note: `test_epistemic_production_reuse_audit` now requires the fail-closed `split_multi_membership_modal_coupling_rejects_gpu_batching` marker so unsafe multi-membership modal coupling cannot be documented as accepted split GPU batching evidence.

Mixed-membership audit note: the source audit now also requires
`accepted_mixed_memberships_match_gpt_oracle_parity`, proving same-rule
`know`/`possible` tuple membership uses the existing GPU tuple-source and
final-row filtering path before being cited as accepted semantic-parity
evidence.

Negated mixed-membership audit note: the source audit also requires
`accepted_negated_mixed_memberships_match_gpt_oracle_parity`, proving same-rule
`not know`/`not possible` tuple membership uses the same GPU tuple-source and
negated final-row filtering path before being cited as accepted semantic-parity
evidence.

All-operator mixed-membership audit note: the source audit also requires
`accepted_all_operator_mixed_memberships_match_gpt_oracle_parity`, proving a
same-rule `know`/`possible`/`not know`/`not possible` tuple-membership matrix
uses the same GPU tuple-source and mixed-polarity final-row filtering path
before being cited as accepted semantic-parity evidence.
It also requires
`accepted_all_operator_mixed_membership_gates_solver_lifecycle_path` and
`accepted_all_operator_mixed_membership_conditions_probabilistic_evidence`, so
the same all-operator accepted world-view evidence reaches the solver and
probabilistic production adapters before being cited as production-reuse
evidence.
The deeper production-reuse markers
`accepted_all_operator_mixed_membership_gates_solver_reuse_maxsat_and_portfolio_paths`
and
`accepted_all_operator_mixed_membership_gates_solver_search_and_scheduler_paths`
extend that same evidence through solver learned-clause reuse, MaxSAT,
portfolio, MaxSAT search pruning, weighted MaxSAT encoding, and scheduler
adapters.
The probabilistic marker
`accepted_all_operator_mixed_membership_gates_probabilistic_program_gradient_and_pir_paths`
extends that same evidence through parsed-program conditioning, gradient, and
PIR/CNF adapters.
The possible/not-know split-batch solver marker extends the same arity-four
batch evidence through MaxSAT search pruning, weighted MaxSAT
encoding/scheduler, and portfolio adapters without adding a parallel solver
engine.
The not-possible split-batch solver marker now extends the `know`/`not
possible` arity-four batch evidence through those same adapters without adding
a parallel solver engine.
The all-binary split-batch solver marker now extends the four-component
`know`/`possible`/`not possible`/`not know` batch evidence through those same
adapters without adding a parallel solver engine.

## Forbidden Parallel Engines

The accepted paths audited above must not introduce epistemic-only replacements
for existing engines. The source audit checks for the absence of new accepted
path declarations or calls for:

- epistemic-only WCOJ planners or dispatch engines;
- epistemic-only relation or tuple stores;
- epistemic-only solver-search engines;
- epistemic-only probability-inference engines;
- CPU oracle solver or bounded fixture probability calls on accepted paths.

This is the specific `M090_CERT.13` claim: zero new epistemic-only WCOJ,
solver-search, probability-inference, or tuple-store engines are present in the
accepted paths audited by the source test.

## Validation

| Command | Expected result |
|---|---|
| `cargo test -p xlog-runtime --test test_epistemic_production_reuse_audit -- --nocapture` | PASS after this artifact exists and accepted-path source markers are present. |
| `cargo test -p xlog-prob --test epistemic_prob_production_reuse -- --nocapture` | PASS after this artifact exists and probabilistic production source markers are present. |
| `cargo fmt --check` | PASS |
| `git diff --check` | PASS |

## Non-Closure Notes

- This audit strengthens `M090_CERT.13`; it does not close `G090_CERT`.
- `M090_GPU.11`, `M090_SOLVER.9`, `M090_PROB.8`, and broader accepted
  execution traces remain incomplete.
- CPU semantic-oracle fixtures remain scaffolding evidence only and cannot
  satisfy release gates.
- No closure-board edit, merge, push, or tag is implied.
