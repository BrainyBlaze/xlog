# Source Clarity Closure Board

Scope: Git-tracked source and documentation files only. Superpowers docs, plan files, evidence directories, generated artifacts, result/output directories, build output, and other worktrees are excluded.

Scanned files: 1430
Unresolved files: 249
Comment/prose artifact occurrences: 1556
Code/identifier artifact occurrences: 683

Resolved means this board found no remaining opaque task/milestone labels or external consumer names in the eligible portions of that file.

Stable links to excluded plan/evidence artifact paths are not counted as source-code shorthand; their filenames remain unchanged to avoid breaking historical references.

Technical transfer terminology such as `D2H`, `DTOH`, `H2D`, and `D2D` is allowed when it refers to device-to-host, host-to-device, or device-to-device transfer behavior.

Hardware cache terminology such as `L1` and `L2` is allowed when it refers to GPU level-1 or level-2 cache behavior.

## Term Meanings Used For Resolution

| Artifact | Meaning used during cleanup |
|---|---|
| `DTS` / `DTS-DLM` | external consumer |
| `D1` | aggregate-fused WCOJ group-by-root aggregate fusion direction |
| `D4` | Decision-DNNF knowledge compiler / GPU Decision-DNNF compilation |
| `G91` | epistemic compatibility mode |
| `M8` | native bounded exact-induction engine |
| `M8 Phase 1` | native bounded exact-induction engine |
| `M_INT.11` | peak VRAM gate for certification-suite and benchmark memory snapshots |
| `M1` | manual Compute Sanitizer acceptance gate for DirectCudaResource sanitizer visibility |
| `C7` | compiled epistemic execution-plan/EIR JSON dump |
| `v085` | legacy v0.8.5 language milestone/examples |
| `W2.5` | WCOJ cost-model default flip to the cardinality model |
| `W3.1` | generic full-row WCOJ layout sort+dedup accessors |
| `W3.2` | general-arity WCOJ clique kernel family and template-call-only source-audit contract |
| `W6.4` | K=7/K=8 extension of the general-arity WCOJ clique template surface |
| `W4.2` | nested-loop inner join production operator and small single-key dispatch path |
| `W4.3` | sort-merge inner join provider-level operator for pre-sorted single-key inputs |
| `W42-14` | single-key join column byte-length validation before kernel launch |
| `W42-15` | checked Cartesian-product capacity guard before nested-loop join allocation |
| `W43-4` | empty and single-row fast path for sort-merge sortedness detection |
| `W43-14` | sort-merge join operator-only scope after benchmark rejection of executor dispatch |
| `D2` | GPU Free Join level-synchronous factorized join execution |
| `L1` | GPU level-1 cache |
| `S1` | aggregate-fused WCOJ triangle group-by-root count measurement gate |
| `S2` | provider-level Free Join frontier-engine spike and measurement gates |
| `S1b` | aggregate-fused WCOJ triangle aggregate and u64-key count widening gate |
| `S1c` | aggregate-fused WCOJ 4-cycle group-by-root count fusion and u64-key triangle aggregate widening |
| `S1d` | aggregate-fused WCOJ 4-cycle group-by-root sum/min/max variants and recorded-groupby U64 value-column min/max widening |
| `S1e` | aggregate-fused WCOJ K-clique group-by-root count fusion |
| `W5.2` | hub-skewed WCOJ triangle benchmark fixture |
| `W66` | bounded CUDA Graph capture/replay for Stage-4 CSM hot-loop paths |
| `G38` | GPU skew-classifier integration path |
| `W5.4` | widened-frontier replay determinism certification |
| `L2` | GPU level-2 cache |
| `G01` | XGCF circuit forward kernel certification category |
| `G02` | XGCF circuit backward kernel certification category |
| `G02.1` | circuit backward adjoint propagation checks |
| `G02.2` | circuit backward gradient accumulation checks |
| `G02.3` | circuit backward numerical verification checks |
| `G03` | XGCF weight injection pattern certification category |
| `G03.1` | weight buffer operation checks |
| `G03.2` | weight value handling checks |
| `G04` | XGCF transfer efficiency certification category |
| `G04.1` | transfer size analysis checks |
| `G04.2` | transfer-vs-compute ratio checks |
| `G04.3` | hot-path verification checks |
| `G05` | XGCF circuit-cache GPU integration certification category |
| `G05.1` | cache-hit GPU behavior checks |
| `G05.2` | cache-key correctness checks |
| `G06` | XGCF PTX kernel robustness certification category |
| `G06.1` | circuit-size edge-case checks |
| `G06.2` | variable-count limit checks |
| `G06.3` | numerical-stability checks |
| `G07` | GPU CDCL SAT/UNSAT verifier certification category |
| `G08` | device-resident row-count invariant certification category |
| `C01-C25` | core CUDA infrastructure certification categories |
| `G01-G08` | GPU-tier probabilistic, neural, and solver certification categories |
| `C01` | toolchain/PTX/SASS validation category |
| `C02` | launch-configuration validation category |
| `C03` | pointer, indexing, and bounds validation category |
| `C04` | address-space validation category |
| `C05` | global-memory hazard validation category |
| `C06` | shared-memory validation category |
| `C07` | local-memory and stack validation category |
| `C08` | synchronization and ordering validation category |
| `C09` | warp-level execution validation category |
| `C10` | block/grid coordination validation category |
| `C11` | control-flow and predication validation category |
| `C12` | atomic-operation validation category |
| `C13` | floating-point validation category |
| `C14` | integer edge-case validation category |
| `C15` | determinism validation category |
| `C16` | async-pipeline validation category |
| `C17` | caching and coherence validation category |
| `C18` | host-device integration validation category |
| `C19` | multi-stream concurrency validation category |
| `C20` | multi-GPU validation category |
| `C21` | hardware-reliability validation category |
| `C22` | algorithm-specific validation category |
| `C23` | testing blind-spot validation category |
| `C24` | edge-case matrix validation category |
| `C25` | float-filter validation category |
| task/milestone labels such as `W2.5`, `G39`, `M37-A`, `S1e`, `FRS-042`, `P0.2`, `D3` | replace with the concrete feature, gate, bug, or milestone meaning recovered from plans, boards, history, or code |

## File Board

| File path | Artifacts found in comments/prose count | Artifacts found in code/naming count | Resolved | Representative comment/prose artifacts | Representative code/naming artifacts |
|---|---:|---:|---|---|---|
| `.cargo/config.toml` | 0 | 0 | true |  |  |
| `.github/FUNDING.yml` | 0 | 0 | true |  |  |
| `.github/ISSUE_TEMPLATE/bug_report.md` | 0 | 0 | true |  |  |
| `.github/ISSUE_TEMPLATE/bug_report.yml` | 0 | 0 | true |  |  |
| `.github/ISSUE_TEMPLATE/config.yml` | 0 | 0 | true |  |  |
| `.github/ISSUE_TEMPLATE/feature_request.md` | 0 | 0 | true |  |  |
| `.github/ISSUE_TEMPLATE/feature_request.yml` | 0 | 0 | true |  |  |
| `.github/PULL_REQUEST_TEMPLATE.md` | 0 | 0 | true |  |  |
| `.github/actionlint.yaml` | 0 | 0 | true |  |  |
| `.github/dependabot.yml` | 0 | 0 | true |  |  |
| `.github/workflows/bench.yml` | 0 | 0 | true |  |  |
| `.github/workflows/ci.yml` | 0 | 0 | true |  |  |
| `.github/workflows/cuda-ci.yml` | 0 | 0 | true |  |  |
| `.github/workflows/fuzz.yml` | 0 | 0 | true |  |  |
| `.github/workflows/github-release.yml` | 0 | 0 | true |  |  |
| `.github/workflows/python-publish.yml` | 0 | 0 | true |  |  |
| `.github/workflows/release-plz.yml` | 0 | 0 | true |  |  |
| `AGENTS.md` | 0 | 0 | true |  |  |
| `CHANGELOG.md` | 0 | 0 | true |  |  |
| `CLAUDE.md` | 0 | 0 | true |  |  |
| `CODE_OF_CONDUCT.md` | 0 | 0 | true |  |  |
| `CONTRIBUTING.md` | 0 | 0 | true |  |  |
| `Cargo.toml` | 0 | 0 | true |  |  |
| `JCODEMUNCH.md` | 0 | 0 | true |  |  |
| `LEAN-CTX.md` | 0 | 0 | true |  |  |
| `Makefile` | 0 | 0 | true |  |  |
| `README.md` | 0 | 0 | true |  |  |
| `ROADMAP.md` | 0 | 0 | true |  |  |
| `SECURITY.md` | 0 | 0 | true |  |  |
| `SUPPORT.md` | 0 | 0 | true |  |  |
| `crates/pyxlog/Cargo.toml` | 0 | 0 | true |  |  |
| `crates/pyxlog/README.md` | 0 | 0 | true |  |  |
| `crates/pyxlog/pyproject.toml` | 0 | 0 | true |  |  |
| `crates/pyxlog/python/pyxlog/__init__.py` | 0 | 0 | true |  |  |
| `crates/pyxlog/python/pyxlog/_kernel_paths.py` | 0 | 0 | true |  |  |
| `crates/pyxlog/python/pyxlog/ilp/__init__.py` | 0 | 0 | true |  |  |
| `crates/pyxlog/python/pyxlog/ilp/backend.py` | 0 | 0 | true |  |  |
| `crates/pyxlog/python/pyxlog/ilp/entropy.py` | 0 | 0 | true |  |  |
| `crates/pyxlog/python/pyxlog/ilp/exact_induce.py` | 0 | 0 | true |  |  |
| `crates/pyxlog/python/pyxlog/ilp/exact_small.py` | 0 | 0 | true |  |  |
| `crates/pyxlog/python/pyxlog/ilp/exceptions.py` | 0 | 0 | true |  |  |
| `crates/pyxlog/python/pyxlog/ilp/holdout.py` | 0 | 0 | true |  |  |
| `crates/pyxlog/python/pyxlog/ilp/inventory.py` | 0 | 0 | true |  |  |
| `crates/pyxlog/python/pyxlog/ilp/neurosymbolic.py` | 0 | 0 | true |  |  |
| `crates/pyxlog/python/pyxlog/ilp/promoter.py` | 0 | 0 | true |  |  |
| `crates/pyxlog/python/pyxlog/ilp/temperature.py` | 0 | 0 | true |  |  |
| `crates/pyxlog/python/pyxlog/ilp/trainer.py` | 0 | 0 | true |  |  |
| `crates/pyxlog/python/pyxlog/ilp/types.py` | 0 | 0 | true |  |  |
| `crates/pyxlog/python/pyxlog/runtime_audit.py` | 0 | 0 | true |  |  |
| `crates/pyxlog/python/pyxlog/transfer_diagnostics.py` | 0 | 0 | true |  |  |
| `crates/pyxlog/src/dlpack.rs` | 0 | 0 | true |  |  |
| `crates/pyxlog/src/ilp.rs` | 0 | 0 | true |  |  |
| `crates/pyxlog/src/ilp_exact.rs` | 0 | 0 | true |  |  |
| `crates/pyxlog/src/ilp_gpu.rs` | 0 | 0 | true |  |  |
| `crates/pyxlog/src/lib.rs` | 0 | 0 | true |  |  |
| `crates/pyxlog/src/logic.rs` | 0 | 0 | true |  |  |
| `crates/pyxlog/src/neural.rs` | 0 | 0 | true |  |  |
| `crates/pyxlog/src/neural_registry.rs` | 0 | 0 | true |  |  |
| `crates/pyxlog/src/program.rs` | 0 | 0 | true |  |  |
| `crates/pyxlog/src/training.rs` | 0 | 0 | true |  |  |
| `crates/pyxlog/src/types.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cli/Cargo.toml` | 0 | 0 | true |  |  |
| `crates/xlog-cli/src/bin/xlog-epistemic-evidence.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cli/src/main.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cli/tests/explain_cli_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cli/tests/generated_rule_diagnostics.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cli/tests/interactive_cli_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cli/tests/prob_cli_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cli/tests/run_cli_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-core/Cargo.toml` | 0 | 0 | true |  |  |
| `crates/xlog-core/src/config.rs` | 0 | 0 | true |  |  |
| `crates/xlog-core/src/error.rs` | 0 | 0 | true |  |  |
| `crates/xlog-core/src/lib.rs` | 0 | 0 | true |  |  |
| `crates/xlog-core/src/symbol.rs` | 0 | 0 | true |  |  |
| `crates/xlog-core/src/traits.rs` | 0 | 0 | true |  |  |
| `crates/xlog-core/src/types.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/Cargo.toml` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c01_toolchain.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c02_launch_config.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c03_pointer_bounds.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c04_address_space.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c05_global_memory.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c06_shared_memory.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c07_local_memory.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c08_synchronization.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c09_warp_level.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c10_block_grid.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c11_control_flow.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c12_atomics.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c13_floating_point.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c14_integer.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c15_determinism.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c16_async_pipeline.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c17_caching.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c18_host_device.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c19_multi_stream.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c20_multi_gpu.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c21_hardware.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c22_algorithms.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c23_blind_spots.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c24_edge_matrix.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/c25_float_filter.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/g01_circuit_forward.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/g02_circuit_backward.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/g03_weight_injection.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/g04_transfer_efficiency.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/g05_circuit_cache.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/g06_ptx_robustness.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/g07_sat_cdcl.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/g08_device_counts.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/categories/mod.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/harness/diagnostics.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/harness/generators.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/harness/mod.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/harness/provider.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/harness/validators.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/harness/xgcf.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/lib.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/src/properties.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/tests/category_isolation.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/tests/certification_suite.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/tests/certification_vram_snapshot_gate.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/tests/count_mask_device.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/tests/ilp_coo_fill_from_mask.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/tests/ilp_csr_histogram.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/tests/ilp_reduce_sum.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/tests/properties.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/tests/quick_smoke.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/tests/require_cuda_guard.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/tests/test_free_join_spike.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/tests/test_wcoj_4cycle_groupby_root_agg.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/tests/test_wcoj_4cycle_groupby_root_count.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/tests/test_wcoj_clique_groupby_root_count.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/tests/test_wcoj_groupby_root_agg.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/tests/test_wcoj_groupby_root_count.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda-tests/tests/test_wcoj_metadata.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/Cargo.toml` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/build.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/examples/comprehensive_test.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/CMakeLists.txt` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/arith.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/cache.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/circuit.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/cnf.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/d4.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/dedup.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/epistemic.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/filter.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/groupby.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/ilp.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/ilp_credit.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/ilp_exact.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/join.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/mc_eval.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/mc_resident.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/mc_sample.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/neural.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/pack.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/pir.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/sat.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/scan.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/set_ops.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/sort.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/totalorder.cuh` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/wcoj.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/kernels/weights.cu` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/arrow_device.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/bin/nvrtc_ptx.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/cuda_compat.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/cuda_graph.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/device.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/device_pool.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/device_runtime/async_resource.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/device_runtime/budget.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/device_runtime/direct.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/device_runtime/logging.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/device_runtime/mod.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/device_runtime/resource.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/device_runtime/runtime.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/device_runtime/stream_pool.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/dlpack.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/kernel_manifest_data.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/launch.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/lib.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/memory.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/multi_gpu_memory.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/provider/arithmetic.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/provider/filter.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/provider/fj.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/provider/groupby.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/provider/ilp.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/provider/ilp_exact.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/provider/io.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/provider/kernel_loading.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/provider/kernel_paths.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/provider/launch_safe.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/provider/mod.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/provider/probabilistic.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/provider/relational.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/provider/transfer.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/provider/wcoj.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/provider/wcoj_metadata.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/provider/wcoj_project.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/type_seam.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/wcoj_metadata.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/src/wcoj_phase_timing.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/arrow_device_ffi.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/arrow_device_import.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/arrow_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/build_script_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/common/mod.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/compact_device_count.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/decision_dnnf_provider_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/device_row_counts.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/device_runtime_singleton.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/dlpack_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/filter_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/groupby_gpu.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/groupby_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/ilp_kernel_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/join_collision_test.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/join_v2_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/large_prefix_sum_test.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/multi_gpu_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/pack_keys_gpu.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/pir_provider_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/ptx_validation.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/runtime_cold_race.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/scan_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/set_ops_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/sort_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_async_reap_race.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_budget_via_runtime.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_compute_ranks_tail_block_oob.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_csm_env_dispatch.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_d2h_counter.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_deterministic_d2h_gate.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_full_row_set_algebra.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_logging_via_runtime.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_membership_mask.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_mt_sort_hj_alloc_ordering.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_provider_launch_recorder.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_provider_runtime_routing.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_provider_with_runtime.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_runtime_cross_stream_use_after_free.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_runtime_direct_allocator_parallel_stress.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_runtime_stream_ordered_alloc_facade.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_runtime_stream_ordered_alloc_lifetime.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_setup_provider_with_runtime.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_sort_dedup_u64.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_w33_hg_metadata_storage.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_w66_cuda_graph_smoke.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_wcoj_4cycle_layout.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_wcoj_4cycle_u32.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_wcoj_4cycle_u64.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_wcoj_clique5.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_wcoj_clique6.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_wcoj_layout_fast_path.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_wcoj_layout_sort_roundtrip.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_wcoj_layout_sort_u32.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_wcoj_layout_sort_u64.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_wcoj_layout_u32.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_wcoj_layout_u64.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_wcoj_plan_provider_cert.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_wcoj_project.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_wcoj_triangle_u32.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/test_wcoj_triangle_u64.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/tracked_slice_into_bytes.rs` | 0 | 0 | true |  |  |
| `crates/xlog-cuda/tests/type_coverage_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-gpu/Cargo.toml` | 0 | 0 | true |  |  |
| `crates/xlog-gpu/benches/logic_bench.rs` | 0 | 0 | true |  |  |
| `crates/xlog-gpu/src/biokg.rs` | 0 | 0 | true |  |  |
| `crates/xlog-gpu/src/lib.rs` | 0 | 0 | true |  |  |
| `crates/xlog-gpu/src/logic.rs` | 0 | 0 | true |  |  |
| `crates/xlog-gpu/tests/biokg_streaming_relation_loader.rs` | 0 | 0 | true |  |  |
| `crates/xlog-gpu/tests/logic_runner.rs` | 0 | 0 | true |  | G91 retained as documented epistemic compatibility mode. |
| `crates/xlog-gpu/tests/relation_delta_planner_telemetry.rs` | 0 | 0 | true |  |  |
| `crates/xlog-gpu/tests/v032_gpu_certification.rs` | 0 | 0 | true |  |  |
| `crates/xlog-gpu/tests/v085_lists.rs` | 0 | 0 | true |  |  |
| `crates/xlog-gpu/tests/v085_meta.rs` | 0 | 0 | true |  |  |
| `crates/xlog-induce/Cargo.toml` | 0 | 0 | true |  |  |
| `crates/xlog-induce/src/index.rs` | 0 | 0 | true |  |  |
| `crates/xlog-induce/src/lib.rs` | 0 | 0 | true |  |  |
| `crates/xlog-induce/src/provenance.rs` | 0 | 0 | true |  |  |
| `crates/xlog-induce/src/reduce.rs` | 0 | 0 | true |  |  |
| `crates/xlog-induce/src/score.rs` | 0 | 0 | true |  |  |
| `crates/xlog-induce/src/types.rs` | 0 | 0 | true |  |  |
| `crates/xlog-induce/src/validate.rs` | 0 | 0 | true |  |  |
| `crates/xlog-induce/tests/rule_induction_provenance.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/Cargo.toml` | 0 | 0 | true |  |  |
| `crates/xlog-integration/benches/fixtures/external_consumer_analog.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/benches/fixtures/paper_class.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/benches/nested_loop_production_bench.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/benches/sort_merge_production_bench.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/benches/skewed_multiway_bench.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/benches/wcoj_4cycle_bench.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/benches/wcoj_paper_class.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/benches/wcoj_triangle_bench.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/benches/wcoj_histogram_guided_superhub.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/benches/wcoj_w37_helper_split.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/benches/wcoj_w38_stream_mux.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/src/bin/wcoj_phase_report.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/src/bin/xlog_run.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/src/lib.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/src/wcoj_dispatch.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/e2e_integration_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/executor_config_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/ilp_integration_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/real_world_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/require_cuda_guard.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/test_a3_a4_stress.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/test_cross_mode_determinism.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/test_epistemic_gpu_wcoj_execution.rs` | 0 | 0 | true |  | G91 retained as documented epistemic compatibility mode. |
| `crates/xlog-integration/tests/test_free_join_e2e.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/test_external_consumer_surface_preservation.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/test_multiway_walker_contract.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/test_selectivity_pass_reordering.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/test_variable_ordering.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/test_heat_selectivity.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/test_external_consumer_analog_fixture.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/test_nested_loop_dispatch.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/test_sort_merge_dispatch.rs` | 0 | 0 | true | - | - |
| `crates/xlog-integration/tests/test_deep_recursive_wcoj.rs` | 0 | 0 | true | - | - |
| `crates/xlog-integration/tests/test_same_generation_gpu.rs` | 0 | 0 | true | - | - |
| `crates/xlog-integration/tests/test_skewed_multiway_gpu.rs` | 0 | 0 | true | - | - |
| `crates/xlog-integration/tests/test_kclique_measurement_pilot.rs` | 0 | 0 | true | - | - |
| `crates/xlog-integration/tests/test_chain_promoter_dispatch.rs` | 0 | 0 | true | - | - |
| `crates/xlog-integration/tests/test_sort_labels.rs` | 0 | 0 | true | - | - |
| `crates/xlog-integration/tests/test_recursive_setop_profile.rs` | 0 | 0 | true | - | - |
| `crates/xlog-integration/tests/test_wcoj_4cycle_adaptive_dispatch.rs` | 0 | 0 | true | - | - |
| `crates/xlog-integration/tests/test_wcoj_4cycle_dispatch_stream_reuse.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/test_wcoj_4cycle_executor_wiring.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/test_wcoj_4cycle_rir_shape_cert.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/test_wcoj_adaptive_default_on.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/test_wcoj_adaptive_dispatch.rs` | 0 | 0 | true | - | - |
| `crates/xlog-integration/tests/test_wcoj_cardinality_cost_model.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/test_wcoj_clique_dispatch.rs` | 0 | 0 | true | - | - |
| `crates/xlog-integration/tests/test_wcoj_clique_groupby_fusion.rs` | 0 | 0 | true | - | - |
| `crates/xlog-integration/tests/test_wcoj_dispatch.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/test_wcoj_dispatch_stream_reuse.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/test_wcoj_executor_wiring.rs` | 0 | 0 | true | - | - |
| `crates/xlog-integration/tests/test_wcoj_groupby_fusion.rs` | 0 | 0 | true | - | - |
| `crates/xlog-integration/tests/test_wcoj_groupby_fusion_recursive.rs` | 0 | 0 | true | - | - |
| `crates/xlog-integration/tests/test_wcoj_record_join_result_feedback.rs` | 0 | 0 | true | - | - |
| `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs` | 0 | 0 | true | - | - |
| `crates/xlog-integration/tests/test_wcoj_rir_shape_cert.rs` | 0 | 0 | true |  |  |
| `crates/xlog-integration/tests/test_widened_frontier_replay.rs` | 0 | 0 | true | - | - |
| `crates/xlog-ir/Cargo.toml` | 0 | 0 | true |  |  |
| `crates/xlog-ir/src/eir.rs` | 0 | 0 | true | - | G91 is a public Gelfond-1991-style epistemic semantics name, not a task code. |
| `crates/xlog-ir/src/epistemic_plan.rs` | 0 | 0 | true |  |  |
| `crates/xlog-ir/src/lib.rs` | 0 | 0 | true |  |  |
| `crates/xlog-ir/src/metadata.rs` | 0 | 0 | true |  |  |
| `crates/xlog-ir/src/plan.rs` | 0 | 0 | true | - | - |
| `crates/xlog-ir/src/rir.rs` | 0 | 0 | true | - | - |
| `crates/xlog-ir/tests/test_kclique_variable_order.rs` | 0 | 0 | true |  |  |
| `crates/xlog-ir/tests/test_multiway_plan.rs` | 0 | 0 | true |  |  |
| `crates/xlog-ir/tests/test_multiway_rir.rs` | 0 | 0 | true | - | - |
| `crates/xlog-logic/Cargo.toml` | 0 | 0 | true |  |  |
| `crates/xlog-logic/examples/optimizer_demo.rs` | 0 | 0 | true | - | - |
| `crates/xlog-logic/src/ast.rs` | 0 | 0 | true | - | G91 is a public Gelfond-1991-style epistemic semantics name, not a task code. |
| `crates/xlog-logic/src/compile.rs` | 0 | 0 | true | - | - |
| `crates/xlog-logic/src/compiler_config.rs` | 0 | 0 | true | - | - |
| `crates/xlog-logic/src/diagnostics.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/eir.rs` | 0 | 0 | true | - | G91 is a public Gelfond-1991-style epistemic semantics name, not a task code. |
| `crates/xlog-logic/src/epistemic.rs` | 0 | 0 | true | - | G91 is a public Gelfond-1991-style epistemic semantics name, not a task code. |
| `crates/xlog-logic/src/expand.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/function.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/hypergraph/eligibility.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/hypergraph/explain.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/hypergraph/fixpoint.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/hypergraph/inference.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/hypergraph/ir.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/hypergraph/mod.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/hypergraph/plan.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/hypergraph/reference.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/hypergraph/scc.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/hypergraph/typed.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/hypergraph/var_order.rs` | 0 | 0 | true | - | Replaced benchmark-task and PR markers with descriptive planner history. |
| `crates/xlog-logic/src/incremental_parse.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/lib.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/list_normalize.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/lower.rs` | 0 | 0 | true | - | Replaced generic multiway and language-completeness goal labels with behavior wording. |
| `crates/xlog-logic/src/magic_sets.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/meta_normalize.rs` | 0 | 0 | true | - | Replaced language-completeness goal diagnostics with finite meta subset wording. |
| `crates/xlog-logic/src/module.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/module_diagnostics.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/optimizer.rs` | 0 | 0 | true | - | Replaced optimizer task labels with selectivity, helper-splitting, and shape-agnostic fallback wording. |
| `crates/xlog-logic/src/optimizer/stream_schedule_pass.rs` | 0 | 0 | true | - | Replaced stream-scheduling task labels with phase and occupancy wording. |
| `crates/xlog-logic/src/parser.rs` | 0 | 0 | true | - | S5 is modal-logic terminology and G91 is the public Gelfond-1991 compatibility mode, not task code. |
| `crates/xlog-logic/src/promote.rs` | 0 | 0 | true | - | Replaced promoter task, slice, paper-label, authorization, and lock references with recursive, aggregate, K-clique, and generic Free Join behavior wording. |
| `crates/xlog-logic/src/proof_trace.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/resolver.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/stratify.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/typeinfer.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/src/wcoj_var_ordering.rs` | 0 | 0 | true | - | Replaced variable-ordering task labels with leader-cardinality, heat-aware, join-result feedback, default dispatch, and canonical permutation wording. |
| `crates/xlog-logic/tests/differentiable_proof_trace.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/function_integration_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/function_parse_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/integration/full_v032.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/integration/main.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/integration/module_udf.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/integration/symbol_module.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/integration/symbol_udf.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/integration_tests.rs` | 0 | 0 | true | - | Replaced learnable-rule test task labels with explicit parser, IR relation coverage, optimizer, and head-validation regression wording. |
| `crates/xlog-logic/tests/logic/aggregates.xlog` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/logic/learnable.xlog` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/logic/stratified.xlog` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/logic/tc.xlog` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/module_boundary_diagnostics.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/module_integration_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/module_parse_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/modules/basic/helper.xlog` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/modules/basic/main.xlog` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/modules/circular/a.xlog` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/modules/circular/b.xlog` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/modules/nested/lib/utils.xlog` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/modules/nested/main.xlog` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/modules/transitive/base.xlog` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/modules/transitive/main.xlog` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/modules/transitive/mid.xlog` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/modules/visibility/internal.xlog` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/modules/visibility/main.xlog` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/optimizer_integration.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/optimizer_real_world.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/parse_neural.rs` | 0 | 0 | true | - | Replaced terse neural-addition digit variables with `LeftDigit` and `RightDigit` in the parser fixture. |
| `crates/xlog-logic/tests/symbol_integration_test.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/test_epistemic_eir.rs` | 0 | 0 | true | - | S5 is modal-logic terminology and G91 is the public Gelfond-1991 compatibility mode; removed the release item label from the modal-chain comment. |
| `crates/xlog-logic/tests/test_epistemic_examples.rs` | 0 | 0 | true | - | G91 is the public Gelfond-1991 compatibility mode used by the epistemic example test. |
| `crates/xlog-logic/tests/test_epistemic_executable_plan.rs` | 0 | 0 | true | - | G91 is the public Gelfond-1991 compatibility mode; removed execution-plan release, item, lock, and bundle labels from comments and the K-clique test name. |
| `crates/xlog-logic/tests/test_epistemic_faeel.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/test_epistemic_faeel_foundedness.rs` | 0 | 0 | true | - | G91 is the public Gelfond-1991 compatibility mode; replaced foundedness release, item, pilot, and section-code labels with behavior wording. |
| `crates/xlog-logic/tests/test_epistemic_g91.rs` | 0 | 0 | true | - | G91 is the public Gelfond-1991 compatibility mode under test; removed release and bundle labels from the FAEEL-vs-G91 comment. |
| `crates/xlog-logic/tests/test_epistemic_gpt.rs` | 0 | 0 | true | - | G91 is the public Gelfond-1991 compatibility mode used by the GPT compatibility-mode test. |
| `crates/xlog-logic/tests/test_epistemic_gpu_plan.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/test_epistemic_split.rs` | 0 | 0 | true | - | G91 is the public Gelfond-1991 compatibility mode and S5 is modal-logic terminology; replaced split/stratified execution release, item, bundle, task, section-code, and pilot labels with behavior wording. |
| `crates/xlog-logic/tests/test_epistemic_world_view.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/test_hg_eligibility_executor_context.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/test_hg_kclique_planner.rs` | 0 | 0 | true | - | Replaced the milestone-derived W5.2/W52 benchmark fixture names and assertion text with skewed multiway benchmark wording. |
| `crates/xlog-logic/tests/test_hypergraph_certification.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/test_hypergraph_fixpoint.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/test_hypergraph_plan.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/test_hypergraph_planner.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/test_hypergraph_pr9_contracts.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/test_hypergraph_reference.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/test_hypergraph_scc.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/test_hypergraph_scc_inference.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/test_hypergraph_typed.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/test_promote_multiway.rs` | 0 | 0 | true | - | Replaced release/slice and D2 labels with compiler-pipeline, dedicated multiway dispatch, and general Free Join wording. |
| `crates/xlog-logic/tests/test_v085_approx_pragmas.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/test_v085_incremental_parse.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/test_v085_lists.rs` | 0 | 0 | true | - | Replaced version-coded list normalizer API usage and example paths with list-builtin normalization and language-completeness example wording. |
| `crates/xlog-logic/tests/test_v085_magic_sets.rs` | 0 | 0 | true | - | Replaced the version-coded magic-set rewrite API usage and example path with behavior-named magic-set rewriting and language-completeness example wording. |
| `crates/xlog-logic/tests/test_v085_meta.rs` | 0 | 0 | true | - | Replaced version-coded meta/list normalizer API usage, diagnostic assertions, and example paths with meta/list normalization and language-completeness wording. |
| `crates/xlog-logic/tests/test_v085_naf.rs` | 0 | 0 | true | - | Replaced the version-coded negation diagnostic assertion and example path with negation-safety and language-completeness wording. |
| `crates/xlog-logic/tests/test_v085_types.rs` | 0 | 0 | true | - | Replaced version-coded type/term test names and diagnostics with language-completeness and extended-term wording. |
| `crates/xlog-logic/tests/test_leader_cardinality_var_order.rs` | 0 | 0 | true | - | Replaced milestone and slice labels with leader-cardinality variable-ordering and original binary-fallback projection wording; renamed the test file accordingly. |
| `crates/xlog-logic/tests/test_heat_aware_var_order.rs` | 0 | 0 | true | - | Replaced milestone and plan-part labels with heat-aware variable-ordering, leader-cardinality baseline, and runtime-snapshot wording; renamed the test file accordingly. |
| `crates/xlog-logic/tests/test_kclique_promoter.rs` | 0 | 0 | true | - | Replaced milestone, phase, authorization, and internal gate labels with K-clique promoter behavior, recursive-scan, filter-rejection, and runtime histogram refresh wording; renamed the test file accordingly. |
| `crates/xlog-logic/tests/test_kclique_cost_gate.rs` | 0 | 0 | true | - | Replaced milestone-derived W5.2/w52 routing cert labels with skewed multiway benchmark wording; expanded dILP in the local test name and renamed the test file accordingly. |
| `crates/xlog-logic/tests/test_w67b_dispatch_plan.rs` | 0 | 0 | true |  |  |
| `crates/xlog-logic/tests/test_kclique_helper_split.rs` | 0 | 0 | true | - | - |
| `crates/xlog-neural/Cargo.toml` | 0 | 0 | true |  |  |
| `crates/xlog-neural/src/batch.rs` | 0 | 0 | true | - | Replaced abbreviated digit variables in the neural batching doc example with descriptive left/right digit names. |
| `crates/xlog-neural/src/bridge.rs` | 0 | 0 | true |  |  |
| `crates/xlog-neural/src/handle.rs` | 0 | 0 | true |  |  |
| `crates/xlog-neural/src/lib.rs` | 0 | 0 | true |  |  |
| `crates/xlog-neural/src/registry.rs` | 0 | 0 | true |  |  |
| `crates/xlog-neural/src/tensor_source.rs` | 0 | 0 | true |  |  |
| `crates/xlog-neural/tests/batch_test.rs` | 0 | 0 | true |  |  |
| `crates/xlog-neural/tests/bridge_test.rs` | 0 | 0 | true |  |  |
| `crates/xlog-neural/tests/registry_test.rs` | 0 | 0 | true |  |  |
| `crates/xlog-neural/tests/tensor_source_test.rs` | 0 | 0 | true |  |  |
| `crates/xlog-neural/tests/test_error_conversion.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/Cargo.toml` | 0 | 0 | true |  |  |
| `crates/xlog-prob/benches/prob_bench.rs` | 0 | 0 | true | - | Replaced the bare D4 compiler note with exact-inference Decision-DNNF knowledge compiler wording. |
| `crates/xlog-prob/src/aggregates.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/cnf.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/compilation/disk_cache.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/compilation/gpu_cache.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/compilation/gpu_cnf.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/compilation/gpu_d4/build.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/compilation/gpu_d4/frontier.rs` | 0 | 0 | true | - | Replaced prose GPU D4 and Phase 1 labels with GPU-native Decision-DNNF frontier-expansion wording; D4 remains only in ABI/kernel identifiers. |
| `crates/xlog-prob/src/compilation/gpu_d4/mod.rs` | 0 | 0 | true | - | Replaced prose GPU D4 and Phase 1 labels with GPU-native Decision-DNNF knowledge-compilation wording; D4 remains only in module/kernel identifiers. |
| `crates/xlog-prob/src/compilation/gpu_pir.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/compilation/gpu_pir_intern.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/compilation/gpu_weights.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/compilation/mod.rs` | 0 | 0 | true | - | Replaced bare D4 stage comments with GPU-native Decision-DNNF compiler wording; D4 remains only in established function/field identifiers. |
| `crates/xlog-prob/src/compilation/sparse_matrix.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/compilation/validation.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/decision_order.rs` | 0 | 0 | true | - | Replaced bare D4 decision-order comments with Decision-DNNF compiler and GPU-native branching-heuristic wording. |
| `crates/xlog-prob/src/epistemic.rs` | 0 | 0 | true | - | Replaced GPU D4 adapter comments with GPU-native Decision-DNNF wording and removed a stale v0.9 fixture label from the module docs. |
| `crates/xlog-prob/src/epistemic_production.rs` | 0 | 0 | true | - | Expanded G91 prose to Gelfond-1991 compatibility-mode wording while preserving public G91 semantics identifiers; replaced the G090 goal label and bare D4 backend text with production-goal and GPU-native Decision-DNNF wording. |
| `crates/xlog-prob/src/exact.rs` | 0 | 0 | true | - | Replaced bare D4 prose and error text with GPU-native Decision-DNNF wording; retained established compile_gpu_d4 implementation identifiers as public compiler-path names. |
| `crates/xlog-prob/src/exact_gpu.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/gpu.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/kc/ddnnf.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/kc/mod.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/lib.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/mc/buffers.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/mc/evidence.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/mc/mod.rs` | 0 | 0 | true | - | Replaced P3/Phase labels with explicit Monte Carlo approximate-engine opt-in and bounded non-monotone SCC semantics wording. |
| `crates/xlog-prob/src/mc/resident.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/mc/results.rs` | 0 | 2 | true | Replaced historical language-completeness gate helper plus stale release/gate diagnostics with host Monte Carlo high-level-term lowering/materialization wording. | G085/v085(1), v0.8.5(1) |
| `crates/xlog-prob/src/neural_fast_path.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/pir.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/provenance.rs` | 4 | 37 | true | Replaced proof-placeholder comments with explicit proof names; replaced legacy language milestone helper names and stale release/gate diagnostics with provenance high-level-term, univ-normalization, count-lift, and exact probabilistic aggregate wording. | P1(2), P2(2) | G085/v085(30), v0.8.5(5), agg_lift(1), prob_aggregate(1) |
| `crates/xlog-prob/src/wfs.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/src/xgcf.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/cdcl_q1_status_simple.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/cdcl_q2_status.rs` | 1 | 5 | true | Replaced the Decision-DNNF compiler shorthand comment, generated digit variable names, and local compile expectation string with descriptive wording. | D4(1) | D1(2), D2(2), d4(1) |
| `crates/xlog-prob/tests/cnf_cross_process.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/cnf_determinism.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/common/mod.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/disk_cache_cross_process.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/epistemic_prob.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/epistemic_prob_gpu_accepted_evidence.rs` | 0 | 0 | true | - | G91 is the public Gelfond-1991 compatibility mode under test, not a task code. |
| `crates/xlog-prob/tests/epistemic_prob_production_reuse.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/exact_ddnnf.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/exact_ddnnf_gpu_grads.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_backward_fused_parity.rs` | 4 | 0 | true | Replaced level shorthand labels with explicit graph level numbers in the circuit layout comment. | L0(1), L1(1), L2(1), L3(1) |
| `crates/xlog-prob/tests/gpu_cache_compile_and_verify.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_cache_store.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_circuit_cache.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_cnf.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_cnf_hash.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_d4_compile_and_verify.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_d4_decision_order_hint.rs` | 0 | 0 | true | Replaced local D4 shorthand with Decision-DNNF wording; retained the existing profile field and historical evidence path. | - |
| `crates/xlog-prob/tests/gpu_d4_validate_cnf.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_d4_var_presence.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_equivalence_padded_phi.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_equivalence_smoke.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_eval_device_only.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_exact_cache_integration.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_mc_device_counts.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_mc_vs_cpu.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_pir_intern.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_pir_layout.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_query_var_mapping.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_query_vars_device.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_weights.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_workspace_verify.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_xgcf.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_xgcf_cached.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/gpu_xgcf_from_device.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/mc.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/mc_engine_contract.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/mc_gpu_native.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/mc_resident.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/neural_fast_path.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/no_cpu_d4_in_exact.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/provenance_tc.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/template_addition_cnf_valid.rs` | 0 | 4 | true | Replaced generated addition digit variables with descriptive left/right digit names. | D1(2), D2(2) |
| `crates/xlog-prob/tests/test_provenance_primitives.rs` | 0 | 0 | true |  |  |
| `crates/xlog-prob/tests/aggregate_lifting.rs` | 0 | 2 | true | Renamed the legacy milestone test file and replaced the stale aggregate-lift diagnostic assertion with current count aggregate lifting wording. | v085(2) |
| `crates/xlog-prob/tests/mc_approx.rs` | 0 | 1 | true | Renamed the legacy milestone test file to describe Monte Carlo approximate-inference coverage. | v085(1) |
| `crates/xlog-prob/tests/prob_aggregates.rs` | 0 | 1 | true | Renamed the legacy milestone test file and replaced stale probabilistic aggregate diagnostic assertions with current exact aggregate wording. | v085(1) |
| `crates/xlog-runtime/Cargo.toml` | 4 | 0 | true | Replaced recursive-stats trace milestone comments and test target metadata with behavior-based wording. | W2.3(4) |
| `crates/xlog-runtime/src/executor/delta.rs` | 0 | 0 | true |  |  |
| `crates/xlog-runtime/src/executor/epistemic_workspace.rs` | 10 | 1 | true | Removed generic Free Join milestone labels, expanded transfer-direction prose, and replaced the constraint-rejection task label with behavior wording. | G91(1) allowed as public Gelfond-1991 compatibility semantics. |
| `crates/xlog-runtime/src/executor/expression.rs` | 0 | 0 | true |  |  |
| `crates/xlog-runtime/src/executor/join_cache.rs` | 0 | 0 | true |  |  |
| `crates/xlog-runtime/src/executor/mod.rs` | 16 | 0 | true | Replaced runtime counter, recursive stats trace, ILP accessor, release-label, and authorization-label comments with behavior-based wording. |  |
| `crates/xlog-runtime/src/executor/node_dispatch.rs` | 12 | 0 | true | Replaced nested-loop dispatch, aggregate-fusion, fallback, and ILP tensor-mask milestone comments with behavior-based wording. |  |
| `crates/xlog-runtime/src/executor/recursive.rs` | 16 | 0 | true | Replaced recursive stats trace, recursive WCOJ, Free Join, paper shorthand, and k-clique milestone comments with behavior-based wording. |  |
| `crates/xlog-runtime/src/executor/rewrite.rs` | 18 | 1 | true | Replaced occurrence-rewrite, input/fallback symmetry, shape-agnosticism, and release/slice milestone comments; renamed milestone-labeled test module. |  |
| `crates/xlog-runtime/src/executor/wcoj_cost_model.rs` | 0 | 2 | true |  | Replaced W2.5-coded unit test names with cardinality-default and env opt-out behavior names. |
| `crates/xlog-runtime/src/executor/wcoj_dispatch.rs` | 98 | 6 | true | Replaced WCOJ dispatch, feedback, Free Join, aggregate-fusion, chain, K-clique, matcher, release/slice, and sibling-task labels with behavior-based wording; renamed milestone-labeled pipeline and matcher-test identifiers. |  |
| `crates/xlog-runtime/src/executor/wcoj_phase_timing.rs` | 0 | 0 | true |  |  |
| `crates/xlog-runtime/src/ilp_registry.rs` | 0 | 0 | true |  |  |
| `crates/xlog-runtime/src/lib.rs` | 0 | 0 | true |  |  |
| `crates/xlog-runtime/src/profiler.rs` | 0 | 0 | true |  |  |
| `crates/xlog-runtime/src/relation.rs` | 0 | 0 | true |  |  |
| `crates/xlog-runtime/src/statistics.rs` | 0 | 0 | true |  |  |
| `crates/xlog-runtime/tests/statistics_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-runtime/tests/test_epistemic_gpu_workspace.rs` | 66 | 70 | true | Replaced release, item, EGB, K1-K4, scope, and lock labels in comments/doc-comments. `KD45`/`S5` remains as public epistemic terminology. | Replaced task-coded test/helper names and assertion strings. Kept `G91` public Gelfond-1991 enum text and `K5` clique fixture terminology. Expanded hot-path transfer assertion to `device-to-host`. |
| `crates/xlog-runtime/tests/test_epistemic_production_reuse_audit.rs` | 0 | 1 | true |  | Expanded assertion text from `D2H` to `device-to-host`; retained runtime transfer API field names. |
| `crates/xlog-runtime/tests/test_leader_input_permutation_tables.rs` | 6 | 7 | true | Replaced W2.1/W2.2 and Part B/Part C plan labels with runtime slot-preparation and permutation-table descriptions. | Renamed milestone-coded file and `part_b_*` tests to leader-input permutation and rotation names. |
| `crates/xlog-runtime/tests/recursive_stats_trace.rs` | 20 | 6 | true | Renamed the legacy milestone test file and replaced recursive-stats, multi-recursive dispatch, semi-naive occurrence, and dispatch-certified fixture prose/variables with behavior-based wording. | W2.3(8), W4.1(7), P1(5) | W2.3(6) |
| `crates/xlog-runtime/tests/test_cost_model_default_resolution.rs` | 0 | 4 | true |  | Renamed W2.5-coded file/tests and expanded the assertion text to the cardinality default behavior. |
| `crates/xlog-solve/Cargo.toml` | 0 | 0 | true |  |  |
| `crates/xlog-solve/benches/solver_bench.rs` | 0 | 0 | true |  |  |
| `crates/xlog-solve/examples/real_world.rs` | 0 | 0 | true |  |  |
| `crates/xlog-solve/src/gpu_cdcl.rs` | 0 | 0 | true |  |  |
| `crates/xlog-solve/src/gpu_cnf.rs` | 0 | 0 | true |  |  |
| `crates/xlog-solve/src/instance.rs` | 0 | 0 | true |  |  |
| `crates/xlog-solve/src/lib.rs` | 0 | 0 | true |  |  |
| `crates/xlog-solve/src/production.rs` | 11 | 8 | true | Expanded G91, D2H, H2D, and release-version comments to Gelfond-1991 compatibility, device-to-host/host-to-device, and production metric wording. | Expanded D2H, H2D, and release-version assertion text to production contract wording. |
| `crates/xlog-solve/src/proof.rs` | 0 | 0 | true |  |  |
| `crates/xlog-solve/src/service.rs` | 0 | 0 | true |  |  |
| `crates/xlog-solve/src/solver.rs` | 0 | 0 | true |  |  |
| `crates/xlog-solve/tests/gpu_cdcl_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-solve/tests/gpu_cdcl_workspace.rs` | 0 | 0 | true |  |  |
| `crates/xlog-solve/tests/gpu_cnf_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-solve/tests/gpu_solver_accepted_evidence.rs` | 0 | 9 | true |  | Expanded G91-coded test/helper names and assertion strings to Gelfond-1991 compatibility wording; kept public pragma, enum, and trace-field API names. |
| `crates/xlog-solve/tests/gpu_solver_production_reuse.rs` | 0 | 0 | true |  |  |
| `crates/xlog-solve/tests/integration_test.rs` | 3 | 0 | true | Expanded P0/P1/P2 pigeonhole comments to pigeon-number descriptions. |  |
| `crates/xlog-solve/tests/no_dtoh_in_gpu_cdcl.rs` | 0 | 0 | true |  |  |
| `crates/xlog-solve/tests/real_world_tests.rs` | 0 | 0 | true |  |  |
| `crates/xlog-solve/tests/solver_service_semantics.rs` | 0 | 0 | true |  |  |
| `crates/xlog-stats/Cargo.toml` | 0 | 0 | true |  |  |
| `crates/xlog-stats/benches/stats_bench.rs` | 0 | 0 | true |  |  |
| `crates/xlog-stats/src/lib.rs` | 0 | 0 | true |  |  |
| `crates/xlog-stats/src/manager.rs` | 0 | 0 | true |  |  |
| `crates/xlog-stats/src/stats.rs` | 0 | 0 | true |  |  |
| `docs/ARCHITECTURE.md` | 25 | 0 | true | Expanded G91, P-tier, E-tier, D4, version, and external-consumer labels into Gelfond 1991, named inference tiers, Decision-DNNF, current-surface, and external-consumer wording. | Replaced code-block D1/D2 example variables and M8/D4-style labels with descriptive names. |
| `docs/BENCHMARKS.md` | 5 | 0 | true | Expanded D4/d4 and versioned release labels to Decision-DNNF and current benchmark-surface wording; removed stale versioned scope link. |  |
| `docs/RELEASE-EXCLUDE.md` | 1 | 0 | true | Explained the literal `docs/v065-closure-board.md` path as the legacy release closure board while preserving the packaging manifest path. |  |
| `docs/architecture/adaptive-indexing.md` | 0 | 0 | true |  |  |
| `docs/architecture/arithmetic-expressions.md` | 0 | 0 | true |  |  |
| `docs/architecture/bounded-exact-induction.md` | 23 | 3 | true | Expanded transfer shorthand, external-consumer milestone labels, versioned evidence labels, and certification-code labels to device-to-host/device-to-device, external-consumer phase, current feature, and numbered-category wording. | Replaced code-block D2H/D2D/H2D transfer labels with descriptive transfer directions while preserving exact API names where needed. |
| `docs/architecture/cli-reference.md` | 4 | 0 | true | Replaced versioned command-contract and diagnostics-note labels with current CLI surface and generated-rule diagnostics wording. |  |
| `docs/architecture/cuda-certification.md` | 51 | 2 | true | Expanded C/G certification category codes and external-consumer milestone wording to core category, GPU-tier category, toolchain category, and external-consumer scorer descriptions. | Replaced code-block C/G category ranges with descriptive category wording while preserving exact source filenames. |
| `docs/architecture/cudf-interop.md` | 0 | 0 | true |  |  |
| `docs/architecture/device-runtime.md` | 0 | 0 | true |  |  |
| `docs/architecture/dilp-showcase-report.md` | 0 | 0 | true |  |  |
| `docs/architecture/dilp-training.md` | 15 | 0 | true | Expanded P0, D2H, DTOH, versioned, and external-consumer diagnostics wording to GPU-resident, device-to-host, and external-consumer descriptions. |  |
| `docs/architecture/epistemic-semantics.md` | 32 | 0 | true | G91/g91(13), release labels(11), goal labels(3), EGB tags(2), H2D/D2H(2), task-coded helper prefix(1) | Clarified Gelfond 1991 compatibility prose, production WCOJ reuse, accepted runtime surface, and transfer direction wording. |
| `docs/architecture/gpu-execution.md` | 0 | 0 | true |  |  |
| `docs/architecture/language-completeness.md` | 33 | 1 | true | release labels(13), G085 feature codes(19), branch/file naming artifact(1) | Renamed file and replaced release/milestone labels with language-completeness feature-area prose. |
| `docs/architecture/living-world-diagnostics.md` | 26 | 1 | true | release labels(3), scoped test labels(4), tracking IDs(17), external-consumer label(1), filename artifact(1) | Renamed file and replaced tracking IDs with descriptive diagnostic surfaces and verification areas. |
| `docs/architecture/lwm-diagnostics-provenance.md` | 0 | 0 | true |  |  |
| `docs/architecture/multi-gpu-join.md` | 4 | 0 | true | release labels(2), P2P(2) | Replaced release-target labels with design/current-roadmap wording and expanded peer-to-peer copy terminology. |
| `docs/architecture/python-bindings.md` | 34 | 4 | true | release labels(16), diagnostics doc path labels(6), external-consumer names(6), DTOH(4), P2b(1), M37-A+B(1) | D1(2), D2(2) |
| `docs/architecture/query-optimizer.md` | 11 | 0 | true | release labels(6), runtime evidence codes(2), version-coded fixture path labels(2), D2H(1) | Replaced version/evidence-node labels with production runtime, exact-induction evidence, substrate handoff, and device-to-host wording. |
| `docs/architecture/recorded-launch-migration.md` | 0 | 0 | true |  |  |
| `docs/architecture/rfc-tensorized-ilp.md` | 99 | 3 | true | draft/release labels(81), validation milestone labels(8), D2H(10), Wave labels(3) | D2H/draft labels in code-block comments(3) |
| `docs/architecture/solver-services.md` | 5 | 0 | true | D4(2), release labels(3) | Replaced Decision-DNNF shorthand and release labels with descriptive solver-service wording. |
| `docs/architecture/external-consumer-diagnostics.md` | 14 | 1 | true | external-consumer names(7), tracking IDs(6), release label(1), filename artifact(1) | Renamed file and replaced consumer-specific issue codes with reusable diagnostic surface descriptions. |
| `docs/architecture/xlog-prob.md` | 32 | 0 | true | Decision-DNNF shorthand(13), release/date labels(9), phase labels(3), transfer shorthand(4), K1(1), P3(1), v085 evidence label(1) | Expanded Decision-DNNF, Monte Carlo, production-gate, phase/status, and transfer wording. |
| `docs/certification/2026-01-12-cuda-certification-results.md` | 89 | 0 | true | certification category codes(84), bug codes(4), implementation-version label(1) | Replaced category, bug, range, performance, and implementation-version codes with descriptive certification wording. |
| `docs/certification/2026-01-14-cuda-certification-results.md` | 85 | 0 | true | certification category codes(85) | Replaced category, range, performance, and longest-category codes with descriptive certification wording. |
| `docs/certification/2026-01-22-neural-symbolic-certification-report.md` | 108 | 1 | true | certification category codes(95), release labels(6), implementation-plan/spec labels(4), document ID(1), certification version label(1), runtime issue label(1) | filename artifact(1) |
| `docs/certification/neural-symbolic-gpu-certification-spec.md` | 65 | 18 | true | neural-symbolic release labels(7), certification category codes(54), external-compiler labels(4) | module/filter category labels(16), external-compiler code label(1), filename artifact(1) |
| `docs/epistemic-solver-semantics-guide.md` | 43 | 0 | true | Gelfond 1991 shorthand(15), release labels(16), transfer shorthand(5), legacy WCOJ/goal labels(3), external-compiler label(1), roadmap/splitting codes(3) |  |
| `docs/language-reference.md` | 77 | 5 | true | release/version labels(56), implementation checkpoint labels(12), Gelfond 1991 shorthand(7), external-compiler label(1), transfer shorthand(1) | example digit variables(4), diagnostic release label(1) |
| `docs/release-process.md` | 0 | 0 | true |  |  |
| `docs/reports/2026-05-31-mc-gpu-resident-engine-bundles-1-3.md` | 0 | 0 | true |  |  |
| `docs/reports/2026-05-31-epistemic-completion-supervisor-report.md` | 45 | 1 | true | release/branch labels(24), executor-bundle codes(11), semantic-case labels(5), Gelfond 1991 shorthand(4), constraint gap code(1) | filename artifact(1) |
| `docs/reports/2026-06-01-full-semantic-completion-supervisor-report.md` | 38 | 1 | true | release/branch labels(9), delivery item codes(13), Gelfond 1991 shorthand(4), modal-literal placeholders(2), modal-logic shorthand(1), fixture matrix labels(7), status labels(2) | filename artifact(1) |
| `docs/reports/2026-06-02-epistemic-final-closure-record.md` | 6 | 1 | true | release/branch labels(2), status labels(2), Gelfond 1991 shorthand(1), interior-negation item code(1) | filename artifact(1) |
| `docs/v065-closure-board.md` | 228 | 0 | true | legacy closure-board task/release/provenance codes(228) | Excluded from rewrite as a historical closure/evidence artifact; task scope excludes plans/evidence artifacts. |
| `docs/wcoj-architecture-guide.md` | 73 | 0 | true | integration phase labels(14), aggregate-fusion slice labels(19), goal/work-package labels(26), recursive-contract labels(5), WCOJ task labels(6), approval/profiler/authorization labels(3) |  |
| `docs/wcoj-user-guide.md` | 10 | 1 | true | goal/work-package labels(5), integration phase labels(3), skew-classifier path labels(2) | flowchart integration phase label(1) |
| `docs/whitepaper/Makefile` | 0 | 0 | true |  |  |
| `docs/whitepaper/README.md` | 0 | 0 | true |  |  |
| `docs/whitepaper/examples/01_hello.xlog` | 0 | 0 | true |  |  |
| `docs/whitepaper/examples/02_types_err.xlog` | 0 | 0 | true |  |  |
| `docs/whitepaper/examples/02_types_ok.xlog` | 0 | 0 | true |  |  |
| `docs/whitepaper/examples/03_arith.xlog` | 0 | 0 | true |  |  |
| `docs/whitepaper/examples/04_udf.xlog` | 0 | 0 | true |  |  |
| `docs/whitepaper/examples/06_agg.xlog` | 0 | 0 | true |  |  |
| `docs/whitepaper/examples/graph_lib.xlog` | 0 | 0 | true |  |  |
| `docs/whitepaper/examples/graph_main.xlog` | 0 | 0 | true |  |  |
| `docs/whitepaper/main.tex` | 0 | 0 | true |  |  |
| `docs/whitepaper/sections/00_abstract.tex` | 0 | 0 | true |  |  |
| `docs/whitepaper/sections/01_introduction.tex` | 4 | 0 | true | external Decision-DNNF compiler shorthand(3), epistemic release-line label(1) |  |
| `docs/whitepaper/sections/02_architecture.tex` | 9 | 0 | true | transfer shorthand(3), external Decision-DNNF compiler shorthand(1), release/runtime labels(5) |  |
| `docs/whitepaper/sections/03_language.tex` | 0 | 0 | true |  |  |
| `docs/whitepaper/sections/04_gpu_datalog.tex` | 3 | 0 | true | transfer shorthand(1), runtime/release labels(2) |  |
| `docs/whitepaper/sections/05_probabilistic.tex` | 9 | 0 | true | external Decision-DNNF compiler shorthand(8), aggregate release label(1) |  |
| `docs/whitepaper/sections/065_epistemic.tex` | 3 | 0 | true | Gelfond 1991 shorthand(1), epistemic release-line labels(2) |  |
| `docs/whitepaper/sections/06_neural_symbolic.tex` | 28 | 0 | true | digit variable shorthand(8), Decision-DNNF compiler shorthand(3), release labels(5), evidence-run labels(8), beta/schema labels(4) |  |
| `docs/whitepaper/sections/07_interop.tex` | 0 | 0 | true |  |  |
| `docs/whitepaper/sections/08_evaluation.tex` | 27 | 0 | true | runtime/release labels(6), Decision-DNNF compiler shorthand(2), certification category codes(8), evidence-run labels(7), WCOJ evidence path goal/work/external-consumer labels(3), beta status label(1) |  |
| `docs/whitepaper/sections/09_related_work.tex` | 5 | 0 | true | user-defined-function shorthand(1), external Decision-DNNF compiler shorthand(1), evidence-run labels(2), stage label(1) |  |
| `docs/whitepaper/sections/10_limitations.tex` | 13 | 0 | true | release-line labels(8), beta status labels(2), shorthand labels(3) |  |
| `docs/whitepaper/sections/A_performance_contract.tex` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/GOALS.md` | 5 | 0 | true | G1(1), G2(1), G3(1), G4(1), G5(1) | Excluded from rewrite as a goal/planning artifact outside the source/docs cleanup scope. |
| `examples/BFO/universal_case_reasoner/GQM.md` | 8 | 0 | true | transfer shorthand(2), priority label(1), release-line labels(5) | Excluded from rewrite as a GQM planning artifact outside the source/docs cleanup scope. |
| `examples/BFO/universal_case_reasoner/README.md` | 38 | 0 | true | branch label(1), priority/gate IDs(14), release-line labels(8), transfer shorthand(2), shorthand labels(13) |  |
| `examples/BFO/universal_case_reasoner/REQUIREMENTS.md` | 60 | 0 | true | priority/gate IDs(32), release-line labels(6), transfer shorthand(2), shorthand labels(20) |  |
| `examples/BFO/universal_case_reasoner/VALIDATION_PLAN.md` | 41 | 0 | true | priority/gate IDs(7), release-line labels(8), transfer shorthand(6), shorthand labels(20) | Excluded from rewrite as a validation-plan artifact outside the source/docs cleanup scope. |
| `examples/BFO/universal_case_reasoner/WORKER_BRIEF.md` | 12 | 0 | true | branch label(1), priority labels(2), release-line labels(6), shorthand labels(3) | Excluded from rewrite as a worker/task brief outside the source/docs cleanup scope. |
| `examples/BFO/universal_case_reasoner/XLOG_FINDINGS.md` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/bfo/kernel.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/domains/domain_inventory.json` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/ablation_ranker.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/dilp_proof_paths.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/epistemic_dilp_candidate_scoring.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/epistemic_dilp_proof_schema_generator.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/epistemic_dilp_proof_schema_promotion.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/epistemic_dilp_proof_schema_selection.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/epistemic_dilp_proof_trace.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/epistemic_generalization_abstention.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/epistemic_generalization_candidate_generator.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/epistemic_generalization_candidate_scoring.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/epistemic_generalization_decision.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/epistemic_generalization_explanation.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/epistemic_generalization_failure_chain_support.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/epistemic_generalization_ranker.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/epistemic_generalization_selector.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/epistemic_generalization_variant_candidate_generator.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/epistemic_showcase_transfer_candidate_scoring.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/modules/abstention_policy.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/modules/candidate_feature_components.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/modules/component_support.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/modules/decision_support.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/modules/dilp_schema_support_policy.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/modules/dilp_trace_support_policy.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/modules/explanation_support.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/modules/failure_chain_closure.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/epistemic/modules/selector_acceptance.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/neural_ranker.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/programs/production_ranker.xlog` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/proposed_fixes.md` | 21 | 0 | true | issue-code labels(12), release/project labels(2), transfer shorthand(2), shorthand labels(5) |  |
| `examples/BFO/universal_case_reasoner/repro/repro_001_nn4_scoring_not_dilp.py` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/repro/repro_002_proof_paths_without_gradients.py` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/repro/repro_003_missing_rule_inventory_audit.py` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/repro/repro_004_scalar_transfer_contract_is_example_side.py` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/repro/repro_005_adapter_boundary_diagnostics_are_example_side.py` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/repro/repro_006_aggregate_transfer_metrics_need_validator.py` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/tests/test_ablation_smoke.py` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/tests/test_bfo_fixture_smoke.py` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/tests/test_neural_smoke.py` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/tests/test_production_transfer.py` | 0 | 29 | true |  | release/bundle labels(16), generalization blocker codes(6), public benchmark shorthand labels(4), root-cause-analysis shorthand labels(2), differentiable ILP shorthand(1) |
| `examples/BFO/universal_case_reasoner/tests/test_runtime_contract_smoke.py` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/tests/test_validation_contract.py` | 0 | 52 | true |  | priority schema labels(12), release-line labels(2), generalization gate labels(15), differentiable inductive logic labels(8), validator requirement codes(14), stale branch label(1) |
| `examples/BFO/universal_case_reasoner/tests/test_xlog_findings_artifacts.py` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/tools/run_ablation_smoke.py` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/tools/run_bfo_fixture_smoke.py` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/tools/run_neural_smoke.py` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/tools/run_production_transfer.py` | 0 | 16 | true |  | release-line labels(10), differentiable inductive logic gate labels(6) |
| `examples/BFO/universal_case_reasoner/tools/run_runtime_contract_smoke.py` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/tools/validate_universal_case_reasoner.py` | 0 | 28 | true |  | priority schema/release-line/transfer shorthand labels(26), stale branch label(1), stale language showcase path(1) |
| `examples/BFO/universal_case_reasoner/validate.sh` | 0 | 0 | true |  |  |
| `examples/BFO/universal_case_reasoner/validation_summary.json` | 0 | 17 | true |  | P0(5), v080(4), v085(4), v086(2), D2H(2); excluded generated validation/evidence artifact |
| `examples/BFO/universal_case_reasoner/xlog_issue_ledger.json` | 0 | 14 | true |  | release-line labels(7), transfer shorthand labels(6), project acronym phrase(1) |
| `examples/README.md` | 20 | 0 | true | release-line labels(19), external-consumer directory label(1); G91 retained as allowed XLOG terminology |  |
| `examples/epistemic/01-eir-boundary.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/02-g91-compatibility.xlog` | 0 | 0 | true | G91 retained as allowed XLOG epistemic-mode terminology and exact pragma value. |  |
| `examples/epistemic/03-faeel-default.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/04-gpt-candidate-filter.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/05-splitting.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/06-eir-candidate-enumeration.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/07-tuple-key-membership.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/08-repeated-variable.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/09-joint-multi-epistemic.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/10-epistemic-constraint.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/11-faeel-foundedness.xlog` | 2 | 0 | true | release/task labels(2) |  |
| `examples/epistemic/12-bound-variable-splitting.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13-nested-modal-chain-collapse.xlog` | 2 | 0 | true | release/task labels(2); G91 and S5 retained as allowed epistemic/modal terminology |  |
| `examples/epistemic/13b-nested-modal-chain-filter.xlog` | 7 | 0 | true | release/task labels(2), comment shorthand(5); G91 retained as allowed epistemic-mode terminology |  |
| `examples/epistemic/13c-nested-modal-chain-negated.xlog` | 9 | 0 | true | release/task labels(2), comment shorthand(7); G91 retained as allowed epistemic-mode terminology |  |
| `examples/epistemic/13d-nested-modal-chain-faeel-unfounded.xlog` | 16 | 0 | true | release/task labels(2), emphasis shorthand and bare example-number references(14); G91 retained as allowed epistemic-mode terminology |  |
| `examples/epistemic/13e-nested-modal-chain-g91-accepted.xlog` | 14 | 0 | true | release/task labels(2), emphasis shorthand and bare example-number references(12); G91 retained as allowed epistemic-mode terminology |  |
| `examples/epistemic/13f-nested-modal-interior-negation-absent.xlog` | 3 | 0 | true | release/task labels(3) |  |
| `examples/epistemic/13f-nested-modal-interior-negation.xlog` | 5 | 0 | true | release/task labels(4), emphasis shorthand(1); S5 retained as allowed modal-logic terminology |  |
| `examples/epistemic/13fw-nested-modal-interior-negation-g91-absent.xlog` | 3 | 0 | true | release/task labels(3); G91 retained as allowed epistemic-mode terminology |  |
| `examples/epistemic/13fw-nested-modal-interior-negation-g91-present.xlog` | 3 | 0 | true | release/task labels(3); G91 retained as allowed epistemic-mode terminology |  |
| `examples/epistemic/13g-nested-modal-negation-matrix-know-know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13h-nested-modal-negation-matrix-know-know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13i-nested-modal-negation-matrix-know-possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13j-nested-modal-negation-matrix-know-possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13k-nested-modal-negation-matrix-possible-know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13l-nested-modal-negation-matrix-possible-know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13m-nested-modal-negation-matrix-possible-possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13n-nested-modal-negation-matrix-possible-possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13o-nested-modal-negation-matrix-know-know-present-nonderived.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13p-nested-modal-negation-matrix-know-know-absent-nonderived.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13q-nested-modal-negation-matrix-know-possible-present-nonderived.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13r-nested-modal-negation-matrix-know-possible-absent-nonderived.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13s-nested-modal-negation-matrix-possible-know-present-nonderived.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13t-nested-modal-negation-matrix-possible-know-absent-nonderived.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13u-nested-modal-negation-matrix-possible-possible-present-nonderived.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13v-nested-modal-negation-matrix-possible-possible-absent-nonderived.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13w-nested-modal-negation-matrix-g91-know-know-absent-nonderived.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13w-nested-modal-negation-matrix-g91-know-know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13w-nested-modal-negation-matrix-g91-know-know-present-nonderived.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13w-nested-modal-negation-matrix-g91-know-know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13w-nested-modal-negation-matrix-g91-know-possible-absent-nonderived.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13w-nested-modal-negation-matrix-g91-know-possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13w-nested-modal-negation-matrix-g91-know-possible-present-nonderived.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13w-nested-modal-negation-matrix-g91-know-possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13w-nested-modal-negation-matrix-g91-possible-know-absent-nonderived.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13w-nested-modal-negation-matrix-g91-possible-know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13w-nested-modal-negation-matrix-g91-possible-know-present-nonderived.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13w-nested-modal-negation-matrix-g91-possible-know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13w-nested-modal-negation-matrix-g91-possible-possible-absent-nonderived.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13w-nested-modal-negation-matrix-g91-possible-possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13w-nested-modal-negation-matrix-g91-possible-possible-present-nonderived.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/13w-nested-modal-negation-matrix-g91-possible-possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/14-mixed-literal-membership.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/15-recursive-epistemic-chain.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/15-recursive-epistemic-closure.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/16-cross-component-coupling.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/17-cross-component-chained-stratified.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/18-cross-component-joint-shared-modal.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/19-access-control-mixed-modal.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/20-supply-chain-recursive-reach.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/21-incident-triage-joint-modal.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/22-recursion-through-modal-fixpoint.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/23-compound-modal-key-membership.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/23b-unbounded-cons-modal-key-rejected.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/24-transitive-determined-modal-stratified.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/25-recursion-over-determined-modal.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/26-negated-modal-over-invariant-recursive.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/27-augmented-multi-head-differing-arity.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/28-determined-multicol-binding-modal.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/29-negated-modal-over-determined-derived.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/30-possible-binding-over-determined.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/31-faeel-unfounded-self-support-empty-extension.xlog` | 24 | 0 | true | release/task labels, emphasis shorthand, and bare example-number references(24); G91 retained as allowed epistemic-mode terminology |  |
| `examples/epistemic/32-g91-self-support-accepted.xlog` | 14 | 0 | true | release/task labels(2), emphasis shorthand and bare example-number references(12); G91 retained as allowed epistemic-mode terminology |  |
| `examples/epistemic/33-negated-modal-through-recursion-wfs.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33a-negated-modal-through-recursion-wfs-faeel-not-know.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33a-negated-modal-through-recursion-wfs-faeel-not-possible.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33a-negated-modal-through-recursion-wfs-g91-not-know.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33a-negated-modal-through-recursion-wfs-g91-not-possible.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33b-negated-modal-through-recursion-wfs-with-edb-negation.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33c-negated-modal-through-recursion-wfs-matrix-faeel-not-know-seed-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33c-negated-modal-through-recursion-wfs-matrix-faeel-not-know-seed-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33c-negated-modal-through-recursion-wfs-matrix-faeel-not-possible-seed-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33c-negated-modal-through-recursion-wfs-matrix-faeel-not-possible-seed-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33c-negated-modal-through-recursion-wfs-matrix-g91-not-know-seed-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33c-negated-modal-through-recursion-wfs-matrix-g91-not-know-seed-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33c-negated-modal-through-recursion-wfs-matrix-g91-not-possible-seed-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33c-negated-modal-through-recursion-wfs-matrix-g91-not-possible-seed-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33d-negated-modal-through-recursion-wfs-edb-negation-matrix-faeel-not-know-seed-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33d-negated-modal-through-recursion-wfs-edb-negation-matrix-faeel-not-know-seed-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33d-negated-modal-through-recursion-wfs-edb-negation-matrix-faeel-not-possible-seed-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33d-negated-modal-through-recursion-wfs-edb-negation-matrix-faeel-not-possible-seed-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33d-negated-modal-through-recursion-wfs-edb-negation-matrix-g91-not-know-seed-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33d-negated-modal-through-recursion-wfs-edb-negation-matrix-g91-not-know-seed-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33d-negated-modal-through-recursion-wfs-edb-negation-matrix-g91-not-possible-seed-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33d-negated-modal-through-recursion-wfs-edb-negation-matrix-g91-not-possible-seed-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33e-negated-modal-through-recursion-wfs-edb-negation-load-bearing-faeel-not-know-allowed.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33e-negated-modal-through-recursion-wfs-edb-negation-load-bearing-faeel-not-know-banned.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33e-negated-modal-through-recursion-wfs-edb-negation-load-bearing-faeel-not-possible-allowed.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33e-negated-modal-through-recursion-wfs-edb-negation-load-bearing-faeel-not-possible-banned.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33e-negated-modal-through-recursion-wfs-edb-negation-load-bearing-g91-not-know-allowed.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33e-negated-modal-through-recursion-wfs-edb-negation-load-bearing-g91-not-know-banned.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33e-negated-modal-through-recursion-wfs-edb-negation-load-bearing-g91-not-possible-allowed.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/33e-negated-modal-through-recursion-wfs-edb-negation-load-bearing-g91-not-possible-banned.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/34-variable-keyed-constraint-prunes.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/35-variable-keyed-constraint-survives.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/36-multi-literal-distinct-var-constraint.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/37-negated-modal-over-recursive-stratified.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/38-diagonal-modal-constraint.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/39-diagonal-modal-constraint-satisfied.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/40-shared-variable-join-constraint.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/41-negated-difference-constraint.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42-same-name-multi-arity-modal-disambiguation.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42a-same-name-multi-arity-literal-a1-know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42a-same-name-multi-arity-literal-a1-know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42a-same-name-multi-arity-literal-a1-not-know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42a-same-name-multi-arity-literal-a1-not-know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42a-same-name-multi-arity-literal-a1-not-possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42a-same-name-multi-arity-literal-a1-not-possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42a-same-name-multi-arity-literal-a1-possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42a-same-name-multi-arity-literal-a1-possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42a-same-name-multi-arity-literal-a2-know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42a-same-name-multi-arity-literal-a2-know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42a-same-name-multi-arity-literal-a2-not-know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42a-same-name-multi-arity-literal-a2-not-know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42a-same-name-multi-arity-literal-a2-not-possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42a-same-name-multi-arity-literal-a2-not-possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42a-same-name-multi-arity-literal-a2-possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42a-same-name-multi-arity-literal-a2-possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-00-know-present--know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-01-know-present--know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-02-know-absent--know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-03-know-absent--know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-04-know-present--possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-05-know-present--possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-06-know-absent--possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-07-know-absent--possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-08-know-present--not-know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-09-know-present--not-know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-10-know-absent--not-know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-11-know-absent--not-know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-12-know-present--not-possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-13-know-present--not-possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-14-know-absent--not-possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-15-know-absent--not-possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-16-possible-present--know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-17-possible-present--know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-18-possible-absent--know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-19-possible-absent--know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-20-possible-present--possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-21-possible-present--possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-22-possible-absent--possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-23-possible-absent--possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-24-possible-present--not-know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-25-possible-present--not-know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-26-possible-absent--not-know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-27-possible-absent--not-know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-28-possible-present--not-possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-29-possible-present--not-possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-30-possible-absent--not-possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-31-possible-absent--not-possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-32-not-know-present--know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-33-not-know-present--know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-34-not-know-absent--know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-35-not-know-absent--know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-36-not-know-present--possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-37-not-know-present--possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-38-not-know-absent--possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-39-not-know-absent--possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-40-not-know-present--not-know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-41-not-know-present--not-know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-42-not-know-absent--not-know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-43-not-know-absent--not-know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-44-not-know-present--not-possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-45-not-know-present--not-possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-46-not-know-absent--not-possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-47-not-know-absent--not-possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-48-not-possible-present--know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-49-not-possible-present--know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-50-not-possible-absent--know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-51-not-possible-absent--know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-52-not-possible-present--possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-53-not-possible-present--possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-54-not-possible-absent--possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-55-not-possible-absent--possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-56-not-possible-present--not-know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-57-not-possible-present--not-know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-58-not-possible-absent--not-know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-59-not-possible-absent--not-know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-60-not-possible-present--not-possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-61-not-possible-present--not-possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-62-not-possible-absent--not-possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/42b-same-name-multi-arity-cross-63-not-possible-absent--not-possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/43-g91-possible-recursion-self-support.xlog` | 0 | 0 | true | G91 retained as allowed epistemic-mode terminology |  |
| `examples/epistemic/44a-single-modal-truth-table-faeel-know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/44a-single-modal-truth-table-faeel-know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/44a-single-modal-truth-table-faeel-not-know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/44a-single-modal-truth-table-faeel-not-know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/44a-single-modal-truth-table-faeel-not-possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/44a-single-modal-truth-table-faeel-not-possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/44a-single-modal-truth-table-faeel-possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/44a-single-modal-truth-table-faeel-possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/44a-single-modal-truth-table-g91-know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/44a-single-modal-truth-table-g91-know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/44a-single-modal-truth-table-g91-not-know-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/44a-single-modal-truth-table-g91-not-know-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/44a-single-modal-truth-table-g91-not-possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/44a-single-modal-truth-table-g91-not-possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/44a-single-modal-truth-table-g91-possible-absent.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/44a-single-modal-truth-table-g91-possible-present.xlog` | 0 | 0 | true |  |  |
| `examples/epistemic/README.md` | 120 | 0 | true | release/task labels, historical bundle/case codes, emphasis shorthand, and bare example references(120); G91, FAEEL, EIR, WFS, S5, EDB, GPU, and CPU retained as allowed XLOG/domain terminology |  |
| `examples/neural/01_minimal/dataset.json` | 0 | 0 | true |  |  |
| `examples/neural/01_minimal/minimal.xlog` | 4 | 4 | true | digit variable shorthand(4) | digit variable shorthand(4) |
| `examples/neural/01_minimal/train.py` | 4 | 4 | true | digit variable shorthand(4) | digit variable shorthand(4) |
| `examples/neural/02_coins/README.md` | 0 | 0 | true |  |  |
| `examples/neural/02_coins/dataset.json` | 0 | 0 | true |  |  |
| `examples/neural/02_coins/model.py` | 0 | 0 | true |  |  |
| `examples/neural/02_coins/program.xlog` | 0 | 0 | true |  |  |
| `examples/neural/02_coins/train.py` | 0 | 0 | true |  |  |
| `examples/neural/03_mnist_multidigit/README.md` | 0 | 0 | true |  |  |
| `examples/neural/03_mnist_multidigit/dataset.json` | 0 | 0 | true |  |  |
| `examples/neural/03_mnist_multidigit/model.py` | 0 | 0 | true |  |  |
| `examples/neural/03_mnist_multidigit/program.xlog` | 0 | 8 | true |  | digit variable shorthand(8) |
| `examples/neural/03_mnist_multidigit/train.py` | 0 | 0 | true |  |  |
| `examples/neural/04_hwf/README.md` | 0 | 0 | true |  |  |
| `examples/neural/04_hwf/dataset.json` | 0 | 0 | true |  |  |
| `examples/neural/04_hwf/model.py` | 0 | 0 | true |  |  |
| `examples/neural/04_hwf/program.xlog` | 0 | 0 | true |  |  |
| `examples/neural/04_hwf/train.py` | 0 | 0 | true |  |  |
| `examples/neural/05_poker/README.md` | 0 | 0 | true |  |  |
| `examples/neural/05_poker/dataset.json` | 0 | 0 | true |  |  |
| `examples/neural/05_poker/model.py` | 0 | 0 | true |  |  |
| `examples/neural/05_poker/program.xlog` | 0 | 0 | true |  |  |
| `examples/neural/05_poker/train.py` | 0 | 0 | true |  |  |
| `examples/neural/06_clutrr/README.md` | 0 | 0 | true |  |  |
| `examples/neural/06_clutrr/dataset.json` | 0 | 0 | true |  |  |
| `examples/neural/06_clutrr/model.py` | 0 | 0 | true |  |  |
| `examples/neural/06_clutrr/program.xlog` | 0 | 0 | true |  |  |
| `examples/neural/06_clutrr/train.py` | 0 | 0 | true |  |  |
| `examples/neural/README.md` | 4 | 8 | true | release/milestone labels(4) | digit variable shorthand(8) |
| `examples/neural/baseline/deepproblog/CLUTRR/__init__.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/CLUTRR/architecture.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/CLUTRR/clutrr.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/CLUTRR/data/ATTRIBUTION.md` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/CLUTRR/data/__init__.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/CLUTRR/data/data_2d5007e7/config.json` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/CLUTRR/data/data_47b0ffea/config.json` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/CLUTRR/data/data_6b1c2f15/config.json` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/CLUTRR/data/data_a7d9402e/config.json` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/CLUTRR/data/data_a9dcffad/config.json` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/CLUTRR/data/data_b11e6a4f/config.json` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/CLUTRR/data/data_f70b574f/config.json` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Coins/README.md` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Coins/__init__.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Coins/coins.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Coins/data/__init__.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Coins/data/blender_files/__init__.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Coins/data/blender_files/render_script.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Coins/data/dataset.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Coins/data/generate_data.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Coins/data/render.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/__init__.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/add.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/__init__.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/extract.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/generate_all.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train24_test128/debug.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train24_test128/dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train24_test128/test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train24_test128/train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train2_test64/debug.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train2_test64/dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train2_test64/test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train2_test64/train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train2_test64_dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train2_test64_test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train2_test64_train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train2_test8/debug.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train2_test8/dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train2_test8/test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train2_test8/train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train2_test8_dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train2_test8_test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train2_test8_train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train4_test64/debug.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train4_test64/dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train4_test64/test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train4_test64/train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train4_test64_dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train4_test64_test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train4_test64_train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train4_test8/debug.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train4_test8/dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train4_test8/test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train4_test8/train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train4_test8_dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train4_test8_test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train4_test8_train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train8_test64/debug.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train8_test64/dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train8_test64/test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train8_test64/train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train8_test64_dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train8_test64_test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train8_test64_train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train8_test8/debug.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train8_test8/dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train8_test8/test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train8_test8/train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train8_test8_dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train8_test8_test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Add/data/train8_test8_train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/README.md` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/__init__.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/__init__.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/extract.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/generate.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train2_test64/debug.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train2_test64/dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train2_test64/test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train2_test64/train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train2_test64_dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train2_test64_test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train2_test64_train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train2_test8/debug.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train2_test8/dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train2_test8/test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train2_test8/train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train2_test8_dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train2_test8_test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train2_test8_train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train3_test64/debug.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train3_test64/dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train3_test64/test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train3_test64/train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train3_test64_dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train3_test64_test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train3_test64_train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train3_test8/debug.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train3_test8/dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train3_test8/test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train3_test8/train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train3_test8_dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train3_test8_test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train3_test8_train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train4_test64/debug.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train4_test64/dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train4_test64/test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train4_test64/train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train4_test64_dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train4_test64_test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train4_test64_train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train4_test8/debug.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train4_test8/dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train4_test8/test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train4_test8/train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train4_test8_dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train4_test8_test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train4_test8_train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train5_test64_dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train5_test64_test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train5_test64_train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train5_test8_dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train5_test8_test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train5_test8_train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train6_test64_dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train6_test64_test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train6_test64_train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train6_test8_dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train6_test8_test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train6_test8_train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train7_test64_dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train7_test64_test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train7_test64_train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train7_test8_dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train7_test8_test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train7_test8_train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train8_test64_dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train8_test64_test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train8_test64_train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train8_test8_dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train8_test8_test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/data/train8_test8_train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/Sort/sort.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/WAP/__init__.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/WAP/data/README.md` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/WAP/data/__init__.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/WAP/data/convert.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/WAP/data/debug.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/WAP/data/dev.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/WAP/data/questions.json` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/WAP/data/test.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/WAP/data/train.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/WAP/data/vocab_636.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/WAP/data/vocab_746.txt` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/WAP/wap.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/WAP/wap_network.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Forth/__init__.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/HWF/README.md` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/HWF/architecture.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/HWF/data/__init__.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/HWF/data/download_hwf.sh` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/HWF/data/split_train_val.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/HWF/hwf.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/HWF/network.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/MNIST/README.md` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/MNIST/__init__.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/MNIST/addition.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/MNIST/addition_noisy.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/MNIST/data/__init__.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/MNIST/models/pretrained/create_pretrained.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/MNIST/network.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/MNIST/neural_baseline/baseline.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/MNIST/neural_baseline/baseline_models.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/MNIST/neural_baseline/multidigit_baseline.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Poker/README.md` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Poker/__init__.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Poker/cfg/__init__.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Poker/cfg/generate_cfg.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Poker/data/__init__.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Poker/data/generate.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Poker/data/render.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/Poker/poker.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/__init__.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/_xlog_runtime.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/minimal/__init__.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/minimal/addition_minimal.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/minimal/data.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/deepproblog/minimal/network.py` | 0 | 0 | true |  |  |
| `examples/neural/baseline/scallop/mnist_addition.py` | 0 | 0 | true |  |  |
| `examples/neural/tests/test_batch_parity.py` | 0 | 4 | true | L2 retained as standard norm terminology | digit variable shorthand(4) |
| `examples/prob/01-wet-conditioning.xlog` | 0 | 0 | true |  |  |
| `examples/prob/02-annotated-disjunction.xlog` | 0 | 0 | true |  |  |
| `examples/prob/03-recursive-reachability.xlog` | 0 | 0 | true |  |  |
| `examples/prob/04-recursive-mc.xlog` | 0 | 0 | true |  |  |
| `examples/prob/README.md` | 2 | 0 | true | phase/task labels(2) |  |
| `examples/python/01_dlpack_reachability_torch.py` | 0 | 0 | true |  |  |
| `examples/python/02_prob_wet_conditioning_torch.py` | 0 | 0 | true |  |  |
| `examples/python/03_prob_mc_nonmonotone_torch.py` | 0 | 0 | true |  |  |
| `examples/python/README.md` | 0 | 0 | true |  |  |
| `examples/external-consumer-python/01_async_streaming_reachability/program.xlog` | 0 | 0 | true |  |  |
| `examples/external-consumer-python/01_async_streaming_reachability/run.py` | 0 | 1 | true |  | release-coded CUDA requirement message(1) |
| `examples/external-consumer-python/02_wmir_relation_deltas/program.xlog` | 0 | 0 | true |  |  |
| `examples/external-consumer-python/02_wmir_relation_deltas/run.py` | 0 | 1 | true |  | release-coded CUDA requirement message(1) |
| `examples/external-consumer-python/03_neural_bridge_topk_belnap/program.xlog` | 0 | 0 | true |  |  |
| `examples/external-consumer-python/03_neural_bridge_topk_belnap/run.py` | 0 | 1 | true |  | release-coded CUDA requirement message(1) |
| `examples/external-consumer-python/04_native_exact_induction/program.xlog` | 0 | 0 | true |  |  |
| `examples/external-consumer-python/04_native_exact_induction/run.py` | 0 | 1 | true |  | release-coded CUDA requirement message(1) |
| `examples/external-consumer-python/05_probabilistic_async_diagnostics/program.xlog` | 0 | 0 | true |  |  |
| `examples/external-consumer-python/05_probabilistic_async_diagnostics/run.py` | 0 | 1 | true |  | release-coded CUDA requirement message(1) |
| `examples/external-consumer-python/README.md` | 9 | 3 | true | release labels, task/gate labels, external-consumer path labels, and phase shorthand(9) | release validator/evidence path labels(3) |
| `examples/language-completeness/aggregate_lifting/count_lift.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/approx/aggregate_mc.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/lists/cons_patterns.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/lists/membership.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/lists/transforms.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/magic_sets/reach_bound.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/meta/findall.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/meta/inspection.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/meta/maplist.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/naf/closed_world.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/prob_aggregates/finite_outcomes.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/01_list_typed_relation/README.md` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/01_list_typed_relation/expected.json` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/01_list_typed_relation/program.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/02_findall_aggregate/README.md` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/02_findall_aggregate/expected.json` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/02_findall_aggregate/program.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/03_maplist_static_predref/README.md` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/03_maplist_static_predref/expected.json` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/03_maplist_static_predref/program.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/04_magic_reach_explain/README.md` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/04_magic_reach_explain/expected.json` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/04_magic_reach_explain/program.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/05_prob_aggregate_exact/README.md` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/05_prob_aggregate_exact/expected.json` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/05_prob_aggregate_exact/program.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/06_prob_aggregate_mc/README.md` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/06_prob_aggregate_mc/expected.json` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/06_prob_aggregate_mc/program.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/07_aggregate_lifting/README.md` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/07_aggregate_lifting/expected.json` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/07_aggregate_lifting/program.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/08_approx_confidence/README.md` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/08_approx_confidence/expected.json` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/08_approx_confidence/program.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/09_repl_watch_explain/README.md` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/09_repl_watch_explain/expected.json` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/09_repl_watch_explain/program.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/10_scientific_incremental/README.md` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/10_scientific_incremental/expected.json` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/10_scientific_incremental/program.xlog` | 0 | 0 | true |  |  |
| `examples/language-completeness/showcase/README.md` | 4 | 1 | true | release/suite/sub-goal labels(4) | release-specific validator path label(1) |
| `examples/v086-runtime/01_dts_delta_optimizer/README.md` | 1 | 0 | true | release label(1) |  |
| `examples/v086-runtime/01_dts_delta_optimizer/expected.json` | 0 | 8 | true |  | external consumer and predicate-name artifacts(8) |
| `examples/v086-runtime/01_dts_delta_optimizer/program.xlog` | 0 | 25 | true |  | external consumer predicate-name artifacts(25) |
| `examples/v086-runtime/02_neutral_material_flow/README.md` | 0 | 0 | true |  |  |
| `examples/v086-runtime/02_neutral_material_flow/expected.json` | 0 | 1 | true |  | external consumer name(1) |
| `examples/v086-runtime/02_neutral_material_flow/program.xlog` | 0 | 0 | true |  |  |
| `examples/v086-runtime/03_neutral_signal_diagnostics/README.md` | 0 | 0 | true |  |  |
| `examples/v086-runtime/03_neutral_signal_diagnostics/expected.json` | 0 | 1 | true |  | external consumer name(1) |
| `examples/v086-runtime/03_neutral_signal_diagnostics/program.xlog` | 0 | 0 | true |  |  |
| `examples/v086-runtime/04_v090_substrate_primitives/README.md` | 2 | 0 | true | release labels(2) |  |
| `examples/v086-runtime/04_v090_substrate_primitives/expected.json` | 0 | 2 | true |  | release substrate contract labels(2) |
| `examples/v086-runtime/04_v090_substrate_primitives/program.xlog` | 2 | 0 | true | release labels(2) |  |
| `examples/v086-runtime/05_pyxlog_session_compatibility/README.md` | 1 | 0 | true | release label(1) |  |
| `examples/v086-runtime/05_pyxlog_session_compatibility/expected.json` | 0 | 0 | true |  |  |
| `examples/v086-runtime/05_pyxlog_session_compatibility/program.xlog` | 0 | 0 | true |  |  |
| `examples/v086-runtime/README.md` | 6 | 1 | true | release labels and external-consumer name(6) | release-coded validator path(1) |
| `examples/xlog/00-basics/01_tc_reachability.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/00-basics/02_stratified_isolated.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/00-basics/03_constraints_acyclic.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/00-basics/04_comparisons_and_equality.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/10-arithmetic/01_arithmetic_demo.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/10-arithmetic/02_builtins_and_precedence.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/15-float-predicates/01_nan_handling.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/15-float-predicates/02_infinity_detection.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/15-float-predicates/03_signed_zero.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/15-float-predicates/04_data_quality_pipeline.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/15-float-predicates/05_statistical_analysis.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/15-float-predicates/README.md` | 0 | 0 | true |  |  |
| `examples/xlog/20-graphs/01_triangle_detection.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/20-graphs/02_social_network_recommendations.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/20-graphs/03_network_connectivity.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/30-security/01_rbac_permissions.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/30-security/02_rbac_separation_of_duties.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/40-supply-chain/01_bom_explosion_leaf_parts.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/40-supply-chain/02_bom_quantities_and_totals.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/50-program-analysis/01_points_to_copy.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/50-program-analysis/02_call_graph.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/60-database-style/01_local_orders_join.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/60-database-style/02_star_schema_sales_agg.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/70-aggregates/01_out_degree_count.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/70-aggregates/02_multi_key_sum.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/70-aggregates/03_logsumexp.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/70-aggregates/04_min_max_latency_stats.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/70-aggregates/05_average_from_sum_count.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/80-v032-showcase/01-enterprise/README.md` | 10 | 1 | true | release label and reader-facing abbreviations(10) | release-coded run path(1) |
| `examples/xlog/80-v032-showcase/01-enterprise/finance/compensation.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/80-v032-showcase/01-enterprise/hr/employees.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/80-v032-showcase/01-enterprise/main.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/80-v032-showcase/01-enterprise/org/hierarchy.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/80-v032-showcase/02-knowledge-graph/README.md` | 40 | 1 | true | stale research-publication wording and abbreviations(40) | release-coded run path(1) |
| `examples/xlog/80-v032-showcase/02-knowledge-graph/entities/movies.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/80-v032-showcase/02-knowledge-graph/inference/reasoning.xlog` | 9 | 0 | true | user-defined-function abbreviation comments(9) |  |
| `examples/xlog/80-v032-showcase/02-knowledge-graph/main.xlog` | 4 | 0 | true | release label and abbreviation comments(4) |  |
| `examples/xlog/80-v032-showcase/02-knowledge-graph/ontology/schema.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/80-v032-showcase/03-game-analytics/README.md` | 24 | 1 | true | release label, game-stat abbreviations, and shorthand snippet variables(24) | release-coded run path(1) |
| `examples/xlog/80-v032-showcase/03-game-analytics/achievements/system.xlog` | 2 | 0 | true | user-defined-function abbreviation comments(2) |  |
| `examples/xlog/80-v032-showcase/03-game-analytics/main.xlog` | 21 | 113 | true | release label and abbreviation comments(14), schema-comment shorthand(7) | local variable and predicate-name shorthand(113) |
| `examples/xlog/80-v032-showcase/03-game-analytics/matches/history.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/80-v032-showcase/03-game-analytics/players/profiles.xlog` | 7 | 37 | true | identifier and regional shorthand comments(7) | local variable shorthand(37) |
| `examples/xlog/80-v032-showcase/03-game-analytics/ranking/elo.xlog` | 28 | 0 | true | Elo/Kill-death-assist/user-defined-function abbreviation comments(28) |  |
| `examples/xlog/80-v032-showcase/04-supply-chain/README.md` | 33 | 1 | true | release label and supply-chain shorthand(33) | release-coded run path(1) |
| `examples/xlog/80-v032-showcase/04-supply-chain/cost/calculator.xlog` | 6 | 13 | true | user-defined-function shorthand comments(6) | local variable shorthand(13) |
| `examples/xlog/80-v032-showcase/04-supply-chain/inventory/stock.xlog` | 8 | 45 | true | bill-of-materials shorthand comments(8) | local variable shorthand(45) |
| `examples/xlog/80-v032-showcase/04-supply-chain/main.xlog` | 5 | 75 | true | release/user-defined-function/bill-of-materials comments(5) | local variable shorthand(75) |
| `examples/xlog/80-v032-showcase/04-supply-chain/shipping/routes.xlog` | 0 | 44 | true |  | local variable shorthand(44) |
| `examples/xlog/80-v032-showcase/README.md` | 57 | 3 | true | release labels, stale-domain snippets, and shorthand terms(57) | release-coded run paths(3) |
| `examples/xlog/90-negative-tests/01_constraint_cycle_violation.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/90-negative-tests/02_stratification_negation_cycle.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/90-negative-tests/03_arithmetic_type_mismatch.xlog` | 0 | 0 | true |  |  |
| `examples/xlog/90-negative-tests/04_is_target_already_bound.xlog` | 0 | 0 | true |  |  |
| `fuzz/Cargo.toml` | 0 | 0 | true |  |  |
| `fuzz/fuzz_targets/fuzz_compiler.rs` | 0 | 0 | true |  |  |
| `fuzz/fuzz_targets/fuzz_parser.rs` | 0 | 0 | true |  |  |
| `fuzz/fuzz_targets/fuzz_type_inference.rs` | 0 | 0 | true |  |  |
| `python/examples/external_consumer_exact_induce_parity.py` | 1 | 1 | true | task/milestone label(1) | external-consumer filename(1) |
| `python/examples/dts_fit_benchmark.py` | 0 | 0 | true |  |  |
| `python/examples/ilp_showcase.py` | 0 | 0 | true |  |  |
| `python/pyproject.toml` | 0 | 0 | true |  |  |
| `python/tests/conftest.py` | 0 | 0 | true |  |  |
| `python/tests/fixtures/external_consumer_frozen_inputs.json` | 0 | 33 | true |  | external-consumer and task labels; opaque relation and witness variable names(33) |
| `python/tests/fixtures/manifest_sample.txt` | 0 | 0 | true |  |  |
| `python/tests/neural_test_env.py` | 0 | 0 | true |  |  |
| `python/tests/test_arrow_device_import.py` | 0 | 0 | true |  |  |
| `python/tests/test_backward.py` | 0 | 0 | true |  |  |
| `python/tests/test_batch_eval.py` | 1 | 33 | true | shorthand sum result in comment(1) | shorthand digit, token, value, tag, number, and sum variables(33) |
| `python/tests/test_batched_forward.py` | 0 | 14 | true |  | shorthand digit, sum, output-dimension, optimizer, and layer names(14) |
| `python/tests/test_circuit_cache.py` | 1 | 35 | true | Decision-DNNF compiler shorthand(1) | shorthand digit, sum, class-count, input-dimension, layer, loss, and query names(35) |
| `python/tests/test_coins.py` | 0 | 0 | true |  |  |
| `python/tests/test_dataset_manifest.py` | 0 | 0 | true |  |  |
| `python/tests/test_embeddings.py` | 2 | 26 | true | task label and gradient shorthand(2) | shorthand embedded variables, vocabulary, dimension, declaration, network, and API-adjacent names(26) |
| `python/tests/test_example_02_coins.py` | 0 | 0 | true |  |  |
| `python/tests/test_example_03_mnist_multidigit.py` | 0 | 0 | true |  |  |
| `python/tests/test_example_04_hwf.py` | 0 | 0 | true |  |  |
| `python/tests/test_example_05_poker.py` | 0 | 0 | true |  |  |
| `python/tests/test_example_05_poker_filename_parsing.py` | 0 | 0 | true |  |  |
| `python/tests/test_example_05_poker_model_focus.py` | 0 | 0 | true |  |  |
| `python/tests/test_example_06_clutrr.py` | 0 | 0 | true |  |  |
| `python/tests/test_gpu_native_forward_backward_returns_tensor.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_artifact.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_backend.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_beta_gate.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_candidates.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_credit_gpu.py` | 8 | 10 | true | device-to-host shorthand and task section labels(8) | device-to-host shorthand in names/messages(10) |
| `python/tests/test_ilp_device_to_host_gate.py` | 7 | 11 | true | device-to-host shorthand in docstrings and comments(7) | device-to-host shorthand in path and test names(11) |
| `python/tests/test_ilp_device_queries.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_entropy.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_exact_induce.py` | 8 | 67 | true | milestone label, device-to-host shorthand, and tuple-field shorthand(8) | device-to-host shorthand and local request/result variable shorthands(67) |
| `python/tests/test_ilp_exact_induce_default_dispatch.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_ga_reliability.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_hard_negatives.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_holdout.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_multistart.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_performance.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_promoter.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_recursive.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_reliability.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_reset.py` | 1 | 2 | true | device-to-host shorthand in docstring(1) | device-to-host shorthand in test name and assertion message(2) |
| `python/tests/test_ilp_robustness.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_rule_inventory.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_showcase.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_sparse.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_sparse_guard.py` | 2 | 5 | true | device-to-host shorthand in docstring(2) | device-to-host shorthand in test names and assertion message(5) |
| `python/tests/test_ilp_temperature.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_trainer.py` | 0 | 0 | true |  |  |
| `python/tests/test_ilp_types.py` | 0 | 0 | true |  |  |
| `python/tests/test_kernel_packaging_layout.py` | 0 | 0 | true |  |  |
| `python/tests/test_label_mapping.py` | 0 | 0 | true |  |  |
| `python/tests/test_logic_external_consumer_frozen_replay_determinism.py` | 0 | 9 | true |  | external-consumer and task labels in env, path, and test naming(9) |
| `python/tests/test_logic_recursive_support_gpu.py` | 1 | 161 | true | midpoint variable shorthand in comment(1) | body-predicate variables, midpoint variable, and opaque relation names(161) |
| `python/tests/test_logic_relation_store.py` | 0 | 0 | true |  |  |
| `python/tests/test_loss.py` | 0 | 0 | true |  |  |
| `python/tests/test_mc_device_results.py` | 0 | 0 | true |  |  |
| `python/tests/test_minimal_example.py` | 0 | 0 | true |  |  |
| `python/tests/test_minimal_example_fast_training.py` | 0 | 0 | true |  |  |
| `python/tests/test_minimal_example_summary.py` | 0 | 0 | true |  |  |
| `python/tests/test_multi_network.py` | 0 | 0 | true |  |  |
| `python/tests/test_negation.py` | 0 | 0 | true |  |  |
| `python/tests/test_network_registry.py` | 0 | 42 | true |  | digit, output, network, optimizer, and multi-input shorthand names(42) |
| `python/tests/test_neural_accuracy_gates.py` | 0 | 0 | true |  |  |
| `python/tests/test_neural_registry.py` | 0 | 0 | true |  |  |
| `python/tests/test_nn4_cuda_no_host_transfer_contract.py` | 0 | 0 | true |  |  |
| `python/tests/test_nn4_dilp_training_surface.py` | 0 | 0 | true |  |  |
| `python/tests/test_nn4_training_lineage.py` | 0 | 0 | true |  |  |
| `python/tests/test_preflight_release_publish_cli.py` | 0 | 0 | true |  |  |
| `python/tests/test_pyxlog_evidence_api.py` | 0 | 0 | true |  |  |
| `python/tests/test_tensor_source.py` | 8 | 30 | true | image-shape shorthand in comments(8) | digit, word, tag, embedding, class, and input variable shorthands(30) |
| `python/tests/test_train_model_tensor.py` | 5 | 12 | true | Decision-DNNF shorthand and section-code comments(5) | digit variable shorthands(12) |
| `python/tests/test_training.py` | 0 | 0 | true |  |  |
| `python/tests/test_transfer_metric_diagnostics.py` | 0 | 0 | true |  |  |
| `python/tests/test_typed_batch_upload.py` | 0 | 0 | true |  |  |
| `python/tests/test_bridge_source.py` | 0 | 13 | true |  | release-coded path/function names, evidence path strings, and bridge milestone codes(13) |
| `python/tests/test_delta_source.py` | 0 | 12 | true |  | release-coded path/function names, evidence path strings, delta milestone codes, and opaque fixture name(12) |
| `python/tests/test_external_consumer_cert.py` | 0 | 9 | true |  | release-coded and external-consumer path, import, and test names(9) |
| `python/tests/test_exact_source.py` | 0 | 8 | true |  | release-coded test/evidence names and external consumer literals(8) |
| `python/tests/test_external_consumer_examples_source.py` | 0 | 6 | true |  | release-coded test, suite, validator, and evidence names(6) |
| `python/tests/test_v080_pyapi_source.py` | 0 | 0 | true |  |  |
| `python/tests/test_v085_examples_source.py` | 0 | 4 | false |  | v085(4) |
| `python/tests/test_v086_adaptive_reoptimization_source.py` | 0 | 1 | false |  | v086(1) |
| `python/tests/test_v086_chain_smem_profile_source.py` | 0 | 1 | false |  | v086(1) |
| `python/tests/test_v086_chain_smem_source.py` | 0 | 1 | false |  | v086(1) |
| `python/tests/test_v086_consumers_source.py` | 0 | 27 | false |  | release-coded file/test labels(27) |
| `python/tests/test_v086_cse_source.py` | 0 | 1 | false |  | v086(1) |
| `python/tests/test_v086_delta_coalescing.py` | 0 | 2 | false |  | v086(2) |
| `python/tests/test_v086_exact_types_runtime.py` | 0 | 0 | true |  |  |
| `python/tests/test_v086_exact_types_source.py` | 0 | 1 | false |  | v086(1) |
| `python/tests/test_v086_persistent_hash_index_source.py` | 0 | 1 | false |  | v086(1) |
| `python/tests/test_v086_pyxlog_persistent_index_runtime.py` | 0 | 0 | true |  |  |
| `python/tests/test_v086_relation_callbacks.py` | 0 | 2 | false |  | v086(2) |
| `python/tests/test_v086_relation_callbacks_runtime.py` | 0 | 0 | true |  |  |
| `python/tests/test_v087_lwm_source.py` | 0 | 0 | true |  |  |
| `python/tests/test_v088_lwm_source.py` | 0 | 0 | true |  |  |
| `python/tests/test_validate_examples_cli.py` | 0 | 0 | true |  |  |
| `python/tests/test_validate_package_metadata.py` | 0 | 0 | true |  |  |
| `python/tests/test_validate_release_gpu_cli.py` | 0 | 0 | true |  |  |
| `python/tests/test_validation_staging.py` | 0 | 0 | true |  |  |
| `python/tests/test_xlog_doctor.py` | 0 | 0 | true |  |  |
| `release-plz.toml` | 0 | 0 | true |  |  |
| `scripts/__init__.py` | 0 | 0 | true |  |  |
| `scripts/cache_ablation.py` | 0 | 0 | true |  |  |
| `scripts/check_warning_free.sh` | 0 | 0 | true |  |  |
| `scripts/collect_track_a_results.sh` | 0 | 0 | true |  |  |
| `scripts/install_pyxlog_for_python.py` | 0 | 0 | true |  |  |
| `scripts/install_rust_toolchain.sh` | 0 | 0 | true |  |  |
| `scripts/measure_v086_chain_smem.py` | 0 | 0 | true |  |  |
| `scripts/neural_datasets.py` | 0 | 0 | true |  |  |
| `scripts/neural_training.py` | 0 | 0 | true |  |  |
| `scripts/package_cli_release.sh` | 0 | 0 | true |  |  |
| `scripts/preflight_release_publish.sh` | 0 | 0 | true |  |  |
| `scripts/repro/dts_dlm_p0_session_fuzz.py` | 1 | 6 | false | P0.4(1) | P0.1(2), P0.2(2), P0.4(2) |
| `scripts/run_prob_examples.py` | 0 | 0 | true |  |  |
| `scripts/run_xlog_examples.sh` | 0 | 0 | true |  |  |
| `scripts/rustc-wrapper.sh` | 0 | 0 | true |  |  |
| `scripts/stage_kernels.py` | 0 | 0 | true |  |  |
| `scripts/stage_pyxlog_kernels.sh` | 0 | 0 | true |  |  |
| `scripts/track_a_runner.py` | 0 | 0 | true |  |  |
| `scripts/track_b_runner.py` | 0 | 0 | true |  |  |
| `scripts/external_consumer_cert.py` | 1 | 6 | true | release label in module docstring(1) | release-coded/external-consumer path and manifest labels(6) |
| `scripts/v080_pyxlog_runtime_probe.py` | 0 | 0 | true |  |  |
| `scripts/validate_examples.py` | 0 | 0 | true |  |  |
| `scripts/validate_package_metadata.py` | 0 | 0 | true |  |  |
| `scripts/validate_release_gpu.sh` | 0 | 0 | true |  |  |
| `scripts/validate_external_consumer_examples.py` | 0 | 5 | true |  | release-coded validator, suite, evidence, and output labels(5) |
| `scripts/validate_v085_examples.py` | 0 | 2 | false |  | v085(2) |
| `scripts/validate_v086_examples.py` | 0 | 57 | false |  | release-coded validator/path labels(57) |
| `scripts/validation_staging.py` | 0 | 0 | true |  |  |
| `scripts/xlog_doctor.py` | 0 | 0 | true |  |  |
| `tools/exdev-shim/build.sh` | 0 | 0 | true |  |  |
| `tools/exdev-shim/exdev_shim.c` | 0 | 0 | true |  |  |
