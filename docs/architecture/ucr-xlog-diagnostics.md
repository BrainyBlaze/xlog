# Universal Case Reasoner Diagnostics

This document records the v0.8.9 XLOG changes made after the BFO Universal Case
Reasoner stress test. The example exposed six reusable gaps that should belong
to XLOG or pyxlog rather than one project-specific validator.

The reusable surface now covers joint neural/symbolic training, differentiable
proof traces, learned-rule inventories, CUDA host-transfer audits, module
boundary diagnostics, and grouped transfer metrics. The BFO example remains
responsible only for domain semantics, evidence files, and scientific thresholds.

## Issue Map

| Finding | Upstream owner | Runtime surface | Regression |
| --- | --- | --- | --- |
| `UCR-XLOG-001` | `pyxlog.ilp` | `train_neurosymbolic_program(...)` trains `nn/4` outputs and symbolic rule weights from one source | `python/tests/test_nn4_dilp_training_surface.py` |
| `UCR-XLOG-002` | `xlog-logic` | `DifferentiableProofTraceMap` exports stable proof IDs, support atoms, clause weights, and gradients | `crates/xlog-logic/tests/differentiable_proof_trace.rs` |
| `UCR-XLOG-003` | `pyxlog.ilp` | `RuleInventory` and `PromotionResult.rule_inventory` record selected/rejected clauses, folds, held-out domains, and kernel checksums | `python/tests/test_ilp_rule_inventory.py` |
| `UCR-XLOG-004` | `pyxlog` | `CudaExecutionAudit` fails closed on tensor host materialization, scalar extraction, score-row downloads, or recorded transfers | `python/tests/test_nn4_cuda_no_host_transfer_contract.py` |
| `UCR-XLOG-005` | `xlog-logic` | `diagnose_module_boundaries(...)` checks frozen kernels, adapter-only facts, held-out modules, and candidate provenance | `crates/xlog-logic/tests/module_boundary_diagnostics.rs` |
| `UCR-XLOG-006` | `pyxlog` | `compute_transfer_diagnostics(...)` recomputes grouped F1, bootstrap intervals, baseline uplift, paired tests, and missing groups | `python/tests/test_transfer_metric_diagnostics.py` |

The machine-readable ledger and minimal reproducers live under
`examples/BFO/universal_case_reasoner/`.

## Core XLOG Additions

`crates/xlog-logic/src/proof_trace.rs` adds a deterministic proof trace map for
training-time credit assignment. Callers insert proof specs containing an answer
key, clause ID, support atoms, and a symbolic weight. XLOG assigns a stable
proof ID, accumulates binary logistic loss gradients by answer key, and can apply
those gradients to clause weights.

`crates/xlog-logic/src/module_diagnostics.rs` adds reusable boundary checks for
systems that freeze a reusable kernel and attach domain adapters around it. The
diagnostic reports writes to frozen predicates, rules inside adapter-only
modules, declarations from held-out modules, and candidate sources derived from
held-out labels.

Both modules are exported from `xlog_logic` so downstream validation code can use
them without reimplementing example-local checks.

## pyxlog Additions

`pyxlog.ilp.neurosymbolic.train_neurosymbolic_program(...)` accepts one source
containing `nn(...)`, `trainable_rule(...)`, and `train(...)` declarations. It
binds registered PyTorch networks to symbolic rule weights, runs the requested
training objective, and returns neural gradient evidence, symbolic gradients,
final symbolic weights, and a learned-rule inventory.

`pyxlog.ilp.inventory.build_rule_inventory(...)` builds a stable audit artifact
from a learned ILP artifact. `train_and_promote(...)` now accepts fold,
held-out-domain, and base-kernel checksum metadata and returns the inventory on
`PromotionResult`. This makes promotion evidence reusable for cross-domain
transfer audits.

`pyxlog.runtime_audit.CudaExecutionAudit` is a hot-loop audit context for CUDA
ranking paths. It records host-to-device and device-to-host counts, detects
score-row host downloads, and raises `HostMaterializationError` in
`forbid_host_materialization` mode when audited code calls `.cpu()`, `.tolist()`,
or `.item()` inside a GPU-only path.

`pyxlog.transfer_diagnostics.compute_transfer_diagnostics(...)` recomputes
grouped classification metrics from raw prediction records. It returns macro F1,
minimum group F1, per-group rows, bootstrap confidence intervals, baseline
uplift, paired sign-test results, and missing-domain or missing-variant failures.

The top-level `pyxlog` import now tolerates a missing `pyxlog._native` extension
for pure-Python helper tests. Native-backed APIs still fail explicitly when the
extension is not available.

## Example And Packaging Scope

`examples/BFO/universal_case_reasoner/` contains the BFO UCR validation package:
requirements, goals, validation plan, source programs, evidence JSON, minimal
reproducers, project-specific tests, and the resolved `xlog_issue_ledger.json`.
Those files document the domain-specific behavior that remains outside core
XLOG.

The pyxlog kernel staging script now builds the release pyxlog target before it
resolves the release `OUT_DIR`. This prevents stale release kernel artifacts from
being selected when package validation follows a fresh source change. The layout
regression lives in `python/tests/test_kernel_packaging_layout.py`.
