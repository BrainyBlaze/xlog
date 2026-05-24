# XLOG Findings: BFO Universal Case Reasoner

This document records what Project 3 revealed about XLOG while stress-testing
robust cross-domain transfer, reusable BFO kernel boundaries, `nn/4` CUDA
ranking, and leave-one-domain-out generalization.

## What did this project reveal about XLOG?

XLOG is strong enough to support a production-grade cross-domain ranking demo:
the same BFO-oriented example can call a CUDA PyTorch model through `nn/4`, keep
domain adapters outside the BFO kernel, and validate leave-one-domain-out
generalization from raw prediction records.

The more important finding was the boundary of that claim. The v0.8.9 work now
adds the reusable surfaces that the UCR issue ledger exposed: joint `nn/4` plus
symbolic rule-weight training in pyxlog, differentiable proof traces in
`xlog-logic`, learned-rule inventories, reusable CUDA host-transfer audits,
module-boundary diagnostics, and grouped transfer metrics.

## What broke, flaked, or required workaround code?

- `UCR-XLOG-001`: resolved by
  `pyxlog.ilp.neurosymbolic.train_neurosymbolic_program`.
- `UCR-XLOG-002`: resolved by `xlog_logic::DifferentiableProofTraceMap`.
- `UCR-XLOG-003`: resolved by `pyxlog.ilp.inventory.build_rule_inventory` and
  `train_and_promote(...).rule_inventory`.
- `UCR-XLOG-004`: resolved by `pyxlog.runtime_audit.CudaExecutionAudit`.
- `UCR-XLOG-005`: resolved by `xlog_logic::diagnose_module_boundaries`.
- `UCR-XLOG-006`: resolved by
  `pyxlog.transfer_diagnostics.compute_transfer_diagnostics`.

## What XLOG feature is missing or weak?

The previously missing feature was integrated differentiable rule learning. The
v0.8.9 pyxlog surface now accepts one neuro-symbolic source containing `nn/4`,
`trainable_rule`, and `train(...)` declarations, then trains neural parameters
and symbolic rule weights together while emitting a learned-rule inventory.

The previously weak feature was diagnostics. v0.8.9 adds reusable XLOG/pyxlog
diagnostic contracts for proof traces, frozen/adaptor module boundaries,
held-out candidate provenance, no-host-transfer CUDA loops, and grouped transfer
metrics so future examples do not need to rebuild the same audit layer.

## What should be fixed in core XLOG vs pyxlog vs the example?

Core XLOG now owns the first reusable declarative pieces: differentiable proof
trace maps with stable proof IDs and reusable module-boundary diagnostics for
frozen kernels and adapter-only modules.

pyxlog now owns the first reusable training and audit APIs: CUDA execution audit
contexts, joint neural/symbolic optimizer integration, rule-inventory JSON,
grouped transfer metrics, bootstrap confidence intervals, paired sign tests,
and fail-closed host materialization checks for hot `nn/4` ranking loops.

The example should stay responsible for BFO-specific semantics: domain catalogs,
external Hugging Face source contracts, root-cause truth selection, intervention
labels, explanation payloads, and the scientific acceptance thresholds.

## What minimal test should enter upstream?

The upstream tests are small, deterministic fixtures rather than the full UCR
production soak:

- `UCR-XLOG-001`: `python/tests/test_nn4_dilp_training_surface.py`
  should prove an `nn/4` predicate and a learned symbolic clause train together.
- `UCR-XLOG-002`: `crates/xlog-logic/tests/differentiable_proof_trace.rs`
  should prove proof IDs, clause weights, and gradients are exported.
- `UCR-XLOG-003`: `python/tests/test_ilp_rule_inventory.py` should prove
  learned-rule inventories include fold safety and kernel checksums.
- `UCR-XLOG-004`: `python/tests/test_nn4_cuda_no_host_transfer_contract.py`
  should fail when CUDA scores are materialized on host inside a ranking loop.
- `UCR-XLOG-005`: `crates/xlog-logic/tests/module_boundary_diagnostics.rs`
  should fail when a held-out adapter mutates a frozen core module or leaks
  held-out labels into candidate generation.
- `UCR-XLOG-006`: `python/tests/test_transfer_metric_diagnostics.py`
  should recompute grouped LODO metrics and fail missing-domain aggregates.

## Findings Ledger

The machine-readable ledger is `xlog_issue_ledger.json`. Minimal reproducers
are under `repro/`, and concrete runtime/pyxlog/example fixes are in
`proposed_fixes.md`.

| ID | Component | Severity | Minimal reproducer | Evidence |
| --- | --- | --- | --- | --- |
| `UCR-XLOG-001` | `nn/4` | blocker | `python examples/BFO/universal_case_reasoner/repro/repro_001_nn4_scoring_not_dilp.py` | `programs/production_ranker.xlog` |
| `UCR-XLOG-002` | compiler | blocker | `python examples/BFO/universal_case_reasoner/repro/repro_002_proof_paths_without_gradients.py` | `programs/dilp_proof_paths.xlog` |
| `UCR-XLOG-003` | pyxlog | major | `python examples/BFO/universal_case_reasoner/repro/repro_003_missing_rule_inventory_audit.py` | `tests/test_production_transfer.py` |
| `UCR-XLOG-004` | CUDA runtime | major | `python examples/BFO/universal_case_reasoner/repro/repro_004_scalar_transfer_contract_is_example_side.py` | `tests/test_production_transfer.py` |
| `UCR-XLOG-005` | diagnostics | major | `python examples/BFO/universal_case_reasoner/repro/repro_005_adapter_boundary_diagnostics_are_example_side.py` | `tools/validate_universal_case_reasoner.py` |
| `UCR-XLOG-006` | aggregates | improvement | `python examples/BFO/universal_case_reasoner/repro/repro_006_aggregate_transfer_metrics_need_validator.py` | `tools/validate_universal_case_reasoner.py` |
