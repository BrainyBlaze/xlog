# Resolved XLOG, pyxlog, and Example Fixes

This file maps the BFO reasoner findings to the upstream work and regression
tests that resolve the Universal Case Reasoner XLOG issue ledger.

## Core XLOG

### Differentiable proof traces

Implemented `xlog_logic::DifferentiableProofTraceMap` for selected rules and
query answers. It exposes stable proof IDs, clause IDs, support atoms, symbolic
weights, binary logistic loss, and a gradient hook usable by pyxlog training.

Suggested regression test:
`crates/xlog-logic/tests/differentiable_proof_trace.rs`

Assertion: a `proof_path(Case, Root, Clause)` query returns proof IDs and clause
weights that receive nonzero gradients during a toy training step.

### Module boundary diagnostics

Implemented `xlog_logic::diagnose_module_boundaries` with manifests for frozen
kernels and adapter-only fact modules. The diagnostic reports when a domain
adapter mutates a frozen BFO predicate, declares an adapter rule, or marks
held-out labels as candidate sources.

Suggested regression test:
`crates/xlog-logic/tests/module_boundary_diagnostics.rs`

Assertion: a frozen core plus two adapters compiles when adapters only add
facts, and fails when a held-out adapter mutates the core or leaks held-out
labels into candidate generation.

## pyxlog

### Joint nn/4 plus symbolic rule-weight training

Implemented `pyxlog.ilp.neurosymbolic.train_neurosymbolic_program`, a trainable
neuro-symbolic program API that accepts `nn/4` predicates, candidate clauses,
symbolic rule weights, and a loss specification. The API returns trained neural
gradient evidence, symbolic gradients, symbolic rule weights, and an inventory.

Suggested regression test:
`python/tests/test_nn4_dilp_training_surface.py`

Assertion: a minimal `nn/4` predicate and a learnable symbolic clause train in
the same optimizer step and both neural and symbolic parameters receive finite,
nonzero gradients.

### Learned rule inventory JSON

Implemented `pyxlog.ilp.inventory.build_rule_inventory` and
`train_and_promote(...).rule_inventory`, recording selected and rejected
clauses, fold membership, excluded held-out domains, symbolic weights, score
deltas, and base-kernel checksums.

Suggested regression test:
`python/tests/test_ilp_rule_inventory.py`

Assertion: `train_and_promote` emits a machine-readable rule inventory proving
the held-out fold was excluded and the frozen kernel checksum stayed unchanged.

### CUDA no-host-transfer audit context

Implemented `pyxlog.runtime_audit.CudaExecutionAudit`, a runtime audit context
around `nn/4` execution. It reports device-to-host and host-to-device counts, score-row host
materialization, scalar extraction, and fail-closed violations for the audited
hot loop.

Suggested regression test:
`python/tests/test_nn4_cuda_no_host_transfer_contract.py`

Assertion: a CUDA ranking loop reports zero hot-loop transfers and fails the
audit when `.cpu()`, `.tolist()`, scalar extraction, or score-row download occurs
inside the ranked selection path.

### Grouped transfer metrics

Implemented `pyxlog.transfer_diagnostics.compute_transfer_diagnostics` for
grouped classification and transfer evaluation: macro F1, minimum per-group F1,
baseline uplift, bootstrap confidence intervals, paired sign tests, variant
coverage, and missing-group failure modes.

Suggested regression test:
`python/tests/test_transfer_metric_diagnostics.py`

Assertion: raw records grouped by held-out domain and adversarial variant
produce the same macro F1 and missing-domain failures as the Universal Case Reasoner validator.

## Universal Case Reasoner Example

The example should continue to own domain-specific evidence and scientific
thresholds:

- External Hugging Face source contracts and root-cause truth fields.
- BFO domain catalogs, interventions, risk states, and explanation payloads.
- Leave-one-domain-out split policy and adversarial variant construction.
- Project-specific acceptance values: macro F1, minimum domain F1, uplift,
  p95 latency, soak duration, and drift thresholds.

The example is no longer the owner of generic proof-gradient, rule-inventory,
CUDA host-transfer audit, or grouped transfer-metric machinery. Those reusable
surfaces now live in core XLOG or pyxlog and can be reused by the Universal Case Reasoner.

## Regression Candidate Map

| Finding | Repo location | Minimal assertion |
| --- | --- | --- |
| Joint neural and symbolic training | `python/tests/test_nn4_dilp_training_surface.py` | Joint `nn/4` and symbolic rule weights train with nonzero gradients. |
| Differentiable proof traces | `crates/xlog-logic/tests/differentiable_proof_trace.rs` | Proof IDs and clause weights are exported and differentiable. |
| Learned rule inventory | `python/tests/test_ilp_rule_inventory.py` | Learned rule inventory records selected clauses, held-out exclusion, and kernel checksum. |
| CUDA host-transfer audit | `python/tests/test_nn4_cuda_no_host_transfer_contract.py` | Hot-loop CUDA ranking fails on host materialization or scalar extraction. |
| Module boundary diagnostics | `crates/xlog-logic/tests/module_boundary_diagnostics.rs` | Frozen core and adapter-only module diagnostics catch mutation and label leakage. |
| Grouped transfer metrics | `python/tests/test_transfer_metric_diagnostics.py` | Grouped leave-one-domain-out metrics and missing-domain failures are recomputed from raw records. |
