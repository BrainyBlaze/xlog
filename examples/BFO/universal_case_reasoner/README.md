# BFO Universal Case Reasoner

Branch: `feat/bfo-universal-case-reasoner`

This directory is the branch-local contract for a BFO-governed
neuro-symbolic case reasoner. The target system must use one stable BFO kernel
to reason over structurally analogous cases across medicine, manufacturing,
cybersecurity, scientific or lab operations, and cloud operations / quality
incidents.

## Current Status

This example has a strict evidence path for the five-domain BFO/neuro-symbolic
demo, and the strict validator now fails closed on the stronger robust
generalization claim:

- `VALIDATION_PLAN.md` refines the local validation plan against `GOALS.md`,
  `GQM.md`, and `REQUIREMENTS.md`.
- `validate.sh` writes `validation_summary.json` and returns non-zero while any
  P0 gate is unproven.
- `bfo/kernel.xlog` declares the shared BFO categories, relation families, and
  root-cause/intervention/explanation rules.
- `domains/domain_inventory.json` defines five thin domain adapters and the
  held-out domain protocol.
- `tools/run_neural_smoke.py` invokes a real CUDA PyTorch model through XLOG
  `nn/4` and proves ranking impact for the local smoke.
- `tools/run_bfo_fixture_smoke.py` executes the shared BFO rules over all five
  domain fixtures through `pyxlog.LogicProgram`.
- `tools/run_ablation_smoke.py` records neural-only, domain-symbolic,
  shared-symbolic, and neuro-symbolic ablations for the held-out fixture and
  measures uplift over the strongest baseline.
- `tools/run_runtime_contract_smoke.py` records relation-delta equivalence,
  cached hot-loop transfer counters, and five-run byte-identical replay.
- `programs/production_ranker.xlog` declares the production neural observation
  predicate used by the transfer run.
- `tools/run_production_transfer.py` writes
  `evidence/production_transfer.json` with five-domain zero-shot holdout
  transfer, real Hugging Face source provenance, externally supplied RCA/fault
  truth fields, per-case predictions, confusion inputs, computed ablations,
  production scale/profile metrics, v0.8.0/v0.8.5/v0.8.6 bundle reuse probes,
  and 30-minute soak evidence.
- `programs/dilp_proof_paths.xlog` and `dilp_report` add example-level DILP
  evidence: XLOG proof-path clauses, CUDA joint neural/symbolic rule-weight
  training, learned rule inventories, clause ablations, proof gradients, and
  held-out-safe rule induction.
- Robust generalization gates `GEN-001` through `GEN-010` require
  leave-one-domain-out evaluation over every domain, at least 100 held-out HF
  cases per domain, frozen model/rule manifests, unseen dataset-family
  transfer, strong baselines, confidence intervals, and adversarial
  domain-shift variants.
- `validation_summary.json` is the strict gate output written by
  `./validate.sh --strict --gpu-required`.

Smoke runners remain intentionally separate from production evidence. They prove
local surfaces, but only `evidence/production_transfer.json` with `scope:
"production"` can close `TRANSFER-002`, `TRANSFER-003`, `NEURAL-002`,
`PERF-001`, `PERF-002`, and GQM `Q5`-`Q8`/`Q12`.

The current production evidence should be read as a production-grade
XLOG/CUDA neuro-symbolic cross-domain ranking and transfer demo. It closes the
corrected robust-generalization contract when every `GEN-*` gate passes. It now
also carries example-level DILP evidence when every `DILP-*` gate passes:
XLOG proof-path clauses feed CUDA joint neural/symbolic rule-weight training,
learned inventories, clause ablations, proof gradients, and held-out-safe rule
  induction. v0.8.9 adds reusable XLOG/pyxlog surfaces for the six UCR issue
  ledger gaps; broader full-language DILP productization remains outside this
  example.

## Production Goal

Build a domain-transfer reasoner that separates domain adapters from upper
ontology structure. With no per-domain BFO core edits, the finished system must:

1. Ingest at least five domains through thin adapters.
2. Infer root causes, failure chains, risk states, interventions, and
   explanations.
3. Explain cross-domain analogies through shared BFO categories and relations.
4. Hold out at least one domain during rule evolution.
5. Validate zero-shot or few-shot transfer on the held-out domain.
6. Outperform domain-specific baselines on transfer consistency and
   explanation coverage.

## Local Artifacts

| File | Purpose |
| --- | --- |
| `GOALS.md` | GDSP goal tree, questions, and leaf artifacts. |
| `GQM.md` | Measurement goal, hypotheses, metrics, data collection, and analysis plan. |
| `REQUIREMENTS.md` | P0 production gates and evidence schema. |
| `WORKER_BRIEF.md` | Branch ownership, read-first sources, first implementation step, and completion standard. |
| `README.md` | Orientation and handoff entrypoint. |

## Required Deliverables

This directory contains:

- A stable BFO kernel shared by every domain.
- At least five thin domain adapters containing domain facts and mappings only.
- Root-cause, failure-chain, risk-state, intervention, and explanation rules.
- A real CUDA PyTorch neural component invoked through XLOG `nn/4`.
- A DILP evidence path that learns symbolic rule weights over XLOG proof-path
  clauses jointly with the neural predicate.
- Real Hugging Face dataset rows and externally supplied RCA/fault truth fields
  for every production-transfer domain.
- Neural-only, domain-symbolic, shared-symbolic, and neuro-symbolic baselines.
- A holdout protocol proving zero-shot or few-shot transfer.
- A core mutation audit proving the BFO kernel checksum is identical across
  domains.
- Executable reuse evidence for the merged v0.8.0 session bridge, v0.8.5
  language contract, and v0.8.6 runtime optimizer/session APIs.
- `validate.sh`, where `./validate.sh --strict --gpu-required` is the
  authoritative gate.
- `validation_summary.json` with raw metrics, evidence paths, PASS/FAIL for
  every P0 gate, and exact blockers for unsupported requirements.
- `generalization_report` plus raw generalization prediction records that close
  `GEN-001` through `GEN-010` before making a robust-generalization claim.
- `XLOG_FINDINGS.md`, `xlog_issue_ledger.json`, `proposed_fixes.md`, and
  `repro/` so the project reports what it revealed about XLOG, which runtime or
  pyxlog features are weak, and what upstream regression tests should be added.

## Validation Contract

Strict validation must fail closed. Unsupported runtime capabilities, missing
CUDA/PyTorch/`pyxlog`/`nn/4` integration, CPU-only semantic evaluation, mocked
neural output, two-domain toy transfer, per-domain core edits, or hidden host
hot loops are production failures with evidence.

The target acceptance thresholds are defined in `REQUIREMENTS.md`. Load-bearing
requirements include:

- At least 12 BFO upper categories and 8 BFO relation families.
- Adapter/core rule ratio `<= 0.25` per domain.
- Held-out root-cause F1 `>= 0.90`.
- Accepted intervention precision `>= 0.95`.
- Neuro-symbolic uplift `>= 15%` over the strongest baseline.
- Fixed-seed byte-identical results across 5 runs.
- Data-plane D2H transfers in the hot loop: `0`.
- Data-plane H2D transfers after initial load: `0`.
- p95 core indexed query latency `<= 50 ms`.

## Scope Boundary

This branch owns primarily:

```text
examples/BFO/universal_case_reasoner/
```

Do not edit shared BFO libraries, repo-wide runtime code, common helper
directories, or sibling `examples/BFO/*` projects unless a separate instruction
explicitly authorizes that scope. The v0.8.6 bundle-reuse path also depends on
the repo-level pyxlog kernel staging script; fixes there are limited to making
the merged runtime bundle reproducible for this production evidence path.

## Production Evidence

Regenerate the production evidence with:

```bash
python tools/run_production_transfer.py \
  --output evidence/production_transfer.json \
  --latency-samples 30
./validate.sh --strict --gpu-required
```

The production evidence must record the held-out `cybersecurity_intrusion`
domain, all five Hugging Face dataset IDs, per-domain prediction records,
held-out confusion counts, intervention predictions, explanation validity,
invalid fixture rejection records, computed baseline/uplift metrics, and bundle
reuse details. Summary-only metric fields are not accepted by the validator.
Robust generalization additionally requires raw leave-one-domain-out records for
every domain and cannot be closed by the single configured showcase holdout.

## Non-Negotiables

- No per-domain edits to the BFO core.
- No two-domain toy transfer.
- No mocked neural output.
- No CPU-only semantic evaluator.
- No hidden host hot loops.
- No edits outside `examples/BFO/universal_case_reasoner/` without separate
  approval.
