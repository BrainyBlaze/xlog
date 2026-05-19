# G086_CONSUMERS Evidence

## GDSP

- Consumer goal: certify that DTS-DLM, Mistaber-style neutral workloads,
  v0.9.0 epistemic/solver work, and public pyxlog users can consume the
  composed v0.8.6 feature set without private hooks or fixture-only paths.
- Existing subsystem reused: `.xlog` parser, `xlog-cli run`, `xlog-cli explain`,
  v0.8.0 DTS validator, v0.8.5 language validator, pyxlog source guards, and
  committed v0.8.6 runtime/provider evidence.
- Scope boundary: the new examples are ordinary `.xlog` programs. Runtime
  feature measurements are referenced from the production feature-node evidence
  rather than collected through a new helper engine. The validator now records
  this as a certification limit instead of treating feature declarations as
  direct behavior proof.

## GQM Questions

- Q086_CONSUMERS.1: DTS-DLM-shaped fixtures exercise delta, exact-pair, and
  optimizer-reuse paths.
- Q086_CONSUMERS.2: two neutral scientific/engineering fixtures run without
  project-specific terms in the `.xlog` programs.
- Q086_CONSUMERS.3: v0.9.0 substrate primitives are documented through exact,
  shared-memory, CSE, adaptive, and persistent-index feature coverage.
- Q086_CONSUMERS.4: public v0.8.0 and v0.8.5 example validators and source
  guards remain green under the local worktree pyxlog package.

## Commands

```bash
python scripts/validate_v086_examples.py
pytest -q python/tests/test_v086_consumers_source.py
```

## Raw Measurements

`validation_summary.json` records:

- five v0.8.6 example run/explain command lines, exit codes, and durations;
- `example_execution_status=PASS` and
  `consumer_certification_status=BLOCKED`;
- feature coverage source as `expected_json_declarations`;
- `feature_node_behavior_proofs.persistent_hash_index=PASS` with
  `speedup_ratio=3.206` from the runtime performance fixture;
- v0.8.0 compatibility validator output in
  `compat_v080_validation_summary.json`;
- v0.8.5 compatibility validator output in
  `compat_v085_validation_summary.json`;
- pyxlog v0.8.0/v0.8.5 source guard output;
- v0.8.6 feature-node transfer and performance evidence for delta, exact
  types, chain shared-memory scoring, CSE, adaptive re-optimization, and
  persistent hash indexes.

## Metric Interpretation

| Metric | Status | Evidence |
|---|---|---|
| M086_CONSUMERS.1 DTS-DLM | BLOCKED | `01_dts_delta_optimizer` passed run/explain, but exact-induction, adaptive, and persistent-index coverage remains declaration-plus-linked-evidence rather than direct consumer fixture behavior. Persistent-index now has direct feature-node performance proof, but not per-consumer fixture dispatch proof. |
| M086_CONSUMERS.2 Mistaber | PASS | `02_neutral_material_flow` and `03_neutral_signal_diagnostics` passed and contain no `mistaber` term in program source. |
| M086_CONSUMERS.3 v0.9.0 substrate | BLOCKED | `04_v090_substrate_primitives` passed run/explain, but exact, chain shared-memory, adaptive, and persistent-index coverage is label-derived and linked to feature evidence rather than directly executed by the `.xlog` fixture. Persistent-index now has direct feature-node performance proof, but not v0.9.0 fixture dispatch proof. |
| M086_CONSUMERS.4 pyxlog compatibility | PASS | v0.8.0 and v0.8.5 validators plus their source guards passed. |
| M086_CONSUMERS.5 production path | PASS | Validator runs examples through `xlog-cli run/explain`; no private hooks or fixture-only bypass are used. |
| M086_CONSUMERS.6 reuse audit | PASS | Summary names reused subsystems and committed feature evidence paths; no duplicate engine/helper path is introduced. |

## Open Gaps

- Feature coverage is label-derived from `expected.json`. The examples prove
  CLI parser/RIR/run/explain behavior and link to feature-node evidence, but
  they do not directly prove each declared runtime feature inside each consumer
  fixture.
- Public pyxlog session compatibility remains green, but persistent hash-index
  reuse across pyxlog session mutation and reevaluation is not directly proven.
