# v0.8.5 Showcase Examples — MC Engine-Contract Re-Validation (2026-06-10)

Re-run of `scripts/validate_v085_examples.py` after the MC fail-closed /
engine-label change:

- The GPU-resident MC engine rejects negation/aggregates with a typed
  `ResidentRejection`; the CPU oracle is now explicit opt-in
  (`--allow-cpu-oracle`) and results carry `mc_engine`.
- Showcase examples `06_prob_aggregate_mc` and `08_approx_confidence` (MC
  aggregates) were re-pinned to declare the oracle explicitly: their
  `expected.json` now passes `extra_args: ["--allow-cpu-oracle"]` and asserts
  `mc_engine: "cpu-oracle"`. Probabilities/CI semantics are unchanged.
- All 10 examples: PASS (overall PASS). Binary: prebuilt
  `target/release/xlog` (host-io), commit of record in the session changeset.

The original 2026-05-19 closure evidence is preserved untouched at
`docs/evidence/2026-05-19-v085-examples/`; that run predates the engine
label and exercised examples 06/08 through the then-silent CPU fallback.

Artifact: `validation_summary.json` (this directory).
