# v0.8.5 Examples Evidence

Sub-goal: `G085_EXAMPLES`

## Scope

- Added the validator-owned showcase suite under
  `examples/v085-language/showcase/`.
- Each numbered example has `program.xlog`, `expected.json`, and README notes.
- The suite covers finite types, lists, safe meta-predicates, NAF, magic sets,
  probabilistic aggregates, aggregate lifting, approximate inference pragmas,
  incremental parsing, and CLI explain/REPL/watch smoke paths.
- The scientific fixture uses engineering-style predicate names without
  domain-specific legacy terminology.

## Validation Command

```text
python3 scripts/validate_v085_examples.py \
  --output docs/evidence/2026-05-19-v085-examples/validation_summary.json
```

Fresh result on this branch: exit 0; `example_count=10`;
`interaction_count=10`; every `per_example[*].status` is `PASS`.

## Acceptance Notes

- `M085_EXAMPLES.1`: at least 10 advanced examples are present.
- `M085_EXAMPLES.2`: feature coverage is recorded in `feature_coverage`.
- `M085_EXAMPLES.3`: at least 5 examples combine two or more new features.
- `M085_EXAMPLES.4`: the validator executes explain JSON for all examples and
  executes REPL/watch diagnostics for the CLI smoke example.
- `M085_EXAMPLES.5`: validation evidence is written to `validation_summary.json`.
