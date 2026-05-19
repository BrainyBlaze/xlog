# v0.8.5 CLI Evidence

Sub-goal: `G085_CLI`

## Scope

- Added `xlog repl` and `xlog watch` command surfaces.
- `repl` reads multiline stdin, preserves the accumulated parser-session state,
  and reports statement/rule/query counts without requiring GPU access.
- `watch` supports `--once`, `--explain`, and `--debounce-ms`; the smoke path
  parses through the incremental session and emits typed errors through the
  normal CLI result path.
- Expanded `xlog explain --format json` with deterministic `parse`, `ast`,
  `stratification`, `rir`, `optimizer`, `wcoj`, `magic_sets`, `probability`, and
  `aggregate_lifting` sections.
- No new CLI dependency was added.

## Checks

```text
cargo test -p xlog-cli --test explain_cli_tests
# 2 passed

cargo test -p xlog-cli --test interactive_cli_tests
# 2 passed
```

## Acceptance Notes

- `M085_CLI.1`: `explain`, `repl`, and `watch` subcommands exist.
- `M085_CLI.2`: `explain` keeps text/json/dot formats under clap validation.
- `M085_CLI.3`: Explain JSON includes parse, AST, stratification, RIR,
  optimizer, WCOJ, magic-set, and probability sections.
- `M085_CLI.4`: REPL smoke parses a multiline fact/rule/query session.
- `M085_CLI.5`: Watch smoke runs `--once --explain` with parser-session output.
- `M085_CLI.6`: No new dependency was introduced.
