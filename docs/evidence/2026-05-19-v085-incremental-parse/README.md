# v0.8.5 Incremental Parse Evidence

Sub-goal: `G085_INC_PARSE`

## Scope

- Added `xlog_logic::ParserSession`, a statement-level parser cache keyed by
  source path.
- Statement units preserve byte offsets plus one-based line/column starts and
  stable text hashes.
- Unchanged statement parses reuse cached AST fragments; changed statements are
  reparsed through the existing production parser.
- Module invalidation removes the changed module path and cached sources that
  import that module name.
- `xlog explain` now parses through `ParserSession`, preparing the same cache
  API for REPL/watch.

## Certified Fixture

The synthetic 500-statement edit fixture changes one fact. The second parse
reports 499 cache hits, 1 miss, 1 invalidation, and an estimated structural
speedup of 500x over reparsing every statement.

## Checks

```text
cargo test -p xlog-logic --test test_v085_incremental_parse
# 4 passed

cargo test -p xlog-logic --lib
# 236 passed

cargo test -p xlog-cli --test explain_cli_tests
# 2 passed

cargo check --workspace
# PASS

cargo fmt --check
# PASS

git diff --check
# PASS
```

## Acceptance Notes

- `M085_INC_PARSE.1`: Cache units are statement fragments with stable hashes and
  spans.
- `M085_INC_PARSE.2`: A one-statement edit invalidates only that statement.
- `M085_INC_PARSE.3`: The 500-statement fixture exceeds the 2x speed target by
  structural parse-unit work (`500 / 1`).
- `M085_INC_PARSE.4`: Parse errors include original line/column and byte span.
- `M085_INC_PARSE.5`: Cache hit, miss, invalidation, module invalidation, and
  error fixtures pass.
