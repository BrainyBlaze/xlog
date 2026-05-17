# W3.3 HG Block-Slice Production Evidence

Branch: `feat/w33-hg-block-slice-prod`
Base: `main` at `f62188b7`
Evidence HEAD: `0abfb358`
Goal slice: `G37/G1`

## Benchmark Gate

Command:

```bash
cargo bench -p xlog-integration --bench wcoj_w33_superhub -- --output-format bencher
```

Raw cells:

| Cell | Public provider route | HG block-slice | Ratio | Verdict |
|---|---:|---:|---:|---|
| `uniform-50K` | 768121 ns/iter | 215499 ns/iter | 3.565x HG | PASS M1.2, faster than baseline |
| `superhub-50K` | 837386 ns/iter | 151531 ns/iter | 5.526x HG | PASS M1.1 >= 2.0x |

Row equality:

| Cell | Result |
|---|---|
| `uniform-50K` | `W33_ROW_EQUALITY uniform-50K PASS rows=0` |
| `superhub-50K` | `W33_ROW_EQUALITY superhub-50K PASS rows=29539` |

## Metric Grid

| Metric | Evidence | Verdict |
|---|---|---|
| M1.1 super-hub speedup | `837386 / 151531 = 5.526x` | PASS |
| M1.2 uniform non-regression | `768121 / 215499 = 3.565x` HG-faster | PASS |
| M1.3 row-set equality | benchmark row equality before timing | PASS |
| M1.4 Algorithm-2 source audit | `cargo test -p xlog-cuda --test test_w33_hg_source_audit --release -- --nocapture` => 6/0 | PASS |
| M1.5 no per-call decision surface | source audit removed-symbol cert 6/0 plus removed-symbol scan outside audit returned no matches | PASS |
| M1.6 metadata storage | `cargo test -p xlog-cuda --test test_w33_hg_metadata_storage --release -- --nocapture` => 1/0 | PASS |
| M1.7 delta-outermost leader | `cargo test -p xlog-runtime delta_outermost_leader_selection --release -- --nocapture` => 2/0; full recursive cert 8/0 | PASS |

## Verification

Commands run after the S1.6/S1.7 commits:

```bash
cargo test -p xlog-runtime --lib --release -- --nocapture
cargo test -p xlog-integration --test test_w21_variable_ordering --release -- --nocapture
cargo test -p xlog-integration --test test_w26_heat_selectivity --release -- --nocapture
cargo test -p xlog-integration --test test_wcoj_recursive_dispatch --release -- --nocapture
cargo test -p xlog-cuda --test test_w33_hg_source_audit --release -- --nocapture
cargo test -p xlog-cuda --test test_w33_hg_metadata_storage --release -- --nocapture
cargo bench -p xlog-integration --bench wcoj_w33_superhub --no-run
cargo fmt --check --all && git diff --check
```

Observed counts:

| Command | Result |
|---|---|
| `xlog-runtime --lib` | 118/0 |
| `test_w21_variable_ordering` | 11/0 |
| `test_w26_heat_selectivity` | 7/0 |
| `test_wcoj_recursive_dispatch` | 8/0 |
| `test_w33_hg_source_audit` | 6/0 |
| `test_w33_hg_metadata_storage` | 1/0 |
| `wcoj_w33_superhub --no-run` | EXIT 0 |
| `fmt + diff check` | EXIT 0 |

Added-line process scan: no matches.
