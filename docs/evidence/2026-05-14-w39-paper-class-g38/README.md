# W3.9 Paper-Class Harness Evidence

**Goal document:** `docs/plans/2026-05-14-supervisor-goal-038.md`  
**Sub-goal:** G_W39 / W3.9 production-scale harness  
**Branch context:** `bench-spike/w35-line6-fanout-g38` with G_W35 and G_W36 graceful flags

## Scope

This slice adds the W39 harness surface requested by S_W39.1-S_W39.5:

| Requirement | Artifact |
|---|---|
| `crates/xlog-integration/benches/wcoj_paper_class.rs` | Added. |
| `crates/xlog-integration/benches/fixtures/paper_class.rs` | Added. |
| `call_graph_edge_analog(scale)` | Added. |
| `andersen_analog(scale)` | Added. |
| `ddisasm_analog(scale)` | Added. |
| `add_fixture_module(module_path)` API | Added as `FixtureRegistry::add_fixture_module`. |
| G_W35 flag | `GRACEFUL`. |
| G_W36 flag | `GRACEFUL`. |

The harness compares provider-direct binary hash-chain execution against provider-direct WCOJ execution, checks row-set equality for every fixture, reports direct 10-trial timing, Criterion timing, fixture path flags, peak VRAM, and recursive VRAM growth for the recursive fixture.

## Commands

```text
cargo fmt --all
EXIT 0

cargo test -p xlog-integration --bench wcoj_paper_class --no-run
EXIT 0

cargo bench -p xlog-integration --bench wcoj_paper_class -- --output-format bencher
EXIT 0
```

## Measurements

Final measurements are in `measurements.tsv`.

| Fixture | Row Equality | Bundle Paths | Direct Ratio | Criterion Ratio | CV Status | VRAM | Verdict |
|---|---|---:|---:|---:|---|---:|---|
| `call_graph_edge_analog` | PASS, 342 rows | 3/5 | 2.286666x | 2.030334x | RED, CV > 5% | 148,532 | RED |
| `andersen_analog` | PASS, 308 rows | 3/5 | 2.397090x | 1.767575x | RED, CV > 5% | 148,532 | RED |
| `ddisasm_analog` | PASS, 575 rows | 3/5 | 1.845573x | 1.589231x | RED, CV > 5% | 148,532 | RED |

Geometric mean direct ratio: `2.162749x`, below the `5.0x` gate.

## Verdict

G_W39 is not accepted in this branch state.

Passing items:

- M_W39.1: three named fixtures exist.
- M_W39.3: ratios are reported for every fixture.
- M_W39.5: row-set equality passed for every fixture.
- M_W39.7: fixture-module API exists.
- M_W39.8: peak VRAM is below 38 GB.
- M_W39.9: recursive fixture reports 0% per-iteration growth.

Blocking items:

- M_W39.2: bundle path coverage is only 3/5. G1 metadata is invoked, and G_W35/G_W36 are graceful flags, but G4 helper-split and G5 stream-mux are not integrated in this worktree.
- M_W39.4: geometric mean speedup is 2.162749x, below the 5.0x gate.
- M_W39.6: reproducibility CV is above 5%.

Do not advance to G_INT from this W39 result. The next required action is an integration branch/worktree that brings in the G4/G5 predecessor branches and reruns W39 there.
