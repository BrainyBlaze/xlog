# W3.9 Integrated Paper-Class Harness Evidence

**Goal document:** `docs/plans/2026-05-14-supervisor-goal-038.md`  
**Sub-goal:** G_W39 / W3.9 production-scale harness  
**Branch:** `feat/w3-bundle-integration`  
**Integrated predecessors:** `bench-spike/w35-line6-fanout-g38 @ c142ae62`, `feat/w37-helper-split-aot-g37`, `feat/w38-stream-mux-aot-g37`

## Scope

This rerun uses the integration branch after merging the G4 helper-split and G5 stream-mux predecessor branches. It supersedes the earlier W39 branch-context result where bundle-path coverage was only 3/5.

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
| `call_graph_edge_analog` | PASS, 342 rows | 5/5 | 2.326465x | 1.978058x | RED, CV > 5% | 148,532 | RED |
| `andersen_analog` | PASS, 308 rows | 5/5 | 2.422292x | 1.762747x | RED, CV > 5% | 148,532 | RED |
| `ddisasm_analog` | PASS, 575 rows | 5/5 | 2.019127x | 1.581565x | RED, CV > 5% | 148,532 | RED |

Geometric mean direct ratio: `2.249204x`, below the `5.0x` gate.

## Verdict

G_W39 remains RED after integration.

Passing items:

- M_W39.1: three named fixtures exist.
- M_W39.2: bundle path coverage is 5/5 in the integration branch.
- M_W39.3: ratios are reported for every fixture.
- M_W39.5: row-set equality passed for every fixture.
- M_W39.7: fixture-module API exists.
- M_W39.8: peak VRAM is below 38 GB.
- M_W39.9: recursive fixture reports 0% per-iteration growth.

Blocking items:

- M_W39.4: geometric mean speedup is 2.249204x, below the 5.0x gate.
- M_W39.6: reproducibility CV is above 5%.

Do not advance to G_INT from this W39 result.
