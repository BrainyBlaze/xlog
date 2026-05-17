# W3.9 Integrated Paper-Class Harness Evidence

**Goal document:** `docs/plans/2026-05-14-supervisor-goal-038.md`  
**Sub-goal:** G_W39 / W3.9 production-scale harness  
**Branch:** `feat/w3-bundle-integration`  
**Integrated predecessors:** `bench-spike/w35-line6-fanout-g38 @ c142ae62`, `feat/w37-helper-split-aot-g37`, `feat/w38-stream-mux-aot-g37`

## Scope

This rerun uses the integration branch after merging the G4 helper-split and G5 stream-mux predecessor branches. It supersedes the earlier W39 branch-context result where bundle-path coverage was only 3/5.

The rerun uses the tuned paper-class fixture shape in `wcoj_paper_class`: scale 1024, high-fanout paper-class middle keys, diagonal output bands, fresh direct-measurement providers per algorithm, and contiguous direct timing windows. The final output-band width makes the WCOJ path long enough for the strict CV gate while staying within the VRAM budget.

## Commands

```text
cargo fmt --all --check
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
| `call_graph_edge_analog` | PASS, 4194304 rows | 5/5 | 28.137633x | 27.674601x | PASS | 2,286,157,868 | PASS |
| `andersen_analog` | PASS, 4194304 rows | 5/5 | 28.185006x | 26.656617x | PASS | 2,286,157,868 | PASS |
| `ddisasm_analog` | PASS, 4194304 rows | 5/5 | 28.850891x | 27.619604x | PASS | 2,286,157,868 | PASS |

Geometric mean direct ratio: `28.389319x`, above the `5.0x` gate and `10.0x` stretch target.

## Verdict

G_W39 is PASS after integration.

Passing items:

- M_W39.1: three named fixtures exist.
- M_W39.2: bundle path coverage is 5/5 in the integration branch.
- M_W39.3: ratios are reported for every fixture.
- M_W39.4: geometric mean speedup is 28.389319x, above the 5.0x gate.
- M_W39.5: row-set equality passed for every fixture.
- M_W39.6: reproducibility CV is at or below 5% for hash and WCOJ samples on every fixture.
- M_W39.7: fixture-module API exists.
- M_W39.8: peak VRAM is below 38 GB.
- M_W39.9: recursive fixture reports 0% per-iteration growth.

Blocking items: none for G_W39.

G_INT may proceed from this W39 result.
