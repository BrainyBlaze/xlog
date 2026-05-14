# G38 G_INT M_INT.1 Successor Evidence

**Goal document:** `docs/plans/2026-05-14-supervisor-goal-038.md`
**Sub-goal:** G_INT
**Metric:** corrected M_INT.1 W3.4 successor re-validation
**Branch:** `feat/w3-bundle-integration`
**HEAD:** `c733c667`
**Date:** 2026-05-14

## Contract Source

The main-checkout goal-038 document has a supervisor correction replacing the
missing `wcoj_w34_kernel_fusion` target with the successor
`wcoj_w33_superhub` bench on the W3.4-canonical `superhub-50K` fixture.

Corrected target:

```text
cargo bench -p xlog-integration --bench wcoj_w33_superhub
```

Gate:

```text
superhub-50K ratio >= 1.51x
```

Rationale: the original W3.4 fused-count surface and old
`wcoj_fusion_bench.rs` were retired by W33 commits `738ab6f2` and `0754a30d`.
The current successor surface is the HG triangle path exercised by
`wcoj_w33_superhub`.

## Commands

```text
cargo test -p xlog-integration --bench wcoj_w33_superhub --no-run
EXIT 0
```

```text
cargo bench -p xlog-integration --bench wcoj_w33_superhub -- --output-format bencher
EXIT 0
```

## Measurements

| Cell | Row equality | Public-provider route median | HG block-slice median | Ratio | Gate | Verdict |
|---|---|---:|---:|---:|---:|---|
| `uniform-50K` | PASS, 0 rows | 621,539 ns | 126,157 ns | 4.926710x | diagnostic | PASS |
| `superhub-50K` | PASS, 29,539 rows | 689,666 ns | 171,057 ns | 4.031791x | >= 1.51x | PASS |

Raw emitted lines:

```text
W33_ROW_EQUALITY uniform-50K PASS rows=0
W33_INPUT_ROWS uniform-50K total_input_rows=147642 total_work=849607 block_work_unit=1024
test wcoj_w33_superhub/public_provider_route/uniform-50K ... bench:      621539 ns/iter (+/- 36216)
test wcoj_w33_superhub/hg_block_slice/uniform-50K ... bench:      126157 ns/iter (+/- 11399)
W33_MEASURED_CELL uniform-50K rows=0
W33_ROW_EQUALITY superhub-50K PASS rows=29539
W33_INPUT_ROWS superhub-50K total_input_rows=89609 total_work=123172 block_work_unit=1024
test wcoj_w33_superhub/public_provider_route/superhub-50K ... bench:      689666 ns/iter (+/- 53675)
test wcoj_w33_superhub/hg_block_slice/superhub-50K ... bench:      171057 ns/iter (+/- 10447)
W33_MEASURED_CELL superhub-50K rows=29539
```

## Verdict

Corrected G_INT M_INT.1 passes.

The `superhub-50K` successor ratio is `4.031791x`, above the corrected
`>= 1.51x` gate. Per S_INT.3, G_INT may proceed to M_INT.2.
