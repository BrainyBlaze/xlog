# G_W39_DTSDLM Evidence

**Goal:** Goal-039 G_W39_DTSDLM, Phase-1 G_W39 harness extension.

**Worktree:** `.worktrees/g39-w6-bundle-integration`

**Branch:** `feat/w6-bundle-integration-g39`

## Scope

This slice adds the DTS-DLM analog fixture to the paper-class WCOJ harness:

- `crates/xlog-integration/benches/fixtures/dts_dlm_analog.rs`
- `crates/xlog-integration/benches/fixtures/paper_class.rs`
- `crates/xlog-integration/tests/test_w39_dts_dlm_analog_fixture.rs`

The analog is registered through the fixture registry, marks itself recursive, and
exports fixture-owned bundle-path metadata:

```text
g1_metadata=PASS g2_branch=GRACEFUL g3_branch=GRACEFUL g4_helper_split=PASS g5_stream_mux=PASS g_w63_chain_promoter=PASS g_w66_cuda_graph=PASS invoked=7/7
```

The benchmark driver now consumes fixture-owned bundle metadata and the registry
expected fixture count. There is no DTS-DLM-specific benchmark-driver branch.
This is a one-time generic pluggability hardening of the pre-existing driver
assumption that the fixture set always had exactly three entries.

## Verification Commands

```text
cargo fmt --check
cargo test -p xlog-integration --test test_w39_dts_dlm_analog_fixture -- --nocapture
cargo test -p xlog-integration --bench wcoj_paper_class --no-run
cargo bench -p xlog-integration --bench wcoj_paper_class -- --output-format bencher
```

Full benchmark log captured at:

```text
/tmp/g39_w39d_wcoj_paper_class_fixture_metadata.log
```

## Raw Results

| Fixture | Rows | Bundle paths | Hash mean ns | WCOJ mean ns | Speedup | Hash CV | WCOJ CV | Peak bytes | cudaMemGetInfo peak delta | Recursive growth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| call_graph_edge_analog | 4194304 | 5/5 | 91881588.500 | 3101673.500 | 29.623230 | 0.019385 | 0.014141 | 2286157868 | 2317352960 | NA |
| andersen_analog | 4194304 | 5/5 | 91214141.500 | 3383475.400 | 26.958713 | 0.026342 | 0.020770 | 2286157868 | 2283798528 | NA |
| ddisasm_analog | 4194304 | 5/5 | 95620794.800 | 3321802.900 | 28.785812 | 0.025058 | 0.023277 | 2286157868 | 2283798528 | 0.000000 |
| dts_dlm_analog | 4194304 | 7/7 | 89421108.100 | 3361345.200 | 26.602774 | 0.018268 | 0.042162 | 2286157868 | 2283798528 | 0.000000 |

Geomean speedup across all four fixtures:

```text
W39_GEOMEAN ratio=27.964641 gate=5.000000 stretch=10.000000
```

VRAM gate:

```text
gate_bytes=40802189312
total_bytes=12820480000
```

## Metric Status

| Metric | Status | Evidence |
|---|---|---|
| M_W39D.1 DTS-DLM analog fixture committed | PASS | `dts_dlm_analog(scale)` exists and is registered. |
| M_W39D.2 Bundle paths | PASS | DTS fixture reports 7/7 with G2/G3 graceful flags and G_W66 CUDA Graph path. |
| M_W39D.3 DTS-DLM analog speedup | PASS | `26.602774x`, above 5.0x gate and 10.0x stretch. |
| M_W39D.4 Four-fixture geomean | PASS | `27.964641x`, above 5.0x gate and 10.0x stretch. |
| M_W39D.5 Determinism | PASS | Row-set equality PASS for all four fixtures. |
| M_W39D.6 Reproducibility | PASS | All hash and WCOJ CVs are <= 0.05 across 10 runs. |
| M_W39D.7 Pluggable API cleanliness | PASS_WITH_NOTE | DTS-DLM fixture data and path coverage live in fixture metadata; the driver has no DTS-specific branch. A generic driver hardening replaced the old hard-coded 3-fixture assumption. |
| M_W39D.8 Peak VRAM | PASS | Max observed allocation peak `2286157868` bytes and max `cudaMemGetInfo` delta `2317352960` bytes, below 38 GiB gate. |

## Acceptance

G_W39_DTSDLM is green for integration with the M_W39D.7 note above. The note is
intentional: the prior harness contained a hard-coded three-fixture assertion, so
supporting a fourth pluggable fixture required removing that generic assumption
while keeping DTS-DLM specifics inside fixture metadata.
