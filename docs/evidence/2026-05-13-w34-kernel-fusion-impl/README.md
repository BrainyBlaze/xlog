# W3.4 Kernel Fusion Production Implementation

## Status

G32 production implementation for W3.4 layout+count fusion.

- Parent SHA: `7bb56e4d`
- Branch: `feat/w34-kernel-fusion-impl`
- Branch HEAD SHA: single G32 implementation commit; immutable hash is reported in the REVIEW REQUEST because a single commit cannot contain its own final hash.
- Predecessor evidence: G31 `bench-spike/w34-kernel-fusion @ 0276fd8d`
- Chosen threshold: `W34_FUSION_THRESHOLD = 4096`
- Env override: `XLOG_WCOJ_W34_THRESHOLD`
- Verdict: W3.4 implementation is closure-ready for G33 closure proposal.

Raw Criterion output:

- `criterion_wcoj_fusion_bench.txt`

## Implementation Summary

| file | LOC | summary |
|---|---:|---|
| `crates/xlog-core/src/config.rs` | +16 | Added `W34_FUSION_THRESHOLD` and `ENV_WCOJ_W34_THRESHOLD`. |
| `crates/xlog-core/src/lib.rs` | +3/-1 | Re-exported the W3.4 threshold constants. |
| `crates/xlog-cuda/kernels/wcoj.cu` | +38 | Added `wcoj_triangle_fused_lc_count`, the layout+count fused count kernel. |
| `crates/xlog-cuda/src/kernel_manifest_data.rs` | +2 | Added the fused kernel to the WCOJ module manifest. |
| `crates/xlog-cuda/src/provider/mod.rs` | +45 | Added fused/unfused W3.4 routing counters and the fused kernel symbol. |
| `crates/xlog-cuda/src/provider/wcoj.rs` | +332 | Promoted `wcoj_triangle_fused_lc_u32_recorded` to production. |
| `crates/xlog-runtime/src/executor/wcoj_dispatch.rs` | +62/-3 | Added the row-count threshold fork and env override resolution. |
| `crates/xlog-integration/benches/wcoj_fusion_bench.rs` | +341 | Added production V3 bench for `superhub-1K` and `superhub-50K`. |
| `crates/xlog-cuda-tests/tests/test_wcoj_w34_fusion.rs` | new | Added the W3.4 routing/correctness cert grid. |

Dispatch metric: total logical input rows across the three canonical triangle slots. This intentionally stays a row-count proxy, matching G32; no CompilerConfig, CostModel, or broader dispatch policy was changed.

The production fork is limited to the canonical 4-byte triangle WCOJ path with no non-default `var_order`. U64 and rotated/non-default leader paths continue through the existing unfused WCOJ pipeline.

## Threshold Sweep

Bench cells:

- `superhub-1K`: total input rows `2730`, threshold-routed path is unfused for every sweep candidate.
- `superhub-50K`: total input rows `89609`, threshold-routed path is fused for every sweep candidate.

| threshold | 1K route | 1K ratio | 1K verdict | 50K route | 50K ratio | 50K verdict |
|---:|---|---:|---|---|---:|---|
| 4096 | unfused | 1.305x | PASS >= 0.95x | fused | 1.590x | PASS >= 1.3x |
| 8192 | unfused | 1.305x | PASS >= 0.95x | fused | 1.590x | PASS >= 1.3x |
| 16384 | unfused | 1.305x | PASS >= 0.95x | fused | 1.590x | PASS >= 1.3x |
| 32768 | unfused | 1.305x | PASS >= 0.95x | fused | 1.590x | PASS >= 1.3x |
| 65536 | unfused | 1.305x | PASS >= 0.95x | fused | 1.590x | PASS >= 1.3x |

Rationale: all candidate thresholds satisfy both gates on the measured cells. `4096` is selected because it is the smallest locked candidate that keeps the small fixture below threshold while maximizing eligibility for the G31-validated fused path.

## Final V3 Measurements

Criterion estimates are from `target/criterion/.../new/estimates.json`; raw console output is captured in `criterion_wcoj_fusion_bench.txt`.

| fixture | baseline median | baseline 95% CI | routed median | routed 95% CI | delta | delta % | ratio | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `superhub-1K` | 1.138559 ms | [1.086388, 1.454979] ms | 0.872198 ms | [0.866579, 0.876890] ms | -266.361 us | -23.40% | 1.305x | PASS no regression |
| `superhub-50K` | 1.806765 ms | [1.798958, 1.814529] ms | 1.136455 ms | [1.133343, 1.140128] ms | -670.310 us | -37.10% | 1.590x | PASS >= 1.3x |

Row equality before timing:

| fixture | verdict | output rows |
|---|---:|---:|
| `superhub-1K` | PASS | 82 |
| `superhub-50K` | PASS | 29,539 |

## Cert Grid

`cargo test -p xlog-cuda-tests --test test_wcoj_w34_fusion --release`:

| cert | assertion | result |
|---|---|---:|
| A | Above-threshold fixture routes to fused and matches reference rows. | PASS |
| B | Below-threshold fixture routes to unfused and matches reference rows. | PASS |
| C1 | `XLOG_WCOJ_W34_THRESHOLD=u32::MAX` forces unfused on a large fixture. | PASS |
| C2 | `XLOG_WCOJ_W34_THRESHOLD=0` forces fused on a small fixture and preserves rows. | PASS |
| D/E | Default threshold is `4096`; W2.5 default cost model remains `Cardinality`. | PASS |

Provider counters exposed:

- `CudaKernelProvider::wcoj_triangle_fused_dispatch_count()`
- `CudaKernelProvider::wcoj_triangle_unfused_dispatch_count()`

## Prior Regressions

The G32 plan listed the prior regression command under `xlog-cuda-tests`, but the named prior files are owned by `xlog-integration` and `xlog-cuda`. The actual owning-crate commands were run:

| command | result |
|---|---:|
| `cargo test --release -p xlog-integration --test test_wcoj_cardinality_cost_model --test test_wcoj_record_join_result_feedback --test test_wcoj_recursive_dispatch` | EXIT 0; 7/0 + 3/0 + 8/0 |
| `cargo test --release -p xlog-cuda --test test_wcoj_layout_u32 --test test_wcoj_clique5 --test test_wcoj_clique6` | EXIT 0; 9/0 + 3/0 + 3/0 |

## Final Gates

| command | result |
|---|---:|
| `cargo fmt --check --all` | EXIT 0 |
| `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` | EXIT 0 |
| `cargo test -p xlog-cuda-tests --test certification_suite --release` | EXIT 0; 1/0 |
| `cargo test -p xlog-cuda-tests --test test_wcoj_w34_fusion --release` | EXIT 0; 5/0 |
| `cargo bench --no-run --bench wcoj_fusion_bench` | EXIT 0 |

Scope checks:

- No `docs/v065-closure-board.md` edit.
- No closure proposal.
- No forbidden roadmap-version or rejected-pattern strings in the G32 diff.
- No push or tag performed.
- Branch remains unmerged pending G33 closure proposal.
