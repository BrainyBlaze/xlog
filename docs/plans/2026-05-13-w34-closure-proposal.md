# W3.4 Closure Proposal

## 1. Status & Predecessor Chain

This proposal stages W3.4 from `OPEN` to `DONE` for user approval. It is not a merge, push, or tag request.

Predecessor chain:

| step | branch / commit | status | purpose |
|---|---|---|---|
| G30 | `main @ 7bb56e4d` | on main | Path C bundle expansion and closure-board baseline. |
| G31 | `bench-spike/w34-kernel-fusion @ 0276fd8d` | unmerged evidence branch | Layout+count fusion spike selected candidate A; `superhub-50K` row equality PASS and 1.491x speedup. |
| G32 | `feat/w34-kernel-fusion-impl @ 70d2cf5e` | unmerged implementation branch | Production threshold dispatch, env override, cert grid, bench evidence. |
| G33 | `feat/w34-closure-proposal-iteration-1` | this proposal branch | Closure proposal plus staged board edit; final commit hash reported in the REVIEW REQUEST. |

W3.4 acceptance contract from `docs/v065-closure-board.md @ 7bb56e4d`:

> Bench: fused kernel shows **>= 1.3x speedup** vs. 2-kernel sequence on a fixture where materialize is the long pole; deterministic. No regression on a small fixture where fusion penalty exceeds savings (must auto-disable below threshold).

## 2. Acceptance Evidence

### Performance Gate

G32 live threshold dispatch on `superhub-50K` routes to the fused layout+count path and preserves the W3.4 performance gate.

| fixture | baseline median | baseline 95% CI | routed median | routed 95% CI | delta | delta % | ratio | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `superhub-50K` | 1.806765 ms | [1.798958, 1.814529] ms | 1.136455 ms | [1.133343, 1.140128] ms | -670.310 us | -37.10% | 1.590x | PASS >= 1.3x |

Source: `docs/evidence/2026-05-13-w34-kernel-fusion-impl/README.md`.

### Auto-Disable Gate

G32 live threshold dispatch on `superhub-1K` keeps the small fixture below threshold (`2730 < 4096`) and routes to the unfused path. The no-small-fixture-regression gate remains preserved.

| fixture | total input rows | selected route | fused counter | unfused counter | baseline median | routed median | ratio | verdict |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `superhub-1K` | 2,730 | unfused | 0 | advances | 1.138559 ms | 0.872198 ms | 1.305x | PASS >= 0.95x |

Routing is pinned by `CudaKernelProvider::wcoj_triangle_fused_dispatch_count()` and `CudaKernelProvider::wcoj_triangle_unfused_dispatch_count()`.

### Cert Grid

`cargo test -p xlog-cuda-tests --test test_wcoj_w34_fusion --release` passes 5/5:

| cert | assertion | result |
|---|---|---:|
| A | Above-threshold fixture routes to fused and matches reference rows. | PASS |
| B | Below-threshold fixture routes to unfused and matches reference rows. | PASS |
| C1 | `XLOG_WCOJ_W34_THRESHOLD=u32::MAX` forces unfused on a large fixture. | PASS |
| C2 | `XLOG_WCOJ_W34_THRESHOLD=0` forces fused on a small fixture and preserves rows. | PASS |
| D/E | Default threshold is `4096`; W2.5 default cost model remains `Cardinality`. | PASS |

### Prior Regression Sweep

The G32 prior sweep covered the existing W2.5/W2.4/W4-recursive/W3.1/W3.2 surfaces:

| surface | command / test files | result |
|---|---|---:|
| W2.5 + W2.4 + recursive dispatch | `xlog-integration`: `test_wcoj_cardinality_cost_model`, `test_wcoj_record_join_result_feedback`, `test_wcoj_recursive_dispatch` | EXIT 0; 7/0 + 3/0 + 8/0 |
| W3.1 + W3.2 K5/K6 | `xlog-cuda`: `test_wcoj_layout_u32`, `test_wcoj_clique5`, `test_wcoj_clique6` | EXIT 0; 9/0 + 3/0 + 3/0 |

## 3. Implementation Summary

G32 production implementation file summary:

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

Threshold value: `W34_FUSION_THRESHOLD = 4096`.

Env override: `XLOG_WCOJ_W34_THRESHOLD`, parsed as `u32`; invalid values fall back to the const.

Production fork scope: canonical 4-byte triangle WCOJ with default `var_order` only. U64 paths, rotated/non-default leader paths, 4-cycle, K5, and K6 remain on their existing pipelines.

## 4. Scope Discipline

This closure proposal closes W3.4 only: production layout+count fusion with threshold auto-disable and the cert grid that pins routing and correctness.

Closing W3.4 does not close W3.3. W3.3 remains `OPEN` under its own `>= 2.0x` full-bundle wall-time gate.

Closing W3.4 does not promise fused dispatch for non-triangle shapes, U64 inputs, rotated leaders, non-default `var_order`, 4-cycle, K5, or K6. Those surfaces remain owned by their existing board items or future W3.5/W3.6/G37 bundle work.

## 5. Final Gates

Final gates were rerun on the G33 proposal branch HEAD after adding only the closure proposal and board edit:

| command | result |
|---|---:|
| `cargo fmt --check --all` | EXIT 0 |
| `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` | EXIT 0 |
| `cargo test -p xlog-cuda-tests --test certification_suite --release` | EXIT 0; 1/0 |
| `cargo test -p xlog-cuda-tests --test test_wcoj_w34_fusion --release` | EXIT 0; 5/0 |
| `cargo test --release -p xlog-integration --test test_wcoj_cardinality_cost_model --test test_wcoj_record_join_result_feedback --test test_wcoj_recursive_dispatch` | EXIT 0; 7/0 + 3/0 + 8/0 |
| `cargo test --release -p xlog-cuda --test test_wcoj_layout_u32 --test test_wcoj_clique5 --test test_wcoj_clique6` | EXIT 0; 9/0 + 3/0 + 3/0 |

## 6. Branch State

Expected post-commit branch state:

| check | expected result |
|---|---:|
| `git rev-list --count main..HEAD` | 2 |
| `git diff feat/w34-kernel-fusion-impl..HEAD -- crates/` | byte-empty |
| `git merge-base --is-ancestor HEAD main` | false |
| `git tag --points-at HEAD` | empty |
| `git ls-remote --heads origin "feat/w34*"` | empty |

No FF-merge, push, or tag is performed by this proposal.

## 7. Response Options

**Response 1 - Accept as DONE (recommended).** FF-merge `feat/w34-closure-proposal-iteration-1` to `main`; W3.4 is marked `DONE` on main. Tally on main becomes 14 DONE + 1 IN-PROGRESS + 11 OPEN = 26. G31 evidence, G32 implementation, and G33 closure proposal remain in the linear history.

**Response 2 - Reject closure.** W3.4 stays `OPEN`; user specifies revised acceptance or evidence requirements.

**Response 3 - Defer closure.** W3.4 stays `OPEN`; closure is deferred to a later iteration with a stated reason.
