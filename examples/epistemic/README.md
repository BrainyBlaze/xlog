# Epistemic Semantics Examples

These examples are production `xlog run` pilots and semantic fixtures. The
high-level GPU runner detects accepted epistemic programs and routes them through
the epistemic GPU runtime; the lower-level direct RIR lowering boundary still
rejects raw epistemic literals with `UnsupportedEpistemicConstruct`.

`01-05` are the v0.9.0 epistemic-surface pilots. `06-13` are the v0.9.1 epistemic
executor showcase: each demonstrates one completed bundle end-to-end through
`xlog run` and is validated with a deterministic output marker by
`test_xlog_run_epistemic_examples` or a typed negative diagnostic test.

Run the examples:

```bash
# v0.9.0 surface
cargo run -q -p xlog-cli -- run examples/epistemic/01-eir-boundary.xlog
cargo run -q -p xlog-cli -- run examples/epistemic/05-splitting.xlog
# v0.9.1 executor showcase
cargo run -q -p xlog-cli -- run examples/epistemic/06-eir-candidate-enumeration.xlog
cargo run -q -p xlog-cli -- run examples/epistemic/09-joint-multi-epistemic.xlog
cargo run -q -p xlog-cli -- run examples/epistemic/12-bound-variable-splitting.xlog
# validate all of them through xlog run (requires CUDA):
XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-cli --test run_cli_tests test_xlog_run_epistemic_examples
```

| Example | Fixture path | Showcases |
|---|---|---|
| EIR boundary | `01-eir-boundary.xlog` | v0.9.0 EIR literal preservation |
| G91 compatibility | `02-g91-compatibility.xlog` | v0.9.0 G91 mode |
| FAEEL default | `03-faeel-default.xlog` | v0.9.0 FAEEL founded knowledge |
| Generate-Propagate-Test | `04-gpt-candidate-filter.xlog` | v0.9.0 GPT |
| Epistemic splitting | `05-splitting.xlog` | v0.9.0 independent components |
| Candidate enumeration | `06-eir-candidate-enumeration.xlog` | v0.9.1 EGB-01 EIR candidate enumeration + EGB-02 bound membership → `believed={1,3}` |
| Tuple-key membership | `07-tuple-key-membership.xlog` | v0.9.1 EGB-02 multi-column bound membership → `matched={(1,2),(3,3)}` |
| Repeated variable | `08-repeated-variable.xlog` | v0.9.1 EGB-02 repeated-variable equality → `reflexive={3}` |
| Joint multi-epistemic | `09-joint-multi-epistemic.xlog` | v0.9.1 EGB-06 joint modal conjunction → `both_known={1}` |
| Epistemic constraint | `10-epistemic-constraint.xlog` | v0.9.1 EGB-04 constraint prunes world view → `accepted` empty (Ok, not failure) |
| FAEEL foundedness | `11-faeel-foundedness.xlog` | v0.9.1 EGB-07 founded self-support → `founded={()}`|
| Bound-variable splitting | `12-bound-variable-splitting.xlog` | v0.9.1 EGB-05/EGB-06 split routing with bound modal membership → `both_known={1}`, `safe_alt={2}` |
| Nested modal rejection | `13-nested-modal-rejected.xlog` | v0.9.1 EGB-03 typed fail-closed diagnostic for `know possible p()` |
| Cross-component coupling (accepted) | `16-cross-component-coupling.xlog` | v0.9.2 Bundle 3 safe coupling: ordinary `report` consumes epistemic-derived `trusted`, coalesced single-output → `trusted={1,3}` |
| Cross-component coupling (rejected) | `17-cross-component-coupling-rejected.xlog` | v0.9.2 Bundle 3 typed fail-closed diagnostic: a modal literal over an epistemic-derived head couples two epistemic outputs (`cross-component epistemic coupling`, names `trusted`/`flagged` + `DerivedPredicate`) |

Notes: examples with one epistemic output head use the single-plan path; examples
with independent epistemic output heads route through split GPU execution from
`xlog run`. A coalesced component that nonetheless carries more than one epistemic
output head (cross-component modal coupling) is unsound to split and cannot be
jointly materialized, so it fails closed with a typed `cross-component epistemic
coupling` diagnostic naming the coupled heads and the merge reason. Nested modal
operators and the other goal-mandated fail-closed fragments (see
`docs/plans/2026-05-29-v091-epistemic-executor-completion-status.md`) are
intentionally rejected and covered by negative pilots.
