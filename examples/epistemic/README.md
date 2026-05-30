# Epistemic Semantics Examples

These examples are production `xlog run` pilots and semantic fixtures. The
high-level GPU runner detects accepted epistemic programs and routes them through
the epistemic GPU runtime; the lower-level direct RIR lowering boundary still
rejects raw epistemic literals with `UnsupportedEpistemicConstruct`.

`01-05` are the v0.9.0 epistemic-surface pilots. `06-11` are the v0.9.1 epistemic
executor showcase: each demonstrates one completed bundle end-to-end through
`xlog run` and is validated with a deterministic output marker by
`test_xlog_run_epistemic_examples`.

Run the examples:

```bash
# v0.9.0 surface
cargo run -q -p xlog-cli -- run examples/epistemic/01-eir-boundary.xlog
cargo run -q -p xlog-cli -- run examples/epistemic/05-splitting.xlog
# v0.9.1 executor showcase
cargo run -q -p xlog-cli -- run examples/epistemic/06-eir-candidate-enumeration.xlog
cargo run -q -p xlog-cli -- run examples/epistemic/09-joint-multi-epistemic.xlog
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

Notes: each `06-11` file has a single query head — the single-plan path
materializes one output relation, so the showcase keeps one head per file.
Nested modal operators and the other goal-mandated fail-closed fragments
(see `docs/plans/2026-05-29-v091-epistemic-executor-completion-status.md`) are
intentionally rejected and therefore are not `xlog run` success pilots; they are
covered by negative pilots in the test suites.
