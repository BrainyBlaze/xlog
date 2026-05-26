# Epistemic Semantics Examples

These examples are v0.9.0 production `xlog run` pilots and semantic fixtures.
The high-level GPU runner detects accepted epistemic programs and routes them
through the epistemic GPU runtime; the lower-level direct RIR lowering boundary
still rejects raw epistemic literals with `UnsupportedEpistemicConstruct`.

Run the examples:

```bash
cargo run -q -p xlog-cli -- run examples/epistemic/01-eir-boundary.xlog
cargo run -q -p xlog-cli -- run examples/epistemic/02-g91-compatibility.xlog
cargo run -q -p xlog-cli -- run examples/epistemic/03-faeel-default.xlog
cargo run -q -p xlog-cli -- run examples/epistemic/04-gpt-candidate-filter.xlog
cargo run -q -p xlog-cli -- run examples/epistemic/05-splitting.xlog
cargo test -p xlog-logic --test test_epistemic_examples
cargo test -p xlog-cli --test run_cli_tests test_xlog_run_epistemic_examples
```

| Example | Fixture path |
|---|---|
| EIR boundary | `01-eir-boundary.xlog` |
| G91 compatibility | `02-g91-compatibility.xlog` |
| FAEEL default | `03-faeel-default.xlog` |
| Generate-Propagate-Test | `04-gpt-candidate-filter.xlog` |
| Epistemic splitting | `05-splitting.xlog` |
