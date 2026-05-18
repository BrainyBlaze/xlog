# Epistemic Semantics Examples

These examples are v0.9.0 semantic fixtures. They are runnable through the
Rust fixture harness, not the production `xlog run` path, because direct RIR
lowering of epistemic literals is still intentionally rejected.

Run the examples:

```bash
cargo test -p xlog-logic --test test_epistemic_examples
```

| Example | Fixture path |
|---|---|
| EIR boundary | `01-eir-boundary.xlog` |
| G91 compatibility | `02-g91-compatibility.xlog` |
| FAEEL default | `03-faeel-default.xlog` |
| Generate-Propagate-Test | `04-gpt-candidate-filter.xlog` |
| Epistemic splitting | `05-splitting.xlog` |
