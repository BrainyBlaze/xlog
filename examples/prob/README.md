# Probabilistic Examples (`xlog-prob`)

These `.xlog` files exercise the Phase 4 probabilistic surface:

- probabilistic facts (`p::atom.`)
- annotated disjunctions (`p1::a1; p2::a2.`)
- evidence (`evidence(atom, true|false).`)
- probabilistic queries (`query(atom).`)

Run them via the Python bindings (`crates/xlog-gpu-py`) or from a small Rust driver using `xlog_prob::exact::ExactDdnnfProgram`.

