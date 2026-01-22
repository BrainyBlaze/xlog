# Probabilistic Examples (`xlog-prob`)

These `.xlog` files exercise the Phase 4 probabilistic surface:

- probabilistic facts (`p::atom.`)
- annotated disjunctions (`p1::a1; p2::a2.`)
- evidence (`evidence(atom, true|false).`)
- probabilistic queries (`query(atom).`)
- explicit P3 opt-in for non-monotone recursion (`#pragma prob_engine = mc`)

Run them via the Python bindings (`crates/pyxlog`) or from a small Rust driver using:

- `xlog_prob::exact::ExactDdnnfProgram` (exact)
- `xlog_prob::mc::McProgram` (approximate, explicit opt-in via `prob_engine=mc`)
