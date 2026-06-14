# Probabilistic Examples (`xlog-prob`)

These `.xlog` files exercise the probabilistic-programming surface:

- probabilistic facts (`p::atom.`)
- annotated disjunctions (`p1::a1; p2::a2.`)
- evidence (`evidence(atom, true|false).`)
- probabilistic queries (`query(atom).`)
- explicit Monte Carlo opt-in for non-monotone recursion (`#pragma prob_engine = mc`)

Run them via the Python bindings (`crates/pyxlog`) or from a small Rust driver using:

- `xlog_prob::exact::ExactDdnnfProgram` (exact)
- `xlog_prob::mc::McProgram` (approximate, explicit opt-in via `prob_engine=mc`)
