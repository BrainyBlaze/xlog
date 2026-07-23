# Errors and diagnostics

XLOG's typed error surface: XlogError variants, CLI exit codes, the fail-closed rejections you may hit, and how they map to Python exceptions.

XLOG fails closed: when a program asks for something the engine cannot run
soundly on the device, the engine stops with a typed error that names the rule
it violated. It does not silently fall back to a slower or semantically
different path. This page catalogs the error types, the exit codes, and the
specific rejections you are most likely to encounter.

## Error types

Every Rust-level error is a variant of a single enum, `XlogError` (defined in
`crates/xlog-core/src/error.rs`). The enum is marked non-exhaustive, so new
variants may be added without a breaking change.

| Variant | Meaning |
|---|---|
| `Parse` | Parse error from the Datalog frontend. |
| `StratificationCycle` | Stratification failed: a cycle through negation, reported with the predicates involved. |
| `UnsafeVariable` | Domain safety violation: a variable is not bound in any positive body literal. |
| `ResourceExhausted` | GPU memory budget exceeded; carries the operation context, the estimated bytes, and the budget. |
| `CompileCapacityExceeded` | The knowledge-compilation phase (the step, named D4, that turns the program into a compiled Boolean circuit) declined the input: the Boolean formula it was asked to compile — a formula in conjunctive normal form, or CNF — was too large to compile safely. Compiling it anyway would overrun the fixed-capacity output buffers and trigger a CUDA launch error that leaves the GPU unusable for the rest of the process. Means "too big to compile"; catchable; distinct from the verify-phase signal below. **(unreleased)** |
| `VerifyBudgetExceeded` | The GPU verifier that checks whether two formulas are equivalent declined. Its conflict-driven SAT search (CDCL) ran out of the per-verify conflict budget before reaching a definite answer. An indeterminate result is never trusted as a proof. **(unreleased)** |
| `Kernel` | GPU kernel launch or execution error. |
| `Type` | Type checking or inference error. |
| `Compilation` | Compilation pipeline error (also carries the resident MC rejections and the exact aggregate cap declines described below). |
| `UnsupportedEpistemicConstruct` | An epistemic construct known to the frontend but unsupported in the given context; names the construct and the context. |
| `Execution` | Runtime execution error. |

## Exit codes

The `xlog` CLI returns `0` on success and `1` on any error. Every failure — parse,
compile, execution, I/O, or an exhausted memory budget — surfaces as exit code `1`
with a descriptive message on stderr. There are no other exit codes.

## Fail-closed rejections you may hit

### Resident Monte Carlo rejection

The production Monte Carlo engine — which estimates probabilities by sampling
many random possible worlds — runs entirely on the GPU ("resident") within fixed
memory bounds. At compile time it checks every rule and fact against the model of
what the device can run; anything outside that model is rejected with a typed
`ResidentRejection` (surfaced as a `Compilation` error of the form
`resident MC engine rejected program [kind=...] construct=... context=...`).
There is no silent CPU fallback.

| Rejection kind | Trigger |
|---|---|
| `negation` | A body literal uses negation. |
| `epistemic_literal` | A body literal is epistemic (`know` / `possible`). |
| `non_relational_literal` | A body literal is a comparison, arithmetic, or `univ` (non-relational). |
| `arity_too_high` | A predicate arity exceeds the cap of 3. |
| `body_too_long` | A rule body has more than 3 literals. |
| `too_many_vars` | A rule uses more than 8 distinct variables. |
| `unbounded_term` | A term is not a variable or ground constant (list, compound, functor, aggregate). |
| `domain_too_large` | The bounded constant domain exceeds 256. |
| `universe_too_large` | The bounded atom universe exceeds 65536 slots. |
| `inconsistent_arity` | A predicate appears with inconsistent arity. |
| `annotated_disjunction_unsupported` | The program uses an annotated disjunction the resident engine cannot ground. |

<Note>
On the CLI, `xlog prob --allow-cpu-oracle` lets a rejected program run on a labeled
CPU oracle instead; the result is tagged `mc_engine: "cpu-oracle"` and is never
GPU-native evidence. Without the flag, a rejected program fails. See the
[CLI reference](/reference/cli).
</Note>

### Exact aggregate caps

The exact engine (`exact_ddnnf`) computes probabilities exactly rather than by
sampling. It evaluates aggregates over finite probabilistic domains using
dynamic programming, and that evaluation is capped per aggregate group:

- **Count-only aggregates**: at most **64** uncertain rows per group.
- **All other aggregates** (`sum`, `min`, `max`, `logsumexp`): at most **16**
  uncertain rows per group.

Rows whose provenance is deterministically true or false do not count against the
cap — only rows whose membership is genuinely uncertain do. Over the cap, the
compile fails with a `Compilation` error that names the predicate, the group key,
and the cap, and tells you the way out: `use prob_engine = mc or reduce the finite
aggregate domain`. The Monte Carlo engine has no such cap because it samples
worlds instead of enumerating outcome formulas. See
[Probabilistic engines](/probabilistic/engines) for choosing between the two.

### Compile and verify budgets (unreleased)

These two budgets let the engine refuse an oversized problem cleanly instead of
crashing the GPU. They currently live on the `main` branch and are not yet in a
published release. The knowledge-compilation phase (D4) and the verify phase
decline oversized instances rather than risk a CUDA launch failure that would
leave the GPU unusable for the rest of the process:

- A CNF (Boolean formula in conjunctive normal form) whose variable or clause
  capacity exceeds `XLOG_D4_VERIFY_MAX_VARS` / `XLOG_D4_VERIFY_MAX_CLAUSES`
  declines **before any kernel launch** with `CompileCapacityExceeded`. Both
  bounds default to unbounded.
- A verify whose SAT search exhausts `XLOG_D4_VERIFY_MAX_CONFLICTS` declines
  with `VerifyBudgetExceeded` — an indeterminate search result is never
  reported as a proof. The default budget of `0` means unlimited.

Both declines are catchable, and the caller can skip the query or fall back to
the approximate `mc` engine. See
[Environment variables](/reference/environment-variables) for the knobs.

<Warning>
A fail-closed decline is a diagnostic, not a result. It blocks an unsound or
context-poisoning execution and explains why; it does not mean the query was
answered.
</Warning>

## Python exceptions

`pyxlog` maps the Rust error surface onto standard Python exception types:

| Exception | Raised for |
|---|---|
| `ValueError` | Invalid parameters: an unknown `prob_engine` (expected `exact_ddnnf` or `mc`), an unknown `sampling_method` (expected `rejection` or `evidence_clamping`), `memory_mb=0`, row counts exceeding `u32` range, and neural input shape or value validation. |
| `MemoryError` | The per-call memory limit: when `memory_mb` is passed to an evaluation and the provider's allocated bytes already exceed it, the call raises before evaluating. |
| `RuntimeError` | Any `XlogError` propagated from the engine (parse, compilation, kernel, execution, resource exhaustion — including the resident MC rejection and aggregate cap messages above), and host-output calls on a build without the `host-io` feature. |
| `IlpConfigError` (a `ValueError` subclass) | `induce_exact(backend="python")` without `XLOG_ALLOW_PYTHON_ILP_REFERENCE=1`. The Python reference scorer exists only to cross-check results (parity-only) and is never a production path. |

## See also

- [CLI reference](/reference/cli) — subcommands, flags, and the exit-code contract
- [Probabilistic engines](/probabilistic/engines) — when to use `exact_ddnnf` vs `mc`
- [Environment variables](/reference/environment-variables) — budgets and kill switches
