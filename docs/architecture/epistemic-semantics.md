# Epistemic Semantics And EIR

This document records the v0.9.0 Epistemic Intermediate Representation (EIR)
boundary. EIR exists so epistemic constructs stay explicit until a semantic
mode can evaluate them; they must not be hidden as ordinary predicate rewrites.

## Source Surface

The initial frontend surface is intentionally small:

- `#pragma epistemic_mode = faeel`
- `#pragma epistemic_mode = g91`
- `know atom(...)`
- `possible atom(...)`
- `not know atom(...)`
- `not possible atom(...)`

`faeel` is the default mode when no pragma is present. `g91` is an explicit
compatibility mode. Nested epistemic operators such as `know possible p(X)` are
recognized as unsupported epistemic constructs and return a typed diagnostic.

## Frontend Representation

`crates/xlog-logic/src/ast.rs` represents epistemic constructs explicitly:

- `EpistemicMode` stores the selected semantics mode in `Directives`.
- `EpistemicOp` stores `know` versus `possible`.
- `EpistemicLiteral` stores operator, explicit negation, and the atom under the
  operator.
- `BodyLiteral::Epistemic` keeps epistemic literals separate from ordinary
  positive and negated atoms.

## EIR Boundary

`crates/xlog-ir/src/eir.rs` defines the crate-level EIR boundary:

- `EirProgram`
- `EirRule`
- `EirBodyLiteral`
- `EirEpistemicLiteral`
- `EirEpistemicMode`
- `EirEpistemicOp`

`xlog_logic::build_eir` converts parsed AST to EIR without lowering to RIR. This
is the required entry point for future G91, FAEEL, Generate-Propagate-Test, and
epistemic splitting work.

## Lowering Boundary

Current RIR lowering rejects `BodyLiteral::Epistemic` with
`XlogError::UnsupportedEpistemicConstruct { construct: "RIR lowering boundary",
... }`.

That rejection is intentional for `G090_EIR`: until G91/FAEEL execution exists,
epistemic programs must flow through EIR-specific planning rather than the
stable Datalog lowering path. Non-epistemic programs continue using the existing
parser, stratifier, RIR lowering, runtime, and probabilistic paths.

The probabilistic WFS/provenance code also rejects epistemic literals with
typed `UnsupportedEpistemicConstruct` errors. `G090_PROB` owns the later
semantic contract for combining epistemic assumptions with probabilistic
queries and circuit updates.

## G91 Compatibility Fixture Semantics

`crates/xlog-logic/src/epistemic.rs` contains the current bounded fixture
evaluator for mode-selection tests. It is not the full production epistemic
executor. It exists to make the G91 compatibility mode testable before the
later FAEEL and Generate-Propagate-Test sub-goals land.

The fixture evaluator uses an `EpistemicInterpretation` with two predicate/arity
sets:

- `known`: facts known in both modes;
- `possible`: compatibility-only possible facts.

For `know p(...)`, both G91 and FAEEL require `p/arity` to be in `known`.
For `possible p(...)`, G91 accepts either `known` or `possible`; FAEEL accepts
only `known` in this bounded fixture layer. That gives a deterministic golden
distinction without routing epistemic programs through RIR.

Non-epistemic programs remain isolated from mode selection. A program with no
`BodyLiteral::Epistemic` compiles to the same RIR plan under the default mode
and under `#pragma epistemic_mode = g91`.
