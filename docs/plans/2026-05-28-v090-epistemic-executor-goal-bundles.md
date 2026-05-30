# v0.9 Epistemic Executor Completion Goal Bundles

**Date:** 2026-05-28.
**Scope:** Completion bundles for the known bounded-executor limitations in the
v0.9 epistemic surface.
**Status:** Goal-definition artifact only. This document defines required
semantics, externally visible behavior, result obligations, and validation
evidence. It intentionally does not prescribe a specific algorithm, data
structure, kernel layout, or solver encoding.

## Purpose

XLOG v0.9.0-rc has a real epistemic language surface:

- `know atom(...)`
- `possible atom(...)`
- `not know atom(...)`
- `not possible atom(...)`
- `#pragma epistemic_mode = faeel`
- `#pragma epistemic_mode = g91`

The current implementation preserves these constructs explicitly in AST/EIR and
routes accepted bounded programs through `xlog run` into the GPU epistemic
runtime. The remaining work is to turn the bounded executor into a general
enough accepted executor for load-bearing epistemic logic programming while
preserving the existing fail-closed safety boundaries.

## Cross-Cutting Locks

These locks apply to every bundle below.

1. **No hidden CPU fallback.** CPU may parse, plan, orchestrate launches, format
   diagnostics, and transfer final requested results. Candidate enumeration,
   world-view validation, tuple membership, and accepted result materialization
   must not fall back to CPU in accepted runtime paths.
2. **No fake predicate rewriting.** `know` and `possible` remain first-class
   epistemic literals through EIR until the semantic boundary is proven.
3. **Direct raw RIR lowering remains a rejection boundary.** Removing this
   boundary is not a goal. Accepted programs must enter through the explicit
   epistemic execution route.
4. **No parallel side engines.** Accepted execution must reuse the existing
   runtime, relation layouts, WCOJ/planner surfaces, solver services, and
   probabilistic evidence paths where those surfaces apply.
5. **Typed fail-closed behavior remains required.** Inputs outside the accepted
   semantic fragment must produce typed diagnostics with source context and a
   reason, not silent partial execution.
6. **Semantic oracle tests are not release evidence.** CPU or fixture oracle
   tests may define expected semantics, but release evidence must include real
   runtime pilots over the production path.

## Bundle EGB-01: Arbitrary EIR Candidate-World Enumeration

### Current Limitation

Generate-Propagate-Test accepts explicit candidate lists in the bounded fixture
layer. The current runtime can execute certified bounded pilots, but it does not
yet enumerate candidate worlds from arbitrary EIR programs.

### Semantic Goal

For an accepted EIR program, XLOG must derive the candidate epistemic
assumption space from the program itself, then evaluate candidates against the
program's stable-model/world-view semantics.

The observable semantics are:

1. Generate candidate assignments for the epistemic literals that affect the
   program.
2. Propagate each candidate into a reduced ordinary program.
3. Evaluate the stable models of each reduced program through the production
   runtime path.
4. Test each candidate against the original epistemic assumptions.
5. Accept exactly the candidates whose world views satisfy the selected mode.

### Expected Accepted Behavior

- Programs with multiple epistemic literals no longer require hand-supplied
  candidate lists.
- Candidate counts, rejected counts, accepted counts, and rejection reasons are
  deterministic for deterministic inputs.
- Empty accepted-world-view results are distinguishable from execution failure.
- Resource exhaustion is reported before unsafe partial execution when the
  candidate space exceeds configured bounds.

### Expected Rejected Behavior

- Programs whose candidate space cannot be represented within configured
  runtime limits fail with a resource diagnostic.
- Programs using constructs outside the accepted epistemic fragment still fail
  closed with `UnsupportedEpistemicConstruct`.

### Required Evidence

- A pilot where candidate worlds are generated from EIR rather than supplied by
  a test fixture.
- A pilot with both accepted and rejected candidates and stable rejection
  reasons.
- A resource-limit pilot proving fail-closed behavior.
- Runtime traces showing GPU candidate generation, propagation, world-view
  validation, result materialization, and zero forbidden CPU fallback counters.

### KPIs

| KPI | Target |
|---|---|
| EGB01.K1 candidate source | 100 percent of accepted pilots derive candidates from EIR, not fixture input |
| EGB01.K2 trace completeness | generated, propagated, tested, accepted, rejected, and reason counts are emitted |
| EGB01.K3 determinism | repeated deterministic runs produce identical candidate and result sets |
| EGB01.K4 fallback safety | forbidden CPU fallback counters remain zero |

## Bundle EGB-02: Tuple-Key And Bound-Value Membership Semantics

### Current Limitation

Ground and fixed tuple-key paths are the strongest accepted surface. Bound-output
metadata exists for variable tuple keys, but the full semantics for general
bound variables, anonymous positions, aggregates, compound terms, lists, and
probabilistic evidence tuple keys are not complete across the accepted runtime
surface.

### Semantic Goal

Tuple membership must answer whether the atom under a modal literal is present
in the relevant stable-model tuple source using the source atom's value-level
terms.

The required semantics are:

- Ground terms match equal typed values.
- Bound variables match the value already bound by the reduced output row.
- Anonymous terms match any value without contributing an equality requirement.
- Repeated variables impose equality across all occurrences.
- Unsupported term classes remain rejected until their finite semantics are
  defined and certified.

### Expected Accepted Behavior

- `know edge(1, 2)` checks ground tuple membership.
- `known_child(X) :- node(X), know child(X).` checks `child(X)` using the
  value of `X` from the reduced output row.
- `possible edge(X, Y)` preserves type-correct matching for both bound columns.
- Repeated-variable literals such as `know same(X, X)` enforce value equality.
- Empty tuple sources produce semantically correct `possible` and
  `not possible` outcomes.

### Expected Rejected Behavior

- Unbound variables in modal atoms fail with a typed diagnostic.
- Aggregate tuple keys fail unless the aggregate value has a defined finite
  bound and a materialized value source.
- Compound/list/predref tuple keys fail unless their finite equality semantics
  are explicitly defined and accepted.
- Type mismatches between bound output columns and tuple source columns fail
  before result materialization.

### Required Evidence

- GPU pilots for ground keys, single bound variables, multiple bound variables,
  repeated variables, anonymous positions, and empty tuple sources.
- Negative pilots for unbound variables, type mismatches, and unsupported term
  classes.
- Evidence that membership uses device relation columns and row-count buffers,
  not host tuple scans.

### KPIs

| KPI | Target |
|---|---|
| EGB02.K1 arity coverage | arity 0, 1, 2, and 3 tuple-key pilots pass through GPU membership |
| EGB02.K2 bound variables | bound-variable pilots produce exact expected rows |
| EGB02.K3 negative safety | unsupported or ill-typed keys fail with typed diagnostics |
| EGB02.K4 device execution | tuple-key column reads and match traces are device-backed |

## Bundle EGB-03: Nested Modal Operator Semantics

### Current Limitation

Nested modal syntax such as `know possible p()` is recognized only to return a
typed unsupported diagnostic. The language does not currently assign accepted
semantics to modal operators applied to modal formulas.

### Semantic Goal

If nested modal operators become accepted, their truth conditions must be
defined over world views without flattening them into ordinary atoms.

At minimum, the semantics must define:

- whether nesting is permitted for both `know` and `possible`;
- how explicit negation interacts with nesting;
- whether nested formulas quantify over the same world view or over a derived
  accessibility relation;
- how nested formulas participate in candidate generation and testing.

### Expected Accepted Behavior

- Accepted nested forms evaluate deterministically according to the selected
  epistemic mode.
- `know possible p()` and `possible know p()` are not treated as equivalent
  unless the semantics explicitly imply equivalence.
- Nested modal formulas are preserved in the semantic representation until
  their truth value is evaluated.

### Expected Rejected Behavior

- If the accepted fragment remains single-level for a milestone, nested forms
  must continue to fail closed with typed diagnostics.
- Ambiguous combinations of `not`, `know`, and `possible` must not be accepted
  by parser precedence accident.

### Required Evidence

- A mode-specific truth table or oracle matrix for every accepted nested form.
- Parser and EIR evidence that nested formulas are represented explicitly.
- Runtime pilots proving nested modal truth values affect accepted results.
- Negative pilots for forms intentionally outside the accepted nested fragment.

### KPIs

| KPI | Target |
|---|---|
| EGB03.K1 semantic matrix | every accepted nested form has an explicit expected truth table |
| EGB03.K2 representation | accepted nested formulas are not rewritten into fake ordinary predicates |
| EGB03.K3 mode coverage | FAEEL and G91 behavior is separately tested or explicitly scoped |
| EGB03.K4 diagnostics | non-accepted nested forms fail with stable typed diagnostics |

## Bundle EGB-04: Epistemic Integrity Constraints

### Current Limitation

Constraints containing epistemic literals, such as `:- know unsafe().`, are
rejected before GPT or GPU lowering. They are not silently lowered through the
reduced ordinary program.

### Semantic Goal

Epistemic constraints must constrain accepted world views, not individual
ordinary derivation rows only. A candidate world view is accepted only if every
epistemic integrity constraint evaluates to false under the selected epistemic
semantics.

### Expected Accepted Behavior

- `:- know unsafe().` rejects any candidate world view where `unsafe` is true in
  every accepted model.
- `:- possible contradiction().` rejects any candidate world view where
  `contradiction` appears in at least one accepted model.
- `:- not possible required().` rejects world views where `required` appears in
  no accepted model.
- Constraint failures are visible as semantic rejection reasons, not runtime
  crashes.

### Expected Rejected Behavior

- Constraints whose modal atoms contain unsupported tuple keys fail closed.
- Constraints requiring unsupported nested modal semantics fail closed unless
  EGB-03 has accepted that fragment.
- Constraints that would require CPU-only world-view scans in the accepted path
  fail certification.

### Required Evidence

- Positive pilots where epistemic constraints prune only the invalid world
  views.
- Negative pilots where every candidate is rejected and the program reports an
  empty accepted result rather than a runtime failure.
- Trace evidence that constraint evaluation participates in GPU world-view
  validation or an explicitly certified equivalent production path.

### KPIs

| KPI | Target |
|---|---|
| EGB04.K1 pruning correctness | valid and invalid candidates are separated exactly |
| EGB04.K2 reason visibility | rejected candidates include constraint-specific reasons |
| EGB04.K3 no silent lowering | source audit shows no ordinary-RIR constraint rewrite for epistemic constraints |
| EGB04.K4 production evidence | accepted constraint pilots have zero forbidden CPU fallback counters |

## Bundle EGB-05: Safe Split Dependency And Coupling Semantics

### Current Limitation

The split planner can identify deterministic components and compile valid
bounded split components, but it rejects or coalesces unsafe coupling. The goal
is not to make every split independent; it is to make split behavior complete,
explainable, and semantically equivalent to unsplit execution.

### Semantic Goal

Splitting must preserve the result set and world-view semantics of the unsplit
program. Components may be solved independently only when no ordinary,
negated, constraint, or epistemic dependency can make their world-view
acceptance mutually dependent.

### Expected Accepted Behavior

- Shared extensional inputs do not force components together.
- Derived dependencies force components together when one component consumes
  another component's head predicate.
- Integrity constraints coalesce exactly the components whose heads they
  mention.
- Valid split execution produces the same accepted results as unsplit execution.

### Expected Rejected Behavior

- Programs that require cross-component epistemic solving fail closed until
  EGB-06 accepts that semantic class.
- Split certificates that cannot prove recomposition equivalence are rejected.
- Source-order-dependent accidental splits are rejected unless recomposition is
  stable and equivalent.

### Required Evidence

- Paired split/unsplit pilots with identical outputs.
- Diagnostics explaining why components were split, coalesced, or rejected.
- Recomposition evidence that all source rules are covered exactly once in the
  semantic order required by the program.

### KPIs

| KPI | Target |
|---|---|
| EGB05.K1 equivalence | valid split pilots match unsplit outputs exactly |
| EGB05.K2 coverage | recomposition covers every relevant source rule exactly once |
| EGB05.K3 diagnostics | split/coalesce/reject decisions are explainable in traces |
| EGB05.K4 fallback safety | split pilots keep forbidden CPU fallback counters at zero |

## Bundle EGB-06: Cross-Component Multi-Epistemic Predicate Solving

### Current Limitation

Rules that couple more than one distinct epistemic body predicate are rejected
by the bounded split layer. Example:

```xlog
h() :- know p(), possible q().
```

Such a rule may require joint reasoning over multiple epistemic components.

### Semantic Goal

Accepted multi-epistemic-predicate rules must be solved as a joint epistemic
condition over the candidate world view. The result must be identical to
unsplit execution for the same program.

### Expected Accepted Behavior

- A rule with multiple modal predicates derives its head only when the full
  modal conjunction is true under the selected mode.
- Mixed operators such as `know p(), possible q()` preserve operator-specific
  semantics for each atom.
- Negated modal literals such as `not know p(), possible q()` participate in the
  same candidate test.
- Solver traces expose that the rule was handled jointly, not split into
  unsound independent pieces.

### Expected Rejected Behavior

- Programs whose joint modal condition cannot be represented within resource
  limits fail with a resource diagnostic.
- Programs whose joint condition depends on unsupported tuple-key or nested
  modal semantics fail with the relevant typed diagnostic.

### Required Evidence

- Pilots covering same-rule modal conjunctions, mixed operators, negated modal
  literals, and empty/contradictory candidate sets.
- Equivalence pilots comparing joint execution against an oracle semantic
  matrix.
- Runtime traces showing joint candidate/world-view evaluation without CPU
  fallback.

### KPIs

| KPI | Target |
|---|---|
| EGB06.K1 modal conjunction correctness | all operator combinations in the accepted fragment produce expected heads |
| EGB06.K2 unsplit parity | joint execution matches unsplit semantic oracle results |
| EGB06.K3 trace clarity | traces identify jointly evaluated modal predicates and outcomes |
| EGB06.K4 resource safety | over-budget joint solving fails before partial execution |

## Bundle EGB-07: FAEEL Founded Self-Support Completion

### Current Limitation

Default FAEEL rejects direct self-support such as `p() :- possible p().` unless
there is independent founded support. Nonzero-arity self-possible rules fail
closed until tuple-level foundedness can be proven for each bound key. G91
explicitly allows compatibility-style self-support.

### Semantic Goal

FAEEL must accept only founded epistemic conclusions. Any self-supporting modal
rule is accepted only when the program proves an independent, non-circular
foundation for the same tuple key. G91 remains the explicit compatibility mode
for semantics that intentionally allow self-supported possibility.

### Expected Accepted Behavior

- FAEEL rejects unfounded `p() :- possible p().`.
- FAEEL accepts self-reference only when independent founded support exists for
  the same predicate and tuple key.
- Nonzero-arity foundedness is checked per tuple key, not per predicate name
  only.
- G91 behavior remains separately testable and does not weaken FAEEL defaults.

### Expected Rejected Behavior

- Cyclic modal support with no independent foundation is rejected in FAEEL.
- Tuple-level foundedness that cannot be proven on the accepted path fails
  closed.
- Accidental acceptance caused by predicate-level support without matching tuple
  keys is a correctness failure.

### Required Evidence

- FAEEL pilots for zero-arity and nonzero-arity founded and unfounded cases.
- G91 compatibility pilots proving explicitly selected G91 behavior remains
  unchanged.
- Tuple-key evidence showing foundedness is checked at value level for
  nonzero-arity predicates.
- Negative pilots for modal cycles and contradictory foundedness claims.

### KPIs

| KPI | Target |
|---|---|
| EGB07.K1 FAEEL foundedness | all founded/unfounded FAEEL pilots produce expected acceptance or rejection |
| EGB07.K2 tuple granularity | nonzero-arity foundedness is proven per tuple key |
| EGB07.K3 G91 separation | G91-only accepted self-support does not pass under default FAEEL |
| EGB07.K4 diagnostic quality | rejected self-support reports the missing foundation precisely |

## Bundle Ordering

Recommended dependency order:

1. EGB-02 tuple-key membership, because arbitrary world enumeration and
   foundedness need correct value-level modal atom checks.
2. EGB-01 candidate-world enumeration, because later constraints and joint
   solving need generated candidates from EIR.
3. EGB-07 FAEEL foundedness, because it defines default-mode acceptance.
4. EGB-04 epistemic constraints, because constraints prune candidate world
   views after candidate semantics are available.
5. EGB-05 safe split semantics, because it can certify independent components
   after candidate and tuple semantics are correct.
6. EGB-06 joint multi-epistemic solving, because it is the generalization of
   split rejection cases.
7. EGB-03 nested modal operators, because it expands the language formula
   grammar and should only land after single-level semantics are complete.

## Global Completion Definition

The limitations are resolved only when:

- every accepted bundle has production-path pilots, not only source or oracle
  tests;
- every pilot reports deterministic semantic outputs and runtime traces;
- unsupported forms remain typed fail-closed with stable diagnostics;
- accepted paths show zero forbidden CPU fallback counters;
- `xlog run` exercises the same paths documented in the language reference;
- release docs distinguish completed semantics from remaining scoped-out
  fragments.

