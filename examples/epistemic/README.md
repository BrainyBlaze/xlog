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
| Nested modal chain collapse | `13-nested-modal-chain-collapse.xlog` | v0.9.2 ITEM C: `know possible p()` EXECUTES via sound KD45/S5 chain-collapse to `possible p()` (inner op wins); over EDB `p` (determined) → `q={()}`, rows: 1 |
| Nested modal chain filter | `13b-nested-modal-chain-filter.xlog` | v0.9.2 ITEM C: `know know reachable(X)` (KK≡K) gates `node` by `reachable` → `gated={1,3}` (node 2 dropped; load-bearing) |
| Nested modal chain negated | `13c-nested-modal-chain-negated.xlog` | v0.9.2 ITEM C: `not know possible blocked(X)` (leading-not distributes, inner `possible` ≡ ordinary over EDB) → `allowed={1,3}` (nodes 2,4 blocked-out; load-bearing) |
| Nested chain FAEEL-unfounded | `13d-nested-modal-chain-faeel-unfounded.xlog` | v0.9.2 ITEM C per-mode: `p():-possible possible p()` (MM≡M) is the chain form of ex31 → FAEEL `rows: 0` (unfounded, absent) |
| Nested chain G91-accepted | `13e-nested-modal-chain-g91-accepted.xlog` | v0.9.2 ITEM C per-mode: SAME chain under G91 → `rows: 1`. 13d (FAEEL 0) vs 13e (G91 1) is the inherited mode-difference (mirrors 31 vs 32) |
| Nested modal interior-negation (rejected) | `13f-nested-modal-interior-negation-rejected.xlog` | v0.9.2 ITEM C boundary: `know not possible p()` ≡ `K ¬M p` is a modal-over-negated-modal compound (C2) with NO sound collapse → typed `UnsupportedEpistemicConstruct` "interior negation between modal operators" |
| Mixed-literal membership | `14-mixed-literal-membership.xlog` | v0.9.2 Bundle 1 EGB-02B global modal gate + per-row bound membership compose conjunctively → `reachable={1,3}` |
| Recursive epistemic closure | `15-recursive-epistemic-closure.xlog` | v0.9.2 Bundle 2 Case-A recursive fixpoint: `reach(X,Z):-reach(X,Y),know edge(Y,Z)` → `{(1,2),(2,3)}` plus derived `(1,3)` |
| Recursive epistemic chain | `15-recursive-epistemic-chain.xlog` | v0.9.2 Bundle 2 Case-A multi-hop: 4-chain → full closure incl. 3-hop `(1,4)` (proves not single-pass) |
| Cross-component coupling (accepted) | `16-cross-component-coupling.xlog` | v0.9.2 Bundle 3 safe coupling: ordinary `report` consumes epistemic-derived `trusted`, coalesced single-output → `trusted={1,3}` |
| Cross-component chained (stratified) | `17-cross-component-chained-stratified.xlog` | v0.9.2 STRATIFIED execution: `flagged :- know trusted` over the DETERMINED epistemic head `trusted :- know vetted` (EDB). `trusted={1,3}` is materialized as a lower stratum; the higher stratum gates `know trusted` against it → `flagged={1,3}` (node 2 gated out) |
| Cross-component JOINT-solving | `18-cross-component-joint-shared-modal.xlog` | v0.9.2 Bundle 3 multi-output joint-solving: two heads sharing base modal `q` materialize against one shared world view → `known={1,2}`, `maybe={2}` (both heads displayed) |
| Access-control mixed modal (FULL) | `19-access-control-mixed-modal.xlog` | v0.9.2 EGB-02B robust: ONE rule combines a GLOBAL `know gateway_online()` gate + per-row `possible cleared(P)` + per-row `not possible revoked(P)`, composed conjunctively → `granted={1,2}` (carol/3 not cleared and dave/4 revoked are gated out; differs from ungated `principal={1,2,3,4}`) |
| Supply-chain recursive reach (FULL) | `20-supply-chain-recursive-reach.xlog` | v0.9.2 Case-A robust: ordinary recursion in `sources_from` gated by `know certified` over the INVARIANT EDB → 10-tuple closure incl. 4-hop derived `(1,5)`; supplier 6 (reachable only via an uncertified link) is fully gated out (no tuple ends in 6; differs from the ungated raw-graph closure) |
| Incident triage joint-solving (FULL) | `21-incident-triage-joint-modal.xlog` | v0.9.2 Bundle 3 robust JOINT-solving: THREE epistemic heads share base modal `compromised`, exercising all three modalities → `quarantine={1,2}` (`know`), `watch={2}` (`possible`; monitored/3 gated out), `clear={3,4,5}` (`not possible`); all three heads displayed |
| Recursion through modal (FIXPOINT) | `22-recursion-through-modal-fixpoint.xlog` | v0.9.2 ITEM A Case-B recursive epistemic FIXPOINT: a POSITIVE `know reach` over a relation that CO-EVOLVES with the recursion is executed to its FAEEL FOUNDED least fixpoint (modal truth + ordinary derivation co-evolve). The modal feeds a non-mirror `trust`, so the gate is load-bearing → founded `reach={(1,2),(1,3)}`, which DIFFERS from base-only `{(1,2)}` and from ungated `{(1,1),(1,2),(1,3)}` (the unfounded `trust(3,1)` is excluded) |
| Structured modal key (accepted) | `23-compound-modal-key-membership.xlog` | v0.9.2 ITEM D: a STRUCTURED finite+typed modal tuple-key. The 1-element list `[H]` in `know watched([H])` is flattened ELEMENT-WISE onto `watched`'s scalar u32 key column and matched on the GPU (no host matcher, no CPU fallback), GATING `host` by `watched` membership → `alert={1}` (gate load-bearing: node 2 dropped; ungated `alert=host={1,2}`). Multi-element `[A,B]`/compound `f(A,B)` flatten the same way over an arity-2 relation; anonymous `_` is a per-column wildcard |
| Unbounded modal key (rejected, finiteness) | `23b-unbounded-cons-modal-key-rejected.xlog` | v0.9.2 ITEM D boundary: an UNBOUNDED structured key — a `cons` `[H \| T]` whose tail length is not statically fixed — has no finite, typed GPU key-column set. Fails closed with a precise `ResourceExhausted` FINITENESS diagnostic (names the cons tail; points at the fixed-arity list alternative), NOT a blanket `UnsupportedEpistemicConstruct` |
| Transitive determined modal (stratified) | `24-transitive-determined-modal-stratified.xlog` | v0.9.2 STRATIFIED: a modal over an ORDINARY relation transitively derived from a DETERMINED epistemic head (`b :- know r`, `r :- a`, `a :- know p`). `r` is determined-in-principle (ordinary over the determined `a`); the lower stratum materializes gated `a`, `r :- a` is computed over the materialized base (locally invariant), the higher stratum gates `know r` against the base `r` → `b={1}` (gate load-bearing: ungated `b=node={1,2}`) |
| Recursion over determined modal | `25-recursion-over-determined-modal.xlog` | v0.9.2 STRATIFIED recursion: `reach :- reach, know a` over the DETERMINED epistemic head `a :- know certified` (EDB). `a` is materialized as a lower stratum; the recursive stratum gates over the now-base `a` → `reach={(1,2),(2,3),(1,3)}` (derived `(1,3)` proves multi-hop fixpoint) |
| Negated modal over invariant (recursive) | `26-negated-modal-over-invariant-recursive.xlog` | v0.9.2 Case-A: `not know blocked(Y)` over EDB `blocked` ≡ ordinary `not blocked(Y)` (anti-join, no modal gating) in a recursive program → `reach={(1,2),(3,4)}` (blocked node 3 severs the chain; differs from ungated closure) |
| Augmented multi-head (differing arity) | `27-augmented-multi-head-differing-arity.xlog` | v0.9.2 PER-HEAD augmented projection: two heads share base modal `edge/2` but differ in public arity and both need projection (`one_hop(X) :- node(X), know edge(X,Y)` arity 1; `pair(X,Y) :- color(X), possible edge(X,Y)` arity 2). Each head materializes from its own reduced buffer projected by its own public arity → `one_hop={1,2}`, `pair={(2,20),(2,21),(3,30)}` |
| Determined multi-column binding modal (stratified) | `28-determined-multicol-binding-modal.xlog` | v0.9.2 STRATIFIED: a modal that BINDS an output variable over a DETERMINED MULTI-COLUMN epistemic head (`out(X) :- node(X), know r(X,Y)`, `r(X,Y) :- edge(X,Y), know flag(X)`). `r` is determined-in-principle (its modal bottoms out in EDB `flag`, acyclic); the lower stratum materializes gated `r={(1,2),(1,3)}`, the higher stratum binds `Y` via the EGB-02 membership/join against the materialized base and projects it away → `out={1}` (gate load-bearing: dropping the modal gives `out=node={1,2,3}`; de-modalizing `know r`→`r` keeps `out={1}`, the `know R≡R` soundness for determined R). This completes the determined-modal family: every modal target is determined→handled (any arity, direct-epistemic or transitive-ordinary) or non-determined (recursive/through-the-SCC)→correctly fail-closed |
| Negated modal over determined-derived | `29-negated-modal-over-determined-derived.xlog` | v0.9.2 COMPLETENESS CELL: a NEGATED modal over a DETERMINED epistemic-DERIVED head. `a :- know p` (EDB `p`) is determined, so `not know a ≡ not possible a ≡ not a` (stratified anti-join over the materialized base `a={1,2}`) → `q_know={3}`, `q_poss={3}` (equal results prove the modal equivalence; node 3 is the only node not in `a`). Pushes example 26's `not know R≡not R` one level up from EDB to a determined DERIVED head |
| Possible binding over determined | `30-possible-binding-over-determined.xlog` | v0.9.2 COMPLETENESS CELL: the `possible` twin of example 28 — a BINDING `possible r(X,Y)` over a DETERMINED multi-column epistemic head. Proves the modal operator is irrelevant for a determined target (`possible r ≡ know r ≡ r`); stratifies identically → `out={1}` (gate load-bearing: ungated `out=node={1,2,3}`) |
| FAEEL-unfounded self-support (empty extension) | `31-faeel-unfounded-self-support-empty-extension.xlog` | v0.9.2 ITEM B: `p() :- possible p()` with no independent founded support is UNFOUNDED under default FAEEL, so `p` is ABSENT from the founded model. The program EXECUTES to its EXACT empty founded extension → `rows: 0`, exit 0 (NOT a rejection). The founded extension is computed on the GPU/runtime path (the circular self-support rule is excluded from the reduced founded base). Contrast example 11 (founded → accepted, rows: 1) and example 32 (same program under G91 → accepted, rows: 1) |
| G91 self-support accepted (mode-difference pair to 31) | `32-g91-self-support-accepted.xlog` | v0.9.2 ITEM B: the SAME `p() :- possible p()` under explicit `#pragma epistemic_mode = g91` ACCEPTS circular self-support → `rows: 1`. The ONLY change from example 31 is the mode pragma; the FAEEL rows:0 vs G91 rows:1 divergence is the exact FAEEL-vs-G91 mode-difference evidence |
| Negated modal through recursion (rejected) | `33-negated-modal-through-recursion-rejected.xlog` | v0.9.2 ITEM A soundness floor: a NEGATED modal `not know reach` over a relation that CO-EVOLVES with the recursion is NON-MONOTONE (the gated complement shrinks as `reach` grows), so the monotone founded-least-fixpoint reduction cannot express it. Fails closed → typed `UnsupportedEpistemicConstruct` `recursive epistemic program` (names `not know`, "non-monotone"). Contrast example 22 (POSITIVE modal co-evolving → accepted Case-B fixpoint) |
| Variable-keyed constraint (prunes) | `34-variable-keyed-constraint-prunes.xlog` | v0.9.2 ITEM E: a VARIABLE-KEYED epistemic integrity constraint `:- know flagged(X).` ranges X EXISTENTIALLY over the modal relation's tuple-key domain — pruned iff EXISTS a binding for which `know flagged(X)` holds (`flagged` non-empty in the accepted model). The single-occurrence key variable lowers to an Anonymous WILDCARD column and the existing GPU wildcard tuple-key matcher evaluates the existential entirely on device (no host scan, no CPU fallback). `flagged={7,9,11}` is multi-tuple, so a ground `know flagged(c)` could not express "some flagged value exists" → the range is load-bearing. Body holds → world view pruned → `report` absent (`rows: 0`, exit 0, NOT a failure) |
| Variable-keyed constraint (survives, companion) | `35-variable-keyed-constraint-survives.xlog` | v0.9.2 ITEM E: the COMPANION to 34 — SAME `:- know flagged(X).` constraint but `flagged` EMPTY, so no binding satisfies `know flagged(X)`, the existential body is false and the world view SURVIVES → `report` holds (`rows: 1`). The 34 (`rows: 0`) vs 35 (`rows: 1`) difference is the EXACT load-bearing effect of the variable-keyed existential range (same program shape, only the modal extension differs) |
| Multi-literal distinct-var constraint | `36-multi-literal-distinct-var-constraint.xlog` | v0.9.2 ITEM E multi-literal (DISTINCT independent variables): `:- know watch(X), know hot(Y).` factors to (∃X: know watch(X)) AND (∃Y: know hot(Y)) = "watch non-empty AND hot non-empty". Each literal lowers to an INDEPENDENT Anonymous wildcard; the GPU constraint kernel ANDs the two existential assumption bits. Both non-empty → body holds → pruned (`rows: 0`). This is the independent-existential conjunction; the shared-variable JOIN (`:- know p(X), possible q(X)`), diagonal (`:- know p(X,X)`), and negated-difference (`:- q(X), not know p(X)`) are now ALSO supported — see examples 38–41. The second-literal load-bearing flip (empty `hot` → survives) is asserted in the device suite |
| Negated modal over recursive (stratified) | `37-negated-modal-over-recursive-stratified.xlog` | v0.9.2 ITEM A1: a NEGATED modal `not know reach` over a recursive relation that is STRATIFIED (reach is in a strictly-lower stratum, not co-evolving with the negation) executes on GPU — `not know reach ≡ not reach` (anti-join) once reach is materialized → `unreachable = node² − reach = {(1,1),(2,1),(2,2),(3,1),(3,2),(3,3)}`. Contrast example 33 (genuine negation cycle → formal WFS/no-host-solver bound) |
| Diagonal modal constraint (prunes) | `38-diagonal-modal-constraint.xlog` | v0.9.2 ITEM E1: a DIAGONAL epistemic constraint `:- know route(X,X)` (a single modal literal repeating X across its key columns) forbids a self-loop. Desugared at normalization to `__epi_diag_0(X) :- route(X,X)` + `:- know __epi_diag_0(X)`, routing through the variable-keyed prune-to-empty path (no new kernel). `route(1,1)` self-loop → world view pruned → `safe` empty (`rows: 0`). Sound for base/determined targets (`know R ≡ R`); a modal-derived target falls through to the existing rejection |
| Diagonal modal constraint (satisfied) | `39-diagonal-modal-constraint-satisfied.xlog` | v0.9.2 ITEM E1: companion to 38 with NO self-loop → constraint satisfied → `safe = {5}`. The 38-vs-39 flip (only the diagonal tuple differs) is the load-bearing prune evidence |
| Shared-variable join constraint | `40-shared-variable-join-constraint.xlog` | v0.9.2 ITEM E1: `:- know p(X), possible q(X)` — two modal literals sharing X — forbids `p ∩ q ≠ ∅`. Desugared to `__epi_join_0(X) :- p(X), q(X)` + `:- know __epi_join_0(X)` (sound for EDB: `know p ≡ p`, `possible q ≡ q`). `p ∩ q = {2}` non-empty → `report` pruned empty. Removing the constraint keeps `report = {5}` (load-bearing) |
| Negated-difference constraint | `41-negated-difference-constraint.xlog` | v0.9.2 ITEM E1: `:- q(X), not know p(X)` — positive `q` binds X (range-restriction), negated modal subtracts known-`p` — forbids `q \ p ≠ ∅`. Desugared to `__epi_join_0(X) :- q(X), not p(X)` + `:- know __epi_join_0(X)`. `q \ p = {3}` non-empty → `report` pruned empty. (Contrast the standalone unsafe `:- not know p(X)`, a NAF safety error.) |

Notes: examples with one epistemic output head use the single-plan path; examples
with independent epistemic output heads route through split GPU execution from
`xlog run`. A coalesced component with more than one epistemic output head sharing
a base modal predicate is JOINT-SOLVED with multi-output materialization — each
head materialized against one shared accepted world view (examples 18, 21).

A modal literal over a DETERMINED head — one whose every defining rule ranges only
over invariant/EDB or already-determined relations, acyclically — is resolved by
v0.9.2 STRATIFIED execution: the determined head is materialized (gated) into the
relation store as a lower stratum and the higher stratum gates against the
materialized base relation through the existing membership filter — no
resolve-into-body, no double-gating. The determined-closure is TRANSITIVE across
ordinary derivations: an ORDINARY predicate is determined when every rule defining
it ranges only over determined/invariant relations (e.g. `r :- a` with `a` a
determined epistemic head), so a modal `know r` over such an `r` also stratifies
(example 24 — the ordinary `r :- a` is deferred to the stratum where `a` is gated
base, computed once from the gated extension). This covers chained coupling
(example 17), recursion whose modal ranges over a determined head (example 25), and
transitive determined-ordinary coupling (example 24). A NEGATED modal over an
invariant relation reduces cleanly to ordinary negation, so it executes in
recursive/coupling contexts (example 26).

Augmented-projection multi-head coupling (heads of differing arity sharing a base
modal) is resolved by PER-HEAD output projection (example 27): each coupled head
materializes from its own reduced relation buffer projected by its own public arity,
reading only the store/world-view boundary. A positive modal over an INVARIANT
relation that is the sole binder of an output variable is resolved into a positive
ordinary join atom (a sound, machine-checked-invariant consequence), which also
makes single-head augmented modals over invariant relations executable.

Structured modal tuple-keys (example 23) are FLATTENED element-wise onto the GPU
when they are FINITE and TYPED: a fixed-arity list `[a, b]` or compound `f(a, b)`
of scalar/Symbol-typed elements expands into one GPU key column per element and
reuses the existing device tuple-key matcher (bound variable → BOUND_OUTPUT,
anonymous `_` → WILDCARD, ground literal → GROUND). The flattened arity must equal
the modal relation's arity. Only genuinely UNBOUNDED or UNTYPED structured forms (a
`cons` with a statically-unknown tail, a nested structure, a predref, an aggregate)
stay rejected, and they reject with a precise `ResourceExhausted` finiteness
diagnostic rather than a blanket unsupported-construct error (example 23b).

Nested modal operators (ITEM C) are no longer rejected wholesale. A bare modal
CHAIN — a sequence of `know`/`possible` operators over an atom, with an optional
single LEADING `not` — collapses to a single epistemic literal under the KD45/S5
modal axioms XLOG's autoepistemic operators assume: the operator ADJACENT to the
atom wins (`know possible p ≡ possible p`, `know know p ≡ know p`,
`possible possible p ≡ possible p`), and a leading `not` distributes. The collapse
is a SOUND modal-logic equivalence applied as AST normalization — it routes through
the existing single-level epistemic path with no new world-of-worlds evaluator —
and holds in BOTH modes (FAEEL/G91 differ only in which world views are admissible,
which the collapsed single-level literal inherits unchanged; examples 13/13b/13c
execute, 13d/13e show the inherited FAEEL-vs-G91 mode split). The remaining
fail-closed boundary for nesting is C2 modal-over-a-negated-modal compound formulas
— an INTERIOR negation (`know not possible p ≡ K ¬M p`, example 13f) or an
atom-adjacent negation (`know possible not p ≡ M ¬p`) — which have no sound collapse
to a single `op atom` literal and stay rejected with a typed
`UnsupportedEpistemicConstruct` ("interior negation"/"negated atom") rather than a
wrong collapse.

The remaining fail-closed cases are the other GENUINELY-UNDEFINED forms, covered by
negative pilots: circular modality (a modal over the relation being recursively
computed, example 22), FAEEL-unfounded self-support (the defined rejection; G91
accepts it), and unbounded/untyped modal tuple-keys (example 23b).
