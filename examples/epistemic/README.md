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
| Nested modal rejection | `13-nested-modal-rejected.xlog` | v0.9.1 EGB-03 typed fail-closed diagnostic for `know possible p()` |
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
| Compound modal key (rejected) | `23-compound-modal-key-rejected.xlog` | v0.9.2 boundary: a list/compound modal tuple-key `know watched([H])` fails closed → typed `UnsupportedEpistemicConstruct` `epistemic GPU tuple-key expectation` (names `List([Variable("H")])`) |
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

The remaining fail-closed cases are the GENUINELY-UNDEFINED forms, covered by
negative pilots: circular modality (a modal over the relation being recursively
computed, example 22), FAEEL-unfounded self-support (the defined rejection; G91
accepts it), syntactic nested modal operators (example 13), and compound/list modal
tuple-keys (example 23).
