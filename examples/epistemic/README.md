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
| Cross-component coupling (rejected) | `17-cross-component-coupling-rejected.xlog` | v0.9.2 Bundle 3 typed fail-closed diagnostic: a modal literal over an epistemic-derived head couples two epistemic outputs (`cross-component epistemic coupling`, names `trusted`/`flagged` + `DerivedPredicate`) |
| Cross-component JOINT-solving | `18-cross-component-joint-shared-modal.xlog` | v0.9.2 Bundle 3 multi-output joint-solving: two heads sharing base modal `q` materialize against one shared world view → `known={1,2}`, `maybe={2}` (both heads displayed) |
| Access-control mixed modal (FULL) | `19-access-control-mixed-modal.xlog` | v0.9.2 EGB-02B robust: ONE rule combines a GLOBAL `know gateway_online()` gate + per-row `possible cleared(P)` + per-row `not possible revoked(P)`, composed conjunctively → `granted={1,2}` (carol/3 not cleared and dave/4 revoked are gated out; differs from ungated `principal={1,2,3,4}`) |
| Supply-chain recursive reach (FULL) | `20-supply-chain-recursive-reach.xlog` | v0.9.2 Case-A robust: ordinary recursion in `sources_from` gated by `know certified` over the INVARIANT EDB → 10-tuple closure incl. 4-hop derived `(1,5)`; supplier 6 (reachable only via an uncertified link) is fully gated out (no tuple ends in 6; differs from the ungated raw-graph closure) |
| Incident triage joint-solving (FULL) | `21-incident-triage-joint-modal.xlog` | v0.9.2 Bundle 3 robust JOINT-solving: THREE epistemic heads share base modal `compromised`, exercising all three modalities → `quarantine={1,2}` (`know`), `watch={2}` (`possible`; monitored/3 gated out), `clear={3,4,5}` (`not possible`); all three heads displayed |
| Recursion through modal (rejected) | `22-recursion-through-modal-rejected.xlog` | v0.9.2 boundary: a modal literal `know reach` over a relation entangled in the recursive SCC (not invariant) fails closed → typed `UnsupportedEpistemicConstruct` `recursive epistemic program` (names `know reach`, "not invariant") |
| Compound modal key (rejected) | `23-compound-modal-key-rejected.xlog` | v0.9.2 boundary: a list/compound modal tuple-key `know watched([H])` fails closed → typed `UnsupportedEpistemicConstruct` `epistemic GPU tuple-key expectation` (names `List([Variable("H")])`) |

Notes: examples with one epistemic output head use the single-plan path; examples
with independent epistemic output heads route through split GPU execution from
`xlog run`. A coalesced component with more than one epistemic output head sharing
a base modal predicate is JOINT-SOLVED with multi-output materialization — each
head materialized against one shared accepted world view (examples 18, 21). Coupling
over an epistemic-derived head (nested/stratified) and augmented-projection coupling
currently fail closed with a typed `cross-component epistemic coupling` diagnostic
(see the v0.9.2 status doc). Genuinely-undefined or out-of-scope forms — circular
modality (a modal over the relation being recursively computed), FAEEL-unfounded
self-support (the defined rejection; G91 accepts it), and syntactic nested modal
operators — are covered by negative pilots.
