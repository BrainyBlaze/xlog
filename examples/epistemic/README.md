# Epistemic Semantics Examples

These examples are production `xlog run` examples and semantic fixtures. The
high-level GPU runner detects accepted epistemic programs and routes them through
the epistemic GPU runtime; the lower-level direct RIR lowering boundary still
rejects raw epistemic literals with `UnsupportedEpistemicConstruct`.

`01-05` cover the epistemic surface. `06-13` cover executable epistemic
semantics: each demonstrates one accepted behavior end-to-end through
`xlog run` and is validated with a deterministic output marker by
`test_xlog_run_epistemic_examples` or a typed negative diagnostic test.

Run the examples:

```bash
# epistemic surface examples
cargo run -q -p xlog-cli -- run examples/epistemic/01-eir-boundary.xlog
cargo run -q -p xlog-cli -- run examples/epistemic/05-splitting.xlog
# executable epistemic examples
cargo run -q -p xlog-cli -- run examples/epistemic/06-eir-candidate-enumeration.xlog
cargo run -q -p xlog-cli -- run examples/epistemic/09-joint-multi-epistemic.xlog
cargo run -q -p xlog-cli -- run examples/epistemic/12-bound-variable-splitting.xlog
# validate all of them through xlog run (requires CUDA):
XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-cli --test run_cli_tests test_xlog_run_epistemic_examples
```

| Example | Fixture path | Showcases |
|---|---|---|
| EIR boundary | `01-eir-boundary.xlog` | EIR literal preservation |
| G91 compatibility | `02-g91-compatibility.xlog` | G91 compatibility mode |
| FAEEL default | `03-faeel-default.xlog` | FAEEL founded knowledge |
| Generate-Propagate-Test | `04-gpt-candidate-filter.xlog` | Generate-Propagate-Test execution |
| Epistemic splitting | `05-splitting.xlog` | independent epistemic components |
| Candidate enumeration | `06-eir-candidate-enumeration.xlog` | EIR candidate enumeration with bound membership -> `believed={1,3}` |
| Tuple-key membership | `07-tuple-key-membership.xlog` | multi-column bound membership -> `matched={(1,2),(3,3)}` |
| Repeated variable | `08-repeated-variable.xlog` | repeated-variable equality -> `reflexive={3}` |
| Joint multi-epistemic | `09-joint-multi-epistemic.xlog` | joint modal conjunction -> `both_known={1}` |
| Epistemic constraint | `10-epistemic-constraint.xlog` | constraint prunes the world view -> `accepted` empty (Ok, not failure) |
| FAEEL foundedness | `11-faeel-foundedness.xlog` | founded self-support -> `founded={()}`|
| Bound-variable splitting | `12-bound-variable-splitting.xlog` | split routing with bound modal membership -> `both_known={1}`, `safe_alt={2}` |
| Nested modal chain collapse | `13-nested-modal-chain-collapse.xlog` | `know possible p()` executes via sound KD45/S5 chain collapse to `possible p()` (inner operator wins); over determined EDB `p` -> `q={()}`, rows: 1 |
| Nested modal chain filter | `13b-nested-modal-chain-filter.xlog` | `know know reachable(X)` (KK == K) gates `node` by `reachable` -> `gated={1,3}` (node 2 dropped; load-bearing) |
| Nested modal chain negated | `13c-nested-modal-chain-negated.xlog` | `not know possible blocked(X)` distributes the leading negation and reduces the inner `possible` over EDB to ordinary negation -> `allowed={1,3}` (nodes 2 and 4 blocked out; load-bearing) |
| Nested chain FAEEL-unfounded | `13d-nested-modal-chain-faeel-unfounded.xlog` | `p() :- possible possible p()` (MM == M) is the chain form of `31-faeel-unfounded-self-support-empty-extension.xlog` -> FAEEL `rows: 0` (unfounded, absent) |
| Nested chain G91-accepted | `13e-nested-modal-chain-g91-accepted.xlog` | The same chain under G91 returns one row; paired with the FAEEL companion, it mirrors the single-level `31-faeel-unfounded-self-support-empty-extension.xlog` and `32-g91-self-support-accepted.xlog` mode difference |
| Nested modal interior-negation | `13f-nested-modal-interior-negation.xlog` | `know not possible p()` dualizes to `not possible p()` and executes through `xlog run`; because `p()` is present, `q` is empty (`rows: 0`) |
| Nested modal interior-negation mini-matrix | `13f-nested-modal-interior-negation-absent.xlog`, `13fw-nested-modal-interior-negation-g91-*.xlog` | Companion cells for interior-negation modal duality: target `{present,absent}` x mode `{FAEEL,G91}`. Present target cells keep `q` empty; absent target cells derive `q`, proving the duality in both directions and both modes |
| Nested modal negation matrix | `13g`-`13v-nested-modal-negation-matrix-*.xlog` | Exhaustive FAEEL matrix: all `{know,possible}^2` chains with leading/interior/atom-adjacent negation combinations over present and absent atoms; derived slices emit the exact four expected `holds(ID)` rows, while non-derived companion slices prove the complementary cases stay empty |
| Nested modal negation matrix G91 companion | `13w-nested-modal-negation-matrix-g91-*.xlog` | Both-mode companion matrix: replays the same finite two-operator negation matrix under explicit G91, proving chain-collapse/duality normalization is not accidentally FAEEL-only |
| Mixed-literal membership | `14-mixed-literal-membership.xlog` | global modal gate plus per-row bound membership compose conjunctively -> `reachable={1,3}` |
| Recursive epistemic closure | `15-recursive-epistemic-closure.xlog` | recursive fixpoint over an invariant modal target: `reach(X,Z):-reach(X,Y),know edge(Y,Z)` -> `{(1,2),(2,3)}` plus derived `(1,3)` |
| Recursive epistemic chain | `15-recursive-epistemic-chain.xlog` | multi-hop recursive closure: 4-chain -> full closure including 3-hop `(1,4)` (proves not single-pass) |
| Cross-component coupling (accepted) | `16-cross-component-coupling.xlog` | safe cross-component coupling: ordinary `report` consumes epistemic-derived `trusted`, coalesced single-output -> `trusted={1,3}` |
| Cross-component chained (stratified) | `17-cross-component-chained-stratified.xlog` | stratified execution: `flagged :- know trusted` over the determined epistemic head `trusted :- know vetted` (EDB). `trusted={1,3}` is materialized as a lower stratum; the higher stratum gates `know trusted` against it -> `flagged={1,3}` (node 2 gated out) |
| Cross-component joint solving | `18-cross-component-joint-shared-modal.xlog` | multi-output joint solving: two heads sharing base modal `q` materialize against one shared world view -> `known={1,2}`, `maybe={2}` (both heads displayed) |
| Access-control mixed modal | `19-access-control-mixed-modal.xlog` | one rule combines a global `know gateway_online()` gate, per-row `possible cleared(P)`, and per-row `not possible revoked(P)`, composed conjunctively -> `granted={1,2}` (carol/3 not cleared and dave/4 revoked are gated out; differs from ungated `principal={1,2,3,4}`) |
| Supply-chain recursive reach | `20-supply-chain-recursive-reach.xlog` | ordinary recursion in `sources_from` gated by `know certified` over invariant EDB -> 10-tuple closure including 4-hop derived `(1,5)`; supplier 6 (reachable only via an uncertified link) is fully gated out (no tuple ends in 6; differs from the ungated raw-graph closure) |
| Incident triage joint solving | `21-incident-triage-joint-modal.xlog` | three epistemic heads share base modal `compromised`, exercising all three modalities -> `quarantine={1,2}` (`know`), `watch={2}` (`possible`; monitored/3 gated out), `clear={3,4,5}` (`not possible`); all three heads displayed |
| Recursion through modal fixpoint | `22-recursion-through-modal-fixpoint.xlog` | recursive epistemic fixpoint: a positive `know reach` over a relation that co-evolves with recursion executes to its FAEEL founded least fixpoint (modal truth plus ordinary derivation co-evolve). The modal feeds a non-mirror `trust`, so the gate is load-bearing -> founded `reach={(1,2),(1,3)}`, which differs from base-only `{(1,2)}` and from ungated `{(1,1),(1,2),(1,3)}` (the unfounded `trust(3,1)` is excluded) |
| Structured modal key (accepted) | `23-compound-modal-key-membership.xlog` | structured finite typed modal tuple-key: the 1-element list `[H]` in `know watched([H])` is flattened element-wise onto `watched`'s scalar u32 key column and matched on the GPU (no host matcher, no CPU fallback), gating `host` by `watched` membership -> `alert={1}` (gate load-bearing: node 2 dropped; ungated `alert=host={1,2}`). Multi-element `[A,B]`/compound `f(A,B)` flatten the same way over an arity-2 relation; anonymous `_` is a per-column wildcard |
| Unbounded modal key (rejected, finiteness) | `23b-unbounded-cons-modal-key-rejected.xlog` | structured-key finiteness boundary: an unbounded key, a `cons` `[H \| T]` whose tail length is not statically fixed, has no finite typed GPU key-column set. It fails closed with a precise `ResourceExhausted` finiteness diagnostic (names the cons tail; points at the fixed-arity list alternative), not a blanket `UnsupportedEpistemicConstruct` |
| Transitive determined modal (stratified) | `24-transitive-determined-modal-stratified.xlog` | stratified execution for a modal over an ordinary relation transitively derived from a determined epistemic head (`b :- know r`, `r :- a`, `a :- know p`). `r` is determined-in-principle (ordinary over the determined `a`); the lower stratum materializes gated `a`, `r :- a` is computed over the materialized base (locally invariant), the higher stratum gates `know r` against the base `r` -> `b={1}` (gate load-bearing: ungated `b=node={1,2}`) |
| Recursion over determined modal | `25-recursion-over-determined-modal.xlog` | stratified recursion: `reach :- reach, know a` over the determined epistemic head `a :- know certified` (EDB). `a` is materialized as a lower stratum; the recursive stratum gates over the now-base `a` -> `reach={(1,2),(2,3),(1,3)}` (derived `(1,3)` proves multi-hop fixpoint) |
| Negated modal over invariant (recursive) | `26-negated-modal-over-invariant-recursive.xlog` | `not know blocked(Y)` over EDB `blocked` is equivalent to ordinary `not blocked(Y)` (anti-join, no modal gating) in a recursive program -> `reach={(1,2),(3,4)}` (blocked node 3 severs the chain; differs from ungated closure) |
| Augmented multi-head (differing arity) | `27-augmented-multi-head-differing-arity.xlog` | per-head augmented projection: two heads share base modal `edge/2` but differ in public arity and both need projection (`one_hop(X) :- node(X), know edge(X,Y)` arity 1; `pair(X,Y) :- color(X), possible edge(X,Y)` arity 2). Each head materializes from its own reduced buffer projected by its own public arity -> `one_hop={1,2}`, `pair={(2,20),(2,21),(3,30)}` |
| Determined multi-column binding modal (stratified) | `28-determined-multicol-binding-modal.xlog` | stratified execution for a modal that binds an output variable over a determined multi-column epistemic head (`out(X) :- node(X), know r(X,Y)`, `r(X,Y) :- edge(X,Y), know flag(X)`). `r` is determined-in-principle (its modal bottoms out in EDB `flag`, acyclic); the lower stratum materializes gated `r={(1,2),(1,3)}`, the higher stratum binds `Y` through membership/join against the materialized base and projects it away -> `out={1}` (gate load-bearing: dropping the modal gives `out=node={1,2,3}`; de-modalizing `know r` to `r` keeps `out={1}`, the `know R == R` soundness for determined R). This completes the determined-modal family: every modal target is determined and handled (any arity, direct-epistemic or transitive-ordinary) or non-determined (recursive/through-the-SCC) and correctly fail-closed |
| Negated modal over determined-derived | `29-negated-modal-over-determined-derived.xlog` | negated modal over a determined epistemic-derived head. `a :- know p` (EDB `p`) is determined, so `not know a == not possible a == not a` (stratified anti-join over the materialized base `a={1,2}`) -> `q_know={3}`, `q_poss={3}` (equal results prove the modal equivalence; node 3 is the only node not in `a`). Pushes `not know R == not R` one level up from EDB to a determined derived head |
| Possible binding over determined | `30-possible-binding-over-determined.xlog` | the `possible` twin of the determined multi-column binding example: a binding `possible r(X,Y)` over a determined multi-column epistemic head. Proves the modal operator is irrelevant for a determined target (`possible r == know r == r`); stratifies identically -> `out={1}` (gate load-bearing: ungated `out=node={1,2,3}`) |
| FAEEL-unfounded self-support (empty extension) | `31-faeel-unfounded-self-support-empty-extension.xlog` | `p() :- possible p()` with no independent founded support is unfounded under default FAEEL, so `p` is absent from the founded model. The program executes to its empty founded extension -> `rows: 0`, exit 0 (not a rejection). The founded extension is computed on the GPU/runtime path (the circular self-support rule is excluded from the reduced founded base). Contrast `11-faeel-foundedness.xlog` (founded -> accepted, rows: 1) and `32-g91-self-support-accepted.xlog` (same program under G91 -> accepted, rows: 1) |
| G91 self-support accepted | `32-g91-self-support-accepted.xlog` | the same `p() :- possible p()` under explicit `#pragma epistemic_mode = g91` accepts circular self-support -> `rows: 1`. The only change from the FAEEL companion is the mode pragma; the FAEEL rows:0 vs G91 rows:1 divergence is the FAEEL-vs-G91 mode-difference evidence |
| Negated modal through recursion (GPU WFS) | `33-negated-modal-through-recursion-wfs.xlog` | a negated modal `not know reach` over a relation that co-evolves with recursion reduces to a non-monotone ordinary SCC. The sound semantics is WFS, and the accepted fixture routes through the GPU-native alternating-fixpoint path: `reach={(1,2)}` while undefined tuples are absent. Host grounding and host WFS remain disallowed |
| Negated modal through recursion WFS matrix | `33a-negated-modal-through-recursion-wfs-*.xlog` | operator/mode matrix for cyclic-negated WFS: mode `{FAEEL,G91}` x modal form `{not know,not possible}`. Each shard uses the same seeded 3-node graph and expects only `reach={(1,2)}` to be emitted; all other `vertex x vertex` reach tuples are WFS-undefined and therefore absent. The full matrix is asserted by `test_xlog_run_negated_modal_through_recursion_uses_gpu_wfs_engine` through production `xlog run` |
| Negated modal through recursion WFS seed-state exhaustive matrix | `33c-negated-modal-through-recursion-wfs-matrix-*.xlog` | exhaustive seed-state matrix for plain cyclic WFS: mode `{FAEEL,G91}` x modal form `{not know,not possible}` x seed support `{present,absent}`. Seed-present shards emit exactly `reach={(1,2)}`; seed-absent shards emit no true reach rows; every other `vertex x vertex` tuple is false/undefined and absent |
| Negated modal through recursion WFS plus EDB negation | `33b-negated-modal-through-recursion-wfs-with-edb-negation.xlog` | mixed-negation case: the cyclic WFS component contains both `not know reach(X,Y)` and ordinary `not banned(Y)`. The accepted fixture proves the GPU WFS path handles ordinary EDB anti-joins inside the same reduced non-monotone SCC: seed-derived `reach={(1,2)}` is emitted, non-banned unseeded pairs remain undefined, and banned target-column pairs are false/absent. Host WFS remains disallowed |
| Negated modal through recursion WFS plus EDB-negation exhaustive matrix | `33d-negated-modal-through-recursion-wfs-edb-negation-matrix-*.xlog` | exhaustive mixed-negation matrix: mode `{FAEEL,G91}` x modal form `{not know,not possible}` x seed support `{present,absent}`, with ordinary `not banned(Y)` in the same reduced non-monotone SCC. This proves the GPU-backed WFS plan handles the modal cycle and ordinary anti-join together across every finite cell, not just the canonical fixture |
| Negated modal through recursion WFS plus EDB-negation load-bearing matrix | `33e-negated-modal-through-recursion-wfs-edb-negation-load-bearing-*.xlog` | behavioral mixed-negation matrix: mode `{FAEEL,G91}` x modal form `{not know,not possible}` x target state `{allowed,banned}`. `not banned(2)` controls the only seed support, so allowed cells emit exactly `reach={(1,2)}` while banned cells emit no reach rows; this makes the ordinary EDB anti-join axis runtime-observable |
| Variable-keyed constraint (prunes) | `34-variable-keyed-constraint-prunes.xlog` | variable-keyed epistemic integrity constraint `:- know flagged(X).` ranges X existentially over the modal relation's tuple-key domain, so it prunes iff some binding satisfies `know flagged(X)` (`flagged` non-empty in the accepted model). The single-occurrence key variable lowers to an anonymous wildcard column and the existing GPU wildcard tuple-key matcher evaluates the existential entirely on device (no host scan, no CPU fallback). `flagged={7,9,11}` is multi-tuple, so a ground `know flagged(c)` could not express "some flagged value exists" -> the range is load-bearing. Body holds -> world view pruned -> `report` absent (`rows: 0`, exit 0, not a failure) |
| Variable-keyed constraint (survives, companion) | `35-variable-keyed-constraint-survives.xlog` | companion to the pruning fixture: the same `:- know flagged(X).` constraint but `flagged` is empty, so no binding satisfies `know flagged(X)`, the existential body is false and the world view survives -> `report` holds (`rows: 1`). The difference from `34-variable-keyed-constraint-prunes.xlog` is the load-bearing effect of the variable-keyed existential range (same program shape, only the modal extension differs) |
| Multi-literal distinct-var constraint | `36-multi-literal-distinct-var-constraint.xlog` | multi-literal constraint with distinct independent variables: `:- know watch(X), know hot(Y).` factors to `(exists X: know watch(X)) and (exists Y: know hot(Y))`, meaning "watch non-empty and hot non-empty". Each literal lowers to an independent anonymous wildcard; the GPU constraint kernel combines the two existential assumption bits. Both non-empty -> body holds -> pruned (`rows: 0`). This is the independent-existential conjunction; the shared-variable join (`:- know p(X), possible q(X)`), diagonal (`:- know p(X,X)`), and negated-difference (`:- q(X), not know p(X)`) are also supported by examples 38-41. The second-literal load-bearing flip (empty `hot` -> survives) is asserted in the device suite |
| Negated modal over recursive (stratified) | `37-negated-modal-over-recursive-stratified.xlog` | a negated modal `not know reach` over a recursive relation that is stratified (reach is in a strictly-lower stratum, not co-evolving with the negation) executes on GPU: `not know reach == not reach` (anti-join) once reach is materialized -> `unreachable = node^2 - reach = {(1,1),(2,1),(2,2),(3,1),(3,2),(3,3)}`. Contrast `33-negated-modal-through-recursion-wfs.xlog` (genuine negation cycle -> GPU-backed WFS alternating fixpoint; host WFS not admitted) |
| Diagonal modal constraint (prunes) | `38-diagonal-modal-constraint.xlog` | diagonal epistemic constraint `:- know route(X,X)` (a single modal literal repeating X across its key columns) forbids a self-loop. Desugared at normalization to `__epi_diag_0(X) :- route(X,X)` plus `:- know __epi_diag_0(X)`, routing through the variable-keyed prune-to-empty path (no new kernel). `route(1,1)` self-loop -> world view pruned -> `safe` empty (`rows: 0`). Sound for base/determined targets (`know R == R`); a modal-derived target falls through to the existing rejection |
| Diagonal modal constraint (satisfied) | `39-diagonal-modal-constraint-satisfied.xlog` | companion to the diagonal pruning fixture with no self-loop -> constraint satisfied -> `safe = {5}`. The only behavioral difference is the diagonal tuple, proving the load-bearing prune effect |
| Shared-variable join constraint | `40-shared-variable-join-constraint.xlog` | `:- know p(X), possible q(X)` has two modal literals sharing X and forbids non-empty `p` intersection `q`. Desugared to `__epi_join_0(X) :- p(X), q(X)` plus `:- know __epi_join_0(X)` (sound for EDB: `know p == p`, `possible q == q`). `p` intersection `q` is `{2}`, so `report` is pruned empty. Removing the constraint keeps `report = {5}` (load-bearing) |
| Negated-difference constraint | `41-negated-difference-constraint.xlog` | `:- q(X), not know p(X)` has positive `q` bind X (range restriction), then the negated modal subtracts known `p`; it forbids non-empty `q \ p`. Desugared to `__epi_join_0(X) :- q(X), not p(X)` plus `:- know __epi_join_0(X)`. `q \ p = {3}` is non-empty -> `report` pruned empty. Contrast the standalone unsafe `:- not know p(X)`, a NAF safety error |
| Same-name multi-arity modal disambiguation | `42-same-name-multi-arity-modal-disambiguation.xlog` | `know p(X), possible p(X,Y)` over same-name `p/1` and `p/2` executes through full `xlog run` by loading arity-qualified modal source relations. `p/1={1,2}`, `p/2={(1,10),(2,99)}`, `seed={(1,10),(2,20),(3,30)}` -> exactly `a={(1,10)}`; `(2,20)` is removed by the arity-2 filter and `(3,30)` by the arity-1 filter |
| Same-name multi-arity exhaustive matrix | `42a-same-name-multi-arity-literal-*.xlog`, `42b-same-name-multi-arity-cross-*.xlog` | exhaustive production-path matrix for same-name multi-arity modal disambiguation. `42a*` covers all single-literal cells: arity `{1,2}` x operator `{know,possible}` x polarity `{positive,negated}` x queried tuple `{present,absent}`. `42b*` covers all cross-arity conjunction cells: unary modal form `{K,M,not K,not M}` x binary modal form `{K,M,not K,not M}` x unary queried tuple `{present,absent}` x binary queried tuple `{present,absent}`. Every shard uses nonmatching distractor facts in both `p/1` and `p/2`, so absent means the queried tuple is absent rather than the relation/schema being absent. The full matrix is asserted by `test_xlog_run_same_name_multi_arity_exhaustive_matrix` through production `xlog run` |
| Single-modal truth-table exhaustive matrix | `44a-single-modal-truth-table-*.xlog` | base-case matrix for one modal literal over a determined EDB target. Covers mode `{FAEEL,G91}` x modal form `{know,possible,not know,not possible}` x queried tuple `{present,absent}`. Both modes should agree on determined targets; every shard includes a nonmatching `p(2)` distractor so absent means `p(1)` is absent, not that `p/1` or its schema vanished. The full matrix is asserted by `test_xlog_run_single_modal_truth_table_exhaustive_matrix` through production `xlog run` |
| G91 possible recursion self-support | `43-g91-possible-recursion-self-support.xlog` | G91 recursive compatibility: positive `possible reach(X,Y)` over the co-evolving recursive target is treated as the G91 self-support assumption, not the FAEEL founded least-fixpoint atom. The full `xlog run` path returns and tests every tuple in `linked = vertex x vertex` for `{1,2,3}` |

Notes: examples with one epistemic output head use the single-plan path; examples
with independent epistemic output heads route through split GPU execution from
`xlog run`. A coalesced component with more than one epistemic output head sharing
a base modal predicate is joint-solved with multi-output materialization; each
head materialized against one shared accepted world view (examples 18, 21).

A modal literal over a determined head, one whose every defining rule ranges only
over invariant/EDB or already-determined relations, acyclically, is resolved by
stratified execution: the determined head is materialized (gated) into the
relation store as a lower stratum and the higher stratum gates against the
materialized base relation through the existing membership filter; no
resolve-into-body, no double-gating. The determined-closure is transitive across
ordinary derivations: an ordinary predicate is determined when every rule defining
it ranges only over determined/invariant relations (e.g. `r :- a` with `a` a
determined epistemic head), so a modal `know r` over such an `r` also stratifies
(example 24; the ordinary `r :- a` is deferred to the stratum where `a` is gated
base, computed once from the gated extension). This covers chained coupling
(example 17), recursion whose modal ranges over a determined head (example 25), and
transitive determined-ordinary coupling (example 24). A negated modal over an
invariant relation reduces cleanly to ordinary negation, so it executes in
recursive/coupling contexts (example 26).

Augmented-projection multi-head coupling (heads of differing arity sharing a base
modal) is resolved by per-head output projection (example 27): each coupled head
materializes from its own reduced relation buffer projected by its own public arity,
reading only the store/world-view boundary. A positive modal over an invariant
relation that is the sole binder of an output variable is resolved into a positive
ordinary join atom (a sound, machine-checked-invariant consequence), which also
makes single-head augmented modals over invariant relations executable.

Structured modal tuple-keys (example 23) are flattened element-wise onto the GPU
when they are finite and typed: a fixed-arity list `[a, b]` or compound `f(a, b)`
of scalar/Symbol-typed elements expands into one GPU key column per element and
reuses the existing device tuple-key matcher (bound variable → BOUND_OUTPUT,
anonymous `_` → WILDCARD, ground literal → GROUND). The flattened arity must equal
the modal relation's arity. Only genuinely unbounded or untyped structured forms (a
`cons` with a statically-unknown tail, a nested structure, a predref, an aggregate)
stay rejected, and they reject with a precise `ResourceExhausted` finiteness
diagnostic rather than a blanket unsupported-construct error (example 23b).

Nested modal operators are no longer rejected wholesale. A modal chain,
a sequence of `know`/`possible` operators over an atom, with optional leading,
interior, or atom-adjacent negation, normalizes to one single-level epistemic
literal. The operator adjacent to the atom wins (`know possible p ≡ possible p`,
`know know p ≡ know p`, `possible possible p ≡ possible p`); negation parity is
preserved, and atom-adjacent negation dualizes the innermost operator
(`know possible not p` normalizes to `not know p`, `know not possible p`
normalizes to `not possible p`). The normalization is an AST rewrite that routes
through the existing single-level epistemic path with no world-of-worlds evaluator
and holds in both modes; examples 13/13b/13c execute, 13d/13e show the inherited
FAEEL-vs-G91 mode split, 13g-13v exhaust all 64 two-operator negation cells under
FAEEL, and 13w* replays those cells under explicit G91.

The remaining fail-closed cases are genuinely unsafe, unbounded, or outside the
defined GPU-native execution families: cyclic modal coupling with no founded
order, unsafe unbound negated modal variables, and unbounded/untyped modal
tuple-keys (example 23b). Recursive modal semantics with a defined
founded/G91/WFS interpretation execute instead:
`22-recursion-through-modal-fixpoint.xlog` is the FAEEL founded recursive
fixpoint, `31-faeel-unfounded-self-support-empty-extension.xlog` is the defined
empty FAEEL extension, `32-g91-self-support-accepted.xlog` is the G91
self-support companion, `43-g91-possible-recursion-self-support.xlog` is G91
positive possible recursion, and the `33*` WFS fixtures are accepted cyclic
negated-modal GPU WFS.
