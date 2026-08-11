use assert_cmd::cargo::cargo_bin_cmd;
use cudarc::driver::result::mem_get_info;
use std::path::Path;
use tempfile::TempDir;
use xlog_cuda::CudaDevice;

#[test]
fn test_xlog_run_basic() {
    // cudarc panics on init when CUDA driver/runtime is unavailable; use xlog-cuda's safe wrapper.
    // Keep _device alive so the CUDA context survives through mem_get_info().
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    // CUDA context is alive via _device — memory query failure is now unexpected.
    let (_free, total) =
        mem_get_info().expect("mem_get_info should succeed while CudaDevice is alive");

    let total_mb = total / (1024 * 1024);
    if total_mb < 16_384 {
        println!("SKIPPED: GPU memory {} MB < required 16384 MB", total_mb);
        return;
    }

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let program = repo_root.join("examples/xlog/00-basics/01_tc_reachability.xlog");

    let mut cmd = cargo_bin_cmd!("xlog");
    cmd.args([
        "run",
        program.to_str().expect("valid path"),
        "--memory-mb",
        "16384",
    ]);
    cmd.assert().success();
}

#[test]
fn test_xlog_run_epistemic_examples() {
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let examples = [
        ("01-eir-boundary.xlog", "believed", "| 1  |"),
        ("02-g91-compatibility.xlog", "accepted", "rows: 1"),
        ("03-faeel-default.xlog", "accepted", "rows: 1"),
        ("04-gpt-candidate-filter.xlog", "accepted", "rows: 1"),
        ("05-splitting.xlog", "left", "rows: 1"),
        ("05-splitting.xlog", "right", "rows: 1"),
        // Epistemic executor showcase programs, each validated through the production
        // `xlog run` path with a deterministic output marker.
        ("06-eir-candidate-enumeration.xlog", "believed", "| 3  |"),
        ("07-tuple-key-membership.xlog", "matched", "| 3  | 3  |"),
        ("08-repeated-variable.xlog", "reflexive", "| 3  |"),
        ("09-joint-multi-epistemic.xlog", "both_known", "| 1  |"),
        ("10-epistemic-constraint.xlog", "accepted", "rows: 0"),
        ("11-faeel-foundedness.xlog", "founded", "rows: 1"),
        ("12-bound-variable-splitting.xlog", "both_known", "| 1  |"),
        ("12-bound-variable-splitting.xlog", "safe_alt", "| 2  |"),
        // Mixed per-row (`possible edge(X)`) and global (`know global_flag()`)
        // modal membership composed conjunctively.
        ("14-mixed-literal-membership.xlog", "reachable", "| 1  |"),
        ("14-mixed-literal-membership.xlog", "reachable", "| 3  |"),
        // Recursive epistemic fixpoint where ordinary recursion is gated by a modal
        // literal over an invariant relation and delegated to the existing
        // recursive/semi-naive engine. The derived multi-hop tuples prove the
        // fixpoint is not single-pass.
        // 3-vertex closure: base (1,2),(2,3) plus DERIVED (1,3).
        (
            "15-recursive-epistemic-closure.xlog",
            "reach",
            "| 1  | 2  |",
        ),
        (
            "15-recursive-epistemic-closure.xlog",
            "reach",
            "| 2  | 3  |",
        ),
        (
            "15-recursive-epistemic-closure.xlog",
            "reach",
            "| 1  | 3  |",
        ),
        // 4-chain: base hops plus 2-hop (1,3),(2,4) and the 3-hop DERIVED (1,4).
        ("15-recursive-epistemic-chain.xlog", "reach", "| 1  | 4  |"),
        ("15-recursive-epistemic-chain.xlog", "reach", "| 2  | 4  |"),
        // Cross-component epistemic coupling in the accepted safe case.
        // The ordinary head `report` consumes the epistemic-derived head `trusted`,
        // coupling two locally-splittable components through a derived dependency.
        // Single epistemic output head -> joint single-output path materializes the
        // gated `trusted = {1, 3}` exactly (node 2 is not vetted, so not known).
        ("16-cross-component-coupling.xlog", "trusted", "| 1  |"),
        ("16-cross-component-coupling.xlog", "trusted", "| 3  |"),
        // Stratified epistemic execution: chained coupling over a determined
        // epistemic-derived head. `trusted` (gated by `know vetted` over EDB) is a
        // strictly-lower stratum materialized into the store; `flagged :- know
        // trusted` then gates against the materialized `trusted = {1, 3}` via the
        // existing tuple-membership filter. flagged = {1, 3}; node 2 is gated out.
        (
            "17-cross-component-chained-stratified.xlog",
            "flagged",
            "| 1  |",
        ),
        (
            "17-cross-component-chained-stratified.xlog",
            "flagged",
            "| 3  |",
        ),
        // Cross-component epistemic joint solving with multiple outputs. Two
        // epistemic heads share one base modal predicate `q`
        // (SharedModalPredicate coalesce) but neither feeds the other, so the
        // coalesced multi-head component is JOINT-SOLVED: one candidate enumeration
        // + world-view validation over the COMBINED modals (`know q`, `possible q`),
        // then BOTH heads materialized against the SAME accepted world view. Unlike
        // the single-producer case, this program materializes and DISPLAYS
        // MULTIPLE coupled epistemic heads through `xlog run`:
        //   known(X) :- node(X),  know q(X).      -> {1, 2}
        //   maybe(X) :- color(X), possible q(X).  -> {2}
        (
            "18-cross-component-joint-shared-modal.xlog",
            "known",
            "| 1  |",
        ),
        (
            "18-cross-component-joint-shared-modal.xlog",
            "known",
            "| 2  |",
        ),
        (
            "18-cross-component-joint-shared-modal.xlog",
            "maybe",
            "| 2  |",
        ),
        // Access-control mixed modal gating. One accepted
        // rule combines a GLOBAL `know gateway_online()` gate, a PER-ROW
        // `possible cleared(P)`, and a PER-ROW negated `not possible revoked(P)`,
        // composed conjunctively. granted = {1, 2}; carol(3) and dave(4) are
        // gated out (absence checked below).
        ("19-access-control-mixed-modal.xlog", "granted", "| 1  |"),
        ("19-access-control-mixed-modal.xlog", "granted", "| 2  |"),
        // Supply-chain recursive epistemic fixpoint: ordinary recursion in
        // `sources_from` gated by `know certified` over the INVARIANT EDB. 10 tuples
        // incl. 4-hop derived
        // (1,5); supplier 6 is reachable only via an uncertified link and is
        // fully gated out (no `| 6  |` row; absence checked below).
        (
            "20-supply-chain-recursive-reach.xlog",
            "sources_from",
            "| 1  | 5  |",
        ),
        (
            "20-supply-chain-recursive-reach.xlog",
            "sources_from",
            "| 2  | 5  |",
        ),
        // Incident-triage joint solving: three epistemic heads share the base modal
        // `compromised`; all three modalities are used and all three heads materialize
        // against the same accepted world view.
        //   quarantine = {1,2}, watch = {2}, clear = {3,4,5}.
        (
            "21-incident-triage-joint-modal.xlog",
            "quarantine",
            "| 1  |",
        ),
        (
            "21-incident-triage-joint-modal.xlog",
            "quarantine",
            "| 2  |",
        ),
        ("21-incident-triage-joint-modal.xlog", "watch", "| 2  |"),
        ("21-incident-triage-joint-modal.xlog", "clear", "| 3  |"),
        ("21-incident-triage-joint-modal.xlog", "clear", "| 5  |"),
        // Stratified recursion over a determined epistemic-derived head:
        // `a` (gated by `know certified` over EDB) is materialized as a lower
        // stratum, then `reach` recurses via `know a` over the now-base relation.
        // reach = {(1,2),(2,3),(1,3)}; the derived (1,3) proves multi-hop fixpoint.
        (
            "25-recursion-over-determined-modal.xlog",
            "reach",
            "| 1  | 2  |",
        ),
        (
            "25-recursion-over-determined-modal.xlog",
            "reach",
            "| 2  | 3  |",
        ),
        (
            "25-recursion-over-determined-modal.xlog",
            "reach",
            "| 1  | 3  |",
        ),
        // Negated modal over an invariant relation in a recursive context:
        // `not know blocked(Y)` == ordinary `not blocked(Y)` anti-join. blocked
        // node 3 severs the chain -> reach = {(1,2),(3,4)}.
        (
            "26-negated-modal-over-invariant-recursive.xlog",
            "reach",
            "| 1  | 2  |",
        ),
        (
            "26-negated-modal-over-invariant-recursive.xlog",
            "reach",
            "| 3  | 4  |",
        ),
        // Augmented multi-head coupling with per-head output projection at differing
        // arity. Two epistemic heads share the base
        // modal `edge/2` (SharedModalPredicate coalesce -> ONE joint component) but
        // have DIFFERING public arity and BOTH need projection:
        //   one_hop(X)    :- node(X),  know edge(X, Y).      -> {1, 2}  (arity 1)
        //   pair(X, Y)    :- color(X), possible edge(X, Y).  -> {(2,20),(2,21),(3,30)}
        // Each head materializes from its OWN reduced buffer with its OWN row-filter,
        // projecting its OWN `public_head_arity` columns. one_hop drops the edge
        // target (projection load-bearing); pair is filtered by `color` (gate
        // load-bearing). Previously fail-closed with a cross-component coupling
        // diagnostic; now joint-solved.
        (
            "27-augmented-multi-head-differing-arity.xlog",
            "one_hop",
            "| 1  |",
        ),
        (
            "27-augmented-multi-head-differing-arity.xlog",
            "one_hop",
            "| 2  |",
        ),
        (
            "27-augmented-multi-head-differing-arity.xlog",
            "pair",
            "| 2  | 20 |",
        ),
        (
            "27-augmented-multi-head-differing-arity.xlog",
            "pair",
            "| 3  | 30 |",
        ),
        // Modal over an ordinary predicate transitively derived from a determined
        // epistemic head, stratified. `r :- a` (a determined
        // via `know p` over EDB) makes `r` determined; the lower stratum materializes
        // gated `a`, `r :- a` is computed over the base, and `know r` gates against the
        // materialized base `r`. b = node intersect r = {1} (node 2 has no `know p`).
        (
            "24-transitive-determined-modal-stratified.xlog",
            "b",
            "| 1  |",
        ),
        // Modal binding an output variable over a determined multi-column epistemic
        // head. `r(X,Y) :- edge(X,Y), know flag(X)` is determined;
        // `out(X) :- node(X), know r(X,Y)` binds the extra column Y. The lower stratum
        // materializes gated `r = {(1,2),(1,3)}`; the higher stratum gates `know r`
        // against that base, projecting away Y. out = {X in node : exists Y r(X,Y)} = {1}.
        ("28-determined-multicol-binding-modal.xlog", "out", "| 1  |"),
        // Negated modal (`not know` / `not possible`) over a determined
        // epistemic-derived head. `a` (gated by `know p` over EDB `p`) is
        // determined, so `not know a == not possible a == not a` (ordinary stratified
        // negation over the materialized base `a = {1,2}`). Both heads = {3} (node 3 is
        // the only node not in `a`); the equal results prove the modal equivalence.
        (
            "29-negated-modal-over-determined-derived.xlog",
            "q_know",
            "| 3  |",
        ),
        (
            "29-negated-modal-over-determined-derived.xlog",
            "q_poss",
            "| 3  |",
        ),
        // The `possible` counterpart: a binding `possible r(X, Y)` over a
        // determined multi-column epistemic head. Proves the
        // modal operator is irrelevant for a determined target (`possible r == know r ==
        // r`); stratifies identically to 28. out = {1} (gate load-bearing: ungated node
        // = {1,2,3}).
        ("30-possible-binding-over-determined.xlog", "out", "| 1  |"),
        // FAEEL unfounded self-support executes to its exact empty founded extension
        // instead of rejecting. `p() :- possible p()` is supported only
        // by circular modal self-support with no independent founded derivation, so `p`
        // is absent from the founded model -> rows: 0, exit 0.
        (
            "31-faeel-unfounded-self-support-empty-extension.xlog",
            "p",
            "rows: 0",
        ),
        // Mode-difference pair: the same unfounded self-support program under explicit
        // G91 compatibility mode accepts circular self-support -> rows: 1. The
        // FAEEL companion rows:0 vs G91 rows:1 is the exact semantic divergence.
        ("32-g91-self-support-accepted.xlog", "p", "rows: 1"),
        // Structured finite typed modal tuple-key. The one-element
        // list `[H]` flattens element-wise into `watched`'s scalar u32 key column,
        // so `know watched([H])` GATES `host` by `watched` membership. Load-bearing
        // (gated != ungated): only watched hosts survive -> alert = {1} (node 2 is
        // dropped). The structured-key flattening runs entirely on the GPU.
        ("23-compound-modal-key-membership.xlog", "alert", "| 1  |"),
        // Variable-keyed epistemic integrity constraint:
        // `:- know flagged(X).` ranges X EXISTENTIALLY over the modal relation's
        // tuple-key domain on the GPU world-view path. In ex34 `flagged` carries
        // MULTIPLE tuples {7, 9, 11}, so the existential body holds and the world
        // view is pruned -> report = {} (rows: 0). A ground `know flagged(c)`
        // could not express "some flagged value exists", so the variable range is
        // load-bearing.
        (
            "34-variable-keyed-constraint-prunes.xlog",
            "report",
            "rows: 0",
        ),
        // ex35 is the COMPANION: the SAME variable-keyed constraint does NOT prune
        // when `flagged` is EMPTY (no binding satisfies `know flagged(X)`), so the
        // world view survives -> report holds (rows: 1). ex34 rows:0 vs ex35 rows:1
        // is the EXACT load-bearing effect of the variable-keyed existential
        // constraint (same program shape, only the modal extension differs).
        (
            "35-variable-keyed-constraint-survives.xlog",
            "report",
            "rows: 1",
        ),
        // Multi-literal epistemic integrity constraint with distinct independent variables:
        // `:- know watch(X), know hot(Y).` factors to "watch non-empty AND hot
        // non-empty". Both relations are non-empty, so the conjunctive existential
        // body holds and the world view is pruned -> report absent (rows: 0). This
        // is the independent-existential conjunction (NOT a shared-variable join,
        // which fails closed as unimplemented scope by design). The empty-hot
        // survive flip (second literal load-bearing) is asserted in the device
        // suite (egb_e_distinct_variable_multi_literal_constraint_survives_*).
        (
            "36-multi-literal-distinct-var-constraint.xlog",
            "report",
            "rows: 0",
        ),
        // Nested modal operators execute via sound chain collapse. A modal chain
        // collapses according to the supported modal logic to the operator adjacent
        // to the atom;
        // a leading `not` distributes. The collapse routes through the existing
        // single-level epistemic path (no new evaluator). ex13 `know possible p()`
        // collapses to `possible p()` over EDB `p` (determined) -> q holds, rows: 1.
        ("13-nested-modal-chain-collapse.xlog", "q", "rows: 1"),
        // `know know reachable(X)` collapses to `know reachable(X)` and gates
        // `node` by `reachable`.
        // Load-bearing: gated = node intersect reachable = {1, 3} (node 2 dropped).
        ("13b-nested-modal-chain-filter.xlog", "gated", "| 1  |"),
        ("13b-nested-modal-chain-filter.xlog", "gated", "| 3  |"),
        // 13c: `not know possible blocked(X)` -- leading negation distributes,
        // chain collapses to `possible` -> `not possible blocked == not blocked`
        // (anti-join over EDB). Load-bearing: allowed = node \ blocked = {1, 3}.
        ("13c-nested-modal-chain-negated.xlog", "allowed", "| 1  |"),
        ("13c-nested-modal-chain-negated.xlog", "allowed", "| 3  |"),
        // `p() :- possible possible p()` collapses to `possible p()` and mirrors
        // the FAEEL unfounded self-support example.
        // The collapse forwards the per-mode foundedness difference: under FAEEL the
        // circular self-support is unfounded -> p absent -> rows: 0.
        (
            "13d-nested-modal-chain-faeel-unfounded.xlog",
            "p",
            "rows: 0",
        ),
        // The same chain program under explicit G91 compatibility mode accepts
        // self-support -> rows: 1. The FAEEL rows:0 vs G91 rows:1 pair is the exact
        // per-mode divergence of the collapsed chain.
        ("13e-nested-modal-chain-g91-accepted.xlog", "p", "rows: 1"),
        // 13f: `know not possible p()` dualizes to `not possible p()`. Because p()
        // is present, q is absent, but the program succeeds through `xlog run`.
        ("13f-nested-modal-interior-negation.xlog", "q", "rows: 0"),
        (
            "13f-nested-modal-interior-negation-absent.xlog",
            "q",
            "rows: 1",
        ),
        (
            "13fw-nested-modal-interior-negation-g91-present.xlog",
            "q",
            "rows: 0",
        ),
        (
            "13fw-nested-modal-interior-negation-g91-absent.xlog",
            "q",
            "rows: 1",
        ),
        // The committed matrix examples exhaust every two-operator modal chain over
        // `{know, possible}` with leading/interior/atom-adjacent negation placements,
        // split by operator pair, present/absent target, and derived/non-derived
        // outcome to keep each example within the accepted candidate-generation
        // bound. Together these slices cover all 64 source forms.
        (
            "13g-nested-modal-negation-matrix-know-know-present.xlog",
            "holds",
            "| 100",
        ),
        (
            "13g-nested-modal-negation-matrix-know-know-present.xlog",
            "holds",
            "| 103",
        ),
        (
            "13g-nested-modal-negation-matrix-know-know-present.xlog",
            "holds",
            "| 105",
        ),
        (
            "13g-nested-modal-negation-matrix-know-know-present.xlog",
            "holds",
            "| 106",
        ),
        (
            "13h-nested-modal-negation-matrix-know-know-absent.xlog",
            "holds",
            "| 1101",
        ),
        (
            "13h-nested-modal-negation-matrix-know-know-absent.xlog",
            "holds",
            "| 1102",
        ),
        (
            "13h-nested-modal-negation-matrix-know-know-absent.xlog",
            "holds",
            "| 1104",
        ),
        (
            "13h-nested-modal-negation-matrix-know-know-absent.xlog",
            "holds",
            "| 1107",
        ),
        (
            "13i-nested-modal-negation-matrix-know-possible-present.xlog",
            "holds",
            "| 200",
        ),
        (
            "13i-nested-modal-negation-matrix-know-possible-present.xlog",
            "holds",
            "| 203",
        ),
        (
            "13i-nested-modal-negation-matrix-know-possible-present.xlog",
            "holds",
            "| 205",
        ),
        (
            "13i-nested-modal-negation-matrix-know-possible-present.xlog",
            "holds",
            "| 206",
        ),
        (
            "13j-nested-modal-negation-matrix-know-possible-absent.xlog",
            "holds",
            "| 1201",
        ),
        (
            "13j-nested-modal-negation-matrix-know-possible-absent.xlog",
            "holds",
            "| 1202",
        ),
        (
            "13j-nested-modal-negation-matrix-know-possible-absent.xlog",
            "holds",
            "| 1204",
        ),
        (
            "13j-nested-modal-negation-matrix-know-possible-absent.xlog",
            "holds",
            "| 1207",
        ),
        (
            "13k-nested-modal-negation-matrix-possible-know-present.xlog",
            "holds",
            "| 300",
        ),
        (
            "13k-nested-modal-negation-matrix-possible-know-present.xlog",
            "holds",
            "| 303",
        ),
        (
            "13k-nested-modal-negation-matrix-possible-know-present.xlog",
            "holds",
            "| 305",
        ),
        (
            "13k-nested-modal-negation-matrix-possible-know-present.xlog",
            "holds",
            "| 306",
        ),
        (
            "13l-nested-modal-negation-matrix-possible-know-absent.xlog",
            "holds",
            "| 1301",
        ),
        (
            "13l-nested-modal-negation-matrix-possible-know-absent.xlog",
            "holds",
            "| 1302",
        ),
        (
            "13l-nested-modal-negation-matrix-possible-know-absent.xlog",
            "holds",
            "| 1304",
        ),
        (
            "13l-nested-modal-negation-matrix-possible-know-absent.xlog",
            "holds",
            "| 1307",
        ),
        (
            "13m-nested-modal-negation-matrix-possible-possible-present.xlog",
            "holds",
            "| 400",
        ),
        (
            "13m-nested-modal-negation-matrix-possible-possible-present.xlog",
            "holds",
            "| 403",
        ),
        (
            "13m-nested-modal-negation-matrix-possible-possible-present.xlog",
            "holds",
            "| 405",
        ),
        (
            "13m-nested-modal-negation-matrix-possible-possible-present.xlog",
            "holds",
            "| 406",
        ),
        (
            "13n-nested-modal-negation-matrix-possible-possible-absent.xlog",
            "holds",
            "| 1401",
        ),
        (
            "13n-nested-modal-negation-matrix-possible-possible-absent.xlog",
            "holds",
            "| 1402",
        ),
        (
            "13n-nested-modal-negation-matrix-possible-possible-absent.xlog",
            "holds",
            "| 1404",
        ),
        (
            "13n-nested-modal-negation-matrix-possible-possible-absent.xlog",
            "holds",
            "| 1407",
        ),
        (
            "13o-nested-modal-negation-matrix-know-know-present-nonderived.xlog",
            "holds",
            "+----+\n| c0 |\n+----+\n+----+",
        ),
        (
            "13p-nested-modal-negation-matrix-know-know-absent-nonderived.xlog",
            "holds",
            "+----+\n| c0 |\n+----+\n+----+",
        ),
        (
            "13q-nested-modal-negation-matrix-know-possible-present-nonderived.xlog",
            "holds",
            "+----+\n| c0 |\n+----+\n+----+",
        ),
        (
            "13r-nested-modal-negation-matrix-know-possible-absent-nonderived.xlog",
            "holds",
            "+----+\n| c0 |\n+----+\n+----+",
        ),
        (
            "13s-nested-modal-negation-matrix-possible-know-present-nonderived.xlog",
            "holds",
            "+----+\n| c0 |\n+----+\n+----+",
        ),
        (
            "13t-nested-modal-negation-matrix-possible-know-absent-nonderived.xlog",
            "holds",
            "+----+\n| c0 |\n+----+\n+----+",
        ),
        (
            "13u-nested-modal-negation-matrix-possible-possible-present-nonderived.xlog",
            "holds",
            "+----+\n| c0 |\n+----+\n+----+",
        ),
        (
            "13v-nested-modal-negation-matrix-possible-possible-absent-nonderived.xlog",
            "holds",
            "+----+\n| c0 |\n+----+\n+----+",
        ),
        // Negated modal `not know reach` over a genuinely recursive relation in a
        // strictly lower stratum than the negating head executes on the GPU production
        // path as ordinary stratified negation. reach =
        // transitive closure of link {(1,2),(2,3)} = {(1,2),(2,3),(1,3)}; unreachable =
        // node x node MINUS reach = 6 pairs. (1,1) is a self-pair excluded from reach
        // (modal gate load-bearing); (3,1) confirms the anti-join against the recursive
        // closure. The cyclic counterpart stays bounded by well-founded semantics.
        (
            "37-negated-modal-over-recursive-stratified.xlog",
            "unreachable",
            "| 1  | 1  |",
        ),
        (
            "37-negated-modal-over-recursive-stratified.xlog",
            "unreachable",
            "| 3  | 1  |",
        ),
        // Same-name multi-arity modal predicates resolve through full `xlog run`
        // by loading p/1 and p/2 under arity-qualified
        // store keys. This base example proves the load-bearing conjunction; the
        // exhaustive finite matrix lives in 42a*/42b* and is asserted by
        // test_xlog_run_same_name_multi_arity_exhaustive_matrix.
        (
            "42-same-name-multi-arity-modal-disambiguation.xlog",
            "a",
            "| 1  | 10 |",
        ),
        (
            "42-same-name-multi-arity-modal-disambiguation.xlog",
            "a",
            "!| 2  | 20 |",
        ),
        (
            "42-same-name-multi-arity-modal-disambiguation.xlog",
            "a",
            "!| 3  | 30 |",
        ),
        // G91 compatibility-mode possible recursion: positive `possible` over the
        // co-evolving recursive target is the compatibility self-support assumption. The full
        // `xlog run` path returns the complete 3 x 3 vertex relation.
        (
            "43-g91-possible-recursion-self-support.xlog",
            "linked",
            "| 1  | 1  |",
        ),
        (
            "43-g91-possible-recursion-self-support.xlog",
            "linked",
            "| 1  | 2  |",
        ),
        (
            "43-g91-possible-recursion-self-support.xlog",
            "linked",
            "| 1  | 3  |",
        ),
        (
            "43-g91-possible-recursion-self-support.xlog",
            "linked",
            "| 2  | 1  |",
        ),
        (
            "43-g91-possible-recursion-self-support.xlog",
            "linked",
            "| 2  | 2  |",
        ),
        (
            "43-g91-possible-recursion-self-support.xlog",
            "linked",
            "| 2  | 3  |",
        ),
        (
            "43-g91-possible-recursion-self-support.xlog",
            "linked",
            "| 3  | 1  |",
        ),
        (
            "43-g91-possible-recursion-self-support.xlog",
            "linked",
            "| 3  | 2  |",
        ),
        (
            "43-g91-possible-recursion-self-support.xlog",
            "linked",
            "| 3  | 3  |",
        ),
    ];

    for (example, expected_relation, expected_value) in examples {
        let program = repo_root.join("examples/epistemic").join(example);
        let output = cargo_bin_cmd!("xlog")
            .args([
                "run",
                program.to_str().expect("valid path"),
                "--memory-mb",
                "1024",
            ])
            .output()
            .expect("run xlog binary");
        assert!(
            output.status.success(),
            "{} failed:\nstdout:\n{}\nstderr:\n{}",
            example,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains(expected_relation),
            "{} did not emit relation {}:\n{}",
            example,
            expected_relation,
            stdout
        );
        if let Some(forbidden_value) = expected_value.strip_prefix('!') {
            assert!(
                !stdout.contains(forbidden_value),
                "{} emitted forbidden value marker {}:\n{}",
                example,
                forbidden_value,
                stdout
            );
        } else {
            assert!(
                stdout.contains(expected_value),
                "{} did not emit expected value marker {}:\n{}",
                example,
                expected_value,
                stdout
            );
        }
    }
}

#[test]
fn test_xlog_run_transitive_determined_modal_stratifies_accepted() {
    // A modal over an ordinary predicate transitively derived from a determined
    // epistemic head (`b :- know r`, `r :- a`, `a :- know p`) is accepted via
    // stratification. `r` is determined-in-principle (ordinary over
    // the determined `a`), so the lower stratum materializes the gated `a`, the
    // ordinary `r :- a` is computed over the materialized base (making `r` locally
    // invariant), and the higher stratum gates `know r` against the materialized base
    // `r` via the existing tuple-membership filter. EXACT tuples: with node={1,2}, p={1},
    // a=r={1}, b = node intersect r = {1}. The gate is load-bearing: dropping `know r`
    // would give b = node = {1,2}; the gate restricts b to {1}.
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    let (success, stdout, stderr) =
        run_epistemic_example("24-transitive-determined-modal-stratified.xlog");
    assert!(
        success,
        "transitive-determined stratified example must succeed, stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains('b'),
        "must surface the queried head `b`:\n{stdout}"
    );
    // b = {1} exactly: node 1 present (gate satisfied), node 2 absent (r empty there).
    assert!(
        stdout.contains("| 1  |"),
        "b must contain the gated node 1:\n{stdout}"
    );
    assert!(
        !stdout.contains("| 2  |"),
        "b must NOT contain node 2 (the `know r` gate is load-bearing):\n{stdout}"
    );
}

#[test]
fn test_xlog_run_determined_multicol_binding_modal_stratifies_accepted() {
    // A modal that binds an output variable over a determined multi-column epistemic
    // head is accepted via stratification. The full program:
    //   r(X, Y) :- edge(X, Y), know flag(X).   -- determined multi-column epistemic head
    //   out(X)  :- node(X), know r(X, Y).        -- modal binds the extra output column Y
    // Previously failed closed with `UnsafeVariable("Y")`. The lower stratum materializes
    // the gated `r = {(1,2),(1,3)}`; the higher stratum gates `know r` against that base
    // and projects away the binding column. EXACT: with node={1,2,3}, flag={1},
    // out = {X in node : exists Y r(X,Y)} = {1}. The gate is load-bearing: dropping the
    // modal literal gives out = node = {1,2,3}; the gate restricts out to {1}.
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    let (success, stdout, stderr) =
        run_epistemic_example("28-determined-multicol-binding-modal.xlog");
    assert!(
        success,
        "determined-multicol binding modal example must succeed, stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("out"),
        "must surface the queried head `out`:\n{stdout}"
    );
    // out = {1} exactly: node 1 present (r(1,_) exists), nodes 2 and 3 absent (no r rows).
    assert!(
        stdout.contains("| 1  |"),
        "out must contain the gated node 1:\n{stdout}"
    );
    assert!(
        !stdout.contains("| 2  |"),
        "out must NOT contain node 2 (the `know r(X,Y)` gate is load-bearing):\n{stdout}"
    );
    assert!(
        !stdout.contains("| 3  |"),
        "out must NOT contain node 3 (the `know r(X,Y)` gate is load-bearing):\n{stdout}"
    );
}

/// Helper: run an epistemic example through the production `xlog run` path and
/// return (success, stdout, stderr).
fn run_epistemic_example(example: &str) -> (bool, String, String) {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let program = repo_root.join("examples/epistemic").join(example);
    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run xlog binary");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn deterministic_modal_truth(form: &str, tuple_present: bool) -> bool {
    match form {
        "know" | "possible" => tuple_present,
        "not-know" | "not-possible" => !tuple_present,
        other => panic!("unknown modal form {other}"),
    }
}

fn same_name_modal_truth(form: &str, tuple_present: bool) -> bool {
    deterministic_modal_truth(form, tuple_present)
}

fn assert_holds_row_example(example: &str, should_hold: bool, label: &str) {
    let (ok, stdout, stderr) = run_epistemic_example(example);
    assert!(
        ok,
        "{label} example {example} must succeed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("holds"),
        "{label} example {example} must emit holds relation:\n{stdout}"
    );
    let has_row = stdout.contains("| 1  |");
    assert_eq!(
        has_row, should_hold,
        "{label} example {example} expected holds row presence {should_hold}:\n{stdout}"
    );
}

fn assert_same_name_matrix_example(example: &str, should_hold: bool) {
    assert_holds_row_example(example, should_hold, "same-name matrix");
}

fn assert_wfs_reach_shape(example: &str, label: &str, seed_present: bool) {
    let (ok, stdout, stderr) = run_epistemic_example(example);
    assert!(
        ok,
        "{label} must execute, stdout:\n{stdout}\nstderr:\n{stderr}"
    );

    let row_present = |x: u32, y: u32| {
        let compact = format!("| {x} | {y} |");
        let padded = format!("| {x}  | {y}  |");
        stdout.contains(&compact) || stdout.contains(&padded)
    };

    if seed_present {
        assert!(
            row_present(1, 2),
            "{label}: WFS true seed reach tuple must be present:\n{stdout}"
        );
    }

    for (x, y) in [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 1),
        (3, 2),
        (3, 3),
    ] {
        if seed_present && x == 1 && y == 2 {
            continue;
        }
        assert!(
            !row_present(x, y),
            "{label}: WFS undefined tuple ({x},{y}) must be absent:\n{stdout}"
        );
    }
}

#[test]
fn test_xlog_run_same_name_multi_arity_exhaustive_matrix() {
    // Exhaustive production-path matrix for same-name multi-arity modal predicates.
    //
    // Single-literal cells cover:
    //   arity {1,2} x modal {know,possible} x polarity {positive,negated}
    //   x queried-tuple state {present,absent}.
    //
    // Cross-arity cells cover every conjunction:
    //   unary modal form {know, possible, not-know, not-possible}
    //   x binary modal form {know, possible, not-know, not-possible}
    //   x unary queried-tuple state {present,absent}
    //   x binary queried-tuple state {present,absent}.
    //
    // Each committed example keeps nonmatching distractor facts in both p/1 and
    // p/2 so absent means "the queried tuple is absent", not "the relation or
    // schema is absent." This proves the arity-qualified source relation is the
    // load-bearing discriminator through full `xlog run`.
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    let forms = ["know", "possible", "not-know", "not-possible"];
    let states = [("present", true), ("absent", false)];

    for arity in [1, 2] {
        for form in forms.iter().copied() {
            for (state, tuple_present) in states.iter().copied() {
                let example =
                    format!("42a-same-name-multi-arity-literal-a{arity}-{form}-{state}.xlog");
                assert_same_name_matrix_example(
                    example.as_str(),
                    same_name_modal_truth(form, tuple_present),
                );
            }
        }
    }

    let mut cell_index = 0usize;
    for unary_form in forms.iter().copied() {
        for binary_form in forms.iter().copied() {
            for (unary_state, unary_present) in states.iter().copied() {
                for (binary_state, binary_present) in states.iter().copied() {
                    let example = format!(
                        "42b-same-name-multi-arity-cross-{cell_index:02}-{unary_form}-{unary_state}--{binary_form}-{binary_state}.xlog"
                    );
                    assert_same_name_matrix_example(
                        example.as_str(),
                        same_name_modal_truth(unary_form, unary_present)
                            && same_name_modal_truth(binary_form, binary_present),
                    );
                    cell_index += 1;
                }
            }
        }
    }
    assert_eq!(cell_index, 64);
}

#[test]
fn test_xlog_run_single_modal_truth_table_exhaustive_matrix() {
    // Exhaustive base truth table for one modal literal over a determined EDB target.
    //
    // Covers:
    //   mode {FAEEL, G91 compatibility} x modal form
    //   {know, possible, not-know, not-possible} x queried tuple {present, absent}.
    //
    // Both modes should agree for a determined target; the mode axis is still explicit so
    // future regressions cannot silently make only one mode work. Each committed example has
    // a nonmatching p(2) distractor, so "absent" means p(1) is absent, not that p/1 or its
    // schema vanished.
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    let forms = ["know", "possible", "not-know", "not-possible"];
    let states = [("present", true), ("absent", false)];

    for mode in ["faeel", "g91"] {
        for form in forms.iter().copied() {
            for (state, tuple_present) in states.iter().copied() {
                let example = format!("44a-single-modal-truth-table-{mode}-{form}-{state}.xlog");
                assert_holds_row_example(
                    example.as_str(),
                    deterministic_modal_truth(form, tuple_present),
                    "single-modal truth-table matrix",
                );
            }
        }
    }
}

#[test]
fn test_xlog_run_nested_modal_negation_matrix_g91_companion() {
    // Both-mode guard: the default-mode examples exhaust the finite two-operator
    // negation matrix. These explicit G91 compatibility-mode companions replay the
    // same source forms so the chain-collapse/duality rewrite is not accidentally
    // mode-specific.
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    for (example, expected_rows) in [
        (
            "13w-nested-modal-negation-matrix-g91-know-know-present.xlog",
            ["| 100", "| 103", "| 105", "| 106"],
        ),
        (
            "13w-nested-modal-negation-matrix-g91-know-know-absent.xlog",
            ["| 1101", "| 1102", "| 1104", "| 1107"],
        ),
        (
            "13w-nested-modal-negation-matrix-g91-know-possible-present.xlog",
            ["| 200", "| 203", "| 205", "| 206"],
        ),
        (
            "13w-nested-modal-negation-matrix-g91-know-possible-absent.xlog",
            ["| 1201", "| 1202", "| 1204", "| 1207"],
        ),
        (
            "13w-nested-modal-negation-matrix-g91-possible-know-present.xlog",
            ["| 300", "| 303", "| 305", "| 306"],
        ),
        (
            "13w-nested-modal-negation-matrix-g91-possible-know-absent.xlog",
            ["| 1301", "| 1302", "| 1304", "| 1307"],
        ),
        (
            "13w-nested-modal-negation-matrix-g91-possible-possible-present.xlog",
            ["| 400", "| 403", "| 405", "| 406"],
        ),
        (
            "13w-nested-modal-negation-matrix-g91-possible-possible-absent.xlog",
            ["| 1401", "| 1402", "| 1404", "| 1407"],
        ),
    ] {
        let (ok, stdout, stderr) = run_epistemic_example(example);
        assert!(
            ok,
            "{example} must execute:\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
        assert!(
            stdout.contains("holds"),
            "{example} must emit holds relation:\n{stdout}"
        );
        for expected in expected_rows {
            assert!(
                stdout.contains(expected),
                "{example} must emit expected G91 matrix row {expected}:\n{stdout}"
            );
        }
    }

    for example in [
        "13w-nested-modal-negation-matrix-g91-know-know-present-nonderived.xlog",
        "13w-nested-modal-negation-matrix-g91-know-know-absent-nonderived.xlog",
        "13w-nested-modal-negation-matrix-g91-know-possible-present-nonderived.xlog",
        "13w-nested-modal-negation-matrix-g91-know-possible-absent-nonderived.xlog",
        "13w-nested-modal-negation-matrix-g91-possible-know-present-nonderived.xlog",
        "13w-nested-modal-negation-matrix-g91-possible-know-absent-nonderived.xlog",
        "13w-nested-modal-negation-matrix-g91-possible-possible-present-nonderived.xlog",
        "13w-nested-modal-negation-matrix-g91-possible-possible-absent-nonderived.xlog",
    ] {
        let (ok, stdout, stderr) = run_epistemic_example(example);
        assert!(
            ok,
            "{example} must execute:\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
        assert!(
            stdout.contains("+----+\n| c0 |\n+----+\n+----+"),
            "{example} must keep the complementary G91 matrix cell empty:\n{stdout}"
        );
    }
}

#[test]
fn test_xlog_run_epistemic_examples_modal_gating_filters() {
    // Prove the accepted examples are real modal filtering, not store extraction:
    // the gated output must strictly omit the rows that fail the modal gate, so the
    // result differs from the ungated relation.
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    // 19 access-control mixed modal: granted = {1,2}. carol(3) fails
    // `possible cleared`; dave(4) fails `not possible revoked`. Both gated out.
    let (ok, stdout, stderr) = run_epistemic_example("19-access-control-mixed-modal.xlog");
    assert!(ok, "19 must succeed:\n{stdout}\n{stderr}");
    assert!(stdout.contains("granted"), "19 emits granted:\n{stdout}");
    assert!(
        stdout.contains("| 1  |") && stdout.contains("| 2  |"),
        "19 keeps 1,2:\n{stdout}"
    );
    assert!(
        !stdout.contains("| 3  |"),
        "19 must GATE OUT carol(3) (not cleared):\n{stdout}"
    );
    assert!(
        !stdout.contains("| 4  |"),
        "19 must GATE OUT dave(4) (revoked):\n{stdout}"
    );

    // 20 supply-chain recursive reach: certified closure OMITS every tuple
    // ending in supplier 6 (reachable only via an uncertified link).
    let (ok, stdout, stderr) = run_epistemic_example("20-supply-chain-recursive-reach.xlog");
    assert!(ok, "20 must succeed:\n{stdout}\n{stderr}");
    assert!(
        stdout.contains("| 1  | 5  |"),
        "20 derives 4-hop (1,5):\n{stdout}"
    );
    assert!(
        !stdout.contains("| 6  |"),
        "20 must GATE OUT supplier 6 (uncertified link):\n{stdout}"
    );

    // 21 incident triage joint-solving: watch = {2} only; monitored(3) is gated
    // out because compromised(3) is false (proves `possible` filters per-row).
    let (ok, stdout, stderr) = run_epistemic_example("21-incident-triage-joint-modal.xlog");
    assert!(ok, "21 must succeed:\n{stdout}\n{stderr}");
    assert!(stdout.contains("quarantine") && stdout.contains("watch") && stdout.contains("clear"));
    // watch keeps 2 but drops monitored(3). The `clear` head separately prints
    // a `| 3  |` row, so scope the absence check to the `watch` block.
    let watch_block = stdout
        .split("watch")
        .nth(1)
        .and_then(|s| s.split("clear").next())
        .expect("watch block present before clear block");
    assert!(
        watch_block.contains("| 2  |"),
        "21 watch keeps 2:\n{stdout}"
    );
    assert!(
        !watch_block.contains("| 3  |"),
        "21 watch must GATE OUT monitored(3) (not compromised):\n{stdout}"
    );

    // 17 stratified chained coupling: flagged = {1, 3}. node 2 is not vetted, so
    // not trusted (lower stratum gates it out), so not flagged -- proves the
    // higher stratum gates against the GATED (materialized) `trusted`, not the
    // ungated `node`.
    let (ok, stdout, stderr) = run_epistemic_example("17-cross-component-chained-stratified.xlog");
    assert!(ok, "17 must succeed:\n{stdout}\n{stderr}");
    assert!(stdout.contains("flagged"), "17 emits flagged:\n{stdout}");
    assert!(
        stdout.contains("| 1  |") && stdout.contains("| 3  |"),
        "17 keeps 1,3:\n{stdout}"
    );
    assert!(
        !stdout.contains("| 2  |"),
        "17 must GATE OUT node 2 (not vetted -> not trusted -> not flagged):\n{stdout}"
    );

    // 26 negated-modal-over-invariant recursion: reach = {(1,2),(3,4)}. The gate
    // `not know blocked(Y)` drops every tuple whose TARGET (second column) is the
    // blocked node 3, so (2,3) is absent and the chain through 3 never extends.
    // The ungated edge closure would be {(1,2),(2,3),(3,4),(1,3),(2,4),(1,4)}; the
    // gated result strictly OMITS all tuples ending in 3 AND all multi-hop tuples
    // that must pass through 3, proving the negated modal actually filters.
    let (ok, stdout, stderr) =
        run_epistemic_example("26-negated-modal-over-invariant-recursive.xlog");
    assert!(ok, "26 must succeed:\n{stdout}\n{stderr}");
    assert!(
        stdout.contains("| 1  | 2  |") && stdout.contains("| 3  | 4  |"),
        "26 keeps (1,2),(3,4):\n{stdout}"
    );
    assert!(
        !stdout.contains("| 2  | 3  |"),
        "26 must GATE OUT (2,3): target node 3 is blocked:\n{stdout}"
    );
    assert!(
        !stdout.contains("| 1  | 3  |") && !stdout.contains("| 1  | 4  |"),
        "26 must OMIT multi-hop tuples that pass through blocked node 3:\n{stdout}"
    );
}

#[test]
fn test_xlog_run_recursion_through_modal_computes_founded_fixpoint() {
    // A positive modal over a relation that co-evolves with the program's ordinary
    // recursion executes to its FAEEL founded least fixpoint instead of rejecting.
    // The modal feeds a non-mirror relation `trust`, so the
    // modal gate is load-bearing: founded reach = {(1,2),(1,3)}.
    //   (1,2): seed-founded.
    //   (1,3): reach(1,2) + trust(2,3); trust(2,3) founded because know reach(1,2)
    //          holds. This tuple exists ONLY because the modal co-evolves into the
    //          recursion -- it is absent from a base-only result.
    // The unfounded candidate trust(3,1) (gated by `know reach(3,3)`, unfounded) is
    // correctly excluded, so (1,1) never appears (it would under an ungated reading).
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    let (ok, stdout, stderr) = run_epistemic_example("22-recursion-through-modal-fixpoint.xlog");
    assert!(
        ok,
        "recursive epistemic fixpoint must execute, stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    // EXACT founded tuples: {(1,2),(1,3)}.
    assert!(
        stdout.contains("| 1  | 2  |"),
        "founded must contain (1,2):\n{stdout}"
    );
    assert!(
        stdout.contains("| 1  | 3  |"),
        "founded must contain (1,3) (co-evolution through the modal):\n{stdout}"
    );
    // NOT base-only: (1,3) present above proves the modal added a founded tuple.
    // NOT ungated: the unfounded reach(3,3) must not self-fulfill trust(3,1), so
    // (1,1) is absent.
    assert!(
        !stdout.contains("| 1  | 1  |"),
        "foundedness must EXCLUDE (1,1) (unfounded reach(3,3) cannot self-fulfill \
         trust(3,1)):\n{stdout}"
    );
}

#[test]
fn test_xlog_run_negated_modal_through_recursion_uses_gpu_wfs_engine() {
    // A negated modal whose target cycles through recursion via negation executes
    // through the GPU-backed WFS alternating-fixpoint path.
    //
    // Covers every cyclic-negated WFS modal cell across:
    //   mode {FAEEL, G91 compatibility}
    //   x modal form {not-know, not-possible}
    //   x seed state {present,absent}
    //   x ordinary EDB negation {absent,present-in-SCC}.
    //
    // True atoms are emitted; false/undefined atoms are absent.
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    assert_wfs_reach_shape(
        "33-negated-modal-through-recursion-wfs.xlog",
        "canonical FAEEL not-know WFS fixture",
        true,
    );
    assert_wfs_reach_shape(
        "33b-negated-modal-through-recursion-wfs-with-edb-negation.xlog",
        "canonical FAEEL WFS with ordinary EDB negation",
        true,
    );

    for mode in ["faeel", "g91"] {
        for form in ["not-know", "not-possible"] {
            for (state, seed_present) in [("seed-present", true), ("seed-absent", false)] {
                let plain = format!(
                    "33c-negated-modal-through-recursion-wfs-matrix-{mode}-{form}-{state}.xlog"
                );
                let plain_label = format!("{mode} {form} {state} plain cyclic WFS matrix cell");
                assert_wfs_reach_shape(&plain, &plain_label, seed_present);

                let edb = format!(
                    "33d-negated-modal-through-recursion-wfs-edb-negation-matrix-{mode}-{form}-{state}.xlog"
                );
                let edb_label = format!("{mode} {form} {state} WFS plus EDB-negation matrix cell");
                assert_wfs_reach_shape(&edb, &edb_label, seed_present);
            }
        }
    }

    for mode in ["faeel", "g91"] {
        for form in ["not-know", "not-possible"] {
            for (state, should_keep_seed) in [("allowed", true), ("banned", false)] {
                let example = format!(
                    "33e-negated-modal-through-recursion-wfs-edb-negation-load-bearing-{mode}-{form}-{state}.xlog"
                );
                let label =
                    format!("{mode} {form} {state} WFS plus load-bearing EDB-negation cell");
                assert_wfs_reach_shape(&example, &label, should_keep_seed);
            }
        }
    }
}

#[test]
fn test_xlog_run_compound_modal_key_reports_typed_epistemic_diagnostic() {
    // Structured finite typed modal keys (fixed-arity list/compound of scalar
    // elements) are accepted and flattened onto the GPU. What stays rejected is a
    // genuinely unbounded structured key: a `cons` `[H | T]` whose tail length is
    // not statically fixed has no finite, typed GPU key-column set. It must FAIL
    // CLOSED with a precise FINITENESS
    // (resource) diagnostic, NOT a blanket "unsupported construct".
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    let (ok, stdout, stderr) = run_epistemic_example("23b-unbounded-cons-modal-key-rejected.xlog");
    assert!(
        !ok,
        "unbounded-cons modal-key example must fail closed, stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    // Honest finiteness/resource bound, NOT "UnsupportedEpistemicConstruct".
    assert!(
        stderr.contains("ResourceExhausted"),
        "unbounded structured key must fail with a finiteness/resource diagnostic:\n{stderr}"
    );
    assert!(
        !stderr.contains("UnsupportedEpistemicConstruct"),
        "rejection must be a precise finiteness bound, not a blanket unsupported construct:\n{stderr}"
    );
    assert!(
        stderr.contains("cons") && stderr.contains("tail length is not statically fixed"),
        "diagnostic must name the unbounded `cons` tail as the finiteness wall:\n{stderr}"
    );
    assert!(
        stderr.contains("fixed-arity list literal"),
        "diagnostic must point the user at the finite-typed alternative:\n{stderr}"
    );
}

#[test]
fn test_xlog_run_diagonal_modal_constraint_prunes() {
    // Diagonal modal constraint: `:- know route(X, X).` repeats X across its own key
    // columns. It is resolved by a sound program-level desugaring (ordinary diagonal extraction
    // `__epi_diag_0(X) :- route(X, X)` + single-occurrence `:- know __epi_diag_0(X)`), which
    // routes through the existing variable-keyed world-view constraint path and PRUNES the
    // world view to empty when a self-loop exists -- no new kernel. 38 has a self-loop
    // `route(1, 1)` -> `safe` pruned empty; 39 has none -> `safe = {5}`. The flip is
    // load-bearing (only the diagonal tuple differs between the two programs).
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    let (ok, stdout, stderr) = run_epistemic_example("38-diagonal-modal-constraint.xlog");
    assert!(
        ok,
        "38 must succeed (Ok with a pruned-empty world view, NOT an error):\n{stdout}\n{stderr}"
    );
    assert!(
        stdout.contains("safe"),
        "38 emits the safe relation:\n{stdout}"
    );
    assert!(
        !stdout.contains("| 5  |"),
        "38 must PRUNE safe to empty -- the self-loop route(1,1) fires `:- know route(X,X)`:\n{stdout}"
    );

    let (ok, stdout, stderr) = run_epistemic_example("39-diagonal-modal-constraint-satisfied.xlog");
    assert!(ok, "39 must succeed:\n{stdout}\n{stderr}");
    assert!(
        stdout.contains("| 5  |"),
        "39 keeps safe = {{5}} (no self-loop -> the diagonal constraint is satisfied):\n{stdout}"
    );
}

#[test]
fn test_xlog_run_shared_variable_join_constraints_prune() {
    // Shared-variable epistemic constraints: `:- know p(X), possible q(X).` (40,
    // intersection) and `:- q(X), not know p(X).` (41, set difference) are resolved
    // by a sound program-level desugaring (ordinary join/difference extraction
    // `__epi_join_0(X) :- ...` + single-occurrence `:- know __epi_join_0(X)`), routing through
    // the existing variable-keyed world-view constraint path -- no new kernel. In both
    // programs the helper relation is non-empty (p∩q={2}; q\p={3}), so the constraint fires
    // and `report` is PRUNED to empty. Removing the constraint would leave `report = {5}`
    // (gate(5) is gated by `know gate` -> survives), so the prune is load-bearing.
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    for (example, why) in [
        (
            "40-shared-variable-join-constraint.xlog",
            "join p∩q={2} fires -> report pruned empty",
        ),
        (
            "41-negated-difference-constraint.xlog",
            "difference q\\p={3} fires -> report pruned empty",
        ),
    ] {
        let (ok, stdout, stderr) = run_epistemic_example(example);
        assert!(
            ok,
            "{example} must succeed (Ok with a pruned-empty world view):\n{stdout}\n{stderr}"
        );
        assert!(
            stdout.contains("report"),
            "{example} emits report:\n{stdout}"
        );
        assert!(!stdout.contains("| 5  |"), "{example}: {why}:\n{stdout}");
    }
}

#[test]
fn test_xlog_run_faeel_unfounded_self_support_executes_to_empty_extension() {
    // A self-supported possible rule (`p() :- possible p().`) with no independent
    // founded support is unfounded under
    // default FAEEL, so `p` is ABSENT from the founded model. The program EXECUTES to
    // its exact empty founded extension (`rows: 0`, exit 0) -- it is NOT rejected with
    // an unsupported-construct error.
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    let (ok, stdout, stderr) =
        run_epistemic_example("31-faeel-unfounded-self-support-empty-extension.xlog");
    assert!(
        ok,
        "FAEEL-unfounded self-support example must EXECUTE (exit 0), not fail closed; \
         stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    // Exact empty founded extension.
    assert!(
        stdout.contains("p\nrows: 0"),
        "FAEEL unfounded self-support must materialize the EMPTY founded extension \
         (rows: 0):\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    // It must NOT be the old rejection.
    assert!(
        !stderr.contains("FAEEL foundedness guard"),
        "FAEEL unfounded self-support must no longer be rejected:\n{stderr}"
    );

    // Mode-difference evidence: the same program under explicit G91 compatibility
    // mode accepts circular self-support and emits rows: 1.
    let (g91_ok, g91_stdout, g91_stderr) =
        run_epistemic_example("32-g91-self-support-accepted.xlog");
    assert!(
        g91_ok,
        "G91 self-support pair must execute (exit 0); stdout:\n{g91_stdout}\nstderr:\n{g91_stderr}"
    );
    assert!(
        g91_stdout.contains("rows: 1"),
        "G91 mode must ACCEPT circular self-support (rows: 1), the FAEEL-vs-G91 mode \
         difference:\nstdout:\n{g91_stdout}\nstderr:\n{g91_stderr}"
    );
}

#[test]
fn test_xlog_run_warns_on_ignored_imported_module_pragma() {
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    let root = std::env::temp_dir().join(format!("xlog_run_import_pragma_{}", std::process::id()));
    let modules = root.join("modules");
    std::fs::create_dir_all(&modules).expect("create module dir");
    std::fs::write(
        modules.join("inputs.xlog"),
        "#pragma magic_sets = auto\npred edge(u32, u32).\nedge(1, 2).\n",
    )
    .expect("write inputs module");
    let program = root.join("main.xlog");
    std::fs::write(
        &program,
        "use inputs.\npred reach(u32, u32).\nreach(X, Y) :- edge(X, Y).\n?- reach(X, Y).\n",
    )
    .expect("write main program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid path"),
            "--module-path",
            modules.to_str().expect("valid module path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run xlog with imported-module pragma");
    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains(
            "warning[W0510]: `#pragma magic_sets` in imported module `inputs` is ignored"
        ),
        "{stderr}"
    );
    assert_eq!(stderr.matches("warning[W0510]").count(), 1, "{stderr}");
}

#[test]
fn test_xlog_run_unions_compatible_predicates_from_separate_modules() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };

    let fixture = TempDir::new().expect("create fixture directory");
    std::fs::write(fixture.path().join("first.xlog"), "shared(from_first).\n")
        .expect("write first module");
    std::fs::write(fixture.path().join("second.xlog"), "shared(from_second).\n")
        .expect("write second module");
    let program = fixture.path().join("main.xlog");
    std::fs::write(&program, "use first.\nuse second.\n?- shared(X).\n")
        .expect("write main program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--module-path",
            fixture.path().to_str().expect("valid module path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run xlog with compatible predicate contributions");

    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert_eq!(stdout.matches("from_first").count(), 1, "{stdout}");
    assert_eq!(stdout.matches("from_second").count(), 1, "{stdout}");
}

#[test]
fn test_xlog_run_unions_body_inferred_predicates_from_separate_modules() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };

    let fixture = TempDir::new().expect("create fixture directory");
    std::fs::write(
        fixture.path().join("first.xlog"),
        "shared(X) :- first_mid(X).\n\
         first_mid(X) :- first_source(X).\n\
         first_source(from_first).\n",
    )
    .expect("write first module");
    std::fs::write(
        fixture.path().join("second.xlog"),
        "shared(X) :- second_mid(X).\n\
         second_mid(X) :- second_source(X).\n\
         second_source(from_second).\n",
    )
    .expect("write second module");
    let program = fixture.path().join("main.xlog");
    std::fs::write(&program, "use first.\nuse second.\n?- shared(X).\n")
        .expect("write main program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--module-path",
            fixture.path().to_str().expect("valid module path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run xlog with body-inferred predicate contributions");

    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert_eq!(stdout.matches("from_first").count(), 1, "{stdout}");
    assert_eq!(stdout.matches("from_second").count(), 1, "{stdout}");
}

#[test]
fn test_xlog_run_uses_imported_schema_for_undeclared_contributions() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };

    let fixture = TempDir::new().expect("create fixture directory");
    std::fs::write(fixture.path().join("schema.xlog"), "pred shared(i64).\n")
        .expect("write schema module");
    std::fs::write(fixture.path().join("small.xlog"), "shared(1).\n")
        .expect("write small-value module");
    std::fs::write(fixture.path().join("wide.xlog"), "shared(5000000000).\n")
        .expect("write wide-value module");
    let program = fixture.path().join("main.xlog");
    std::fs::write(
        &program,
        "use schema.\nuse small.\nuse wide.\n?- shared(X).\n",
    )
    .expect("write main program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--module-path",
            fixture.path().to_str().expect("valid module path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run xlog with an imported predicate schema");

    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let mut values = stdout
        .lines()
        .filter_map(|line| line.trim().trim_matches('|').trim().parse::<i64>().ok())
        .collect::<Vec<_>>();
    values.sort_unstable();
    assert_eq!(values, vec![1, 5_000_000_000], "{stdout}");
}

#[test]
fn test_xlog_run_epistemic_imports_keep_undeclared_predicate_arities_distinct() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };

    let fixture = TempDir::new().expect("create fixture directory");
    std::fs::write(fixture.path().join("unary.xlog"), "shared(1).\n").expect("write unary module");
    std::fs::write(fixture.path().join("binary.xlog"), "shared(one, two).\n")
        .expect("write binary module");
    let program = fixture.path().join("main.xlog");
    std::fs::write(
        &program,
        "#pragma epistemic_mode = faeel\n\
         use unary.\n\
         use binary.\n\
         accepted() :- know shared(1), know shared(one, two).\n\
         ?- accepted().\n",
    )
    .expect("write main program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--module-path",
            fixture.path().to_str().expect("valid module path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run xlog with imported multi-arity predicates");

    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("accepted"), "{stdout}");
    assert!(stdout.contains("rows: 1"), "{stdout}");
}

#[test]
fn test_xlog_run_projects_declared_head_constants_with_the_relation_schema() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };

    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("typed_head_constant.xlog");
    std::fs::write(
        &program,
        "pred source(f64).\n\
         pred real(f64).\n\
         seed().\n\
         source(2.0).\n\
         real(1) :- seed().\n\
         real(X) :- source(X).\n\
         ?- real(X).\n",
    )
    .expect("write typed head-constant program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run xlog with a typed rule-head constant");

    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let mut values = stdout
        .lines()
        .filter_map(|line| line.trim().trim_matches('|').trim().parse::<f64>().ok())
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    assert_eq!(values, vec![1.0, 2.0], "{stdout}");
}

#[test]
fn test_xlog_run_uses_exact_multi_arity_schema_for_augmented_epistemic_head() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };

    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("exact_epistemic_schema.xlog");
    std::fs::write(
        &program,
        "#pragma epistemic_mode = faeel\n\
         pred node(symbol).\n\
         pred source(symbol, i64).\n\
         pred source(u32).\n\
         pred result(symbol).\n\
         node(key).\n\
         source(key, 5000000000).\n\
         source(1).\n\
         result(X) :- node(X), know source(X, Y).\n\
         ?- result(X).\n",
    )
    .expect("write exact epistemic schema program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run xlog with an augmented epistemic head");

    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("result"), "{stdout}");
    assert_eq!(stdout.matches("key").count(), 1, "{stdout}");
}

#[test]
fn test_xlog_run_epistemic_schema_identity_and_stratum_regressions() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };

    let fixtures = [
        (
            "inferred_hidden_column.xlog",
            "#pragma epistemic_mode = faeel\n\
             pred node(symbol).\n\
             pred result(symbol).\n\
             node(key).\n\
             raw(key, 5000000000).\n\
             edge(X, Y) :- raw(X, Y).\n\
             result(X) :- node(X), know edge(X, Y).\n\
             ?- result(X).\n",
            vec!["result", "key"],
        ),
        (
            "arithmetic_hidden_column.xlog",
            "#pragma epistemic_mode = faeel\n\
             pred node(symbol).\n\
             pred allowed(u64).\n\
             pred result(symbol).\n\
             node(key).\n\
             allowed(2).\n\
             result(X) :- node(X), Y is cast(1, u64), not know allowed(Y).\n\
             ?- result(X).\n",
            vec!["result", "key"],
        ),
        (
            "constraint_arity_census.xlog",
            "#pragma epistemic_mode = faeel\n\
             p(a).\n\
             result(X) :- p(X), know p(X).\n\
             :- p(X, Y).\n\
             ?- result(X).\n",
            vec!["result", "a"],
        ),
        (
            "recursive_stratum_schema.xlog",
            "#pragma epistemic_mode = faeel\n\
             pred node(u32).\n\
             pred edge(u32, u32).\n\
             pred accepted_edge(u32, u32).\n\
             pred reach(u32, u32).\n\
             node(1). node(2). node(3).\n\
             edge(1, 2). edge(2, 3).\n\
             accepted_edge(X, Y) :- node(X), node(Y), know edge(X, Y).\n\
             reach(X, Y) :- node(X), node(Y), know accepted_edge(X, Y).\n\
             reach(X, Z) :- reach(X, Y), node(Z), know accepted_edge(Y, Z).\n\
             ?- reach(X, Z).\n",
            vec!["| 1  | 2  |", "| 2  | 3  |", "| 1  | 3  |"],
        ),
    ];

    let fixture = TempDir::new().expect("create fixture directory");
    for (name, source, expected) in fixtures {
        let program = fixture.path().join(name);
        std::fs::write(&program, source).expect("write epistemic regression program");
        let output = cargo_bin_cmd!("xlog")
            .args([
                "run",
                program.to_str().expect("valid program path"),
                "--memory-mb",
                "1024",
            ])
            .output()
            .expect("run epistemic regression program");
        assert!(
            output.status.success(),
            "xlog run failed for {name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
        for marker in expected {
            assert!(
                stdout.contains(marker),
                "missing {marker:?} in {name}:\n{stdout}"
            );
        }
    }
}

#[test]
fn test_xlog_run_scopes_joint_ground_modal_gates_to_their_output_heads() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };

    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("scoped_joint_ground_gates.xlog");
    std::fs::write(
        &program,
        "#pragma epistemic_mode = faeel\n\
         p(b).\n\
         a() :- know p(a).\n\
         b() :- know p(b).\n\
         ?- a().\n\
         ?- b().\n",
    )
    .expect("write scoped ground-gate program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run scoped ground-gate program");
    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("a\nrows: 0"), "{stdout}");
    assert!(stdout.contains("b\nrows: 1"), "{stdout}");
}

#[test]
fn test_xlog_run_equivalent_modal_filters_preserve_union_rows() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };

    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("equivalent_modal_filter_union.xlog");
    std::fs::write(
        &program,
        "#pragma epistemic_mode = faeel\n\
         pred target(u32).\n\
         pred left(u32).\n\
         pred right(u32).\n\
         pred result(u32).\n\
         target(1).\n\
         target(2).\n\
         left(1).\n\
         right(2).\n\
         result(X) :- left(X), know target(X).\n\
         result(Y) :- right(Y), possible target(Y).\n\
         ?- result(Value).\n",
    )
    .expect("write equivalent modal-filter union program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run equivalent modal-filter union program");
    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("| 1  |"), "{stdout}");
    assert!(stdout.contains("| 2  |"), "{stdout}");
}

#[test]
fn test_xlog_run_ignores_removed_unfounded_arity_in_relation_identity() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };

    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("removed_unfounded_arity.xlog");
    std::fs::write(
        &program,
        "#pragma epistemic_mode = faeel\n\
         dom(a).\n\
         p(a, b).\n\
         p(X) :- dom(X), possible p(X).\n\
         ?- p(X, Y).\n",
    )
    .expect("write removed-unfounded-arity program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run removed-unfounded-arity program");
    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("| a  | b  |"), "{stdout}");
}

#[test]
fn test_xlog_run_excludes_removed_unfounded_rules_from_active_gpu_plan() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };

    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("active_plan_after_foundedness.xlog");
    std::fs::write(
        &program,
        "#pragma epistemic_mode = faeel\n\
         dom(a).\n\
         p(a, b).\n\
         q(c).\n\
         p(X) :- dom(X), possible p(X).\n\
         result(X) :- q(X), know q(X).\n\
         ?- result(X).\n",
    )
    .expect("write active plan foundedness program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run active plan foundedness program");
    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("| c  |"), "{stdout}");
}

#[test]
fn test_xlog_run_preserves_predecessor_supported_recursive_modal_rule() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };

    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture
        .path()
        .join("predecessor_supported_recursive_modal.xlog");
    std::fs::write(
        &program,
        "#pragma epistemic_mode = faeel\n\
         pred edge(u32, u32).\n\
         pred p(u32).\n\
         edge(1, 2).\n\
         p(1).\n\
         p(Y) :- p(X), edge(X, Y), possible p(X).\n\
         ?- p(Y).\n",
    )
    .expect("write predecessor-supported recursive modal program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run predecessor-supported recursive modal program");
    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("| 1  |"), "{stdout}");
    assert!(stdout.contains("| 2  |"), "{stdout}");
}

#[test]
fn test_xlog_run_propagates_founded_modal_predecessor() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };

    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("founded_modal_predecessor.xlog");
    std::fs::write(
        &program,
        "#pragma epistemic_mode = faeel\n\
         pred edge(u32, u32).\n\
         pred p(u32).\n\
         edge(1, 2).\n\
         p(1).\n\
         p(Y) :- edge(X, Y), possible p(X).\n\
         ?- p(Y).\n",
    )
    .expect("write founded modal-predecessor program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run founded modal-predecessor program");
    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("| 1  |"), "{stdout}");
    assert!(stdout.contains("| 2  |"), "{stdout}");
}

#[test]
fn test_xlog_run_excludes_unfounded_modal_tuple_cycle() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };

    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("unfounded_modal_tuple_cycle.xlog");
    std::fs::write(
        &program,
        "#pragma epistemic_mode = faeel\n\
         pred pair(u32, u32).\n\
         pred p(u32, u32).\n\
         pair(1, 2).\n\
         pair(2, 1).\n\
         p(X, Y) :- pair(X, Y), possible p(Y, X).\n\
         ?- p(X, Y).\n",
    )
    .expect("write unfounded modal tuple-cycle program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run unfounded modal tuple-cycle program");
    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.starts_with("p\n"), "{stdout}");
    assert!(!stdout.contains("| 1  | 2  |"), "{stdout}");
    assert!(!stdout.contains("| 2  | 1  |"), "{stdout}");
}

#[test]
fn test_xlog_run_solves_multi_relation_modal_cycle_from_founded_seed() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };

    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture
        .path()
        .join("founded_multi_relation_modal_cycle.xlog");
    std::fs::write(
        &program,
        "#pragma epistemic_mode = faeel\n\
         pred seed().\n\
         pred p().\n\
         pred q().\n\
         seed().\n\
         p() :- seed().\n\
         q() :- possible p().\n\
         p() :- possible q().\n\
         ?- p().\n\
         ?- q().\n",
    )
    .expect("write founded multi-relation modal-cycle program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run founded multi-relation modal-cycle program");
    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("p\nrows: 1"), "{stdout}");
    assert!(stdout.contains("q\nrows: 1"), "{stdout}");
}

#[test]
fn test_xlog_run_aggregate_outputs_match_declared_runtime_schemas() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };

    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("aggregate_runtime_schemas.xlog");
    std::fs::write(
        &program,
        "pred edge(u32, u64).\n\
         pred summary(u32, u64, u64, u64).\n\
         edge(1, 5000000000).\n\
         edge(1, 6000000000).\n\
         summary(X, count(Y), min(Y), max(Y)) :- edge(X, Y).\n\
         ?- summary(X, Count, Minimum, Maximum).\n",
    )
    .expect("write aggregate schema program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run aggregate schema program");
    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(
        stdout.lines().any(|line| {
            line.split('|')
                .any(|field| field.trim().parse::<u64>() == Ok(2))
        }),
        "{stdout}"
    );
    assert!(stdout.contains("5000000000"), "{stdout}");
    assert!(stdout.contains("6000000000"), "{stdout}");
}

#[test]
fn test_xlog_run_g91_exact_head_possible_union_preserves_self_supported_rows() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };

    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("g91_exact_head_possible_union.xlog");
    std::fs::write(
        &program,
        "#pragma epistemic_mode = g91\n\
         pred seed(u32).\n\
         pred node(u32).\n\
         pred p(u32).\n\
         seed(1).\n\
         node(2).\n\
         p(X) :- seed(X).\n\
         p(X) :- node(X), possible p(X).\n\
         ?- p(X).\n",
    )
    .expect("write G91 exact-head possibility program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run G91 exact-head possibility program");
    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("| 1  |"), "{stdout}");
    assert!(stdout.contains("| 2  |"), "{stdout}");
}

#[test]
fn test_xlog_run_faeel_constrained_support_excludes_unfounded_union_rows() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };

    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("constrained_founded_support.xlog");
    std::fs::write(
        &program,
        "#pragma epistemic_mode = faeel\n\
         pred seed(u32).\n\
         pred p(u32).\n\
         seed(1).\n\
         seed(2).\n\
         p(X) :- seed(X), X = 1.\n\
         p(X) :- seed(X), possible p(X).\n\
         ?- p(X).\n",
    )
    .expect("write constrained founded-support program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run constrained founded-support program");
    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("| 1  |"), "{stdout}");
    assert!(!stdout.contains("| 2  |"), "{stdout}");
}

#[test]
fn test_xlog_run_preserves_non_bijective_modal_union_rows_in_fixpoint() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };

    let fixture = TempDir::new().expect("create fixture directory");
    let program = fixture.path().join("repeated_head_variable_union.xlog");
    std::fs::write(
        &program,
        "#pragma epistemic_mode = faeel\n\
         pred domain(symbol).\n\
         pred other(symbol, symbol).\n\
         pred p(symbol, symbol).\n\
         domain(a).\n\
         other(c, d).\n\
         p(X, X) :- domain(X).\n\
         p(A, B) :- other(A, B).\n\
         p(X, X) :- domain(X), know p(X, X).\n\
         ?- p(A, B).\n",
    )
    .expect("write non-bijective modal-union program");

    let output = cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run non-bijective modal-union program");
    assert!(
        output.status.success(),
        "xlog run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("| a  | a  |"), "{stdout}");
    assert!(stdout.contains("| c  | d  |"), "{stdout}");
}

fn run_inline_xlog_program(
    fixture: &TempDir,
    filename: &str,
    source: &str,
) -> std::process::Output {
    let program = fixture.path().join(filename);
    std::fs::write(&program, source).expect("write inline XLOG program");
    cargo_bin_cmd!("xlog")
        .args([
            "run",
            program.to_str().expect("valid program path"),
            "--memory-mb",
            "1024",
        ])
        .output()
        .expect("run inline XLOG program")
}

#[test]
fn test_xlog_run_validates_source_before_founded_rule_elision() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };
    let fixture = TempDir::new().expect("create fixture directory");

    for (filename, source, expected) in [
        (
            "unbounded_modal_variable.xlog",
            "#pragma epistemic_mode = faeel\npred p(u32).\np(X) :- possible p(X).\n?- p(X).\n",
            "UnsafeVariable(\"X\")",
        ),
        (
            "unsafe_negation_before_elision.xlog",
            "pred p().\npred q(u32).\np() :- possible p(), not q(X).\n?- p().\n",
            "unbound variable X in negated atom",
        ),
        (
            "type_error_before_elision.xlog",
            "pred p().\npred q(u32).\np() :- possible p(), q(a).\n?- p().\n",
            "Symbol literal",
        ),
    ] {
        let output = run_inline_xlog_program(&fixture, filename, source);
        assert!(
            !output.status.success(),
            "{filename} must fail static validation before foundedness elision"
        );
        let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
        assert!(stderr.contains(expected), "{filename}: {stderr}");
    }

    let output = run_inline_xlog_program(
        &fixture,
        "arithmetic_bound_dead_signature.xlog",
        "#pragma epistemic_mode = faeel\n\
         p(a, b).\n\
         p(X) :- X is 1, possible p(X).\n\
         ?- p(X, Y).\n",
    );
    assert!(
        output.status.success(),
        "arithmetic-bound dead signature must not block p/2: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("| a  | b  |"), "{stdout}");
}

#[test]
fn test_xlog_run_preserves_active_gates_and_constraints_after_founded_elision() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };
    let fixture = TempDir::new().expect("create fixture directory");
    let output = run_inline_xlog_program(
        &fixture,
        "active_modal_contract_after_elision.xlog",
        "#pragma epistemic_mode = faeel\n\
         pred missing().\n\
         pred present().\n\
         pred circular().\n\
         pred blocked().\n\
         pred allowed().\n\
         present().\n\
         circular() :- possible circular().\n\
         blocked() :- know missing().\n\
         allowed() :- know present().\n\
         :- know missing().\n\
         ?- circular().\n\
         ?- blocked().\n\
         ?- allowed().\n",
    );
    assert!(
        output.status.success(),
        "founded dispatch must retain active gates and the satisfied constraint: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("circular\nrows: 0"), "{stdout}");
    assert!(stdout.contains("blocked\nrows: 0"), "{stdout}");
    assert!(stdout.contains("allowed\nrows: 1"), "{stdout}");

    let output = run_inline_xlog_program(
        &fixture,
        "violated_modal_constraint_after_elision.xlog",
        "#pragma epistemic_mode = faeel\n\
         pred present().\n\
         pred circular().\n\
         present().\n\
         circular() :- possible circular().\n\
         :- know present().\n\
         ?- circular().\n",
    );
    assert!(
        !output.status.success(),
        "an active modal constraint must still reject the founded model"
    );
    let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
    assert!(stderr.contains("Constraint 0 violated"), "{stderr}");
}

#[test]
fn test_xlog_run_reports_ordinary_constraint_violation_after_modal_evaluation() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };
    let fixture = TempDir::new().expect("create fixture directory");
    let output = run_inline_xlog_program(
        &fixture,
        "ordinary_constraint_after_modal_evaluation.xlog",
        "#pragma epistemic_mode = faeel\n\
         pred observed(symbol, symbol, symbol, symbol).\n\
         pred normative(symbol, symbol, symbol, symbol).\n\
         pred modal_actual(symbol, symbol, symbol, symbol, symbol).\n\
         pred expected(symbol, symbol, symbol, symbol, symbol).\n\
         pred actual(symbol, symbol, symbol, symbol, symbol).\n\
         pred missing(symbol, symbol, symbol, symbol, symbol).\n\
         observed(world, item, present, claim).\n\
         expected(known, world, item, absent, claim).\n\
         normative(W, I, O, C) :- observed(W, I, O, C).\n\
         modal_actual(known, W, I, O, C) :-\n\
           normative(W, I, O, C),\n\
           know normative(W, I, O, C).\n\
         actual(M, W, I, O, C) :- modal_actual(M, W, I, O, C).\n\
         missing(M, W, I, O, C) :-\n\
           expected(M, W, I, O, C),\n\
           not actual(M, W, I, O, C).\n\
         :- missing(M, W, I, O, C).\n\
         ?- modal_actual(M, W, I, O, C).\n",
    );
    assert!(
        !output.status.success(),
        "an ordinary constraint over modal output must reject a missing expectation"
    );
    let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("Constraint 0 violated: :- missing(M, W, I, O, C)."),
        "{stderr}"
    );
    assert!(
        !stderr.contains("epistemic GPU transfer budget"),
        "the constraint failure must not be replaced by a transfer-budget diagnostic: {stderr}"
    );
}

#[test]
fn test_xlog_run_scopes_g91_compatibility_to_exact_tuple_support() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };
    let fixture = TempDir::new().expect("create fixture directory");
    let output = run_inline_xlog_program(
        &fixture,
        "tuple_scoped_g91_compatibility.xlog",
        "#pragma epistemic_mode = g91\n\
         pred edge(u32, u32).\n\
         pred p(u32).\n\
         pred missing().\n\
         pred q().\n\
         pred result().\n\
         pred cycle().\n\
         pred domain(u32).\n\
         pred left(u32).\n\
         pred right(u32).\n\
         edge(1, 2).\n\
         p(99).\n\
         domain(7).\n\
         p(Y) :- edge(X, Y), possible p(X).\n\
         q() :- know missing().\n\
         result() :- possible q().\n\
         cycle() :- possible cycle().\n\
         left(X) :- domain(X), possible right(X).\n\
         right(X) :- domain(X), possible left(X).\n\
         ?- p(X).\n\
         ?- result().\n\
         ?- cycle().\n\
         ?- left(X).\n\
         ?- right(X).\n",
    );
    assert!(
        output.status.success(),
        "tuple-scoped G91 program must execute: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("| 99 |"), "{stdout}");
    assert!(!stdout.contains("| 2  |"), "{stdout}");
    assert!(stdout.contains("result\nrows: 0"), "{stdout}");
    assert!(stdout.contains("cycle\nrows: 1"), "{stdout}");
    assert_eq!(stdout.matches("| 7  |").count(), 2, "{stdout}");
}

#[test]
fn test_xlog_run_preserves_authored_queries_through_modal_cycle_reduction() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };
    let fixture = TempDir::new().expect("create fixture directory");
    let output = run_inline_xlog_program(
        &fixture,
        "authored_query_metadata.xlog",
        "#pragma epistemic_mode = faeel\n\
         p(a).\n\
         p(a, b).\n\
         q() :- possible q().\n\
         ordinary_loop() :- ordinary_loop().\n\
         ?- p(X).\n\
         ?- p(X, Y).\n\
         ?- q().\n",
    );
    assert!(
        output.status.success(),
        "multi-arity modal-cycle query program must execute: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(
        !stdout.contains("p/1") && !stdout.contains("p/2"),
        "{stdout}"
    );
    assert!(!stdout.contains("computed_0"), "{stdout}");
    assert_eq!(stdout.matches("p\n").count(), 2, "{stdout}");
    assert!(stdout.contains("q\nrows: 0"), "{stdout}");

    let output = run_inline_xlog_program(
        &fixture,
        "authored_query_metadata_case_a.xlog",
        "#pragma epistemic_mode = faeel\n\
         p(a).\n\
         p(a, b).\n\
         base().\n\
         recursive() :- know base().\n\
         recursive() :- recursive().\n\
         ?- p(X).\n\
         ?- p(X, Y).\n\
         ?- recursive().\n",
    );
    assert!(
        output.status.success(),
        "Case-A query presentation must execute: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(
        !stdout.contains("p/1") && !stdout.contains("p/2"),
        "{stdout}"
    );
    assert!(!stdout.contains("computed_0"), "{stdout}");
    assert_eq!(stdout.matches("p\n").count(), 2, "{stdout}");
    assert!(stdout.contains("recursive\nrows: 1"), "{stdout}");
}

#[test]
fn test_xlog_run_nullary_negative_modal_cycles_use_well_founded_truth() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };
    let fixture = TempDir::new().expect("create fixture directory");
    for (mode, pragma) in [
        ("faeel", "#pragma epistemic_mode = faeel\n"),
        ("g91", "#pragma epistemic_mode = g91\n"),
    ] {
        let source = format!(
            "{pragma}pred p().\npred q().\npred self_cycle().\npred dependent().\n\
             p() :- not possible q().\nq() :- not possible p().\n\
             self_cycle() :- not possible self_cycle().\n\
             dependent() :- not self_cycle().\n\
             ?- p().\n?- q().\n?- self_cycle().\n?- dependent().\n"
        );
        let output = run_inline_xlog_program(
            &fixture,
            &format!("nullary_negative_cycle_{mode}.xlog"),
            &source,
        );
        assert!(
            output.status.success(),
            "{mode} nullary WFS program must execute: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
        assert!(stdout.contains("p\nrows: 0"), "{mode}: {stdout}");
        assert!(stdout.contains("q\nrows: 0"), "{mode}: {stdout}");
        assert!(stdout.contains("self_cycle\nrows: 0"), "{mode}: {stdout}");
        assert!(stdout.contains("dependent\nrows: 0"), "{mode}: {stdout}");
    }
}

#[test]
fn test_xlog_run_validates_structured_modal_arity_before_exact_elision() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };
    let fixture = TempDir::new().expect("create fixture directory");

    for mode in ["faeel", "g91"] {
        let source = format!(
            "#pragma epistemic_mode = {mode}\n\
             pred p(list<symbol>).\n\
             p([a, b]) :- possible p([a, b]).\n\
             ?- p(X).\n"
        );
        let output = run_inline_xlog_program(
            &fixture,
            &format!("structured_modal_arity_{mode}.xlog"),
            &source,
        );
        assert!(
            !output.status.success(),
            "{mode} must reject the authored structured-key mismatch before reduction"
        );
        let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
        assert!(stderr.contains("epistemic modal tuple key"), "{stderr}");
        assert!(stderr.contains("target arity 1"), "{stderr}");
        assert!(stderr.contains("binding arity 2"), "{stderr}");
    }
}

#[test]
fn test_xlog_run_accepts_modal_binder_over_acyclic_dependency_diamond() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };
    let fixture = TempDir::new().expect("create fixture directory");
    let output = run_inline_xlog_program(
        &fixture,
        "invariant_dependency_diamond.xlog",
        "pred base(u32).\n\
         pred left(u32).\n\
         pred right(u32).\n\
         pred joined(u32).\n\
         pred out(u32).\n\
         base(1).\n\
         left(X) :- base(X).\n\
         right(X) :- base(X).\n\
         joined(X) :- left(X), right(X).\n\
         out(X) :- possible joined(X).\n\
         ?- out(X).\n",
    );
    assert!(
        output.status.success(),
        "the shared acyclic dependency must remain invariant: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("out\n"), "{stdout}");
    assert!(stdout.contains("| 1  |"), "{stdout}");
}

#[test]
fn test_xlog_run_g91_compatibility_is_scoped_to_concrete_cycle_tuples() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };
    let fixture = TempDir::new().expect("create fixture directory");

    let disjoint = run_inline_xlog_program(
        &fixture,
        "g91_disjoint_mutual_domains.xlog",
        "#pragma epistemic_mode = g91\n\
         pred domain_p(u32).\n\
         pred domain_q(u32).\n\
         pred p(u32).\n\
         pred q(u32).\n\
         domain_p(1).\n\
         domain_q(2).\n\
         p(X) :- domain_p(X), possible q(X).\n\
         q(X) :- domain_q(X), possible p(X).\n\
         ?- p(X).\n\
         ?- q(X).\n",
    );
    assert!(
        disjoint.status.success(),
        "disjoint compatibility cycle must execute: {}",
        String::from_utf8_lossy(&disjoint.stderr)
    );
    let stdout = String::from_utf8(disjoint.stdout).expect("utf8 stdout");
    assert!(stdout.starts_with("p\n"), "{stdout}");
    assert!(stdout.contains("\nq\n"), "{stdout}");
    assert!(!stdout.contains("| 1  |"), "{stdout}");
    assert!(!stdout.contains("| 2  |"), "{stdout}");

    let shared = run_inline_xlog_program(
        &fixture,
        "g91_shared_mutual_domain.xlog",
        "#pragma epistemic_mode = g91\n\
         pred domain(u32).\n\
         pred p(u32).\n\
         pred q(u32).\n\
         domain(7).\n\
         p(X) :- domain(X), possible q(X).\n\
         q(X) :- domain(X), possible p(X).\n\
         ?- p(X).\n\
         ?- q(X).\n",
    );
    assert!(
        shared.status.success(),
        "shared compatibility tuple must execute: {}",
        String::from_utf8_lossy(&shared.stderr)
    );
    let stdout = String::from_utf8(shared.stdout).expect("utf8 stdout");
    assert_eq!(stdout.matches("| 7  |").count(), 2, "{stdout}");

    let tuple_changing_return = run_inline_xlog_program(
        &fixture,
        "g91_tuple_changing_return_path.xlog",
        "#pragma epistemic_mode = g91\n\
         pred domain(u32).\n\
         pred edge(u32, u32).\n\
         pred p(u32).\n\
         pred q(u32).\n\
         domain(1).\n\
         edge(2, 1).\n\
         p(X) :- domain(X), possible q(X).\n\
         q(Y) :- edge(X, Y), possible p(X).\n\
         ?- p(X).\n\
         ?- q(X).\n",
    );
    assert!(
        tuple_changing_return.status.success(),
        "tuple-changing return path must execute: {}",
        String::from_utf8_lossy(&tuple_changing_return.stderr)
    );
    let stdout = String::from_utf8(tuple_changing_return.stdout).expect("utf8 stdout");
    assert!(!stdout.contains("| 1  |"), "{stdout}");
    assert!(!stdout.contains("| 2  |"), "{stdout}");

    let founded_union = run_inline_xlog_program(
        &fixture,
        "g91_founded_union_and_incompatible_cycle.xlog",
        "#pragma epistemic_mode = g91\n\
         pred domain_p(u32).\n\
         pred domain_q(u32).\n\
         pred p(u32).\n\
         pred q(u32).\n\
         domain_p(1).\n\
         domain_q(2).\n\
         p(9).\n\
         p(X) :- domain_p(X), possible q(X).\n\
         q(X) :- domain_q(X), possible p(X).\n\
         ?- p(X).\n\
         ?- q(X).\n",
    );
    assert!(
        founded_union.status.success(),
        "founded rule union must execute: {}",
        String::from_utf8_lossy(&founded_union.stderr)
    );
    let stdout = String::from_utf8(founded_union.stdout).expect("utf8 stdout");
    assert!(stdout.contains("| 9  |"), "{stdout}");
    assert!(!stdout.contains("| 1  |"), "{stdout}");
    assert!(!stdout.contains("| 2  |"), "{stdout}");
}

#[test]
fn test_xlog_run_g91_compatibility_iterates_to_the_greatest_tuple_fixpoint() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };
    let fixture = TempDir::new().expect("create fixture directory");
    let output = run_inline_xlog_program(
        &fixture,
        "g91_three_relation_descending_fixpoint.xlog",
        "#pragma epistemic_mode = g91\n\
         pred domain_p(u32).\n\
         pred domain_q(u32).\n\
         pred domain_r(u32).\n\
         pred p(u32).\n\
         pred q(u32).\n\
         pred r(u32).\n\
         domain_p(1). domain_p(2). domain_p(3).\n\
         domain_q(1). domain_q(2).\n\
         domain_r(1).\n\
         p(X) :- domain_p(X), possible q(X).\n\
         q(X) :- domain_q(X), possible r(X).\n\
         r(X) :- domain_r(X), possible p(X).\n\
         ?- p(X).\n\
         ?- q(X).\n\
         ?- r(X).\n",
    );
    assert!(
        output.status.success(),
        "descending compatibility fixpoint must execute: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert_eq!(stdout.matches("| 1  |").count(), 3, "{stdout}");
    assert!(!stdout.contains("| 2  |"), "{stdout}");
    assert!(!stdout.contains("| 3  |"), "{stdout}");

    let limited = run_inline_xlog_program(
        &fixture,
        "g91_compatibility_iteration_limit.xlog",
        "#pragma epistemic_mode = g91\n\
         #pragma max_recursion_depth = 1\n\
         pred domain_p(u32).\n\
         pred domain_q(u32).\n\
         pred domain_r(u32).\n\
         pred p(u32).\n\
         pred q(u32).\n\
         pred r(u32).\n\
         domain_p(1). domain_p(2). domain_p(3).\n\
         domain_q(1). domain_q(2).\n\
         domain_r(1).\n\
         p(X) :- domain_p(X), possible q(X).\n\
         q(X) :- domain_q(X), possible r(X).\n\
         r(X) :- domain_r(X), possible p(X).\n\
         ?- p(X).\n",
    );
    assert!(!limited.status.success());
    let stderr = String::from_utf8(limited.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("did not converge within 1 refinement iterations"),
        "{stderr}"
    );
}

#[test]
fn test_xlog_run_g91_modal_binder_order_does_not_change_compatibility() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };
    let fixture = TempDir::new().expect("create fixture directory");
    for (name, body) in [
        ("recursive_first", "possible p(X), possible base(X)"),
        ("finite_first", "possible base(X), possible p(X)"),
    ] {
        let source = format!(
            "#pragma epistemic_mode = g91\n\
             pred base(u32).\n\
             pred p(u32).\n\
             base(1).\n\
             p(X) :- {body}.\n\
             ?- p(X).\n"
        );
        let output =
            run_inline_xlog_program(&fixture, &format!("g91_modal_binder_{name}.xlog"), &source);
        assert!(
            output.status.success(),
            "{name} must execute: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
        assert!(stdout.contains("| 1  |"), "{name}: {stdout}");
    }

    let constrained = run_inline_xlog_program(
        &fixture,
        "g91_compatibility_ordinary_constraint.xlog",
        "#pragma epistemic_mode = g91\n\
         pred base(u32).\n\
         pred p(u32).\n\
         base(1).\n\
         p(X) :- base(X), possible p(X).\n\
         :- p(1).\n\
         ?- p(X).\n",
    );
    assert!(
        !constrained.status.success(),
        "the converged compatibility result must still enforce ordinary constraints"
    );
    let stderr = String::from_utf8(constrained.stderr).expect("utf8 stderr");
    assert!(stderr.contains("Constraint 0 violated"), "{stderr}");
}

#[test]
fn test_xlog_run_combines_g91_compatibility_with_unrelated_gpu_wfs() {
    let _device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {error}");
            return;
        }
    };
    let fixture = TempDir::new().expect("create fixture directory");
    let output = run_inline_xlog_program(
        &fixture,
        "g91_compatibility_with_wfs.xlog",
        "#pragma epistemic_mode = g91\n\
         pred base(u32).\n\
         pred p(u32).\n\
         pred left().\n\
         pred right().\n\
         base(1).\n\
         p(X) :- base(X), possible p(X).\n\
         left() :- not possible right().\n\
         right() :- not possible left().\n\
         :- p(2).\n\
         :- not left().\n\
         ?- p(X).\n\
         ?- left().\n\
         ?- right().\n",
    );
    assert!(
        output.status.success(),
        "positive compatibility and unrelated WFS must compose: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("| 1  |"), "{stdout}");
    assert!(stdout.contains("left\nrows: 0"), "{stdout}");
    assert!(stdout.contains("right\nrows: 0"), "{stdout}");

    let violated = run_inline_xlog_program(
        &fixture,
        "g91_compatibility_with_wfs_constraint_violation.xlog",
        "#pragma epistemic_mode = g91\n\
         pred base(u32).\n\
         pred p(u32).\n\
         pred left().\n\
         pred right().\n\
         base(1).\n\
         p(X) :- base(X), possible p(X).\n\
         left() :- not possible right().\n\
         right() :- not possible left().\n\
         :- p(1).\n\
         ?- p(X).\n",
    );
    assert!(
        !violated.status.success(),
        "a constraint violated after nested WFS convergence must fail"
    );
    let stderr = String::from_utf8(violated.stderr).expect("utf8 stderr");
    assert!(stderr.contains("Constraint 0 violated"), "{stderr}");

    let false_negation = run_inline_xlog_program(
        &fixture,
        "g91_compatibility_with_wfs_false_negation_constraint.xlog",
        "#pragma epistemic_mode = g91\n\
         pred base(u32).\n\
         pred p(u32).\n\
         pred left().\n\
         pred right().\n\
         pred definitely_false().\n\
         base(1).\n\
         p(X) :- base(X), possible p(X).\n\
         left() :- not possible right().\n\
         right() :- not possible left().\n\
         :- not definitely_false().\n\
         ?- p(X).\n",
    );
    assert!(
        !false_negation.status.success(),
        "negation of a WFS-false atom makes the constraint body true"
    );
    let stderr = String::from_utf8(false_negation.stderr).expect("utf8 stderr");
    assert!(stderr.contains("Constraint 0 violated"), "{stderr}");
}
