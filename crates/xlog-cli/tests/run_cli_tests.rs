use assert_cmd::cargo::cargo_bin_cmd;
use cudarc::driver::result::mem_get_info;
use std::path::Path;
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
        // v0.9.1 epistemic executor showcase (EGB-01/02/04/05/06/07), each validated
        // through the production `xlog run` path with a deterministic output marker.
        ("06-eir-candidate-enumeration.xlog", "believed", "| 3  |"),
        ("07-tuple-key-membership.xlog", "matched", "| 3  | 3  |"),
        ("08-repeated-variable.xlog", "reflexive", "| 3  |"),
        ("09-joint-multi-epistemic.xlog", "both_known", "| 1  |"),
        ("10-epistemic-constraint.xlog", "accepted", "rows: 0"),
        ("11-faeel-foundedness.xlog", "founded", "rows: 1"),
        ("12-bound-variable-splitting.xlog", "both_known", "| 1  |"),
        ("12-bound-variable-splitting.xlog", "safe_alt", "| 2  |"),
        // v0.9.2 EGB-02B: mixed per-row (`possible edge(X)`) + global
        // (`know global_flag()`) modal membership composed conjunctively.
        ("14-mixed-literal-membership.xlog", "reachable", "| 1  |"),
        ("14-mixed-literal-membership.xlog", "reachable", "| 3  |"),
        // v0.9.2 Case-A recursive epistemic fixpoint: ordinary recursion gated by a
        // modal literal over an INVARIANT relation, delegated to the existing
        // recursive/semi-naive engine. The derived multi-hop tuples prove the
        // fixpoint is NOT single-pass.
        // 3-vertex closure: base (1,2),(2,3) plus DERIVED (1,3).
        (
            "15-recursive-epistemic-closure.xlog",
            "__xlog_query_0",
            "| 1  | 2  |",
        ),
        (
            "15-recursive-epistemic-closure.xlog",
            "__xlog_query_0",
            "| 2  | 3  |",
        ),
        (
            "15-recursive-epistemic-closure.xlog",
            "__xlog_query_0",
            "| 1  | 3  |",
        ),
        // 4-chain: base hops plus 2-hop (1,3),(2,4) and the 3-hop DERIVED (1,4).
        (
            "15-recursive-epistemic-chain.xlog",
            "__xlog_query_0",
            "| 1  | 4  |",
        ),
        (
            "15-recursive-epistemic-chain.xlog",
            "__xlog_query_0",
            "| 2  | 4  |",
        ),
        // v0.9.2 Bundle 3: cross-component epistemic coupling (ACCEPTED safe case).
        // The ordinary head `report` consumes the epistemic-derived head `trusted`,
        // coupling two locally-splittable components through a derived dependency.
        // Single epistemic output head -> joint single-output path materializes the
        // gated `trusted = {1, 3}` exactly (node 2 is not vetted, so not known).
        ("16-cross-component-coupling.xlog", "trusted", "| 1  |"),
        ("16-cross-component-coupling.xlog", "trusted", "| 3  |"),
        // v0.9.2 STRATIFIED epistemic execution: chained coupling over a DETERMINED
        // epistemic-derived head. `trusted` (gated by `know vetted` over EDB) is a
        // strictly-lower stratum materialized into the store; `flagged :- know
        // trusted` then gates against the materialized `trusted = {1, 3}` via the
        // existing EGB-02 membership filter. flagged = {1, 3}; node 2 is gated out.
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
        // v0.9.2 Bundle 3 COMPLETION: cross-component epistemic JOINT-SOLVING
        // (multi-output). Two epistemic heads share ONE base modal predicate `q`
        // (SharedModalPredicate coalesce) but neither feeds the other, so the
        // coalesced multi-head component is JOINT-SOLVED: one candidate enumeration
        // + world-view validation over the COMBINED modals (`know q`, `possible q`),
        // then BOTH heads materialized against the SAME accepted world view. Unlike
        // example 16 (single producer head), this pilot materializes and DISPLAYS
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
        // v0.9.2 EGB-02B (FULL): access-control mixed modal gating. ONE accepted
        // rule combines a GLOBAL `know gateway_online()` gate, a PER-ROW
        // `possible cleared(P)`, and a PER-ROW negated `not possible revoked(P)`,
        // composed conjunctively. granted = {1, 2}; carol(3) and dave(4) are
        // gated out (absence checked below).
        ("19-access-control-mixed-modal.xlog", "granted", "| 1  |"),
        ("19-access-control-mixed-modal.xlog", "granted", "| 2  |"),
        // v0.9.2 Case-A recursive epistemic fixpoint (FULL): supply-chain
        // provenance. Ordinary recursion in `sources_from` gated by `know
        // certified` over the INVARIANT EDB. 10 tuples incl. 4-hop derived
        // (1,5); supplier 6 is reachable only via an uncertified link and is
        // fully gated out (no `| 6  |` row; absence checked below).
        (
            "20-supply-chain-recursive-reach.xlog",
            "__xlog_query_0",
            "| 1  | 5  |",
        ),
        (
            "20-supply-chain-recursive-reach.xlog",
            "__xlog_query_0",
            "| 2  | 5  |",
        ),
        // v0.9.2 Bundle 3 JOINT-SOLVING (FULL): incident triage. THREE epistemic
        // heads share the base modal `compromised`; all three modalities used and
        // all three heads materialized against the SAME accepted world view.
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
        // v0.9.2 STRATIFIED recursion over a DETERMINED epistemic-derived head:
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
        // v0.9.2 NEGATED modal over an INVARIANT relation in a recursive context:
        // `not know blocked(Y)` == ordinary `not blocked(Y)` anti-join. blocked
        // node 3 severs the chain -> reach = {(1,2),(3,4)}.
        (
            "26-negated-modal-over-invariant-recursive.xlog",
            "__xlog_query_0",
            "| 1  | 2  |",
        ),
        (
            "26-negated-modal-over-invariant-recursive.xlog",
            "__xlog_query_0",
            "| 3  | 4  |",
        ),
        // v0.9.2 SCOPE-LIMIT CLOSED: AUGMENTED multi-head coupling with PER-HEAD
        // output projection at DIFFERING arity. Two epistemic heads share the base
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
        // v0.9.2 SCOPE-LIMIT CLOSED: modal over an ORDINARY predicate transitively
        // derived from a DETERMINED epistemic head, stratified. `r :- a` (a determined
        // via `know p` over EDB) makes `r` determined; the lower stratum materializes
        // gated `a`, `r :- a` is computed over the base, and `know r` gates against the
        // materialized base `r`. b = node intersect r = {1} (node 2 has no `know p`).
        (
            "24-transitive-determined-modal-stratified.xlog",
            "b",
            "| 1  |",
        ),
        // v0.9.2 SCOPE-LIMIT CLOSED: a modal BINDING an output variable over a DETERMINED
        // MULTI-COLUMN epistemic head. `r(X,Y) :- edge(X,Y), know flag(X)` is determined;
        // `out(X) :- node(X), know r(X,Y)` binds the extra column Y. The lower stratum
        // materializes gated `r = {(1,2),(1,3)}`; the higher stratum gates `know r`
        // against that base, projecting away Y. out = {X in node : exists Y r(X,Y)} = {1}.
        ("28-determined-multicol-binding-modal.xlog", "out", "| 1  |"),
        // v0.9.2 COMPLETENESS CELL: a NEGATED modal (`not know` / `not possible`) over
        // a DETERMINED epistemic-DERIVED head. `a` (gated by `know p` over EDB `p`) is
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
        // v0.9.2 COMPLETENESS CELL: the `possible` twin of example 28 -- a BINDING
        // `possible r(X, Y)` over a DETERMINED multi-column epistemic head. Proves the
        // modal operator is irrelevant for a determined target (`possible r == know r ==
        // r`); stratifies identically to 28. out = {1} (gate load-bearing: ungated node
        // = {1,2,3}).
        ("30-possible-binding-over-determined.xlog", "out", "| 1  |"),
        // v0.9.2 ITEM B: FAEEL unfounded self-support EXECUTES to its EXACT empty
        // founded extension (NOT a rejection). `p() :- possible p()` is supported only
        // by circular modal self-support with no independent founded derivation, so `p`
        // is absent from the founded model -> rows: 0, exit 0.
        (
            "31-faeel-unfounded-self-support-empty-extension.xlog",
            "p",
            "rows: 0",
        ),
        // v0.9.2 ITEM B mode-difference pair: the SAME unfounded self-support program
        // under explicit G91 accepts circular self-support -> rows: 1. (31 FAEEL rows:0
        // vs 32 G91 rows:1 is the exact FAEEL-vs-G91 semantic divergence.)
        ("32-g91-self-support-accepted.xlog", "p", "rows: 1"),
        // v0.9.2 ITEM D: a STRUCTURED finite+typed modal tuple-key. The 1-element
        // list `[H]` flattens element-wise into `watched`'s scalar u32 key column,
        // so `know watched([H])` GATES `host` by `watched` membership. Load-bearing
        // (gated != ungated): only watched hosts survive -> alert = {1} (node 2 is
        // dropped). The structured-key flattening runs entirely on the GPU.
        ("23-compound-modal-key-membership.xlog", "alert", "| 1  |"),
        // v0.9.2 ITEM E: a VARIABLE-KEYED epistemic integrity constraint
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
        // v0.9.2 ITEM E multi-literal (DISTINCT independent variables):
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
        // v0.9.2 ITEM C: NESTED MODAL OPERATORS execute via sound chain-collapse.
        // A modal chain collapses (KD45/S5) to the operator ADJACENT to the atom;
        // a leading `not` distributes. The collapse routes through the existing
        // single-level epistemic path (no new evaluator). ex13 `know possible p()`
        // collapses to `possible p()` over EDB `p` (determined) -> q holds, rows: 1.
        ("13-nested-modal-chain-collapse.xlog", "q", "rows: 1"),
        // 13b: `know know reachable(X)` (KK == K) gates `node` by `reachable`.
        // Load-bearing: gated = node intersect reachable = {1, 3} (node 2 dropped).
        ("13b-nested-modal-chain-filter.xlog", "gated", "| 1  |"),
        ("13b-nested-modal-chain-filter.xlog", "gated", "| 3  |"),
        // 13c: `not know possible blocked(X)` -- leading negation distributes,
        // chain collapses to `possible` -> `not possible blocked == not blocked`
        // (anti-join over EDB). Load-bearing: allowed = node \ blocked = {1, 3}.
        ("13c-nested-modal-chain-negated.xlog", "allowed", "| 1  |"),
        ("13c-nested-modal-chain-negated.xlog", "allowed", "| 3  |"),
        // 13d: `p() :- possible possible p()` (MM == M) is the CHAIN form of ex31.
        // The collapse forwards the per-mode foundedness difference: under FAEEL the
        // circular self-support is unfounded -> p absent -> rows: 0.
        (
            "13d-nested-modal-chain-faeel-unfounded.xlog",
            "p",
            "rows: 0",
        ),
        // 13e: the SAME chain program under explicit G91 ACCEPTS self-support ->
        // rows: 1. 13d (FAEEL rows:0) vs 13e (G91 rows:1) is the exact per-mode
        // divergence of the collapsed chain (mirrors 31 vs 32 with a chain).
        ("13e-nested-modal-chain-g91-accepted.xlog", "p", "rows: 1"),
        // 13f: `know not possible p()` dualizes to `not possible p()`. Because p()
        // is present, q is absent, but the program succeeds through `xlog run`.
        (
            "13f-nested-modal-interior-negation-rejected.xlog",
            "q",
            "rows: 0",
        ),
        // 13g-13v exhaust every two-operator modal chain over `{know, possible}` with
        // leading/interior/atom-adjacent negation placements, split by operator
        // pair, present/absent target, and derived/non-derived outcome to keep each
        // example within the accepted candidate-generation bound. Together these
        // slices cover all 64 source forms.
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
        // v0.9.2 WALL A1 (ACCEPTED): a NEGATED modal `not know reach` over a GENUINELY
        // RECURSIVE relation in a strictly LOWER stratum than the negating head
        // EXECUTES on the GPU production path as ordinary stratified negation. reach =
        // transitive closure of link {(1,2),(2,3)} = {(1,2),(2,3),(1,3)}; unreachable =
        // node x node MINUS reach = 6 pairs. (1,1) is a self-pair excluded from reach
        // (modal gate load-bearing); (3,1) confirms the anti-join against the recursive
        // closure. Contrast example 33 (the cyclic twin) which stays WFS-bounded.
        (
            "37-negated-modal-over-recursive-stratified.xlog",
            "__xlog_query_0",
            "| 1  | 1  |",
        ),
        (
            "37-negated-modal-over-recursive-stratified.xlog",
            "__xlog_query_0",
            "| 3  | 1  |",
        ),
        // v0.9.2 EGB-06 K2/K4: same-name multi-arity modal predicates resolve
        // through full `xlog run` by loading p/1 and p/2 under arity-qualified
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
        // v0.9.2 G91 possible recursion: positive `possible` over the co-evolving
        // recursive target is the compatibility self-support assumption. The full
        // `xlog run` path returns the complete 3 x 3 vertex relation.
        (
            "43-g91-possible-recursion-self-support.xlog",
            "__xlog_query_0",
            "| 1  | 1  |",
        ),
        (
            "43-g91-possible-recursion-self-support.xlog",
            "__xlog_query_0",
            "| 1  | 2  |",
        ),
        (
            "43-g91-possible-recursion-self-support.xlog",
            "__xlog_query_0",
            "| 1  | 3  |",
        ),
        (
            "43-g91-possible-recursion-self-support.xlog",
            "__xlog_query_0",
            "| 2  | 1  |",
        ),
        (
            "43-g91-possible-recursion-self-support.xlog",
            "__xlog_query_0",
            "| 2  | 2  |",
        ),
        (
            "43-g91-possible-recursion-self-support.xlog",
            "__xlog_query_0",
            "| 2  | 3  |",
        ),
        (
            "43-g91-possible-recursion-self-support.xlog",
            "__xlog_query_0",
            "| 3  | 1  |",
        ),
        (
            "43-g91-possible-recursion-self-support.xlog",
            "__xlog_query_0",
            "| 3  | 2  |",
        ),
        (
            "43-g91-possible-recursion-self-support.xlog",
            "__xlog_query_0",
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
    // v0.9.2 SCOPE-LIMIT CLOSED: a modal over an ORDINARY predicate transitively
    // derived from a DETERMINED epistemic head (`b :- know r`, `r :- a`, `a :- know p`)
    // is now ACCEPTED via stratification. `r` is determined-in-principle (ordinary over
    // the determined `a`), so the lower stratum materializes the gated `a`, the
    // ordinary `r :- a` is computed over the materialized base (making `r` locally
    // invariant), and the higher stratum gates `know r` against the materialized base
    // `r` via the existing EGB-02 filter. EXACT tuples: with node={1,2}, p={1},
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
    // v0.9.2 SCOPE-LIMIT CLOSED: a modal that BINDS an output variable over a DETERMINED
    // MULTI-COLUMN epistemic head is now ACCEPTED via stratification. The full program:
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

fn same_name_modal_truth(form: &str, tuple_present: bool) -> bool {
    match form {
        "know" | "possible" => tuple_present,
        "not-know" | "not-possible" => !tuple_present,
        other => panic!("unknown same-name modal form {other}"),
    }
}

fn assert_same_name_matrix_example(example: &str, should_hold: bool) {
    let (ok, stdout, stderr) = run_epistemic_example(example);
    assert!(
        ok,
        "same-name matrix example {example} must succeed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("holds"),
        "same-name matrix example {example} must emit holds relation:\n{stdout}"
    );
    let has_row = stdout.contains("| 1  |");
    assert_eq!(
        has_row, should_hold,
        "same-name matrix example {example} expected holds row presence {should_hold}:\n{stdout}"
    );
}

#[test]
fn test_xlog_run_same_name_multi_arity_exhaustive_matrix() {
    // v0.9.2 EGB-06 K2/K4 exhaustive production-path matrix.
    //
    // Single-literal cells cover:
    //   arity {1,2} x modal {know,possible} x polarity {positive,negated}
    //   x queried-tuple state {present,absent}.
    //
    // Cross-arity cells cover every conjunction:
    //   unary modal form {K,M,not K,not M}
    //   x binary modal form {K,M,not K,not M}
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
fn test_xlog_run_v092_examples_modal_gating_filters() {
    // v0.9.2 anti-gaming: prove the accepted examples are REAL modal filtering,
    // not store extraction -- the gated output must strictly OMIT the rows that
    // fail the modal gate (so the result DIFFERS from the ungated relation).
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
    // v0.9.2 ITEM A: a POSITIVE modal over a relation that CO-EVOLVES with the
    // program's ordinary recursion (Case B) is EXECUTED to its FAEEL founded least
    // fixpoint, NOT rejected. The modal feeds a non-mirror relation `trust`, so the
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
        "Case-B recursive epistemic fixpoint must EXECUTE, stdout:\n{stdout}\nstderr:\n{stderr}"
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
fn test_xlog_run_negated_modal_through_recursion_uses_wfs_engine() {
    // v0.9.2 A1 closure: a NEGATED modal whose target CYCLES through recursion via
    // negation now executes through the explicit WFS engine. True atoms are emitted;
    // undefined atoms are absent.
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    let (ok, stdout, stderr) = run_epistemic_example("33-negated-modal-through-recursion-wfs.xlog");
    assert!(
        ok,
        "negated-modal-through-recursion WFS example must execute, stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("| 1 | 2 |") || stdout.contains("| 1  | 2  |"),
        "WFS true seed reach tuple must be present:\n{stdout}"
    );
    for (x, y) in [
        (1, 1),
        (1, 3),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 1),
        (3, 2),
        (3, 3),
    ] {
        let compact = format!("| {x} | {y} |");
        let padded = format!("| {x}  | {y}  |");
        assert!(
            !stdout.contains(&compact) && !stdout.contains(&padded),
            "WFS undefined tuple ({x},{y}) must be absent:\n{stdout}"
        );
    }
}

#[test]
fn test_xlog_run_compound_modal_key_reports_typed_epistemic_diagnostic() {
    // v0.9.2 ITEM D BOUNDARY: structured finite+typed modal keys (fixed-arity
    // list/compound of scalar elements) are now ACCEPTED and flattened onto the
    // GPU (see 23-compound-modal-key-membership.xlog + the accepted examples
    // list). What stays REJECTED is a genuinely UNBOUNDED structured key -- a
    // `cons` `[H | T]` whose tail length is not statically fixed has no finite,
    // typed GPU key-column set. It must FAIL CLOSED with a precise FINITENESS
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
    // v0.9.2 E1 (diagonal): `:- know route(X, X).` -- a single modal literal repeating X
    // across its OWN key columns -- was rejected as an unimplemented shared-variable join.
    // It is now resolved by a sound PROGRAM-level desugaring (ordinary diagonal extraction
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
    // v0.9.2 E1 (join + negated-difference): a shared-variable epistemic constraint --
    // `:- know p(X), possible q(X).` (40, intersection) and `:- q(X), not know p(X).` (41,
    // set difference) -- was rejected as an unimplemented shared-variable join. Both are now
    // resolved by a sound PROGRAM-level desugaring (ordinary join/difference extraction
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
    // v0.9.2 ITEM B (mandate headline): a self-supported possible rule
    // (`p() :- possible p().`) with no independent founded support is UNFOUNDED under
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
        stdout.contains("rows: 0"),
        "FAEEL unfounded self-support must materialize the EMPTY founded extension \
         (rows: 0):\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    // It must NOT be the old rejection.
    assert!(
        !stderr.contains("FAEEL foundedness guard"),
        "FAEEL unfounded self-support must no longer be rejected:\n{stderr}"
    );

    // Mode-difference evidence: the SAME program under explicit G91 ACCEPTS the
    // circular self-support and emits rows: 1.
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
