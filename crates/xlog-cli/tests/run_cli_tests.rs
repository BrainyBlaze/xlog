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
        assert!(
            stdout.contains(expected_value),
            "{} did not emit expected value marker {}:\n{}",
            example,
            expected_value,
            stdout
        );
    }
}

#[test]
fn test_xlog_run_nested_modal_reports_typed_epistemic_diagnostic() {
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
    let program = repo_root
        .join("examples/epistemic")
        .join("13-nested-modal-rejected.xlog");
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
        !output.status.success(),
        "nested modal example must fail closed, stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("UnsupportedEpistemicConstruct"), "{stderr}");
    assert!(stderr.contains("nested epistemic literal"), "{stderr}");
    assert!(stderr.contains("know possible p()"), "{stderr}");
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
fn test_xlog_run_recursion_through_modal_reports_typed_epistemic_diagnostic() {
    // v0.9.2 BOUNDARY: a modal literal over a relation entangled in the program's
    // recursive SCC (NOT invariant) must FAIL CLOSED with a typed diagnostic.
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    let (ok, stdout, stderr) = run_epistemic_example("22-recursion-through-modal-rejected.xlog");
    assert!(
        !ok,
        "recursion-through-modal example must fail closed, stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(stderr.contains("UnsupportedEpistemicConstruct"), "{stderr}");
    assert!(stderr.contains("recursive epistemic program"), "{stderr}");
    // Names the offending modal predicate and the invariance requirement.
    assert!(stderr.contains("know reach"), "{stderr}");
    assert!(stderr.contains("not invariant"), "{stderr}");
}

#[test]
fn test_xlog_run_compound_modal_key_reports_typed_epistemic_diagnostic() {
    // v0.9.2 BOUNDARY: a list/compound modal key cannot be encoded as a GPU
    // tuple-key column and must FAIL CLOSED with a typed diagnostic naming the
    // offending term.
    let _device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIPPED: CUDA runtime unavailable (no GPU or driver not loaded)");
            return;
        }
    };

    let (ok, stdout, stderr) = run_epistemic_example("23-compound-modal-key-rejected.xlog");
    assert!(
        !ok,
        "compound-modal-key example must fail closed, stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(stderr.contains("UnsupportedEpistemicConstruct"), "{stderr}");
    assert!(
        stderr.contains("epistemic GPU tuple-key expectation"),
        "{stderr}"
    );
    // Names the offending compound term. The diagnostic context is itself a
    // Debug-escaped string, so the inner quotes appear as literal backslashes;
    // match them with a raw string literal.
    assert!(
        stderr.contains(r#"List([Variable(\"H\")])"#),
        "diagnostic must name the offending list term:\n{stderr}"
    );
}
