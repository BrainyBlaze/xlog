use xlog_logic::{parse_program, stratify, Compiler};

#[test]
fn compiles_stratified_closed_world_naf_with_list_meta_and_is_binders() {
    let src = r#"
        pred node(id: u32).
        pred edge(src: u32, dst: u32).
        pred blocked(id: u32).
        pred bag(xs: list<u32>).

        node(1).
        node(2).
        num(1).
        edge(1, 2).
        blocked(2).
        blocked64(2).
        bag([1, 2]).

        leaf(X) :- node(X), not edge(X, _).
        unblocked_member(X) :- bag(L), member(X, L), not blocked(X).
        next_clear(Z) :- num(X), Z is X + cast(1, u32), not blocked64(Z).
        source_fanout_ok(1) :- findall(Y, edge(1, Y), Ys), length(Ys, 1), not blocked(1).

        ?- unblocked_member(X).
    "#;

    Compiler::new()
        .compile(src)
        .expect("stratified closed-world NAF should compile");
}

#[test]
fn rejects_8_unsafe_or_unstratified_naf_fixtures_with_typed_diagnostics() {
    let invalid = [
        (
            "unbound second variable",
            "node(1). bad(X) :- node(X), not edge(X, Y).",
            "unbound variable Y",
        ),
        (
            "negation before binder",
            "node(1). bad(X) :- not edge(X, _), node(X).",
            "unbound variable X",
        ),
        (
            "unbound first variable",
            "node(1). bad(X) :- node(X), not edge(Y, X).",
            "unbound variable Y",
        ),
        (
            "anonymous does not bind named variable",
            "node(1). bad(X) :- node(X), not edge(_, Y).",
            "unbound variable Y",
        ),
        (
            "self cycle",
            "node(1). p(X) :- node(X), not p(X).",
            "stratified",
        ),
        (
            "mutual cycle",
            "node(1). p(X) :- q(X). q(X) :- node(X), not p(X).",
            "stratified",
        ),
        ("nullary self cycle", "p() :- not p().", "stratified"),
        (
            "meta binder order",
            "bad(X) :- not blocked(X), var(Unused).",
            "unbound variable X",
        ),
    ];

    assert_eq!(invalid.len(), 8);

    for (label, src, needle) in invalid {
        let err = Compiler::new().compile(src).expect_err(label);
        let msg = err.to_string();
        assert!(
            msg.contains("v0.8.5 naf error") && msg.contains(needle),
            "{label}: expected NAF diagnostic containing {needle:?}, got {msg}"
        );
    }
}

#[test]
fn probabilistic_nonmonotone_wfs_remains_separate_from_deterministic_naf() {
    let deterministic = parse_program(
        r#"
        p() :- not q().
        q() :- not p().
    "#,
    )
    .expect("parse deterministic");
    assert!(
        stratify(&deterministic).is_err(),
        "deterministic cycle through negation must remain unstratified"
    );

    let probabilistic = parse_program(
        r#"
        0.5::bias().
        p() :- bias(), not q().
        q() :- not p().
        query(p()).
    "#,
    )
    .expect("parse probabilistic");
    stratify(&probabilistic).expect("probabilistic nonmonotone WFS profile is allowed");
}

#[test]
fn committed_naf_example_compiles() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root");
    let path = "examples/v085-language/naf/closed_world.xlog";
    let full_path = repo_root.join(path);
    let src = std::fs::read_to_string(&full_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", full_path.display()));
    Compiler::new()
        .compile(&src)
        .unwrap_or_else(|err| panic!("{path} failed to compile: {err}"));
}
