//! The fact fast path in `parse_program` must be observationally identical to
//! the pure-pest reference parser: same AST (including statement order), same
//! errors, on the whole in-repo `.xlog` corpus and on adversarial snippets.

use std::path::{Path, PathBuf};
use xlog_logic::parse_program;
use xlog_logic::parser::{parse_program_reference, parse_program_with_stats};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root")
}

fn collect_xlog(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if matches!(name, "target" | ".git" | "node_modules" | ".worktrees") {
                continue;
            }
            collect_xlog(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("xlog") {
            out.push(path);
        }
    }
}

fn assert_same(src: &str, label: &str) {
    let fast = parse_program(src);
    let reference = parse_program_reference(src);
    assert_eq!(
        format!("{fast:?}"),
        format!("{reference:?}"),
        "fast-path parse differs from reference for {label}"
    );
}

#[test]
fn corpus_matches_reference() {
    let mut files = Vec::new();
    collect_xlog(&repo_root(), &mut files);
    assert!(files.len() > 100, "corpus too small: {} files", files.len());
    let mut fast_facts_total = 0usize;
    for path in &files {
        let src = std::fs::read_to_string(path).expect("read xlog");
        assert_same(&src, &path.display().to_string());
        if let Ok((_, stats)) = parse_program_with_stats(&src) {
            fast_facts_total += stats.fast_facts;
        }
    }
    assert!(
        fast_facts_total > 0,
        "fast path never triggered on the corpus"
    );
}

#[test]
fn synthetic_fact_programs_match_reference_and_take_fast_path() {
    let mut src = String::new();
    for i in 0..2000 {
        src.push_str(&format!("event({i}, {}, {}).\n", i % 97, i % 13));
        src.push_str(&format!("a({i}, {}).\n", i + 1));
        src.push_str(&format!(
            "name({i}, \"node {i}\", sym_{i}, {}.5, -{i}).\n",
            i % 7
        ));
        if i % 100 == 0 {
            src.push_str("// a comment line\n");
            src.push_str(&format!("weird(X, _, {i}) .\n"));
        }
    }
    src.push_str("happens(E, T) :- event(E, T, _).\n");
    src.push_str("pair(A, B) :- event(A, T, K), event(B, T, K).\n");
    src.push_str("learnable(W_h) :: h(X, Y) :- bL(X, Z), bR(Z, Y).\n");
    assert_same(&src, "synthetic");
    let (program, stats) = parse_program_with_stats(&src).expect("parse");
    assert!(stats.fast_facts >= 6000, "fast facts: {}", stats.fast_facts);
    // order preserved: first rule is the first fact, rules come last
    assert_eq!(program.rules[0].head.predicate, "event");
    assert_eq!(program.rules.last().unwrap().head.predicate, "pair");
}

#[test]
fn adversarial_snippets_match_reference() {
    let cases: &[&str] = &[
        "p(1).",
        "p(1). q(2).",
        "p(1).q(2).",
        "p (1 , 2) .",
        "p(1)\n.\nq(3).",
        "p(1.5). p(-2). p(-2.25). p(1). p(0).",
        "p(1.). q(2).",
        "p(.5).",
        "p(1.5.3).",
        "p(X). p(_). p(X_1, Y).",
        "p(\"a.b\"). p(\"// not a comment\"). p(\"unterminated",
        "p(\"multi\nline\").",
        "p(\"\").",
        "// comment with p(1).\nq(2).",
        "p( // comment inside\n 1).",
        "p(1) // trailing\n. q(2).",
        "p(1)// comment\n q(2).",
        "evidence(foo(1), true).",
        "evidence(x, true).",
        "query(foo(1)).",
        "query(x).",
        "pred p(int).",
        "pred(int).",
        "domain d : int.",
        "use graph.",
        "use(1).",
        "func(1).",
        "learnable(W) :: h(X) :- b(X).",
        "nn(m, [X], Y) :: out(X, Y).",
        "0.5 :: p(1).",
        "0.5 :: p(1); 0.5 :: p(2).",
        "p(f(1)). p([1,2]). p([H|T]). p([]).",
        "p(1, f(2), 3).",
        "p(true). p(false). p(true, 1).",
        "p(1, 2",
        "p(1, 2).)",
        "p(1,,2).",
        "p(1) q(2).",
        "p(1). :- q(X). ?- r(Y).",
        "#pragma magic_sets = on\np(1).\nq(X) :- p(X).",
        "#pragma prob_confidence = 0.95\np(1).",
        "p(1).\n#pragma magic_sets = on\nq(2).",
        "p(5abc).",
        "p(abc5). p(a_b). p(aB).",
        "p(_foo).",
        "p(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20).",
        "p(12345678901234567890).",
        "p(-).",
        "p(--1).",
        "p(1e5).",
        "p(\u{00e9}).",
        "p(1). \u{00e9}",
        "",
        "   \n\t ",
        "// only a comment",
        ".",
        "p().",
        "p( ).",
        "p(1)..",
        "p(1) :- q(1).",
        "p(1) :- q(1). r(2).",
        "r(2). p(1) :- q(1).",
        "x(1) :- y(1.5). z(2).",
        "x(X) :- X > 1. z(2).",
        "x(X) :- X = 1.5 . z(2).",
        // univ `=..`: its dots are not statement terminators; the right-hand
        // side must never be mistaken for a fact (found in review)
        "r(X) :- X =.. f(1).
q(g(1)).",
        "r(X) :- X =.. f(A).
evidence(q(1), true).",
        "r(X) :- X =.. f(1).
query(r(Y)).",
        "r(X) :- X =..
  f(1).
  q(f(2)).",
        "r(X) :- X =.. f(1).
s(2).",
        "r(X) :- X =.. [F|A].
s(2).",
        "r(X) :- f(1) =.. X. s(2).",
        "r(X) :- X = .. f(1). s(2).",
        "p(1). r(X) :- X =.. f(1). s(2). t(3).",
        // a removed fact must not let its neighbours fuse (found in review)
        "u(X) :- X > 1.nn(1).0.5::c(1).",
        "u(X) :- X > 1.nn(1).0.5::c(1); 0.5::c(2). q(2).",
        "p(1).q(2).0.5::c(1).",
        "a(1).
0.5::c(1).
b(2).",
    ];
    for case in cases {
        assert_same(case, &format!("{case:?}"));
    }
}

/// Valid programs with simple facts must take the fast path without falling
/// back to the reference parser (a fallback is correct but parses the program
/// three times, so it would hide a performance regression).
#[test]
fn valid_programs_take_the_fast_path_without_fallback() {
    let cases: &[&str] = &[
        "p(1). q(X) :- p(X).",
        "r(X) :- X =.. f(1).
q(2).",
        "r(X) :- X =.. [F|A].
q(2).",
        "u(X) :- X > 1.nn(1).0.5::c(1).",
        "pred p(u32).
p(1).
#pragma magic_sets = on
q(X) :- p(X).",
        "// header
a(1, \"s\", sym).
b(2.5, -3, X, _).
c(X) :- a(X, _, _), b(_, _, X, _).",
    ];
    for case in cases {
        let (_, stats) = parse_program_with_stats(case).expect("valid program");
        assert!(!stats.fell_back, "fell back on {case:?}");
        assert!(stats.fast_facts > 0, "no fast facts on {case:?}");
    }
}

/// With facts blanked out of the residual, pest reports errors at the original
/// positions; the reference and fast paths must agree on the message verbatim
/// (covered implicitly by `assert_same`, spelled out here for a multi-line
/// program with facts before the error).
#[test]
fn residual_errors_carry_the_original_positions() {
    let src = "p(1).
q(2).
r(X) :- s(X) t(X).
";
    let fast = parse_program(src).expect_err("syntax error");
    let reference = parse_program_reference(src).expect_err("syntax error");
    assert_eq!(fast.to_string(), reference.to_string());
    assert!(fast.to_string().contains("3:"), "{fast}");
}
