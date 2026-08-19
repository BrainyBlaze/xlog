//! Randomised differential test: random statement sequences and token soup,
//! fast-path `parse_program` vs the pure-pest reference. Deterministic LCG;
//! `XLOG_PARSE_DIFF_N` overrides the program count (default 20 000 each).
use xlog_logic::parse_program;
use xlog_logic::parser::parse_program_reference;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[(self.next() as usize) % xs.len()]
    }
}

const TOKENS: &[&str] = &[
    "p",
    "q",
    "r",
    "evidence",
    "query",
    "pred",
    "func",
    "nn",
    "learnable",
    "use",
    "domain",
    "X",
    "Y",
    "_",
    "_a",
    "1",
    "-1",
    "1.5",
    "-2.5",
    "0",
    "1.",
    ".5",
    "12",
    "\"s\"",
    "\"a.b\"",
    "\"//\"",
    "\"",
    "(",
    ")",
    "(",
    ")",
    ",",
    ".",
    ".",
    ".",
    ":-",
    "?-",
    "::",
    "=..",
    "=",
    "<",
    ">",
    "!=",
    "==",
    "is",
    "+",
    "-",
    "*",
    "[",
    "]",
    "|",
    "//",
    "\n",
    " ",
    " ",
    " ",
    "\t",
    "#pragma",
    "magic_sets",
    "on",
    "prob_confidence",
    "0.9",
    "not",
    "know",
    "possible",
    "count",
    "sum",
    "true",
    "false",
    "f",
    "g",
    "u32",
    ":",
    "private",
    "{",
    "}",
    "/",
    "\u{e9}",
    "p(1).",
    "q(a, B).",
    "r(x, \"y\", 2.5, -3, Z, _).",
    "a(f(1)).",
    "b([1, 2]).",
    "s(X) :- p(X).",
    "t(X) :- p(X), X =.. [F|A].",
    "0.5::c(1).",
    "pred p(u32).",
    "#pragma magic_sets = on\n",
    "// c.\n",
    "evidence(p(1), true).",
    "query(p(1)).",
];

const STMTS: &[&str] = &[
    "p(1).",
    "p(1,2).",
    "q(a, B).",
    "r(x, \"y\", 2.5, -3, Z, _).",
    "a(f(1)).",
    "b([1, 2]).",
    "c([H|T]).",
    "d().",
    "e(\"a.b\").",
    "e(\"//\").",
    "s(X) :- p(X).",
    "s(X) :- p(X), q(X, Y).",
    "t(X) :- p(X), X =.. [F|A].",
    "t(X) :- X =.. f(1).",
    "t(X) :- X =.. f(A, b).",
    "t(X) :- f(1) =.. X.",
    "t(X) :- X =.. Y.",
    "t(X) :- X =.. [f, 1].",
    "u(X) :- X > 1.5.",
    "u(X) :- X > 1.",
    "u(X) :- X is 1.5 + 2.",
    "u(X) :- X = \"1.\".",
    "v(X) :- not p(X).",
    "w(X) :- know p(X).",
    "w(X) :- not know p(X).",
    "x(X, count(Y)) :- p(X, Y).",
    "0.5::c(1).",
    "0.5::c(1); 0.5::c(2).",
    "pred p(u32).",
    "pred p(x: u32, y: list<u32>).",
    "domain d : u32.",
    "use graph.",
    "use a/b::{c, d}.",
    "func sq(X) = X * X.",
    "func f(X) = if X < 0 then 0 - X else X.",
    "#pragma magic_sets = on\n",
    "#pragma prob_confidence = 0.9\n",
    "#pragma prob_samples = 5\n",
    "// c.\n",
    "// \"\n",
    "evidence(p(1), true).",
    "query(p(1)).",
    "?- p(X).",
    ":- p(X), q(X).",
    "learnable(W) :: h(X) :- b(X).",
    "nn(m, [X], Y) :: out(X, Y).",
    "nn(m, [X], Y, [a, b]) :: out(X, Y).",
    "evidence(1).",
    "query(x).",
    "nn(1).",
    "pred(1).",
    "use(1).",
    "func(1).",
    "p(1)",
    "p(1",
    "p(1) q(2).",
    "p(.5).",
    "p(1.).",
    "p(_a).",
    "p(a(1).",
    ".",
    "..",
    "p(1)..",
    "p(X(1)).",
    "p(\"unterminated",
    "p(1\u{e9}).",
    "s(X) :- .",
    ":-.",
    "?-.",
    "p (1 , 2) .",
    "p(1)\n.",
    "p( // c\n 1).",
    "p(1) // c\n.",
    "t(X) :- X =.. f(1).\nq(g(1)).",
    "t(X) :- X =.. \n f(1). r(f(2)).",
];

fn run(n: usize, seed: u64, gen: impl Fn(&mut Lcg) -> String) -> (usize, Vec<String>) {
    let mut rng = Lcg(seed);
    let mut mismatches: Vec<String> = Vec::new();
    let mut ok_programs = 0usize;
    for _ in 0..n {
        let src = gen(&mut rng);
        let fast = parse_program(&src);
        let reference = parse_program_reference(&src);
        if reference.is_ok() {
            ok_programs += 1;
        }
        let f = format!("{fast:?}");
        let r = format!("{reference:?}");
        if f != r {
            mismatches.push(format!("CASE {src:?}"));
        }
    }
    (ok_programs, mismatches)
}

fn n() -> usize {
    std::env::var("XLOG_PARSE_DIFF_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20000)
}

fn report(label: &str, n: usize, ok: usize, mismatches: &[String]) {
    eprintln!(
        "{label}: programs: {n}, valid per reference: {ok}, mismatches: {}",
        mismatches.len()
    );
    for m in mismatches.iter().take(25) {
        eprintln!("{m}");
    }
}

#[test]
fn random_statement_sequences_match_reference() {
    let n = n();
    let seps = [" ", "\n", "", "\t", "  ", "\n\n", " // c\n", "\r\n"];
    let (ok, mismatches) = run(n, 0xDEADBEEFCAFEBABE, |rng| {
        let len = 1 + (rng.next() as usize % 8);
        let mut src = String::new();
        for _ in 0..len {
            src.push_str(rng.pick(&seps));
            src.push_str(rng.pick(STMTS));
        }
        src.push_str(rng.pick(&seps));
        src
    });
    let non_univ: Vec<String> = mismatches
        .iter()
        .filter(|m| !m.contains("=.."))
        .cloned()
        .collect();
    eprintln!("mismatches without `=..`: {}", non_univ.len());
    report("statements", n, ok, &non_univ);
    assert!(mismatches.is_empty(), "{} mismatches", mismatches.len());
}

#[test]
fn random_token_soup_matches_reference() {
    let n = n();
    let (ok, mismatches) = run(n, 0x9E3779B97F4A7C15, |rng| {
        let len = 1 + (rng.next() as usize % 14);
        let mut src = String::new();
        for _ in 0..len {
            src.push_str(rng.pick(TOKENS));
            if rng.next() % 3 == 0 {
                src.push(' ');
            }
        }
        src
    });
    report("soup", n, ok, &mismatches);
    assert!(mismatches.is_empty(), "{} mismatches", mismatches.len());
}
