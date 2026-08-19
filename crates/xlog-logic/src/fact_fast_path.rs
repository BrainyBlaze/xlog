//! Hand-written fast path for simple ground facts inside `parse_program`.
//!
//! In the pest grammar a fact is the 13th `statement` alternative, so every
//! `p(1, 2).` line is first tried as a function definition, a rule (whose head
//! parses completely before `:-` fails), a probabilistic fact, ... and only then
//! as a fact — and every token allocates a `Pair`. Fact-heavy programs (ILP
//! event logs, EDB dumps) therefore spend most of their frontend time in the
//! parser.
//!
//! This module scans the source once, recognises statements of the shape
//!
//! ```text
//! ident "(" [ term { "," term } ] ")" "."
//! ```
//!
//! where every `term` is an integer, a float, a symbol (ident), a variable, `_`
//! or a `"string"`, builds the very same AST nodes `build_term`/`build_fact`
//! would, and leaves *everything else* (rules, declarations, pragmas, facts with
//! compound/list arguments, anything with a comment between tokens, ...) to pest
//! as a "residual" source. The caller parses the residual with pest and merges
//! the two statement streams back in source order.
//!
//! Soundness argument (why the residual cannot change what pest sees):
//!
//! * The residual is the original source with every recognised fact **blanked
//!   out** — replaced byte for byte by spaces (newlines kept). It has the same
//!   length, so byte offsets in the residual are byte offsets in the original
//!   (no remapping), pest's error positions are the original ones, and the
//!   tokens around a removed fact can never fuse (`X > 1.` `nn(1).` `0.5::c(1).`
//!   stays `X > 1.` `      ` `0.5::c(1).` rather than becoming `1.0.5`).
//!   Blanks are grammar WHITESPACE, which is implicit between statements.
//! * A statement boundary is a `.` outside string literals and comments that is
//!   not between two ASCII digits and is not part of the univ operator `=..`.
//!   In the grammar `.` occurs only as a statement terminator, inside
//!   `float_num`/`prob_num` (digits on both sides, atomic rules, no
//!   whitespace), inside `string_lit` (`"..."`, no escapes), inside `//`
//!   comments and in `=..` — the scanner handles each of these explicitly.
//! * The fast path only accepts a *complete* fact at a statement start. Every
//!   grammar alternative tried before `fact` either starts with a keyword that
//!   cannot be followed by `(...)` `.` in a way that parses (`pred`, `domain`,
//!   `use`, `func`, `nn`, `learnable` — all need more tokens), or is
//!   `evidence(...)`/`query(...)`, which are excluded by name here. For the
//!   remaining text pest itself would pick `fact`.
//! * Symbols are interned when a fact is merged into the program, i.e. in
//!   source order interleaved with pest's statements, so symbol ids are the
//!   ones the reference parser would assign.
//! * If the residual fails to parse or build for any reason, the caller re-runs
//!   the pure-pest parser on the original source, so error messages (and their
//!   positions) are exactly the reference ones.
//!
//! `crates/xlog-logic/tests/parse_fast_facts*.rs` check all of this against the
//! reference parser on the in-repo corpus, adversarial snippets and randomised
//! statement sequences.

use xlog_core::symbol;

use crate::ast::{Atom, Rule as AstRule, Term};

/// One fact recognised by the fast path, with its byte offset in the original
/// source (used to merge with pest's statements in source order).
#[derive(Debug)]
pub(crate) struct FastFact<'a> {
    /// Byte offset of the fact's first byte in the original source.
    pub offset: usize,
    /// The fact; symbol terms are placeholders until [`FastFact::into_rule`].
    rule: AstRule,
    /// `(term index, symbol text)` for every symbol term of the head, interned
    /// only when the fact is merged so that interning order is source order.
    symbols: Vec<(usize, &'a str)>,
}

impl FastFact<'_> {
    /// Finish the fact: intern its symbols (now, in merge order) and return it.
    pub(crate) fn into_rule(self) -> AstRule {
        let mut rule = self.rule;
        for (index, text) in self.symbols {
            rule.head.terms[index] = Term::Symbol(symbol::intern(text));
        }
        rule
    }
}

/// Result of scanning a source text.
#[derive(Debug, Default)]
pub(crate) struct FactScan<'a> {
    /// Facts handled here, in source order.
    pub facts: Vec<FastFact<'a>>,
    /// The source with those facts blanked out, for pest. Same length as the
    /// source, so offsets coincide. Empty when no fact was recognised.
    pub residual: String,
    /// Bytes occupied by the recognised facts (what pest does not re-parse).
    pub fact_bytes: usize,
}

/// Statement-start predicates the grammar treats specially (they come before
/// `fact` in `statement` and are fact-shaped): never fast-path them.
fn is_reserved_statement_head(ident: &str) -> bool {
    matches!(ident, "evidence" | "query")
}

#[inline]
fn is_ws(b: u8) -> bool {
    // Exactly the grammar's WHITESPACE rule: " " | "\t" | "\r" | "\n".
    matches!(b, b' ' | b'\t' | b'\r' | b'\n')
}

#[inline]
fn is_ident_start(b: u8) -> bool {
    b.is_ascii_lowercase()
}

#[inline]
fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Skip whitespace and `//` comments (what pest skips implicitly between
/// statements). Returns the offset of the next significant byte.
fn skip_ws_and_comments(src: &[u8], mut pos: usize) -> usize {
    loop {
        while pos < src.len() && is_ws(src[pos]) {
            pos += 1;
        }
        if pos + 1 < src.len() && src[pos] == b'/' && src[pos + 1] == b'/' {
            while pos < src.len() && src[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        return pos;
    }
}

/// Skip whitespace only (inside a fact we bail on comments instead of
/// modelling pest's implicit COMMENT between tokens).
fn skip_ws(src: &[u8], mut pos: usize) -> usize {
    while pos < src.len() && is_ws(src[pos]) {
        pos += 1;
    }
    pos
}

/// From `pos` (a statement start that is not a fast fact), find the offset just
/// past the statement terminator `.`; `src.len()` if there is none.
fn skip_statement(src: &[u8], mut pos: usize) -> usize {
    let mut in_string = false;
    while pos < src.len() {
        let b = src[pos];
        if in_string {
            if b == b'"' {
                in_string = false;
            }
            pos += 1;
            continue;
        }
        match b {
            b'"' => in_string = true,
            // The univ operator `=..` is the one token besides float/prob
            // literals that contains `.`: neither dot is a terminator.
            b'=' if pos + 2 < src.len() && src[pos + 1] == b'.' && src[pos + 2] == b'.' => {
                pos += 3;
                continue;
            }
            b'/' if pos + 1 < src.len() && src[pos + 1] == b'/' => {
                while pos < src.len() && src[pos] != b'\n' {
                    pos += 1;
                }
                continue;
            }
            b'.' => {
                let decimal = pos > 0
                    && src[pos - 1].is_ascii_digit()
                    && pos + 1 < src.len()
                    && src[pos + 1].is_ascii_digit();
                if !decimal {
                    return pos + 1;
                }
            }
            _ => {}
        }
        pos += 1;
    }
    src.len()
}

/// Try to recognise a simple ground fact starting exactly at `start`.
/// Returns the fact and the offset just past its terminating `.`.
fn try_fact(source: &str, start: usize) -> Option<(FastFact<'_>, usize)> {
    let src = source.as_bytes();
    let mut pos = start;
    if pos >= src.len() || !is_ident_start(src[pos]) {
        return None;
    }
    while pos < src.len() && is_ident_continue(src[pos]) {
        pos += 1;
    }
    let predicate = &source[start..pos];
    if is_reserved_statement_head(predicate) {
        return None;
    }
    pos = skip_ws(src, pos);
    if pos >= src.len() || src[pos] != b'(' {
        return None;
    }
    pos += 1;
    let mut terms = Vec::new();
    let mut symbols = Vec::new();
    pos = skip_ws(src, pos);
    if pos < src.len() && src[pos] == b')' {
        pos += 1;
    } else {
        loop {
            let (term, next) = try_term(source, pos)?;
            match term {
                FastTerm::Term(term) => terms.push(term),
                FastTerm::Symbol(text) => {
                    symbols.push((terms.len(), text));
                    terms.push(Term::Symbol(0));
                }
            }
            pos = skip_ws(src, next);
            match src.get(pos) {
                Some(b',') => {
                    pos = skip_ws(src, pos + 1);
                }
                Some(b')') => {
                    pos += 1;
                    break;
                }
                _ => return None,
            }
        }
    }
    pos = skip_ws(src, pos);
    if pos >= src.len() || src[pos] != b'.' {
        return None;
    }
    let fact = FastFact {
        offset: start,
        rule: AstRule {
            head: Atom {
                predicate: predicate.to_string(),
                terms,
            },
            body: vec![],
        },
        symbols,
    };
    Some((fact, pos + 1))
}

/// A recognised term: either final, or a symbol whose interning is deferred.
enum FastTerm<'a> {
    Term(Term),
    Symbol(&'a str),
}

/// Recognise one simple term at `pos`: variable, `_`, integer, float, string
/// literal or symbol. Returns the term and the offset just past it. Anything
/// else (compound, list, cons, comment, ...) returns `None`.
fn try_term(source: &str, pos: usize) -> Option<(FastTerm<'_>, usize)> {
    let src = source.as_bytes();
    let b = *src.get(pos)?;
    if b == b'_' {
        // `anonymous` is exactly "_"; the caller then requires `,` or `)`,
        // which rejects `_foo` just like the grammar does.
        return Some((FastTerm::Term(Term::Anonymous), pos + 1));
    }
    if b.is_ascii_uppercase() {
        let mut end = pos + 1;
        while end < src.len() && is_ident_continue(src[end]) {
            end += 1;
        }
        return Some((
            FastTerm::Term(Term::Variable(source[pos..end].to_string())),
            end,
        ));
    }
    if b == b'-' || b.is_ascii_digit() {
        let mut end = pos;
        if src[end] == b'-' {
            end += 1;
        }
        let digits_start = end;
        while end < src.len() && src[end].is_ascii_digit() {
            end += 1;
        }
        if end == digits_start {
            return None;
        }
        // float_num = "-"? digits "." digits — atomic, digits on both sides.
        if end + 1 < src.len() && src[end] == b'.' && src[end + 1].is_ascii_digit() {
            end += 1;
            while end < src.len() && src[end].is_ascii_digit() {
                end += 1;
            }
            let val: f64 = source[pos..end].parse().ok()?;
            return Some((FastTerm::Term(Term::Float(val)), end));
        }
        let val: i64 = source[pos..end].parse().ok()?;
        return Some((FastTerm::Term(Term::Integer(val)), end));
    }
    if b == b'"' {
        let rel = src[pos + 1..].iter().position(|&c| c == b'"')?;
        let end = pos + 1 + rel;
        return Some((
            FastTerm::Term(Term::String(source[pos + 1..end].to_string())),
            end + 1,
        ));
    }
    if is_ident_start(b) {
        let mut end = pos + 1;
        while end < src.len() && is_ident_continue(src[end]) {
            end += 1;
        }
        // `ident "("` would be a compound term: leave it to pest.
        let after = skip_ws(src, end);
        if after < src.len() && src[after] == b'(' {
            return None;
        }
        return Some((FastTerm::Symbol(&source[pos..end]), end));
    }
    None
}

/// Append `source[start..end]` to the residual with every byte blanked to a
/// space, newlines kept (line numbers in pest diagnostics stay intact).
fn push_blanked(residual: &mut String, source: &str, start: usize, end: usize) {
    const SPACES: &str = "                                                                ";
    for line in source[start..end].split_inclusive('\n') {
        let (body, newline) = match line.strip_suffix('\n') {
            Some(body) => (body, true),
            None => (line, false),
        };
        let mut remaining = body.len();
        while remaining > 0 {
            let chunk = remaining.min(SPACES.len());
            residual.push_str(&SPACES[..chunk]);
            remaining -= chunk;
        }
        if newline {
            residual.push('\n');
        }
    }
}

/// Scan `source`, extracting simple ground facts and producing the residual.
pub(crate) fn scan(source: &str) -> FactScan<'_> {
    let src = source.as_bytes();
    let mut out = FactScan::default();
    let mut pos = 0usize;
    // Start of the source run not yet copied into the residual. The residual
    // is only materialised once the first fact is found, so programs without
    // simple facts pay for the scan but not for a copy of their source.
    let mut copied_to = 0usize;
    while pos < src.len() {
        let stmt_start = skip_ws_and_comments(src, pos);
        if stmt_start >= src.len() {
            break;
        }
        if let Some((fact, end)) = try_fact(source, stmt_start) {
            if out.facts.is_empty() {
                out.residual.reserve(source.len());
            }
            out.residual.push_str(&source[copied_to..stmt_start]);
            push_blanked(&mut out.residual, source, stmt_start, end);
            out.fact_bytes += end - stmt_start;
            out.facts.push(fact);
            copied_to = end;
            pos = end;
        } else {
            pos = skip_statement(src, stmt_start);
        }
    }
    if !out.facts.is_empty() {
        out.residual.push_str(&source[copied_to..]);
        debug_assert_eq!(out.residual.len(), source.len());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognises_simple_facts_and_leaves_the_rest() {
        let src = "p(1, -2, 3.5, \"s\", sym, X, _).\nq(X) :- p(X, _, _, _, _, _, _).\nr().\n";
        let scan = scan(src);
        assert_eq!(scan.facts.len(), 2);
        let facts: Vec<AstRule> = scan.facts.into_iter().map(FastFact::into_rule).collect();
        assert_eq!(facts[0].head.predicate, "p");
        assert_eq!(
            facts[0].head.terms,
            vec![
                Term::Integer(1),
                Term::Integer(-2),
                Term::Float(3.5),
                Term::String("s".to_string()),
                Term::Symbol(symbol::intern("sym")),
                Term::Variable("X".to_string()),
                Term::Anonymous,
            ]
        );
        assert_eq!(facts[1].head.predicate, "r");
        assert_eq!(scan.residual.len(), src.len());
        assert_eq!(scan.residual.trim(), "q(X) :- p(X, _, _, _, _, _, _).");
    }

    #[test]
    fn leaves_compound_list_comment_and_reserved_to_pest() {
        for src in [
            "p(f(1)).",
            "p([1]).",
            "p( // c\n 1).",
            "evidence(foo(1), true).",
            "query(x).",
            "p(1) :- q(1).",
            "p(1)",
            "p(_foo).",
            "p(1.).",
        ] {
            let scan = scan(src);
            assert!(scan.facts.is_empty(), "{src:?} should not be fast-pathed");
            assert!(
                scan.residual.is_empty(),
                "no residual is built without facts"
            );
        }
    }

    #[test]
    fn statement_skipping_respects_strings_comments_decimals_and_univ() {
        let src = "x(X) :- X > 1.5, y(\"a.b\"). // c.\np(1).";
        let first = scan(src);
        assert_eq!(first.facts.len(), 1);
        assert_eq!(first.facts[0].offset, src.find("p(1)").unwrap());

        // `=..` is one token: its dots are not terminators, so `f(1)` below is
        // the univ's right-hand side, not a fact.
        let src = "r(X) :- X =.. f(1).\nq(g(1)).";
        let second = scan(src);
        assert!(second.facts.is_empty(), "{:?}", second.facts);
    }

    #[test]
    fn blanking_keeps_offsets_and_never_fuses_neighbours() {
        let src = "u(X) :- X > 1.nn(1).0.5::c(1).";
        let scan = scan(src);
        assert_eq!(scan.facts.len(), 1);
        assert_eq!(scan.residual.len(), src.len());
        assert_eq!(scan.residual, "u(X) :- X > 1.      0.5::c(1).");
        assert_eq!(scan.fact_bytes, "nn(1).".len());
    }
}
