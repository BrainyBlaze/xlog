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
//! Soundness argument (why the residual split cannot change what pest sees):
//!
//! * A statement boundary is a `.` outside string literals and comments that is
//!   not between two ASCII digits. In the grammar `.` occurs only as a
//!   statement terminator, inside `float_num`/`prob_num` (digits on both sides,
//!   atomic rules, no whitespace), inside `string_lit` (`"..."`, no escapes) and
//!   inside `//` comments — so the rule is exact, not a heuristic.
//! * The fast path only accepts a *complete* fact at a statement start. Every
//!   grammar alternative tried before `fact` either starts with a keyword that
//!   cannot be followed by `(...)` `.` in a way that parses (`pred`, `domain`,
//!   `use`, `func`, `nn`, `learnable` — all need more tokens), or is
//!   `evidence(...)`/`query(...)`, which are excluded by name here. For the
//!   remaining text pest itself would pick `fact`.
//! * If the residual fails to parse or build for any reason, the caller re-runs
//!   the pure-pest parser on the original source, so error messages (and their
//!   positions) are exactly the reference ones.

use xlog_core::symbol;

use crate::ast::{Atom, Rule as AstRule, Term};

/// One fact recognised by the fast path, with its byte offset in the original
/// source (used to merge with pest's statements in source order).
#[derive(Debug)]
pub(crate) struct FastFact {
    pub offset: usize,
    pub rule: AstRule,
}

/// Result of scanning a source text.
#[derive(Debug, Default)]
pub(crate) struct FactScan {
    /// Facts handled here, in source order.
    pub facts: Vec<FastFact>,
    /// The source with those facts cut out, for pest.
    pub residual: String,
    /// `(residual_offset, original_offset)` for every copied segment, in order,
    /// so a pest span start inside `residual` maps back to the original.
    pub segments: Vec<(usize, usize)>,
}

impl FactScan {
    /// Map a byte offset inside `residual` back to the original source.
    pub(crate) fn original_offset(&self, residual_offset: usize) -> usize {
        let idx = self
            .segments
            .partition_point(|(res_start, _)| *res_start <= residual_offset);
        let (res_start, orig_start) = self.segments[idx.saturating_sub(1)];
        orig_start + (residual_offset - res_start)
    }
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
/// Returns the rule and the offset just past its terminating `.`.
fn try_fact(source: &str, start: usize) -> Option<(AstRule, usize)> {
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
    pos = skip_ws(src, pos);
    if pos < src.len() && src[pos] == b')' {
        pos += 1;
    } else {
        loop {
            let (term, next) = try_term(source, pos)?;
            terms.push(term);
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
    let rule = AstRule {
        head: Atom {
            predicate: predicate.to_string(),
            terms,
        },
        body: vec![],
    };
    Some((rule, pos + 1))
}

/// Recognise one simple term at `pos`: variable, `_`, integer, float, string
/// literal or symbol. Returns the term and the offset just past it. Anything
/// else (compound, list, cons, comment, ...) returns `None`.
fn try_term(source: &str, pos: usize) -> Option<(Term, usize)> {
    let src = source.as_bytes();
    let b = *src.get(pos)?;
    if b == b'_' {
        // `anonymous` is exactly "_"; the caller then requires `,` or `)`,
        // which rejects `_foo` just like the grammar does.
        return Some((Term::Anonymous, pos + 1));
    }
    if b.is_ascii_uppercase() {
        let mut end = pos + 1;
        while end < src.len() && is_ident_continue(src[end]) {
            end += 1;
        }
        return Some((Term::Variable(source[pos..end].to_string()), end));
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
            return Some((Term::Float(val), end));
        }
        let val: i64 = source[pos..end].parse().ok()?;
        return Some((Term::Integer(val), end));
    }
    if b == b'"' {
        let rel = src[pos + 1..].iter().position(|&c| c == b'"')?;
        let end = pos + 1 + rel;
        return Some((Term::String(source[pos + 1..end].to_string()), end + 1));
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
        return Some((Term::Symbol(symbol::intern(&source[pos..end])), end));
    }
    None
}

/// Scan `source`, extracting simple ground facts and producing the residual.
pub(crate) fn scan(source: &str) -> FactScan {
    let src = source.as_bytes();
    let mut out = FactScan::default();
    let mut pos = 0usize;
    let mut seg_start = 0usize;
    while pos < src.len() {
        let stmt_start = skip_ws_and_comments(src, pos);
        if stmt_start >= src.len() {
            break;
        }
        if let Some((rule, end)) = try_fact(source, stmt_start) {
            if stmt_start > seg_start {
                out.segments.push((out.residual.len(), seg_start));
                out.residual.push_str(&source[seg_start..stmt_start]);
            }
            out.facts.push(FastFact {
                offset: stmt_start,
                rule,
            });
            seg_start = end;
            pos = end;
        } else {
            pos = skip_statement(src, stmt_start);
        }
    }
    if seg_start < src.len() {
        out.segments.push((out.residual.len(), seg_start));
        out.residual.push_str(&source[seg_start..]);
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
        assert_eq!(scan.facts[0].rule.head.predicate, "p");
        assert_eq!(
            scan.facts[0].rule.head.terms,
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
        assert_eq!(scan.facts[1].rule.head.predicate, "r");
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
            assert_eq!(scan.residual, src);
        }
    }

    #[test]
    fn statement_skipping_respects_strings_comments_and_decimals() {
        let src = "x(X) :- X > 1.5, y(\"a.b\"). // c.\np(1).";
        let scan = scan(src);
        assert_eq!(scan.facts.len(), 1);
        assert_eq!(scan.facts[0].offset, src.find("p(1)").unwrap());
    }

    #[test]
    fn offsets_map_back_to_the_original() {
        let src = "a(1). q(X) :- a(X). b(2). r(Y) :- b(Y).";
        let scan = scan(src);
        let q = scan.residual.find("q(X)").unwrap();
        let r = scan.residual.find("r(Y)").unwrap();
        assert_eq!(scan.original_offset(q), src.find("q(X)").unwrap());
        assert_eq!(scan.original_offset(r), src.find("r(Y)").unwrap());
    }
}
