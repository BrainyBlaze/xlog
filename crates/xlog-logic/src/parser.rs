//! Parser for XLOG programs using Pest

use pest::Parser;
use pest_derive::Parser;
use xlog_core::{XlogError, Result};

#[derive(Parser)]
#[grammar = "grammar.pest"]
pub struct XlogParser;

/// Parse result containing the parsed pairs
pub type ParseResult<'a> = pest::iterators::Pairs<'a, Rule>;

/// Parse an XLOG program string
pub fn parse_program(input: &str) -> Result<ParseResult<'_>> {
    XlogParser::parse(Rule::program, input)
        .map_err(|e| XlogError::Parse(e.to_string()))
}

/// Parse a single statement
pub fn parse_statement(input: &str) -> Result<ParseResult<'_>> {
    XlogParser::parse(Rule::statement, input)
        .map_err(|e| XlogError::Parse(e.to_string()))
}

/// Parse a single atom
pub fn parse_atom(input: &str) -> Result<ParseResult<'_>> {
    XlogParser::parse(Rule::atom, input)
        .map_err(|e| XlogError::Parse(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fact() {
        let input = "edge(1, 2).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse fact: {:?}", result.err());
    }

    #[test]
    fn test_parse_rule() {
        let input = "reach(X, Y) :- edge(X, Y).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse rule: {:?}", result.err());
    }

    #[test]
    fn test_parse_recursive_rule() {
        let input = "reach(X, Z) :- reach(X, Y), edge(Y, Z).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse recursive rule: {:?}", result.err());
    }

    #[test]
    fn test_parse_negation() {
        let input = "isolated(X) :- node(X), not edge(X, Y).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse negation: {:?}", result.err());
    }

    #[test]
    fn test_parse_aggregate() {
        let input = "out_degree(X, count(Y)) :- edge(X, Y).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse aggregate: {:?}", result.err());
    }

    #[test]
    fn test_parse_constraint() {
        let input = ":- reach(X, X).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse constraint: {:?}", result.err());
    }

    #[test]
    fn test_parse_query() {
        let input = "?- reach(1, N).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse query: {:?}", result.err());
    }

    #[test]
    fn test_parse_full_program() {
        let input = r#"
            edge(1, 2).
            edge(2, 3).
            edge(3, 4).
            reach(X, Y) :- edge(X, Y).
            reach(X, Z) :- reach(X, Y), edge(Y, Z).
            ?- reach(1, N).
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse full program: {:?}", result.err());
    }

    #[test]
    fn test_parse_comparison() {
        let input = "small(X) :- value(X), X < 10.";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse comparison: {:?}", result.err());
    }

    #[test]
    fn test_parse_pred_decl() {
        let input = "pred edge(u32, u32).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse pred decl: {:?}", result.err());
    }
}
