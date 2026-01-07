//! Integration tests for xlog-logic

#[allow(unused_imports)]
use xlog_logic::{parse_program, stratify};

#[test]
fn test_parse_tc_program() {
    let input = include_str!("logic/tc.xlog");
    let result = parse_program(input);
    assert!(result.is_ok(), "Failed to parse TC program: {:?}", result.err());
}

#[test]
fn test_parse_stratified_program() {
    let input = include_str!("logic/stratified.xlog");
    let result = parse_program(input);
    assert!(result.is_ok(), "Failed to parse stratified program: {:?}", result.err());
}

#[test]
fn test_parse_aggregate_program() {
    let input = include_str!("logic/aggregates.xlog");
    let result = parse_program(input);
    assert!(result.is_ok(), "Failed to parse aggregate program: {:?}", result.err());
}

// Note: Full execution tests require xlog-cuda and xlog-runtime
// which depend on CUDA hardware. These will be added in later tasks.
