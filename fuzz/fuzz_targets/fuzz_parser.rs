//! Fuzz target for the XLOG parser.
//!
//! This target feeds arbitrary byte sequences to the parser to find
//! crashes, hangs, and assertion failures.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Try to interpret the bytes as UTF-8 and parse
    if let Ok(input) = std::str::from_utf8(data) {
        // The parser should never panic on any input
        let _ = xlog_logic::parse_program(input);
    }
});
