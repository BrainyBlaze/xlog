//! Symbol interning order must be source order, exactly as in the reference
//! parser: the fast path defers interning of fact symbols until the fact is
//! merged between pest's statements. This file holds a single test so that it
//! runs in its own process with a fresh symbol table (and unique symbol names
//! in case the table is shared).

use xlog_core::symbol;
use xlog_logic::parse_program;

#[test]
fn fact_symbols_are_interned_in_source_order() {
    // rule (pest residual) mentions `zq_order_first` before the fact mentions
    // `zq_order_second`; a scan-time interning would invert the ids.
    let src = "r(X) :- q(X, zq_order_first).\np(zq_order_second).\ns(zq_order_third) :- p(_).\n";
    parse_program(src).expect("parse");
    let first = symbol::intern("zq_order_first");
    let second = symbol::intern("zq_order_second");
    let third = symbol::intern("zq_order_third");
    assert!(first < second && second < third, "{first} {second} {third}");
}
