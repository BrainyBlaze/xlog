//! Fuzz target for the XLOG compiler.
//!
//! This target generates valid-looking XLOG programs and compiles them
//! to find edge cases in the compiler.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

/// A fuzzer-generated XLOG program structure
#[derive(Debug, Arbitrary)]
struct FuzzProgram {
    /// Number of predicates (0-10)
    num_predicates: u8,
    /// Number of facts per predicate (0-5)
    facts_per_pred: u8,
    /// Number of rules (0-10)
    num_rules: u8,
    /// Include negation in some rules
    include_negation: bool,
    /// Include recursion
    include_recursion: bool,
    /// Include aggregates
    include_aggregates: bool,
    /// Random seed for deterministic generation
    seed: u32,
}

impl FuzzProgram {
    fn to_source(&self) -> String {
        let mut source = String::new();
        let mut rng_state = self.seed as u64;

        // Simple LCG for deterministic randomness
        let mut next_rand = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            rng_state
        };

        let num_preds = (self.num_predicates % 10).max(1) as usize;
        let facts_per = (self.facts_per_pred % 5) as usize;
        let num_rules = (self.num_rules % 10) as usize;

        // Generate predicate declarations
        let types = ["i64", "u32", "f64"];
        let mut pred_arities: Vec<usize> = Vec::new();

        for i in 0..num_preds {
            let arity = ((next_rand() % 4) + 1) as usize;
            pred_arities.push(arity);

            source.push_str("pred p");
            source.push_str(&i.to_string());
            source.push('(');
            for j in 0..arity {
                if j > 0 {
                    source.push_str(", ");
                }
                let typ = types[(next_rand() % 3) as usize];
                source.push_str(typ);
            }
            source.push_str(").\n");
        }
        source.push('\n');

        // Generate facts
        for i in 0..num_preds {
            let arity = pred_arities[i];
            for _ in 0..facts_per {
                source.push_str("p");
                source.push_str(&i.to_string());
                source.push('(');
                for j in 0..arity {
                    if j > 0 {
                        source.push_str(", ");
                    }
                    let val = (next_rand() % 100) as i64;
                    source.push_str(&val.to_string());
                }
                source.push_str(").\n");
            }
        }
        source.push('\n');

        // Generate rules
        for _ in 0..num_rules {
            let head_pred = (next_rand() % num_preds as u64) as usize;
            let head_arity = pred_arities[head_pred];

            // Head
            source.push_str("p");
            source.push_str(&head_pred.to_string());
            source.push('(');
            for j in 0..head_arity {
                if j > 0 {
                    source.push_str(", ");
                }
                let var_name = (b'A' + (j % 26) as u8) as char;
                source.push(var_name);
            }
            source.push_str(") :- ");

            // Body with 1-3 literals
            let num_body_lits = ((next_rand() % 3) + 1) as usize;
            for k in 0..num_body_lits {
                if k > 0 {
                    source.push_str(", ");
                }

                // Negation?
                if self.include_negation && next_rand() % 4 == 0 {
                    source.push_str("not ");
                }

                // Pick a body predicate
                let body_pred = if self.include_recursion && next_rand() % 3 == 0 {
                    head_pred // Self-reference for recursion
                } else {
                    (next_rand() % num_preds as u64) as usize
                };

                let body_arity = pred_arities[body_pred];
                source.push_str("p");
                source.push_str(&body_pred.to_string());
                source.push('(');
                for j in 0..body_arity {
                    if j > 0 {
                        source.push_str(", ");
                    }
                    // Mix of variables and constants
                    if next_rand() % 3 == 0 {
                        let val = (next_rand() % 100) as i64;
                        source.push_str(&val.to_string());
                    } else {
                        let var_name = (b'A' + ((j + k) % 26) as u8) as char;
                        source.push(var_name);
                    }
                }
                source.push(')');
            }

            source.push_str(".\n");
        }

        // Generate aggregate rules if enabled
        if self.include_aggregates && num_preds > 0 {
            source.push_str("\npred agg_result(i64, i64).\n");

            // count aggregate
            let agg_pred = (next_rand() % num_preds as u64) as usize;
            let agg_arity = pred_arities[agg_pred];
            if agg_arity >= 1 {
                source.push_str("agg_result(G, count(V)) :- p");
                source.push_str(&agg_pred.to_string());
                source.push('(');
                source.push_str("G");
                for j in 1..agg_arity {
                    source.push_str(", ");
                    if j == 1 {
                        source.push_str("V");
                    } else {
                        source.push('_');
                    }
                }
                source.push_str(").\n");
            }
        }

        // Add a query
        if num_preds > 0 {
            source.push_str("\n?- p0(");
            let arity = pred_arities[0];
            for j in 0..arity {
                if j > 0 {
                    source.push_str(", ");
                }
                let var_name = (b'X' + (j % 3) as u8) as char;
                source.push(var_name);
            }
            source.push_str(").\n");
        }

        source
    }
}

fuzz_target!(|program: FuzzProgram| {
    let source = program.to_source();

    // Parse and compile - should not panic
    match xlog_logic::parse_program(&source) {
        Ok(ast) => {
            // Successfully parsed, now try to compile
            let mut compiler = xlog_logic::Compiler::new();
            let _ = compiler.compile_program(&ast);
        }
        Err(_) => {
            // Parse errors are expected for some inputs
        }
    }
});
