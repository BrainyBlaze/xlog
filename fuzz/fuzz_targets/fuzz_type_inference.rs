//! Fuzz target for XLOG type inference.
//!
//! This target generates programs with various type combinations
//! to stress-test the type inference system.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

/// Scalar type for fuzzing
#[derive(Debug, Clone, Copy, Arbitrary)]
enum FuzzType {
    I32,
    I64,
    U32,
    U64,
    F32,
    F64,
    Bool,
}

impl FuzzType {
    fn to_str(self) -> &'static str {
        match self {
            FuzzType::I32 => "i32",
            FuzzType::I64 => "i64",
            FuzzType::U32 => "u32",
            FuzzType::U64 => "u64",
            FuzzType::F32 => "f32",
            FuzzType::F64 => "f64",
            FuzzType::Bool => "bool",
        }
    }

    fn sample_value(self, seed: u64) -> String {
        match self {
            FuzzType::I32 | FuzzType::I64 => {
                let val = (seed % 1000) as i64 - 500;
                val.to_string()
            }
            FuzzType::U32 | FuzzType::U64 => {
                let val = seed % 1000;
                val.to_string()
            }
            FuzzType::F32 | FuzzType::F64 => {
                let val = ((seed % 1000) as f64) / 10.0 - 50.0;
                format!("{:.2}", val)
            }
            FuzzType::Bool => {
                if seed % 2 == 0 { "0" } else { "1" }.to_string()
            }
        }
    }
}

/// Comparison operation for fuzzing
#[derive(Debug, Clone, Copy, Arbitrary)]
enum FuzzCmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl FuzzCmpOp {
    fn to_str(self) -> &'static str {
        match self {
            FuzzCmpOp::Eq => "=",
            FuzzCmpOp::Ne => "!=",
            FuzzCmpOp::Lt => "<",
            FuzzCmpOp::Le => "<=",
            FuzzCmpOp::Gt => ">",
            FuzzCmpOp::Ge => ">=",
        }
    }
}

/// Arithmetic operation for fuzzing
#[derive(Debug, Clone, Copy, Arbitrary)]
enum FuzzArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

impl FuzzArithOp {
    fn to_str(self) -> &'static str {
        match self {
            FuzzArithOp::Add => "+",
            FuzzArithOp::Sub => "-",
            FuzzArithOp::Mul => "*",
            FuzzArithOp::Div => "/",
            FuzzArithOp::Mod => "%",
        }
    }
}

/// A fuzzer-generated program for type inference testing
#[derive(Debug, Arbitrary)]
struct FuzzTypeProgram {
    /// Types for predicate columns
    pred_types: Vec<FuzzType>,
    /// Include arithmetic expressions
    include_arith: bool,
    /// Include comparisons with mixed types
    include_mixed_cmp: bool,
    /// Comparison operations to test
    cmp_ops: Vec<FuzzCmpOp>,
    /// Arithmetic operations to test
    arith_ops: Vec<FuzzArithOp>,
    /// Seed for value generation
    seed: u32,
}

impl FuzzTypeProgram {
    fn to_source(&self) -> String {
        let mut source = String::new();
        let mut rng_state = self.seed as u64;

        let mut next_rand = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            rng_state
        };

        // Limit column count
        let num_cols = self.pred_types.len().min(5).max(1);

        // Predicate declaration
        source.push_str("pred data(");
        for (i, typ) in self.pred_types.iter().take(num_cols).enumerate() {
            if i > 0 {
                source.push_str(", ");
            }
            source.push_str(typ.to_str());
        }
        source.push_str(").\n");

        // Some facts
        for _ in 0..3 {
            source.push_str("data(");
            for (i, typ) in self.pred_types.iter().take(num_cols).enumerate() {
                if i > 0 {
                    source.push_str(", ");
                }
                source.push_str(&typ.sample_value(next_rand()));
            }
            source.push_str(").\n");
        }
        source.push('\n');

        // Rules with comparisons
        if !self.cmp_ops.is_empty() && num_cols >= 2 {
            source.push_str("pred cmp_result(");
            for (i, typ) in self.pred_types.iter().take(num_cols).enumerate() {
                if i > 0 {
                    source.push_str(", ");
                }
                source.push_str(typ.to_str());
            }
            source.push_str(").\n");

            for (idx, op) in self.cmp_ops.iter().take(4).enumerate() {
                source.push_str("cmp_result(");
                for i in 0..num_cols {
                    if i > 0 {
                        source.push_str(", ");
                    }
                    let var = (b'A' + i as u8) as char;
                    source.push(var);
                }
                source.push_str(") :- data(");
                for i in 0..num_cols {
                    if i > 0 {
                        source.push_str(", ");
                    }
                    let var = (b'A' + i as u8) as char;
                    source.push(var);
                }
                source.push_str("), ");

                // Add comparison
                let left_var = (b'A' + (idx % num_cols) as u8) as char;
                let right = if self.include_mixed_cmp {
                    self.pred_types[idx % num_cols].sample_value(next_rand())
                } else {
                    let right_var = (b'A' + ((idx + 1) % num_cols) as u8) as char;
                    right_var.to_string()
                };
                source.push(left_var);
                source.push(' ');
                source.push_str(op.to_str());
                source.push(' ');
                source.push_str(&right);
                source.push_str(".\n");
            }
            source.push('\n');
        }

        // Rules with arithmetic (is expressions)
        if self.include_arith && !self.arith_ops.is_empty() && num_cols >= 1 {
            let first_numeric_type = self.pred_types.iter()
                .take(num_cols)
                .find(|t| !matches!(t, FuzzType::Bool))
                .copied()
                .unwrap_or(FuzzType::I64);

            source.push_str(&format!("pred arith_result({}, {}).\n",
                first_numeric_type.to_str(),
                first_numeric_type.to_str()));

            for op in self.arith_ops.iter().take(4) {
                // Skip modulo for floats
                if matches!(first_numeric_type, FuzzType::F32 | FuzzType::F64)
                    && matches!(op, FuzzArithOp::Mod) {
                    continue;
                }

                source.push_str("arith_result(A, R) :- data(");
                for i in 0..num_cols {
                    if i > 0 {
                        source.push_str(", ");
                    }
                    let var = (b'A' + i as u8) as char;
                    source.push(var);
                }
                source.push_str("), R is A ");
                source.push_str(op.to_str());
                source.push(' ');
                let operand = first_numeric_type.sample_value(next_rand() | 1); // Avoid division by zero
                source.push_str(&operand);
                source.push_str(".\n");
            }
            source.push('\n');
        }

        // Query
        source.push_str("?- data(");
        for i in 0..num_cols {
            if i > 0 {
                source.push_str(", ");
            }
            let var = (b'X' + (i % 3) as u8) as char;
            source.push(var);
        }
        source.push_str(").\n");

        source
    }
}

fuzz_target!(|program: FuzzTypeProgram| {
    let source = program.to_source();

    // Parse and compile - type inference should not panic
    match xlog_logic::parse_program(&source) {
        Ok(ast) => {
            let mut compiler = xlog_logic::Compiler::new();
            // Type inference happens during compilation
            let _ = compiler.compile_program(&ast);
        }
        Err(_) => {
            // Parse errors are expected for some generated inputs
        }
    }
});
