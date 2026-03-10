//! Function registry and validation for user-defined functions.

use crate::ast::{ArithExpr, CompOp, CondExpr, FuncBody, FuncDef, Program};
use std::collections::{HashMap, HashSet};
use xlog_core::ScalarType;

/// Errors related to functions
#[derive(Debug, Clone)]
pub enum FunctionError {
    /// Duplicate function definition
    DuplicateDefinition { name: String },
    /// Recursive function without base case
    RecursionWithoutBaseCase { name: String },
    /// Undefined function called
    UndefinedFunction { name: String },
    /// Maximum recursion depth exceeded
    MaxRecursionDepth { name: String, depth: u32 },
    /// Function name conflicts with predicate
    NameConflict { name: String },
}

impl std::fmt::Display for FunctionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FunctionError::DuplicateDefinition { name } => {
                write!(f, "error[E0501]: duplicate function definition `{}`", name)
            }
            FunctionError::RecursionWithoutBaseCase { name } => {
                writeln!(
                    f,
                    "error[E0502]: recursive function `{}` without base case",
                    name
                )?;
                write!(
                    f,
                    "  = help: use conditional form: `if <condition> then <base> else <recursive>`"
                )
            }
            FunctionError::UndefinedFunction { name } => {
                write!(f, "error[E0503]: undefined function `{}`", name)
            }
            FunctionError::MaxRecursionDepth { name, depth } => {
                write!(
                    f,
                    "error[E0504]: maximum recursion depth ({}) exceeded in function `{}`",
                    depth, name
                )
            }
            FunctionError::NameConflict { name } => {
                write!(
                    f,
                    "error[E0505]: `{}` is already defined as a predicate",
                    name
                )
            }
        }
    }
}

impl std::error::Error for FunctionError {}

/// Type errors
#[derive(Debug, Clone)]
pub enum TypeError {
    /// Type mismatch
    Mismatch {
        expected: ScalarType,
        found: ScalarType,
        location: String,
    },
    /// Cannot infer type
    CannotInfer { name: String },
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeError::Mismatch {
                expected,
                found,
                location,
            } => {
                writeln!(f, "error[E0506]: type mismatch in {}", location)?;
                write!(f, "  expected {:?}, found {:?}", expected, found)
            }
            TypeError::CannotInfer { name } => {
                write!(f, "error[E0507]: cannot infer type for `{}`", name)
            }
        }
    }
}

impl std::error::Error for TypeError {}

impl From<FunctionError> for xlog_core::XlogError {
    fn from(e: FunctionError) -> Self {
        xlog_core::XlogError::Compilation(e.to_string())
    }
}

impl From<TypeError> for xlog_core::XlogError {
    fn from(e: TypeError) -> Self {
        xlog_core::XlogError::Type(e.to_string())
    }
}

/// Warning for potentially infinite recursion
#[derive(Debug, Clone)]
pub struct RecursionWarning {
    pub func_name: String,
    pub message: String,
}

impl std::fmt::Display for RecursionWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "warning[W0502]: potentially infinite recursion in `{}`",
            self.func_name
        )?;
        writeln!(f, "  {}", self.message)?;
        write!(
            f,
            "  = note: base case may be unreachable with given recursive call"
        )
    }
}

/// Registry of user-defined functions
#[derive(Debug, Default)]
pub struct FunctionRegistry {
    functions: HashMap<String, FuncDef>,
    call_graph: HashMap<String, HashSet<String>>,
}

impl FunctionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a function
    pub fn register(&mut self, func: FuncDef) -> Result<(), FunctionError> {
        if self.functions.contains_key(&func.name) {
            return Err(FunctionError::DuplicateDefinition {
                name: func.name.clone(),
            });
        }

        // Build call graph
        let calls = Self::extract_calls(&func.body);
        self.call_graph.insert(func.name.clone(), calls);
        self.functions.insert(func.name.clone(), func);

        Ok(())
    }

    /// Get a function by name
    pub fn get(&self, name: &str) -> Option<&FuncDef> {
        self.functions.get(name)
    }

    /// Check if a function exists
    pub fn contains(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    /// Extract function calls from a body
    fn extract_calls(body: &FuncBody) -> HashSet<String> {
        let mut calls = HashSet::new();
        Self::extract_calls_from_body(body, &mut calls);
        calls
    }

    fn extract_calls_from_body(body: &FuncBody, calls: &mut HashSet<String>) {
        match body {
            FuncBody::Arithmetic(expr) => Self::extract_calls_from_expr(expr, calls),
            FuncBody::Conditional(cond) => {
                Self::extract_calls_from_expr(&cond.cond_left, calls);
                Self::extract_calls_from_expr(&cond.cond_right, calls);
                Self::extract_calls_from_body(&cond.then_branch, calls);
                Self::extract_calls_from_body(&cond.else_branch, calls);
            }
            FuncBody::Predicate { .. } => {
                // Predicate bodies don't contain function calls in expressions
            }
        }
    }

    fn extract_calls_from_expr(expr: &ArithExpr, calls: &mut HashSet<String>) {
        match expr {
            ArithExpr::FuncCall { name, args } => {
                calls.insert(name.clone());
                for arg in args {
                    Self::extract_calls_from_expr(arg, calls);
                }
            }
            ArithExpr::Add(l, r)
            | ArithExpr::Sub(l, r)
            | ArithExpr::Mul(l, r)
            | ArithExpr::Div(l, r)
            | ArithExpr::Mod(l, r)
            | ArithExpr::Min(l, r)
            | ArithExpr::Max(l, r)
            | ArithExpr::Pow(l, r) => {
                Self::extract_calls_from_expr(l, calls);
                Self::extract_calls_from_expr(r, calls);
            }
            ArithExpr::Abs(e) | ArithExpr::Cast(e, _) => {
                Self::extract_calls_from_expr(e, calls);
            }
            ArithExpr::Variable(_) | ArithExpr::Integer(_) | ArithExpr::Float(_) => {}
            ArithExpr::Conditional {
                cond_left,
                cond_right,
                then_expr,
                else_expr,
                ..
            } => {
                Self::extract_calls_from_expr(cond_left, calls);
                Self::extract_calls_from_expr(cond_right, calls);
                Self::extract_calls_from_expr(then_expr, calls);
                Self::extract_calls_from_expr(else_expr, calls);
            }
        }
    }

    /// Check if a function is recursive (calls itself directly or indirectly)
    pub fn is_recursive(&self, name: &str) -> bool {
        self.reaches(name, name, &mut HashSet::new())
    }

    fn reaches(&self, from: &str, target: &str, visited: &mut HashSet<String>) -> bool {
        if visited.contains(from) {
            return false;
        }
        visited.insert(from.to_string());

        if let Some(calls) = self.call_graph.get(from) {
            if calls.contains(target) {
                return true;
            }
            for call in calls {
                if self.reaches(call, target, visited) {
                    return true;
                }
            }
        }
        false
    }

    /// Validate all functions
    pub fn validate(&self) -> Result<(), FunctionError> {
        for (name, func) in &self.functions {
            // Check that all called functions exist
            if let Some(calls) = self.call_graph.get(name) {
                for call in calls {
                    if !self.functions.contains_key(call) && !is_builtin(call) {
                        return Err(FunctionError::UndefinedFunction { name: call.clone() });
                    }
                }
            }

            // Check recursive functions have base case
            if self.is_recursive(name) && !Self::has_base_case(&func.body) {
                return Err(FunctionError::RecursionWithoutBaseCase { name: name.clone() });
            }
        }
        Ok(())
    }

    fn has_base_case(body: &FuncBody) -> bool {
        matches!(body, FuncBody::Conditional(_))
    }

    /// Build registry from a program
    pub fn from_program(program: &Program) -> Result<Self, FunctionError> {
        let mut registry = Self::new();

        // Check for name conflicts with predicates
        let pred_names: HashSet<_> = program.predicates.iter().map(|p| p.name.clone()).collect();

        for func in &program.functions {
            if pred_names.contains(&func.name) {
                return Err(FunctionError::NameConflict {
                    name: func.name.clone(),
                });
            }
            registry.register(func.clone())?;
        }

        registry.validate()?;
        Ok(registry)
    }

    /// Get all registered functions
    pub fn functions(&self) -> impl Iterator<Item = &FuncDef> {
        self.functions.values()
    }

    /// Analyze recursive function for potential infinite recursion
    pub fn analyze_recursion(&self, func: &FuncDef) -> Option<RecursionWarning> {
        if !self.is_recursive(&func.name) {
            return None;
        }

        match &func.body {
            FuncBody::Conditional(cond) => self.check_convergence(func, cond),
            _ => None,
        }
    }

    fn check_convergence(&self, func: &FuncDef, cond: &CondExpr) -> Option<RecursionWarning> {
        // Find recursive calls in else branch
        let recursive_calls = Self::find_recursive_calls_in_body(&func.name, &cond.else_branch);

        for call_args in recursive_calls {
            if call_args.is_empty() {
                continue;
            }

            // Simple pattern check: if condition is var <= k and recursive uses var + n
            // This is a warning sign (moving away from base case)
            if let (ArithExpr::Variable(var), CompOp::Le | CompOp::Lt) =
                (&cond.cond_left, cond.cond_op)
            {
                if let ArithExpr::Add(left, right) = &call_args[0] {
                    if let (ArithExpr::Variable(arg_var), ArithExpr::Integer(n)) =
                        (left.as_ref(), right.as_ref())
                    {
                        if arg_var == var && *n > 0 {
                            return Some(RecursionWarning {
                                func_name: func.name.clone(),
                                message: format!(
                                    "recursive call increases `{}`, but base case requires it to decrease",
                                    var
                                ),
                            });
                        }
                    }
                }
            }
        }

        None
    }

    fn find_recursive_calls_in_body(name: &str, body: &FuncBody) -> Vec<Vec<ArithExpr>> {
        let mut calls = Vec::new();
        match body {
            FuncBody::Arithmetic(expr) => {
                Self::find_recursive_calls_in_expr(name, expr, &mut calls);
            }
            FuncBody::Conditional(cond) => {
                Self::find_recursive_calls_in_expr(name, &cond.cond_left, &mut calls);
                Self::find_recursive_calls_in_expr(name, &cond.cond_right, &mut calls);
                calls.extend(Self::find_recursive_calls_in_body(name, &cond.then_branch));
                calls.extend(Self::find_recursive_calls_in_body(name, &cond.else_branch));
            }
            FuncBody::Predicate { .. } => {}
        }
        calls
    }

    fn find_recursive_calls_in_expr(name: &str, expr: &ArithExpr, calls: &mut Vec<Vec<ArithExpr>>) {
        match expr {
            ArithExpr::FuncCall {
                name: fn_name,
                args,
            } if fn_name == name => {
                calls.push(args.clone());
            }
            ArithExpr::Add(l, r)
            | ArithExpr::Sub(l, r)
            | ArithExpr::Mul(l, r)
            | ArithExpr::Div(l, r)
            | ArithExpr::Mod(l, r)
            | ArithExpr::Min(l, r)
            | ArithExpr::Max(l, r)
            | ArithExpr::Pow(l, r) => {
                Self::find_recursive_calls_in_expr(name, l, calls);
                Self::find_recursive_calls_in_expr(name, r, calls);
            }
            ArithExpr::Abs(e) | ArithExpr::Cast(e, _) => {
                Self::find_recursive_calls_in_expr(name, e, calls);
            }
            ArithExpr::FuncCall { args, .. } => {
                for arg in args {
                    Self::find_recursive_calls_in_expr(name, arg, calls);
                }
            }
            ArithExpr::Conditional {
                cond_left,
                cond_right,
                then_expr,
                else_expr,
                ..
            } => {
                Self::find_recursive_calls_in_expr(name, cond_left, calls);
                Self::find_recursive_calls_in_expr(name, cond_right, calls);
                Self::find_recursive_calls_in_expr(name, then_expr, calls);
                Self::find_recursive_calls_in_expr(name, else_expr, calls);
            }
            _ => {}
        }
    }

    /// Validate all functions, collecting warnings
    pub fn validate_with_warnings(&self) -> (Result<(), FunctionError>, Vec<RecursionWarning>) {
        let mut warnings = Vec::new();

        for func in self.functions.values() {
            if let Some(warning) = self.analyze_recursion(func) {
                warnings.push(warning);
            }
        }

        (self.validate(), warnings)
    }
}

/// Check if a name is a built-in function
fn is_builtin(name: &str) -> bool {
    matches!(name, "abs" | "min" | "max" | "pow" | "cast")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::FuncParam;
    use xlog_core::XlogError;

    #[test]
    fn test_function_error_into_xlog() {
        let err = FunctionError::UndefinedFunction {
            name: "foo".to_string(),
        };
        let xlog_err: XlogError = err.into();
        let msg = xlog_err.to_string();
        assert!(msg.contains("foo"), "Expected 'foo' in: {msg}");
    }

    #[test]
    fn test_type_error_into_xlog() {
        let err = TypeError::CannotInfer {
            name: "X".to_string(),
        };
        let xlog_err: XlogError = err.into();
        let msg = xlog_err.to_string();
        assert!(msg.contains("X"), "Expected 'X' in: {msg}");
    }

    fn make_arith_func(name: &str, body: ArithExpr) -> FuncDef {
        FuncDef {
            name: name.to_string(),
            params: vec![FuncParam {
                name: "X".to_string(),
                typ: None,
            }],
            return_type: None,
            body: FuncBody::Arithmetic(body),
            is_private: false,
        }
    }

    #[test]
    fn test_register_function() {
        let mut reg = FunctionRegistry::new();
        let func = make_arith_func("square", ArithExpr::Variable("X".to_string()));
        assert!(reg.register(func).is_ok());
    }

    #[test]
    fn test_duplicate_error() {
        let mut reg = FunctionRegistry::new();
        let func = make_arith_func("f", ArithExpr::Variable("X".to_string()));
        reg.register(func.clone()).unwrap();
        let result = reg.register(func);
        assert!(matches!(
            result,
            Err(FunctionError::DuplicateDefinition { .. })
        ));
    }

    #[test]
    fn test_recursive_detection() {
        let mut reg = FunctionRegistry::new();

        // f calls itself
        let f = FuncDef {
            name: "f".to_string(),
            params: vec![],
            return_type: None,
            body: FuncBody::Arithmetic(ArithExpr::FuncCall {
                name: "f".to_string(),
                args: vec![],
            }),
            is_private: false,
        };
        reg.register(f).unwrap();

        assert!(reg.is_recursive("f"));
    }

    #[test]
    fn test_get_function() {
        let mut reg = FunctionRegistry::new();
        let func = make_arith_func("square", ArithExpr::Variable("X".to_string()));
        reg.register(func).unwrap();

        assert!(reg.get("square").is_some());
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn test_contains_function() {
        let mut reg = FunctionRegistry::new();
        let func = make_arith_func("square", ArithExpr::Variable("X".to_string()));
        reg.register(func).unwrap();

        assert!(reg.contains("square"));
        assert!(!reg.contains("nonexistent"));
    }

    #[test]
    fn test_undefined_function_error() {
        let mut reg = FunctionRegistry::new();

        // Function that calls an undefined function
        let f = FuncDef {
            name: "f".to_string(),
            params: vec![],
            return_type: None,
            body: FuncBody::Arithmetic(ArithExpr::FuncCall {
                name: "undefined_func".to_string(),
                args: vec![],
            }),
            is_private: false,
        };
        reg.register(f).unwrap();

        let result = reg.validate();
        assert!(matches!(
            result,
            Err(FunctionError::UndefinedFunction { .. })
        ));
    }

    #[test]
    fn test_builtin_function_allowed() {
        let mut reg = FunctionRegistry::new();

        // Function that calls built-in functions
        let f = FuncDef {
            name: "f".to_string(),
            params: vec![FuncParam {
                name: "X".to_string(),
                typ: None,
            }],
            return_type: None,
            body: FuncBody::Arithmetic(ArithExpr::FuncCall {
                name: "abs".to_string(),
                args: vec![ArithExpr::Variable("X".to_string())],
            }),
            is_private: false,
        };
        reg.register(f).unwrap();

        // Should not error because abs is a built-in
        assert!(reg.validate().is_ok());
    }

    #[test]
    fn test_indirect_recursion() {
        let mut reg = FunctionRegistry::new();

        // f calls g
        let f = FuncDef {
            name: "f".to_string(),
            params: vec![],
            return_type: None,
            body: FuncBody::Arithmetic(ArithExpr::FuncCall {
                name: "g".to_string(),
                args: vec![],
            }),
            is_private: false,
        };

        // g calls f (indirect recursion)
        let g = FuncDef {
            name: "g".to_string(),
            params: vec![],
            return_type: None,
            body: FuncBody::Arithmetic(ArithExpr::FuncCall {
                name: "f".to_string(),
                args: vec![],
            }),
            is_private: false,
        };

        reg.register(f).unwrap();
        reg.register(g).unwrap();

        assert!(reg.is_recursive("f"));
        assert!(reg.is_recursive("g"));
    }

    #[test]
    fn test_functions_iterator() {
        let mut reg = FunctionRegistry::new();
        let f1 = make_arith_func("f1", ArithExpr::Variable("X".to_string()));
        let f2 = make_arith_func("f2", ArithExpr::Variable("X".to_string()));
        reg.register(f1).unwrap();
        reg.register(f2).unwrap();

        let names: HashSet<_> = reg.functions().map(|f| f.name.as_str()).collect();
        assert!(names.contains("f1"));
        assert!(names.contains("f2"));
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_type_error_display() {
        let err = TypeError::Mismatch {
            expected: ScalarType::I64,
            found: ScalarType::F64,
            location: "function f".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("E0506"));
        assert!(msg.contains("type mismatch"));

        let err2 = TypeError::CannotInfer {
            name: "X".to_string(),
        };
        let msg2 = err2.to_string();
        assert!(msg2.contains("E0507"));
        assert!(msg2.contains("cannot infer"));
    }

    #[test]
    fn test_recursion_warning_display() {
        let warning = RecursionWarning {
            func_name: "fib".to_string(),
            message: "recursive call increases `N`".to_string(),
        };
        let msg = warning.to_string();
        assert!(msg.contains("W0502"));
        assert!(msg.contains("infinite recursion"));
        assert!(msg.contains("fib"));
    }

    #[test]
    fn test_analyze_non_recursive() {
        let mut reg = FunctionRegistry::new();
        let func = make_arith_func("square", ArithExpr::Variable("X".to_string()));
        reg.register(func.clone()).unwrap();

        // Non-recursive functions shouldn't trigger warnings
        assert!(reg.analyze_recursion(&func).is_none());
    }

    #[test]
    fn test_analyze_recursive_with_proper_convergence() {
        use crate::ast::CondExpr;

        let mut reg = FunctionRegistry::new();

        // Proper factorial: if N <= 1 then 1 else N * fact(N - 1)
        let factorial = FuncDef {
            name: "fact".to_string(),
            params: vec![FuncParam {
                name: "N".to_string(),
                typ: None,
            }],
            return_type: None,
            body: FuncBody::Conditional(CondExpr {
                cond_left: ArithExpr::Variable("N".to_string()),
                cond_op: CompOp::Le,
                cond_right: ArithExpr::Integer(1),
                then_branch: Box::new(FuncBody::Arithmetic(ArithExpr::Integer(1))),
                else_branch: Box::new(FuncBody::Arithmetic(ArithExpr::Mul(
                    Box::new(ArithExpr::Variable("N".to_string())),
                    Box::new(ArithExpr::FuncCall {
                        name: "fact".to_string(),
                        args: vec![ArithExpr::Sub(
                            Box::new(ArithExpr::Variable("N".to_string())),
                            Box::new(ArithExpr::Integer(1)),
                        )],
                    }),
                ))),
            }),
            is_private: false,
        };

        reg.register(factorial.clone()).unwrap();

        // Proper convergence (N - 1) shouldn't trigger warning
        assert!(reg.analyze_recursion(&factorial).is_none());
    }

    #[test]
    fn test_analyze_recursive_with_divergence() {
        use crate::ast::CondExpr;

        let mut reg = FunctionRegistry::new();

        // Bad function: if N <= 1 then 1 else f(N + 1)
        // This increases N, which diverges from the base case
        let bad_func = FuncDef {
            name: "badfunc".to_string(),
            params: vec![FuncParam {
                name: "N".to_string(),
                typ: None,
            }],
            return_type: None,
            body: FuncBody::Conditional(CondExpr {
                cond_left: ArithExpr::Variable("N".to_string()),
                cond_op: CompOp::Le,
                cond_right: ArithExpr::Integer(1),
                then_branch: Box::new(FuncBody::Arithmetic(ArithExpr::Integer(1))),
                else_branch: Box::new(FuncBody::Arithmetic(ArithExpr::FuncCall {
                    name: "badfunc".to_string(),
                    args: vec![ArithExpr::Add(
                        Box::new(ArithExpr::Variable("N".to_string())),
                        Box::new(ArithExpr::Integer(1)),
                    )],
                })),
            }),
            is_private: false,
        };

        reg.register(bad_func.clone()).unwrap();

        // Should trigger a warning about potential infinite recursion
        let warning = reg.analyze_recursion(&bad_func);
        assert!(warning.is_some());
        assert!(warning.unwrap().message.contains("increases"));
    }

    #[test]
    fn test_validate_with_warnings() {
        use crate::ast::CondExpr;

        let mut reg = FunctionRegistry::new();

        // Bad function that will generate a warning
        let bad_func = FuncDef {
            name: "diverging".to_string(),
            params: vec![FuncParam {
                name: "X".to_string(),
                typ: None,
            }],
            return_type: None,
            body: FuncBody::Conditional(CondExpr {
                cond_left: ArithExpr::Variable("X".to_string()),
                cond_op: CompOp::Lt,
                cond_right: ArithExpr::Integer(0),
                then_branch: Box::new(FuncBody::Arithmetic(ArithExpr::Integer(0))),
                else_branch: Box::new(FuncBody::Arithmetic(ArithExpr::FuncCall {
                    name: "diverging".to_string(),
                    args: vec![ArithExpr::Add(
                        Box::new(ArithExpr::Variable("X".to_string())),
                        Box::new(ArithExpr::Integer(1)),
                    )],
                })),
            }),
            is_private: false,
        };

        reg.register(bad_func).unwrap();

        let (result, warnings) = reg.validate_with_warnings();
        assert!(result.is_ok());
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].func_name == "diverging");
    }
}
