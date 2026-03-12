//! Module system types for XLOG.

use std::collections::HashSet;
use std::path::PathBuf;

use crate::ast::Program;

/// A module path like ["utils", "math"]
pub(crate) type ModulePath = Vec<String>;

/// Convert module path to string for display
pub(crate) fn module_path_to_string(path: &[String]) -> String {
    path.join("/")
}

/// A loaded module with metadata
#[derive(Debug)]
pub struct LoadedModule {
    /// Module path
    pub path: ModulePath,
    /// Source file location
    pub source_file: PathBuf,
    /// Public predicate names
    pub exports: HashSet<String>,
    /// Public function names
    pub function_exports: HashSet<String>,
    /// The parsed program content
    pub program: Program,
}

impl LoadedModule {
    pub fn new(path: ModulePath, source_file: PathBuf, program: Program) -> Self {
        Self {
            path,
            source_file,
            exports: HashSet::new(),
            function_exports: HashSet::new(),
            program,
        }
    }
}

/// Errors that can occur during module resolution
#[derive(Debug, Clone)]
pub enum ModuleError {
    /// Module file not found
    NotFound {
        path: ModulePath,
        searched: Vec<PathBuf>,
    },
    /// Circular import detected
    CircularImport { cycle: Vec<ModulePath> },
    /// Name conflict between imports
    ImportConflict {
        name: String,
        module1: ModulePath,
        module2: ModulePath,
    },
    /// Attempted to import private predicate
    PrivatePredicate { name: String, module: ModulePath },
    /// Predicate not found in module
    PredicateNotFound { name: String, module: ModulePath },
    /// Parse error in module
    ParseError { path: PathBuf, message: String },
}

impl std::fmt::Display for ModuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModuleError::NotFound { path, searched } => {
                writeln!(
                    f,
                    "error[E0400]: module not found: `{}`",
                    module_path_to_string(path)
                )?;
                writeln!(f, "  = note: searched in:")?;
                for s in searched {
                    writeln!(f, "          - {}", s.display())?;
                }
                write!(
                    f,
                    "  = help: check the module path spelling or add to --module-path"
                )
            }
            ModuleError::CircularImport { cycle } => {
                writeln!(f, "error[E0401]: circular import detected")?;
                for (i, path) in cycle.iter().enumerate() {
                    if i < cycle.len() - 1 {
                        writeln!(
                            f,
                            "  {} imports {}",
                            module_path_to_string(path),
                            module_path_to_string(&cycle[i + 1])
                        )?;
                    }
                }
                write!(f, "  = help: extract shared predicates into a third module")
            }
            ModuleError::ImportConflict {
                name,
                module1,
                module2,
            } => {
                writeln!(f, "error[E0402]: ambiguous import `{}`", name)?;
                writeln!(
                    f,
                    "  `{}` first imported from {}",
                    name,
                    module_path_to_string(module1)
                )?;
                writeln!(
                    f,
                    "  `{}` also exported by {}",
                    name,
                    module_path_to_string(module2)
                )?;
                write!(
                    f,
                    "  = help: use selective imports: `use {}::{{...}}.`",
                    module_path_to_string(module1)
                )
            }
            ModuleError::PrivatePredicate { name, module } => {
                write!(
                    f,
                    "error[E0403]: cannot import private predicate `{}` from {}",
                    name,
                    module_path_to_string(module)
                )
            }
            ModuleError::PredicateNotFound { name, module } => {
                write!(
                    f,
                    "error[E0404]: predicate `{}` not found in module {}",
                    name,
                    module_path_to_string(module)
                )
            }
            ModuleError::ParseError { path, message } => {
                write!(f, "error: parse error in {:?}: {}", path, message)
            }
        }
    }
}

impl std::error::Error for ModuleError {}

impl From<ModuleError> for xlog_core::XlogError {
    fn from(e: ModuleError) -> Self {
        xlog_core::XlogError::Compilation(e.to_string())
    }
}

/// Generate internal qualified name for a predicate
/// E.g., (["utils", "math"], "abs") -> "__utils_math__abs"
#[allow(dead_code)] // reserved API: module system not yet wired
pub(crate) fn internal_name(module_path: &[String], predicate: &str) -> String {
    if module_path.is_empty() {
        predicate.to_string()
    } else {
        format!("__{}__{}", module_path.join("_"), predicate)
    }
}

/// Extract module and predicate from internal name
/// E.g., "__utils_math__abs" -> (["utils", "math"], "abs")
#[allow(dead_code)] // reserved API: module system not yet wired
pub(crate) fn parse_internal_name(internal: &str) -> (Vec<String>, String) {
    if internal.starts_with("__") {
        if let Some(pos) = internal.rfind("__") {
            if pos > 2 {
                let module_part = &internal[2..pos];
                let pred_part = &internal[pos + 2..];
                let modules: Vec<String> = module_part.split('_').map(String::from).collect();
                return (modules, pred_part.to_string());
            }
        }
    }
    (vec![], internal.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_path_to_string() {
        assert_eq!(
            module_path_to_string(&["utils".into(), "math".into()]),
            "utils/math"
        );
        assert_eq!(module_path_to_string(&["single".into()]), "single");
    }

    #[test]
    fn test_loaded_module_new() {
        let module = LoadedModule::new(
            vec!["test".to_string()],
            PathBuf::from("/test.xlog"),
            Program::default(),
        );
        assert_eq!(module.path, vec!["test"]);
        assert!(module.exports.is_empty());
    }

    #[test]
    fn test_module_error_display() {
        let err = ModuleError::NotFound {
            path: vec!["missing".to_string()],
            searched: vec![PathBuf::from("/a/missing.xlog")],
        };
        let msg = err.to_string();
        assert!(msg.contains("module not found"));
        assert!(msg.contains("missing"));
    }

    #[test]
    fn test_internal_name() {
        assert_eq!(internal_name(&[], "foo"), "foo");
        assert_eq!(
            internal_name(&["utils".into(), "math".into()], "abs"),
            "__utils_math__abs"
        );
        assert_eq!(internal_name(&["single".into()], "pred"), "__single__pred");
    }

    #[test]
    fn test_parse_internal_name() {
        assert_eq!(parse_internal_name("foo"), (vec![], "foo".to_string()));
        assert_eq!(
            parse_internal_name("__utils_math__abs"),
            (
                vec!["utils".to_string(), "math".to_string()],
                "abs".to_string()
            )
        );
        assert_eq!(
            parse_internal_name("__single__pred"),
            (vec!["single".to_string()], "pred".to_string())
        );
    }

    #[test]
    fn test_module_error_into_xlog() {
        let err = ModuleError::ParseError {
            path: std::path::PathBuf::from("/test.xlog"),
            message: "unexpected EOF".to_string(),
        };
        let xlog_err: xlog_core::XlogError = err.into();
        let msg = xlog_err.to_string();
        assert!(
            msg.contains("unexpected EOF"),
            "Expected 'unexpected EOF' in: {msg}"
        );
    }
}
