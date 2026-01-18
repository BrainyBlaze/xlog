//! Module system types for XLOG.

use std::collections::HashSet;
use std::path::PathBuf;

/// A module path like ["utils", "math"]
pub type ModulePath = Vec<String>;

/// Convert module path to string for display
pub fn module_path_to_string(path: &[String]) -> String {
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
}

impl LoadedModule {
    pub fn new(path: ModulePath, source_file: PathBuf) -> Self {
        Self {
            path,
            source_file,
            exports: HashSet::new(),
            function_exports: HashSet::new(),
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
    CircularImport {
        cycle: Vec<ModulePath>,
    },
    /// Name conflict between imports
    ImportConflict {
        name: String,
        module1: ModulePath,
        module2: ModulePath,
    },
    /// Attempted to import private predicate
    PrivatePredicate {
        name: String,
        module: ModulePath,
    },
    /// Predicate not found in module
    PredicateNotFound {
        name: String,
        module: ModulePath,
    },
    /// Parse error in module
    ParseError {
        path: PathBuf,
        message: String,
    },
}

impl std::fmt::Display for ModuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModuleError::NotFound { path, searched } => {
                writeln!(f, "error[E0400]: module not found: `{}`", module_path_to_string(path))?;
                writeln!(f, "  = note: searched in:")?;
                for s in searched {
                    writeln!(f, "          - {}", s.display())?;
                }
                write!(f, "  = help: check the module path spelling or add to --module-path")
            }
            ModuleError::CircularImport { cycle } => {
                writeln!(f, "error[E0401]: circular import detected")?;
                for (i, path) in cycle.iter().enumerate() {
                    if i < cycle.len() - 1 {
                        writeln!(f, "  {} imports {}",
                            module_path_to_string(path),
                            module_path_to_string(&cycle[i + 1]))?;
                    }
                }
                write!(f, "  = help: extract shared predicates into a third module")
            }
            ModuleError::ImportConflict { name, module1, module2 } => {
                writeln!(f, "error[E0402]: ambiguous import `{}`", name)?;
                writeln!(f, "  `{}` first imported from {}", name, module_path_to_string(module1))?;
                writeln!(f, "  `{}` also exported by {}", name, module_path_to_string(module2))?;
                write!(f, "  = help: use selective imports: `use {}::{{...}}.`", module_path_to_string(module1))
            }
            ModuleError::PrivatePredicate { name, module } => {
                write!(f, "error[E0403]: cannot import private predicate `{}` from {}",
                    name, module_path_to_string(module))
            }
            ModuleError::PredicateNotFound { name, module } => {
                write!(f, "error[E0404]: predicate `{}` not found in module {}",
                    name, module_path_to_string(module))
            }
            ModuleError::ParseError { path, message } => {
                write!(f, "error: parse error in {:?}: {}", path, message)
            }
        }
    }
}

impl std::error::Error for ModuleError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_path_to_string() {
        assert_eq!(module_path_to_string(&["utils".into(), "math".into()]), "utils/math");
        assert_eq!(module_path_to_string(&["single".into()]), "single");
    }

    #[test]
    fn test_loaded_module_new() {
        let module = LoadedModule::new(
            vec!["test".to_string()],
            PathBuf::from("/test.xlog"),
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
}
