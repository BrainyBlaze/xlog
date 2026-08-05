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
    /// Logical path recorded when this canonical source was first loaded.
    ///
    /// A later alias returned by `ModuleResolver::get_module` can differ from
    /// this representative path.
    pub path: ModulePath,
    /// Filesystem spelling used when this canonical source was first loaded.
    ///
    /// This path is not guaranteed to be canonical when the module was reached
    /// through a relative path or symbolic link.
    pub source_file: PathBuf,
    /// Public predicate names
    pub exports: HashSet<String>,
    /// Public function names
    pub function_exports: HashSet<String>,
    /// The parsed program content
    pub program: Program,
}

impl LoadedModule {
    /// Create a new loaded module (exports initially empty).
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
#[non_exhaustive]
pub enum ModuleError {
    /// Module file not found
    NotFound {
        /// Logical module path that failed to resolve.
        path: ModulePath,
        /// Filesystem locations that were searched.
        searched: Vec<PathBuf>,
    },
    /// Circular import detected
    CircularImport {
        /// Ordered import cycle that was discovered.
        cycle: Vec<ModulePath>,
    },
    /// Conflicting definitions for an imported function.
    ImportConflict {
        /// Function name that has multiple definitions.
        name: String,
        /// Module containing the first definition.
        module1: ModulePath,
        /// Module containing the conflicting definition.
        module2: ModulePath,
    },
    /// Attempted to import private predicate
    PrivatePredicate {
        /// Predicate name that is not exported.
        name: String,
        /// Module that owns the private predicate.
        module: ModulePath,
    },
    /// Selected item not exported by a module
    PredicateNotFound {
        /// Predicate or function name that is not exported.
        name: String,
        /// Module that was expected to export the item.
        module: ModulePath,
    },
    /// An imported module contains program-level constructs that are entry-only.
    UnsupportedImportedContent {
        /// Module containing the unsupported constructs.
        module: ModulePath,
        /// Deterministically ordered construct categories.
        constructs: Vec<String>,
    },
    /// An exported item depends on module-local support that the import filters out.
    HiddenDependency {
        /// Module containing the exported item.
        module: ModulePath,
        /// Exported predicate or function whose implementation is incomplete.
        export: String,
        /// Private or selectively omitted dependency.
        dependency: String,
    },
    /// A context-free API request used a logical path that names several loaded files.
    AmbiguousModulePath {
        /// Logical module path whose source cannot be inferred.
        path: ModulePath,
        /// Canonical source files registered for the logical path.
        candidates: Vec<PathBuf>,
    },
    /// The entry program or imported modules declare one predicate with
    /// incompatible schemas.
    IncompatiblePredicateDeclaration {
        /// Predicate whose declarations differ.
        name: String,
        /// Module containing the first declaration.
        module1: ModulePath,
        /// Module containing the incompatible declaration.
        module2: ModulePath,
    },
    /// The entry program or imported modules define one domain alias with
    /// incompatible scalar types.
    IncompatibleDomainDeclaration {
        /// Domain alias whose declarations differ.
        name: String,
        /// Module containing the first declaration.
        module1: ModulePath,
        /// Module containing the incompatible declaration.
        module2: ModulePath,
    },
    /// An imported module defines one exported function name more than once.
    DuplicateImportedFunction {
        /// Duplicate function name.
        name: String,
        /// Module containing the duplicate definitions.
        module: ModulePath,
    },
    /// An imported module declares one predicate as both public and private.
    ConflictingPredicateVisibility {
        /// Predicate with contradictory visibility declarations.
        name: String,
        /// Module containing the declarations.
        module: ModulePath,
    },
    /// Parse error in module
    ParseError {
        /// Source file path that failed to parse.
        path: PathBuf,
        /// Human-readable parse failure message.
        message: String,
    },
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
                writeln!(
                    f,
                    "error[E0402]: conflicting definitions for imported function `{name}`"
                )?;
                writeln!(
                    f,
                    "  function `{}` is defined by module `{}`",
                    name,
                    module_path_to_string(module1)
                )?;
                writeln!(
                    f,
                    "  function `{}` is also defined by module `{}`",
                    name,
                    module_path_to_string(module2)
                )?;
                write!(
                    f,
                    "  = help: import only one definition of function `{name}` with selective `use` declarations"
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
                    "error[E0404]: item `{}` is not exported by module {}",
                    name,
                    module_path_to_string(module)
                )
            }
            ModuleError::UnsupportedImportedContent { module, constructs } => {
                writeln!(
                    f,
                    "error[E0405]: imported module `{}` contains unsupported program-level constructs: {}",
                    module_path_to_string(module),
                    constructs.join(", ")
                )?;
                write!(f, "  = help: declare these constructs in the entry file")
            }
            ModuleError::HiddenDependency {
                module,
                export,
                dependency,
            } => {
                writeln!(
                    f,
                    "error[E0406]: exported item `{}` in module `{}` depends on hidden item `{}`",
                    export,
                    module_path_to_string(module),
                    dependency
                )?;
                write!(
                    f,
                    "  = help: imported exports cannot depend on private or selectively omitted module items"
                )
            }
            ModuleError::AmbiguousModulePath { path, candidates } => {
                writeln!(
                    f,
                    "error[E0407]: module path `{}` identifies multiple loaded files",
                    module_path_to_string(path)
                )?;
                writeln!(f, "  = note: loaded candidates:")?;
                for candidate in candidates {
                    writeln!(f, "          - {}", candidate.display())?;
                }
                write!(
                    f,
                    "  = help: load the entry file or root module before validating or merging its imports"
                )
            }
            ModuleError::IncompatiblePredicateDeclaration {
                name,
                module1,
                module2,
            } => {
                writeln!(
                    f,
                    "error[E0408]: incompatible declarations for predicate `{name}`"
                )?;
                writeln!(
                    f,
                    "  `{name}` is declared by {} and {} with different schemas",
                    module_path_to_string(module1),
                    module_path_to_string(module2)
                )?;
                write!(
                    f,
                    "  = help: all declarations in the entry program and resolved import closure must use identical arity, column names, and resolved types"
                )
            }
            ModuleError::IncompatibleDomainDeclaration {
                name,
                module1,
                module2,
            } => {
                writeln!(
                    f,
                    "error[E0409]: incompatible declarations for domain alias `{name}`"
                )?;
                writeln!(
                    f,
                    "  `{name}` is declared by {} and {} with different scalar types",
                    module_path_to_string(module1),
                    module_path_to_string(module2)
                )?;
                write!(
                    f,
                    "  = help: a domain alias must resolve to one scalar type throughout the entry program and resolved import closure"
                )
            }
            ModuleError::DuplicateImportedFunction { name, module } => {
                writeln!(
                    f,
                    "error[E0410]: imported module `{}` defines function `{name}` more than once",
                    module_path_to_string(module)
                )?;
                write!(f, "  = help: keep exactly one definition for each function")
            }
            ModuleError::ConflictingPredicateVisibility { name, module } => {
                writeln!(
                    f,
                    "error[E0411]: imported module `{}` declares predicate `{name}` as both public and private",
                    module_path_to_string(module)
                )?;
                write!(
                    f,
                    "  = help: use one visibility for every declaration of a predicate"
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
    fn ambiguous_module_path_error_lists_candidates() {
        let err = ModuleError::AmbiguousModulePath {
            path: vec!["support".to_string()],
            candidates: vec![
                PathBuf::from("/modules/left/support.xlog"),
                PathBuf::from("/modules/right/support.xlog"),
            ],
        };

        let message = err.to_string();

        assert!(message.contains("error[E0407]"));
        assert!(message.contains("module path `support`"));
        assert!(message.contains("/modules/left/support.xlog"));
        assert!(message.contains("/modules/right/support.xlog"));
    }

    #[test]
    fn incompatible_predicate_declaration_error_names_both_modules() {
        let err = ModuleError::IncompatiblePredicateDeclaration {
            name: "external".to_string(),
            module1: vec!["first".to_string()],
            module2: vec!["second".to_string()],
        };

        let message = err.to_string();

        assert!(message.contains("error[E0408]"));
        assert!(message.contains("predicate `external`"));
        assert!(message.contains("first and second"));
    }

    #[test]
    fn incompatible_domain_declaration_error_names_both_modules() {
        let err = ModuleError::IncompatibleDomainDeclaration {
            name: "key".to_string(),
            module1: vec!["first".to_string()],
            module2: vec!["second".to_string()],
        };

        let message = err.to_string();

        assert!(message.contains("error[E0409]"));
        assert!(message.contains("domain alias `key`"));
        assert!(message.contains("first and second"));
    }

    #[test]
    fn duplicate_imported_function_error_names_module() {
        let err = ModuleError::DuplicateImportedFunction {
            name: "normalize".to_string(),
            module: vec!["library".to_string()],
        };

        let message = err.to_string();

        assert!(message.contains("error[E0410]"));
        assert!(message.contains("function `normalize`"));
        assert!(message.contains("module `library`"));
    }

    #[test]
    fn conflicting_predicate_visibility_error_names_module() {
        let err = ModuleError::ConflictingPredicateVisibility {
            name: "shared".to_string(),
            module: vec!["library".to_string()],
        };

        let message = err.to_string();

        assert!(message.contains("error[E0411]"));
        assert!(message.contains("predicate `shared`"));
        assert!(message.contains("module `library`"));
        assert!(message.contains("both public and private"));
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
    fn imported_function_conflict_error_names_both_definitions() {
        let err = ModuleError::ImportConflict {
            name: "normalize".to_string(),
            module1: vec!["first".to_string()],
            module2: vec!["second".to_string()],
        };

        let message = err.to_string();

        assert!(message
            .contains("error[E0402]: conflicting definitions for imported function `normalize`"));
        assert!(message.contains("function `normalize` is defined by module `first`"));
        assert!(message.contains("function `normalize` is also defined by module `second`"));
        assert!(message.contains("import only one definition of function `normalize`"));
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
