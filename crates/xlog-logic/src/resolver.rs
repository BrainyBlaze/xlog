//! Module resolution for XLOG programs.

use crate::ast::Program;
use crate::module::{LoadedModule, ModuleError, ModulePath, module_path_to_string};
use crate::parser::parse_program;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Resolves and loads modules
pub struct ModuleResolver {
    /// Directories to search for modules
    search_paths: Vec<PathBuf>,
    /// Already loaded modules (path string -> module)
    loaded: HashMap<String, LoadedModule>,
    /// Currently loading (for cycle detection)
    loading: Vec<ModulePath>,
}

impl ModuleResolver {
    /// Create a new resolver with given search paths
    pub fn new(search_paths: Vec<PathBuf>) -> Self {
        Self {
            search_paths,
            loaded: HashMap::new(),
            loading: Vec::new(),
        }
    }

    /// Find the file for a module path
    pub fn find_module_file(&self, base_dir: &Path, module_path: &[String]) -> Option<PathBuf> {
        let relative_path = format!("{}.xlog", module_path.join("/"));

        // Try relative to base_dir first
        let candidate = base_dir.join(&relative_path);
        if candidate.exists() {
            return Some(candidate);
        }

        // Try search paths
        for search_path in &self.search_paths {
            let candidate = search_path.join(&relative_path);
            if candidate.exists() {
                return Some(candidate);
            }
        }

        None
    }

    /// Get the list of searched paths for error reporting
    fn searched_paths(&self, base_dir: &Path, module_path: &[String]) -> Vec<PathBuf> {
        let relative_path = format!("{}.xlog", module_path.join("/"));
        let mut searched = vec![base_dir.join(&relative_path)];
        for sp in &self.search_paths {
            searched.push(sp.join(&relative_path));
        }
        searched
    }

    /// Check if we're in a circular import
    fn check_cycle(&self, module_path: &[String]) -> Option<Vec<ModulePath>> {
        let path_str = module_path_to_string(module_path);
        for (i, loading_path) in self.loading.iter().enumerate() {
            if module_path_to_string(loading_path) == path_str {
                // Found cycle - return the cycle path
                let mut cycle: Vec<ModulePath> = self.loading[i..].to_vec();
                cycle.push(module_path.to_vec());
                return Some(cycle);
            }
        }
        None
    }

    /// Extract exports from a parsed program
    pub fn extract_exports(program: &Program) -> HashSet<String> {
        let mut exports = HashSet::new();

        // Add declared predicates that aren't private
        for pred in &program.predicates {
            if !pred.is_private {
                exports.insert(pred.name.clone());
            }
        }

        // Add rule heads (all rules define public predicates unless declared private)
        for rule in &program.rules {
            // Check if this predicate was declared as private
            let is_private = program.predicates
                .iter()
                .any(|p| p.name == rule.head.predicate && p.is_private);
            if !is_private {
                exports.insert(rule.head.predicate.clone());
            }
        }

        exports
    }

    /// Load a module from a path
    pub fn load_module(
        &mut self,
        base_dir: &Path,
        module_path: &[String],
    ) -> Result<&LoadedModule, ModuleError> {
        let path_key = module_path_to_string(module_path);

        // Already loaded?
        if self.loaded.contains_key(&path_key) {
            return Ok(self.loaded.get(&path_key).unwrap());
        }

        // Check for cycle
        if let Some(cycle) = self.check_cycle(module_path) {
            return Err(ModuleError::CircularImport { cycle });
        }

        // Find the file
        let source_file = self.find_module_file(base_dir, module_path)
            .ok_or_else(|| ModuleError::NotFound {
                path: module_path.to_vec(),
                searched: self.searched_paths(base_dir, module_path),
            })?;

        // Mark as loading
        self.loading.push(module_path.to_vec());

        // Read and parse
        let source = fs::read_to_string(&source_file)
            .map_err(|e| ModuleError::ParseError {
                path: source_file.clone(),
                message: e.to_string(),
            })?;

        let program = parse_program(&source)
            .map_err(|e| ModuleError::ParseError {
                path: source_file.clone(),
                message: e.to_string(),
            })?;

        // Extract exports
        let exports = Self::extract_exports(&program);

        // Recursively load imports
        let module_dir = source_file.parent().unwrap_or(base_dir);
        for import in &program.imports {
            self.load_module(module_dir, &import.module_path)?;
        }

        // Remove from loading
        self.loading.pop();

        // Store loaded module
        let module = LoadedModule {
            path: module_path.to_vec(),
            source_file,
            exports,
            function_exports: HashSet::new(),
        };

        self.loaded.insert(path_key.clone(), module);
        Ok(self.loaded.get(&path_key).unwrap())
    }

    /// Check if a predicate can be imported from a module
    pub fn check_import(
        &self,
        module_path: &[String],
        predicate: &str,
    ) -> Result<(), ModuleError> {
        let path_key = module_path_to_string(module_path);
        let module = self.loaded.get(&path_key)
            .ok_or_else(|| ModuleError::NotFound {
                path: module_path.to_vec(),
                searched: vec![],
            })?;

        if !module.exports.contains(predicate) {
            return Err(ModuleError::PredicateNotFound {
                name: predicate.to_string(),
                module: module_path.to_vec(),
            });
        }

        Ok(())
    }

    /// Validate all imports in a program
    pub fn validate_imports(
        &self,
        program: &Program,
    ) -> Result<HashMap<String, ModulePath>, ModuleError> {
        let mut imported_names: HashMap<String, ModulePath> = HashMap::new();

        for use_decl in &program.imports {
            let module = self.loaded.get(&module_path_to_string(&use_decl.module_path))
                .expect("module should be loaded");

            let names_to_import: Vec<String> = match &use_decl.imports {
                Some(specific) => specific.clone(),
                None => module.exports.iter().cloned().collect(),
            };

            for name in names_to_import {
                // Check if predicate exists and is public
                if !module.exports.contains(&name) {
                    return Err(ModuleError::PredicateNotFound {
                        name: name.clone(),
                        module: use_decl.module_path.clone(),
                    });
                }

                // Check for conflicts
                if let Some(prev_module) = imported_names.get(&name) {
                    if prev_module != &use_decl.module_path {
                        return Err(ModuleError::ImportConflict {
                            name,
                            module1: prev_module.clone(),
                            module2: use_decl.module_path.clone(),
                        });
                    }
                }

                imported_names.insert(name, use_decl.module_path.clone());
            }
        }

        Ok(imported_names)
    }

    /// Get a loaded module by path
    pub fn get_module(&self, module_path: &[String]) -> Option<&LoadedModule> {
        self.loaded.get(&module_path_to_string(module_path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_module(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(format!("{}.xlog", name));
        let mut file = fs::File::create(&path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn test_find_module_file() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "graph", "edge(1, 2).");

        let resolver = ModuleResolver::new(vec![]);
        let found = resolver.find_module_file(tmp.path(), &["graph".into()]);
        assert!(found.is_some());
    }

    #[test]
    fn test_module_not_found() {
        let tmp = TempDir::new().unwrap();
        let mut resolver = ModuleResolver::new(vec![]);

        let result = resolver.load_module(tmp.path(), &["nonexistent".into()]);
        assert!(matches!(result, Err(ModuleError::NotFound { .. })));
    }

    #[test]
    fn test_circular_import() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "a", "use b.");
        create_test_module(tmp.path(), "b", "use a.");

        let mut resolver = ModuleResolver::new(vec![]);
        let result = resolver.load_module(tmp.path(), &["a".into()]);
        assert!(matches!(result, Err(ModuleError::CircularImport { .. })));
    }

    #[test]
    fn test_load_simple_module() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "math", r#"
            pred add(u32, u32, u32).
            add(1, 2, 3).
        "#);

        let mut resolver = ModuleResolver::new(vec![]);
        let result = resolver.load_module(tmp.path(), &["math".into()]);
        assert!(result.is_ok());
        let module = result.unwrap();
        assert!(module.exports.contains("add"));
    }

    #[test]
    fn test_private_not_exported() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "graph", r#"
            pred edge(u32, u32).
            private pred helper(u32).
            edge(1, 2).
            helper(1).
        "#);

        let mut resolver = ModuleResolver::new(vec![]);
        let result = resolver.load_module(tmp.path(), &["graph".into()]);
        assert!(result.is_ok());
        let module = result.unwrap();
        assert!(module.exports.contains("edge"));
        assert!(!module.exports.contains("helper"));
    }

    #[test]
    fn test_search_paths() {
        let tmp = TempDir::new().unwrap();
        let lib_dir = tmp.path().join("lib");
        fs::create_dir(&lib_dir).unwrap();
        create_test_module(&lib_dir, "stdlib", "helper(1).");

        let resolver = ModuleResolver::new(vec![lib_dir.clone()]);
        let found = resolver.find_module_file(tmp.path(), &["stdlib".into()]);
        assert!(found.is_some());
        assert!(found.unwrap().starts_with(&lib_dir));
    }
}
