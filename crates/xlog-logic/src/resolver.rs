//! Module resolution for XLOG programs.

mod extraction;
mod manifest;

pub use extraction::{
    AggregateOperator, ComparisonOperator, EpistemicOperator, ExecutableAnnotatedDisjunction,
    ExecutableArithmeticExpression, ExecutableAtom, ExecutableBodyLiteral, ExecutableConstraint,
    ExecutableDomain, ExecutableEvidence, ExecutableFunction, ExecutableFunctionBody,
    ExecutableFunctionParameter, ExecutableLearnableRule, ExecutableNeuralLabel,
    ExecutableNeuralPredicate, ExecutablePredicateColumn, ExecutableProbabilisticFact,
    ExecutableProbabilisticQuery, ExecutableProbability, ExecutableProgram, ExecutableQuery,
    ExecutableRelation, ExecutableRelationDefinition, ExecutableRelationDefinitionKind,
    ExecutableRule, ExecutableScalarType, ExecutableScc, ExecutableTerm, ExecutableTypeReference,
    ExecutableWeightedAtom, RelationDependency, RelationDependencyKind,
    RelationDependencyProducerKind, ResolvedProgramExtraction, ResolvedProgramExtractionError,
};
pub use manifest::{
    ResolvedConstructCount, ResolvedImportManifest, ResolvedModuleManifest,
    ResolvedProgramManifest, ResolvedProgramManifestError, ResolvedSourceObject,
    ResolvedSourceObjectKind, ResolvedSourceObjectProvenance, ResolvedSourceSpan,
};

use crate::ast::{
    ArithExpr, BodyLiteral, DomainDecl, FuncBody, PredDecl, Program, Rule, Term, TypeRef,
};
use crate::lower::Lowerer;
use crate::meta_normalize::static_meta_predicate_dependency;
use crate::module::{module_path_to_string, LoadedModule, ModuleError, ModulePath};
use crate::parser::parse_program;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
#[cfg(target_os = "linux")]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use xlog_core::{ScalarType, XlogError};

/// Predicate and function names mapped to the source module selected by import validation.
pub type ValidatedImports = (HashMap<String, ModulePath>, HashMap<String, ModulePath>);

fn collect_term_predicate_dependencies(term: &Term, predicates: &mut HashSet<String>) {
    match term {
        Term::List(items) => {
            for item in items {
                collect_term_predicate_dependencies(item, predicates);
            }
        }
        Term::Cons { head, tail } => {
            collect_term_predicate_dependencies(head, predicates);
            collect_term_predicate_dependencies(tail, predicates);
        }
        Term::Compound { args, .. } => {
            for argument in args {
                collect_term_predicate_dependencies(argument, predicates);
            }
        }
        Term::PredRef(name) => {
            predicates.insert(name.clone());
        }
        Term::Variable(_)
        | Term::Anonymous
        | Term::Integer(_)
        | Term::Float(_)
        | Term::String(_)
        | Term::Symbol(_)
        | Term::Aggregate(_) => {}
    }
}

fn collect_arithmetic_function_dependencies(
    expression: &ArithExpr,
    functions: &mut HashSet<String>,
) {
    match expression {
        ArithExpr::Add(left, right)
        | ArithExpr::Sub(left, right)
        | ArithExpr::Mul(left, right)
        | ArithExpr::Div(left, right)
        | ArithExpr::Mod(left, right)
        | ArithExpr::Min(left, right)
        | ArithExpr::Max(left, right)
        | ArithExpr::Pow(left, right) => {
            collect_arithmetic_function_dependencies(left, functions);
            collect_arithmetic_function_dependencies(right, functions);
        }
        ArithExpr::Abs(inner) | ArithExpr::Cast(inner, _) => {
            collect_arithmetic_function_dependencies(inner, functions);
        }
        ArithExpr::FuncCall { name, args } => {
            functions.insert(name.clone());
            for argument in args {
                collect_arithmetic_function_dependencies(argument, functions);
            }
        }
        ArithExpr::Conditional {
            cond_left,
            cond_right,
            then_expr,
            else_expr,
            ..
        } => {
            collect_arithmetic_function_dependencies(cond_left, functions);
            collect_arithmetic_function_dependencies(cond_right, functions);
            collect_arithmetic_function_dependencies(then_expr, functions);
            collect_arithmetic_function_dependencies(else_expr, functions);
        }
        ArithExpr::Variable(_) | ArithExpr::Integer(_) | ArithExpr::Float(_) => {}
    }
}

fn collect_body_dependencies(
    body: &[BodyLiteral],
    predicates: &mut HashSet<String>,
    functions: &mut HashSet<String>,
) {
    for literal in body {
        match literal {
            BodyLiteral::Positive(atom) => {
                predicates.insert(atom.predicate.clone());
                if let Some(dependency) = static_meta_predicate_dependency(atom) {
                    predicates.insert(dependency);
                }
                for term in &atom.terms {
                    collect_term_predicate_dependencies(term, predicates);
                }
            }
            BodyLiteral::Negated(atom) => {
                predicates.insert(atom.predicate.clone());
                for term in &atom.terms {
                    collect_term_predicate_dependencies(term, predicates);
                }
            }
            BodyLiteral::Epistemic(literal) => {
                predicates.insert(literal.atom.predicate.clone());
                for term in &literal.atom.terms {
                    collect_term_predicate_dependencies(term, predicates);
                }
            }
            BodyLiteral::Comparison(comparison) => {
                collect_term_predicate_dependencies(&comparison.left, predicates);
                collect_term_predicate_dependencies(&comparison.right, predicates);
            }
            BodyLiteral::IsExpr(expression) => {
                collect_arithmetic_function_dependencies(&expression.expr, functions);
            }
            BodyLiteral::Univ(univ) => {
                collect_term_predicate_dependencies(&univ.term, predicates);
                collect_term_predicate_dependencies(&univ.parts, predicates);
            }
        }
    }
}

fn collect_function_body_dependencies(
    body: &FuncBody,
    predicates: &mut HashSet<String>,
    functions: &mut HashSet<String>,
) {
    match body {
        FuncBody::Arithmetic(expression) => {
            collect_arithmetic_function_dependencies(expression, functions);
        }
        FuncBody::Conditional(expression) => {
            collect_arithmetic_function_dependencies(&expression.cond_left, functions);
            collect_arithmetic_function_dependencies(&expression.cond_right, functions);
            collect_function_body_dependencies(&expression.then_branch, predicates, functions);
            collect_function_body_dependencies(&expression.else_branch, predicates, functions);
        }
        FuncBody::Predicate { body, .. } => {
            collect_body_dependencies(body, predicates, functions);
        }
    }
}

/// Predicate and function names classified within one lexical import scope.
#[derive(Clone, Default)]
struct ModuleItems {
    predicates: HashSet<String>,
    functions: HashSet<String>,
}

/// Names supplied by at least one visible provider, and names filtered out by
/// every provider in the same lexical import scope.
#[derive(Default)]
struct ImportScope {
    visible: ModuleItems,
    hidden: ModuleItems,
}

/// Function provider contributed by a resolved import branch.
#[derive(Clone)]
struct ImportProvider {
    module: ModulePath,
    source: PathBuf,
}

#[derive(Clone)]
struct ImportedPredicateDeclaration {
    module: ModulePath,
    schema: PredicateDeclarationSchema,
}

#[derive(Clone, PartialEq, Eq)]
struct PredicateDeclarationSchema {
    columns: Vec<(Option<String>, TypeRef)>,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PredicateKey {
    name: String,
    arity: usize,
}

#[derive(Clone)]
struct InferredPredicateContribution {
    provider: ImportProvider,
    rule: Rule,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum InferredColumn {
    Unknown,
    Known(ScalarType),
    Conflicting,
}

#[derive(Clone)]
struct ImportedDomainDeclaration {
    module: ModulePath,
    declaration: DomainDecl,
}

#[derive(Default)]
struct ImportProviders {
    functions: BTreeMap<String, ImportProvider>,
    predicate_declarations: BTreeMap<String, ImportedPredicateDeclaration>,
    inferred_predicate_contributions: BTreeMap<PredicateKey, Vec<InferredPredicateContribution>>,
    domain_declarations: BTreeMap<String, ImportedDomainDeclaration>,
}

/// One import declaration resolved in the context of its owning source file.
#[derive(Clone)]
struct ResolvedImport {
    source: PathBuf,
    module_path: ModulePath,
    imports: Option<Vec<String>>,
}

struct ResolvedImportGroup {
    source: PathBuf,
    module_path: ModulePath,
    imported_items: Option<HashSet<String>>,
}

#[derive(Hash, PartialEq, Eq)]
struct ImportMergeKey {
    source: PathBuf,
    imported_items: Option<Vec<String>>,
}

/// A `#pragma` directive declared in an imported module.
///
/// Pragmas are entry-file-scoped: a directive declared in an imported
/// module never affects compilation. These records exist so callers can
/// surface the dropped pragmas as warnings instead of ignoring them
/// silently.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct IgnoredImportPragma {
    /// Module path string (e.g. `rules/common/base`).
    pub module: String,
    /// Pragma key as written in source (e.g. `magic_sets`).
    pub pragma: &'static str,
}

impl std::fmt::Display for IgnoredImportPragma {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "warning[W0510]: `#pragma {}` in imported module `{}` is ignored",
            self.pragma, self.module
        )?;
        write!(
            f,
            "  = note: pragmas apply only when declared in the entry file"
        )
    }
}

/// Resolves and loads modules
pub struct ModuleResolver {
    /// Directories to search for modules
    search_paths: Vec<PathBuf>,
    /// Loaded modules keyed by canonical source-file identity.
    loaded: HashMap<PathBuf, LoadedModule>,
    /// Exact UTF-8 source bytes parsed for each canonical loaded module.
    loaded_source_texts: HashMap<PathBuf, String>,
    /// Logical path spellings mapped to their resolved source files. Bare
    /// aliases retain first-load lookup behavior for public inspection APIs;
    /// resolved programs use contextual paths that identify each import edge.
    module_aliases: HashMap<String, Vec<PathBuf>>,
    /// Import edges resolved separately for every loaded source file.
    resolved_imports: HashMap<PathBuf, Vec<ResolvedImport>>,
    /// Entry source loaded from its exact filesystem path. It is kept outside
    /// the logical module map so its filename cannot collide with a `use` path.
    entry: Option<LoadedModule>,
    /// Canonical source identity and resolved import edges for the entry file.
    entry_source: Option<PathBuf>,
    /// Exact UTF-8 source bytes parsed for the entry file.
    entry_source_text: Option<String>,
    entry_resolved_imports: Vec<ResolvedImport>,
    /// Source identity of the most recent public `load_module` root.
    root_module: Option<PathBuf>,
    /// Source identities and display paths currently loading, for cycle detection.
    loading: Vec<(PathBuf, ModulePath)>,
    /// Path key of the entry module, when known. The entry module's own
    /// pragmas are authoritative and excluded from the ignored-pragma
    /// listing.
    entry_module: Option<String>,
}

#[cfg(target_os = "linux")]
fn linux_proc_fd_identity(source_file: &Path) -> Option<std::io::Result<PathBuf>> {
    let descriptor = source_file.strip_prefix("/proc/self/fd").ok()?.to_str()?;
    if descriptor.is_empty() || !descriptor.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    Some(fs::metadata(source_file).map(|metadata| {
        PathBuf::from(format!(
            "/proc/self/fd/@xlog-source-{:x}-{:x}",
            metadata.dev(),
            metadata.ino()
        ))
    }))
}

impl ModuleResolver {
    /// Create a new resolver with given search paths
    pub fn new(search_paths: Vec<PathBuf>) -> Self {
        Self {
            search_paths,
            loaded: HashMap::new(),
            loaded_source_texts: HashMap::new(),
            module_aliases: HashMap::new(),
            resolved_imports: HashMap::new(),
            entry: None,
            entry_source: None,
            entry_source_text: None,
            entry_resolved_imports: Vec::new(),
            root_module: None,
            loading: Vec::new(),
            entry_module: None,
        }
    }

    /// Record which loaded module is the compilation entry point.
    ///
    /// The entry module's pragmas are the ones the compiler honors;
    /// [`Self::ignored_import_pragmas`] skips it.
    pub fn mark_entry_module(&mut self, path_key: &str) {
        self.entry_module = Some(path_key.to_string());
    }

    /// List `#pragma` directives declared in imported (non-entry) modules.
    ///
    /// Pragmas are entry-file-scoped, so anything an imported module
    /// declares is dropped at merge time. Callers surface these records as
    /// warnings so the scoping is never silent. The result is sorted by
    /// module path, then pragma name, for deterministic output.
    ///
    /// Nested imports resolve relative to the importer's directory, so one
    /// file can be loaded under several module-path spellings. Warnings are
    /// deduplicated on the canonical source file (one warning per file per
    /// pragma), keeping the alphabetically-first module label; the entry
    /// file itself never warns under any spelling.
    pub fn ignored_import_pragmas(&self) -> Vec<IgnoredImportPragma> {
        let entry_source = self
            .entry_module
            .as_deref()
            .and_then(|path| self.module_aliases.get(path))
            .and_then(|sources| sources.first().cloned())
            .or_else(|| self.entry_source.clone());

        let mut candidates: Vec<(PathBuf, IgnoredImportPragma)> = Vec::new();
        for (path_key, sources) in &self.module_aliases {
            for source in sources {
                if entry_source.as_ref() == Some(source) {
                    continue;
                }
                let Some(module) = self.loaded.get(source) else {
                    continue;
                };
                for pragma in module.program.directives.set_pragma_names() {
                    candidates.push((
                        source.clone(),
                        IgnoredImportPragma {
                            module: path_key.clone(),
                            pragma,
                        },
                    ));
                }
            }
        }

        candidates.sort_by(|a, b| a.1.cmp(&b.1));
        let mut seen: HashSet<(PathBuf, &'static str)> = HashSet::new();
        let mut ignored = Vec::with_capacity(candidates.len());
        for (source, warning) in candidates {
            if seen.insert((source, warning.pragma)) {
                ignored.push(warning);
            }
        }
        ignored
    }

    fn source_identity(source_file: &Path) -> Result<PathBuf, ModuleError> {
        let identity = match fs::canonicalize(source_file) {
            Ok(identity) => Ok(identity),
            Err(error) => {
                #[cfg(target_os = "linux")]
                {
                    linux_proc_fd_identity(source_file).unwrap_or(Err(error))
                }
                #[cfg(not(target_os = "linux"))]
                {
                    Err(error)
                }
            }
        };
        identity.map_err(|error| ModuleError::ParseError {
            path: source_file.to_path_buf(),
            message: format!("failed to resolve source-file identity: {error}"),
        })
    }

    fn resolve_module_file(
        &self,
        base_dir: &Path,
        module_path: &[String],
    ) -> Option<(PathBuf, bool)> {
        let relative_path = format!("{}.xlog", module_path.join("/"));
        let candidate = base_dir.join(&relative_path);
        if candidate.exists() {
            return Some((candidate, true));
        }

        for search_path in &self.search_paths {
            let candidate = search_path.join(&relative_path);
            if candidate.exists() {
                return Some((candidate, false));
            }
        }

        None
    }

    /// Find the file for a module path
    pub fn find_module_file(&self, base_dir: &Path, module_path: &[String]) -> Option<PathBuf> {
        self.resolve_module_file(base_dir, module_path)
            .map(|(path, _)| path)
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
    fn check_cycle(&self, source: &Path, module_path: &[String]) -> Option<Vec<ModulePath>> {
        for (i, (loading_source, _)) in self.loading.iter().enumerate() {
            if loading_source == source {
                let mut cycle: Vec<ModulePath> = self.loading[i..]
                    .iter()
                    .map(|(_, path)| path.clone())
                    .collect();
                cycle.push(module_path.to_vec());
                return Some(cycle);
            }
        }
        None
    }

    /// Extract exports from a parsed program
    /// Returns (predicate exports, function exports)
    pub fn extract_exports(program: &Program) -> (HashSet<String>, HashSet<String>) {
        let mut pred_exports = HashSet::new();
        let mut func_exports = HashSet::new();

        // Add declared predicates that aren't private
        for pred in &program.predicates {
            if !pred.is_private {
                pred_exports.insert(pred.name.clone());
            }
        }

        // Add rule heads (all rules define public predicates unless declared private)
        for rule in &program.rules {
            // Check if this predicate was declared as private
            let is_private = program
                .predicates
                .iter()
                .any(|p| p.name == rule.head.predicate && p.is_private);
            if !is_private {
                pred_exports.insert(rule.head.predicate.clone());
            }
        }

        // Add functions that aren't private
        for func in &program.functions {
            if !func.is_private {
                func_exports.insert(func.name.clone());
            }
        }

        (pred_exports, func_exports)
    }

    /// Load a module from a logical path and make it the resolution root.
    ///
    /// A successful call replaces any exact-file entry anchor used by
    /// [`Self::validate_imports`] and [`Self::merge_imports`].
    pub fn load_module(
        &mut self,
        base_dir: &Path,
        module_path: &[String],
    ) -> Result<&LoadedModule, ModuleError> {
        let (source, _) = self.load_module_resolved(base_dir, None, module_path)?;
        self.entry = None;
        self.entry_source = None;
        self.entry_source_text = None;
        self.entry_resolved_imports.clear();
        self.root_module = Some(source.clone());
        Ok(self.loaded.get(&source).expect("module was just resolved"))
    }

    /// Load the compilation entry from its exact filesystem path.
    ///
    /// Imported modules still use normal `.xlog` module lookup. The entry file
    /// itself may use any extension accepted by the caller. It is tracked
    /// separately from logical imports so a same-stem `.xlog` module remains
    /// addressable through `use`. Nested imports in an imported module resolve
    /// beside its canonical source target, making symbolic-link aliases
    /// independent of load order.
    pub fn load_entry_file(&mut self, entry_file: &Path) -> Result<&LoadedModule, ModuleError> {
        let base_dir = entry_file.parent().unwrap_or(Path::new("."));
        let module_name = entry_file
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("main")
            .to_string();
        let module_path = vec![module_name.clone()];

        let (module, source_text) =
            Self::parse_module_file(&module_path, entry_file.to_path_buf())?;
        let entry_source = Self::source_identity(entry_file)?;
        let module_dir = module.source_file.parent().unwrap_or(base_dir);
        self.loading
            .push((entry_source.clone(), module_path.clone()));
        let resolved_imports = (|| {
            let mut resolved = Vec::with_capacity(module.program.imports.len());
            for import in &module.program.imports {
                let (source, resolved_path) =
                    self.load_module_resolved(module_dir, Some(&module_path), &import.module_path)?;
                resolved.push(ResolvedImport {
                    source,
                    module_path: resolved_path,
                    imports: import.imports.clone(),
                });
            }
            Ok(resolved)
        })();
        self.loading.pop();
        let resolved_imports = resolved_imports?;
        self.entry_module = None;
        self.entry_source = Some(entry_source);
        self.entry_source_text = Some(source_text);
        self.entry_resolved_imports = resolved_imports;
        self.root_module = None;
        self.entry = Some(module);
        Ok(self.entry.as_ref().expect("entry module was just loaded"))
    }

    fn contextual_module_path(
        parent_path: Option<&[String]>,
        declared_path: &[String],
        resolved_relative_to_parent: bool,
    ) -> ModulePath {
        if !resolved_relative_to_parent {
            return declared_path.to_vec();
        }

        let mut resolved = parent_path
            .and_then(|path| path.split_last().map(|(_, prefix)| prefix.to_vec()))
            .unwrap_or_default();
        resolved.extend_from_slice(declared_path);
        resolved
    }

    fn record_module_alias(&mut self, module_path: &[String], source: &Path) {
        let sources = self
            .module_aliases
            .entry(module_path_to_string(module_path))
            .or_default();
        if !sources.iter().any(|candidate| candidate == source) {
            sources.push(source.to_path_buf());
        }
    }

    fn load_module_resolved(
        &mut self,
        base_dir: &Path,
        parent_path: Option<&[String]>,
        declared_path: &[String],
    ) -> Result<(PathBuf, ModulePath), ModuleError> {
        let (source_file, resolved_relative_to_parent) = self
            .resolve_module_file(base_dir, declared_path)
            .ok_or_else(|| ModuleError::NotFound {
                path: declared_path.to_vec(),
                searched: self.searched_paths(base_dir, declared_path),
            })?;
        let source = Self::source_identity(&source_file)?;
        let contextual_path =
            Self::contextual_module_path(parent_path, declared_path, resolved_relative_to_parent);

        if let Some(cycle) = self.check_cycle(&source, &contextual_path) {
            return Err(ModuleError::CircularImport { cycle });
        }

        if self.loaded.contains_key(&source) {
            self.record_module_alias(declared_path, &source);
            self.record_module_alias(&contextual_path, &source);
            return Ok((source, contextual_path));
        }

        self.loading.push((source.clone(), contextual_path.clone()));

        let loaded = (|| {
            let (module, source_text) = Self::parse_module_file(&contextual_path, source_file)?;
            // Canonical aliases share one module identity and therefore one
            // deterministic dependency closure. Resolve nested imports beside
            // the canonical source instead of whichever alias loaded first.
            let module_dir = source.parent().unwrap_or(base_dir).to_path_buf();
            let mut resolved_imports = Vec::with_capacity(module.program.imports.len());
            for import in &module.program.imports {
                let (target_source, resolved_path) = self.load_module_resolved(
                    &module_dir,
                    Some(&contextual_path),
                    &import.module_path,
                )?;
                resolved_imports.push(ResolvedImport {
                    source: target_source,
                    module_path: resolved_path,
                    imports: import.imports.clone(),
                });
            }
            Ok((module, source_text, resolved_imports))
        })();

        self.loading.pop();
        let (module, source_text, resolved_imports) = loaded?;
        let primary_path = module.path.clone();

        self.record_module_alias(declared_path, &source);
        self.record_module_alias(&contextual_path, &source);
        self.resolved_imports
            .insert(source.clone(), resolved_imports);
        self.loaded_source_texts.insert(source.clone(), source_text);
        self.loaded.insert(source.clone(), module);
        Ok((source, primary_path))
    }

    fn parse_module_file(
        module_path: &[String],
        source_file: PathBuf,
    ) -> Result<(LoadedModule, String), ModuleError> {
        let source = fs::read_to_string(&source_file).map_err(|error| ModuleError::ParseError {
            path: source_file.clone(),
            message: error.to_string(),
        })?;
        let program = parse_program(&source).map_err(|error| ModuleError::ParseError {
            path: source_file.clone(),
            message: error.to_string(),
        })?;
        let (exports, function_exports) = Self::extract_exports(&program);

        Ok((
            LoadedModule {
                path: module_path.to_vec(),
                source_file,
                exports,
                function_exports,
                program,
            },
            source,
        ))
    }

    /// Check if a predicate can be imported from a module
    ///
    /// This compatibility inspection uses the first source registered for the
    /// logical path. Semantic validation follows importer-scoped resolved edges
    /// and does not use this alias lookup.
    pub fn check_import(&self, module_path: &[String], predicate: &str) -> Result<(), ModuleError> {
        let (_, module) = self
            .first_loaded_module_for_alias(module_path)
            .ok_or_else(|| ModuleError::NotFound {
                path: module_path.to_vec(),
                searched: vec![],
            })?;

        Self::validate_consistent_predicate_visibility(&module.program, module_path)?;
        if !module.exports.contains(predicate) {
            return Err(ModuleError::PredicateNotFound {
                name: predicate.to_string(),
                module: module_path.to_vec(),
            });
        }

        Ok(())
    }

    /// Validate all imports in a program.
    ///
    /// Import edges come from the most recently loaded entry file or root module
    /// when its declarations match `program`. A context-free request is accepted
    /// only when every logical path identifies one loaded source file.
    ///
    /// Returns predicate and function names with the first resolved source module
    /// retained as a representative. Entry declarations and selected public
    /// import declarations participate in declaration compatibility checks. For
    /// signatures without a participating declaration, inferred head-column
    /// types from entry and selected public import clauses are validated before
    /// merging. Constants, head variables typed by ordinary body atoms or
    /// built-in arithmetic bindings, and aggregate result types supply evidence;
    /// unanchored variables do not.
    pub fn validate_imports(&self, program: &Program) -> Result<ValidatedImports, ModuleError> {
        let imports = self.resolved_imports_for_program(program)?;
        let validated = self.validate_resolved_imports(&imports)?;
        self.validate_program_against_imports(program, &imports)?;
        Ok(validated)
    }

    fn validate_resolved_imports(
        &self,
        imports: &[ResolvedImport],
    ) -> Result<ValidatedImports, ModuleError> {
        let mut imported_predicates: HashMap<String, ModulePath> = HashMap::new();
        let mut imported_functions: HashMap<String, ModulePath> = HashMap::new();
        let mut function_providers: HashMap<String, ImportProvider> = HashMap::new();

        for resolved_import in imports {
            let module =
                self.loaded
                    .get(&resolved_import.source)
                    .ok_or_else(|| ModuleError::NotFound {
                        path: resolved_import.module_path.clone(),
                        searched: vec![],
                    })?;

            // Combine all available exports for wildcard imports
            let all_exports: HashSet<String> = module
                .exports
                .iter()
                .chain(module.function_exports.iter())
                .cloned()
                .collect();

            let mut names_to_import: Vec<String> = match &resolved_import.imports {
                Some(specific) => specific.clone(),
                None => all_exports.iter().cloned().collect(),
            };
            names_to_import.sort();

            for name in names_to_import {
                // Check if name exists as predicate or function
                let is_predicate = module.exports.contains(&name);
                let is_function = module.function_exports.contains(&name);

                if !is_predicate && !is_function {
                    return Err(ModuleError::PredicateNotFound {
                        name: name.clone(),
                        module: resolved_import.module_path.clone(),
                    });
                }

                if is_predicate {
                    imported_predicates
                        .entry(name.clone())
                        .or_insert_with(|| resolved_import.module_path.clone());
                }

                if is_function {
                    if let Some(previous) = function_providers.get(&name) {
                        if previous.source != resolved_import.source {
                            return Err(ModuleError::ImportConflict {
                                name,
                                module1: previous.module.clone(),
                                module2: resolved_import.module_path.clone(),
                            });
                        }
                    } else {
                        function_providers.insert(
                            name.clone(),
                            ImportProvider {
                                module: resolved_import.module_path.clone(),
                                source: resolved_import.source.clone(),
                            },
                        );
                    }
                    imported_functions
                        .entry(name.clone())
                        .or_insert_with(|| resolved_import.module_path.clone());
                }
            }
        }

        Ok((imported_predicates, imported_functions))
    }

    fn resolved_imports_for_program(
        &self,
        program: &Program,
    ) -> Result<Vec<ResolvedImport>, ModuleError> {
        if self
            .entry
            .as_ref()
            .is_some_and(|entry| entry.program.imports == program.imports)
        {
            return Ok(self.entry_resolved_imports.clone());
        }

        if let Some(root_source) = &self.root_module {
            if let Some(root) = self.loaded.get(root_source) {
                if root.program.imports == program.imports {
                    return Ok(self
                        .resolved_imports
                        .get(root_source)
                        .cloned()
                        .unwrap_or_default());
                }
            }
        }

        program
            .imports
            .iter()
            .map(|use_decl| {
                let sources = self
                    .module_aliases
                    .get(&module_path_to_string(&use_decl.module_path))
                    .ok_or_else(|| ModuleError::NotFound {
                        path: use_decl.module_path.clone(),
                        searched: vec![],
                    })?;
                if sources.len() > 1 {
                    let mut candidates = sources.clone();
                    candidates.sort();
                    return Err(ModuleError::AmbiguousModulePath {
                        path: use_decl.module_path.clone(),
                        candidates,
                    });
                }
                let source = sources
                    .first()
                    .filter(|source| self.loaded.contains_key(*source))
                    .ok_or_else(|| ModuleError::NotFound {
                        path: use_decl.module_path.clone(),
                        searched: vec![],
                    })?;
                Ok(ResolvedImport {
                    source: source.clone(),
                    module_path: use_decl.module_path.clone(),
                    imports: use_decl.imports.clone(),
                })
            })
            .collect()
    }

    fn first_loaded_module_for_alias(
        &self,
        module_path: &[String],
    ) -> Option<(&PathBuf, &LoadedModule)> {
        let source = self
            .module_aliases
            .get(&module_path_to_string(module_path))?
            .first()?;
        self.loaded.get_key_value(source)
    }

    /// Get a loaded logical import by module path.
    ///
    /// If several importer contexts registered the same logical path for
    /// different files, this compatibility view returns the first registered
    /// source. Semantic validation and merging use resolved import edges.
    pub fn get_module(&self, module_path: &[String]) -> Option<&LoadedModule> {
        self.first_loaded_module_for_alias(module_path)
            .map(|(_, module)| module)
    }

    /// Return the entry source loaded from its exact path, if present.
    pub fn entry(&self) -> Option<&LoadedModule> {
        self.entry.as_ref()
    }

    /// Check whether at least one source is registered for a logical import path.
    pub fn is_loaded(&self, module_path: &str) -> bool {
        self.module_aliases.contains_key(module_path)
    }

    /// Get all registered logical import aliases (for testing).
    pub fn loaded_modules(&self) -> Vec<&str> {
        self.module_aliases.keys().map(String::as_str).collect()
    }

    fn imported_item_set(imports: &Option<Vec<String>>) -> Option<HashSet<String>> {
        match imports {
            Some(items) if !items.is_empty() => Some(items.iter().cloned().collect()),
            _ => None,
        }
    }

    fn combined_import_selections(imports: &[ResolvedImport]) -> Vec<ResolvedImportGroup> {
        // Repeated declarations in one source file form one selection. Keep
        // this combination lexical: an unrelated importing module must not
        // make an omitted dependency visible here.
        let mut combined = Vec::<ResolvedImportGroup>::new();
        let mut indexes = HashMap::<PathBuf, usize>::new();
        for resolved_import in imports {
            let selection = Self::imported_item_set(&resolved_import.imports);
            if let Some(index) = indexes.get(&resolved_import.source).copied() {
                let existing = &mut combined[index].imported_items;
                match (existing.as_mut(), selection) {
                    (Some(existing_names), Some(names)) => existing_names.extend(names),
                    (_, None) => *existing = None,
                    (None, Some(_)) => {}
                }
            } else {
                indexes.insert(resolved_import.source.clone(), combined.len());
                combined.push(ResolvedImportGroup {
                    source: resolved_import.source.clone(),
                    module_path: resolved_import.module_path.clone(),
                    imported_items: selection,
                });
            }
        }
        combined
    }

    fn local_predicate_names(program: &Program) -> HashSet<String> {
        program
            .predicates
            .iter()
            .map(|predicate| predicate.name.clone())
            .chain(program.rules.iter().map(|rule| rule.head.predicate.clone()))
            .collect()
    }

    fn local_function_names(program: &Program) -> HashSet<String> {
        program
            .functions
            .iter()
            .map(|function| function.name.clone())
            .collect()
    }

    fn hidden_local_function_names(
        program: &Program,
        imported_items: Option<&HashSet<String>>,
    ) -> HashSet<String> {
        let private_functions = program
            .functions
            .iter()
            .filter(|function| function.is_private)
            .map(|function| function.name.clone())
            .collect::<HashSet<_>>();
        Self::local_function_names(program)
            .into_iter()
            .filter(|name| {
                private_functions.contains(name)
                    || imported_items.is_some_and(|items| !items.contains(name))
            })
            .collect()
    }

    fn hidden_local_items(
        program: &Program,
        imported_items: Option<&HashSet<String>>,
    ) -> ModuleItems {
        let private_predicates = program
            .predicates
            .iter()
            .filter(|predicate| predicate.is_private)
            .map(|predicate| predicate.name.clone())
            .collect::<HashSet<_>>();
        let predicates = Self::local_predicate_names(program)
            .into_iter()
            .filter(|name| {
                private_predicates.contains(name)
                    || imported_items.is_some_and(|items| !items.contains(name))
            })
            .collect();

        let functions = Self::hidden_local_function_names(program, imported_items);

        ModuleItems {
            predicates,
            functions,
        }
    }

    fn visible_local_functions(
        program: &Program,
        imported_items: Option<&HashSet<String>>,
    ) -> HashSet<String> {
        let hidden = Self::hidden_local_function_names(program, imported_items);
        Self::local_function_names(program)
            .difference(&hidden)
            .cloned()
            .collect()
    }

    fn local_inferred_predicate_contributions(
        program: &Program,
        excluded_predicates: &HashSet<String>,
        provider: ImportProvider,
    ) -> BTreeMap<PredicateKey, Vec<InferredPredicateContribution>> {
        let mut contributions = BTreeMap::<PredicateKey, Vec<InferredPredicateContribution>>::new();

        for rule in &program.rules {
            let key = PredicateKey {
                name: rule.head.predicate.clone(),
                arity: rule.head.arity(),
            };
            if excluded_predicates.contains(&key.name)
                || program.predicates.iter().any(|declaration| {
                    declaration.name == key.name && declaration.arity() == key.arity
                })
            {
                continue;
            }

            let contribution = InferredPredicateContribution {
                provider: provider.clone(),
                rule: rule.clone(),
            };
            let existing = contributions.entry(key).or_default();
            if !existing.iter().any(|candidate| {
                candidate.provider.source == contribution.provider.source
                    && candidate.rule == contribution.rule
            }) {
                existing.push(contribution);
            }
        }

        contributions
    }

    fn merge_inferred_predicate_contributions(
        contributions: &mut BTreeMap<PredicateKey, Vec<InferredPredicateContribution>>,
        incoming: BTreeMap<PredicateKey, Vec<InferredPredicateContribution>>,
    ) {
        for (key, incoming_contributions) in incoming {
            let existing = contributions.entry(key).or_default();
            for contribution in incoming_contributions {
                if !existing.iter().any(|candidate| {
                    candidate.provider.source == contribution.provider.source
                        && candidate.rule == contribution.rule
                }) {
                    existing.push(contribution);
                }
            }
        }
    }

    fn storage_scalar_type(typ: &TypeRef) -> Option<ScalarType> {
        match typ {
            TypeRef::Scalar(typ) => Some(*typ),
            TypeRef::List(_) | TypeRef::Term | TypeRef::Compound | TypeRef::PredRef => {
                Some(ScalarType::U64)
            }
            TypeRef::Domain(_) => None,
        }
    }

    fn inferred_rule_columns(
        schema_inference: &Lowerer,
        rule: &Rule,
        schemas: &BTreeMap<PredicateKey, Vec<InferredColumn>>,
    ) -> xlog_core::Result<Vec<Option<ScalarType>>> {
        schema_inference.infer_rule_head_column_types_before_function_expansion(
            rule,
            |atom, index| {
                schemas
                    .get(&PredicateKey {
                        name: atom.predicate.clone(),
                        arity: atom.arity(),
                    })
                    .and_then(|columns| columns.get(index))
                    .and_then(|column| match column {
                        InferredColumn::Known(typ) => Some(*typ),
                        InferredColumn::Unknown | InferredColumn::Conflicting => None,
                    })
            },
        )
    }

    fn predicate_schema_inference_error(
        key: &PredicateKey,
        contribution: &InferredPredicateContribution,
        error: XlogError,
    ) -> ModuleError {
        ModuleError::PredicateSchemaInferenceFailed {
            name: key.name.clone(),
            arity: key.arity,
            module: contribution.provider.module.clone(),
            source: Box::new(contribution.provider.source.clone()),
            message: error.to_string(),
        }
    }

    fn inferred_predicate_schemas(
        providers: &ImportProviders,
    ) -> Result<BTreeMap<PredicateKey, Vec<InferredColumn>>, ModuleError> {
        let mut schemas = BTreeMap::<PredicateKey, Vec<InferredColumn>>::new();
        let mut declared = BTreeSet::<PredicateKey>::new();

        for (name, declaration) in &providers.predicate_declarations {
            let key = PredicateKey {
                name: name.clone(),
                arity: declaration.schema.columns.len(),
            };
            let columns = declaration
                .schema
                .columns
                .iter()
                .map(|(_, typ)| {
                    Self::storage_scalar_type(typ)
                        .map(InferredColumn::Known)
                        .unwrap_or(InferredColumn::Unknown)
                })
                .collect();
            declared.insert(key.clone());
            schemas.insert(key, columns);
        }

        let total_columns = providers
            .inferred_predicate_contributions
            .keys()
            .filter(|key| !declared.contains(*key))
            .map(|key| key.arity)
            .sum::<usize>();
        let max_iterations = total_columns.saturating_mul(2).saturating_add(1);
        let schema_inference = Lowerer::new();
        let mut converged = false;
        for _ in 0..max_iterations {
            let mut changed = false;
            for (key, contributions) in &providers.inferred_predicate_contributions {
                if declared.contains(key) {
                    continue;
                }
                for contribution in contributions {
                    let columns = Self::inferred_rule_columns(
                        &schema_inference,
                        &contribution.rule,
                        &schemas,
                    )
                    .map_err(|error| {
                        Self::predicate_schema_inference_error(key, contribution, error)
                    })?;
                    let schema = schemas
                        .entry(key.clone())
                        .or_insert_with(|| vec![InferredColumn::Unknown; key.arity]);
                    for (slot, inferred) in schema.iter_mut().zip(columns) {
                        let Some(inferred) = inferred else {
                            continue;
                        };
                        let updated = match *slot {
                            InferredColumn::Unknown => InferredColumn::Known(inferred),
                            InferredColumn::Known(existing) if existing != inferred => {
                                InferredColumn::Conflicting
                            }
                            InferredColumn::Known(_) | InferredColumn::Conflicting => continue,
                        };
                        *slot = updated;
                        changed = true;
                    }
                }
            }
            if !changed {
                converged = true;
                break;
            }
        }
        debug_assert!(
            converged,
            "predicate schema inference exceeded its monotonic transition bound"
        );

        Ok(schemas)
    }

    fn validate_inferred_predicate_contributions(
        providers: &ImportProviders,
    ) -> Result<(), ModuleError> {
        let schemas = Self::inferred_predicate_schemas(providers)?;
        let schema_inference = Lowerer::new();
        for (key, contributions) in &providers.inferred_predicate_contributions {
            if providers
                .predicate_declarations
                .get(&key.name)
                .is_some_and(|declaration| declaration.schema.columns.len() == key.arity)
            {
                continue;
            }

            let inferred = contributions
                .iter()
                .map(|contribution| {
                    Self::inferred_rule_columns(&schema_inference, &contribution.rule, &schemas)
                        .map(|columns| (contribution, columns))
                        .map_err(|error| {
                            Self::predicate_schema_inference_error(key, contribution, error)
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            for (left_index, (left, left_columns)) in inferred.iter().enumerate() {
                for (right, right_columns) in inferred.iter().skip(left_index + 1) {
                    if left.provider.source == right.provider.source {
                        continue;
                    }
                    for (column_index, (left_type, right_type)) in
                        left_columns.iter().zip(right_columns).enumerate()
                    {
                        if let (Some(type1), Some(type2)) = (left_type, right_type) {
                            if type1 != type2 {
                                return Err(ModuleError::IncompatibleInferredPredicateSchema {
                                    name: key.name.clone(),
                                    arity: key.arity,
                                    column: column_index + 1,
                                    type1: *type1,
                                    type2: *type2,
                                    module1: left.provider.module.clone(),
                                    module2: right.provider.module.clone(),
                                    source1: Box::new(left.provider.source.clone()),
                                    source2: Box::new(right.provider.source.clone()),
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn merge_function_providers(
        providers: &mut BTreeMap<String, ImportProvider>,
        incoming: BTreeMap<String, ImportProvider>,
    ) -> Result<(), ModuleError> {
        for (name, provider) in incoming {
            if let Some(existing) = providers.get(&name) {
                if existing.source != provider.source {
                    return Err(ModuleError::ImportConflict {
                        name,
                        module1: existing.module.clone(),
                        module2: provider.module,
                    });
                }
            } else {
                providers.insert(name, provider);
            }
        }
        Ok(())
    }

    fn record_predicate_declaration(
        declarations: &mut BTreeMap<String, ImportedPredicateDeclaration>,
        name: String,
        incoming: ImportedPredicateDeclaration,
    ) -> Result<(), ModuleError> {
        if let Some(existing) = declarations.get(&name) {
            if existing.schema != incoming.schema {
                return Err(ModuleError::IncompatiblePredicateDeclaration {
                    name,
                    module1: existing.module.clone(),
                    module2: incoming.module,
                });
            }
        } else {
            declarations.insert(name, incoming);
        }
        Ok(())
    }

    fn normalized_type_ref(
        domains: &BTreeMap<String, ImportedDomainDeclaration>,
        typ: &TypeRef,
    ) -> TypeRef {
        match typ {
            TypeRef::Domain(name) => domains
                .get(name)
                .map(|domain| TypeRef::Scalar(domain.declaration.typ))
                .unwrap_or_else(|| typ.clone()),
            TypeRef::List(element) => {
                TypeRef::List(Box::new(Self::normalized_type_ref(domains, element)))
            }
            _ => typ.clone(),
        }
    }

    fn predicate_declaration_schema(
        domains: &BTreeMap<String, ImportedDomainDeclaration>,
        declaration: &PredDecl,
    ) -> PredicateDeclarationSchema {
        PredicateDeclarationSchema {
            columns: declaration
                .schema_columns()
                .iter()
                .map(|column| {
                    (
                        column.name.clone(),
                        Self::normalized_type_ref(domains, &column.typ),
                    )
                })
                .collect(),
        }
    }

    fn merge_predicate_declaration_map(
        declarations: &mut BTreeMap<String, ImportedPredicateDeclaration>,
        incoming: BTreeMap<String, ImportedPredicateDeclaration>,
    ) -> Result<(), ModuleError> {
        for (name, declaration) in incoming {
            Self::record_predicate_declaration(declarations, name, declaration)?;
        }
        Ok(())
    }

    fn record_domain_declaration(
        declarations: &mut BTreeMap<String, ImportedDomainDeclaration>,
        name: String,
        incoming: ImportedDomainDeclaration,
    ) -> Result<(), ModuleError> {
        if let Some(existing) = declarations.get(&name) {
            if existing.declaration.typ != incoming.declaration.typ {
                return Err(ModuleError::IncompatibleDomainDeclaration {
                    name,
                    module1: existing.module.clone(),
                    module2: incoming.module,
                });
            }
        } else {
            declarations.insert(name, incoming);
        }
        Ok(())
    }

    fn merge_domain_declaration_map(
        declarations: &mut BTreeMap<String, ImportedDomainDeclaration>,
        incoming: BTreeMap<String, ImportedDomainDeclaration>,
    ) -> Result<(), ModuleError> {
        for (name, declaration) in incoming {
            Self::record_domain_declaration(declarations, name, declaration)?;
        }
        Ok(())
    }

    fn validate_unique_imported_functions(
        program: &Program,
        module_path: &[String],
    ) -> Result<(), ModuleError> {
        let mut counts = HashMap::<&str, usize>::new();
        for function in &program.functions {
            *counts.entry(&function.name).or_default() += 1;
        }
        for function in program
            .functions
            .iter()
            .filter(|function| !function.is_private)
        {
            if counts.get(function.name.as_str()).copied().unwrap_or(0) > 1 {
                return Err(ModuleError::DuplicateImportedFunction {
                    name: function.name.clone(),
                    module: module_path.to_vec(),
                });
            }
        }
        Ok(())
    }

    fn validate_consistent_predicate_visibility(
        program: &Program,
        module_path: &[String],
    ) -> Result<(), ModuleError> {
        let mut visibility = HashMap::<&str, bool>::new();
        for declaration in &program.predicates {
            match visibility.get(declaration.name.as_str()) {
                Some(is_private) if *is_private != declaration.is_private => {
                    return Err(ModuleError::ConflictingPredicateVisibility {
                        name: declaration.name.clone(),
                        module: module_path.to_vec(),
                    });
                }
                Some(_) => {}
                None => {
                    visibility.insert(declaration.name.as_str(), declaration.is_private);
                }
            }
        }
        Ok(())
    }

    fn import_providers_for_module(
        &self,
        source: &Path,
        module_path: &[String],
        imported_items: Option<&HashSet<String>>,
    ) -> Result<ImportProviders, ModuleError> {
        let loaded_module = self
            .loaded
            .get(source)
            .ok_or_else(|| ModuleError::NotFound {
                path: module_path.to_vec(),
                searched: vec![],
            })?;
        let nested_imports = self.imports_for_source(source);

        Self::validate_consistent_predicate_visibility(&loaded_module.program, module_path)?;
        Self::validate_unique_imported_functions(&loaded_module.program, module_path)?;
        self.validate_resolved_imports(nested_imports)?;
        let mut providers = self.import_providers_from_imports(nested_imports)?;
        for declaration in &loaded_module.program.domains {
            Self::record_domain_declaration(
                &mut providers.domain_declarations,
                declaration.name.clone(),
                ImportedDomainDeclaration {
                    module: module_path.to_vec(),
                    declaration: declaration.clone(),
                },
            )?;
        }
        for declaration in loaded_module
            .program
            .predicates
            .iter()
            .filter(|declaration| {
                !declaration.is_private
                    && imported_items.is_none_or(|items| items.contains(&declaration.name))
            })
        {
            let schema =
                Self::predicate_declaration_schema(&providers.domain_declarations, declaration);
            Self::record_predicate_declaration(
                &mut providers.predicate_declarations,
                declaration.name.clone(),
                ImportedPredicateDeclaration {
                    module: module_path.to_vec(),
                    schema,
                },
            )?;
        }
        let excluded_predicates =
            Self::hidden_local_items(&loaded_module.program, imported_items).predicates;
        let local_contributions = Self::local_inferred_predicate_contributions(
            &loaded_module.program,
            &excluded_predicates,
            ImportProvider {
                module: module_path.to_vec(),
                source: source.to_path_buf(),
            },
        );
        Self::merge_inferred_predicate_contributions(
            &mut providers.inferred_predicate_contributions,
            local_contributions,
        );
        let mut local_functions = BTreeMap::new();
        for name in Self::visible_local_functions(&loaded_module.program, imported_items) {
            local_functions.insert(
                name,
                ImportProvider {
                    module: module_path.to_vec(),
                    source: source.to_path_buf(),
                },
            );
        }
        // Functions have one body. Keeping the earlier merged definition
        // would silently discard this module's body, so distinct providers
        // are an import conflict even within one branch.
        Self::merge_function_providers(&mut providers.functions, local_functions)?;
        Ok(providers)
    }

    fn import_providers_from_imports(
        &self,
        imports: &[ResolvedImport],
    ) -> Result<ImportProviders, ModuleError> {
        let mut providers = ImportProviders::default();
        for group in Self::combined_import_selections(imports) {
            let incoming = self.import_providers_for_module(
                &group.source,
                &group.module_path,
                group.imported_items.as_ref(),
            )?;
            Self::merge_domain_declaration_map(
                &mut providers.domain_declarations,
                incoming.domain_declarations,
            )?;
            Self::merge_predicate_declaration_map(
                &mut providers.predicate_declarations,
                incoming.predicate_declarations,
            )?;
            Self::merge_inferred_predicate_contributions(
                &mut providers.inferred_predicate_contributions,
                incoming.inferred_predicate_contributions,
            );
            Self::merge_function_providers(&mut providers.functions, incoming.functions)?;
        }
        Ok(providers)
    }

    fn provider_for_program(&self, program: &Program) -> ImportProvider {
        if let Some(entry) = &self.entry {
            if entry.program.imports == program.imports {
                return ImportProvider {
                    module: entry.path.clone(),
                    source: self
                        .entry_source
                        .clone()
                        .unwrap_or_else(|| entry.source_file.clone()),
                };
            }
        }
        if let Some(root_source) = &self.root_module {
            if let Some(root) = self.loaded.get(root_source) {
                if root.program.imports == program.imports {
                    return ImportProvider {
                        module: root.path.clone(),
                        source: root_source.clone(),
                    };
                }
            }
        }
        ImportProvider {
            module: vec!["<program>".to_string()],
            source: PathBuf::from("<program>"),
        }
    }

    fn validate_program_against_imports(
        &self,
        program: &Program,
        imports: &[ResolvedImport],
    ) -> Result<(), ModuleError> {
        let program_provider = self.provider_for_program(program);
        let module_path = program_provider.module.clone();
        let mut providers = self.import_providers_from_imports(imports)?;

        for declaration in &program.domains {
            Self::record_domain_declaration(
                &mut providers.domain_declarations,
                declaration.name.clone(),
                ImportedDomainDeclaration {
                    module: module_path.clone(),
                    declaration: declaration.clone(),
                },
            )?;
        }
        for declaration in &program.predicates {
            let schema =
                Self::predicate_declaration_schema(&providers.domain_declarations, declaration);
            Self::record_predicate_declaration(
                &mut providers.predicate_declarations,
                declaration.name.clone(),
                ImportedPredicateDeclaration {
                    module: module_path.clone(),
                    schema,
                },
            )?;
        }
        let entry_contributions = Self::local_inferred_predicate_contributions(
            program,
            &HashSet::new(),
            program_provider,
        );
        Self::merge_inferred_predicate_contributions(
            &mut providers.inferred_predicate_contributions,
            entry_contributions,
        );
        for function in &program.functions {
            if let Some(imported) = providers.functions.get(&function.name) {
                return Err(ModuleError::ImportConflict {
                    name: function.name.clone(),
                    module1: imported.module.clone(),
                    module2: module_path,
                });
            }
        }

        Self::validate_inferred_predicate_contributions(&providers)?;

        Ok(())
    }

    fn import_scope_for_module(
        &self,
        source: &Path,
        module_path: &[String],
        imported_items: Option<&HashSet<String>>,
    ) -> Result<ImportScope, ModuleError> {
        let loaded_module = self
            .loaded
            .get(source)
            .ok_or_else(|| ModuleError::NotFound {
                path: module_path.to_vec(),
                searched: vec![],
            })?;
        let nested_imports = self.imports_for_source(source);
        let local_predicates = Self::local_predicate_names(&loaded_module.program);
        let local_functions = Self::local_function_names(&loaded_module.program);
        let local_hidden = Self::hidden_local_items(&loaded_module.program, imported_items);
        let mut scope = self.import_scope_from_imports(nested_imports)?;
        scope.visible.predicates.extend(
            local_predicates
                .difference(&local_hidden.predicates)
                .cloned(),
        );
        scope
            .visible
            .functions
            .extend(local_functions.difference(&local_hidden.functions).cloned());
        scope.hidden.predicates.extend(local_hidden.predicates);
        scope.hidden.functions.extend(local_hidden.functions);
        // A visible provider in this module's import closure satisfies the
        // name even when another provider keeps an item with that name hidden.
        scope
            .hidden
            .predicates
            .retain(|name| !scope.visible.predicates.contains(name));
        scope
            .hidden
            .functions
            .retain(|name| !scope.visible.functions.contains(name));
        Ok(scope)
    }

    fn import_scope_from_imports(
        &self,
        imports: &[ResolvedImport],
    ) -> Result<ImportScope, ModuleError> {
        let mut scope = ImportScope::default();
        for group in Self::combined_import_selections(imports) {
            let imported_scope = self.import_scope_for_module(
                &group.source,
                &group.module_path,
                group.imported_items.as_ref(),
            )?;
            scope
                .visible
                .predicates
                .extend(imported_scope.visible.predicates);
            scope
                .visible
                .functions
                .extend(imported_scope.visible.functions);
            scope
                .hidden
                .predicates
                .extend(imported_scope.hidden.predicates);
            scope
                .hidden
                .functions
                .extend(imported_scope.hidden.functions);
        }
        scope
            .hidden
            .predicates
            .retain(|name| !scope.visible.predicates.contains(name));
        scope
            .hidden
            .functions
            .retain(|name| !scope.visible.functions.contains(name));
        Ok(scope)
    }

    fn imports_for_source(&self, source: &Path) -> &[ResolvedImport] {
        self.resolved_imports
            .get(source)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    fn validate_visible_dependencies(
        program: &Program,
        imported_items: Option<&HashSet<String>>,
        imported_scope: &ImportScope,
        module_path: &[String],
    ) -> Result<(), ModuleError> {
        let local_predicates = Self::local_predicate_names(program);
        let local_functions = Self::local_function_names(program);
        let local_hidden = Self::hidden_local_items(program, imported_items);
        let mut hidden_predicates = local_hidden.predicates.clone();
        hidden_predicates.extend(
            imported_scope
                .hidden
                .predicates
                .difference(&local_predicates)
                .cloned(),
        );
        let mut hidden_functions = local_hidden.functions.clone();
        hidden_functions.extend(
            imported_scope
                .hidden
                .functions
                .difference(&local_functions)
                .cloned(),
        );

        for rule in &program.rules {
            if local_hidden.predicates.contains(&rule.head.predicate) {
                continue;
            }
            let mut predicate_dependencies = HashSet::new();
            let mut function_dependencies = HashSet::new();
            collect_body_dependencies(
                &rule.body,
                &mut predicate_dependencies,
                &mut function_dependencies,
            );
            let mut hidden_dependencies = predicate_dependencies
                .intersection(&hidden_predicates)
                .chain(function_dependencies.intersection(&hidden_functions))
                .cloned()
                .collect::<Vec<_>>();
            hidden_dependencies.sort();
            if let Some(dependency) = hidden_dependencies.into_iter().next() {
                return Err(ModuleError::HiddenDependency {
                    module: module_path.to_vec(),
                    export: rule.head.predicate.clone(),
                    dependency,
                });
            }
        }

        for function in &program.functions {
            if local_hidden.functions.contains(&function.name) {
                continue;
            }
            let mut predicate_dependencies = HashSet::new();
            let mut function_dependencies = HashSet::new();
            collect_function_body_dependencies(
                &function.body,
                &mut predicate_dependencies,
                &mut function_dependencies,
            );
            let mut hidden_dependencies = predicate_dependencies
                .intersection(&hidden_predicates)
                .chain(function_dependencies.intersection(&hidden_functions))
                .cloned()
                .collect::<Vec<_>>();
            hidden_dependencies.sort();
            if let Some(dependency) = hidden_dependencies.into_iter().next() {
                return Err(ModuleError::HiddenDependency {
                    module: module_path.to_vec(),
                    export: function.name.clone(),
                    dependency,
                });
            }
        }

        Ok(())
    }

    fn validate_supported_import_content(
        program: &Program,
        module_path: &[String],
    ) -> Result<(), ModuleError> {
        let Program {
            imports: _,
            functions: _,
            domains: _,
            predicates: _,
            rules: _,
            constraints,
            authored_constraint_source_bound: _,
            queries: _,
            prob_facts,
            annotated_disjunctions,
            evidence,
            prob_queries: _,
            neural_predicates,
            learnable_rules,
            directives: _,
        } = program;

        let mut constructs = Vec::new();
        if !prob_facts.is_empty() {
            constructs.push("probabilistic facts".to_string());
        }
        if !annotated_disjunctions.is_empty() {
            constructs.push("annotated disjunctions".to_string());
        }
        if !evidence.is_empty() {
            constructs.push("evidence statements".to_string());
        }
        if !constraints.is_empty() {
            constructs.push("integrity constraints".to_string());
        }
        if !neural_predicates.is_empty() {
            constructs.push("neural predicate declarations".to_string());
        }
        if !learnable_rules.is_empty() {
            constructs.push("learnable rule templates".to_string());
        }
        constructs.sort();

        if constructs.is_empty() {
            Ok(())
        } else {
            Err(ModuleError::UnsupportedImportedContent {
                module: module_path.to_vec(),
                constructs,
            })
        }
    }

    fn import_merge_key(source: &Path, imported_items: Option<&HashSet<String>>) -> ImportMergeKey {
        let imported_items = imported_items.map(|items| {
            let mut sorted = items.iter().cloned().collect::<Vec<_>>();
            sorted.sort();
            sorted
        });
        ImportMergeKey {
            source: source.to_path_buf(),
            imported_items,
        }
    }

    fn merge_import_group_with_report<F>(
        &self,
        program: &mut Program,
        imports: &[ResolvedImport],
        merged_imports: &mut HashSet<ImportMergeKey>,
        on_merge: &mut F,
    ) -> Result<(), ModuleError>
    where
        F: FnMut(&Path, &crate::ast::ProgramMergeReport),
    {
        for group in Self::combined_import_selections(imports) {
            let loaded_module =
                self.loaded
                    .get(&group.source)
                    .ok_or_else(|| ModuleError::NotFound {
                        path: group.module_path.clone(),
                        searched: vec![],
                    })?;
            let nested_imports = self.imports_for_source(&group.source);

            self.merge_import_group_with_report(program, nested_imports, merged_imports, on_merge)?;

            let imported_scope = self.import_scope_from_imports(nested_imports)?;
            Self::validate_supported_import_content(&loaded_module.program, &group.module_path)?;
            Self::validate_visible_dependencies(
                &loaded_module.program,
                group.imported_items.as_ref(),
                &imported_scope,
                &group.module_path,
            )?;
            let merge_key = Self::import_merge_key(&group.source, group.imported_items.as_ref());
            if merged_imports.insert(merge_key) {
                let report = program
                    .merge_from_with_report(&loaded_module.program, group.imported_items.as_ref());
                on_merge(&group.source, &report);
            }
        }
        Ok(())
    }

    fn merge_import_group(
        &self,
        program: &mut Program,
        imports: &[ResolvedImport],
        merged_imports: &mut HashSet<ImportMergeKey>,
    ) -> Result<(), ModuleError> {
        self.merge_import_group_with_report(program, imports, merged_imports, &mut |_, _| {})
    }

    /// Merge supported deterministic content from every resolved import.
    ///
    /// Resolution follows the importer-scoped edges recorded by the matching
    /// entry file or root module. Without that anchor, every logical path must
    /// identify one loaded source. Imported entry-only content, incomplete
    /// exports, incompatible participating declarations, conflicting inferred
    /// head-column types for undeclared predicate signatures, and conflicting
    /// function definitions are rejected rather than silently omitted. Selected
    /// public predicate clauses from separate import branches are then merged
    /// into one relation.
    ///
    /// # Arguments
    /// * `program` - The main program with imports to resolve
    ///
    /// # Returns
    /// The program with all imports merged in
    pub fn merge_imports(&self, mut program: Program) -> Result<Program, ModuleError> {
        let imports = self.resolved_imports_for_program(&program)?;
        self.validate_resolved_imports(&imports)?;
        self.validate_program_against_imports(&program, &imports)?;
        let entry_rules = std::mem::take(&mut program.rules);
        let mut merged_imports = HashSet::new();
        self.merge_import_group(&mut program, &imports, &mut merged_imports)?;
        program.rules.extend(entry_rules);

        Ok(program)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    #[cfg(target_os = "linux")]
    use std::os::fd::AsRawFd;
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
    fn test_load_entry_file_uses_supplied_path() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "helper", "helper_fact(1).");
        let entry = tmp.path().join("program.datalog");
        fs::write(&entry, "use helper.\nentry_fact(1).\n").unwrap();

        let mut resolver = ModuleResolver::new(vec![]);
        let loaded = resolver.load_entry_file(&entry).unwrap();

        assert_eq!(loaded.source_file, entry);
        assert_eq!(loaded.path, vec!["program"]);
        assert_eq!(resolver.entry().unwrap().source_file, entry);
        assert!(resolver.is_loaded("helper"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_load_entry_file_from_open_unlinked_proc_fd() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "helper", "helper_fact(1).");
        let entry_path = tmp.path().join("entry.xlog");
        fs::write(&entry_path, "use helper.\nentry_fact(1).\n").unwrap();
        let entry = fs::File::open(&entry_path).unwrap();
        let entry_proc_path = PathBuf::from(format!("/proc/self/fd/{}", entry.as_raw_fd()));
        assert_eq!(
            ModuleResolver::source_identity(&entry_proc_path).unwrap(),
            fs::canonicalize(&entry_path).unwrap()
        );
        fs::remove_file(&entry_path).unwrap();
        let duplicate = entry.try_clone().unwrap();
        let duplicate_proc_path = PathBuf::from(format!("/proc/self/fd/{}", duplicate.as_raw_fd()));

        assert!(fs::canonicalize(&entry_proc_path).is_err());
        assert_eq!(
            ModuleResolver::source_identity(&entry_proc_path).unwrap(),
            ModuleResolver::source_identity(&duplicate_proc_path).unwrap()
        );

        let mut resolver = ModuleResolver::new(vec![tmp.path().to_path_buf()]);
        let loaded = resolver.load_entry_file(&entry_proc_path).unwrap();

        assert_eq!(loaded.source_file, entry_proc_path);
        assert!(resolver.is_loaded("helper"));
    }

    #[test]
    fn test_load_entry_file_distinguishes_same_stem_import() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "main", "imported_fact(1).");
        let entry = tmp.path().join("main.datalog");
        let entry_source = "use main.\nentry_fact(1).\n";
        fs::write(&entry, entry_source).unwrap();

        let mut resolver = ModuleResolver::new(vec![]);
        let loaded = resolver.load_entry_file(&entry).unwrap();

        assert_eq!(loaded.source_file, entry);
        assert_eq!(
            resolver
                .get_module(&["main".into()])
                .expect("same-stem import should be loaded")
                .source_file,
            tmp.path().join("main.xlog")
        );
        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap();
        assert!(merged
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "imported_fact"));
    }

    #[test]
    fn test_module_not_found() {
        let tmp = TempDir::new().unwrap();
        let mut resolver = ModuleResolver::new(vec![]);

        let result = resolver.load_module(tmp.path(), &["nonexistent".into()]);
        assert!(matches!(result, Err(ModuleError::NotFound { .. })));
    }

    #[test]
    fn merge_imports_returns_not_found_for_unloaded_module() {
        let resolver = ModuleResolver::new(vec![]);
        let program = parse_program("use nonexistent.").unwrap();

        let result = resolver.merge_imports(program);

        assert!(matches!(
            result,
            Err(ModuleError::NotFound { path, searched })
                if path == vec!["nonexistent"] && searched.is_empty()
        ));
    }

    #[test]
    fn merge_imports_unions_compatible_predicates_from_separate_modules() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "first", "pred shared(u32). shared(1).");
        create_test_module(tmp.path(), "second", "pred shared(u32). shared(2).");
        let entry_source = "use first.\nuse second.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let (predicate_imports, function_imports) = resolver
            .validate_imports(&parse_program(entry_source).unwrap())
            .unwrap();
        assert_eq!(predicate_imports.get("shared"), Some(&vec!["first".into()]));
        assert!(function_imports.is_empty());

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap();
        let shared_facts = merged
            .rules
            .iter()
            .filter(|rule| rule.head.predicate == "shared" && rule.is_fact())
            .collect::<Vec<_>>();

        assert_eq!(shared_facts.len(), 2);
        assert!(shared_facts
            .iter()
            .any(|rule| rule.head.terms == vec![Term::Integer(1)]));
        assert!(shared_facts
            .iter()
            .any(|rule| rule.head.terms == vec![Term::Integer(2)]));
        assert_eq!(
            merged
                .predicates
                .iter()
                .filter(|declaration| declaration.name == "shared")
                .count(),
            1
        );
    }

    #[test]
    fn merge_imports_unions_compatible_undeclared_predicates() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "first", "shared(from_first).");
        create_test_module(tmp.path(), "second", "shared(from_second).");
        let entry_source = "use first.\nuse second.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap();
        let shared_facts = merged
            .rules
            .iter()
            .filter(|rule| rule.head.predicate == "shared" && rule.is_fact())
            .collect::<Vec<_>>();

        assert_eq!(shared_facts.len(), 2);
        assert!(shared_facts
            .iter()
            .any(|rule| rule.head.terms
                == vec![Term::Symbol(xlog_core::symbol::intern("from_first"))]));
        assert!(shared_facts
            .iter()
            .any(|rule| rule.head.terms
                == vec![Term::Symbol(xlog_core::symbol::intern("from_second"))]));
        assert!(!merged
            .predicates
            .iter()
            .any(|declaration| declaration.name == "shared"));
    }

    #[test]
    fn merge_import_validation_keeps_undeclared_predicate_arities_distinct() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "unary", "shared(1).");
        create_test_module(tmp.path(), "binary", "shared(one, two).");
        let entry_source = "use unary.\nuse binary.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap();
        let arities = merged
            .rules
            .iter()
            .filter(|rule| rule.head.predicate == "shared")
            .map(|rule| rule.head.arity())
            .collect::<Vec<_>>();

        assert_eq!(arities, vec![1, 2]);
    }

    #[test]
    fn merge_imports_rejects_incompatible_undeclared_predicate_schemas() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "first", "shared(row, 1).");
        create_test_module(tmp.path(), "second", "shared(row, from_second).");
        let entry_source = "use first.\nuse second.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let program = parse_program(entry_source).unwrap();
        let errors = [
            resolver
                .validate_imports(&program)
                .expect_err("validation must reject different inferred column types"),
            resolver
                .merge_imports(program)
                .expect_err("merge must reject different inferred column types"),
        ];

        for error in errors {
            let message = error.to_string();
            assert!(message.contains("error[E0412]"), "{message}");
            assert!(message.contains("shared/2"), "{message}");
            assert!(message.contains("column 2"), "{message}");
            assert!(message.contains("first"), "{message}");
            assert!(message.contains("second"), "{message}");
            assert!(message.contains("u32"), "{message}");
            assert!(message.contains("symbol"), "{message}");
        }
    }

    #[test]
    fn merge_imports_rejects_numeric_inference_conflicts_in_either_import_order() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "small", "shared(1).");
        create_test_module(tmp.path(), "wide", "shared(5000000000).");

        for (entry_name, entry_source) in [
            ("small_first", "use small.\nuse wide.\n"),
            ("wide_first", "use wide.\nuse small.\n"),
        ] {
            create_test_module(tmp.path(), entry_name, entry_source);
            let mut resolver = ModuleResolver::new(vec![]);
            resolver
                .load_module(tmp.path(), &[entry_name.into()])
                .unwrap();

            let error = resolver
                .merge_imports(parse_program(entry_source).unwrap())
                .expect_err("numeric inference must not depend on import order");
            let message = error.to_string();

            assert!(message.contains("error[E0412]"), "{message}");
            assert!(message.contains("shared/1"), "{message}");
            assert!(message.contains("module `small`"), "{message}");
            assert!(message.contains("module `wide`"), "{message}");
            assert!(message.contains("u32"), "{message}");
            assert!(message.contains("i64"), "{message}");
        }
    }

    #[test]
    fn merge_imports_uses_an_explicit_schema_for_undeclared_contributions() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "small", "shared(1).");
        create_test_module(tmp.path(), "wide", "shared(5000000000).");
        let entry_source = "use small.\nuse wide.\npred shared(i64).\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .expect("the explicit schema controls both imported facts");

        assert_eq!(
            merged
                .rules
                .iter()
                .filter(|rule| rule.head.predicate == "shared")
                .count(),
            2
        );
    }

    #[test]
    fn merge_imports_rejects_transitive_inferred_schema_conflicts() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "left_provider", "shared(1).");
        create_test_module(tmp.path(), "left_wrapper", "use left_provider.\n");
        create_test_module(tmp.path(), "right_provider", "shared(from_right).");
        create_test_module(tmp.path(), "right_wrapper", "use right_provider.\n");
        let entry_source = "use left_wrapper.\nuse right_wrapper.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let error = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .expect_err("transitive contributions must be schema-compatible");
        let message = error.to_string();

        assert!(message.contains("error[E0412]"), "{message}");
        assert!(message.contains("shared/1"), "{message}");
        assert!(message.contains("left_provider"), "{message}");
        assert!(message.contains("right_provider"), "{message}");
    }

    #[test]
    fn merge_imports_rejects_constant_head_rule_schema_conflicts() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "first",
            "first_source(ready).\nshared(1) :- first_source(ready).",
        );
        create_test_module(
            tmp.path(),
            "second",
            "second_source(ready).\nshared(from_second) :- second_source(ready).",
        );
        let entry_source = "use first.\nuse second.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let error = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .expect_err("known constant head types must agree across modules");

        assert!(error.to_string().contains("error[E0412]"));
    }

    #[test]
    fn merge_imports_rejects_body_inferred_rule_head_conflicts() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "first",
            "first_source(1).\nshared(X) :- first_source(X).",
        );
        create_test_module(
            tmp.path(),
            "second",
            "second_source(from_second).\nshared(X) :- second_source(X).",
        );
        let entry_source = "use first.\nuse second.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let error = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .expect_err("body-derived head types must agree across modules");

        let message = error.to_string();
        assert!(message.contains("error[E0412]"), "{message}");
        assert!(message.contains("shared/1"), "{message}");
        assert!(message.contains("u32"), "{message}");
        assert!(message.contains("symbol"), "{message}");
    }

    #[test]
    fn merge_imports_rejects_arithmetic_inferred_rule_head_conflicts() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "first", "shared(X) :- X is cast(1, u32).");
        create_test_module(tmp.path(), "second", "shared(X) :- X is cast(1, u64).");
        let entry_source = "use first.\nuse second.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let error = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .expect_err("arithmetic-derived head types must agree across modules");
        let message = error.to_string();

        assert!(message.contains("error[E0412]"), "{message}");
        assert!(message.contains("shared/1"), "{message}");
        assert!(message.contains("module `first`"), "{message}");
        assert!(message.contains("module `second`"), "{message}");
        assert!(message.contains("u32"), "{message}");
        assert!(message.contains("u64"), "{message}");
    }

    #[test]
    fn merge_imports_rejects_aggregate_inferred_rule_head_conflicts() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "first",
            "first_source(1).\nshared(min(X)) :- first_source(X).",
        );
        create_test_module(
            tmp.path(),
            "second",
            "second_source(1.0).\nshared(logsumexp(X)) :- second_source(X).",
        );
        let entry_source = "use first.\nuse second.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let error = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .expect_err("aggregate-derived head types must agree across modules");
        let message = error.to_string();

        assert!(message.contains("error[E0412]"), "{message}");
        assert!(message.contains("shared/1"), "{message}");
        assert!(message.contains("module `first`"), "{message}");
        assert!(message.contains("module `second`"), "{message}");
        assert!(message.contains("u32"), "{message}");
        assert!(message.contains("f64"), "{message}");
    }

    #[test]
    fn merge_imports_attributes_invalid_schema_inference_to_its_module() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "library",
            "shared(X) :- X is cast(1, u32) + cast(1, u64).",
        );
        let entry_source = "use library.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let error = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .expect_err("invalid arithmetic evidence must fail during schema inference");
        let message = error.to_string();

        assert!(message.contains("error[E0413]"), "{message}");
        assert!(message.contains("shared/1"), "{message}");
        assert!(message.contains("module `library`"), "{message}");
        assert!(message.contains("Type mismatch in arithmetic"), "{message}");
    }

    #[test]
    fn merge_imports_defers_user_function_schema_evidence_until_expansion() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "first",
            "func first_value(X) = cast(X, u32).\nshared(X) :- X is first_value(1).",
        );
        create_test_module(
            tmp.path(),
            "second",
            "func second_value(X) = cast(X, u64).\nshared(X) :- X is second_value(1).",
        );
        let entry_source = "use first.\nuse second.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .expect("user-defined calls are expanded after module resolution");

        assert_eq!(
            merged
                .rules
                .iter()
                .filter(|rule| rule.head.predicate == "shared")
                .count(),
            2
        );
    }

    #[test]
    fn merge_imports_retains_independent_schema_evidence_beside_user_functions() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "first",
            "func first_value(X) = cast(X, u32).\nshared(1, X) :- X is first_value(1).",
        );
        create_test_module(
            tmp.path(),
            "second",
            "func second_value(X) = cast(X, u32).\nshared(from_second, X) :- X is second_value(1).",
        );
        let entry_source = "use first.\nuse second.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let error = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .expect_err("a user function must not hide independent head-column evidence");
        let message = error.to_string();

        assert!(message.contains("error[E0412]"), "{message}");
        assert!(message.contains("shared/2"), "{message}");
        assert!(message.contains("module `first`"), "{message}");
        assert!(message.contains("module `second`"), "{message}");
        assert!(message.contains("u32"), "{message}");
        assert!(message.contains("symbol"), "{message}");
    }

    #[test]
    fn merge_imports_reports_invalid_builtin_evidence_beside_user_functions() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "library",
            "func value(X) = cast(X, u32).\nshared(X, Y) :- X is value(1), Y is cast(1, u32) + cast(1, u64).",
        );
        let entry_source = "use library.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let error = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .expect_err("a user function must not hide invalid built-in arithmetic evidence");
        let message = error.to_string();

        assert!(message.contains("error[E0413]"), "{message}");
        assert!(message.contains("shared/2"), "{message}");
        assert!(message.contains("module `library`"), "{message}");
        assert!(message.contains("Type mismatch in arithmetic"), "{message}");
    }

    #[test]
    fn merge_imports_reports_invalid_builtin_subexpressions_beside_user_functions() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "library",
            "func value(X) = cast(X, u32).\nshared(Y) :- Y is value(1) + (cast(1, u32) + cast(1, u64)).",
        );
        let entry_source = "use library.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let error = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .expect_err("an unknown user-function operand must not hide an invalid sibling");
        let message = error.to_string();

        assert!(message.contains("error[E0413]"), "{message}");
        assert!(message.contains("shared/1"), "{message}");
        assert!(message.contains("module `library`"), "{message}");
        assert!(message.contains("Type mismatch in arithmetic"), "{message}");
    }

    #[test]
    fn merge_imports_reports_invalid_user_function_argument_evidence() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "library",
            "func value(X) = cast(X, u32).\nshared(Y) :- Y is value(cast(1, u32) + cast(1, u64)).",
        );
        let entry_source = "use library.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let error = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .expect_err("invalid built-in arithmetic inside a user-function argument must fail");
        let message = error.to_string();

        assert!(message.contains("error[E0413]"), "{message}");
        assert!(message.contains("shared/1"), "{message}");
        assert!(message.contains("module `library`"), "{message}");
        assert!(message.contains("Type mismatch in arithmetic"), "{message}");
    }

    #[test]
    fn merge_imports_reports_invalid_partial_arithmetic_evidence() {
        let cases = [
            (
                "symbols(foo).\nshared(Y) :- symbols(S), Y is value(1) + S.",
                "Arithmetic requires numeric type",
            ),
            (
                "floats(1.0).\nshared(Y) :- floats(F), Y is value(1) % F.",
                "Modulo (%) not supported for floating point",
            ),
            (
                "symbols(foo).\nshared(Y) :- symbols(S), Y is value(1) % S.",
                "Modulo (%) requires integer operands",
            ),
        ];

        for (body, expected_detail) in cases {
            let tmp = TempDir::new().unwrap();
            create_test_module(
                tmp.path(),
                "library",
                &format!("func value(X) = cast(X, u32).\n{body}"),
            );
            let entry_source = "use library.\n";
            create_test_module(tmp.path(), "entry", entry_source);

            let mut resolver = ModuleResolver::new(vec![]);
            resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

            let error = resolver
                .merge_imports(parse_program(entry_source).unwrap())
                .expect_err("a deferred operand must not hide an invalid known operand");
            let message = error.to_string();

            assert!(message.contains("error[E0413]"), "{message}");
            assert!(message.contains("shared/1"), "{message}");
            assert!(message.contains("module `library`"), "{message}");
            assert!(message.contains(expected_detail), "{message}");
        }
    }

    #[test]
    fn merge_imports_propagates_signature_types_through_rule_chains() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "first",
            "shared(X) :- first_mid(X).\nfirst_mid(X) :- z_typed(X).\nz_typed(1).",
        );
        create_test_module(
            tmp.path(),
            "second",
            "shared(X) :- second_mid(X).\nsecond_mid(X) :- z_typed(prefix, X).\nz_typed(prefix, from_second).",
        );
        let entry_source = "use first.\nuse second.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let error = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .expect_err("transitive body-derived types must agree across modules");
        let message = error.to_string();

        assert!(message.contains("error[E0412]"), "{message}");
        assert!(message.contains("shared/1"), "{message}");
        assert!(message.contains("u32"), "{message}");
        assert!(message.contains("symbol"), "{message}");
    }

    #[test]
    fn merge_imports_leaves_unanchored_rule_head_variables_unconstrained() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "first",
            "first_cycle(X) :- first_cycle(X).\nshared(X) :- first_cycle(X).",
        );
        create_test_module(
            tmp.path(),
            "second",
            "second_cycle(X) :- second_cycle(X).\nshared(X) :- second_cycle(X).",
        );
        let entry_source = "use first.\nuse second.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .expect("unanchored variables do not supply concrete schema evidence");

        assert_eq!(
            merged
                .rules
                .iter()
                .filter(|rule| rule.head.predicate == "shared")
                .count(),
            2
        );
    }

    #[test]
    fn merge_imports_uses_column_schema_arity_for_programmatic_declarations() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "small", "shared(1).");
        create_test_module(tmp.path(), "wide", "shared(5000000000).");
        let entry_source = "use small.\nuse wide.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();
        let mut program = parse_program(entry_source).unwrap();
        program.predicates.push(PredDecl {
            name: "shared".to_string(),
            types: Vec::new(),
            columns: vec![crate::ast::PredColumn {
                name: Some("value".to_string()),
                typ: TypeRef::Scalar(ScalarType::I64),
            }],
            is_private: false,
        });

        let merged = resolver
            .merge_imports(program)
            .expect("the effective declaration schema must control imported facts");

        assert_eq!(
            merged
                .rules
                .iter()
                .filter(|rule| rule.head.predicate == "shared")
                .count(),
            2
        );
    }

    #[test]
    fn merge_imports_distinguishes_same_stem_entry_and_import_in_schema_errors() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "main", "shared(1).");
        let entry = tmp.path().join("main.datalog");
        fs::write(&entry, "use main.\nshared(from_entry).\n").unwrap();

        let mut resolver = ModuleResolver::new(vec![]);
        let program = resolver.load_entry_file(&entry).unwrap().program.clone();

        let error = resolver
            .merge_imports(program)
            .expect_err("same-stem sources with different schemas must be distinguishable");
        let message = error.to_string();

        assert!(message.contains("error[E0412]"), "{message}");
        assert!(message.contains("main.datalog"), "{message}");
        assert!(message.contains("main.xlog"), "{message}");
    }

    #[test]
    fn merge_imports_excludes_selectively_omitted_schema_conflicts() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "first", "shared(from_first).\nomitted(1).");
        create_test_module(
            tmp.path(),
            "second",
            "shared(from_second).\nomitted(from_second).",
        );
        let entry_source = "use first::{shared}.\nuse second::{shared}.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .expect("omitted predicates do not participate in schema validation");

        assert_eq!(
            merged
                .rules
                .iter()
                .filter(|rule| rule.head.predicate == "shared")
                .count(),
            2
        );
        assert!(!merged
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "omitted"));
    }

    #[test]
    fn merge_imports_rejects_selected_inferred_schema_conflicts() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "first", "shared(1).\nfirst_only(ok).");
        create_test_module(
            tmp.path(),
            "second",
            "shared(from_second).\nsecond_only(ok).",
        );
        let entry_source = "use first::{shared}.\nuse second::{shared}.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let error = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .expect_err("selected incompatible contributions must be rejected");

        assert!(error.to_string().contains("error[E0412]"));
    }

    #[test]
    fn merge_imports_excludes_private_schema_conflicts() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "first",
            "private pred hidden(u32).\nhidden(1).\nshared(from_first).",
        );
        create_test_module(
            tmp.path(),
            "second",
            "private pred hidden(symbol).\nhidden(from_second).\nshared(from_second).",
        );
        let entry_source = "use first.\nuse second.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .expect("private predicates do not participate in import validation");

        assert_eq!(
            merged
                .rules
                .iter()
                .filter(|rule| rule.head.predicate == "shared")
                .count(),
            2
        );
        assert!(!merged
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "hidden"));
    }

    #[test]
    fn merge_imports_validates_entry_clause_schema_against_imports() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "library", "shared(1).");
        let entry_source = "use library.\nshared(from_entry).\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let error = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .expect_err("entry and imported contributions must agree");
        let message = error.to_string();

        assert!(message.contains("error[E0412]"), "{message}");
        assert!(message.contains("library"), "{message}");
        assert!(message.contains("entry"), "{message}");
    }

    #[test]
    fn merge_imports_unions_selectively_imported_compatible_predicates() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "first",
            "pred shared(u32). pred first_only(u32). shared(1). first_only(10).",
        );
        create_test_module(
            tmp.path(),
            "second",
            "pred shared(u32). pred second_only(u32). shared(2). second_only(20).",
        );
        let entry_source = "use first::{shared}.\nuse second::{shared}.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap();
        let shared_facts = merged
            .rules
            .iter()
            .filter(|rule| rule.head.predicate == "shared" && rule.is_fact())
            .count();

        assert_eq!(shared_facts, 2);
        assert!(!merged.rules.iter().any(|rule| {
            rule.head.predicate == "first_only" || rule.head.predicate == "second_only"
        }));
        assert!(!merged.predicates.iter().any(|declaration| {
            declaration.name == "first_only" || declaration.name == "second_only"
        }));
    }

    #[test]
    fn merge_imports_unions_compatible_predicates_across_wrapper_branches() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "left_provider",
            concat!(
                "pred left_source(symbol).\n",
                "pred shared(symbol).\n",
                "left_source(left).\n",
                "shared(X) :- left_source(X).",
            ),
        );
        create_test_module(
            tmp.path(),
            "left_wrapper",
            "use left_provider.\npred left_result(symbol).\nleft_result(X) :- shared(X).",
        );
        create_test_module(
            tmp.path(),
            "right_provider",
            concat!(
                "pred right_source(symbol).\n",
                "pred shared(symbol).\n",
                "right_source(right).\n",
                "shared(X) :- right_source(X).",
            ),
        );
        create_test_module(
            tmp.path(),
            "right_wrapper",
            "use right_provider.\npred right_result(symbol).\nright_result(X) :- shared(X).",
        );
        let entry_source = "use left_wrapper.\nuse right_wrapper.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap();
        let shared_rules = merged
            .rules
            .iter()
            .filter(|rule| rule.head.predicate == "shared" && !rule.is_fact())
            .collect::<Vec<_>>();

        assert_eq!(shared_rules.len(), 2);
        assert!(merged
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "left_result"));
        assert!(merged
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "right_result"));
    }

    #[test]
    fn merge_imports_allows_local_extension_of_one_imported_provider() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "provider",
            "pred shared(symbol). shared(provider_value).",
        );
        create_test_module(
            tmp.path(),
            "wrapper",
            "use provider.\npred shared(symbol).\nshared(wrapper_value).",
        );
        let entry_source = "use wrapper.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap();
        let shared_facts = merged
            .rules
            .iter()
            .filter(|rule| rule.head.predicate == "shared" && rule.is_fact())
            .count();

        assert_eq!(shared_facts, 2);
    }

    #[test]
    fn merge_imports_rejects_local_redefinition_of_an_imported_function() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "base", "func shared(X) = X + 1.");
        create_test_module(tmp.path(), "wrapper", "use base.\nfunc shared(X) = X + 2.");
        let entry_source = "use wrapper.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let result = resolver.merge_imports(parse_program(entry_source).unwrap());

        assert!(matches!(
            result,
            Err(ModuleError::ImportConflict {
                name,
                module1,
                module2,
            }) if name == "shared"
                && module1 == vec!["base"]
                && module2 == vec!["wrapper"]
        ));
    }

    #[test]
    fn merge_imports_rejects_function_definitions_in_separate_branches() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "first", "func shared(X) = X + 1.");
        create_test_module(tmp.path(), "second", "func shared(X) = X + 2.");
        let entry_source = "use first.\nuse second.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let result = resolver.merge_imports(parse_program(entry_source).unwrap());

        assert!(matches!(
            result,
            Err(ModuleError::ImportConflict {
                name,
                module1,
                module2,
            }) if name == "shared"
                && module1 == vec!["first"]
                && module2 == vec!["second"]
        ));
    }

    #[test]
    fn merge_imports_rejects_function_definitions_across_wrapper_branches() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "left_provider", "func shared(X) = X + 1.");
        create_test_module(tmp.path(), "left_wrapper", "use left_provider.\n");
        create_test_module(tmp.path(), "right_provider", "func shared(X) = X + 2.");
        create_test_module(tmp.path(), "right_wrapper", "use right_provider.\n");
        let entry_source = "use left_wrapper.\nuse right_wrapper.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let result = resolver.merge_imports(parse_program(entry_source).unwrap());

        assert!(matches!(
            result,
            Err(ModuleError::ImportConflict {
                name,
                module1,
                module2,
            }) if name == "shared"
                && module1 == vec!["left_provider"]
                && module2 == vec!["right_provider"]
        ));
    }

    #[test]
    fn merge_imports_rejects_entry_redefinition_of_an_imported_function() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "library", "func shared(X) = X + 1.");
        let entry_source = "use library.\nfunc shared(X) = X + 2.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let result = resolver.merge_imports(parse_program(entry_source).unwrap());

        assert!(matches!(
            result,
            Err(ModuleError::ImportConflict {
                name,
                module1,
                module2,
            }) if name == "shared"
                && module1 == vec!["library"]
                && module2 == vec!["entry"]
        ));
    }

    #[test]
    fn merge_imports_does_not_treat_declarations_as_providers() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "first",
            "pred external(symbol).\npred first_result(symbol).\nfirst_result(X) :- external(X).",
        );
        create_test_module(
            tmp.path(),
            "second",
            "pred external(symbol).\npred second_result(symbol).\nsecond_result(X) :- external(X).",
        );
        let entry_source = "use first.\nuse second.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap();

        assert!(merged
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "first_result"));
        assert!(merged
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "second_result"));
    }

    #[test]
    fn merge_imports_rejects_incompatible_predicate_declarations() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "first", "pred external(u32). external(1).");
        create_test_module(
            tmp.path(),
            "second",
            "pred external(symbol). external(value).",
        );
        let entry_source = "use first.\nuse second.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let result = resolver.merge_imports(parse_program(entry_source).unwrap());

        assert!(matches!(
            result,
            Err(ModuleError::IncompatiblePredicateDeclaration {
                name,
                module1,
                module2,
            }) if name == "external"
                && module1 == vec!["first"]
                && module2 == vec!["second"]
        ));
    }

    #[test]
    fn merge_imports_rejects_incompatible_declaration_only_modules() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "first", "pred external(u32).");
        create_test_module(tmp.path(), "second", "pred external(symbol).");
        let entry_source = "use first.\nuse second.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let result = resolver.merge_imports(parse_program(entry_source).unwrap());

        assert!(matches!(
            result,
            Err(ModuleError::IncompatiblePredicateDeclaration {
                name,
                module1,
                module2,
            }) if name == "external"
                && module1 == vec!["first"]
                && module2 == vec!["second"]
        ));
    }

    #[test]
    fn merge_imports_rejects_entry_declaration_incompatible_with_import() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "library",
            "pred external(symbol). external(value).",
        );
        let entry_source = "use library.\npred external(u32).\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let result = resolver.merge_imports(parse_program(entry_source).unwrap());

        assert!(matches!(
            result,
            Err(ModuleError::IncompatiblePredicateDeclaration {
                name,
                module1,
                module2,
            }) if name == "external"
                && module1 == vec!["library"]
                && module2 == vec!["entry"]
        ));
    }

    #[test]
    fn merge_imports_rejects_incompatible_domain_declarations() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "first",
            "domain key : u32.\npred external(key).",
        );
        create_test_module(
            tmp.path(),
            "second",
            "domain key : symbol.\npred external(key).",
        );
        let entry_source = "use first.\nuse second.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let result = resolver.merge_imports(parse_program(entry_source).unwrap());

        assert!(matches!(
            result,
            Err(ModuleError::IncompatibleDomainDeclaration {
                name,
                module1,
                module2,
            }) if name == "key"
                && module1 == vec!["first"]
                && module2 == vec!["second"]
        ));
    }

    #[test]
    fn merge_imports_rejects_entry_domain_incompatible_with_import() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "library", "domain key : symbol.");
        let entry_source = "use library.\ndomain key : u32.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let result = resolver.merge_imports(parse_program(entry_source).unwrap());

        assert!(matches!(
            result,
            Err(ModuleError::IncompatibleDomainDeclaration {
                name,
                module1,
                module2,
            }) if name == "key"
                && module1 == vec!["library"]
                && module2 == vec!["entry"]
        ));
    }

    #[test]
    fn merge_imports_rejects_duplicate_functions_within_an_imported_module() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "library",
            "func shared(X) = X + 1.\nfunc shared(X) = X + 2.",
        );
        let entry_source = "use library.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let result = resolver.merge_imports(parse_program(entry_source).unwrap());

        assert!(matches!(
            result,
            Err(ModuleError::DuplicateImportedFunction { name, module })
                if name == "shared" && module == vec!["library"]
        ));
    }

    #[test]
    fn merge_imports_rejects_unselected_duplicate_public_functions() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "library",
            "func shared(X) = X + 1.\nfunc shared(X) = X + 2.\nfunc other(X) = X.",
        );
        let entry_source = "use library::{other}.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let result = resolver.merge_imports(parse_program(entry_source).unwrap());

        assert!(matches!(
            result,
            Err(ModuleError::DuplicateImportedFunction { name, module })
                if name == "shared" && module == vec!["library"]
        ));
    }

    #[test]
    fn merge_imports_rejects_private_and_public_definitions_of_an_exported_function() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "library",
            "private func shared(X) = X + 1.\nfunc shared(X) = X + 2.",
        );
        let entry_source = "use library.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let result = resolver.merge_imports(parse_program(entry_source).unwrap());

        assert!(matches!(
            result,
            Err(ModuleError::DuplicateImportedFunction { name, module })
                if name == "shared" && module == vec!["library"]
        ));
    }

    #[test]
    fn validation_rejects_mixed_visibility_for_one_imported_predicate() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "library",
            "private pred shared(u32).\npred shared(u32).\nshared(1).",
        );
        let entry_source = "use library.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();
        let program = parse_program(entry_source).unwrap();

        assert!(matches!(
            resolver.check_import(&["library".to_string()], "shared"),
            Err(ModuleError::ConflictingPredicateVisibility { name, module })
                if name == "shared" && module == vec!["library"]
        ));
        assert!(matches!(
            resolver.validate_imports(&program),
            Err(ModuleError::ConflictingPredicateVisibility { name, module })
                if name == "shared" && module == vec!["library"]
        ));
        assert!(matches!(
            resolver.merge_imports(program),
            Err(ModuleError::ConflictingPredicateVisibility { name, module })
                if name == "shared" && module == vec!["library"]
        ));
    }

    #[test]
    fn merge_imports_accepts_equivalent_predicate_schemas_using_distinct_domains() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "first",
            "domain first_key : u32.\npred external(first_key).",
        );
        create_test_module(
            tmp.path(),
            "second",
            "domain second_key : u32.\npred external(second_key).",
        );
        let entry_source = "use first.\nuse second.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap();

        assert!(merged
            .predicates
            .iter()
            .any(|declaration| declaration.name == "external"));
    }

    #[test]
    fn merge_imports_normalizes_a_predicate_schema_with_an_imported_domain() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "types", "domain key : u32.");
        create_test_module(tmp.path(), "wrapper", "use types.\npred external(key).");
        create_test_module(tmp.path(), "second", "pred external(u32).");
        let entry_source = "use wrapper.\nuse second.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap();

        assert!(merged
            .predicates
            .iter()
            .any(|declaration| declaration.name == "external"));
    }

    #[test]
    fn merge_imports_normalizes_an_entry_schema_with_an_imported_domain() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "types", "domain key : u32.");
        create_test_module(tmp.path(), "provider", "pred external(u32).");
        let entry_source = "use types.\nuse provider.\npred external(key).\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap();

        assert!(merged
            .predicates
            .iter()
            .any(|declaration| declaration.name == "external"));
    }

    #[test]
    fn merge_imports_allows_entry_to_extend_an_imported_predicate() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "library", "pred shared(u32). shared(1).");
        let entry_source = "use library.\npred shared(u32).\nshared(2).\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap();
        let shared_facts = merged
            .rules
            .iter()
            .filter(|rule| rule.head.predicate == "shared" && rule.is_fact())
            .count();

        assert_eq!(shared_facts, 2);
    }

    #[test]
    fn merge_imports_resolves_same_relative_name_per_importer() {
        let tmp = TempDir::new().unwrap();
        let left = tmp.path().join("left");
        let right = tmp.path().join("right");
        fs::create_dir_all(&left).unwrap();
        fs::create_dir_all(&right).unwrap();
        create_test_module(
            &left,
            "support",
            "pred left_local(symbol). left_local(left).",
        );
        create_test_module(
            &left,
            "wrapper",
            "use support.\npred left_result(symbol).\nleft_result(X) :- left_local(X).",
        );
        create_test_module(
            &right,
            "support",
            "pred right_local(symbol). right_local(right).",
        );
        create_test_module(
            &right,
            "wrapper",
            "use support.\npred right_result(symbol).\nright_result(X) :- right_local(X).",
        );
        let entry_source = "use left/wrapper.\nuse right/wrapper.\n";
        let entry = create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_entry_file(&entry).unwrap();

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap();

        assert!(merged
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "left_local"));
        assert!(merged
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "right_local"));
    }

    #[test]
    fn merge_imports_deduplicates_aliases_of_one_source_file() {
        let tmp = TempDir::new().unwrap();
        let util = tmp.path().join("util");
        fs::create_dir_all(&util).unwrap();
        create_test_module(
            &util,
            "helpers",
            "pred shared(symbol). shared(helper_value).",
        );
        create_test_module(
            &util,
            "wrapper",
            "use helpers.\npred wrapped(symbol).\nwrapped(X) :- shared(X).",
        );
        let entry_source = "use util/wrapper.\nuse util/helpers.\n";
        let entry = create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_entry_file(&entry).unwrap();

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap();
        let shared_facts = merged
            .rules
            .iter()
            .filter(|rule| rule.head.predicate == "shared" && rule.is_fact())
            .count();

        assert_eq!(shared_facts, 1);
        assert!(merged
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "wrapped"));
    }

    #[cfg(unix)]
    #[test]
    fn merge_imports_resolves_symlinked_module_dependencies_from_canonical_source() {
        let tmp = TempDir::new().unwrap();
        let real = tmp.path().join("real");
        let alias = tmp.path().join("alias");
        fs::create_dir_all(&real).unwrap();
        fs::create_dir_all(&alias).unwrap();
        create_test_module(
            &real,
            "support",
            "pred support_value(u32). support_value(1).",
        );
        create_test_module(
            &alias,
            "support",
            "pred support_value(u32). support_value(2).",
        );
        let shared = create_test_module(
            &real,
            "shared",
            "use support.\npred shared_value(symbol).\nshared_value(X) :- support_value(X).",
        );
        std::os::unix::fs::symlink(&shared, alias.join("shared.xlog")).unwrap();
        let entry_source = "use alias/shared.\n";
        let entry = create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_entry_file(&entry).unwrap();

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap();
        let support_values = merged
            .rules
            .iter()
            .filter(|rule| rule.head.predicate == "support_value" && rule.is_fact())
            .map(|rule| rule.head.terms.clone())
            .collect::<Vec<_>>();

        assert_eq!(support_values, vec![vec![Term::Integer(1)]]);
    }

    #[test]
    fn merge_imports_keeps_importer_local_resolution_across_search_paths() {
        let tmp = TempDir::new().unwrap();
        let entry_dir = tmp.path().join("entry");
        let module_dir = tmp.path().join("modules");
        fs::create_dir_all(&entry_dir).unwrap();
        fs::create_dir_all(&module_dir).unwrap();
        create_test_module(
            &entry_dir,
            "support",
            "pred entry_support(symbol). entry_support(local).",
        );
        create_test_module(
            &module_dir,
            "support",
            "pred wrapper_support(symbol). wrapper_support(search_path).",
        );
        create_test_module(
            &module_dir,
            "wrapper",
            "use support.\npred wrapped(symbol).\nwrapped(X) :- wrapper_support(X).",
        );
        let entry_source = "use support.\nuse wrapper.\n";
        let entry = create_test_module(&entry_dir, "main", entry_source);

        let mut resolver = ModuleResolver::new(vec![module_dir]);
        resolver.load_entry_file(&entry).unwrap();

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap();

        for predicate in ["entry_support", "wrapper_support", "wrapped"] {
            assert!(
                merged
                    .rules
                    .iter()
                    .any(|rule| rule.head.predicate == predicate),
                "missing rules for {predicate}"
            );
        }
    }

    #[test]
    fn validate_imports_rejects_an_unanchored_ambiguous_module_path() {
        let tmp = TempDir::new().unwrap();
        let left = tmp.path().join("left");
        let right = tmp.path().join("right");
        fs::create_dir_all(&left).unwrap();
        fs::create_dir_all(&right).unwrap();
        create_test_module(&left, "support", "pred left(symbol). left(value).");
        create_test_module(&right, "support", "pred right(symbol). right(value).");

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(&left, &["support".into()]).unwrap();
        resolver.load_module(&right, &["support".into()]).unwrap();

        let first = resolver
            .get_module(&["support".into()])
            .expect("first source remains available through the compatibility API");
        assert_eq!(first.source_file, left.join("support.xlog"));
        assert!(resolver.check_import(&["support".into()], "left").is_ok());
        assert!(resolver.check_import(&["support".into()], "right").is_err());
        assert_eq!(
            resolver
                .loaded_modules()
                .into_iter()
                .filter(|alias| *alias == "support")
                .count(),
            1
        );

        let result = resolver.validate_imports(&parse_program("use support.").unwrap());

        assert!(matches!(
            result,
            Err(ModuleError::AmbiguousModulePath { path, candidates })
                if path == vec!["support"] && candidates.len() == 2
        ));
    }

    #[test]
    fn load_entry_file_detects_an_import_of_itself() {
        let tmp = TempDir::new().unwrap();
        let entry = create_test_module(tmp.path(), "entry", "use entry.");
        let mut resolver = ModuleResolver::new(vec![]);

        let result = resolver.load_entry_file(&entry);

        assert!(matches!(result, Err(ModuleError::CircularImport { .. })));
    }

    #[cfg(unix)]
    #[test]
    fn load_entry_file_detects_a_symlink_import_of_itself() {
        let tmp = TempDir::new().unwrap();
        let entry = create_test_module(tmp.path(), "entry", "use alias.");
        std::os::unix::fs::symlink(&entry, tmp.path().join("alias.xlog")).unwrap();
        let mut resolver = ModuleResolver::new(vec![]);

        let result = resolver.load_entry_file(&entry);

        assert!(matches!(result, Err(ModuleError::CircularImport { .. })));
    }

    #[test]
    fn merge_imports_deduplicates_function_aliases_of_one_source_file() {
        let tmp = TempDir::new().unwrap();
        let util = tmp.path().join("util");
        fs::create_dir_all(&util).unwrap();
        create_test_module(&util, "helpers", "func normalize(X) = X + 1.");
        create_test_module(&util, "wrapper", "use helpers.\n");
        let entry_source = "use util/wrapper.\nuse util/helpers.\n";
        let entry = create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_entry_file(&entry).unwrap();

        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap();

        assert_eq!(
            merged
                .functions
                .iter()
                .filter(|function| function.name == "normalize")
                .count(),
            1
        );
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
        create_test_module(
            tmp.path(),
            "math",
            r#"
            pred add(u32, u32, u32).
            add(1, 2, 3).
        "#,
        );

        let mut resolver = ModuleResolver::new(vec![]);
        let result = resolver.load_module(tmp.path(), &["math".into()]);
        assert!(result.is_ok());
        let module = result.unwrap();
        assert!(module.exports.contains("add"));
    }

    #[test]
    fn test_private_not_exported() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "graph",
            r#"
            pred edge(u32, u32).
            private pred helper(u32).
            edge(1, 2).
            helper(1).
        "#,
        );

        let mut resolver = ModuleResolver::new(vec![]);
        let result = resolver.load_module(tmp.path(), &["graph".into()]);
        assert!(result.is_ok());
        let module = result.unwrap();
        assert!(module.exports.contains("edge"));
        assert!(!module.exports.contains("helper"));
    }

    #[test]
    fn test_merge_rejects_export_with_private_support() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "library",
            r#"
            private pred hidden(u32).
            pred visible(u32).
            hidden(1).
            visible(X) :- hidden(X).
        "#,
        );
        let entry_source = "use library.\nquery(visible(1)).\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();
        let error = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap_err()
            .to_string();

        assert!(error.contains("error[E0406]"), "{error}");
        assert!(error.contains("`visible`"), "{error}");
        assert!(error.contains("`hidden`"), "{error}");
        assert!(error.contains("`library`"), "{error}");
    }

    #[test]
    fn test_merge_rejects_export_with_selectively_omitted_support() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "library",
            r#"
            pred support(u32).
            pred visible(u32).
            support(1).
            visible(X) :- support(X).
        "#,
        );
        let entry_source = "use library::{visible}.\nquery(visible(1)).\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();
        let error = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap_err()
            .to_string();

        assert!(error.contains("error[E0406]"), "{error}");
        assert!(error.contains("`visible`"), "{error}");
        assert!(error.contains("`support`"), "{error}");
        assert!(error.contains("`library`"), "{error}");
    }

    #[test]
    fn test_merge_combines_separate_selective_imports() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "library",
            r#"
            pred support(u32).
            pred visible(u32).
            support(1).
            visible(X) :- support(X).
        "#,
        );
        let entry_source = "use library::{visible}.\nuse library::{support}.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();
        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap();

        assert!(merged
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "visible"));
        assert!(merged
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "support"));
    }

    #[test]
    fn test_merge_accepts_visible_provider_for_same_named_private_item() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "private_provider",
            "private pred hidden(u32).\nhidden(1).\n",
        );
        create_test_module(
            tmp.path(),
            "public_provider",
            "pred hidden(u32).\nhidden(2).\n",
        );
        create_test_module(
            tmp.path(),
            "wrapper",
            "use private_provider.\nuse public_provider.\npred visible(u32).\nvisible(X) :- hidden(X).\n",
        );
        let entry_source = "use wrapper.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();
        let merged = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap();

        assert!(merged
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "visible"));
        assert!(merged.rules.iter().any(|rule| {
            rule.head.predicate == "hidden" && rule.head.terms == vec![Term::Integer(2)]
        }));
    }

    #[test]
    fn test_unrelated_import_does_not_satisfy_selective_dependency() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "catalog",
            "pred support(u32).\npred marker(u32).\nsupport(1).\nmarker(1).\n",
        );
        create_test_module(
            tmp.path(),
            "wrapper_a",
            "use catalog::{marker}.\npred visible(u32).\nvisible(X) :- support(X).\n",
        );
        create_test_module(tmp.path(), "wrapper_b", "use catalog::{support}.\n");
        let entry_source = "use wrapper_a.\nuse wrapper_b.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();
        let error = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap_err()
            .to_string();

        assert!(error.contains("error[E0406]"), "{error}");
        assert!(error.contains("`visible`"), "{error}");
        assert!(error.contains("`support`"), "{error}");
        assert!(error.contains("`wrapper_a`"), "{error}");
    }

    #[test]
    fn test_merge_rejects_unknown_selected_item() {
        let tmp = TempDir::new().unwrap();
        create_test_module(tmp.path(), "library", "known(1).\n");
        let entry_source = "use library::{missing}.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();
        let error = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap_err()
            .to_string();

        assert!(error.contains("error[E0404]"), "{error}");
        assert!(error.contains("`missing`"), "{error}");
        assert!(error.contains("module library"), "{error}");
    }

    #[test]
    fn test_merge_rejects_transitive_private_support() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "base",
            "private pred hidden(u32).\nhidden(1).\n",
        );
        create_test_module(
            tmp.path(),
            "wrapper",
            "use base.\npred visible(u32).\nvisible(X) :- hidden(X).\n",
        );
        let entry_source = "use wrapper.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();
        let error = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap_err()
            .to_string();

        assert!(error.contains("error[E0406]"), "{error}");
        assert!(error.contains("`visible`"), "{error}");
        assert!(error.contains("`hidden`"), "{error}");
        assert!(error.contains("`wrapper`"), "{error}");
    }

    #[test]
    fn test_merge_rejects_exported_function_with_private_support() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "library",
            r#"
            private func hidden(X) = X + 1.
            func visible(X) = hidden(X) * 2.
        "#,
        );
        let entry_source = "use library.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();
        let error = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap_err()
            .to_string();

        assert!(error.contains("error[E0406]"), "{error}");
        assert!(error.contains("`visible`"), "{error}");
        assert!(error.contains("`hidden`"), "{error}");
        assert!(error.contains("`library`"), "{error}");
    }

    #[test]
    fn test_merge_rejects_private_safe_meta_dependencies() {
        let tmp = TempDir::new().unwrap();
        for (export, rule) in [
            ("all_hidden", "all_hidden() :- maplist(hidden, [1])."),
            (
                "collect_hidden",
                "collect_hidden(Values) :- findall(X, hidden(X), Values).",
            ),
        ] {
            create_test_module(
                tmp.path(),
                "library",
                &format!("private pred hidden(u32).\nhidden(1).\n{rule}\n"),
            );
            let entry_source = "use library.\n";
            create_test_module(tmp.path(), "entry", entry_source);

            let mut resolver = ModuleResolver::new(vec![]);
            resolver.load_module(tmp.path(), &["entry".into()]).unwrap();
            let error = resolver
                .merge_imports(parse_program(entry_source).unwrap())
                .unwrap_err()
                .to_string();

            assert!(error.contains("error[E0406]"), "{error}");
            assert!(error.contains(&format!("`{export}`")), "{error}");
            assert!(error.contains("`hidden`"), "{error}");
        }
    }

    #[test]
    fn test_merge_rejects_imported_program_level_constructs() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "library",
            r#"
            pred coin(symbol).
            pred left(symbol).
            pred right(symbol).
            0.5::coin(heads).
            0.4::left(choice); 0.6::right(choice).
            evidence(coin(heads), true).
            :- coin(heads), not left(choice).
            nn(classifier, [X], Y, [yes, no]) :: neural_label(X, Y).
            learnable(W) :: learned(X) :- source(X).
        "#,
        );
        let entry_source = "use library.\n";
        create_test_module(tmp.path(), "entry", entry_source);

        let mut resolver = ModuleResolver::new(vec![]);
        resolver.load_module(tmp.path(), &["entry".into()]).unwrap();
        let error = resolver
            .merge_imports(parse_program(entry_source).unwrap())
            .unwrap_err()
            .to_string();

        assert!(error.contains("error[E0405]"), "{error}");
        assert!(error.contains("`library`"), "{error}");
        assert!(
            error.contains(
                "annotated disjunctions, evidence statements, integrity constraints, learnable rule templates, neural predicate declarations, probabilistic facts"
            ),
            "{error}"
        );
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

    #[test]
    fn test_function_exports() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "mathfuncs",
            r#"
            func square(X) = X * X.
            func cube(X) = X * X * X.
            private func helper(X) = X.
        "#,
        );

        let mut resolver = ModuleResolver::new(vec![]);
        let result = resolver.load_module(tmp.path(), &["mathfuncs".into()]);
        assert!(result.is_ok());
        let module = result.unwrap();

        // Public functions should be exported
        assert!(module.function_exports.contains("square"));
        assert!(module.function_exports.contains("cube"));

        // Private function should not be exported
        assert!(!module.function_exports.contains("helper"));
    }

    #[test]
    fn test_mixed_exports() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "mixed",
            r#"
            pred value(i64).
            value(42).
            func double(X) = X * 2.
        "#,
        );

        let mut resolver = ModuleResolver::new(vec![]);
        let result = resolver.load_module(tmp.path(), &["mixed".into()]);
        assert!(result.is_ok());
        let module = result.unwrap();

        // Both predicate and function exports should be present
        assert!(module.exports.contains("value"));
        assert!(module.function_exports.contains("double"));
    }

    #[test]
    fn test_ignored_import_pragmas_lists_imported_module_pragmas_only() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "entry",
            r#"
            #pragma magic_sets = auto
            use lib.
            result(1).
        "#,
        );
        create_test_module(
            tmp.path(),
            "lib",
            r#"
            #pragma magic_sets = auto
            #pragma prob_seed = 7
            helper(1).
        "#,
        );

        let mut resolver = ModuleResolver::new(vec![]);
        resolver
            .load_module(tmp.path(), &["entry".into()])
            .expect("load entry");
        resolver.mark_entry_module("entry");

        // The entry file's own pragma is authoritative and excluded; the
        // imported module's pragmas are listed sorted by module then name.
        let ignored = resolver.ignored_import_pragmas();
        assert_eq!(
            ignored,
            vec![
                IgnoredImportPragma {
                    module: "lib".to_string(),
                    pragma: "magic_sets",
                },
                IgnoredImportPragma {
                    module: "lib".to_string(),
                    pragma: "prob_seed",
                },
            ]
        );

        // Verbatim: the rendered warning format is part of the contract.
        assert_eq!(
            ignored[0].to_string(),
            "warning[W0510]: `#pragma magic_sets` in imported module `lib` is ignored\n  \
             = note: pragmas apply only when declared in the entry file"
        );
    }

    #[test]
    fn test_ignored_import_pragmas_empty_without_module_pragmas() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "entry",
            r#"
            #pragma magic_sets = on
            use quiet.
            result(1).
        "#,
        );
        create_test_module(tmp.path(), "quiet", "helper(1).");

        let mut resolver = ModuleResolver::new(vec![]);
        resolver
            .load_module(tmp.path(), &["entry".into()])
            .expect("load entry");
        resolver.mark_entry_module("entry");

        assert!(resolver.ignored_import_pragmas().is_empty());
    }

    #[test]
    fn test_ignored_import_pragmas_sorted_across_modules() {
        let tmp = TempDir::new().unwrap();
        create_test_module(
            tmp.path(),
            "entry",
            r#"
            use zeta.
            use alpha.
            result(1).
        "#,
        );
        create_test_module(
            tmp.path(),
            "zeta",
            r#"
            #pragma prob_seed = 3
            z(1).
        "#,
        );
        create_test_module(
            tmp.path(),
            "alpha",
            r#"
            #pragma magic_sets = off
            a(1).
        "#,
        );

        let mut resolver = ModuleResolver::new(vec![]);
        resolver
            .load_module(tmp.path(), &["entry".into()])
            .expect("load entry");
        resolver.mark_entry_module("entry");

        // Deterministic cross-module order: sorted by module path first.
        assert_eq!(
            resolver.ignored_import_pragmas(),
            vec![
                IgnoredImportPragma {
                    module: "alpha".to_string(),
                    pragma: "magic_sets",
                },
                IgnoredImportPragma {
                    module: "zeta".to_string(),
                    pragma: "prob_seed",
                },
            ]
        );
    }

    #[test]
    fn test_ignored_import_pragmas_dedups_two_spellings_of_one_file() {
        let tmp = TempDir::new().unwrap();
        let util = tmp.path().join("util");
        fs::create_dir_all(&util).unwrap();
        create_test_module(
            tmp.path(),
            "entry",
            r#"
            use util/b.
            use util/helpers.
            result(1).
        "#,
        );
        create_test_module(
            tmp.path(),
            "util/b",
            r#"
            use helpers.
            b_pred(1).
        "#,
        );
        create_test_module(
            tmp.path(),
            "util/helpers",
            r#"
            #pragma magic_sets = auto
            helper(1).
        "#,
        );

        let mut resolver = ModuleResolver::new(vec![]);
        resolver
            .load_module(tmp.path(), &["entry".into()])
            .expect("load entry");
        resolver.mark_entry_module("entry");

        // The nested `use helpers.` (resolved relative to util/) loads the
        // same file under a second path key; one file must warn once, under
        // the alphabetically-first label.
        assert_eq!(
            resolver.ignored_import_pragmas(),
            vec![IgnoredImportPragma {
                module: "helpers".to_string(),
                pragma: "magic_sets",
            }]
        );
    }
}
