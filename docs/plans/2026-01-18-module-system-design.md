# Module System Design (v0.3.2)

> **Status:** Approved
> **Author:** Claude + Human
> **Date:** 2026-01-18
> **Target:** v0.3.2 (Language Core)

---

## Overview

This document specifies the module system for XLOG, enabling organization of large programs into reusable, encapsulated units. The design prioritizes simplicity, explicit dependencies, and clear error messages.

---

## Core Concepts

### File-to-Module Mapping

Every `.xlog` file is a module. The module name is derived from the file path relative to the project root or search path:

```
src/
  main.xlog          → module: main
  graph.xlog         → module: graph
  utils/
    math.xlog        → module: utils/math
    string.xlog      → module: utils/string
```

No explicit `module` declaration is needed. The filesystem is the module hierarchy.

### Imports

All external predicates must be explicitly imported before use:

```prolog
% main.xlog
use graph.                      % import all public predicates from graph
use utils/math::{abs, clamp}.   % import specific predicates from utils/math

reachable(X, Y) :- edge(X, Y).  % edge came from graph
clamped(X, Y) :- val(X, V), Y is clamp(V, 0, 100).
```

Import forms:
- `use module.` — import all public predicates
- `use module::{pred1, pred2}.` — import specific predicates

### Visibility

Predicates are public by default. Use `private` to hide implementation details:

```prolog
% graph.xlog
pred edge(u32, u32).           % public
pred reach(u32, u32).          % public
private pred helper(u32).       % not importable

reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).
private aux(X) :- helper(X).   % private rule
```

---

## File Resolution

### Search Order

When resolving `use utils/math.`, XLOG searches in order:

1. **Relative to current file** — if importing from `/project/src/main.xlog`, check `/project/src/utils/math.xlog`
2. **Module path directories** — each directory in `--module-path` flag, left to right

Example:
```bash
xlog run src/main.xlog --module-path /shared/lib:/home/user/xlog-stdlib
```

For `use utils/math.` from `src/main.xlog`:
1. `src/utils/math.xlog`
2. `/shared/lib/utils/math.xlog`
3. `/home/user/xlog-stdlib/utils/math.xlog`

First match wins. Error if not found in any location.

### Circular Import Detection

Circular imports are rejected at compile time:

```
error[E0401]: circular import detected
  --> src/a.xlog:1:1
   |
 1 | use b.
   | ^^^^^^ a.xlog imports b.xlog
   |
  --> src/b.xlog:1:1
   |
 1 | use a.
   | ^^^^^^ b.xlog imports a.xlog
   |
   = help: extract shared predicates into a third module
```

### Name Conflict Detection

Importing the same predicate name from multiple modules is an error:

```
error[E0402]: ambiguous import `edge`
  --> src/main.xlog:2:1
   |
 1 | use graph.
   |     ----- `edge` first imported here
 2 | use network.
   |     ^^^^^^^ `edge` also exported by network
   |
   = help: use selective imports: `use graph::{edge}.`
```

---

## Grammar Changes

Additions to `grammar.pest`:

```pest
// Module path: graph or utils/math or deep/nested/module
module_path = @{ ident ~ ("/" ~ ident)* }

// Import statements
import_list = { "{" ~ ident ~ ("," ~ ident)* ~ "}" }
use_stmt = { "use" ~ module_path ~ (":" ~ import_list)? ~ "." }

// Private modifier for predicates and rules
private_mod = { "private" }
pred_decl = { private_mod? ~ "pred" ~ ident ~ "(" ~ type_list? ~ ")" ~ "." }

// Update statement to include use_stmt
statement = {
    use_stmt          // NEW
    | domain_decl
    | pred_decl       // updated with private_mod
    | pragma
    | rule_def
    | prob_fact
    | annotated_disjunction
    | evidence_stmt
    | prob_query
    | fact
    | constraint
    | query
}
```

---

## AST Changes

Additions to `ast.rs`:

```rust
/// Import statement
#[derive(Debug, Clone, PartialEq)]
pub struct UseDecl {
    pub module_path: Vec<String>,  // ["utils", "math"]
    pub imports: Option<Vec<String>>,  // None = all, Some([...]) = selective
    pub span: Span,  // for error reporting
}

/// Updated predicate declaration
#[derive(Debug, Clone, PartialEq)]
pub struct PredDecl {
    pub name: String,
    pub types: Vec<ScalarType>,
    pub is_private: bool,  // NEW
}

/// Updated Program
pub struct Program {
    pub imports: Vec<UseDecl>,  // NEW
    pub domains: Vec<DomainDecl>,
    pub predicates: Vec<PredDecl>,
    pub rules: Vec<Rule>,
    pub constraints: Vec<Constraint>,
    pub queries: Vec<Query>,
    pub prob_facts: Vec<ProbFact>,
    pub annotated_disjunctions: Vec<AnnotatedDisjunction>,
    pub evidence: Vec<Evidence>,
    pub prob_queries: Vec<ProbQuery>,
    pub directives: Directives,
}
```

---

## Compilation Pipeline

### Module Resolver

New module `resolver.rs` handles module loading, import resolution, and visibility checking:

```rust
pub struct ModuleResolver {
    search_paths: Vec<PathBuf>,
    loaded: HashMap<ModulePath, LoadedModule>,
    loading: HashSet<ModulePath>,  // for cycle detection
}

pub struct LoadedModule {
    pub path: ModulePath,
    pub source_file: PathBuf,
    pub program: Program,
    pub exports: HashSet<String>,  // public predicate names
}

impl ModuleResolver {
    /// Resolve and load a module, detecting cycles
    pub fn resolve(&mut self, from: &Path, module_path: &[String])
        -> Result<&LoadedModule, ModuleError>;

    /// Check if predicate is accessible from importing module
    pub fn check_visibility(&self, module: &ModulePath, predicate: &str)
        -> Result<(), VisibilityError>;
}
```

### Compilation Phases

1. **Parse** — Parse entry file, collect `use` statements
2. **Resolve** — Recursively load imported modules, detect cycles
3. **Merge** — Build unified symbol table with qualified names internally
4. **Visibility Check** — Verify all referenced predicates are accessible
5. **Lower** — Existing lowering, but with fully-qualified internal names
6. **Compile** — Unchanged from current pipeline

### Internal Naming

Internally, all predicates get qualified names to avoid collision:

```
graph:edge/2      → __graph__edge
utils/math:abs/1  → __utils_math__abs
main:query/1      → __main__query
```

User-facing output uses original names with module prefix when ambiguous.

---

## Testing Strategy

### Test Cases

| Category | Test |
|----------|------|
| Basic import | `use graph.` loads and imports all public predicates |
| Selective import | `use graph::{edge}.` imports only `edge` |
| Nested module | `use utils/math.` resolves from subdirectory |
| Private visibility | Importing private predicate fails with clear error |
| Circular detection | `a → b → a` produces cycle error |
| Name conflict | Same predicate from two modules errors |
| Not found | Missing module produces helpful error with search paths tried |
| Search path order | Relative takes precedence over `--module-path` |
| Transitive deps | `main → graph → utils` all resolve correctly |

### Test File Structure

```
crates/xlog-logic/tests/
  modules/
    basic/
      main.xlog
      helper.xlog
    nested/
      main.xlog
      lib/
        utils.xlog
    circular/
      a.xlog
      b.xlog
    visibility/
      main.xlog
      internal.xlog
```

### Error Messages

All module errors include:
- Source location (file:line:col)
- What was attempted
- Why it failed
- Actionable suggestion (`= help:`)

Example for missing module:
```
error[E0400]: module not found: `utils/strings`
  --> src/main.xlog:3:5
   |
 3 | use utils/strings.
   |     ^^^^^^^^^^^^^ module not found
   |
   = note: searched in:
           - src/utils/strings.xlog
           - /stdlib/utils/strings.xlog
   = help: check the module path spelling or add to --module-path
```

---

## Implementation Plan

### Files to Create

| File | Purpose |
|------|---------|
| `crates/xlog-logic/src/module.rs` | `ModulePath`, `LoadedModule` types |
| `crates/xlog-logic/src/resolver.rs` | `ModuleResolver`, import resolution |
| `crates/xlog-logic/tests/module_tests.rs` | Module system integration tests |

### Files to Modify

| File | Changes |
|------|---------|
| `grammar.pest` | Add `use_stmt`, `module_path`, `private_mod`, `import_list` |
| `ast.rs` | Add `UseDecl`, update `PredDecl` with `is_private`, add `imports` to `Program` |
| `parser.rs` | Parse new grammar rules into AST |
| `lib.rs` | Export new modules, update public API |
| `compile.rs` | Integrate resolver into compilation pipeline |
| `lower.rs` | Use qualified internal names for predicates |

### Implementation Order

1. Grammar + AST changes (parser compiles but ignores imports)
2. `module.rs` — core types
3. `resolver.rs` — file loading, cycle detection
4. Parser updates — wire up `use_stmt` parsing
5. Visibility checking — enforce `private`
6. Integration — wire resolver into `compile.rs`
7. Tests — comprehensive module test suite
8. CLI — add `--module-path` flag

### Dependencies

- No external crate dependencies needed
- Uses existing `pest` parser infrastructure
- Uses existing error types from `xlog-core`

---

## Non-Goals (Deferred)

- **Module aliases** (`use graph as g.`) — can add in future version
- **Re-exports** (`pub use internal::{foo}.`) — can add if needed
- **Index files** (`mod.xlog`) — current directory-based approach is sufficient
- **Glob imports** (`use graph::*.`) — explicit imports preferred

---

## Summary

The module system provides:

- **File-based modules** — no boilerplate declarations
- **Explicit imports** — clear dependency tracking
- **Public by default** — `private` keyword for encapsulation
- **Directory nesting** — `utils/math` maps to `utils/math.xlog`
- **Slash separator** — intuitive filesystem-like paths
- **Conflict detection** — errors on ambiguous imports
- **Cycle detection** — clean error on circular dependencies
- **Search paths** — `--module-path` for shared libraries
