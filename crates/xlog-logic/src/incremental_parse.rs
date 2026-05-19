//! Statement-level parser session cache for incremental workflows.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::Duration;

use xlog_core::{Result, XlogError};

use crate::ast::{Directives, Program};
use crate::parser::parse_program;

/// Byte and line/column span for one source statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatementSpan {
    /// Byte offset where the statement starts.
    pub start: usize,
    /// Byte offset where the statement ends.
    pub end: usize,
    /// One-based line where the statement starts.
    pub line: usize,
    /// One-based column where the statement starts.
    pub column: usize,
}

/// One statement unit discovered in a source file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatementUnit {
    /// Statement text.
    pub text: String,
    /// Statement source span.
    pub span: StatementSpan,
    /// Stable hash of the statement text.
    pub hash: u64,
}

/// Cache statistics from one incremental parse.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ParseCacheStats {
    /// Number of unchanged statements reused from cache.
    pub hits: usize,
    /// Number of statements parsed fresh.
    pub misses: usize,
    /// Number of prior cached statements invalidated for this source.
    pub invalidated: usize,
    /// Number of cached source files invalidated through module invalidation.
    pub module_invalidations: usize,
    /// Total statement count in the parsed source.
    pub statement_count: usize,
    /// Full-parse work estimate, in statement units.
    pub full_parse_units: usize,
    /// Incremental work estimate, in statement units.
    pub incremental_parse_units: usize,
    /// Wall-clock time spent in the incremental parse call.
    pub elapsed: Duration,
}

impl ParseCacheStats {
    /// Structural speedup estimate based on avoided statement parses.
    pub fn estimated_speedup(&self) -> f64 {
        if self.incremental_parse_units == 0 {
            return self.full_parse_units.max(1) as f64;
        }
        self.full_parse_units as f64 / self.incremental_parse_units as f64
    }
}

/// Result from parsing source through a [`ParserSession`].
#[derive(Debug, Clone)]
pub struct IncrementalParseResult {
    /// Parsed program assembled from statement fragments.
    pub program: Program,
    /// Statement units discovered in source order.
    pub statements: Vec<StatementUnit>,
    /// Cache statistics for this parse.
    pub stats: ParseCacheStats,
}

#[derive(Debug, Clone)]
struct CachedStatement {
    hash: u64,
    text: String,
    program: Program,
}

#[derive(Debug, Clone, Default)]
struct CachedSource {
    statements: Vec<CachedStatement>,
    imports: Vec<Vec<String>>,
}

/// Incremental parser cache keyed by source path.
#[derive(Debug, Default)]
pub struct ParserSession {
    sources: HashMap<PathBuf, CachedSource>,
    module_invalidations: usize,
}

impl ParserSession {
    /// Create an empty parser session.
    pub fn new() -> Self {
        Self::default()
    }

    /// Split source text into statement units with byte and line/column spans.
    pub fn split_statements(source: &str) -> Vec<StatementUnit> {
        split_statements(source)
    }

    /// Parse source associated with a path, reusing unchanged statement parses.
    pub fn parse_path(
        &mut self,
        path: impl AsRef<Path>,
        source: &str,
    ) -> Result<IncrementalParseResult> {
        let started = std::time::Instant::now();
        let path = path.as_ref().to_path_buf();
        let units = split_statements(source);
        let previous = self.sources.get(&path);

        let mut stats = ParseCacheStats {
            statement_count: units.len(),
            full_parse_units: units.len(),
            module_invalidations: self.module_invalidations,
            ..ParseCacheStats::default()
        };
        self.module_invalidations = 0;

        let mut parsed_statements = Vec::with_capacity(units.len());
        for (idx, unit) in units.iter().enumerate() {
            if let Some(prev) = previous.and_then(|src| src.statements.get(idx)) {
                if prev.hash == unit.hash && prev.text == unit.text {
                    stats.hits += 1;
                    parsed_statements.push(prev.clone());
                    continue;
                }
            }

            stats.misses += 1;
            let program = parse_program(&unit.text).map_err(|err| {
                XlogError::Parse(format!(
                    "incremental parse error at {}:{} (bytes {}..{}): {}",
                    unit.span.line, unit.span.column, unit.span.start, unit.span.end, err
                ))
            })?;
            parsed_statements.push(CachedStatement {
                hash: unit.hash,
                text: unit.text.clone(),
                program,
            });
        }

        if let Some(previous) = previous {
            let retained = parsed_statements
                .iter()
                .filter(|stmt| {
                    previous
                        .statements
                        .iter()
                        .any(|prev| prev.hash == stmt.hash && prev.text == stmt.text)
                })
                .count();
            stats.invalidated = previous.statements.len().saturating_sub(retained);
        }

        stats.incremental_parse_units = stats.misses;
        stats.elapsed = started.elapsed();

        let mut program = Program::new();
        let mut imports = Vec::new();
        for cached in &parsed_statements {
            append_program(&mut program, cached.program.clone());
            imports.extend(cached.program.imports.iter().map(|u| u.module_path.clone()));
        }

        self.sources.insert(
            path,
            CachedSource {
                statements: parsed_statements,
                imports,
            },
        );

        Ok(IncrementalParseResult {
            program,
            statements: units,
            stats,
        })
    }

    /// Invalidate one module path and cached sources that import it by final path segment.
    pub fn invalidate_module(&mut self, path: impl AsRef<Path>) -> usize {
        let path = path.as_ref();
        let module_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(str::to_string);
        let mut removed = Vec::new();
        for (cached_path, source) in &self.sources {
            let direct = cached_path == path;
            let dependent = module_name.as_ref().is_some_and(|name| {
                source
                    .imports
                    .iter()
                    .any(|parts| parts.last().is_some_and(|part| part == name))
            });
            if direct || dependent {
                removed.push(cached_path.clone());
            }
        }
        let count = removed.len();
        for path in removed {
            self.sources.remove(&path);
        }
        self.module_invalidations = self.module_invalidations.saturating_add(count);
        count
    }

    /// Return the number of cached source files.
    pub fn cached_source_count(&self) -> usize {
        self.sources.len()
    }
}

fn split_statements(source: &str) -> Vec<StatementUnit> {
    let line_starts = line_starts(source);
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut in_comment = false;
    let mut line_start = 0usize;

    for (idx, ch) in source.char_indices() {
        if in_comment {
            if ch == '\n' {
                in_comment = false;
                let segment = source[start..idx].trim_start();
                if segment.starts_with("#pragma") {
                    push_statement(source, &line_starts, start, idx, &mut out);
                    start = idx + ch.len_utf8();
                } else if segment.starts_with("//") || segment.is_empty() {
                    start = idx + ch.len_utf8();
                }
                line_start = idx + ch.len_utf8();
            }
            continue;
        }

        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        if ch == '"' {
            in_string = true;
            continue;
        }

        if ch == '/' && source[idx..].starts_with("//") {
            if source[start..idx].trim().is_empty() {
                start = idx;
            }
            in_comment = true;
            continue;
        }

        if ch == '\n' {
            if source[start..idx].trim_start().starts_with("#pragma") {
                push_statement(source, &line_starts, start, idx, &mut out);
                start = idx + ch.len_utf8();
            }
            line_start = idx + ch.len_utf8();
            continue;
        }

        if ch == '.' && !is_decimal_point(source, idx) {
            push_statement(source, &line_starts, start, idx + ch.len_utf8(), &mut out);
            start = idx + ch.len_utf8();
        }

        if idx == line_start && ch.is_whitespace() {
            line_start = idx + ch.len_utf8();
        }
    }

    if source[start..].trim().is_empty() {
        return out;
    }
    push_statement(source, &line_starts, start, source.len(), &mut out);
    out
}

fn push_statement(
    source: &str,
    line_starts: &[usize],
    start: usize,
    end: usize,
    out: &mut Vec<StatementUnit>,
) {
    let text = source[start..end].trim().to_string();
    if text.is_empty() || text.starts_with("//") {
        return;
    }
    let trimmed_start = source[start..end]
        .find(|c: char| !c.is_whitespace())
        .map(|offset| start + offset)
        .unwrap_or(start);
    let (line, column) = line_col(line_starts, trimmed_start);
    out.push(StatementUnit {
        hash: stable_hash(&text),
        text,
        span: StatementSpan {
            start: trimmed_start,
            end,
            line,
            column,
        },
    });
}

fn line_starts(source: &str) -> Vec<usize> {
    let mut starts = vec![0];
    for (idx, ch) in source.char_indices() {
        if ch == '\n' {
            starts.push(idx + ch.len_utf8());
        }
    }
    starts
}

fn line_col(line_starts: &[usize], byte: usize) -> (usize, usize) {
    let idx = match line_starts.binary_search(&byte) {
        Ok(idx) => idx,
        Err(idx) => idx.saturating_sub(1),
    };
    (idx + 1, byte.saturating_sub(line_starts[idx]) + 1)
}

fn is_decimal_point(source: &str, idx: usize) -> bool {
    let prev = source[..idx].chars().next_back();
    let next = source[idx + 1..].chars().next();
    matches!((prev, next), (Some(a), Some(b)) if a.is_ascii_digit() && b.is_ascii_digit())
}

fn stable_hash(text: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

fn append_program(target: &mut Program, fragment: Program) {
    target.imports.extend(fragment.imports);
    target.functions.extend(fragment.functions);
    target.domains.extend(fragment.domains);
    target.predicates.extend(fragment.predicates);
    target.rules.extend(fragment.rules);
    target.constraints.extend(fragment.constraints);
    target.queries.extend(fragment.queries);
    target.prob_facts.extend(fragment.prob_facts);
    target
        .annotated_disjunctions
        .extend(fragment.annotated_disjunctions);
    target.evidence.extend(fragment.evidence);
    target.prob_queries.extend(fragment.prob_queries);
    target.neural_predicates.extend(fragment.neural_predicates);
    target.learnable_rules.extend(fragment.learnable_rules);
    merge_directives(&mut target.directives, fragment.directives);
}

fn merge_directives(target: &mut Directives, fragment: Directives) {
    if fragment.prob_engine.is_some() {
        target.prob_engine = fragment.prob_engine;
    }
    if fragment.prob_cache.is_some() {
        target.prob_cache = fragment.prob_cache;
    }
    if fragment.prob_samples.is_some() {
        target.prob_samples = fragment.prob_samples;
    }
    if fragment.prob_seed.is_some() {
        target.prob_seed = fragment.prob_seed;
    }
    if fragment.prob_confidence.is_some() {
        target.prob_confidence = fragment.prob_confidence;
    }
    if fragment.prob_method.is_some() {
        target.prob_method = fragment.prob_method;
    }
    if fragment.prob_max_nonmonotone_iterations.is_some() {
        target.prob_max_nonmonotone_iterations = fragment.prob_max_nonmonotone_iterations;
    }
    if fragment.max_recursion_depth.is_some() {
        target.max_recursion_depth = fragment.max_recursion_depth;
    }
    if fragment.magic_sets.is_some() {
        target.magic_sets = fragment.magic_sets;
    }
}
