use clap::{Parser, Subcommand, ValueEnum};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arrow::csv::WriterBuilder;
use arrow::util::pretty::pretty_format_batches;
use xlog_core::{symbol, MemoryBudget, Result, XlogError};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_gpu::logic::LogicProgram;
use xlog_logic::ast::{BodyLiteral, CompOp, ProbEngine, Program, Term};
use xlog_logic::compile::load_modules;
#[cfg(feature = "host-io")]
use xlog_logic::parse_program;
use xlog_logic::{rewrite_v085_magic_sets, MagicSetReport, MagicSetStatus, ParserSession};
use xlog_logic::{stratify, Compiler};
use xlog_logic::{QueryProofTrace, RuleProvenance};
#[cfg(feature = "host-io")]
use xlog_prob::exact::ExactDdnnfProgram;
#[cfg(feature = "host-io")]
use xlog_prob::exact::GpuConfig;
#[cfg(feature = "host-io")]
use xlog_prob::mc::{McEvalConfig, McProgram, McSamplingMethod};
use xlog_prob::provenance::{AggregateLiftReport, Value};

#[derive(Parser)]
#[command(author, version, about = "XLOG CLI")]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Run(RunArgs),
    Prob(ProbArgs),
    Explain(ExplainArgs),
    Repl(ReplArgs),
    Watch(WatchArgs),
}

#[derive(Parser)]
struct RunArgs {
    source: PathBuf,
    #[arg(long, default_value = "0")]
    device: usize,
    #[arg(long, default_value = "1024")]
    memory_mb: u64,
    #[arg(long)]
    input: Vec<String>,
    #[arg(long, value_enum, default_value = "pretty")]
    output: OutputFormat,
    #[arg(long)]
    output_dir: Option<PathBuf>,
    /// Show execution statistics (timing, memory usage)
    #[arg(long)]
    stats: bool,
    /// Stats output format (human or json)
    #[arg(long, value_enum, default_value = "human")]
    stats_format: StatsFormat,
    /// Additional directories to search for modules (colon-separated)
    #[arg(long, value_delimiter = ':')]
    module_path: Vec<PathBuf>,
}

#[derive(Copy, Clone, ValueEnum, Default)]
enum StatsFormat {
    #[default]
    Human,
    Json,
}

#[derive(Parser)]
struct ProbArgs {
    source: PathBuf,
    #[arg(long, default_value = "0")]
    device: usize,
    #[arg(long, default_value = "1024")]
    memory_mb: u64,
    #[arg(long, value_enum)]
    prob_engine: Option<ProbEngineCli>,
    #[arg(long)]
    samples: Option<usize>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    confidence: Option<f64>,
    #[arg(long, value_enum)]
    prob_method: Option<ProbMethodCli>,
    #[arg(long, alias = "max-nonmonotone-iterations")]
    prob_max_nonmonotone_iterations: Option<usize>,
    #[arg(long, value_enum, default_value = "pretty")]
    output: ProbOutputFormat,
    #[arg(long)]
    output_dir: Option<PathBuf>,
    /// Additional directories to search for modules (colon-separated)
    #[arg(long, value_delimiter = ':')]
    module_path: Vec<PathBuf>,
}

#[derive(Parser)]
struct ExplainArgs {
    source: PathBuf,
    #[arg(long, value_enum, default_value = "text")]
    format: ExplainFormat,
}

#[derive(Parser)]
struct ReplArgs {
    /// Additional directories to search for modules (colon-separated)
    #[arg(long, value_delimiter = ':')]
    module_path: Vec<PathBuf>,
}

#[derive(Parser)]
struct WatchArgs {
    source: PathBuf,
    #[arg(long, default_value = "250")]
    debounce_ms: u64,
    #[arg(long)]
    explain: bool,
    #[arg(long)]
    once: bool,
}

#[derive(Copy, Clone, ValueEnum)]
enum ExplainFormat {
    Text,
    Json,
    Dot,
}

#[derive(Copy, Clone, ValueEnum)]
enum OutputFormat {
    Pretty,
    Csv,
    Arrow,
}

#[derive(Copy, Clone, ValueEnum)]
enum ProbOutputFormat {
    Pretty,
    Csv,
    Arrow,
    Json,
}

#[derive(Copy, Clone, ValueEnum)]
enum ProbEngineCli {
    #[value(name = "exact_ddnnf")]
    ExactDdnnf,
    Mc,
}

#[derive(Copy, Clone, ValueEnum)]
enum ProbMethodCli {
    Rejection,
    #[value(name = "evidence_clamping")]
    EvidenceClamping,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run_deterministic(args),
        Command::Prob(args) => run_probabilistic(args),
        Command::Explain(args) => explain(args),
        Command::Repl(args) => repl(args),
        Command::Watch(args) => watch(args),
    }
}

fn explain(args: ExplainArgs) -> Result<()> {
    let source = std::fs::read_to_string(&args.source).map_err(|e| {
        XlogError::Execution(format!("Failed to read {}: {}", args.source.display(), e))
    })?;
    let mut parser_session = ParserSession::new();
    let parsed = parser_session.parse_path(&args.source, &source)?;
    let report = build_explain_report(parsed)?;
    match args.format {
        ExplainFormat::Text => print_explain_text(&report),
        ExplainFormat::Json => print_explain_json(&report),
        ExplainFormat::Dot => print_magic_dot(&report.magic_sets),
    }
    Ok(())
}

fn repl(args: ReplArgs) -> Result<()> {
    let _ = args.module_path;
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| XlogError::Execution(format!("Failed to read stdin: {}", e)))?;
    let mut parser_session = ParserSession::new();
    let parsed = parser_session.parse_path("<repl>", &input)?;
    println!(
        "repl: statements={} cache_hits={} cache_misses={}",
        parsed.stats.statement_count, parsed.stats.hits, parsed.stats.misses
    );
    println!(
        "state: rules={} queries={} prob_queries={}",
        parsed.program.rules.len(),
        parsed.program.queries.len(),
        parsed.program.prob_queries.len()
    );
    Ok(())
}

fn watch(args: WatchArgs) -> Result<()> {
    let mut parser_session = ParserSession::new();
    loop {
        let source = std::fs::read_to_string(&args.source).map_err(|e| {
            XlogError::Execution(format!("Failed to read {}: {}", args.source.display(), e))
        })?;
        let parsed = parser_session.parse_path(&args.source, &source)?;
        println!(
            "watch: statements={} cache_hits={} cache_misses={}",
            parsed.stats.statement_count, parsed.stats.hits, parsed.stats.misses
        );
        if args.explain {
            let report = build_explain_report(parsed)?;
            print_explain_text(&report);
        }
        if args.once {
            break;
        }
        std::thread::sleep(Duration::from_millis(args.debounce_ms));
    }
    Ok(())
}

struct ExplainReport {
    program: Program,
    parse_stats: xlog_logic::ParseCacheStats,
    magic_sets: MagicSetReport,
    aggregate_lifting: Vec<AggregateLiftReport>,
    generated_rule_diagnostics: Vec<GeneratedRuleDiagnostic>,
    rule_provenance: Vec<RuleProvenance>,
    proof_traces: Vec<QueryProofTrace>,
    stratification_status: String,
    stratification_count: usize,
    rir_status: String,
    rir_sccs: usize,
    optimizer_status: String,
    optimizer_memory_peak: u64,
}

fn build_explain_report(parsed: xlog_logic::IncrementalParseResult) -> Result<ExplainReport> {
    let program = parsed.program;
    let magic_rewrite = rewrite_v085_magic_sets(&program)?;
    let rule_provenance = xlog_logic::rule_provenance(&program, Some(&magic_rewrite.program));
    let proof_traces = xlog_logic::query_proof_traces(&program, &rule_provenance);
    let magic_sets = magic_rewrite.report;
    let aggregate_lifting = explain_aggregate_lifting(&program)?;
    let generated_rule_diagnostics = explain_generated_rule_diagnostics(&program);
    let (stratification_status, stratification_count) = match stratify(&program) {
        Ok(strata) => ("ok".to_string(), strata.len()),
        Err(err) => (format!("error: {}", err), 0),
    };
    let mut compiler = Compiler::new();
    let (rir_status, rir_sccs, optimizer_status, optimizer_memory_peak) =
        match compiler.compile_program(&program) {
            Ok(plan) => (
                "ok".to_string(),
                plan.sccs.len(),
                "ok".to_string(),
                plan.est_memory_peak,
            ),
            Err(err) => (format!("error: {}", err), 0, "not_available".to_string(), 0),
        };
    Ok(ExplainReport {
        program,
        parse_stats: parsed.stats,
        magic_sets,
        aggregate_lifting,
        generated_rule_diagnostics,
        rule_provenance,
        proof_traces,
        stratification_status,
        stratification_count,
        rir_status,
        rir_sccs,
        optimizer_status,
        optimizer_memory_peak,
    })
}

fn explain_aggregate_lifting(program: &Program) -> Result<Vec<AggregateLiftReport>> {
    let has_probabilistic_source =
        !program.prob_facts.is_empty() || !program.annotated_disjunctions.is_empty();
    let has_aggregate_rule = program.proper_rules().any(|rule| rule.has_aggregation());
    if !(has_probabilistic_source && has_aggregate_rule) {
        return Ok(Vec::new());
    }
    Ok(xlog_prob::provenance::extract_from_program(program)?.aggregate_lifting)
}

struct GeneratedRuleDiagnostic {
    rule_head: String,
    source_relation: String,
    row_decisions: Vec<GeneratedRuleRowDecision>,
}

struct GeneratedRuleRowDecision {
    row_key: String,
    accepted: bool,
    failed_predicates: Vec<String>,
    threshold_comparisons: Vec<ThresholdComparison>,
    aggregate_inputs: Vec<String>,
}

struct ThresholdComparison {
    predicate: String,
    left: String,
    op: String,
    right: String,
    left_value: String,
    right_value: String,
    passed: bool,
}

fn explain_generated_rule_diagnostics(program: &Program) -> Vec<GeneratedRuleDiagnostic> {
    let mut diagnostics = Vec::new();
    for rule in program
        .rules
        .iter()
        .filter(|rule| !rule.body.is_empty() && generated_rule_candidate(rule))
    {
        let Some(source_atom) = rule.body.iter().find_map(|literal| match literal {
            BodyLiteral::Positive(atom) if atom.predicate.starts_with("generated_") => Some(atom),
            _ => None,
        }) else {
            continue;
        };
        let mut row_decisions = Vec::new();
        for fact in program.rules.iter().filter(|fact| {
            fact.body.is_empty()
                && fact.head.predicate == source_atom.predicate
                && fact.head.terms.len() == source_atom.terms.len()
        }) {
            let mut bindings = HashMap::new();
            for (pattern, value) in source_atom.terms.iter().zip(&fact.head.terms) {
                if let Term::Variable(name) = pattern {
                    bindings.insert(name.clone(), value.clone());
                }
            }

            let mut threshold_comparisons = Vec::new();
            for comparison in rule.body.iter().filter_map(|literal| match literal {
                BodyLiteral::Comparison(comparison) => Some(comparison),
                _ => None,
            }) {
                let left_value = bound_term(&comparison.left, &bindings);
                let right_value = bound_term(&comparison.right, &bindings);
                let passed = left_value
                    .as_ref()
                    .zip(right_value.as_ref())
                    .and_then(|(left, right)| compare_terms(left, comparison.op, right))
                    .unwrap_or(false);
                threshold_comparisons.push(ThresholdComparison {
                    predicate: format!(
                        "{} {} {}",
                        term_label(&comparison.left),
                        comp_op_label(comparison.op),
                        term_label(&comparison.right)
                    ),
                    left: term_label(&comparison.left),
                    op: comp_op_label(comparison.op).to_string(),
                    right: term_label(&comparison.right),
                    left_value: left_value
                        .as_ref()
                        .map(term_label)
                        .unwrap_or_else(|| "unbound".to_string()),
                    right_value: right_value
                        .as_ref()
                        .map(term_label)
                        .unwrap_or_else(|| "unbound".to_string()),
                    passed,
                });
            }
            let mut failed_predicates = predicate_failures(program, rule, source_atom, &bindings);
            failed_predicates.extend(
                threshold_comparisons
                    .iter()
                    .filter(|comparison| !comparison.passed)
                    .map(|comparison| comparison.predicate.clone()),
            );
            row_decisions.push(GeneratedRuleRowDecision {
                row_key: fact
                    .head
                    .terms
                    .first()
                    .map(term_label)
                    .unwrap_or_else(|| source_atom.predicate.clone()),
                accepted: failed_predicates.is_empty(),
                failed_predicates,
                threshold_comparisons,
                aggregate_inputs: vec![format!(
                    "{}({})",
                    source_atom.predicate,
                    fact.head
                        .terms
                        .iter()
                        .map(term_label)
                        .collect::<Vec<_>>()
                        .join(", ")
                )],
            });
        }

        if !row_decisions.is_empty() {
            diagnostics.push(GeneratedRuleDiagnostic {
                rule_head: rule.head.predicate.clone(),
                source_relation: source_atom.predicate.clone(),
                row_decisions,
            });
        }
    }
    diagnostics
}

fn generated_rule_candidate(rule: &xlog_logic::ast::Rule) -> bool {
    rule.head.predicate.starts_with("generated_")
        || rule.body.iter().any(|literal| match literal {
            BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                atom.predicate.starts_with("generated_")
            }
            BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) | BodyLiteral::Univ(_) => false,
        })
}

fn predicate_failures(
    program: &Program,
    rule: &xlog_logic::ast::Rule,
    source_atom: &xlog_logic::ast::Atom,
    bindings: &HashMap<String, Term>,
) -> Vec<String> {
    let mut failures = Vec::new();
    for literal in &rule.body {
        match literal {
            BodyLiteral::Positive(atom) if atom.predicate != source_atom.predicate => {
                if !matching_fact_exists(program, atom, bindings) {
                    failures.push(atom_label(atom));
                }
            }
            BodyLiteral::Negated(atom) => {
                if matching_fact_exists(program, atom, bindings) {
                    failures.push(format!("not {}", atom_label(atom)));
                }
            }
            BodyLiteral::Positive(_)
            | BodyLiteral::Comparison(_)
            | BodyLiteral::IsExpr(_)
            | BodyLiteral::Univ(_) => {}
        }
    }
    failures
}

fn matching_fact_exists(
    program: &Program,
    atom: &xlog_logic::ast::Atom,
    bindings: &HashMap<String, Term>,
) -> bool {
    program.rules.iter().any(|fact| {
        fact.body.is_empty()
            && fact.head.predicate == atom.predicate
            && fact.head.terms.len() == atom.terms.len()
            && atom
                .terms
                .iter()
                .zip(&fact.head.terms)
                .all(|(pattern, value)| match bound_term(pattern, bindings) {
                    Some(bound) => bound == *value,
                    None => false,
                })
    })
}

fn atom_label(atom: &xlog_logic::ast::Atom) -> String {
    format!(
        "{}({})",
        atom.predicate,
        atom.terms
            .iter()
            .map(term_label)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn bound_term(term: &Term, bindings: &HashMap<String, Term>) -> Option<Term> {
    match term {
        Term::Variable(name) => bindings.get(name).cloned(),
        _ => Some(term.clone()),
    }
}

fn compare_terms(left: &Term, op: CompOp, right: &Term) -> Option<bool> {
    match (left, right) {
        (Term::Integer(left), Term::Integer(right)) => Some(compare_i64(*left, op, *right)),
        (Term::String(left), Term::String(right)) => match op {
            CompOp::Eq => Some(left == right),
            CompOp::Ne => Some(left != right),
            _ => None,
        },
        (Term::Symbol(left), Term::Symbol(right)) => match op {
            CompOp::Eq => Some(left == right),
            CompOp::Ne => Some(left != right),
            _ => None,
        },
        _ => None,
    }
}

fn compare_i64(left: i64, op: CompOp, right: i64) -> bool {
    match op {
        CompOp::Eq => left == right,
        CompOp::Ne => left != right,
        CompOp::Lt => left < right,
        CompOp::Le => left <= right,
        CompOp::Gt => left > right,
        CompOp::Ge => left >= right,
    }
}

fn comp_op_label(op: CompOp) -> &'static str {
    match op {
        CompOp::Eq => "==",
        CompOp::Ne => "!=",
        CompOp::Lt => "<",
        CompOp::Le => "<=",
        CompOp::Gt => ">",
        CompOp::Ge => ">=",
    }
}

fn term_label(term: &Term) -> String {
    match term {
        Term::Variable(name) => name.clone(),
        Term::Anonymous => "_".to_string(),
        Term::Integer(value) => value.to_string(),
        Term::Float(value) => value.to_string(),
        Term::String(value) => value.clone(),
        Term::Symbol(id) => symbol::resolve(*id),
        Term::List(items) => format!(
            "[{}]",
            items.iter().map(term_label).collect::<Vec<_>>().join(", ")
        ),
        Term::Cons { head, tail } => format!("{}|{}", term_label(head), term_label(tail)),
        Term::Compound { functor, args } => format!(
            "{}({})",
            functor,
            args.iter().map(term_label).collect::<Vec<_>>().join(", ")
        ),
        Term::PredRef(name) => name.clone(),
        Term::Aggregate(agg) => format!("{:?}({})", agg.op, agg.variable),
    }
}

fn print_explain_text(report: &ExplainReport) {
    println!("parse:");
    println!("  statements: {}", report.parse_stats.statement_count);
    println!("ast:");
    println!("  rules: {}", report.program.rules.len());
    println!("  queries: {}", report.program.queries.len());
    println!("stratification:");
    println!("  status: {}", report.stratification_status);
    println!("  strata: {}", report.stratification_count);
    println!("rir:");
    println!("  status: {}", report.rir_status);
    println!("  sccs: {}", report.rir_sccs);
    println!("optimizer:");
    println!("  status: {}", report.optimizer_status);
    println!("  est_memory_peak: {}", report.optimizer_memory_peak);
    println!("wcoj:");
    println!("  status: reported");
    print_magic_text(&report.magic_sets);
    if !report.aggregate_lifting.is_empty() {
        println!("aggregate_lifting:");
        for entry in &report.aggregate_lifting {
            println!(
                "  - predicate: {} operator: {} status: {} domain: {} uncertain: {} cap: {}",
                entry.predicate,
                entry.operator,
                entry.status.as_str(),
                entry.domain_size,
                entry.uncertain_rows,
                entry.cap
            );
        }
    }
    if !report.rule_provenance.is_empty() {
        println!("rule_provenance:");
        for entry in &report.rule_provenance {
            println!(
                "  - id: {} source_kind: {} head: {}",
                entry.rule_id,
                entry.source_kind.as_str(),
                entry.head
            );
        }
    }
    if !report.proof_traces.is_empty() {
        println!("proof_traces:");
        for entry in &report.proof_traces {
            println!(
                "  - query: {} rules: {} source_facts: {}",
                entry.query,
                entry.rule_ids.len(),
                entry.source_facts.len()
            );
        }
    }
}

fn print_magic_text(report: &MagicSetReport) {
    println!("magic_sets:");
    println!("  status: {}", magic_status_label(report.status));
    if !report.adorned_predicates.is_empty() {
        println!("  adorned_predicates:");
        for pred in &report.adorned_predicates {
            println!("    - {}", pred);
        }
    }
    if !report.generated_predicates.is_empty() {
        println!("  generated_predicates:");
        for pred in &report.generated_predicates {
            println!("    - {}", pred);
        }
    }
    if !report.declined_reasons.is_empty() {
        println!("  declined_reasons:");
        for reason in &report.declined_reasons {
            println!("    - {}", reason);
        }
    }
}

fn print_explain_json(report: &ExplainReport) {
    println!("{{");
    println!("  \"parse\": {{");
    println!(
        "    \"statements\": {},",
        report.parse_stats.statement_count
    );
    println!("    \"cache_hits\": {},", report.parse_stats.hits);
    println!("    \"cache_misses\": {}", report.parse_stats.misses);
    println!("  }},");
    println!("  \"ast\": {{");
    println!("    \"rules\": {},", report.program.rules.len());
    println!("    \"queries\": {},", report.program.queries.len());
    println!(
        "    \"prob_queries\": {}",
        report.program.prob_queries.len()
    );
    println!("  }},");
    println!("  \"stratification\": {{");
    println!(
        "    \"status\": \"{}\",",
        json_escape(&report.stratification_status)
    );
    println!("    \"strata\": {}", report.stratification_count);
    println!("  }},");
    println!("  \"rir\": {{");
    println!("    \"status\": \"{}\",", json_escape(&report.rir_status));
    println!("    \"sccs\": {}", report.rir_sccs);
    println!("  }},");
    println!("  \"optimizer\": {{");
    println!(
        "    \"status\": \"{}\",",
        json_escape(&report.optimizer_status)
    );
    println!("    \"est_memory_peak\": {}", report.optimizer_memory_peak);
    println!("  }},");
    println!("  \"wcoj\": {{");
    println!("    \"status\": \"reported\"");
    println!("  }},");
    println!("  \"magic_sets\": {{");
    println!(
        "    \"status\": \"{}\",",
        json_escape(magic_status_label(report.magic_sets.status))
    );
    println!(
        "    \"adorned_predicates\": {},",
        json_string_array(&report.magic_sets.adorned_predicates)
    );
    println!(
        "    \"generated_predicates\": {},",
        json_string_array(&report.magic_sets.generated_predicates)
    );
    println!(
        "    \"declined_reasons\": {}",
        json_string_array(&report.magic_sets.declined_reasons)
    );
    println!("  }},");
    println!("  \"probability\": {{");
    println!(
        "    \"engine\": \"{}\",",
        match report.program.prob_engine() {
            ProbEngine::ExactDdnnf => "exact_ddnnf",
            ProbEngine::Mc => "mc",
        }
    );
    println!(
        "    \"aggregate_lifting_count\": {}",
        report.aggregate_lifting.len()
    );
    println!("  }},");
    println!("  \"aggregate_lifting\": [");
    for (idx, entry) in report.aggregate_lifting.iter().enumerate() {
        let suffix = if idx + 1 == report.aggregate_lifting.len() {
            ""
        } else {
            ","
        };
        println!("    {{");
        println!(
            "      \"predicate\": \"{}\",",
            json_escape(&entry.predicate)
        );
        println!(
            "      \"group_key\": {},",
            json_value_array(&entry.group_key)
        );
        println!("      \"operator\": \"{}\",", json_escape(&entry.operator));
        println!(
            "      \"finite_domain_source\": \"{}\",",
            json_escape(&entry.finite_domain_source)
        );
        println!(
            "      \"deterministic_rows\": {},",
            entry.deterministic_rows
        );
        println!("      \"uncertain_rows\": {},", entry.uncertain_rows);
        println!("      \"domain_size\": {},", entry.domain_size);
        println!("      \"cap\": {},", entry.cap);
        println!("      \"status\": \"{}\",", entry.status.as_str());
        println!("      \"reason\": \"{}\",", json_escape(&entry.reason));
        println!("      \"naive_outcomes\": {},", entry.naive_outcomes);
        println!(
            "      \"dynamic_programming_states\": {}",
            entry.dynamic_programming_states
        );
        println!("    }}{}", suffix);
    }
    println!("  ],");
    print_rule_provenance_json(&report.rule_provenance);
    println!(",");
    print_proof_traces_json(&report.proof_traces);
    println!(",");
    print_generated_rule_diagnostics_json(&report.generated_rule_diagnostics);
    println!("}}");
}

fn print_rule_provenance_json(entries: &[RuleProvenance]) {
    println!("  \"rule_provenance\": [");
    for (idx, entry) in entries.iter().enumerate() {
        let suffix = if idx + 1 == entries.len() { "" } else { "," };
        println!("    {{");
        println!("      \"rule_id\": \"{}\",", json_escape(&entry.rule_id));
        println!("      \"head\": \"{}\",", json_escape(&entry.head));
        println!(
            "      \"source_kind\": \"{}\",",
            json_escape(entry.source_kind.as_str())
        );
        println!(
            "      \"source_span\": {},",
            json_optional_string(entry.source_span.as_deref())
        );
        println!(
            "      \"generation_trace_hash\": {},",
            json_optional_string(entry.generation_trace_hash.as_deref())
        );
        println!(
            "      \"support_relation_ids\": {},",
            json_string_array(&entry.support_relation_ids)
        );
        println!(
            "      \"counterexample_relation_ids\": {}",
            json_string_array(&entry.counterexample_relation_ids)
        );
        println!("    }}{}", suffix);
    }
    println!("  ]");
}

fn print_proof_traces_json(entries: &[QueryProofTrace]) {
    println!("  \"proof_traces\": [");
    for (idx, entry) in entries.iter().enumerate() {
        let suffix = if idx + 1 == entries.len() { "" } else { "," };
        println!("    {{");
        println!("      \"query_id\": \"{}\",", json_escape(&entry.query_id));
        println!("      \"query\": \"{}\",", json_escape(&entry.query));
        println!(
            "      \"answer_relation\": \"{}\",",
            json_escape(&entry.answer_relation)
        );
        println!(
            "      \"rule_ids\": {},",
            json_string_array(&entry.rule_ids)
        );
        println!(
            "      \"source_facts\": {},",
            json_string_array(&entry.source_facts)
        );
        println!(
            "      \"rejected_alternatives\": {}",
            json_string_array(&entry.rejected_alternatives)
        );
        println!("    }}{}", suffix);
    }
    println!("  ]");
}

fn print_generated_rule_diagnostics_json(entries: &[GeneratedRuleDiagnostic]) {
    println!("  \"generated_rule_diagnostics\": [");
    for (idx, entry) in entries.iter().enumerate() {
        let suffix = if idx + 1 == entries.len() { "" } else { "," };
        println!("    {{");
        println!(
            "      \"rule_head\": \"{}\",",
            json_escape(&entry.rule_head)
        );
        println!(
            "      \"source_relation\": \"{}\",",
            json_escape(&entry.source_relation)
        );
        println!("      \"row_decisions\": [");
        for (row_idx, row) in entry.row_decisions.iter().enumerate() {
            let row_suffix = if row_idx + 1 == entry.row_decisions.len() {
                ""
            } else {
                ","
            };
            println!("        {{");
            println!("          \"row_key\": \"{}\",", json_escape(&row.row_key));
            println!("          \"accepted\": {},", row.accepted);
            println!(
                "          \"failed_predicates\": {},",
                json_string_array(&row.failed_predicates)
            );
            println!("          \"threshold_comparisons\": [");
            for (comparison_idx, comparison) in row.threshold_comparisons.iter().enumerate() {
                let comparison_suffix = if comparison_idx + 1 == row.threshold_comparisons.len() {
                    ""
                } else {
                    ","
                };
                println!("            {{");
                println!(
                    "              \"predicate\": \"{}\",",
                    json_escape(&comparison.predicate)
                );
                println!(
                    "              \"left\": \"{}\",",
                    json_escape(&comparison.left)
                );
                println!("              \"op\": \"{}\",", json_escape(&comparison.op));
                println!(
                    "              \"right\": \"{}\",",
                    json_escape(&comparison.right)
                );
                println!(
                    "              \"left_value\": \"{}\",",
                    json_escape(&comparison.left_value)
                );
                println!(
                    "              \"right_value\": \"{}\",",
                    json_escape(&comparison.right_value)
                );
                println!("              \"passed\": {}", comparison.passed);
                println!("            }}{}", comparison_suffix);
            }
            println!("          ],");
            println!(
                "          \"aggregate_inputs\": {}",
                json_string_array(&row.aggregate_inputs)
            );
            println!("        }}{}", row_suffix);
        }
        println!("      ]");
        println!("    }}{}", suffix);
    }
    println!("  ]");
}

fn print_magic_dot(report: &MagicSetReport) {
    println!("digraph xlog_magic_sets {{");
    println!(
        "  status [label=\"status: {}\"];",
        magic_status_label(report.status)
    );
    for pred in &report.generated_predicates {
        println!("  \"{}\" [shape=box];", dot_escape(pred));
    }
    for pred in &report.adorned_predicates {
        println!("  \"{}\" [shape=ellipse];", dot_escape(pred));
    }
    println!("}}");
}

fn magic_status_label(status: MagicSetStatus) -> &'static str {
    match status {
        MagicSetStatus::Disabled => "disabled",
        MagicSetStatus::Applied => "applied",
        MagicSetStatus::Declined => "declined",
    }
}

fn json_string_array(items: &[String]) -> String {
    let values = items
        .iter()
        .map(|item| format!("\"{}\"", json_escape(item)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{}]", values)
}

fn json_value_array(items: &[Value]) -> String {
    let values = items.iter().map(json_value).collect::<Vec<_>>().join(", ");
    format!("[{}]", values)
}

fn json_value(value: &Value) -> String {
    match value {
        Value::I64(v) => v.to_string(),
        Value::F64(bits) => {
            let v = f64::from_bits(*bits);
            if v.is_finite() {
                v.to_string()
            } else {
                format!("\"{}\"", json_escape(&v.to_string()))
            }
        }
        Value::Symbol(id) => format!("\"{}\"", json_escape(&symbol::resolve(*id))),
        Value::String(s) => format!("\"{}\"", json_escape(s)),
    }
}

fn json_optional_string(value: Option<&str>) -> String {
    match value {
        Some(value) => format!("\"{}\"", json_escape(value)),
        None => "null".to_string(),
    }
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn dot_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn make_provider(device: usize, memory_mb: u64) -> Result<Arc<CudaKernelProvider>> {
    let device = Arc::new(CudaDevice::new(device)?);
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(memory_mb * 1024 * 1024),
    ));
    Ok(Arc::new(CudaKernelProvider::new(device, memory)?))
}

fn parse_inputs(inputs: &[String]) -> Result<HashMap<String, PathBuf>> {
    let mut out = HashMap::new();
    for entry in inputs {
        let (name, path) = entry.split_once('=').ok_or_else(|| {
            XlogError::Execution(format!("Invalid --input '{}', expected rel=path", entry))
        })?;
        out.insert(name.to_string(), PathBuf::from(path));
    }
    Ok(out)
}

fn run_deterministic(args: RunArgs) -> Result<()> {
    let provider = make_provider(args.device, args.memory_mb)?;
    let source = std::fs::read_to_string(&args.source).map_err(|e| {
        XlogError::Execution(format!("Failed to read {}: {}", args.source.display(), e))
    })?;

    // Check if the source has any imports that need resolution
    let has_imports = source.contains("use ");

    // Load and merge modules if there are imports
    let program = if has_imports {
        let resolver = load_modules(&args.source, args.module_path.clone())
            .map_err(|e| XlogError::Execution(format!("Module resolution failed: {}", e)))?;
        LogicProgram::compile_with_resolver(&source, &resolver)?
    } else {
        LogicProgram::compile(&source)?
    };
    let mut inputs = HashMap::new();
    for (name, path) in parse_inputs(&args.input)? {
        let buf = provider.read_arrow_ipc_stream_file(&path)?;
        inputs.insert(name, buf);
    }

    let result = program.evaluate_with_options(provider.clone(), inputs, args.stats)?;

    // Emit query results
    emit_logic_results(
        provider.as_ref(),
        &result.queries,
        args.output,
        args.output_dir.as_deref(),
    )?;

    // Emit stats if requested
    if args.stats {
        if let Some(stats) = result.stats {
            let stats_output = match args.stats_format {
                StatsFormat::Human => stats.format_human(),
                StatsFormat::Json => stats.format_json(),
            };
            eprintln!("{}", stats_output);
        }
        // Symbol table statistics
        eprintln!(
            "Symbols: {} interned ({} bytes)",
            symbol::count(),
            symbol::memory_usage()
        );
    }

    Ok(())
}

fn run_probabilistic(args: ProbArgs) -> Result<()> {
    #[cfg(not(feature = "host-io"))]
    {
        let _ = args;
        return Err(XlogError::Execution(
            "Host output is disabled (feature \"host-io\" is OFF). Use device-resident APIs (DLPack) or rebuild with --features host-io.".to_string(),
        ));
    }

    #[cfg(feature = "host-io")]
    {
        let source = std::fs::read_to_string(&args.source).map_err(|e| {
            XlogError::Execution(format!("Failed to read {}: {}", args.source.display(), e))
        })?;
        let parsed_program = parse_program(&source)?;

        // Validate module imports if any search paths are provided
        if !args.module_path.is_empty() {
            let _ = load_modules(&args.source, args.module_path.clone())
                .map_err(|e| XlogError::Execution(format!("Module resolution failed: {}", e)))?;
        }

        let mut config = GpuConfig::default();
        config.device_ordinal = args.device;
        config.memory_bytes = args.memory_mb * 1024 * 1024;

        match resolve_prob_engine(&args, &parsed_program) {
            ProbEngineCli::ExactDdnnf => {
                let prog = ExactDdnnfProgram::compile_source_with_gpu(&source, config)?;
                let result = prog.evaluate()?;
                emit_prob_exact(result, args.output, args.output_dir.as_deref())
            }
            ProbEngineCli::Mc => {
                let prog = McProgram::compile_source_with_gpu(&source, config)?;
                let mut cfg = McEvalConfig::from_directives(&parsed_program.directives)?;
                apply_mc_cli_overrides(&args, &mut cfg)?;
                let result = prog.evaluate(cfg)?;
                emit_prob_mc(result, args.output, args.output_dir.as_deref())
            }
        }
    }
}

#[cfg(feature = "host-io")]
fn resolve_prob_engine(args: &ProbArgs, program: &Program) -> ProbEngineCli {
    args.prob_engine
        .unwrap_or_else(|| match program.directives.prob_engine_or_default() {
            ProbEngine::ExactDdnnf => ProbEngineCli::ExactDdnnf,
            ProbEngine::Mc => ProbEngineCli::Mc,
        })
}

#[cfg(feature = "host-io")]
fn apply_mc_cli_overrides(args: &ProbArgs, cfg: &mut McEvalConfig) -> Result<()> {
    if let Some(samples) = args.samples {
        cfg.samples = samples;
    }
    if let Some(seed) = args.seed {
        cfg.seed = seed;
    }
    if let Some(confidence) = args.confidence {
        cfg.confidence = confidence;
    }
    if let Some(iterations) = args.prob_max_nonmonotone_iterations {
        cfg.max_nonmonotone_iterations = iterations;
    }
    if let Some(method) = args.prob_method {
        cfg.sampling_method = Some(match method {
            ProbMethodCli::Rejection => McSamplingMethod::Rejection,
            ProbMethodCli::EvidenceClamping => McSamplingMethod::EvidenceClamping,
        });
    }
    cfg.validate()
}

fn emit_logic_results(
    provider: &CudaKernelProvider,
    queries: &[xlog_gpu::logic::LogicQueryResult],
    format: OutputFormat,
    output_dir: Option<&Path>,
) -> Result<()> {
    for (i, q) in queries.iter().enumerate() {
        let batch = provider.to_arrow_record_batch(&q.buffer)?;
        match format {
            OutputFormat::Pretty => {
                let formatted = pretty_format_batches(&[batch])
                    .map_err(|e| XlogError::Execution(format!("Pretty print failed: {}", e)))?;
                println!("{}\n{}", q.relation_name, formatted);
            }
            OutputFormat::Csv => {
                let mut out = Vec::new();
                {
                    let mut writer = WriterBuilder::new().build(&mut out);
                    writer
                        .write(&batch)
                        .map_err(|e| XlogError::Execution(format!("CSV write failed: {}", e)))?;
                }
                println!("{}\n{}", q.relation_name, String::from_utf8_lossy(&out));
            }
            OutputFormat::Arrow => {
                let dir = output_dir.unwrap_or_else(|| Path::new("."));
                let path = dir.join(format!("query_{}.arrow", i));
                provider.write_arrow_ipc_stream_file(&q.buffer, &path)?;
                println!("wrote {}", path.display());
            }
        }
    }
    Ok(())
}

#[cfg(feature = "host-io")]
fn emit_prob_exact(
    result: xlog_prob::exact::ExactResult,
    format: ProbOutputFormat,
    output_dir: Option<&Path>,
) -> Result<()> {
    if matches!(format, ProbOutputFormat::Json) {
        print_prob_exact_json(result);
        return Ok(());
    }

    let mut atoms = Vec::new();
    let mut probs = Vec::new();
    let mut log_probs = Vec::new();
    for q in result.query_probs {
        atoms.push(atom_to_string(&q.atom));
        probs.push(q.prob);
        log_probs.push(q.log_prob);
    }

    let batch = arrow::record_batch::RecordBatch::try_from_iter(vec![
        (
            "atom",
            Arc::new(arrow::array::StringArray::from(atoms)) as Arc<dyn arrow::array::Array>,
        ),
        (
            "prob",
            Arc::new(arrow::array::Float64Array::from(probs)) as Arc<dyn arrow::array::Array>,
        ),
        (
            "log_prob",
            Arc::new(arrow::array::Float64Array::from(log_probs)) as Arc<dyn arrow::array::Array>,
        ),
    ])
    .map_err(|e| XlogError::Execution(format!("Failed to build prob batch: {}", e)))?;

    emit_batch(
        "prob",
        &batch,
        prob_output_as_batch_format(format),
        output_dir,
    )
}

#[cfg(feature = "host-io")]
fn emit_prob_mc(
    result: xlog_prob::mc::McResult,
    format: ProbOutputFormat,
    output_dir: Option<&Path>,
) -> Result<()> {
    if matches!(format, ProbOutputFormat::Json) {
        print_prob_mc_json(result);
        return Ok(());
    }

    let total_samples = result.total_samples as u64;
    let evidence_samples = result.evidence_samples as u64;
    let seed = result.seed;
    let confidence = result.confidence;
    let sampling_method = result.sampling_method.as_str().to_string();

    let mut atoms = Vec::new();
    let mut probs = Vec::new();
    let mut log_probs = Vec::new();
    let mut stderr = Vec::new();
    let mut ci_low = Vec::new();
    let mut ci_high = Vec::new();
    let mut total_samples_col = Vec::new();
    let mut evidence_samples_col = Vec::new();
    let mut seed_col = Vec::new();
    let mut confidence_col = Vec::new();
    let mut sampling_method_col = Vec::new();
    for q in result.query_estimates {
        atoms.push(atom_to_string(&q.atom));
        probs.push(q.prob);
        log_probs.push(q.log_prob);
        stderr.push(q.stderr);
        ci_low.push(q.ci_low);
        ci_high.push(q.ci_high);
        total_samples_col.push(total_samples);
        evidence_samples_col.push(evidence_samples);
        seed_col.push(seed);
        confidence_col.push(confidence);
        sampling_method_col.push(sampling_method.clone());
    }

    let batch = arrow::record_batch::RecordBatch::try_from_iter(vec![
        (
            "atom",
            Arc::new(arrow::array::StringArray::from(atoms)) as Arc<dyn arrow::array::Array>,
        ),
        (
            "prob",
            Arc::new(arrow::array::Float64Array::from(probs)) as Arc<dyn arrow::array::Array>,
        ),
        (
            "log_prob",
            Arc::new(arrow::array::Float64Array::from(log_probs)) as Arc<dyn arrow::array::Array>,
        ),
        (
            "stderr",
            Arc::new(arrow::array::Float64Array::from(stderr)) as Arc<dyn arrow::array::Array>,
        ),
        (
            "ci_low",
            Arc::new(arrow::array::Float64Array::from(ci_low)) as Arc<dyn arrow::array::Array>,
        ),
        (
            "ci_high",
            Arc::new(arrow::array::Float64Array::from(ci_high)) as Arc<dyn arrow::array::Array>,
        ),
        (
            "total_samples",
            Arc::new(arrow::array::UInt64Array::from(total_samples_col))
                as Arc<dyn arrow::array::Array>,
        ),
        (
            "evidence_samples",
            Arc::new(arrow::array::UInt64Array::from(evidence_samples_col))
                as Arc<dyn arrow::array::Array>,
        ),
        (
            "seed",
            Arc::new(arrow::array::UInt64Array::from(seed_col)) as Arc<dyn arrow::array::Array>,
        ),
        (
            "confidence",
            Arc::new(arrow::array::Float64Array::from(confidence_col))
                as Arc<dyn arrow::array::Array>,
        ),
        (
            "sampling_method",
            Arc::new(arrow::array::StringArray::from(sampling_method_col))
                as Arc<dyn arrow::array::Array>,
        ),
    ])
    .map_err(|e| XlogError::Execution(format!("Failed to build mc batch: {}", e)))?;

    emit_batch(
        "prob",
        &batch,
        prob_output_as_batch_format(format),
        output_dir,
    )
}

#[cfg(feature = "host-io")]
fn prob_output_as_batch_format(format: ProbOutputFormat) -> OutputFormat {
    match format {
        ProbOutputFormat::Pretty => OutputFormat::Pretty,
        ProbOutputFormat::Csv => OutputFormat::Csv,
        ProbOutputFormat::Arrow => OutputFormat::Arrow,
        ProbOutputFormat::Json => unreachable!("json output is handled before batch emission"),
    }
}

#[cfg(feature = "host-io")]
fn print_prob_exact_json(result: xlog_prob::exact::ExactResult) {
    println!("{{");
    println!("  \"engine\": \"exact_ddnnf\",");
    println!("  \"queries\": [");
    let len = result.query_probs.len();
    for (idx, q) in result.query_probs.into_iter().enumerate() {
        let suffix = if idx + 1 == len { "" } else { "," };
        println!("    {{");
        println!(
            "      \"atom\": \"{}\",",
            json_escape(&atom_to_string(&q.atom))
        );
        println!("      \"prob\": {},", q.prob);
        println!("      \"log_prob\": {}", q.log_prob);
        println!("    }}{}", suffix);
    }
    println!("  ]");
    println!("}}");
}

#[cfg(feature = "host-io")]
fn print_prob_mc_json(result: xlog_prob::mc::McResult) {
    let total_samples = result.total_samples;
    let evidence_samples = result.evidence_samples;
    let seed = result.seed;
    let confidence = result.confidence;
    let sampling_method = result.sampling_method.as_str();
    println!("{{");
    println!("  \"engine\": \"mc\",");
    println!("  \"total_samples\": {},", total_samples);
    println!("  \"evidence_samples\": {},", evidence_samples);
    println!("  \"seed\": {},", seed);
    println!("  \"confidence\": {},", confidence);
    println!("  \"sampling_method\": \"{}\",", sampling_method);
    println!("  \"queries\": [");
    let len = result.query_estimates.len();
    for (idx, q) in result.query_estimates.into_iter().enumerate() {
        let suffix = if idx + 1 == len { "" } else { "," };
        println!("    {{");
        println!(
            "      \"atom\": \"{}\",",
            json_escape(&atom_to_string(&q.atom))
        );
        println!("      \"prob\": {},", q.prob);
        println!("      \"log_prob\": {},", q.log_prob);
        println!("      \"stderr\": {},", q.stderr);
        println!("      \"ci_low\": {},", q.ci_low);
        println!("      \"ci_high\": {},", q.ci_high);
        println!("      \"total_samples\": {},", total_samples);
        println!("      \"evidence_samples\": {}", evidence_samples);
        println!("    }}{}", suffix);
    }
    println!("  ]");
    println!("}}");
}

#[cfg(feature = "host-io")]
fn emit_batch(
    name: &str,
    batch: &arrow::record_batch::RecordBatch,
    format: OutputFormat,
    output_dir: Option<&Path>,
) -> Result<()> {
    match format {
        OutputFormat::Pretty => {
            let formatted = pretty_format_batches(&[batch.clone()])
                .map_err(|e| XlogError::Execution(format!("Pretty print failed: {}", e)))?;
            println!("{}\n{}", name, formatted);
        }
        OutputFormat::Csv => {
            let mut out = Vec::new();
            {
                let mut writer = WriterBuilder::new().build(&mut out);
                writer
                    .write(batch)
                    .map_err(|e| XlogError::Execution(format!("CSV write failed: {}", e)))?;
            }
            println!("{}\n{}", name, String::from_utf8_lossy(&out));
        }
        OutputFormat::Arrow => {
            let dir = output_dir.unwrap_or_else(|| Path::new("."));
            let path = dir.join(format!("{}_prob.arrow", name));
            let mut out = Vec::new();
            let mut writer =
                arrow::ipc::writer::StreamWriter::try_new(&mut out, &batch.schema())
                    .map_err(|e| XlogError::Execution(format!("Arrow writer failed: {}", e)))?;
            writer
                .write(batch)
                .map_err(|e| XlogError::Execution(format!("Arrow write failed: {}", e)))?;
            writer
                .finish()
                .map_err(|e| XlogError::Execution(format!("Arrow finish failed: {}", e)))?;
            std::fs::write(&path, out)
                .map_err(|e| XlogError::Execution(format!("Arrow write file failed: {}", e)))?;
            println!("wrote {}", path.display());
        }
    }
    Ok(())
}

#[cfg(feature = "host-io")]
fn atom_to_string(atom: &xlog_prob::provenance::GroundAtom) -> String {
    use xlog_prob::provenance::Value;

    if atom.args.is_empty() {
        return format!("{}()", atom.predicate);
    }

    let mut out = String::new();
    out.push_str(&atom.predicate);
    out.push('(');
    for (i, arg) in atom.args.iter().enumerate() {
        if i != 0 {
            out.push_str(", ");
        }
        match arg {
            Value::I64(v) => out.push_str(&v.to_string()),
            Value::F64(bits) => out.push_str(&f64::from_bits(*bits).to_string()),
            Value::Symbol(sym) => out.push_str(&symbol::resolve(*sym)),
            Value::String(v) => out.push_str(v),
        }
    }
    out.push(')');
    out
}
