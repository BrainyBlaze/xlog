use clap::{Parser, Subcommand, ValueEnum};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::csv::WriterBuilder;
use arrow::util::pretty::pretty_format_batches;
use xlog_core::{symbol, MemoryBudget, Result, XlogError};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_gpu::logic::LogicProgram;
use xlog_logic::compile::load_modules;
use xlog_logic::{parse_program, rewrite_v085_magic_sets, MagicSetReport, MagicSetStatus};
#[cfg(feature = "host-io")]
use xlog_prob::exact::ExactDdnnfProgram;
#[cfg(feature = "host-io")]
use xlog_prob::exact::GpuConfig;
#[cfg(feature = "host-io")]
use xlog_prob::mc::{McEvalConfig, McProgram};

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
    #[arg(long, value_enum, default_value = "exact_ddnnf")]
    prob_engine: ProbEngineCli,
    #[arg(long, default_value = "10000")]
    samples: usize,
    #[arg(long, default_value = "0")]
    seed: u64,
    #[arg(long, default_value = "0.95")]
    confidence: f64,
    #[arg(long, value_enum, default_value = "pretty")]
    output: OutputFormat,
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
enum ProbEngineCli {
    #[value(name = "exact_ddnnf")]
    ExactDdnnf,
    Mc,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run_deterministic(args),
        Command::Prob(args) => run_probabilistic(args),
        Command::Explain(args) => explain(args),
    }
}

fn explain(args: ExplainArgs) -> Result<()> {
    let source = std::fs::read_to_string(&args.source).map_err(|e| {
        XlogError::Execution(format!("Failed to read {}: {}", args.source.display(), e))
    })?;
    let program = parse_program(&source)?;
    let rewrite = rewrite_v085_magic_sets(&program)?;
    match args.format {
        ExplainFormat::Text => print_magic_text(&rewrite.report),
        ExplainFormat::Json => print_magic_json(&rewrite.report),
        ExplainFormat::Dot => print_magic_dot(&rewrite.report),
    }
    Ok(())
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

fn print_magic_json(report: &MagicSetReport) {
    println!("{{");
    println!("  \"magic_sets\": {{");
    println!(
        "    \"status\": \"{}\",",
        json_escape(magic_status_label(report.status))
    );
    println!(
        "    \"adorned_predicates\": {},",
        json_string_array(&report.adorned_predicates)
    );
    println!(
        "    \"generated_predicates\": {},",
        json_string_array(&report.generated_predicates)
    );
    println!(
        "    \"declined_reasons\": {}",
        json_string_array(&report.declined_reasons)
    );
    println!("  }}");
    println!("}}");
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

        // Validate module imports if any search paths are provided
        if !args.module_path.is_empty() {
            let _ = load_modules(&args.source, args.module_path.clone())
                .map_err(|e| XlogError::Execution(format!("Module resolution failed: {}", e)))?;
        }

        let mut config = GpuConfig::default();
        config.device_ordinal = args.device;
        config.memory_bytes = args.memory_mb * 1024 * 1024;

        match args.prob_engine {
            ProbEngineCli::ExactDdnnf => {
                let prog = ExactDdnnfProgram::compile_source_with_gpu(&source, config)?;
                let result = prog.evaluate()?;
                emit_prob_exact(result, args.output, args.output_dir.as_deref())
            }
            ProbEngineCli::Mc => {
                let prog = McProgram::compile_source_with_gpu(&source, config)?;
                let mut cfg = McEvalConfig::default();
                cfg.samples = args.samples;
                cfg.seed = args.seed;
                cfg.confidence = args.confidence;
                let result = prog.evaluate(cfg)?;
                emit_prob_mc(result, args.output, args.output_dir.as_deref())
            }
        }
    }
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
    format: OutputFormat,
    output_dir: Option<&Path>,
) -> Result<()> {
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

    emit_batch("prob", &batch, format, output_dir)
}

#[cfg(feature = "host-io")]
fn emit_prob_mc(
    result: xlog_prob::mc::McResult,
    format: OutputFormat,
    output_dir: Option<&Path>,
) -> Result<()> {
    let mut atoms = Vec::new();
    let mut probs = Vec::new();
    let mut log_probs = Vec::new();
    let mut stderr = Vec::new();
    let mut ci_low = Vec::new();
    let mut ci_high = Vec::new();
    for q in result.query_estimates {
        atoms.push(atom_to_string(&q.atom));
        probs.push(q.prob);
        log_probs.push(q.log_prob);
        stderr.push(q.stderr);
        ci_low.push(q.ci_low);
        ci_high.push(q.ci_high);
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
    ])
    .map_err(|e| XlogError::Execution(format!("Failed to build mc batch: {}", e)))?;

    emit_batch("prob", &batch, format, output_dir)
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
