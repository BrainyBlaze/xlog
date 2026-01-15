use clap::{Parser, Subcommand, ValueEnum};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::csv::WriterBuilder;
use arrow::util::pretty::pretty_format_batches;
use xlog_core::{MemoryBudget, Result, XlogError};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_gpu::logic::LogicProgram;

#[derive(Parser)]
#[command(author, version, about = "XLOG CLI")]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Run(RunArgs),
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
}

#[derive(Copy, Clone, ValueEnum)]
enum OutputFormat {
    Pretty,
    Csv,
    Arrow,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run_deterministic(args),
    }
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
        XlogError::Execution(format!(
            "Failed to read {}: {}",
            args.source.display(),
            e
        ))
    })?;

    let program = LogicProgram::compile(&source)?;
    let mut inputs = HashMap::new();
    for (name, path) in parse_inputs(&args.input)? {
        let buf = provider.read_arrow_ipc_stream_file(&path)?;
        inputs.insert(name, buf);
    }

    let result = program.evaluate(provider.clone(), inputs)?;
    emit_logic_results(
        provider.as_ref(),
        &result.queries,
        args.output,
        args.output_dir.as_deref(),
    )
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
