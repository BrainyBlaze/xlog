//! Run a `.xlog` program end-to-end on a CUDA device.
//!
//! Usage:
//! `cargo run -p xlog-integration --release --bin xlog_run -- <file.xlog> [--device N] [--memory-mb MB] [--limit N]`

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use xlog_core::{symbol, MemoryBudget, Result, ScalarType, XlogError};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::{
    compile::load_modules, expand_program_functions, parse_program, BodyLiteral, Compiler, Query,
    Term,
};
use xlog_runtime::Executor;

fn usage() -> String {
    [
        "Usage:",
        "  xlog_run <file.xlog> [--device N] [--memory-mb MB] [--limit N]",
        "",
        "Examples:",
        "  cargo run -p xlog-integration --release --bin xlog_run -- examples/xlog/00-basics/01_tc_reachability.xlog",
        "  cargo run -p xlog-integration --release --bin xlog_run -- examples/xlog/70-aggregates/01_out_degree_count.xlog --limit 20",
    ]
    .join("\n")
}

fn module_search_paths(entry_path: &Path) -> Vec<std::path::PathBuf> {
    let Some(base_dir) = entry_path.parent() else {
        return vec![Path::new(".").to_path_buf()];
    };

    let mut paths = Vec::new();
    for dir in base_dir.ancestors() {
        if !paths.iter().any(|p| p == dir) {
            paths.push(dir.to_path_buf());
        }
        if dir.join(".git").exists() {
            break;
        }
    }
    paths
}

fn parse_args() -> Result<(String, usize, usize, usize)> {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        return Err(XlogError::Execution(usage()));
    };

    let mut device: usize = 0;
    let mut memory_mb: usize = 1024;
    let mut limit: usize = 50;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => {
                let v = args.next().ok_or_else(|| {
                    XlogError::Execution("Missing value for --device".to_string())
                })?;
                device = v
                    .parse()
                    .map_err(|_| XlogError::Execution(format!("Invalid --device value: {}", v)))?;
            }
            "--memory-mb" => {
                let v = args.next().ok_or_else(|| {
                    XlogError::Execution("Missing value for --memory-mb".to_string())
                })?;
                memory_mb = v.parse().map_err(|_| {
                    XlogError::Execution(format!("Invalid --memory-mb value: {}", v))
                })?;
            }
            "--limit" => {
                let v = args
                    .next()
                    .ok_or_else(|| XlogError::Execution("Missing value for --limit".to_string()))?;
                limit = v
                    .parse()
                    .map_err(|_| XlogError::Execution(format!("Invalid --limit value: {}", v)))?;
            }
            "--help" | "-h" => return Err(XlogError::Execution(usage())),
            other => {
                return Err(XlogError::Execution(format!(
                    "Unknown argument: {}\n\n{}",
                    other,
                    usage()
                )));
            }
        }
    }

    Ok((path, device, memory_mb, limit))
}

fn push_term_bytes(out: &mut Vec<u8>, term: &Term, typ: ScalarType) -> Result<()> {
    match (typ, term) {
        (ScalarType::U32, Term::Integer(v)) => {
            if *v < 0 || *v > u32::MAX as i64 {
                return Err(XlogError::Execution(format!("u32 out of range: {}", v)));
            }
            out.extend_from_slice(&(*v as u32).to_le_bytes());
        }
        (ScalarType::U64, Term::Integer(v)) => {
            if *v < 0 {
                return Err(XlogError::Execution(format!(
                    "u64 out of range (negative): {}",
                    v
                )));
            }
            out.extend_from_slice(&(*v as u64).to_le_bytes());
        }
        (ScalarType::I32, Term::Integer(v)) => {
            if *v < i32::MIN as i64 || *v > i32::MAX as i64 {
                return Err(XlogError::Execution(format!("i32 out of range: {}", v)));
            }
            out.extend_from_slice(&(*v as i32).to_le_bytes());
        }
        (ScalarType::I64, Term::Integer(v)) => {
            out.extend_from_slice(&v.to_le_bytes());
        }
        (ScalarType::F32, Term::Float(v)) => {
            out.extend_from_slice(&(*v as f32).to_le_bytes());
        }
        (ScalarType::F64, Term::Float(v)) => {
            out.extend_from_slice(&v.to_le_bytes());
        }
        (ScalarType::F32, Term::Integer(v)) => {
            out.extend_from_slice(&(*v as f32).to_le_bytes());
        }
        (ScalarType::F64, Term::Integer(v)) => {
            out.extend_from_slice(&(*v as f64).to_le_bytes());
        }
        (ScalarType::Bool, Term::Integer(v)) => {
            let b = match *v {
                0 => 0u8,
                1 => 1u8,
                other => {
                    return Err(XlogError::Execution(format!(
                        "bool expects 0/1, got {}",
                        other
                    )));
                }
            };
            out.push(b);
        }
        (ScalarType::Bool, Term::Symbol(id)) => {
            let s = symbol::resolve(*id);
            if s == "true" || s == "false" {
                out.push(if s == "true" { 1u8 } else { 0u8 });
            } else {
                return Err(XlogError::Execution(format!(
                    "Expected boolean symbol 'true' or 'false', got '{}'",
                    s
                )));
            }
        }
        (ScalarType::Symbol, Term::String(s)) => {
            out.extend_from_slice(&symbol::intern(s).to_le_bytes());
        }
        (ScalarType::Symbol, Term::Symbol(id)) => {
            // Symbol is already interned, just use the ID directly
            out.extend_from_slice(&id.to_le_bytes());
        }
        (_, Term::Variable(v)) => {
            return Err(XlogError::Execution(format!(
                "Fact cannot contain variable {}",
                v
            )));
        }
        (_, Term::Anonymous) => {
            return Err(XlogError::Execution(
                "Fact cannot contain anonymous wildcard '_'".to_string(),
            ));
        }
        (_, Term::Aggregate(_)) => {
            return Err(XlogError::Execution(
                "Fact cannot contain aggregate".to_string(),
            ));
        }
        (expected, got) => {
            return Err(XlogError::Execution(format!(
                "Type mismatch in fact: expected {:?}, got {:?}",
                expected, got
            )));
        }
    }

    Ok(())
}

fn query_output_vars(Query { atom }: &Query) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for term in &atom.terms {
        if let Term::Variable(name) = term {
            if seen.insert(name.as_str()) {
                out.push(name.clone());
            }
        }
    }
    out
}

fn format_term(term: &Term) -> String {
    match term {
        Term::Variable(v) => v.clone(),
        Term::Anonymous => "_".to_string(),
        Term::Integer(i) => i.to_string(),
        Term::Float(f) => f.to_string(),
        Term::String(s) => format!("{:?}", s),
        Term::Symbol(id) => symbol::resolve(*id),
        Term::Aggregate(a) => format!("{:?}({})", a.op, a.variable),
    }
}

fn format_query(q: &Query) -> String {
    let args = q
        .atom
        .terms
        .iter()
        .map(format_term)
        .collect::<Vec<_>>()
        .join(", ");
    format!("?- {}({}).", q.atom.predicate, args)
}

fn format_constraint(body: &[BodyLiteral]) -> String {
    let lits = body
        .iter()
        .map(|lit| match lit {
            BodyLiteral::Positive(a) => {
                let args = a
                    .terms
                    .iter()
                    .map(format_term)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}({})", a.predicate, args)
            }
            BodyLiteral::Negated(a) => {
                let args = a
                    .terms
                    .iter()
                    .map(format_term)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("not {}({})", a.predicate, args)
            }
            BodyLiteral::Comparison(c) => format!("{:?} {:?} {:?}", c.left, c.op, c.right),
            BodyLiteral::IsExpr(is) => format!("{} is {:?}", is.target, is.expr),
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(":- {}.", lits)
}

fn decode_column_to_strings(
    provider: &CudaKernelProvider,
    buf: &xlog_cuda::CudaBuffer,
    col_idx: usize,
) -> Result<Vec<String>> {
    let num_rows = buf.num_rows() as usize;
    let typ = buf
        .schema()
        .column_type(col_idx)
        .ok_or_else(|| XlogError::Execution(format!("Missing type for column {}", col_idx)))?;
    let size = typ.size_bytes();
    let col = buf
        .column(col_idx)
        .ok_or_else(|| XlogError::Execution(format!("Missing column {}", col_idx)))?;

    let mut bytes = vec![0u8; num_rows * size];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(col, &mut bytes)
        .map_err(|e| XlogError::Kernel(format!("Failed to download column: {}", e)))?;

    let mut out = Vec::with_capacity(num_rows);
    match typ {
        ScalarType::U32 => {
            for chunk in bytes.chunks_exact(4) {
                out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).to_string());
            }
        }
        ScalarType::Symbol => {
            for chunk in bytes.chunks_exact(4) {
                let id = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                // Avoid crashing the example runner when a relation contains a non-interned symbol ID.
                // Keep output printable while preserving the raw identifier for debugging.
                let resolved = symbol::resolve_checked(id).unwrap_or_else(|| format!("sym#{}", id));
                out.push(resolved);
            }
        }
        ScalarType::I32 => {
            for chunk in bytes.chunks_exact(4) {
                out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).to_string());
            }
        }
        ScalarType::F32 => {
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).to_string());
            }
        }
        ScalarType::U64 => {
            for chunk in bytes.chunks_exact(8) {
                out.push(
                    u64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                    .to_string(),
                );
            }
        }
        ScalarType::I64 => {
            for chunk in bytes.chunks_exact(8) {
                out.push(
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                    .to_string(),
                );
            }
        }
        ScalarType::F64 => {
            for chunk in bytes.chunks_exact(8) {
                out.push(
                    f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                    .to_string(),
                );
            }
        }
        ScalarType::Bool => {
            for b in bytes {
                out.push(if b != 0 { "true" } else { "false" }.to_string());
            }
        }
    }

    Ok(out)
}

fn main() -> Result<()> {
    let (path, device_id, memory_mb, limit) = parse_args()?;
    let source = fs::read_to_string(&path)
        .map_err(|e| XlogError::Execution(format!("Failed to read {}: {}", path, e)))?;

    // Parse and resolve module imports
    let entry_path = Path::new(&path);

    let mut program = parse_program(&source)?;

    // If the program has imports, resolve them using the module system
    if !program.imports.is_empty() {
        let resolver = load_modules(entry_path, module_search_paths(entry_path))
            .map_err(|e| XlogError::Compilation(format!("Module resolution failed: {}", e)))?;
        program = resolver
            .merge_imports(program)
            .map_err(|e| XlogError::Compilation(format!("Module merge failed: {}", e)))?;
    }

    // Expand user-defined functions (inline UDF calls in rules)
    if !program.functions.is_empty() {
        let max_depth = program.directives.max_recursion_depth.unwrap_or(1000);
        program = expand_program_functions(&program, max_depth)
            .map_err(|e| XlogError::Compilation(format!("Function expansion failed: {}", e)))?;
    }

    let mut compiler = Compiler::new();
    let plan = compiler.compile_program(&program)?;

    let device = Arc::new(CudaDevice::new(device_id).map_err(|e| {
        XlogError::Execution(format!("Failed to open CUDA device {}: {}", device_id, e))
    })?);
    let budget = MemoryBudget::with_limit((memory_mb as u64) * 1024 * 1024);
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    let provider = Arc::new(CudaKernelProvider::new(device, memory)?);

    let mut executor = Executor::new(provider.clone());

    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }

    // Load EDB facts (one GPU buffer per predicate).
    let mut rows_by_pred: HashMap<String, Vec<Vec<Term>>> = HashMap::new();
    for fact in program.facts() {
        rows_by_pred
            .entry(fact.head.predicate.clone())
            .or_default()
            .push(fact.head.terms.clone());
    }

    for (pred, rows) in rows_by_pred {
        let schema = compiler.schemas().get(&pred).ok_or_else(|| {
            XlogError::Execution(format!("Missing inferred schema for predicate {}", pred))
        })?;

        if rows.iter().any(|r| r.len() != schema.arity()) {
            return Err(XlogError::Execution(format!(
                "Fact arity mismatch for {} (expected {} columns)",
                pred,
                schema.arity()
            )));
        }

        let mut columns: Vec<Vec<u8>> = vec![Vec::new(); schema.arity()];
        for row in &rows {
            for (col_idx, term) in row.iter().enumerate() {
                let typ = schema.column_type(col_idx).unwrap_or(ScalarType::U64);
                push_term_bytes(&mut columns[col_idx], term, typ)?;
            }
        }

        let slices: Vec<&[u8]> = columns.iter().map(|c| c.as_slice()).collect();
        let buf = provider.create_buffer_from_slices(&slices, schema.clone())?;
        executor.store_mut().put(&pred, buf);
    }

    // Ensure every relation has an initial buffer so scans never fail.
    for (name, schema) in compiler.schemas() {
        if !executor.store().contains(name) {
            executor
                .store_mut()
                .put(name, provider.create_empty_buffer(schema.clone())?);
        }
    }

    executor.execute_plan(&plan)?;

    // Enforce constraints (if any): any non-empty `__xlog_constraint_i` is a violation.
    let mut violated: Vec<usize> = Vec::new();
    for i in 0..program.constraints.len() {
        let name = format!("__xlog_constraint_{}", i);
        let buf = executor.store().get(&name).ok_or_else(|| {
            XlogError::Execution(format!(
                "Missing constraint result relation {} (compiler bug?)",
                name
            ))
        })?;
        if !buf.is_empty() && buf.num_rows() > 0 {
            violated.push(i);
        }
    }

    if !violated.is_empty() {
        eprintln!("Constraint violations:");
        for &i in &violated {
            eprintln!(
                "  [{}] {}",
                i,
                format_constraint(&program.constraints[i].body)
            );
        }
        return Err(XlogError::Execution(format!(
            "{} constraint(s) violated",
            violated.len()
        )));
    }

    // Print queries (if any).
    for (i, query) in program.queries.iter().enumerate() {
        let name = format!("__xlog_query_{}", i);
        let buf = executor.store().get(&name).ok_or_else(|| {
            XlogError::Execution(format!(
                "Missing query result relation {} (compiler bug?)",
                name
            ))
        })?;

        println!("{}", format_query(query));

        let vars = query_output_vars(query);
        if vars.is_empty() {
            println!("  {}", if buf.num_rows() > 0 { "YES" } else { "NO" });
            continue;
        }

        if buf.is_empty() {
            println!("  (0 rows)");
            continue;
        }

        let mut columns: Vec<Vec<String>> = Vec::with_capacity(buf.arity());
        for col_idx in 0..buf.arity() {
            columns.push(decode_column_to_strings(&provider, buf, col_idx)?);
        }

        let rows = buf.num_rows() as usize;
        let shown = rows.min(limit);
        for row_idx in 0..shown {
            let mut parts = Vec::with_capacity(vars.len());
            for (col_idx, var) in vars.iter().enumerate() {
                parts.push(format!("{}={}", var, columns[col_idx][row_idx]));
            }
            println!("  {}", parts.join(", "));
        }
        if shown < rows {
            println!("  ... ({} more rows; use --limit)", rows - shown);
        }
    }

    Ok(())
}
