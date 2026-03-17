//! GPU buffer allocation and sample management.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use cudarc::driver::{CudaView, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, ScalarType, Schema, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{mc_eval_kernels, MC_EVAL_MODULE};
use xlog_cuda::{CudaBuffer, CudaKernelProvider};
use xlog_logic::ast::{
    AggOp, Atom, BodyLiteral, Evidence, PredDecl, ProbFact, ProbQuery, Program, Term,
};
use xlog_runtime::Executor;

use crate::provenance::{atom_key_from_ground_atom, validate_prob, GroundAtom, Value};

use super::{
    AdDecisionDevice, AdSpec, AdTableDevice, GpuMcPlan, McProgram, ProbFactSpec, ProbTableDevice,
};

/// Plan for per-sample executor reset.
///
/// Instead of cloning/restoring the entire store each sample, we classify
/// relations as either *preserve* (deterministic base facts that are never
/// overwritten) or *clear* (everything else).  Preserved relations stay
/// in-place across samples; cleared relations are re-created as empty
/// buffers.
///
/// A predicate that has **both** deterministic facts and dynamic writes
/// (probabilistic / AD / rule-head) is placed in `clear`, and its
/// deterministic base facts are re-loaded each sample from `reload_base`.
pub(super) struct McSampleResetPlan {
    /// Relations to keep untouched (pure deterministic base facts).
    pub(super) preserve: Vec<String>,
    /// Relations to clear to empty buffers each sample (dynamic/sampled/rule-derived).
    pub(super) clear: Vec<(String, Schema)>,
    /// Deterministic base-fact buffers for predicates that are both
    /// deterministic AND dynamic.  These are cloned into the store each
    /// sample after clearing, before `build_sample_buffers` merges sampled
    /// rows on top.
    pub(super) reload_base: Vec<(String, CudaBuffer)>,
}

pub(super) fn upload_slice<T: cudarc::driver::DeviceRepr>(
    provider: &Arc<CudaKernelProvider>,
    src: &[T],
    dst: &mut TrackedCudaSlice<T>,
    label: &str,
) -> Result<()> {
    if src.is_empty() {
        return Ok(());
    }
    provider
        .device()
        .inner()
        .htod_sync_copy_into(src, dst)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload {}: {}", label, e)))
}

pub(super) fn load_deterministic_facts(
    program: &Program,
    schemas: &HashMap<String, Schema>,
    provider: &Arc<CudaKernelProvider>,
    executor: &mut Executor,
) -> Result<()> {
    let mut rows_by_pred: HashMap<String, Vec<Vec<Value>>> = HashMap::new();
    for fact in program.facts() {
        let atom = atom_key_from_ground_atom(&fact.head)?;
        rows_by_pred
            .entry(atom.predicate.clone())
            .or_default()
            .push(atom.args);
    }

    for (pred, rows) in rows_by_pred {
        let schema = schemas.get(&pred).ok_or_else(|| {
            XlogError::Execution(format!(
                "Missing schema for deterministic predicate {}",
                pred
            ))
        })?;
        let buffer = build_buffer_from_rows(provider, schema, &rows)?;
        let deduped = dedup_relation(provider, &buffer)?;
        executor.put_relation(&pred, deduped);
    }

    Ok(())
}

/// Build a reset plan from the compiled MC program and the GPU execution plan.
///
/// A predicate is "preserve-safe" iff it:
/// - has at least one deterministic fact (from `program.facts()`)
/// - is NOT a probabilistic fact predicate (`self.prob_facts`)
/// - is NOT an annotated disjunction choice predicate (`self.annotated_disjunctions`)
/// - is NOT a rule head predicate (`program.rules`, excluding facts)
/// - is NOT a query temp relation (`__xlog_query_*`)
pub(super) fn build_sample_reset_plan(
    gpu_plan: &GpuMcPlan,
    mc_program: &McProgram,
    provider: &Arc<CudaKernelProvider>,
    executor: &Executor,
) -> Result<McSampleResetPlan> {
    // Collect the set of predicates that have deterministic facts.
    let mut det_preds: HashSet<String> = HashSet::new();
    for fact in gpu_plan.program.facts() {
        det_preds.insert(fact.head.predicate.clone());
    }

    // Collect predicates that are "dynamic" — written by sampling or rules.
    let mut dynamic_preds: HashSet<String> = HashSet::new();

    // Probabilistic fact predicates.
    for pf in &mc_program.prob_facts {
        dynamic_preds.insert(pf.atom.predicate.clone());
    }

    // Annotated disjunction choice predicates.
    for ad in &mc_program.annotated_disjunctions {
        for choice in &ad.choices {
            dynamic_preds.insert(choice.predicate.clone());
        }
    }

    // Rule head predicates (proper rules only, not facts).
    for rule in gpu_plan.program.proper_rules() {
        dynamic_preds.insert(rule.head.predicate.clone());
    }

    // Query temp relations.
    for (name, _) in &gpu_plan.schemas {
        if name.starts_with("__xlog_query_") {
            dynamic_preds.insert(name.clone());
        }
    }

    let mut preserve: Vec<String> = Vec::new();
    let mut clear: Vec<(String, Schema)> = Vec::new();
    let mut reload_base: Vec<(String, CudaBuffer)> = Vec::new();

    for (name, schema) in &gpu_plan.schemas {
        let is_det = det_preds.contains(name);
        let is_dyn = dynamic_preds.contains(name);

        if is_det && !is_dyn {
            // Pure deterministic — preserve in-place.
            preserve.push(name.clone());
        } else {
            // Dynamic (possibly also deterministic) — clear each sample.
            clear.push((name.clone(), schema.clone()));

            if is_det && is_dyn {
                // Has both deterministic and dynamic facts: snapshot the
                // deterministic base buffer so we can re-load it each sample.
                let buf = executor.store().get(name).ok_or_else(|| {
                    XlogError::Execution(format!("Missing relation {} for reload snapshot", name))
                })?;
                let cloned = if buf.is_empty() {
                    provider.create_empty_buffer(schema.clone())?
                } else {
                    clone_buffer_device(provider, buf)?
                };
                reload_base.push((name.clone(), cloned));
            }
        }
    }

    Ok(McSampleResetPlan {
        preserve,
        clear,
        reload_base,
    })
}

pub(super) fn clone_buffer_device(
    provider: &Arc<CudaKernelProvider>,
    buffer: &CudaBuffer,
) -> Result<CudaBuffer> {
    if buffer.is_empty() {
        return provider.create_empty_buffer(buffer.schema().clone());
    }

    let mut result_columns = Vec::with_capacity(buffer.arity());
    for col_idx in 0..buffer.arity() {
        let col_type_size = buffer
            .schema()
            .column_type(col_idx)
            .map(|t| t.size_bytes())
            .unwrap_or(4);
        let bytes = (buffer.num_rows() as usize) * col_type_size;
        let Some(src_col) = buffer.column(col_idx) else {
            continue;
        };
        let mut dst_col = provider.memory().alloc::<u8>(bytes)?;
        if bytes > 0 {
            provider
                .device()
                .inner()
                .dtod_copy(src_col, &mut dst_col)
                .map_err(|e| {
                    XlogError::Execution(format!("Failed to clone column on device: {}", e))
                })?;
        }
        result_columns.push(dst_col.into());
    }

    let mut d_num_rows = provider.memory().alloc::<u32>(1)?;
    provider
        .device()
        .inner()
        .dtod_copy(buffer.num_rows_device(), &mut d_num_rows)
        .map_err(|e| XlogError::Execution(format!("Failed to copy row count: {}", e)))?;
    Ok(CudaBuffer::from_columns(
        result_columns,
        buffer.num_rows(),
        d_num_rows,
        buffer.schema().clone(),
    ))
}

fn build_zero_arity_buffer(
    provider: &Arc<CudaKernelProvider>,
    row_count: u32,
    schema: &Schema,
) -> Result<CudaBuffer> {
    let mut d_num_rows = provider.memory().alloc::<u32>(1)?;
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[row_count], &mut d_num_rows)
        .map_err(|e| XlogError::Kernel(format!("Failed to set row count: {}", e)))?;
    Ok(CudaBuffer::from_columns(
        Vec::new(),
        row_count as u64,
        d_num_rows,
        schema.clone(),
    ))
}

pub(super) fn dedup_relation(
    provider: &Arc<CudaKernelProvider>,
    buffer: &CudaBuffer,
) -> Result<CudaBuffer> {
    let rows = buffer.num_rows();
    if rows == 0 {
        return provider.create_empty_buffer(buffer.schema().clone());
    }
    let key_cols: Vec<usize> = (0..buffer.arity()).collect();
    provider.dedup(buffer, &key_cols)
}

pub(super) fn build_prob_tables_device(
    program: &McProgram,
    provider: &Arc<CudaKernelProvider>,
    schemas: &HashMap<String, Schema>,
) -> Result<(Vec<ProbTableDevice>, Vec<AdTableDevice>, AdDecisionDevice)> {
    let mut prob_rows_by_pred: HashMap<String, Vec<(Vec<Value>, u32)>> = HashMap::new();
    for pf in &program.prob_facts {
        prob_rows_by_pred
            .entry(pf.atom.predicate.clone())
            .or_default()
            .push((pf.atom.args.clone(), pf.var_idx as u32));
    }

    let mut prob_tables: Vec<ProbTableDevice> = Vec::new();
    for (pred, rows) in prob_rows_by_pred {
        let mut tuples: Vec<Vec<Value>> = Vec::with_capacity(rows.len());
        let mut var_idx: Vec<u32> = Vec::with_capacity(rows.len());
        for (tuple, idx) in rows {
            tuples.push(tuple);
            var_idx.push(idx);
        }

        let schema = match schemas.get(&pred) {
            Some(schema) => schema.clone(),
            None => infer_schema_from_values(&tuples)?,
        };

        let buffer = build_buffer_from_rows(provider, &schema, &tuples)?;
        let mut d_var_idx = provider.memory().alloc::<u32>(var_idx.len())?;
        upload_slice(provider, &var_idx, &mut d_var_idx, "prob var indices")?;

        prob_tables.push(ProbTableDevice {
            predicate: pred,
            buffer,
            var_idx: d_var_idx,
        });
    }

    #[derive(Debug)]
    struct AdRow {
        args: Vec<Value>,
        offset: u32,
        len: u32,
        pos: u32,
    }

    let mut decision_vars_flat: Vec<u32> = Vec::new();
    let mut ad_rows_by_pred: HashMap<String, Vec<AdRow>> = HashMap::new();

    for ad in &program.annotated_disjunctions {
        let offset = decision_vars_flat.len() as u32;
        let len = ad.decision_vars.len() as u32;
        decision_vars_flat.extend(ad.decision_vars.iter().map(|v| *v as u32));

        let choices_len = ad.choices.len();
        for (idx, atom) in ad.choices.iter().enumerate() {
            let pos = if ad.has_none {
                idx as u32
            } else if idx + 1 == choices_len {
                len
            } else {
                idx as u32
            };
            ad_rows_by_pred
                .entry(atom.predicate.clone())
                .or_default()
                .push(AdRow {
                    args: atom.args.clone(),
                    offset,
                    len,
                    pos,
                });
        }
    }

    let mut ad_tables: Vec<AdTableDevice> = Vec::new();
    for (pred, rows) in ad_rows_by_pred {
        let mut tuples: Vec<Vec<Value>> = Vec::with_capacity(rows.len());
        let mut offsets: Vec<u32> = Vec::with_capacity(rows.len());
        let mut lengths: Vec<u32> = Vec::with_capacity(rows.len());
        let mut positions: Vec<u32> = Vec::with_capacity(rows.len());

        for row in rows {
            tuples.push(row.args);
            offsets.push(row.offset);
            lengths.push(row.len);
            positions.push(row.pos);
        }

        let schema = match schemas.get(&pred) {
            Some(schema) => schema.clone(),
            None => infer_schema_from_values(&tuples)?,
        };

        let buffer = build_buffer_from_rows(provider, &schema, &tuples)?;
        let mut d_offsets = provider.memory().alloc::<u32>(offsets.len())?;
        let mut d_lengths = provider.memory().alloc::<u32>(lengths.len())?;
        let mut d_positions = provider.memory().alloc::<u32>(positions.len())?;
        upload_slice(provider, &offsets, &mut d_offsets, "AD offsets")?;
        upload_slice(provider, &lengths, &mut d_lengths, "AD lengths")?;
        upload_slice(provider, &positions, &mut d_positions, "AD positions")?;

        ad_tables.push(AdTableDevice {
            predicate: pred,
            buffer,
            decision_offsets: d_offsets,
            decision_lengths: d_lengths,
            choice_positions: d_positions,
        });
    }

    let mut d_decision_vars = provider.memory().alloc::<u32>(decision_vars_flat.len())?;
    upload_slice(
        provider,
        &decision_vars_flat,
        &mut d_decision_vars,
        "AD decision vars",
    )?;

    Ok((
        prob_tables,
        ad_tables,
        AdDecisionDevice {
            decision_vars: d_decision_vars,
        },
    ))
}

pub(super) fn build_sample_buffers(
    provider: &Arc<CudaKernelProvider>,
    sample_bits: &CudaView<'_, u8>,
    prob_tables: &[ProbTableDevice],
    ad_tables: &[AdTableDevice],
    ad_decisions: &AdDecisionDevice,
) -> Result<Vec<(String, CudaBuffer)>> {
    if sample_bits.len() == 0 && (!prob_tables.is_empty() || ad_decisions.decision_vars.len() > 0) {
        return Err(XlogError::Execution(
            "MC sample bits empty but probabilistic variables exist".to_string(),
        ));
    }

    let device = provider.device().inner();
    let mut out: Vec<(String, CudaBuffer)> = Vec::new();

    let block_size = 256u32;

    for table in prob_tables {
        if table.buffer.is_empty() {
            continue;
        }
        let n_u64 = table.buffer.num_rows();
        let n: u32 = n_u64.try_into().map_err(|_| {
            XlogError::Execution(format!(
                "Prob table {} rows {} exceed u32",
                table.predicate, n_u64
            ))
        })?;

        let mut d_mask = provider.memory().alloc::<u8>(n as usize)?;
        let num_blocks = (n + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernel = device
            .get_func(MC_EVAL_MODULE, mc_eval_kernels::MC_EVAL_MASK_VAR)
            .ok_or_else(|| XlogError::Kernel("mc_eval_mask_var kernel not found".to_string()))?;

        // SAFETY: mc_eval_mask_var(sample_bits, var_idx, n, out_mask)
        unsafe {
            kernel
                .clone()
                .launch(config, (sample_bits, &table.var_idx, n, &mut d_mask))
        }
        .map_err(|e| XlogError::Kernel(format!("mc_eval_mask_var failed: {}", e)))?;

        let filtered = provider.compact_buffer_by_device_mask_counted(&table.buffer, &d_mask)?;
        let deduped = dedup_relation(provider, &filtered)?;
        out.push((table.predicate.clone(), deduped));
    }

    for table in ad_tables {
        if table.buffer.is_empty() {
            continue;
        }
        let n_u64 = table.buffer.num_rows();
        let n: u32 = n_u64.try_into().map_err(|_| {
            XlogError::Execution(format!(
                "AD table {} rows {} exceed u32",
                table.predicate, n_u64
            ))
        })?;

        let mut d_mask = provider.memory().alloc::<u8>(n as usize)?;
        let num_blocks = (n + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernel = device
            .get_func(MC_EVAL_MODULE, mc_eval_kernels::MC_EVAL_MASK_AD)
            .ok_or_else(|| {
                XlogError::Kernel("mc_eval_mask_ad_choice kernel not found".to_string())
            })?;

        // SAFETY: mc_eval_mask_ad_choice(sample_bits, decision_vars, offsets, lengths, positions, n, out_mask)
        unsafe {
            kernel.clone().launch(
                config,
                (
                    sample_bits,
                    &ad_decisions.decision_vars,
                    &table.decision_offsets,
                    &table.decision_lengths,
                    &table.choice_positions,
                    n,
                    &mut d_mask,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("mc_eval_mask_ad_choice failed: {}", e)))?;

        let filtered = provider.compact_buffer_by_device_mask_counted(&table.buffer, &d_mask)?;
        let deduped = dedup_relation(provider, &filtered)?;
        out.push((table.predicate.clone(), deduped));
    }

    Ok(out)
}

pub(super) fn build_buffer_from_rows(
    provider: &Arc<CudaKernelProvider>,
    schema: &Schema,
    rows: &[Vec<Value>],
) -> Result<CudaBuffer> {
    if schema.arity() == 0 {
        for row in rows {
            if !row.is_empty() {
                return Err(XlogError::Execution(
                    "Zero-arity buffer row should be empty".to_string(),
                ));
            }
        }
        if rows.is_empty() {
            return provider.create_empty_buffer(schema.clone());
        }
        let row_count = u32::try_from(rows.len()).map_err(|_| {
            XlogError::Execution(format!(
                "Row count {} exceeds u32::MAX for zero-arity buffer",
                rows.len()
            ))
        })?;
        return build_zero_arity_buffer(provider, row_count, schema);
    }

    if rows.is_empty() {
        return provider.create_empty_buffer(schema.clone());
    }

    let mut columns: Vec<Vec<u8>> = Vec::with_capacity(schema.arity());
    for col_idx in 0..schema.arity() {
        let col_size = schema
            .column_type(col_idx)
            .map(|t| t.size_bytes())
            .unwrap_or(4);
        columns.push(Vec::with_capacity(rows.len() * col_size));
    }

    for row in rows {
        if row.len() != schema.arity() {
            return Err(XlogError::Execution(format!(
                "Row arity {} does not match schema arity {}",
                row.len(),
                schema.arity()
            )));
        }
        for (idx, value) in row.iter().enumerate() {
            let col_type = schema
                .column_type(idx)
                .ok_or_else(|| XlogError::Execution(format!("Missing column type for {}", idx)))?;
            push_value_bytes(&mut columns[idx], col_type, value)?;
        }
    }

    let slices: Vec<&[u8]> = columns.iter().map(|c| c.as_slice()).collect();
    provider.create_buffer_from_slices(&slices, schema.clone())
}

pub(super) fn augment_schemas_for_program(
    program: &Program,
    schemas: &mut HashMap<String, Schema>,
) {
    for fact in program.facts() {
        ensure_schema_for_atom(&fact.head, schemas);
    }

    for rule in &program.rules {
        for lit in &rule.body {
            match lit {
                BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                    ensure_schema_for_atom(atom, schemas);
                }
                BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) => {}
            }
        }
    }

    for pf in &program.prob_facts {
        ensure_schema_for_atom(&pf.atom, schemas);
    }

    for ad in &program.annotated_disjunctions {
        for choice in &ad.choices {
            ensure_schema_for_atom(&choice.atom, schemas);
        }
    }

    for ProbQuery { atom } in &program.prob_queries {
        ensure_schema_for_atom(atom, schemas);
    }

    for Evidence { atom, .. } in &program.evidence {
        ensure_schema_for_atom(atom, schemas);
    }
}

pub(super) fn ensure_predicate_decls(program: &mut Program) -> Result<()> {
    let mut declared: HashMap<String, Vec<ScalarType>> = HashMap::new();
    for pred in &program.predicates {
        declared.insert(pred.name.clone(), pred.types.clone());
    }

    let mut inferred: HashMap<String, Vec<ScalarType>> = HashMap::new();

    let mut record_atom = |atom: &Atom| {
        let types: Vec<ScalarType> = atom.terms.iter().map(infer_term_scalar_type).collect();
        match inferred.get(&atom.predicate) {
            Some(existing) if *existing != types => Err(XlogError::Compilation(format!(
                "Inconsistent predicate types for {}",
                atom.predicate
            ))),
            Some(_) => Ok(()),
            None => {
                inferred.insert(atom.predicate.clone(), types);
                Ok(())
            }
        }
    };

    for fact in program.facts() {
        record_atom(&fact.head)?;
    }

    for rule in &program.rules {
        record_atom(&rule.head)?;
        for lit in &rule.body {
            match lit {
                BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                    record_atom(atom)?;
                }
                BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) => {}
            }
        }
    }

    for pf in &program.prob_facts {
        record_atom(&pf.atom)?;
    }

    for ad in &program.annotated_disjunctions {
        for choice in &ad.choices {
            record_atom(&choice.atom)?;
        }
    }

    for ProbQuery { atom } in &program.prob_queries {
        record_atom(atom)?;
    }
    for Evidence { atom, .. } in &program.evidence {
        record_atom(atom)?;
    }

    for (pred, types) in inferred {
        if let Some(existing) = declared.get(&pred) {
            if existing != &types {
                return Err(XlogError::Compilation(format!(
                    "Predicate {} declared with {:?} but inferred {:?}",
                    pred, existing, types
                )));
            }
            continue;
        }
        program.predicates.push(PredDecl {
            name: pred,
            types,
            is_private: false,
        });
    }

    Ok(())
}

fn ensure_schema_for_atom(atom: &Atom, schemas: &mut HashMap<String, Schema>) {
    if schemas.contains_key(&atom.predicate) {
        return;
    }

    let columns: Vec<(String, ScalarType)> = atom
        .terms
        .iter()
        .enumerate()
        .map(|(i, term)| (format!("c{}", i), infer_term_scalar_type(term)))
        .collect();
    schemas.insert(atom.predicate.clone(), Schema::new(columns));
}

fn infer_term_scalar_type(term: &Term) -> ScalarType {
    match term {
        Term::Variable(_) | Term::Anonymous => ScalarType::U64,
        Term::Integer(i) => {
            if *i >= 0 && *i <= u32::MAX as i64 {
                ScalarType::U32
            } else {
                ScalarType::I64
            }
        }
        Term::Float(_) => ScalarType::F64,
        Term::String(_) | Term::Symbol(_) => ScalarType::Symbol,
        Term::Aggregate(agg) => match agg.op {
            AggOp::Count => ScalarType::U32,
            AggOp::Sum => ScalarType::U64,
            AggOp::Min | AggOp::Max => ScalarType::U32,
            AggOp::LogSumExp => ScalarType::F64,
        },
    }
}

fn infer_schema_from_values(rows: &[Vec<Value>]) -> Result<Schema> {
    if rows.is_empty() {
        return Err(XlogError::Execution(
            "Cannot infer schema from empty rows".to_string(),
        ));
    }
    let arity = rows[0].len();
    let mut types: Vec<Option<ScalarType>> = vec![None; arity];

    for row in rows {
        if row.len() != arity {
            return Err(XlogError::Execution(format!(
                "Row arity {} does not match inferred arity {}",
                row.len(),
                arity
            )));
        }
        for (idx, value) in row.iter().enumerate() {
            let ty = scalar_type_from_value(value);
            match types[idx] {
                Some(existing) if existing != ty => {
                    return Err(XlogError::Execution(format!(
                        "Inconsistent types for column {}: {:?} vs {:?}",
                        idx, existing, ty
                    )))
                }
                None => types[idx] = Some(ty),
                _ => {}
            }
        }
    }

    let columns: Vec<(String, ScalarType)> = types
        .into_iter()
        .enumerate()
        .map(|(i, ty)| (format!("c{}", i), ty.unwrap_or(ScalarType::U64)))
        .collect();
    Ok(Schema::new(columns))
}

fn scalar_type_from_value(value: &Value) -> ScalarType {
    match value {
        Value::I64(v) => {
            if *v >= 0 && *v <= u32::MAX as i64 {
                ScalarType::U32
            } else {
                ScalarType::I64
            }
        }
        Value::F64(_) => ScalarType::F64,
        Value::Symbol(_) | Value::String(_) => ScalarType::Symbol,
    }
}

pub(super) fn extend_prob_facts_with_coin(
    program: &Program,
    prob_facts: &mut Vec<ProbFact>,
) -> Result<()> {
    let mut seen: HashSet<GroundAtom> = HashSet::new();
    for pf in prob_facts.iter() {
        seen.insert(atom_key_from_ground_atom(&pf.atom)?);
    }

    for rule in &program.rules {
        for lit in &rule.body {
            let BodyLiteral::Positive(atom) = lit else {
                continue;
            };
            if atom.predicate != "coin" || atom.terms.len() != 1 {
                continue;
            }
            let Term::Float(prob) = atom.terms[0] else {
                continue;
            };
            let key = atom_key_from_ground_atom(atom)?;
            if seen.insert(key) {
                prob_facts.push(ProbFact {
                    prob,
                    atom: atom.clone(),
                });
            }
        }
    }

    Ok(())
}

fn push_value_bytes(out: &mut Vec<u8>, col_type: ScalarType, value: &Value) -> Result<()> {
    match col_type {
        ScalarType::U32 => match value {
            Value::I64(v) => {
                let v_u32 = u32::try_from(*v).map_err(|_| {
                    XlogError::Execution(format!("Value {} out of range for u32", v))
                })?;
                out.extend_from_slice(&v_u32.to_le_bytes());
            }
            Value::Symbol(v) => {
                out.extend_from_slice(&v.to_le_bytes());
            }
            _ => {
                return Err(XlogError::Execution(
                    "Expected integer-compatible value for u32".to_string(),
                ))
            }
        },
        ScalarType::U64 => match value {
            Value::I64(v) => {
                let v_u64 = u64::try_from(*v).map_err(|_| {
                    XlogError::Execution(format!("Value {} out of range for u64", v))
                })?;
                out.extend_from_slice(&v_u64.to_le_bytes());
            }
            _ => {
                return Err(XlogError::Execution(
                    "Expected integer-compatible value for u64".to_string(),
                ))
            }
        },
        ScalarType::I32 => match value {
            Value::I64(v) => {
                let v_i32 = i32::try_from(*v).map_err(|_| {
                    XlogError::Execution(format!("Value {} out of range for i32", v))
                })?;
                out.extend_from_slice(&v_i32.to_le_bytes());
            }
            _ => {
                return Err(XlogError::Execution(
                    "Expected integer-compatible value for i32".to_string(),
                ))
            }
        },
        ScalarType::I64 => match value {
            Value::I64(v) => {
                out.extend_from_slice(&v.to_le_bytes());
            }
            _ => {
                return Err(XlogError::Execution(
                    "Expected integer-compatible value for i64".to_string(),
                ))
            }
        },
        ScalarType::F32 => match value {
            Value::F64(bits) => {
                let v = f64::from_bits(*bits) as f32;
                out.extend_from_slice(&v.to_le_bytes());
            }
            Value::I64(v) => {
                let v = *v as f32;
                out.extend_from_slice(&v.to_le_bytes());
            }
            _ => {
                return Err(XlogError::Execution(
                    "Expected numeric value for f32".to_string(),
                ))
            }
        },
        ScalarType::F64 => match value {
            Value::F64(bits) => {
                let v = f64::from_bits(*bits);
                out.extend_from_slice(&v.to_le_bytes());
            }
            Value::I64(v) => {
                let v = *v as f64;
                out.extend_from_slice(&v.to_le_bytes());
            }
            _ => {
                return Err(XlogError::Execution(
                    "Expected numeric value for f64".to_string(),
                ))
            }
        },
        ScalarType::Bool => match value {
            Value::I64(v) => {
                let b = match *v {
                    0 => 0u8,
                    1 => 1u8,
                    _ => {
                        return Err(XlogError::Execution(
                            "Boolean value must be 0 or 1".to_string(),
                        ))
                    }
                };
                out.push(b);
            }
            _ => {
                return Err(XlogError::Execution(
                    "Expected integer-compatible value for bool".to_string(),
                ))
            }
        },
        ScalarType::Symbol => match value {
            Value::Symbol(v) => {
                out.extend_from_slice(&v.to_le_bytes());
            }
            Value::String(s) => {
                let id = xlog_core::symbol::intern(s);
                out.extend_from_slice(&id.to_le_bytes());
            }
            _ => {
                return Err(XlogError::Execution(
                    "Expected symbol/string value for symbol column".to_string(),
                ))
            }
        },
    }
    Ok(())
}

pub(super) fn compile_sampling_plan(
    prob_facts: &[ProbFact],
    annotated_disjunctions: &[xlog_logic::ast::AnnotatedDisjunction],
) -> Result<(Vec<f32>, Vec<ProbFactSpec>, Vec<AdSpec>)> {
    let mut probs: Vec<f32> = Vec::new();
    let mut fact_specs: Vec<ProbFactSpec> = Vec::new();
    let mut ad_specs: Vec<AdSpec> = Vec::new();

    for pf in prob_facts {
        validate_prob(pf.prob, "probabilistic fact")?;
        let atom = atom_key_from_ground_atom(&pf.atom)?;
        let var_idx = probs.len();
        probs.push(pf.prob as f32);
        fact_specs.push(ProbFactSpec { var_idx, atom });
    }

    for ad in annotated_disjunctions {
        if ad.choices.is_empty() {
            return Err(XlogError::Compilation(
                "Annotated disjunction must contain at least one choice".to_string(),
            ));
        }

        let mut choice_atoms: Vec<GroundAtom> = Vec::with_capacity(ad.choices.len());
        let mut choice_probs: Vec<f64> = Vec::with_capacity(ad.choices.len());
        for pf in &ad.choices {
            validate_prob(pf.prob, "annotated disjunction choice")?;
            choice_atoms.push(atom_key_from_ground_atom(&pf.atom)?);
            choice_probs.push(pf.prob);
        }

        let sum: f64 = choice_probs.iter().copied().sum();
        let eps = 1e-12;
        if sum > 1.0 + eps {
            return Err(XlogError::Compilation(format!(
                "Annotated disjunction probabilities sum to {} (> 1.0)",
                sum
            )));
        }

        let has_none = (1.0 - sum) > eps;
        let mut probs_full: Vec<f64> = choice_probs.clone();
        if has_none {
            probs_full.push((1.0 - sum).max(0.0));
        }

        // Encode categorical choice as a chain of Bernoulli decisions (same as provenance lowering).
        let m = probs_full.len();
        let mut decision_vars: Vec<usize> = Vec::new();
        if m > 1 {
            let mut remaining = 1.0f64;
            for i in 0..(m - 1) {
                let p_i = probs_full[i];
                let cond_true = if remaining <= 0.0 {
                    0.0
                } else {
                    p_i / remaining
                };
                validate_prob(cond_true, "annotated disjunction conditional")?;
                probs.push(cond_true as f32);
                decision_vars.push(probs.len() - 1);
                remaining -= p_i;
            }
        }

        ad_specs.push(AdSpec {
            decision_vars,
            choices: choice_atoms,
            has_none,
        });
    }

    Ok((probs, fact_specs, ad_specs))
}
