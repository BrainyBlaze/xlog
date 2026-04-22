// ---------------------------------------------------------------------------
// ILP (Inductive Logic Programming) Python bindings
// ---------------------------------------------------------------------------

use std::collections::HashMap as StdHashMap;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySequence};

use xlog_core::{symbol, RelId, ScalarType, Schema};
use xlog_cuda::{CudaKernelProvider, JoinType};
use xlog_ir::{ExecutionPlan, RirNode};
use xlog_logic::ast::{Program as AstProgram, Term};
use xlog_prob::exact::GpuConfig;
use xlog_runtime::ilp_registry::IlpMask;
use xlog_runtime::{read_device_row_count, Executor};

use xlog_cuda::type_seam::GpuScalar;

use super::{
    dlpack_capsule_from_tensor, dlpack_from_py, ilp_gpu, provider_from_config, types,
    CompiledIlpProgram, IlpProgramFactory, IlpTaggedCreditDeviceResult,
};

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

struct RelationExampleGroup {
    relation: String,
    query_buf: xlog_cuda::CudaBuffer,
    num_rows: u32,
}

fn collect_dlpack_columns(
    dlpack_columns: &Bound<'_, PyAny>,
    err_msg: &str,
) -> PyResult<Vec<xlog_cuda::DlpackManagedTensor>> {
    let seq = dlpack_columns
        .downcast::<PySequence>()
        .map_err(|_| PyValueError::new_err(err_msg.to_string()))?;
    let mut tensors = Vec::with_capacity(seq.len()? as usize);
    for item in seq.iter()? {
        tensors.push(dlpack_from_py(&item?)?);
    }
    Ok(tensors)
}

/// Allocate zero-initialized loss (scalar) and grad (num_cands) on GPU and
/// export both as DLPack capsules. Shared by both f32 and f64 paths.
fn build_zero_typed<T>(
    provider: &CudaKernelProvider,
    py: Python<'_>,
    num_cands: u32,
    scalar_type: ScalarType,
) -> PyResult<(PyObject, PyObject)>
where
    T: GpuScalar + cudarc::driver::ValidAsZeroBits,
{
    let mut d_grad = provider
        .memory()
        .alloc::<T>(num_cands as usize)
        .map_err(|e| types::gpu_err("alloc grad", e))?;
    if num_cands > 0 {
        provider
            .device()
            .inner()
            .memset_zeros(&mut d_grad)
            .map_err(|e| types::gpu_err("zero grad", e))?;
    }
    let mut d_loss = provider
        .memory()
        .alloc::<T>(1)
        .map_err(|e| types::gpu_err("alloc loss", e))?;
    provider
        .device()
        .inner()
        .memset_zeros(&mut d_loss)
        .map_err(|e| types::gpu_err("zero loss", e))?;
    ilp_gpu::export_loss_grad_device(provider, py, d_loss, d_grad, num_cands, scalar_type)
}

fn export_device_bool_tensor(
    provider: &CudaKernelProvider,
    py: Python<'_>,
    values: xlog_cuda::memory::TrackedCudaSlice<u8>,
    rows: usize,
) -> PyResult<PyObject> {
    let rows_u32 = u32::try_from(rows)
        .map_err(|_| PyValueError::new_err(format!("Row count {} exceeds u32::MAX", rows)))?;

    let mut d_num_rows = provider.memory().alloc::<u32>(1).map_err(types::xlog_err)?;
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[rows_u32], &mut d_num_rows)
        .map_err(types::xlog_err)?;

    let buffer = xlog_cuda::CudaBuffer::from_columns(
        vec![values.into_bytes().into()],
        rows as u64,
        d_num_rows,
        Schema::new(vec![("col0".to_string(), ScalarType::Bool)]),
    );
    let tensor = provider
        .to_dlpack_table(buffer)
        .column(0)
        .map_err(types::xlog_err)?;
    dlpack_capsule_from_tensor(py, tensor)
}

fn export_device_u32_tensor_as_i32(
    provider: &CudaKernelProvider,
    py: Python<'_>,
    values: xlog_cuda::memory::TrackedCudaSlice<u32>,
    rows: usize,
) -> PyResult<PyObject> {
    let rows_u32 = u32::try_from(rows)
        .map_err(|_| PyValueError::new_err(format!("Row count {} exceeds u32::MAX", rows)))?;

    let mut d_num_rows = provider.memory().alloc::<u32>(1).map_err(types::xlog_err)?;
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[rows_u32], &mut d_num_rows)
        .map_err(types::xlog_err)?;

    // PyTorch does not support unsigned 32-bit DLPack tensors; export as i32.
    let buffer = xlog_cuda::CudaBuffer::from_columns(
        vec![values.into_bytes().into()],
        rows as u64,
        d_num_rows,
        Schema::new(vec![("col0".to_string(), ScalarType::I32)]),
    );
    let tensor = provider
        .to_dlpack_table(buffer)
        .column(0)
        .map_err(types::xlog_err)?;
    dlpack_capsule_from_tensor(py, tensor)
}

fn empty_tagged_credit_device_result(
    provider: &CudaKernelProvider,
    py: Python<'_>,
    num_facts: usize,
) -> PyResult<IlpTaggedCreditDeviceResult> {
    let mut d_row_offsets = provider
        .memory()
        .alloc::<u32>(num_facts + 1)
        .map_err(types::xlog_err)?;
    provider
        .device()
        .inner()
        .memset_zeros(&mut d_row_offsets)
        .map_err(types::xlog_err)?;
    let d_empty_indices = provider.memory().alloc::<u32>(0).map_err(types::xlog_err)?;
    let d_empty_i = provider.memory().alloc::<u32>(0).map_err(types::xlog_err)?;
    let d_empty_j = provider.memory().alloc::<u32>(0).map_err(types::xlog_err)?;
    let d_empty_k = provider.memory().alloc::<u32>(0).map_err(types::xlog_err)?;

    Ok(IlpTaggedCreditDeviceResult {
        fact_row_offsets: export_device_u32_tensor_as_i32(
            provider,
            py,
            d_row_offsets,
            num_facts + 1,
        )?,
        entry_indices: export_device_u32_tensor_as_i32(provider, py, d_empty_indices, 0)?,
        entry_i: export_device_u32_tensor_as_i32(provider, py, d_empty_i, 0)?,
        entry_j: export_device_u32_tensor_as_i32(provider, py, d_empty_j, 0)?,
        entry_k: export_device_u32_tensor_as_i32(provider, py, d_empty_k, 0)?,
    })
}

fn push_term_bytes(out: &mut Vec<u8>, term: &Term, typ: ScalarType) -> xlog_core::Result<()> {
    use xlog_core::XlogError;
    match (typ, term) {
        (ScalarType::U32, Term::Integer(v)) => {
            let v = u32::try_from(*v)
                .map_err(|_| XlogError::Execution(format!("u32 out of range: {}", v)))?;
            out.extend_from_slice(&v.to_le_bytes());
        }
        (ScalarType::U64, Term::Integer(v)) => {
            let v = u64::try_from(*v)
                .map_err(|_| XlogError::Execution(format!("u64 out of range: {}", v)))?;
            out.extend_from_slice(&v.to_le_bytes());
        }
        (ScalarType::I32, Term::Integer(v)) => {
            let v = i32::try_from(*v)
                .map_err(|_| XlogError::Execution(format!("i32 out of range: {}", v)))?;
            out.extend_from_slice(&v.to_le_bytes());
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
                    "Expected boolean symbol, got '{}'",
                    s
                )));
            }
        }
        (ScalarType::Symbol, Term::String(s)) => {
            out.extend_from_slice(&symbol::intern(s).to_le_bytes());
        }
        (ScalarType::Symbol, Term::Symbol(id)) => {
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
                "Fact cannot contain anonymous wildcard '_'".into(),
            ));
        }
        (_, Term::Aggregate(_)) => {
            return Err(XlogError::Execution("Fact cannot contain aggregate".into()));
        }
        (expected, got) => {
            return Err(XlogError::Execution(format!(
                "Type mismatch: expected {:?}, got {:?}",
                expected, got
            )));
        }
    }
    Ok(())
}

/// Pack `i64` fact values into typed byte columns according to schema.
/// Returns one `Vec<u8>` per column with correctly-encoded LE bytes.
/// Rejects F32/F64 columns (not supported in batch APIs).
pub(crate) fn pack_i64_columns_typed(
    relation: &str,
    facts: &[Vec<i64>],
    schema: &Schema,
) -> PyResult<Vec<Vec<u8>>> {
    let arity = schema.arity();
    for (idx, fact) in facts.iter().enumerate() {
        if fact.len() != arity {
            return Err(PyValueError::new_err(format!(
                "Relation '{}': fact {} has {} values, expected {}",
                relation,
                idx,
                fact.len(),
                arity,
            )));
        }
    }

    let mut columns: Vec<Vec<u8>> = (0..arity)
        .map(|col_idx| {
            let elem_size = schema
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            Vec::with_capacity(facts.len() * elem_size)
        })
        .collect();

    for fact in facts {
        for (col_idx, &val) in fact.iter().enumerate() {
            let col_type = schema.column_type(col_idx);
            let col = &mut columns[col_idx];
            match col_type {
                Some(ScalarType::U32) => {
                    let v = u32::try_from(val).map_err(|_| {
                        PyValueError::new_err(format!(
                            "Relation '{}' column {} (U32): value {} out of range [0, {}]",
                            relation,
                            col_idx,
                            val,
                            u32::MAX,
                        ))
                    })?;
                    col.extend_from_slice(&v.to_le_bytes());
                }
                Some(ScalarType::I32) => {
                    let v = i32::try_from(val).map_err(|_| {
                        PyValueError::new_err(format!(
                            "Relation '{}' column {} (I32): value {} out of range [{}, {}]",
                            relation,
                            col_idx,
                            val,
                            i32::MIN,
                            i32::MAX,
                        ))
                    })?;
                    col.extend_from_slice(&v.to_le_bytes());
                }
                Some(ScalarType::U64) => {
                    let v = u64::try_from(val).map_err(|_| PyValueError::new_err(format!(
                        "Relation '{}' column {} (U64): value {} is negative; U64 requires non-negative values",
                        relation, col_idx, val,
                    )))?;
                    col.extend_from_slice(&v.to_le_bytes());
                }
                Some(ScalarType::I64) => {
                    col.extend_from_slice(&val.to_le_bytes());
                }
                Some(ScalarType::Bool) => match val {
                    0 => col.push(0u8),
                    1 => col.push(1u8),
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "Relation '{}' column {} (Bool): value {} not in {{0, 1}}",
                            relation, col_idx, val,
                        )))
                    }
                },
                Some(ScalarType::Symbol) => {
                    let v = u32::try_from(val).map_err(|_| {
                        PyValueError::new_err(format!(
                            "Relation '{}' column {} (Symbol): value {} out of range [0, {}]",
                            relation,
                            col_idx,
                            val,
                            u32::MAX,
                        ))
                    })?;
                    col.extend_from_slice(&v.to_le_bytes());
                }
                Some(ScalarType::F32) => {
                    return Err(PyValueError::new_err(format!(
                        "Relation '{}' column {} (F32): float columns not supported in batch APIs",
                        relation, col_idx,
                    )));
                }
                Some(ScalarType::F64) => {
                    return Err(PyValueError::new_err(format!(
                        "Relation '{}' column {} (F64): float columns not supported in batch APIs",
                        relation, col_idx,
                    )));
                }
                None => {
                    return Err(PyValueError::new_err(format!(
                        "Relation '{}' column {}: no type in schema",
                        relation, col_idx,
                    )));
                }
            }
        }
    }

    Ok(columns)
}

pub(crate) fn load_facts_into_store(
    ast: &AstProgram,
    provider: &CudaKernelProvider,
    executor: &mut Executor,
    schemas: &HashMap<String, Schema>,
) -> xlog_core::Result<()> {
    use xlog_core::XlogError;
    let mut rows_by_pred: HashMap<&str, Vec<&[Term]>> = HashMap::new();
    for fact in ast.facts() {
        rows_by_pred
            .entry(fact.head.predicate.as_str())
            .or_default()
            .push(&fact.head.terms);
    }

    for (pred, rows) in rows_by_pred {
        let schema = schemas.get(pred).ok_or_else(|| {
            XlogError::Execution(format!("Missing schema for fact predicate {}", pred))
        })?;

        if rows.iter().any(|r| r.len() != schema.arity()) {
            return Err(XlogError::Execution(format!(
                "Fact arity mismatch for {} (expected {})",
                pred,
                schema.arity()
            )));
        }

        let mut columns: Vec<Vec<u8>> = vec![Vec::new(); schema.arity()];
        for row in &rows {
            for (col_idx, term) in row.iter().enumerate() {
                let typ = schema.column_type(col_idx).ok_or_else(|| {
                    XlogError::Execution(format!("Missing type for col {}", col_idx))
                })?;
                push_term_bytes(&mut columns[col_idx], term, typ)?;
            }
        }

        let slices: Vec<&[u8]> = columns.iter().map(|c| c.as_slice()).collect();
        let fact_buf = provider.create_buffer_from_slices(&slices, schema.clone())?;

        let existing = executor.store().get(pred).ok_or_else(|| {
            XlogError::Execution(format!(
                "Missing base relation {} while loading facts",
                pred
            ))
        })?;
        let merged = provider.union(existing, &fact_buf)?;
        executor.store_mut().put(pred, merged);
    }
    Ok(())
}

/// Extracted TensorMaskedJoin metadata from the execution plan.
struct TmjMeta {
    left_keys: Vec<usize>,
    right_keys: Vec<usize>,
    head_projection: Vec<usize>,
    schema_size: usize,
    head_rel_name: String,
}

fn walk_tmj(node: &RirNode, target_mask: Option<&str>) -> Option<TmjMeta> {
    match node {
        RirNode::TensorMaskedJoin {
            mask_name,
            left_keys,
            right_keys,
            head_projection,
            schema_size,
            head_rel_name,
            ..
        } => {
            if target_mask.is_none() || target_mask == Some(mask_name.as_str()) {
                Some(TmjMeta {
                    left_keys: left_keys.clone(),
                    right_keys: right_keys.clone(),
                    head_projection: head_projection.clone(),
                    schema_size: *schema_size,
                    head_rel_name: head_rel_name.clone(),
                })
            } else {
                None
            }
        }
        RirNode::Fixpoint {
            base, recursive, ..
        } => walk_tmj(base, target_mask).or_else(|| walk_tmj(recursive, target_mask)),
        RirNode::Union { inputs } => inputs.iter().find_map(|n| walk_tmj(n, target_mask)),
        RirNode::Filter { input, .. }
        | RirNode::Project { input, .. }
        | RirNode::Distinct { input, .. }
        | RirNode::GroupBy { input, .. } => walk_tmj(input, target_mask),
        RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
            walk_tmj(left, target_mask).or_else(|| walk_tmj(right, target_mask))
        }
        _ => None,
    }
}

fn extract_tmj_meta(plan: &ExecutionPlan) -> TmjMeta {
    extract_tmj_meta_for_mask(plan, None)
}

fn extract_tmj_meta_for_mask(plan: &ExecutionPlan, mask_name: Option<&str>) -> TmjMeta {
    for scc_rules in &plan.rules_by_scc {
        for rule in scc_rules {
            if let Some(meta) = walk_tmj(&rule.body, mask_name) {
                return meta;
            }
        }
    }
    TmjMeta {
        left_keys: vec![],
        right_keys: vec![],
        head_projection: vec![],
        schema_size: 0,
        head_rel_name: String::new(),
    }
}

fn strip_learnable_declarations(source: &str) -> String {
    source
        .lines()
        .filter(|line| !line.trim_start().starts_with("learnable("))
        .collect::<Vec<_>>()
        .join("\n")
}

fn extract_learnable_declarations(source: &str) -> String {
    source
        .lines()
        .filter(|line| line.trim_start().starts_with("learnable("))
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// IlpProgramFactory
// ---------------------------------------------------------------------------

#[pymethods]
impl IlpProgramFactory {
    #[staticmethod]
    #[pyo3(signature = (source, device=0, memory_mb=512, max_active_rules=None))]
    pub fn compile(
        source: &str,
        device: usize,
        memory_mb: u64,
        max_active_rules: Option<usize>,
    ) -> PyResult<CompiledIlpProgram> {
        // Validate max_active_rules range
        if let Some(max) = max_active_rules {
            if !(16..=128).contains(&max) {
                return Err(PyValueError::new_err(format!(
                    "max_active_rules must be between 16 and 128, got {}",
                    max
                )));
            }
        }

        let ast = xlog_logic::parse_program(source).map_err(types::val_err)?;

        let base_source = strip_learnable_declarations(source);
        let learnable_source = extract_learnable_declarations(source);

        let mut compiler = xlog_logic::Compiler::new();
        if let Some(max) = max_active_rules {
            compiler.set_max_active_rules(max);
        }
        let plan = compiler.compile_program(&ast).map_err(types::xlog_err)?;

        let mut rel_index: Vec<(RelId, String)> = compiler
            .rel_ids()
            .iter()
            .map(|(name, id)| (*id, name.clone()))
            .collect();
        rel_index.sort_by_key(|(id, _)| id.0);
        let schemas = compiler.schemas().clone();

        let mut config = GpuConfig::default();
        config.device_ordinal = device;
        config.memory_bytes = memory_mb * 1024 * 1024;
        let provider = Arc::new(provider_from_config(config).map_err(types::xlog_err)?);

        let mut executor = Executor::new(provider.clone());

        for (name, rel_id) in compiler.rel_ids() {
            executor.register_relation(*rel_id, name);
        }

        for (name, schema) in &schemas {
            let empty = provider
                .create_empty_buffer(schema.clone())
                .map_err(types::xlog_err)?;
            executor.store_mut().put(name, empty);
        }

        load_facts_into_store(&ast, &provider, &mut executor, &schemas).map_err(types::xlog_err)?;

        executor.execute_plan(&plan).map_err(types::xlog_err)?;

        let tmj = extract_tmj_meta(&plan);

        let active_rules = max_active_rules.unwrap_or(32);

        Ok(CompiledIlpProgram {
            base_source,
            _learnable_source: learnable_source,
            ast,
            executor,
            provider,
            plan,
            rel_index,
            schemas,
            left_keys: tmj.left_keys,
            right_keys: tmj.right_keys,
            head_projection: tmj.head_projection,
            compiled_schema_size: tmj.schema_size,
            head_rel_name: tmj.head_rel_name,
            max_active_rules: active_rules,
            candidate_map: None,
            candidate_order: None,
            relation_overrides: HashMap::new(),
            coo_chunk_budget: 16 * 1024 * 1024,
            strict_zero_dtoh: false,
        })
    }
}

// ---------------------------------------------------------------------------
// CompiledIlpProgram — #[pymethods] block
// ---------------------------------------------------------------------------

#[pymethods]
impl CompiledIlpProgram {
    /// Upload candidate (i,j,k) -> index mapping. Called once per attempt.
    pub fn set_candidate_map(&mut self, candidates: Vec<(u32, u32, u32)>) -> PyResult<()> {
        let mut map = HashMap::with_capacity(candidates.len());
        for (cidx, &(i, j, k)) in candidates.iter().enumerate() {
            map.insert((i, j, k), cidx as u32);
        }
        self.candidate_map = Some(map);
        self.candidate_order = Some(candidates);
        Ok(())
    }

    /// Length of current candidate map (0 if not set).
    pub fn candidate_map_len(&self) -> usize {
        self.candidate_map.as_ref().map_or(0, |m| m.len())
    }

    pub fn debug_ilp_mask_kind(&self, name: String) -> Option<String> {
        self.executor.ilp_registry().get_mask(&name).map(|mask| {
            match mask {
                IlpMask::Dense { .. } => "dense",
                IlpMask::Sparse { .. } => "sparse_host",
                IlpMask::SparseDevice { .. } => "sparse_device",
            }
            .to_string()
        })
    }

    /// Set the per-chunk temp allocation budget in bytes. The final merged
    /// COO buffer is exact-NNZ sized and may exceed this budget. Default: 16 MB.
    pub fn set_coo_chunk_budget(&mut self, bytes: u64) {
        self.coo_chunk_budget = bytes;
    }

    /// Deprecated: use `set_coo_chunk_budget`. Kept for one release cycle.
    #[allow(deprecated)]
    pub fn set_coo_memory_cap(&mut self, bytes: u64) {
        self.coo_chunk_budget = bytes;
    }

    /// Enable strict zero-D2H mode. When true, raises RuntimeError instead
    /// of falling back to the chunked COO path (which uses D2H transfers).
    /// Use for zero-D2H benchmarks and CI gates.
    pub fn set_strict_zero_dtoh(&mut self, strict: bool) {
        self.strict_zero_dtoh = strict;
    }

    /// Upload a named relation as DLPack tensor columns into the ILP program store.
    ///
    /// This enables tensor-native ILP data upload: compile a schema-only source
    /// (predicate declarations only, no facts), then upload GPU tensor data via
    /// DLPack without host materialization.
    pub fn put_relation(
        &mut self,
        name: String,
        dlpack_columns: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if name.starts_with("__") {
            return Err(PyValueError::new_err(format!(
                "Relation {} is internal and cannot be stored in a compiled ILP program",
                name
            )));
        }
        let schema = self.schemas.get(&name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown relation {} (not present in compiled schemas)",
                name
            ))
        })?;
        let tensors = collect_dlpack_columns(
            dlpack_columns,
            &format!("Relation {} must be a sequence of DLPack columns", name),
        )?;
        let buffer = self
            .provider
            .from_dlpack_tensors_with_schema(schema.clone(), tensors)
            .map_err(types::xlog_err)?;
        let live_buffer = self
            .provider
            .clone_buffer(&buffer)
            .map_err(types::xlog_err)?;
        self.relation_overrides.insert(name.clone(), buffer);
        self.executor.put_relation(&name, live_buffer);
        Ok(())
    }

    /// GPU-resident ILP loss + gradient computation.
    ///
    /// Builds a sparse CSR structure from retained per-entry membership masks
    /// and launches forward (credit gather + NLL loss) and backward (gradient
    /// scatter) CUDA kernels.  Returns `(loss_dlpack, grad_dlpack)` where both
    /// are GPU-resident tensors exported via DLPack.
    ///
    /// # Arguments
    /// * `positives` — list of `(relation, [col_values])` positive examples
    /// * `negatives` — list of `(relation, [col_values])` negative examples
    /// * `cand_probs_obj` — DLPack/PyTorch tensor of candidate probabilities on GPU
    pub fn compute_ilp_loss_grad_gpu<'py>(
        &mut self,
        py: Python<'py>,
        positives: Vec<(String, Vec<i64>)>,
        negatives: Vec<(String, Vec<i64>)>,
        cand_probs_obj: &Bound<'py, PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        // ── Phase A: Validation + DLPack import ──
        let candidate_map = self.candidate_map.clone().ok_or_else(|| {
            PyRuntimeError::new_err(
                "candidate_map not set — call set_candidate_map() before compute_ilp_loss_grad_gpu()",
            )
        })?;
        let num_cands = candidate_map.len() as u32;

        let managed = dlpack_from_py(cand_probs_obj)?;
        let cand_buf = self
            .provider
            .from_dlpack_tensors(vec![managed])
            .map_err(|e| types::gpu_err("DLPack import", e))?;

        // Determine dtype from imported buffer
        let cand_schema = cand_buf.schema().clone();
        let cand_dtype = cand_schema
            .column_type(0)
            .ok_or_else(|| PyRuntimeError::new_err("cand_probs has no column type"))?;
        let is_f64 = match cand_dtype {
            ScalarType::F32 => false,
            ScalarType::F64 => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "cand_probs must be F32 or F64, got {:?}",
                    other
                )));
            }
        };

        let cand_rows = read_device_row_count(&self.provider, &cand_buf)
            .map_err(|e| types::gpu_err("row count", e))?;
        if cand_rows != num_cands as usize {
            return Err(PyValueError::new_err(format!(
                "cand_probs length ({}) != candidate_map length ({})",
                cand_rows, num_cands
            )));
        }

        // ── Phase B: Build fact list, group by relation ──
        let num_pos = positives.len();
        let num_neg = negatives.len();
        let num_facts = (num_pos + num_neg) as u32;

        // Build all_facts: (relation, values, is_positive, global_fact_idx)
        struct FactInfo {
            relation: String,
            values: Vec<i64>,
            is_positive: bool,
        }
        let mut all_facts: Vec<FactInfo> = Vec::with_capacity(num_pos + num_neg);
        for (rel, vals) in positives {
            all_facts.push(FactInfo {
                relation: rel,
                values: vals,
                is_positive: true,
            });
        }
        for (rel, vals) in negatives {
            all_facts.push(FactInfo {
                relation: rel,
                values: vals,
                is_positive: false,
            });
        }

        // Handle empty facts edge case: return zero loss + zero grad
        if num_facts == 0 {
            return self.build_zero_loss_grad(py, num_cands, is_f64);
        }

        // Build is_positive array for upload
        let is_positive_host: Vec<u8> = all_facts
            .iter()
            .map(|f| if f.is_positive { 1u8 } else { 0u8 })
            .collect();

        // Group facts by relation, preserving global fact_idx
        let mut groups: HashMap<String, Vec<(u32, Vec<i64>)>> = HashMap::new();
        for (global_idx, fact) in all_facts.iter().enumerate() {
            groups
                .entry(fact.relation.clone())
                .or_default()
                .push((global_idx as u32, fact.values.clone()));
        }

        if self.strict_zero_dtoh && self.executor.ilp_registry().has_sparse_device_mask() {
            self.evaluate_ilp_plan(py)?;
        }

        // Get ILP tagged result
        let tagged = self
            .executor
            .ilp_last_result()
            .ok_or_else(|| PyRuntimeError::new_err("No ILP result — call evaluate() first"))?;

        // ── Phase C: Device-side COO build (zero D2H) ──
        //
        // Two-pass approach over (relation, candidate) tasks:
        //   Pass 1: compute GPU membership masks
        //   Pass 2: scatter COO entries at host-computed offsets via device kernel
        //
        // The key insight is that each task's num_query is known on the host
        // (it equals the number of facts in that relation group), so we can
        // compute COO write offsets entirely on the host without any D2H reads.
        // We over-allocate COO arrays at upper_bound (sum of all num_query)
        // and fill sentinel values for unused slots.

        let mut tasks: Vec<ilp_gpu::CooTask> = Vec::new();
        let mut fact_indices_buffers: Vec<xlog_cuda::memory::TrackedCudaSlice<u32>> = Vec::new();

        for (relation, facts_with_idx) in &groups {
            let k_idx = self
                .rel_index
                .iter()
                .position(|(_, name)| name == relation)
                .ok_or_else(|| {
                    PyValueError::new_err(format!("Relation '{}' not in ILP schema", relation))
                })? as u32;

            let relevant_entries: Vec<&xlog_runtime::ilp_registry::IlpTagEntry> = tagged
                .entries
                .iter()
                .filter(|e| e.k == k_idx && e.num_rows > 0 && e.buffer.is_some())
                .collect();

            if relevant_entries.is_empty() {
                continue;
            }

            let first_buf = relevant_entries[0]
                .buffer
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("internal: filtered entry has no buffer"))?;
            let arity = first_buf.arity();
            if arity == 0 {
                continue;
            }
            let schema = first_buf.schema().clone();

            let fact_values: Vec<Vec<i64>> =
                facts_with_idx.iter().map(|(_, v)| v.clone()).collect();
            let col_bytes = pack_i64_columns_typed(relation, &fact_values, &schema)?;
            let col_slices: Vec<&[u8]> = col_bytes.iter().map(|c| c.as_slice()).collect();
            let query_buf = self
                .provider
                .create_buffer_from_slices(&col_slices, schema)
                .map_err(|e| types::gpu_err("create_buffer", e))?;

            let keys: Vec<usize> = (0..arity).collect();
            let num_query = fact_values.len() as u32;

            // Upload global fact indices for this relation group (H2D, allowed).
            // Shared across all entries in this relation group via index.
            let global_indices: Vec<u32> = facts_with_idx.iter().map(|(idx, _)| *idx).collect();
            let mut d_fact_indices = self
                .provider
                .memory()
                .alloc::<u32>(num_query as usize)
                .map_err(|e| types::gpu_err("alloc fact_indices", e))?;
            self.provider
                .device()
                .inner()
                .htod_sync_copy_into(&global_indices, &mut d_fact_indices)
                .map_err(|e| types::gpu_err("htod fact_indices", e))?;
            let fi_idx = fact_indices_buffers.len();
            fact_indices_buffers.push(d_fact_indices);

            for entry in &relevant_entries {
                let cidx = match candidate_map.get(&(entry.i, entry.j, entry.k)) {
                    Some(c) => *c,
                    None => continue,
                };

                let entry_buf = entry.buffer.as_ref().ok_or_else(|| {
                    PyRuntimeError::new_err("internal: filtered entry has no buffer")
                })?;
                let d_mask = self
                    .provider
                    .membership_mask_device(&query_buf, entry_buf, &keys, &keys)
                    .map_err(|e| types::gpu_err("membership_mask", e))?;

                tasks.push(ilp_gpu::CooTask {
                    cidx,
                    num_query,
                    d_mask,
                    fact_indices_idx: fi_idx,
                });
            }
        }

        let num_tasks = tasks.len();
        let upper_bound: u32 = tasks.iter().map(|t| t.num_query).sum();

        if upper_bound == 0 || num_tasks == 0 {
            return self.build_loss_grad_empty_coo(
                py,
                &is_positive_host,
                num_facts,
                num_cands,
                is_f64,
            );
        }

        // Check if COO allocation would exceed memory cap.
        // Each COO entry uses 8 bytes (4 for fact_idx + 4 for cand_idx).
        let coo_bytes = (upper_bound as u64) * 8;
        let needs_chunking = coo_bytes > self.coo_chunk_budget;

        // Upload is_positive once (shared across all paths, H2D allowed)
        let mut d_is_positive = self
            .provider
            .memory()
            .alloc::<u8>(num_facts as usize)
            .map_err(|e| types::gpu_err("alloc is_positive", e))?;
        self.provider
            .device()
            .inner()
            .htod_sync_copy_into(&is_positive_host, &mut d_is_positive)
            .map_err(|e| types::gpu_err("htod is_positive", e))?;

        let cand_col = cand_buf
            .column(0)
            .ok_or_else(|| PyRuntimeError::new_err("cand_probs has no column"))?;

        // ── Phase C: COO construction ──
        let (mut d_coo_facts, mut d_coo_cands, actual_nnz) = if !needs_chunking {
            ilp_gpu::build_coo_single(
                &self.provider,
                &tasks,
                &fact_indices_buffers,
                num_facts,
                num_cands,
                upper_bound,
            )?
        } else {
            match ilp_gpu::build_coo_chunked(
                &self.provider,
                &tasks,
                &fact_indices_buffers,
                num_facts,
                self.coo_chunk_budget,
            )? {
                Some(result) => result,
                None => {
                    return self.build_loss_grad_empty_coo(
                        py,
                        &is_positive_host,
                        num_facts,
                        num_cands,
                        is_f64,
                    );
                }
            }
        };

        // ── Phase D: Sort COO + device-side CSR build ──
        let d_row_offsets = ilp_gpu::sort_and_build_csr(
            &self.provider,
            &mut d_coo_facts,
            &mut d_coo_cands,
            actual_nnz,
            num_facts,
        )?;

        // ── Phase E: Forward + backward + device-side reduction ──
        ilp_gpu::forward_backward_reduce(
            &self.provider,
            py,
            &d_row_offsets,
            &d_coo_cands,
            cand_col,
            &d_is_positive,
            num_facts,
            num_cands,
            is_f64,
        )
    }

    /// GPU-resident ILP loss + gradient computation from relation-native
    /// positive/negative examples stored as DLPack column groups.
    ///
    /// `positives_by_relation` and `negatives_by_relation` must be Python dicts
    /// mapping relation name -> sequence of DLPack columns with schema-matching
    /// dtypes. No host fact tuple materialization occurs on this path.
    pub fn compute_ilp_loss_grad_gpu_relations<'py>(
        &mut self,
        py: Python<'py>,
        positives_by_relation: &Bound<'py, PyAny>,
        negatives_by_relation: &Bound<'py, PyAny>,
        cand_probs_obj: &Bound<'py, PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        let candidate_map = self.candidate_map.clone().ok_or_else(|| {
            PyRuntimeError::new_err(
                "candidate_map not set — call set_candidate_map() before compute_ilp_loss_grad_gpu_relations()",
            )
        })?;
        let num_cands = candidate_map.len() as u32;

        let managed = dlpack_from_py(cand_probs_obj)?;
        let cand_buf = self
            .provider
            .from_dlpack_tensors(vec![managed])
            .map_err(|e| types::gpu_err("DLPack import", e))?;

        let cand_schema = cand_buf.schema().clone();
        let cand_dtype = cand_schema
            .column_type(0)
            .ok_or_else(|| PyRuntimeError::new_err("cand_probs has no column type"))?;
        let is_f64 = match cand_dtype {
            ScalarType::F32 => false,
            ScalarType::F64 => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "cand_probs must be F32 or F64, got {:?}",
                    other
                )));
            }
        };

        let cand_rows = read_device_row_count(&self.provider, &cand_buf)
            .map_err(|e| types::gpu_err("row count", e))?;
        if cand_rows != num_cands as usize {
            return Err(PyValueError::new_err(format!(
                "cand_probs length ({}) != candidate_map length ({})",
                cand_rows, num_cands
            )));
        }

        let positive_groups = self.collect_relation_example_groups(
            positives_by_relation,
            "positives_by_relation must be a dict[str, sequence[dlpack]]",
        )?;
        let negative_groups = self.collect_relation_example_groups(
            negatives_by_relation,
            "negatives_by_relation must be a dict[str, sequence[dlpack]]",
        )?;

        let num_pos: u32 = positive_groups.iter().map(|g| g.num_rows).sum();
        let num_neg: u32 = negative_groups.iter().map(|g| g.num_rows).sum();
        let num_facts = num_pos + num_neg;

        if num_facts == 0 {
            return self.build_zero_loss_grad(py, num_cands, is_f64);
        }

        let mut is_positive_host = Vec::with_capacity(num_facts as usize);
        is_positive_host.extend(std::iter::repeat_n(1u8, num_pos as usize));
        is_positive_host.extend(std::iter::repeat_n(0u8, num_neg as usize));

        if self.strict_zero_dtoh && self.executor.ilp_registry().has_sparse_device_mask() {
            self.evaluate_ilp_plan(py)?;
        }

        let tagged = self
            .executor
            .ilp_last_result()
            .ok_or_else(|| PyRuntimeError::new_err("No ILP result — call evaluate() first"))?;

        let mut tasks: Vec<ilp_gpu::CooTask> = Vec::new();
        let mut fact_indices_buffers: Vec<xlog_cuda::memory::TrackedCudaSlice<u32>> = Vec::new();
        let mut next_global_idx: u32 = 0;

        for group in positive_groups.iter().chain(negative_groups.iter()) {
            if group.num_rows == 0 {
                continue;
            }
            let k_idx = self
                .rel_index
                .iter()
                .position(|(_, name)| name == &group.relation)
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Relation '{}' not in ILP schema",
                        group.relation
                    ))
                })? as u32;

            let relevant_entries: Vec<&xlog_runtime::ilp_registry::IlpTagEntry> = tagged
                .entries
                .iter()
                .filter(|e| e.k == k_idx && e.num_rows > 0 && e.buffer.is_some())
                .collect();
            if relevant_entries.is_empty() {
                next_global_idx = next_global_idx.saturating_add(group.num_rows);
                continue;
            }

            let arity = group.query_buf.arity();
            if arity == 0 {
                next_global_idx = next_global_idx.saturating_add(group.num_rows);
                continue;
            }
            let keys: Vec<usize> = (0..arity).collect();

            let global_indices: Vec<u32> =
                (next_global_idx..next_global_idx + group.num_rows).collect();
            let mut d_fact_indices = self
                .provider
                .memory()
                .alloc::<u32>(group.num_rows as usize)
                .map_err(|e| types::gpu_err("alloc fact_indices", e))?;
            self.provider
                .device()
                .inner()
                .htod_sync_copy_into(&global_indices, &mut d_fact_indices)
                .map_err(|e| types::gpu_err("htod fact_indices", e))?;
            let fi_idx = fact_indices_buffers.len();
            fact_indices_buffers.push(d_fact_indices);

            for entry in &relevant_entries {
                let cidx = match candidate_map.get(&(entry.i, entry.j, entry.k)) {
                    Some(c) => *c,
                    None => continue,
                };
                let entry_buf = entry.buffer.as_ref().ok_or_else(|| {
                    PyRuntimeError::new_err("internal: filtered entry has no buffer")
                })?;
                let d_mask = self
                    .provider
                    .membership_mask_device(&group.query_buf, entry_buf, &keys, &keys)
                    .map_err(|e| types::gpu_err("membership_mask", e))?;
                tasks.push(ilp_gpu::CooTask {
                    cidx,
                    num_query: group.num_rows,
                    d_mask,
                    fact_indices_idx: fi_idx,
                });
            }

            next_global_idx = next_global_idx.saturating_add(group.num_rows);
        }

        let num_tasks = tasks.len();
        let upper_bound: u32 = tasks.iter().map(|t| t.num_query).sum();
        if upper_bound == 0 || num_tasks == 0 {
            return self.build_loss_grad_empty_coo(
                py,
                &is_positive_host,
                num_facts,
                num_cands,
                is_f64,
            );
        }

        let coo_bytes = (upper_bound as u64) * 8;
        let needs_chunking = coo_bytes > self.coo_chunk_budget;

        let mut d_is_positive = self
            .provider
            .memory()
            .alloc::<u8>(num_facts as usize)
            .map_err(|e| types::gpu_err("alloc is_positive", e))?;
        self.provider
            .device()
            .inner()
            .htod_sync_copy_into(&is_positive_host, &mut d_is_positive)
            .map_err(|e| types::gpu_err("htod is_positive", e))?;

        let cand_col = cand_buf
            .column(0)
            .ok_or_else(|| PyRuntimeError::new_err("cand_probs has no column"))?;

        let (mut d_coo_facts, mut d_coo_cands, actual_nnz) = if !needs_chunking {
            ilp_gpu::build_coo_single(
                &self.provider,
                &tasks,
                &fact_indices_buffers,
                num_facts,
                num_cands,
                upper_bound,
            )?
        } else {
            match ilp_gpu::build_coo_chunked(
                &self.provider,
                &tasks,
                &fact_indices_buffers,
                num_facts,
                self.coo_chunk_budget,
            )? {
                Some(result) => result,
                None => {
                    return self.build_loss_grad_empty_coo(
                        py,
                        &is_positive_host,
                        num_facts,
                        num_cands,
                        is_f64,
                    );
                }
            }
        };

        let d_row_offsets = ilp_gpu::sort_and_build_csr(
            &self.provider,
            &mut d_coo_facts,
            &mut d_coo_cands,
            actual_nnz,
            num_facts,
        )?;

        ilp_gpu::forward_backward_reduce(
            &self.provider,
            py,
            &d_row_offsets,
            &d_coo_cands,
            cand_col,
            &d_is_positive,
            num_facts,
            num_cands,
            is_f64,
        )
    }

    pub fn set_rule_mask(
        &mut self,
        name: String,
        mask_hard_flat: &Bound<'_, PyAny>,
        mask_soft_flat: &Bound<'_, PyAny>,
        schema_size: usize,
    ) -> PyResult<()> {
        if self.compiled_schema_size > 0 && schema_size != self.compiled_schema_size {
            return Err(PyValueError::new_err(format!(
                "schema_size mismatch: mask has N={} but compiled program expects N={}",
                schema_size, self.compiled_schema_size,
            )));
        }

        let hard_dmt = dlpack_from_py(mask_hard_flat)?;
        let soft_dmt = dlpack_from_py(mask_soft_flat)?;

        let hard_buf = self
            .provider
            .from_dlpack_tensors(vec![hard_dmt])
            .map_err(types::xlog_err)?;
        let soft_buf = self
            .provider
            .from_dlpack_tensors(vec![soft_dmt])
            .map_err(types::xlog_err)?;

        self.executor
            .ilp_registry_mut()
            .insert_mask(name, hard_buf, soft_buf, schema_size);
        Ok(())
    }

    /// Sparse mask API: candidate IDs + DLPack soft probabilities (GPU tensor).
    ///
    /// `candidate_ids` must be exactly `[0..C)` where C is the candidate count
    /// for this mask under the provided recursion policy.
    ///
    /// `soft_probs_dlpack` is a DLPack capsule (CUDA f64 tensor) passed from PyTorch.
    /// Rust imports it zero-copy, downloads values (not counted by the D2H counter),
    /// performs deterministic top-k (desc soft value, then lower id), and
    /// stores a sparse IlpMask (no dense N^3 materialization).
    #[pyo3(signature = (name, candidate_ids, soft_probs_dlpack, budget, allow_recursive=false))]
    pub fn set_rule_mask_sparse(
        &mut self,
        name: String,
        candidate_ids: Vec<u32>,
        soft_probs_dlpack: &Bound<'_, PyAny>,
        budget: usize,
        allow_recursive: bool,
    ) -> PyResult<()> {
        if self.strict_zero_dtoh {
            return Err(PyRuntimeError::new_err(
                "strict_zero_dtoh forbids legacy set_rule_mask_sparse; use set_rule_mask_sparse_selected instead",
            ));
        }

        let tmj = extract_tmj_meta_for_mask(&self.plan, Some(&name));
        let n = tmj.schema_size;
        if n == 0 {
            return Err(PyValueError::new_err(format!(
                "no learnable mask '{}' found",
                name
            )));
        }
        if self.compiled_schema_size > 0 && n != self.compiled_schema_size {
            return Err(PyValueError::new_err(format!(
                "schema_size mismatch for '{}': plan N={} compiled N={}",
                name, n, self.compiled_schema_size
            )));
        }

        let expected_c = self.expected_candidate_count(&name, allow_recursive)?;
        if candidate_ids.len() != expected_c {
            return Err(PyValueError::new_err(format!(
                "candidate_ids length {} != expected candidate count {}",
                candidate_ids.len(),
                expected_c
            )));
        }
        for (idx, &cid) in candidate_ids.iter().enumerate() {
            if cid != idx as u32 {
                return Err(PyValueError::new_err(format!(
                    "candidate_ids must be [0..{}), got id {} at position {}",
                    expected_c, cid, idx
                )));
            }
        }

        // Import DLPack tensor (zero-copy, stays on GPU)
        let soft_dmt = dlpack_from_py(soft_probs_dlpack)?;
        let soft_buf = self
            .provider
            .from_dlpack_tensors(vec![soft_dmt])
            .map_err(types::xlog_err)?;

        // Download f64 values (control-plane, NOT tracked by D2H counter)
        let soft_probs = self
            .provider
            .download_column_untracked::<f64>(&soft_buf, 0)
            .map_err(types::xlog_err)?;

        if soft_probs.len() != candidate_ids.len() {
            return Err(PyValueError::new_err(format!(
                "soft_probs tensor length {} != candidate_ids length {}",
                soft_probs.len(),
                candidate_ids.len()
            )));
        }

        let candidate_triples = self.candidate_triples_for_mask(&name, allow_recursive)?;
        if candidate_triples.len() != expected_c {
            return Err(PyRuntimeError::new_err(format!(
                "internal candidate count mismatch: triples={} expected={}",
                candidate_triples.len(),
                expected_c
            )));
        }

        // Convert to f32 for top-k ranking in insert_mask_from_sparse
        let active_soft: Vec<f32> = soft_probs.iter().map(|&v| v as f32).collect();

        self.executor
            .ilp_registry_mut()
            .insert_mask_from_sparse(name, n, &candidate_triples, &active_soft, budget)
            .map_err(types::xlog_err)
    }

    /// Preferred sparse mask API: caller preselects candidate IDs and passes
    /// only the selected subset plus aligned soft probabilities.
    ///
    /// This path avoids the full-vector soft-probability download in
    /// `set_rule_mask_sparse(...)`. The selected IDs are mapped directly to the
    /// existing candidate triples and stored in sparse-mask order.
    #[pyo3(signature = (name, selected_candidate_ids, selected_soft_probs_dlpack, allow_recursive=false))]
    pub fn set_rule_mask_sparse_selected(
        &mut self,
        name: String,
        selected_candidate_ids: Vec<u32>,
        selected_soft_probs_dlpack: &Bound<'_, PyAny>,
        allow_recursive: bool,
    ) -> PyResult<()> {
        if self.strict_zero_dtoh {
            return Err(PyRuntimeError::new_err(
                "strict_zero_dtoh forbids set_rule_mask_sparse_selected; use explicit compatibility export instead",
            ));
        }
        let tmj = extract_tmj_meta_for_mask(&self.plan, Some(&name));
        let n = tmj.schema_size;
        if n == 0 {
            return Err(PyValueError::new_err(format!(
                "no learnable mask '{}' found",
                name
            )));
        }
        if self.compiled_schema_size > 0 && n != self.compiled_schema_size {
            return Err(PyValueError::new_err(format!(
                "schema_size mismatch for '{}': plan N={} compiled N={}",
                name, n, self.compiled_schema_size
            )));
        }

        let candidate_triples = self.candidate_triples_for_mask(&name, allow_recursive)?;
        let expected_c = candidate_triples.len();

        let soft_dmt = dlpack_from_py(selected_soft_probs_dlpack)?;
        let soft_buf = self
            .provider
            .from_dlpack_tensors(vec![soft_dmt])
            .map_err(types::xlog_err)?;
        let selected_len = usize::try_from(soft_buf.num_rows())
            .map_err(|_| PyValueError::new_err("selected soft_probs length overflow"))?;

        if selected_len != selected_candidate_ids.len() {
            return Err(PyValueError::new_err(format!(
                "selected soft_probs length {} != selected_candidate_ids length {}",
                selected_len,
                selected_candidate_ids.len()
            )));
        }

        let mut seen = HashSet::with_capacity(selected_candidate_ids.len());
        let mut selected_entries = Vec::with_capacity(selected_candidate_ids.len());
        for (pos, &cid) in selected_candidate_ids.iter().enumerate() {
            let idx = usize::try_from(cid).map_err(|_| {
                PyValueError::new_err(format!("candidate id {} overflows usize", cid))
            })?;
            if idx >= expected_c {
                return Err(PyValueError::new_err(format!(
                    "selected candidate id {} out of range [0, {}) at position {}",
                    cid, expected_c, pos
                )));
            }
            if !seen.insert(cid) {
                return Err(PyValueError::new_err(format!(
                    "duplicate selected candidate id {} at position {}",
                    cid, pos
                )));
            }
            selected_entries.push(candidate_triples[idx]);
        }

        self.executor
            .ilp_registry_mut()
            .insert_selected_mask(name, n, &selected_entries);
        Ok(())
    }

    /// Strict sparse mask API: selected candidate IDs stay as a device tensor on
    /// the Python side. Rust resolves them against the fixed candidate order
    /// uploaded via `set_candidate_map(...)`.
    #[pyo3(signature = (name, selected_candidate_ids_dlpack, selected_soft_probs_dlpack, allow_recursive=false))]
    pub fn set_rule_mask_sparse_selected_device(
        &mut self,
        name: String,
        selected_candidate_ids_dlpack: &Bound<'_, PyAny>,
        selected_soft_probs_dlpack: &Bound<'_, PyAny>,
        allow_recursive: bool,
    ) -> PyResult<()> {
        self.set_rule_mask_sparse_selected_device_impl(
            name,
            selected_candidate_ids_dlpack,
            selected_soft_probs_dlpack,
            allow_recursive,
            true,
        )
    }

    pub fn evaluate(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.strict_zero_dtoh && self.executor.ilp_registry().has_sparse_device_mask() {
            return Err(PyRuntimeError::new_err(
                "SparseDevice evaluate() is incompatible with strict_zero_dtoh; \
use train_only(..., strict_gpu_native=True) or export an explicit compatibility mask first",
            ));
        }
        self.evaluate_ilp_plan(py)
    }

    /// Reset mutable runtime state for ILP attempt reuse.
    ///
    /// Clears ILP registry (masks/tagged results), executor store,
    /// join index cache, stats, and profiler. Then re-registers schemas
    /// with empty buffers, reloads base facts from AST, and re-executes
    /// the plan. Preserves all immutable compile artifacts (AST, plan,
    /// schemas, rel_index, provider, TMJ metadata, max_active_rules).
    ///
    /// After reset, the program is in the same state as a fresh compile()
    /// with the same source — ready for set_rule_mask / evaluate cycles.
    pub fn reset_runtime(&mut self, py: Python<'_>) -> PyResult<()> {
        let _result: xlog_core::Result<()> = py.allow_threads(|| {
            // 1. Clear all mutable state (ILP registry, store, caches, stats)
            self.executor.reset_for_ilp();

            // 2. Re-register schemas with empty buffers
            for (name, schema) in &self.schemas {
                let empty = self.provider.create_empty_buffer(schema.clone())?;
                self.executor.store_mut().put(name, empty);
            }

            // 3. Reload base facts from preserved AST
            load_facts_into_store(&self.ast, &self.provider, &mut self.executor, &self.schemas)?;

            // 3b. Reapply persistent relation uploads on top of AST/base facts.
            self.apply_relation_overrides()?;

            // 4. Re-execute plan (populates derived relations)
            self.executor.execute_plan(&self.plan)?;

            Ok(())
        });
        _result.map_err(types::xlog_err)?;

        // 5. Reset D2H transfer counter
        self.provider.reset_d2h_transfer_count();

        Ok(())
    }

    pub fn get_tagged_results(&self) -> PyResult<Vec<(u32, u32, u32, u32)>> {
        self.ensure_host_semantic_compat(
            "get_tagged_results()",
            "disable strict_zero_dtoh for host materialization",
        )?;
        match self.executor.ilp_last_result() {
            Some(result) => Ok(result
                .entries
                .iter()
                .map(|e| (e.i, e.j, e.k, e.num_rows))
                .collect()),
            None => Ok(Vec::new()),
        }
    }

    pub fn fact_exists(&self, relation: &str, values: Vec<i64>) -> PyResult<bool> {
        self.ensure_host_semantic_compat("fact_exists()", "batch_fact_membership_device(...)")?;
        let buf =
            self.executor.store().get(relation).ok_or_else(|| {
                PyValueError::new_err(format!("Relation '{}' not found", relation))
            })?;

        Self::fact_exists_in_buffer(&self.provider, buf, &values).map_err(types::xlog_err)
    }

    /// Return all facts in the named relation as a list of int lists.
    #[pyo3(signature = (rel_name))]
    pub fn relation_facts(&self, rel_name: String) -> PyResult<Vec<Vec<i64>>> {
        self.ensure_host_semantic_compat(
            "relation_facts()",
            "disable strict_zero_dtoh for host materialization",
        )?;
        let buf =
            self.executor.store().get(&rel_name).ok_or_else(|| {
                PyValueError::new_err(format!("Relation '{}' not found", rel_name))
            })?;

        let num_rows =
            read_device_row_count(&self.provider, buf).map_err(types::xlog_err)? as usize;
        if num_rows == 0 {
            return Ok(Vec::new());
        }

        // Download all columns (reuse fact_exists_in_buffer pattern)
        let schema = buf.schema();
        let mut columns: Vec<Vec<i64>> = Vec::new();
        for col_idx in 0..buf.arity() {
            let col_type = schema.column_type(col_idx).ok_or_else(|| {
                PyRuntimeError::new_err(format!("Column {} type not found in schema", col_idx))
            })?;
            let col_i64: Vec<i64> = match col_type {
                ScalarType::I64 => self
                    .provider
                    .download_column::<i64>(buf, col_idx)
                    .map_err(types::xlog_err)?,
                ScalarType::I32 => self
                    .provider
                    .download_column::<i32>(buf, col_idx)
                    .map_err(types::xlog_err)?
                    .into_iter()
                    .map(|v| v as i64)
                    .collect(),
                ScalarType::U32 | ScalarType::Symbol => self
                    .provider
                    .download_column::<u32>(buf, col_idx)
                    .map_err(types::xlog_err)?
                    .into_iter()
                    .map(|v| v as i64)
                    .collect(),
                ScalarType::U64 => self
                    .provider
                    .download_column::<u64>(buf, col_idx)
                    .map_err(types::xlog_err)?
                    .into_iter()
                    .map(|v| v as i64)
                    .collect(),
                ScalarType::Bool => self
                    .provider
                    .download_column::<bool>(buf, col_idx)
                    .map_err(types::xlog_err)?
                    .into_iter()
                    .map(|v| if v { 1i64 } else { 0i64 })
                    .collect(),
                ScalarType::F32 | ScalarType::F64 => {
                    return Err(PyRuntimeError::new_err(format!(
                        "relation_facts does not support float column type {:?}",
                        col_type
                    )));
                }
            };
            columns.push(col_i64);
        }

        let mut result = Vec::with_capacity(num_rows);
        for r in 0..num_rows {
            let mut row = Vec::with_capacity(buf.arity());
            for c in 0..buf.arity() {
                row.push(columns[c][r]);
            }
            result.push(row);
        }
        Ok(result)
    }

    /// Sample up to `max_n` derived facts for `head_rel` that are NOT in `exclude`.
    ///
    /// Returns `list[list[int]]` — each inner list is a tuple of column values.
    /// Uses the same column-download pattern as `relation_facts`.
    #[pyo3(signature = (head_rel, exclude, max_n))]
    pub fn sample_false_positives(
        &self,
        head_rel: String,
        exclude: Vec<(String, Vec<i64>)>,
        max_n: usize,
    ) -> PyResult<Vec<Vec<i64>>> {
        self.ensure_host_semantic_compat(
            "sample_false_positives()",
            "disable strict_zero_dtoh for host materialization",
        )?;
        // Build exclude set: only consider tuples for the requested relation
        let exclude_set: HashSet<Vec<i64>> = exclude
            .into_iter()
            .filter(|(rel, _)| rel == &head_rel)
            .map(|(_, vals)| vals)
            .collect();

        // Download all facts using the same pattern as relation_facts
        let buf =
            self.executor.store().get(&head_rel).ok_or_else(|| {
                PyValueError::new_err(format!("Relation '{}' not found", head_rel))
            })?;

        let num_rows =
            read_device_row_count(&self.provider, buf).map_err(types::xlog_err)? as usize;
        if num_rows == 0 {
            return Ok(Vec::new());
        }

        let schema = buf.schema();
        let mut columns: Vec<Vec<i64>> = Vec::new();
        for col_idx in 0..buf.arity() {
            let col_type = schema.column_type(col_idx).ok_or_else(|| {
                PyRuntimeError::new_err(format!("Column {} type not found in schema", col_idx))
            })?;
            let col_i64: Vec<i64> = match col_type {
                ScalarType::I64 => self
                    .provider
                    .download_column::<i64>(buf, col_idx)
                    .map_err(types::xlog_err)?,
                ScalarType::I32 => self
                    .provider
                    .download_column::<i32>(buf, col_idx)
                    .map_err(types::xlog_err)?
                    .into_iter()
                    .map(|v| v as i64)
                    .collect(),
                ScalarType::U32 | ScalarType::Symbol => self
                    .provider
                    .download_column::<u32>(buf, col_idx)
                    .map_err(types::xlog_err)?
                    .into_iter()
                    .map(|v| v as i64)
                    .collect(),
                ScalarType::U64 => self
                    .provider
                    .download_column::<u64>(buf, col_idx)
                    .map_err(types::xlog_err)?
                    .into_iter()
                    .map(|v| v as i64)
                    .collect(),
                ScalarType::Bool => self
                    .provider
                    .download_column::<bool>(buf, col_idx)
                    .map_err(types::xlog_err)?
                    .into_iter()
                    .map(|v| if v { 1i64 } else { 0i64 })
                    .collect(),
                ScalarType::F32 | ScalarType::F64 => {
                    return Err(PyRuntimeError::new_err(format!(
                        "sample_false_positives does not support float column type {:?}",
                        col_type
                    )));
                }
            };
            columns.push(col_i64);
        }

        // Filter out excluded tuples and cap at max_n
        let mut result = Vec::with_capacity(max_n.min(num_rows));
        for r in 0..num_rows {
            if result.len() >= max_n {
                break;
            }
            let mut row = Vec::with_capacity(buf.arity());
            for c in 0..buf.arity() {
                row.push(columns[c][r]);
            }
            if !exclude_set.contains(&row) {
                result.push(row);
            }
        }
        Ok(result)
    }

    pub fn tagged_entries_containing_fact(
        &self,
        relation: &str,
        values: Vec<i64>,
    ) -> PyResult<Vec<(u32, u32, u32)>> {
        self.ensure_host_semantic_compat(
            "tagged_entries_containing_fact()",
            "batch_tagged_credit_device(...)",
        )?;
        let k_idx = self
            .rel_index
            .iter()
            .position(|(_, name)| name == relation)
            .ok_or_else(|| {
                PyValueError::new_err(format!("Relation '{}' not in ILP schema", relation))
            })? as u32;

        let tagged = match self.executor.ilp_last_result() {
            Some(t) => t,
            None => return Ok(Vec::new()),
        };

        let mut result = Vec::new();
        for entry in &tagged.entries {
            if entry.k != k_idx || entry.num_rows == 0 {
                continue;
            }

            let (_, left_name) = &self.rel_index[entry.i as usize];
            let (_, right_name) = &self.rel_index[entry.j as usize];

            let left_buf = match self.executor.store().get(left_name) {
                Some(buf) if buf.arity() > 0 => buf,
                _ => continue,
            };
            let right_buf = match self.executor.store().get(right_name) {
                Some(buf) if buf.arity() > 0 => buf,
                _ => continue,
            };

            // Arity guard: same as executor (skip if join keys exceed columns)
            let left_max = self.left_keys.iter().copied().max().unwrap_or(0);
            let right_max = self.right_keys.iter().copied().max().unwrap_or(0);
            if left_buf.arity() <= left_max || right_buf.arity() <= right_max {
                continue;
            }

            let joined = self
                .provider
                .hash_join_v2(
                    left_buf,
                    right_buf,
                    &self.left_keys,
                    &self.right_keys,
                    JoinType::Inner,
                )
                .map_err(types::xlog_err)?;

            // Apply head_projection: check projected columns against values,
            // not the raw join output (which has more columns than the head).
            let found = if !self.head_projection.is_empty()
                && self.head_projection.len() == values.len()
            {
                Self::fact_exists_projected(&self.provider, &joined, &values, &self.head_projection)
                    .map_err(types::xlog_err)?
            } else {
                Self::fact_exists_in_buffer(&self.provider, &joined, &values)
                    .map_err(types::xlog_err)?
            };

            if found {
                result.push((entry.i, entry.j, entry.k));
            }
        }
        Ok(result)
    }

    pub fn ilp_schema_size(&self) -> usize {
        self.rel_index.len()
    }

    pub fn ilp_relation_names(&self) -> Vec<String> {
        self.rel_index
            .iter()
            .map(|(_, name)| name.clone())
            .collect()
    }

    /// Return declared predicate types from source `pred` declarations.
    ///
    /// Output is a list of `(name, types)` tuples so callers can
    /// deterministically inspect whether metadata is available for
    /// relations used during promotion.
    pub fn relation_type_annotations(&self) -> Vec<(String, Vec<String>)> {
        self.ast
            .predicates
            .iter()
            .map(|pred| {
                let types = pred.types.iter().map(types::scalar_type_name).collect();
                (pred.name.clone(), types)
            })
            .collect()
    }

    /// Return the set of valid (i,j,k) candidates for the given learnable mask.
    ///
    /// Pruning rules:
    /// - k must be the head relation for this mask
    /// - At least one of (i,j) must have nonzero tuples in the store
    /// - Template+template body pairs (both have zero tuples) are pruned
    /// - If allow_recursive is false: i==k_head or j==k_head are pruned
    ///   (unless head already has base facts)
    ///
    /// Returns list of dicts: [{id, i, j, k, left_name, right_name, head_name}]
    /// IDs assigned 0..C-1 after sorting by (k, i, j) ascending.
    #[pyo3(signature = (mask_name, allow_recursive=false))]
    fn valid_candidates(
        &self,
        py: Python<'_>,
        mask_name: String,
        allow_recursive: bool,
    ) -> PyResult<Vec<StdHashMap<String, PyObject>>> {
        let candidates = self.candidate_triples_for_mask(&mask_name, allow_recursive)?;

        let result: Vec<StdHashMap<String, PyObject>> = candidates
            .iter()
            .enumerate()
            .map(|(id, &(i, j, k))| {
                let mut d = StdHashMap::new();
                d.insert("id".into(), id.into_py(py));
                d.insert("i".into(), i.into_py(py));
                d.insert("j".into(), j.into_py(py));
                d.insert("k".into(), k.into_py(py));
                d.insert(
                    "left_name".into(),
                    self.rel_index[i as usize].1.clone().into_py(py),
                );
                d.insert(
                    "right_name".into(),
                    self.rel_index[j as usize].1.clone().into_py(py),
                );
                d.insert(
                    "head_name".into(),
                    self.rel_index[k as usize].1.clone().into_py(py),
                );
                d
            })
            .collect();

        Ok(result)
    }

    pub fn commit_induced_rule(&mut self, rule_source: &str) -> PyResult<()> {
        let new_base = format!("{}\n{}", self.base_source, rule_source);

        let ast = xlog_logic::parse_program(&new_base).map_err(types::val_err)?;
        let mut compiler = xlog_logic::Compiler::new();
        compiler.set_max_active_rules(self.max_active_rules);
        let plan = compiler.compile_program(&ast).map_err(types::xlog_err)?;
        let schemas = compiler.schemas().clone();

        self.executor.reset_for_mc();
        for (name, rel_id) in compiler.rel_ids() {
            self.executor.register_relation(*rel_id, name);
        }
        for (name, schema) in &schemas {
            let empty = self
                .provider
                .create_empty_buffer(schema.clone())
                .map_err(types::xlog_err)?;
            self.executor.store_mut().put(name, empty);
        }
        load_facts_into_store(&ast, &self.provider, &mut self.executor, &schemas)
            .map_err(types::xlog_err)?;
        self.apply_relation_overrides().map_err(types::xlog_err)?;
        self.executor.execute_plan(&plan).map_err(types::xlog_err)?;

        self.base_source = new_base;
        self.ast = ast;
        let tmj = extract_tmj_meta(&plan);
        self.left_keys = tmj.left_keys;
        self.right_keys = tmj.right_keys;
        self.head_projection = tmj.head_projection;
        self.compiled_schema_size = tmj.schema_size;
        self.head_rel_name = tmj.head_rel_name;
        self.plan = plan;
        self.schemas = schemas;
        Ok(())
    }

    /// GPU-side batch fact membership check.
    /// Uploads `facts` (list of value-lists) to a temporary CudaBuffer,
    /// semi-joins against the named relation, returns per-fact boolean mask.
    /// Zero download_column_* calls — only downloads the u8 mask.
    pub fn batch_fact_membership_device(
        &self,
        py: Python<'_>,
        relation: &str,
        facts: Vec<Vec<i64>>,
    ) -> PyResult<PyObject> {
        let buf =
            self.executor.store().get(relation).ok_or_else(|| {
                PyValueError::new_err(format!("Relation '{}' not found", relation))
            })?;

        if facts.is_empty() {
            let empty = self
                .provider
                .memory()
                .alloc::<u8>(0)
                .map_err(types::xlog_err)?;
            return export_device_bool_tensor(&self.provider, py, empty, 0);
        }
        if buf.arity() == 0 {
            let mut zeros = self
                .provider
                .memory()
                .alloc::<u8>(facts.len())
                .map_err(types::xlog_err)?;
            self.provider
                .device()
                .inner()
                .memset_zeros(&mut zeros)
                .map_err(types::xlog_err)?;
            return export_device_bool_tensor(&self.provider, py, zeros, facts.len());
        }

        let col_bytes = pack_i64_columns_typed(relation, &facts, buf.schema())?;
        let col_slices: Vec<&[u8]> = col_bytes.iter().map(|c| c.as_slice()).collect();
        let query_buf = self
            .provider
            .create_buffer_from_slices(&col_slices, buf.schema().clone())
            .map_err(types::xlog_err)?;

        let keys: Vec<usize> = (0..buf.arity()).collect();
        let mask = self
            .provider
            .membership_mask_device(&query_buf, buf, &keys, &keys)
            .map_err(types::xlog_err)?;
        export_device_bool_tensor(&self.provider, py, mask, facts.len())
    }

    pub fn batch_fact_membership(
        &self,
        relation: &str,
        facts: Vec<Vec<i64>>,
    ) -> PyResult<Vec<bool>> {
        self.ensure_host_semantic_compat(
            "batch_fact_membership()",
            "batch_fact_membership_device(...)",
        )?;
        if facts.is_empty() {
            return Ok(Vec::new());
        }

        let buf =
            self.executor.store().get(relation).ok_or_else(|| {
                PyValueError::new_err(format!("Relation '{}' not found", relation))
            })?;

        let arity = buf.arity();
        if arity == 0 {
            return Ok(vec![false; facts.len()]);
        }

        // Schema-aware typed upload
        let col_bytes = pack_i64_columns_typed(relation, &facts, buf.schema())?;
        let col_slices: Vec<&[u8]> = col_bytes.iter().map(|c| c.as_slice()).collect();
        let query_buf = self
            .provider
            .create_buffer_from_slices(&col_slices, buf.schema().clone())
            .map_err(types::xlog_err)?;

        // All columns are keys (full-tuple match)
        let keys: Vec<usize> = (0..arity).collect();

        self.provider
            .membership_mask(&query_buf, buf, &keys, &keys)
            .map_err(types::xlog_err)
    }

    /// GPU-side batch credit assignment.
    ///
    /// For each fact in `facts`, returns the list of (i,j,k) entries whose
    /// join result contains that fact. Uses membership_mask against retained
    /// per-entry buffers — zero download_column_* calls.
    pub fn batch_tagged_credit(
        &self,
        relation: &str,
        facts: Vec<Vec<i64>>,
    ) -> PyResult<Vec<Vec<(u32, u32, u32)>>> {
        self.ensure_host_semantic_compat(
            "batch_tagged_credit()",
            "batch_tagged_credit_device(...)",
        )?;
        if facts.is_empty() {
            return Ok(Vec::new());
        }

        // Find k index for this relation
        let k_idx = self
            .rel_index
            .iter()
            .position(|(_, name)| name == relation)
            .ok_or_else(|| {
                PyValueError::new_err(format!("Relation '{}' not in ILP schema", relation))
            })? as u32;

        let tagged = match self.executor.ilp_last_result() {
            Some(t) => t,
            None => return Ok(vec![Vec::new(); facts.len()]),
        };

        // Filter entries to those matching target relation k, with retained buffers
        let relevant_entries: Vec<&xlog_runtime::ilp_registry::IlpTagEntry> = tagged
            .entries
            .iter()
            .filter(|e| e.k == k_idx && e.num_rows > 0 && e.buffer.is_some())
            .collect();

        if relevant_entries.is_empty() {
            return Ok(vec![Vec::new(); facts.len()]);
        }

        // Determine arity from the first entry's buffer
        let first_buf = relevant_entries[0]
            .buffer
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("internal: filtered entry has no buffer"))?;
        let arity = first_buf.arity();
        if arity == 0 {
            return Ok(vec![Vec::new(); facts.len()]);
        }

        // Schema-aware typed upload
        let schema = first_buf.schema().clone();
        let col_bytes = pack_i64_columns_typed(relation, &facts, &schema)?;
        let col_slices: Vec<&[u8]> = col_bytes.iter().map(|c| c.as_slice()).collect();
        let query_buf = self
            .provider
            .create_buffer_from_slices(&col_slices, schema)
            .map_err(types::xlog_err)?;

        let keys: Vec<usize> = (0..arity).collect();

        // For each relevant entry, compute membership mask against query facts
        let mut per_fact_credits: Vec<Vec<(u32, u32, u32)>> = vec![Vec::new(); facts.len()];

        for entry in &relevant_entries {
            let entry_buf = entry
                .buffer
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("internal: filtered entry has no buffer"))?;
            let mask = self
                .provider
                .membership_mask(&query_buf, entry_buf, &keys, &keys)
                .map_err(types::xlog_err)?;

            for (fact_idx, &found) in mask.iter().enumerate() {
                if found {
                    per_fact_credits[fact_idx].push((entry.i, entry.j, entry.k));
                }
            }
        }

        Ok(per_fact_credits)
    }

    /// GPU-side batch credit assignment.
    ///
    /// Returns a CSR-style device representation:
    /// - `fact_row_offsets`: len = num_facts + 1
    /// - `entry_indices`: COO candidate indices, sorted by fact row
    /// - `entry_i/j/k`: metadata arrays indexed by `entry_indices`
    ///
    /// Zero DTOH calls on the query path. Uses the non-chunked COO builder
    /// to avoid reading device-side nnz metadata back to the host.
    pub fn batch_tagged_credit_device(
        &self,
        py: Python<'_>,
        relation: &str,
        facts: Vec<Vec<i64>>,
    ) -> PyResult<IlpTaggedCreditDeviceResult> {
        if facts.is_empty() {
            return empty_tagged_credit_device_result(&self.provider, py, 0);
        }

        let k_idx = self
            .rel_index
            .iter()
            .position(|(_, name)| name == relation)
            .ok_or_else(|| {
                PyValueError::new_err(format!("Relation '{}' not in ILP schema", relation))
            })? as u32;

        let tagged = match self.executor.ilp_last_result() {
            Some(t) => t,
            None => return empty_tagged_credit_device_result(&self.provider, py, facts.len()),
        };

        let relevant_entries: Vec<&xlog_runtime::ilp_registry::IlpTagEntry> = tagged
            .entries
            .iter()
            .filter(|e| e.k == k_idx && e.num_rows > 0 && e.buffer.is_some())
            .collect();

        if relevant_entries.is_empty() {
            return empty_tagged_credit_device_result(&self.provider, py, facts.len());
        }

        let first_buf = relevant_entries[0]
            .buffer
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("internal: filtered entry has no buffer"))?;
        let arity = first_buf.arity();
        if arity == 0 {
            return empty_tagged_credit_device_result(&self.provider, py, facts.len());
        }

        let schema = first_buf.schema().clone();
        let col_bytes = pack_i64_columns_typed(relation, &facts, &schema)?;
        let col_slices: Vec<&[u8]> = col_bytes.iter().map(|c| c.as_slice()).collect();
        let query_buf = self
            .provider
            .create_buffer_from_slices(&col_slices, schema)
            .map_err(types::xlog_err)?;

        let keys: Vec<usize> = (0..arity).collect();
        let num_facts = u32::try_from(facts.len())
            .map_err(|_| PyValueError::new_err("facts length exceeds u32::MAX"))?;
        let num_entries = u32::try_from(relevant_entries.len())
            .map_err(|_| PyValueError::new_err("entry count exceeds u32::MAX"))?;
        let upper_bound = num_facts
            .checked_mul(num_entries)
            .ok_or_else(|| PyValueError::new_err("credit upper bound overflow"))?;

        let fact_indices_host: Vec<u32> = (0..num_facts).collect();
        let mut d_fact_indices = self
            .provider
            .memory()
            .alloc::<u32>(num_facts as usize)
            .map_err(|e| types::gpu_err("alloc fact_indices", e))?;
        self.provider
            .device()
            .inner()
            .htod_sync_copy_into(&fact_indices_host, &mut d_fact_indices)
            .map_err(|e| types::gpu_err("htod fact_indices", e))?;

        let mut entry_i_host = Vec::with_capacity(relevant_entries.len());
        let mut entry_j_host = Vec::with_capacity(relevant_entries.len());
        let mut entry_k_host = Vec::with_capacity(relevant_entries.len());
        let mut tasks = Vec::with_capacity(relevant_entries.len());

        for (entry_idx, entry) in relevant_entries.iter().enumerate() {
            let entry_buf = entry
                .buffer
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("internal: filtered entry has no buffer"))?;
            let d_mask = self
                .provider
                .membership_mask_device(&query_buf, entry_buf, &keys, &keys)
                .map_err(|e| types::gpu_err("membership_mask", e))?;
            tasks.push(ilp_gpu::CooTask {
                d_mask,
                fact_indices_idx: 0,
                cidx: entry_idx as u32,
                num_query: num_facts,
            });
            entry_i_host.push(entry.i);
            entry_j_host.push(entry.j);
            entry_k_host.push(entry.k);
        }

        let (mut d_coo_facts, mut d_coo_cands, actual_nnz) = ilp_gpu::build_coo_single(
            &self.provider,
            &tasks,
            &[d_fact_indices],
            num_facts,
            num_entries,
            upper_bound,
        )?;
        let d_row_offsets = ilp_gpu::sort_and_build_csr(
            &self.provider,
            &mut d_coo_facts,
            &mut d_coo_cands,
            actual_nnz,
            num_facts,
        )?;

        let mut d_entry_i = self
            .provider
            .memory()
            .alloc::<u32>(entry_i_host.len())
            .map_err(|e| types::gpu_err("alloc entry_i", e))?;
        let mut d_entry_j = self
            .provider
            .memory()
            .alloc::<u32>(entry_j_host.len())
            .map_err(|e| types::gpu_err("alloc entry_j", e))?;
        let mut d_entry_k = self
            .provider
            .memory()
            .alloc::<u32>(entry_k_host.len())
            .map_err(|e| types::gpu_err("alloc entry_k", e))?;
        self.provider
            .device()
            .inner()
            .htod_sync_copy_into(&entry_i_host, &mut d_entry_i)
            .map_err(|e| types::gpu_err("htod entry_i", e))?;
        self.provider
            .device()
            .inner()
            .htod_sync_copy_into(&entry_j_host, &mut d_entry_j)
            .map_err(|e| types::gpu_err("htod entry_j", e))?;
        self.provider
            .device()
            .inner()
            .htod_sync_copy_into(&entry_k_host, &mut d_entry_k)
            .map_err(|e| types::gpu_err("htod entry_k", e))?;

        Ok(IlpTaggedCreditDeviceResult {
            fact_row_offsets: export_device_u32_tensor_as_i32(
                &self.provider,
                py,
                d_row_offsets,
                facts.len() + 1,
            )?,
            entry_indices: export_device_u32_tensor_as_i32(
                &self.provider,
                py,
                d_coo_cands,
                upper_bound as usize,
            )?,
            entry_i: export_device_u32_tensor_as_i32(
                &self.provider,
                py,
                d_entry_i,
                entry_i_host.len(),
            )?,
            entry_j: export_device_u32_tensor_as_i32(
                &self.provider,
                py,
                d_entry_j,
                entry_j_host.len(),
            )?,
            entry_k: export_device_u32_tensor_as_i32(
                &self.provider,
                py,
                d_entry_k,
                entry_k_host.len(),
            )?,
        })
    }

    pub fn d2h_transfer_count(&self) -> u64 {
        self.provider.d2h_transfer_count()
    }

    pub fn reset_d2h_transfer_count(&self) {
        self.provider.reset_d2h_transfer_count()
    }

    pub fn host_transfer_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = self.provider.host_transfer_stats();
        let dict = PyDict::new_bound(py);
        dict.set_item("dtoh_bytes", stats.dtoh_bytes)?;
        dict.set_item("htod_bytes", stats.htod_bytes)?;
        dict.set_item("dtoh_calls", stats.dtoh_calls)?;
        dict.set_item("htod_calls", stats.htod_calls)?;
        Ok(dict.into())
    }

    pub fn reset_host_transfer_stats(&self) {
        self.provider.reset_host_transfer_stats()
    }
}

// ---------------------------------------------------------------------------
// CompiledIlpProgram — plain impl block (GPU export helpers + internal)
// ---------------------------------------------------------------------------

impl CompiledIlpProgram {
    fn collect_relation_example_groups(
        &self,
        relations_obj: &Bound<'_, PyAny>,
        err_msg: &str,
    ) -> PyResult<Vec<RelationExampleGroup>> {
        let dict = relations_obj
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err(err_msg.to_string()))?;
        let mut groups = Vec::with_capacity(dict.len());
        for (name_obj, columns_obj) in dict.iter() {
            let relation: String = name_obj.extract()?;
            let schema = self.schemas.get(&relation).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Unknown relation {} (not present in compiled schemas)",
                    relation
                ))
            })?;
            let tensors = collect_dlpack_columns(
                &columns_obj,
                &format!("Relation {} must be a sequence of DLPack columns", relation),
            )?;
            let query_buf = self
                .provider
                .from_dlpack_tensors_with_schema(schema.clone(), tensors)
                .map_err(types::xlog_err)?;
            let num_rows = u32::try_from(query_buf.num_rows())
                .map_err(|_| PyValueError::new_err("relation row count exceeds u32::MAX"))?;
            groups.push(RelationExampleGroup {
                relation,
                query_buf,
                num_rows,
            });
        }
        Ok(groups)
    }

    fn apply_relation_overrides(&mut self) -> xlog_core::Result<()> {
        let relation_names: Vec<String> = self.relation_overrides.keys().cloned().collect();
        for name in relation_names {
            let stored = self
                .relation_overrides
                .get(&name)
                .expect("relation override disappeared during runtime reset");
            let live = self.provider.clone_buffer(stored)?;
            self.executor.put_relation(&name, live);
        }
        Ok(())
    }

    fn evaluate_ilp_plan(&mut self, py: Python<'_>) -> PyResult<()> {
        let result: xlog_core::Result<xlog_cuda::CudaBuffer> = py.allow_threads(|| {
            self.executor.reset_for_mc();
            for (name, schema) in &self.schemas {
                let empty = self.provider.create_empty_buffer(schema.clone())?;
                self.executor.store_mut().put(name, empty);
            }
            load_facts_into_store(&self.ast, &self.provider, &mut self.executor, &self.schemas)?;
            self.apply_relation_overrides()?;
            self.executor.execute_plan(&self.plan)
        });
        result.map_err(types::xlog_err)?;
        Ok(())
    }

    fn set_rule_mask_sparse_selected_device_impl(
        &mut self,
        name: String,
        selected_candidate_ids_dlpack: &Bound<'_, PyAny>,
        selected_soft_probs_dlpack: &Bound<'_, PyAny>,
        allow_recursive: bool,
        validate_ids: bool,
    ) -> PyResult<()> {
        let tmj = extract_tmj_meta_for_mask(&self.plan, Some(&name));
        let n = tmj.schema_size;
        if n == 0 {
            return Err(PyValueError::new_err(format!(
                "no learnable mask '{}' found",
                name
            )));
        }
        if self.compiled_schema_size > 0 && n != self.compiled_schema_size {
            return Err(PyValueError::new_err(format!(
                "schema_size mismatch for '{}': plan N={} compiled N={}",
                name, n, self.compiled_schema_size
            )));
        }

        let _ = allow_recursive;
        let candidate_order = self.candidate_order.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "candidate order not set — call set_candidate_map() before strict sparse mask updates",
            )
        })?;
        let expected_c = candidate_order.len();

        let ids_dmt = dlpack_from_py(selected_candidate_ids_dlpack)?;
        let ids_buf = self
            .provider
            .from_dlpack_tensors(vec![ids_dmt])
            .map_err(types::xlog_err)?;
        let soft_dmt = dlpack_from_py(selected_soft_probs_dlpack)?;
        let soft_buf = self
            .provider
            .from_dlpack_tensors(vec![soft_dmt])
            .map_err(types::xlog_err)?;

        let selected_len = usize::try_from(ids_buf.num_rows())
            .map_err(|_| PyValueError::new_err("selected candidate ids length overflow"))?;
        let soft_len = usize::try_from(soft_buf.num_rows())
            .map_err(|_| PyValueError::new_err("selected soft_probs length overflow"))?;
        if soft_len != selected_len {
            return Err(PyValueError::new_err(format!(
                "selected soft_probs length {} != selected candidate ids length {}",
                soft_len, selected_len
            )));
        }

        if validate_ids {
            self.provider
                .validate_selected_ids(&ids_buf, expected_c)
                .map_err(types::val_err)?;
        }

        let active_flags = self
            .provider
            .build_selected_id_mask(&ids_buf, expected_c)
            .map_err(types::xlog_err)?;

        self.executor
            .ilp_registry_mut()
            .insert_selected_mask_device(
                name,
                n,
                candidate_order.clone(),
                active_flags,
                selected_len,
            );
        Ok(())
    }

    fn ensure_host_semantic_compat(&self, api: &str, device_hint: &str) -> PyResult<()> {
        if self.strict_zero_dtoh {
            return Err(PyRuntimeError::new_err(format!(
                "strict_zero_dtoh forbids {}; use {} instead",
                api, device_hint
            )));
        }
        Ok(())
    }

    // ─── GPU loss/grad export helpers ──────────────────────────────────

    /// Build zero loss (scalar) + zero grad (num_cands) on GPU, export via DLPack.
    fn build_zero_loss_grad(
        &self,
        py: Python<'_>,
        num_cands: u32,
        is_f64: bool,
    ) -> PyResult<(PyObject, PyObject)> {
        if is_f64 {
            build_zero_typed::<f64>(&self.provider, py, num_cands, ScalarType::F64)
        } else {
            build_zero_typed::<f32>(&self.provider, py, num_cands, ScalarType::F32)
        }
    }

    /// Handle the case where COO is empty but we have facts.
    /// Facts with no covering entries get: positive -> -log(eps), negative -> -log(1.0) = 0.
    /// We still run the kernels with an empty CSR for correctness.
    fn build_loss_grad_empty_coo(
        &self,
        py: Python<'_>,
        is_positive_host: &[u8],
        num_facts: u32,
        num_cands: u32,
        is_f64: bool,
    ) -> PyResult<(PyObject, PyObject)> {
        // Build CSR with all-zero row_offsets (every row has 0 non-zeros)
        let row_offsets = vec![0u32; (num_facts + 1) as usize];
        let mut d_row_offsets = self
            .provider
            .memory()
            .alloc::<u32>((num_facts + 1) as usize)
            .map_err(|e| types::gpu_err("alloc", e))?;
        self.provider
            .device()
            .inner()
            .htod_sync_copy_into(&row_offsets, &mut d_row_offsets)
            .map_err(|e| types::gpu_err("htod", e))?;

        // Empty col_indices
        let d_col_indices = self
            .provider
            .memory()
            .alloc::<u32>(0)
            .map_err(|e| types::gpu_err("alloc", e))?;

        let mut d_is_positive = self
            .provider
            .memory()
            .alloc::<u8>(num_facts as usize)
            .map_err(|e| types::gpu_err("alloc", e))?;
        self.provider
            .device()
            .inner()
            .htod_sync_copy_into(is_positive_host, &mut d_is_positive)
            .map_err(|e| types::gpu_err("htod", e))?;

        // Build a single-column CudaBuffer with 0 elements to represent an empty cand_probs
        // for the kernel launch (won't be read since row ranges are all empty).
        // We need a dummy CudaColumn. Use the actual cand count = num_cands.
        // Since COO is empty the kernel won't access cand_probs, but we need to pass something.
        let dummy_col: xlog_cuda::CudaColumn = if is_f64 {
            self.provider
                .memory()
                .alloc::<f64>(num_cands.max(1) as usize)
                .map_err(|e| types::gpu_err("alloc dummy", e))?
                .into_bytes()
                .into()
        } else {
            self.provider
                .memory()
                .alloc::<f32>(num_cands.max(1) as usize)
                .map_err(|e| types::gpu_err("alloc dummy", e))?
                .into_bytes()
                .into()
        };

        ilp_gpu::forward_backward_reduce(
            &self.provider,
            py,
            &d_row_offsets,
            &d_col_indices,
            &dummy_col,
            &d_is_positive,
            num_facts,
            num_cands,
            is_f64,
        )
    }

    /// Returns sorted (i,j,k) candidate triples for the given learnable mask.
    /// Pruning logic must stay aligned with `valid_candidates`.
    fn candidate_triples_for_mask(
        &self,
        mask_name: &str,
        allow_recursive: bool,
    ) -> PyResult<Vec<(u32, u32, u32)>> {
        let tmj = extract_tmj_meta_for_mask(&self.plan, Some(mask_name));
        let n = tmj.schema_size;
        if n == 0 {
            return Err(PyValueError::new_err(format!(
                "no learnable mask '{}' found in compiled program",
                mask_name
            )));
        }
        let head_name = &tmj.head_rel_name;
        let k_head = self
            .rel_index
            .iter()
            .position(|(_, name)| name == head_name)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "head relation '{}' not in rel_index for mask '{}'",
                    head_name, mask_name
                ))
            })? as u32;

        // Identify which relations currently have nonzero tuples in store.
        let has_tuples: Vec<bool> = self
            .rel_index
            .iter()
            .map(|(_, name)| {
                self.executor
                    .store()
                    .get(name)
                    .map(|buf| buf.num_rows() > 0)
                    .unwrap_or(false)
            })
            .collect();

        let mut triples: Vec<(u32, u32, u32)> = Vec::new();
        for i in 0..n as u32 {
            for j in 0..n as u32 {
                let k = k_head;

                // Prune template+template (both no tuples).
                if !has_tuples[i as usize] && !has_tuples[j as usize] {
                    continue;
                }

                // Keep behavior aligned with existing alpha candidate pruning:
                // recursive body refs are allowed only if head already has tuples.
                if !allow_recursive && (i == k || j == k) && !has_tuples[k as usize] {
                    continue;
                }

                triples.push((i, j, k));
            }
        }
        triples.sort_by_key(|&(i, j, k)| (k, i, j));
        Ok(triples)
    }

    fn expected_candidate_count(&self, mask_name: &str, allow_recursive: bool) -> PyResult<usize> {
        Ok(self
            .candidate_triples_for_mask(mask_name, allow_recursive)?
            .len())
    }

    fn fact_exists_in_buffer(
        provider: &CudaKernelProvider,
        buf: &xlog_cuda::CudaBuffer,
        values: &[i64],
    ) -> xlog_core::Result<bool> {
        use xlog_core::XlogError;
        let num_rows = read_device_row_count(provider, buf)? as usize;
        if num_rows == 0 {
            return Ok(false);
        }
        if values.len() != buf.arity() {
            return Ok(false);
        }

        let schema = buf.schema();
        let mut columns: Vec<Vec<i64>> = Vec::new();
        for col_idx in 0..buf.arity() {
            let col_type = schema.column_type(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("Column {} type not found in schema", col_idx))
            })?;
            let col_i64: Vec<i64> = match col_type {
                ScalarType::I64 => provider.download_column::<i64>(buf, col_idx)?,
                ScalarType::I32 => provider
                    .download_column::<i32>(buf, col_idx)?
                    .into_iter()
                    .map(|v| v as i64)
                    .collect(),
                ScalarType::U32 | ScalarType::Symbol => provider
                    .download_column::<u32>(buf, col_idx)?
                    .into_iter()
                    .map(|v| v as i64)
                    .collect(),
                ScalarType::U64 => {
                    let col_u64 = provider.download_column::<u64>(buf, col_idx)?;
                    col_u64.into_iter().map(|v| v as i64).collect()
                }
                ScalarType::Bool => provider
                    .download_column::<bool>(buf, col_idx)?
                    .into_iter()
                    .map(|v| if v { 1i64 } else { 0i64 })
                    .collect(),
                ScalarType::F32 | ScalarType::F64 => {
                    return Err(XlogError::Kernel(format!(
                        "fact_exists does not support float column type {:?}",
                        col_type
                    )));
                }
            };
            columns.push(col_i64);
        }

        for row in 0..num_rows {
            let mut matches = true;
            for (col_idx, val) in values.iter().enumerate() {
                if columns[col_idx][row] != *val {
                    matches = false;
                    break;
                }
            }
            if matches {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Like fact_exists_in_buffer but checks only the projected columns.
    /// `projection[i]` is the column index in `buf` that corresponds to
    /// `values[i]` in the head relation.
    fn fact_exists_projected(
        provider: &CudaKernelProvider,
        buf: &xlog_cuda::CudaBuffer,
        values: &[i64],
        projection: &[usize],
    ) -> xlog_core::Result<bool> {
        use xlog_core::XlogError;
        let num_rows = read_device_row_count(provider, buf)? as usize;
        if num_rows == 0 {
            return Ok(false);
        }

        let schema = buf.schema();
        let mut columns: Vec<Vec<i64>> = Vec::new();
        for &col_idx in projection {
            if col_idx >= buf.arity() {
                return Ok(false);
            }
            let col_type = schema.column_type(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("Column {} type not found in schema", col_idx))
            })?;
            let col_i64: Vec<i64> = match col_type {
                ScalarType::I64 => provider.download_column::<i64>(buf, col_idx)?,
                ScalarType::I32 => provider
                    .download_column::<i32>(buf, col_idx)?
                    .into_iter()
                    .map(|v| v as i64)
                    .collect(),
                ScalarType::U32 | ScalarType::Symbol => provider
                    .download_column::<u32>(buf, col_idx)?
                    .into_iter()
                    .map(|v| v as i64)
                    .collect(),
                ScalarType::U64 => provider
                    .download_column::<u64>(buf, col_idx)?
                    .into_iter()
                    .map(|v| v as i64)
                    .collect(),
                ScalarType::Bool => provider
                    .download_column::<bool>(buf, col_idx)?
                    .into_iter()
                    .map(|v| if v { 1i64 } else { 0i64 })
                    .collect(),
                ScalarType::F32 | ScalarType::F64 => {
                    return Err(XlogError::Kernel(format!(
                        "fact_exists does not support float column type {:?}",
                        col_type
                    )));
                }
            };
            columns.push(col_i64);
        }

        for row in 0..num_rows {
            let mut matches = true;
            for (i, val) in values.iter().enumerate() {
                if columns[i][row] != *val {
                    matches = false;
                    break;
                }
            }
            if matches {
                return Ok(true);
            }
        }
        Ok(false)
    }
}
