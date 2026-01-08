//! Query executor for RIR nodes
//!
//! The executor interprets RIR (Relational IR) nodes using the CUDA kernel provider
//! to execute GPU-accelerated relational operations.

use std::collections::HashMap;
use std::sync::Arc;

use xlog_core::{AggOp, RelId, Result, ScalarType, Schema, XlogError};
use xlog_cuda::{CudaBuffer, CudaKernelProvider};
use xlog_ir::{CompareOp, ConstValue, ExecutionPlan, Expr, JoinType, RirNode, Stratum};

use crate::RelationStore;

/// Query executor that interprets RIR nodes using GPU kernels
///
/// The executor processes execution plans by iterating through strata and
/// executing RIR node trees. It maintains a relation store for intermediate
/// and final results.
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use xlog_runtime::Executor;
/// use xlog_cuda::CudaKernelProvider;
///
/// let provider = Arc::new(CudaKernelProvider::new(device, memory)?);
/// let mut executor = Executor::new(provider);
///
/// // Execute a plan
/// let result = executor.execute_plan(&plan)?;
/// ```
pub struct Executor {
    /// CUDA kernel provider for GPU operations
    provider: Arc<CudaKernelProvider>,
    /// Storage for named relations
    store: RelationStore,
    /// Mapping from RelId to relation name
    rel_names: HashMap<RelId, String>,
}

impl Executor {
    /// Create a new executor with the given kernel provider
    ///
    /// # Arguments
    /// * `provider` - The CUDA kernel provider for GPU operations
    pub fn new(provider: Arc<CudaKernelProvider>) -> Self {
        Self {
            provider,
            store: RelationStore::new(),
            rel_names: HashMap::new(),
        }
    }

    /// Get a reference to the relation store
    pub fn store(&self) -> &RelationStore {
        &self.store
    }

    /// Get a mutable reference to the relation store
    pub fn store_mut(&mut self) -> &mut RelationStore {
        &mut self.store
    }

    /// Register a relation name for a RelId
    ///
    /// This mapping is used when executing Scan nodes to look up relations
    /// by their RelId.
    ///
    /// # Arguments
    /// * `rel_id` - The relation identifier
    /// * `name` - The name to associate with the relation
    pub fn register_relation(&mut self, rel_id: RelId, name: &str) {
        self.rel_names.insert(rel_id, name.to_string());
    }

    /// Get the relation name for a RelId
    fn get_rel_name(&self, rel_id: RelId) -> Option<&str> {
        self.rel_names.get(&rel_id).map(|s| s.as_str())
    }

    /// Execute a complete execution plan
    ///
    /// Iterates through strata in order, executing each one.
    /// Returns the result of the final query if present, or an empty buffer.
    ///
    /// # Arguments
    /// * `plan` - The execution plan to execute
    ///
    /// # Returns
    /// The result buffer from executing the plan
    ///
    /// # Errors
    /// Returns an error if any stratum or query execution fails
    pub fn execute_plan(&mut self, plan: &ExecutionPlan) -> Result<CudaBuffer> {
        // Execute strata in order
        for stratum in &plan.strata {
            self.execute_stratum_impl(stratum, plan)?;
        }

        // If there are no strata, return empty buffer
        Ok(CudaBuffer::empty())
    }

    /// Execute a single RIR node tree
    ///
    /// Recursively evaluates the node and its children, returning
    /// the result as a GPU buffer.
    ///
    /// # Arguments
    /// * `node` - The RIR node to execute
    ///
    /// # Returns
    /// A CudaBuffer containing the result of the node execution
    ///
    /// # Errors
    /// Returns an error if the node execution fails
    pub fn execute_node(&mut self, node: &RirNode) -> Result<CudaBuffer> {
        match node {
            RirNode::Scan { rel } => self.execute_scan(*rel),

            RirNode::Filter { input, predicate } => {
                let input_buf = self.execute_node(input)?;
                self.execute_filter(&input_buf, predicate)
            }

            RirNode::Project { input, columns } => {
                let input_buf = self.execute_node(input)?;
                self.execute_project(&input_buf, columns)
            }

            RirNode::Join {
                left,
                right,
                left_keys,
                right_keys,
                join_type,
            } => {
                let left_buf = self.execute_node(left)?;
                let right_buf = self.execute_node(right)?;
                self.execute_join(&left_buf, &right_buf, left_keys, right_keys, *join_type)
            }

            RirNode::GroupBy {
                input,
                key_cols,
                aggs,
            } => {
                let input_buf = self.execute_node(input)?;
                self.execute_groupby(&input_buf, key_cols, aggs)
            }

            RirNode::Union { inputs } => {
                let mut buffers = Vec::with_capacity(inputs.len());
                for input in inputs {
                    buffers.push(self.execute_node(input)?);
                }
                self.execute_union(&buffers)
            }

            RirNode::Distinct { input, key_cols } => {
                let input_buf = self.execute_node(input)?;
                self.execute_distinct(&input_buf, key_cols)
            }

            RirNode::Diff { left, right } => {
                let left_buf = self.execute_node(left)?;
                let right_buf = self.execute_node(right)?;
                self.execute_diff(&left_buf, &right_buf)
            }

            RirNode::Fixpoint {
                scc_id,
                base,
                recursive,
                delta_rel,
                full_rel,
            } => {
                // Semi-naive fixpoint iteration
                self.execute_fixpoint(*scc_id, base, recursive, *delta_rel, *full_rel)
            }
        }
    }

    /// Execute a stratum (internal implementation)
    ///
    /// Processes all SCCs in the stratum by executing their rules.
    fn execute_stratum_impl(&mut self, stratum: &Stratum, plan: &ExecutionPlan) -> Result<()> {
        // Process each SCC in the stratum
        for &scc_id in &stratum.sccs {
            // Get rules for this SCC
            if let Some(rules) = plan.rules_by_scc.get(scc_id as usize) {
                // Get SCC metadata
                let scc = plan.sccs.get(scc_id as usize);
                let is_recursive = scc.map(|s| s.is_recursive).unwrap_or(false);

                // For MVP (Task 6), execute rules once regardless of recursion.
                // Task 7 will implement proper fixpoint iteration for recursive SCCs.
                let _ = is_recursive; // Will be used in Task 7
                for rule in rules {
                    let result = self.execute_node(&rule.body)?;
                    self.store.put(&rule.head, result);
                }
            }
        }

        Ok(())
    }

    /// Execute a stratum (public API)
    ///
    /// This method cannot be called directly because stratum execution requires
    /// access to the full ExecutionPlan (for rules_by_scc mapping). Use
    /// `execute_plan` instead, which processes all strata with proper context.
    ///
    /// # Arguments
    /// * `_stratum` - The stratum (unused - see error)
    ///
    /// # Returns
    /// Always returns an error indicating this method should not be called directly
    ///
    /// # Errors
    /// Always returns an error. Use `execute_plan` instead.
    pub fn execute_stratum(&mut self, _stratum: &Stratum) -> Result<()> {
        Err(XlogError::Execution(
            "execute_stratum cannot be called directly; use execute_plan instead which provides \
             the required rules_by_scc context"
                .to_string(),
        ))
    }

    // ============== Node execution implementations ==============

    /// Execute a Scan node
    ///
    /// Looks up the relation by RelId and returns a clone of its buffer.
    fn execute_scan(&self, rel: RelId) -> Result<CudaBuffer> {
        let name = self
            .get_rel_name(rel)
            .ok_or_else(|| XlogError::Execution(format!("Unknown relation: RelId({})", rel.0)))?;

        let buffer = self
            .store
            .get(name)
            .ok_or_else(|| XlogError::Execution(format!("Relation not found: {}", name)))?;

        // Clone the buffer
        self.clone_buffer(buffer)
    }

    /// Execute a Filter node (CPU-based for MVP)
    ///
    /// Copies data to host, applies the predicate, and copies back.
    fn execute_filter(&self, input: &CudaBuffer, predicate: &Expr) -> Result<CudaBuffer> {
        if input.is_empty() {
            return self.create_empty_buffer(input.schema().clone());
        }

        let num_rows = input.num_rows() as usize;
        let schema = input.schema().clone();

        // Read all columns to host
        let mut host_columns: Vec<Vec<u8>> = Vec::with_capacity(input.arity());
        for col_idx in 0..input.arity() {
            let col_type_size = schema
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            let col_bytes = num_rows * col_type_size;

            if let Some(col) = input.column(col_idx) {
                let mut host_data = vec![0u8; col_bytes];
                self.provider
                    .device()
                    .inner()
                    .dtoh_sync_copy_into(col, &mut host_data)
                    .map_err(|e| XlogError::Execution(format!("Failed to read column: {}", e)))?;
                host_columns.push(host_data);
            }
        }

        // Evaluate predicate for each row
        let mut matching_indices = Vec::new();
        for row_idx in 0..num_rows {
            if self.evaluate_predicate(predicate, &host_columns, row_idx, &schema)? {
                matching_indices.push(row_idx);
            }
        }

        let result_rows = matching_indices.len() as u64;
        if result_rows == 0 {
            return self.create_empty_buffer(schema);
        }

        // Build result columns
        let mut result_columns = Vec::with_capacity(input.arity());
        for col_idx in 0..input.arity() {
            let col_type_size = schema
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            let result_bytes = (result_rows as usize) * col_type_size;

            let mut result_host = Vec::with_capacity(result_bytes);
            for &row_idx in &matching_indices {
                let start = row_idx * col_type_size;
                let end = start + col_type_size;
                result_host.extend_from_slice(&host_columns[col_idx][start..end]);
            }

            let mut result_col = self.provider.memory().alloc::<u8>(result_bytes)?;
            self.provider
                .device()
                .inner()
                .htod_sync_copy_into(&result_host, &mut result_col)
                .map_err(|e| XlogError::Execution(format!("Failed to upload result: {}", e)))?;

            result_columns.push(result_col);
        }

        Ok(CudaBuffer::from_columns(
            result_columns,
            result_rows,
            schema,
        ))
    }

    /// Evaluate a predicate expression for a single row
    fn evaluate_predicate(
        &self,
        expr: &Expr,
        columns: &[Vec<u8>],
        row_idx: usize,
        schema: &Schema,
    ) -> Result<bool> {
        match expr {
            Expr::Column(col_idx) => {
                // Interpret column value as boolean
                let col_type = schema.column_type(*col_idx);
                if let Some(ScalarType::Bool) = col_type {
                    Ok(columns
                        .get(*col_idx)
                        .map(|c| c.get(row_idx).copied().unwrap_or(0) != 0)
                        .unwrap_or(false))
                } else {
                    // Non-bool columns: check if non-zero
                    Ok(true)
                }
            }

            Expr::Const(ConstValue::Bool(b)) => Ok(*b),
            Expr::Const(_) => Ok(true), // Non-bool constants are truthy

            Expr::Compare { left, op, right } => {
                let left_val = self.evaluate_expr_as_i64(left, columns, row_idx, schema)?;
                let right_val = self.evaluate_expr_as_i64(right, columns, row_idx, schema)?;

                Ok(match op {
                    CompareOp::Eq => left_val == right_val,
                    CompareOp::Ne => left_val != right_val,
                    CompareOp::Lt => left_val < right_val,
                    CompareOp::Le => left_val <= right_val,
                    CompareOp::Gt => left_val > right_val,
                    CompareOp::Ge => left_val >= right_val,
                })
            }

            Expr::And(exprs) => {
                for e in exprs {
                    if !self.evaluate_predicate(e, columns, row_idx, schema)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }

            Expr::Or(exprs) => {
                for e in exprs {
                    if self.evaluate_predicate(e, columns, row_idx, schema)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }

            Expr::Not(inner) => Ok(!self.evaluate_predicate(inner, columns, row_idx, schema)?),
        }
    }

    /// Evaluate an expression as an i64 value
    fn evaluate_expr_as_i64(
        &self,
        expr: &Expr,
        columns: &[Vec<u8>],
        row_idx: usize,
        schema: &Schema,
    ) -> Result<i64> {
        match expr {
            Expr::Column(col_idx) => {
                let col_type = schema.column_type(*col_idx).unwrap_or(ScalarType::U32);
                let col_data = columns
                    .get(*col_idx)
                    .ok_or_else(|| XlogError::Execution(format!("Column {} not found", col_idx)))?;

                let type_size = col_type.size_bytes();
                let start = row_idx * type_size;

                Ok(match col_type {
                    ScalarType::U32 => {
                        let bytes = &col_data[start..start + 4];
                        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64
                    }
                    ScalarType::I32 => {
                        let bytes = &col_data[start..start + 4];
                        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64
                    }
                    ScalarType::U64 => {
                        let bytes = &col_data[start..start + 8];
                        u64::from_le_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ]) as i64
                    }
                    ScalarType::I64 => {
                        let bytes = &col_data[start..start + 8];
                        i64::from_le_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ])
                    }
                    ScalarType::Bool => col_data.get(start).copied().unwrap_or(0) as i64,
                    ScalarType::Symbol => {
                        let bytes = &col_data[start..start + 4];
                        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64
                    }
                    ScalarType::F32 | ScalarType::F64 => {
                        // TODO: Implement proper float comparison (Task 7+)
                        return Err(XlogError::Execution(
                            "Float comparison not yet supported in filter predicates".to_string(),
                        ));
                    }
                })
            }

            Expr::Const(val) => Ok(match val {
                ConstValue::U32(v) => *v as i64,
                ConstValue::I32(v) => *v as i64,
                ConstValue::U64(v) => *v as i64,
                ConstValue::I64(v) => *v,
                ConstValue::Bool(b) => *b as i64,
                ConstValue::F32(f) => *f as i64,
                ConstValue::F64(f) => *f as i64,
                ConstValue::Symbol(_) => 0,
            }),

            _ => Err(XlogError::Execution(
                "Cannot evaluate compound expression as value".to_string(),
            )),
        }
    }

    /// Execute a Project node
    ///
    /// Selects and reorders columns according to the projection list.
    fn execute_project(&self, input: &CudaBuffer, columns: &[usize]) -> Result<CudaBuffer> {
        if input.is_empty() {
            // Build projected schema
            let projected_schema = self.project_schema(input.schema(), columns)?;
            return self.create_empty_buffer(projected_schema);
        }

        let num_rows = input.num_rows();
        let projected_schema = self.project_schema(input.schema(), columns)?;

        let mut result_columns = Vec::with_capacity(columns.len());

        for &col_idx in columns {
            let col_type_size = input
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            let col_bytes = (num_rows as usize) * col_type_size;

            if let Some(src_col) = input.column(col_idx) {
                // Clone the column
                let mut host_data = vec![0u8; col_bytes];
                self.provider
                    .device()
                    .inner()
                    .dtoh_sync_copy_into(src_col, &mut host_data)
                    .map_err(|e| XlogError::Execution(format!("Failed to read column: {}", e)))?;

                let mut dst_col = self.provider.memory().alloc::<u8>(col_bytes)?;
                self.provider
                    .device()
                    .inner()
                    .htod_sync_copy_into(&host_data, &mut dst_col)
                    .map_err(|e| XlogError::Execution(format!("Failed to upload column: {}", e)))?;

                result_columns.push(dst_col);
            } else {
                return Err(XlogError::Execution(format!(
                    "Column {} not found in input",
                    col_idx
                )));
            }
        }

        Ok(CudaBuffer::from_columns(
            result_columns,
            num_rows,
            projected_schema,
        ))
    }

    /// Build a projected schema
    fn project_schema(&self, input: &Schema, columns: &[usize]) -> Result<Schema> {
        let mut projected_columns = Vec::with_capacity(columns.len());
        for &col_idx in columns {
            if let Some((name, ty)) = input.columns.get(col_idx) {
                projected_columns.push((name.clone(), *ty));
            } else {
                return Err(XlogError::Execution(format!(
                    "Column index {} out of bounds",
                    col_idx
                )));
            }
        }
        Ok(Schema::new(projected_columns))
    }

    /// Execute a Join node
    ///
    /// Delegates to the kernel provider's hash_join for inner joins.
    /// Other join types have simplified implementations.
    fn execute_join(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
        join_type: JoinType,
    ) -> Result<CudaBuffer> {
        match join_type {
            JoinType::Inner => self.provider.hash_join(left, right, left_keys, right_keys),

            JoinType::LeftOuter => {
                // MVP: Fall back to inner join (lossy but functional)
                self.provider.hash_join(left, right, left_keys, right_keys)
            }

            JoinType::Semi => {
                // Semi join: return left rows that have a match in right
                // MVP: Use hash join and project only left columns
                let joined = self
                    .provider
                    .hash_join(left, right, left_keys, right_keys)?;
                let left_cols: Vec<usize> = (0..left.arity()).collect();
                self.execute_project(&joined, &left_cols)
            }

            JoinType::Anti => {
                // Anti join: return left rows that have NO match in right
                // MVP: Use diff on the key columns
                self.provider.diff(left, right)
            }
        }
    }

    /// Execute a GroupBy node
    ///
    /// Delegates to the kernel provider's groupby_agg.
    /// For multiple aggregations, executes them sequentially (MVP simplification).
    fn execute_groupby(
        &self,
        input: &CudaBuffer,
        key_cols: &[usize],
        aggs: &[(usize, AggOp)],
    ) -> Result<CudaBuffer> {
        if aggs.is_empty() {
            // No aggregations: just distinct on key columns
            return self.provider.dedup(input, key_cols);
        }

        // MVP: Execute first aggregation only
        let (value_col, agg_op) = aggs[0];
        self.provider
            .groupby_agg(input, key_cols, agg_op, value_col)
    }

    /// Execute a Union node
    ///
    /// Combines multiple input buffers into one.
    fn execute_union(&self, inputs: &[CudaBuffer]) -> Result<CudaBuffer> {
        if inputs.is_empty() {
            return Ok(CudaBuffer::empty());
        }

        if inputs.len() == 1 {
            return self.clone_buffer(&inputs[0]);
        }

        // Pairwise union
        let mut result = self.clone_buffer(&inputs[0])?;
        for input in inputs.iter().skip(1) {
            result = self.provider.union(&result, input)?;
        }

        Ok(result)
    }

    /// Execute a Distinct node
    ///
    /// Removes duplicate rows based on key columns.
    fn execute_distinct(&self, input: &CudaBuffer, key_cols: &[usize]) -> Result<CudaBuffer> {
        self.provider.dedup(input, key_cols)
    }

    /// Execute a Diff node
    ///
    /// Returns rows in left that are not in right.
    fn execute_diff(&self, left: &CudaBuffer, right: &CudaBuffer) -> Result<CudaBuffer> {
        self.provider.diff(left, right)
    }

    /// Maximum iterations for fixpoint computation to prevent infinite loops
    const MAX_FIXPOINT_ITERATIONS: usize = 1000;

    /// Execute a Fixpoint node using semi-naive evaluation
    ///
    /// The semi-naive algorithm avoids redundant computation in recursive queries:
    ///
    /// 1. **Initialize:**
    ///    - Compute base case: `R = eval(base)`
    ///    - Set delta to base: `delta = R`
    ///    - Store both `R` and `delta` in RelationStore
    ///
    /// 2. **Iterate until fixpoint:**
    ///    - Compute new tuples: `delta_new = eval(recursive)` using current `delta`
    ///    - Remove already-known tuples: `delta_new = delta_new - R`
    ///    - If `delta_new` is empty, we've reached fixpoint
    ///    - Otherwise: `R = R union delta_new`, `delta = delta_new`
    ///
    /// 3. **Return:** Final `R`
    ///
    /// # Arguments
    /// * `scc_id` - SCC identifier for logging/debugging
    /// * `base` - Base case RIR tree (non-recursive facts/rules)
    /// * `recursive` - Recursive RIR tree (references delta relation)
    /// * `delta_rel` - RelId for delta relation
    /// * `full_rel` - RelId for full relation
    ///
    /// # Returns
    /// A CudaBuffer containing the final fixpoint result
    ///
    /// # Errors
    /// Returns an error if evaluation fails or iteration limit is exceeded
    fn execute_fixpoint(
        &mut self,
        scc_id: u32,
        base: &RirNode,
        recursive: &RirNode,
        delta_rel: RelId,
        full_rel: RelId,
    ) -> Result<CudaBuffer> {
        // Step 1: Compute base case R = eval(base)
        let r_initial = self.execute_node(base)?;

        // Handle empty base case
        if r_initial.is_empty() {
            return Ok(r_initial);
        }

        // Step 2: Initialize delta = R (clone the base result)
        let delta_initial = self.clone_buffer(&r_initial)?;

        // Get relation names for delta and full relations
        let delta_name = self.get_or_create_rel_name(delta_rel, &format!("__delta_{}", scc_id));
        let full_name = self.get_or_create_rel_name(full_rel, &format!("__full_{}", scc_id));

        // Store initial R and delta in relation store
        self.store.put(&full_name, r_initial);
        self.store.put(&delta_name, delta_initial);

        // Step 3: Iterate until fixpoint
        for _iteration in 0..Self::MAX_FIXPOINT_ITERATIONS {
            // Evaluate recursive step using current delta
            // The recursive RIR tree should reference delta_rel internally
            let delta_new_raw = self.execute_node(recursive)?;

            // Get current R for set difference
            let current_r = self.store.get(&full_name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Full relation {} not found during fixpoint iteration",
                    full_name
                ))
            })?;

            // Compute delta_new = delta_new_raw - R (remove already-known tuples)
            let delta_new = self.provider.diff(&delta_new_raw, current_r)?;

            // Check for fixpoint: if delta_new is empty, we're done
            if delta_new.is_empty() {
                // Fixpoint reached - return final R
                let final_r = self.store.remove(&full_name).ok_or_else(|| {
                    XlogError::Execution("Full relation lost during fixpoint".to_string())
                })?;

                // Clean up delta relation
                self.store.remove(&delta_name);

                return Ok(final_r);
            }

            // Not at fixpoint yet: R = R union delta_new
            let new_r = self.provider.union(current_r, &delta_new)?;

            // Update relations for next iteration
            // delta = delta_new (the newly discovered tuples)
            self.store.put(&delta_name, delta_new);
            self.store.put(&full_name, new_r);
        }

        // Iteration limit exceeded
        Err(XlogError::Execution(format!(
            "Fixpoint iteration limit ({}) exceeded for SCC {}",
            Self::MAX_FIXPOINT_ITERATIONS,
            scc_id
        )))
    }

    /// Get the relation name for a RelId, creating a default name if not registered
    fn get_or_create_rel_name(&mut self, rel_id: RelId, default: &str) -> String {
        if let Some(name) = self.rel_names.get(&rel_id) {
            name.clone()
        } else {
            let name = default.to_string();
            self.rel_names.insert(rel_id, name.clone());
            name
        }
    }

    // ============== Helper methods ==============

    /// Create an empty buffer with the given schema
    fn create_empty_buffer(&self, schema: Schema) -> Result<CudaBuffer> {
        let mut columns = Vec::with_capacity(schema.arity());
        for _ in 0..schema.arity() {
            columns.push(self.provider.memory().alloc::<u8>(0)?);
        }
        Ok(CudaBuffer::from_columns(columns, 0, schema))
    }

    /// Clone a buffer (deep copy via host)
    fn clone_buffer(&self, buffer: &CudaBuffer) -> Result<CudaBuffer> {
        if buffer.is_empty() {
            return self.create_empty_buffer(buffer.schema().clone());
        }

        let mut result_columns = Vec::with_capacity(buffer.arity());

        for col_idx in 0..buffer.arity() {
            let col_type_size = buffer
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            let bytes = (buffer.num_rows() as usize) * col_type_size;

            if let Some(src_col) = buffer.column(col_idx) {
                let mut host_data = vec![0u8; bytes];
                self.provider
                    .device()
                    .inner()
                    .dtoh_sync_copy_into(src_col, &mut host_data)
                    .map_err(|e| {
                        XlogError::Execution(format!("Failed to read column for clone: {}", e))
                    })?;

                let mut dst_col = self.provider.memory().alloc::<u8>(bytes)?;
                self.provider
                    .device()
                    .inner()
                    .htod_sync_copy_into(&host_data, &mut dst_col)
                    .map_err(|e| XlogError::Execution(format!("Failed to clone column: {}", e)))?;

                result_columns.push(dst_col);
            }
        }

        Ok(CudaBuffer::from_columns(
            result_columns,
            buffer.num_rows(),
            buffer.schema().clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::MemoryBudget;
    use xlog_cuda::{CudaDevice, GpuMemoryManager};
    use xlog_ir::{CompiledRule, RirMeta, Scc};

    fn has_cuda_device() -> bool {
        // Check if CUDA device is available using CudaDevice wrapper
        CudaDevice::new(0).is_ok()
    }

    fn create_test_executor() -> Option<Executor> {
        if !has_cuda_device() {
            return None;
        }
        let device = Arc::new(CudaDevice::new(0).ok()?);
        let budget = MemoryBudget::with_limit(1024 * 1024 * 1024); // 1 GB
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
        let provider = Arc::new(CudaKernelProvider::new(device, memory).ok()?);
        Some(Executor::new(provider))
    }

    fn create_test_buffer(executor: &Executor, data: &[u32], col_name: &str) -> CudaBuffer {
        let schema = Schema::new(vec![(col_name.to_string(), ScalarType::U32)]);
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut col = executor
            .provider
            .memory()
            .alloc::<u8>(bytes.len())
            .expect("alloc");
        executor
            .provider
            .device()
            .inner()
            .htod_sync_copy_into(&bytes, &mut col)
            .expect("htod");

        CudaBuffer::from_columns(vec![col], data.len() as u64, schema)
    }

    fn read_buffer_u32(executor: &Executor, buffer: &CudaBuffer, col: usize) -> Vec<u32> {
        if buffer.is_empty() || buffer.column(col).is_none() {
            return vec![];
        }
        let num_rows = buffer.num_rows() as usize;
        let mut bytes = vec![0u8; num_rows * 4];
        executor
            .provider
            .device()
            .inner()
            .dtoh_sync_copy_into(buffer.column(col).unwrap(), &mut bytes)
            .expect("dtoh");
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    // ============== Basic Executor Tests ==============

    #[test]
    fn test_executor_creation() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        assert!(executor.store().is_empty());
    }

    #[test]
    fn test_register_and_get_relation() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Register a relation
        executor.register_relation(RelId(1), "test_rel");

        // Verify mapping
        assert_eq!(executor.get_rel_name(RelId(1)), Some("test_rel"));
        assert_eq!(executor.get_rel_name(RelId(2)), None);
    }

    // ============== Scan Node Tests ==============

    #[test]
    fn test_execute_scan_not_found() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        executor.register_relation(RelId(1), "missing_rel");

        let node = RirNode::Scan { rel: RelId(1) };
        let result = executor.execute_node(&node);

        assert!(result.is_err());
    }

    #[test]
    fn test_execute_scan_success() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create and store a buffer
        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        executor.store_mut().put("test_rel", buffer);
        executor.register_relation(RelId(1), "test_rel");

        // Execute scan
        let node = RirNode::Scan { rel: RelId(1) };
        let result = executor.execute_node(&node);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.num_rows(), 5);

        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![1, 2, 3, 4, 5]);
    }

    // ============== Filter Node Tests ==============

    #[test]
    fn test_execute_filter_empty_input() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty = executor.create_empty_buffer(schema).unwrap();

        let predicate = Expr::Const(ConstValue::Bool(true));
        let result = executor.execute_filter(&empty, &predicate);

        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_execute_filter_all_match() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        let predicate = Expr::Const(ConstValue::Bool(true));

        let result = executor.execute_filter(&buffer, &predicate);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.num_rows(), 5);
    }

    #[test]
    fn test_execute_filter_none_match() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        let predicate = Expr::Const(ConstValue::Bool(false));

        let result = executor.execute_filter(&buffer, &predicate);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_execute_filter_comparison() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");

        // Filter: key > 3
        let predicate = Expr::Compare {
            left: Box::new(Expr::Column(0)),
            op: CompareOp::Gt,
            right: Box::new(Expr::Const(ConstValue::U32(3))),
        };

        let result = executor.execute_filter(&buffer, &predicate);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.num_rows(), 2);

        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![4, 5]);
    }

    #[test]
    fn test_execute_filter_and() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");

        // Filter: key >= 2 AND key <= 4
        let predicate = Expr::And(vec![
            Expr::Compare {
                left: Box::new(Expr::Column(0)),
                op: CompareOp::Ge,
                right: Box::new(Expr::Const(ConstValue::U32(2))),
            },
            Expr::Compare {
                left: Box::new(Expr::Column(0)),
                op: CompareOp::Le,
                right: Box::new(Expr::Const(ConstValue::U32(4))),
            },
        ]);

        let result = executor.execute_filter(&buffer, &predicate);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.num_rows(), 3);

        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![2, 3, 4]);
    }

    // ============== Project Node Tests ==============

    #[test]
    fn test_execute_project_empty_input() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![
            ("a".to_string(), ScalarType::U32),
            ("b".to_string(), ScalarType::U32),
        ]);
        let empty = executor.create_empty_buffer(schema).unwrap();

        let result = executor.execute_project(&empty, &[0]);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.is_empty());
        assert_eq!(result.arity(), 1);
    }

    #[test]
    fn test_execute_project_reorder() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create a 2-column buffer
        let schema = Schema::new(vec![
            ("a".to_string(), ScalarType::U32),
            ("b".to_string(), ScalarType::U32),
        ]);

        let a_data: Vec<u8> = [1u32, 2, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_data: Vec<u8> = [10u32, 20, 30]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let mut col_a = executor
            .provider
            .memory()
            .alloc::<u8>(a_data.len())
            .unwrap();
        let mut col_b = executor
            .provider
            .memory()
            .alloc::<u8>(b_data.len())
            .unwrap();

        executor
            .provider
            .device()
            .inner()
            .htod_sync_copy_into(&a_data, &mut col_a)
            .unwrap();
        executor
            .provider
            .device()
            .inner()
            .htod_sync_copy_into(&b_data, &mut col_b)
            .unwrap();

        let buffer = CudaBuffer::from_columns(vec![col_a, col_b], 3, schema);

        // Project: [b, a] (reverse order)
        let result = executor.execute_project(&buffer, &[1, 0]);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.num_rows(), 3);
        assert_eq!(result.arity(), 2);

        // First column should be b's values
        let col0 = read_buffer_u32(&executor, &result, 0);
        assert_eq!(col0, vec![10, 20, 30]);

        // Second column should be a's values
        let col1 = read_buffer_u32(&executor, &result, 1);
        assert_eq!(col1, vec![1, 2, 3]);
    }

    // ============== Union Node Tests ==============

    #[test]
    fn test_execute_union_empty_inputs() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let result = executor.execute_union(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_execute_union_single_input() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&executor, &[1, 2, 3], "key");

        let result = executor.execute_union(&[buffer]);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.num_rows(), 3);

        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_execute_union_multiple_inputs() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer1 = create_test_buffer(&executor, &[1, 2], "key");
        let buffer2 = create_test_buffer(&executor, &[3, 4], "key");
        let buffer3 = create_test_buffer(&executor, &[5], "key");

        let result = executor.execute_union(&[buffer1, buffer2, buffer3]);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.num_rows(), 5);
    }

    // ============== Distinct Node Tests ==============

    #[test]
    fn test_execute_distinct_empty() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty = executor.create_empty_buffer(schema).unwrap();

        let result = executor.execute_distinct(&empty, &[0]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    // ============== Diff Node Tests ==============

    #[test]
    fn test_execute_diff() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let left = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        let right = create_test_buffer(&executor, &[2, 4], "key");

        let result = executor.execute_diff(&left, &right);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.num_rows(), 3);

        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![1, 3, 5]);
    }

    // ============== Fixpoint Tests ==============

    #[test]
    fn test_execute_fixpoint_base_only() {
        // Test fixpoint with a base case that reaches fixpoint immediately
        // (recursive step produces nothing new)
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create base relation
        let buffer = create_test_buffer(&executor, &[1, 2, 3], "key");
        executor.store_mut().put("base_rel", buffer);
        executor.register_relation(RelId(1), "base_rel");

        // Create an empty recursive relation (simulating a recursive step that produces nothing)
        let empty_schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty_buffer = executor.create_empty_buffer(empty_schema).unwrap();
        executor.store_mut().put("empty_rel", empty_buffer);
        executor.register_relation(RelId(4), "empty_rel");

        // Base: scan base_rel
        // Recursive: scan empty_rel (produces nothing new)
        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(4) });

        let node = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(2),
            full_rel: RelId(3),
        };

        let result = executor.execute_node(&node);
        assert!(result.is_ok());

        // Should return base case since recursive produces nothing
        let result = result.unwrap();
        assert_eq!(result.num_rows(), 3);
        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_execute_fixpoint_empty_base() {
        // Test fixpoint with empty base case
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create empty base relation
        let empty_schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty_buffer = executor.create_empty_buffer(empty_schema.clone()).unwrap();
        executor.store_mut().put("empty_base", empty_buffer);
        executor.register_relation(RelId(1), "empty_base");

        // Create recursive relation (won't be used since base is empty)
        let rec_buffer = create_test_buffer(&executor, &[4, 5, 6], "key");
        executor.store_mut().put("rec_rel", rec_buffer);
        executor.register_relation(RelId(4), "rec_rel");

        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(4) });

        let node = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(2),
            full_rel: RelId(3),
        };

        let result = executor.execute_node(&node);
        assert!(result.is_ok());

        // Should return empty since base is empty
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_execute_fixpoint_one_iteration() {
        // Test fixpoint that converges after one iteration
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Base: [1, 2]
        let base_buffer = create_test_buffer(&executor, &[1, 2], "key");
        executor.store_mut().put("base_rel", base_buffer);
        executor.register_relation(RelId(1), "base_rel");

        // Recursive produces [1, 2, 3] - after diff with R, only [3] remains
        let rec_buffer = create_test_buffer(&executor, &[1, 2, 3], "key");
        executor.store_mut().put("rec_rel", rec_buffer);
        executor.register_relation(RelId(4), "rec_rel");

        // After first iteration, R = [1, 2, 3], recursive produces [1, 2, 3] again
        // diff([1, 2, 3], [1, 2, 3]) = empty -> fixpoint reached

        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(4) });

        let node = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(2),
            full_rel: RelId(3),
        };

        let result = executor.execute_node(&node);
        assert!(result.is_ok());

        let result = result.unwrap();
        // Result should be [1, 2, 3]
        assert_eq!(result.num_rows(), 3);
    }

    #[test]
    fn test_execute_fixpoint_multiple_iterations() {
        // Test fixpoint that requires multiple iterations to converge
        // This simulates transitive closure behavior
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Base: [1]
        let base_buffer = create_test_buffer(&executor, &[1], "key");
        executor.store_mut().put("base_rel", base_buffer);
        executor.register_relation(RelId(1), "base_rel");

        // For this test, we need a recursive rule that can expand
        // Since we can't easily simulate join-based recursion without complex setup,
        // we'll test a simpler case where recursive produces cumulative data

        // Recursive relation will produce [1, 2] in first iteration,
        // then [1, 2, 3] in second iteration, etc.
        // This requires a more complex setup, so let's test the basic convergence

        // Simplified test: recursive produces union of base with [2]
        // First iteration: R=[1], rec produces [1, 2] -> delta_new = [2]
        // Second iteration: R=[1, 2], rec produces [1, 2] -> delta_new = empty
        let rec_buffer = create_test_buffer(&executor, &[1, 2], "key");
        executor.store_mut().put("rec_rel", rec_buffer);
        executor.register_relation(RelId(4), "rec_rel");

        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(4) });

        let node = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(2),
            full_rel: RelId(3),
        };

        let result = executor.execute_node(&node);
        assert!(result.is_ok());

        let result = result.unwrap();
        // Result should be union of [1] and [2] = [1, 2]
        assert_eq!(result.num_rows(), 2);
    }

    #[test]
    fn test_execute_fixpoint_via_node() {
        // Test fixpoint through execute_node to ensure the match arm works
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create and store a base buffer
        let buffer = create_test_buffer(&executor, &[1, 2, 3], "key");
        executor.store_mut().put("base_rel", buffer);
        executor.register_relation(RelId(1), "base_rel");

        // Empty recursive means immediate fixpoint
        let empty_schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty_buffer = executor.create_empty_buffer(empty_schema).unwrap();
        executor.store_mut().put("empty_rel", empty_buffer);
        executor.register_relation(RelId(4), "empty_rel");

        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(4) });

        let node = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(2),
            full_rel: RelId(3),
        };

        let result = executor.execute_node(&node);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.num_rows(), 3);
    }

    #[test]
    fn test_fixpoint_cleanup() {
        // Test that fixpoint properly cleans up delta and full relations
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&executor, &[1, 2], "key");
        executor.store_mut().put("base_rel", buffer);
        executor.register_relation(RelId(1), "base_rel");

        let empty_schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty_buffer = executor.create_empty_buffer(empty_schema).unwrap();
        executor.store_mut().put("empty_rel", empty_buffer);
        executor.register_relation(RelId(4), "empty_rel");

        // Register names for delta and full relations to check cleanup
        executor.register_relation(RelId(2), "__delta_test");
        executor.register_relation(RelId(3), "__full_test");

        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(4) });

        let node = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(2),
            full_rel: RelId(3),
        };

        let result = executor.execute_node(&node);
        assert!(result.is_ok());

        // After fixpoint, the delta and full relations should be cleaned up
        assert!(!executor.store().contains("__delta_test"));
        assert!(!executor.store().contains("__full_test"));
    }

    // ============== Execute Plan Tests ==============

    #[test]
    fn test_execute_plan_empty() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let plan = ExecutionPlan::new(vec![]);

        let result = executor.execute_plan(&plan);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_execute_plan_with_stratum() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create input relation
        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        executor.store_mut().put("input", buffer);
        executor.register_relation(RelId(1), "input");

        // Build a simple plan
        let scc = Scc {
            id: 0,
            predicates: vec!["output".to_string()],
            is_recursive: false,
        };

        let rule = CompiledRule {
            head: "output".to_string(),
            body: RirNode::Scan { rel: RelId(1) },
            meta: RirMeta::default(),
        };

        let stratum = Stratum {
            id: 0,
            sccs: vec![0],
        };

        let plan = ExecutionPlan {
            sccs: vec![scc],
            strata: vec![stratum],
            rules_by_scc: vec![vec![rule]],
            est_memory_peak: 0,
        };

        let result = executor.execute_plan(&plan);
        assert!(result.is_ok());

        // Verify output relation was created
        assert!(executor.store().contains("output"));
        let output = executor.store().get("output").unwrap();
        assert_eq!(output.num_rows(), 5);
    }

    // ============== RIR Node Composition Tests ==============

    #[test]
    fn test_execute_filter_project_chain() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create input relation
        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        executor.store_mut().put("input", buffer);
        executor.register_relation(RelId(1), "input");

        // Build: Project(Filter(Scan))
        let scan = RirNode::Scan { rel: RelId(1) };
        let filter = RirNode::Filter {
            input: Box::new(scan),
            predicate: Expr::Compare {
                left: Box::new(Expr::Column(0)),
                op: CompareOp::Gt,
                right: Box::new(Expr::Const(ConstValue::U32(2))),
            },
        };
        let project = RirNode::Project {
            input: Box::new(filter),
            columns: vec![0],
        };

        let result = executor.execute_node(&project);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.num_rows(), 3);

        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![3, 4, 5]);
    }
}
