//! Integration tests for ILP TensorMaskedJoin execution.
//!
//! These tests require a CUDA device to run.
//! Covers RFC T3.1 (identity mask), T3.2 (empty mask), T3.3 (no mask noop).

#![allow(clippy::arc_with_non_send_sync)]

use std::collections::HashMap;
use std::sync::Arc;
use xlog_core::{MemoryBudget, RelId, ScalarType, Schema};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::{parse_program, Compiler};
use xlog_runtime::{read_device_row_count, Executor};

fn setup() -> Option<(Arc<CudaKernelProvider>, Compiler)> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1024 * 1024 * 1024),
    ));
    let provider = Arc::new(CudaKernelProvider::new(device, memory).ok()?);
    let compiler = Compiler::new();
    Some((provider, compiler))
}

/// Helper: full relation setup per RD-15 + fact loading per RD-26.
///
/// Registers all relations, creates empty buffers for all schemas, then
/// loads base facts. Also creates empty buffers for any relations in
/// rel_ids that don't have schemas (e.g. learnable rule heads that the
/// schema inference doesn't cover), using a default 2-column U32 schema.
fn setup_executor_with_facts(
    provider: &Arc<CudaKernelProvider>,
    compiler: &Compiler,
    ast: &xlog_logic::ast::Program,
    executor: &mut Executor,
) {
    // Step 1: Register relations
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }

    // Step 2: Pre-seed schemas from compiler
    for (name, schema) in compiler.schemas() {
        let empty = provider.create_empty_buffer(schema.clone()).unwrap();
        executor.store_mut().put(name, empty);
    }

    // Step 3: Create empty buffers for relations without schemas
    // (e.g. learnable rule head predicates like "reach" that have rel_ids
    // but no schema from fact inference)
    let default_schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ]);
    for (name, _rel_id) in compiler.rel_ids() {
        if executor.store().get(name).is_none() {
            let empty = provider
                .create_empty_buffer(default_schema.clone())
                .unwrap();
            executor.store_mut().put(name, empty);
        }
    }

    // Step 4: Load base facts (RD-26)
    load_test_facts(provider, compiler, ast, executor);
}

/// Simplified test-side fact loader for integration tests.
fn load_test_facts(
    provider: &Arc<CudaKernelProvider>,
    compiler: &Compiler,
    ast: &xlog_logic::ast::Program,
    executor: &mut Executor,
) {
    let mut fact_groups: HashMap<String, Vec<Vec<i64>>> = HashMap::new();
    for rule in ast.facts() {
        let pred = &rule.head.predicate;
        let terms: Vec<i64> = rule
            .head
            .terms
            .iter()
            .map(|t| match t {
                xlog_logic::ast::Term::Integer(v) => *v,
                xlog_logic::ast::Term::Symbol(id) => *id as i64,
                _ => panic!("Test helper only handles Integer/Symbol terms"),
            })
            .collect();
        fact_groups.entry(pred.clone()).or_default().push(terms);
    }

    for (pred, rows) in &fact_groups {
        if rows.is_empty() {
            continue;
        }
        let arity = rows[0].len();

        let schema = compiler.schemas().get(pred).cloned().unwrap_or_else(|| {
            Schema::new(
                (0..arity)
                    .map(|i| (format!("c{}", i), ScalarType::U32))
                    .collect(),
            )
        });

        // Build column-major byte slices, encoding according to schema type
        let mut columns: Vec<Vec<u8>> = vec![Vec::new(); arity];
        for row in rows {
            for (col_idx, val) in row.iter().enumerate() {
                let col_type = schema.column_type(col_idx).unwrap_or(ScalarType::U32);
                match col_type {
                    ScalarType::U32 => {
                        columns[col_idx].extend_from_slice(&(*val as u32).to_ne_bytes());
                    }
                    ScalarType::I64 => {
                        columns[col_idx].extend_from_slice(&(*val).to_ne_bytes());
                    }
                    ScalarType::U64 => {
                        columns[col_idx].extend_from_slice(&(*val as u64).to_ne_bytes());
                    }
                    _ => {
                        // Default to u32 for other types
                        columns[col_idx].extend_from_slice(&(*val as u32).to_ne_bytes());
                    }
                }
            }
        }

        let col_refs: Vec<&[u8]> = columns.iter().map(|c| c.as_slice()).collect();
        let buf = provider
            .create_buffer_from_slices(&col_refs, schema)
            .unwrap();

        // Union with existing (which is the empty seed)
        if let Some(existing) = executor.store().get(pred) {
            let merged = provider.union_gpu(existing, &buf).unwrap();
            executor.store_mut().put(pred, merged);
        } else {
            executor.store_mut().put(pred, buf);
        }
    }
}

/// T3.3: No mask registered -- TensorMaskedJoin returns no-op with correct
/// head schema, no store corruption.
#[test]
fn test_tmj_no_mask_noop() {
    let Some((provider, mut compiler)) = setup() else {
        return;
    };

    let input = r#"
        b1(1, 2).
        b1(2, 3).
        b2(1, 2).
        b2(2, 3).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    "#;

    let ast = parse_program(input).unwrap();
    let plan = compiler.compile_program(&ast).unwrap();

    let mut executor = Executor::new(provider.clone());
    setup_executor_with_facts(&provider, &compiler, &ast, &mut executor);

    // Execute without setting a mask -- should no-op gracefully (RD-12)
    let result = executor.execute_plan(&plan);
    assert!(
        result.is_ok(),
        "No-mask execution failed: {:?}",
        result.err()
    );

    // Last ILP result should be empty
    let tagged = executor.ilp_last_result();
    assert!(tagged.is_some());
    assert!(tagged.unwrap().entries.is_empty());

    // Verify b1 facts were loaded (RD-26 fact loading worked)
    let b1_buf = executor.store().get("b1");
    assert!(b1_buf.is_some(), "b1 relation should exist in store");
    let b1_rows = read_device_row_count(&provider, b1_buf.unwrap()).unwrap();
    assert_eq!(b1_rows, 2, "b1 should have 2 facts");
}

/// T3.2: Empty mask (all zeros) -> no derivations.
#[test]
fn test_tmj_empty_mask_no_derivations() {
    let Some((provider, mut compiler)) = setup() else {
        return;
    };

    let input = r#"
        b1(1, 2).
        b1(2, 3).
        b2(1, 2).
        b2(2, 3).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    "#;

    let ast = parse_program(input).unwrap();
    let plan = compiler.compile_program(&ast).unwrap();

    let mut executor = Executor::new(provider.clone());
    setup_executor_with_facts(&provider, &compiler, &ast, &mut executor);

    // Determine schema size from compiled rel_ids
    let n = compiler.rel_ids().len();
    let total = n * n * n;

    // Create all-zero mask (no active rules)
    let hard_data = vec![0.0f32; total];
    let soft_data = vec![0.0f32; total];
    let schema_1d = Schema::new(vec![("c0".to_string(), ScalarType::F32)]);
    let hard_buf = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&hard_data)], schema_1d.clone())
        .unwrap();
    let soft_buf = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&soft_data)], schema_1d)
        .unwrap();

    executor
        .ilp_registry_mut()
        .insert_mask("W".to_string(), hard_buf, soft_buf, n);

    let result = executor.execute_plan(&plan);
    assert!(
        result.is_ok(),
        "Empty-mask execution failed: {:?}",
        result.err()
    );

    let tagged = executor.ilp_last_result().unwrap();
    assert!(
        tagged.entries.is_empty(),
        "Empty mask should produce no results"
    );
}

/// T3.1: Identity mask (b1 join b2 for reach) -> correct join results.
/// Sets mask so that reach(X,Y) :- b1(X,Z), b2(Z,Y) is active.
#[test]
fn test_tmj_identity_mask_correct_join() {
    let Some((provider, mut compiler)) = setup() else {
        return;
    };

    let input = r#"
        b1(1, 2).
        b1(2, 3).
        b2(1, 2).
        b2(2, 3).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    "#;

    let ast = parse_program(input).unwrap();
    let plan = compiler.compile_program(&ast).unwrap();

    let mut executor = Executor::new(provider.clone());
    setup_executor_with_facts(&provider, &compiler, &ast, &mut executor);

    // Build rel_index to find b1, b2, and reach indices
    let mut rel_index: Vec<(RelId, String)> = compiler
        .rel_ids()
        .iter()
        .map(|(name, id)| (*id, name.clone()))
        .collect();
    rel_index.sort_by_key(|(id, _)| id.0);
    let n = rel_index.len();

    let b1_idx = rel_index.iter().position(|(_, name)| name == "b1").unwrap();
    let b2_idx = rel_index.iter().position(|(_, name)| name == "b2").unwrap();
    let reach_idx = rel_index
        .iter()
        .position(|(_, name)| name == "reach")
        .unwrap();

    // Create mask: activate (b1, b2) -> reach
    // Flat index = b1_idx * N^2 + b2_idx * N + reach_idx
    let total = n * n * n;
    let mut hard_data = vec![0.0f32; total];
    let mut soft_data = vec![0.0f32; total];
    let flat_idx = b1_idx * n * n + b2_idx * n + reach_idx;
    hard_data[flat_idx] = 1.0;
    soft_data[flat_idx] = 0.95;

    let schema_1d = Schema::new(vec![("c0".to_string(), ScalarType::F32)]);
    let hard_buf = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&hard_data)], schema_1d.clone())
        .unwrap();
    let soft_buf = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&soft_data)], schema_1d)
        .unwrap();

    executor
        .ilp_registry_mut()
        .insert_mask("W".to_string(), hard_buf, soft_buf, n);

    let result = executor.execute_plan(&plan);
    assert!(result.is_ok(), "Mask execution failed: {:?}", result.err());

    // Tag metadata should have one entry: (b1, b2) -> reach
    let tagged = executor.ilp_last_result().unwrap();
    assert!(
        !tagged.entries.is_empty(),
        "Identity mask should produce results"
    );

    let entry = &tagged.entries[0];
    assert_eq!(entry.i as usize, b1_idx);
    assert_eq!(entry.j as usize, b2_idx);
    assert_eq!(entry.k as usize, reach_idx);

    // b1 join b2 on shared variable gives reach rows:
    //   b1(1,2) join b2(2,3) -> reach(1,3)
    //   b1(2,3) join b2(?,?) -- b1.c1=3 needs b2.c0=3, which doesn't exist
    // So we expect at least 1 row
    assert!(entry.num_rows >= 1, "Join should produce at least 1 row");
}

// T3.4: Tag metadata matches active rules
#[test]
fn test_tmj_tag_metadata_correct() {
    let Some((provider, mut compiler)) = setup() else {
        return;
    };

    let input = r#"
        edge(1, 2).
        edge(2, 3).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    "#;
    let ast = parse_program(input).unwrap();
    let plan = compiler.compile_program(&ast).unwrap();
    let mut executor = Executor::new(provider.clone());
    setup_executor_with_facts(&provider, &compiler, &ast, &mut executor);

    let mut rel_index: Vec<(RelId, String)> = compiler
        .rel_ids()
        .iter()
        .map(|(name, id)| (*id, name.clone()))
        .collect();
    rel_index.sort_by_key(|(id, _)| id.0);
    let n = rel_index.len();

    let edge_idx = rel_index
        .iter()
        .position(|(_, name)| name == "edge")
        .unwrap();
    let reach_idx = rel_index
        .iter()
        .position(|(_, name)| name == "reach")
        .unwrap();

    let total = n * n * n;
    let mut hard_data = vec![0.0f32; total];
    let mut soft_data = vec![0.0f32; total];
    let flat_idx = edge_idx * n * n + edge_idx * n + reach_idx;
    hard_data[flat_idx] = 1.0;
    soft_data[flat_idx] = 0.95;

    let schema_1d = Schema::new(vec![("c0".to_string(), ScalarType::F32)]);
    let hard_buf = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&hard_data)], schema_1d.clone())
        .unwrap();
    let soft_buf = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&soft_data)], schema_1d)
        .unwrap();
    executor
        .ilp_registry_mut()
        .insert_mask("W".to_string(), hard_buf, soft_buf, n);

    executor.execute_plan(&plan).unwrap();

    let tagged = executor.ilp_last_result().unwrap();
    assert_eq!(tagged.entries.len(), 1);
    let e = &tagged.entries[0];
    assert_eq!(e.i as usize, edge_idx, "Tag i should be edge index");
    assert_eq!(e.j as usize, edge_idx, "Tag j should be edge index");
    assert_eq!(e.k as usize, reach_idx, "Tag k should be reach index");
    assert!(e.num_rows > 0, "Tag num_rows should be positive");
}

// T3.7: Diff against existing facts — only new facts added
#[test]
fn test_tmj_diff_no_duplicate_facts() {
    let Some((provider, mut compiler)) = setup() else {
        return;
    };

    let input = r#"
        edge(1, 2).
        edge(2, 3).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    "#;
    let ast = parse_program(input).unwrap();
    let plan = compiler.compile_program(&ast).unwrap();
    let mut executor = Executor::new(provider.clone());
    setup_executor_with_facts(&provider, &compiler, &ast, &mut executor);

    let mut rel_index: Vec<(RelId, String)> = compiler
        .rel_ids()
        .iter()
        .map(|(name, id)| (*id, name.clone()))
        .collect();
    rel_index.sort_by_key(|(id, _)| id.0);
    let n = rel_index.len();

    let edge_idx = rel_index
        .iter()
        .position(|(_, name)| name == "edge")
        .unwrap();
    let reach_idx = rel_index
        .iter()
        .position(|(_, name)| name == "reach")
        .unwrap();

    let total = n * n * n;
    let mut hard_data = vec![0.0f32; total];
    let mut soft_data = vec![0.0f32; total];
    hard_data[edge_idx * n * n + edge_idx * n + reach_idx] = 1.0;
    soft_data[edge_idx * n * n + edge_idx * n + reach_idx] = 0.95;

    let schema_1d = Schema::new(vec![("c0".to_string(), ScalarType::F32)]);

    // First execution
    let h1 = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&hard_data)], schema_1d.clone())
        .unwrap();
    let s1 = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&soft_data)], schema_1d.clone())
        .unwrap();
    executor
        .ilp_registry_mut()
        .insert_mask("W".to_string(), h1, s1, n);
    executor.execute_plan(&plan).unwrap();

    let reach_rows_1 = executor
        .store()
        .get("reach")
        .map(|b| read_device_row_count(&provider, b).unwrap())
        .unwrap_or(0);

    // Second execution with same mask
    let h2 = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&hard_data)], schema_1d.clone())
        .unwrap();
    let s2 = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&soft_data)], schema_1d)
        .unwrap();
    executor
        .ilp_registry_mut()
        .insert_mask("W".to_string(), h2, s2, n);
    executor.execute_plan(&plan).unwrap();

    let reach_rows_2 = executor
        .store()
        .get("reach")
        .map(|b| read_device_row_count(&provider, b).unwrap())
        .unwrap_or(0);

    assert_eq!(
        reach_rows_1, reach_rows_2,
        "Re-execution with same mask must not duplicate facts (diff_gpu)"
    );
}
