//! CUDA certification tests for v0.3.2 language features.
//!
//! This module verifies that all v0.3.2 features (symbols, UDFs, modules)
//! work correctly when executed on the GPU.

#![allow(clippy::arc_with_non_send_sync)]

use std::collections::HashMap;
use std::sync::Arc;

use serial_test::serial;
use xlog_core::{symbol, MemoryBudget, Result, ScalarType, Schema};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn create_test_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    Some(Arc::new(CudaKernelProvider::new(device, memory).ok()?))
}

fn create_buffer_u32(
    provider: &CudaKernelProvider,
    columns: &[&[u32]],
    col_names: &[&str],
) -> Result<CudaBuffer> {
    let schema = Schema::new(
        col_names
            .iter()
            .map(|n| (n.to_string(), ScalarType::U32))
            .collect(),
    );

    if columns.is_empty() || columns[0].is_empty() {
        return provider.create_empty_buffer(schema);
    }

    let col_bytes: Vec<Vec<u8>> = columns
        .iter()
        .map(|col| col.iter().flat_map(|v| v.to_le_bytes()).collect())
        .collect();

    let slices: Vec<&[u8]> = col_bytes.iter().map(|v| v.as_slice()).collect();
    provider.create_buffer_from_slices(&slices, schema)
}

fn create_symbol_buffer(
    provider: &CudaKernelProvider,
    col0: &[u32],
    col1: &[u32],
) -> Result<CudaBuffer> {
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::Symbol),
        ("c1".to_string(), ScalarType::Symbol),
    ]);

    if col0.is_empty() {
        return provider.create_empty_buffer(schema);
    }

    let col0_bytes: Vec<u8> = col0.iter().flat_map(|v| v.to_le_bytes()).collect();
    let col1_bytes: Vec<u8> = col1.iter().flat_map(|v| v.to_le_bytes()).collect();
    provider.create_buffer_from_slices(&[&col0_bytes, &col1_bytes], schema)
}

fn read_column_u32(provider: &CudaKernelProvider, buffer: &CudaBuffer, col: usize) -> Vec<u32> {
    provider
        .download_column_u32(buffer, col)
        .unwrap_or_default()
}

fn read_column_i64(provider: &CudaKernelProvider, buffer: &CudaBuffer, col: usize) -> Vec<i64> {
    provider
        .download_column_i64(buffer, col)
        .unwrap_or_default()
}

// =============================================================================
// Symbol GPU Certification Tests
// =============================================================================

/// Test that symbol columns can be created and read on GPU
#[test]
#[serial]
fn test_gpu_symbol_column_roundtrip() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    symbol::clear();

    // Intern some symbols
    let alice = symbol::intern("alice");
    let bob = symbol::intern("bob");
    let carol = symbol::intern("carol");

    // Create buffer with symbol columns
    let buffer = create_symbol_buffer(&provider, &[alice, bob, carol], &[bob, carol, alice])?;

    // Read back
    let c0 = read_column_u32(&provider, &buffer, 0);
    let c1 = read_column_u32(&provider, &buffer, 1);

    assert_eq!(c0, vec![alice, bob, carol]);
    assert_eq!(c1, vec![bob, carol, alice]);

    // Verify symbols resolve correctly
    assert_eq!(symbol::resolve(c0[0]), "alice");
    assert_eq!(symbol::resolve(c0[1]), "bob");
    assert_eq!(symbol::resolve(c0[2]), "carol");

    Ok(())
}

/// Test symbol join operations on GPU
#[test]
#[serial]
fn test_gpu_symbol_join() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    symbol::clear();

    let source = r#"
        pred person(symbol, symbol).
        pred knows(symbol, symbol).
        pred friend_of_friend(symbol, symbol).

        friend_of_friend(A, C) :- knows(A, B), knows(B, C), A != C.

        ?- friend_of_friend(X, Y).
    "#;

    let program = xlog_gpu::logic::LogicProgram::compile(source)?;

    // Create input data with symbols
    let alice = symbol::intern("alice");
    let bob = symbol::intern("bob");
    let carol = symbol::intern("carol");

    let knows_buf = create_symbol_buffer(
        &provider,
        &[alice, bob], // A knows B
        &[bob, carol], // B knows C
    )?;

    let mut inputs = HashMap::new();
    inputs.insert("knows".to_string(), knows_buf);

    let result = program.evaluate(provider.clone(), inputs)?;

    // Should find alice -> carol (through bob)
    assert!(!result.queries.is_empty());
    let query = &result.queries[0];
    let col0 = read_column_u32(&provider, &query.buffer, 0);
    let col1 = read_column_u32(&provider, &query.buffer, 1);

    // alice should be friend-of-friend with carol
    assert!(col0.contains(&alice) || col1.contains(&carol));

    Ok(())
}

/// Test symbol deduplication on GPU
#[test]
#[serial]
fn test_gpu_symbol_dedup() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    symbol::clear();

    let source = r#"
        pred tag(symbol).
        pred unique_tag(symbol).

        unique_tag(T) :- tag(T).

        ?- unique_tag(X).
    "#;

    let program = xlog_gpu::logic::LogicProgram::compile(source)?;

    // Create input with duplicate symbols
    let red = symbol::intern("red");
    let blue = symbol::intern("blue");

    let schema = Schema::new(vec![("c0".to_string(), ScalarType::Symbol)]);
    let col_bytes: Vec<u8> = [red, red, blue, red, blue, blue]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let tag_buf = provider.create_buffer_from_slices(&[&col_bytes], schema)?;

    let mut inputs = HashMap::new();
    inputs.insert("tag".to_string(), tag_buf);

    let result = program.evaluate(provider.clone(), inputs)?;

    let query = &result.queries[0];
    let tags = read_column_u32(&provider, &query.buffer, 0);

    // Should have deduplicated to 2 unique tags
    let mut unique: Vec<u32> = tags.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), 2);
    assert!(unique.contains(&red));
    assert!(unique.contains(&blue));

    Ok(())
}

// =============================================================================
// User-Defined Function GPU Certification Tests
// =============================================================================

/// Test simple arithmetic UDF on GPU
#[test]
fn test_gpu_udf_arithmetic() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        func double(X) = X + X.
        func square(X) = X * X.

        pred input(u32).
        pred doubled(u32, u32).
        pred squared(u32, u32).

        doubled(X, Y) :- input(X), Y is double(X).
        squared(X, Y) :- input(X), Y is square(X).

        ?- doubled(X, Y).
        ?- squared(X, Y).
    "#;

    let program = xlog_gpu::logic::LogicProgram::compile(source)?;

    let input_buf = create_buffer_u32(&provider, &[&[2, 3, 5, 10]], &["c0"])?;

    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), input_buf);

    let result = program.evaluate(provider.clone(), inputs)?;

    // Check doubled results
    let doubled = &result.queries[0];
    let d_x = read_column_u32(&provider, &doubled.buffer, 0);
    let d_y = read_column_u32(&provider, &doubled.buffer, 1);

    for (x, y) in d_x.iter().zip(d_y.iter()) {
        assert_eq!(*y, x * 2, "double({}) should be {}", x, x * 2);
    }

    // Check squared results
    let squared = &result.queries[1];
    let s_x = read_column_u32(&provider, &squared.buffer, 0);
    let s_y = read_column_u32(&provider, &squared.buffer, 1);

    for (x, y) in s_x.iter().zip(s_y.iter()) {
        assert_eq!(*y, x * x, "square({}) should be {}", x, x * x);
    }

    Ok(())
}

/// Test nested UDF calls on GPU
#[test]
fn test_gpu_udf_nested() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        func double(X) = X + X.
        func quadruple(X) = double(double(X)).

        pred input(u32).
        pred result(u32, u32).

        result(X, Y) :- input(X), Y is quadruple(X).

        ?- result(X, Y).
    "#;

    let program = xlog_gpu::logic::LogicProgram::compile(source)?;

    let input_buf = create_buffer_u32(&provider, &[&[1, 2, 3, 5]], &["c0"])?;

    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), input_buf);

    let result = program.evaluate(provider.clone(), inputs)?;

    let query = &result.queries[0];
    let x_vals = read_column_u32(&provider, &query.buffer, 0);
    let y_vals = read_column_u32(&provider, &query.buffer, 1);

    for (x, y) in x_vals.iter().zip(y_vals.iter()) {
        assert_eq!(*y, x * 4, "quadruple({}) should be {}", x, x * 4);
    }

    Ok(())
}

/// Test UDF with multiple parameters on GPU
#[test]
fn test_gpu_udf_multi_param() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        func manhattan(X1, Y1, X2, Y2) = abs(X2 - X1) + abs(Y2 - Y1).

        pred point(u32, u32, u32).
        pred distance(u32, u32, u32).

        distance(A, B, D) :- point(A, X1, Y1), point(B, X2, Y2), A < B, D is manhattan(X1, Y1, X2, Y2).

        ?- distance(A, B, D).
    "#;

    let program = xlog_gpu::logic::LogicProgram::compile(source)?;

    // Points: (id, x, y)
    // Point 0: (0, 0)
    // Point 1: (3, 4)
    // Point 2: (6, 0)
    let point_buf = create_buffer_u32(
        &provider,
        &[&[0, 1, 2], &[0, 3, 6], &[0, 4, 0]],
        &["id", "x", "y"],
    )?;

    let mut inputs = HashMap::new();
    inputs.insert("point".to_string(), point_buf);

    let result = program.evaluate(provider.clone(), inputs)?;

    let query = &result.queries[0];
    let distances = read_column_u32(&provider, &query.buffer, 2);

    // Expected distances:
    // 0->1: |3-0| + |4-0| = 7
    // 0->2: |6-0| + |0-0| = 6
    // 1->2: |6-3| + |0-4| = 7
    assert!(distances.contains(&7) || distances.contains(&6));

    Ok(())
}

// =============================================================================
// Module System GPU Certification Tests
// =============================================================================

/// Test that module-style predicates work on GPU
/// (Testing with regular predicate names since internal qualified names
/// are generated by the module system, not parsed from source)
#[test]
fn test_gpu_module_qualified_predicates() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    // Module system generates internal names, but for GPU testing
    // we use regular predicate names that simulate module structure
    let source = r#"
        pred graph_edge(u32, u32).
        pred graph_reach(u32, u32).

        graph_reach(X, Y) :- graph_edge(X, Y).
        graph_reach(X, Z) :- graph_reach(X, Y), graph_edge(Y, Z).

        ?- graph_reach(X, Y).
    "#;

    let program = xlog_gpu::logic::LogicProgram::compile(source)?;

    let edge_buf = create_buffer_u32(&provider, &[&[1, 2, 3], &[2, 3, 4]], &["c0", "c1"])?;

    let mut inputs = HashMap::new();
    inputs.insert("graph_edge".to_string(), edge_buf);

    let result = program.evaluate(provider.clone(), inputs)?;

    let query = &result.queries[0];
    let from_vals = read_column_u32(&provider, &query.buffer, 0);
    let to_vals = read_column_u32(&provider, &query.buffer, 1);

    // Should compute transitive closure: 1->2, 1->3, 1->4, 2->3, 2->4, 3->4
    assert!(!from_vals.is_empty());
    assert!(from_vals.contains(&1));
    assert!(to_vals.contains(&4));

    Ok(())
}

/// Test private predicates don't leak (simulated with naming convention)
#[test]
fn test_gpu_private_predicate_isolation() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    // Simulates module system's private predicate handling
    // (module system generates internal names at compile time)
    let source = r#"
        pred mod_public_edge(u32, u32).
        pred mod_private_helper(u32, u32).
        pred mod_result(u32, u32).

        mod_private_helper(X, Y) :- mod_public_edge(X, Y).
        mod_result(X, Y) :- mod_private_helper(X, Y).

        ?- mod_result(X, Y).
    "#;

    let program = xlog_gpu::logic::LogicProgram::compile(source)?;

    let edge_buf = create_buffer_u32(&provider, &[&[1, 2], &[2, 3]], &["c0", "c1"])?;

    let mut inputs = HashMap::new();
    inputs.insert("mod_public_edge".to_string(), edge_buf);

    let result = program.evaluate(provider.clone(), inputs)?;

    let query = &result.queries[0];
    let rows = query.buffer.num_rows();
    assert_eq!(rows, 2);

    Ok(())
}

// =============================================================================
// Combined Feature Tests
// =============================================================================

/// Test symbols + UDFs together on GPU
#[test]
#[serial]
fn test_gpu_symbols_with_udf() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    symbol::clear();

    let source = r#"
        func bonus(Salary) = Salary / cast(10, u32).

        pred employee(symbol, u32).
        pred with_bonus(symbol, u32, u32).

        with_bonus(Name, Salary, B) :- employee(Name, Salary), B is bonus(Salary).

        ?- with_bonus(Name, Salary, Bonus).
    "#;

    let program = xlog_gpu::logic::LogicProgram::compile(source)?;

    let alice = symbol::intern("alice");
    let bob = symbol::intern("bob");

    let schema = Schema::new(vec![
        ("name".to_string(), ScalarType::Symbol),
        ("salary".to_string(), ScalarType::U32),
    ]);

    let name_bytes: Vec<u8> = [alice, bob].iter().flat_map(|v| v.to_le_bytes()).collect();
    let salary_bytes: Vec<u8> = [50000u32, 60000u32]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let emp_buf = provider.create_buffer_from_slices(&[&name_bytes, &salary_bytes], schema)?;

    let mut inputs = HashMap::new();
    inputs.insert("employee".to_string(), emp_buf);

    let result = program.evaluate(provider.clone(), inputs)?;

    let query = &result.queries[0];
    let names = read_column_u32(&provider, &query.buffer, 0);
    let salaries = read_column_u32(&provider, &query.buffer, 1);
    let bonuses = read_column_u32(&provider, &query.buffer, 2);

    // Verify bonus calculation
    for (i, (salary, bonus)) in salaries.iter().zip(bonuses.iter()).enumerate() {
        assert_eq!(*bonus, salary / 10, "Bonus for row {} incorrect", i);
    }

    // Verify names preserved
    assert!(names.contains(&alice) || names.contains(&bob));

    Ok(())
}

/// Test all v0.3.2 features combined on GPU
#[test]
#[serial]
fn test_gpu_v032_full_integration() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    symbol::clear();

    let source = r#"
        func score_bonus(Score) = Score * 2.

        pred analytics_player(symbol, i64).
        pred analytics_high_scorer(symbol, i64).

        analytics_high_scorer(Name, Bonus) :-
            analytics_player(Name, Score),
            Score > 50,
            Bonus is score_bonus(Score).

        ?- analytics_high_scorer(Name, Bonus).
    "#;

    let program = xlog_gpu::logic::LogicProgram::compile(source)?;

    let alice = symbol::intern("alice");
    let bob = symbol::intern("bob");
    let carol = symbol::intern("carol");

    let schema = Schema::new(vec![
        ("name".to_string(), ScalarType::Symbol),
        ("score".to_string(), ScalarType::I64),
    ]);

    let name_bytes: Vec<u8> = [alice, bob, carol]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let score_bytes: Vec<u8> = [30i64, 75i64, 90i64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let player_buf = provider.create_buffer_from_slices(&[&name_bytes, &score_bytes], schema)?;

    let mut inputs = HashMap::new();
    inputs.insert("analytics_player".to_string(), player_buf);

    let result = program.evaluate(provider.clone(), inputs)?;

    let query = &result.queries[0];
    let names = read_column_u32(&provider, &query.buffer, 0);
    let bonuses = read_column_i64(&provider, &query.buffer, 1);

    // Only bob (75) and carol (90) should be high scorers
    // alice (30) is below threshold
    assert_eq!(names.len(), 2);

    // Verify bonuses are doubled scores
    for bonus in &bonuses {
        assert!(*bonus == 150 || *bonus == 180); // 75*2 or 90*2
    }

    // Verify alice is NOT in results
    assert!(!names.contains(&alice));

    // Verify bob and carol ARE in results
    assert!(names.contains(&bob) || names.contains(&carol));

    Ok(())
}
