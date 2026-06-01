#![allow(clippy::arc_with_non_send_sync)]
use std::sync::Arc;

use xlog_core::{MemoryBudget, Result, ScalarType, Schema};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn create_test_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    Some(Arc::new(CudaKernelProvider::new(device, memory).ok()?))
}

fn create_edge_buffer(provider: &CudaKernelProvider, edges: &[(u32, u32)]) -> Result<CudaBuffer> {
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ]);

    if edges.is_empty() {
        return provider.create_empty_buffer(schema);
    }

    let col0: Vec<u8> = edges.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
    let col1: Vec<u8> = edges.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
    provider.create_buffer_from_slices(&[&col0, &col1], schema)
}

fn read_pairs(provider: &CudaKernelProvider, buffer: &CudaBuffer) -> Vec<(u32, u32)> {
    let c0 = provider
        .download_column::<u32>(buffer, 0)
        .unwrap_or_default();
    let c1 = provider
        .download_column::<u32>(buffer, 1)
        .unwrap_or_default();
    c0.into_iter().zip(c1).collect()
}

#[test]
fn test_logic_program_runs_with_gpu_inputs() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        pred edge(u32, u32).
        pred reach(u32, u32).

        reach(X, Y) :- edge(X, Y).
        reach(X, Z) :- reach(X, Y), edge(Y, Z).

        ?- reach(X, Y).
    "#;

    let program = xlog_gpu::logic::LogicProgram::compile(source)?;

    let edge_buf = create_edge_buffer(&provider, &[(1, 2), (2, 3)])?;
    let mut inputs = std::collections::HashMap::new();
    inputs.insert("edge".to_string(), edge_buf);

    let result = program.evaluate(provider.clone(), inputs)?;
    let query0 = &result.queries[0];
    let pairs = read_pairs(&provider, &query0.buffer);

    let mut got = pairs;
    got.sort_unstable();
    got.dedup();

    let mut expected = vec![(1, 2), (1, 3), (2, 3)];
    expected.sort_unstable();

    assert_eq!(got, expected);
    Ok(())
}

/// v0.9.2 ITEM A: a Case-B recursive epistemic program (positive `know` over a
/// relation that CO-EVOLVES with the recursion) EXECUTES to its FAEEL founded least
/// fixpoint on the production runtime path, NOT a rejection.
///
/// The modal feeds a NON-MIRROR relation `trust`, so the modal gate is load-bearing:
///   reach(X,Y) :- seed(X,Y).               -- founding base
///   reach(X,Z) :- reach(X,Y), trust(Y,Z).  -- ordinary recursion
///   trust(2,3) :- know reach(1,2).         -- founded (know reach(1,2) holds)
///   trust(3,1) :- know reach(3,3).         -- UNFOUNDED (reach(3,3) never derived)
///
/// EXACT founded model: reach = {(1,2),(1,3)}.
///   (1,2): seed-founded.
///   (1,3): reach(1,2)+trust(2,3); trust(2,3) founded via know reach(1,2).
/// The unfounded candidate trust(3,1) is excluded, so (1,1) (which an UNGATED reading
/// would add via trust(3,1)) is absent -- foundedness is load-bearing. This founded
/// result differs from BOTH base-only {(1,2)} and ungated {(1,1),(1,2),(1,3)}.
#[test]
fn test_case_b_recursive_epistemic_fixpoint_founded_tuples() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        #pragma epistemic_mode = faeel
        pred node(u32).
        pred seed(u32, u32).
        pred trust(u32, u32).
        pred reach(u32, u32).

        node(1). node(2). node(3).
        seed(1, 2).

        reach(X, Y) :- seed(X, Y).
        reach(X, Z) :- reach(X, Y), trust(Y, Z).
        trust(2, 3) :- know reach(1, 2).
        trust(3, 1) :- know reach(3, 3).

        ?- reach(X, Y).
    "#;

    let program = xlog_gpu::logic::LogicProgram::compile(source)?;

    // NON-VACUOUS dispatch proof: the Case-B program must compile to the ORDINARY
    // recursive plan (modal resolved into the SCC), NOT the single-pass epistemic plan.
    // This matters because the epistemic CPU-fallback counters (cpu_candidate_enumerations,
    // cpu_world_view_validations, cpu_fallbacks) exist ONLY on the single-pass epistemic
    // execution path; the ordinary semi-naive engine has no epistemic CPU code to count,
    // so it is CPU-fallback-free BY CONSTRUCTION. The provenance reduction tag
    // ("case_a_recursive") is the discriminating, non-vacuous proof of WHICH path ran:
    // a regression that rerouted Case-B through the single-pass planner (and could then
    // incur epistemic CPU fallbacks) would change this tag and FAIL here.
    let plan_json = program
        .epistemic_plan_json()
        .expect("Case-B epistemic program must carry a provenance summary");
    assert!(
        plan_json.contains("\"reduction\":\"case_a_recursive\""),
        "Case-B must route through the ORDINARY recursive reduction (no single-pass \
         epistemic CPU-fallback surface), got: {plan_json}"
    );
    assert!(
        plan_json.contains("\"plan_kind\":\"epistemic_reduced_ordinary\""),
        "Case-B plan kind must be the reduced-ordinary engine, got: {plan_json}"
    );
    // The reduced-ordinary plan carries NO epistemic GPU candidate-enumeration units
    // (those would be the CPU-fallback-bearing surface); the units list is empty.
    assert!(
        plan_json.contains("\"units\":[]"),
        "reduced-ordinary Case-B plan carries no epistemic candidate-enumeration units: \
         {plan_json}"
    );

    // All relations are in-program EDB facts; no GPU input buffers needed.
    let result = program.evaluate(provider.clone(), std::collections::HashMap::new())?;
    let query0 = &result.queries[0];
    let mut got = read_pairs(&provider, &query0.buffer);
    got.sort_unstable();
    got.dedup();

    // EXACT founded least fixpoint.
    assert_eq!(
        got,
        vec![(1, 2), (1, 3)],
        "founded reach must be exactly {{(1,2),(1,3)}} -- co-evolution adds (1,3), \
         foundedness excludes the unfounded trust(3,1) so (1,1) is absent"
    );

    Ok(())
}

/// v0.9.2 ITEM A MUTATION PROBE: neutralizing the founded modal gate FLIPS the result,
/// proving the co-evolution + foundedness are load-bearing (not cosmetic).
///
/// The accepted program above gates `trust` behind `know reach(...)`; foundedness then
/// excludes the unfounded `trust(3,1)`. Here the SAME program is mutated to UNGATE the
/// modal (both `trust` facts asserted unconditionally, as if every `know reach` were
/// true). The founded answer {(1,2),(1,3)} FLIPS to {(1,1),(1,2),(1,3)} -- the spurious
/// (1,1) appears precisely because the unfounded trust(3,1) is no longer excluded. A
/// founded fixpoint that returned this set would be a soundness violation; the accepted
/// path must NOT.
#[test]
fn test_case_b_ungated_mutation_flips_founded_result() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    // MUTATION: drop the `know reach(...)` gates -> trust asserted unconditionally.
    let ungated = r#"
        pred node(u32).
        pred seed(u32, u32).
        pred trust(u32, u32).
        pred reach(u32, u32).

        node(1). node(2). node(3).
        seed(1, 2).
        trust(2, 3).
        trust(3, 1).

        reach(X, Y) :- seed(X, Y).
        reach(X, Z) :- reach(X, Y), trust(Y, Z).

        ?- reach(X, Y).
    "#;

    let program = xlog_gpu::logic::LogicProgram::compile(ungated)?;
    let result = program.evaluate(provider.clone(), std::collections::HashMap::new())?;
    let query0 = &result.queries[0];
    let mut got = read_pairs(&provider, &query0.buffer);
    got.sort_unstable();
    got.dedup();

    // The ungated result is STRICTLY LARGER than the founded {(1,2),(1,3)}: the
    // spurious (1,1) appears. This is the FLIP the gate prevents.
    assert_eq!(
        got,
        vec![(1, 1), (1, 2), (1, 3)],
        "ungating the modal must FLIP the result to include the spurious (1,1)"
    );
    assert!(
        got.contains(&(1, 1)),
        "the mutation must introduce the unfounded (1,1) that the founded gate excludes"
    );

    Ok(())
}
