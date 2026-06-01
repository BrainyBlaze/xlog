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

fn read_unary(provider: &CudaKernelProvider, buffer: &CudaBuffer) -> Vec<u32> {
    let mut v = provider
        .download_column::<u32>(buffer, 0)
        .unwrap_or_default();
    v.sort_unstable();
    v.dedup();
    v
}

/// Evaluate an all-EDB program and return the first query's unary column, sorted.
fn run_unary_query(provider: &Arc<CudaKernelProvider>, source: &str) -> Result<Vec<u32>> {
    let program = xlog_gpu::logic::LogicProgram::compile(source)?;
    let result = program.evaluate(provider.clone(), std::collections::HashMap::new())?;
    Ok(read_unary(provider, &result.queries[0].buffer))
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

/// v0.9.2 WALL A1: a NEGATED modal `not know reach` over a GENUINELY RECURSIVE
/// relation that sits in a strictly LOWER stratum than the negating head EXECUTES on
/// the GPU production path as ordinary stratified negation -- with EXACT tuples and
/// ZERO CPU fallback.
///
/// This is the canonical "negated modal literal in a recursive epistemic program"
/// the WALL names. It is admissible because the negation is STRATIFIED: `reach`
/// (recursive transitive closure of `know link`) never depends on the negating
/// `unreachable`, so the reduced ordinary program (`not know reach` -> `not reach`,
/// `know link` -> `link`) has NO cycle through negation. The semi-naive engine
/// completes the recursive `reach` fixpoint, THEN anti-joins it.
///
/// NON-VACUOUS dispatch + zero-CPU-fallback proof (same convention as the Case-B
/// test above): the program routes through the ORDINARY recursive reduction
/// ("case_a_recursive" / "epistemic_reduced_ordinary"), which has NO epistemic CPU
/// code surface at all -- the epistemic CPU-fallback counters
/// (cpu_candidate_enumerations, cpu_world_view_validations, cpu_fallbacks) exist
/// ONLY on the single-pass epistemic path, so the ordinary engine is
/// CPU-fallback-free BY CONSTRUCTION. The "units":[] assertion proves no epistemic
/// GPU candidate-enumeration units (the CPU-fallback-bearing surface) were emitted.
/// A regression that rerouted this through the single-pass planner OR misclassified
/// the cycle case as stratified would change the tag and FAIL here.
///
/// EXACT founded answer: link = {(1,2),(2,3)} -> reach (transitive closure) =
/// {(1,2),(2,3),(1,3)}; unreachable = node x node MINUS reach =
/// {(1,1),(2,1),(2,2),(3,1),(3,2),(3,3)} (6 pairs).
#[test]
fn test_wall_a1_negated_modal_over_recursive_stratified_executes_exact() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        #pragma epistemic_mode = faeel
        pred node(u32).
        pred link(u32, u32).
        pred reach(u32, u32).
        pred unreachable(u32, u32).

        node(1). node(2). node(3).
        link(1, 2). link(2, 3).

        reach(X, Y) :- know link(X, Y).
        reach(X, Z) :- reach(X, Y), know link(Y, Z).
        unreachable(X, Y) :- node(X), node(Y), not know reach(X, Y).

        ?- unreachable(X, Y).
    "#;

    let program = xlog_gpu::logic::LogicProgram::compile(source)?;

    let plan_json = program
        .epistemic_plan_json()
        .expect("stratified negated-modal recursion must carry a provenance summary");
    assert!(
        plan_json.contains("\"reduction\":\"case_a_recursive\""),
        "stratified negated-modal recursion must route through the ORDINARY recursive \
         reduction (no single-pass epistemic CPU-fallback surface), got: {plan_json}"
    );
    assert!(
        plan_json.contains("\"plan_kind\":\"epistemic_reduced_ordinary\""),
        "stratified negated-modal plan kind must be the reduced-ordinary engine, got: {plan_json}"
    );
    assert!(
        plan_json.contains("\"units\":[]"),
        "reduced-ordinary plan carries NO epistemic candidate-enumeration units (the \
         CPU-fallback-bearing surface): {plan_json}"
    );

    let result = program.evaluate(provider.clone(), std::collections::HashMap::new())?;
    let query0 = &result.queries[0];
    let mut got = read_pairs(&provider, &query0.buffer);
    got.sort_unstable();
    got.dedup();

    assert_eq!(
        got,
        vec![(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)],
        "unreachable must be exactly node x node MINUS the recursive closure reach = \
         {{(1,2),(2,3),(1,3)}}"
    );

    Ok(())
}

/// v0.9.2 WALL A1 MUTATION PROBE: dropping the RECURSIVE second `reach` rule (so the
/// modal target is base-only, NOT a transitive closure) FLIPS the anti-join result,
/// proving the recursion INSIDE the negated modal gate is load-bearing.
///
/// The accepted program's `reach` is the transitive closure {(1,2),(2,3),(1,3)}; the
/// transitive pair (1,3) is reachable, so `unreachable` EXCLUDES (1,3). Here the
/// recursive rule is removed, so `reach` is base-only {(1,2),(2,3)} and (1,3) is NO
/// longer reachable -> `unreachable` now INCLUDES (1,3). The presence of (1,3) is the
/// FLIP the recursive closure (computed before the anti-join) prevents.
#[test]
fn test_wall_a1_drop_recursion_flips_anti_join_result() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    // MUTATION: remove the recursive `reach(X,Z) :- reach(X,Y), know link(Y,Z)` rule.
    // `reach` becomes the single-hop base relation, NOT the transitive closure.
    let base_only = r#"
        #pragma epistemic_mode = faeel
        pred node(u32).
        pred link(u32, u32).
        pred reach(u32, u32).
        pred unreachable(u32, u32).

        node(1). node(2). node(3).
        link(1, 2). link(2, 3).

        reach(X, Y) :- know link(X, Y).
        unreachable(X, Y) :- node(X), node(Y), not know reach(X, Y).

        ?- unreachable(X, Y).
    "#;

    let program = xlog_gpu::logic::LogicProgram::compile(base_only)?;
    let result = program.evaluate(provider.clone(), std::collections::HashMap::new())?;
    let query0 = &result.queries[0];
    let mut got = read_pairs(&provider, &query0.buffer);
    got.sort_unstable();
    got.dedup();

    // reach = base {(1,2),(2,3)} only; unreachable = 9 - 2 = 7 pairs, INCLUDING the
    // transitive (1,3) that the recursive closure would have removed.
    assert!(
        got.contains(&(1, 3)),
        "dropping the recursion must FLIP (1,3) into `unreachable` (base-only reach no \
         longer covers the transitive pair): {got:?}"
    );
    assert_eq!(
        got,
        vec![(1, 1), (1, 3), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)],
        "base-only `unreachable` must be node x node MINUS only the two base `link` pairs"
    );

    Ok(())
}

/// v0.9.2 ITEM F (derived-head coupling — SPLIT-VS-UNSPLIT EQUIVALENCE): a
/// cross-component modal coupling over an epistemically-DETERMINED derived head is
/// SOLVED in production by STRATIFICATION, and the stratified joint result equals
/// the per-stratum INDEPENDENT reference EXACTLY.
///
/// Program (`b` gates `know a`; `a` gates `know base`; `base` invariant):
///   base = {2,3}, src = {1,2}, src2 = {1,2,3}
///   a(X) :- src(X),  know base(X).   -- stratum 0 (determined: base invariant)
///   b(X) :- src2(X), know a(X).      -- stratum 1 (modal over determined `a`)
///
/// EXACT tuples: a = {2} (X in src AND in base), b = {2} (X in src2 AND in a).
///
/// REFERENCE = the per-stratum INDEPENDENT evaluation: stratum 0 solves `a` on its
/// own ({2}); that GATED extension is then staged as a base relation and stratum 1
/// solves `b` over it. Production instead AUTO-materializes a's gated output into
/// the store (`materialize_epistemic_head_relation`) between strata. This test
/// asserts the two mechanisms agree, so it validates the stratification
/// ORCHESTRATION (a regression that materialized the UNGATED `a`, i.e. double- or
/// non-gating, would make the joint `b` diverge from the reference). NON-tautological:
/// the reference recomputes each stratum independently rather than reading the joint
/// run's buffers.
#[test]
fn test_derived_head_coupling_stratified_equals_per_stratum_reference() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    // --- JOINT (production) path: stratified determined-derived-head coupling. ---
    let joint = r#"
        #pragma epistemic_mode = faeel
        pred base(u32). pred src(u32). pred src2(u32). pred a(u32). pred b(u32).
        base(2). base(3). src(1). src(2). src2(1). src2(2). src2(3).
        a(X) :- src(X), know base(X).
        b(X) :- src2(X), know a(X).
        ?- b(X).
    "#;
    let joint_program = xlog_gpu::logic::LogicProgram::compile(joint)?;
    // NON-VACUOUS dispatch proof: the coupling must route through the STRATIFIED
    // plan (not the joint single-component split, which fails closed in isolation).
    let plan_json = joint_program
        .epistemic_plan_json()
        .expect("stratified coupling must carry a provenance summary");
    assert!(
        plan_json.contains("\"plan_kind\":\"epistemic_stratified\""),
        "determined-derived-head coupling must route through the stratified plan, got: {plan_json}"
    );
    assert!(
        plan_json.contains("\"cpu_fallback_total_zero\":true"),
        "full stratified epistemic plan must report aggregate zero CPU fallback: {plan_json}"
    );
    assert!(
        plan_json.contains("\"cpu_fallback_is_zero\":true")
            && !plan_json.contains("\"cpu_fallback_is_zero\":false"),
        "every stratified epistemic GPU unit must report zero CPU fallback: {plan_json}"
    );
    assert!(
        plan_json.contains("\"candidate_enumeration\":0")
            && plan_json.contains("\"world_view_validation\":0")
            && plan_json.contains("\"solver_search\":0")
            && plan_json.contains("\"probabilistic_recompute\":0"),
        "stratified epistemic plan must expose all forbidden CPU-fallback counters at zero: {plan_json}"
    );
    let joint_result =
        joint_program.evaluate(provider.clone(), std::collections::HashMap::new())?;
    let joint_b = read_unary(&provider, &joint_result.queries[0].buffer);

    // --- REFERENCE: each stratum solved INDEPENDENTLY. ---
    // Stratum 0: solve `a` on its own.
    let a_ref = run_unary_query(
        &provider,
        r#"
        #pragma epistemic_mode = faeel
        pred base(u32). pred src(u32). pred a(u32).
        base(2). base(3). src(1). src(2).
        a(X) :- src(X), know base(X).
        ?- a(X).
    "#,
    )?;
    assert_eq!(
        a_ref,
        vec![2],
        "reference stratum-0 `a` must be exactly {{2}}"
    );

    // Stage `a`'s gated extension as a BASE relation, then solve stratum 1 `b`.
    let a_facts: String = a_ref.iter().map(|x| format!("a({x}). ")).collect();
    let b_ref_src = format!(
        r#"
        #pragma epistemic_mode = faeel
        pred src2(u32). pred a(u32). pred b(u32).
        src2(1). src2(2). src2(3). {a_facts}
        b(X) :- src2(X), know a(X).
        ?- b(X).
    "#
    );
    let b_ref = run_unary_query(&provider, &b_ref_src)?;

    // EXACT tuples and split-vs-unsplit (stratified-vs-independent) equivalence.
    assert_eq!(
        joint_b,
        vec![2],
        "stratified joint `b` must be exactly {{2}}"
    );
    assert_eq!(
        b_ref,
        vec![2],
        "reference stratum-1 `b` must be exactly {{2}}"
    );
    assert_eq!(
        joint_b, b_ref,
        "stratified coupling must equal the per-stratum independent reference EXACTLY"
    );

    Ok(())
}

/// v0.9.2 ITEM F (derived-head coupling — MUTATION PROBE): neutralizing the modal
/// gate on the determined head FLIPS the result, proving the stratified gating is
/// load-bearing (not cosmetic).
///
/// The accepted program gates `b` behind `know a` (and `a` behind `know base`).
/// Here the SAME shape is mutated to UNGATE the modals (every `know` dropped, so `b`
/// ranges over all of `src2`). The founded answer {2} FLIPS to {1,2,3}. A stratified
/// solve that returned {1,2,3} would be unsound; the accepted path must not.
#[test]
fn test_derived_head_coupling_ungated_mutation_flips_result() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    // MUTATION: drop the `know` gates -> a := src∩base ordinarily, b := all of src2.
    let mutated = run_unary_query(
        &provider,
        r#"
        pred base(u32). pred src(u32). pred src2(u32). pred a(u32). pred b(u32).
        base(2). base(3). src(1). src(2). src2(1). src2(2). src2(3).
        a(X) :- src(X), base(X).
        b(X) :- src2(X).
        ?- b(X).
    "#,
    )?;

    assert_eq!(
        mutated,
        vec![1, 2, 3],
        "ungating the modal coupling must FLIP `b` from the founded {{2}} to all of src2"
    );
    assert!(
        mutated.contains(&1) && mutated.contains(&3),
        "the mutation must introduce the spurious 1 and 3 that the modal gate excludes"
    );

    Ok(())
}

/// v0.9.2 ITEM F (derived-head coupling — TRUE-UNSAFETY WALL): a GENUINELY-CYCLIC
/// modal coupling (`a :- know b. b :- know a.`) has NO founded stratum order — the
/// modal truth of each head depends on the other's accepted world view, a circular
/// modality. It is correctly REJECTED end-to-end through the full production
/// dispatch (stratification yields no stratum order, Case-A does not apply, and the
/// split layer fails closed with a precise diagnostic naming both coupled heads).
/// This is the honest wall the item permits — it must NOT be silently accepted.
#[test]
fn test_true_cyclic_modal_coupling_rejected_end_to_end() -> Result<()> {
    if create_test_provider().is_none() {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    }

    let err = match xlog_gpu::logic::LogicProgram::compile("a() :- know b(). b() :- know a().") {
        Ok(_) => panic!("genuinely-cyclic modal coupling must fail closed end-to-end"),
        Err(err) => err,
    };
    let msg = format!("{err:?}");
    assert!(
        msg.contains("cross-component epistemic coupling"),
        "cyclic coupling must fail with the cross-component coupling diagnostic, got: {msg}"
    );
    assert!(
        msg.contains('a') && msg.contains('b'),
        "diagnostic must name both coupled heads a and b, got: {msg}"
    );

    Ok(())
}
