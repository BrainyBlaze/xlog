use std::{fs, sync::Arc};

use xlog_core::{MemoryBudget, Result, ScalarType, Schema};
use xlog_cuda::{CudaBuffer, CudaKernelProvider};

fn create_test_provider() -> Option<Arc<CudaKernelProvider>> {
    Some(Arc::new(
        xlog_cuda::CudaProviderBuilder::new(0, MemoryBudget::with_limit(1024 * 1024 * 1024))
            .build()
            .ok()?,
    ))
}

#[test]
fn normalize_program_for_execution_prepares_authored_identity_before_transforms() -> Result<()> {
    let program = xlog_logic::parse_program(
        r#"
        :- p(0).
        :- p(1).
        "#,
    )?;

    let normalized = xlog_gpu::logic::normalize_program_for_execution(program)?;
    assert_eq!(normalized.authored_constraint_source_bound, Some(2));
    assert_eq!(normalized.constraints[0].authored_index, Some(0));
    assert_eq!(normalized.constraints[1].authored_index, Some(1));
    Ok(())
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

fn assert_gpu_reject_unsupported_policy(plan_json: &str) {
    assert!(
        plan_json.contains("\"execution_backend\":\"gpu\""),
        "epistemic plan must declare the GPU execution backend: {plan_json}"
    );
    assert!(
        plan_json.contains("\"fallback_policy\":\"reject_unsupported\""),
        "epistemic plan must reject unsupported execution shapes: {plan_json}"
    );
    for removed_key in [
        "cpu_fallbacks",
        "cpu_fallback_is_zero",
        "cpu_fallback_total_zero",
    ] {
        assert!(
            !plan_json.contains(&format!("\"{removed_key}\"")),
            "epistemic plan must not present `{removed_key}` as measured evidence: {plan_json}"
        );
    }
}

#[test]
fn test_single_pass_plan_summary_declares_gpu_reject_unsupported_policy() -> Result<()> {
    let program = xlog_gpu::logic::LogicProgram::compile(
        r#"
            pred seed(u32). pred gate(u32). pred out(u32).
            seed(7). gate(7).
            out(X) :- seed(X), know gate(X).
            ?- out(X).
        "#,
    )?;
    let plan_json = program
        .epistemic_plan_json()
        .expect("single-pass epistemic plan summary");

    assert_gpu_reject_unsupported_policy(&plan_json);
    Ok(())
}

/// Evaluate an all-EDB program and return the first query's unary column, sorted.
fn run_unary_query(provider: &Arc<CudaKernelProvider>, source: &str) -> Result<Vec<u32>> {
    let program = xlog_gpu::logic::LogicProgram::compile(source)?;
    let result = program.evaluate(provider.clone(), std::collections::HashMap::new())?;
    Ok(read_unary(provider, &result.queries[0].buffer))
}

#[test]
fn test_xlog_gpu_manifest_does_not_depend_on_xlog_prob_host_wfs() -> Result<()> {
    let manifest = fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
        .map_err(|err| xlog_core::XlogError::Execution(err.to_string()))?;
    assert!(
        !manifest.contains("xlog-prob"),
        "xlog-gpu must not depend on xlog-prob; accepted GPU WFS execution must not link the host HashMap/HashSet WFS path"
    );
    Ok(())
}

#[test]
fn test_xlog_gpu_logic_source_does_not_reintroduce_host_wfs_solver() -> Result<()> {
    let source = fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/logic.rs"))
        .map_err(|err| xlog_core::XlogError::Execution(err.to_string()))?;

    for forbidden in [
        "use xlog_prob",
        "xlog_prob::",
        "evaluate_wfs_rules",
        "evaluate_wfs_program",
        "ground_wfs_program",
        "LogicExecutionPlan::EpistemicWfs(",
        "WfsRule",
        "WfsLiteral",
        "WfsConfig",
        "PirGraph",
    ] {
        assert!(
            !source.contains(forbidden),
            "xlog-gpu accepted WFS execution must stay GPU-native; forbidden host-WFS token `{forbidden}` appeared in src/logic.rs"
        );
    }

    Ok(())
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

/// Co-evolving recursive epistemic fixpoint: a positive `know` over a relation that
/// evolves with the recursion executes to its FAEEL founded least fixpoint on the
/// production runtime path, not a rejection.
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
fn test_coevolving_recursive_epistemic_fixpoint_founded_tuples() -> Result<()> {
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

    // NON-VACUOUS dispatch proof: the co-evolving recursive program must compile to
    // the ORDINARY recursive plan (modal resolved into the SCC), NOT the single-pass
    // epistemic plan.
    // The structural policy is explicit even though this family has no single-pass
    // candidate units. The reduction tag remains the non-vacuous proof of which path ran.
    let plan_json = program
        .epistemic_plan_json()
        .expect("co-evolving recursive epistemic program must carry a provenance summary");
    assert_gpu_reject_unsupported_policy(&plan_json);
    assert!(
        plan_json.contains("\"reduction\":\"ordinary_recursive_modal_reduction\""),
        "co-evolving recursive fixpoint must route through the ORDINARY recursive reduction under the typed \
         GPU/reject-unsupported policy, got: {plan_json}"
    );
    assert!(
        plan_json.contains("\"plan_kind\":\"epistemic_reduced_ordinary\""),
        "co-evolving recursive fixpoint plan kind must be the reduced-ordinary engine, got: {plan_json}"
    );
    // The reduced-ordinary plan carries NO epistemic GPU candidate-enumeration units
    // because this family executes the founded ordinary reduction; the units list is empty.
    assert!(
        plan_json.contains("\"units\":[]"),
        "reduced-ordinary recursive modal plan carries no epistemic candidate-enumeration units: \
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

/// Ungated mutation probe: neutralizing the founded modal gate flips the result,
/// proving the co-evolution and foundedness are load-bearing, not cosmetic.
///
/// The accepted program above gates `trust` behind `know reach(...)`; foundedness then
/// excludes the unfounded `trust(3,1)`. Here the SAME program is mutated to UNGATE the
/// modal (both `trust` facts asserted unconditionally, as if every `know reach` were
/// true). The founded answer {(1,2),(1,3)} FLIPS to {(1,1),(1,2),(1,3)} -- the spurious
/// (1,1) appears precisely because the unfounded trust(3,1) is no longer excluded. A
/// founded fixpoint that returned this set would be a soundness violation; the accepted
/// path must NOT.
#[test]
fn test_coevolving_recursive_ungated_mutation_flips_founded_result() -> Result<()> {
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

/// Stratified negated modal over a recursive relation: `not know reach` sits above
/// the recursive `reach` stratum and executes on the GPU production path as ordinary
/// stratified negation, with exact tuples under the typed GPU/reject-unsupported policy.
///
/// This is the canonical "negated modal literal in a recursive epistemic program"
/// this fixture covers. It is admissible because the negation is STRATIFIED: `reach`
/// (recursive transitive closure of `know link`) never depends on the negating
/// `unreachable`, so the reduced ordinary program (`not know reach` -> `not reach`,
/// `know link` -> `link`) has NO cycle through negation. The semi-naive engine
/// completes the recursive `reach` fixpoint, THEN anti-joins it.
///
/// NON-VACUOUS dispatch proof: the program routes through the ORDINARY recursive reduction
/// (`epistemic_reduced_ordinary`) under the same explicit GPU/reject-unsupported policy.
/// The "units":[] assertion proves no single-pass epistemic candidate units were emitted.
/// A regression that rerouted this through the single-pass planner OR misclassified
/// the cycle case as stratified would change the tag and FAIL here.
///
/// EXACT founded answer: link = {(1,2),(2,3)} -> reach (transitive closure) =
/// {(1,2),(2,3),(1,3)}; unreachable = node x node MINUS reach =
/// {(1,1),(2,1),(2,2),(3,1),(3,2),(3,3)} (6 pairs).
#[test]
fn test_stratified_negated_modal_over_recursive_relation_executes_exact() -> Result<()> {
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
    assert_gpu_reject_unsupported_policy(&plan_json);
    assert!(
        plan_json.contains("\"reduction\":\"ordinary_recursive_modal_reduction\""),
        "stratified negated-modal recursion must route through the ORDINARY recursive \
         reduction under the typed GPU/reject-unsupported policy, got: {plan_json}"
    );
    assert!(
        plan_json.contains("\"plan_kind\":\"epistemic_reduced_ordinary\""),
        "stratified negated-modal plan kind must be the reduced-ordinary engine, got: {plan_json}"
    );
    assert!(
        plan_json.contains("\"units\":[]"),
        "reduced-ordinary plan carries no single-pass epistemic candidate-enumeration units: \
         {plan_json}"
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

/// Cyclic negated modal recursion: a negated modal over a co-evolving recursive target
/// whose reduced ordinary program has a cycle through negation must route to the
/// GPU-native WFS alternating-fixpoint plan, not host WFS or the ordinary stratified path.
///
/// Matrix coverage:
///   mode {FAEEL, G91}
///   x modal form {not know,not possible}
///   x seed state {present,absent}
///   x ordinary EDB negation {absent,present-in-SCC}.
///
/// EXACT WFS answer for seed-present cells: reach = {(1,2)}. Seed-absent cells
/// emit no true reach rows. Every other vertex x vertex reach tuple is false or
/// WFS-undefined and therefore absent from `xlog run` output.
fn wfs_cycle_source(
    mode: &str,
    modal: &str,
    seed_present: bool,
    include_edb_negation: bool,
) -> String {
    let seed_fact = if seed_present { "seed(1, 2)." } else { "" };
    let banned_decl = if include_edb_negation {
        "pred banned(u32)."
    } else {
        ""
    };
    let banned_fact = if include_edb_negation {
        "banned(3)."
    } else {
        ""
    };
    let edb_guard = if include_edb_negation {
        ", not banned(Y)"
    } else {
        ""
    };

    format!(
        r#"
        #pragma epistemic_mode = {mode}
        pred vertex(u32).
        pred seed(u32, u32).
        {banned_decl}
        pred linked(u32, u32).
        pred reach(u32, u32).

        vertex(1). vertex(2). vertex(3).
        {seed_fact}
        {banned_fact}

        reach(X, Y) :- linked(X, Y).
        reach(X, Z) :- reach(X, Y), linked(Y, Z).
        linked(X, Y) :- vertex(X), vertex(Y), {modal} reach(X, Y){edb_guard}.
        linked(X, Y) :- seed(X, Y).

        ?- reach(X, Y).
        "#
    )
}

#[test]
fn test_cyclic_negated_modal_wfs_plan_kind_matrix_compiles_without_cuda() -> Result<()> {
    for (mode, modal, seed_present, include_edb_negation) in [
        ("faeel", "not know", true, false),
        ("faeel", "not know", false, false),
        ("faeel", "not know", true, true),
        ("faeel", "not know", false, true),
        ("faeel", "not possible", true, false),
        ("faeel", "not possible", false, false),
        ("faeel", "not possible", true, true),
        ("faeel", "not possible", false, true),
        ("g91", "not know", true, false),
        ("g91", "not know", false, false),
        ("g91", "not know", true, true),
        ("g91", "not know", false, true),
        ("g91", "not possible", true, false),
        ("g91", "not possible", false, false),
        ("g91", "not possible", true, true),
        ("g91", "not possible", false, true),
    ] {
        let source = wfs_cycle_source(mode, modal, seed_present, include_edb_negation);
        let program = xlog_gpu::logic::LogicProgram::compile(source.as_str())?;
        let plan_json = program
            .epistemic_plan_json()
            .expect("cyclic negated-modal recursion must carry a WFS provenance summary");
        assert_gpu_reject_unsupported_policy(&plan_json);
        assert!(
            plan_json.contains("\"reduction\":\"wfs_gpu_recursive\""),
            "{mode}/{modal}: cyclic negated-modal recursion must use GPU WFS reduction, got: {plan_json}"
        );
        assert!(
            plan_json.contains("\"plan_kind\":\"epistemic_wfs_gpu\""),
            "{mode}/{modal}: plan kind must be GPU-backed WFS, got: {plan_json}"
        );
        assert!(
            plan_json.contains("\"units\":[]"),
            "{mode}/{modal}: GPU WFS reduction must not emit single-pass epistemic candidate units: {plan_json}"
        );
        assert!(
            plan_json.contains("\"reach\":{\"upper\":\"__wfs_upper_reach\",\"lower\":\"__wfs_lower_reach\"}"),
            "{mode}/{modal}: WFS plan JSON must expose deterministic fixed relation rewrites, got: {plan_json}"
        );
        if include_edb_negation {
            assert!(
                plan_json.contains(
                    "\"banned\":{\"upper\":\"__wfs_upper_banned\",\"lower\":\"__wfs_lower_banned\"}"
                ),
                "{mode}/{modal}: WFS plan JSON must expose ordinary EDB negation fixed relation rewrites, got: {plan_json}"
            );
        }
        assert!(
            plan_json.contains("\"wfs_convergence_predicates\":[\"linked\",\"reach\"]"),
            "{mode}/{modal}: WFS plan JSON must expose the convergence predicate set, got: {plan_json}"
        );
        assert!(
            plan_json.contains("\"wfs_gpu_passes\":[\"overapprox\",\"lower\",\"upper\"]"),
            "{mode}/{modal}: WFS plan JSON must expose the alternating GPU pass structure, got: {plan_json}"
        );
        assert!(
            !plan_json.contains("host_wfs_fallback_allowed"),
            "{mode}/{modal}: WFS plan JSON must omit synthetic host-fallback claims, got: {plan_json}"
        );
    }

    Ok(())
}

#[test]
fn undeclared_nullary_negative_modal_cycle_uses_inferred_wfs_schema() -> Result<()> {
    let source = r#"
        #pragma epistemic_mode = faeel

        p() :- not possible p().

        ?- p().
    "#;

    let program = xlog_gpu::logic::LogicProgram::compile(source)?;
    let plan_json = program
        .epistemic_plan_json()
        .expect("negative modal cycle must carry a WFS provenance summary");
    assert!(
        plan_json.contains("\"plan_kind\":\"epistemic_wfs_gpu\""),
        "undeclared nullary cycle must use GPU-backed WFS: {plan_json}"
    );

    let Some(provider) = create_test_provider() else {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed");
        }
        eprintln!("Skipping runtime assertion: no CUDA device");
        return Ok(());
    };
    let result = program.evaluate(provider, std::collections::HashMap::new())?;
    assert_eq!(result.queries.len(), 1);
    assert_eq!(result.queries[0].buffer.arity(), 0);
    assert_eq!(result.queries[0].buffer.num_rows(), 0);

    Ok(())
}

#[test]
fn wfs_constraints_use_three_valued_truth_after_convergence() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed");
        }
        eprintln!("Skipping runtime assertion: no CUDA device");
        return Ok(());
    };

    let undefined_negation = xlog_gpu::logic::LogicProgram::compile(
        r#"
            #pragma epistemic_mode = faeel
            pred left().
            pred right().
            left() :- not possible right().
            right() :- not possible left().
            :- not left().
            ?- left().
        "#,
    )?;
    undefined_negation.evaluate(provider.clone(), std::collections::HashMap::new())?;

    for (name, constraint) in [
        ("positive true atom", "fact()"),
        ("negated false atom", "not definitely_false()"),
    ] {
        let source = format!(
            r#"
                #pragma epistemic_mode = faeel
                pred left().
                pred right().
                pred fact().
                pred definitely_false().
                fact().
                left() :- not possible right().
                right() :- not possible left().
                :- {constraint}.
                ?- left().
            "#,
        );
        let program = xlog_gpu::logic::LogicProgram::compile(&source)?;
        let error = match program.evaluate(provider.clone(), std::collections::HashMap::new()) {
            Ok(_) => panic!("{name} must violate the WFS constraint"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("Constraint 0 violated"),
            "{error}"
        );
    }

    Ok(())
}

#[test]
fn iterative_epistemic_plans_return_collected_profiling_stats() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed");
        }
        eprintln!("Skipping runtime assertion: no CUDA device");
        return Ok(());
    };

    for (name, source, expected_rows) in [
        (
            "Gelfond-1991 compatibility",
            r#"
                #pragma epistemic_mode = g91
                pred base(u32).
                pred p(u32).
                base(1).
                p(X) :- base(X), possible p(X).
                ?- p(X).
            "#,
            1,
        ),
        (
            "well-founded negation",
            r#"
                #pragma epistemic_mode = faeel
                pred p().
                p() :- not possible p().
                ?- p().
            "#,
            0,
        ),
    ] {
        let program = xlog_gpu::logic::LogicProgram::compile(source)?;
        let result = program.evaluate_with_options(
            provider.clone(),
            std::collections::HashMap::new(),
            true,
        )?;
        let stats = result
            .stats
            .unwrap_or_else(|| panic!("{name} must return profiling statistics"));
        assert!(!stats.strata.is_empty(), "{name}: {stats:?}");
        assert_eq!(stats.total_output_rows, expected_rows, "{name}: {stats:?}");
    }

    Ok(())
}

#[test]
fn g91_constraint_errors_preserve_authored_predicate_names() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed");
        }
        eprintln!("Skipping runtime assertion: no CUDA device");
        return Ok(());
    };
    let program = xlog_gpu::logic::LogicProgram::compile(
        r#"
            #pragma epistemic_mode = g91
            p(a).
            p(a, b).
            cycle() :- possible cycle().
            :- p(a).
            ?- cycle().
        "#,
    )?;
    let error = match program.evaluate(provider, std::collections::HashMap::new()) {
        Ok(_) => panic!("the authored p(a) constraint must be violated"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(
        message.contains("Constraint 0 violated: :- p(a)."),
        "{message}"
    );
    assert!(!message.contains("p/1"), "{message}");
    Ok(())
}

#[test]
fn test_cyclic_negated_modal_wfs_plan_clamps_zero_iteration_bound_without_cuda() -> Result<()> {
    let source = r#"
        #pragma epistemic_mode = faeel
        #pragma max_recursion_depth = 0
        pred vertex(u32).
        pred seed(u32, u32).
        pred linked(u32, u32).
        pred reach(u32, u32).

        vertex(1). vertex(2). vertex(3).
        seed(1, 2).

        reach(X, Y) :- linked(X, Y).
        reach(X, Z) :- reach(X, Y), linked(Y, Z).
        linked(X, Y) :- vertex(X), vertex(Y), not know reach(X, Y).
        linked(X, Y) :- seed(X, Y).

        ?- reach(X, Y).
    "#;

    let program = xlog_gpu::logic::LogicProgram::compile(source)?;
    let plan_json = program
        .epistemic_plan_json()
        .expect("cyclic negated-modal recursion must carry a WFS provenance summary");
    assert!(
        plan_json.contains("\"plan_kind\":\"epistemic_wfs_gpu\""),
        "zero-depth WFS fixture must still route to GPU WFS, got: {plan_json}"
    );
    assert!(
        plan_json.contains("\"max_iterations\":1"),
        "zero max_recursion_depth must clamp to one WFS alternating pass, got: {plan_json}"
    );
    assert!(
        plan_json.contains("\"wfs_fixed_relations\":{\"reach\":"),
        "zero-depth WFS fixture must still expose its fixed relation map, got: {plan_json}"
    );
    assert!(
        plan_json.contains("\"wfs_convergence_predicates\":[\"linked\",\"reach\"]"),
        "zero-depth WFS fixture must expose the convergence predicate set, got: {plan_json}"
    );
    assert!(
        plan_json.contains("\"wfs_gpu_passes\":[\"overapprox\",\"lower\",\"upper\"]"),
        "zero-depth WFS fixture must expose the alternating GPU pass structure, got: {plan_json}"
    );
    assert!(
        !plan_json.contains("host_wfs_fallback_allowed"),
        "zero-depth WFS fixture must omit synthetic host-fallback claims, got: {plan_json}"
    );

    Ok(())
}

#[test]
fn test_cyclic_negated_modal_wfs_plan_exposes_multiple_fixed_relation_maps_without_cuda(
) -> Result<()> {
    let source = r#"
        #pragma epistemic_mode = faeel
        pred vertex(u32).
        pred seed(u32, u32).
        pred linked(u32, u32).
        pred blocked(u32, u32).
        pred reach(u32, u32).
        pred avoid(u32, u32).

        vertex(1). vertex(2). vertex(3).
        seed(1, 2).

        reach(X, Y) :- linked(X, Y).
        reach(X, Z) :- reach(X, Y), linked(Y, Z).
        avoid(X, Y) :- blocked(X, Y).

        linked(X, Y) :- vertex(X), vertex(Y), not know reach(X, Y), not possible avoid(X, Y).
        linked(X, Y) :- seed(X, Y).
        blocked(X, Y) :- vertex(X), vertex(Y), not know reach(Y, X).

        ?- reach(X, Y).
    "#;

    let program = xlog_gpu::logic::LogicProgram::compile(source)?;
    let plan_json = program
        .epistemic_plan_json()
        .expect("multi-negated cyclic WFS program must carry a WFS provenance summary");
    assert!(
        plan_json.contains("\"plan_kind\":\"epistemic_wfs_gpu\""),
        "multi-negated WFS fixture must route to GPU WFS, got: {plan_json}"
    );
    assert!(
        plan_json.contains(
            "\"avoid\":{\"upper\":\"__wfs_upper_avoid\",\"lower\":\"__wfs_lower_avoid\"}"
        ),
        "WFS plan JSON must expose private fixed relations for avoid/2, got: {plan_json}"
    );
    assert!(
        plan_json.contains(
            "\"reach\":{\"upper\":\"__wfs_upper_reach\",\"lower\":\"__wfs_lower_reach\"}"
        ),
        "WFS plan JSON must expose private fixed relations for reach/2, got: {plan_json}"
    );
    assert!(
        plan_json.contains(
            "\"wfs_convergence_predicates\":[\"avoid\",\"blocked\",\"linked\",\"reach\"]"
        ),
        "multi-negated WFS fixture must expose every intensional convergence predicate, got: {plan_json}"
    );
    assert!(
        plan_json.contains("\"wfs_gpu_passes\":[\"overapprox\",\"lower\",\"upper\"]"),
        "multi-negated WFS fixture must expose the alternating GPU pass structure, got: {plan_json}"
    );
    assert!(
        !plan_json.contains("host_wfs_fallback_allowed"),
        "multi-negated WFS fixture must omit synthetic host-fallback claims, got: {plan_json}"
    );

    Ok(())
}

#[test]
fn test_cyclic_negated_modal_wfs_plan_exposes_fixed_relation_for_ordinary_edb_negation_without_cuda(
) -> Result<()> {
    let source = r#"
        #pragma epistemic_mode = faeel
        pred vertex(u32).
        pred seed(u32, u32).
        pred banned(u32).
        pred linked(u32, u32).
        pred reach(u32, u32).

        vertex(1). vertex(2). vertex(3).
        seed(1, 2).
        banned(3).

        reach(X, Y) :- linked(X, Y).
        reach(X, Z) :- reach(X, Y), linked(Y, Z).
        linked(X, Y) :- vertex(X), vertex(Y), not know reach(X, Y), not banned(Y).
        linked(X, Y) :- seed(X, Y).

        ?- reach(X, Y).
    "#;

    let program = xlog_gpu::logic::LogicProgram::compile(source)?;
    let plan_json = program
        .epistemic_plan_json()
        .expect("mixed modal/ordinary-negation WFS program must carry a WFS provenance summary");
    assert!(
        plan_json.contains("\"plan_kind\":\"epistemic_wfs_gpu\""),
        "mixed modal/ordinary-negation WFS fixture must route to GPU WFS, got: {plan_json}"
    );
    assert!(
        plan_json.contains("\"banned\":{\"upper\":\"__wfs_upper_banned\",\"lower\":\"__wfs_lower_banned\"}"),
        "WFS plan JSON must expose private fixed relations for ordinary EDB negation banned/1, got: {plan_json}"
    );
    assert!(
        plan_json.contains("\"reach\":{\"upper\":\"__wfs_upper_reach\",\"lower\":\"__wfs_lower_reach\"}"),
        "WFS plan JSON must expose private fixed relations for recursive modal target reach/2, got: {plan_json}"
    );
    assert!(
        plan_json.contains("\"wfs_convergence_predicates\":[\"linked\",\"reach\"]"),
        "mixed modal/ordinary-negation WFS fixture must expose convergence predicates, got: {plan_json}"
    );
    assert!(
        plan_json.contains("\"wfs_gpu_passes\":[\"overapprox\",\"lower\",\"upper\"]"),
        "mixed modal/ordinary-negation WFS fixture must expose the alternating GPU pass structure, got: {plan_json}"
    );
    assert!(
        !plan_json.contains("host_wfs_fallback_allowed"),
        "mixed modal/ordinary-negation WFS fixture must omit synthetic host-fallback claims, got: {plan_json}"
    );

    Ok(())
}

#[test]
fn test_cyclic_negated_modal_cycle_routes_to_gpu_wfs_matrix() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    for (mode, modal, seed_present, include_edb_negation) in [
        ("faeel", "not know", true, false),
        ("faeel", "not know", false, false),
        ("faeel", "not know", true, true),
        ("faeel", "not know", false, true),
        ("faeel", "not possible", true, false),
        ("faeel", "not possible", false, false),
        ("faeel", "not possible", true, true),
        ("faeel", "not possible", false, true),
        ("g91", "not know", true, false),
        ("g91", "not know", false, false),
        ("g91", "not know", true, true),
        ("g91", "not know", false, true),
        ("g91", "not possible", true, false),
        ("g91", "not possible", false, false),
        ("g91", "not possible", true, true),
        ("g91", "not possible", false, true),
    ] {
        let source = wfs_cycle_source(mode, modal, seed_present, include_edb_negation);

        let program = xlog_gpu::logic::LogicProgram::compile(source.as_str())?;
        let plan_json = program
            .epistemic_plan_json()
            .expect("cyclic negated-modal recursion must carry a WFS provenance summary");
        assert!(
            plan_json.contains("\"reduction\":\"wfs_gpu_recursive\""),
            "{mode}/{modal}: cyclic negated-modal recursion must use GPU WFS reduction, got: {plan_json}"
        );
        assert!(
            plan_json.contains("\"plan_kind\":\"epistemic_wfs_gpu\""),
            "{mode}/{modal}: plan kind must be GPU-backed WFS, got: {plan_json}"
        );
        assert!(
            plan_json.contains("\"units\":[]"),
            "{mode}/{modal}: GPU WFS reduction must not emit single-pass epistemic candidate units: {plan_json}"
        );
        assert!(
            plan_json.contains("\"reach\":{\"upper\":\"__wfs_upper_reach\",\"lower\":\"__wfs_lower_reach\"}"),
            "{mode}/{modal}: runtime WFS plan JSON must expose deterministic fixed relation rewrites, got: {plan_json}"
        );
        if include_edb_negation {
            assert!(
                plan_json.contains(
                    "\"banned\":{\"upper\":\"__wfs_upper_banned\",\"lower\":\"__wfs_lower_banned\"}"
                ),
                "{mode}/{modal}: runtime WFS plan JSON must expose ordinary EDB negation fixed relation rewrites, got: {plan_json}"
            );
        }
        assert!(
            plan_json.contains("\"wfs_convergence_predicates\":[\"linked\",\"reach\"]"),
            "{mode}/{modal}: runtime WFS plan JSON must expose convergence predicates, got: {plan_json}"
        );
        assert!(
            plan_json.contains("\"wfs_gpu_passes\":[\"overapprox\",\"lower\",\"upper\"]"),
            "{mode}/{modal}: runtime WFS plan JSON must expose the alternating GPU pass structure, got: {plan_json}"
        );
        assert!(
            !plan_json.contains("host_wfs_fallback_allowed"),
            "{mode}/{modal}: runtime WFS plan JSON must omit synthetic host-fallback claims, got: {plan_json}"
        );

        let result = program.evaluate(provider.clone(), std::collections::HashMap::new())?;
        let query0 = &result.queries[0];
        let mut got = read_pairs(&provider, &query0.buffer);
        got.sort_unstable();
        got.dedup();

        let expected = if seed_present { vec![(1, 2)] } else { vec![] };
        assert_eq!(
            got, expected,
            "{mode}/{modal}: WFS true extension must match the seed-state matrix cell; \
             false/undefined reach tuples must be absent"
        );
    }

    Ok(())
}

/// WFS fixed-relation anti-collision guard: user predicates may legally have names near
/// the internal WFS fixed relations. Source-level predicates cannot start with `__`
/// under the language grammar, so this fixture uses legal near-collision names and
/// verifies the compiler keeps its private fixed names in the internal namespace instead
/// of reading user-owned relations.
///
/// The user-owned `wfs_upper_reach` / `wfs_lower_reach` predicates below derive
/// every vertex pair. If the WFS transform accidentally reused those names as its
/// fixed upper/lower approximation buffers, the negated modal gate would see the
/// wrong relation and the WFS result would change. The correct GPU WFS answer remains
/// exactly reach = {(1,2)}.
#[test]
fn test_cyclic_negated_modal_wfs_fixed_relation_names_avoid_user_collisions() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        #pragma epistemic_mode = faeel
        pred vertex(u32).
        pred seed(u32, u32).
        pred linked(u32, u32).
        pred reach(u32, u32).
        pred wfs_upper_reach(u32, u32).
        pred wfs_lower_reach(u32, u32).

        vertex(1). vertex(2). vertex(3).
        seed(1, 2).

        wfs_upper_reach(X, Y) :- vertex(X), vertex(Y).
        wfs_lower_reach(X, Y) :- vertex(X), vertex(Y).

        reach(X, Y) :- linked(X, Y).
        reach(X, Z) :- reach(X, Y), linked(Y, Z).
        linked(X, Y) :- vertex(X), vertex(Y), not know reach(X, Y).
        linked(X, Y) :- seed(X, Y).

        ?- reach(X, Y).
    "#;

    let program = xlog_gpu::logic::LogicProgram::compile(source)?;
    let plan_json = program
        .epistemic_plan_json()
        .expect("cyclic negated-modal recursion must carry a WFS provenance summary");
    assert!(
        plan_json.contains("\"reduction\":\"wfs_gpu_recursive\""),
        "collision fixture must still route to GPU WFS, got: {plan_json}"
    );
    assert!(
        plan_json.contains("\"plan_kind\":\"epistemic_wfs_gpu\""),
        "collision fixture must keep the GPU WFS plan kind, got: {plan_json}"
    );
    assert!(
        !plan_json.contains("\"upper\":\"wfs_upper_reach\"")
            && !plan_json.contains("\"lower\":\"wfs_lower_reach\""),
        "collision fixture must not reuse user-owned wfs_* predicates as private fixed relations: {plan_json}"
    );
    assert!(
        plan_json.contains("\"upper\":\"__wfs_upper_reach\"")
            && plan_json.contains("\"lower\":\"__wfs_lower_reach\""),
        "collision fixture must expose private internal WFS fixed names: {plan_json}"
    );
    assert!(
        !plan_json.contains("host_wfs_fallback_allowed"),
        "collision fixture must omit synthetic host-fallback claims, got: {plan_json}"
    );

    let result = program.evaluate(provider.clone(), std::collections::HashMap::new())?;
    let query0 = &result.queries[0];
    let mut got = read_pairs(&provider, &query0.buffer);
    got.sort_unstable();
    got.dedup();

    assert_eq!(
        got,
        vec![(1, 2)],
        "user-owned __wfs_* predicates must not pollute private WFS fixed relations"
    );

    Ok(())
}

/// Stratified negated-modal mutation probe: dropping the RECURSIVE second `reach` rule
/// (so the modal target is base-only, NOT a transitive closure) flips the anti-join
/// result, proving the recursion inside the negated modal gate is load-bearing.
///
/// The accepted program's `reach` is the transitive closure {(1,2),(2,3),(1,3)}; the
/// transitive pair (1,3) is reachable, so `unreachable` EXCLUDES (1,3). Here the
/// recursive rule is removed, so `reach` is base-only {(1,2),(2,3)} and (1,3) is NO
/// longer reachable -> `unreachable` now INCLUDES (1,3). The presence of (1,3) is the
/// FLIP the recursive closure (computed before the anti-join) prevents.
#[test]
fn test_stratified_negated_modal_drop_recursion_flips_anti_join_result() -> Result<()> {
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

/// Derived-head modal coupling, split-vs-unsplit equivalence: a cross-component modal
/// coupling over an epistemically-determined derived head is solved in production by
/// stratification, and the stratified joint result equals the per-stratum independent
/// reference exactly.
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
    assert_gpu_reject_unsupported_policy(&plan_json);
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

#[test]
fn stratified_ordinary_constraint_is_enforced() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed");
        }
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        #pragma epistemic_mode = faeel
        pred base(u32). pred src(u32). pred src2(u32). pred a(u32). pred b(u32).
        base(2). src(2). src2(2).
        a(X) :- src(X), know base(X).
        b(X) :- src2(X), know a(X).
        b(X) :- b(X), know a(X).
        :- b(X).
        ?- b(X).
    "#;
    let mut authored = xlog_logic::parse_program(source)?;
    authored.constraints[0].authored_index = Some(3);
    authored.prepare_authored_constraint_identity(4)?;
    let program = xlog_gpu::logic::LogicProgram::compile_program(authored)?;
    let plan_json = program.epistemic_plan_json().expect("epistemic plan");
    assert!(plan_json.contains("\"plan_kind\":\"epistemic_stratified\""));

    let error = match program.evaluate(provider, std::collections::HashMap::new()) {
        Ok(_) => panic!("the ordinary stratum constraint must reject b(2)"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("Constraint 3 violated: :- b(X)."),
        "{error}"
    );
    Ok(())
}

#[test]
fn stratified_cross_stratum_ordinary_constraint_false_body_succeeds() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed");
        }
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        #pragma epistemic_mode = faeel
        pred base(u32). pred src(u32). pred a(u32). pred b(u32).
        base(1). src(1). src(2).
        a(X) :- src(X), know base(X).
        b(X) :- src(X), know a(X).
        :- a(2), b(2).
        ?- b(X).
    "#;
    let mut authored = xlog_logic::parse_program(source)?;
    authored.constraints[0].authored_index = Some(3);
    authored.prepare_authored_constraint_identity(4)?;
    let program = xlog_gpu::logic::LogicProgram::compile_program(authored)?;
    assert!(program
        .epistemic_plan_json()
        .expect("epistemic plan")
        .contains("\"plan_kind\":\"epistemic_stratified\""));
    let result = program.evaluate(provider.clone(), std::collections::HashMap::new())?;
    assert_eq!(read_unary(&provider, &result.queries[0].buffer), vec![1]);
    Ok(())
}

#[test]
fn stratified_cross_stratum_ordinary_constraint_true_body_reports_authored_identity() -> Result<()>
{
    let Some(provider) = create_test_provider() else {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed");
        }
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        #pragma epistemic_mode = faeel
        pred base(u32). pred src(u32). pred a(u32). pred b(u32).
        base(1). src(1).
        a(X) :- src(X), know base(X).
        b(X) :- src(X), know a(X).
        :- a(X), b(X).
        ?- b(X).
    "#;
    let mut authored = xlog_logic::parse_program(source)?;
    authored.constraints[0].authored_index = Some(3);
    authored.prepare_authored_constraint_identity(4)?;
    let program = xlog_gpu::logic::LogicProgram::compile_program(authored)?;
    assert!(program
        .epistemic_plan_json()
        .expect("epistemic plan")
        .contains("\"plan_kind\":\"epistemic_stratified\""));

    let error = match program.evaluate(provider, std::collections::HashMap::new()) {
        Ok(_) => panic!("the true cross-stratum ordinary constraint must be rejected"),
        Err(error) => error,
    };
    match error {
        xlog_core::XlogError::Execution(message) => {
            assert_eq!(message, "Constraint 3 violated: :- a(X), b(X).")
        }
        other => panic!("expected typed execution error, got {other}"),
    }
    Ok(())
}

#[test]
fn stratified_post_ordinary_closure_false_constraint_succeeds() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed");
        }
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        #pragma epistemic_mode = faeel
        pred base(u32). pred src(u32). pred a(u32). pred b(u32). pred c(u32).
        base(1). src(1).
        a(X) :- src(X), know base(X).
        b(X) :- src(X), know a(X).
        c(X) :- b(X).
        :- c(2).
        ?- b(X).
    "#;
    let mut authored = xlog_logic::parse_program(source)?;
    authored.constraints[0].authored_index = Some(3);
    authored.prepare_authored_constraint_identity(4)?;
    let program = xlog_gpu::logic::LogicProgram::compile_program(authored)?;
    let result = program.evaluate(provider.clone(), std::collections::HashMap::new())?;
    assert_eq!(read_unary(&provider, &result.queries[0].buffer), vec![1]);
    Ok(())
}

#[test]
fn stratified_plan_json_exposes_ordinary_post_stage() -> Result<()> {
    let source = r#"
        #pragma epistemic_mode = faeel
        pred base(u32). pred src(u32). pred a(u32). pred b(u32). pred c(u32).
        base(1). src(1).
        a(X) :- src(X), know base(X).
        b(X) :- src(X), know a(X).
        c(X) :- b(X).
        ?- c(X).
    "#;
    let program = xlog_gpu::logic::LogicProgram::compile(source)?;
    let plan_json = program.epistemic_plan_json().expect("epistemic plan");
    assert!(
        plan_json.contains(
            "\"unit\":\"ordinary_post\",\"stage_kind\":\"ordinary_closure_and_constraints\""
        ),
        "the public plan must expose the ordinary post-stratification stage: {plan_json}"
    );
    Ok(())
}

#[test]
fn stratified_post_ordinary_query_surfaces_authored_relation() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed");
        }
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        #pragma epistemic_mode = faeel
        pred base(u32). pred src(u32). pred a(u32). pred b(u32). pred c(u32).
        base(1). src(1).
        a(X) :- src(X), know base(X).
        b(X) :- src(X), know a(X).
        c(X) :- b(X).
        ?- c(X).
    "#;
    let program = xlog_gpu::logic::LogicProgram::compile(source)?;

    let result =
        program.evaluate_with_options(provider.clone(), std::collections::HashMap::new(), true)?;
    assert_eq!(result.queries.len(), 1);
    assert_eq!(result.queries[0].relation_name, "c");
    assert_eq!(read_unary(&provider, &result.queries[0].buffer), vec![1]);
    let stats = result.stats.expect("profiled stratified evaluation stats");
    assert!(
        stats.strata.len() >= 3,
        "profiling must retain both modal strata plus the ordinary post stage: {stats:?}"
    );
    Ok(())
}

#[test]
fn stratified_post_mixed_queries_preserve_authored_order() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed");
        }
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        #pragma epistemic_mode = faeel
        pred base(u32). pred src(u32). pred a(u32). pred b(u32). pred c(u32).
        base(1). src(1).
        a(X) :- src(X), know base(X).
        b(X) :- src(X), know a(X).
        c(X) :- b(X).
        ?- c(X).
        ?- b(X).
    "#;
    let program = xlog_gpu::logic::LogicProgram::compile(source)?;
    let result = program.evaluate(provider.clone(), std::collections::HashMap::new())?;
    assert_eq!(result.queries.len(), 2);
    assert_eq!(result.queries[0].relation_name, "c");
    assert_eq!(read_unary(&provider, &result.queries[0].buffer), vec![1]);
    assert_eq!(result.queries[1].relation_name, "b");
    assert_eq!(read_unary(&provider, &result.queries[1].buffer), vec![1]);
    Ok(())
}

#[test]
fn stratified_post_preserves_projection_and_nullary_query_shape() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed");
        }
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        #pragma epistemic_mode = faeel
        pred base(u32). pred src(u32). pred a(u32). pred b(u32). pred pair(u32, u32).
        pred ready().
        base(1). src(1).
        a(X) :- src(X), know base(X).
        b(X) :- src(X), know a(X).
        pair(X, Y) :- b(X), base(Y).
        ?- pair(Y, X).
    "#;
    let mut authored = xlog_logic::parse_program(source)?;
    authored.rules.push(xlog_logic::ast::Rule {
        head: xlog_logic::ast::Atom {
            predicate: "ready".to_string(),
            terms: Vec::new(),
        },
        body: vec![xlog_logic::ast::BodyLiteral::Positive(
            xlog_logic::ast::Atom {
                predicate: "b".to_string(),
                terms: vec![xlog_logic::ast::Term::Integer(1)],
            },
        )],
    });
    authored.queries.push(xlog_logic::ast::Query {
        atom: xlog_logic::ast::Atom {
            predicate: "ready".to_string(),
            terms: Vec::new(),
        },
    });
    let program = xlog_gpu::logic::LogicProgram::compile_program(authored)?;
    let result = program.evaluate(provider, std::collections::HashMap::new())?;
    assert_eq!(result.queries.len(), 2);
    assert_eq!(result.queries[0].relation_name, "pair");
    assert_eq!(result.queries[0].columns, vec!["Y", "X"]);
    assert_eq!(result.queries[1].relation_name, "ready");
    assert!(result.queries[1].columns.is_empty());
    assert_eq!(result.queries[1].buffer.schema().arity(), 0);
    assert_eq!(result.queries[1].buffer.num_rows(), 1);
    Ok(())
}

#[test]
fn stratified_modal_query_preserves_projection_and_nullary_shape() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed");
        }
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        #pragma epistemic_mode = faeel
        pred edge(u32, u32). pred src(u32). pred a(u32, u32). pred b(u32, u32).
        edge(1, 2). src(1).
        a(X, Y) :- src(X), know edge(X, Y).
        b(X, Y) :- src(X), know a(X, Y).
        ?- b(Y, X).
        ?- b(1, Y).
    "#;
    let mut authored = xlog_logic::parse_program(source)?;
    authored.queries.push(xlog_logic::ast::Query {
        atom: xlog_logic::ast::Atom {
            predicate: "b".to_string(),
            terms: vec![
                xlog_logic::ast::Term::Integer(1),
                xlog_logic::ast::Term::Integer(2),
            ],
        },
    });
    authored.queries.push(xlog_logic::ast::Query {
        atom: xlog_logic::ast::Atom {
            predicate: "b".to_string(),
            terms: vec![
                xlog_logic::ast::Term::Integer(999),
                xlog_logic::ast::Term::Integer(999),
            ],
        },
    });
    authored.queries.push(xlog_logic::ast::Query {
        atom: xlog_logic::ast::Atom {
            predicate: "b".to_string(),
            terms: vec![
                xlog_logic::ast::Term::Variable("X".to_string()),
                xlog_logic::ast::Term::Variable("X".to_string()),
            ],
        },
    });
    let program = xlog_gpu::logic::LogicProgram::compile_program(authored)?;
    let result = program.evaluate(provider, std::collections::HashMap::new())?;
    assert_eq!(result.queries.len(), 5);
    assert_eq!(result.queries[0].relation_name, "b");
    assert_eq!(result.queries[0].columns, vec!["Y", "X"]);
    assert_eq!(result.queries[1].relation_name, "b");
    assert_eq!(result.queries[1].columns, vec!["Y"]);
    assert_eq!(result.queries[1].buffer.schema().arity(), 1);
    assert_eq!(result.queries[2].relation_name, "b");
    assert!(result.queries[2].columns.is_empty());
    assert_eq!(result.queries[2].buffer.schema().arity(), 0);
    assert_eq!(result.queries[2].buffer.num_rows(), 1);
    assert_eq!(result.queries[3].relation_name, "b");
    assert!(result.queries[3].columns.is_empty());
    assert_eq!(result.queries[3].buffer.schema().arity(), 0);
    assert_eq!(result.queries[3].buffer.num_rows(), 0);
    assert_eq!(result.queries[4].relation_name, "b");
    assert_eq!(result.queries[4].columns, vec!["X"]);
    assert_eq!(result.queries[4].buffer.schema().arity(), 1);
    assert_eq!(result.queries[4].buffer.num_rows(), 0);
    Ok(())
}

#[test]
fn stratified_queries_across_modal_strata_do_not_collide() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed");
        }
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        #pragma epistemic_mode = faeel
        pred base(u32). pred src(u32). pred edge(u32, u32).
        pred a(u32). pred b(u32, u32).
        base(1). src(1). edge(1, 2).
        a(X) :- src(X), know base(X).
        b(X, Y) :- edge(X, Y), know a(X).
        ?- a(X).
        ?- b(X, Y).
    "#;
    let program = xlog_gpu::logic::LogicProgram::compile(source)?;
    let result = program.evaluate(provider.clone(), std::collections::HashMap::new())?;

    assert_eq!(result.queries.len(), 2);
    assert_eq!(result.queries[0].relation_name, "a");
    assert_eq!(read_unary(&provider, &result.queries[0].buffer), vec![1]);
    assert_eq!(result.queries[1].relation_name, "b");
    assert_eq!(result.queries[1].columns, vec!["X", "Y"]);
    assert_eq!(result.queries[1].buffer.schema().arity(), 2);
    assert_eq!(
        read_pairs(&provider, &result.queries[1].buffer),
        vec![(1, 2)]
    );
    Ok(())
}

#[test]
fn stratified_post_ordinary_closure_true_constraint_reports_authored_identity() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed");
        }
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        #pragma epistemic_mode = faeel
        pred base(u32). pred src(u32). pred a(u32). pred b(u32). pred c(u32).
        base(1). src(1).
        a(X) :- src(X), know base(X).
        b(X) :- src(X), know a(X).
        c(X) :- b(X).
        :- c(X).
        ?- b(X).
    "#;
    let mut authored = xlog_logic::parse_program(source)?;
    authored.constraints[0].authored_index = Some(3);
    authored.prepare_authored_constraint_identity(4)?;
    let program = xlog_gpu::logic::LogicProgram::compile_program(authored)?;

    let error = match program.evaluate(provider, std::collections::HashMap::new()) {
        Ok(_) => panic!("the post-stratification ordinary closure must derive c(1)"),
        Err(error) => error,
    };
    match error {
        xlog_core::XlogError::Execution(message) => {
            assert_eq!(message, "Constraint 3 violated: :- c(X).")
        }
        other => panic!("expected typed execution error, got {other}"),
    }
    Ok(())
}

#[test]
fn stratified_post_constraint_preserves_declared_augmented_head_schema() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed");
        }
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        #pragma epistemic_mode = faeel
        pred node(u32). pred edge(u32, u32). pred a(u32). pred b(u32). pred c(u32).
        node(1). edge(1, 7).
        a(X) :- node(X), know edge(X, Y).
        b(X) :- node(X), know a(X).
        c(X) :- b(X).
        :- c(X).
        ?- b(X).
    "#;
    let mut authored = xlog_logic::parse_program(source)?;
    authored.constraints[0].authored_index = Some(3);
    authored.prepare_authored_constraint_identity(4)?;
    let program = xlog_gpu::logic::LogicProgram::compile_program(authored)?;
    assert!(program
        .epistemic_plan_json()
        .expect("epistemic plan")
        .contains("\"plan_kind\":\"epistemic_stratified\""));

    let error = match program.evaluate(provider, std::collections::HashMap::new()) {
        Ok(_) => panic!("the deferred closure must derive c(1)"),
        Err(error) => error,
    };
    match error {
        xlog_core::XlogError::Execution(message) => {
            assert_eq!(message, "Constraint 3 violated: :- c(X).")
        }
        other => panic!("expected authored constraint error, got {other}"),
    }
    assert_eq!(
        program.schema("a").expect("compiled a schema").arity(),
        2,
        "the post stage must not replace a's widened tuple-key schema"
    );
    Ok(())
}

#[test]
fn stratified_multi_head_post_constraint_false_preserves_all_outputs() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed");
        }
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        #pragma epistemic_mode = faeel
        pred base(u32). pred other_base(u32). pred src(u32). pred shared(u32).
        pred a(u32). pred d(u32). pred b(u32). pred c(u32).
        base(1). other_base(1). src(1). shared(2).
        a(X) :- src(X), know base(X).
        d(X) :- src(X), know other_base(X).
        b(X) :- src(X), know a(X).
        c(X) :- src(X), know d(X).
        :- shared(99).
        ?- b(X). ?- c(X).
    "#;
    let mut authored = xlog_logic::parse_program(source)?;
    authored.constraints[0].authored_index = Some(3);
    authored.prepare_authored_constraint_identity(4)?;
    let program = xlog_gpu::logic::LogicProgram::compile_program(authored)?;
    let plan_json = program.epistemic_plan_json().expect("epistemic plan");
    assert!(plan_json.contains("\"unit\":\"stratum[1].split[0]\""));
    assert!(plan_json.contains("\"unit\":\"stratum[1].split[1]\""));
    let result = program.evaluate(provider.clone(), std::collections::HashMap::new())?;
    let outputs = result
        .queries
        .iter()
        .map(|query| {
            (
                query.relation_name.as_str(),
                read_unary(&provider, &query.buffer),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    assert_eq!(outputs.get("b"), Some(&vec![1]));
    assert_eq!(outputs.get("c"), Some(&vec![1]));
    Ok(())
}

#[test]
fn stratified_multi_head_post_constraint_on_fact_reports_authored_identity() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA initialization failed");
        }
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let source = r#"
        #pragma epistemic_mode = faeel
        pred base(u32). pred other_base(u32). pred src(u32). pred shared(u32).
        pred a(u32). pred d(u32). pred b(u32). pred c(u32).
        base(1). other_base(1). src(1). shared(2).
        a(X) :- src(X), know base(X).
        d(X) :- src(X), know other_base(X).
        b(X) :- src(X), know a(X).
        c(X) :- src(X), know d(X).
        :- shared(2).
        ?- b(X). ?- c(X).
    "#;
    let mut authored = xlog_logic::parse_program(source)?;
    authored.constraints[0].authored_index = Some(3);
    authored.prepare_authored_constraint_identity(4)?;
    let program = xlog_gpu::logic::LogicProgram::compile_program(authored)?;
    let plan_json = program.epistemic_plan_json().expect("epistemic plan");
    assert!(plan_json.contains("\"unit\":\"stratum[1].split[0]\""));
    assert!(plan_json.contains("\"unit\":\"stratum[1].split[1]\""));

    let error = match program.evaluate(provider, std::collections::HashMap::new()) {
        Ok(_) => panic!("the fact-only post-stratification constraint must reject"),
        Err(error) => error,
    };
    match error {
        xlog_core::XlogError::Execution(message) => {
            assert_eq!(message, "Constraint 3 violated: :- shared(2).")
        }
        other => panic!("expected typed execution error, got {other}"),
    }
    Ok(())
}

/// Derived-head modal coupling mutation probe: neutralizing the modal gate on the
/// determined head flips the result, proving the stratified gating is load-bearing, not
/// cosmetic.
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

/// A positive FAEEL modal cycle has a founded least-fixpoint interpretation even when
/// neither predicate has independent support. The production dispatcher must route the
/// cycle through recursive modal reduction and return the empty founded extension.
#[test]
fn test_positive_faeel_modal_cycle_computes_empty_founded_extension() -> Result<()> {
    let Some(provider) = create_test_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };
    let program = xlog_gpu::logic::LogicProgram::compile(
        "#pragma epistemic_mode = faeel\n\
         pred a(). pred b().\n\
         a() :- know b().\n\
         b() :- know a().\n\
         ?- a(). ?- b().\n",
    )?;
    let plan_json = program
        .epistemic_plan_json()
        .expect("recursive modal reduction provenance");
    assert!(
        plan_json.contains("ordinary_recursive_modal_reduction"),
        "positive FAEEL cycle must use founded recursive reduction: {plan_json}"
    );
    let result = program.evaluate(provider, std::collections::HashMap::new())?;
    assert_eq!(result.queries.len(), 2);
    assert_eq!(result.queries[0].relation_name, "a");
    assert_eq!(result.queries[1].relation_name, "b");
    assert!(result
        .queries
        .iter()
        .all(|query| query.buffer.num_rows() == 0));

    Ok(())
}
