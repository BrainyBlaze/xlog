//! v0.6.2 minimal env-gated GPU 3-way WCOJ triangle dispatch.
//!
//! Single public entry:
//! [`try_wcoj_triangle_u32_dispatch`] (env-driven) and
//! [`try_wcoj_triangle_u32_dispatch_with_gate`] (boolean-driven,
//! for tests). The slice is intentionally narrow:
//!
//!   * **Env flag only.** `XLOG_USE_WCOJ_TRIANGLE_U32=1` (or
//!     `true`/`TRUE`) opts in. Anything else (unset, `0`,
//!     `false`, etc.) means the helper returns `Ok(None)`
//!     unconditionally — the caller takes the existing
//!     binary-join path.
//!   * **Recognizes exactly one shape.** A rule of the form
//!     `tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z)` over u32
//!     2-column relations: three positive 2-arity body atoms
//!     covering the head's three distinct variables in
//!     head-position order. No negation, no comparison filters,
//!     no recursion (head predicate not in body), no
//!     reversed-axis atoms (e.g. `e1(Y, X)`), no constants in
//!     atom args. The planner must also return
//!     [`xlog_logic::hypergraph::RulePlan::MultiwayCandidate`].
//!   * **Silent fallback.** Any mismatch — gate off, shape
//!     mismatch, planner verdict not multiway, missing input
//!     buffer, non-u32 schema — returns `Ok(None)` without an
//!     error or log line. The caller is expected to silently
//!     route to the existing binary-join path. This keeps the
//!     env flag truly opt-in and prevents the helper from
//!     accidentally diverting work it can't handle.
//!   * **Strict GPU pipeline on dispatch.** When all checks pass,
//!     the helper builds three sorted+deduped layouts via
//!     [`xlog_cuda::CudaKernelProvider::wcoj_layout_u32_recorded`]
//!     and runs
//!     [`xlog_cuda::CudaKernelProvider::wcoj_triangle_u32_recorded`]
//!     on the configured `launch_stream`. All
//!     [`xlog_cuda::launch::LaunchRecorder`] discipline carries
//!     through unchanged.
//!
//! What this slice deliberately does NOT do:
//!
//!   * No automatic detection at the executor level — callers
//!     pass the rule + input buffers explicitly. Wiring this
//!     into [`xlog_runtime::Executor`] is the next slice.
//!   * No recursion / SCC mixed execution.
//!   * No cost model.
//!   * No Symbol or u64 column variants.
//!   * No histogram-guided block dispatch.

use std::collections::{BTreeMap, BTreeSet};

use xlog_core::{Result, ScalarType};
use xlog_cuda::device_runtime::StreamId;
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::CudaKernelProvider;
use xlog_logic::ast::{BodyLiteral, Rule, Term};
use xlog_logic::hypergraph::{plan_rule, RefRelation, RefRelationStore, RefValue, RulePlan};

/// Env variable controlling the dispatch gate. Treated as ON
/// when set to `"1"` or case-insensitive `"true"`; anything else
/// (unset, `"0"`, `"false"`, empty string, …) means OFF.
pub const ENV_USE_WCOJ_TRIANGLE_U32: &str = "XLOG_USE_WCOJ_TRIANGLE_U32";

/// Env-driven entry. Reads `XLOG_USE_WCOJ_TRIANGLE_U32` and
/// delegates to [`try_wcoj_triangle_u32_dispatch_with_gate`].
///
/// Tests should prefer [`try_wcoj_triangle_u32_dispatch_with_gate`]
/// to avoid races on the process-global env var; the env-driven
/// form exists for production callers that want to opt in via
/// configuration alone.
pub fn try_wcoj_triangle_u32_dispatch(
    rule: &Rule,
    inputs: &BTreeMap<String, CudaBuffer>,
    provider: &CudaKernelProvider,
    launch_stream: StreamId,
) -> Result<Option<CudaBuffer>> {
    let gate = std::env::var(ENV_USE_WCOJ_TRIANGLE_U32)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    try_wcoj_triangle_u32_dispatch_with_gate(gate, rule, inputs, provider, launch_stream)
}

/// Test-friendly form that takes the gate as an explicit boolean.
/// Production callers use [`try_wcoj_triangle_u32_dispatch`] which
/// reads the env var.
pub fn try_wcoj_triangle_u32_dispatch_with_gate(
    gate_enabled: bool,
    rule: &Rule,
    inputs: &BTreeMap<String, CudaBuffer>,
    provider: &CudaKernelProvider,
    launch_stream: StreamId,
) -> Result<Option<CudaBuffer>> {
    if !gate_enabled {
        return Ok(None);
    }
    // Match the triangle shape — silently bail on any mismatch.
    let Some(matched) = match_triangle_shape(rule, inputs) else {
        return Ok(None);
    };

    // Run the planner on a synthetic store carrying the inputs'
    // schemas. plan_rule needs schemas (not rows) for typed
    // gating; rows are not consulted at the gate phase.
    let mut plan_store: RefRelationStore = BTreeMap::new();
    for atom_match in &matched.atoms {
        // Skip duplicates: the same predicate may legitimately
        // appear in multiple slots (e.g. all three atoms over a
        // single edge relation), and a BTreeMap insert collapses
        // them naturally. Schemas are guaranteed identical
        // because they all came from the same `inputs[name]`.
        plan_store.insert(
            atom_match.predicate.clone(),
            schema_only_relation(&inputs[&atom_match.predicate]),
        );
    }
    let plan = plan_rule(rule, &plan_store).ok();
    match plan {
        Some(RulePlan::MultiwayCandidate { .. }) => {}
        _ => return Ok(None),
    }

    // Construct sorted+deduped layouts for each slot and run
    // the WCOJ triangle kernel.
    let buf_xy = provider.wcoj_layout_u32_recorded(matched.e_xy, launch_stream)?;
    let buf_yz = provider.wcoj_layout_u32_recorded(matched.e_yz, launch_stream)?;
    let buf_xz = provider.wcoj_layout_u32_recorded(matched.e_xz, launch_stream)?;
    let result = provider.wcoj_triangle_u32_recorded(&buf_xy, &buf_yz, &buf_xz, launch_stream)?;
    Ok(Some(result))
}

/// Matched atom slots after pattern recognition. Each `&CudaBuffer`
/// borrows from the caller's `inputs` map.
struct MatchedAtoms<'a> {
    atoms: Vec<MatchedAtom>,
    e_xy: &'a CudaBuffer,
    e_yz: &'a CudaBuffer,
    e_xz: &'a CudaBuffer,
}

struct MatchedAtom {
    predicate: String,
}

/// Attempt to match the rule + inputs against the v1 triangle
/// shape. Returns `Some(...)` if every check passes; `None`
/// otherwise. All checks are pure validation — no buffer
/// modification, no kernel launches, no errors.
fn match_triangle_shape<'a>(
    rule: &Rule,
    inputs: &'a BTreeMap<String, CudaBuffer>,
) -> Option<MatchedAtoms<'a>> {
    // Head must have exactly 3 distinct variable terms.
    let head_vars = head_vars_or_none(rule)?;

    // Body must have exactly 3 BodyLiteral::Positive atoms; no
    // negation, no comparison filters, no is-expr.
    if rule.body.len() != 3 {
        return None;
    }
    let mut positive_atoms = Vec::with_capacity(3);
    for literal in &rule.body {
        match literal {
            BodyLiteral::Positive(atom) => positive_atoms.push(atom),
            _ => return None,
        }
    }

    // Head predicate must NOT appear in the body — rules out
    // direct recursion. Indirect recursion through SCC structure
    // is also unsupported in this slice; since we only see one
    // rule at a time, "head predicate equals any body
    // predicate" is the conservative check.
    let head_predicate = &rule.head.predicate;
    for atom in &positive_atoms {
        if &atom.predicate == head_predicate {
            return None;
        }
    }

    // Each atom must be 2-arity, with both args being head
    // variables (not constants, anonymous, or non-head vars).
    // Collect (head_position_0, head_position_1) per atom; the
    // pair must be in ascending order — i.e. atom args are in
    // head-position order, not reversed.
    //
    // After this collection, the three atoms together must
    // cover EXACTLY the three pairs {0,1}, {1,2}, {0,2} (the
    // triangle's three edges over head positions).
    let mut atom_specs: Vec<(usize, usize, &xlog_logic::ast::Atom)> = Vec::with_capacity(3);
    for atom in &positive_atoms {
        if atom.terms.len() != 2 {
            return None;
        }
        let pos0 = head_pos_of_var(&head_vars, &atom.terms[0])?;
        let pos1 = head_pos_of_var(&head_vars, &atom.terms[1])?;
        if pos0 >= pos1 {
            // Same variable twice (`e(X, X)`) or reversed
            // axis (`e(Y, X)`). Both are out-of-scope for v1
            // dispatch.
            return None;
        }
        atom_specs.push((pos0, pos1, atom));
    }
    let pair_set: BTreeSet<(usize, usize)> = atom_specs.iter().map(|(a, b, _)| (*a, *b)).collect();
    let mut expected: BTreeSet<(usize, usize)> = BTreeSet::new();
    expected.insert((0, 1));
    expected.insert((1, 2));
    expected.insert((0, 2));
    if pair_set != expected {
        return None;
    }

    // Look up each input buffer by predicate name. Validate
    // every buffer is 2-column u32.
    let mut e_xy: Option<&CudaBuffer> = None;
    let mut e_yz: Option<&CudaBuffer> = None;
    let mut e_xz: Option<&CudaBuffer> = None;
    let mut atoms_out: Vec<MatchedAtom> = Vec::with_capacity(3);
    for (pos0, pos1, atom) in &atom_specs {
        let buf = inputs.get(&atom.predicate)?;
        if !is_two_col_u32(buf) {
            return None;
        }
        atoms_out.push(MatchedAtom {
            predicate: atom.predicate.clone(),
        });
        match (*pos0, *pos1) {
            (0, 1) => e_xy = Some(buf),
            (1, 2) => e_yz = Some(buf),
            (0, 2) => e_xz = Some(buf),
            _ => unreachable!("pair_set check above guarantees these three"),
        }
    }
    Some(MatchedAtoms {
        atoms: atoms_out,
        e_xy: e_xy?,
        e_yz: e_yz?,
        e_xz: e_xz?,
    })
}

/// Return the rule head's variable names if every head term is a
/// distinct `Term::Variable`; `None` otherwise.
fn head_vars_or_none(rule: &Rule) -> Option<Vec<String>> {
    if rule.head.terms.len() != 3 {
        return None;
    }
    let mut out = Vec::with_capacity(3);
    for term in &rule.head.terms {
        match term {
            Term::Variable(name) => out.push(name.clone()),
            _ => return None,
        }
    }
    // Distinctness.
    let unique: BTreeSet<&String> = out.iter().collect();
    if unique.len() != 3 {
        return None;
    }
    Some(out)
}

/// Map an atom-arg term to its position in the head variable list
/// (0, 1, or 2). Returns `None` for anything that isn't a
/// head-list variable.
fn head_pos_of_var(head_vars: &[String], term: &Term) -> Option<usize> {
    match term {
        Term::Variable(name) => head_vars.iter().position(|v| v == name),
        _ => None,
    }
}

/// True when `buf` has 2 columns and both columns are 4-byte
/// keys ([`ScalarType::U32`] or [`ScalarType::Symbol`]). Both
/// share the same 4-byte physical layout, so the WCOJ kernel
/// reads them identically; the cross-relation type-compatibility
/// check (so symbol IDs don't accidentally join with raw u32s)
/// is enforced by the planner upstream via
/// `xlog_logic::hypergraph::analyze_typed`.
fn is_two_col_u32(buf: &CudaBuffer) -> bool {
    if buf.arity() != 2 {
        return false;
    }
    for col_idx in 0..2 {
        match buf.schema.column_type(col_idx) {
            Some(ScalarType::U32) | Some(ScalarType::Symbol) => {}
            _ => return false,
        }
    }
    true
}

/// Build a synthetic schema-only [`RefRelation`] for the planner's
/// view. Rows are intentionally empty: `plan_rule` only consults
/// schemas for typed gating, never rows.
fn schema_only_relation(buf: &CudaBuffer) -> RefRelation {
    let arity = buf.arity();
    let schema: Vec<ScalarType> = (0..arity)
        .map(|i| buf.schema.column_type(i).unwrap_or(ScalarType::U32))
        .collect();
    RefRelation {
        schema,
        rows: Vec::<Vec<RefValue>>::new(),
    }
}
