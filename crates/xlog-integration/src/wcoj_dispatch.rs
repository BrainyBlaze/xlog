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
//!     `tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z)` over
//!     2-column WCOJ-eligible relations (U32, Symbol, or U64
//!     keys — see `WcojKeyWidth`): three positive 2-arity body
//!     atoms covering the head's three distinct variables in
//!     head-position order. No negation, no comparison filters,
//!     no recursion (head predicate not in body), no
//!     reversed-axis atoms (e.g. `e1(Y, X)`), no constants in
//!     atom args. The planner must also return
//!     [`xlog_logic::hypergraph::RulePlan::MultiwayCandidate`].
//!   * **Width uniformity.** All three slots must share a key
//!     width. A mixed-width triangle (e.g. e1 U32, e2 U64) is
//!     rejected at this dispatch level — the binary-join chain
//!     handles it.
//!   * **Silent fallback.** Any mismatch — gate off, shape
//!     mismatch, planner verdict not multiway, missing input
//!     buffer, unsupported scalar type, mixed-width slots —
//!     returns `Ok(None)` without an error or log line. The
//!     caller is expected to silently route to the existing
//!     binary-join path. This keeps the env flag truly opt-in
//!     and prevents the helper from accidentally diverting
//!     work it can't handle.
//!   * **Strict GPU pipeline on dispatch.** When all checks pass,
//!     the helper builds three sorted+deduped layouts and runs
//!     the matching WCOJ triangle kernel on the configured
//!     `launch_stream` — `wcoj_layout_u32_recorded` /
//!     `wcoj_triangle_u32_recorded` for 4-byte keys, the
//!     `_u64_recorded` siblings for 8-byte keys. All
//!     [`xlog_cuda::launch::LaunchRecorder`] discipline carries
//!     through unchanged.
//!
//! What this slice deliberately does NOT do:
//!
//!   * No automatic detection at the executor level — callers
//!     pass the rule + input buffers explicitly. Executor
//!     wiring lives in `xlog-runtime`.
//!   * No recursion / SCC mixed execution.
//!   * No cost model.
//!   * No mixed-width admission (U32+U64 triangle stays on the
//!     binary-join path).
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
    // the WCOJ triangle kernel. Branch by width: 4-byte
    // (U32 / Symbol) and 8-byte (U64) inputs go to parallel
    // provider entries. Mixed-width triangles never reach this
    // point — `match_triangle_shape` requires all three slots
    // to share a width.
    let result = match matched.width {
        WcojKeyWidth::FourByte => {
            let buf_xy = provider.wcoj_layout_u32_recorded(matched.e_xy, launch_stream)?;
            let buf_yz = provider.wcoj_layout_u32_recorded(matched.e_yz, launch_stream)?;
            let buf_xz = provider.wcoj_layout_u32_recorded(matched.e_xz, launch_stream)?;
            provider.wcoj_triangle_u32_recorded(&buf_xy, &buf_yz, &buf_xz, launch_stream)?
        }
        WcojKeyWidth::EightByte => {
            let buf_xy = provider.wcoj_layout_u64_recorded(matched.e_xy, launch_stream)?;
            let buf_yz = provider.wcoj_layout_u64_recorded(matched.e_yz, launch_stream)?;
            let buf_xz = provider.wcoj_layout_u64_recorded(matched.e_xz, launch_stream)?;
            provider.wcoj_triangle_u64_recorded(&buf_xy, &buf_yz, &buf_xz, launch_stream)?
        }
    };
    Ok(Some(result))
}

/// Matched atom slots after pattern recognition. Each `&CudaBuffer`
/// borrows from the caller's `inputs` map. `width` is the key
/// width shared by all three slots (mixed-width triangles never
/// reach this struct — see `match_triangle_shape`).
struct MatchedAtoms<'a> {
    atoms: Vec<MatchedAtom>,
    e_xy: &'a CudaBuffer,
    e_yz: &'a CudaBuffer,
    e_xz: &'a CudaBuffer,
    width: WcojKeyWidth,
}

struct MatchedAtom {
    predicate: String,
}

/// Physical key width for a WCOJ-eligible binary relation.
/// `FourByte` covers `U32` and `Symbol` (bit-identical layout);
/// `EightByte` covers `U64`. Other scalar types are not
/// WCOJ-eligible at this dispatch level (the planner upstream
/// is the source of truth for cross-relation type compatibility;
/// `analyze_typed` rejects them before we get here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WcojKeyWidth {
    FourByte,
    EightByte,
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
    // every buffer is a 2-column WCOJ-eligible relation
    // (4-byte U32/Symbol or 8-byte U64) AND that all three
    // slots share the same width — mixed-width triangles fall
    // back here so the binary-join chain handles them.
    let mut e_xy: Option<&CudaBuffer> = None;
    let mut e_yz: Option<&CudaBuffer> = None;
    let mut e_xz: Option<&CudaBuffer> = None;
    let mut atoms_out: Vec<MatchedAtom> = Vec::with_capacity(3);
    let mut shared_width: Option<WcojKeyWidth> = None;
    for (pos0, pos1, atom) in &atom_specs {
        let buf = inputs.get(&atom.predicate)?;
        let w = classify_two_col_wcoj_width(buf)?;
        match shared_width {
            None => shared_width = Some(w),
            Some(prev) if prev == w => {}
            Some(_) => return None,
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
        width: shared_width?,
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

/// Classify a binary [`CudaBuffer`]'s key width for WCOJ
/// dispatch. Returns
///
/// * `Some(WcojKeyWidth::FourByte)` when both columns are
///   `U32` or `Symbol` (bit-identical 4-byte physical layout —
///   the WCOJ kernel reads them identically),
/// * `Some(WcojKeyWidth::EightByte)` when both columns are
///   `U64`,
/// * `None` for any other arity / type combination, including
///   mixed widths within a single buffer (e.g. one column U32
///   and the other U64).
///
/// Cross-relation type-compatibility (so a Symbol column never
/// joins with a U32 column with the same bit pattern, and a
/// U32 column never joins with a U64 column) is enforced by
/// the planner upstream via `xlog_logic::hypergraph::analyze_typed`;
/// the dispatch helper additionally requires width-uniformity
/// across the three slots in `match_triangle_shape`.
fn classify_two_col_wcoj_width(buf: &CudaBuffer) -> Option<WcojKeyWidth> {
    if buf.arity() != 2 {
        return None;
    }
    let c0 = buf.schema.column_type(0)?;
    let c1 = buf.schema.column_type(1)?;
    let width0 = scalar_wcoj_width(c0)?;
    let width1 = scalar_wcoj_width(c1)?;
    if width0 != width1 {
        return None;
    }
    Some(width0)
}

/// Map a single [`ScalarType`] to its WCOJ key width, or `None`
/// if the type is not supported by any current WCOJ entry.
fn scalar_wcoj_width(ty: ScalarType) -> Option<WcojKeyWidth> {
    match ty {
        ScalarType::U32 | ScalarType::Symbol => Some(WcojKeyWidth::FourByte),
        ScalarType::U64 => Some(WcojKeyWidth::EightByte),
        _ => None,
    }
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
