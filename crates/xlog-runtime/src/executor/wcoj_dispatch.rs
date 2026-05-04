//! v0.6.2 WCOJ triangle dispatch — runtime hook.
//!
//! Wires the GPU 3-way WCOJ kernel into the executor's per-rule
//! loop. **Default-on** (post-A2-lite default flip): the
//! adaptive classifier runs on every matching non-recursive
//! triangle rule and dispatches WCOJ when the per-key skew
//! score clears [`WCOJ_ADAPTIVE_SKEW_THRESHOLD`]. Production
//! callers leave `RuntimeConfig::default()` and accept the
//! adaptive path.
//!
//! Override knobs (config + env, highest precedence first):
//!
//!   1. **Hard kill switch** — `wcoj_triangle_dispatch_disabled` /
//!      [`ENV_DISABLE_WCOJ_TRIANGLE`]. Pins all dispatch off,
//!      including force. Ops emergency knob.
//!   2. **Force-WCOJ** — `wcoj_triangle_dispatch=Some(true)` /
//!      [`ENV_USE_WCOJ_TRIANGLE_U32`]. Bypasses classifier.
//!   3. **Explicit force-off** —
//!      `wcoj_triangle_dispatch=Some(false)`. Used by bench
//!      `Mode::Off` cells and any test that wants binary-join.
//!   4. **Adaptive opt-out** —
//!      `wcoj_triangle_dispatch_adaptive=Some(false)`. Disables
//!      the default-on classifier without a global env var.
//!   5. **Default**: classifier runs.
//!
//! ## Recognized RIR shape (v0.6.5)
//!
//! The hook now consumes [`RirNode::MultiWayJoin`], produced by
//! [`xlog_logic::promote::promote_multiway`] after the optimizer
//! pass in [`xlog_logic::Compiler::compile_program_with_stats_snapshot`].
//! The promoter rewrites the canonical lowered+optimized triangle
//! tree to a `MultiWayJoin` whose structure encodes the same
//! semantic invariants as the v0.6.2 strict tree-pattern matcher:
//!
//! * `inputs` is a 3-element vec of `Scan` nodes in WCOJ slot
//!   order `[xy, yz, xz]`.
//! * `slot_vars` is exactly `[[Some(0), Some(1)], [Some(1), Some(2)],
//!   [Some(0), Some(2)]]` — variable-class ids for X, Y, Z.
//! * `output_columns` is exactly
//!   `[Column(0), Column(1), Column(3)]` (matching the certified
//!   GPU kernel's (X, Y, Z) emit order).
//! * `fallback` is the post-optimizer binary-join tree, executed
//!   verbatim when this hook declines.
//!
//! Anything else (rotated/computed projection, non-canonical
//! slot_vars, non-Scan inputs, recursive SCC, missing input
//! buffers, unsupported scalar types, mixed-width slots, or no
//! runtime-backed memory manager) returns `Ok(None)` and the
//! caller takes the embedded `fallback` path.
//!
//! Width branching: 4-byte (U32 / Symbol) inputs go to
//! `wcoj_layout_u32_recorded` + `wcoj_triangle_u32_recorded`;
//! 8-byte (U64) inputs go to the `_u64_recorded` siblings. All
//! three slots must share a width — mixed-width triangles fall
//! back to the binary-join path.
//!
//! ## Failure handling
//!
//! Per slice spec: "failure in helper must not corrupt store
//! state." If the WCOJ pipeline (layout construction or kernel
//! launch) returns an error, the hook converts it to `Ok(None)`
//! and the caller falls back to the existing path. The store is
//! never partially mutated; the dispatch hook only writes when the
//! full pipeline succeeds, and the writeback is the caller's
//! responsibility.
//!
//! ## Hook surface
//!
//! The dispatcher exposes two entry points per shape (slice 4):
//!
//! * `try_dispatch_wcoj_*(rule)` — keyed on `&CompiledRule`,
//!   used by the non-recursive arm in `execute_stratum_impl`.
//! * `try_dispatch_wcoj_*_on_body(body)` — keyed on `&RirNode`,
//!   used by the recursive arm via
//!   `Executor::execute_wcoj_or_fallback_node` on both seeding
//!   and per-variant evaluation. The slice 4 promoter gates
//!   recursive bodies on per-rule recursive-Scan count (≤ 1
//!   promotes; ≥ 2 stays binary-join — see slice 4.2 deferral).
//!
//! ## Out of scope (per slice spec)
//!
//! * Cost model — slice 5.
//! * Mixed-width admission (a triangle with both U32 and U64
//!   slots stays on the binary-join path).
//! * Histogram-guided block dispatch (B1 heavy-row offload).
//! * Multi-recursive WCOJ (≥ 2 in-SCC body Scans) — slice 4.2.

use xlog_core::{RelId, Result, ScalarType, Schema};
use xlog_cuda::device_runtime::StreamId;
use xlog_cuda::CudaBuffer;
use xlog_ir::{
    rir::{ProjectExpr, VariableOrder},
    CompiledRule, RirNode,
};

use super::Executor;

#[cfg(feature = "wcoj-phase-timing")]
use std::time::Instant;

/// Env variable controlling the WCOJ triangle dispatch. Treated
/// as ON when set to `"1"` or case-insensitive `"true"`; anything
/// else (unset, `"0"`, `"false"`, empty string, …) means OFF.
pub const ENV_USE_WCOJ_TRIANGLE_U32: &str = "XLOG_USE_WCOJ_TRIANGLE_U32";

/// Resolve the dispatch gate. Config override (set by tests)
/// takes precedence over the env var. Production callers leave
/// the override as `None` and configure via env.
pub(super) fn wcoj_gate_enabled(config_override: Option<bool>) -> bool {
    if let Some(v) = config_override {
        return v;
    }
    std::env::var(ENV_USE_WCOJ_TRIANGLE_U32)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Env variable controlling the adaptive WCOJ dispatch.
/// `"1"` / case-insensitive `"true"` → ON. Anything else
/// (including unset) is *not* a hard off — the resolver
/// defaults to ON when this env is unset (post-default-on
/// flip). To explicitly disable adaptive, use
/// `RuntimeConfig::wcoj_triangle_dispatch_adaptive = Some(false)`.
pub const ENV_USE_WCOJ_TRIANGLE_ADAPTIVE: &str = "XLOG_USE_WCOJ_TRIANGLE_ADAPTIVE";

/// Env variable for the hard kill switch. `"1"` / case-
/// insensitive `"true"` → kill. Beats every other flag.
pub const ENV_DISABLE_WCOJ_TRIANGLE: &str = "XLOG_DISABLE_WCOJ_TRIANGLE";

/// Resolve the adaptive dispatch gate. Precedence:
///   * `config_override = Some(b)` → `b` (test-only knob).
///   * `XLOG_USE_WCOJ_TRIANGLE_ADAPTIVE=1` → `true`.
///   * `XLOG_USE_WCOJ_TRIANGLE_ADAPTIVE` set to any other
///     value (`"0"`, `"false"`, …) → `false`.
///   * Unset → `true` (default-on flip).
pub(super) fn wcoj_adaptive_enabled(config_override: Option<bool>) -> bool {
    if let Some(v) = config_override {
        return v;
    }
    match std::env::var(ENV_USE_WCOJ_TRIANGLE_ADAPTIVE) {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
        // Default-on: when env is unset, adaptive runs.
        Err(_) => true,
    }
}

/// Resolve the kill switch. Same precedence shape as
/// `wcoj_gate_enabled` (config override > env > false).
/// Returns `true` when dispatch should be hard-disabled.
pub(super) fn wcoj_disabled(config_override: Option<bool>) -> bool {
    if let Some(v) = config_override {
        return v;
    }
    std::env::var(ENV_DISABLE_WCOJ_TRIANGLE)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

// -----------------------------------------------------------------
// v0.6.5 slice 2 — 4-cycle dispatch gates.
//
// Width-neutral env naming: `XLOG_USE_WCOJ_4CYCLE` controls the
// force gate across u32 / u64 / Symbol. Triangle's `_U32` suffix is
// historical debt; we do NOT propagate that pattern to 4-cycle.
//
// Adaptive resolution differs from triangle: 4-cycle is **opt-in by
// default**. Unset env + `None` config → `false`. Default-on is
// gated on bench evidence and lives in a separate follow-up slice.
// -----------------------------------------------------------------

/// Force-gate env. `"1"` / case-insensitive `"true"` → ON.
pub const ENV_USE_WCOJ_4CYCLE: &str = "XLOG_USE_WCOJ_4CYCLE";

/// Adaptive opt-in env. Default off (slice 2 ships explicit-only).
pub const ENV_USE_WCOJ_4CYCLE_ADAPTIVE: &str = "XLOG_USE_WCOJ_4CYCLE_ADAPTIVE";

/// Kill switch env.
pub const ENV_DISABLE_WCOJ_4CYCLE: &str = "XLOG_DISABLE_WCOJ_4CYCLE";

/// Resolve the 4-cycle force gate (config override > env > false).
pub(super) fn wcoj_4cycle_gate_enabled(config_override: Option<bool>) -> bool {
    if let Some(v) = config_override {
        return v;
    }
    std::env::var(ENV_USE_WCOJ_4CYCLE)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Resolve the 4-cycle adaptive opt-in. Precedence:
///   * `config_override = Some(b)` → `b`.
///   * `XLOG_USE_WCOJ_4CYCLE_ADAPTIVE=1` → `true`.
///   * Anything else (including unset) → `false`.
///
/// **Differs from triangle**: triangle defaults adaptive to `true`
/// when env is unset (default-on flip after baseline evidence).
/// 4-cycle defaults to `false` until its own baseline evidence
/// supports a default-on flip in a follow-up slice.
pub(super) fn wcoj_4cycle_adaptive_enabled(config_override: Option<bool>) -> bool {
    if let Some(v) = config_override {
        return v;
    }
    std::env::var(ENV_USE_WCOJ_4CYCLE_ADAPTIVE)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Resolve the 4-cycle kill switch (config > env > false).
pub(super) fn wcoj_4cycle_disabled(config_override: Option<bool>) -> bool {
    if let Some(v) = config_override {
        return v;
    }
    std::env::var(ENV_DISABLE_WCOJ_4CYCLE)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Threshold at which a classifier score routes the rule to the
/// WCOJ pipeline rather than the binary-join fallback. Locked
/// from the v0.6.2 baseline probe in
/// `docs/evidence/2026-05-01-wcoj-bench-baseline/`: uniform/empty
/// fixtures score ≤ 0.04, super-hub fixtures score ≥ 0.18.
/// Threshold of 0.10 sits in the gap with ≥1.7× headroom on each
/// side — robust to bench/kernel noise.
pub(super) const WCOJ_ADAPTIVE_SKEW_THRESHOLD: f64 = 0.10;

/// v0.6.5 slice 2 — threshold for the 4-cycle adaptive
/// classifier. Reduction across the four join positions is
/// `max(score_per_position)`, which keeps the score in the same
/// `[0, 1]` range as the triangle classifier — so the same `0.10`
/// threshold transfers directly. Bench evidence under
/// `docs/evidence/2026-05-?-wcoj-4cycle-bench-baseline/`
/// (slice 2 step 10) verifies the gap has ≥1.7× headroom on
/// each side; if the evidence shows a different threshold is
/// warranted, lock the new value before merging.
pub(super) const WCOJ_ADAPTIVE_4CYCLE_SKEW_THRESHOLD: f64 = 0.10;

/// Resolved dispatch mode after consulting both gates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DispatchMode {
    /// Force-WCOJ: classifier is bypassed entirely; dispatch
    /// fires whenever the RIR + buffers + width all match.
    /// Set by `wcoj_triangle_dispatch=Some(true)` or env=1.
    Force,
    /// Adaptive: run the GPU skew classifier; dispatch only
    /// when the score clears [`WCOJ_ADAPTIVE_SKEW_THRESHOLD`].
    /// Set by `wcoj_triangle_dispatch_adaptive=Some(true)` (and
    /// force is not on).
    Adaptive,
}

/// Three rel IDs extracted from a matched triangle RIR. The
/// names correspond to the WCOJ kernel's slot semantics.
pub(super) struct TriangleRirMatch {
    /// Rel for the (X, Y) edge — left subtree of the inner join,
    /// joined with `rel_yz` on Y.
    pub rel_xy: RelId,
    /// Rel for the (Y, Z) edge — right subtree of the inner join.
    pub rel_yz: RelId,
    /// Rel for the (X, Z) closing edge — right subtree of the
    /// outer join, joined with the inner join's output on (X, Z).
    pub rel_xz: RelId,
}

/// Pattern-match a `RirNode::MultiWayJoin` whose structure is the
/// canonical triangle shape. Returns the three scan rel IDs in
/// WCOJ slot order on a successful match; `None` for any deviation.
///
/// The match is intentionally strict over `inputs`, `slot_vars`,
/// AND `output_columns`. v0.6.5 slice 1 only certifies the
/// canonical (X, Y, Z) emit order; rotated head projections,
/// non-Scan inputs, or non-canonical variable classes decline
/// dispatch and the caller takes the embedded `fallback` path.
///
/// Future slices generalize the matcher in tandem with kernel
/// generalization (4-way, n-way) — never one without the other.
pub(super) fn match_multiway_triangle(body: &RirNode) -> Option<TriangleRirMatch> {
    let RirNode::MultiWayJoin {
        inputs,
        slot_vars,
        output_columns,
        ..
    } = body
    else {
        return None;
    };
    if inputs.len() != 3 {
        return None;
    }
    if !slot_vars_match_canonical_triangle(slot_vars) {
        return None;
    }
    if !output_columns_match_canonical_triangle(output_columns) {
        return None;
    }
    let rel_xy = scan_rel(&inputs[0])?;
    let rel_yz = scan_rel(&inputs[1])?;
    let rel_xz = scan_rel(&inputs[2])?;
    Some(TriangleRirMatch {
        rel_xy,
        rel_yz,
        rel_xz,
    })
}

/// Confirm `slot_vars` is the canonical
/// `[[A, B], [B, C], [A, C]]` triangle shape with three distinct
/// variable-class ids. Anything else (rotated, dropped, repeated)
/// fails the match.
fn slot_vars_match_canonical_triangle(slot_vars: &[Vec<Option<u32>>]) -> bool {
    if slot_vars.len() != 3 {
        return false;
    }
    let s0 = &slot_vars[0];
    let s1 = &slot_vars[1];
    let s2 = &slot_vars[2];
    if s0.len() != 2 || s1.len() != 2 || s2.len() != 2 {
        return false;
    }
    let (a, b) = match (s0[0], s0[1]) {
        (Some(a), Some(b)) if a != b => (a, b),
        _ => return false,
    };
    let c = match (s1[0], s1[1]) {
        (Some(b1), Some(c)) if b1 == b && c != a && c != b => c,
        _ => return false,
    };
    matches!((s2[0], s2[1]), (Some(a2), Some(c2)) if a2 == a && c2 == c)
}

/// Confirm `output_columns` is one of the valid head-extraction
/// layouts. The GPU kernel writes triples in canonical
/// `(X, Y, Z)` order; the project columns describe the
/// binary-join-intermediate layout the head extracts from.
///
/// W2.2 — accepted layouts:
///   * `[Column(0), Column(1), Column(3)]` — Y-shared /
///     X-shared inner pair (binary intermediate cols
///     [X, Y, Y, Z, X, Z] / [X, Y, X, Z, Y, Z]).
///   * `[Column(0), Column(2), Column(3)]` — Z-shared inner
///     pair (binary intermediate cols [X, Z, Y, Z, X, Y]).
fn output_columns_match_canonical_triangle(cols: &[ProjectExpr]) -> bool {
    if cols.len() != 3 {
        return false;
    }
    let cols_pattern = (
        matches!(cols[0], ProjectExpr::Column(0)),
        matches!(cols[1], ProjectExpr::Column(1)) || matches!(cols[1], ProjectExpr::Column(2)),
        matches!(cols[2], ProjectExpr::Column(3)),
    );
    cols_pattern == (true, true, true)
}

// -----------------------------------------------------------------
// v0.6.5 slice 2 — 4-cycle matcher.
//
// Mirrors the triangle matcher with shape-locked qualifier per the
// slice 2 walker contract.
// -----------------------------------------------------------------

/// Four rel IDs extracted from a matched 4-cycle RIR.
pub(super) struct FourCycleRirMatch {
    pub rel_e1: RelId,
    pub rel_e2: RelId,
    pub rel_e3: RelId,
    pub rel_e4: RelId,
}

/// Pattern-match a `RirNode::MultiWayJoin` whose structure is the
/// canonical 4-cycle shape. Returns the four scan rel IDs in WCOJ
/// slot order on a successful match; `None` for any deviation.
///
/// The match is intentionally strict over `inputs`, `slot_vars`,
/// AND `output_columns`. v0.6.5 slice 2 only certifies the
/// canonical (W, X, Y, Z) emit order.
pub(super) fn match_multiway_4cycle(body: &RirNode) -> Option<FourCycleRirMatch> {
    let RirNode::MultiWayJoin {
        inputs,
        slot_vars,
        output_columns,
        ..
    } = body
    else {
        return None;
    };
    if inputs.len() != 4 {
        return None;
    }
    if !slot_vars_match_canonical_4cycle(slot_vars) {
        return None;
    }
    if !output_columns_match_canonical_4cycle(output_columns) {
        return None;
    }
    let rel_e1 = scan_rel(&inputs[0])?;
    let rel_e2 = scan_rel(&inputs[1])?;
    let rel_e3 = scan_rel(&inputs[2])?;
    let rel_e4 = scan_rel(&inputs[3])?;
    Some(FourCycleRirMatch {
        rel_e1,
        rel_e2,
        rel_e3,
        rel_e4,
    })
}

/// Confirm `slot_vars` is the canonical
/// `[[A, B], [B, C], [C, D], [D, A]]` 4-cycle shape with four
/// distinct variable-class ids closing the cycle (slot 3's second
/// var equals slot 0's first var).
fn slot_vars_match_canonical_4cycle(slot_vars: &[Vec<Option<u32>>]) -> bool {
    if slot_vars.len() != 4 {
        return false;
    }
    for s in slot_vars {
        if s.len() != 2 {
            return false;
        }
    }
    let (a, b) = match (slot_vars[0][0], slot_vars[0][1]) {
        (Some(a), Some(b)) if a != b => (a, b),
        _ => return false,
    };
    let c = match (slot_vars[1][0], slot_vars[1][1]) {
        (Some(b1), Some(c)) if b1 == b && c != a && c != b => c,
        _ => return false,
    };
    let d = match (slot_vars[2][0], slot_vars[2][1]) {
        (Some(c1), Some(d)) if c1 == c && d != a && d != b && d != c => d,
        _ => return false,
    };
    matches!(
        (slot_vars[3][0], slot_vars[3][1]),
        (Some(d2), Some(a2)) if d2 == d && a2 == a
    )
}

/// Confirm `output_columns` is the certified `(W, X, Y, Z)` emit
/// order. The GPU kernel writes quads in this order.
/// W2.2 — 4-cycle accepted output_column layouts:
///   * `[Column(0), Column(1), Column(3), Column(5)]` —
///     Default grouping `(WX⋈XY) + (YZ⋈ZW)`.
///   * `[Column(5), Column(0), Column(1), Column(3)]` — Alt
///     grouping `(XY⋈YZ) + (ZW⋈WX)` (binary intermediate
///     col 5 = W from inner-right; (W, X, Y, Z) extracts
///     from cols [5, 0, 1, 3]).
fn output_columns_match_canonical_4cycle(cols: &[ProjectExpr]) -> bool {
    if cols.len() != 4 {
        return false;
    }
    let exact = |idx: usize, want: usize| matches!(cols[idx], ProjectExpr::Column(c) if c == want);
    // Default layout.
    let default_layout = exact(0, 0) && exact(1, 1) && exact(2, 3) && exact(3, 5);
    // Alt layout.
    let alt_layout = exact(0, 5) && exact(1, 0) && exact(2, 1) && exact(3, 3);
    default_layout || alt_layout
}

/// Extract the `RelId` from a leaf `Scan` node, or `None` for
/// any non-Scan child. v0.6.5 slice 1 only admits Scan leaves;
/// future slices may admit `Filter { Scan }` or projected
/// scans, but always in tandem with kernel support.
fn scan_rel(node: &RirNode) -> Option<RelId> {
    match node {
        RirNode::Scan { rel } => Some(*rel),
        _ => None,
    }
}

/// Physical key width for a WCOJ-eligible binary relation at
/// the RIR-level dispatch. `FourByte` covers `U32` and `Symbol`
/// (bit-identical layout); `EightByte` covers `U64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WcojKeyWidth {
    FourByte,
    EightByte,
}

/// Classify a binary [`CudaBuffer`]'s key width for WCOJ
/// dispatch, mirroring `xlog_integration::wcoj_dispatch`'s
/// AST-level helper. Returns `Some(width)` for 2-column buffers
/// whose columns are both 4-byte (U32/Symbol) or both 8-byte
/// (U64); `None` for any other arity / type combination,
/// including mixed-width within a single buffer.
///
/// Cross-relation type compatibility is enforced upstream by
/// the planner via `analyze_typed`. The executor only sees
/// lowered RIR at this point, so this classifier is the last
/// width-uniformity check before the GPU launch — any
/// divergence vs. the binary-join path is caught by the
/// wiring/cert row-set-equality tests.
fn classify_two_col_wcoj_width(buf: &CudaBuffer) -> Option<WcojKeyWidth> {
    if buf.arity() != 2 {
        return None;
    }
    let c0 = buf.schema.column_type(0)?;
    let c1 = buf.schema.column_type(1)?;
    let w0 = scalar_wcoj_width(c0)?;
    let w1 = scalar_wcoj_width(c1)?;
    if w0 != w1 {
        return None;
    }
    Some(w0)
}

fn scalar_wcoj_width(ty: xlog_core::ScalarType) -> Option<WcojKeyWidth> {
    match ty {
        xlog_core::ScalarType::U32 | xlog_core::ScalarType::Symbol => Some(WcojKeyWidth::FourByte),
        xlog_core::ScalarType::U64 => Some(WcojKeyWidth::EightByte),
        _ => None,
    }
}

/// W2.1: convert `kernel_output_cols` (a `Vec<ProjectExpr>`) into
/// the `Vec<usize>` permutation that
/// `wcoj_project_output_columns_recorded` consumes. Triangle and
/// 4-cycle kernel_output_cols entries are always
/// `ProjectExpr::Column(_)` per the locked permutation tables in
/// `xlog_logic::wcoj_var_ordering`; anything else is a planner bug.
/// W2.6 step 5 — derive the slot-0 ⋈ slot-1 feedback pair AND
/// the underlying-relation key columns from `var_order`.
///
/// Returns `(rel_a, rel_b, left_keys, right_keys)` where keys
/// are NATIVE (pre-swap) column indices on the underlying
/// relations — `record_join_result` stores keys against native
/// indexing. For triangle non-default leaders, slot 1 is a
/// 2-col SWAPPED view of the underlying relation; the kernel
/// invariant `slot0.col1 ≡ slot1.col0` holds for the views
/// but maps to native key index 1 on BOTH sides.
///
/// **Locked rotated-feedback table** (W2.6 plan §"Step 5"):
///
/// | Shape    | Leader            | (rel_a, rel_b)      | (left_keys, right_keys) |
/// |----------|-------------------|---------------------|-------------------------|
/// | Triangle | 0 (e_xy default)  | (slot[0], slot[1])  | [1] / [0] (no swap) |
/// | Triangle | 1 (e_yz)          | (slot[1], slot[2])  | **[1] / [1]** (slot 1 = e_xz↔) |
/// | Triangle | 2 (e_xz)          | (slot[2], slot[1])  | **[1] / [1]** (slot 1 = e_yz↔) |
/// | 4-cycle  | 0..3 (rotation)   | (slot[i], slot[i+1])| [1] / [0] (no swap) |
///
/// Returns `None` only if `slot_rels.len() < 2` (defensive).
fn feedback_pair_from_var_order(
    slot_rels: &[RelId],
    var_order: Option<&VariableOrder>,
) -> Option<(RelId, RelId, Vec<usize>, Vec<usize>)> {
    if slot_rels.len() < 2 {
        return None;
    }
    let Some(vo) = var_order else {
        // Default config / no rotation — bit-identical W2.4
        // behavior: canonical (slot_rels[0], slot_rels[1]) with
        // keys [1] / [0].
        return Some((slot_rels[0], slot_rels[1], vec![1], vec![0]));
    };
    let leader_idx = vo.leader_idx as usize;
    match slot_rels.len() {
        3 => {
            // Triangle: locked table per W2.6 plan §"Step 5".
            match leader_idx {
                0 => Some((slot_rels[0], slot_rels[1], vec![1], vec![0])),
                1 => {
                    // Leader e_yz: slot 0 = rel_yz native, slot 1 =
                    // rel_xz **swapped** view. Native rel_xz has Z
                    // at col1, so [1]/[1].
                    Some((slot_rels[1], slot_rels[2], vec![1], vec![1]))
                }
                2 => {
                    // Leader e_xz: slot 0 = rel_xz native, slot 1 =
                    // rel_yz **swapped** view. Native rel_yz has Z
                    // at col1, so [1]/[1].
                    Some((slot_rels[2], slot_rels[1], vec![1], vec![1]))
                }
                _ => None,
            }
        }
        4 => {
            // 4-cycle: rotation-only, all slots in native layout,
            // keys [1]/[0] for every leader.
            if leader_idx >= 4 {
                return None;
            }
            let slot1_input_idx = (leader_idx + 1) % 4;
            Some((
                slot_rels[leader_idx],
                slot_rels[slot1_input_idx],
                vec![1],
                vec![0],
            ))
        }
        _ => None,
    }
}

fn perm_indices_from_kernel_output_cols(cols: &[ProjectExpr]) -> Result<Vec<usize>> {
    let mut out = Vec::with_capacity(cols.len());
    for c in cols {
        match c {
            ProjectExpr::Column(idx) => out.push(*idx),
            other => {
                return Err(xlog_core::XlogError::Kernel(format!(
                    "perm_indices_from_kernel_output_cols: \
                     W2.1 kernel_output_cols must be ProjectExpr::Column(_), got {:?}",
                    other
                )));
            }
        }
    }
    Ok(out)
}

/// W2.1: build the canonical triangle head schema `(X, Y, Z)`
/// from the canonical promoter inputs. Used as the
/// `head_schema` argument to
/// `wcoj_project_output_columns_recorded` on the W2.1 path.
fn build_triangle_head_schema(buf_xy: &CudaBuffer, buf_yz: &CudaBuffer) -> Result<Schema> {
    let x_type = buf_xy.schema.column_type(0).ok_or_else(|| {
        xlog_core::XlogError::Kernel("build_triangle_head_schema: e_xy.col0 type missing".into())
    })?;
    let y_type = buf_xy.schema.column_type(1).ok_or_else(|| {
        xlog_core::XlogError::Kernel("build_triangle_head_schema: e_xy.col1 type missing".into())
    })?;
    let z_type = buf_yz.schema.column_type(1).ok_or_else(|| {
        xlog_core::XlogError::Kernel("build_triangle_head_schema: e_yz.col1 type missing".into())
    })?;
    Ok(Schema::new(vec![
        ("col0".to_string(), x_type),
        ("col1".to_string(), y_type),
        ("col2".to_string(), z_type),
    ]))
}

/// W2.1: build the canonical 4-cycle head schema
/// `(W, X, Y, Z)` from the canonical promoter inputs.
fn build_4cycle_head_schema(
    buf_e1: &CudaBuffer,
    buf_e2: &CudaBuffer,
    buf_e3: &CudaBuffer,
) -> Result<Schema> {
    // `[e_wx, e_xy, e_yz, e_zw]` — canonical promoter order.
    // W = e_wx.col0, X = e_wx.col1 (= e_xy.col0), Y = e_xy.col1
    // (= e_yz.col0), Z = e_yz.col1 (= e_zw.col0).
    let w_type = buf_e1.schema.column_type(0).ok_or_else(|| {
        xlog_core::XlogError::Kernel("build_4cycle_head_schema: e_wx.col0 type missing".into())
    })?;
    let x_type = buf_e1.schema.column_type(1).ok_or_else(|| {
        xlog_core::XlogError::Kernel("build_4cycle_head_schema: e_wx.col1 type missing".into())
    })?;
    let y_type = buf_e2.schema.column_type(1).ok_or_else(|| {
        xlog_core::XlogError::Kernel("build_4cycle_head_schema: e_xy.col1 type missing".into())
    })?;
    let z_type = buf_e3.schema.column_type(1).ok_or_else(|| {
        xlog_core::XlogError::Kernel("build_4cycle_head_schema: e_yz.col1 type missing".into())
    })?;
    // Suppress the unused-import warning when ScalarType isn't
    // referenced in this scope (kept here for explicitness in case
    // a future change adds a width check).
    let _: ScalarType = w_type;
    Ok(Schema::new(vec![
        ("col0".to_string(), w_type),
        ("col1".to_string(), x_type),
        ("col2".to_string(), y_type),
        ("col3".to_string(), z_type),
    ]))
}

impl Executor {
    /// Try to dispatch a single non-recursive rule through the
    /// GPU WCOJ triangle kernel. Returns `Ok(Some(buffer))` if
    /// the dispatch fires and produces a result; `Ok(None)`
    /// otherwise (gate off, shape mismatch, missing buffer,
    /// non-4-byte-key schema, missing runtime, or kernel error — every
    /// failure mode is silent fallback).
    ///
    /// On `Ok(Some(_))`, the caller is responsible for installing
    /// the buffer into the relation store via the same path the
    /// existing binary-join branch uses.
    pub(super) fn try_dispatch_wcoj_triangle(
        &mut self,
        rule: &CompiledRule,
    ) -> Result<Option<CudaBuffer>> {
        // Slice 4: body-keyed entry. Rule-keyed callers stay
        // byte-identical via this thin wrapper.
        self.try_dispatch_wcoj_triangle_on_body(&rule.body)
    }

    /// W2.4 — read the WCOJ output buffer's logical row count.
    /// Returns `None` when the cache isn't populated. **Never
    /// returns `Some(0)` for an unknown row count** — only for
    /// an observed-empty output. The distinction matters for
    /// `record_wcoj_feedback`: an unknown count must skip the
    /// EMA update, not record selectivity 0.
    fn wcoj_output_rows(buf: &CudaBuffer) -> Option<u64> {
        // `CudaBuffer::cached_row_count` returns `Option<u32>`;
        // widen to `u64` for the `StatsManager` API.
        buf.cached_row_count().map(u64::from)
    }

    /// W2.4 + W2.6 — wire successful WCOJ dispatches back into
    /// `StatsManager` so the cardinality cost model's future
    /// `binary_est` reads reflect observed selectivity.
    ///
    /// **W2.6 routing**: the `(rel_a, rel_b, left_keys, right_keys)`
    /// quadruple is derived from the dispatched plan's
    /// `var_order` via `feedback_pair_from_var_order`, NOT
    /// hardcoded:
    ///
    /// * `var_order = None` (default config): returns the
    ///   pre-W2.6 W2.4 pair — `(slot_rels[0], slot_rels[1])`
    ///   with keys `[1] / [0]`. Bit-identical to slice 1-5 +
    ///   W2.4.
    /// * `var_order = Some(_)` (W2.1 LeaderCardinality or W2.6
    ///   HeatAware non-default leader): returns the rotated
    ///   pair from the locked feedback table — triangle
    ///   non-default leaders use rotated `(slot_rels[0],
    ///   slot_rels[1])` with keys `[1] / [1]` (Z-shared edges
    ///   in canonical layout join on col 1 of both rels);
    ///   4-cycle is rotation-only with keys `[1] / [0]`.
    ///
    /// `CardinalityAwareCostModel::should_dispatch_*` still
    /// reads via `estimate_join_cardinality` on the canonical
    /// default-leader pair — but on a non-default-leader run
    /// the dispatched layout's actual edge is what we observe,
    /// and that's what gets recorded under the rotated key.
    /// The W2.1 + W2.6 cost models look up rotated edges
    /// correspondingly; the writer ↔ reader pair stays
    /// coherent under each leader topology.
    ///
    /// Skips the recording when:
    ///   * `slot_rels.len() < 2` — not enough slots for a
    ///     binary inner pair (defensive).
    ///   * `output_rows == None` — unknown logical row count;
    ///     recording 0 would poison the EMA.
    ///   * `feedback_pair_from_var_order` returns `None` — the
    ///     leader rotation isn't in the locked feedback table
    ///     (conservative; never write under uncertainty).
    ///   * Any of `(rel_a, rel_b)` has missing or zero
    ///     cardinality — `populated_cards` analog from slice 5;
    ///     unknown inputs would compute a meaningless
    ///     `input_card_product`.
    ///
    /// Recording an observed-empty output (`Some(0)`) IS
    /// correct — the EMA tightens future estimates toward zero,
    /// so WCOJ becomes less likely on the same inputs next
    /// call (the kernel produced nothing useful).
    ///
    /// The triangle / 4-cycle output is a strict subset of the
    /// inner-join intermediate (the third / additional atoms
    /// further filter it). The recorded selectivity is
    /// therefore an UPPER BOUND on the true binary
    /// selectivity, which is the correct conservative direction
    /// for the cost model: it under-claims the WCOJ kernel's
    /// win rather than over-claiming.
    fn record_wcoj_feedback(
        &mut self,
        slot_rels: &[RelId],
        var_order: Option<&VariableOrder>,
        output_rows: Option<u64>,
    ) {
        if slot_rels.len() < 2 {
            return;
        }
        let Some(out_rows) = output_rows else {
            return;
        };
        // W2.6: derive the (slot 0, slot 1) feedback pair AND
        // the underlying-relation key columns from `var_order`.
        // For `var_order = None` (default config), this returns
        // the canonical W2.4 pair + keys [1]/[0] — bit-identical
        // to pre-W2.6 behavior. For Some(_), the pair may be
        // rotated (triangle non-default leaders use rotated pair
        // + [1]/[1] keys; 4-cycle is rotation-only [1]/[0]).
        let Some((rel_a, rel_b, left_keys, right_keys)) =
            feedback_pair_from_var_order(slot_rels, var_order)
        else {
            return;
        };
        let card_a = self
            .stats
            .get_relation_stats(rel_a)
            .map(|s| s.cardinality)
            .filter(|c| *c > 0);
        let card_b = self
            .stats
            .get_relation_stats(rel_b)
            .map(|s| s.cardinality)
            .filter(|c| *c > 0);
        let (Some(a), Some(b)) = (card_a, card_b) else {
            return;
        };
        let input_rows = a.saturating_mul(b);
        // `record_join_result` takes owned `Vec<usize>` for the
        // key columns (signature predates this slice).
        self.stats
            .record_join_result(rel_a, rel_b, left_keys, right_keys, input_rows, out_rows);
    }

    /// Slice 4 entry point — same gate / pattern-match / dispatch
    /// logic as `try_dispatch_wcoj_triangle`, keyed on `body`
    /// rather than `&CompiledRule`. The recursive engine calls
    /// this on the rewritten variant body (one Scan's RelId
    /// swapped to a delta RelId); the slice 1–3 wrapper above
    /// preserves the rule-keyed surface for non-recursive callers.
    pub(super) fn try_dispatch_wcoj_triangle_on_body(
        &mut self,
        body: &RirNode,
    ) -> Result<Option<CudaBuffer>> {
        #[cfg(feature = "wcoj-phase-timing")]
        let wall_start = Instant::now();
        // 1. Gate resolution. Decision tree (highest → lowest):
        //
        //    a. Hard kill switch
        //       (`wcoj_triangle_dispatch_disabled` /
        //       `XLOG_DISABLE_WCOJ_TRIANGLE=1`) → no dispatch.
        //       Beats every other flag including force.
        //    b. If `wcoj_triangle_dispatch` resolves to true
        //       (config Some(true) or env=1) → force WCOJ;
        //       classifier is bypassed entirely (mode = Force).
        //    c. Force = Some(false) → explicit off.
        //    d. Else if `wcoj_triangle_dispatch_adaptive`
        //       resolves to true (config / env / default-on) →
        //       run classifier; dispatch only when score ≥
        //       threshold (mode = Adaptive).
        //    e. Else → no dispatch.
        if wcoj_disabled(self.config.wcoj_triangle_dispatch_disabled) {
            return Ok(None);
        }
        let force_override = self.config.wcoj_triangle_dispatch;
        let force_on = wcoj_gate_enabled(force_override);
        let mode = if force_on {
            DispatchMode::Force
        } else {
            // Force-Some(false) is "explicitly off" — adaptive
            // does NOT resurrect it. Only when force is None or
            // env-default-off do we consult the adaptive gate.
            let force_explicit_off = matches!(force_override, Some(false));
            if force_explicit_off {
                return Ok(None);
            }
            let adaptive_override = self.config.wcoj_triangle_dispatch_adaptive;
            if wcoj_adaptive_enabled(adaptive_override) {
                DispatchMode::Adaptive
            } else {
                return Ok(None);
            }
        };

        // 2. Pattern-match the canonical-triangle MultiWayJoin.
        let Some(matched) = match_multiway_triangle(body) else {
            return Ok(None);
        };

        // 3. Resolve rel IDs to predicate names.
        // get_rel_name returns Option<&str> — bind to owned String
        // so the borrow doesn't conflict with later &mut self uses.
        let name_xy = match self.get_rel_name(matched.rel_xy) {
            Some(s) => s.to_string(),
            None => return Ok(None),
        };
        let name_yz = match self.get_rel_name(matched.rel_yz) {
            Some(s) => s.to_string(),
            None => return Ok(None),
        };
        let name_xz = match self.get_rel_name(matched.rel_xz) {
            Some(s) => s.to_string(),
            None => return Ok(None),
        };

        // 4. Look up input buffers + classify their key widths.
        // All three slots must be WCOJ-eligible AND share the
        // same width — mixed-width triangles fall back here so
        // the binary-join path handles them.
        let buf_xy = match self.store.get(&name_xy) {
            Some(b) => b,
            None => return Ok(None),
        };
        let buf_yz = match self.store.get(&name_yz) {
            Some(b) => b,
            None => return Ok(None),
        };
        let buf_xz = match self.store.get(&name_xz) {
            Some(b) => b,
            None => return Ok(None),
        };
        let width = match (
            classify_two_col_wcoj_width(buf_xy),
            classify_two_col_wcoj_width(buf_yz),
            classify_two_col_wcoj_width(buf_xz),
        ) {
            (Some(a), Some(b), Some(c)) if a == b && b == c => a,
            _ => return Ok(None),
        };

        // 5. Resolve the cached executor WCOJ launch stream.
        // Acquire-once / reuse-forever (mirrors
        // `CudaKernelProvider::recorded_op_stream`). Acquiring
        // per-invocation would silently drain the
        // `StreamPool` (default cap 16, grow-only) on long-
        // lived runtimes — once exhausted, subsequent
        // dispatches would silently fall back to binary-join
        // and the dispatch counter would stop incrementing.
        // Without a runtime-backed manager, the recorded WCOJ
        // primitives can't run — fall back silently.
        if self.provider.memory().runtime().is_none() {
            return Ok(None);
        }
        let launch_stream = match self.wcoj_dispatch_stream_or_init() {
            Some(s) => s,
            None => return Ok(None),
        };

        // 6. Adaptive mode only: run the classifier on the same
        // launch_stream as the eventual WCOJ pipeline. Classifier
        // failures (Ok(None) from the provider) silently fall
        // back to binary-join — classifier is optimization, not
        // correctness. A score below
        // `WCOJ_ADAPTIVE_SKEW_THRESHOLD` likewise falls back.
        // v0.6.5 slice 3: route the adaptive decision through the
        // WcojCostModel seam. Default impl is SkewClassifierCostModel,
        // which is a verbatim wrap of the v0.6.5 slice 2 inline
        // logic — dispatch counts are preserved bit-for-bit.
        #[cfg(feature = "wcoj-phase-timing")]
        let mut classifier_ms: f32 = 0.0;
        if mode == DispatchMode::Adaptive {
            #[cfg(feature = "wcoj-phase-timing")]
            let cls_start = Instant::now();
            // Slice 5: factory selects per RuntimeConfig precedence.
            // Default (slice 1–4 behavior) is `SkewClassifierCostModel`;
            // opt-in via `XLOG_WCOJ_COST_MODEL=cardinality` or
            // `RuntimeConfig::with_wcoj_cost_model(...)` selects
            // `CardinalityAwareCostModel`.
            let model = super::wcoj_cost_model::build_wcoj_cost_model(&self.config);
            let slot_rels = [matched.rel_xy, matched.rel_yz, matched.rel_xz];
            let ctx = super::wcoj_cost_model::WcojDispatchCtx {
                stats: &self.stats,
                launch_stream,
                width,
                slot_rels: &slot_rels,
            };
            let scorer = super::wcoj_cost_model::TriangleScorer {
                provider: self.provider.as_ref(),
                e_xy: buf_xy,
                e_yz: buf_yz,
                e_xz: buf_xz,
            };
            let dispatch = model.should_dispatch_triangle(&ctx, &scorer);
            #[cfg(feature = "wcoj-phase-timing")]
            {
                classifier_ms = cls_start.elapsed().as_secs_f64() as f32 * 1000.0;
            }
            if !dispatch {
                return Ok(None);
            }
        }

        // W2.1: extract var_order from the matched MultiWayJoin
        // body. None preserves slice 1/2/W2.2 default-leader
        // dispatch bit-identically.
        let var_order_opt: Option<&VariableOrder> = match body {
            RirNode::MultiWayJoin { var_order, .. } => var_order.as_ref(),
            _ => None,
        };

        // 7. Run layout + triangle. Convert any kernel error to
        // silent fallback per slice spec ("failure must not
        // corrupt store state"). The WCOJ helpers don't write
        // to the store, so an error here only loses the work
        // we just did — the binary-join path picks it up.
        #[cfg(feature = "wcoj-phase-timing")]
        let mut layout_times: [f32; 3] = [0.0; 3];
        let dispatch_result = self.run_wcoj_triangle_pipeline(
            buf_xy,
            buf_yz,
            buf_xz,
            launch_stream,
            width,
            var_order_opt,
            #[cfg(feature = "wcoj-phase-timing")]
            &mut layout_times,
        );
        match dispatch_result {
            Ok(buf) => {
                // W2.4 + W2.6 — record observed selectivity into
                // StatsManager for the cardinality cost model.
                // The (rel_a, rel_b, left_keys, right_keys) pair
                // is derived from `var_order_opt` via
                // `feedback_pair_from_var_order`:
                //   * `var_order = None` (default config) →
                //     canonical `(rel_xy, rel_yz)` keys
                //     `[1]/[0]`. Bit-identical to slice 1-5 +
                //     W2.4.
                //   * `var_order = Some(_)` (W2.1 / W2.6
                //     non-default leader) → rotated pair per
                //     the locked W2.6 step-5 feedback table.
                //     Triangle non-default leaders use rotated
                //     `(slot_rels[0], slot_rels[1])` with keys
                //     `[1]/[1]` (Z-shared edges in canonical
                //     layout join on col 1 of both rels).
                // Helper handles skip-on-missing-data and is
                // called BEFORE the counter increment so a
                // helper panic doesn't advance the counter.
                let output_rows = Self::wcoj_output_rows(&buf);
                let slot_rels = [matched.rel_xy, matched.rel_yz, matched.rel_xz];
                self.record_wcoj_feedback(&slot_rels, var_order_opt, output_rows);
                self.wcoj_triangle_dispatch_count += 1;
                #[cfg(feature = "wcoj-phase-timing")]
                {
                    let triangle_timing = self
                        .provider
                        .take_wcoj_triangle_phase_timing()
                        .unwrap_or_default();
                    let wall_ms = wall_start.elapsed().as_secs_f64() as f32 * 1000.0;
                    let timing = super::wcoj_phase_timing::WcojDispatchPhaseTiming::new(
                        classifier_ms,
                        layout_times[0],
                        layout_times[1],
                        layout_times[2],
                        triangle_timing,
                        wall_ms,
                    );
                    if let Ok(mut g) = self.last_wcoj_phase_timing.lock() {
                        *g = Some(timing);
                    }
                }
                Ok(Some(buf))
            }
            Err(_) => Ok(None),
        }
    }

    /// Inner pipeline: 3× layout construction + triangle kernel.
    /// Split out so [`try_dispatch_wcoj_triangle`] can map any
    /// error to `Ok(None)` cleanly. Branches by `width` between
    /// the parallel u32 and u64 provider entries.
    ///
    /// Under feature `wcoj-phase-timing`, fills the optional
    /// `layout_times_ms` slot with `[layout_xy, layout_yz, layout_xz]`
    /// wall times in milliseconds. The triangle's per-phase GPU
    /// times are pulled from the provider via
    /// `take_wcoj_triangle_phase_timing` after this returns.
    fn run_wcoj_triangle_pipeline(
        &self,
        buf_xy: &CudaBuffer,
        buf_yz: &CudaBuffer,
        buf_xz: &CudaBuffer,
        launch_stream: StreamId,
        width: WcojKeyWidth,
        var_order: Option<&VariableOrder>,
        #[cfg(feature = "wcoj-phase-timing")] layout_times_ms: &mut [f32; 3],
    ) -> Result<CudaBuffer> {
        // W2.1: when the cost model selected a non-default leader,
        // run the rotated/swapped path. Layout helper sees the
        // (possibly col-swapped) leader-rotated inputs; kernel
        // emits in (a, b, c) order; final projection helper remaps
        // to the canonical (X, Y, Z) head order.
        if let Some(vo) = var_order {
            return self.run_wcoj_triangle_pipeline_w21(
                buf_xy,
                buf_yz,
                buf_xz,
                launch_stream,
                width,
                vo,
            );
        }
        #[cfg(feature = "wcoj-phase-timing")]
        let mut time_layout =
            |f: &dyn Fn() -> Result<CudaBuffer>, slot: usize| -> Result<CudaBuffer> {
                let s = Instant::now();
                let r = f()?;
                layout_times_ms[slot] = s.elapsed().as_secs_f64() as f32 * 1000.0;
                Ok(r)
            };
        match width {
            WcojKeyWidth::FourByte => {
                #[cfg(feature = "wcoj-phase-timing")]
                let (layout_xy, layout_yz, layout_xz) = {
                    let xy = time_layout(
                        &|| {
                            self.provider
                                .wcoj_layout_u32_recorded(buf_xy, launch_stream)
                        },
                        0,
                    )?;
                    let yz = time_layout(
                        &|| {
                            self.provider
                                .wcoj_layout_u32_recorded(buf_yz, launch_stream)
                        },
                        1,
                    )?;
                    let xz = time_layout(
                        &|| {
                            self.provider
                                .wcoj_layout_u32_recorded(buf_xz, launch_stream)
                        },
                        2,
                    )?;
                    (xy, yz, xz)
                };
                #[cfg(not(feature = "wcoj-phase-timing"))]
                let layout_xy = self
                    .provider
                    .wcoj_layout_u32_recorded(buf_xy, launch_stream)?;
                #[cfg(not(feature = "wcoj-phase-timing"))]
                let layout_yz = self
                    .provider
                    .wcoj_layout_u32_recorded(buf_yz, launch_stream)?;
                #[cfg(not(feature = "wcoj-phase-timing"))]
                let layout_xz = self
                    .provider
                    .wcoj_layout_u32_recorded(buf_xz, launch_stream)?;
                self.provider.wcoj_triangle_u32_recorded(
                    &layout_xy,
                    &layout_yz,
                    &layout_xz,
                    launch_stream,
                )
            }
            WcojKeyWidth::EightByte => {
                #[cfg(feature = "wcoj-phase-timing")]
                let (layout_xy, layout_yz, layout_xz) = {
                    let xy = time_layout(
                        &|| {
                            self.provider
                                .wcoj_layout_u64_recorded(buf_xy, launch_stream)
                        },
                        0,
                    )?;
                    let yz = time_layout(
                        &|| {
                            self.provider
                                .wcoj_layout_u64_recorded(buf_yz, launch_stream)
                        },
                        1,
                    )?;
                    let xz = time_layout(
                        &|| {
                            self.provider
                                .wcoj_layout_u64_recorded(buf_xz, launch_stream)
                        },
                        2,
                    )?;
                    (xy, yz, xz)
                };
                #[cfg(not(feature = "wcoj-phase-timing"))]
                let layout_xy = self
                    .provider
                    .wcoj_layout_u64_recorded(buf_xy, launch_stream)?;
                #[cfg(not(feature = "wcoj-phase-timing"))]
                let layout_yz = self
                    .provider
                    .wcoj_layout_u64_recorded(buf_yz, launch_stream)?;
                #[cfg(not(feature = "wcoj-phase-timing"))]
                let layout_xz = self
                    .provider
                    .wcoj_layout_u64_recorded(buf_xz, launch_stream)?;
                self.provider.wcoj_triangle_u64_recorded(
                    &layout_xy,
                    &layout_yz,
                    &layout_xz,
                    launch_stream,
                )
            }
        }
    }

    /// W2.1 — pipeline for non-default leaders. Uses the locked
    /// permutation tables on `var_order` to:
    /// 1. Rotate canonical inputs `[buf_xy, buf_yz, buf_xz]` so the
    ///    leader sits at slot 0.
    /// 2. Apply col-swap (via `wcoj_project_2col_swap_recorded`) to
    ///    any non-leader slot whose `LookupPerm.swap_cols` is true.
    ///    Triangle e_yz / e_xz leaders need swaps; 4-cycle is
    ///    rotation-only (no swap entries).
    /// 3. Run `wcoj_layout_*_recorded` on each slot input.
    /// 4. Run `wcoj_triangle_*_recorded`. Kernel emits 3 columns
    ///    in leader's `(a, b, c)` order.
    /// 5. Apply `wcoj_project_output_columns_recorded` with
    ///    `var_order.kernel_output_cols` to re-permute the
    ///    kernel-direct output into the canonical head order
    ///    `(X, Y, Z)`.
    ///
    /// Phase timing is intentionally NOT instrumented on this
    /// path — perf validation of the W2.1 threshold is W5.2 work
    /// (per the W2.1 plan §"Risk & Open Questions / Q1").
    fn run_wcoj_triangle_pipeline_w21(
        &self,
        buf_xy: &CudaBuffer,
        buf_yz: &CudaBuffer,
        buf_xz: &CudaBuffer,
        launch_stream: StreamId,
        width: WcojKeyWidth,
        var_order: &VariableOrder,
    ) -> Result<CudaBuffer> {
        let canonical: [&CudaBuffer; 3] = [buf_xy, buf_yz, buf_xz];
        let slot_inputs = self.prepare_leader_inputs(&canonical, var_order, launch_stream)?;
        if slot_inputs.len() != 3 {
            return Err(xlog_core::XlogError::Kernel(
                "run_wcoj_triangle_pipeline_w21: prepare_leader_inputs must return 3 slots"
                    .to_string(),
            ));
        }

        // Build the canonical (X, Y, Z) head schema from the
        // canonical promoter inputs (NOT the rotated kernel
        // inputs). The kernel will emit in (a, b, c) order under
        // the rotated leader; the final projection helper maps
        // back to head order using kernel_output_cols.
        let head_schema = build_triangle_head_schema(buf_xy, buf_yz)?;
        let perm = perm_indices_from_kernel_output_cols(&var_order.kernel_output_cols)?;

        let kernel_out: CudaBuffer = match width {
            WcojKeyWidth::FourByte => {
                let l0 = self
                    .provider
                    .wcoj_layout_u32_recorded(&slot_inputs[0], launch_stream)?;
                let l1 = self
                    .provider
                    .wcoj_layout_u32_recorded(&slot_inputs[1], launch_stream)?;
                let l2 = self
                    .provider
                    .wcoj_layout_u32_recorded(&slot_inputs[2], launch_stream)?;
                self.provider
                    .wcoj_triangle_u32_recorded(&l0, &l1, &l2, launch_stream)?
            }
            WcojKeyWidth::EightByte => {
                let l0 = self
                    .provider
                    .wcoj_layout_u64_recorded(&slot_inputs[0], launch_stream)?;
                let l1 = self
                    .provider
                    .wcoj_layout_u64_recorded(&slot_inputs[1], launch_stream)?;
                let l2 = self
                    .provider
                    .wcoj_layout_u64_recorded(&slot_inputs[2], launch_stream)?;
                self.provider
                    .wcoj_triangle_u64_recorded(&l0, &l1, &l2, launch_stream)?
            }
        };

        self.provider.wcoj_project_output_columns_recorded(
            &kernel_out,
            &perm,
            head_schema,
            launch_stream,
        )
    }

    /// Number of times the WCOJ triangle hook produced a result
    /// and the executor installed it. Used by tests to assert
    /// that the WCOJ path actually ran (vs. silently falling
    /// back to the existing binary-join path with the same
    /// answer).
    pub fn wcoj_triangle_dispatch_count(&self) -> u64 {
        self.wcoj_triangle_dispatch_count
    }

    /// v0.6.5 slice 2 — count of times the WCOJ 4-cycle hook
    /// produced a result and the executor installed it. Tracked
    /// separately from triangle so tests can pin which shape
    /// dispatched.
    pub fn wcoj_4cycle_dispatch_count(&self) -> u64 {
        self.wcoj_4cycle_dispatch_count
    }

    /// v0.6.5 slice 2 — try to dispatch a non-recursive rule
    /// through the GPU 4-cycle WCOJ kernel.
    ///
    /// Decision tree (highest → lowest):
    ///   1. Hard kill switch (`wcoj_4cycle_dispatch_disabled` /
    ///      `XLOG_DISABLE_WCOJ_4CYCLE=1`) → no dispatch.
    ///   2. Force gate (`wcoj_4cycle_dispatch=Some(true)` /
    ///      `XLOG_USE_WCOJ_4CYCLE=1`) → kernel runs.
    ///   3. Force-Some(false) → no dispatch.
    ///   4. Adaptive opt-in (config / env, default off) →
    ///      classifier integration lands in slice 2 step 9;
    ///      until then, the adaptive branch returns Ok(None)
    ///      (no dispatch). Per the slice 2 plan, ship force +
    ///      adaptive together; this step just plumbs the gates.
    ///
    /// Returns `Ok(Some(buffer))` on dispatch; `Ok(None)`
    /// silently otherwise. The caller installs the buffer or
    /// descends into `MultiWayJoin.fallback`.
    pub(super) fn try_dispatch_wcoj_4cycle(
        &mut self,
        rule: &CompiledRule,
    ) -> Result<Option<CudaBuffer>> {
        // Slice 4: body-keyed entry. Rule-keyed callers stay
        // byte-identical via this thin wrapper.
        self.try_dispatch_wcoj_4cycle_on_body(&rule.body)
    }

    /// Slice 4 entry point — same gate / pattern-match / dispatch
    /// logic as `try_dispatch_wcoj_4cycle`, keyed on `body` rather
    /// than `&CompiledRule`. See
    /// `try_dispatch_wcoj_triangle_on_body` for the rationale.
    pub(super) fn try_dispatch_wcoj_4cycle_on_body(
        &mut self,
        body: &RirNode,
    ) -> Result<Option<CudaBuffer>> {
        // 1. Kill switch.
        if wcoj_4cycle_disabled(self.config.wcoj_4cycle_dispatch_disabled) {
            return Ok(None);
        }
        // 2. Force gate.
        let force_override = self.config.wcoj_4cycle_dispatch;
        let force_on = wcoj_4cycle_gate_enabled(force_override);
        let mode = if force_on {
            DispatchMode::Force
        } else {
            // Force-Some(false) is explicit off — adaptive does
            // NOT resurrect it.
            if matches!(force_override, Some(false)) {
                return Ok(None);
            }
            let adaptive_override = self.config.wcoj_4cycle_dispatch_adaptive;
            if wcoj_4cycle_adaptive_enabled(adaptive_override) {
                DispatchMode::Adaptive
            } else {
                return Ok(None);
            }
        };

        // 3. Match the canonical 4-cycle MultiWayJoin.
        let Some(matched) = match_multiway_4cycle(body) else {
            return Ok(None);
        };

        // 4. Resolve rel IDs to predicate names.
        let name_e1 = match self.get_rel_name(matched.rel_e1) {
            Some(s) => s.to_string(),
            None => return Ok(None),
        };
        let name_e2 = match self.get_rel_name(matched.rel_e2) {
            Some(s) => s.to_string(),
            None => return Ok(None),
        };
        let name_e3 = match self.get_rel_name(matched.rel_e3) {
            Some(s) => s.to_string(),
            None => return Ok(None),
        };
        let name_e4 = match self.get_rel_name(matched.rel_e4) {
            Some(s) => s.to_string(),
            None => return Ok(None),
        };

        // 5. Look up input buffers + classify their key widths.
        // All four slots must share the same width.
        let buf_e1 = match self.store.get(&name_e1) {
            Some(b) => b,
            None => return Ok(None),
        };
        let buf_e2 = match self.store.get(&name_e2) {
            Some(b) => b,
            None => return Ok(None),
        };
        let buf_e3 = match self.store.get(&name_e3) {
            Some(b) => b,
            None => return Ok(None),
        };
        let buf_e4 = match self.store.get(&name_e4) {
            Some(b) => b,
            None => return Ok(None),
        };
        let width = match (
            classify_two_col_wcoj_width(buf_e1),
            classify_two_col_wcoj_width(buf_e2),
            classify_two_col_wcoj_width(buf_e3),
            classify_two_col_wcoj_width(buf_e4),
        ) {
            (Some(a), Some(b), Some(c), Some(d)) if a == b && b == c && c == d => a,
            _ => return Ok(None),
        };

        // 6. Resolve the cached WCOJ launch stream (shared with
        // triangle dispatch — slice 2's stream rename made this
        // shape-agnostic).
        if self.provider.memory().runtime().is_none() {
            return Ok(None);
        }
        let launch_stream = match self.wcoj_dispatch_stream_or_init() {
            Some(s) => s,
            None => return Ok(None),
        };

        // 7. v0.6.5 slice 3: route the adaptive decision through
        // the WcojCostModel seam. Default impl is
        // SkewClassifierCostModel — verbatim wrap of the v0.6.5
        // slice 2 inline logic; dispatch counts preserved.
        if mode == DispatchMode::Adaptive {
            // Slice 5: factory selects per RuntimeConfig precedence.
            let model = super::wcoj_cost_model::build_wcoj_cost_model(&self.config);
            let slot_rels = [
                matched.rel_e1,
                matched.rel_e2,
                matched.rel_e3,
                matched.rel_e4,
            ];
            let ctx = super::wcoj_cost_model::WcojDispatchCtx {
                stats: &self.stats,
                launch_stream,
                width,
                slot_rels: &slot_rels,
            };
            let scorer = super::wcoj_cost_model::Cycle4Scorer {
                provider: self.provider.as_ref(),
                e1: buf_e1,
                e2: buf_e2,
                e3: buf_e3,
                e4: buf_e4,
            };
            let dispatch = model.should_dispatch_4cycle(&ctx, &scorer);
            if !dispatch {
                return Ok(None);
            }
        }

        // W2.1: extract var_order. None preserves slice 2/W2.2
        // default-leader dispatch bit-identically.
        let var_order_opt: Option<&VariableOrder> = match body {
            RirNode::MultiWayJoin { var_order, .. } => var_order.as_ref(),
            _ => None,
        };

        // 8. Run layout (4× per slot) + 4-cycle kernel. Failure
        // → silent fallback per slice contract.
        let dispatch_result = self.run_wcoj_4cycle_pipeline(
            buf_e1,
            buf_e2,
            buf_e3,
            buf_e4,
            launch_stream,
            width,
            var_order_opt,
        );
        match dispatch_result {
            Ok(buf) => {
                // W2.4 + W2.6 — record observed selectivity.
                // The (rel_a, rel_b, left_keys, right_keys)
                // pair is derived from `var_order_opt` via
                // `feedback_pair_from_var_order`:
                //   * `var_order = None` (default config) →
                //     canonical `(rel_e1, rel_e2)` keys
                //     `[1]/[0]`. Bit-identical to slice 1-5 +
                //     W2.4.
                //   * `var_order = Some(_)` (W2.1 / W2.6
                //     non-default leader) → rotated pair from
                //     the locked feedback table. 4-cycle is
                //     rotation-only (every cycle edge is
                //     `[1]/[0]` in canonical layout), so the
                //     keys stay `[1]/[0]` while the pair
                //     itself rotates.
                let output_rows = Self::wcoj_output_rows(&buf);
                let slot_rels = [
                    matched.rel_e1,
                    matched.rel_e2,
                    matched.rel_e3,
                    matched.rel_e4,
                ];
                self.record_wcoj_feedback(&slot_rels, var_order_opt, output_rows);
                self.wcoj_4cycle_dispatch_count += 1;
                Ok(Some(buf))
            }
            Err(_) => Ok(None),
        }
    }

    /// Inner pipeline for 4-cycle: 4× layout construction + kernel.
    fn run_wcoj_4cycle_pipeline(
        &self,
        buf_e1: &CudaBuffer,
        buf_e2: &CudaBuffer,
        buf_e3: &CudaBuffer,
        buf_e4: &CudaBuffer,
        launch_stream: StreamId,
        width: WcojKeyWidth,
        var_order: Option<&VariableOrder>,
    ) -> Result<CudaBuffer> {
        if let Some(vo) = var_order {
            return self.run_wcoj_4cycle_pipeline_w21(
                buf_e1,
                buf_e2,
                buf_e3,
                buf_e4,
                launch_stream,
                width,
                vo,
            );
        }
        match width {
            WcojKeyWidth::FourByte => {
                let layout_e1 = self
                    .provider
                    .wcoj_layout_u32_recorded(buf_e1, launch_stream)?;
                let layout_e2 = self
                    .provider
                    .wcoj_layout_u32_recorded(buf_e2, launch_stream)?;
                let layout_e3 = self
                    .provider
                    .wcoj_layout_u32_recorded(buf_e3, launch_stream)?;
                let layout_e4 = self
                    .provider
                    .wcoj_layout_u32_recorded(buf_e4, launch_stream)?;
                self.provider.wcoj_4cycle_u32_recorded(
                    &layout_e1,
                    &layout_e2,
                    &layout_e3,
                    &layout_e4,
                    launch_stream,
                )
            }
            WcojKeyWidth::EightByte => {
                let layout_e1 = self
                    .provider
                    .wcoj_layout_u64_recorded(buf_e1, launch_stream)?;
                let layout_e2 = self
                    .provider
                    .wcoj_layout_u64_recorded(buf_e2, launch_stream)?;
                let layout_e3 = self
                    .provider
                    .wcoj_layout_u64_recorded(buf_e3, launch_stream)?;
                let layout_e4 = self
                    .provider
                    .wcoj_layout_u64_recorded(buf_e4, launch_stream)?;
                self.provider.wcoj_4cycle_u64_recorded(
                    &layout_e1,
                    &layout_e2,
                    &layout_e3,
                    &layout_e4,
                    launch_stream,
                )
            }
        }
    }

    /// W2.1 — pipeline for non-default 4-cycle leaders. All
    /// 4-cycle leaders are rotation-only (no col-swap entries
    /// in `lookup_perms`); kernel emits in `(a, b, c, d)` order
    /// per the rotated leader; final projection helper remaps
    /// to canonical `(W, X, Y, Z)` head order.
    fn run_wcoj_4cycle_pipeline_w21(
        &self,
        buf_e1: &CudaBuffer,
        buf_e2: &CudaBuffer,
        buf_e3: &CudaBuffer,
        buf_e4: &CudaBuffer,
        launch_stream: StreamId,
        width: WcojKeyWidth,
        var_order: &VariableOrder,
    ) -> Result<CudaBuffer> {
        let canonical: [&CudaBuffer; 4] = [buf_e1, buf_e2, buf_e3, buf_e4];
        let slot_inputs = self.prepare_leader_inputs(&canonical, var_order, launch_stream)?;
        if slot_inputs.len() != 4 {
            return Err(xlog_core::XlogError::Kernel(
                "run_wcoj_4cycle_pipeline_w21: prepare_leader_inputs must return 4 slots"
                    .to_string(),
            ));
        }

        let head_schema = build_4cycle_head_schema(buf_e1, buf_e2, buf_e3)?;
        let perm = perm_indices_from_kernel_output_cols(&var_order.kernel_output_cols)?;

        let kernel_out: CudaBuffer = match width {
            WcojKeyWidth::FourByte => {
                let l0 = self
                    .provider
                    .wcoj_layout_u32_recorded(&slot_inputs[0], launch_stream)?;
                let l1 = self
                    .provider
                    .wcoj_layout_u32_recorded(&slot_inputs[1], launch_stream)?;
                let l2 = self
                    .provider
                    .wcoj_layout_u32_recorded(&slot_inputs[2], launch_stream)?;
                let l3 = self
                    .provider
                    .wcoj_layout_u32_recorded(&slot_inputs[3], launch_stream)?;
                self.provider
                    .wcoj_4cycle_u32_recorded(&l0, &l1, &l2, &l3, launch_stream)?
            }
            WcojKeyWidth::EightByte => {
                let l0 = self
                    .provider
                    .wcoj_layout_u64_recorded(&slot_inputs[0], launch_stream)?;
                let l1 = self
                    .provider
                    .wcoj_layout_u64_recorded(&slot_inputs[1], launch_stream)?;
                let l2 = self
                    .provider
                    .wcoj_layout_u64_recorded(&slot_inputs[2], launch_stream)?;
                let l3 = self
                    .provider
                    .wcoj_layout_u64_recorded(&slot_inputs[3], launch_stream)?;
                self.provider
                    .wcoj_4cycle_u64_recorded(&l0, &l1, &l2, &l3, launch_stream)?
            }
        };

        self.provider.wcoj_project_output_columns_recorded(
            &kernel_out,
            &perm,
            head_schema,
            launch_stream,
        )
    }

    /// W2.1 — produce **owned, materialized** kernel slot inputs
    /// from a canonical-order input array and a `VariableOrder`.
    ///
    /// **Public** runtime helper. Production callers are
    /// `run_wcoj_*_pipeline_w21` (this module); the W2.1 plan
    /// §"Part B" runtime tests in
    /// `crates/xlog-runtime/tests/test_w21_part_b.rs` invoke it
    /// directly to assert per-slot schema + content against a CPU
    /// reference. Public visibility is intentional: there is no
    /// other reasonable seam for tests to inspect rotation +
    /// col-swap behavior, and the helper has well-defined
    /// owned-buffer semantics that external callers can rely on.
    ///
    /// Returns a `Vec<CudaBuffer>` of length `canonical.len()` (3
    /// for triangle, 4 for 4-cycle). Slot 0 is the leader; slots
    /// 1.. follow `var_order.lookup_perms[i].input_idx` mapping.
    /// Triangle non-default leaders may col-swap selected slots
    /// per the locked permutation table; 4-cycle is rotation-only
    /// and rejects swap requests with a kernel error.
    ///
    /// Each returned `CudaBuffer` is owned: swapped slots are
    /// DtoD-copied via `wcoj_project_2col_swap_recorded`; non-
    /// swapped slots use the double-swap clone path below to give
    /// every slot a uniform owned-buffer return type.
    ///
    /// **Lifetime contract**: returned buffers are independent of
    /// `canonical[*]`. Callers may pass references through to
    /// `wcoj_layout_*_recorded` without aliasing concerns.
    pub fn prepare_leader_inputs(
        &self,
        canonical: &[&CudaBuffer],
        var_order: &VariableOrder,
        launch_stream: StreamId,
    ) -> Result<Vec<CudaBuffer>> {
        let n = canonical.len();
        if !(n == 3 || n == 4) {
            return Err(xlog_core::XlogError::Kernel(format!(
                "prepare_leader_inputs: canonical inputs must be 3 (triangle) or 4 (4-cycle), got {n}"
            )));
        }
        let leader_idx = var_order.leader_idx as usize;
        if leader_idx >= n {
            return Err(xlog_core::XlogError::Kernel(format!(
                "prepare_leader_inputs: leader_idx {leader_idx} out of range for arity {n}"
            )));
        }
        if var_order.lookup_perms.len() != n - 1 {
            return Err(xlog_core::XlogError::Kernel(format!(
                "prepare_leader_inputs: lookup_perms.len() = {} must equal {} (arity - 1)",
                var_order.lookup_perms.len(),
                n - 1
            )));
        }
        // 4-cycle defense: no col-swaps allowed (locked table).
        if n == 4 {
            for lp in &var_order.lookup_perms {
                if lp.swap_cols {
                    return Err(xlog_core::XlogError::Kernel(
                        "prepare_leader_inputs: 4-cycle does not support col-swaps".to_string(),
                    ));
                }
            }
        }

        // Slot 0: clone the leader via the swap helper called twice
        // (cancels out → owned pass-through). The simpler path for
        // production is just passing `canonical[leader_idx]` by
        // reference, but since the production callers consume the
        // returned `Vec<CudaBuffer>` by index, we materialize an
        // owned copy. Triangle leaders never have swap_cols on
        // their own slot; we use `wcoj_project_2col_swap_recorded`
        // twice to produce an owned copy with identical layout.
        //
        // For clarity and to avoid the extra DtoD: leader slot 0 is
        // produced by single swap-twice, lookups by either single
        // swap (when swap_cols) or single swap-twice (when not).
        //
        // Cost: one extra DtoD copy per slot vs. the previous
        // inline-references implementation. The W2.1 path is opt-in
        // and the DtoD overhead is small relative to the layout +
        // kernel cost; perf validation is W5.2 anyway.
        let mut slots: Vec<CudaBuffer> = Vec::with_capacity(n);
        // Slot 0 = leader, no swap.
        slots.push(self.clone_buffer_via_swap(canonical[leader_idx], launch_stream)?);
        for lp in &var_order.lookup_perms {
            let src = canonical[lp.input_idx as usize];
            let buf = if lp.swap_cols {
                self.provider
                    .wcoj_project_2col_swap_recorded(src, launch_stream)?
            } else {
                self.clone_buffer_via_swap(src, launch_stream)?
            };
            slots.push(buf);
        }
        Ok(slots)
    }

    /// Clone a 2-col `CudaBuffer` via a double-swap through the
    /// existing recorded helper. Two swaps cancel — the result is a
    /// fresh owned buffer with the same column order, schema, and
    /// content as `src`. Used by `prepare_leader_inputs` to give
    /// every slot a uniform owned-buffer return type.
    fn clone_buffer_via_swap(
        &self,
        src: &CudaBuffer,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let once = self
            .provider
            .wcoj_project_2col_swap_recorded(src, launch_stream)?;
        self.provider
            .wcoj_project_2col_swap_recorded(&once, launch_stream)
    }

    /// Resolve the cached WCOJ launch stream, lazily initializing
    /// it on first call by acquiring one stream from the runtime
    /// pool. Subsequent calls reuse the same stream — mirrors
    /// [`xlog_cuda::CudaKernelProvider::recorded_op_stream`]
    /// (provider/mod.rs).
    ///
    /// **Shared across WCOJ shapes** (v0.6.5 slice 2): triangle
    /// and 4-cycle dispatch both go through this resolver and
    /// reuse the same stream. Renamed from
    /// `wcoj_triangle_stream_or_init` when 4-cycle dispatch
    /// landed.
    ///
    /// Returns `None` only when (a) the manager has no runtime,
    /// or (b) the very first acquisition fails (pool already
    /// at cap from other consumers). After that first success
    /// the cached id keeps resolving for the executor's lifetime.
    pub fn wcoj_dispatch_stream_or_init(&self) -> Option<StreamId> {
        if let Some(s) = self.wcoj_dispatch_stream.get() {
            return Some(*s);
        }
        let runtime = self.provider.memory().runtime()?;
        let stream = runtime.stream_pool().acquire().ok()?;
        let _ = self.wcoj_dispatch_stream.set(stream);
        self.wcoj_dispatch_stream.get().copied()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, OnceLock};

    use super::{
        match_multiway_triangle, wcoj_adaptive_enabled, wcoj_disabled, wcoj_gate_enabled,
        ENV_DISABLE_WCOJ_TRIANGLE, ENV_USE_WCOJ_TRIANGLE_ADAPTIVE, ENV_USE_WCOJ_TRIANGLE_U32,
    };
    use xlog_core::RelId;
    use xlog_ir::rir::ProjectExpr;
    use xlog_ir::RirNode;

    fn canonical_multiway() -> RirNode {
        RirNode::MultiWayJoin {
            inputs: vec![
                RirNode::Scan { rel: RelId(1) },
                RirNode::Scan { rel: RelId(2) },
                RirNode::Scan { rel: RelId(3) },
            ],
            slot_vars: vec![
                vec![Some(0u32), Some(1)],
                vec![Some(1u32), Some(2)],
                vec![Some(0u32), Some(2)],
            ],
            output_columns: vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(3),
            ],
            fallback: Box::new(RirNode::Unit),
            var_order: None,
        }
    }

    #[test]
    fn match_canonical_returns_three_rels() {
        let node = canonical_multiway();
        let m = match_multiway_triangle(&node).expect("must match canonical triangle");
        assert_eq!(m.rel_xy, RelId(1));
        assert_eq!(m.rel_yz, RelId(2));
        assert_eq!(m.rel_xz, RelId(3));
    }

    #[test]
    fn match_rejects_non_multiway_body() {
        let node = RirNode::Scan { rel: RelId(1) };
        assert!(match_multiway_triangle(&node).is_none());
    }

    #[test]
    fn match_rejects_rotated_output_columns() {
        let mut node = canonical_multiway();
        if let RirNode::MultiWayJoin { output_columns, .. } = &mut node {
            *output_columns = vec![
                ProjectExpr::Column(1),
                ProjectExpr::Column(0),
                ProjectExpr::Column(3),
            ];
        }
        assert!(match_multiway_triangle(&node).is_none());
    }

    /// W2.2: triangle with Z-shared output_columns layout
    /// `[Column(0), Column(2), Column(3)]` must match. The
    /// matcher's output-column relaxation in W2.2 accepts both
    /// `[0, 1, 3]` (Y/X-shared) and `[0, 2, 3]` (Z-shared).
    #[test]
    fn match_accepts_w22_z_shared_triangle_output_columns() {
        let mut node = canonical_multiway();
        if let RirNode::MultiWayJoin { output_columns, .. } = &mut node {
            *output_columns = vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(2),
                ProjectExpr::Column(3),
            ];
        }
        let m = match_multiway_triangle(&node)
            .expect("W2.2 matcher must accept the Z-shared output-column layout");
        assert_eq!(m.rel_xy, RelId(1));
        assert_eq!(m.rel_yz, RelId(2));
        assert_eq!(m.rel_xz, RelId(3));
    }

    /// W2.2: triangle output_columns `[Column(0), Column(3), Column(3)]`
    /// MUST be rejected — second col must be 1 or 2, not 3.
    #[test]
    fn match_rejects_invalid_w22_triangle_output_columns() {
        let mut node = canonical_multiway();
        if let RirNode::MultiWayJoin { output_columns, .. } = &mut node {
            *output_columns = vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(3),
                ProjectExpr::Column(3),
            ];
        }
        assert!(match_multiway_triangle(&node).is_none());
    }

    #[test]
    fn match_rejects_arity_mismatched_output_columns() {
        let mut node = canonical_multiway();
        if let RirNode::MultiWayJoin { output_columns, .. } = &mut node {
            *output_columns = vec![ProjectExpr::Column(0), ProjectExpr::Column(1)];
        }
        assert!(match_multiway_triangle(&node).is_none());
    }

    #[test]
    fn match_rejects_malformed_slot_vars() {
        // [[A,B],[B,C],[A,B]] — last slot is wrong (should be [A,C]).
        let mut node = canonical_multiway();
        if let RirNode::MultiWayJoin { slot_vars, .. } = &mut node {
            *slot_vars = vec![
                vec![Some(0u32), Some(1)],
                vec![Some(1u32), Some(2)],
                vec![Some(0u32), Some(1)],
            ];
        }
        assert!(match_multiway_triangle(&node).is_none());
    }

    #[test]
    fn match_rejects_repeated_var_in_slot() {
        let mut node = canonical_multiway();
        if let RirNode::MultiWayJoin { slot_vars, .. } = &mut node {
            // [[A, A], …] — repeated var in slot 0.
            *slot_vars = vec![
                vec![Some(0u32), Some(0)],
                vec![Some(1u32), Some(2)],
                vec![Some(0u32), Some(2)],
            ];
        }
        assert!(match_multiway_triangle(&node).is_none());
    }

    #[test]
    fn match_rejects_non_scan_input() {
        let mut node = canonical_multiway();
        if let RirNode::MultiWayJoin { inputs, .. } = &mut node {
            inputs[0] = RirNode::Unit;
        }
        assert!(match_multiway_triangle(&node).is_none());
    }

    #[test]
    fn match_rejects_input_arity_mismatch() {
        let mut node = canonical_multiway();
        if let RirNode::MultiWayJoin { inputs, .. } = &mut node {
            inputs.pop();
        }
        assert!(match_multiway_triangle(&node).is_none());
    }

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvSnapshot {
        force: Option<String>,
        adaptive: Option<String>,
        disable: Option<String>,
    }

    impl EnvSnapshot {
        fn capture_and_clear() -> Self {
            let snapshot = Self {
                force: std::env::var(ENV_USE_WCOJ_TRIANGLE_U32).ok(),
                adaptive: std::env::var(ENV_USE_WCOJ_TRIANGLE_ADAPTIVE).ok(),
                disable: std::env::var(ENV_DISABLE_WCOJ_TRIANGLE).ok(),
            };

            // SAFETY: The caller holds `env_lock`, serializing mutation of
            // these process-global WCOJ env vars.
            unsafe {
                std::env::remove_var(ENV_USE_WCOJ_TRIANGLE_U32);
                std::env::remove_var(ENV_USE_WCOJ_TRIANGLE_ADAPTIVE);
                std::env::remove_var(ENV_DISABLE_WCOJ_TRIANGLE);
            }

            snapshot
        }
    }

    impl Drop for EnvSnapshot {
        fn drop(&mut self) {
            // SAFETY: The snapshot is dropped before `env_lock` is released,
            // so restoration is serialized even if the test body panics.
            unsafe {
                match self.force.take() {
                    Some(v) => std::env::set_var(ENV_USE_WCOJ_TRIANGLE_U32, v),
                    None => std::env::remove_var(ENV_USE_WCOJ_TRIANGLE_U32),
                }
                match self.adaptive.take() {
                    Some(v) => std::env::set_var(ENV_USE_WCOJ_TRIANGLE_ADAPTIVE, v),
                    None => std::env::remove_var(ENV_USE_WCOJ_TRIANGLE_ADAPTIVE),
                }
                match self.disable.take() {
                    Some(v) => std::env::set_var(ENV_DISABLE_WCOJ_TRIANGLE, v),
                    None => std::env::remove_var(ENV_DISABLE_WCOJ_TRIANGLE),
                }
            }
        }
    }

    fn with_wcoj_env<R>(f: impl FnOnce() -> R) -> R {
        let _guard = env_lock().lock().expect("WCOJ env lock poisoned");
        let _snapshot = EnvSnapshot::capture_and_clear();
        f()
    }

    fn set_env(name: &str, value: &str) {
        // SAFETY: Callers are inside `with_wcoj_env`, which serializes and
        // restores these process-global WCOJ env vars.
        unsafe {
            std::env::set_var(name, value);
        }
    }

    #[test]
    fn adaptive_resolver_defaults_on_when_env_unset() {
        with_wcoj_env(|| {
            assert!(wcoj_adaptive_enabled(None));
            assert!(wcoj_adaptive_enabled(Some(true)));
            assert!(!wcoj_adaptive_enabled(Some(false)));
        });
    }

    #[test]
    fn adaptive_resolver_env_can_disable_or_enable() {
        with_wcoj_env(|| {
            set_env(ENV_USE_WCOJ_TRIANGLE_ADAPTIVE, "0");
            assert!(!wcoj_adaptive_enabled(None));

            set_env(ENV_USE_WCOJ_TRIANGLE_ADAPTIVE, "false");
            assert!(!wcoj_adaptive_enabled(None));

            set_env(ENV_USE_WCOJ_TRIANGLE_ADAPTIVE, "true");
            assert!(wcoj_adaptive_enabled(None));

            set_env(ENV_USE_WCOJ_TRIANGLE_ADAPTIVE, "1");
            assert!(wcoj_adaptive_enabled(None));
        });
    }

    #[test]
    fn config_overrides_adaptive_env() {
        with_wcoj_env(|| {
            set_env(ENV_USE_WCOJ_TRIANGLE_ADAPTIVE, "0");
            assert!(wcoj_adaptive_enabled(Some(true)));

            set_env(ENV_USE_WCOJ_TRIANGLE_ADAPTIVE, "1");
            assert!(!wcoj_adaptive_enabled(Some(false)));
        });
    }

    #[test]
    fn kill_switch_resolver_honors_env_and_config_precedence() {
        with_wcoj_env(|| {
            assert!(!wcoj_disabled(None));

            set_env(ENV_DISABLE_WCOJ_TRIANGLE, "1");
            assert!(wcoj_disabled(None));
            assert!(!wcoj_disabled(Some(false)));

            set_env(ENV_DISABLE_WCOJ_TRIANGLE, "0");
            assert!(!wcoj_disabled(None));
            assert!(wcoj_disabled(Some(true)));
        });
    }

    #[test]
    fn force_resolver_config_still_overrides_env() {
        with_wcoj_env(|| {
            set_env(ENV_USE_WCOJ_TRIANGLE_U32, "1");
            assert!(wcoj_gate_enabled(None));
            assert!(!wcoj_gate_enabled(Some(false)));

            set_env(ENV_USE_WCOJ_TRIANGLE_U32, "0");
            assert!(!wcoj_gate_enabled(None));
            assert!(wcoj_gate_enabled(Some(true)));
        });
    }

    // -------------------------------------------------------------
    // v0.6.5 slice 2 — 4-cycle env-resolver + matcher tests.
    // -------------------------------------------------------------

    use super::{
        match_multiway_4cycle, wcoj_4cycle_adaptive_enabled, wcoj_4cycle_disabled,
        wcoj_4cycle_gate_enabled, ENV_DISABLE_WCOJ_4CYCLE, ENV_USE_WCOJ_4CYCLE,
        ENV_USE_WCOJ_4CYCLE_ADAPTIVE,
    };

    struct EnvSnapshot4Cycle {
        force: Option<String>,
        adaptive: Option<String>,
        disable: Option<String>,
    }

    impl EnvSnapshot4Cycle {
        fn capture_and_clear() -> Self {
            let snap = Self {
                force: std::env::var(ENV_USE_WCOJ_4CYCLE).ok(),
                adaptive: std::env::var(ENV_USE_WCOJ_4CYCLE_ADAPTIVE).ok(),
                disable: std::env::var(ENV_DISABLE_WCOJ_4CYCLE).ok(),
            };
            // SAFETY: caller holds env_lock.
            unsafe {
                std::env::remove_var(ENV_USE_WCOJ_4CYCLE);
                std::env::remove_var(ENV_USE_WCOJ_4CYCLE_ADAPTIVE);
                std::env::remove_var(ENV_DISABLE_WCOJ_4CYCLE);
            }
            snap
        }
    }

    impl Drop for EnvSnapshot4Cycle {
        fn drop(&mut self) {
            // SAFETY: caller holds env_lock.
            unsafe {
                match self.force.take() {
                    Some(v) => std::env::set_var(ENV_USE_WCOJ_4CYCLE, v),
                    None => std::env::remove_var(ENV_USE_WCOJ_4CYCLE),
                }
                match self.adaptive.take() {
                    Some(v) => std::env::set_var(ENV_USE_WCOJ_4CYCLE_ADAPTIVE, v),
                    None => std::env::remove_var(ENV_USE_WCOJ_4CYCLE_ADAPTIVE),
                }
                match self.disable.take() {
                    Some(v) => std::env::set_var(ENV_DISABLE_WCOJ_4CYCLE, v),
                    None => std::env::remove_var(ENV_DISABLE_WCOJ_4CYCLE),
                }
            }
        }
    }

    fn with_4cycle_env<R>(f: impl FnOnce() -> R) -> R {
        let _guard = env_lock().lock().expect("4-cycle env lock poisoned");
        let _snap = EnvSnapshot4Cycle::capture_and_clear();
        f()
    }

    #[test]
    fn force_4cycle_resolver_defaults_off_when_env_unset() {
        with_4cycle_env(|| {
            assert!(!wcoj_4cycle_gate_enabled(None));
            assert!(wcoj_4cycle_gate_enabled(Some(true)));
            assert!(!wcoj_4cycle_gate_enabled(Some(false)));
        });
    }

    #[test]
    fn force_4cycle_resolver_env_can_enable() {
        with_4cycle_env(|| {
            set_env(ENV_USE_WCOJ_4CYCLE, "1");
            assert!(wcoj_4cycle_gate_enabled(None));
            set_env(ENV_USE_WCOJ_4CYCLE, "true");
            assert!(wcoj_4cycle_gate_enabled(None));
            set_env(ENV_USE_WCOJ_4CYCLE, "0");
            assert!(!wcoj_4cycle_gate_enabled(None));
        });
    }

    /// **Locks the slice 2 contract**: 4-cycle adaptive opt-in
    /// defaults OFF, unlike triangle's default-on. If a future
    /// slice flips this, that change must update this test
    /// explicitly with bench evidence.
    #[test]
    fn adaptive_4cycle_resolver_defaults_off_when_env_unset() {
        with_4cycle_env(|| {
            assert!(
                !wcoj_4cycle_adaptive_enabled(None),
                "4-cycle adaptive must be OPT-IN by default (unlike triangle's default-on)"
            );
            assert!(wcoj_4cycle_adaptive_enabled(Some(true)));
            assert!(!wcoj_4cycle_adaptive_enabled(Some(false)));
        });
    }

    #[test]
    fn adaptive_4cycle_resolver_env_can_enable() {
        with_4cycle_env(|| {
            set_env(ENV_USE_WCOJ_4CYCLE_ADAPTIVE, "1");
            assert!(wcoj_4cycle_adaptive_enabled(None));
            set_env(ENV_USE_WCOJ_4CYCLE_ADAPTIVE, "0");
            assert!(!wcoj_4cycle_adaptive_enabled(None));
            set_env(ENV_USE_WCOJ_4CYCLE_ADAPTIVE, "true");
            assert!(wcoj_4cycle_adaptive_enabled(None));
        });
    }

    #[test]
    fn kill_4cycle_resolver_honors_env_and_config() {
        with_4cycle_env(|| {
            assert!(!wcoj_4cycle_disabled(None));
            set_env(ENV_DISABLE_WCOJ_4CYCLE, "1");
            assert!(wcoj_4cycle_disabled(None));
            assert!(!wcoj_4cycle_disabled(Some(false)));
            set_env(ENV_DISABLE_WCOJ_4CYCLE, "0");
            assert!(wcoj_4cycle_disabled(Some(true)));
        });
    }

    fn canonical_4cycle_multiway() -> RirNode {
        RirNode::MultiWayJoin {
            inputs: vec![
                RirNode::Scan { rel: RelId(1) },
                RirNode::Scan { rel: RelId(2) },
                RirNode::Scan { rel: RelId(3) },
                RirNode::Scan { rel: RelId(4) },
            ],
            slot_vars: vec![
                vec![Some(0u32), Some(1)],
                vec![Some(1u32), Some(2)],
                vec![Some(2u32), Some(3)],
                vec![Some(3u32), Some(0)],
            ],
            output_columns: vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(3),
                ProjectExpr::Column(5),
            ],
            fallback: Box::new(RirNode::Unit),
            var_order: None,
        }
    }

    #[test]
    fn match_4cycle_canonical_returns_four_rels() {
        let node = canonical_4cycle_multiway();
        let m = match_multiway_4cycle(&node).expect("must match canonical 4-cycle");
        assert_eq!(m.rel_e1, RelId(1));
        assert_eq!(m.rel_e2, RelId(2));
        assert_eq!(m.rel_e3, RelId(3));
        assert_eq!(m.rel_e4, RelId(4));
    }

    #[test]
    fn match_4cycle_rejects_non_multiway() {
        assert!(match_multiway_4cycle(&RirNode::Scan { rel: RelId(1) }).is_none());
    }

    #[test]
    fn match_4cycle_rejects_triangle_shape() {
        // Triangle is 3 inputs — 4-cycle matcher must reject.
        let triangle = RirNode::MultiWayJoin {
            inputs: vec![
                RirNode::Scan { rel: RelId(1) },
                RirNode::Scan { rel: RelId(2) },
                RirNode::Scan { rel: RelId(3) },
            ],
            slot_vars: vec![
                vec![Some(0u32), Some(1)],
                vec![Some(1u32), Some(2)],
                vec![Some(0u32), Some(2)],
            ],
            output_columns: vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(3),
            ],
            fallback: Box::new(RirNode::Unit),
            var_order: None,
        };
        assert!(match_multiway_4cycle(&triangle).is_none());
    }

    #[test]
    fn match_4cycle_rejects_rotated_output_columns() {
        let mut node = canonical_4cycle_multiway();
        if let RirNode::MultiWayJoin { output_columns, .. } = &mut node {
            output_columns.swap(0, 1);
        }
        assert!(match_multiway_4cycle(&node).is_none());
    }

    /// W2.2: 4-cycle Alt-grouping output_columns
    /// `[Column(5), Column(0), Column(1), Column(3)]` must
    /// match. The W2.2 matcher relaxation accepts both
    /// Default `[0, 1, 3, 5]` and Alt `[5, 0, 1, 3]`.
    #[test]
    fn match_4cycle_accepts_w22_alt_grouping_output_columns() {
        let mut node = canonical_4cycle_multiway();
        if let RirNode::MultiWayJoin { output_columns, .. } = &mut node {
            *output_columns = vec![
                ProjectExpr::Column(5),
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(3),
            ];
        }
        let m = match_multiway_4cycle(&node)
            .expect("W2.2 matcher must accept the Alt-grouping output-column layout");
        // RelIds preserved positionally from the body's
        // MultiWayJoin.inputs (which are in canonical
        // semantic order [WX, XY, YZ, ZW] per W2.2 step 2a).
        assert_eq!(m.rel_e1, RelId(1));
        assert_eq!(m.rel_e2, RelId(2));
        assert_eq!(m.rel_e3, RelId(3));
        assert_eq!(m.rel_e4, RelId(4));
    }

    /// W2.2: 4-cycle output_columns `[1, 0, 3, 5]` (only swap
    /// of cols 0 and 1 vs Default) must STILL be rejected —
    /// it's neither Default nor Alt.
    #[test]
    fn match_4cycle_rejects_invalid_w22_output_columns() {
        let mut node = canonical_4cycle_multiway();
        if let RirNode::MultiWayJoin { output_columns, .. } = &mut node {
            *output_columns = vec![
                ProjectExpr::Column(1),
                ProjectExpr::Column(0),
                ProjectExpr::Column(3),
                ProjectExpr::Column(5),
            ];
        }
        assert!(match_multiway_4cycle(&node).is_none());
    }

    #[test]
    fn match_4cycle_rejects_arity_mismatched_output_columns() {
        let mut node = canonical_4cycle_multiway();
        if let RirNode::MultiWayJoin { output_columns, .. } = &mut node {
            output_columns.pop();
        }
        assert!(match_multiway_4cycle(&node).is_none());
    }

    #[test]
    fn match_4cycle_rejects_unclosed_cycle() {
        // Slot 3's second var is supposed to equal slot 0's first
        // var (closing the cycle). Replace with a fresh id.
        let mut node = canonical_4cycle_multiway();
        if let RirNode::MultiWayJoin { slot_vars, .. } = &mut node {
            slot_vars[3] = vec![Some(3), Some(99)];
        }
        assert!(match_multiway_4cycle(&node).is_none());
    }

    #[test]
    fn match_4cycle_rejects_non_scan_input() {
        let mut node = canonical_4cycle_multiway();
        if let RirNode::MultiWayJoin { inputs, .. } = &mut node {
            inputs[0] = RirNode::Unit;
        }
        assert!(match_multiway_4cycle(&node).is_none());
    }

    #[test]
    fn match_4cycle_rejects_input_arity_mismatch() {
        let mut node = canonical_4cycle_multiway();
        if let RirNode::MultiWayJoin { inputs, .. } = &mut node {
            inputs.push(RirNode::Scan { rel: RelId(5) });
        }
        assert!(match_multiway_4cycle(&node).is_none());
    }
}
