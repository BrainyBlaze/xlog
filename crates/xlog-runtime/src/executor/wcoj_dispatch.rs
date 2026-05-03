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
//! ## Out of scope (per slice spec)
//!
//! * Recursive / SCC mixed execution — the executor's recursive
//!   branch is unchanged. We hook only the non-recursive branch.
//! * Cost model.
//! * Mixed-width admission (a triangle with both U32 and U64
//!   slots stays on the binary-join path).
//! * Histogram-guided block dispatch (B1 heavy-row offload).

use xlog_core::{RelId, Result};
use xlog_cuda::device_runtime::StreamId;
use xlog_cuda::CudaBuffer;
use xlog_ir::{rir::ProjectExpr, CompiledRule, RirNode};

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

/// Confirm `output_columns` is the certified `(X, Y, Z)` emit
/// order. The GPU kernel writes triples in this order; a rotated
/// or computed projection would silently produce wrong results.
fn output_columns_match_canonical_triangle(cols: &[ProjectExpr]) -> bool {
    cols.len() == 3
        && matches!(cols[0], ProjectExpr::Column(0))
        && matches!(cols[1], ProjectExpr::Column(1))
        && matches!(cols[2], ProjectExpr::Column(3))
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
fn output_columns_match_canonical_4cycle(cols: &[ProjectExpr]) -> bool {
    cols.len() == 4
        && matches!(cols[0], ProjectExpr::Column(0))
        && matches!(cols[1], ProjectExpr::Column(1))
        && matches!(cols[2], ProjectExpr::Column(3))
        && matches!(cols[3], ProjectExpr::Column(5))
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
        let Some(matched) = match_multiway_triangle(&rule.body) else {
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
        #[cfg(feature = "wcoj-phase-timing")]
        let mut classifier_ms: f32 = 0.0;
        if mode == DispatchMode::Adaptive {
            #[cfg(feature = "wcoj-phase-timing")]
            let cls_start = Instant::now();
            let score = match width {
                WcojKeyWidth::FourByte => self.provider.wcoj_triangle_skew_score_u32(
                    buf_xy,
                    buf_yz,
                    buf_xz,
                    launch_stream,
                ),
                WcojKeyWidth::EightByte => self.provider.wcoj_triangle_skew_score_u64(
                    buf_xy,
                    buf_yz,
                    buf_xz,
                    launch_stream,
                ),
            };
            #[cfg(feature = "wcoj-phase-timing")]
            {
                classifier_ms = cls_start.elapsed().as_secs_f64() as f32 * 1000.0;
            }
            match score {
                Ok(Some(s)) if s >= WCOJ_ADAPTIVE_SKEW_THRESHOLD => {
                    // Above threshold → fall through to dispatch.
                }
                Ok(Some(_)) | Ok(None) => return Ok(None),
                Err(_) => return Ok(None),
            }
        }

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
            #[cfg(feature = "wcoj-phase-timing")]
            &mut layout_times,
        );
        match dispatch_result {
            Ok(buf) => {
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
        #[cfg(feature = "wcoj-phase-timing")] layout_times_ms: &mut [f32; 3],
    ) -> Result<CudaBuffer> {
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
        let Some(matched) = match_multiway_4cycle(&rule.body) else {
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

        // 7. Adaptive mode only: run the skew classifier on the
        // same launch_stream as the eventual WCOJ pipeline.
        // Classifier failures (Ok(None)) silently fall back to
        // binary-join — classifier is optimization, not
        // correctness. Score below threshold also falls back.
        if mode == DispatchMode::Adaptive {
            let score = match width {
                WcojKeyWidth::FourByte => self.provider.wcoj_4cycle_skew_score_u32(
                    buf_e1,
                    buf_e2,
                    buf_e3,
                    buf_e4,
                    launch_stream,
                ),
                WcojKeyWidth::EightByte => self.provider.wcoj_4cycle_skew_score_u64(
                    buf_e1,
                    buf_e2,
                    buf_e3,
                    buf_e4,
                    launch_stream,
                ),
            };
            match score {
                Ok(Some(s)) if s >= WCOJ_ADAPTIVE_4CYCLE_SKEW_THRESHOLD => {
                    // Above threshold → fall through to dispatch.
                }
                Ok(Some(_)) | Ok(None) => return Ok(None),
                Err(_) => return Ok(None),
            }
        }

        // 8. Run layout (4× per slot) + 4-cycle kernel. Failure
        // → silent fallback per slice contract.
        let dispatch_result =
            self.run_wcoj_4cycle_pipeline(buf_e1, buf_e2, buf_e3, buf_e4, launch_stream, width);
        match dispatch_result {
            Ok(buf) => {
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
    ) -> Result<CudaBuffer> {
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
    pub(super) fn wcoj_dispatch_stream_or_init(&self) -> Option<StreamId> {
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
