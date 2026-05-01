//! v0.6.2 env-gated WCOJ triangle dispatch — runtime hook.
//!
//! Wires the existing GPU 3-way WCOJ kernel into the executor's
//! per-rule loop. The hook is opt-in via the env variable
//! [`ENV_USE_WCOJ_TRIANGLE_U32`] (or via
//! [`xlog_core::RuntimeConfig::wcoj_triangle_dispatch`] for tests).
//!
//! ## Recognized RIR shape
//!
//! The hook pattern-matches the exact RIR tree that
//! [`xlog_logic::Lowerer`] produces for a triangle rule of the form
//! `tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z)`:
//!
//! ```text
//! Project {
//!     input: Join {
//!         left: Join {
//!             left: Scan(e_xy),
//!             right: Scan(e_yz),
//!             left_keys: [1],          // e_xy.col1 == Y
//!             right_keys: [0],         // e_yz.col0 == Y
//!             join_type: Inner,
//!         },
//!         right: Scan(e_xz),
//!         left_keys: [0, 3],           // X, Z (cols 0 and 3 of inner join's output)
//!         right_keys: [0, 1],          // e_xz.col0 == X, e_xz.col1 == Z
//!         join_type: Inner,
//!     },
//!     columns: [Column(0), Column(1), Column(3)],  // X, Y, Z
//! }
//! ```
//!
//! Anything else (different shape, non-Inner join, recursive SCC,
//! 2-arity heads, missing input buffers, unsupported scalar types,
//! mixed-width slots within the same triangle, or no runtime-backed
//! memory manager) returns `Ok(None)` and the caller takes the
//! existing binary-join path.
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
//! * Histogram-guided block dispatch.
//! * Default-on behavior (env var must be explicitly set).

use xlog_core::{RelId, Result};
use xlog_cuda::device_runtime::StreamId;
use xlog_cuda::CudaBuffer;
use xlog_ir::{rir::ProjectExpr, CompiledRule, JoinType, RirNode};

use super::Executor;

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
/// `"1"` / case-insensitive `"true"` → ON. Anything else → OFF.
/// Force-WCOJ ([`ENV_USE_WCOJ_TRIANGLE_U32`]) takes precedence
/// over this env: when force is on, the classifier is bypassed.
pub const ENV_USE_WCOJ_TRIANGLE_ADAPTIVE: &str = "XLOG_USE_WCOJ_TRIANGLE_ADAPTIVE";

/// Resolve the adaptive dispatch gate. Same precedence shape as
/// the force gate above (config override > env > false).
pub(super) fn wcoj_adaptive_enabled(config_override: Option<bool>) -> bool {
    if let Some(v) = config_override {
        return v;
    }
    std::env::var(ENV_USE_WCOJ_TRIANGLE_ADAPTIVE)
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
const WCOJ_ADAPTIVE_SKEW_THRESHOLD: f64 = 0.10;

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

/// Pattern-match the canonical triangle RIR. Returns the three
/// scan rel IDs in WCOJ slot order on a successful match;
/// `None` for any deviation.
///
/// The pattern is intentionally strict: any future RIR shape
/// change in the lowerer falls back silently rather than running
/// a kernel against the wrong tree. When the lowerer is
/// generalized, this matcher gets a corresponding slice.
pub(super) fn match_triangle_rir(body: &RirNode) -> Option<TriangleRirMatch> {
    // Outer Project { input: Join, columns: [Column(0), Column(1), Column(3)] }.
    let RirNode::Project {
        input: outer_input,
        columns,
    } = body
    else {
        return None;
    };
    if columns.len() != 3 {
        return None;
    }
    let expected_cols = [0usize, 1, 3];
    for (i, expr) in columns.iter().enumerate() {
        match expr {
            ProjectExpr::Column(idx) if *idx == expected_cols[i] => {}
            _ => return None,
        }
    }
    // Outer Join — Inner, left_keys [0, 3], right_keys [0, 1].
    let RirNode::Join {
        left: l1,
        right: r1,
        left_keys: lk1,
        right_keys: rk1,
        join_type: jt1,
    } = outer_input.as_ref()
    else {
        return None;
    };
    if !matches!(jt1, JoinType::Inner) {
        return None;
    }
    if lk1.as_slice() != [0usize, 3] || rk1.as_slice() != [0usize, 1] {
        return None;
    }
    // Right side of outer Join: Scan(e_xz).
    let RirNode::Scan { rel: rel_xz } = r1.as_ref() else {
        return None;
    };
    // Inner Join — Inner, left_keys [1], right_keys [0].
    let RirNode::Join {
        left: l2,
        right: r2,
        left_keys: lk2,
        right_keys: rk2,
        join_type: jt2,
    } = l1.as_ref()
    else {
        return None;
    };
    if !matches!(jt2, JoinType::Inner) {
        return None;
    }
    if lk2.as_slice() != [1usize] || rk2.as_slice() != [0usize] {
        return None;
    }
    // Inner Join's leaves: Scan(e_xy), Scan(e_yz).
    let RirNode::Scan { rel: rel_xy } = l2.as_ref() else {
        return None;
    };
    let RirNode::Scan { rel: rel_yz } = r2.as_ref() else {
        return None;
    };
    Some(TriangleRirMatch {
        rel_xy: *rel_xy,
        rel_yz: *rel_yz,
        rel_xz: *rel_xz,
    })
}

/// Physical key width for a WCOJ-eligible binary relation at
/// the RIR-level dispatch. `FourByte` covers `U32` and `Symbol`
/// (bit-identical layout); `EightByte` covers `U64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WcojKeyWidth {
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
        // 1. Gate resolution. Decision tree:
        //
        //    a. If `wcoj_triangle_dispatch` resolves to true
        //       (config Some(true) or env=1) → force WCOJ;
        //       classifier is bypassed entirely (mode = Force).
        //    b. Else if `wcoj_triangle_dispatch_adaptive`
        //       resolves to true → run classifier; dispatch only
        //       when score ≥ threshold (mode = Adaptive).
        //    c. Else → no dispatch.
        //
        // Force takes precedence so test/microbench callers that
        // already pass `Some(true)` keep their existing
        // semantics unchanged (and silently sidestep classifier
        // overhead).
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

        // 2. Pattern-match the triangle RIR.
        let Some(matched) = match_triangle_rir(&rule.body) else {
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
        let launch_stream = match self.wcoj_triangle_stream_or_init() {
            Some(s) => s,
            None => return Ok(None),
        };

        // 6. Adaptive mode only: run the classifier on the same
        // launch_stream as the eventual WCOJ pipeline. Classifier
        // failures (Ok(None) from the provider) silently fall
        // back to binary-join — classifier is optimization, not
        // correctness. A score below
        // `WCOJ_ADAPTIVE_SKEW_THRESHOLD` likewise falls back.
        if mode == DispatchMode::Adaptive {
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
        let dispatch_result =
            self.run_wcoj_triangle_pipeline(buf_xy, buf_yz, buf_xz, launch_stream, width);
        match dispatch_result {
            Ok(buf) => {
                self.wcoj_triangle_dispatch_count += 1;
                Ok(Some(buf))
            }
            Err(_) => Ok(None),
        }
    }

    /// Inner pipeline: 3× layout construction + triangle kernel.
    /// Split out so [`try_dispatch_wcoj_triangle`] can map any
    /// error to `Ok(None)` cleanly. Branches by `width` between
    /// the parallel u32 and u64 provider entries.
    fn run_wcoj_triangle_pipeline(
        &self,
        buf_xy: &CudaBuffer,
        buf_yz: &CudaBuffer,
        buf_xz: &CudaBuffer,
        launch_stream: StreamId,
        width: WcojKeyWidth,
    ) -> Result<CudaBuffer> {
        match width {
            WcojKeyWidth::FourByte => {
                let layout_xy = self
                    .provider
                    .wcoj_layout_u32_recorded(buf_xy, launch_stream)?;
                let layout_yz = self
                    .provider
                    .wcoj_layout_u32_recorded(buf_yz, launch_stream)?;
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
                let layout_xy = self
                    .provider
                    .wcoj_layout_u64_recorded(buf_xy, launch_stream)?;
                let layout_yz = self
                    .provider
                    .wcoj_layout_u64_recorded(buf_yz, launch_stream)?;
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

    /// Resolve the cached WCOJ launch stream, lazily initializing
    /// it on first call by acquiring one stream from the runtime
    /// pool. Subsequent calls reuse the same stream — mirrors
    /// [`xlog_cuda::CudaKernelProvider::recorded_op_stream`]
    /// (provider/mod.rs).
    ///
    /// Returns `None` only when (a) the manager has no runtime,
    /// or (b) the very first acquisition fails (pool already
    /// at cap from other consumers). After that first success
    /// the cached id keeps resolving for the executor's lifetime.
    fn wcoj_triangle_stream_or_init(&self) -> Option<StreamId> {
        if let Some(s) = self.wcoj_triangle_stream.get() {
            return Some(*s);
        }
        let runtime = self.provider.memory().runtime()?;
        let stream = runtime.stream_pool().acquire().ok()?;
        let _ = self.wcoj_triangle_stream.set(stream);
        self.wcoj_triangle_stream.get().copied()
    }
}
