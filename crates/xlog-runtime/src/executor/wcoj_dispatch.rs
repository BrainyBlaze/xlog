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
//! 2-arity heads, missing or non-4-byte-key input buffers, no
//! runtime-backed memory manager) returns `Ok(None)` and the
//! caller takes the existing binary-join path.
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
//! * u64 key types.
//! * Histogram-guided block dispatch.
//! * Default-on behavior (env var must be explicitly set).

use std::sync::Arc;

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

/// True when `buf` has 2 columns and both are 4-byte keys
/// ([`xlog_core::ScalarType::U32`] or
/// [`xlog_core::ScalarType::Symbol`]). Both share the same 4-byte
/// physical layout; the WCOJ kernel reads either identically.
/// Cross-relation type compatibility (e.g., that a Symbol column
/// doesn't accidentally join with a U32 column with the same bit
/// pattern) is the planner's job — but the executor only sees
/// the lowered RIR by this point. For v1 we trust that the
/// existing pre-WCOJ binary-join path produces the same result
/// either way, so any divergence is caught by the
/// row-set-equality checks in the wiring/cert tests.
fn is_two_col_u32(buf: &CudaBuffer) -> bool {
    if buf.arity() != 2 {
        return false;
    }
    for col_idx in 0..2 {
        match buf.schema.column_type(col_idx) {
            Some(xlog_core::ScalarType::U32) | Some(xlog_core::ScalarType::Symbol) => {}
            _ => return false,
        }
    }
    true
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
        // 1. Gate.
        let override_value = self.config.wcoj_triangle_dispatch;
        if !wcoj_gate_enabled(override_value) {
            return Ok(None);
        }

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

        // 4. Look up input buffers + validate 4-byte-key schemas.
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
        if !is_two_col_u32(buf_xy) || !is_two_col_u32(buf_yz) || !is_two_col_u32(buf_xz) {
            return Ok(None);
        }

        // 5. Acquire a launch stream from the runtime pool.
        // Without a runtime-backed manager, the recorded WCOJ
        // primitives can't run — fall back silently.
        let runtime = match self.provider.memory().runtime() {
            Some(r) => Arc::clone(r),
            None => return Ok(None),
        };
        let launch_stream = match runtime.stream_pool().acquire() {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };

        // 6. Run layout + triangle. Convert any kernel error to
        // silent fallback per slice spec ("failure must not
        // corrupt store state"). The WCOJ helpers don't write
        // to the store, so an error here only loses the work
        // we just did — the binary-join path picks it up.
        let dispatch_result =
            self.run_wcoj_triangle_pipeline(buf_xy, buf_yz, buf_xz, launch_stream);
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
    /// error to `Ok(None)` cleanly.
    fn run_wcoj_triangle_pipeline(
        &self,
        buf_xy: &CudaBuffer,
        buf_yz: &CudaBuffer,
        buf_xz: &CudaBuffer,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let layout_xy = self
            .provider
            .wcoj_layout_u32_recorded(buf_xy, launch_stream)?;
        let layout_yz = self
            .provider
            .wcoj_layout_u32_recorded(buf_yz, launch_stream)?;
        let layout_xz = self
            .provider
            .wcoj_layout_u32_recorded(buf_xz, launch_stream)?;
        self.provider
            .wcoj_triangle_u32_recorded(&layout_xy, &layout_yz, &layout_xz, launch_stream)
    }

    /// Number of times the WCOJ triangle hook produced a result
    /// and the executor installed it. Used by tests to assert
    /// that the WCOJ path actually ran (vs. silently falling
    /// back to the existing binary-join path with the same
    /// answer).
    pub fn wcoj_triangle_dispatch_count(&self) -> u64 {
        self.wcoj_triangle_dispatch_count
    }
}
