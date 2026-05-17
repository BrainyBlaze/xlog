//! Stable textual explain output for a (hypergraph, eligibility,
//! variable order) triple.
//!
//! The format is deterministic and intended for snapshot tests in
//! PR 1, debugging in PR 2+, and a CLI subcommand later. The format
//! is **not** a stable API for downstream tools yet — keep
//! consumers inside the workspace.
//!
//! Format:
//! ```text
//! rule head=<predicate>
//!   vertices: [<v0> <v1> ...]
//!   hyperedges:
//!     <predicate>(<arg0>, <arg1>, ...)
//!     ...
//!   filters: <comparison_count>
//!   eligibility: <Eligible | Ineligible>
//!     <boundary> ...
//!   variable-order(<name>): [<v0> <v1> ...]
//! ```
//! where each `<arg>` is either `?<varname>` for a variable position
//! or `_` for a constant / anonymous wildcard. Vertices and the
//! variable-order line use names rather than `VertexId`s so the
//! output is stable across construction-order changes that don't
//! change the source rule.

use super::eligibility::{Boundary, Eligibility};
use super::ir::HypergraphRule;
use super::var_order::VariableOrder;
use std::fmt::Write;

/// Render a stable textual explanation of `hg` plus its eligibility
/// verdict and a variable order computed via `vo`. Pure: no IO, no
/// hidden state.
pub fn explain(hg: &HypergraphRule, eligibility: &Eligibility, vo: &dyn VariableOrder) -> String {
    let mut out = String::new();

    writeln!(out, "rule head={}", hg.head_predicate).unwrap();

    // Vertices.
    if hg.vertices.is_empty() {
        writeln!(out, "  vertices: []").unwrap();
    } else {
        let names: Vec<&str> = hg.vertices.iter().map(|v| v.name.as_str()).collect();
        writeln!(out, "  vertices: [{}]", names.join(" ")).unwrap();
    }

    // Hyperedges.
    if hg.hyperedges.is_empty() {
        writeln!(out, "  hyperedges: <none>").unwrap();
    } else {
        writeln!(out, "  hyperedges:").unwrap();
        for edge in &hg.hyperedges {
            let args: Vec<String> = edge
                .vertex_positions
                .iter()
                .map(|p| match p {
                    Some(vid) => format!("?{}", hg.vertex(*vid).name),
                    None => "_".to_string(),
                })
                .collect();
            writeln!(out, "    {}({})", edge.predicate, args.join(", ")).unwrap();
        }
    }

    // Filters.
    writeln!(out, "  filters: {}", hg.comparison_count).unwrap();

    // Eligibility.
    match eligibility {
        Eligibility::Eligible => {
            writeln!(out, "  eligibility: Eligible").unwrap();
        }
        Eligibility::Ineligible(boundaries) => {
            writeln!(out, "  eligibility: Ineligible").unwrap();
            for b in boundaries {
                writeln!(out, "    {}", format_boundary(b)).unwrap();
            }
        }
    }

    // Variable order.
    let order = vo.order(hg);
    let order_names: Vec<&str> = order
        .iter()
        .map(|vid| hg.vertex(*vid).name.as_str())
        .collect();
    writeln!(
        out,
        "  variable-order({}): [{}]",
        vo.name(),
        order_names.join(" ")
    )
    .unwrap();

    out
}

fn format_boundary(b: &Boundary) -> String {
    match b {
        Boundary::GroundFact => "GroundFact".to_string(),
        Boundary::HeadAggregation => "HeadAggregation".to_string(),
        Boundary::BodyNegation => "BodyNegation".to_string(),
        Boundary::BodyIsExpr => "BodyIsExpr".to_string(),
        Boundary::InsufficientPositiveAtoms { positive_count } => {
            format!("InsufficientPositiveAtoms(positive_count={positive_count})")
        }
        Boundary::JoinKeysExceedBinaryFallbackLimit {
            context,
            count,
            limit,
        } => {
            format!(
                "JoinKeysExceedBinaryFallbackLimit(context={context:?}, count={count}, limit={limit})"
            )
        }
        Boundary::UnsupportedKeyType { var, ty } => {
            format!("UnsupportedKeyType(var={var}, ty={ty:?})")
        }
    }
}
