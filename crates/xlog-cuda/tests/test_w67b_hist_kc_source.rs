// crates/xlog-cuda/tests/test_w67b_hist_kc_source.rs
//! Goal-038-B Authorization 5 G_HIST_KC source-audit certs.

use std::fs;
use std::path::PathBuf;

fn workspace_source(path: &str) -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("crate lives under workspace/crates");
    let full = root.join(path);
    fs::read_to_string(&full).unwrap_or_else(|e| panic!("failed to read {}: {}", full.display(), e))
}

fn wcoj_cu_source() -> String {
    workspace_source("crates/xlog-cuda/kernels/wcoj.cu")
}

fn provider_source() -> String {
    workspace_source("crates/xlog-cuda/src/provider/wcoj.rs")
}

fn recursive_source() -> String {
    workspace_source("crates/xlog-runtime/src/executor/recursive.rs")
}

fn phase_report_source() -> String {
    workspace_source("crates/xlog-integration/src/bin/wcoj_phase_report.rs")
}

fn w52_bench_source() -> String {
    workspace_source("crates/xlog-integration/benches/w52_skewed_multiway_bench.rs")
}

fn promoter_source() -> String {
    workspace_source("crates/xlog-logic/src/promote.rs")
}

fn extract_device_template(src: &str, name: &str) -> String {
    let needle = format!("void {}", name);
    extract_body_after(src, &needle).unwrap_or_else(|| panic!("{name} must exist"))
}

fn extract_body_after(src: &str, needle: &str) -> Option<String> {
    let start = src.find(needle)?;
    let after_name = start + needle.len();
    let mut depth_paren = 0i32;
    let mut saw_paren = false;
    let mut after_args = None;
    for (offset, b) in src[after_name..].bytes().enumerate() {
        if b == b'(' {
            depth_paren += 1;
            saw_paren = true;
        } else if b == b')' {
            depth_paren -= 1;
            if saw_paren && depth_paren == 0 {
                after_args = Some(after_name + offset + 1);
                break;
            }
        }
    }
    let mut depth = 0i32;
    let mut body_start = None;
    let mut body_end = None;
    let bytes = src.as_bytes();
    let mut i = after_args?;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            if body_start.is_none() {
                body_start = Some(i + 1);
            }
            depth += 1;
        } else if bytes[i] == b'}' {
            depth -= 1;
            if depth == 0 {
                body_end = Some(i);
                break;
            }
        }
        i += 1;
    }
    Some(src[body_start?..body_end?].to_string())
}

#[test]
fn kclique_metadata_provider_entries_are_present() {
    let src = provider_source();
    for entry in [
        "wcoj_clique5_metadata_recorded_u32",
        "wcoj_clique5_metadata_recorded_u64",
        "wcoj_clique6_metadata_recorded_u32",
        "wcoj_clique6_metadata_recorded_u64",
    ] {
        assert!(src.contains(entry), "{entry} must be a provider entry");
    }
    assert!(
        src.contains("wcoj_build_metadata_u32_recorded")
            && src.contains("wcoj_build_metadata_u64_recorded"),
        "K-clique metadata entries must reuse Phase-1 G1 metadata builders"
    );
}

#[test]
fn kclique_hg_kernel_templates_accept_runtime_histogram_params() {
    let src = wcoj_cu_source();
    for sig in [
        "const T* unique_keys",
        "const uint32_t* fan_out",
        "const uint32_t* prefix_sum",
        "uint32_t total",
    ] {
        assert!(
            src.contains(sig),
            "K-clique template signature must include {sig}"
        );
    }

    for template in [
        "wcoj_clique_template_count_hg_grid_t",
        "wcoj_clique_template_materialize_hg_grid_t",
    ] {
        let body = extract_device_template(&src, template);
        assert!(
            body.contains("prefix_sum") && body.contains("fan_out") && body.contains("total"),
            "{template} must consume metadata-derived prefix/fanout/total"
        );
        assert!(
            body.contains("leader_work_total"),
            "{template} must derive its block range from metadata total, not raw leader_count"
        );
        assert!(
            !body.contains("block_start >= leader_count"),
            "{template} must not bound the grid slice directly by leader_count"
        );
    }
}

#[test]
fn kclique_provider_builds_metadata_before_kernel_launch() {
    let src = provider_source();
    let metadata_pos = src
        .find("let leader_metadata")
        .expect("provider must build leader metadata");
    let count_launch_pos = src
        .find("let count_kernel")
        .expect("provider must still launch count kernel");
    assert!(
        metadata_pos < count_launch_pos,
        "leader metadata must be built before K-clique count launch"
    );
    for token in [
        "rec_count.read(&leader_metadata.unique_keys)",
        "rec_count.read(&leader_metadata.fan_out)",
        "rec_count.read(&leader_metadata.prefix_sum)",
        "rec_mat.read(&leader_metadata.unique_keys)",
        "rec_mat.read(&leader_metadata.fan_out)",
        "rec_mat.read(&leader_metadata.prefix_sum)",
    ] {
        assert!(
            src.contains(token),
            "provider recorder must include {token}"
        );
    }
}

#[test]
fn recursive_merge_phase_refresh_hook_is_present() {
    let src = recursive_source();
    assert!(
        src.contains("refresh_kclique_edge_metadata_after_merge"),
        "recursive Merge phase must refresh K-clique edge metadata"
    );
    assert!(
        src.contains("record_kclique_histogram_refresh_time"),
        "recursive Merge phase must record histogram-refresh timing"
    );
}

#[test]
fn phase_report_exposes_kclique_histogram_refresh_costs() {
    let src = phase_report_source();
    for token in [
        "measure_recursive_k5_histogram_refresh",
        "print_kclique_histogram_refresh_report",
        "kclique_histogram_refresh_count",
        "kclique_histogram_refresh_nanos",
        "kclique_metadata_build_count",
        "kclique_metadata_build_nanos",
        "merge_histogram_refresh",
        "leader_metadata_build",
    ] {
        assert!(
            src.contains(token),
            "wcoj_phase_report must expose {token} for S_HIST_KC.7"
        );
    }
}

#[test]
fn w52_bench_reports_kclique_metadata_cost_ratio() {
    let src = w52_bench_source();
    for token in [
        "report_kclique_metadata_cost",
        "reset_kclique_metadata_build_metrics",
        "kclique_metadata_build_count",
        "kclique_metadata_build_nanos",
        "metadata_ratio",
        "ratio <= 0.05",
    ] {
        assert!(
            src.contains(token),
            "W5.2 bench must expose {token} for M_HIST_KC.7"
        );
    }
}

#[test]
fn kclique_promoter_admits_recursive_bodies_for_hist_refresh() {
    let src = promoter_source();
    assert!(
        !src.contains("if recursive_scan_count(&rule.body, &head_rel_set) == 0"),
        "G_HIST_KC recursive K-clique fixtures must reach MultiWayJoin promotion"
    );
    assert!(
        src.contains("try_promote_clique_k(&rule.body, 5, stats)")
            && src.contains("try_promote_clique_k(&rule.body, 6, stats)"),
        "K-clique promotion must remain active for k=5 and k=6"
    );
}

#[test]
fn paper_phase1_citation_comment_is_present() {
    let required = "Paper §5 Algorithm 1 Phase 1: Histograms maintained alongside data; refreshed during Merge per Authorization 5 (2026-05-17)";
    assert!(
        wcoj_cu_source().contains(required),
        "K-clique kernel must carry the paper §5 Phase 1 citation"
    );
    assert!(
        provider_source().contains(required),
        "K-clique provider path must carry the paper §5 Phase 1 citation"
    );
}
