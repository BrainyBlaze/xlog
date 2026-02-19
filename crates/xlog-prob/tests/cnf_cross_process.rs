//! Cross-process determinism test for CNF encoding.
//!
//! Spawns two child processes that each build a PIR graph through
//! HashMap-simulated non-deterministic interning (mimicking provenance.rs),
//! encode CNF, and write DIMACS output to a file. The parent asserts that
//! both processes produced byte-identical output.

use std::collections::HashMap;
use std::process::Command;

use xlog_prob::cnf::encode_cnf;
use xlog_prob::pir::{LeafId, PirGraph, PirNodeId};

/// Build a non-trivial PIR graph through HashMap-based "interning",
/// mimicking the non-deterministic node creation order in provenance.rs.
///
/// HashMap iteration order varies across processes due to Rust's per-process
/// random hash seeding (SipHash). This means `pir.lit()` calls happen in
/// different orders, assigning different PirNodeId values to the same leaves.
fn build_and_encode() -> String {
    // Simulate provenance's HashMap<PirKey, PirNodeId> interning.
    let mut intern: HashMap<String, ()> = HashMap::new();
    for i in 0..10u32 {
        intern.insert(format!("node_{}", i), ());
    }

    let mut pir = PirGraph::new();

    // Create leaf nodes in HashMap iteration order (non-deterministic across processes).
    // Each "name" maps to a fixed LeafId, but the PirNodeId assignment order varies.
    let mut leaf_by_id: HashMap<u32, PirNodeId> = HashMap::new();
    for (name, _) in &intern {
        let idx: u32 = name.strip_prefix("node_").unwrap().parse().unwrap();
        leaf_by_id.insert(idx, pir.lit(LeafId::new(idx)));
    }

    // Build deterministic structure using LeafId as the semantic key.
    // The logical graph is always: Or(And(leaf_0..leaf_4), And(leaf_5..leaf_9))
    let left: Vec<PirNodeId> = (0..5).map(|i| leaf_by_id[&i]).collect();
    let right: Vec<PirNodeId> = (5..10).map(|i| leaf_by_id[&i]).collect();
    let and_left = pir.and(left);
    let and_right = pir.and(right);
    let root = pir.or(vec![and_left, and_right]);

    encode_cnf(&pir, &[root]).unwrap().cnf.to_dimacs()
}

/// Child-mode entry point. When env var XLOG_CNF_OUTPUT_PATH is set,
/// build + encode and write DIMACS to that path. Otherwise no-op.
#[test]
fn cnf_cross_process_child() {
    let path = match std::env::var("XLOG_CNF_OUTPUT_PATH") {
        Ok(p) => p,
        Err(_) => return, // No-op when run directly
    };
    let dimacs = build_and_encode();
    std::fs::write(&path, dimacs.as_bytes()).unwrap();
}

/// Spawn two separate processes that each build the same logical PIR graph
/// (through non-deterministic HashMap interning) and encode CNF.
/// Assert that both produce byte-identical DIMACS output.
#[test]
fn cnf_encoding_is_identical_across_processes() {
    let exe = std::env::current_exe().unwrap();
    let tmp = std::env::temp_dir();
    let path_a = tmp.join("xlog_cnf_cross_a.dimacs");
    let path_b = tmp.join("xlog_cnf_cross_b.dimacs");

    // Clean up any stale files from previous runs.
    let _ = std::fs::remove_file(&path_a);
    let _ = std::fs::remove_file(&path_b);

    let run = |label: &str, path: &std::path::Path| {
        let output = Command::new(&exe)
            .arg("--exact")
            .arg("cnf_cross_process_child")
            .env("XLOG_CNF_OUTPUT_PATH", path.to_str().unwrap())
            .output()
            .unwrap_or_else(|e| panic!("{label}: failed to spawn: {e}"));
        assert!(
            output.status.success(),
            "{label}: child process failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    };

    run("process_a", &path_a);
    run("process_b", &path_b);

    let a = std::fs::read_to_string(&path_a)
        .unwrap_or_else(|e| panic!("failed to read process_a output: {e}"));
    let b = std::fs::read_to_string(&path_b)
        .unwrap_or_else(|e| panic!("failed to read process_b output: {e}"));

    // Cleanup.
    let _ = std::fs::remove_file(&path_a);
    let _ = std::fs::remove_file(&path_b);

    assert!(!a.is_empty(), "process_a produced empty DIMACS");
    assert_eq!(a, b, "CNF DIMACS output differs between processes");
}
