//! Verify-size guard calibration support — emit the GpuCnf var_cap /
//! clause_cap (the upper bounds a pre-launch verify-size guard keys on) for
//! the dense correlated reachability programs at the d-DNNF-verify explosion
//! boundary (n=5 ok, n=6 = 47s but completes, n=7 = CUDA launch-fail in the
//! d-DNNF compile). Reads the caps right after CNF ENCODING — before the
//! compile that crashes at n=7 — so the n=7 caps are measurable without
//! poisoning the CUDA context. Functional (no d-DNNF compile, no perf
//! timing); local-safe. Remote-GPU-only rule does not apply (no compile/eval).

#![cfg(feature = "host-io")]

use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::{encode_cnf_gpu, GpuPirGraph, GpuPirRoots};
use xlog_prob::pir::PirNodeId;
use xlog_prob::provenance::extract_from_source;

fn try_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(2 * 1024 * 1024 * 1024),
    ));
    CudaKernelProvider::new(device, memory).ok().map(Arc::new)
}

/// Near-complete probabilistic digraph on 1..=n (i<j edges), reach(1,n)
/// query — the dense correlated fixture whose D4 verify explodes.
fn dense_correlated_source(n: u32) -> String {
    let mut s = String::new();
    for i in 1..=n {
        for j in (i + 1)..=n {
            s.push_str(&format!("0.5::edge({i},{j}).\n"));
        }
    }
    s.push_str("reach(X,Y) :- edge(X,Y).\n");
    s.push_str("reach(X,Z) :- reach(X,Y), edge(Y,Z).\n");
    s.push_str(&format!("query(reach(1,{n})).\n"));
    s
}

#[test]
fn d4_verify_calibration_caps() {
    let Some(provider) = try_provider() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    eprintln!("[verify-size calibration] GpuCnf (var_cap, clause_cap) at the dense-correlated d-DNNF-verify boundary:");
    for n in [5u32, 6, 7, 8] {
        let src = dense_correlated_source(n);
        let prov = match extract_from_source(&src) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  n={n}: provenance extract failed: {e}");
                continue;
            }
        };
        let roots: Vec<PirNodeId> = prov
            .queries
            .iter()
            .filter_map(|a| prov.query_formula(&a.predicate, &a.args))
            .collect();
        if roots.is_empty() {
            eprintln!("  n={n}: no query roots");
            continue;
        }
        let gpu_pir = match GpuPirGraph::from_host(&prov.pir, &provider) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("  n={n}: GpuPirGraph::from_host failed: {e}");
                continue;
            }
        };
        let gpu_roots = match GpuPirRoots::from_host(&roots, &provider) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  n={n}: GpuPirRoots::from_host failed: {e}");
                continue;
            }
        };
        // CNF encode only — NO D4 compile (which is where n>=7 crashes).
        match encode_cnf_gpu(&gpu_pir, &gpu_roots, &provider) {
            Ok(enc) => eprintln!(
                "  n={n} ({} edges, pir_nodes={}): var_cap={}, clause_cap={}",
                n * (n - 1) / 2,
                prov.pir.len(),
                enc.cnf.var_cap,
                enc.cnf.clause_cap
            ),
            Err(e) => eprintln!("  n={n}: encode_cnf_gpu failed: {e}"),
        }
    }
    eprintln!("[verify-size calibration] n=6 is the 'completes at 47s' upper-anchor; n=7 is the launch-fail point. The verify-size default sits between the largest known-good (cert max var_cap=15, a floor) and the n=7 cap.");
}
