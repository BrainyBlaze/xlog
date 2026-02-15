use std::collections::HashSet;
use std::sync::Arc;

use cudarc::driver::{DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::MemoryBudget;
use xlog_cuda::provider::{cnf_kernels, CNF_MODULE};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::compilation::{encode_cnf_gpu, GpuPirGraph, GpuPirRoots};
use xlog_prob::pir::PirNode;
use xlog_prob::provenance::extract_from_source;

fn try_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    match CudaKernelProvider::new(device, memory) {
        Ok(p) => Some(Arc::new(p)),
        Err(e) => {
            eprintln!("Skipping test: failed to create provider: {}", e);
            None
        }
    }
}

fn template_source_addition(num_labels: usize) -> String {
    // Mirrors `pyxlog::CompiledProgram::generate_template_source` for `addition`.
    //
    // Note: probabilities are dummy placeholders; they will be overwritten by the neural fast path.
    assert!(num_labels > 0);

    let p = 1.0f64 / (num_labels as f64);
    let scale = 0.9999999; // force an implicit none-branch
    let normalized_p = (p * scale).max(1e-10);

    let mut source = String::new();
    for input_pos in 0..2usize {
        for label_idx in 0..num_labels {
            if label_idx > 0 {
                source.push_str("; ");
            }
            source.push_str(&format!(
                "{:.10}::digit({}, {})",
                normalized_p, input_pos, label_idx
            ));
        }
        source.push_str(".\n");
    }

    source.push_str("addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.\n");

    let max_sum = 2usize * (num_labels.saturating_sub(1));
    for sum in 0..=max_sum {
        source.push_str(&format!("query(addition(0, 1, {})).\n", sum));
    }

    source
}

fn grid_dim(n: u32, block: u32) -> u32 {
    let mut grid = (n + block - 1) / block;
    if grid == 0 {
        grid = 1;
    }
    if grid > 65_535 {
        grid = 65_535;
    }
    grid
}

fn compute_reachable_device(
    provider: &Arc<CudaKernelProvider>,
    pir: &GpuPirGraph,
    roots: &GpuPirRoots,
) -> Vec<u32> {
    let num_nodes = pir.node_type.len();
    let num_nodes_u32 = u32::try_from(num_nodes).expect("num_nodes overflow");
    let num_roots_u32 = u32::try_from(roots.roots.len()).expect("num_roots overflow");

    let memory = provider.memory();
    let device = provider.device().inner();

    let mut reachable = memory.alloc::<u32>(num_nodes).unwrap();
    let mut queue = memory.alloc::<u32>(num_nodes).unwrap();
    let mut queue_ready = memory.alloc::<u32>(num_nodes).unwrap();
    let mut head = memory.alloc::<u32>(1).unwrap();
    let mut tail = memory.alloc::<u32>(1).unwrap();
    let mut in_flight = memory.alloc::<u32>(1).unwrap();

    device.memset_zeros(&mut reachable).unwrap();
    device.memset_zeros(&mut queue_ready).unwrap();
    device.memset_zeros(&mut head).unwrap();
    device.memset_zeros(&mut tail).unwrap();
    device.memset_zeros(&mut in_flight).unwrap();

    let reach_init_fn = device
        .get_func(CNF_MODULE, cnf_kernels::CNF_REACHABILITY_INIT)
        .expect("cnf_reachability_init not found");
    let reach_bfs_fn = device
        .get_func(CNF_MODULE, cnf_kernels::CNF_REACHABILITY_BFS)
        .expect("cnf_reachability_bfs not found");

    let block = 256u32;
    let grid_roots = grid_dim(num_roots_u32, block);
    unsafe {
        reach_init_fn
            .clone()
            .launch(
                LaunchConfig {
                    grid_dim: (grid_roots, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &roots.roots,
                    num_roots_u32,
                    num_nodes_u32,
                    &mut reachable,
                    &mut queue,
                    &mut queue_ready,
                    &mut head,
                    &mut tail,
                    &mut in_flight,
                ),
            )
            .unwrap();
    }

    let grid_nodes = grid_dim(num_nodes_u32, block);
    unsafe {
        reach_bfs_fn
            .clone()
            .launch(
                LaunchConfig {
                    grid_dim: (grid_nodes, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &pir.node_type,
                    &pir.child_offsets,
                    &pir.children,
                    &pir.decision_child_false,
                    &pir.decision_child_true,
                    num_nodes_u32,
                    &mut reachable,
                    &mut queue,
                    &mut queue_ready,
                    &mut head,
                    &mut tail,
                    &mut in_flight,
                ),
            )
            .unwrap();
    }

    provider.device().synchronize().unwrap();

    let mut host = vec![0u32; num_nodes];
    device.dtoh_sync_copy_into(&reachable, &mut host).unwrap();
    host
}

fn validate_dimacs_host(
    var_cap: u32,
    clause_cap: u32,
    lit_cap: u32,
    nv: u32,
    nc: u32,
    nl: u32,
    offsets: &[u32],
    lits: &[i32],
) -> Result<(), String> {
    if nv > var_cap || nc > clause_cap || nl > lit_cap {
        return Err(format!(
            "counts exceed caps: nv={} var_cap={} nc={} clause_cap={} nl={} lit_cap={}",
            nv, var_cap, nc, clause_cap, nl, lit_cap
        ));
    }
    if offsets.len() != (nc as usize) + 1 {
        return Err(format!(
            "offsets length mismatch: got {} expected {}",
            offsets.len(),
            (nc as usize) + 1
        ));
    }
    if lits.len() != (nl as usize) {
        return Err(format!(
            "lits length mismatch: got {} expected {}",
            lits.len(),
            nl
        ));
    }

    if nc > 0 && offsets[0] != 0 {
        return Err(format!("offsets[0] != 0 (got {})", offsets[0]));
    }
    if offsets[nc as usize] != nl {
        return Err(format!(
            "offsets[nc] != nl (offsets[{}]={}, nl={})",
            nc, offsets[nc as usize], nl
        ));
    }

    for c in 0..(nc as usize) {
        let s = offsets[c] as usize;
        let e = offsets[c + 1] as usize;
        if s > e || e > (nl as usize) {
            return Err(format!(
                "bad clause range at c={}: s={} e={} nl={}",
                c, s, e, nl
            ));
        }
        for i in s..e {
            let lit = lits[i];
            if lit == 0 {
                return Err(format!(
                    "lit==0 at i={} (clause {} range {}..{} lits={:?})",
                    i,
                    c,
                    s,
                    e,
                    &lits[s..e]
                ));
            }
            let v = lit.unsigned_abs();
            if v == 0 || v > nv {
                return Err(format!(
                    "var out of bounds at i={} (clause {} range {}..{} lits={:?}): lit={} abs={} nv={}",
                    i,
                    c,
                    s,
                    e,
                    &lits[s..e],
                    lit,
                    v,
                    nv
                ));
            }
        }
    }
    Ok(())
}

#[test]
fn template_addition_cnf_is_valid_dimacs() {
    let Some(provider) = try_provider() else {
        return;
    };

    let source = template_source_addition(10);
    let provenance = extract_from_source(&source).expect("extract provenance");

    let mut roots_set: HashSet<xlog_prob::pir::PirNodeId> = HashSet::new();
    for atom in &provenance.queries {
        let Some(id) = provenance.query_formula(&atom.predicate, &atom.args) else {
            panic!("query formula missing for {}", atom.predicate);
        };
        roots_set.insert(id);
    }
    let mut roots: Vec<_> = roots_set.into_iter().collect();
    roots.sort_by_key(|id| id.as_u32());

    let gpu_pir = GpuPirGraph::from_host(&provenance.pir, &provider).expect("upload pir");
    let gpu_roots = GpuPirRoots::from_host(&roots, &provider).expect("upload roots");

    // Debugging guardrail: reachability must be closed under child edges.
    // If this fails, CNF emission may read node_var==0 for an unreachable child and emit lit==0.
    let reachable = compute_reachable_device(&provider, &gpu_pir, &gpu_roots);
    let reachable2 = compute_reachable_device(&provider, &gpu_pir, &gpu_roots);
    assert_eq!(
        reachable, reachable2,
        "cnf_reachability_bfs produced non-deterministic reachable set"
    );
    for (i, node) in provenance.pir.nodes().iter().enumerate() {
        if reachable[i] == 0 {
            continue;
        }
        match node {
            PirNode::And { children } | PirNode::Or { children } => {
                for child in children {
                    let cid = child.as_u32() as usize;
                    if reachable.get(cid).copied().unwrap_or(0) == 0 {
                        panic!(
                            "device reachability not closed: node {} -> child {} not reachable (tag={:?})",
                            i,
                            cid,
                            node
                        );
                    }
                }
            }
            PirNode::Decision {
                child_false,
                child_true,
                ..
            } => {
                let f = child_false.as_u32() as usize;
                let t = child_true.as_u32() as usize;
                if reachable.get(f).copied().unwrap_or(0) == 0 {
                    panic!(
                        "device reachability not closed: decision node {} -> false child {} not reachable",
                        i, f
                    );
                }
                if reachable.get(t).copied().unwrap_or(0) == 0 {
                    panic!(
                        "device reachability not closed: decision node {} -> true child {} not reachable",
                        i, t
                    );
                }
            }
            _ => {}
        }
    }

    let encoding = encode_cnf_gpu(&gpu_pir, &gpu_roots, &provider).expect("encode_cnf_gpu");

    let device = provider.device().inner();
    let mut node_vars = vec![0u32; encoding.vars.node_var.len()];
    device
        .dtoh_sync_copy_into(&encoding.vars.node_var, &mut node_vars)
        .unwrap();
    for (i, node) in provenance.pir.nodes().iter().enumerate() {
        if reachable.get(i).copied().unwrap_or(0) == 0 {
            continue;
        }
        if node_vars.get(i).copied().unwrap_or(0) == 0 {
            panic!("reachable node_var is zero: node {} tag={:?}", i, node);
        }
        match node {
            PirNode::And { children } | PirNode::Or { children } => {
                for child in children {
                    let cid = child.as_u32() as usize;
                    if node_vars.get(cid).copied().unwrap_or(0) == 0 {
                        panic!(
                            "reachable node has child with node_var=0: node {} -> child {} tag={:?}",
                            i,
                            cid,
                            node
                        );
                    }
                }
            }
            PirNode::Decision {
                child_false,
                child_true,
                ..
            } => {
                let f = child_false.as_u32() as usize;
                let t = child_true.as_u32() as usize;
                if node_vars.get(f).copied().unwrap_or(0) == 0 {
                    panic!(
                        "reachable decision node has false child with node_var=0: node {} -> child_false {}",
                        i, f
                    );
                }
                if node_vars.get(t).copied().unwrap_or(0) == 0 {
                    panic!(
                        "reachable decision node has true child with node_var=0: node {} -> child_true {}",
                        i, t
                    );
                }
            }
            _ => {}
        }
    }

    let mut num_vars = [0u32; 1];
    let mut num_clauses = [0u32; 1];
    let mut num_lits = [0u32; 1];
    device
        .dtoh_sync_copy_into(&encoding.cnf.num_vars, &mut num_vars)
        .unwrap();
    device
        .dtoh_sync_copy_into(&encoding.cnf.num_clauses, &mut num_clauses)
        .unwrap();
    device
        .dtoh_sync_copy_into(&encoding.cnf.num_lits, &mut num_lits)
        .unwrap();

    let nv = num_vars[0];
    let nc = num_clauses[0];
    let nl = num_lits[0];

    let offsets_view = encoding.cnf.clause_offsets.slice(0..(nc as usize + 1));
    let lits_view = encoding.cnf.literals.slice(0..(nl as usize));
    let mut offsets = vec![0u32; (nc as usize) + 1];
    let mut lits = vec![0i32; nl as usize];
    device
        .dtoh_sync_copy_into(&offsets_view, &mut offsets)
        .unwrap();
    device.dtoh_sync_copy_into(&lits_view, &mut lits).unwrap();

    if let Err(msg) = validate_dimacs_host(
        encoding.cnf.var_cap,
        encoding.cnf.clause_cap,
        encoding.cnf.lit_cap,
        nv,
        nc,
        nl,
        &offsets,
        &lits,
    ) {
        panic!(
            "CNF validation failed: {msg}\n  nv={nv} nc={nc} nl={nl}\n  var_cap={} clause_cap={} lit_cap={}\n  offsets[0..min(8,nc+1)]={:?}",
            encoding.cnf.var_cap,
            encoding.cnf.clause_cap,
            encoding.cnf.lit_cap,
            &offsets[..offsets.len().min(8)],
        );
    }
}
