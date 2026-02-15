//! GPU-only exact compilation helpers (no host reads in this module).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use cudarc::driver::DeviceSlice;
use xlog_core::{MemoryBudget, Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

use crate::compilation::gpu_cache::{GpuCircuitCache, GpuCircuitCacheHandle};
use crate::compilation::gpu_weights::{
    apply_query_vars_device, build_evidence_by_var_gpu, build_weights_gpu, map_nodes_to_vars_gpu,
    restore_query_vars_device, GpuWeights,
};
use crate::compilation::{
    compile_gpu_d4_and_verify_cached, encode_cnf_gpu, DeviceRandomVarList, GpuPirGraph, GpuPirRoots,
};
use crate::exact::{
    build_weight_sources, collect_random_vars_device, default_cache_config, default_compile_config,
    upload_f64, upload_u32, upload_u8, GpuConfig,
};
use crate::provenance::{GroundAtom, Provenance};

pub struct ExactGpuState {
    provider: Option<Arc<CudaKernelProvider>>,
    cache: Option<Mutex<GpuCircuitCache>>,
    handle: Option<GpuCircuitCacheHandle>,
    weights: Option<GpuWeights>,
    max_var: u32,
    query_vars_device: Option<TrackedCudaSlice<u32>>,
    query_indices: Vec<usize>,
    queries: Vec<GroundAtom>,
}

impl ExactGpuState {
    fn empty(queries: Vec<GroundAtom>) -> Self {
        Self {
            provider: None,
            cache: None,
            handle: None,
            weights: None,
            max_var: 0,
            query_vars_device: None,
            query_indices: Vec::new(),
            queries,
        }
    }

    pub fn provider(&self) -> Option<&Arc<CudaKernelProvider>> {
        self.provider.as_ref()
    }

    pub fn lock_cache(&self) -> Option<std::sync::MutexGuard<'_, GpuCircuitCache>> {
        self.cache.as_ref().map(|cache| {
            cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        })
    }

    pub fn handle(&self) -> Option<&GpuCircuitCacheHandle> {
        self.handle.as_ref()
    }

    pub fn weights(&self) -> Option<&GpuWeights> {
        self.weights.as_ref()
    }

    pub fn max_var(&self) -> u32 {
        self.max_var
    }

    pub fn query_vars_device(&self) -> Option<&TrackedCudaSlice<u32>> {
        self.query_vars_device.as_ref()
    }

    pub fn query_indices(&self) -> &[usize] {
        &self.query_indices
    }

    pub fn queries(&self) -> &[GroundAtom] {
        &self.queries
    }

    pub fn allocate_query_restore(&self) -> Result<Option<TrackedCudaSlice<f64>>> {
        let Some(provider) = self.provider.as_ref() else {
            return Ok(None);
        };
        let Some(query_vars) = self.query_vars_device.as_ref() else {
            return Ok(None);
        };
        let buf = provider.memory().alloc::<f64>(query_vars.len())?;
        Ok(Some(buf))
    }

    pub fn apply_query_vars(
        &self,
        cache: &mut GpuCircuitCache,
        saved: &mut TrackedCudaSlice<f64>,
    ) -> Result<()> {
        let Some(provider) = self.provider.as_ref() else {
            return Ok(());
        };
        let Some(query_vars) = self.query_vars_device.as_ref() else {
            return Ok(());
        };
        let (_, log_false) = cache.var_log_weights_mut();
        apply_query_vars_device(provider, query_vars, self.max_var, log_false, saved)
    }

    pub fn restore_query_vars(
        &self,
        cache: &mut GpuCircuitCache,
        saved: &TrackedCudaSlice<f64>,
    ) -> Result<()> {
        let Some(provider) = self.provider.as_ref() else {
            return Ok(());
        };
        let Some(query_vars) = self.query_vars_device.as_ref() else {
            return Ok(());
        };
        let (_, log_false) = cache.var_log_weights_mut();
        restore_query_vars_device(provider, query_vars, self.max_var, log_false, saved)
    }
}

pub fn compile_provenance_gpu_only(
    provenance: &Provenance,
    config: GpuConfig,
) -> Result<ExactGpuState> {
    if config.memory_bytes == 0 {
        return Err(XlogError::Kernel(
            "GPU memory budget must be non-zero".to_string(),
        ));
    }

    let mut roots_set: HashSet<crate::pir::PirNodeId> = HashSet::new();
    let mut evidence_formulas: Vec<(crate::pir::PirNodeId, bool, GroundAtom)> = Vec::new();
    let mut evidence_atoms: HashMap<GroundAtom, bool> = HashMap::new();
    for (atom, value) in &provenance.evidence {
        if let Some(prev) = evidence_atoms.insert(atom.clone(), *value) {
            if prev != *value {
                return Err(XlogError::Execution(format!(
                    "Exact inference error: conflicting evidence for {}",
                    display_atom(atom)
                )));
            }
        }

        let formula = provenance.query_formula(&atom.predicate, &atom.args);
        match formula {
            Some(id) => {
                roots_set.insert(id);
                evidence_formulas.push((id, *value, atom.clone()));
            }
            None => {
                if *value {
                    return Err(XlogError::Execution(format!(
                        "Exact inference error: evidence atom is never derivable: {}",
                        display_atom(atom)
                    )));
                }
            }
        }
    }

    let mut queries: Vec<GroundAtom> = Vec::new();
    let mut query_nodes: Vec<(usize, crate::pir::PirNodeId)> = Vec::new();
    for atom in &provenance.queries {
        let formula = provenance.query_formula(&atom.predicate, &atom.args);
        if let Some(id) = formula {
            roots_set.insert(id);
            query_nodes.push((queries.len(), id));
        }
        queries.push(atom.clone());
    }

    // Ensure ALL probabilistic variable nodes (Decision, Lit, NegLit) are reachable
    // so they get CNF variables. Required for template/neural fast-path slot mapping.
    for (idx, node) in provenance.pir.nodes().iter().enumerate() {
        match node {
            crate::pir::PirNode::Decision { .. }
            | crate::pir::PirNode::Lit { .. }
            | crate::pir::PirNode::NegLit { .. } => {
                roots_set.insert(crate::pir::PirNodeId::from_u32(idx as u32));
            }
            _ => {}
        }
    }

    let mut roots: Vec<crate::pir::PirNodeId> = roots_set.into_iter().collect();
    roots.sort();

    if roots.is_empty() {
        return Ok(ExactGpuState::empty(queries));
    }

    let device = Arc::new(CudaDevice::new(config.device_ordinal)?);
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(config.memory_bytes),
    ));
    let provider = Arc::new(CudaKernelProvider::new(device, memory)?);

    let gpu_pir = GpuPirGraph::from_host(&provenance.pir, &provider)?;
    let gpu_roots = GpuPirRoots::from_host(&roots, &provider)?;
    let encoding = encode_cnf_gpu(&gpu_pir, &gpu_roots, &provider)?;
    if encoding.vars.max_var != encoding.cnf.var_cap {
        return Err(XlogError::Compilation(format!(
            "Exact inference error: CNF var_cap {} != vars.max_var {}",
            encoding.cnf.var_cap, encoding.vars.max_var
        )));
    }

    let (leaf_probs_host, choice_true_host, choice_false_host) = build_weight_sources(provenance)?;
    let leaf_probs = upload_f64(&provider, &leaf_probs_host)?;
    let choice_true = upload_f64(&provider, &choice_true_host)?;
    let choice_false = upload_f64(&provider, &choice_false_host)?;

    let evidence_by_var = if evidence_formulas.is_empty() {
        let mut evidence = provider
            .memory()
            .alloc::<u8>((encoding.vars.max_var as usize) + 1)?;
        provider
            .device()
            .inner()
            .memset_zeros(&mut evidence)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero evidence buffer: {}", e)))?;
        evidence
    } else {
        let mut nodes: Vec<u32> = Vec::with_capacity(evidence_formulas.len());
        let mut vals: Vec<u8> = Vec::with_capacity(evidence_formulas.len());
        for (node, value, _atom) in &evidence_formulas {
            nodes.push(node.as_u32());
            vals.push(if *value { 1u8 } else { 2u8 });
        }
        let evidence_nodes = upload_u32(&provider, &nodes)?;
        let evidence_vals = upload_u8(&provider, &vals)?;
        build_evidence_by_var_gpu(
            &encoding.vars.node_var,
            &evidence_nodes,
            &evidence_vals,
            encoding.vars.max_var,
            &provider,
        )?
    };

    let weights = build_weights_gpu(
        &encoding.vars,
        &leaf_probs,
        &choice_true,
        &choice_false,
        &evidence_by_var,
        &provider,
    )?;

    let random_var_count = leaf_probs_host
        .len()
        .checked_add(choice_true_host.len())
        .ok_or_else(|| XlogError::Compilation("random var count overflow".to_string()))?;
    let random_var_count = u32::try_from(random_var_count)
        .map_err(|_| XlogError::Compilation("random var count exceeds u32".to_string()))?;
    let num_leaf_probs = u32::try_from(leaf_probs_host.len())
        .map_err(|_| XlogError::Compilation("leaf_probs count exceeds u32".to_string()))?;
    let num_choice_probs = u32::try_from(choice_true_host.len())
        .map_err(|_| XlogError::Compilation("choice_probs count exceeds u32".to_string()))?;
    let (random_var_list, actual_random_var_count) = collect_random_vars_device(
        &provider,
        &encoding.vars,
        num_leaf_probs,
        num_choice_probs,
        random_var_count,
    )?;
    let random_vars = DeviceRandomVarList::from_device(random_var_list, actual_random_var_count)?;
    let compile_config = default_compile_config(&encoding.cnf, config.memory_bytes)?;
    let cache_config = default_cache_config(&encoding.cnf, &compile_config)?;

    let mut cache = GpuCircuitCache::new(&provider, cache_config)?;
    let handle = compile_gpu_d4_and_verify_cached(
        &encoding.cnf,
        &encoding.decision_var_limit,
        &provider,
        &compile_config,
        &mut cache,
        &random_vars,
    )?;
    cache.store_weights(&handle, &weights.log_true, &weights.log_false)?;

    let (query_indices, query_vars_device) = if query_nodes.is_empty() {
        (Vec::new(), None)
    } else {
        let mut node_ids: Vec<u32> = Vec::with_capacity(query_nodes.len());
        let mut indices: Vec<usize> = Vec::with_capacity(query_nodes.len());
        for (idx, node) in &query_nodes {
            indices.push(*idx);
            node_ids.push(node.as_u32());
        }
        let node_ids_device = upload_u32(&provider, &node_ids)?;
        let vars_device = map_nodes_to_vars_gpu(
            &encoding.vars.node_var,
            &node_ids_device,
            encoding.vars.max_var,
            &provider,
        )?;
        (indices, Some(vars_device))
    };

    Ok(ExactGpuState {
        provider: Some(provider),
        cache: Some(Mutex::new(cache)),
        handle: Some(handle),
        weights: Some(weights),
        max_var: encoding.vars.max_var,
        query_vars_device,
        query_indices,
        queries,
    })
}

fn display_atom(atom: &GroundAtom) -> String {
    if atom.args.is_empty() {
        format!("{}()", atom.predicate)
    } else {
        format!("{}({} args)", atom.predicate, atom.args.len())
    }
}
