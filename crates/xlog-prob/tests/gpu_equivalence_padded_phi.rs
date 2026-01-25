use std::sync::Arc;

use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

use xlog_prob::compilation::{validate_equivalence_gpu, GpuEquivalenceConfig};
use xlog_prob::gpu::GpuXgcf;
use xlog_prob::xgcf::{Xgcf, XgcfNodeType};

use xlog_solve::GpuCnf;

fn try_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024); // 1 GiB
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    match CudaKernelProvider::new(device, memory) {
        Ok(p) => Some(Arc::new(p)),
        Err(e) => {
            eprintln!("Skipping test: failed to create CUDA kernel provider: {}", e);
            None
        }
    }
}

#[test]
fn gpu_equivalence_accepts_padded_phi_caps() {
    let Some(provider) = try_provider() else {
        return;
    };

    // Construct a CNF where the device-resident exact size is 1 clause / 1 literal:
    //   phi_exact = (x0)
    //
    // But the allocated buffers have larger *capacities* that include a second clause:
    //   phi_padded = (x0) ∧ (¬x0)
    //
    // Production GPU-native builders may allocate capacities > exact sizes, so the verifier must
    // rely on device-resident `num_*` and ignore the padded region.
    let var_cap = 1u32;
    let clause_cap = 2u32;
    let lit_cap = 2u32;

    let memory = provider.memory();
    let device = provider.device().inner();

    let mut d_num_vars = memory.alloc::<u32>(1).expect("alloc num_vars");
    let mut d_num_clauses = memory.alloc::<u32>(1).expect("alloc num_clauses");
    let mut d_num_lits = memory.alloc::<u32>(1).expect("alloc num_lits");
    let mut d_offsets = memory
        .alloc::<u32>((clause_cap as usize) + 1)
        .expect("alloc offsets");
    let mut d_lits = memory.alloc::<i32>(lit_cap as usize).expect("alloc lits");

    device
        .htod_sync_copy_into(&[1u32], &mut d_num_vars)
        .expect("upload num_vars");
    device
        .htod_sync_copy_into(&[1u32], &mut d_num_clauses)
        .expect("upload num_clauses");
    device
        .htod_sync_copy_into(&[1u32], &mut d_num_lits)
        .expect("upload num_lits");
    device
        .htod_sync_copy_into(&[0u32, 1u32, 2u32], &mut d_offsets)
        .expect("upload offsets");
    device
        .htod_sync_copy_into(&[1i32, -1i32], &mut d_lits)
        .expect("upload lits");

    let phi = GpuCnf {
        var_cap,
        clause_cap,
        lit_cap,
        num_vars: d_num_vars,
        num_clauses: d_num_clauses,
        num_lits: d_num_lits,
        clause_offsets: d_offsets,
        literals: d_lits,
    };

    // C = x0 (single literal circuit).
    let xgcf = Xgcf {
        node_type: vec![XgcfNodeType::Lit],
        child_offsets: vec![0, 0],
        child_indices: vec![],
        lit: vec![1],
        decision_var: vec![0],
        decision_child_false: vec![0],
        decision_child_true: vec![0],
        roots: vec![0],
        level_offsets: vec![0, 1],
        level_nodes: vec![0],
    };
    let circuit = GpuXgcf::upload(&provider, &xgcf).expect("GpuXgcf upload");

    validate_equivalence_gpu(&phi, &circuit, &provider, GpuEquivalenceConfig::default())
        .expect("equivalence should hold even when phi capacities exceed exact sizes");
}

