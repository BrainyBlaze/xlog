//! Tests for the ILP CUDA kernel (extract_nonzero_indices)

mod common;
use common::setup_provider;
use xlog_core::{ScalarType, Schema};
use xlog_cuda::CudaKernelProvider;

fn make_mask_buffer(provider: &CudaKernelProvider, data: &[f32]) -> xlog_cuda::CudaBuffer {
    let schema = Schema::new(vec![("c0".to_string(), ScalarType::F32)]);
    provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(data)], schema)
        .expect("create mask buffer")
}

#[test]
fn test_extract_nonzero_3x3x3_single_active() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let n = 3;
    let total = n * n * n; // 27
    let mut hard = vec![0.0f32; total];
    let mut soft = vec![0.0f32; total];

    // Set (i=0, j=1, k=2) active: flat index = 0*9 + 1*3 + 2 = 5
    hard[5] = 1.0;
    soft[5] = 0.9;

    let hard_buf = make_mask_buffer(&provider, &hard);
    let soft_buf = make_mask_buffer(&provider, &soft);

    let result = provider
        .extract_active_rule_indices(&hard_buf, &soft_buf, n, 32)
        .expect("kernel launch");

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], (0, 1, 2));
}

#[test]
fn test_extract_nonzero_budget_cap_top_priority() {
    // RFC T2.3: 50 non-zeros, max=10 → top 10 by soft-mask priority.
    // We use N=4 (64 total) with 50 active, cap at 10, and verify the
    // returned entries are exactly the 10 with highest soft-mask values.
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let n = 4;
    let total = n * n * n; // 64
    let mut hard = vec![0.0f32; total];
    // Assign distinct priorities so top-K is deterministic
    let mut soft = vec![0.0f32; total];
    // Activate exactly 50 elements (indices 0..50)
    for idx in 0..50 {
        hard[idx] = 1.0;
        soft[idx] = (idx + 1) as f32; // priority 1..50
    }

    let hard_buf = make_mask_buffer(&provider, &hard);
    let soft_buf = make_mask_buffer(&provider, &soft);

    // Budget cap = 10
    let result = provider
        .extract_active_rule_indices(&hard_buf, &soft_buf, n, 10)
        .expect("kernel launch");

    assert_eq!(result.len(), 10, "Budget cap must truncate to 10");

    // The top 10 by priority should be flat indices 40..49 (priority 41..50).
    // Convert returned (i,j,k) back to flat indices and verify they are
    // the 10 highest-priority entries.
    let flat_indices: Vec<usize> = result
        .iter()
        .map(|(i, j, k)| (*i as usize) * n * n + (*j as usize) * n + (*k as usize))
        .collect();
    for &fi in &flat_indices {
        assert!(
            (40..50).contains(&fi),
            "Expected top-10 entries (flat indices 40..49), got {}",
            fi
        );
    }
}

#[test]
fn test_extract_nonzero_empty_mask() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let n = 3;
    let total = n * n * n;
    let hard = vec![0.0f32; total];
    let soft = vec![0.0f32; total];

    let hard_buf = make_mask_buffer(&provider, &hard);
    let soft_buf = make_mask_buffer(&provider, &soft);

    let result = provider
        .extract_active_rule_indices(&hard_buf, &soft_buf, n, 32)
        .expect("kernel launch");

    assert!(result.is_empty());
}

// T2.4: ILP module loads successfully
#[test]
fn test_ilp_module_loads() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    use xlog_cuda::provider::{ilp_kernels, ILP_MODULE};
    let func = provider
        .device()
        .inner()
        .get_func(ILP_MODULE, ilp_kernels::EXTRACT_NONZERO_INDICES);
    assert!(
        func.is_some(),
        "extract_nonzero_indices kernel must be loadable"
    );
}

// T2.2: Multi-element extraction
#[test]
fn test_extract_nonzero_multiple_active() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let n = 3;
    let total = n * n * n;
    let mut hard = vec![0.0f32; total];
    let mut soft = vec![0.0f32; total];

    // Activate 3 entries with distinct priorities
    // (0,1,2) at flat 5, priority 0.9
    hard[5] = 1.0;
    soft[5] = 0.9;
    // (1,0,1) at flat 10, priority 0.5
    hard[10] = 1.0;
    soft[10] = 0.5;
    // (2,2,0) at flat 24, priority 0.8
    hard[24] = 1.0;
    soft[24] = 0.8;

    let hard_buf = make_mask_buffer(&provider, &hard);
    let soft_buf = make_mask_buffer(&provider, &soft);

    let result = provider
        .extract_active_rule_indices(&hard_buf, &soft_buf, n, 32)
        .expect("kernel launch");

    assert_eq!(result.len(), 3);
    // Sorted by priority descending: (0,1,2)=0.9, (2,2,0)=0.8, (1,0,1)=0.5
    assert_eq!(result[0], (0, 1, 2));
    assert_eq!(result[1], (2, 2, 0));
    assert_eq!(result[2], (1, 0, 1));
}
