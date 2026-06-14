// crates/xlog-runtime/tests/test_leader_input_permutation_tables.rs
//! Leader-input permutation table acceptance gate (7 runtime tests).
//!
//! These tests invoke the
//! `prepare_leader_inputs` helper directly with synthesized
//! `VariableOrder` values and assert the **observable
//! consequences** (pointer-identity is invalid because the helper
//! materializes owned buffers):
//!
//!   * **Per-slot schema** matches the locked permutation table
//!     (e.g., triangle e_yz-leader: slot 0 schema cols are
//!     `(Y, Z)`, slot 1 = `(Z, X)` after the col-swap, slot 2 =
//!     `(Y, X)` after the col-swap).
//!   * **Per-slot column content** matches a CPU-computed
//!     reference (download via `cuMemcpyDtoH_v2` → host vec
//!     equality with the expected swapped/rotated columns).
//!   * **`var_order.kernel_output_cols`** matches the locked
//!     `head_proj` from the table.
//!   * **`var_order.leader_idx`** equals the requested leader.
//!
//! Coverage: triangle 3 leaders (e_xy / e_yz / e_xz) + 4-cycle 4
//! leaders (e_wx / e_xy / e_yz / e_zw) = 7 tests.

use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{MemoryBudget, RuntimeConfig, ScalarType, Schema, XlogError};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LogRecord, LoggingResource,
    LoggingSink, SinkError, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_ir::rir::{LookupPerm, ProjectExpr, VariableOrder};
use xlog_runtime::Executor;

// ---------------------------------------------------------------
// Locked permutation tables — constructed by hand here (NOT
// through xlog-logic helpers) to keep this runtime slot-preparation cert
// independent of xlog-logic. Each test cross-checks against the
// locked permutation tables.
// ---------------------------------------------------------------

fn build_triangle_var_order(leader_idx: u8) -> VariableOrder {
    let (lookup_perms, kernel_output_cols) = match leader_idx {
        0 => (
            vec![
                LookupPerm {
                    input_idx: 1,
                    swap_cols: false,
                },
                LookupPerm {
                    input_idx: 2,
                    swap_cols: false,
                },
            ],
            vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(2),
            ],
        ),
        1 => (
            vec![
                LookupPerm {
                    input_idx: 2,
                    swap_cols: true,
                },
                LookupPerm {
                    input_idx: 0,
                    swap_cols: true,
                },
            ],
            vec![
                ProjectExpr::Column(2),
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
            ],
        ),
        2 => (
            vec![
                LookupPerm {
                    input_idx: 1,
                    swap_cols: true,
                },
                LookupPerm {
                    input_idx: 0,
                    swap_cols: false,
                },
            ],
            vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(2),
                ProjectExpr::Column(1),
            ],
        ),
        _ => panic!("triangle leader_idx must be in [0, 3): got {leader_idx}"),
    };
    VariableOrder::legacy(leader_idx, lookup_perms, kernel_output_cols)
}

fn build_cycle4_var_order(leader_idx: u8) -> VariableOrder {
    if leader_idx >= 4 {
        panic!("4-cycle leader_idx must be in [0, 4): got {leader_idx}");
    }
    let lookup_perms: Vec<LookupPerm> = (1..4)
        .map(|k| LookupPerm {
            input_idx: ((leader_idx as usize + k) % 4) as u8,
            swap_cols: false,
        })
        .collect();
    let kernel_output_cols = match leader_idx {
        0 => vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(2),
            ProjectExpr::Column(3),
        ],
        1 => vec![
            ProjectExpr::Column(3),
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(2),
        ],
        2 => vec![
            ProjectExpr::Column(2),
            ProjectExpr::Column(3),
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
        ],
        3 => vec![
            ProjectExpr::Column(1),
            ProjectExpr::Column(2),
            ProjectExpr::Column(3),
            ProjectExpr::Column(0),
        ],
        _ => unreachable!(),
    };
    VariableOrder::legacy(leader_idx, lookup_perms, kernel_output_cols)
}

// ---------------------------------------------------------------
// Fixture infrastructure for runtime-backed slot-preparation tests.
// ---------------------------------------------------------------

struct DiscardSink;
impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

#[allow(dead_code)]
struct RuntimeBackedFixture {
    device: Arc<CudaDevice>,
    runtime: Arc<XlogDeviceRuntime>,
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    pool: Arc<StreamPool>,
    executor: Executor,
}

fn make_runtime_fixture() -> Option<RuntimeBackedFixture> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
    );
    let logging: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(LoggingResource::new(
        async_resource,
        Arc::new(DiscardSink) as Arc<dyn LoggingSink>,
    ));
    let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
        Box::new(GlobalDeviceBudget::new(logging, 64 * 1024 * 1024));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        Arc::clone(&pool),
        budget,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        MemoryBudget::with_limit(64 * 1024 * 1024),
        Arc::clone(&runtime),
    ));
    let provider =
        Arc::new(CudaKernelProvider::with_runtime(Arc::clone(&device), Arc::clone(&memory)).ok()?);
    let executor = Executor::new_with_config(Arc::clone(&provider), RuntimeConfig::default());
    Some(RuntimeBackedFixture {
        device,
        runtime,
        memory,
        provider,
        pool,
        executor,
    })
}

fn upload_binary_named(
    memory: &Arc<GpuMemoryManager>,
    rows: &[(u32, u32)],
    name_a: &str,
    name_b: &str,
) -> CudaBuffer {
    let n = rows.len() as u32;
    let bytes_per_col = (n as usize) * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc d_num_rows");
    let device = memory.device().inner();
    if !rows.is_empty() {
        let bs0: Vec<u8> = rows.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
        let bs1: Vec<u8> = rows.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
        device.htod_sync_copy_into(&bs0, &mut col0).unwrap();
        device.htod_sync_copy_into(&bs1, &mut col1).unwrap();
    }
    device.htod_sync_copy_into(&[n], &mut d_num_rows).unwrap();
    let schema = Schema::new(vec![
        (name_a.to_string(), ScalarType::U32),
        (name_b.to_string(), ScalarType::U32),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        n as u64,
        d_num_rows,
        schema,
        n,
    )
}

fn download_pairs(buf: &CudaBuffer) -> Vec<(u32, u32)> {
    let n = buf.cached_row_count().unwrap() as usize;
    if n == 0 {
        return Vec::new();
    }
    assert_eq!(buf.arity(), 2);
    let mut col0 = vec![0u8; n * 4];
    let mut col1 = vec![0u8; n * 4];
    unsafe {
        sys::cuMemcpyDtoH_v2(
            col0.as_mut_ptr() as *mut _,
            *buf.column(0).unwrap().device_ptr(),
            col0.len(),
        );
        sys::cuMemcpyDtoH_v2(
            col1.as_mut_ptr() as *mut _,
            *buf.column(1).unwrap().device_ptr(),
            col1.len(),
        );
    }
    (0..n)
        .map(|i| {
            let a = u32::from_le_bytes(col0[i * 4..i * 4 + 4].try_into().unwrap());
            let b = u32::from_le_bytes(col1[i * 4..i * 4 + 4].try_into().unwrap());
            (a, b)
        })
        .collect()
}

fn schema_names(buf: &CudaBuffer) -> Vec<String> {
    buf.schema()
        .columns
        .iter()
        .map(|(n, _)| n.clone())
        .collect()
}

// ---------------------------------------------------------------
// Triangle dispatch routing per leader (3 tests)
//
// Canonical promoter inputs `[e_xy, e_yz, e_xz]`. The fixtures
// pre-stamp schema names that encode the variable each column
// represents, so per-slot schema assertions can name-check the
// rotated/swapped layout.
// ---------------------------------------------------------------

fn triangle_canonical_inputs(memory: &Arc<GpuMemoryManager>) -> [CudaBuffer; 3] {
    let e_xy = upload_binary_named(memory, &[(1, 2), (3, 4)], "X", "Y");
    let e_yz = upload_binary_named(memory, &[(2, 5), (4, 6)], "Y", "Z");
    let e_xz = upload_binary_named(memory, &[(1, 5), (3, 6)], "X", "Z");
    [e_xy, e_yz, e_xz]
}

#[test]
fn triangle_e_xy_default_leader_slot_permutation() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let canonical = triangle_canonical_inputs(&fix.memory);
    let canonical_refs: [&CudaBuffer; 3] = [&canonical[0], &canonical[1], &canonical[2]];
    let var_order = build_triangle_var_order(0);
    // Locked: leader_idx = 0, kernel_output_cols = [Col(0), Col(1), Col(2)].
    assert_eq!(var_order.leader_idx, 0);
    assert_eq!(
        var_order.kernel_output_cols,
        vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(2),
        ]
    );
    let stream = fix
        .executor
        .wcoj_dispatch_stream_or_init()
        .expect("stream id");
    let slots = fix
        .executor
        .prepare_leader_inputs(&canonical_refs, &var_order, stream)
        .expect("prepare must succeed");
    assert_eq!(slots.len(), 3);
    // Slot 0 = e_xy (X, Y) unchanged.
    assert_eq!(schema_names(&slots[0]), vec!["X", "Y"]);
    assert_eq!(download_pairs(&slots[0]), vec![(1, 2), (3, 4)]);
    // Slot 1 = e_yz (Y, Z).
    assert_eq!(schema_names(&slots[1]), vec!["Y", "Z"]);
    assert_eq!(download_pairs(&slots[1]), vec![(2, 5), (4, 6)]);
    // Slot 2 = e_xz (X, Z).
    assert_eq!(schema_names(&slots[2]), vec!["X", "Z"]);
    assert_eq!(download_pairs(&slots[2]), vec![(1, 5), (3, 6)]);
}

#[test]
fn triangle_e_yz_leader_slot_permutation() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let canonical = triangle_canonical_inputs(&fix.memory);
    let canonical_refs: [&CudaBuffer; 3] = [&canonical[0], &canonical[1], &canonical[2]];
    let var_order = build_triangle_var_order(1);
    // Locked table for e_yz leader: slots [e_yz, e_xz↔, e_xy↔],
    // kernel_output_cols = [Col(2), Col(0), Col(1)].
    assert_eq!(var_order.leader_idx, 1);
    assert_eq!(
        var_order.kernel_output_cols,
        vec![
            ProjectExpr::Column(2),
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
        ]
    );
    assert_eq!(
        var_order.lookup_perms,
        vec![
            LookupPerm {
                input_idx: 2,
                swap_cols: true
            },
            LookupPerm {
                input_idx: 0,
                swap_cols: true
            },
        ]
    );
    let stream = fix
        .executor
        .wcoj_dispatch_stream_or_init()
        .expect("stream id");
    let slots = fix
        .executor
        .prepare_leader_inputs(&canonical_refs, &var_order, stream)
        .expect("prepare must succeed");
    assert_eq!(slots.len(), 3);
    // Slot 0 = e_yz unchanged → (Y, Z).
    assert_eq!(schema_names(&slots[0]), vec!["Y", "Z"]);
    assert_eq!(download_pairs(&slots[0]), vec![(2, 5), (4, 6)]);
    // Slot 1 = e_xz with cols swapped → (Z, X).
    assert_eq!(schema_names(&slots[1]), vec!["Z", "X"]);
    assert_eq!(download_pairs(&slots[1]), vec![(5, 1), (6, 3)]);
    // Slot 2 = e_xy with cols swapped → (Y, X).
    assert_eq!(schema_names(&slots[2]), vec!["Y", "X"]);
    assert_eq!(download_pairs(&slots[2]), vec![(2, 1), (4, 3)]);
}

#[test]
fn triangle_e_xz_leader_slot_permutation() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let canonical = triangle_canonical_inputs(&fix.memory);
    let canonical_refs: [&CudaBuffer; 3] = [&canonical[0], &canonical[1], &canonical[2]];
    let var_order = build_triangle_var_order(2);
    // Locked: slots [e_xz, e_yz↔, e_xy], kernel_output_cols =
    // [Col(0), Col(2), Col(1)].
    assert_eq!(var_order.leader_idx, 2);
    assert_eq!(
        var_order.kernel_output_cols,
        vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(2),
            ProjectExpr::Column(1),
        ]
    );
    assert_eq!(
        var_order.lookup_perms,
        vec![
            LookupPerm {
                input_idx: 1,
                swap_cols: true
            },
            LookupPerm {
                input_idx: 0,
                swap_cols: false
            },
        ]
    );
    let stream = fix
        .executor
        .wcoj_dispatch_stream_or_init()
        .expect("stream id");
    let slots = fix
        .executor
        .prepare_leader_inputs(&canonical_refs, &var_order, stream)
        .expect("prepare must succeed");
    assert_eq!(slots.len(), 3);
    // Slot 0 = e_xz unchanged → (X, Z).
    assert_eq!(schema_names(&slots[0]), vec!["X", "Z"]);
    assert_eq!(download_pairs(&slots[0]), vec![(1, 5), (3, 6)]);
    // Slot 1 = e_yz with cols swapped → (Z, Y).
    assert_eq!(schema_names(&slots[1]), vec!["Z", "Y"]);
    assert_eq!(download_pairs(&slots[1]), vec![(5, 2), (6, 4)]);
    // Slot 2 = e_xy unchanged → (X, Y).
    assert_eq!(schema_names(&slots[2]), vec!["X", "Y"]);
    assert_eq!(download_pairs(&slots[2]), vec![(1, 2), (3, 4)]);
}

#[test]
fn prepare_leader_inputs_rejects_out_of_range_lookup_perm() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let canonical = triangle_canonical_inputs(&fix.memory);
    let canonical_refs: [&CudaBuffer; 3] = [&canonical[0], &canonical[1], &canonical[2]];
    let var_order = VariableOrder::legacy(
        0,
        vec![
            LookupPerm {
                input_idx: 3,
                swap_cols: false,
            },
            LookupPerm {
                input_idx: 1,
                swap_cols: false,
            },
        ],
        vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(2),
        ],
    );
    let stream = fix
        .executor
        .wcoj_dispatch_stream_or_init()
        .expect("stream id");
    let err = match fix
        .executor
        .prepare_leader_inputs(&canonical_refs, &var_order, stream)
    {
        Ok(_) => panic!("out-of-range lookup input must reject before indexing canonical inputs"),
        Err(err) => err,
    };

    match err {
        XlogError::Kernel(message) => {
            assert!(message.contains("lookup_perms[0].input_idx 3 out of range for arity 3"));
        }
        other => panic!("expected kernel error for malformed lookup perm, got {other:?}"),
    }
}

// ---------------------------------------------------------------
// 4-cycle dispatch routing per leader (4 tests, all rotation-only)
//
// Canonical promoter inputs `[e_wx, e_xy, e_yz, e_zw]`.
// ---------------------------------------------------------------

fn cycle4_canonical_inputs(memory: &Arc<GpuMemoryManager>) -> [CudaBuffer; 4] {
    let e_wx = upload_binary_named(memory, &[(10, 20)], "W", "X");
    let e_xy = upload_binary_named(memory, &[(20, 30)], "X", "Y");
    let e_yz = upload_binary_named(memory, &[(30, 40)], "Y", "Z");
    let e_zw = upload_binary_named(memory, &[(40, 10)], "Z", "W");
    [e_wx, e_xy, e_yz, e_zw]
}

fn assert_cycle4_slot_layout(
    fix: &RuntimeBackedFixture,
    leader_idx: u8,
    expected_kernel_output_cols: &[ProjectExpr; 4],
    // expected_slot_schemas[i] = (name_a, name_b) for slot i after rotation.
    expected_slot_schemas: &[(&str, &str); 4],
    // Each slot is rotation-only, so the content is the corresponding
    // canonical input verbatim.
    expected_slot_rows: &[&[(u32, u32)]; 4],
) {
    let canonical = cycle4_canonical_inputs(&fix.memory);
    let canonical_refs: [&CudaBuffer; 4] =
        [&canonical[0], &canonical[1], &canonical[2], &canonical[3]];
    let var_order = build_cycle4_var_order(leader_idx);
    assert_eq!(var_order.leader_idx, leader_idx);
    assert_eq!(var_order.kernel_output_cols, expected_kernel_output_cols);
    // 4-cycle is rotation-only: every lookup_perm must have swap == false.
    assert!(var_order.lookup_perms.iter().all(|p| !p.swap_cols));
    let stream = fix
        .executor
        .wcoj_dispatch_stream_or_init()
        .expect("stream id");
    let slots = fix
        .executor
        .prepare_leader_inputs(&canonical_refs, &var_order, stream)
        .expect("prepare must succeed");
    assert_eq!(slots.len(), 4);
    for i in 0..4 {
        assert_eq!(
            schema_names(&slots[i]),
            vec![
                expected_slot_schemas[i].0.to_string(),
                expected_slot_schemas[i].1.to_string()
            ],
            "slot {} schema mismatch",
            i
        );
        assert_eq!(
            download_pairs(&slots[i]),
            expected_slot_rows[i].to_vec(),
            "slot {} content mismatch",
            i
        );
    }
}

#[test]
fn cycle4_e_wx_default_leader_slot_rotation() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    assert_cycle4_slot_layout(
        &fix,
        0,
        &[
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(2),
            ProjectExpr::Column(3),
        ],
        &[("W", "X"), ("X", "Y"), ("Y", "Z"), ("Z", "W")],
        &[&[(10, 20)], &[(20, 30)], &[(30, 40)], &[(40, 10)]],
    );
}

#[test]
fn cycle4_e_xy_leader_slot_rotation() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    assert_cycle4_slot_layout(
        &fix,
        1,
        &[
            ProjectExpr::Column(3),
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(2),
        ],
        &[("X", "Y"), ("Y", "Z"), ("Z", "W"), ("W", "X")],
        &[&[(20, 30)], &[(30, 40)], &[(40, 10)], &[(10, 20)]],
    );
}

#[test]
fn cycle4_e_yz_leader_slot_rotation() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    assert_cycle4_slot_layout(
        &fix,
        2,
        &[
            ProjectExpr::Column(2),
            ProjectExpr::Column(3),
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
        ],
        &[("Y", "Z"), ("Z", "W"), ("W", "X"), ("X", "Y")],
        &[&[(30, 40)], &[(40, 10)], &[(10, 20)], &[(20, 30)]],
    );
}

#[test]
fn cycle4_e_zw_leader_slot_rotation() {
    let Some(fix) = make_runtime_fixture() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    assert_cycle4_slot_layout(
        &fix,
        3,
        &[
            ProjectExpr::Column(1),
            ProjectExpr::Column(2),
            ProjectExpr::Column(3),
            ProjectExpr::Column(0),
        ],
        &[("Z", "W"), ("W", "X"), ("X", "Y"), ("Y", "Z")],
        &[&[(40, 10)], &[(10, 20)], &[(20, 30)], &[(30, 40)]],
    );
}
