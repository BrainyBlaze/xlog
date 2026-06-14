use std::collections::HashMap;
use std::sync::Arc;

use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{
    AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, StreamPool, XlogDeviceRuntime,
};
use xlog_cuda::memory::GpuMemoryManager;
use xlog_cuda::{CudaDevice, CudaKernelProvider};
use xlog_gpu::logic::LogicProgram;
use xlog_runtime::RelationStore;

struct EnvGuard {
    old_hash_join: Option<String>,
    old_csm: Option<String>,
    old_graph: Option<String>,
}

impl EnvGuard {
    fn recursive_setop_profile() -> Self {
        let old_hash_join = std::env::var("XLOG_USE_RECORDED_HASH_JOIN").ok();
        let old_csm = std::env::var("XLOG_USE_RECORDED_CSM").ok();
        let old_graph = std::env::var("XLOG_USE_CSM_CUDA_GRAPH").ok();
        std::env::set_var("XLOG_USE_RECORDED_HASH_JOIN", "1");
        std::env::set_var("XLOG_USE_RECORDED_CSM", "1");
        std::env::set_var("XLOG_USE_CSM_CUDA_GRAPH", "0");
        Self {
            old_hash_join,
            old_csm,
            old_graph,
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        restore_env("XLOG_USE_RECORDED_HASH_JOIN", &self.old_hash_join);
        restore_env("XLOG_USE_RECORDED_CSM", &self.old_csm);
        restore_env("XLOG_USE_CSM_CUDA_GRAPH", &self.old_graph);
    }
}

fn restore_env(key: &str, old: &Option<String>) {
    match old {
        Some(value) => std::env::set_var(key, value),
        None => std::env::remove_var(key),
    }
}

fn provider() -> Option<Arc<CudaKernelProvider>> {
    let device = Arc::new(CudaDevice::new(0).ok()?);
    let stream_pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
    let budget = MemoryBudget::with_limit(2 * 1024 * 1024 * 1024);
    let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
        AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&stream_pool)),
    );
    let budgeted: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(GlobalDeviceBudget::new(
        async_resource,
        budget.device_bytes as usize,
    ));
    let runtime = Arc::new(XlogDeviceRuntime::with_resource(
        Arc::clone(&device),
        0,
        stream_pool,
        budgeted,
    ));
    let memory = Arc::new(GpuMemoryManager::with_runtime(
        Arc::clone(&device),
        budget,
        runtime,
    ));
    Some(Arc::new(
        CudaKernelProvider::with_runtime(device, memory).ok()?,
    ))
}

fn schema3() -> Schema {
    Schema::new(vec![
        ("c0".to_string(), ScalarType::I64),
        ("c1".to_string(), ScalarType::I64),
        ("c2".to_string(), ScalarType::I64),
    ])
}

fn schema1() -> Schema {
    Schema::new(vec![("c0".to_string(), ScalarType::I64)])
}

fn upload_i64(
    provider: &CudaKernelProvider,
    cols: &[Vec<i64>],
    schema: Schema,
) -> xlog_core::Result<xlog_cuda::CudaBuffer> {
    let bytes: Vec<Vec<u8>> = cols
        .iter()
        .map(|col| col.iter().flat_map(|v| v.to_le_bytes()).collect())
        .collect();
    let slices: Vec<&[u8]> = bytes.iter().map(Vec::as_slice).collect();
    provider.create_buffer_from_slices(&slices, schema)
}

#[test]
fn recursive_union_does_not_rededup_union_output() {
    let Some(provider) = provider() else {
        eprintln!("CUDA unavailable; skipping");
        return;
    };
    let source = r#"
pred wmir_committed(i64, i64, i64).
pred wmir_body_0(i64, i64, i64).
pred wmir_body_1(i64, i64, i64).
pred wmir_len_2(i64).
pred usable(i64, i64, i64).
pred support_2(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64).

usable(P, A0, A1) :- wmir_committed(P, A0, A1).

support_2(Head, V0, V1, RId, Body0Pred, V0, V2, Body1Pred, V2, V1) :-
    wmir_len_2(RId),
    wmir_body_0(RId, Head, Body0Pred),
    usable(Body0Pred, V0, V2),
    wmir_body_1(RId, Head, Body1Pred),
    usable(Body1Pred, V2, V1).

usable(Head, A0, A1) :- support_2(Head, A0, A1, _, _, _, _, _, _, _).

?- support_2(H, A0, A1, R, Body0Pred, Body0Arg0, Body0Arg1, Body1Pred, Body1Arg0, Body1Arg1).
?- usable(P, A0, A1).
"#;
    let program = LogicProgram::compile(source).expect("compile");
    let mut store = RelationStore::new(Arc::clone(&provider));
    let rows = 96u64;
    let mut pred = Vec::with_capacity((rows * 2) as usize);
    let mut arg0 = Vec::with_capacity((rows * 2) as usize);
    let mut arg1 = Vec::with_capacity((rows * 2) as usize);
    for y in 0..rows {
        pred.push(1i64);
        arg0.push(100_000i64 + y as i64);
        arg1.push(y as i64);
    }
    for y in 0..rows {
        pred.push(2i64);
        arg0.push(y as i64);
        arg1.push(200_000i64 + y as i64);
    }
    store.put(
        "wmir_committed",
        upload_i64(&provider, &[pred, arg0, arg1], schema3()).expect("wmir_committed"),
    );
    store.put(
        "wmir_body_0",
        upload_i64(&provider, &[vec![0], vec![3], vec![1]], schema3()).expect("wmir_body_0"),
    );
    store.put(
        "wmir_body_1",
        upload_i64(&provider, &[vec![0], vec![3], vec![2]], schema3()).expect("wmir_body_1"),
    );
    store.put(
        "wmir_len_2",
        upload_i64(&provider, &[vec![0]], schema1()).expect("wmir_len_2"),
    );
    let _env = EnvGuard::recursive_setop_profile();
    let result = program
        .evaluate_with_relation_store(Arc::clone(&provider), &store, true)
        .expect("evaluate");
    let stats = result.stats.expect("stats");
    let mut totals = HashMap::<String, (usize, u64)>::new();
    for stratum in stats.strata {
        for op in stratum.ops {
            let entry = totals.entry(op.op_name).or_default();
            entry.0 += 1;
            entry.1 += op.duration_us;
        }
    }
    let dedup_ops = totals.get("dedup").copied().unwrap_or_default().0;
    assert_eq!(
        dedup_ops, 0,
        "recursive set maintenance must not re-dedup union_gpu output; totals={totals:?}"
    );
}
