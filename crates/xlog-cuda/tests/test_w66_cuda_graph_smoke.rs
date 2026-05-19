mod common;

use cudarc::driver::sys;
use std::sync::{Mutex, OnceLock};
use xlog_core::{ScalarType, Schema};
use xlog_cuda::cuda_graph::{CapturedCudaGraph, CudaGraphNodeKind};
use xlog_cuda::device_runtime::Access;

const BYTES: usize = 4096;
const PATTERN: u8 = 0x5a;
const UPDATED_PATTERN: u8 = 0xa5;

#[test]
fn cuda_graph_replays_runtime_backed_memset_on_launch_stream() {
    let Some(handles) = common::setup_provider_with_runtime() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let launch_stream_id = handles
        .runtime
        .stream_pool()
        .acquire()
        .expect("acquire non-default launch stream");
    let launch_stream = handles
        .runtime
        .stream_pool()
        .resolve(launch_stream_id)
        .expect("launch stream resolves");
    let buf = handles.memory.alloc::<u8>(BYTES).expect("alloc graph buf");

    handles
        .runtime
        .prepare_first_use(&buf, launch_stream_id, Access::Write)
        .expect("prepare graph write");

    let graph = CapturedCudaGraph::capture_on_stream(&launch_stream, || unsafe {
        let result =
            sys::cuMemsetD8Async(*buf.device_ptr(), PATTERN, BYTES, launch_stream.cu_stream());
        if result == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(xlog_core::XlogError::Kernel(format!(
                "cuMemsetD8Async failed: {result:?}"
            )))
        }
    })
    .expect("capture memset graph");
    assert_eq!(graph.node_count().expect("graph node count"), 1);
    let nodes = graph.nodes().expect("graph node inventory");
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].kind, CudaGraphNodeKind::Memset);
    assert!(!graph.graph().is_null(), "capture should return a graph");
    assert!(
        !graph.exec().is_null(),
        "instantiate should return an executable graph"
    );
    let mut memset_params = graph
        .memset_node_params(nodes[0])
        .expect("memset node params");
    assert_eq!(memset_params.value, PATTERN as u32);
    assert_eq!(memset_params.width, BYTES);
    memset_params.value = UPDATED_PATTERN as u32;
    graph
        .set_memset_node_params(nodes[0], &memset_params, &launch_stream)
        .expect("update memset node params");
    for _ in 0..3 {
        graph.launch(&launch_stream).expect("graph launch");
    }

    handles
        .runtime
        .finish_first_use(&buf, launch_stream_id, Access::Write)
        .expect("finish graph write");
    launch_stream.synchronize().expect("graph stream sync");

    let host = handles
        .provider
        .dtoh_small_metadata_untracked(&buf, BYTES)
        .expect("read graph result metadata");
    assert_eq!(host, vec![UPDATED_PATTERN; BYTES]);
}

#[test]
fn csm_inner_join_uses_bounded_cuda_graph_when_enabled() {
    let _guard = env_lock().lock().expect("env lock");
    let _env = EnvVarRestore::set("XLOG_USE_CSM_CUDA_GRAPH", "1");

    let Some(handles) = common::setup_provider_with_runtime() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };

    let launch_stream_id = handles
        .runtime
        .stream_pool()
        .acquire()
        .expect("acquire launch stream");
    let schema = Schema::new(vec![("k".to_string(), ScalarType::U32)]);
    let left = handles
        .provider
        .create_buffer_from_slice(&[1u32, 2, 3, 2], schema.clone())
        .expect("left buffer");
    let right = handles
        .provider
        .create_buffer_from_slice(&[2u32, 2, 4], schema)
        .expect("right buffer");

    let captures_before = handles.provider.csm_cuda_graph_captures();
    let launches_before = handles.provider.csm_cuda_graph_launches();
    let fallbacks_before = handles.provider.csm_cuda_graph_fallbacks();
    let cache_hits_before = handles.provider.csm_cuda_graph_cache_hits();
    let result = handles
        .provider
        .hash_join_inner_v2_count_scan_materialize_recorded(
            &left,
            &right,
            &[0],
            &[0],
            Some(16),
            launch_stream_id,
        )
        .expect("CSM graph join");
    let launch_stream = handles
        .runtime
        .stream_pool()
        .resolve(launch_stream_id)
        .expect("launch stream resolves");
    launch_stream.synchronize().expect("sync graph join");

    let schema2 = Schema::new(vec![("k".to_string(), ScalarType::U32)]);
    let left2 = handles
        .provider
        .create_buffer_from_slice(&[5u32, 7, 5, 9], schema2.clone())
        .expect("left2 buffer");
    let right2 = handles
        .provider
        .create_buffer_from_slice(&[5u32, 8, 5], schema2)
        .expect("right2 buffer");
    let result2 = handles
        .provider
        .hash_join_inner_v2_count_scan_materialize_recorded(
            &left2,
            &right2,
            &[0],
            &[0],
            Some(16),
            launch_stream_id,
        )
        .expect("cached CSM graph join");
    launch_stream.synchronize().expect("sync cached graph join");

    assert_eq!(
        handles.provider.csm_cuda_graph_captures() - captures_before,
        1,
        "bounded CSM graph path must capture exactly once"
    );
    assert_eq!(
        handles.provider.csm_cuda_graph_launches() - launches_before,
        2,
        "bounded CSM graph path must launch both capture and cached replays"
    );
    assert_eq!(
        handles.provider.csm_cuda_graph_cache_hits() - cache_hits_before,
        1,
        "second same-topology join must replay the cached graph"
    );
    assert_eq!(
        handles.provider.csm_cuda_graph_fallbacks(),
        fallbacks_before,
        "bounded fixture must not fall back to non-graph CSM"
    );
    assert_eq!(result.num_rows(), 4);
    let left_keys = handles
        .provider
        .download_column::<u32>(&result, 0)
        .expect("left result key");
    let right_keys = handles
        .provider
        .download_column::<u32>(&result, 1)
        .expect("right result key");
    assert_eq!(left_keys, vec![2, 2, 2, 2]);
    assert_eq!(right_keys, vec![2, 2, 2, 2]);

    assert_eq!(result2.num_rows(), 4);
    let left_keys2 = handles
        .provider
        .download_column::<u32>(&result2, 0)
        .expect("left2 result key");
    let right_keys2 = handles
        .provider
        .download_column::<u32>(&result2, 1)
        .expect("right2 result key");
    assert_eq!(left_keys2, vec![5, 5, 5, 5]);
    assert_eq!(right_keys2, vec![5, 5, 5, 5]);
}

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct EnvVarRestore {
    key: &'static str,
    old: Option<String>,
}

impl EnvVarRestore {
    fn set(key: &'static str, value: &str) -> Self {
        let old = std::env::var(key).ok();
        std::env::set_var(key, value);
        Self { key, old }
    }
}

impl Drop for EnvVarRestore {
    fn drop(&mut self) {
        match &self.old {
            Some(v) => std::env::set_var(self.key, v),
            None => std::env::remove_var(self.key),
        }
    }
}
