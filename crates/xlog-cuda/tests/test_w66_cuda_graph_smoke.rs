mod common;

use cudarc::driver::sys;
use xlog_cuda::cuda_graph::CapturedCudaGraph;
use xlog_cuda::device_runtime::Access;

const BYTES: usize = 4096;
const PATTERN: u8 = 0x5a;

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
    assert!(!graph.graph().is_null(), "capture should return a graph");
    assert!(
        !graph.exec().is_null(),
        "instantiate should return an executable graph"
    );
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
    assert_eq!(host, vec![PATTERN; BYTES]);
}
