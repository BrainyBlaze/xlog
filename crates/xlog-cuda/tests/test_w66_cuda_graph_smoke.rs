mod common;

use cudarc::driver::sys;
use xlog_cuda::device_runtime::Access;

const BYTES: usize = 4096;
const PATTERN: u8 = 0x5a;

fn assert_cuda_success(site: &str, result: sys::CUresult) {
    assert_eq!(
        result,
        sys::CUresult::CUDA_SUCCESS,
        "{site} failed: {result:?}"
    );
}

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

    let mut graph: sys::CUgraph = std::ptr::null_mut();
    let mut exec: sys::CUgraphExec = std::ptr::null_mut();

    unsafe {
        assert_cuda_success(
            "cuStreamBeginCapture_v2",
            sys::cuStreamBeginCapture_v2(
                launch_stream.cu_stream(),
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            ),
        );
        assert_cuda_success(
            "cuMemsetD8Async",
            sys::cuMemsetD8Async(*buf.device_ptr(), PATTERN, BYTES, launch_stream.cu_stream()),
        );
        assert_cuda_success(
            "cuStreamEndCapture",
            sys::cuStreamEndCapture(launch_stream.cu_stream(), &mut graph),
        );
        assert!(!graph.is_null(), "capture should return a graph");
        assert_cuda_success(
            "cuGraphInstantiateWithFlags",
            sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0),
        );
        assert!(
            !exec.is_null(),
            "instantiate should return an executable graph"
        );
        for _ in 0..3 {
            assert_cuda_success(
                "cuGraphLaunch",
                sys::cuGraphLaunch(exec, launch_stream.cu_stream()),
            );
        }
    }

    handles
        .runtime
        .finish_first_use(&buf, launch_stream_id, Access::Write)
        .expect("finish graph write");
    launch_stream.synchronize().expect("graph stream sync");

    unsafe {
        assert_cuda_success("cuGraphExecDestroy", sys::cuGraphExecDestroy(exec));
        assert_cuda_success("cuGraphDestroy", sys::cuGraphDestroy(graph));
    }

    let host = handles
        .provider
        .dtoh_small_metadata_untracked(&buf, BYTES)
        .expect("read graph result metadata");
    assert_eq!(host, vec![PATTERN; BYTES]);
}
