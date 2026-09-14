use std::sync::Arc;

use cudarc::driver::{sys, CudaStream};
use xlog_core::MemoryBudget;
use xlog_cuda::{
    device_runtime::{StreamId, XlogDeviceRuntime},
    launch::{LaunchEnqueueError, LaunchRecorder},
    memory::TrackedCudaSlice,
    CudaKernelProvider, CudaProviderBuilder,
};

const WORDS: usize = 1 << 18;
const MEMORY_LIMIT: u64 = 64 * 1024 * 1024;

#[derive(Debug, Eq, PartialEq)]
struct InjectedAfterEnqueue;

unsafe fn copy_on_stream(
    stream: &CudaStream,
    source: &TrackedCudaSlice<u32>,
    destination: &mut TrackedCudaSlice<u32>,
) -> Result<(), String> {
    // SAFETY: the caller guarantees both pointers are live for `source.len()`
    // words and that `stream` is the runtime-owned stream for this operation.
    let result = unsafe {
        sys::cuMemcpyDtoDAsync_v2(
            destination.device_ptr_value(),
            source.device_ptr_value(),
            source.len() * std::mem::size_of::<u32>(),
            stream.cu_stream(),
        )
    };
    if result == sys::cudaError_enum::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("cuMemcpyDtoDAsync_v2 failed: {result:?}"))
    }
}

fn provider_or_skip() -> Option<CudaKernelProvider> {
    match CudaProviderBuilder::new(0, MemoryBudget::with_limit(MEMORY_LIMIT))
        .with_stream_capacity(2)
        .build()
    {
        Ok(provider) => Some(provider),
        Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("XLOG_REQUIRE_CUDA=1 but CUDA provider construction failed: {error}")
        }
        Err(error) => {
            eprintln!("Skipping: CUDA provider unavailable: {error}");
            None
        }
    }
}

fn recorder_for_copy(
    stream: StreamId,
    source: &TrackedCudaSlice<u32>,
    destination: &TrackedCudaSlice<u32>,
) -> LaunchRecorder {
    let mut recorder = LaunchRecorder::new_strict(stream);
    recorder.read(source).write(destination);
    recorder
}

fn commit_copy(
    runtime: &Arc<XlogDeviceRuntime>,
    stream: StreamId,
    source: &TrackedCudaSlice<u32>,
    destination: &mut TrackedCudaSlice<u32>,
) {
    let recorder = recorder_for_copy(stream, source, destination);

    // SAFETY: the operation reads only `source`, writes only `destination`,
    // both accesses were recorded above, and it enqueues only on the stream
    // supplied by `enqueue`.
    let enqueued = unsafe {
        recorder.enqueue(runtime, |cuda_stream| {
            copy_on_stream(cuda_stream.stream(), source, destination)
        })
    }
    .expect("device copy enqueue");

    enqueued.commit().expect("commit recorded copy");
    runtime
        .stream_pool()
        .resolve(stream)
        .expect("committed stream resolves")
        .synchronize()
        .expect("committed copy completes");
}

fn assert_readback(
    provider: &CudaKernelProvider,
    buffer: &TrackedCudaSlice<u32>,
    expected: &[u32],
) {
    let mut observed = vec![0_u32; expected.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(buffer, &mut observed)
        .expect("device readback");
    assert_eq!(observed, expected);
}

#[test]
fn public_launch_recorder_recovers_after_operation_error_explicit_abort_and_drop() {
    let Some(provider) = provider_or_skip() else {
        return;
    };

    let memory = Arc::clone(provider.memory());
    let runtime = Arc::clone(
        memory
            .runtime()
            .expect("CudaProviderBuilder must attach its runtime"),
    );
    let pool = Arc::clone(runtime.stream_pool());

    let first_stream = pool.acquire().expect("first non-default stream");
    let second_stream = pool.acquire().expect("second non-default stream");
    assert_ne!(first_stream, StreamId::DEFAULT);
    assert_ne!(second_stream, StreamId::DEFAULT);
    assert_ne!(first_stream, second_stream);
    assert!(pool.resolve(first_stream).is_some());
    assert!(pool.resolve(second_stream).is_some());

    let first_pattern = vec![0x1111_1111_u32; WORDS];
    let second_pattern = vec![0x2222_2222_u32; WORDS];
    let third_pattern = vec![0x3333_3333_u32; WORDS];

    let mut first_source = memory.alloc::<u32>(WORDS).expect("first source");
    let mut second_source = memory.alloc::<u32>(WORDS).expect("second source");
    let mut third_source = memory.alloc::<u32>(WORDS).expect("third source");
    let mut destination = memory.alloc::<u32>(WORDS).expect("destination");

    provider
        .device()
        .inner()
        .htod_sync_copy_into(&first_pattern, &mut first_source)
        .expect("upload first pattern");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&second_pattern, &mut second_source)
        .expect("upload second pattern");
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&third_pattern, &mut third_source)
        .expect("upload third pattern");

    commit_copy(&runtime, first_stream, &first_source, &mut destination);
    assert_readback(&provider, &destination, &first_pattern);

    let recorder = recorder_for_copy(first_stream, &second_source, &destination);
    // SAFETY: the closure touches exactly the recorded buffers and uses only
    // the supplied stream.
    let error = match unsafe {
        recorder.enqueue(&runtime, |cuda_stream| {
            copy_on_stream(cuda_stream.stream(), &second_source, &mut destination)
                .expect("real copy enqueued before injected failure");
            Err::<(), InjectedAfterEnqueue>(InjectedAfterEnqueue)
        })
    } {
        Err(error) => error,
        Ok(enqueued) => {
            enqueued.abort().expect("abort unexpected successful guard");
            panic!("injected operation error was not returned");
        }
    };

    match error {
        LaunchEnqueueError::Operation(InjectedAfterEnqueue) => {}
        other => panic!("typed operation error was not preserved: {other:?}"),
    }

    commit_copy(&runtime, second_stream, &third_source, &mut destination);
    assert_readback(&provider, &destination, &third_pattern);

    let recorder = recorder_for_copy(second_stream, &first_source, &destination);
    // SAFETY: the closure touches exactly the recorded buffers and uses only
    // the supplied stream.
    let enqueued = unsafe {
        recorder.enqueue(&runtime, |cuda_stream| {
            copy_on_stream(cuda_stream.stream(), &first_source, &mut destination)
        })
    }
    .expect("enqueue before explicit abort");
    enqueued.abort().expect("explicit abort");

    commit_copy(&runtime, first_stream, &second_source, &mut destination);
    assert_readback(&provider, &destination, &second_pattern);

    let recorder = recorder_for_copy(first_stream, &first_source, &destination);
    // SAFETY: the closure touches exactly the recorded buffers and uses only
    // the supplied stream.
    let enqueued = unsafe {
        recorder.enqueue(&runtime, |cuda_stream| {
            copy_on_stream(cuda_stream.stream(), &first_source, &mut destination)
        })
    }
    .expect("enqueue before automatic drop abort");
    drop(enqueued);

    commit_copy(&runtime, second_stream, &third_source, &mut destination);
    assert_readback(&provider, &destination, &third_pattern);
}
