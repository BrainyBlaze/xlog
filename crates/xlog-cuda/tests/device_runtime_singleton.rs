use std::sync::{Arc, Barrier};
use std::thread;

use xlog_cuda::device_runtime::{AllocTag, StreamId, XlogDeviceRuntime};

fn runtime_or_skip() -> Option<&'static XlogDeviceRuntime> {
    match XlogDeviceRuntime::try_get(0) {
        Ok(runtime) => Some(runtime),
        Err(err) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", err);
            None
        }
    }
}

#[test]
fn get_returns_same_singleton_for_same_ordinal() {
    let Some(first) = runtime_or_skip() else {
        return;
    };
    let second = XlogDeviceRuntime::try_get(0).expect("runtime should stay available");

    assert!(std::ptr::eq(first, second));
    assert_eq!(first.device_ordinal(), 0);
    assert_eq!(second.device_ordinal(), 0);
}

#[test]
fn concurrent_get_returns_same_singleton() {
    let Some(expected) = runtime_or_skip() else {
        return;
    };
    let expected_addr = expected as *const XlogDeviceRuntime as usize;

    let barrier = Arc::new(Barrier::new(8));
    let mut handles = Vec::new();
    for _ in 0..8 {
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            let rt = XlogDeviceRuntime::try_get(0).expect("runtime");
            rt as *const XlogDeviceRuntime as usize
        }));
    }

    for handle in handles {
        assert_eq!(handle.join().expect("thread panicked"), expected_addr);
    }
}

#[test]
fn runtime_allocates_and_deallocates_through_default_resource() {
    let Some(runtime) = runtime_or_skip() else {
        return;
    };

    let before = runtime.bytes_outstanding();
    let block = runtime
        .allocate(1024, StreamId::DEFAULT, AllocTag("runtime-singleton-test"))
        .expect("allocate through runtime");

    assert_eq!(block.device_ordinal, 0);
    assert_eq!(block.bytes, 1024);
    assert_eq!(runtime.bytes_outstanding(), before + 1024);

    runtime
        .deallocate(block)
        .expect("deallocate through runtime");
    assert_eq!(runtime.bytes_outstanding(), before);
}
