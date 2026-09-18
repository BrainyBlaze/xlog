//! Default-device-stream allocation through the shared raw storage owner.
//!
//! Allocation/free use the same explicit mode selected from context capability.
//! Failed initialization retains actual storage; reclamation proves physical
//! release before refunding bytes. Caller stream IDs are block metadata, not
//! proof of arbitrary pool routing. Sanitizer qualification remains separate.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::memory::RawDeviceAllocation;

use super::resource::{
    AllocationAccounting, AllocationRequest, BlockState, DeviceBlock, DeviceMemoryResource,
    Generation, ResourceBudgetSnapshot, ResourceError, ResourceResult,
};
use crate::CudaDevice;

/// Default-device-stream allocation adaptor. The live map retains actual raw
/// owners while callers hold opaque [`DeviceBlock`] identities. Deallocation
/// removes an exact identity and queues its actual owner after unlocking the
/// map. Reclamation waits for shared leases; an unknown physical-free outcome
/// retains the owner and its outstanding byte charge for a later reap.
pub struct DirectCudaResource {
    device: Arc<CudaDevice>,
    device_ordinal: u32,
    /// Live allocation identities and their physical owners.
    live: Mutex<HashMap<u64, (super::resource::BlockId, Arc<RawDeviceAllocation>)>>,
    /// Logical detach may leave graph or operation leases alive. The existing
    /// resource reap contract retires their actual owner after the last lease.
    pending: Mutex<Vec<Option<Arc<RawDeviceAllocation>>>>,
    /// All physical bytes, including retained initialization/free failures.
    /// Only the raw owner's proven release callback decrements this counter.
    bytes_outstanding: Arc<AllocationAccounting>,
}

impl DirectCudaResource {
    /// Construct a resource bound to `device`. `device_ordinal` is the
    /// CUDA ordinal for logging / multi-device disambiguation.
    pub fn new(device: Arc<CudaDevice>, device_ordinal: u32) -> Self {
        Self {
            device,
            device_ordinal,
            live: Mutex::new(HashMap::new()),
            pending: Mutex::new(Vec::new()),
            bytes_outstanding: Arc::default(),
        }
    }

    /// Borrow the device handle. Tests and downstream resources use
    /// this to launch kernels against the same device this resource
    /// allocates on.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

impl DeviceMemoryResource for DirectCudaResource {
    fn materialize(&self, request: AllocationRequest) -> ResourceResult<DeviceBlock> {
        let AllocationRequest {
            bytes,
            stream,
            tag,
            reclamation,
            ..
        } = request;
        if bytes == 0 {
            // Zero-byte allocations are not legal in CUDA; surface as
            // a contract error rather than calling cuMemAlloc(0).
            return Err(ResourceError::Driver(
                "DirectCudaResource: zero-byte allocation not supported".to_string(),
            ));
        }

        reclamation.attach_resource(Arc::clone(&self.bytes_outstanding), bytes)?;
        let allocation = crate::memory::RawDeviceAllocation::allocate(
            Arc::clone(self.device.inner().stream()),
            bytes,
            None,
            Arc::clone(&reclamation),
        )?;
        let ptr = allocation.ptr();
        let generation = Generation::next();
        let identity = super::resource::BlockId {
            ptr,
            generation,
            alloc_stream: stream,
            device_ordinal: self.device_ordinal,
        };
        let mut live = self.live.lock().expect("direct allocation map poisoned");
        if live.contains_key(&ptr) {
            return Err(
                ResourceError::Driver(format!("allocation pointer collision: {ptr:#x}"))
                    .retaining(bytes, reclamation),
            );
        }
        live.insert(ptr, (identity, allocation));
        Ok(DeviceBlock {
            ptr,
            device_ordinal: self.device_ordinal,
            alloc_stream: stream,
            bytes,
            align: std::mem::align_of::<u8>(),
            tag,
            generation,
            state: BlockState::Live,
        })
    }

    fn deallocate(&self, block: DeviceBlock) -> ResourceResult<()> {
        let mut live = self.live.lock().expect("direct allocation map poisoned");
        let (identity, allocation) = live.get(&block.ptr).ok_or(ResourceError::UseAfterFree {
            generation: block.generation,
        })?;
        super::resource::BlockId::from_block(&block).validate_allocation(
            block.bytes,
            *identity,
            allocation.len(),
        )?;
        let (_, allocation) = live.remove(&block.ptr).expect("validated live owner");
        drop(live);
        self.pending
            .lock()
            .expect("direct pending allocation queue poisoned")
            .push(Some(allocation));
        self.reap_pending()
    }

    fn reap_pending(&self) -> ResourceResult<()> {
        let mut pending =
            crate::memory::PendingReclamationBatch::take(&self.pending, |queue, owners| {
                owners.retain(Option::is_some);
                queue.append(owners);
            });
        crate::memory::reap_raw_allocations(&mut pending)
    }

    fn access_dependencies(
        &self,
        block: super::resource::BlockId,
        bytes: usize,
    ) -> ResourceResult<Option<Arc<super::resource::DeviceAccessDependencies>>> {
        let live = self.live.lock().expect("direct allocation map poisoned");
        let (identity, allocation) = live.get(&block.ptr).ok_or(ResourceError::UseAfterFree {
            generation: block.generation,
        })?;
        block.validate_allocation(bytes, *identity, allocation.len())?;
        Ok(Some(allocation.dependencies()))
    }

    fn device_ordinal(&self) -> u32 {
        self.device_ordinal
    }

    fn bytes_outstanding(&self) -> usize {
        self.bytes_outstanding.snapshot().0
    }

    fn allocation_accounting(&self) -> Arc<AllocationAccounting> {
        Arc::clone(&self.bytes_outstanding)
    }

    fn budget_snapshot(&self) -> Option<ResourceBudgetSnapshot> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::super::resource::{AllocTag, StreamId};
    use super::*;

    fn try_device() -> Option<Arc<CudaDevice>> {
        CudaDevice::new(0).ok().map(Arc::new)
    }

    #[test]
    fn allocate_then_deallocate_round_trips() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no CUDA device");
            return;
        };
        let r = DirectCudaResource::new(device, 0);
        assert_eq!(r.bytes_outstanding(), 0);

        let block = r
            .allocate(4096, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .expect("alloc");
        assert_eq!(block.bytes, 4096);
        assert_eq!(block.state, BlockState::Live);
        assert_eq!(r.bytes_outstanding(), 4096);

        r.deallocate(block).expect("dealloc");
        assert_eq!(r.bytes_outstanding(), 0);
    }

    #[test]
    fn zero_byte_allocate_rejects() {
        let Some(device) = try_device() else {
            return;
        };
        let r = DirectCudaResource::new(device, 0);
        let err = r.allocate(0, StreamId::DEFAULT, AllocTag::UNTAGGED);
        assert!(matches!(err, Err(ResourceError::Driver(_))));
        assert_eq!(r.bytes_outstanding(), 0);
    }

    #[test]
    fn deallocate_unknown_block_returns_use_after_free() {
        let Some(device) = try_device() else {
            return;
        };
        let r = DirectCudaResource::new(device, 0);
        let bogus = DeviceBlock {
            ptr: 0xdead_beef,
            device_ordinal: 0,
            alloc_stream: StreamId::DEFAULT,
            bytes: 16,
            align: 1,
            tag: AllocTag::UNTAGGED,
            generation: Generation::next(),
            state: BlockState::Live,
        };
        assert!(matches!(
            r.deallocate(bogus),
            Err(ResourceError::UseAfterFree { .. })
        ));
    }

    /// Locks the contract that DirectCudaResource does NOT
    /// silently accept `record_block_use`. If a caller (e.g. the
    /// future xlog launch builder) calls record_block_use against
    /// a runtime built around DirectCudaResource, the call must
    /// fail loudly with StreamMisuse — not return Ok and quietly
    /// fail to track anything. False safety here would let
    /// downstream code queue cross-stream kernels and drop
    /// blocks while the cross-stream use was never recorded,
    /// reproducing exactly the use-after-free this whole layer
    /// exists to prevent.
    ///
    /// Implementation note: DirectCudaResource inherits the
    /// trait's default `record_block_use` impl (which returns
    /// `StreamMisuse`). It does NOT override. If a future change
    /// adds a real override, it must make the override
    /// genuinely track cross-stream uses (similar to
    /// AsyncCudaResource's implementation) — anything else
    /// regresses this contract.
    #[test]
    fn record_block_use_rejected_with_stream_misuse() {
        let Some(device) = try_device() else {
            return;
        };
        let r = DirectCudaResource::new(device, 0);
        let block = r
            .allocate(64, StreamId::DEFAULT, AllocTag::UNTAGGED)
            .expect("alloc");
        let err = r.record_block_use(&block, StreamId::DEFAULT);
        match err {
            Err(ResourceError::StreamMisuse(msg)) => {
                assert!(
                    msg.contains("unsupported"),
                    "expected 'unsupported' in StreamMisuse message, got {:?}",
                    msg
                );
            }
            other => panic!(
                "DirectCudaResource::record_block_use must return StreamMisuse \
                 to surface unsupported cross-stream tracking; got {:?}",
                other
            ),
        }
        // The block stays live — a failed record_block_use must
        // NOT have removed the entry or dropped the slice.
        assert_eq!(r.bytes_outstanding(), 64);
        r.deallocate(block).expect("dealloc still works");
    }
}
