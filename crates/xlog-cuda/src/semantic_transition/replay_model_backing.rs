//! Immutable numerical allocations shared by original private replay readers.

use super::*;
use std::sync::Weak;

#[derive(PartialEq, Eq)]
struct ReplayBackingOrigin {
    instance: Identity256,
    storage_slot: u64,
    generation: u64,
}

pub(super) struct ReplayModelBacking {
    origin: ReplayBackingOrigin,
    seal: Identity256,
    geometry: Vec<u8>,
    allocation: TrackedCudaSlice<u8>,
    sealed: AtomicBool,
}

#[derive(Default)]
struct ReplayModelBackingState {
    provider: Option<Arc<CudaKernelProvider>>,
    allocations: Vec<Weak<ReplayModelBacking>>,
}

/// Numerical backing custody for one original private execution group.
///
/// Lookup uses original allocation coordinates, not content equality. The first
/// canonical cold restoration owns and seals the allocation; later historical
/// readers lease that same owner. Weak registry entries grant neither write
/// authority nor lifetime: original Sessions, readers and captured consumers
/// retain the strong owners through known final use or unknown completion.
#[derive(Default)]
pub struct SemanticReplayModelBackings {
    state: Mutex<ReplayModelBackingState>,
}

impl SemanticReplayModelBackings {
    pub(super) fn lease(
        &self,
        provider: &Arc<CudaKernelProvider>,
        material: &PublicationMaterial,
    ) -> Result<Vec<Arc<ReplayModelBacking>>, SemanticTransitionError> {
        let mut state = self.state.lock().map_err(|_| {
            publication_input_error("original replay model backing custody is poisoned")
        })?;
        if let Some(original) = &state.provider {
            if !Arc::ptr_eq(original, provider) {
                return Err(publication_input_error(
                    "replay model lease changed its original CUDA provider",
                ));
            }
        } else {
            state.provider = Some(Arc::clone(provider));
        }
        state.allocations.retain(|owner| owner.strong_count() != 0);
        let mut leases = Vec::with_capacity(material.model_memory.allocation_bytes.len());
        for (allocation, &bytes) in material.model_memory.allocation_bytes.iter().enumerate() {
            let mut origin = None;
            let mut seal = None;
            let mut geometry = Vec::new();
            material_u64(&mut geometry, bytes);
            material_u64(
                &mut geometry,
                material
                    .model_memory
                    .storages
                    .iter()
                    .filter(|storage| storage.allocation == allocation as u64)
                    .count() as u64,
            );
            let mut storages = BTreeMap::new();
            for (ordinal, storage) in material.model_memory.storages.iter().enumerate() {
                if storage.allocation != allocation as u64 {
                    continue;
                }
                storages.insert(ordinal as u64, storages.len() as u64);
                material_u64(&mut geometry, storage.byte_offset);
                material_u64(&mut geometry, storage.span_bytes);
            }
            material_u64(
                &mut geometry,
                material
                    .model_memory
                    .views
                    .iter()
                    .filter(|view| storages.contains_key(&view.storage))
                    .count() as u64,
            );
            for view in &material.model_memory.views {
                let Some(&storage) = storages.get(&view.storage) else {
                    continue;
                };
                let range = material
                    .ranges
                    .iter()
                    .find(|item| (item.range.role, item.range.index) == (view.role, view.index))
                    .ok_or(SemanticTransitionError::ObservationMismatch)?
                    .range;
                let coordinate = (range.storage_slot, range.generation);
                if origin
                    .replace(coordinate)
                    .is_some_and(|old| old != coordinate)
                    || seal
                        .replace(range.backing_digest)
                        .is_some_and(|old| old != range.backing_digest)
                {
                    return Err(publication_input_error(
                        "historical model aliases disagree on their original backing generation",
                    ));
                }
                for value in [
                    view.role,
                    view.index,
                    storage,
                    view.byte_offset,
                    range.offset_bytes,
                    range.length_bytes,
                    range.generation,
                ] {
                    material_u64(&mut geometry, value);
                }
                geometry.extend_from_slice(range.digest.as_bytes());
                geometry.extend_from_slice(&publication_abi_bytes(&[
                    material.layouts[&(view.role, view.index)]
                ]));
            }
            let (storage_slot, generation) =
                origin.ok_or(SemanticTransitionError::ObservationMismatch)?;
            let origin = ReplayBackingOrigin {
                instance: material.bank.header.instance,
                storage_slot,
                generation,
            };
            let seal = seal.ok_or(SemanticTransitionError::ObservationMismatch)?;
            let existing = state
                .allocations
                .iter()
                .filter_map(Weak::upgrade)
                .find(|owner| owner.origin == origin);
            let owner = if let Some(owner) = existing {
                if owner.seal != seal
                    || owner.geometry != geometry
                    || owner.allocation.len() as u64 != bytes
                {
                    return Err(publication_input_error(
                        "replay model lease changed the original seal or allocation alias geometry",
                    ));
                }
                if !owner.sealed.load(Ordering::Acquire) {
                    return Err(publication_input_error(
                        "replay model allocation retains an unfinished original restoration",
                    ));
                }
                owner
            } else {
                let owner = Arc::new(ReplayModelBacking {
                    origin,
                    seal,
                    geometry,
                    allocation: allocate_publication(provider, bytes as usize)?,
                    sealed: AtomicBool::new(false),
                });
                state.allocations.push(Arc::downgrade(&owner));
                owner
            };
            leases.push(owner);
        }
        Ok(leases)
    }
}

pub(super) enum PublicationAllocation {
    Writable(TrackedCudaSlice<u8>),
    Generation(Arc<TrackedCudaSlice<u8>>),
    Immutable(Arc<ReplayModelBacking>),
}

impl PublicationAllocation {
    pub(super) fn slice(&self) -> &TrackedCudaSlice<u8> {
        match self {
            Self::Writable(allocation) => allocation,
            Self::Generation(allocation) => allocation,
            Self::Immutable(owner) => &owner.allocation,
        }
    }

    pub(super) fn immutable(&self) -> bool {
        matches!(self, Self::Generation(_) | Self::Immutable(_))
    }

    pub(super) fn initializing(&self) -> bool {
        match self {
            Self::Writable(_) => true,
            Self::Generation(_) => true,
            Self::Immutable(owner) => !owner.sealed.load(Ordering::Acquire),
        }
    }

    pub(super) fn seal_restoration(&self) {
        if let Self::Immutable(owner) = self {
            owner.sealed.store(true, Ordering::Release);
        }
    }

    pub(super) fn retain(&self) -> Self {
        match self {
            Self::Writable(allocation) => Self::Writable(allocation.retain()),
            Self::Generation(owner) => Self::Generation(Arc::clone(owner)),
            Self::Immutable(owner) => Self::Immutable(Arc::clone(owner)),
        }
    }
}

impl std::ops::Deref for PublicationAllocation {
    type Target = TrackedCudaSlice<u8>;

    fn deref(&self) -> &Self::Target {
        self.slice()
    }
}
