//! Immutable numerical allocations shared by original private replay readers.

use super::*;
use std::sync::{OnceLock, Weak};

#[derive(Clone, PartialEq, Eq)]
struct ReplayBackingOrigin {
    instance: Identity256,
    storage_slot: u64,
    generation: u64,
}

pub(super) struct ReplayModelBacking {
    origin: ReplayBackingOrigin,
    seal: Identity256,
    geometry: Vec<u8>,
    allocation: Arc<TrackedCudaSlice<u8>>,
    sealed: AtomicBool,
}

struct LiveModelGeneration {
    provider: Weak<CudaKernelProvider>,
    origin: ReplayBackingOrigin,
    seal: Identity256,
    geometry: Vec<u8>,
    allocation: Weak<TrackedCudaSlice<u8>>,
}

// Cold native exports register genuine owners, not byte-equal replacements.
// Weak entries neither allocate nor extend numerical lifetimes. All strong
// owners remain in original Sessions, snapshots and actual replay consumers.
fn live_model_generations() -> &'static Mutex<Vec<LiveModelGeneration>> {
    static LIVE: OnceLock<Mutex<Vec<LiveModelGeneration>>> = OnceLock::new();
    LIVE.get_or_init(Mutex::default)
}

fn backing_descriptor(
    material: &PublicationMaterial,
    allocation: usize,
) -> Result<(ReplayBackingOrigin, Identity256, Vec<u8>), SemanticTransitionError> {
    let bytes = material.model_memory.allocation_bytes[allocation];
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
    let (storage_slot, generation) = origin.ok_or(SemanticTransitionError::ObservationMismatch)?;
    Ok((
        ReplayBackingOrigin {
            instance: material.bank.header.instance,
            storage_slot,
            generation,
        },
        seal.ok_or(SemanticTransitionError::ObservationMismatch)?,
        geometry,
    ))
}

/// Register only after the complete original cold observation authenticated
/// the actual acquired directory, every typed seal and the full backing bytes.
#[cfg(feature = "semantic-policy")]
pub(super) fn register_original_model_generations(
    provider: &Arc<CudaKernelProvider>,
    material: &PublicationMaterial,
    originals: &[(u64, Arc<TrackedCudaSlice<u8>>)],
) -> Result<(), SemanticTransitionError> {
    if originals.len() != material.model_memory.allocation_bytes.len() {
        return Err(SemanticTransitionError::ObservationMismatch);
    }
    let mut live = live_model_generations().lock().map_err(|_| {
        publication_input_error("original numerical generation custody is poisoned")
    })?;
    live.retain(|entry| entry.provider.strong_count() != 0 && entry.allocation.strong_count() != 0);
    for (index, (slot, allocation)) in originals.iter().enumerate() {
        let (origin, seal, geometry) = backing_descriptor(material, index)?;
        if *slot != origin.storage_slot
            || allocation.len() as u64 != material.model_memory.allocation_bytes[index]
        {
            return Err(SemanticTransitionError::ObservationMismatch);
        }
        let existing = live.iter().find(|entry| {
            entry.origin == origin
                && entry
                    .provider
                    .upgrade()
                    .is_some_and(|owner| Arc::ptr_eq(&owner, provider))
        });
        if let Some(existing) = existing {
            if existing.seal != seal
                || existing.geometry != geometry
                || existing
                    .allocation
                    .upgrade()
                    .is_none_or(|owner| !Arc::ptr_eq(&owner, allocation))
            {
                return Err(publication_input_error(
                    "original numerical generation changed its allocation owner or seals",
                ));
            }
        } else {
            live.push(LiveModelGeneration {
                provider: Arc::downgrade(provider),
                origin,
                seal,
                geometry,
                allocation: Arc::downgrade(allocation),
            });
        }
    }
    Ok(())
}

#[derive(Default)]
struct ReplayModelBackingState {
    provider: Option<Arc<CudaKernelProvider>>,
    allocations: Vec<Weak<ReplayModelBacking>>,
}

/// Numerical backing custody for one original private execution group.
///
/// Lookup uses original allocation coordinates, not content equality. Cold
/// completed-step export registers the genuine immutable source owner, which
/// historical readers lease even on their first import. If no original owner
/// survives, complete cold restoration owns and seals one backing. Weak entries grant neither write
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
            let (origin, seal, geometry) = backing_descriptor(material, allocation)?;
            let original = {
                let mut live = live_model_generations().lock().map_err(|_| {
                    publication_input_error("original numerical generation custody is poisoned")
                })?;
                live.retain(|entry| {
                    entry.provider.strong_count() != 0 && entry.allocation.strong_count() != 0
                });
                let entry = live.iter().find(|entry| {
                    entry.origin == origin
                        && entry
                            .provider
                            .upgrade()
                            .is_some_and(|owner| Arc::ptr_eq(&owner, provider))
                });
                if let Some(entry) = entry {
                    if entry.seal != seal || entry.geometry != geometry {
                        return Err(publication_input_error("historical replay changed the original live numerical geometry or seals"));
                    }
                    entry.allocation.upgrade()
                } else {
                    None
                }
            };
            let existing = state
                .allocations
                .iter()
                .filter_map(Weak::upgrade)
                .find(|owner| owner.origin == origin);
            let owner = if let Some(owner) = existing {
                if owner.seal != seal
                    || owner.geometry != geometry
                    || owner.allocation.len() as u64 != bytes
                    || original
                        .as_ref()
                        .is_some_and(|original| !Arc::ptr_eq(original, &owner.allocation))
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
                let (allocation, sealed) = if let Some(original) = original {
                    if original.len() as u64 != bytes {
                        return Err(SemanticTransitionError::ObservationMismatch);
                    }
                    (original, true)
                } else {
                    // A durable restart may have no live original allocation.
                    // The unchanged complete cold restoration then owns and
                    // seals one allocation, never a byte-equality alias.
                    (
                        Arc::new(allocate_publication(provider, bytes as usize)?),
                        false,
                    )
                };
                let owner = Arc::new(ReplayModelBacking {
                    origin,
                    seal,
                    geometry,
                    allocation,
                    sealed: AtomicBool::new(sealed),
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
    #[cfg(feature = "semantic-policy")]
    pub(super) fn model_generation_owner(&self) -> Option<Arc<TrackedCudaSlice<u8>>> {
        match self {
            Self::Generation(owner) => Some(Arc::clone(owner)),
            Self::Immutable(owner) if owner.sealed.load(Ordering::Acquire) => {
                Some(Arc::clone(&owner.allocation))
            }
            _ => None,
        }
    }

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
