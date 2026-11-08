use super::*;
use crate::types::PreparedDiagnosticHandoff;
use std::sync::Weak;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MaterialStoreIdentity(pub(super) [u64; 2]);

#[derive(Clone)]
pub(super) struct StoredMaterial {
    pub(super) store: MaterialStoreIdentity,
    pub(super) digest: String,
    pub(super) size: u64,
    pub(super) source: Weak<MaterialSourceRegistry>,
}

impl std::fmt::Debug for StoredMaterial {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StoredMaterial")
            .field("store", &self.store)
            .field("digest", &self.digest)
            .field("size", &self.size)
            .finish()
    }
}

impl PartialEq for StoredMaterial {
    fn eq(&self, other: &Self) -> bool {
        self.store == other.store && self.digest == other.digest && self.size == other.size
    }
}

impl Eq for StoredMaterial {}

pub(super) struct ReplayPayload {
    payload: xlog_cuda::SemanticReplayLoadedMaterial,
    source: Weak<MaterialSourceRegistry>,
}

impl ReplayPayload {
    pub(super) fn new(source: &Arc<MaterialSourceRegistry>, bytes: &[u8]) -> PyResult<Arc<Self>> {
        Ok(Arc::new(Self {
            payload: source.copy_payload(bytes)?,
            source: Arc::downgrade(source),
        }))
    }

    pub(super) fn bytes(&self) -> &[u8] {
        self.payload.as_bytes()
    }

    pub(super) fn source(&self) -> PyResult<Arc<MaterialSourceRegistry>> {
        self.source
            .upgrade()
            .ok_or_else(|| invalid("replay payload lost its original material source"))
    }

    pub(super) fn load(&self) -> PyResult<xlog_cuda::SemanticReplayLoadedMaterial> {
        self.source()?.copy_payload(self.bytes())
    }

    pub(super) fn original_loaded(&self) -> PyResult<xlog_cuda::SemanticReplayLoadedMaterial> {
        self.payload
            .require_budget(&self.source()?.native_budget()?)
            .map_err(xlog_err)?;
        self.payload
            .slice_payload(0..self.payload.as_bytes().len())
            .map_err(xlog_err)
    }
}

impl std::fmt::Debug for ReplayPayload {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReplayPayload")
            .field("bytes", &self.bytes().len())
            .finish()
    }
}

impl PartialEq for ReplayPayload {
    fn eq(&self, other: &Self) -> bool {
        self.bytes() == other.bytes()
    }
}

impl Eq for ReplayPayload {}

impl StoredMaterial {
    pub(super) fn read(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let fields = exact_tuple::<4>(value)?;
        if !fields[0].is_exact_instance_of::<PyString>()
            || fields[0].extract::<String>()? != "stored"
        {
            return Err(invalid("stored replay material has another transport tag"));
        }
        Ok(Self {
            store: read_store_identity(&fields[1])?,
            digest: read_digest(&fields[2])?,
            size: positive_u64(&fields[3])?,
            source: Weak::new(),
        })
    }

    pub(super) fn bind(mut self, source: &Arc<MaterialSourceRegistry>) -> PyResult<Self> {
        source.require_reference(&self)?;
        self.source = Arc::downgrade(source);
        Ok(self)
    }

    pub(super) fn source(&self) -> PyResult<Arc<MaterialSourceRegistry>> {
        self.source
            .upgrade()
            .ok_or_else(|| invalid("stored material lost its original source binding"))
    }

    pub(super) fn load(&self) -> PyResult<xlog_cuda::SemanticReplayLoadedMaterial> {
        let source = self.source()?;
        Python::attach(|py| source.resolve(py, self))
    }

    pub(super) fn python_value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok((
            "stored",
            (self.store.0[0], self.store.0[1]),
            &self.digest,
            self.size,
        )
            .into_pyobject(py)?
            .into_any()
            .unbind())
    }

    pub(super) fn digest_bytes(&self) -> PyResult<[u8; 32]> {
        let mut digest = [0; 32];
        if self.digest.len() != 64 {
            return Err(invalid("stored replay digest has another byte extent"));
        }
        for (output, encoded) in digest
            .iter_mut()
            .zip(self.digest.as_bytes().chunks_exact(2))
        {
            let nibble = |value| match value {
                b'0'..=b'9' => Ok(value - b'0'),
                b'a'..=b'f' => Ok(value - b'a' + 10),
                _ => Err(invalid(
                    "stored replay digest is not canonical lowercase SHA256",
                )),
            };
            *output = (nibble(encoded[0])? << 4) | nibble(encoded[1])?;
        }
        Ok(digest)
    }
}

struct MaterialSourceMetadata {
    store: MaterialStoreIdentity,
    roster: Vec<(String, u64)>,
    item_cap: u64,
    store_cap: u64,
    resolved_cap: u64,
    _reservation: Option<xlog_cuda::SemanticReplayMaterialReservation>,
}

impl MaterialSourceMetadata {
    fn size(&self, digest: &str) -> Option<u64> {
        self.roster
            .binary_search_by(|(original, _)| original.as_str().cmp(digest))
            .ok()
            .map(|index| self.roster[index].1)
    }
}

struct OriginalMaterialSource {
    metadata: Arc<MaterialSourceMetadata>,
    resolver: Option<Py<PyAny>>,
    budget: Option<xlog_cuda::SemanticReplayMaterialBudget>,
}

#[derive(Default)]
struct NativeReplayBindings {
    entries: Vec<(String, NativeReplayBinding)>,
    _reservation: Option<xlog_cuda::SemanticReplayMaterialReservation>,
}

#[derive(Default)]
struct CompilerDiagnosticCache {
    available: Option<PreparedDiagnosticHandoff>,
    blocking: Option<PyErr>,
    loaned: bool,
}

struct CompilerDiagnosticAcquisition<'a> {
    acquiring: &'a AtomicBool,
}

impl Drop for CompilerDiagnosticAcquisition<'_> {
    fn drop(&mut self) {
        self.acquiring.store(false, Ordering::Release);
    }
}

fn compiler_diagnostic_custody_refusal() -> xlog_core::XlogError {
    xlog_core::XlogError::AllocationRefused {
        operation: "compiler diagnostic custody",
        reason: xlog_core::error::AllocationRefusal::Poisoned,
    }
}

#[derive(Default)]
pub(super) struct MaterialSourceRegistry {
    original: Mutex<Option<OriginalMaterialSource>>,
    native_bindings: Mutex<NativeReplayBindings>,
    compiler_diagnostic: Mutex<CompilerDiagnosticCache>,
    compiler_diagnostic_acquiring: AtomicBool,
    owner_metadata: Mutex<Option<xlog_cuda::SemanticReplayMaterialReservation>>,
    semantic_world: Mutex<Option<Arc<world::WorldOwnerData>>>,
}

impl MaterialSourceRegistry {
    pub(super) fn bind_semantic_world(&self, owner: &Arc<world::WorldOwnerData>) -> PyResult<()> {
        let mut current = self
            .semantic_world
            .lock()
            .map_err(|_| invalid("common semantic world owner is poisoned"))?;
        if let Some(original) = current.as_ref() {
            if !Arc::ptr_eq(original, owner) {
                return Err(invalid(
                    "material source cannot replace its original common world owner",
                ));
            }
        } else {
            *current = Some(Arc::clone(owner));
        }
        Ok(())
    }

    pub(super) fn semantic_context(
        &self,
        binding: &ColdValue,
    ) -> PyResult<Arc<world::ContextSourceData>> {
        let owner = self
            .semantic_world
            .lock()
            .map_err(|_| invalid("common semantic world owner is poisoned"))?
            .as_ref()
            .map(Arc::clone)
            .ok_or_else(|| invalid("semantic context requires its original common world owner"))?;
        world::resolve_binding(&owner, binding)
    }

    /// Warm the same original cache before acquiring a consumer's own mutex.
    /// This grants no compiler entry and returns only an unbound family.
    pub(super) fn ensure_compiler_diagnostic(&self, py: Python<'_>) -> PyResult<()> {
        let family = self.take_compiler_diagnostic(py)?;
        self.return_compiler_diagnostic(py, family)?;
        // Retiring the previous exception can run a destructor that reenters
        // another compiler and consumes the returned family. Check only after
        // that retirement; never let the caller construct under its own lock.
        let cached = match self.compiler_diagnostic.lock() {
            Ok(cached) => cached,
            Err(poison) => {
                let cached = poison.into_inner();
                return Err(cached
                    .blocking
                    .as_ref()
                    .expect("poisoned compiler entry retains its original outcome")
                    .clone_ref(py));
            }
        };
        if cached.available.is_some() && !cached.loaned {
            Ok(())
        } else {
            Err(cached
                .blocking
                .as_ref()
                .expect("reentered compiler entry retains its original outcome")
                .clone_ref(py))
        }
    }

    /// Lend an original family before compiler entry. After a published failure,
    /// replacement refusal returns that same outcome without starting a scope.
    pub(super) fn take_compiler_diagnostic(
        &self,
        py: Python<'_>,
    ) -> PyResult<PreparedDiagnosticHandoff> {
        let budget = self.native_budget()?;
        let mut cached = match self.compiler_diagnostic.lock() {
            Ok(cached) => cached,
            Err(poison) => {
                let mut cached = poison.into_inner();
                if let Some(error) = &cached.blocking {
                    let error = error.clone_ref(py);
                    drop(cached);
                    return Err(error);
                }
                let family = cached.available.take();
                drop(cached);
                // Poison grants no compiler entry. Its inline refusal uses
                // only an already-acquired original family, never a new one.
                return Err(self.publish_compiler_failure(
                    py,
                    family.expect("poisoned compiler diagnostic requires its original available family or outcome"),
                    compiler_diagnostic_custody_refusal(),
                ));
            }
        };
        if let Some(family) = cached.available.take() {
            assert!(
                !cached.loaned,
                "original compiler diagnostic is already loaned"
            );
            cached.loaned = true;
            drop(cached);
            assert!(
                family.is_unbound_for(py, &budget),
                "cached compiler diagnostic requires its unbound original pool"
            );
            return Ok(family);
        }
        let blocking = cached.blocking.as_ref().map(|error| error.clone_ref(py));
        if cached.loaned || self.compiler_diagnostic_acquiring.load(Ordering::Acquire) {
            drop(cached);
            // The first loan spans only the canonical callback-free Rust
            // compiler. Replacement construction can reenter through Python GC,
            // and must keep its already-published original outcome available.
            return Err(blocking.expect("original compiler diagnostic loan cannot be reentered"));
        }
        let blocking =
            blocking.expect("original compiler diagnostic requires its admitted family or outcome");
        self.compiler_diagnostic_acquiring
            .store(true, Ordering::Release);
        let acquisition = CompilerDiagnosticAcquisition {
            acquiring: &self.compiler_diagnostic_acquiring,
        };
        drop(cached);
        // Python allocation can execute GC callbacks. No cache mutex is held,
        // and the original blocking outcome stays charged through this attempt.
        let result = crate::types::prepare_owned_diagnostic_handoff(py, &budget);
        let mut cached = match self.compiler_diagnostic.lock() {
            Ok(cached) => cached,
            Err(poison) => {
                let cached = poison.into_inner();
                let error = cached.blocking.as_ref().map(|error| error.clone_ref(py));
                drop(cached);
                drop(result);
                drop(acquisition);
                return Err(error.unwrap_or(blocking));
            }
        };
        let result = match result {
            Ok(family) => {
                assert!(
                    family.is_unbound_for(py, &budget),
                    "replacement compiler diagnostic requires its original pool"
                );
                assert!(
                    cached.available.is_none() && !cached.loaned,
                    "replacement compiler diagnostic changed its original entry"
                );
                cached.loaned = true;
                Ok(family)
            }
            Err(error) => {
                drop(cached);
                drop(error);
                drop(acquisition);
                return Err(blocking);
            }
        };
        drop(cached);
        drop(acquisition);
        result
    }

    /// A known successful attempt returns its unused family. A published
    /// diagnostic is permanently bound and must never return to this cache.
    pub(super) fn return_compiler_diagnostic(
        &self,
        py: Python<'_>,
        family: PreparedDiagnosticHandoff,
    ) -> PyResult<()> {
        let budget = self.native_budget()?;
        assert!(
            family.is_unbound_for(py, &budget),
            "only an unbound original compiler diagnostic may be returned"
        );
        let mut cached = match self.compiler_diagnostic.lock() {
            Ok(cached) => cached,
            Err(poison) => {
                let mut cached = poison.into_inner();
                let blocking = cached.blocking.as_ref().map(|error| error.clone_ref(py));
                if blocking.is_some() {
                    cached.loaned = false;
                }
                drop(cached);
                if let Some(error) = blocking {
                    drop(family);
                    return Err(error);
                }
                return Err(self.publish_compiler_failure(
                    py,
                    family,
                    compiler_diagnostic_custody_refusal(),
                ));
            }
        };
        assert!(
            cached.available.is_none() && cached.loaned,
            "original compiler diagnostic was already returned"
        );
        cached.available = Some(family);
        cached.loaned = false;
        let previous = cached.blocking.take();
        drop(cached);
        // BaseException slots may retain arbitrary Python owners. Their last
        // references must not be released while the cache mutex is held.
        drop(previous);
        Ok(())
    }

    pub(super) fn publish_compiler_failure(
        &self,
        py: Python<'_>,
        family: PreparedDiagnosticHandoff,
        error: xlog_core::XlogError,
    ) -> PyErr {
        let error = family.bind_failure(py, error);
        let (mut cached, poisoned) = match self.compiler_diagnostic.lock() {
            Ok(cached) => (cached, false),
            // Preserve the original outcome even when custody is poisoned.
            // Poison is not cleared: subsequent entry/return remain refused.
            Err(poison) => (poison.into_inner(), true),
        };
        if !poisoned {
            assert!(
                cached.available.is_none() && cached.loaned,
                "compiler failure requires its original loaned family"
            );
        }
        let previous = cached.blocking.replace(error.clone_ref(py));
        cached.loaned = false;
        drop(cached);
        drop(previous);
        error
    }

    pub(super) fn retain_process_owner_metadata(
        &self,
        metadata: xlog_cuda::SemanticReplayMaterialReservation,
    ) -> PyResult<()> {
        let mut original = self
            .owner_metadata
            .lock()
            .map_err(|_| invalid("material source owner metadata is poisoned"))?;
        if original.is_some() {
            return Err(invalid(
                "material source owner metadata was already retained",
            ));
        }
        *original = Some(metadata);
        Ok(())
    }
    pub(super) fn opened_store_identity(&self) -> PyResult<[u64; 2]> {
        let retained = self
            .original
            .lock()
            .map_err(|_| invalid("material source registry is poisoned"))?;
        retained
            .as_ref()
            .map(|original| original.metadata.store.0)
            .ok_or_else(|| invalid("material source has no original opened store"))
    }

    pub(super) fn require_store(&self, store: MaterialStoreIdentity) -> PyResult<()> {
        let original = self
            .original
            .lock()
            .map_err(|_| invalid("material source registry is poisoned"))?;
        if original
            .as_ref()
            .is_none_or(|original| original.metadata.store != store)
        {
            return Err(invalid(
                "checkpoint belongs to another original material store",
            ));
        }
        Ok(())
    }

    pub(super) fn require_index_materials(
        &self,
        store: &serde_json::Value,
        materials: &serde_json::Map<String, serde_json::Value>,
    ) -> PyResult<()> {
        let store = store
            .as_object()
            .ok_or_else(|| invalid("initial replay store metadata is absent"))?;
        let identity = store
            .get("identity")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| invalid("initial replay store identity is absent"))?;
        if store.len() != 2
            || store.get("directory").and_then(serde_json::Value::as_str) != Some("materials")
            || identity.len() != 2
        {
            return Err(invalid(
                "initial replay store metadata differs from its canonical original store",
            ));
        }
        let identity = MaterialStoreIdentity([
            identity[0].as_u64().ok_or_else(|| {
                invalid("initial replay store identity requires nonnegative u64 coordinates")
            })?,
            identity[1].as_u64().ok_or_else(|| {
                invalid("initial replay store identity requires nonnegative u64 coordinates")
            })?,
        ]);
        let original = self
            .original
            .lock()
            .map_err(|_| invalid("material source registry is poisoned"))?;
        let original = original
            .as_ref()
            .ok_or_else(|| invalid("initial replay kit has no original material source"))?;
        if original.metadata.store != identity || original.resolver.is_none() {
            return Err(invalid(
                "initial replay kit names another original material store",
            ));
        }
        for (digest, size) in materials {
            validate_digest(digest)?;
            let size = size.as_u64().filter(|size| *size > 0).ok_or_else(|| {
                invalid("initial replay material requires its positive original sealed size")
            })?;
            if original.metadata.size(digest) != Some(size) {
                return Err(invalid(
                    "initial replay material size differs from its original sealed roster",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn require_saved(&self, saved: &Self) -> PyResult<()> {
        let saved = saved
            .original
            .lock()
            .map_err(|_| invalid("material source registry is poisoned"))?
            .as_ref()
            .map(|original| Arc::clone(&original.metadata))
            .ok_or_else(|| invalid("checkpoint has no original material source"))?;
        let retained = self
            .original
            .lock()
            .map_err(|_| invalid("material source registry is poisoned"))?;
        let current = retained
            .as_ref()
            .ok_or_else(|| invalid("Session has no original material source"))?;
        if saved.store != current.metadata.store
            || saved.item_cap != current.metadata.item_cap
            || saved.store_cap != current.metadata.store_cap
            || saved.resolved_cap != current.metadata.resolved_cap
        {
            return Err(invalid(
                "checkpoint changed its original material store or caps",
            ));
        }
        for (digest, size) in &saved.roster {
            if current.metadata.size(digest) != Some(*size) {
                return Err(invalid(
                    "checkpoint material roster is absent from its original sealed store",
                ));
            }
        }
        if current.resolver.is_none() || current.budget.is_none() {
            return Err(invalid(
                "checkpoint material source is not bound to its original pool",
            ));
        }
        Ok(())
    }

    pub(super) fn bind_native(
        &self,
        owner: &mut xlog_cuda::SemanticTransitionSession,
    ) -> PyResult<()> {
        owner
            .bind_replay_material_budget(&self.native_budget()?)
            .map_err(xlog_err)
    }

    pub(super) fn bind_values(self: &Arc<Self>, value: &ColdValue) -> PyResult<ColdValue> {
        Ok(match value {
            ColdValue::Bytes(bytes) => ColdValue::ReplayBytes(ReplayPayload::new(self, bytes)?),
            ColdValue::Stored(stored) => ColdValue::Stored(stored.clone().bind(self)?),
            ColdValue::Sequence(values) => ColdValue::Sequence(
                values
                    .iter()
                    .map(|value| self.bind_values(value))
                    .collect::<PyResult<_>>()?,
            ),
            ColdValue::ReplayBytes(payload) => {
                if !Arc::ptr_eq(self, &payload.source()?) {
                    return Err(invalid(
                        "replay payload changed its original material source",
                    ));
                }
                value.clone()
            }
            _ => value.clone(),
        })
    }

    pub(super) fn bind(
        &self,
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
        item_cap: u64,
        store_cap: u64,
        resolved_cap: u64,
    ) -> PyResult<()> {
        if item_cap == 0 || store_cap == 0 || item_cap > store_cap || resolved_cap == 0 {
            return Err(invalid("material store caps must be positive and ordered"));
        }
        let fields = exact_tuple::<3>(source)?;
        let store = read_store_identity(&fields[0])?;
        if !fields[1].is_exact_instance_of::<PyTuple>() {
            return Err(invalid("sealed material roster requires an exact tuple"));
        }
        if !fields[2].is_callable() {
            return Err(invalid(
                "material source requires its original retained resolver",
            ));
        }
        self.compiler_diagnostic_acquiring
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                invalid("original material source capacity admission is already in progress")
            })?;
        let _admission = CompilerDiagnosticAcquisition {
            acquiring: &self.compiler_diagnostic_acquiring,
        };
        let require_original = |original: Option<&OriginalMaterialSource>| -> PyResult<()> {
            if let Some(original) = original {
                if original.metadata.store != store
                    || original.metadata.item_cap != item_cap
                    || original.metadata.store_cap != store_cap
                    || original.metadata.resolved_cap != resolved_cap
                    || original
                        .resolver
                        .as_ref()
                        .is_some_and(|resolver| !resolver.bind(py).is(&fields[2]))
                {
                    return Err(invalid(
                        "material source changed its original store, caps or cached resolver",
                    ));
                }
            }
            Ok(())
        };
        let (original_metadata, original_resolver, original_budget) = {
            let retained = self
                .original
                .lock()
                .map_err(|_| invalid("material source registry is poisoned"))?;
            require_original(retained.as_ref())?;
            (
                retained
                    .as_ref()
                    .map(|original| Arc::clone(&original.metadata)),
                retained
                    .as_ref()
                    .and_then(|original| original.resolver.as_ref())
                    .map(|resolver| resolver.as_ptr()),
                retained
                    .as_ref()
                    .and_then(|original| original.budget.clone()),
            )
        };
        let budget = match original_budget.as_ref() {
            Some(budget) => budget.clone(),
            None => xlog_cuda::SemanticReplayMaterialBudget::new(resolved_cap).map_err(xlog_err)?,
        };
        // The first error family precedes all roster, payload and compiler
        // allocations in this original pool. Its Python construction can run
        // GC callbacks, so neither original nor cache mutex is held here.
        let diagnostic_handoff = if original_budget.is_none() {
            Some(
                crate::types::prepare_owned_diagnostic_handoff(py, &budget)
                    .map_err(|error| error.into_bootstrap_error())?,
            )
        } else {
            None
        };
        // These guards are declared after the local family so all failure exits
        // unlock before dropping its Python objects or original budget receipt.
        let mut retained = self
            .original
            .lock()
            .map_err(|_| invalid("material source registry is poisoned"))?;
        require_original(retained.as_ref())?;
        let unchanged = match (original_metadata.as_ref(), retained.as_ref()) {
            (None, None) => true,
            (Some(metadata), Some(original)) => {
                Arc::ptr_eq(metadata, &original.metadata)
                    && original_resolver
                        == original.resolver.as_ref().map(|resolver| resolver.as_ptr())
                    && match (original_budget.as_ref(), original.budget.as_ref()) {
                        (None, None) => true,
                        (Some(previous), Some(current)) => previous.same_pool(current),
                        _ => false,
                    }
            }
            _ => false,
        };
        if !unchanged || !fields[2].is_callable() {
            return Err(invalid(
                "material source changed during its original capacity admission",
            ));
        }
        let mut diagnostic_cache = if diagnostic_handoff.is_some() {
            Some(
                self.compiler_diagnostic
                    .lock()
                    .map_err(|_| invalid("original compiler diagnostic cache is poisoned"))?,
            )
        } else {
            None
        };
        let count = fields[1].cast::<PyTuple>()?.len();
        let reservation = reserve_roster(&budget, count)?;
        let mut roster = Vec::with_capacity(count);
        let mut total = 0u64;
        for item in fields[1].cast::<PyTuple>()?.iter() {
            let entry = exact_tuple::<2>(&item)?;
            let digest = read_digest(&entry[0])?;
            let size = positive_u64(&entry[1])?;
            if roster
                .last()
                .is_some_and(|(previous, _): &(String, u64)| previous >= &digest)
            {
                return Err(invalid(
                    "sealed material roster must be strictly sorted without duplicates",
                ));
            }
            if size > item_cap {
                return Err(invalid("sealed material exceeds its original item cap"));
            }
            total = total
                .checked_add(size)
                .ok_or_else(|| invalid("sealed material store size overflow"))?;
            if total > store_cap {
                return Err(invalid(
                    "sealed material roster exceeds its original store cap",
                ));
            }
            roster.push((digest, size));
        }
        if let Some(original) = retained.as_mut() {
            for (digest, size) in &original.metadata.roster {
                if roster
                    .binary_search_by(|(candidate, _)| candidate.cmp(digest))
                    .ok()
                    .is_none_or(|index| roster[index].1 != *size)
                {
                    return Err(invalid("material source extension must preserve its complete original sealed roster"));
                }
            }
            for (digest, size) in &roster {
                if original
                    .metadata
                    .size(digest)
                    .is_some_and(|original| original != *size)
                {
                    return Err(invalid("sealed material digest changed its original size"));
                }
            }
            let mut total = 0u64;
            for (_, size) in &original.metadata.roster {
                total = total
                    .checked_add(*size)
                    .ok_or_else(|| invalid("sealed material store size overflow"))?;
                if total > store_cap {
                    return Err(invalid(
                        "sealed material roster exceeds its original store cap",
                    ));
                }
            }
            for (digest, size) in &roster {
                if original.metadata.size(digest).is_none() {
                    total = total
                        .checked_add(*size)
                        .ok_or_else(|| invalid("sealed material store size overflow"))?;
                    if total > store_cap {
                        return Err(invalid(
                            "sealed material roster exceeds its original store cap",
                        ));
                    }
                }
            }
            if roster
                .iter()
                .any(|(digest, _)| original.metadata.size(digest).is_none())
                || original.metadata._reservation.is_none()
            {
                let capacity = original
                    .metadata
                    .roster
                    .len()
                    .checked_add(roster.len())
                    .ok_or_else(|| invalid("sealed material roster extent overflow"))?;
                let merged_reservation = reserve_roster(&budget, capacity)?;
                let mut merged = Vec::with_capacity(capacity);
                merged.extend(original.metadata.roster.iter().cloned());
                merged.extend(
                    roster
                        .iter()
                        .filter(|(digest, _)| original.metadata.size(digest).is_none())
                        .cloned(),
                );
                merged.sort_unstable_by(|left, right| left.0.cmp(&right.0));
                original.metadata = Arc::new(MaterialSourceMetadata {
                    store,
                    roster: merged,
                    item_cap,
                    store_cap,
                    resolved_cap,
                    _reservation: Some(merged_reservation),
                });
            }
            if original.resolver.is_none() {
                original.budget = Some(budget);
                original.resolver = Some(fields[2].clone().unbind());
            }
        } else {
            *retained = Some(OriginalMaterialSource {
                budget: Some(budget),
                metadata: Arc::new(MaterialSourceMetadata {
                    store,
                    roster,
                    item_cap,
                    store_cap,
                    resolved_cap,
                    _reservation: Some(reservation),
                }),
                resolver: Some(fields[2].clone().unbind()),
            });
        }
        if let Some(family) = diagnostic_handoff {
            let cached = diagnostic_cache
                .as_mut()
                .expect("original diagnostic cache admission");
            assert!(
                cached.available.is_none() && cached.blocking.is_none() && !cached.loaned,
                "original diagnostic family was already admitted"
            );
            cached.available = Some(family);
        }
        Ok(())
    }

    pub(super) fn require_reference(&self, reference: &StoredMaterial) -> PyResult<()> {
        let retained = self
            .original
            .lock()
            .map_err(|_| invalid("material source registry is poisoned"))?;
        let original = retained
            .as_ref()
            .ok_or_else(|| invalid("stored material has no original material source"))?;
        if original.resolver.is_none()
            || original.metadata.store != reference.store
            || original.metadata.size(&reference.digest) != Some(reference.size)
        {
            return Err(invalid(
                "stored material differs from its original bound store and sealed roster",
            ));
        }
        Ok(())
    }

    pub(super) fn resolve(
        &self,
        py: Python<'_>,
        reference: &StoredMaterial,
    ) -> PyResult<xlog_cuda::SemanticReplayLoadedMaterial> {
        let payload = self.resolve_carrier(py, reference)?;
        let budget = self.native_budget()?;
        let loaded = payload.borrow(py).original_loaded(&budget);
        loaded
    }

    pub(super) fn resolve_carrier(
        &self,
        py: Python<'_>,
        reference: &StoredMaterial,
    ) -> PyResult<Py<material_bytes::PySemanticMaterialBytes>> {
        self.require_reference(reference)?;
        let (resolver, budget) = {
            let retained = self
                .original
                .lock()
                .map_err(|_| invalid("material source registry is poisoned"))?;
            let original = retained
                .as_ref()
                .ok_or_else(|| invalid("stored material has no original material source"))?;
            (
                original
                    .resolver
                    .as_ref()
                    .ok_or_else(|| invalid("stored material has no original resolver"))?
                    .clone_ref(py),
                original
                    .budget
                    .as_ref()
                    .ok_or_else(|| invalid("stored material has no original byte pool"))?
                    .clone(),
            )
        };
        self.resolve_carrier_with(py, reference, resolver, &budget)
    }

    fn resolve_carrier_with(
        &self,
        py: Python<'_>,
        reference: &StoredMaterial,
        resolver: Py<PyAny>,
        budget: &xlog_cuda::SemanticReplayMaterialBudget,
    ) -> PyResult<Py<material_bytes::PySemanticMaterialBytes>> {
        // The sealed roster and item/store caps were checked before reserving
        // this original output. No native or registry lock spans the loader.
        let capacity = usize::try_from(reference.size)
            .map_err(|_| invalid("stored material extent exceeds this host"))?;
        let call_extent = python_metadata_tuple_extent(py, 3)?
            .checked_add(python_metadata_text_extent(py, &reference.digest)?)
            .and_then(|bytes| {
                bytes.checked_add(material_bytes::python_instance_extent::<PyInt>(py).ok()?)
            })
            .and_then(|bytes| {
                bytes.checked_add(std::mem::size_of::<u64>() * std::mem::size_of::<u32>())
            })
            .ok_or_else(|| invalid("stored material loader metadata extent overflow"))?;
        let _call_metadata = budget
            .reserve_auxiliary(
                u64::try_from(call_extent)
                    .map_err(|_| invalid("stored material loader metadata exceeds u64"))?,
            )
            .map_err(xlog_err)?;
        let writer = material_bytes::PySemanticMaterialWriter::prepare(py, budget, capacity, 0)?;
        let payload =
            resolver
                .bind(py)
                .call1((&reference.digest, reference.size, writer.clone_ref(py)))?;
        let loaded = writer.borrow(py).require_sealed_material(py, &payload)?;
        let bytes = loaded.as_bytes();
        if u64::try_from(bytes.len()).ok() != Some(reference.size)
            || replay_hash(bytes) != reference.digest
        {
            return Err(invalid(
                "resolved material differs from its original sealed size or SHA256",
            ));
        }
        writer.borrow(py).complete_producer()?;
        Ok(payload
            .cast::<material_bytes::PySemanticMaterialBytes>()?
            .clone()
            .unbind())
    }

    pub(super) fn copy_payload(
        &self,
        bytes: &[u8],
    ) -> PyResult<xlog_cuda::SemanticReplayLoadedMaterial> {
        let budget = self.native_budget()?;
        let size = u64::try_from(bytes.len()).map_err(|_| invalid("replay payload exceeds u64"))?;
        budget
            .reserve(size)
            .map_err(xlog_err)?
            .copy_payload(bytes)
            .map_err(xlog_err)
    }

    pub(super) fn checkpoint_value(
        &self,
    ) -> PyResult<(
        ColdValue,
        Option<xlog_cuda::SemanticReplayMaterialReservation>,
    )> {
        let retained = self
            .original
            .lock()
            .map_err(|_| invalid("material source registry is poisoned"))?;
        let Some(original) = retained.as_ref() else {
            return Ok((ColdValue::None, None));
        };
        let metadata = &original.metadata;
        let decimal_bytes = |mut value: u64| {
            let mut bytes = 1u64;
            while value >= 10 {
                value /= 10;
                bytes += 1;
            }
            // Cover the standard String growth used by integer formatting,
            // including its minimum allocation for short decimal values.
            bytes.max(8).next_power_of_two()
        };
        let count = u64::try_from(metadata.roster.len())
            .map_err(|_| invalid("checkpoint material roster extent does not fit this host"))?;
        let mut bytes = count
            .checked_mul(3)
            .and_then(|count| count.checked_add(7))
            .and_then(|count| count.checked_mul(std::mem::size_of::<ColdValue>() as u64))
            .ok_or_else(|| invalid("checkpoint material metadata extent overflow"))?;
        for value in metadata.store.0.into_iter().chain([
            metadata.item_cap,
            metadata.store_cap,
            metadata.resolved_cap,
        ]) {
            bytes = bytes
                .checked_add(decimal_bytes(value))
                .ok_or_else(|| invalid("checkpoint material metadata extent overflow"))?;
        }
        for (digest, size) in &metadata.roster {
            bytes = bytes
                .checked_add(digest.len() as u64)
                .and_then(|bytes| bytes.checked_add(decimal_bytes(*size)))
                .ok_or_else(|| invalid("checkpoint material metadata extent overflow"))?;
        }
        let reservation = original
            .budget
            .as_ref()
            .ok_or_else(|| invalid("checkpoint material source has not bound its original pool"))?
            .reserve_auxiliary(bytes)
            .map_err(xlog_err)?;
        Ok((
            ColdValue::Sequence(vec![
                ColdValue::Sequence(
                    metadata
                        .store
                        .0
                        .iter()
                        .map(|value| ColdValue::Integer(value.to_string()))
                        .collect(),
                ),
                ColdValue::Sequence(
                    metadata
                        .roster
                        .iter()
                        .map(|(digest, size)| {
                            ColdValue::Sequence(vec![
                                ColdValue::Text(digest.clone()),
                                ColdValue::Integer(size.to_string()),
                            ])
                        })
                        .collect(),
                ),
                ColdValue::Integer(metadata.item_cap.to_string()),
                ColdValue::Integer(metadata.store_cap.to_string()),
                ColdValue::Integer(metadata.resolved_cap.to_string()),
            ]),
            Some(reservation),
        ))
    }

    pub(super) fn restored(value: &ColdValue) -> PyResult<Self> {
        if *value == ColdValue::None {
            return Ok(Self::default());
        }
        let fields = value.fields(5)?;
        let store = fields[0].fields(2)?;
        let metadata = MaterialSourceMetadata {
            store: MaterialStoreIdentity([store[0].unsigned()?, store[1].unsigned()?]),
            roster: fields[1]
                .sequence()?
                .iter()
                .map(|item| {
                    let fields = item.fields(2)?;
                    checked_digest(&fields[0])?;
                    Ok((fields[0].text()?.to_owned(), fields[1].unsigned()?))
                })
                .collect::<PyResult<_>>()?,
            item_cap: fields[2].unsigned()?,
            store_cap: fields[3].unsigned()?,
            resolved_cap: fields[4].unsigned()?,
            _reservation: None,
        };
        if metadata.item_cap == 0
            || metadata.store_cap == 0
            || metadata.resolved_cap == 0
            || metadata.item_cap > metadata.store_cap
            || metadata.roster.len() != fields[1].sequence()?.len()
        {
            return Err(invalid(
                "checkpoint material source has invalid original caps or roster",
            ));
        }
        let mut previous = None::<&str>;
        let mut total = 0u64;
        for item in fields[1].sequence()? {
            let item = item.fields(2)?;
            let digest = item[0].text()?;
            let size = item[1].unsigned()?;
            if size == 0
                || size > metadata.item_cap
                || previous.is_some_and(|previous| previous >= digest)
            {
                return Err(invalid(
                    "checkpoint sealed material roster is not exact sorted bounded metadata",
                ));
            }
            total = total
                .checked_add(size)
                .ok_or_else(|| invalid("checkpoint sealed material size overflow"))?;
            if total > metadata.store_cap {
                return Err(invalid(
                    "checkpoint sealed material roster exceeds its original store cap",
                ));
            }
            previous = Some(digest);
        }
        Ok(Self {
            original: Mutex::new(Some(OriginalMaterialSource {
                metadata: Arc::new(metadata),
                resolver: None,
                budget: None,
            })),
            native_bindings: Mutex::new(NativeReplayBindings::default()),
            compiler_diagnostic: Mutex::new(CompilerDiagnosticCache::default()),
            compiler_diagnostic_acquiring: AtomicBool::new(false),
            owner_metadata: Mutex::new(None),
        })
    }

    pub(super) fn saved_caps(&self) -> PyResult<Option<(u64, u64, u64)>> {
        let retained = self
            .original
            .lock()
            .map_err(|_| invalid("material source registry is poisoned"))?;
        Ok(retained.as_ref().map(|original| {
            (
                original.metadata.item_cap,
                original.metadata.store_cap,
                original.metadata.resolved_cap,
            )
        }))
    }

    pub(super) fn native_budget(&self) -> PyResult<xlog_cuda::SemanticReplayMaterialBudget> {
        let retained = self
            .original
            .lock()
            .map_err(|_| invalid("material source registry is poisoned"))?;
        let original = retained.as_ref().ok_or_else(|| {
            invalid("material source is not bound before native material admission")
        })?;
        if original.resolver.is_none() {
            return Err(invalid("material source must be rebound to its original durable store before native material admission"));
        }
        original
            .budget
            .clone()
            .ok_or_else(|| invalid("material source has no original bound memory pool"))
    }

    pub(super) fn cached_native_binding(&self, key: &str) -> PyResult<Option<NativeReplayBinding>> {
        Ok(self
            .native_bindings
            .lock()
            .map_err(|_| invalid("native replay material cache is poisoned"))?
            .entries
            .iter()
            .find(|(original, _)| original == key)
            .map(|(_, binding)| binding.clone()))
    }

    pub(super) fn bind_native_checkpoint_source(
        &self,
        key: &str,
        digest: [u8; 32],
    ) -> PyResult<()> {
        let mut retained = self
            .native_bindings
            .lock()
            .map_err(|_| invalid("native replay material cache is poisoned"))?;
        let binding = retained
            .entries
            .iter_mut()
            .find(|(original, _)| original == key)
            .map(|(_, binding)| binding)
            .ok_or_else(|| invalid("checkpoint source lost its already-decoded replay owner"))?;
        if binding
            .checkpoint_source
            .is_some_and(|original| original != digest)
        {
            return Err(invalid(
                "original replay changed its verified checkpoint source",
            ));
        }
        binding.checkpoint_source = Some(digest);
        Ok(())
    }

    pub(super) fn retain_native_binding(
        &self,
        key: String,
        binding: NativeReplayBinding,
    ) -> PyResult<NativeReplayBinding> {
        let budget = self.native_budget()?;
        let mut retained = self
            .native_bindings
            .lock()
            .map_err(|_| invalid("native replay material cache is poisoned"))?;
        if let Some((_, original)) = retained
            .entries
            .iter()
            .find(|(original, _)| original == &key)
        {
            return Ok(original.clone());
        }
        let count = retained
            .entries
            .len()
            .checked_add(1)
            .ok_or_else(|| invalid("native replay proof roster extent overflow"))?;
        let bytes = count
            .checked_mul(std::mem::size_of::<(String, NativeReplayBinding)>())
            .ok_or_else(|| invalid("native replay proof roster extent overflow"))?;
        let reservation = budget
            .reserve_auxiliary(
                u64::try_from(bytes)
                    .map_err(|_| invalid("native replay proof roster exceeds u64"))?,
            )
            .map_err(xlog_err)?;
        let result = binding.clone();
        let mut entries = Vec::with_capacity(count);
        entries.append(&mut retained.entries);
        entries.push((key, binding));
        retained.entries = entries;
        retained._reservation = Some(reservation);
        Ok(result)
    }
}

fn reserve_roster(
    budget: &xlog_cuda::SemanticReplayMaterialBudget,
    count: usize,
) -> PyResult<xlog_cuda::SemanticReplayMaterialReservation> {
    let per_entry = std::mem::size_of::<(String, u64)>()
        .checked_add(64)
        .ok_or_else(|| invalid("sealed material metadata extent overflow"))?;
    let bytes = count
        .checked_mul(per_entry)
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<MaterialSourceMetadata>()))
        .and_then(|bytes| bytes.checked_add(2 * std::mem::size_of::<usize>()))
        .ok_or_else(|| invalid("sealed material metadata extent overflow"))?;
    budget
        .reserve_auxiliary(
            u64::try_from(bytes).map_err(|_| invalid("sealed material metadata exceeds u64"))?,
        )
        .map_err(xlog_err)
}

fn exact_tuple<'py, const N: usize>(value: &Bound<'py, PyAny>) -> PyResult<[Bound<'py, PyAny>; N]> {
    if !value.is_exact_instance_of::<PyTuple>() {
        return Err(invalid(
            "material source metadata requires exact builtin tuples",
        ));
    }
    let tuple = value.cast::<PyTuple>()?;
    if tuple.len() != N {
        return Err(invalid("material source metadata has another tuple extent"));
    }
    Ok(std::array::from_fn(|index| {
        tuple
            .get_item(index)
            .expect("validated builtin tuple extent")
    }))
}

fn read_store_identity(value: &Bound<'_, PyAny>) -> PyResult<MaterialStoreIdentity> {
    let fields = exact_tuple::<2>(value)?;
    Ok(MaterialStoreIdentity([
        nonnegative_u64(&fields[0])?,
        nonnegative_u64(&fields[1])?,
    ]))
}

fn nonnegative_u64(value: &Bound<'_, PyAny>) -> PyResult<u64> {
    if !value.is_exact_instance_of::<PyInt>() {
        return Err(invalid(
            "material source integers require exact nonnegative u64 values",
        ));
    }
    value
        .extract()
        .map_err(|_| invalid("material source integer exceeds nonnegative u64"))
}

fn positive_u64(value: &Bound<'_, PyAny>) -> PyResult<u64> {
    let value = nonnegative_u64(value)?;
    if value == 0 {
        return Err(invalid("sealed material size must be positive"));
    }
    Ok(value)
}

fn read_digest(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if !value.is_exact_instance_of::<PyString>() {
        return Err(invalid("sealed material digest requires an exact string"));
    }
    let digest = value.cast::<PyString>()?.to_str()?;
    validate_digest(digest)?;
    Ok(digest.to_owned())
}

pub(super) fn validate_digest(digest: &str) -> PyResult<()> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(invalid(
            "sealed material digest is not canonical lowercase SHA256",
        ));
    }
    Ok(())
}
