//! Retained authority authenticated against the complete immutable donor checkpoint.

use super::*;

pub(super) struct AuthenticatedSourceAuthority {
    pub(super) authority: Arc<TaskAuthority>,
    pub(super) saved_snapshot: AuthoritySnapshot,
    origin: xlog_cuda::SemanticEvaluationForkOrigin,
    registry: Arc<material_source::MaterialSourceRegistry>,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

impl AuthenticatedSourceAuthority {
    pub(super) fn require_origin(
        &self,
        origin: &xlog_cuda::SemanticEvaluationForkOrigin,
        registry: &Arc<material_source::MaterialSourceRegistry>,
    ) -> PyResult<()> {
        if !origin.is_resolved()
            || !Arc::ptr_eq(&self.registry, registry)
            || self.origin.source_material_reference() != origin.source_material_reference()
            || self.origin.source_identity() != origin.source_identity()
            || self.origin.source_task_identity() != origin.source_task_identity()
            || self.origin.member() != origin.member()
        {
            return Err(invalid(
                "retained source authority changed its authenticated native owner",
            ));
        }
        Ok(())
    }
}

pub(super) fn authenticate(
    dormant: &DormantEvaluationSource,
    origin: &xlog_cuda::SemanticEvaluationForkOrigin,
    registry: &Arc<material_source::MaterialSourceRegistry>,
) -> PyResult<Arc<AuthenticatedSourceAuthority>> {
    if dormant
        .source_material_sources
        .as_ref()
        .is_some_and(|original| !Arc::ptr_eq(original, registry))
    {
        return Err(invalid(
            "evaluation fork changed its original source material owner",
        ));
    }
    let retained = {
        dormant
            .authenticated
            .lock()
            .map_err(|_| invalid("authenticated source authority owner is poisoned"))?
            .as_ref()
            .cloned()
    };
    if let Some(retained) = retained {
        retained.require_origin(origin, registry)?;
        return Ok(retained);
    }

    // No cache or native Session mutex crosses source decoding or a CAS resolver.
    // The native origin retains the actual sealed source body throughout this
    // first authentication. Known restoration retirement can then release the
    // archive while preserving this authenticated authority and original origin.
    // Fresh restoration resolves and authenticates the durable source again.
    let fields = dormant.value.fields(7)?;
    let section = |index: usize| -> PyResult<&[u8]> {
        match &fields[index] {
            ColdValue::Bytes(bytes) => Ok(bytes.as_ref()),
            _ => Err(invalid("evaluation source checkpoint section is absent")),
        }
    };
    let referent = CheckpointReferent::decode(section(0)?)?;
    let pool = registry.native_budget()?;
    let native = origin.source_material().map_err(xlog_err)?;
    native.require_budget(&pool).map_err(xlog_err)?;
    let manifest = BorrowedSemanticCheckpointManifest {
        native: native.as_bytes(),
        model: section(1)?,
        task: section(2)?,
        session: section(3)?,
        initial_prefill: section(4)?,
        model_context: section(5)?,
        phase_seal: section(6)?,
    };
    let mut digest = Sha256::new();
    manifest.write_encoded(|part| {
        digest.update(part);
        Ok(())
    })?;
    if <[u8; 32]>::from(digest.finalize()) != referent.checkpoint_digest
        || u64::try_from(manifest.encoded_extent()?).ok() != Some(referent.checkpoint_bytes)
        || checkpoint_publication_bytes(origin.source_identity()) != referent.publication
        || origin.source_task_identity().as_bytes() != &referent.task.identity
    {
        return Err(invalid(
            "evaluation source archive differs from its complete original checkpoint",
        ));
    }

    // Charge the retained parsed authority and saved snapshot before decoding.
    // The bounded decoder separately owns its transient tree and bound-payload
    // projections. The retained native source slice is shared, never copied.
    let plan = checked_checkpoint_cold_value(manifest.task, manifest.task.len(), false, None)?;
    let extent = plan
        .owned_bytes
        .checked_mul(3)
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<TaskAuthority>() as u64))
        .and_then(|bytes| {
            bytes.checked_add(std::mem::size_of::<AuthenticatedSourceAuthority>() as u64)
        })
        .and_then(|bytes| bytes.checked_add((4 * std::mem::size_of::<usize>()) as u64))
        .ok_or_else(|| invalid("retained source authority metadata extent overflow"))?;
    let metadata = pool.reserve_auxiliary(extent).map_err(xlog_err)?;
    let (seed, saved_snapshot, phase, binding) =
        TaskCheckpointSeed::decode_bound(manifest.task, registry)?;
    if seed.dormant_evaluation_source.is_some() || binding != referent.task {
        return Err(invalid(
            "evaluation source task differs from its original checkpoint binding",
        ));
    }
    let mut authority = TaskAuthority::parse(&seed.authority)?;
    authority.bind_initial_sources(&seed.initial_sources, &seed.source_mapping)?;
    if checkpoint_scope_digest(&authority.scope) != referent.scope_digest
        || authority.authority_root().as_bytes() != &referent.authority_root
        || <[u8; 32]>::from(Sha256::digest(&saved_snapshot.canonical)) != referent.snapshot_digest
        || matches!(phase, CheckpointTaskPhase::Imported)
    {
        return Err(invalid(
            "evaluation source authority differs from its original checkpoint",
        ));
    }
    let authenticated = Arc::new(AuthenticatedSourceAuthority {
        authority: Arc::new(authority),
        saved_snapshot,
        origin: origin.clone(),
        registry: Arc::clone(registry),
        _metadata: metadata,
    });
    authenticated.require_origin(origin, registry)?;
    let mut retained = dormant
        .authenticated
        .lock()
        .map_err(|_| invalid("authenticated source authority owner is poisoned"))?;
    if let Some(original) = retained.as_ref() {
        original.require_origin(origin, registry)?;
        return Ok(Arc::clone(original));
    }
    *retained = Some(Arc::clone(&authenticated));
    Ok(authenticated)
}
