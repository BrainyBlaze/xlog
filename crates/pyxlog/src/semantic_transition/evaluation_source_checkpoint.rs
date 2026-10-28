//! Guarded checkpoint acquisition for dormant evaluation donors.

use super::*;

/// Acquire the complete original checkpoint through the registered writer
/// protocol. The caller owns roster-wide accounting and authority refresh.
pub(super) fn load_checkpoint_source(
    py: Python<'_>,
    referent: &CheckpointReferent,
    source_registry: &Arc<material_source::MaterialSourceRegistry>,
    resolver: &Bound<'_, PyAny>,
    limits: CheckpointSourceLimits,
) -> PyResult<xlog_cuda::SemanticReplayLoadedMaterial> {
    if !resolver.is_callable() {
        return Err(invalid(
            "checkpoint source requires its original callable resolver",
        ));
    }
    let length = usize::try_from(referent.checkpoint_bytes)
        .map_err(|_| invalid("checkpoint source extent exceeds this host"))?;
    if length == 0 || length > limits.item_bytes || length > limits.total_bytes {
        return Err(invalid(
            "checkpoint source exceeds its original finite byte limits",
        ));
    }
    let budget = source_registry.native_budget()?;
    let callback_extent = python_metadata_tuple_extent(py, 3)?
        .checked_add(material_bytes::python_instance_extent::<PyBytes>(py)?)
        .and_then(|bytes| bytes.checked_add(referent.checkpoint_digest.len() + 1))
        .and_then(|bytes| {
            bytes.checked_add(material_bytes::python_instance_extent::<PyInt>(py).ok()?)
        })
        .and_then(|bytes| {
            bytes.checked_add(std::mem::size_of::<u64>() * std::mem::size_of::<u32>())
        })
        .ok_or_else(|| invalid("checkpoint loader callback metadata extent overflow"))?;
    let _callback_metadata = budget
        .reserve_auxiliary(
            u64::try_from(callback_extent)
                .map_err(|_| invalid("checkpoint loader callback metadata exceeds u64"))?,
        )
        .map_err(xlog_err)?;
    let writer = material_bytes::PySemanticMaterialWriter::prepare(py, &budget, length, 0)?;
    let returned = resolver.call1((
        PyBytes::new(py, &referent.checkpoint_digest),
        referent.checkpoint_bytes,
        writer.clone_ref(py),
    ))?;
    let source = writer.borrow(py).require_sealed_material(py, &returned)?;
    if source.as_bytes().len() != length {
        return Err(invalid(
            "resolved checkpoint differs from its original sealed extent",
        ));
    }
    referent.verify_source_identity(source.as_bytes())?;
    writer.borrow(py).complete_producer()?;
    Ok(source)
}

fn source_section(source: &DormantEvaluationSource, index: usize) -> PyResult<&[u8]> {
    match &source.value.fields(7)?[index] {
        ColdValue::Bytes(bytes) => Ok(bytes.as_ref()),
        _ => Err(invalid(
            "dormant evaluation source lost an original checkpoint section",
        )),
    }
}

fn snapshot_projection(value: &Bound<'_, PyAny>) -> PyResult<AuthoritySnapshot> {
    AuthoritySnapshot::parse(&ColdValue::read(value, &mut (16 * 1024 * 1024), 0)?)
}

/// Bind a saved fork's native reference to its complete immutable donor before
/// restoration. Return the actual authority pair observed after external I/O.
#[expect(
    clippy::too_many_arguments,
    reason = "donor acquisition keeps its scoped snapshots, original material owner and finite resolver explicit"
)]
pub(super) fn bind_dormant_source(
    py: Python<'_>,
    target_native: &mut xlog_cuda::SemanticReplayLoadedMaterial,
    dormant: &DormantEvaluationSource,
    source_registry: &Arc<material_source::MaterialSourceRegistry>,
    snapshot: &Bound<'_, PyAny>,
    resolve_checkpoint: Option<&Bound<'_, PyAny>>,
    max_checkpoint_bytes: Option<&Bound<'_, PyAny>>,
    max_total_checkpoint_bytes: Option<&Bound<'_, PyAny>>,
    refresh_snapshot: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let budget = source_registry.native_budget()?;
    target_native.require_budget(&budget).map_err(xlog_err)?;
    let proof = target_native.checkpoint_proof().map_err(xlog_err)?;
    let origin = proof
        .evaluation_fork_origin()
        .ok_or_else(|| invalid("dormant evaluation source has no original native fork binding"))?;
    let referent = CheckpointReferent::decode(source_section(dormant, 0)?)?;
    if checkpoint_publication_bytes(origin.source_identity()) != referent.publication
        || origin.source_task_identity().as_bytes() != &referent.task.identity
    {
        return Err(invalid(
            "evaluation donor reference changed its original publication or task",
        ));
    }
    let (source_snapshot, target_projection) = evaluation_authority_snapshots(snapshot)?;
    dormant.require_current_source_snapshot(&source_snapshot, &budget)?;
    let target_snapshot = snapshot_projection(&target_projection)?;
    let (seed, saved_snapshot, _, binding) =
        TaskCheckpointSeed::decode_bound(source_section(dormant, 2)?, source_registry)?;
    if seed.dormant_evaluation_source.is_some() || binding != referent.task {
        return Err(invalid(
            "evaluation donor must be its original ordinary task checkpoint",
        ));
    }
    let mut authority = TaskAuthority::parse(&seed.authority)?;
    authority.bind_initial_sources(&seed.initial_sources, &seed.source_mapping)?;
    source_snapshot.newer_than(&saved_snapshot)?;
    authority.check_use("inference", &saved_snapshot, false)?;
    authority.check_use("inference", &source_snapshot, false)?;
    if origin.is_resolved() {
        dormant.require_native(Some(origin), &source_snapshot, source_registry)?;
        return Ok(snapshot.clone().unbind());
    }
    let resolver = resolve_checkpoint
        .filter(|callback| callback.is_callable())
        .ok_or_else(|| {
            invalid("evaluation restore requires its original bounded checkpoint resolver")
        })?;
    let refresh = refresh_snapshot
        .filter(|callback| callback.is_callable())
        .ok_or_else(|| {
            invalid("evaluation restore requires its original authority pair refresh")
        })?;
    let read_limit = |value: Option<&Bound<'_, PyAny>>, name: &str| -> PyResult<u64> {
        ColdValue::read(
            value.ok_or_else(|| invalid(&format!("evaluation restore requires {name}")))?,
            &mut 128,
            0,
        )?
        .unsigned()
    };
    let limits = CheckpointSourceLimits::new(
        read_limit(max_checkpoint_bytes, "max_checkpoint_bytes")?,
        read_limit(max_total_checkpoint_bytes, "max_total_checkpoint_bytes")?,
    )?;
    let source = load_checkpoint_source(py, &referent, source_registry, resolver, limits)?;
    let manifest = SemanticCheckpointManifest::decode_borrowed(source.as_bytes())?;
    for (index, actual) in [
        manifest.model,
        manifest.task,
        manifest.session,
        manifest.initial_prefill,
        manifest.model_context,
        manifest.phase_seal,
    ]
    .into_iter()
    .enumerate()
    {
        if source_section(dormant, index + 1)? != actual {
            return Err(invalid(
                "resolved donor changed an original checkpoint metadata section",
            ));
        }
    }
    let mut source_native = source
        .slice_payload(checkpoint_section_range(
            source.as_bytes(),
            manifest.native,
        )?)
        .map_err(xlog_err)?;
    referent.verify_source_bound(source.as_bytes(), source_registry, &mut source_native)?;
    let source_proof = source_native.checkpoint_proof().map_err(xlog_err)?;
    if source_proof.evaluation_fork_origin().is_some() {
        return Err(invalid(
            "a saved evaluation fork cannot supply another evaluation donor",
        ));
    }
    let (digest, extent) = origin.source_material_reference();
    if u64::try_from(source_native.as_bytes().len()).ok() != Some(extent)
        || Sha256::digest(source_native.as_bytes()).as_slice() != digest.as_bytes()
    {
        return Err(invalid(
            "resolved donor native material differs from its original reference",
        ));
    }
    // Neither native Session, task state nor authority-pair locks cross this
    // callback. The original sealed donor remains guarded throughout refresh.
    let observed = refresh.call0()?;
    let (latest_source, latest_target_projection) = evaluation_authority_snapshots(&observed)?;
    let latest_target = snapshot_projection(&latest_target_projection)?;
    latest_source.newer_than(&source_snapshot)?;
    latest_target.newer_than(&target_snapshot)?;
    authority.check_use("inference", &latest_source, false)?;
    target_native
        .bind_evaluation_fork_source(source_native)
        .map_err(xlog_err)?;
    // Binding replaces the unresolved origin; a previously captured proof is
    // deliberately not used to authenticate the newly retained source owner.
    let bound_proof = target_native.checkpoint_proof().map_err(xlog_err)?;
    dormant.require_native(
        bound_proof.evaluation_fork_origin(),
        &latest_source,
        source_registry,
    )?;
    Ok(observed.unbind())
}
