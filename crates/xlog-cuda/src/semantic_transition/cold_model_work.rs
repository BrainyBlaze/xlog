//! Model expenditure inside an original admitted cold operation. This owner has
//! no evaluation, backward-tape, content-admission or publication authority.

use super::*;
use crate::semantic_work::ModelWorkPlan;

#[derive(Clone, Debug)]
pub struct SemanticColdModelWork {
    issuer: Arc<()>,
    invocation: Arc<()>,
    token: u64,
    operation_ordinal: u64,
    admission: Arc<[u8]>,
}

/// A single registration region inside the original cold report. Regions
/// append to the same event roster and device slots; they never reset or reopen
/// the enclosing report and grant no numerical evaluation authority.
#[derive(Clone, Debug)]
pub struct SemanticColdModelWorkRegion {
    work: SemanticColdModelWork,
    index: usize,
}

impl SemanticColdModelWork {
    /// Checked bytes for this exact native work-buffer ABI, not measured usage.
    pub fn allocation_bytes(capacity: usize) -> Result<usize, SemanticTransitionError> {
        if capacity == 0 {
            return Err(publication_input_error(
                "cold model work requires a positive original event capacity",
            ));
        }
        capacity
            .checked_mul(size_of::<ModelWorkEvent>() + 3 * size_of::<u64>())
            .and_then(|bytes| bytes.checked_add((15 + 11) * size_of::<u64>()))
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or(SemanticTransitionError::GenerationExhausted)
    }
}

/// The original operation's native tally carried into its actual child graph.
/// Construction is private; neither a raw pointer nor another report can mint
/// this custody. Allocation and stream identity remain those of the issuer.
#[derive(Clone)]
pub struct SemanticColdNativeWork {
    provider: Arc<CudaKernelProvider>,
    domain: ResidentExecutionDomain,
    work: DeviceMemoryView<u64>,
    custody: Arc<()>,
}

impl SemanticColdNativeWork {
    pub(crate) fn graph_custody(
        self,
        provider: &Arc<CudaKernelProvider>,
        domain: &ResidentExecutionDomain,
    ) -> Result<(DeviceMemoryView<u64>, Arc<()>), SemanticTransitionError> {
        if !Arc::ptr_eq(provider, &self.provider) || domain.stream_id() != self.domain.stream_id() {
            return Err(publication_input_error(
                "cold child construction changed its original allocation owner or stream",
            ));
        }
        Ok((self.work, self.custody))
    }
}

/// Actual model and native components, not the complete operation's
/// native work or physical peak. Other cold producers remain independent.
#[derive(Clone, Copy, Debug)]
pub struct SemanticColdModelWorkResult {
    pub model_work: u64,
    pub operation_count: u64,
    pub work_bound: u64,
    pub model_calls: u64,
    pub native_work: u64,
    pub native_events: [u64; 9],
}

/// Authored by the enclosing native operation after its known outcome, never by
/// a callback error or a Python declaration of partial work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticColdModelWorkDisposition {
    Complete,
    KnownRefusal,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RecordingState {
    Waiting,
    Recording,
    Closed,
    Failed,
    Submitting,
    Submitted,
    Completed,
    NotEntered,
}

pub(super) struct ColdModelWorkStorage {
    invocation: Arc<()>,
    aliases: Arc<()>,
    operation_ordinal: u64,
    admission: Arc<[u8]>,
    work: PreparedModelWork,
    native_work: TrackedCudaSlice<u64>,
    report: TrackedCudaSlice<u64>,
    state: RecordingState,
    regions: Vec<RecordingState>,
    plan: Option<ModelWorkPlan>,
    disposition: Option<SemanticColdModelWorkDisposition>,
    result: Option<SemanticColdModelWorkResult>,
    streams: Option<Vec<u64>>,
}

impl ColdModelWorkStorage {
    fn plan(&self) -> Result<&ModelWorkPlan, SemanticTransitionError> {
        self.plan.as_ref().ok_or_else(|| {
            publication_input_error("cold recording requires its admitted operation plan")
        })
    }

    fn plan_mut(&mut self) -> Result<&mut ModelWorkPlan, SemanticTransitionError> {
        self.plan.as_mut().ok_or_else(|| {
            publication_input_error("cold recording requires its admitted operation plan")
        })
    }
}

impl SemanticTransitionSession {
    pub(super) fn cold_native_work(
        &self,
        token: u64,
    ) -> Result<Option<DeviceMemoryView<u64>>, SemanticTransitionError> {
        // Captured steps already own their execution tally. Keep a borrowed
        // cold tally out of capture, but retain it for final-use guards and
        // resource retirement after the actual completed handoff.
        if self
            .steps
            .get(&token)
            .is_some_and(|step| step.prepared.is_some())
            && self
                .prepared_segment
                .as_ref()
                .is_some_and(|build| build.capturing)
        {
            return Ok(None);
        }
        let Some(storage) = self
            .steps
            .get(&token)
            .and_then(|step| step.cold_model_work.as_ref())
        else {
            return Ok(self.graph.borrowed_cold_work());
        };
        match storage.state {
            RecordingState::Waiting | RecordingState::Recording | RecordingState::Closed => {
                Ok(Some(storage.native_work.view()))
            }
            // A completed local report does not hide the next original
            // operation's borrowed tally. Late verification and retirement of
            // this reader must charge their actual commands to that live owner.
            RecordingState::Completed => Ok(self.graph.borrowed_cold_work()),
            _ => Err(publication_input_error(
                "cold native producers cannot run after failure or report submission",
            )),
        }
    }

    /// Mint only from the retained original invocation, before child admission.
    /// The guard counts as a live alias until that actual child stream joins.
    pub fn share_cold_native_work(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<SemanticColdNativeWork, SemanticTransitionError> {
        let storage = self.cold_model_work(handle)?;
        if !matches!(
            storage.state,
            RecordingState::Waiting | RecordingState::Recording
        ) {
            return Err(publication_input_error(
                "cold child construction requires its original open operation",
            ));
        }
        Ok(SemanticColdNativeWork {
            provider: Arc::clone(&self.provider),
            domain: self.domain.clone(),
            work: storage.native_work.view(),
            custody: Arc::clone(&storage.aliases),
        })
    }

    /// Completion removes only a borrowed tally, after the actual reader's
    /// entire consumer roster joins. Unknown completion keeps the same guard.
    pub fn complete_shared_cold_native_work(
        &mut self,
        lease: &SemanticPublishedLease,
        streams: &[u64],
    ) -> Result<(), SemanticTransitionError> {
        self.checked_reader(lease)?;
        let work = self.graph.borrowed_cold_work().ok_or_else(|| {
            publication_input_error("child completion lost its original borrowed tally")
        })?;
        self.complete_step_consumers(lease, streams)?;
        self.graph
            .complete_cold_work(&work)
            .map_err(SemanticTransitionError::Semantic)
    }

    /// Attach the original phase's open tally to an already restored reader,
    /// before its next actual cold preparation. No new counter is allocated.
    pub fn attach_shared_cold_native_work(
        &mut self,
        lease: &SemanticPublishedLease,
        work: SemanticColdNativeWork,
    ) -> Result<(), SemanticTransitionError> {
        self.checked_reader(lease)?;
        let (view, custody) = work.graph_custody(&self.provider, &self.domain)?;
        self.graph
            .begin_borrowed_cold_work(view, custody)
            .map_err(SemanticTransitionError::Semantic)
    }

    /// Remove visibility of this exact borrowed tally before its issuer joins
    /// and reads the report. Detachment is not completion or resource release;
    /// the original work allocation and unfinished report remain retained.
    pub fn detach_shared_cold_native_work(
        &mut self,
        work: &SemanticColdNativeWork,
    ) -> Result<(), SemanticTransitionError> {
        if !Arc::ptr_eq(&self.provider, &work.provider)
            || self.domain.stream_id() != work.domain.stream_id()
        {
            return Err(publication_input_error(
                "cold detachment changed its original allocation owner or stream",
            ));
        }
        self.graph
            .complete_cold_work(&work.work)
            .map_err(SemanticTransitionError::Semantic)
    }

    /// Keep the same open operation alive at its already attached surviving
    /// reader before the private issuer is completely retired. Move the actual
    /// storage and region states: no allocation, copy, reset or new invocation.
    #[cfg(feature = "semantic-policy")]
    pub fn handoff_cold_model_work(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticColdModelWork,
        survivor: &mut SemanticTransitionSession,
        survivor_lease: &SemanticPublishedLease,
        streams: &[u64],
    ) -> Result<(SemanticColdModelWork, Vec<SemanticColdModelWorkRegion>), SemanticTransitionError>
    {
        self.checked_reader(lease)?;
        survivor.checked_reader(survivor_lease)?;
        survivor.require_completed_cold_model_work()?;
        let original = self.cold_model_work(handle)?;
        if lease.token != handle.token
            || !Arc::ptr_eq(&self.provider, &survivor.provider)
            || self.domain.stream_id() != survivor.domain.stream_id()
            || Arc::ptr_eq(&self.publication_issuer, &survivor.publication_issuer)
            || original.state != RecordingState::Recording
            || original.regions.is_empty()
            || original
                .regions
                .iter()
                .any(|state| !matches!(state, RecordingState::Waiting | RecordingState::Closed))
            || survivor
                .checked_step(survivor_lease)?
                .cold_model_work
                .as_ref()
                .is_some_and(|prior| {
                    !Arc::ptr_eq(&prior.admission, &original.admission)
                        || prior.operation_ordinal > original.operation_ordinal
                        || prior.work.actual.len() != original.work.actual.len()
                })
        {
            return Err(publication_input_error(
                "cold ownership handoff requires the same known open operation and attached survivor",
            ));
        }
        // All actual consumers join before changing graph visibility. A failed
        // join leaves the same issuer/report retained, never a second transfer.
        self.quiesce_published_reader(lease, streams)?;
        survivor.quiesce_published_reader(survivor_lease, streams)?;
        let original = self.cold_model_work(handle)?;
        let view = original.native_work.view();
        let custody = Arc::clone(&original.aliases);
        let regions = original.regions.len();
        self.graph
            .require_cold_work_owner(&view, false)
            .map_err(SemanticTransitionError::Semantic)?;
        survivor
            .graph
            .require_cold_work_owner(&view, true)
            .map_err(SemanticTransitionError::Semantic)?;
        // Validation and joins above precede these ownership-only changes.
        // The old storage remains live until the graph ownership is transferred.
        self.graph
            .complete_cold_work(&view)
            .map_err(SemanticTransitionError::Semantic)?;
        survivor
            .graph
            .complete_cold_work(&view)
            .map_err(SemanticTransitionError::Semantic)?;
        survivor
            .graph
            .begin_cold_work(view.clone())
            .map_err(SemanticTransitionError::Semantic)?;
        self.graph
            .begin_borrowed_cold_work(view, custody)
            .map_err(SemanticTransitionError::Semantic)?;
        let storage = self
            .steps
            .get_mut(&lease.token)
            .expect("checked original reader")
            .cold_model_work
            .take()
            .expect("checked original cold report");
        let transferred = SemanticColdModelWork {
            issuer: Arc::clone(&survivor.publication_issuer),
            invocation: Arc::clone(&storage.invocation),
            token: survivor_lease.token,
            operation_ordinal: storage.operation_ordinal,
            admission: Arc::clone(&storage.admission),
        };
        survivor
            .steps
            .get_mut(&survivor_lease.token)
            .expect("checked surviving reader")
            .cold_model_work = Some(storage);
        Ok((
            transferred.clone(),
            (0..regions)
                .map(|index| SemanticColdModelWorkRegion {
                    work: transferred.clone(),
                    index,
                })
                .collect(),
        ))
    }

    /// The actual model recorder must have closed before a prepared graph can
    /// take over numerical execution. Waiting or failed registration is not a
    /// CPU-only or zero-work observation.
    pub fn require_closed_cold_model_work(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        if self.cold_model_work(handle)?.state != RecordingState::Closed {
            return Err(publication_input_error(
                "prepared execution requires its original closed cold preparation recorder",
            ));
        }
        Ok(())
    }

    pub(super) fn require_completed_cold_model_work(&self) -> Result<(), SemanticTransitionError> {
        if self.steps.values().any(|step| {
            step.cold_model_work.as_ref().is_some_and(|storage| {
                storage.state != RecordingState::Completed
                    || Arc::strong_count(&storage.aliases) != 1
            })
        }) {
            return Err(publication_input_error(
                "state-changing work or release requires completed cold model work without scratch aliases",
            ));
        }
        Ok(())
    }

    /// Called only after the original signed admission and physical interval
    /// have begun. Retention precedes every reset or external model callback.
    pub fn prepare_cold_model_work(
        &mut self,
        lease: &SemanticPublishedLease,
        capacity: usize,
        operation_ordinal: u64,
        admission: Arc<[u8]>,
    ) -> Result<SemanticColdModelWork, SemanticTransitionError> {
        self.ensure_quiescent()?;
        self.require_completed_cold_model_work()?;
        self.checked_reader(lease)?;
        let step = self.checked_step(lease)?;
        // A later actual callback is a new invocation, never a reopened
        // recording. The previous report is complete and its scratch aliases
        // are gone before replacement; the enclosing operation retains its
        // returned immutable result and physical interval independently.
        if step.cold_model_work.as_ref().is_some_and(|previous| {
            !Arc::ptr_eq(&previous.admission, &admission)
                || previous.work.actual.len() / 3 != capacity
                || previous.operation_ordinal > operation_ordinal
        }) {
            return Err(publication_input_error(
                "a later cold callback cannot change its original admission, capacity or operation order",
            ));
        }
        if capacity == 0
            || admission.is_empty()
            || self.prepared_segment.is_some()
            || step.prepared.is_some()
        {
            return Err(publication_input_error(
                "cold model work requires its sole admitted operation and positive original capacity",
            ));
        }
        self.retire_completed_evaluation_storage(lease)?;
        let bytes = SemanticColdModelWork::allocation_bytes(capacity)?;
        let mut reservation = self
            .provider
            .memory()
            .reserve_bytes(bytes as u64)
            .map_err(|error| runtime_error("cold model work reservation", error))?;
        let work = PreparedModelWork::allocate(&self.provider, &mut reservation, capacity)?;
        let native_work = reservation
            .alloc(11)
            .map_err(|error| runtime_error("cold semantic work allocation", error))?;
        let report = reservation
            .alloc(15)
            .map_err(|error| runtime_error("cold model work report allocation", error))?;
        let invocation = Arc::new(());
        let handle = SemanticColdModelWork {
            issuer: Arc::clone(&self.publication_issuer),
            invocation: Arc::clone(&invocation),
            token: lease.token,
            operation_ordinal,
            admission: Arc::clone(&admission),
        };
        // Keep the completed storage until all new allocation succeeds. The
        // new invocation is retained before reset or any model callback; an
        // allocation failure grants no rights to replay the prior callback.
        self.steps
            .get_mut(&lease.token)
            .expect("checked original reader")
            .cold_model_work = Some(ColdModelWorkStorage {
            invocation,
            aliases: Arc::new(()),
            operation_ordinal,
            admission,
            work,
            native_work,
            report,
            state: RecordingState::Waiting,
            regions: Vec::new(),
            plan: None,
            disposition: None,
            result: None,
            streams: None,
        });
        self.steps[&lease.token]
            .cold_model_work
            .as_ref()
            .expect("retained cold work")
            .work
            .reset_slots(&self.domain, &mut self.poisoned, 0, capacity)?;
        let storage = self
            .steps
            .get_mut(&lease.token)
            .expect("retained reader")
            .cold_model_work
            .as_mut()
            .expect("retained cold work");
        // Initialize the actual tally once, before the original semantic commands.
        // This fixed instrumentation metadata is not a fabricated work occurrence.
        self.provider
            .htod_launch_metadata_sync_copy_into(&[0u64; 11], &mut storage.native_work)
            .map_err(|error| runtime_error("cold semantic work initialization", error))?;
        self.graph
            .begin_cold_work(storage.native_work.view())
            .map_err(SemanticTransitionError::Semantic)?;
        Ok(handle)
    }

    fn cold_model_work(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<&ColdModelWorkStorage, SemanticTransitionError> {
        self.ensure_quiescent()?;
        let work = self
            .steps
            .get(&handle.token)
            .and_then(|step| step.cold_model_work.as_ref())
            .ok_or_else(|| {
                publication_input_error("cold model work lost its original operation")
            })?;
        if !Arc::ptr_eq(&handle.issuer, &self.publication_issuer)
            || !Arc::ptr_eq(&handle.invocation, &work.invocation)
            || !Arc::ptr_eq(&handle.admission, &work.admission)
            || handle.operation_ordinal != work.operation_ordinal
        {
            return Err(publication_input_error(
                "cold model work belongs to another Session, admission or operation",
            ));
        }
        Ok(work)
    }

    pub fn cold_model_work_stream(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<u64, SemanticTransitionError> {
        self.cold_model_work(handle)?;
        Ok(self.stream.cu_stream() as u64)
    }

    /// Admit the complete ordered model recipe before its first actual recorder
    /// begins. An enclosing region report may already be open without any model
    /// event. Returned quantities are (work bound, event count, call upper).
    pub fn admit_cold_model_work_plan(
        &mut self,
        handle: &SemanticColdModelWork,
        operations: &[(ModelWorkKind, Vec<u64>)],
        region_ends: &[usize],
    ) -> Result<[u64; 3], SemanticTransitionError> {
        self.admit_cold_model_work_plan_in_region(handle, None, operations, region_ends)
    }

    pub fn admit_cold_model_work_region_plan(
        &mut self,
        region: &SemanticColdModelWorkRegion,
        operations: &[(ModelWorkKind, Vec<u64>)],
        region_ends: &[usize],
    ) -> Result<[u64; 3], SemanticTransitionError> {
        self.admit_cold_model_work_plan_in_region(
            &region.work,
            Some(region),
            operations,
            region_ends,
        )
    }

    fn admit_cold_model_work_plan_in_region(
        &mut self,
        handle: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
        operations: &[(ModelWorkKind, Vec<u64>)],
        region_ends: &[usize],
    ) -> Result<[u64; 3], SemanticTransitionError> {
        let result = (|| {
            if let Some(region) = region {
                self.require_cold_model_work_region(region, RecordingState::Waiting)?;
            } else {
                let storage = self.cold_model_work(handle)?;
                if storage.state != RecordingState::Waiting || !storage.regions.is_empty() {
                    return Err(publication_input_error(
                        "cold plan requires its original waiting recorder",
                    ));
                }
            }
            let storage = self.cold_model_work(handle)?;
            if storage.plan.is_some()
                || !storage.work.recording.events().is_empty()
                || storage.regions.iter().any(|state| {
                    !matches!(state, RecordingState::Waiting | RecordingState::NotEntered)
                })
            {
                return Err(publication_input_error(
                    "the original cold operation plan is admitted once",
                ));
            }
            let capacity = storage.work.actual.len() / 3;
            let mut plan =
                ModelWorkPlan::new(capacity, operations, region_ends, storage.regions.len())
                    .map_err(publication_input_error)?;
            for (index, state) in storage.regions.iter().enumerate() {
                if *state == RecordingState::NotEntered {
                    plan.skip_unentered_region(index)
                        .map_err(publication_input_error)?;
                }
            }
            let quantities = plan.quantities();
            self.steps
                .get_mut(&handle.token)
                .expect("checked reader")
                .cold_model_work
                .as_mut()
                .expect("checked cold work")
                .plan = Some(plan);
            Ok(quantities)
        })();
        if result.is_err() {
            self.fail_cold_model_work(handle);
        }
        result
    }

    /// Read the retained complete report plan before admitting dependent work.
    /// Native-authorized skipped regions never change these upper quantities.
    pub fn cold_model_work_plan_quantities(
        &self,
        handle: &SemanticColdModelWork,
    ) -> Result<[u64; 3], SemanticTransitionError> {
        Ok(self.cold_model_work(handle)?.plan()?.quantities())
    }

    pub fn cold_model_work_buffer(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticColdModelWork,
        stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.cold_model_work_buffer_for_region(lease, handle, stream, None)
    }

    pub fn cold_model_work_region_buffer(
        &mut self,
        lease: &SemanticPublishedLease,
        region: &SemanticColdModelWorkRegion,
        stream: u64,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.cold_model_work_buffer_for_region(lease, &region.work, stream, Some(region))
    }

    fn cold_model_work_buffer_for_region(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticColdModelWork,
        stream: u64,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<DlpackManagedTensor, SemanticTransitionError> {
        self.checked_reader(lease)?;
        let storage = self.cold_model_work(handle)?;
        let available = if let Some(region) = region {
            self.require_cold_model_work_region(region, RecordingState::Waiting)?;
            storage.state == RecordingState::Recording
        } else {
            storage.regions.is_empty() && storage.state == RecordingState::Waiting
        };
        if lease.token != handle.token || stream != self.stream.cu_stream() as u64 || !available {
            return Err(publication_input_error(
                "cold model scratch requires its original reader and stream before recording",
            ));
        }
        storage.plan()?;
        // SAFETY: reset initialized all three words of every original slot.
        let view = unsafe { storage.work.actual.view().cast::<u8>() }
            .ok_or(SemanticTransitionError::ObservationMismatch)?;
        let capacity = i64::try_from(storage.work.actual.len() / 3)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        // This is the Session's own stream, already joined by cold completion,
        // not an additional external consumer omitted from the frozen roster.
        let scratch = Arc::clone(&storage.aliases);
        let guard = Arc::clone(&self.checked_step(lease)?.aliases);
        self.export_owned_view(
            view,
            vec![capacity, 3],
            vec![3, 1],
            (1, 64),
            guard,
            stream,
            Some(scratch),
        )
    }

    /// Freeze the finite region roster before the original report begins.
    pub fn prepare_cold_model_work_regions(
        &mut self,
        handle: &SemanticColdModelWork,
        count: usize,
    ) -> Result<Vec<SemanticColdModelWorkRegion>, SemanticTransitionError> {
        let storage = self.cold_model_work(handle)?;
        if count == 0
            || storage.state != RecordingState::Waiting
            || !storage.regions.is_empty()
            || storage.plan.is_some()
        {
            return Err(publication_input_error(
                "cold registration regions require their sole original waiting report",
            ));
        }
        self.steps
            .get_mut(&handle.token)
            .expect("checked original reader")
            .cold_model_work
            .as_mut()
            .expect("checked original work")
            .regions = vec![RecordingState::Waiting; count];
        Ok((0..count)
            .map(|index| SemanticColdModelWorkRegion {
                work: handle.clone(),
                index,
            })
            .collect())
    }

    fn require_cold_model_work_region(
        &self,
        region: &SemanticColdModelWorkRegion,
        expected: RecordingState,
    ) -> Result<(), SemanticTransitionError> {
        let storage = self.cold_model_work(&region.work)?;
        if storage.state != RecordingState::Recording
            || storage.regions.get(region.index) != Some(&expected)
            || storage.regions[..region.index]
                .iter()
                .any(|state| !matches!(state, RecordingState::Closed | RecordingState::NotEntered))
        {
            return Err(publication_input_error(
                "cold registration region changed its original report, order or one-shot state",
            ));
        }
        Ok(())
    }

    pub fn require_closed_cold_model_work_region(
        &self,
        region: &SemanticColdModelWorkRegion,
    ) -> Result<(), SemanticTransitionError> {
        self.require_cold_model_work_region(region, RecordingState::Closed)
    }

    pub fn begin_cold_model_work_region(
        &mut self,
        region: &SemanticColdModelWorkRegion,
    ) -> Result<(), SemanticTransitionError> {
        self.require_cold_model_work_region(region, RecordingState::Waiting)?;
        self.cold_model_work(&region.work)?
            .plan()?
            .require_region_start(region.index)
            .map_err(publication_input_error)?;
        self.steps
            .get_mut(&region.work.token)
            .expect("checked original reader")
            .cold_model_work
            .as_mut()
            .expect("checked original work")
            .regions[region.index] = RecordingState::Recording;
        Ok(())
    }

    pub fn close_cold_model_work_region(
        &mut self,
        region: &SemanticColdModelWorkRegion,
    ) -> Result<(), SemanticTransitionError> {
        self.require_cold_model_work_region(region, RecordingState::Recording)?;
        self.cold_model_work(&region.work)?
            .plan()?
            .require_region_end(region.index)
            .map_err(publication_input_error)?;
        self.steps
            .get_mut(&region.work.token)
            .expect("checked original reader")
            .cold_model_work
            .as_mut()
            .expect("checked original work")
            .regions[region.index] = RecordingState::Closed;
        Ok(())
    }

    pub fn begin_cold_model_work(
        &mut self,
        handle: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        let storage = self.cold_model_work(handle)?;
        if storage.state != RecordingState::Waiting {
            return Err(publication_input_error(
                "original cold recorder begins exactly once",
            ));
        }
        if storage.regions.is_empty() {
            storage.plan()?;
        }
        let storage = self
            .steps
            .get_mut(&handle.token)
            .expect("checked reader")
            .cold_model_work
            .as_mut()
            .expect("checked cold work");
        storage.state = RecordingState::Recording;
        storage.work.uncaptured_recording = true;
        Ok(())
    }

    /// The phase calls this only after known numerical completion, native
    /// cancellation or complete private retirement has joined consumers. The
    /// original report and physical interval remain open; entered, failed or
    /// unknown regions stay unchanged, including at the report's full extent.
    pub fn cancel_unentered_cold_model_work_regions(
        &mut self,
        handle: &SemanticColdModelWork,
        before: usize,
    ) -> Result<(), SemanticTransitionError> {
        let storage = self.cold_model_work(handle)?;
        if storage.state != RecordingState::Recording
            || before > storage.regions.len()
            || storage.regions[..before].iter().any(|state| {
                !matches!(
                    state,
                    RecordingState::Waiting | RecordingState::Closed | RecordingState::NotEntered
                )
            })
        {
            return Err(publication_input_error(
                "known cancellation cannot close an entered, failed or unknown cold region",
            ));
        }
        let storage = self
            .steps
            .get_mut(&handle.token)
            .expect("checked original reader")
            .cold_model_work
            .as_mut()
            .expect("checked original work");
        for (index, state) in storage.regions[..before].iter_mut().enumerate() {
            if *state == RecordingState::Waiting {
                if let Some(plan) = &mut storage.plan {
                    plan.skip_unentered_region(index)
                        .map_err(publication_input_error)?;
                }
                *state = RecordingState::NotEntered;
            }
        }
        Ok(())
    }

    pub fn close_cold_model_work(
        &mut self,
        handle: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        let storage = self.cold_model_work(handle)?;
        if storage.state != RecordingState::Recording
            || storage
                .regions
                .iter()
                .any(|state| !matches!(state, RecordingState::Closed | RecordingState::NotEntered))
        {
            return Err(publication_input_error(
                "cold recorder closes its sole original recording",
            ));
        }
        let storage = self
            .steps
            .get_mut(&handle.token)
            .expect("checked reader")
            .cold_model_work
            .as_mut()
            .expect("checked cold work");
        storage.work.uncaptured_recording = false;
        storage.state = RecordingState::Closed;
        Ok(())
    }

    /// Failure never clears allocations, resets capacity or opens another recorder.
    pub fn fail_cold_model_work(&mut self, handle: &SemanticColdModelWork) {
        if let Some(storage) = self
            .steps
            .get_mut(&handle.token)
            .and_then(|step| step.cold_model_work.as_mut())
            .filter(|storage| Arc::ptr_eq(&storage.invocation, &handle.invocation))
        {
            storage.state = RecordingState::Failed;
            storage.work.uncaptured_recording = false;
        }
    }

    pub fn record_cold_model_work(
        &mut self,
        handle: &SemanticColdModelWork,
        kind: ModelWorkKind,
        dimensions: &[u64],
        device_produced: bool,
    ) -> Result<usize, SemanticTransitionError> {
        self.record_cold_model_work_in_region(handle, None, kind, dimensions, device_produced)
    }

    pub fn record_cold_model_work_region(
        &mut self,
        region: &SemanticColdModelWorkRegion,
        kind: ModelWorkKind,
        dimensions: &[u64],
        device_produced: bool,
    ) -> Result<usize, SemanticTransitionError> {
        self.record_cold_model_work_in_region(
            &region.work,
            Some(region),
            kind,
            dimensions,
            device_produced,
        )
    }

    fn record_cold_model_work_in_region(
        &mut self,
        handle: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
        kind: ModelWorkKind,
        dimensions: &[u64],
        device_produced: bool,
    ) -> Result<usize, SemanticTransitionError> {
        let result = (|| {
            if let Some(region) = region {
                self.require_cold_model_work_region(region, RecordingState::Recording)?;
            } else if !self.cold_model_work(handle)?.regions.is_empty() {
                return Err(publication_input_error(
                    "segmented cold registration requires its actual original region",
                ));
            }
            if self.cold_model_work(handle)?.state != RecordingState::Recording {
                return Err(publication_input_error(
                    "cold model producer requires its active original recorder",
                ));
            }
            self.cold_model_work(handle)?
                .plan()?
                .require_next(kind, dimensions, region.map(|region| region.index))
                .map_err(publication_input_error)?;
            let storage = self
                .steps
                .get_mut(&handle.token)
                .expect("checked reader")
                .cold_model_work
                .as_mut()
                .expect("checked cold work");
            let slot = storage.work.record_operation(
                kind,
                dimensions,
                device_produced,
                &self.domain,
                &mut self.poisoned,
            )?;
            storage.plan_mut()?.consume();
            Ok(slot)
        })();
        if result.is_err() {
            self.fail_cold_model_work(handle);
        }
        result
    }

    pub fn record_cold_model_invocation(
        &mut self,
        handle: &SemanticColdModelWork,
    ) -> Result<(), SemanticTransitionError> {
        self.record_cold_model_invocation_in_region(handle, None)
    }

    pub fn record_cold_model_region_invocation(
        &mut self,
        region: &SemanticColdModelWorkRegion,
    ) -> Result<(), SemanticTransitionError> {
        self.record_cold_model_invocation_in_region(&region.work, Some(region))
    }

    fn record_cold_model_invocation_in_region(
        &mut self,
        handle: &SemanticColdModelWork,
        region: Option<&SemanticColdModelWorkRegion>,
    ) -> Result<(), SemanticTransitionError> {
        let result = (|| {
            if let Some(region) = region {
                self.require_cold_model_work_region(region, RecordingState::Recording)?;
            } else if !self.cold_model_work(handle)?.regions.is_empty() {
                return Err(publication_input_error(
                    "segmented cold invocation requires its actual original region",
                ));
            }
            if self.cold_model_work(handle)?.state != RecordingState::Recording {
                return Err(publication_input_error(
                    "cold model call requires its active original recorder",
                ));
            }
            self.cold_model_work(handle)?
                .plan()?
                .require_next(
                    ModelWorkKind::ModelInvocation,
                    &[],
                    region.map(|region| region.index),
                )
                .map_err(publication_input_error)?;
            let storage = self
                .steps
                .get_mut(&handle.token)
                .expect("checked reader")
                .cold_model_work
                .as_mut()
                .expect("checked cold work");
            storage
                .work
                .record_invocation(&self.domain, &mut self.poisoned)?;
            storage.plan_mut()?.consume();
            Ok(())
        })();
        if result.is_err() {
            self.fail_cold_model_work(handle);
        }
        result
    }

    /// Native completion, after the callback and all original consumer joins.
    /// Model and reached semantic commands are returned together. Other native S
    /// and physical M remain the original operation's mandatory producers.
    pub fn finish_cold_model_work(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticColdModelWork,
        streams: &[u64],
        disposition: SemanticColdModelWorkDisposition,
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        self.checked_reader(lease)?;
        let storage = self.cold_model_work(handle)?;
        if let Some(plan) = &storage.plan {
            plan.require_recorded_trace(storage.work.recording.events())
                .map_err(publication_input_error)?;
            if disposition == SemanticColdModelWorkDisposition::Complete && !plan.complete() {
                return Err(publication_input_error(
                    "successful cold completion requires every non-skipped original plan span",
                ));
            }
        } else if disposition != SemanticColdModelWorkDisposition::KnownRefusal
            || !storage.work.recording.events().is_empty()
            || storage
                .regions
                .iter()
                .any(|state| !matches!(state, RecordingState::Waiting | RecordingState::NotEntered))
        {
            return Err(publication_input_error(
                "cold completion without an admitted plan requires known refusal before every model callback",
            ));
        }
        if matches!(
            storage.state,
            RecordingState::Submitted | RecordingState::Completed
        ) {
            if storage.streams.as_deref() != Some(streams)
                || storage.disposition != Some(disposition)
            {
                return Err(publication_input_error(
                    "cold completion changed its original submitted consumer roster or disposition",
                ));
            }
            return self.resolve_cold_model_work(lease, handle);
        }
        if lease.token != handle.token
            || self.cold_model_work(handle)?.state != RecordingState::Closed
            || Arc::strong_count(&self.cold_model_work(handle)?.aliases) != 1
        {
            return Err(publication_input_error(
                "cold completion requires its original closed recorder without scratch or child aliases",
            ));
        }
        self.quiesce_published_reader(lease, streams)?;
        let kernel = self
            .provider
            .device()
            .inner()
            .get_func(
                "xlog_semantic_transition",
                "semantic_cold_model_work_result",
            )
            .ok_or_else(|| runtime_error("kernel lookup", "cold model work result unavailable"))?;
        let storage = self
            .steps
            .get_mut(&handle.token)
            .expect("checked reader")
            .cold_model_work
            .as_mut()
            .expect("checked cold work");
        storage
            .work
            .recording
            .freeze_cold()
            .map_err(publication_input_error)?;
        storage.state = RecordingState::Submitting;
        storage.streams = Some(streams.to_vec());
        storage.disposition = Some(disposition);
        let work = &storage.work;
        let input = if work.recording.events().is_empty() {
            ModelWorkInput::default()
        } else {
            let mut destination = work.device.view().slice(..work.recording.events().len());
            self.provider
                .htod_launch_metadata_sync_copy_into(work.recording.events(), &mut destination)
                .map_err(|error| runtime_error("cold model work metadata upload", error))?;
            work.descriptor()
        };
        let mut recorder = self.domain.new_strict_recorder();
        work.record_reads(&mut recorder);
        recorder.read(&storage.native_work);
        recorder.write(&storage.report);
        let arguments = (
            input.events,
            input.count,
            input.bound,
            storage.native_work.device_ptr_value(),
            storage.report.device_ptr_value(),
        );
        enqueue_recorded(&self.domain, &mut self.poisoned, recorder, |enqueue| {
            // SAFETY: the retained private inputs and fifteen-word report belong to
            // this one operation; no model alias can modify event metadata.
            unsafe {
                kernel.clone().launch_in(
                    enqueue,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    arguments,
                )
            }
            .map_err(|error| XlogError::Kernel(error.to_string()))
        })?;
        self.steps
            .get_mut(&handle.token)
            .expect("submitted reader")
            .cold_model_work
            .as_mut()
            .expect("submitted cold work")
            .state = RecordingState::Submitted;
        self.resolve_cold_model_work(lease, handle)
    }

    /// Read only the original already submitted report; never replay a callback,
    /// work marker, reset, metadata upload or result kernel on continuation.
    pub fn resolve_cold_model_work(
        &mut self,
        lease: &SemanticPublishedLease,
        handle: &SemanticColdModelWork,
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        self.checked_reader(lease)?;
        let storage = self.cold_model_work(handle)?;
        if lease.token != handle.token
            || !matches!(
                storage.state,
                RecordingState::Submitted | RecordingState::Completed
            )
        {
            return Err(publication_input_error(
                "cold resolution requires its original submitted report",
            ));
        }
        if let Some(result) = storage.result {
            return Ok(result);
        }
        let streams = storage
            .streams
            .as_ref()
            .expect("submitted original streams")
            .clone();
        self.complete_step_consumers(lease, &streams)?;
        let words = self.publication_read(self.cold_model_work(handle)?.report.view())?;
        let storage = self
            .steps
            .get_mut(&handle.token)
            .expect("completed reader")
            .cold_model_work
            .as_mut()
            .expect("completed cold work");
        if words.len() != 15
            || words[0] != 0
            || words[2] != storage.work.recording.events().len() as u64
            || Some(words[3]) != storage.work.recording.frozen_bound()
            || words[1] > words[3]
            || words[4]
                != storage
                    .work
                    .recording
                    .events()
                    .iter()
                    .filter(|event| event.kind == ModelWorkKind::ModelInvocation as u64)
                    .count() as u64
            || words[6..15]
                .iter()
                .try_fold(0u64, |sum, units| sum.checked_add(*units))
                != Some(words[5])
            || words[1].checked_add(words[5]).is_none()
        {
            storage.state = RecordingState::Failed;
            return Err(publication_input_error(
                "cold model work report is incomplete; retain its original operation",
            ));
        }
        let result = SemanticColdModelWorkResult {
            model_work: words[1],
            operation_count: words[2],
            work_bound: words[3],
            model_calls: words[4],
            native_work: words[5],
            native_events: words[6..15]
                .try_into()
                .expect("checked original native event extent"),
        };
        self.graph
            .complete_cold_work(&storage.native_work.view())
            .map_err(SemanticTransitionError::Semantic)?;
        storage.result = Some(result);
        storage.state = RecordingState::Completed;
        Ok(result)
    }
}
