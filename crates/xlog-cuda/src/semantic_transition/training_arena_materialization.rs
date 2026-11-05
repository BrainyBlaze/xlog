use super::native_work_report::{
    add_native_work_ceiling, ceiling_product, ceiling_sum, publication_hash_ceiling,
    selected_model_origin_native_work_ceiling, OriginalNativeWorkAccounting,
};
use super::*;

pub(super) struct OriginalTrainingMaterializationWork {
    pub(super) accounting: Arc<Mutex<OriginalNativeWorkAccounting>>,
    native: Option<TrackedCudaSlice<u64>>,
    report: Option<TrackedCudaSlice<u64>>,
    source: Option<TrackedCudaSlice<u8>>,
    input: Option<
        TrackedCudaSlice<crate::semantic_training_view::TrainingOriginalMaterializationInput>,
    >,
    writes: Vec<state_restoration::OriginalDeviceWrite>,
    recorded: [bool; 4],
    cursor: usize,
    pub(super) command: Option<OriginalNativeCommand>,
    report_command: Option<OriginalNativeCommand>,
    pub(super) control: Option<PublicationRead<PublicationControl>>,
    pub(super) header: Option<PublicationRead<PublicationHeader>>,
    report_read: Option<PublicationRead<u64>>,
    pub(super) result: Option<SemanticColdModelWorkResult>,
    poisoned: bool,
}

impl OriginalTrainingMaterializationWork {
    pub(super) fn new(
        storage: &PublicationStorage,
        directory: &[PublicationRange],
        packed_bytes: usize,
        cohort_preparation: [[u64; 9]; 2],
    ) -> Result<Self, SemanticTransitionError> {
        let n = u64::try_from(directory.len())
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        let t = storage.layouts.len() as u64;
        let rows = ceiling_sum([
            storage.contract_value.window_capacity,
            storage.contract_value.feedback_capacity,
        ])?;
        let mut native = selected_model_origin_native_work_ceiling(storage, directory)?;
        let visits = ceiling_sum([
            ceiling_product(
                n,
                ceiling_sum([
                    ceiling_product(4, n)?,
                    ceiling_product(2, t)?,
                    ceiling_product(2, rows)?,
                    128,
                ])?,
            )?,
            64,
        ])?;
        let mut copied = 0u64;
        for range in directory.iter().filter(|range| {
            publication_mutable_role(range.role) && !matches!(range.role, 1 | 2 | 18..=25 | 31 | 56)
        }) {
            let allocation = storage
                .allocations
                .get(
                    usize::try_from(range.storage_slot)
                        .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
                )
                .ok_or(SemanticTransitionError::ObservationMismatch)?
                .entry();
            copied = ceiling_sum([
                copied,
                allocation
                    .bytes
                    .checked_sub(range.offset_bytes)
                    .ok_or(SemanticTransitionError::ObservationMismatch)?,
            ])?;
        }
        add_native_work_ceiling(
            &mut native,
            [
                3,
                visits,
                0,
                0,
                0,
                0,
                0,
                0,
                ceiling_sum([
                    packed_bytes as u64,
                    copied,
                    size_of::<PublicationBank>() as u64,
                    ceiling_product(n, 80)?,
                    1024,
                    388,
                ])?,
            ],
        )?;
        add_native_work_ceiling(&mut native, publication_hash_ceiling(212)?)?;
        let mut dma = [
            14,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            ceiling_sum([
                packed_bytes as u64,
                11 * 8,
                2 * 15 * 8,
                size_of::<crate::semantic_training_view::TrainingOriginalMaterializationInput>()
                    as u64,
                4 * size_of::<PublicationLease>() as u64,
                size_of::<PublicationControl>() as u64,
                2 * size_of::<PublicationHeader>() as u64,
                ceiling_product(n, size_of::<PublicationRange>() as u64)?,
                32 * size_of::<SourceSlot>() as u64,
            ])?,
        ];
        add_native_work_ceiling(&mut native, cohort_preparation[0])?;
        add_native_work_ceiling(&mut dma, cohort_preparation[1])?;
        native_work_report::native_work_ceiling_units(native)?;
        native_work_report::native_work_ceiling_units(dma)?;
        Ok(Self {
            accounting: Arc::new(Mutex::new(OriginalNativeWorkAccounting::new(native, dma))),
            native: None,
            report: None,
            source: None,
            input: None,
            writes: Vec::new(),
            recorded: [false; 4],
            cursor: 0,
            command: None,
            report_command: None,
            control: None,
            header: None,
            report_read: None,
            result: None,
            poisoned: false,
        })
    }

    pub(super) fn prepare(
        &mut self,
        session: &SemanticTransitionSession,
        packed: &[u8],
        mut input: crate::semantic_training_view::TrainingOriginalMaterializationInput,
        reservation: Option<Arc<SemanticReplayMaterialReservation>>,
    ) -> Result<(), SemanticTransitionError> {
        if self.native.is_none() {
            self.native = Some(allocate_publication(&session.provider, 11)?);
            self.accounting
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?
                .work = Some(self.native.as_ref().expect("retained tally").view());
        }
        if self.report.is_none() {
            self.report = Some(allocate_publication(&session.provider, 15)?);
        }
        if self.source.is_none() {
            self.source = Some(allocate_publication(&session.provider, packed.len())?);
        }
        if self.input.is_none() {
            self.input = Some(allocate_publication(&session.provider, 1)?);
        }
        input.source = self
            .source
            .as_ref()
            .expect("retained original bytes")
            .device_ptr_value();
        while self.writes.len() < 4 {
            let write = match self.writes.len() {
                0 => state_restoration::OriginalDeviceWrite::new_with_owner(
                    &session.stream,
                    &[0u64; 11],
                    self.native.as_ref().expect("retained tally").view(),
                    false,
                    reservation.clone(),
                )?,
                1 => state_restoration::OriginalDeviceWrite::new_with_owner(
                    &session.stream,
                    packed,
                    self.source
                        .as_ref()
                        .expect("retained original bytes")
                        .view(),
                    false,
                    reservation.clone(),
                )?,
                2 => state_restoration::OriginalDeviceWrite::new_with_owner(
                    &session.stream,
                    &[input],
                    self.input.as_ref().expect("retained original input").view(),
                    false,
                    reservation.clone(),
                )?,
                3 => state_restoration::OriginalDeviceWrite::new_with_owner(
                    &session.stream,
                    &[u64::MAX; 15],
                    self.report.as_ref().expect("retained report").view(),
                    false,
                    reservation.clone(),
                )?,
                _ => unreachable!(),
            };
            self.writes.push(write);
        }
        if self.command.is_none() {
            self.command = Some(OriginalNativeCommand::new(&session.domain)?);
        }
        if self.report_command.is_none() {
            self.report_command = Some(OriginalNativeCommand::new(&session.domain)?);
        }
        if self.control.is_none() {
            self.control = Some(
                session.stage_publication_read_with_owner(
                    session
                        .publication
                        .as_ref()
                        .ok_or(SemanticTransitionError::NotBound)?
                        .control
                        .view(),
                    reservation.clone(),
                )?,
            );
        }
        if self.report_read.is_none() {
            self.report_read = Some(session.stage_publication_read_with_owner(
                self.report.as_ref().expect("retained report").view(),
                reservation,
            )?);
        }
        Ok(())
    }

    pub(super) fn upload(
        &mut self,
        session: &SemanticTransitionSession,
    ) -> Result<(), SemanticTransitionError> {
        while self.cursor < self.writes.len() {
            let write = &mut self.writes[self.cursor];
            let resolved = write.resolve(
                &session.domain,
                &session.stream,
                &session.provider,
                &mut self.poisoned,
                true,
            );
            if write.submitted() {
                self.accounting
                    .lock()
                    .map_err(|_| SemanticTransitionError::Poisoned)?
                    .record_dma(write.dma_bytes(), &mut self.recorded[self.cursor])?;
            }
            resolved?;
            if self.cursor == 0 {
                self.accounting
                    .lock()
                    .map_err(|_| SemanticTransitionError::Poisoned)?
                    .initialized = true;
            }
            self.cursor += 1;
        }
        Ok(())
    }

    pub(super) fn known_completion_failure(&self) -> Option<sys::CUresult> {
        self.writes
            .iter()
            .find_map(|write| write.known_completion_failure())
            .or_else(|| {
                [&self.command, &self.report_command]
                    .into_iter()
                    .filter_map(Option::as_ref)
                    .find_map(|command| {
                        command
                            .completion_failure
                            .or_else(|| command.completion.known_completion_failure())
                            .or(command.submission_failure.filter(|status| {
                                crate::device::known_execution_failure(
                                    &cudarc::driver::DriverError(*status),
                                )
                                .is_some()
                            }))
                    })
            })
            .or_else(|| {
                self.control
                    .as_ref()
                    .and_then(PublicationRead::known_completion_failure)
            })
            .or_else(|| {
                self.header
                    .as_ref()
                    .and_then(PublicationRead::known_completion_failure)
            })
            .or_else(|| {
                self.report_read
                    .as_ref()
                    .and_then(PublicationRead::known_completion_failure)
            })
    }

    pub(super) fn input(
        &self,
    ) -> &TrackedCudaSlice<crate::semantic_training_view::TrainingOriginalMaterializationInput>
    {
        self.input.as_ref().expect("prepared materialization input")
    }

    pub(super) fn source(&self) -> &TrackedCudaSlice<u8> {
        self.source
            .as_ref()
            .expect("prepared materialization source")
    }

    pub(super) fn native(&self) -> &TrackedCudaSlice<u64> {
        self.native
            .as_ref()
            .expect("prepared materialization tally")
    }

    pub(super) fn finish_report(
        &mut self,
        session: &mut SemanticTransitionSession,
    ) -> Result<SemanticColdModelWorkResult, SemanticTransitionError> {
        if let Some(result) = self.result {
            return Ok(result);
        }
        let native = self.native().view();
        let report = self.report.as_ref().expect("retained report").view();
        let command = self
            .report_command
            .as_mut()
            .expect("original report command");
        let dispatched = native_work_report::run_native_work_report(
            &session.provider,
            &session.domain,
            command,
            &mut self.poisoned,
            ModelWorkInput::default(),
            &native,
            &report,
            session.domain.new_strict_recorder(),
        );
        let unresolved_submission = if let Err(error) = dispatched {
            if !matches!(
                error,
                SemanticTransitionError::NativeExecutionFailure {
                    status: sys::CUresult::CUDA_ERROR_UNKNOWN | sys::CUresult::CUDA_ERROR_NOT_READY,
                    ..
                }
            ) {
                return Err(error);
            }
            session
                .stream
                .context()
                .bind_to_thread()
                .and_then(|_| session.stream.context().synchronize())
                .map_err(|error| native_runtime_error("original arena report completion", error))?;
            Some(error)
        } else {
            None
        };
        let words = session.resolve_publication_read_with_poison(
            self.report_read.as_mut().expect("original report read"),
            &mut self.poisoned,
        )?;
        // An untouched report does not resolve the original uncertain submission.
        if words.iter().all(|word| *word == u64::MAX) {
            if let Some(error) = unresolved_submission {
                return Err(error);
            }
        }
        let result = SemanticColdModelWorkResult::from_native_report(&words)?;
        let result = self
            .accounting
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .complete(result)?;
        self.result = Some(result);
        Ok(result)
    }
}

pub(super) fn record_materialization_dma(
    accounting: Option<&Arc<Mutex<OriginalNativeWorkAccounting>>>,
    bytes: usize,
    recorded: &mut bool,
) -> Result<(), SemanticTransitionError> {
    if let Some(accounting) = accounting {
        accounting
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .record_dma(bytes, recorded)?;
    }
    Ok(())
}

pub(super) fn metadata_allocation_bytes() -> Result<usize, SemanticTransitionError> {
    let events = usize::try_from(
        crate::device::ExecutionCompletion::single_stream_allocation_bytes().map_err(|error| {
            runtime_error("training materialization completion metadata", error)
        })?,
    )
    .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
    let fixed = size_of::<PendingTrainingMaterialization>()
        + size_of::<OriginalNativeWorkAccounting>()
        + 2 * size_of::<usize>()
        + 3 * size_of::<PublicationRead<u64>>()
        + state_restoration::OriginalDeviceWrite::metadata_allocation_bytes::<u8>()
        + 2 * state_restoration::OriginalDeviceWrite::metadata_allocation_bytes::<u64>()
        + state_restoration::OriginalDeviceWrite::metadata_allocation_bytes::<
            crate::semantic_training_view::TrainingOriginalMaterializationInput,
        >()
        + 2 * events;
    // Original pinned metadata and both cached and returned cold read values.
    fixed
        .checked_add(
            11 * 8
                + 15 * 8
                + size_of::<crate::semantic_training_view::TrainingOriginalMaterializationInput>()
                + 3 * (15 * 8 + size_of::<PublicationControl>() + size_of::<PublicationHeader>()),
        )
        .ok_or(SemanticTransitionError::GenerationExhausted)
}
