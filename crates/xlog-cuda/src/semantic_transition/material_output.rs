//! One fixed material allocation with disjoint initialized-range publication.

use super::*;
use std::cell::UnsafeCell;

// The canonical material codec declares two staging buffers with this bound
// before its original numerical callback. Native readback borrows one buffer
// extent, with the second covering the retained known result; it cannot grow it.
const MODEL_CODEC_STAGING_LIMIT: usize = 8 * 1024 * 1024;

struct MaterialOutputSpan {
    begin: usize,
    end: usize,
    cursor: usize,
}

struct MaterialOutputDeclaration {
    extent: usize,
    spans: Vec<MaterialOutputSpan>,
    suffix_cursor: usize,
    _metadata: SemanticReplayMaterialReservation,
}

struct MaterialOutputLedger {
    model_begin: usize,
    model_capacity: usize,
    suffix_extent: usize,
    declared_suffix_extent: Option<usize>,
    framed: bool,
    prefix_begin: Option<usize>,
    prefix_cursor: usize,
    prefix_published: bool,
    declaration: Option<MaterialOutputDeclaration>,
    model_published: bool,
    full_published: bool,
}

struct MaterialOutputComparison {
    original: SemanticReplayLoadedMaterial,
    budget: SemanticReplayMaterialBudget,
    ledger: Mutex<MaterialOutputLedger>,
    _metadata: SemanticReplayMaterialReservation,
}

pub(super) struct MaterialOutputAllocation {
    bytes: UnsafeCell<Vec<u8>>,
    data: *mut u8,
    extent: usize,
    ledger: Mutex<MaterialOutputLedger>,
}

// SAFETY: the Vec's allocation, length and capacity never change while this
// owner is shared. The sole native ledger serializes writes and rejects every
// write to a published range. Immutable references cover only an initialized,
// published exact range, never the still-writable remainder of the allocation.
unsafe impl Send for MaterialOutputAllocation {}
unsafe impl Sync for MaterialOutputAllocation {}

impl MaterialOutputAllocation {
    pub(super) fn len(&self) -> usize {
        self.extent
    }

    pub(super) fn published_range(&self, range: std::ops::Range<usize>) -> &[u8] {
        // SAFETY: only seal_prefix/seal_model/seal_frame_body/seal_frame mint this owner's ranges;
        // slice_payload and the private byte slices can only narrow them. No
        // reference to the whole mutable backing is materialized here.
        unsafe { std::slice::from_raw_parts(self.data.add(range.start), range.len()) }
    }

    pub(super) fn into_owned(self) -> Result<Vec<u8>, SemanticTransitionError> {
        let ledger = self
            .ledger
            .into_inner()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if !ledger.full_published || ledger.framed {
            return Err(publication_input_error(
                "material output has not sealed its original allocation",
            ));
        }
        Ok(self.bytes.into_inner())
    }
}

/// The original bounded writer. Its loaded ranges share the same canonical
/// byte owner and reservation; there is no independent material serializer.
#[derive(Clone)]
pub struct SemanticReplayMaterialOutput {
    owner: Arc<PublicationMaterialBytesOwner>,
    comparison: Option<Arc<MaterialOutputComparison>>,
}

/// The original codec workspace. Readback borrows this same reservation only
/// after the codec has sealed its model; it never obtains another pool credit.
#[derive(Clone)]
pub struct SemanticReplayMaterialScratch {
    owner: Arc<MaterialScratchOwner>,
}

struct MaterialScratchOwner {
    budget: SemanticReplayMaterialBudget,
    working_bytes: u64,
    reservation: Arc<SemanticReplayMaterialReservation>,
    readback: Mutex<Option<MaterialReadbackBinding>>,
    host_commitment: material_host_memory::MaterialHostCommitment,
    _metadata: SemanticReplayMaterialReservation,
}

struct MaterialReadbackBinding {
    issuer: Arc<()>,
    token: u64,
    output: std::sync::Weak<PublicationMaterialBytesOwner>,
}

/// Fixed original full-backing geometry and one exclusive codec staging loan.
/// Its commands and bytes are derived before admission, not chosen from a
/// later pool balance. A pending observation retains this same owner.
#[derive(Clone)]
pub struct SemanticPublicationReadbackLoan {
    owner: Arc<PublicationReadbackLoanOwner>,
}

struct PublicationReadbackLoanOwner {
    scratch: SemanticReplayMaterialScratch,
    backing_spans: Vec<usize>,
    staging_bytes: usize,
    dma_ceiling: [u64; 9],
    native_ceiling: [u64; 9],
    #[cfg(feature = "semantic-policy")]
    geometry: Arc<publication_readback::CheckpointReadbackGeometry>,
    #[cfg(feature = "semantic-policy")]
    origin: publication_readback::CheckpointReadbackOrigin,
    _metadata: SemanticReplayMaterialReservation,
}

impl SemanticReplayMaterialScratch {
    pub fn enter_codec_workspace(&self) -> Result<(), SemanticTransitionError> {
        self.require_codec_available()?;
        if !self.owner.host_commitment.is_entered() {
            self.owner.budget.require_host_allocation(0)?;
        }
        self.owner.host_commitment.enter()
    }
    pub(super) fn require_budget(
        &self,
        budget: &SemanticReplayMaterialBudget,
    ) -> Result<(), SemanticTransitionError> {
        if !self.owner.budget.same_pool(budget) {
            return Err(publication_input_error(
                "codec workspace changed its original pool",
            ));
        }
        Ok(())
    }
    pub fn require_codec_available(&self) -> Result<(), SemanticTransitionError> {
        if self
            .owner
            .readback
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .is_some()
        {
            return Err(publication_input_error(
                "original codec workspace is lent to its native readback",
            ));
        }
        Ok(())
    }

    pub(super) fn prepare_readback_loan(
        &self,
        backing_spans: &[u64],
        codec_staging_bytes: usize,
        #[cfg(feature = "semantic-policy")] geometry: Arc<
            publication_readback::CheckpointReadbackGeometry,
        >,
        #[cfg(feature = "semantic-policy")] origin: publication_readback::CheckpointReadbackOrigin,
    ) -> Result<SemanticPublicationReadbackLoan, SemanticTransitionError> {
        self.require_codec_available()?;
        let staging = u64::try_from(codec_staging_bytes)
            .map_err(|_| SemanticTransitionError::GenerationExhausted)?;
        if staging == 0
            || codec_staging_bytes > MODEL_CODEC_STAGING_LIMIT
            || staging
                .checked_mul(2)
                .is_none_or(|bytes| bytes > self.owner.working_bytes)
        {
            return Err(publication_input_error(
                "native readback staging exceeds the original two-buffer codec workspace",
            ));
        }
        let metadata_extent = size_of::<PublicationReadbackLoanOwner>()
            .checked_add(2 * size_of::<usize>())
            .and_then(|extent| {
                extent.checked_add(backing_spans.len().checked_mul(size_of::<usize>())?)
            })
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let metadata = self.owner.budget.reserve_auxiliary(
            u64::try_from(metadata_extent)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )?;
        let mut spans = Vec::new();
        spans
            .try_reserve_exact(backing_spans.len())
            .map_err(|_| publication_input_error("native readback geometry allocation failed"))?;
        let mut commands = 0u64;
        let mut bytes = 0u64;
        for &span in backing_spans {
            // Preserve the canonical publication codec's existing field width.
            u32::try_from(span).map_err(|_| SemanticTransitionError::GenerationExhausted)?;
            let count = span / staging + u64::from(span % staging != 0);
            commands = commands
                .checked_add(count)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            bytes = bytes
                .checked_add(span)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            spans.push(
                usize::try_from(span).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
            );
        }
        let model_dma = [commands, 0, 0, 0, 0, 0, 0, 0, bytes];
        #[cfg(feature = "semantic-policy")]
        let (dma_ceiling, native_ceiling) = {
            let mut dma_ceiling = model_dma;
            let mut native_ceiling = model_dma;
            let (metadata_dma, metadata_native) = geometry.ceilings(origin)?;
            native_work_bound::add_native_work_ceiling(&mut dma_ceiling, metadata_dma)?;
            native_work_bound::add_native_work_ceiling(&mut native_ceiling, metadata_native)?;
            (dma_ceiling, native_ceiling)
        };
        #[cfg(not(feature = "semantic-policy"))]
        let (dma_ceiling, native_ceiling) = (model_dma, model_dma);
        Ok(SemanticPublicationReadbackLoan {
            owner: Arc::new(PublicationReadbackLoanOwner {
                scratch: self.clone(),
                backing_spans: spans,
                staging_bytes: codec_staging_bytes,
                dma_ceiling,
                native_ceiling,
                #[cfg(feature = "semantic-policy")]
                geometry,
                #[cfg(feature = "semantic-policy")]
                origin,
                _metadata: metadata,
            }),
        })
    }
}

impl SemanticPublicationReadbackLoan {
    pub(super) fn same_loan(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner)
    }
    pub fn material_scratch(&self) -> SemanticReplayMaterialScratch {
        self.owner.scratch.clone()
    }

    pub(super) fn require_workspace(
        &self,
        working_bytes: u64,
        extent: usize,
    ) -> Result<(), SemanticTransitionError> {
        if self.owner.scratch.owner.working_bytes != working_bytes
            || self.owner.staging_bytes != extent
        {
            return Err(publication_input_error(
                "readback changed its original workspace bounds",
            ));
        }
        Ok(())
    }

    pub(super) fn require_scratch(
        &self,
        scratch: &SemanticReplayMaterialScratch,
        extent: usize,
    ) -> Result<(), SemanticTransitionError> {
        if !Arc::ptr_eq(&self.owner.scratch.owner, &scratch.owner)
            || self.owner.staging_bytes != extent
        {
            return Err(publication_input_error(
                "readback changed its original codec workspace or extent",
            ));
        }
        Ok(())
    }
    pub fn staging_bytes(&self) -> usize {
        self.owner.staging_bytes
    }

    pub fn native_work_ceiling(&self) -> [u64; 9] {
        self.owner.native_ceiling
    }

    pub(super) fn submitted_dma_ceiling(&self) -> [u64; 9] {
        self.owner.dma_ceiling
    }

    #[cfg(feature = "semantic-policy")]
    pub(super) fn require_metadata_geometry(
        &self,
        session: &SemanticTransitionSession,
        lease: &SemanticPublishedLease,
    ) -> Result<(), SemanticTransitionError> {
        self.owner
            .geometry
            .require_actual(self.owner.origin, session, lease)
    }

    pub(super) fn require_staging_host_allocation(
        &self,
        bytes: usize,
    ) -> Result<(), SemanticTransitionError> {
        if bytes
            > self
                .owner
                .staging_bytes
                .checked_mul(2)
                .ok_or(SemanticTransitionError::GenerationExhausted)?
        {
            return Err(publication_input_error(
                "native readback exceeded its original codec staging loan",
            ));
        }
        self.owner.scratch.owner.budget.require_host_allocation(
            u64::try_from(bytes).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )
    }

    pub(super) fn require_backing_spans(
        &self,
        spans: &[u64],
    ) -> Result<(), SemanticTransitionError> {
        if spans.len() != self.owner.backing_spans.len()
            || spans
                .iter()
                .zip(&self.owner.backing_spans)
                .any(|(&actual, &original)| usize::try_from(actual) != Ok(original))
        {
            return Err(publication_input_error(
                "native readback changed its original full backing roster",
            ));
        }
        Ok(())
    }

    pub(super) fn bind_observation(
        &self,
        issuer: &Arc<()>,
        token: u64,
        output: &SemanticReplayMaterialOutput,
    ) -> Result<(), SemanticTransitionError> {
        output.require_budget(&self.owner.scratch.owner.budget)?;
        // The original numerical codec must finish before its staging can be
        // lent to a native observation. Metadata issuance is not this proof.
        output.require_sealed_model()?;
        let mut binding = self
            .owner
            .scratch
            .owner
            .readback
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if let Some(original) = &*binding {
            if !Arc::ptr_eq(&original.issuer, issuer)
                || original.token != token
                || original
                    .output
                    .upgrade()
                    .is_none_or(|original| !Arc::ptr_eq(&original, &output.owner))
            {
                return Err(publication_input_error(
                    "codec staging loan changed its original native observation",
                ));
            }
        } else {
            *binding = Some(MaterialReadbackBinding {
                issuer: Arc::clone(issuer),
                token,
                output: Arc::downgrade(&output.owner),
            });
        }
        Ok(())
    }

    pub(super) fn retained_owner(&self) -> Arc<dyn Send + Sync> {
        Arc::clone(&self.owner.scratch.owner.reservation) as Arc<dyn Send + Sync>
    }
}

impl SemanticReplayMaterialBudget {
    pub fn prepare_material_scratch(
        &self,
        working_bytes: u64,
    ) -> Result<SemanticReplayMaterialScratch, SemanticTransitionError> {
        let reservation = self.reserve_auxiliary(working_bytes)?;
        let metadata_extent = size_of::<MaterialScratchOwner>()
            + 2 * size_of::<usize>()
            + size_of::<SemanticReplayMaterialReservation>()
            + 2 * size_of::<usize>();
        let metadata = self.reserve_auxiliary(
            u64::try_from(metadata_extent)
                .map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )?;
        let host_commitment =
            material_host_memory::MaterialHostCommitment::new(self, working_bytes)?;
        Ok(SemanticReplayMaterialScratch {
            owner: Arc::new(MaterialScratchOwner {
                budget: self.clone(),
                working_bytes,
                reservation: Arc::new(reservation),
                readback: Mutex::new(None),
                host_commitment,
                _metadata: metadata,
            }),
        })
    }
    /// The original writer protocol can verify an already retained material
    /// range without allocating a second payload. Every declared byte must
    /// still be produced and compared before the original range is returned.
    pub fn prepare_material_comparison(
        &self,
        original: &SemanticReplayLoadedMaterial,
    ) -> Result<SemanticReplayMaterialOutput, SemanticTransitionError> {
        original.require_budget(self)?;
        let extent = size_of::<MaterialOutputComparison>()
            .checked_add(2 * size_of::<usize>())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let metadata = self.reserve_auxiliary(
            u64::try_from(extent).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )?;
        Ok(SemanticReplayMaterialOutput {
            owner: Arc::clone(&original.bytes.owner),
            comparison: Some(Arc::new(MaterialOutputComparison {
                original: original.slice_payload(0..original.as_bytes().len())?,
                budget: self.clone(),
                ledger: Mutex::new(MaterialOutputLedger {
                    model_begin: 0,
                    model_capacity: original.as_bytes().len(),
                    suffix_extent: 0,
                    declared_suffix_extent: None,
                    framed: false,
                    prefix_begin: Some(0),
                    prefix_cursor: 0,
                    prefix_published: false,
                    declaration: None,
                    model_published: false,
                    full_published: false,
                }),
                _metadata: metadata,
            })),
        })
    }

    pub fn prepare_material_output(
        &self,
        capacity: usize,
    ) -> Result<SemanticReplayMaterialOutput, SemanticTransitionError> {
        self.prepare_fixed_material_output(0, capacity, 0, false)
    }

    pub fn prepare_framed_material_output(
        &self,
        model_begin: usize,
        model_capacity: usize,
        suffix_extent: usize,
    ) -> Result<SemanticReplayMaterialOutput, SemanticTransitionError> {
        self.prepare_fixed_material_output(model_begin, model_capacity, suffix_extent, true)
    }

    fn prepare_fixed_material_output(
        &self,
        model_begin: usize,
        model_capacity: usize,
        suffix_extent: usize,
        framed: bool,
    ) -> Result<SemanticReplayMaterialOutput, SemanticTransitionError> {
        let extent = model_begin
            .checked_add(model_capacity)
            .and_then(|size| size.checked_add(suffix_extent))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let reservation = self.reserve_auxiliary(
            u64::try_from(extent).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )?;
        let metadata = self.reserve_owned_payload_metadata()?;
        self.require_host_allocation(
            u64::try_from(extent).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(extent)
            .map_err(|_| publication_input_error("material output allocation failed"))?;
        bytes.resize(extent, 0);
        let data = bytes.as_mut_ptr();
        Ok(SemanticReplayMaterialOutput {
            comparison: None,
            owner: Arc::new(PublicationMaterialBytesOwner {
                bytes: PublicationMaterialByteAllocation::Produced(MaterialOutputAllocation {
                    bytes: UnsafeCell::new(bytes),
                    data,
                    extent,
                    ledger: Mutex::new(MaterialOutputLedger {
                        model_begin,
                        model_capacity,
                        suffix_extent,
                        declared_suffix_extent: None,
                        framed,
                        prefix_begin: (!framed).then_some(0),
                        prefix_cursor: 0,
                        prefix_published: false,
                        declaration: None,
                        model_published: false,
                        full_published: false,
                    }),
                }),
                reservation: Some(reservation),
                readback_reservation: None,
                decode_reservation: Mutex::new(None),
                _metadata_reservation: Some(metadata),
                _shared_metadata: None,
            }),
        })
    }
}

impl SemanticReplayMaterialOutput {
    fn require_sealed_model(&self) -> Result<(), SemanticTransitionError> {
        if !self
            .ledger()
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .model_published
        {
            return Err(publication_input_error(
                "native codec staging requires its original sealed model",
            ));
        }
        Ok(())
    }

    /// Native-only projection of a prefix range already initialized by the
    /// sole monotone writer. Later writes begin at prefix_cursor and cannot
    /// modify any byte in this exact borrowed range.
    pub(super) fn initialized_prefix_range(
        &self,
        range: std::ops::Range<usize>,
    ) -> Result<SemanticReplayLoadedMaterial, SemanticTransitionError> {
        self.require_produced_frame()?;
        let ledger = self
            .allocation()
            .ledger
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if !ledger.framed
            || ledger.prefix_begin.is_none_or(|begin| range.start < begin)
            || range.start > range.end
            || range.end > ledger.prefix_cursor
            || range.end > ledger.model_begin
        {
            return Err(publication_input_error(
                "native material range lacks its original initialized prefix",
            ));
        }
        Ok(self.loaded(range))
    }
    fn require_produced_frame(&self) -> Result<(), SemanticTransitionError> {
        if self.comparison.is_some() {
            return Err(publication_input_error(
                "material comparison cannot produce an independent frame",
            ));
        }
        Ok(())
    }

    fn ledger(&self) -> &Mutex<MaterialOutputLedger> {
        self.comparison
            .as_ref()
            .map_or_else(|| &self.allocation().ledger, |original| &original.ledger)
    }

    pub(super) fn same_output(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner)
            && match (&self.comparison, &other.comparison) {
                (None, None) => true,
                (Some(original), Some(current)) => Arc::ptr_eq(original, current),
                _ => false,
            }
    }
    /// Freeze the exact canonical prefix immediately before the fixed model
    /// range. Unused leading capacity is never part of a published payload.
    pub fn declare_frame_prefix(&self, extent: usize) -> Result<usize, SemanticTransitionError> {
        self.require_produced_frame()?;
        let mut ledger = self
            .allocation()
            .ledger
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if !ledger.framed || extent > ledger.model_begin {
            return Err(publication_input_error(
                "checkpoint prefix exceeds its original capacity",
            ));
        }
        let begin = ledger.model_begin - extent;
        if let Some(original) = ledger.prefix_begin {
            if original != begin {
                return Err(publication_input_error(
                    "checkpoint prefix changed its original extent",
                ));
            }
            return Ok(original);
        }
        if ledger.prefix_published || ledger.full_published {
            return Err(publication_input_error(
                "checkpoint prefix is already published",
            ));
        }
        ledger.prefix_begin = Some(begin);
        ledger.prefix_cursor = begin;
        Ok(begin)
    }

    fn allocation(&self) -> &MaterialOutputAllocation {
        match &self.owner.bytes {
            PublicationMaterialByteAllocation::Produced(original) => original,
            _ => unreachable!("only the original fixed producer issues a material output"),
        }
    }

    pub fn require_budget(
        &self,
        budget: &SemanticReplayMaterialBudget,
    ) -> Result<(), SemanticTransitionError> {
        if let Some(original) = &self.comparison {
            return original.original.require_budget(budget);
        }
        if self
            .owner
            .reservation
            .as_ref()
            .is_none_or(|original| !original.budget.same_pool(budget))
        {
            return Err(publication_input_error(
                "material output changed its original byte pool",
            ));
        }
        Ok(())
    }

    pub fn declare_spans(
        &self,
        extent: usize,
        spans: &[(usize, usize)],
    ) -> Result<(), SemanticTransitionError> {
        let mut ledger = self
            .ledger()
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if ledger.declaration.is_some() || ledger.model_published || extent > ledger.model_capacity
        {
            return Err(publication_input_error(
                "material output must declare its original spans once within capacity",
            ));
        }
        if self.comparison.is_some() && extent != ledger.model_capacity {
            return Err(publication_input_error(
                "material comparison must declare the entire original range",
            ));
        }
        let budget = self.comparison.as_ref().map_or_else(
            || {
                &self
                    .owner
                    .reservation
                    .as_ref()
                    .expect("original output reservation")
                    .budget
            },
            |original| &original.budget,
        );
        let size = spans
            .len()
            .checked_mul(size_of::<MaterialOutputSpan>())
            .and_then(|size| size.checked_add(size_of::<MaterialOutputDeclaration>()))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let metadata = budget.reserve_auxiliary(
            u64::try_from(size).map_err(|_| SemanticTransitionError::GenerationExhausted)?,
        )?;
        let mut roster = Vec::new();
        roster
            .try_reserve_exact(spans.len())
            .map_err(|_| publication_input_error("material output span allocation failed"))?;
        let mut cursor = 0;
        for &(begin, end) in spans {
            if begin != cursor || end <= begin || end > extent {
                return Err(publication_input_error(
                    "material output spans must partition the exact initialized extent",
                ));
            }
            roster.push(MaterialOutputSpan {
                begin,
                end,
                cursor: begin,
            });
            cursor = end;
        }
        if cursor != extent {
            return Err(publication_input_error(
                "material output spans do not cover their exact extent",
            ));
        }
        let suffix_cursor = ledger
            .model_begin
            .checked_add(extent)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        ledger.declaration = Some(MaterialOutputDeclaration {
            extent,
            spans: roster,
            suffix_cursor,
            _metadata: metadata,
        });
        Ok(())
    }

    pub fn write_model(&self, offset: usize, bytes: &[u8]) -> Result<(), SemanticTransitionError> {
        let mut ledger = self
            .ledger()
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if ledger.model_published || ledger.full_published {
            return Err(publication_input_error(
                "material output cannot overwrite a published model range",
            ));
        }
        let model_begin = ledger.model_begin;
        let declaration = ledger.declaration.as_mut().ok_or_else(|| {
            publication_input_error("material output requires its original span declaration")
        })?;
        let end = offset
            .checked_add(bytes.len())
            .filter(|&end| end <= declaration.extent)
            .ok_or_else(|| publication_input_error("material write exceeds its original extent"))?;
        if bytes.is_empty() {
            if offset == 0 && declaration.extent == 0
                || declaration.spans.iter().any(|span| span.cursor == offset)
            {
                return Ok(());
            }
            return Err(publication_input_error(
                "empty material write does not match an original span cursor",
            ));
        }
        let index = declaration.spans.partition_point(|span| span.end <= offset);
        let span = declaration
            .spans
            .get_mut(index)
            .ok_or_else(|| publication_input_error("material write has no original span"))?;
        if offset < span.begin || offset != span.cursor || end > span.end {
            return Err(publication_input_error(
                "material write must append within one original span",
            ));
        }
        if let Some(original) = &self.comparison {
            if original.original.as_bytes()[offset..end] != *bytes {
                return Err(publication_input_error(
                    "material producer changed its original retained range",
                ));
            }
        } else {
            // SAFETY: the checked destination is wholly unpublished, within the
            // fixed allocation, and only this ledger can write it. copy permits
            // an overlapping borrowed source without a whole-allocation ref.
            unsafe {
                std::ptr::copy(
                    bytes.as_ptr(),
                    self.allocation().data.add(model_begin + offset),
                    bytes.len(),
                )
            };
        }
        span.cursor = end;
        Ok(())
    }

    /// Freeze the actual initialized suffix within its original finite backing
    /// capacity. Reserved trailing capacity is never a checkpoint payload.
    /// This may run only before the first suffix write; retries retain exactly
    /// the same extent, including after the frame has become immutable.
    pub fn declare_frame_suffix(&self, extent: usize) -> Result<(), SemanticTransitionError> {
        self.require_produced_frame()?;
        let mut ledger = self
            .allocation()
            .ledger
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if !ledger.framed || extent > ledger.suffix_extent {
            return Err(publication_input_error(
                "checkpoint suffix exceeds its original capacity",
            ));
        }
        if let Some(original) = ledger.declared_suffix_extent {
            return if original == extent {
                Ok(())
            } else {
                Err(publication_input_error(
                    "checkpoint suffix changed its original extent",
                ))
            };
        }
        let declaration = ledger.declaration.as_ref().ok_or_else(|| {
            publication_input_error("checkpoint suffix requires its original model extent")
        })?;
        let begin = ledger
            .model_begin
            .checked_add(declaration.extent)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        if ledger.full_published || declaration.suffix_cursor != begin {
            return Err(publication_input_error(
                "checkpoint suffix is already initialized or published",
            ));
        }
        ledger.declared_suffix_extent = Some(extent);
        Ok(())
    }

    pub fn write_frame(&self, offset: usize, bytes: &[u8]) -> Result<(), SemanticTransitionError> {
        self.require_produced_frame()?;
        let allocation = self.allocation();
        let mut ledger = allocation
            .ledger
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if !ledger.framed || ledger.full_published {
            return Err(publication_input_error(
                "material frame write requires its original unsealed frame",
            ));
        }
        let end = offset
            .checked_add(bytes.len())
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let prefix = ledger.prefix_begin.is_some_and(|begin| offset >= begin)
            && !ledger.prefix_published
            && offset == ledger.prefix_cursor
            && end <= ledger.model_begin;
        if prefix {
            // SAFETY: the fixed prefix is never part of a published model span.
            unsafe { std::ptr::copy(bytes.as_ptr(), allocation.data.add(offset), bytes.len()) };
            ledger.prefix_cursor = end;
            return Ok(());
        }
        let model_begin = ledger.model_begin;
        let suffix_extent = ledger
            .declared_suffix_extent
            .unwrap_or(ledger.suffix_extent);
        let declaration = ledger.declaration.as_mut().ok_or_else(|| {
            publication_input_error("material suffix requires the original model extent")
        })?;
        let frame_end = model_begin
            .checked_add(declaration.extent)
            .and_then(|size| size.checked_add(suffix_extent))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        if offset != declaration.suffix_cursor || end > frame_end {
            return Err(publication_input_error(
                "material frame write must append within its original prefix or suffix",
            ));
        }
        // SAFETY: this frozen suffix starts exactly after the finalized model
        // range and therefore cannot modify any published model byte.
        unsafe { std::ptr::copy(bytes.as_ptr(), allocation.data.add(offset), bytes.len()) };
        declaration.suffix_cursor = end;
        Ok(())
    }

    pub fn model_extent(&self) -> Result<usize, SemanticTransitionError> {
        self.ledger()
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?
            .declaration
            .as_ref()
            .map(|original| original.extent)
            .ok_or_else(|| {
                publication_input_error(
                    "material output has not declared its original model extent",
                )
            })
    }

    /// Publish only the completed fixed header and native section. The model
    /// and suffix remain disjoint unpublished ranges of the same allocation.
    pub fn seal_prefix(&self) -> Result<SemanticReplayLoadedMaterial, SemanticTransitionError> {
        self.require_produced_frame()?;
        let mut ledger = self
            .allocation()
            .ledger
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if !ledger.framed
            || ledger.prefix_begin.is_none()
            || ledger.prefix_cursor != ledger.model_begin
        {
            return Err(publication_input_error(
                "checkpoint output has not initialized its entire original prefix",
            ));
        }
        ledger.prefix_published = true;
        Ok(self.loaded(ledger.prefix_begin.expect("original declared prefix")..ledger.model_begin))
    }

    pub fn seal_model(&self) -> Result<SemanticReplayLoadedMaterial, SemanticTransitionError> {
        let mut ledger = self
            .ledger()
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let declaration = ledger.declaration.as_ref().ok_or_else(|| {
            publication_input_error("material output has not declared its original spans")
        })?;
        if declaration.spans.iter().any(|span| span.cursor != span.end) {
            return Err(publication_input_error(
                "material output has not initialized every original model span",
            ));
        }
        let range = ledger.model_begin..(ledger.model_begin + declaration.extent);
        ledger.model_published = true;
        if !ledger.framed {
            ledger.full_published = true;
        }
        match &self.comparison {
            Some(original) => original
                .original
                .slice_payload(0..original.original.as_bytes().len()),
            None => Ok(self.loaded(range)),
        }
    }

    /// Publish the initialized frame body while its disjoint suffix remains
    /// writable. This is the exact original message used by a final signer.
    pub fn seal_frame_body(&self) -> Result<SemanticReplayLoadedMaterial, SemanticTransitionError> {
        self.require_produced_frame()?;
        let mut ledger = self
            .allocation()
            .ledger
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let declaration = ledger.declaration.as_ref().ok_or_else(|| {
            publication_input_error("material output has not declared its original spans")
        })?;
        if !ledger.framed
            || ledger.prefix_begin.is_none()
            || ledger.prefix_cursor != ledger.model_begin
            || declaration.spans.iter().any(|span| span.cursor != span.end)
        {
            return Err(publication_input_error(
                "material output has not initialized its original frame body",
            ));
        }
        let end = ledger
            .model_begin
            .checked_add(declaration.extent)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        ledger.prefix_published = true;
        ledger.model_published = true;
        Ok(self.loaded(ledger.prefix_begin.expect("original declared prefix")..end))
    }

    pub fn require_model_payload(
        &self,
        payload: &SemanticReplayLoadedMaterial,
    ) -> Result<(), SemanticTransitionError> {
        let ledger = self
            .ledger()
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let declaration = ledger.declaration.as_ref().ok_or_else(|| {
            publication_input_error("material output has not declared its original spans")
        })?;
        let end = ledger
            .model_begin
            .checked_add(declaration.extent)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        let range = self
            .comparison
            .as_ref()
            .map_or(ledger.model_begin..end, |original| {
                original.original.bytes.range.clone()
            });
        if !ledger.model_published
            || !Arc::ptr_eq(&self.owner, &payload.bytes.owner)
            || payload.bytes.range != range
        {
            return Err(publication_input_error(
                "material payload changed its original output range",
            ));
        }
        Ok(())
    }

    pub fn seal_frame(&self) -> Result<SemanticReplayLoadedMaterial, SemanticTransitionError> {
        self.require_produced_frame()?;
        let allocation = self.allocation();
        let mut ledger = allocation
            .ledger
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        let declaration = ledger.declaration.as_ref().ok_or_else(|| {
            publication_input_error("material output has not declared its original spans")
        })?;
        let suffix_extent = ledger
            .declared_suffix_extent
            .unwrap_or(ledger.suffix_extent);
        let end = ledger
            .model_begin
            .checked_add(declaration.extent)
            .and_then(|size| size.checked_add(suffix_extent))
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        if !ledger.model_published
            || ledger.prefix_begin.is_none()
            || ledger.prefix_cursor != ledger.model_begin
            || declaration.suffix_cursor
                != ledger.model_begin
                    + declaration.extent
                    + if ledger.framed { suffix_extent } else { 0 }
        {
            return Err(publication_input_error(
                "material output has not initialized its entire original frame",
            ));
        }
        ledger.full_published = true;
        Ok(self.loaded(ledger.prefix_begin.expect("original declared frame")..end))
    }

    fn loaded(&self, range: std::ops::Range<usize>) -> SemanticReplayLoadedMaterial {
        SemanticReplayLoadedMaterial {
            bytes: PublicationMaterialBytes {
                owner: Arc::clone(&self.owner),
                range,
            },
            decoded: None,
        }
    }
}
