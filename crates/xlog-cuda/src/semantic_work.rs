//! Original first-recording work geometry and device-produced work. Units are
//! logical operations, not elapsed time, instructions or FLOPs. Every recorded
//! occurrence is retained; conditional producers supply actual units separately
//! from their cold upper geometry.

use std::collections::HashSet;

#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelWorkKind {
    Pointwise = 1,
    Reduction = 2,
    Scan = 3,
    Fill = 4,
    Contraction = 5,
    Softmax = 6,
    LogSoftmax = 7,
    LogSumExp = 8,
    Scatter = 9,
    Gather = 10,
    CopyBytes = 11,
    Cast = 12,
    SortKeys = 13,
    BucketQueries = 14,
    SavedCopyBytes = 15,
    /// An executed model-call marker, separate from logical device work.
    ModelInvocation = 16,
}

impl ModelWorkKind {
    pub fn from_code(code: u64) -> Option<Self> {
        Some(match code {
            1 => Self::Pointwise,
            2 => Self::Reduction,
            3 => Self::Scan,
            4 => Self::Fill,
            5 => Self::Contraction,
            6 => Self::Softmax,
            7 => Self::LogSoftmax,
            8 => Self::LogSumExp,
            9 => Self::Scatter,
            10 => Self::Gather,
            11 => Self::CopyBytes,
            12 => Self::Cast,
            13 => Self::SortKeys,
            14 => Self::BucketQueries,
            15 => Self::SavedCopyBytes,
            16 => Self::ModelInvocation,
            _ => return None,
        })
    }

    fn geometry_units(self, dimensions: &[u64]) -> Result<u64, &'static str> {
        if dimensions.len() > 8 {
            return Err("model work geometry exceeds eight dimensions");
        }
        if self == Self::ModelInvocation {
            return if dimensions.is_empty() {
                Ok(0)
            } else {
                Err("model invocation requires empty geometry")
            };
        }
        if dimensions.contains(&0) {
            return Ok(0);
        }
        dimensions
            .iter()
            .try_fold(1u64, |units, &extent| units.checked_mul(extent))
            .ok_or("model work geometry exceeds the exact u64 counter")
    }
}

/// Immutable kernel input. Unused dimensions are zero, while a scalar has rank
/// zero and one unit. Saved-copy coordinates name the original content witness
/// and tensor occurrence; ordinary operations have no saved witness.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModelWorkEvent {
    pub kind: u64,
    pub witness: u64,
    pub tensor: u64,
    pub rank: u64,
    pub dimensions: [u64; 8],
    /// Zero for exact geometry, otherwise the original retained device slot
    /// containing [actual units, sticky overflow, producer writes].
    pub actual: u64,
}

// SAFETY: the CUDA ABI is thirteen padding-free u64 words, with no host pointers.
unsafe impl crate::DeviceRepr for ModelWorkEvent {}
const _: () = assert!(std::mem::size_of::<ModelWorkEvent>() == 104);

impl ModelWorkEvent {
    pub(crate) fn operation(kind: ModelWorkKind, dimensions: &[u64]) -> Result<Self, &'static str> {
        if matches!(
            kind,
            ModelWorkKind::SavedCopyBytes | ModelWorkKind::ModelInvocation
        ) {
            return Err(
                "saved copies and model invocations require their original native producers",
            );
        }
        Self::new(kind, u64::MAX, u64::MAX, dimensions)
    }

    pub(crate) fn device_operation(
        kind: ModelWorkKind,
        upper_dimensions: &[u64],
        actual: u64,
    ) -> Result<Self, &'static str> {
        if actual == 0 || !actual.is_multiple_of(8) || actual > u64::MAX - 24 {
            return Err("model work requires its complete aligned native device slot");
        }
        let mut event = Self::operation(kind, upper_dimensions)?;
        event.actual = actual;
        Ok(event)
    }

    pub(crate) fn saved(
        witness: u64,
        tensor: u64,
        dimensions: &[u64],
        element_bytes: u64,
    ) -> Result<Self, &'static str> {
        if witness == u64::MAX
            || tensor == u64::MAX
            || dimensions.len() >= 8
            || !matches!(element_bytes, 1 | 2 | 4 | 8)
        {
            return Err("saved-copy work has invalid original tensor geometry");
        }
        let mut shape = [0; 8];
        shape[..dimensions.len()].copy_from_slice(dimensions);
        shape[dimensions.len()] = element_bytes;
        Self::new(
            ModelWorkKind::SavedCopyBytes,
            witness,
            tensor,
            &shape[..dimensions.len() + 1],
        )
    }

    pub(crate) fn invocation(actual: u64) -> Result<Self, &'static str> {
        let event = Self {
            kind: ModelWorkKind::ModelInvocation as u64,
            witness: u64::MAX,
            tensor: u64::MAX,
            rank: 0,
            dimensions: [0; 8],
            actual,
        };
        event.units()?;
        Ok(event)
    }

    fn new(
        kind: ModelWorkKind,
        witness: u64,
        tensor: u64,
        dimensions: &[u64],
    ) -> Result<Self, &'static str> {
        if dimensions.len() > 8 {
            return Err("model work geometry exceeds eight dimensions");
        }
        let mut event = Self {
            kind: kind as u64,
            witness,
            tensor,
            rank: dimensions.len() as u64,
            dimensions: [0; 8],
            actual: 0,
        };
        event.dimensions[..dimensions.len()].copy_from_slice(dimensions);
        event.units()?;
        Ok(event)
    }

    fn units(&self) -> Result<u64, &'static str> {
        let kind = ModelWorkKind::from_code(self.kind).ok_or("model work unit kind is unknown")?;
        if self.rank > 8
            || self.dimensions[self.rank as usize..]
                .iter()
                .any(|&extent| extent != 0)
            || (kind == ModelWorkKind::SavedCopyBytes
                && (self.witness == u64::MAX
                    || self.tensor == u64::MAX
                    || self.actual != 0
                    || self.rank == 0
                    || !matches!(self.dimensions[self.rank as usize - 1], 1 | 2 | 4 | 8)))
            || (kind != ModelWorkKind::SavedCopyBytes
                && (self.witness != u64::MAX || self.tensor != u64::MAX))
        {
            return Err("model work event has noncanonical geometry or occurrence coordinates");
        }
        if kind == ModelWorkKind::ModelInvocation {
            if self.rank != 0
                || self.actual == 0
                || !self.actual.is_multiple_of(8)
                || self.actual > u64::MAX - 24
            {
                return Err("model invocation requires its original scalar native device slot");
            }
        }
        kind.geometry_units(&self.dimensions[..self.rank as usize])
    }
}

struct ModelWorkGeometry {
    kind: ModelWorkKind,
    rank: usize,
    dimensions: [u64; 8],
}

impl ModelWorkGeometry {
    fn new(kind: ModelWorkKind, dimensions: &[u64]) -> Result<(Self, u64), &'static str> {
        if kind == ModelWorkKind::SavedCopyBytes {
            return Err("saved copies require their original native witness producer");
        }
        let units = kind.geometry_units(dimensions)?;
        let mut geometry = Self {
            kind,
            rank: dimensions.len(),
            dimensions: [0; 8],
        };
        geometry.dimensions[..geometry.rank].copy_from_slice(dimensions);
        Ok((geometry, units))
    }
}

/// Immutable ordered geometry for the original report, admitted before recording.
/// Actual indices exclude spans skipped by native proof of never-entered regions.
pub(crate) struct ModelWorkPlan {
    events: Vec<ModelWorkGeometry>,
    region_ends: Vec<usize>,
    recorded_indices: Vec<usize>,
    bound: u64,
    calls: u64,
    next: usize,
}

impl ModelWorkPlan {
    pub(crate) fn new(
        capacity: usize,
        operations: &[(ModelWorkKind, Vec<u64>)],
        region_ends: &[usize],
        region_count: usize,
    ) -> Result<Self, &'static str> {
        if operations.len() > capacity || u64::try_from(operations.len()).is_err() {
            return Err("model operation plan exceeds its original event capacity");
        }
        if region_ends.len() != region_count
            || region_ends.windows(2).any(|ends| ends[0] > ends[1])
            || region_ends
                .last()
                .is_some_and(|end| *end != operations.len())
        {
            return Err("cold operation boundaries changed the original region roster or extent");
        }
        let mut events = Vec::new();
        events
            .try_reserve_exact(operations.len())
            .map_err(|_| "model operation plan reservation failed")?;
        let mut recorded_indices = Vec::new();
        recorded_indices
            .try_reserve_exact(operations.len())
            .map_err(|_| "model operation trace reservation failed")?;
        let mut bound = 0u64;
        let mut calls = 0u64;
        for (kind, dimensions) in operations {
            let (event, units) = ModelWorkGeometry::new(*kind, dimensions)?;
            bound = bound
                .checked_add(units)
                .ok_or("model operation plan exceeds the exact u64 work bound")?;
            if *kind == ModelWorkKind::ModelInvocation {
                calls = calls
                    .checked_add(1)
                    .ok_or("model operation plan exceeds the exact u64 call bound")?;
            }
            events.push(event);
        }
        Ok(Self {
            events,
            region_ends: region_ends.to_vec(),
            recorded_indices,
            bound,
            calls,
            next: 0,
        })
    }

    pub(crate) fn quantities(&self) -> [u64; 3] {
        [self.bound, self.events.len() as u64, self.calls]
    }

    pub(crate) fn require_next(
        &self,
        kind: ModelWorkKind,
        dimensions: &[u64],
        region: Option<usize>,
    ) -> Result<(), &'static str> {
        if let Some(region) = region {
            let (start, end) = self.region(region)?;
            if self.next < start || self.next >= end {
                return Err("cold model registration crossed its original region span");
            }
        }
        if !self.events.get(self.next).is_some_and(|event| {
            event.kind == kind && &event.dimensions[..event.rank] == dimensions
        }) {
            return Err("model registration changed its admitted operation order or geometry");
        }
        Ok(())
    }

    pub(crate) fn consume(&mut self) {
        self.recorded_indices.push(self.next);
        self.next += 1;
    }

    fn region(&self, index: usize) -> Result<(usize, usize), &'static str> {
        let end = *self
            .region_ends
            .get(index)
            .ok_or("cold recording lost its original region boundary")?;
        let start = if index == 0 {
            0
        } else {
            self.region_ends[index - 1]
        };
        Ok((start, end))
    }

    pub(crate) fn require_region_start(&self, index: usize) -> Result<(), &'static str> {
        if self.next != self.region(index)?.0 {
            return Err("cold region did not begin at its original operation boundary");
        }
        Ok(())
    }

    pub(crate) fn require_region_end(&self, index: usize) -> Result<(), &'static str> {
        if self.next != self.region(index)?.1 {
            return Err("successful cold region did not consume its complete admitted span");
        }
        Ok(())
    }

    pub(crate) fn skip_unentered_region(&mut self, index: usize) -> Result<(), &'static str> {
        self.require_region_start(index)?;
        self.next = self.region(index)?.1;
        Ok(())
    }

    pub(crate) fn require_evaluation_result_copy_region(
        &self,
        index: usize,
    ) -> Result<(), &'static str> {
        self.require_region_start(index)?;
        let (start, end) = self.region(index)?;
        if self.events[start..end]
            .iter()
            .any(|event| event.kind != ModelWorkKind::CopyBytes)
        {
            return Err("cancelled evaluation can skip only its unentered result-copy region");
        }
        Ok(())
    }

    pub(crate) fn complete(&self) -> bool {
        self.next == self.events.len()
    }

    pub(crate) fn require_recorded_trace(
        &self,
        recorded: &[ModelWorkEvent],
    ) -> Result<(), &'static str> {
        if recorded.len() != self.recorded_indices.len()
            || !recorded
                .iter()
                .zip(&self.recorded_indices)
                .all(|(actual, index)| {
                    self.events.get(*index).is_some_and(|planned| {
                        actual.kind == planned.kind as u64
                            && actual.rank as usize == planned.rank
                            && actual.dimensions == planned.dimensions
                    })
                })
        {
            return Err("cold model report changed its authenticated admitted operation trace");
        }
        Ok(())
    }
}

/// Capacity is reserved cold. Neither host vector growth nor device allocation
/// is allowed when original producer callbacks append their first-record events.
pub(crate) struct ModelWorkRecording {
    capacity: usize,
    events: Vec<ModelWorkEvent>,
    saved: HashSet<(u64, u64)>,
    bound: u64,
    minimum: u64,
    frozen: bool,
}

impl ModelWorkRecording {
    pub(crate) fn new(capacity: usize) -> Result<Self, &'static str> {
        if capacity == 0 {
            return Err("model work requires a positive cold event capacity");
        }
        let mut events = Vec::new();
        events
            .try_reserve_exact(capacity)
            .map_err(|_| "model work event reservation failed")?;
        let mut saved = HashSet::new();
        saved
            .try_reserve(capacity)
            .map_err(|_| "saved occurrence reservation failed")?;
        Ok(Self {
            capacity,
            events,
            saved,
            bound: 0,
            minimum: 0,
            frozen: false,
        })
    }

    pub(crate) fn push(&mut self, event: ModelWorkEvent) -> Result<(), &'static str> {
        if self.frozen {
            return Err("original model work is already frozen");
        }
        if self.events.len() == self.capacity {
            return Err("model work exceeds its cold event capacity");
        }
        let bound = self
            .bound
            .checked_add(event.units()?)
            .ok_or("model work sum exceeds the exact u64 counter")?;
        let saved = event.kind == ModelWorkKind::SavedCopyBytes as u64;
        if saved && self.saved.contains(&(event.witness, event.tensor)) {
            return Err("saved-tensor occurrence was already charged");
        }
        self.events.push(event);
        if saved {
            self.saved.insert((event.witness, event.tensor));
        }
        if event.actual == 0 {
            // This exact contribution is a subset of the checked total bound.
            self.minimum += bound - self.bound;
        }
        self.bound = bound;
        Ok(())
    }

    #[cfg(any(test, feature = "semantic-policy"))]
    pub(crate) fn freeze(&mut self) -> Result<u64, &'static str> {
        if self.frozen || self.events.is_empty() || self.bound == 0 {
            return Err("model work requires one nonempty original recording and a single freeze");
        }
        self.freeze_cold()
    }

    /// Cold callbacks may genuinely perform no model work. Their separately
    /// checked recorder lifecycle, not a fabricated event, certifies emptiness.
    #[cfg(any(test, feature = "semantic-policy"))]
    pub(crate) fn freeze_cold(&mut self) -> Result<u64, &'static str> {
        if self.frozen {
            return Err("original cold model work is already frozen");
        }
        self.frozen = true;
        Ok(self.bound)
    }

    #[cfg(feature = "semantic-policy")]
    pub(crate) fn require_model_invocations(&self) -> Result<(), &'static str> {
        if !self
            .events
            .iter()
            .any(|event| event.kind == ModelWorkKind::ModelInvocation as u64)
        {
            return Err("model work lacks its original executed-call marker; zero is not an unobserved call count");
        }
        Ok(())
    }

    pub(crate) fn events(&self) -> &[ModelWorkEvent] {
        &self.events
    }
    pub(crate) fn frozen_bound(&self) -> Option<u64> {
        self.frozen.then_some(self.bound)
    }
}

/// Checked device accumulator. The model and native contributions are kept
/// separate; neither the old edit facts nor an estimated capacity is a charge.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExecutionWork {
    pub raw: u64,
    pub model_once: u64,
    pub native_attempt: u64,
    pub model_events: u64,
    pub model_bound: u64,
    pub overflow: u64,
    /// Fixed native event order follows the producer's unit-tariff registry.
    pub native_events: [u64; 9],
    /// Attribution of existing native units; never added a second time to raw.
    pub candidates: [u64; 3],
    /// Zero is shared work; one through three select an original candidate.
    pub active_candidate: u64,
}

impl ExecutionWork {
    #[cfg(any(test, feature = "semantic-policy"))]
    pub(crate) fn validate_model(
        &self,
        recording: &ModelWorkRecording,
        refused_overflow: bool,
    ) -> bool {
        recording.frozen_bound() == Some(self.model_bound)
            && self.model_once.checked_add(self.native_attempt) == Some(self.raw)
            && self
                .native_events
                .iter()
                .try_fold(0u64, |sum, value| sum.checked_add(*value))
                == Some(self.native_attempt)
            && self
                .candidates
                .iter()
                .try_fold(0u64, |sum, value| sum.checked_add(*value))
                .is_some_and(|sum| sum <= self.native_attempt)
            && self.active_candidate == 0
            && if refused_overflow {
                self.overflow == 1 && self.model_events <= recording.events().len() as u64
            } else {
                self.overflow == 0
                    && self.model_events == recording.events().len() as u64
                    && (recording.minimum..=self.model_bound).contains(&self.model_once)
            }
    }
}

const _: () = assert!(std::mem::size_of::<ExecutionWork>() == 152);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_geometry_counts_declared_units_without_dense_mask_discount() {
        let event = ModelWorkEvent::operation(ModelWorkKind::Contraction, &[2, 3, 5]).unwrap();
        assert_eq!(event.units().unwrap(), 30);
        assert_eq!(
            ModelWorkEvent::operation(ModelWorkKind::Pointwise, &[])
                .unwrap()
                .units()
                .unwrap(),
            1
        );
        assert_eq!(
            ModelWorkEvent::operation(ModelWorkKind::Gather, &[4, 0, 7])
                .unwrap()
                .units()
                .unwrap(),
            0
        );
        assert!(ModelWorkEvent::operation(ModelWorkKind::Contraction, &[u64::MAX, 2]).is_err());
        assert!(ModelWorkEvent::operation(ModelWorkKind::Pointwise, &[1; 9]).is_err());
    }

    #[test]
    fn saved_occurrences_keep_original_witness_coordinates_and_byte_extents() {
        let mut recording = ModelWorkRecording::new(3).unwrap();
        recording
            .push(ModelWorkEvent::saved(7, 0, &[2, 3], 4).unwrap())
            .unwrap();
        recording
            .push(ModelWorkEvent::saved(8, 0, &[2, 3], 4).unwrap())
            .unwrap();
        assert!(recording
            .push(ModelWorkEvent::saved(7, 0, &[2, 3], 4).unwrap())
            .is_err());
        assert_eq!(recording.freeze().unwrap(), 48);
        assert_eq!(recording.events()[0].witness, 7);
        assert_eq!(recording.events()[1].witness, 8);
    }

    #[test]
    fn recording_freezes_once_and_rejects_capacity_and_sum_overflow() {
        let mut recording = ModelWorkRecording::new(1).unwrap();
        let event = ModelWorkEvent::operation(ModelWorkKind::CopyBytes, &[12]).unwrap();
        recording.push(event).unwrap();
        assert!(recording.push(event).is_err());
        assert_eq!(recording.freeze().unwrap(), 12);
        assert!(recording.freeze().is_err());
        assert!(recording.push(event).is_err());
        let mut overflow = ModelWorkRecording::new(2).unwrap();
        overflow
            .push(ModelWorkEvent::operation(ModelWorkKind::CopyBytes, &[u64::MAX]).unwrap())
            .unwrap();
        assert!(overflow
            .push(ModelWorkEvent::operation(ModelWorkKind::Fill, &[1]).unwrap())
            .is_err());
        assert!(ModelWorkRecording::new(0).is_err());
        assert!(ModelWorkRecording::new(1).unwrap().freeze().is_err());
    }

    #[test]
    fn completion_consumes_the_original_frozen_geometry_and_checked_common_sum() {
        let mut recording = ModelWorkRecording::new(1).unwrap();
        recording
            .push(ModelWorkEvent::operation(ModelWorkKind::Contraction, &[3, 7]).unwrap())
            .unwrap();
        let mut actual = ExecutionWork {
            model_once: 21,
            native_attempt: 5,
            raw: 26,
            model_events: 1,
            model_bound: 21,
            native_events: [5, 0, 0, 0, 0, 0, 0, 0, 0],
            ..ExecutionWork::default()
        };
        assert!(!actual.validate_model(&recording, false));
        recording.freeze().unwrap();
        assert!(actual.validate_model(&recording, false));
        actual.raw = 21;
        assert!(!actual.validate_model(&recording, false));
        actual.raw = 26;
        actual.model_events = 0;
        assert!(!actual.validate_model(&recording, false));
        actual.overflow = 1;
        assert!(actual.validate_model(&recording, true));
        actual.model_bound = 22;
        assert!(!actual.validate_model(&recording, true));
    }

    #[test]
    fn native_event_sum_and_candidate_attribution_are_not_additional_work() {
        let mut recording = ModelWorkRecording::new(1).unwrap();
        recording
            .push(ModelWorkEvent::operation(ModelWorkKind::Fill, &[7]).unwrap())
            .unwrap();
        recording.freeze().unwrap();
        let mut actual = ExecutionWork {
            raw: 17,
            model_once: 7,
            model_events: 1,
            model_bound: 7,
            native_attempt: 10,
            native_events: [2, 3, 0, 0, 0, 0, 0, 1, 4],
            candidates: [1, 3, 4],
            ..ExecutionWork::default()
        };
        assert!(actual.validate_model(&recording, false));
        actual.native_events[1] += 1;
        assert!(!actual.validate_model(&recording, false));
        actual.native_events[1] -= 1;
        actual.candidates[0] = 4;
        assert!(!actual.validate_model(&recording, false));
        actual.candidates[0] = 1;
        actual.active_candidate = 2;
        assert!(!actual.validate_model(&recording, false));
    }
}
