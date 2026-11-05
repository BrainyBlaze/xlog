//! Original resource interval shared by admitted and prepared segment owners.
use super::*;
use learning_phase::ResourceObserver;
use xlog_cuda::{SemanticPreparedSegmentNonSubmission, SemanticPreparedStep};

pyo3::create_exception!(
    pyxlog._native, SemanticSegmentRetirementPending, PyRuntimeError,
    "The original segment retirement awaits completion; retain its owner and never repeat completed work."
);

pub(super) fn retirement_error(py: Python<'_>, error: xlog_cuda::SemanticTransitionError) -> PyErr {
    let pending = error.native_completion_pending();
    let cause = xlog_err(error);
    if !pending {
        return cause;
    }
    let error = SemanticSegmentRetirementPending::new_err(
        "the original native retirement is pending; continue that same retirement",
    );
    error.set_cause(py, Some(cause));
    error
}

pub(super) struct OriginalSegmentResourceInterval {
    observer: ResourceObserver,
    ordinal: u64,
    state: Mutex<ResourceIntervalState>,
}

#[derive(Default)]
struct ResourceIntervalState {
    begin_entered: bool,
    begun: bool,
    steps: Option<Vec<SemanticPreparedStep>>,
    capture_stream: Option<u64>,
    cancellation_entered: bool,
    cancellation_complete: bool,
    retirement: Option<NativeResourceRetirement>,
    finish_entered: bool,
    release_entered: bool,
    released: bool,
    interval: Option<serde_json::Value>,
    terminal_failure: Option<PyErr>,
    slot: Option<std::sync::Weak<Mutex<Option<RetiredSegmentResourceInterval>>>>,
}

struct NativeResourceRetirement {
    streams: Vec<u64>,
    cancellation: Option<SemanticPreparedSegmentNonSubmission>,
    whole_only: bool,
}

pub(super) type RetiredResourceSlot = Arc<Mutex<Option<RetiredSegmentResourceInterval>>>;

pub(super) struct RetiredSegmentResourceInterval {
    pub(super) task_use_key: usize,
    pub(super) interval: Arc<OriginalSegmentResourceInterval>,
}

impl OriginalSegmentResourceInterval {
    pub(super) fn new(observer: ResourceObserver, ordinal: u64) -> Self {
        Self {
            observer,
            ordinal,
            state: Mutex::new(ResourceIntervalState::default()),
        }
    }

    fn state(&self) -> PyResult<std::sync::MutexGuard<'_, ResourceIntervalState>> {
        self.state
            .lock()
            .map_err(|_| invalid("original segment resource custody is poisoned"))
    }

    pub(super) fn derive(
        &self,
        py: Python<'_>,
        device: ([u8; 16], u64),
        ordinal: u64,
    ) -> PyResult<Arc<Self>> {
        Ok(Arc::new(Self::new(
            self.observer.derive(py, device)?,
            ordinal,
        )))
    }

    pub(super) fn begin(&self, py: Python<'_>) -> PyResult<()> {
        {
            let mut state = self.state()?;
            if state.begun {
                return Ok(());
            }
            if state.begin_entered {
                return Err(invalid(
                    "unknown original interval beginning cannot issue another interval",
                ));
            }
            state.begin_entered = true;
        }
        self.observer.begin(py, self.ordinal)?;
        self.state()?.begun = true;
        Ok(())
    }

    pub(super) fn require_ordinal(&self, ordinal: u64) -> PyResult<()> {
        if self.ordinal != ordinal {
            return Err(invalid(
                "frozen context changed its original resource coordinate",
            ));
        }
        Ok(())
    }

    pub(super) fn require_begun(&self) -> PyResult<()> {
        if !self.state()?.begun {
            return Err(invalid(
                "segment allocation requires its original resource interval",
            ));
        }
        Ok(())
    }

    pub(super) fn retain_steps(&self, steps: &[SemanticPreparedStep]) -> PyResult<()> {
        let mut state = self.state()?;
        if state.retirement.is_some() {
            return Err(invalid(
                "retired segment cannot replace its original captured roster",
            ));
        }
        if let Some(original) = &state.steps {
            if original.len() != steps.len()
                || original
                    .iter()
                    .zip(steps)
                    .any(|(original, step)| !original.same_handle(step))
            {
                return Err(invalid(
                    "segment resource interval changed its original native roster",
                ));
            }
        } else {
            state.steps = Some(steps.to_vec());
        }
        Ok(())
    }

    pub(super) fn retain_capture_stream(&self, stream: u64) -> PyResult<()> {
        self.require_begun()?;
        let mut state = self.state()?;
        if stream == 0
            || state
                .capture_stream
                .is_some_and(|original| original != stream)
        {
            return Err(invalid(
                "segment resource interval changed its actual recording stream",
            ));
        }
        state.capture_stream = Some(stream);
        Ok(())
    }

    pub(super) fn bind_step(
        &self,
        py: Python<'_>,
        index: usize,
        step: &SemanticPreparedStep,
        stream: u64,
        end: bool,
    ) -> PyResult<()> {
        self.retain_capture_stream(stream)?;
        {
            let state = self.state()?;
            if state
                .steps
                .as_ref()
                .and_then(|steps| steps.get(index))
                .is_none_or(|original| !original.same_handle(step))
            {
                return Err(invalid("resource capture changed its original native step"));
            }
        }
        let index = u64::try_from(index).map_err(|_| invalid("capture step exceeds u64"))?;
        let ordinal = self
            .ordinal
            .checked_add(index)
            .ok_or_else(|| invalid("resource capture ordinal overflowed"))?;
        self.observer
            .bind_step_capture(py, ordinal, index, stream, end)
    }

    pub(super) fn mark_native_retired(
        self: &Arc<Self>,
        streams: &[u64],
        cancellation: Option<&SemanticPreparedSegmentNonSubmission>,
        task_use_key: usize,
        slot: &RetiredResourceSlot,
    ) -> PyResult<()> {
        let mut retired = slot
            .lock()
            .map_err(|_| invalid("original resource retirement retention is poisoned"))?;
        if retired
            .as_ref()
            .is_some_and(|original| !Arc::ptr_eq(&original.interval, self))
        {
            return Err(invalid(
                "another original resource retirement remains unresolved",
            ));
        }
        let mut state = self.state()?;
        if let Some(original) = &state.retirement {
            if original.streams != streams {
                return Err(invalid(
                    "resource retirement changed its actual original consumers",
                ));
            }
            return Ok(());
        }
        let steps = state.steps.as_ref().ok_or_else(|| {
            invalid("resource retirement lost its original planned native roster")
        })?;
        if cancellation.is_some_and(|proof| !proof.matches(steps)) {
            return Err(invalid(
                "resource retirement changed its native non-submission proof",
            ));
        }
        state.retirement = Some(NativeResourceRetirement {
            streams: streams.to_vec(),
            cancellation: cancellation.cloned(),
            whole_only: false,
        });
        state.slot = Some(Arc::downgrade(slot));
        drop(state);
        *retired = Some(RetiredSegmentResourceInterval {
            task_use_key,
            interval: Arc::clone(self),
        });
        Ok(())
    }

    /// Called only after the canonical native terminal deallocator succeeded.
    /// A never-built context has no graph roster to cancel or invent.
    pub(super) fn mark_terminal_closed(
        self: &Arc<Self>,
        py: Python<'_>,
        task_use_key: usize,
        slot: &RetiredResourceSlot,
    ) -> PyResult<()> {
        {
            let state = self.state()?;
            if state.released || state.retirement.is_some() {
                return Ok(());
            }
        }
        let mut retired = slot
            .lock()
            .map_err(|_| invalid("original resource retirement retention is poisoned"))?;
        if retired
            .as_ref()
            .is_some_and(|original| !Arc::ptr_eq(&original.interval, self))
        {
            return Err(invalid(
                "another original resource retirement remains unresolved",
            ));
        }
        let mut state = self.state()?;
        if state.released || state.retirement.is_some() {
            return Ok(());
        }
        if !state.begun || state.steps.is_some() || self.observer.has_step_captures(py)? {
            return Err(invalid(
                "terminal context release lost its original graph retirement",
            ));
        }
        state.steps = Some(Vec::new());
        state.retirement = Some(NativeResourceRetirement {
            streams: Vec::new(),
            cancellation: None,
            whole_only: true,
        });
        state.slot = Some(Arc::downgrade(slot));
        *retired = Some(RetiredSegmentResourceInterval {
            task_use_key,
            interval: Arc::clone(self),
        });
        Ok(())
    }

    pub(super) fn require_retired_streams(&self, streams: &[u64]) -> PyResult<()> {
        if self
            .state()?
            .retirement
            .as_ref()
            .is_none_or(|original| original.streams != streams)
        {
            return Err(invalid(
                "retirement continuation requires its actual completed native consumers",
            ));
        }
        Ok(())
    }

    fn observer_error(&self, py: Python<'_>, cause: PyErr) -> PyErr {
        match self.observer.completion_pending(py) {
            Ok(true) => {
                let error = SemanticSegmentRetirementPending::new_err(
                    "the original observer completion is pending; continue the same retirement owner");
                error.set_cause(py, Some(cause));
                error
            }
            _ => {
                if let Ok(mut state) = self.state() {
                    state.terminal_failure = Some(cause.clone_ref(py));
                }
                cause
            }
        }
    }

    pub(super) fn prepare_cancellation(
        &self,
        py: Python<'_>,
        proof: &SemanticPreparedSegmentNonSubmission,
    ) -> PyResult<()> {
        let (steps, stream) = {
            let state = self.state()?;
            let steps = state
                .steps
                .clone()
                .ok_or_else(|| invalid("cancellation lost its original planned roster"))?;
            if !proof.matches(&steps) {
                return Err(invalid(
                    "cancellation changed its genuine native non-submission proof",
                ));
            }
            (steps, state.capture_stream)
        };
        if !self.state()?.cancellation_complete {
            if let Some(stream) = stream {
                let entered = {
                    let mut state = self.state()?;
                    let entered = state.cancellation_entered;
                    state.cancellation_entered = true;
                    entered
                };
                let result = if entered {
                    self.observer
                        .resolve_step_cancellation(py, self.ordinal, &steps, stream, proof)
                } else {
                    self.observer
                        .cancel_step_captures(py, self.ordinal, &steps, stream, proof)
                };
                result.map_err(|error| self.observer_error(py, error))?;
            } else if self.observer.has_step_captures(py)? {
                return Err(invalid(
                    "unsubmitted resource capture lost its original stream",
                ));
            }
            // Without any entered capture registration there are no
            // observer step obligations. The native proof still covers
            // the entire original planned roster; no step is fabricated.
            self.state()?.cancellation_complete = true;
        }
        Ok(())
    }

    pub(super) fn finish(&self, py: Python<'_>) -> PyResult<serde_json::Value> {
        if let Some(error) = self.state()?.terminal_failure.as_ref() {
            return Err(error.clone_ref(py));
        }
        if self.state()?.released {
            return self.state()?.interval.clone().ok_or_else(|| {
                invalid("completed resource retirement lost its original interval")
            });
        }
        let (steps, cancellation, whole_only) = {
            let state = self.state()?;
            let original = state
                .retirement
                .as_ref()
                .ok_or_else(|| invalid("observer retirement precedes actual native final uses"))?;
            (
                state
                    .steps
                    .clone()
                    .ok_or_else(|| invalid("retirement lost its planned roster"))?,
                original.cancellation.clone(),
                original.whole_only,
            )
        };
        if let Some(proof) = &cancellation {
            self.prepare_cancellation(py, proof)?;
        }
        let finish = {
            let mut state = self.state()?;
            let enter = !state.finish_entered;
            state.finish_entered = true;
            enter
        };
        if finish {
            self.observer
                .finish(py)
                .map_err(|error| self.observer_error(py, error))?;
        }
        if cancellation.is_some() || whole_only {
            self.observer
                .backing_peak(py)
                .map_err(|error| self.observer_error(py, error))?;
        } else {
            self.observer
                .step_backing_peaks(py, steps.len())
                .map_err(|error| self.observer_error(py, error))?;
        }
        if self.state()?.interval.is_none() {
            let interval = self.observer.original_interval_projection()?;
            self.state()?.interval = Some(interval);
        }
        {
            let mut state = self.state()?;
            if state.release_entered {
                return Err(invalid(
                    "unknown original observer release cannot be repeated",
                ));
            }
            state.release_entered = true;
        }
        self.observer.release(py)?;
        let mut state = self.state()?;
        state.released = true;
        // Only CPU evidence and original scalar identities survive closure.
        state.steps = None;
        state.retirement = None;
        let value = state.interval.clone().expect("retained original interval");
        let slot = state.slot.take().and_then(|slot| slot.upgrade());
        drop(state);
        let removed = if let Some(slot) = slot {
            let mut retired = slot
                .lock()
                .map_err(|_| invalid("original resource retirement retention is poisoned"))?;
            if retired
                .as_ref()
                .is_some_and(|original| std::ptr::eq(original.interval.as_ref(), self))
            {
                retired.take()
            } else {
                None
            }
        } else {
            None
        };
        drop(removed);
        Ok(value)
    }

    pub(super) fn projection(&self) -> PyResult<Option<serde_json::Value>> {
        let state = self.state()?;
        Ok(if state.released {
            state.interval.clone()
        } else {
            None
        })
    }
}

/// A CPU-only continuation. It cannot keep a dormant model context, producer,
/// graph or numerical owner alive while their actual release is observed.
#[pyclass(
    name = "SemanticSegmentResourceRetirement",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticSegmentResourceRetirement {
    pub(super) interval: Arc<OriginalSegmentResourceInterval>,
}

#[pymethods]
impl PySemanticSegmentResourceRetirement {
    fn finish_retirement(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        finish_original_retirement(py, &self.interval)
    }

    #[getter]
    fn resource_interval(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.interval.observer.require_original(py)?;
        self.interval
            .projection()?
            .map(|value| native_observation_report(py, value))
            .transpose()
    }
}

/// Both public instructions and inactive context disposal use this same owner.
/// A detached handle lets the caller drop numerical contexts before this call.
pub(super) fn finish_original_retirement(
    py: Python<'_>,
    interval: &Arc<OriginalSegmentResourceInterval>,
) -> PyResult<Py<PyDict>> {
    interval.observer.require_original(py)?;
    native_observation_report(py, interval.finish(py)?)
}
