//! Original frozen source content expressions and their immutable projections.

use super::prepared_import::PySemanticPreparedSourceContent;
use super::*;
use xlog_cuda::{
    SemanticSourceAllocationDescriptor, SemanticSourceContentAllocation,
    SemanticSourceContentEvent, SemanticSourceContentFamily, SemanticSourceContentKind,
    SemanticSourceContentOccurrence, SemanticSourceContentPhase, SemanticSourceTensorDescriptor,
};

#[pyclass(
    name = "SemanticSourceContentFamily",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticSourceContentFamily {
    pub(super) inner: SemanticSourceContentFamily,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

#[pyclass(
    name = "SemanticSourceContentOccurrence",
    module = "pyxlog._native",
    frozen
)]
pub(crate) struct PySemanticSourceContentOccurrence {
    pub(super) inner: SemanticSourceContentOccurrence,
    budget: xlog_cuda::SemanticReplayMaterialBudget,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
    tensor_projection: Mutex<Option<RetainedOccurrenceTuple>>,
    allocation_projection: Mutex<Option<RetainedOccurrenceTuple>>,
}

struct RetainedOccurrenceTuple {
    value: Py<PyTuple>,
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

pub(super) struct OriginalSourceContentOccurrences {
    events: Py<PyTuple>,
    native: Vec<SemanticSourceContentOccurrence>,
    handles: Vec<Py<PySemanticSourceContentOccurrence>>,
    result: Option<RetainedOccurrenceTuple>,
    phases: [Option<RetainedOccurrenceTuple>; 3],
    _metadata: xlog_cuda::SemanticReplayMaterialReservation,
}

pub(super) fn family(
    py: Python<'_>,
    inner: SemanticSourceContentFamily,
    budget: &xlog_cuda::SemanticReplayMaterialBudget,
) -> PyResult<Py<PySemanticSourceContentFamily>> {
    let metadata =
        material_bytes::reserve_python_material::<PySemanticSourceContentFamily>(py, budget)?;
    Py::new(
        py,
        PySemanticSourceContentFamily {
            inner,
            _metadata: metadata,
        },
    )
}

pub(super) fn family_arguments(
    phase: &Bound<'_, PyAny>,
    kind: &Bound<'_, PyAny>,
) -> PyResult<(SemanticSourceContentPhase, SemanticSourceContentKind)> {
    if !phase.is_exact_instance_of::<PyString>() || !kind.is_exact_instance_of::<PyString>() {
        return Err(invalid(
            "source content families require exact builtin strings",
        ));
    }
    // Compare Unicode directly against fixed ASCII spellings. Encoding an
    // arbitrary caller string would create an unowned cached UTF-8 buffer.
    let same = |value: &Bound<'_, PyAny>, spelling: &std::ffi::CStr| unsafe {
        pyo3::ffi::PyUnicode_CompareWithASCIIString(value.as_ptr(), spelling.as_ptr()) == 0
    };
    let phase = if same(phase, c"model") {
        SemanticSourceContentPhase::Model
    } else if same(phase, c"pack-policy") {
        SemanticSourceContentPhase::PackPolicy
    } else if same(phase, c"finalize-continuation") {
        SemanticSourceContentPhase::FinalizeContinuation
    } else {
        return Err(invalid("unknown original source content phase"));
    };
    let kind = if same(kind, c"bind-model") {
        SemanticSourceContentKind::BindModel
    } else if same(kind, c"capture") {
        SemanticSourceContentKind::Capture
    } else if same(kind, c"verify") {
        SemanticSourceContentKind::Verify
    } else {
        return Err(invalid("unknown original source content command family"));
    };
    Ok((phase, kind))
}

pub(super) fn prepared_section_arguments(
    kind: Option<&Bound<'_, PyAny>>,
    bank: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<(SemanticTransitionKind, usize)>> {
    match (kind, bank) {
        (None, None) => Ok(None),
        (Some(kind), Some(bank)) => {
            if !kind.is_exact_instance_of::<PyString>() || !bank.is_exact_instance_of::<PyInt>() {
                return Err(invalid(
                    "prepared content section requires an exact string and integer bank",
                ));
            }
            let same = |spelling: &std::ffi::CStr| unsafe {
                pyo3::ffi::PyUnicode_CompareWithASCIIString(kind.as_ptr(), spelling.as_ptr()) == 0
            };
            let kind = if same(c"recompute") {
                SemanticTransitionKind::Recompute
            } else if same(c"proposal") {
                SemanticTransitionKind::Proposal
            } else {
                return Err(invalid(
                    "prepared content section requires the original Recompute or Proposal",
                ));
            };
            let bank = bank.extract::<usize>()?;
            if bank > 1 {
                return Err(invalid(
                    "prepared content section requires bank zero or one",
                ));
            }
            Ok(Some((kind, bank)))
        }
        _ => Err(invalid(
            "prepared content section requires both kind and bank",
        )),
    }
}

fn prepared_kind_name(
    section: Option<(SemanticTransitionKind, usize)>,
) -> PyResult<Option<&'static str>> {
    match section {
        None => Ok(None),
        Some((SemanticTransitionKind::Recompute, _)) => Ok(Some("recompute")),
        Some((SemanticTransitionKind::Proposal, _)) => Ok(Some("proposal")),
        _ => Err(invalid(
            "original prepared content section has an invalid transition",
        )),
    }
}

pub(super) fn reserve_events(
    events: &Bound<'_, PyAny>,
    pool: &xlog_cuda::SemanticReplayMaterialBudget,
) -> PyResult<xlog_cuda::SemanticReplayMaterialReservation> {
    let mut transport = 16 * 1024 * 1024;
    let rows = exact_tuple(events, None, &mut transport)?;
    let mut extent = rows
        .len()
        .checked_mul(std::mem::size_of::<
            SemanticSourceContentEvent<SemanticSourceContentAllocation>,
        >())
        .ok_or_else(|| invalid("source content input roster extent overflow"))?;
    for row in rows.iter() {
        let row = exact_tuple(&row, Some(5), &mut transport)?;
        for (index, size) in [
            (3, std::mem::size_of::<SemanticSourceTensorDescriptor>()),
            (
                4,
                std::mem::size_of::<
                    SemanticSourceAllocationDescriptor<SemanticSourceContentAllocation>,
                >(),
            ),
        ] {
            let count = exact_tuple(&row.get_item(index)?, None, &mut transport)?.len();
            extent = extent
                .checked_add(
                    count
                        .checked_mul(size)
                        .ok_or_else(|| invalid("source content input geometry extent overflow"))?,
                )
                .ok_or_else(|| invalid("source content input geometry extent overflow"))?;
        }
    }
    pool.reserve_auxiliary(
        u64::try_from(extent).map_err(|_| invalid("source content input roster exceeds u64"))?,
    )
    .map_err(xlog_err)
}

pub(super) fn reserve_occurrences(
    count: usize,
    pool: &xlog_cuda::SemanticReplayMaterialBudget,
) -> PyResult<xlog_cuda::SemanticReplayMaterialReservation> {
    let extent = count
        .checked_mul(
            std::mem::size_of::<SemanticSourceContentOccurrence>()
                + std::mem::size_of::<Py<PySemanticSourceContentOccurrence>>(),
        )
        .ok_or_else(|| invalid("source content handoff roster extent overflow"))?;
    pool.reserve_auxiliary(
        u64::try_from(extent).map_err(|_| invalid("source content handoff roster exceeds u64"))?,
    )
    .map_err(xlog_err)
}

impl OriginalSourceContentOccurrences {
    pub(super) fn new(
        events: &Bound<'_, PyAny>,
        native: Vec<SemanticSourceContentOccurrence>,
        metadata: xlog_cuda::SemanticReplayMaterialReservation,
    ) -> PyResult<Self> {
        Ok(Self {
            events: events.cast::<PyTuple>()?.clone().unbind(),
            handles: Vec::with_capacity(native.len()),
            native,
            result: None,
            phases: [None, None, None],
            _metadata: metadata,
        })
    }

    pub(super) fn require_events(&self, events: &Bound<'_, PyAny>) -> PyResult<()> {
        if self.events.as_ptr() != events.as_ptr() {
            return Err(invalid(
                "source content freeze replaced its original event tuple",
            ));
        }
        Ok(())
    }

    pub(super) fn project(
        &mut self,
        py: Python<'_>,
        pool: &xlog_cuda::SemanticReplayMaterialBudget,
    ) -> PyResult<Py<PyTuple>> {
        if let Some(original) = &self.result {
            return Ok(original.value.clone_ref(py));
        }
        while self.handles.len() < self.native.len() {
            let inner = self.native[self.handles.len()].clone();
            let metadata = material_bytes::reserve_python_material::<
                PySemanticSourceContentOccurrence,
            >(py, pool)?;
            self.handles.push(Py::new(
                py,
                PySemanticSourceContentOccurrence {
                    inner,
                    budget: pool.clone(),
                    _metadata: metadata,
                    tensor_projection: Mutex::new(None),
                    allocation_projection: Mutex::new(None),
                },
            )?);
        }
        let metadata = pool
            .reserve_auxiliary(
                u64::try_from(python_metadata_tuple_extent(py, self.handles.len())?)
                    .map_err(|_| invalid("source content handle tuple exceeds u64"))?,
            )
            .map_err(xlog_err)?;
        let value =
            PyTuple::new(py, self.handles.iter().map(|handle| handle.clone_ref(py)))?.unbind();
        self.result = Some(RetainedOccurrenceTuple {
            value: value.clone_ref(py),
            _metadata: metadata,
        });
        Ok(value)
    }

    pub(super) fn phase_projection(
        &mut self,
        py: Python<'_>,
        pool: &xlog_cuda::SemanticReplayMaterialBudget,
        phase: SemanticSourceContentPhase,
    ) -> PyResult<Py<PyTuple>> {
        self.project(py, pool)?;
        let index = match phase {
            SemanticSourceContentPhase::Model => 0,
            SemanticSourceContentPhase::PackPolicy => 1,
            SemanticSourceContentPhase::FinalizeContinuation => 2,
        };
        if let Some(original) = &self.phases[index] {
            return Ok(original.value.clone_ref(py));
        }
        let count = self
            .handles
            .iter()
            .filter(|handle| handle.borrow(py).inner.phase() == phase)
            .count();
        let metadata = pool
            .reserve_auxiliary(
                u64::try_from(python_metadata_tuple_extent(py, count)?)
                    .map_err(|_| invalid("source content phase projection exceeds u64"))?,
            )
            .map_err(xlog_err)?;
        // Each handle's phase is immutable native geometry in a frozen class;
        // the counted filter therefore supplies exactly this many handles.
        let mut handles = self
            .handles
            .iter()
            .filter(|handle| handle.borrow(py).inner.phase() == phase);
        let value = PyTuple::new(
            py,
            (0..count).map(|_| {
                handles
                    .next()
                    .expect("counted frozen source content phase")
                    .clone_ref(py)
            }),
        )?
        .unbind();
        self.phases[index] = Some(RetainedOccurrenceTuple {
            value: value.clone_ref(py),
            _metadata: metadata,
        });
        Ok(value)
    }
}

fn kind_name(kind: SemanticSourceContentKind) -> &'static str {
    match kind {
        SemanticSourceContentKind::BindModel => "bind-model",
        SemanticSourceContentKind::Capture => "capture",
        SemanticSourceContentKind::Verify => "verify",
    }
}

fn phase_name(phase: SemanticSourceContentPhase) -> &'static str {
    match phase {
        SemanticSourceContentPhase::Model => "model",
        SemanticSourceContentPhase::PackPolicy => "pack-policy",
        SemanticSourceContentPhase::FinalizeContinuation => "finalize-continuation",
    }
}

#[pymethods]
impl PySemanticSourceContentFamily {
    fn kind(&self) -> &'static str {
        kind_name(self.inner.kind())
    }

    fn phase(&self) -> &'static str {
        phase_name(self.inner.phase())
    }

    fn prepared_kind(&self) -> PyResult<Option<&'static str>> {
        prepared_kind_name(self.inner.prepared_section())
    }

    fn prepared_bank(&self) -> Option<usize> {
        self.inner.prepared_section().map(|(_, bank)| bank)
    }
}

#[pymethods]
impl PySemanticSourceContentOccurrence {
    fn kind(&self) -> &'static str {
        kind_name(self.inner.kind())
    }

    fn phase(&self) -> &'static str {
        phase_name(self.inner.phase())
    }

    fn prepared_kind(&self) -> PyResult<Option<&'static str>> {
        prepared_kind_name(self.inner.prepared_section())
    }

    fn prepared_bank(&self) -> Option<usize> {
        self.inner.prepared_section().map(|(_, bank)| bank)
    }

    fn receiver_slot(&self) -> u64 {
        self.inner.receiver_slot()
    }

    fn output_slot(&self) -> Option<u64> {
        self.inner.output_slot()
    }

    fn tensor_layouts(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let mut cache = self
            .tensor_projection
            .lock()
            .map_err(|_| invalid("source occurrence tensor projection custody is poisoned"))?;
        if let Some(original) = &*cache {
            return Ok(original.value.clone_ref(py));
        }
        let count = self.inner.tensor_layouts().len();
        let mut row_extent = python_metadata_tuple_extent(py, 8)?;
        row_extent = row_extent
            .checked_add(2 * python_metadata_tuple_extent(py, 4)?)
            .and_then(|n| n.checked_add(python_metadata_tuple_extent(py, 3).ok()?))
            .and_then(|n| n.checked_add(python_metadata_tuple_extent(py, 7).ok()?))
            .and_then(|n| n.checked_add(22 * source_content_geometry::integer_extent(py).ok()?))
            .and_then(|n| n.checked_add(std::mem::size_of::<Py<PyTuple>>()))
            .ok_or_else(|| invalid("source occurrence tensor projection extent overflow"))?;
        let extent = count
            .checked_mul(row_extent)
            .and_then(|n| n.checked_add(python_metadata_tuple_extent(py, count).ok()?))
            .ok_or_else(|| invalid("source occurrence tensor projection extent overflow"))?;
        let metadata = self
            .budget
            .reserve_auxiliary(
                u64::try_from(extent)
                    .map_err(|_| invalid("source occurrence tensor projection exceeds u64"))?,
            )
            .map_err(xlog_err)?;
        let mut rows = Vec::with_capacity(count);
        for tensor in self.inner.tensor_layouts() {
            let metadata = tensor_metadata_value(
                py,
                (tensor.layout, tensor.logical_begin, tensor.logical_end),
            )?;
            rows.push(
                (
                    metadata.bind(py).get_item(0)?,
                    tensor.logical_begin,
                    tensor.logical_end,
                    tensor.storage_slot,
                    tensor.version_slot,
                    tensor.allocation_slot,
                    tensor.byte_offset,
                )
                    .into_pyobject(py)?
                    .unbind(),
            );
        }
        let value = PyTuple::new(py, rows)?.unbind();
        *cache = Some(RetainedOccurrenceTuple {
            value: value.clone_ref(py),
            _metadata: metadata,
        });
        Ok(value)
    }

    fn allocation_layouts(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let mut cache = self
            .allocation_projection
            .lock()
            .map_err(|_| invalid("source occurrence allocation projection custody is poisoned"))?;
        if let Some(original) = &*cache {
            return Ok(original.value.clone_ref(py));
        }
        let count = self.inner.allocation_layouts().len();
        let extent = count
            .checked_mul(
                python_metadata_tuple_extent(py, 2)?
                    .checked_add(2 * source_content_geometry::integer_extent(py)?)
                    .ok_or_else(|| {
                        invalid("source occurrence allocation projection extent overflow")
                    })?,
            )
            .and_then(|n| n.checked_add(python_metadata_tuple_extent(py, count).ok()?))
            .ok_or_else(|| invalid("source occurrence allocation projection extent overflow"))?;
        let metadata = self
            .budget
            .reserve_auxiliary(
                u64::try_from(extent)
                    .map_err(|_| invalid("source occurrence allocation projection exceeds u64"))?,
            )
            .map_err(xlog_err)?;
        let value = PyTuple::new(py, self.inner.allocation_layouts().iter().copied())?.unbind();
        *cache = Some(RetainedOccurrenceTuple {
            value: value.clone_ref(py),
            _metadata: metadata,
        });
        Ok(value)
    }
}

pub(super) fn exact_tuple<'py>(
    value: &Bound<'py, PyAny>,
    arity: Option<usize>,
    budget: &mut usize,
) -> PyResult<Bound<'py, PyTuple>> {
    if !value.is_exact_instance_of::<PyTuple>() {
        return Err(invalid(
            "source content metadata requires exact immutable tuples",
        ));
    }
    let tuple = value.cast::<PyTuple>()?;
    if arity.is_some_and(|expected| tuple.len() != expected) {
        return Err(invalid(
            "source content metadata has an incorrect tuple length",
        ));
    }
    *budget = budget
        .checked_sub(tuple.len().saturating_add(1))
        .ok_or_else(|| invalid("source content metadata exceeds its transport bound"))?;
    Ok(tuple.clone())
}

fn unsigned(value: &Bound<'_, PyAny>) -> PyResult<u64> {
    if !value.is_exact_instance_of::<PyInt>() {
        return Err(invalid(
            "source content coordinates require exact unsigned integers",
        ));
    }
    value.extract()
}

fn optional_unsigned(value: &Bound<'_, PyAny>) -> PyResult<Option<u64>> {
    if value.is_none() {
        Ok(None)
    } else {
        unsigned(value).map(Some)
    }
}

fn unsigned_array_from_tuple(value: &Bound<'_, PyAny>, budget: &mut usize) -> PyResult<[u64; 4]> {
    let tuple = exact_tuple(value, Some(4), budget)?;
    let mut values = [0; 4];
    for (index, value) in values.iter_mut().enumerate() {
        *value = unsigned(&tuple.get_item(index)?)?;
    }
    Ok(values)
}

pub(super) fn parse_tensor_descriptors(
    value: &Bound<'_, PyAny>,
    budget: &mut usize,
) -> PyResult<Vec<SemanticSourceTensorDescriptor>> {
    let rows = exact_tuple(value, None, budget)?;
    let mut tensors = Vec::with_capacity(rows.len());
    for tensor in rows.iter() {
        let tensor = exact_tuple(&tensor, Some(7), budget)?;
        let layout = exact_tuple(&tensor.get_item(0)?, Some(8), budget)?;
        let mut fields = [0; 6];
        for (index, value) in fields.iter_mut().enumerate() {
            *value = unsigned(&layout.get_item(index)?)?;
        }
        tensors.push(SemanticSourceTensorDescriptor {
            layout: tensor_layout_from_values(
                fields,
                unsigned_array_from_tuple(&layout.get_item(6)?, budget)?,
                unsigned_array_from_tuple(&layout.get_item(7)?, budget)?,
            ),
            logical_begin: unsigned(&tensor.get_item(1)?)?,
            logical_end: unsigned(&tensor.get_item(2)?)?,
            storage_slot: unsigned(&tensor.get_item(3)?)?,
            version_slot: optional_unsigned(&tensor.get_item(4)?)?,
            allocation_slot: unsigned(&tensor.get_item(5)?)?,
            byte_offset: unsigned(&tensor.get_item(6)?)?,
        });
    }
    Ok(tensors)
}

pub(super) fn parse_allocation_descriptors(
    value: &Bound<'_, PyAny>,
    budget: &mut usize,
) -> PyResult<Vec<SemanticSourceAllocationDescriptor<SemanticSourceContentAllocation>>> {
    let rows = exact_tuple(value, None, budget)?;
    let mut allocations = Vec::with_capacity(rows.len());
    for allocation in rows.iter() {
        let allocation = exact_tuple(&allocation, Some(3), budget)?;
        let provenance = allocation.get_item(2)?;
        allocations.push(SemanticSourceAllocationDescriptor {
            slot: unsigned(&allocation.get_item(0)?)?,
            span_bytes: unsigned(&allocation.get_item(1)?)?,
            native_allocation: if provenance.is_none() {
                None
            } else if provenance
                .is_instance_of::<source_content_geometry::PySemanticSourceGeometryAllocation>()
            {
                Some(SemanticSourceContentAllocation::Geometry(
                    provenance
                        .extract::<PyRef<
                            '_,
                            source_content_geometry::PySemanticSourceGeometryAllocation,
                        >>()?
                        .inner
                        .clone(),
                ))
            } else {
                Some(SemanticSourceContentAllocation::Native(
                    provenance
                        .extract::<PyRef<'_, PyNativeTensorAllocation>>()?
                        .provenance
                        .clone(),
                ))
            },
        });
    }
    Ok(allocations)
}

pub(super) fn parse_events(
    events: &Bound<'_, PyAny>,
) -> PyResult<Vec<SemanticSourceContentEvent<SemanticSourceContentAllocation>>> {
    // Reuse the original cold-metadata transport bound; this is not a native
    // operation count, content price, remaining budget or numerical allowance.
    let mut budget = 16 * 1024 * 1024;
    let rows = exact_tuple(events, None, &mut budget)?;
    let mut result = Vec::with_capacity(rows.len());
    for row in rows.iter() {
        let row = exact_tuple(&row, Some(5), &mut budget)?;
        let family = row
            .get_item(0)?
            .extract::<PyRef<'_, PySemanticSourceContentFamily>>()?
            .inner
            .clone();
        let receiver = unsigned(&row.get_item(1)?)?;
        let output = optional_unsigned(&row.get_item(2)?)?;
        let tensors = parse_tensor_descriptors(&row.get_item(3)?, &mut budget)?;
        let allocations = parse_allocation_descriptors(&row.get_item(4)?, &mut budget)?;
        result.push(SemanticSourceContentEvent {
            family,
            receiver_slot: receiver,
            output_slot: output,
            tensors,
            allocations,
        });
    }
    Ok(result)
}

#[pymethods]
impl PySemanticPreparedSourceContent {
    fn content_family(
        &self,
        py: Python<'_>,
        phase: &Bound<'_, PyAny>,
        kind: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PySemanticSourceContentFamily>> {
        self.require_original(py)?;
        let (phase, kind) = family_arguments(phase, kind)?;
        let inner = {
            let (parent, recipe, _) = self.original_binding();
            let parent = parent.borrow(py);
            let session = parent.session.borrow(py);
            let owner = session.owner()?;
            let lease = parent.lease()?;
            owner
                .source_content_family(&lease, recipe, phase, kind)
                .map_err(xlog_err)?
        };
        family(py, inner, &self.material_budget(py)?)
    }

    #[pyo3(signature = (*, events))]
    fn freeze_content_occurrences(
        &self,
        py: Python<'_>,
        events: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        self.require_original(py)?;
        self.content_geometry
            .borrow(py)
            .issued_occurrences(py, events)
    }

    fn pack_policy_occurrences(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.phase_occurrences(py, SemanticSourceContentPhase::PackPolicy)
    }

    fn finalize_continuation_occurrences(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        self.phase_occurrences(py, SemanticSourceContentPhase::FinalizeContinuation)
    }
}

impl PySemanticPreparedSourceContent {
    fn phase_occurrences(
        &self,
        py: Python<'_>,
        phase: SemanticSourceContentPhase,
    ) -> PyResult<Py<PyTuple>> {
        self.require_original(py)?;
        self.content_geometry
            .borrow(py)
            .phase_occurrences(py, phase)
    }
}
