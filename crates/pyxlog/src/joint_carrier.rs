// PyO3 binding for the XLOG-owned joint constraint carrier.
//
// The carrier owns every solver buffer on device; Python consumers
// receive zero-copy DLPack views of the exact producer allocations
// (ownership stays with xlog) and drive the cold-path session
// lifecycle: allocate -> register_schema -> bind_signatures ->
// solve_label_feasibility. The fuel meter lives inside the session,
// so exhaustion saturates across Python calls — a retry repeats the
// identical typed refusal instead of slipping work past the budget.

use std::sync::Mutex;

use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use xlog_cuda::dlpack::{
    export_slice_managed_tensor, DLDataType, K_DLFLOAT, K_DLUINT,
};
use xlog_cuda::{
    CarrierBufferId, CarrierError, CudaDevice, FuelMeter,
    JointConstraintCarrier as NativeCarrier, SolverError,
};

create_exception!(
    pyxlog,
    CarrierRefused,
    PyRuntimeError,
    "Typed refusal from the joint constraint carrier; the message names \
     the exact refused precondition and its literals."
);

create_exception!(
    pyxlog,
    SolverResourceExhausted,
    PyRuntimeError,
    "The device fuel budget is exhausted; the refusal carries the exact \
     spent/limit literals and repeats identically on retry."
);

fn carrier_err(err: CarrierError) -> PyErr {
    match err {
        CarrierError::Solver(SolverError::ResourceExhausted {
            fuel_spent,
            fuel_limit,
        }) => SolverResourceExhausted::new_err(format!(
            "solver fuel exhausted: spent {fuel_spent} of {fuel_limit} \
             node expansions"
        )),
        other => CarrierRefused::new_err(other.to_string()),
    }
}

struct CarrierState {
    carrier: NativeCarrier,
    fuel: FuelMeter,
    device_id: i32,
}

/// Session handle over the device-resident joint constraint carrier.
#[pyclass(unsendable)]
pub struct JointConstraintCarrier {
    state: Mutex<CarrierState>,
}

#[pymethods]
impl JointConstraintCarrier {
    #[new]
    fn new(
        device: usize,
        entities: usize,
        domain_lanes: usize,
        candidates: usize,
        labels: usize,
        fuel_limit: u64,
    ) -> PyResult<Self> {
        let cuda = CudaDevice::new(device).map_err(|e| {
            PyRuntimeError::new_err(format!("CUDA device {device}: {e}"))
        })?;
        let carrier = NativeCarrier::allocate(
            std::sync::Arc::new(cuda),
            entities,
            domain_lanes,
            candidates,
            labels,
        )
        .map_err(carrier_err)?;
        Ok(Self {
            state: Mutex::new(CarrierState {
                carrier,
                fuel: FuelMeter::new(fuel_limit),
                device_id: device as i32,
            }),
        })
    }

    fn register_schema(
        &self,
        catalog_sha: &str,
        solver_identity: &str,
    ) -> PyResult<()> {
        let mut state = self.state.lock().expect("carrier state lock");
        state
            .carrier
            .register_schema(catalog_sha, solver_identity)
            .map_err(carrier_err)
    }

    fn bind_signatures(
        &self,
        head_masks: Vec<u64>,
        tail_masks: Vec<u64>,
    ) -> PyResult<()> {
        let mut state = self.state.lock().expect("carrier state lock");
        state
            .carrier
            .bind_signatures(&head_masks, &tail_masks)
            .map_err(carrier_err)
    }

    /// Export one carrier buffer as a zero-copy DLPack capsule
    /// (`torch.utils.dlpack.from_dlpack` consumes it directly). The
    /// view shares the exact solve allocation; xlog stays the owner.
    ///
    /// Stream-ordering obligation: writes made through the exported
    /// view happen on the consumer's stream. Either call
    /// `note_producer_stream` with that stream's handle after the
    /// writes (the solve then waits on device — the measured-region
    /// path), or synchronize on the host before solving.
    fn export_buffer(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        let (id, dtype) = match name {
            "domains" => (
                CarrierBufferId::Domains,
                DLDataType { code: K_DLUINT, bits: 64, lanes: 1 },
            ),
            "scores" => (
                CarrierBufferId::Scores,
                DLDataType { code: K_DLFLOAT, bits: 32, lanes: 1 },
            ),
            "constraints" => (
                CarrierBufferId::Constraints,
                DLDataType { code: K_DLUINT, bits: 32, lanes: 1 },
            ),
            "outputs" => (
                CarrierBufferId::Outputs,
                DLDataType { code: K_DLUINT, bits: 32, lanes: 1 },
            ),
            "feasible_sets" => (
                CarrierBufferId::FeasibleSets,
                DLDataType { code: K_DLUINT, bits: 64, lanes: 1 },
            ),
            "logical_counts" => (
                CarrierBufferId::LogicalCounts,
                DLDataType { code: K_DLUINT, bits: 32, lanes: 1 },
            ),
            "map_results" => (
                CarrierBufferId::MapResults,
                DLDataType { code: K_DLUINT, bits: 32, lanes: 1 },
            ),
            "solve_status" => (
                CarrierBufferId::SolveStatus,
                DLDataType { code: K_DLUINT, bits: 32, lanes: 1 },
            ),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown carrier buffer {other:?}; valid names: domains, \
                     scores, constraints, outputs, feasible_sets, \
                     logical_counts, map_results, solve_status"
                )))
            }
        };
        let state = self.state.lock().expect("carrier state lock");
        let export = state.carrier.export_buffer(id);
        let tensor = export_slice_managed_tensor(
            export.slice,
            state.device_id,
            dtype,
            export.rows,
            export.cols,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        crate::dlpack_capsule_from_tensor(py, tensor)
    }

    /// Launch the on-device feasibility solve. Orders against
    /// runtime-recorded events plus every producer event noted via
    /// `note_producer_stream`; producer writes not covered by a noted
    /// event or a host sync race the launch.
    fn solve_label_feasibility(&self, abstain_label: u32) -> PyResult<()> {
        let mut state = self.state.lock().expect("carrier state lock");
        let state = &mut *state;
        state
            .carrier
            .solve_label_feasibility(abstain_label, &mut state.fuel)
            .map_err(carrier_err)
    }

    /// Launch the per-candidate exact top-two stage, consuming the
    /// feasibility stage's feasible sets on device. Refuses typed
    /// before feasibility has run. The same caller synchronization
    /// obligation applies between stages: device writes through
    /// exported views (domains, scores) after the feasibility solve
    /// must be synchronized before this call. Results in
    /// `map_results` are the global max-marginal ONLY for
    /// single-candidate components; a set ambiguity flag must never
    /// emit as a unique MAP label.
    fn solve_label_map_top2(&self) -> PyResult<()> {
        let mut state = self.state.lock().expect("carrier state lock");
        let state = &mut *state;
        state
            .carrier
            .solve_label_map_top2(&mut state.fuel)
            .map_err(carrier_err)
    }

    /// Solve every listed multi-candidate component exactly by
    /// complete enumeration. The plan (offsets + candidate indices)
    /// comes from the caller's own pair list; malformed plans refuse
    /// typed. Per-row authority afterwards is `solve_status`:
    /// 2 = component-exact, 3 = refused (stage capacity/fuel — the
    /// memoized stage is the named open cell), 0xFFFFFFFF = poisoned;
    /// untouched rows keep top-two authority.
    fn solve_components_exact(
        &self,
        comp_offsets: Vec<u32>,
        comp_indices: Vec<u32>,
    ) -> PyResult<()> {
        let mut state = self.state.lock().expect("carrier state lock");
        let state = &mut *state;
        state
            .carrier
            .solve_components_exact(&comp_offsets, &comp_indices, &mut state.fuel)
            .map_err(carrier_err)
    }

    /// Record a producer-completion event on an external CUDA stream.
    /// The next solve stage waits on every noted event on device
    /// before launching, replacing any host synchronization between
    /// producer writes through exported views and the solve. Use a
    /// dedicated stream (`torch.cuda.Stream()`, pass its
    /// `cuda_stream`): torch's default stream carries raw handle 0,
    /// which is indistinguishable from a null handle and is refused.
    fn note_producer_stream(&self, external_stream: u64) -> PyResult<()> {
        let mut state = self.state.lock().expect("carrier state lock");
        state
            .carrier
            .note_producer_stream(external_stream)
            .map_err(carrier_err)
    }

    #[getter]
    fn fuel_spent(&self) -> u64 {
        self.state.lock().expect("carrier state lock").fuel.spent()
    }
}
