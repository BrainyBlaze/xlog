use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySequence};

use xlog_cuda::DlpackManagedTensor;
use xlog_gpu::logic as gpu_logic;
use xlog_logic::ast::ProbEngine;
use xlog_neural::{NetworkRegistry, TensorSourceRegistry};
use xlog_prob::exact::{ExactDdnnfProgram, GpuConfig};
use xlog_prob::mc::McProgram;
use xlog_runtime::RelationDelta;

use std::collections::HashMap as StdHashMap;

use super::neural_registry::NeuralPredicateRegistry;
use super::{
    dlpack_capsule_from_tensor, dlpack_from_py, enforce_call_memory_limit,
    parse_prob_engine_override, provider_from_config, provider_memory_stats, types,
    CompiledLogicProgram, CompiledProbProgram, CompiledProgram, LogicDeltaStats, LogicEvalResult,
    LogicProgram, LogicQueryResult, LogicRelationSession, Program, RelationChangeCallback,
};

#[pymethods]
impl Program {
    #[staticmethod]
    #[pyo3(signature = (source, device=0, memory_mb=32768, prob_engine=None))]
    pub fn compile(
        source: &str,
        device: usize,
        memory_mb: u64,
        prob_engine: Option<String>,
    ) -> PyResult<CompiledProgram> {
        if memory_mb == 0 {
            return Err(PyValueError::new_err("memory_mb must be > 0"));
        }

        let mut config = GpuConfig::default();
        config.device_ordinal = device;
        config.memory_bytes = memory_mb * 1024 * 1024;

        // Parse the AST to get prob_engine and neural predicates
        let ast = xlog_logic::parse_program(source).map_err(types::xlog_err)?;

        // Extract declared neural network names
        let declared_networks: HashSet<String> = ast
            .neural_predicates
            .iter()
            .map(|np| np.network.clone())
            .collect();
        // Build by-network form index: network name -> is_embedding
        let mut declared_network_forms: HashMap<String, bool> = HashMap::new();
        for np in &ast.neural_predicates {
            let is_embedding = np.labels.is_none();
            match declared_network_forms.get(&np.network) {
                Some(&existing_form) if existing_form != is_embedding => {
                    return Err(PyValueError::new_err(format!(
                        "network '{}' is declared as both classification and embedding; \
                         each network name must have a single form",
                        np.network
                    )));
                }
                _ => {
                    declared_network_forms.insert(np.network.clone(), is_embedding);
                }
            }
        }

        let neural_registry = NeuralPredicateRegistry::from_ast(&ast).map_err(types::val_err)?;

        let engine = match prob_engine {
            Some(s) => parse_prob_engine_override(&s)?,
            None => ast.prob_engine(),
        };

        let program = match engine {
            ProbEngine::ExactDdnnf => CompiledProbProgram::Exact(
                ExactDdnnfProgram::compile_source_with_gpu(source, config)
                    .map_err(types::xlog_err)?,
            ),
            ProbEngine::Mc => CompiledProbProgram::Mc(
                McProgram::compile_source_with_gpu(source, config).map_err(types::xlog_err)?,
            ),
        };
        let provider = provider_from_config(config).map_err(types::xlog_err)?;

        Ok(CompiledProgram {
            program,
            output_provider: Arc::new(provider),
            network_registry: NetworkRegistry::new(),
            neural_registry,
            declared_networks,
            declared_network_forms,
            tensor_sources: TensorSourceRegistry::new(),
            _source: source.to_string(),
            ast,
            _gpu_config: config,
            _prob_engine: engine,
            query_signature_cache: StdHashMap::new(),
            circuit_cache: StdHashMap::new(),
            circuit_cache_hits: 0,
            circuit_cache_misses: 0,
            template_compile_count: 0,
            batch_queries: true,
            last_compile_profile: None,
        })
    }
}

#[pymethods]
impl LogicProgram {
    #[staticmethod]
    #[pyo3(signature = (source, device=0, memory_mb=32768))]
    pub fn compile(source: &str, device: usize, memory_mb: u64) -> PyResult<CompiledLogicProgram> {
        if memory_mb == 0 {
            return Err(PyValueError::new_err("memory_mb must be > 0"));
        }

        let mut config = GpuConfig::default();
        config.device_ordinal = device;
        config.memory_bytes = memory_mb * 1024 * 1024;

        let program = gpu_logic::LogicProgram::compile(source).map_err(types::xlog_err)?;
        let provider = provider_from_config(config).map_err(types::xlog_err)?;

        Ok(CompiledLogicProgram {
            program,
            provider: Arc::new(provider),
        })
    }
}

#[pymethods]
impl CompiledLogicProgram {
    #[pyo3(signature = (dlpack_inputs=None, memory_mb=None))]
    pub fn evaluate(
        &self,
        py: Python<'_>,
        dlpack_inputs: Option<&Bound<'_, PyDict>>,
        memory_mb: Option<u64>,
    ) -> PyResult<LogicEvalResult> {
        enforce_call_memory_limit(&self.provider, memory_mb)?;
        let mut inputs: HashMap<String, xlog_cuda::CudaBuffer> = HashMap::new();

        if let Some(dict) = dlpack_inputs {
            for (k, v) in dict.iter() {
                let name: String = k.extract()?;
                let schema = self.program.schema(&name).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Unknown input relation {} (not present in compiled schemas)",
                        name
                    ))
                })?;

                let tensors = collect_dlpack_columns(
                    &v,
                    &format!(
                        "Input relation {} must be a sequence of DLPack columns",
                        name
                    ),
                )?;

                let buffer = self
                    .provider
                    .from_dlpack_tensors_with_schema(schema.clone(), tensors)
                    .map_err(types::xlog_err)?;

                inputs.insert(name, buffer);
            }
        }

        let result = self
            .program
            .evaluate(self.provider.clone(), inputs)
            .map_err(types::xlog_err)?;
        pack_logic_result_with_provider(py, &self.provider, result)
    }

    pub fn session(&self) -> PyResult<LogicRelationSession> {
        let relation_store = self
            .program
            .create_relation_store(self.provider.clone())
            .map_err(types::xlog_err)?;
        Ok(LogicRelationSession {
            program: self.program.clone(),
            provider: self.provider.clone(),
            relation_store,
            evaluation_store: None,
            session_runtime: None,
            last_delta_stats: None,
            relation_callbacks: Vec::new(),
            next_relation_callback_id: 1,
            relation_generations: HashMap::new(),
        })
    }

    /// Return memory diagnostics including allocated_bytes and memory_limit_bytes.
    pub fn memory_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        provider_memory_stats(py, &self.provider)
    }

    pub fn rule_provenance(&self, py: Python<'_>) -> PyResult<PyObject> {
        pack_rule_provenance(py, &self.program.rule_provenance())
    }

    pub fn proof_traces(&self, py: Python<'_>) -> PyResult<PyObject> {
        pack_proof_traces(py, &self.program.proof_traces())
    }
}

impl CompiledLogicProgram {}

#[pymethods]
impl LogicRelationSession {
    pub fn put_relation(
        &mut self,
        name: String,
        dlpack_columns: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if name.starts_with("__") {
            return Err(PyValueError::new_err(format!(
                "Relation {} is internal and cannot be stored in a persistent session",
                name
            )));
        }
        let schema = self.program.schema(&name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown relation {} (not present in compiled schemas)",
                name
            ))
        })?;
        let tensors = collect_dlpack_columns(
            dlpack_columns,
            &format!("Relation {} must be a sequence of DLPack columns", name),
        )?;
        let buffer = self
            .provider
            .from_dlpack_tensors_with_schema(schema.clone(), tensors)
            .map_err(types::xlog_err)?;
        self.relation_store.put(&name, buffer);
        self.evaluation_store = None;
        self.session_runtime = None;
        self.last_delta_stats = None;
        Ok(())
    }

    #[pyo3(signature = (memory_mb=None))]
    pub fn evaluate(
        &mut self,
        py: Python<'_>,
        memory_mb: Option<u64>,
    ) -> PyResult<LogicEvalResult> {
        enforce_call_memory_limit(&self.provider, memory_mb)?;
        let result = if let Some(store) = &self.evaluation_store {
            self.program
                .evaluate_cached_relation_store(self.provider.clone(), store)
                .map_err(types::xlog_err)?
        } else {
            if self.session_runtime.is_none() {
                self.session_runtime = Some(
                    self.program
                        .create_session_runtime(self.provider.clone(), &self.relation_store, false)
                        .map_err(types::xlog_err)?,
                );
            }
            let runtime = self.session_runtime.as_mut().ok_or_else(|| {
                PyRuntimeError::new_err("session runtime unavailable during evaluation")
            })?;
            let (result, store) = self
                .program
                .evaluate_with_session_runtime(self.provider.clone(), runtime)
                .map_err(types::xlog_err)?;
            self.evaluation_store = Some(store);
            result
        };
        pack_logic_result_with_provider(py, &self.provider, result)
    }

    pub fn insert_relation(
        &mut self,
        py: Python<'_>,
        name: String,
        dlpack_columns: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let insert = self.relation_delta_buffer(&name, dlpack_columns)?;
        self.apply_single_relation_delta(py, name, Some(insert), None)
    }

    pub fn delete_relation(
        &mut self,
        py: Python<'_>,
        name: String,
        dlpack_columns: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let delete = self.relation_delta_buffer(&name, dlpack_columns)?;
        self.apply_single_relation_delta(py, name, None, Some(delete))
    }

    #[pyo3(signature = (name, insert_columns=None, delete_columns=None))]
    pub fn apply_relation_delta(
        &mut self,
        py: Python<'_>,
        name: String,
        insert_columns: Option<&Bound<'_, PyAny>>,
        delete_columns: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        if insert_columns.is_none() && delete_columns.is_none() {
            return Err(PyValueError::new_err(
                "apply_relation_delta requires insert_columns, delete_columns, or both",
            ));
        }
        let insert = insert_columns
            .map(|columns| self.relation_delta_buffer(&name, columns))
            .transpose()?;
        let delete = delete_columns
            .map(|columns| self.relation_delta_buffer(&name, columns))
            .transpose()?;
        self.apply_single_relation_delta(py, name, insert, delete)
    }

    pub fn apply_relation_delta_batch(
        &mut self,
        py: Python<'_>,
        updates: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let (batch, relation_names) =
            self.parse_relation_delta_batch("apply_relation_delta_batch", updates)?;

        let report = self
            .program
            .apply_relation_delta_batch_with_session_runtime(
                self.provider.clone(),
                &mut self.relation_store,
                &mut self.evaluation_store,
                &mut self.session_runtime,
                batch,
            )
            .map_err(types::xlog_err)?;
        let stats = logic_delta_stats_from_report(report);
        self.last_delta_stats = Some(stats.clone());
        self.fire_relation_callbacks(py, &relation_names, &stats)?;
        pack_delta_stats(py, &stats)
    }

    #[pyo3(signature = (updates, check_equivalence=false))]
    pub fn apply_relation_delta_debug(
        &mut self,
        py: Python<'_>,
        updates: &Bound<'_, PyAny>,
        check_equivalence: bool,
    ) -> PyResult<PyObject> {
        let (batch, relation_names) =
            self.parse_relation_delta_batch("apply_relation_delta_debug", updates)?;
        let delta_start = Instant::now();
        let report = self
            .program
            .apply_relation_delta_batch_with_session_runtime(
                self.provider.clone(),
                &mut self.relation_store,
                &mut self.evaluation_store,
                &mut self.session_runtime,
                batch,
            )
            .map_err(types::xlog_err)?;
        let delta_micros = delta_start.elapsed().as_micros().max(1) as u64;
        let mut stats = logic_delta_stats_from_report(report);
        if check_equivalence {
            let full_start = Instant::now();
            let (_, full_store) = self
                .program
                .evaluate_with_relation_store_and_cache(
                    self.provider.clone(),
                    &self.relation_store,
                    false,
                )
                .map_err(types::xlog_err)?;
            let full_micros = full_start.elapsed().as_micros() as u64;
            let cached_store = self.evaluation_store.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("delta debug missing cached store after delta application")
            })?;
            stats.equivalent_to_full_recompute = Some(
                self.program
                    .relation_stores_query_equivalent(
                        self.provider.as_ref(),
                        &full_store,
                        cached_store,
                    )
                    .map_err(types::xlog_err)?,
            );
            let speedup = full_micros as f64 / delta_micros as f64;
            stats.planner_telemetry.measured_delta_speedup = Some(speedup);
            if speedup >= 1.0 {
                stats
                    .planner_telemetry
                    .planner_advice
                    .push(format!("delta path is faster by {speedup:.2}x"));
            } else {
                stats.planner_telemetry.planner_advice.push(format!(
                    "full recompute may be faster; delta measured {speedup:.2}x"
                ));
            }
        }
        self.last_delta_stats = Some(stats.clone());
        self.fire_relation_callbacks(py, &relation_names, &stats)?;
        pack_delta_stats(py, &stats)
    }

    pub fn delta_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.last_delta_stats {
            Some(stats) => pack_delta_stats(py, stats),
            None => {
                let dict = PyDict::new(py);
                dict.set_item("status", "unavailable")?;
                dict.set_item("reason", "no relation delta has been applied")?;
                Ok(dict.into())
            }
        }
    }

    pub fn rule_provenance(&self, py: Python<'_>) -> PyResult<PyObject> {
        pack_rule_provenance(py, &self.program.rule_provenance())
    }

    pub fn proof_traces(&self, py: Python<'_>) -> PyResult<PyObject> {
        pack_proof_traces(py, &self.program.proof_traces())
    }

    pub fn register_relation_callback(
        &mut self,
        py: Python<'_>,
        callback: PyObject,
    ) -> PyResult<u64> {
        if !callback.bind(py).is_callable() {
            return Err(PyValueError::new_err(
                "register_relation_callback expects a callable",
            ));
        }
        let id = self.next_relation_callback_id;
        self.next_relation_callback_id = self.next_relation_callback_id.saturating_add(1);
        self.relation_callbacks
            .push(RelationChangeCallback { id, callback });
        Ok(id)
    }

    pub fn unregister_relation_callback(&mut self, callback_id: u64) -> bool {
        let before = self.relation_callbacks.len();
        self.relation_callbacks
            .retain(|registered| registered.id != callback_id);
        before != self.relation_callbacks.len()
    }

    pub fn cuda_graph_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item(
            "csm_cuda_graph_captures",
            self.provider.csm_cuda_graph_captures(),
        )?;
        dict.set_item(
            "csm_cuda_graph_launches",
            self.provider.csm_cuda_graph_launches(),
        )?;
        dict.set_item(
            "csm_cuda_graph_fallbacks",
            self.provider.csm_cuda_graph_fallbacks(),
        )?;
        dict.set_item(
            "csm_cuda_graph_cache_hits",
            self.provider.csm_cuda_graph_cache_hits(),
        )?;
        Ok(dict.into())
    }

    pub fn host_transfer_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = self.provider.host_transfer_stats();
        let dict = PyDict::new(py);
        dict.set_item("dtoh_bytes", stats.dtoh_bytes)?;
        dict.set_item("htod_bytes", stats.htod_bytes)?;
        dict.set_item("dtoh_calls", stats.dtoh_calls)?;
        dict.set_item("htod_calls", stats.htod_calls)?;
        Ok(dict.into())
    }

    pub fn join_index_cache_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let stats = self
            .session_runtime
            .as_ref()
            .map(|runtime| runtime.join_index_cache_stats())
            .unwrap_or_default();
        dict.set_item("lookups", stats.lookups)?;
        dict.set_item("hits", stats.hits)?;
        dict.set_item("misses", stats.misses)?;
        dict.set_item("builds", stats.builds)?;
        dict.set_item("evictions", stats.evictions)?;
        dict.set_item("invalidations", stats.invalidations)?;
        dict.set_item("stale_rejections", stats.stale_rejections)?;
        dict.set_item("background_build_requests", stats.background_build_requests)?;
        dict.set_item(
            "background_builds_completed",
            stats.background_builds_completed,
        )?;
        dict.set_item(
            "background_builds_deferred",
            stats.background_builds_deferred,
        )?;
        dict.set_item("entries", stats.entries)?;
        dict.set_item("total_bytes", stats.total_bytes)?;
        Ok(dict.into())
    }

    pub fn reset_host_transfer_stats(&self) {
        self.provider.reset_host_transfer_stats()
    }

    /// Return memory diagnostics including allocated_bytes and memory_limit_bytes.
    pub fn memory_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        provider_memory_stats(py, &self.provider)
    }

    pub fn export_relation(&mut self, py: Python<'_>, name: &str) -> PyResult<Vec<PyObject>> {
        let existing = self.relation_store.get(name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Relation '{}' not found in persistent session",
                name
            ))
        })?;
        let replacement = self
            .provider
            .clone_buffer(existing)
            .map_err(types::xlog_err)?;
        let buffer = self.relation_store.remove(name).ok_or_else(|| {
            PyRuntimeError::new_err(format!("Relation '{}' disappeared during export", name))
        })?;
        self.relation_store.put(name, replacement);
        export_buffer_columns(py, &self.provider, buffer)
    }

    pub fn remove_relation(&mut self, name: &str) -> bool {
        let removed = self.relation_store.remove(name).is_some();
        if removed {
            self.evaluation_store = None;
            self.session_runtime = None;
            self.last_delta_stats = None;
        }
        removed
    }

    pub fn clear_relations(&mut self) {
        self.relation_store.clear();
        self.evaluation_store = None;
        self.session_runtime = None;
        self.last_delta_stats = None;
    }
}

impl LogicRelationSession {
    fn parse_relation_delta_batch(
        &self,
        method_name: &str,
        updates: &Bound<'_, PyAny>,
    ) -> PyResult<(Vec<(String, RelationDelta)>, Vec<String>)> {
        let seq = updates.downcast::<PySequence>().map_err(|_| {
            PyValueError::new_err(format!(
                "{method_name} expects a sequence of update dictionaries"
            ))
        })?;
        let mut batch: Vec<(String, RelationDelta)> = Vec::with_capacity(seq.len()? as usize);
        let mut relation_names: Vec<String> = Vec::new();
        for item in seq.try_iter()? {
            let item = item?;
            let dict = item.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err(format!("{method_name} updates must be dictionaries"))
            })?;
            let name_obj = dict.get_item("name")?.ok_or_else(|| {
                PyValueError::new_err(format!("{method_name} update missing 'name'"))
            })?;
            let name: String = name_obj.extract()?;
            let insert = optional_delta_columns(dict, "insert_columns")
                .map(|columns| self.relation_delta_buffer(&name, &columns))
                .transpose()?;
            let delete = optional_delta_columns(dict, "delete_columns")
                .map(|columns| self.relation_delta_buffer(&name, &columns))
                .transpose()?;
            if insert.is_none() && delete.is_none() {
                return Err(PyValueError::new_err(format!(
                    "{method_name} updates require insert_columns, delete_columns, or both"
                )));
            }
            relation_names.push(name.clone());
            batch.push((name, RelationDelta::new(insert, delete)));
        }
        Ok((batch, relation_names))
    }

    fn relation_delta_buffer(
        &self,
        name: &str,
        dlpack_columns: &Bound<'_, PyAny>,
    ) -> PyResult<xlog_cuda::CudaBuffer> {
        if name.starts_with("__") {
            return Err(PyValueError::new_err(format!(
                "Relation {} is internal and cannot be updated in a persistent session",
                name
            )));
        }
        let schema = self.program.schema(name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown relation {} (not present in compiled schemas)",
                name
            ))
        })?;
        let tensors = collect_dlpack_columns(
            dlpack_columns,
            &format!(
                "Relation {} delta must be a sequence of DLPack columns",
                name
            ),
        )?;
        self.provider
            .from_dlpack_tensors_with_schema(schema.clone(), tensors)
            .map_err(types::xlog_err)
    }

    fn apply_single_relation_delta(
        &mut self,
        py: Python<'_>,
        name: String,
        insert: Option<xlog_cuda::CudaBuffer>,
        delete: Option<xlog_cuda::CudaBuffer>,
    ) -> PyResult<PyObject> {
        let relation_names = vec![name.clone()];
        let mut deltas = HashMap::new();
        deltas.insert(name, RelationDelta::new(insert, delete));
        let report = self
            .program
            .apply_relation_deltas_with_session_runtime(
                self.provider.clone(),
                &mut self.relation_store,
                &mut self.evaluation_store,
                &mut self.session_runtime,
                deltas,
            )
            .map_err(types::xlog_err)?;
        let stats = LogicDeltaStats {
            input_delta_count: report.input_delta_count,
            changed_relations: report.changed_relations,
            changed_relation_names: report.changed_relation_names,
            insert_rows: report.insert_rows,
            delete_rows: report.delete_rows,
            has_deletes: report.has_deletes,
            affected_sccs: report.affected_sccs,
            recomputed_sccs: report.recomputed_sccs,
            incremental_sccs: report.incremental_sccs,
            coalesced_insert_rows: report.coalesced_insert_rows,
            coalesced_delete_rows: report.coalesced_delete_rows,
            canceled_rows: report.canceled_rows,
            equivalent_to_full_recompute: None,
            planner_telemetry: report.planner_telemetry,
            debug_trace: report.debug_trace,
        };
        self.last_delta_stats = Some(stats.clone());
        self.fire_relation_callbacks(py, &relation_names, &stats)?;
        pack_delta_stats(py, &stats)
    }

    fn fire_relation_callbacks(
        &mut self,
        py: Python<'_>,
        relation_names: &[String],
        stats: &LogicDeltaStats,
    ) -> PyResult<()> {
        if self.relation_callbacks.is_empty() || stats.changed_relations == 0 {
            return Ok(());
        }

        let mut seen = HashSet::new();
        let mut events: Vec<(String, u64)> = Vec::new();
        for relation in relation_names {
            if seen.insert(relation.clone()) {
                let generation = self
                    .relation_generations
                    .entry(relation.clone())
                    .and_modify(|current| *current = current.saturating_add(1))
                    .or_insert(1);
                events.push((relation.clone(), *generation));
            }
        }

        for (relation, generation) in events {
            let payload = relation_callback_payload(py, &relation, generation, stats)?;
            for registered in &self.relation_callbacks {
                registered.callback.call1(py, (payload.clone_ref(py),))?;
            }
        }

        Ok(())
    }
}

fn relation_callback_payload(
    py: Python<'_>,
    relation: &str,
    generation: u64,
    stats: &LogicDeltaStats,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("relation", relation)?;
    dict.set_item("generation", generation)?;
    dict.set_item("input_delta_count", stats.input_delta_count)?;
    dict.set_item(
        "changed_relation_names",
        stats.changed_relation_names.clone(),
    )?;
    dict.set_item("insert_rows", stats.insert_rows)?;
    dict.set_item("delete_rows", stats.delete_rows)?;
    dict.set_item("has_deletes", stats.has_deletes)?;
    dict.set_item("coalesced_insert_rows", stats.coalesced_insert_rows)?;
    dict.set_item("coalesced_delete_rows", stats.coalesced_delete_rows)?;
    dict.set_item("canceled_rows", stats.canceled_rows)?;
    dict.set_item("affected_sccs", stats.affected_sccs)?;
    dict.set_item("recomputed_sccs", stats.recomputed_sccs)?;
    dict.set_item("incremental_sccs", stats.incremental_sccs)?;
    dict.set_item("debug_trace", stats.debug_trace.clone())?;
    dict.set_item("telemetry", pack_delta_stats(py, stats)?)?;
    Ok(dict.into())
}

fn optional_delta_columns<'py>(dict: &Bound<'py, PyDict>, key: &str) -> Option<Bound<'py, PyAny>> {
    match dict.get_item(key) {
        Ok(Some(value)) if !value.is_none() => Some(value),
        _ => None,
    }
}

fn logic_delta_stats_from_report(report: gpu_logic::LogicDeltaReport) -> LogicDeltaStats {
    LogicDeltaStats {
        input_delta_count: report.input_delta_count,
        changed_relations: report.changed_relations,
        changed_relation_names: report.changed_relation_names,
        insert_rows: report.insert_rows,
        delete_rows: report.delete_rows,
        has_deletes: report.has_deletes,
        affected_sccs: report.affected_sccs,
        recomputed_sccs: report.recomputed_sccs,
        incremental_sccs: report.incremental_sccs,
        coalesced_insert_rows: report.coalesced_insert_rows,
        coalesced_delete_rows: report.coalesced_delete_rows,
        canceled_rows: report.canceled_rows,
        equivalent_to_full_recompute: None,
        planner_telemetry: report.planner_telemetry,
        debug_trace: report.debug_trace,
    }
}

fn pack_delta_stats(py: Python<'_>, stats: &LogicDeltaStats) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("status", "ok")?;
    dict.set_item("input_delta_count", stats.input_delta_count)?;
    dict.set_item("changed_relations", stats.changed_relations)?;
    dict.set_item(
        "changed_relation_names",
        stats.changed_relation_names.clone(),
    )?;
    dict.set_item("insert_rows", stats.insert_rows)?;
    dict.set_item("delete_rows", stats.delete_rows)?;
    dict.set_item("has_deletes", stats.has_deletes)?;
    dict.set_item("affected_sccs", stats.affected_sccs)?;
    dict.set_item("recomputed_sccs", stats.recomputed_sccs)?;
    dict.set_item("incremental_sccs", stats.incremental_sccs)?;
    dict.set_item("coalesced_insert_rows", stats.coalesced_insert_rows)?;
    dict.set_item("coalesced_delete_rows", stats.coalesced_delete_rows)?;
    dict.set_item("canceled_rows", stats.canceled_rows)?;
    dict.set_item(
        "equivalent_to_full_recompute",
        stats.equivalent_to_full_recompute,
    )?;
    dict.set_item(
        "planner_telemetry",
        pack_delta_planner_telemetry(py, &stats.planner_telemetry)?,
    )?;
    dict.set_item("debug_trace", stats.debug_trace.clone())?;
    Ok(dict.into())
}

fn pack_delta_planner_telemetry(
    py: Python<'_>,
    telemetry: &gpu_logic::DeltaPlannerTelemetry,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("cache_reused", telemetry.cache_reused)?;
    dict.set_item("fallback_decision", telemetry.fallback_decision.clone())?;
    dict.set_item("affected_sccs", telemetry.affected_sccs)?;
    dict.set_item("recomputed_sccs", telemetry.recomputed_sccs)?;
    dict.set_item("incremental_sccs", telemetry.incremental_sccs)?;
    dict.set_item("estimated_delta_speedup", telemetry.estimated_delta_speedup)?;
    dict.set_item("measured_delta_speedup", telemetry.measured_delta_speedup)?;
    dict.set_item("planner_advice", telemetry.planner_advice.clone())?;
    Ok(dict.into())
}

fn pack_rule_provenance(
    py: Python<'_>,
    entries: &[xlog_logic::RuleProvenance],
) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for entry in entries {
        let dict = PyDict::new(py);
        dict.set_item("rule_id", &entry.rule_id)?;
        dict.set_item("head", &entry.head)?;
        dict.set_item("source_kind", entry.source_kind.as_str())?;
        dict.set_item("source_span", entry.source_span.clone())?;
        dict.set_item("generation_trace_hash", entry.generation_trace_hash.clone())?;
        dict.set_item("support_relation_ids", entry.support_relation_ids.clone())?;
        dict.set_item(
            "counterexample_relation_ids",
            entry.counterexample_relation_ids.clone(),
        )?;
        list.append(dict)?;
    }
    Ok(list.into())
}

fn pack_proof_traces(
    py: Python<'_>,
    entries: &[xlog_logic::QueryProofTrace],
) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for entry in entries {
        let dict = PyDict::new(py);
        dict.set_item("query_id", &entry.query_id)?;
        dict.set_item("query", &entry.query)?;
        dict.set_item("answer_relation", &entry.answer_relation)?;
        dict.set_item("rule_ids", entry.rule_ids.clone())?;
        dict.set_item("source_facts", entry.source_facts.clone())?;
        dict.set_item("rejected_alternatives", entry.rejected_alternatives.clone())?;
        list.append(dict)?;
    }
    Ok(list.into())
}

fn collect_dlpack_columns(
    obj: &Bound<'_, PyAny>,
    type_error_message: &str,
) -> PyResult<Vec<DlpackManagedTensor>> {
    let seq = obj
        .downcast::<PySequence>()
        .map_err(|_| PyValueError::new_err(type_error_message.to_string()))?;

    let mut tensors: Vec<DlpackManagedTensor> = Vec::with_capacity(seq.len()?);
    for item in seq.try_iter()? {
        let item = item?;
        tensors.push(dlpack_from_py(&item)?);
    }
    Ok(tensors)
}

fn export_buffer_columns(
    py: Python<'_>,
    provider: &Arc<xlog_cuda::CudaKernelProvider>,
    buffer: xlog_cuda::CudaBuffer,
) -> PyResult<Vec<PyObject>> {
    let arity = buffer.arity();
    let table = provider.to_dlpack_table(buffer);
    let mut tensors: Vec<PyObject> = Vec::with_capacity(arity);
    for col_idx in 0..arity {
        let tensor = table.column(col_idx).map_err(types::xlog_err)?;
        tensors.push(dlpack_capsule_from_tensor(py, tensor)?);
    }
    Ok(tensors)
}

fn pack_logic_result_with_provider(
    py: Python<'_>,
    provider: &Arc<xlog_cuda::CudaKernelProvider>,
    result: gpu_logic::LogicEvalResult,
) -> PyResult<LogicEvalResult> {
    let mut queries: Vec<Py<LogicQueryResult>> = Vec::with_capacity(result.queries.len());

    for q in result.queries {
        let num_rows = provider
            .validated_logical_row_count(&q.buffer)
            .map_err(types::xlog_err)?;
        let is_true = q.columns.is_empty() && num_rows > 0;
        let tensors = if q.columns.is_empty() {
            Vec::new()
        } else {
            export_buffer_columns(py, provider, q.buffer)?
        };

        queries.push(Py::new(
            py,
            LogicQueryResult {
                relation_name: q.relation_name,
                columns: q.columns,
                sort_labels: q.sort_labels,
                tensors,
                num_rows,
                is_true,
            },
        )?);
    }

    Ok(LogicEvalResult { queries })
}
