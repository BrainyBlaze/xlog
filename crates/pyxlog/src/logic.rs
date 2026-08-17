use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySequence};

use xlog_cuda::DlpackManagedTensor;
use xlog_gpu::logic as gpu_logic;
use xlog_logic::ast::ProbEngine;
use xlog_neural::{NetworkRegistry, TensorSourceRegistry};
use xlog_prob::exact::{ExactDdnnfProgram, GpuConfig};
use xlog_prob::mc::McProgram;
use xlog_runtime::RelationDelta;

use std::collections::HashMap as StdHashMap;

use super::neural_registry::NeuralPredicateRegistry;
use super::relation_metadata::{
    metadata_error, pack_session_evidence, relation_schema_fingerprint,
    require_positive_metadata_arity, PreparedInsertEvidence, PreparedRelationMetadataUpdate,
    RelationEvidence, RelationMetadataStore, RelationSnapshot,
};
use super::{
    dlpack_capsule_from_tensor, dlpack_from_py, enforce_call_memory_limit, pack_query_proof_traces,
    pack_rule_provenance, parse_prob_engine_override, provider_from_config, provider_memory_stats,
    types, CompiledLogicProgram, CompiledProbProgram, CompiledProgram, LogicDeltaStats,
    LogicEvalResult, LogicProgram, LogicQueryResult, LogicRelationSession, Program,
    RelationChangeCallback,
};

struct ParsedRelationDeltaUpdate {
    name: String,
    delta: RelationDelta,
    insert_evidence: Option<PreparedInsertEvidence>,
}

enum RelationReplacementMetadata {
    Clear,
    Replace(RelationMetadataStore),
}

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
            domain_source: None,
            domain_ids: Vec::new(),
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
            program: Arc::new(program),
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
                    schema.arity(),
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
            relation_metadata: RelationMetadataStore::default(),
        })
    }

    /// Return memory diagnostics including allocated_bytes and memory_limit_bytes.
    pub fn memory_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        provider_memory_stats(py, &self.provider)
    }

    pub fn rule_provenance(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        pack_rule_provenance(py, &self.program.rule_provenance())
    }

    pub fn proof_traces(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        pack_query_proof_traces(py, &self.program.proof_traces())
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
        let schema = self.relation_replacement_schema(&name)?;
        let buffer = self.detached_relation_replacement_buffer(&name, schema, dlpack_columns)?;
        self.commit_relation_replacement(name, buffer, RelationReplacementMetadata::Clear)
    }

    #[pyo3(signature = (name, dlpack_columns, *, roles, facts))]
    pub fn put_relation_with_provenance(
        &mut self,
        py: Python<'_>,
        name: String,
        dlpack_columns: &Bound<'_, PyAny>,
        roles: &Bound<'_, PyAny>,
        facts: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let schema = self.relation_replacement_schema(&name)?;
        require_positive_metadata_arity(&name, schema)?;
        let arguments = self.program.argument_schema(&name).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "Relation '{name}' compiled argument contract is unavailable"
            ))
        })?;
        let buffer = self.detached_relation_replacement_buffer(&name, schema, dlpack_columns)?;
        let row_count = self
            .provider
            .validated_logical_row_count(&buffer)
            .map_err(types::xlog_err)?;
        let (prospective_metadata, snapshot) = self.relation_metadata.prepare_replacement(
            &name,
            &arguments,
            schema,
            &self.provider,
            &buffer,
            roles,
            facts,
            row_count,
        )?;
        let packed_snapshot = snapshot.pack(py)?;
        self.commit_relation_replacement(
            name,
            buffer,
            RelationReplacementMetadata::Replace(prospective_metadata),
        )?;
        Ok(packed_snapshot)
    }

    pub fn put_relation_from_manifest(
        &mut self,
        py: Python<'_>,
        name: String,
        dlpack_columns: &Bound<'_, PyAny>,
        manifest: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let schema = self.relation_replacement_schema(&name)?.clone();
        require_positive_metadata_arity(&name, &schema)?;
        let arguments = self.program.argument_schema(&name).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "Relation '{name}' compiled argument contract is unavailable"
            ))
        })?;
        let prepared_manifest = self
            .relation_metadata
            .parse_manifest(&name, &arguments, &schema, manifest)?;
        let buffer = self.detached_relation_replacement_buffer(&name, &schema, dlpack_columns)?;
        let row_count = self
            .provider
            .validated_logical_row_count(&buffer)
            .map_err(types::xlog_err)?;
        let (prospective_metadata, snapshot) =
            self.relation_metadata.prepare_manifest_replacement(
                &name,
                &schema,
                &self.provider,
                &buffer,
                row_count,
                prepared_manifest,
            )?;
        let packed_snapshot = snapshot.pack(py)?;
        let metadata = prospective_metadata.map_or(
            RelationReplacementMetadata::Clear,
            RelationReplacementMetadata::Replace,
        );
        self.commit_relation_replacement(name, buffer, metadata)?;
        Ok(packed_snapshot)
    }

    pub fn relation(&self, name: &str) -> PyResult<RelationEvidence> {
        let buffer = self
            .relation_store
            .get(name)
            .ok_or_else(|| PyKeyError::new_err(format!("Relation '{name}' is not stored")))?;
        Ok(RelationEvidence::new(
            self.snapshot_stored_relation(name, buffer)?,
        ))
    }

    #[pyo3(signature = (name=None))]
    pub fn evidence(&self, py: Python<'_>, name: Option<&str>) -> PyResult<Py<PyAny>> {
        if let Some(name) = name {
            if !self.relation_store.contains(name) {
                return Err(PyKeyError::new_err(format!(
                    "Relation '{name}' is not stored"
                )));
            }
        }
        let mut relation_names = self
            .relation_store
            .names()
            .map(str::to_string)
            .collect::<Vec<_>>();
        relation_names.sort();
        let mut snapshots = Vec::with_capacity(relation_names.len());
        for relation_name in relation_names {
            let buffer = self.relation_store.get(&relation_name).ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "Relation '{relation_name}' disappeared while preparing evidence"
                ))
            })?;
            snapshots.push(self.snapshot_stored_relation(&relation_name, buffer)?);
        }
        pack_session_evidence(py, snapshots, name)
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

    #[pyo3(signature = (name, dlpack_columns, *, facts=None))]
    pub fn insert_relation(
        &mut self,
        py: Python<'_>,
        name: String,
        dlpack_columns: &Bound<'_, PyAny>,
        facts: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        self.require_insert_metadata_arity(&name, facts)?;
        let insert = self.relation_delta_buffer(&name, dlpack_columns)?;
        let insert_evidence = self.prepare_insert_evidence(&name, &insert, facts)?;
        self.apply_single_relation_delta(py, name, Some(insert), None, insert_evidence)
    }

    pub fn delete_relation(
        &mut self,
        py: Python<'_>,
        name: String,
        dlpack_columns: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let delete = self.relation_delta_buffer(&name, dlpack_columns)?;
        self.apply_single_relation_delta(py, name, None, Some(delete), None)
    }

    #[pyo3(signature = (name, insert_columns=None, delete_columns=None, *, insert_facts=None))]
    pub fn apply_relation_delta(
        &mut self,
        py: Python<'_>,
        name: String,
        insert_columns: Option<&Bound<'_, PyAny>>,
        delete_columns: Option<&Bound<'_, PyAny>>,
        insert_facts: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if insert_facts.is_some() && insert_columns.is_none() {
            return Err(metadata_error(
                "apply_relation_delta insert_facts requires insert_columns".to_string(),
            ));
        }
        if insert_columns.is_none() && delete_columns.is_none() {
            return Err(PyValueError::new_err(
                "apply_relation_delta requires insert_columns, delete_columns, or both",
            ));
        }
        self.require_insert_metadata_arity(&name, insert_facts)?;
        let insert = insert_columns
            .map(|columns| self.relation_delta_buffer(&name, columns))
            .transpose()?;
        let delete = delete_columns
            .map(|columns| self.relation_delta_buffer(&name, columns))
            .transpose()?;
        let insert_evidence = match insert.as_ref() {
            Some(insert) => self.prepare_insert_evidence(&name, insert, insert_facts)?,
            None => None,
        };
        self.apply_single_relation_delta(py, name, insert, delete, insert_evidence)
    }

    pub fn apply_relation_delta_batch(
        &mut self,
        py: Python<'_>,
        updates: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let parsed = self.parse_relation_delta_batch("apply_relation_delta_batch", updates)?;
        let (batch, metadata_updates, relation_names, schemas, cancellation_capture_relations) =
            split_parsed_relation_updates(&self.program, parsed)?;
        let prepared_batch = self
            .program
            .prepare_relation_delta_batch(
                self.provider.as_ref(),
                batch,
                &cancellation_capture_relations,
            )
            .map_err(types::xlog_err)?;
        let metadata_transition = self.relation_metadata.prepare_batch_transition(
            &self.provider,
            &schemas,
            metadata_updates,
            &prepared_batch,
        )?;
        let data_commit = self
            .program
            .prepare_relation_delta_commit_with_session_runtime(
                self.provider.clone(),
                &mut self.relation_store,
                &mut self.evaluation_store,
                &mut self.session_runtime,
                prepared_batch,
            )
            .map_err(types::xlog_err)?;
        let report = data_commit.commit();
        metadata_transition.commit(&mut self.relation_metadata);
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
    ) -> PyResult<Py<PyAny>> {
        let parsed = self.parse_relation_delta_batch("apply_relation_delta_debug", updates)?;
        let (batch, metadata_updates, relation_names, schemas, cancellation_capture_relations) =
            split_parsed_relation_updates(&self.program, parsed)?;
        let had_derived_state = self.evaluation_store.is_some() || self.session_runtime.is_some();
        let delta_start = Instant::now();
        let prepared_batch = self
            .program
            .prepare_relation_delta_batch(
                self.provider.as_ref(),
                batch,
                &cancellation_capture_relations,
            )
            .map_err(types::xlog_err)?;
        let coalesced_no_op = prepared_batch.net_deltas().is_empty();
        let metadata_transition = self.relation_metadata.prepare_batch_transition(
            &self.provider,
            &schemas,
            metadata_updates,
            &prepared_batch,
        )?;
        let data_commit = self
            .program
            .prepare_relation_delta_commit_with_session_runtime(
                self.provider.clone(),
                &mut self.relation_store,
                &mut self.evaluation_store,
                &mut self.session_runtime,
                prepared_batch,
            )
            .map_err(types::xlog_err)?;
        let delta_micros = delta_start.elapsed().as_micros().max(1) as u64;
        let mut equivalent_to_full_recompute = None;
        let mut measured_full_micros = None;
        if check_equivalence {
            let full_start = Instant::now();
            let prospective_base = data_commit
                .clone_prospective_base_store()
                .map_err(types::xlog_err)?;
            let (_, full_store) = self
                .program
                .evaluate_with_relation_store_and_cache(
                    self.provider.clone(),
                    &prospective_base,
                    false,
                )
                .map_err(types::xlog_err)?;
            let full_micros = full_start.elapsed().as_micros() as u64;
            let equivalent = if coalesced_no_op && !had_derived_state {
                true
            } else {
                self.program
                    .relation_stores_query_equivalent(
                        self.provider.as_ref(),
                        &full_store,
                        data_commit.prospective_derived_store(),
                    )
                    .map_err(types::xlog_err)?
            };
            equivalent_to_full_recompute = Some(equivalent);
            measured_full_micros = Some(full_micros);
        }
        let report = data_commit.commit();
        metadata_transition.commit(&mut self.relation_metadata);
        let mut stats = logic_delta_stats_from_report(report);
        stats.equivalent_to_full_recompute = equivalent_to_full_recompute;
        if let Some(full_micros) = measured_full_micros {
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

    pub fn delta_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
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

    pub fn rule_provenance(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        pack_rule_provenance(py, &self.program.rule_provenance())
    }

    pub fn proof_traces(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        pack_query_proof_traces(py, &self.program.proof_traces())
    }

    pub fn register_relation_callback(
        &mut self,
        py: Python<'_>,
        callback: Py<PyAny>,
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

    pub fn cuda_graph_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
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

    pub fn host_transfer_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let stats = self.provider.host_transfer_stats();
        let dict = PyDict::new(py);
        dict.set_item("dtoh_bytes", stats.dtoh_bytes)?;
        dict.set_item("htod_bytes", stats.htod_bytes)?;
        dict.set_item("dtoh_calls", stats.dtoh_calls)?;
        dict.set_item("htod_calls", stats.htod_calls)?;
        Ok(dict.into())
    }

    pub fn join_index_cache_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
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

    /// Multiway/Free-Join dispatch telemetry for the retained session
    /// executor. Counters accumulate across evaluates within this session;
    /// all zeros before the first evaluate.
    pub fn wcoj_dispatch_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        let stats = self
            .session_runtime
            .as_ref()
            .map(|runtime| runtime.wcoj_dispatch_stats())
            .unwrap_or_default();
        dict.set_item("free_join_dispatch_count", stats.free_join_dispatch_count)?;
        dict.set_item(
            "factorized_delta_dispatch_count",
            stats.factorized_delta_dispatch_count,
        )?;
        dict.set_item(
            "wcoj_groupby_fusion_dispatch_count",
            stats.wcoj_groupby_fusion_dispatch_count,
        )?;
        dict.set_item("wcoj_error_decline_count", stats.wcoj_error_decline_count)?;
        Ok(dict.into())
    }

    pub fn reset_host_transfer_stats(&self) {
        self.provider.reset_host_transfer_stats()
    }

    pub fn set_strict_deterministic_d2h(&self, enabled: bool) {
        if enabled {
            self.provider.enable_strict_deterministic_d2h();
        } else {
            self.provider.disable_strict_deterministic_d2h();
        }
    }

    pub fn strict_deterministic_d2h_enabled(&self) -> bool {
        self.provider.strict_deterministic_d2h_enabled()
    }

    pub fn deterministic_d2h_violation_count(&self) -> u64 {
        self.provider.deterministic_d2h_violation_count()
    }

    pub fn reset_deterministic_d2h_violations(&self) {
        self.provider.reset_deterministic_d2h_violations();
    }

    /// Return memory diagnostics including allocated_bytes and memory_limit_bytes.
    pub fn memory_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        provider_memory_stats(py, &self.provider)
    }

    pub fn export_relation(&mut self, py: Python<'_>, name: &str) -> PyResult<Vec<Py<PyAny>>> {
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
        let stored = self.relation_store.get_mut(name).ok_or_else(|| {
            PyRuntimeError::new_err(format!("Relation '{}' disappeared during export", name))
        })?;
        let buffer = std::mem::replace(stored, replacement);
        export_buffer_columns(py, &self.provider, buffer)
    }

    pub fn export_relation_with_provenance(
        &mut self,
        py: Python<'_>,
        name: &str,
    ) -> PyResult<Py<PyAny>> {
        let schema = self.relation_replacement_schema(name)?;
        require_positive_metadata_arity(name, schema)?;
        let existing = self.relation_store.get(name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Relation '{}' not found in persistent session",
                name
            ))
        })?;
        let manifest = self
            .snapshot_stored_relation(name, existing)?
            .pack_manifest(py)?;
        let columns = self.export_relation(py, name)?;
        let exported = PyDict::new(py);
        exported.set_item("columns", columns)?;
        exported.set_item("manifest", manifest)?;
        Ok(exported.into())
    }

    pub fn remove_relation(&mut self, name: &str) -> bool {
        let removed = self.relation_store.remove(name).is_some();
        if removed {
            self.relation_metadata.clear_relation(name);
            self.evaluation_store = None;
            self.session_runtime = None;
            self.last_delta_stats = None;
        }
        removed
    }

    pub fn clear_relations(&mut self) {
        self.relation_store.clear();
        self.relation_metadata.clear();
        self.evaluation_store = None;
        self.session_runtime = None;
        self.last_delta_stats = None;
    }
}

impl LogicRelationSession {
    fn commit_relation_replacement(
        &mut self,
        name: String,
        buffer: xlog_cuda::CudaBuffer,
        metadata: RelationReplacementMetadata,
    ) -> PyResult<()> {
        let additional = usize::from(!self.relation_store.contains(&name));
        self.relation_store
            .try_reserve_relations(additional)
            .map_err(types::xlog_err)?;
        match metadata {
            RelationReplacementMetadata::Clear => {
                self.relation_metadata.clear_relation(&name);
                self.relation_store.put_owned(name, buffer);
            }
            RelationReplacementMetadata::Replace(prospective) => {
                self.relation_store.put_owned(name, buffer);
                self.relation_metadata = prospective
            }
        }
        self.evaluation_store = None;
        self.session_runtime = None;
        self.last_delta_stats = None;
        Ok(())
    }

    fn snapshot_stored_relation(
        &self,
        name: &str,
        buffer: &xlog_cuda::CudaBuffer,
    ) -> PyResult<RelationSnapshot> {
        let arguments = self.program.argument_schema(name).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "Relation '{name}' compiled argument contract is unavailable"
            ))
        })?;
        let schema_sha256 = relation_schema_fingerprint(name, &arguments)?;
        let row_count = self
            .provider
            .validated_logical_row_count(buffer)
            .map_err(types::xlog_err)?;
        Ok(self
            .relation_metadata
            .snapshot(name, row_count, schema_sha256, arguments.len()))
    }

    fn relation_replacement_schema(&self, name: &str) -> PyResult<&xlog_core::Schema> {
        if name.starts_with("__") {
            return Err(PyValueError::new_err(format!(
                "Relation {name} is internal and cannot be stored in a persistent session"
            )));
        }
        self.program.schema(name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown relation {name} (not present in compiled schemas)"
            ))
        })
    }

    fn relation_replacement_buffer(
        &self,
        name: &str,
        schema: &xlog_core::Schema,
        dlpack_columns: &Bound<'_, PyAny>,
    ) -> PyResult<xlog_cuda::CudaBuffer> {
        let tensors = collect_dlpack_columns(
            dlpack_columns,
            schema.arity(),
            &format!("Relation {name} must be a sequence of DLPack columns"),
        )?;
        self.provider
            .from_dlpack_tensors_with_schema(schema.clone(), tensors)
            .map_err(types::xlog_err)
    }

    fn detached_relation_replacement_buffer(
        &self,
        name: &str,
        schema: &xlog_core::Schema,
        dlpack_columns: &Bound<'_, PyAny>,
    ) -> PyResult<xlog_cuda::CudaBuffer> {
        // Persistent sessions own a stable snapshot. Keeping a producer-backed
        // DLPack pointer here would let later producer writes bypass relation
        // versions and invalidate whole-fact evidence.
        let imported = self.relation_replacement_buffer(name, schema, dlpack_columns)?;
        self.provider
            .clone_buffer(&imported)
            .map_err(types::xlog_err)
    }

    fn parse_relation_delta_batch(
        &self,
        method_name: &str,
        updates: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<ParsedRelationDeltaUpdate>> {
        let seq = updates.cast::<PySequence>().map_err(|_| {
            PyValueError::new_err(format!(
                "{method_name} expects a sequence of update dictionaries"
            ))
        })?;
        let mut parsed = Vec::new();
        for (update_index, item) in seq.try_iter()?.enumerate() {
            let item = item?;
            let dict = item.cast::<PyDict>().map_err(|_| {
                PyValueError::new_err(format!("{method_name} updates must be dictionaries"))
            })?;
            reject_unknown_delta_update_keys(dict, method_name, update_index)?;
            let name_obj = dict.get_item("name")?.ok_or_else(|| {
                PyValueError::new_err(format!("{method_name} update missing 'name'"))
            })?;
            let name: String = name_obj.extract()?;
            let insert_columns = optional_delta_columns(dict, "insert_columns");
            let delete_columns = optional_delta_columns(dict, "delete_columns");
            let insert_facts = dict
                .get_item("insert_facts")?
                .filter(|value| !value.is_none());
            if insert_facts.is_some() && insert_columns.is_none() {
                return Err(metadata_error(format!(
                    "{method_name} update {update_index} insert_facts requires insert_columns"
                )));
            }
            if insert_columns.is_none() && delete_columns.is_none() {
                return Err(PyValueError::new_err(format!(
                    "{method_name} updates require insert_columns, delete_columns, or both"
                )));
            }
            self.require_insert_metadata_arity(&name, insert_facts.as_ref())?;
            let insert = insert_columns
                .map(|columns| self.relation_delta_buffer(&name, &columns))
                .transpose()?;
            let delete = delete_columns
                .map(|columns| self.relation_delta_buffer(&name, &columns))
                .transpose()?;
            let insert_evidence = match insert.as_ref() {
                Some(insert) => {
                    self.prepare_insert_evidence(&name, insert, insert_facts.as_ref())?
                }
                None => None,
            };
            parsed.push(ParsedRelationDeltaUpdate {
                name,
                delta: RelationDelta::new(insert, delete),
                insert_evidence,
            });
        }
        Ok(parsed)
    }

    fn require_insert_metadata_arity(
        &self,
        name: &str,
        facts: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        if facts.is_none() {
            return Ok(());
        }
        require_positive_metadata_arity(name, self.relation_delta_schema(name)?)
    }

    fn prepare_insert_evidence(
        &self,
        name: &str,
        insert: &xlog_cuda::CudaBuffer,
        facts: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<PreparedInsertEvidence>> {
        let Some(facts) = facts else {
            return Ok(None);
        };
        let schema = self.program.schema(name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown relation {name} (not present in compiled schemas)"
            ))
        })?;
        let arguments = self.program.argument_schema(name).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "Relation '{name}' compiled argument contract is unavailable"
            ))
        })?;
        self.relation_metadata
            .prepare_insert_evidence(name, &arguments, schema, &self.provider, insert, facts)
            .map(Some)
    }

    fn relation_delta_buffer(
        &self,
        name: &str,
        dlpack_columns: &Bound<'_, PyAny>,
    ) -> PyResult<xlog_cuda::CudaBuffer> {
        let schema = self.relation_delta_schema(name)?;
        let tensors = collect_dlpack_columns(
            dlpack_columns,
            schema.arity(),
            &format!(
                "Relation {} delta must be a sequence of DLPack columns",
                name
            ),
        )?;
        self.provider
            .from_dlpack_tensors_with_schema(schema.clone(), tensors)
            .map_err(types::xlog_err)
    }

    fn relation_delta_schema(&self, name: &str) -> PyResult<&xlog_core::Schema> {
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
        Ok(schema)
    }

    fn apply_single_relation_delta(
        &mut self,
        py: Python<'_>,
        name: String,
        insert: Option<xlog_cuda::CudaBuffer>,
        delete: Option<xlog_cuda::CudaBuffer>,
        insert_evidence: Option<PreparedInsertEvidence>,
    ) -> PyResult<Py<PyAny>> {
        let relation_names = vec![name.clone()];
        let schema = self.program.schema(&name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown relation {name} (not present in compiled schemas)"
            ))
        })?;
        let metadata_transition = self.relation_metadata.prepare_delta_transition(
            &name,
            schema,
            &self.provider,
            delete.as_ref(),
            insert_evidence,
        )?;
        let mut deltas = HashMap::new();
        deltas.insert(name, RelationDelta::new(insert, delete));
        let data_commit = self
            .program
            .prepare_relation_deltas_commit_with_session_runtime(
                self.provider.clone(),
                &mut self.relation_store,
                &mut self.evaluation_store,
                &mut self.session_runtime,
                deltas,
            )
            .map_err(types::xlog_err)?;
        let report = data_commit.commit();
        metadata_transition.commit(&mut self.relation_metadata);
        let stats = logic_delta_stats_from_report(report);
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

        let effective_relations = stats
            .changed_relation_names
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let mut seen = HashSet::new();
        let mut events: Vec<(String, u64)> = Vec::new();
        for relation in relation_names {
            if effective_relations.contains(relation.as_str()) && seen.insert(relation.clone()) {
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
) -> PyResult<Py<PyAny>> {
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

fn reject_unknown_delta_update_keys(
    dict: &Bound<'_, PyDict>,
    method_name: &str,
    update_index: usize,
) -> PyResult<()> {
    for key in dict.keys().iter() {
        let key = key.extract::<String>().map_err(|_| {
            PyValueError::new_err(format!(
                "{method_name} update {update_index} keys must be strings"
            ))
        })?;
        if !matches!(
            key.as_str(),
            "name" | "insert_columns" | "delete_columns" | "insert_facts"
        ) {
            return Err(PyValueError::new_err(format!(
                "{method_name} update {update_index} has unknown key '{key}'"
            )));
        }
    }
    Ok(())
}

fn split_parsed_relation_updates(
    program: &gpu_logic::LogicProgram,
    parsed: Vec<ParsedRelationDeltaUpdate>,
) -> PyResult<(
    Vec<(String, RelationDelta)>,
    Vec<PreparedRelationMetadataUpdate>,
    Vec<String>,
    BTreeMap<String, xlog_core::Schema>,
    BTreeSet<String>,
)> {
    let mut batch = Vec::with_capacity(parsed.len());
    let mut metadata_updates = Vec::with_capacity(parsed.len());
    let mut relation_names = Vec::with_capacity(parsed.len());
    let mut schemas = BTreeMap::new();
    let mut cancellation_capture_relations = BTreeSet::new();

    for update in parsed {
        if update
            .insert_evidence
            .as_ref()
            .is_some_and(PreparedInsertEvidence::has_fact_keys)
        {
            cancellation_capture_relations.insert(update.name.clone());
        }
        let schema = program.schema(&update.name).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "Relation '{}' schema disappeared while preparing its delta batch",
                update.name
            ))
        })?;
        schemas
            .entry(update.name.clone())
            .or_insert_with(|| schema.clone());
        relation_names.push(update.name.clone());
        metadata_updates.push(PreparedRelationMetadataUpdate::new(
            update.name.clone(),
            update.insert_evidence,
        ));
        batch.push((update.name, update.delta));
    }

    Ok((
        batch,
        metadata_updates,
        relation_names,
        schemas,
        cancellation_capture_relations,
    ))
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

fn pack_delta_stats(py: Python<'_>, stats: &LogicDeltaStats) -> PyResult<Py<PyAny>> {
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
) -> PyResult<Py<PyAny>> {
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

fn collect_dlpack_columns(
    obj: &Bound<'_, PyAny>,
    expected_arity: usize,
    type_error_message: &str,
) -> PyResult<Vec<DlpackManagedTensor>> {
    let seq = obj
        .cast::<PySequence>()
        .map_err(|_| PyValueError::new_err(type_error_message.to_string()))?;

    let mut iterator = seq.try_iter()?;
    let mut items = Vec::with_capacity(expected_arity);
    for column in 0..expected_arity {
        let item = iterator.next().transpose()?.ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "Schema arity {expected_arity} does not match tensor count {column}"
            ))
        })?;
        items.push(item);
    }
    if iterator.next().transpose()?.is_some() {
        return Err(PyRuntimeError::new_err(format!(
            "Schema arity {expected_arity} does not match tensor count greater than {expected_arity}"
        )));
    }
    items.iter().map(dlpack_from_py).collect()
}

fn export_buffer_columns(
    py: Python<'_>,
    provider: &Arc<xlog_cuda::CudaKernelProvider>,
    buffer: xlog_cuda::CudaBuffer,
) -> PyResult<Vec<Py<PyAny>>> {
    let arity = buffer.arity();
    let table = provider.to_dlpack_table(buffer);
    let mut tensors: Vec<Py<PyAny>> = Vec::with_capacity(arity);
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
