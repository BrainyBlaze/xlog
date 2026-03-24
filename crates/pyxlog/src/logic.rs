use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySequence};

use xlog_cuda::DlpackManagedTensor;
use xlog_gpu::logic as gpu_logic;
use xlog_logic::ast::ProbEngine;
use xlog_neural::{NetworkRegistry, TensorSourceRegistry};
use xlog_prob::exact::{ExactDdnnfProgram, GpuConfig};
use xlog_prob::mc::McProgram;

use std::collections::HashMap as StdHashMap;

use super::neural_registry::NeuralPredicateRegistry;
use super::{
    dlpack_capsule_from_tensor, dlpack_from_py, parse_prob_engine_override, provider_from_config,
    types, CompiledLogicProgram, CompiledProbProgram, CompiledProgram, LogicEvalResult,
    LogicProgram, LogicQueryResult, LogicRelationSession, Program,
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

        let config = GpuConfig {
            device_ordinal: device,
            memory_bytes: memory_mb * 1024 * 1024,
            ..Default::default()
        };

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

        let config = GpuConfig {
            device_ordinal: device,
            memory_bytes: memory_mb * 1024 * 1024,
            ..Default::default()
        };

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
    #[pyo3(signature = (dlpack_inputs=None))]
    pub fn evaluate(
        &self,
        py: Python<'_>,
        dlpack_inputs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<LogicEvalResult> {
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
        })
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
        Ok(())
    }

    pub fn evaluate(&self, py: Python<'_>) -> PyResult<LogicEvalResult> {
        let result = self
            .program
            .evaluate_with_relation_store(self.provider.clone(), &self.relation_store, false)
            .map_err(types::xlog_err)?;
        pack_logic_result_with_provider(py, &self.provider, result)
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
        self.relation_store.remove(name).is_some()
    }

    pub fn clear_relations(&mut self) {
        self.relation_store.clear();
    }
}

fn collect_dlpack_columns(
    obj: &Bound<'_, PyAny>,
    type_error_message: &str,
) -> PyResult<Vec<DlpackManagedTensor>> {
    let seq = obj
        .downcast::<PySequence>()
        .map_err(|_| PyValueError::new_err(type_error_message.to_string()))?;

    let mut tensors: Vec<DlpackManagedTensor> = Vec::with_capacity(seq.len()? as usize);
    for item in seq.iter()? {
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
                tensors,
                num_rows,
                is_true,
            },
        )?);
    }

    Ok(LogicEvalResult { queries })
}
