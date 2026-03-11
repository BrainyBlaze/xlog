use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PySequence};
use pyo3::exceptions::PyValueError;

use ::xlog_gpu::logic as gpu_logic;
use xlog_cuda::DlpackManagedTensor;
use xlog_logic::ast::ProbEngine;
use xlog_neural::{NetworkRegistry, TensorSourceRegistry};
use xlog_prob::exact::{ExactDdnnfProgram, GpuConfig};
use xlog_prob::mc::McProgram;

use std::collections::HashMap as StdHashMap;

use super::{
    types,
    CompiledLogicProgram, CompiledProgram, LogicProgram, Program,
    LogicEvalResult, LogicQueryResult,
    CompiledProbProgram,
    dlpack_capsule_from_tensor, dlpack_from_py, provider_from_config, parse_prob_engine_override,
};
use super::neural_registry::NeuralPredicateRegistry;

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
        let ast = xlog_logic::parse_program(source)
            .map_err(types::xlog_err)?;

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

        let neural_registry =
            NeuralPredicateRegistry::from_ast(&ast).map_err(types::val_err)?;

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
                McProgram::compile_source_with_gpu(source, config)
                    .map_err(types::xlog_err)?,
            ),
        };
        let provider =
            provider_from_config(config).map_err(types::xlog_err)?;

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

        let program = gpu_logic::LogicProgram::compile(source)
            .map_err(types::xlog_err)?;
        let provider =
            provider_from_config(config).map_err(types::xlog_err)?;

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

                let seq = v.downcast::<PySequence>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "Input relation {} must be a sequence of DLPack columns",
                        name
                    ))
                })?;

                let mut tensors: Vec<DlpackManagedTensor> = Vec::with_capacity(seq.len()? as usize);
                for item in seq.iter()? {
                    let item = item?;
                    tensors.push(dlpack_from_py(&item)?);
                }

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
        self.pack_logic_result(py, result)
    }
}

impl CompiledLogicProgram {
    fn pack_logic_result(
        &self,
        py: Python<'_>,
        result: gpu_logic::LogicEvalResult,
    ) -> PyResult<LogicEvalResult> {
        let mut queries: Vec<Py<LogicQueryResult>> = Vec::with_capacity(result.queries.len());

        for q in result.queries {
            let num_rows = q.buffer.num_rows() as usize;
            let is_true = q.columns.is_empty() && num_rows > 0;
            let arity = q.buffer.arity();
            let mut tensors: Vec<PyObject> = Vec::new();
            if !q.columns.is_empty() {
                let table = self.provider.to_dlpack_table(q.buffer);
                tensors.reserve(arity);
                for col_idx in 0..arity {
                    let tensor = table
                        .column(col_idx)
                        .map_err(types::xlog_err)?;
                    tensors.push(dlpack_capsule_from_tensor(py, tensor)?);
                }
            }

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
}
