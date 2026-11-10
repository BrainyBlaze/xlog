//! Candidate construction through the canonical function and physical tensor owners.

use super::*;
use std::collections::BTreeSet;
use xlog_ir::tensor::{
    TensorOperation as Op, TensorOutputDependencies, TensorParameter, TensorSignature,
};
use xlog_logic::ast::{FuncBody, FuncDef};
use xlog_neural::causal_snapshot::AcquiredInsertion;

/// Maximum bytes of an authored acquisition document.
pub const ACQUIRED_SOURCE_BYTES: usize = 1_048_576;
/// Total authored numerical operations across all new definitions.
pub const ACQUIRED_NODES: usize = 64;
/// Total unique new FP32 body scalars, excluding the native output projection.
pub const ACQUIRED_SCALARS: usize = 65_536;
/// Feature width at the native residual insertion.
pub const ACQUIRED_WIDTH: usize = 16;

/// Actual callers and physical parameter reads in a numerical call graph.
pub struct DefinitionUsage {
    /// Definitions reached from the requested entry, including external leaves.
    pub definitions: BTreeSet<String>,
    /// Parameter identities read by operations in the resolved bodies.
    pub parameters: BTreeSet<String>,
}

/// Follow canonical call/parameter references without constructing model state.
/// A prefix owner may resolve a known retained callee as an external leaf: an
/// earlier immutable body cannot reference a newly acquired name or constant.
pub fn definition_usage<'a>(
    entrypoint: &str,
    resolve: impl Fn(&str) -> Result<Option<&'a TensorProgram>>,
) -> Result<DefinitionUsage> {
    let mut usage = DefinitionUsage {
        definitions: BTreeSet::new(),
        parameters: BTreeSet::new(),
    };
    let mut pending = vec![entrypoint.to_owned()];
    while let Some(name) = pending.pop() {
        if !usage.definitions.insert(name.clone()) {
            continue;
        }
        let Some(body) = resolve(&name)? else {
            continue;
        };
        for operation in &body.operations {
            if let Some(callee) = operation.callee() {
                pending.push(callee.to_owned());
            }
            if let Some(id) = operation.parameter() {
                let parameter = body
                    .parameters
                    .get(id)
                    .ok_or_else(|| invalid("unknown acquired parameter reference"))?;
                usage.parameters.insert(parameter.name.clone());
            }
        }
    }
    Ok(usage)
}

/// Resolve output dependencies against actual immutable numerical definitions.
pub fn definition_output_dependencies<'a>(
    entrypoint: &str,
    resolve: impl Fn(&str) -> Result<&'a TensorProgram>,
) -> Result<TensorOutputDependencies> {
    definition_dependencies(entrypoint, resolve, false)
}

/// Resolve explicit value-aware facts through the same immutable call graph.
pub fn definition_output_value_dependencies<'a>(
    entrypoint: &str,
    resolve: impl Fn(&str) -> Result<&'a TensorProgram>,
) -> Result<TensorOutputDependencies> {
    definition_dependencies(entrypoint, resolve, true)
}

fn definition_dependencies<'a>(
    entrypoint: &str,
    resolve: impl Fn(&str) -> Result<&'a TensorProgram>,
    value_identities: bool,
) -> Result<TensorOutputDependencies> {
    fn visit<'a>(
        name: &str,
        resolve: &impl Fn(&str) -> Result<&'a TensorProgram>,
        saved: &mut BTreeMap<String, TensorOutputDependencies>,
        active: &mut BTreeSet<String>,
        value_identities: bool,
    ) -> Result<TensorOutputDependencies> {
        if let Some(dependencies) = saved.get(name) {
            return Ok(dependencies.clone());
        }
        if !active.insert(name.to_owned()) {
            return Err(invalid("numerical output dependency graph is cyclic"));
        }
        let body = resolve(name)?;
        let mut callee = |callee: &str| visit(callee, resolve, saved, active, value_identities);
        let mut dependencies = if value_identities {
            body.output_value_dependencies(&mut callee)?
        } else {
            body.output_dependencies(&mut callee)?
        };
        dependencies.definitions.insert(name.to_owned());
        active.remove(name);
        saved.insert(name.to_owned(), dependencies.clone());
        Ok(dependencies)
    }
    visit(
        entrypoint,
        &resolve,
        &mut BTreeMap::new(),
        &mut BTreeSet::new(),
        value_identities,
    )
}

fn invalid(message: impl Into<String>) -> XlogError {
    XlogError::Compilation(message.into())
}

impl NativeCausalModel {
    fn acquisition_down_projection(&self, insertion_layer: usize) -> Result<usize> {
        let name = format!("model.layers.{insertion_layer}.mlp.down_proj.weight");
        self.program()
            .parameters
            .iter()
            .position(|parameter| parameter.name == name)
            .ok_or_else(|| invalid("insertion layer is absent"))
    }

    /// ABI and available identities for a proposal against this exact registry.
    /// No function body or parameter values are supplied by this producer.
    pub fn acquisition_context(&self, insertion_layer: usize) -> Result<serde_json::Value> {
        use serde_json::json;
        let program = self.program();
        let hidden =
            program.parameters[self.acquisition_down_projection(insertion_layer)?].shape[0];
        let signature = TensorSignature::Causal {
            input_width: hidden,
            output_width: ACQUIRED_WIDTH,
        };
        let (params, result) =
            FuncDef::numerical_types(&signature, program.vocabulary, program.max_context)?;
        let mut ordinal = 0usize;
        let entrypoint = loop {
            let name = format!("acquired::{ordinal}");
            let namespace = format!("{name}::");
            if !self
                .functions
                .functions()
                .any(|function| function.name == name || function.name.starts_with(&namespace))
                && !program.parameters.iter().any(|parameter| {
                    parameter.name == name || parameter.name.starts_with(&namespace)
                })
            {
                break name;
            }
            ordinal = ordinal
                .checked_add(1)
                .ok_or_else(|| invalid("acquisition identity overflow"))?;
        };
        let callees = self.functions.functions().filter_map(|function| {
            let FuncBody::Numerical(body) = &function.body else {
                return None;
            };
            if matches!(body.signature, TensorSignature::Decoder) {
                return None;
            }
            Some((function.name.clone(), json!({
                "signature": body.signature, "params": function.params, "return_type": function.return_type,
            })))
        }).collect::<BTreeMap<_, _>>();
        let parameters = program
            .parameters
            .iter()
            .filter(|parameter| parameter.float32)
            .collect::<Vec<_>>();
        Ok(json!({
            "version": 1, "entrypoint": entrypoint,
            "function_prefix": format!("{entrypoint}::function::"),
            "parameter_prefix": format!("{entrypoint}::constant::"),
            "insertion_layer": insertion_layer, "signature": signature,
            "params": params, "return_type": result,
            "vocabulary": program.vocabulary, "max_context": program.max_context,
            "parameters": parameters, "callees": callees,
            "limits": {"source_bytes": ACQUIRED_SOURCE_BYTES, "nodes": ACQUIRED_NODES,
                       "new_scalars": ACQUIRED_SCALARS, "output_width": ACQUIRED_WIDTH},
        }))
    }

    /// Extend the new acquisition contract with exact output facts, not bodies.
    /// The original context remains byte-for-byte unchanged for older policies.
    pub fn acquisition_context_with_output_dependencies(
        &self,
        insertion_layer: usize,
    ) -> Result<serde_json::Value> {
        self.acquisition_context_with_dependencies(insertion_layer, false)
    }

    /// The explicit new context carries semantic identities in the same records.
    pub fn acquisition_context_with_value_dependencies(
        &self,
        insertion_layer: usize,
    ) -> Result<serde_json::Value> {
        self.acquisition_context_with_dependencies(insertion_layer, true)
    }

    fn acquisition_context_with_dependencies(
        &self,
        insertion_layer: usize,
        value_identities: bool,
    ) -> Result<serde_json::Value> {
        let mut context = self.acquisition_context(insertion_layer)?;
        context["version"] = serde_json::json!(if value_identities { 3 } else { 2 });
        context["dependency_contract"] = serde_json::json!(if value_identities {
            "output-dependencies-v2"
        } else {
            "output-dependencies-v1"
        });
        for (name, callee) in context["callees"].as_object_mut().unwrap() {
            let dependencies = definition_dependencies(
                name,
                |name| self.functions.numerical(name),
                value_identities,
            )?;
            callee["output_dependencies"] =
                serde_json::to_value(dependencies).map_err(|error| invalid(error.to_string()))?;
        }
        Ok(context)
    }

    /// Register actual typed definitions, insert their causal feature call, and
    /// allocate new FP32 constants/P with independent zero-step Adam states.
    /// Original definitions, weights, aliases and optimizer versions remain intact.
    pub fn fork_acquired(
        &self,
        definition_json: &str,
        parameter_values: BTreeMap<String, Vec<f32>>,
        insertion_layer: usize,
        train_down_projection: bool,
    ) -> Result<Self> {
        self.fork_acquired_with_dependencies(
            definition_json,
            parameter_values,
            insertion_layer,
            train_down_projection,
            false,
        )
    }

    /// Opt into output-reaching admission while retaining the original constructor.
    pub fn fork_acquired_with_dependencies(
        &self,
        definition_json: &str,
        parameter_values: BTreeMap<String, Vec<f32>>,
        insertion_layer: usize,
        train_down_projection: bool,
        require_output_dependencies: bool,
    ) -> Result<Self> {
        if definition_json.len() > ACQUIRED_SOURCE_BYTES {
            return Err(invalid(
                "acquired definition exceeds the source byte budget",
            ));
        }
        let mut document: BTreeMap<String, serde_json::Value> =
            serde_json::from_str(definition_json)
                .map_err(|error| invalid(format!("acquired definition JSON: {error}")))?;
        let entrypoint = document
            .remove("entrypoint")
            .and_then(|value| value.as_str().map(str::to_owned))
            .ok_or_else(|| invalid("acquired definition requires an entrypoint"))?;
        let definitions: Vec<FuncDef> = serde_json::from_value(
            document
                .remove("definitions")
                .ok_or_else(|| invalid("acquired definition requires canonical definitions"))?,
        )
        .map_err(|error| invalid(format!("acquired definitions: {error}")))?;
        if !document.is_empty() {
            return Err(invalid("unknown acquired fields or empty definitions"));
        }
        self.fork_acquired_definitions_with_dependencies(
            entrypoint,
            definitions,
            parameter_values,
            insertion_layer,
            train_down_projection,
            require_output_dependencies,
        )
    }

    /// Admit decoded definitions through the same registry, allocation and
    /// optimizer path as the JSON entry point. A source decoder owns its byte
    /// limit and elaboration before passing canonical definitions here.
    pub fn fork_acquired_definitions(
        &self,
        entrypoint: String,
        definitions: Vec<FuncDef>,
        parameter_values: BTreeMap<String, Vec<f32>>,
        insertion_layer: usize,
        train_down_projection: bool,
    ) -> Result<Self> {
        self.fork_acquired_definitions_with_dependencies(
            entrypoint,
            definitions,
            parameter_values,
            insertion_layer,
            train_down_projection,
            false,
        )
    }

    /// One registry/allocation path for the explicit old and output contracts.
    pub fn fork_acquired_definitions_with_dependencies(
        &self,
        entrypoint: String,
        definitions: Vec<FuncDef>,
        parameter_values: BTreeMap<String, Vec<f32>>,
        insertion_layer: usize,
        train_down_projection: bool,
        require_output_dependencies: bool,
    ) -> Result<Self> {
        self.fork_acquired_definitions_with_contract(
            entrypoint,
            definitions,
            parameter_values,
            insertion_layer,
            train_down_projection,
            require_output_dependencies,
            false,
        )
    }

    /// Admit the explicit value-aware source through the same physical owner.
    pub fn fork_acquired_definitions_with_value_dependencies(
        &self,
        entrypoint: String,
        definitions: Vec<FuncDef>,
        parameter_values: BTreeMap<String, Vec<f32>>,
        insertion_layer: usize,
        train_down_projection: bool,
    ) -> Result<Self> {
        self.fork_acquired_definitions_with_contract(
            entrypoint,
            definitions,
            parameter_values,
            insertion_layer,
            train_down_projection,
            true,
            true,
        )
    }

    fn fork_acquired_definitions_with_contract(
        &self,
        entrypoint: String,
        definitions: Vec<FuncDef>,
        parameter_values: BTreeMap<String, Vec<f32>>,
        insertion_layer: usize,
        train_down_projection: bool,
        require_output_dependencies: bool,
        value_identities: bool,
    ) -> Result<Self> {
        if definitions.is_empty() {
            return Err(invalid("unknown acquired fields or empty definitions"));
        }
        let down = self.acquisition_down_projection(insertion_layer)?;
        let hidden = self.program().parameters[down].shape[0];
        let mut functions = (*self.functions).clone();
        let mut program = self.program().clone();
        let mut new_parameters = BTreeMap::<String, TensorParameter>::new();
        let mut nodes = 0usize;
        let mut new_names = BTreeSet::new();
        for definition in definitions {
            if definition.name.starts_with("model::") || functions.contains(&definition.name) {
                return Err(invalid(
                    "acquired definition cannot replace an existing identity",
                ));
            }
            new_names.insert(definition.name.clone());
            let FuncBody::Numerical(body) = &definition.body else {
                return Err(invalid("acquired definitions require numerical bodies"));
            };
            if body.vocabulary != program.vocabulary
                || body.max_context != program.max_context
                || matches!(body.signature, TensorSignature::Decoder)
            {
                return Err(invalid(
                    "acquired definition has a different causal contract",
                ));
            }
            nodes = nodes
                .checked_add(body.operations.len())
                .ok_or_else(|| invalid("acquired node count overflow"))?;
            if nodes > ACQUIRED_NODES {
                return Err(invalid("acquired definitions exceed 64 numerical nodes"));
            }
            body.validate_acquired_body()?;
            for parameter in &body.parameters {
                if parameter.name.is_empty() || !parameter.float32 {
                    return Err(invalid("acquired constants require named FP32 storage"));
                }
                parameter.bytes()?;
                if let Some(existing) = program
                    .parameters
                    .iter()
                    .find(|value| value.name == parameter.name)
                {
                    if existing != parameter {
                        return Err(invalid(
                            "shared parameter name has a different physical type",
                        ));
                    }
                } else if let Some(existing) =
                    new_parameters.insert(parameter.name.clone(), parameter.clone())
                {
                    if existing != *parameter {
                        return Err(invalid("new parameter declarations disagree"));
                    }
                }
            }
            functions
                .register(definition)
                .map_err(|error| invalid(error.to_string()))?;
        }
        if functions.numerical(&entrypoint)?.signature
            != (TensorSignature::Causal {
                input_width: hidden,
                output_width: ACQUIRED_WIDTH,
            })
            || self.functions.contains(&entrypoint)
        {
            return Err(invalid(
                "new entrypoint must map this layer's hidden rows to 16 FP32 features",
            ));
        }
        let usage = if require_output_dependencies {
            let dependencies = if value_identities {
                definition_output_value_dependencies(&entrypoint, |name| functions.numerical(name))?
            } else {
                definition_output_dependencies(&entrypoint, |name| functions.numerical(name))?
            };
            if !dependencies.has_input() {
                return Err(invalid(
                    "acquired output must have a possible dependency on tokens or current hidden",
                ));
            }
            DefinitionUsage {
                definitions: dependencies.definitions,
                parameters: dependencies.parameters,
            }
        } else {
            definition_usage(&entrypoint, |name| functions.numerical(name).map(Some))?
        };
        let unused_definitions = new_names.difference(&usage.definitions).collect::<Vec<_>>();
        let unused_constants = new_parameters
            .keys()
            .filter(|name| !usage.parameters.contains(*name))
            .collect::<Vec<_>>();
        if !unused_definitions.is_empty() || !unused_constants.is_empty() {
            if require_output_dependencies {
                return Err(invalid(format!(
                    "new definitions and constants must reach the returned output; unreachable definitions: {unused_definitions:?}; unused constants: {unused_constants:?}"
                )));
            }
            return Err(invalid(format!(
                "new definitions and constants must have actual callers; unreachable definitions: {unused_definitions:?}; unused constants: {unused_constants:?}"
            )));
        }
        let mut constants = 0usize;
        for parameter in new_parameters.values() {
            constants = constants
                .checked_add(parameter.bytes()? / 4)
                .ok_or_else(|| invalid("constant extent overflow"))?;
        }
        if constants > ACQUIRED_SCALARS || new_parameters.keys().ne(parameter_values.keys()) {
            return Err(invalid(
                "new constants exceed the bound or initial array identities differ",
            ));
        }
        for (name, parameter) in &new_parameters {
            let values = &parameter_values[name];
            if values.len() != parameter.bytes()? / 4
                || values.iter().any(|value| !value.is_finite())
            {
                return Err(invalid("initial FP32 array extent or finiteness differs"));
            }
        }
        let mut candidate = self.fork_candidate();
        let mut ids = Vec::new();
        for parameter in new_parameters.values() {
            ids.push(program.parameters.len());
            program.parameters.push(parameter.clone());
        }
        let projection = program.parameters.len();
        let projection_name = format!("{entrypoint}::projection");
        if program
            .parameters
            .iter()
            .any(|parameter| parameter.name == projection_name)
        {
            return Err(invalid("projection identity already exists"));
        }
        program.parameters.push(TensorParameter {
            name: projection_name,
            shape: vec![hidden, ACQUIRED_WIDTH],
            float32: true,
        });
        let anchor = if let Some(insertion) = self
            .adaptation
            .insertions
            .iter()
            .rev()
            .find(|insertion| insertion.layer == insertion_layer)
        {
            insertion.output
        } else {
            let projected = program
                .operations
                .iter()
                .position(
                    |operation| matches!(operation, Op::Linear { weight, .. } if *weight == down),
                )
                .ok_or_else(|| invalid("down projection has no numerical caller"))?;
            program
                .operations
                .iter()
                .position(
                    |operation| matches!(operation, Op::Add { right, .. } if *right == projected),
                )
                .ok_or_else(|| invalid("decoder layer residual is absent"))?
        };
        let old = std::mem::take(&mut program.operations);
        let parameter_ids = (0..program.parameters.len()).collect::<Vec<_>>();
        let mut mapped = Vec::with_capacity(old.len());
        let mut original_positions = Vec::with_capacity(old.len());
        let mut insertion_output = None;
        for (id, operation) in old.iter().enumerate() {
            let value = emit(&mut program, operation.remap(&mapped, &parameter_ids));
            mapped.push(value);
            original_positions.push(value);
            if id == anchor {
                let hidden = emit(&mut program, Op::Float32 { input: value });
                let features = emit(
                    &mut program,
                    Op::Call {
                        function: entrypoint.clone(),
                        arguments: vec![hidden],
                    },
                );
                let delta = emit(
                    &mut program,
                    Op::Linear {
                        input: features,
                        weight: projection,
                    },
                );
                let rounded = emit(&mut program, Op::Bfloat16 { input: delta });
                let output = emit(
                    &mut program,
                    Op::Add {
                        left: value,
                        right: rounded,
                    },
                );
                mapped[id] = output;
                insertion_output = Some(output);
            }
        }
        program.output = mapped[program.output];
        for insertion in &mut candidate.adaptation.insertions {
            insertion.output = original_positions[insertion.output];
        }
        candidate.adaptation.insertions.push(AcquiredInsertion {
            function: entrypoint,
            layer: insertion_layer,
            projection,
            parameters: ids.clone(),
            output: insertion_output.ok_or_else(|| invalid("insertion anchor is absent"))?,
        });
        candidate.entrypoint = format!(
            "model::forward::acquired_{}",
            candidate.adaptation.insertions.len()
        );
        functions
            .register(FuncDef::numerical(candidate.entrypoint.clone(), program)?)
            .map_err(|error| invalid(error.to_string()))?;
        functions.compile_numerical(&candidate.entrypoint, 1, 0, Some((0, 1)))?;
        candidate.functions = Arc::new(functions);
        // Definitions and all physical extents have been admitted before creating
        // any native state. Failed allocation leaves only this unpublished fork.
        for &id in &ids {
            let parameter = &candidate.program().parameters[id];
            let mut tensor = self.executor.allocate_tensor(parameter_shape(parameter)?)?;
            let bytes = parameter_values[&parameter.name]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            self.executor.write_tensor(&mut tensor, 0, &bytes)?;
            candidate
                .optimizers
                .insert(id, self.executor.optimizer(id, &tensor)?);
            candidate.parameters.push(tensor);
        }
        let mut tensor = self.executor.allocate_tensor(parameter_shape(
            &candidate.program().parameters[projection],
        )?)?;
        self.executor
            .write_tensor(&mut tensor, 0, &vec![0u8; hidden * ACQUIRED_WIDTH * 4])?;
        candidate
            .optimizers
            .insert(projection, self.executor.optimizer(projection, &tensor)?);
        candidate.parameters.push(tensor);
        candidate.adaptation.trainable = ids;
        candidate.adaptation.trainable.push(projection);
        candidate.adaptation.down_projection = train_down_projection.then_some(down);
        if train_down_projection {
            if !candidate.optimizers.contains_key(&down) {
                candidate.optimizers.insert(
                    down,
                    self.executor.optimizer(down, &candidate.parameters[down])?,
                );
            }
            candidate.adaptation.trainable.push(down);
        }
        Ok(candidate)
    }

    /// Versioned, read-only physical roles for callers and continuation validation.
    pub fn adaptation_manifest(&self) -> Result<String> {
        let owned = self
            .adaptation
            .insertions
            .iter()
            .flat_map(|insertion| insertion.parameters.iter().copied())
            .collect::<BTreeSet<_>>();
        let projections = self
            .adaptation
            .insertions
            .iter()
            .map(|insertion| insertion.projection)
            .collect::<BTreeSet<_>>();
        let parameters = self.program().parameters.iter().enumerate().map(|(id, parameter)| serde_json::json!({
            "id": id, "name": parameter.name, "shape": parameter.shape, "dtype": if parameter.float32 { "f32" } else { "bf16" },
            "role": if projections.contains(&id) { "projection" } else if owned.contains(&id) { "acquired_constant" }
                else if self.optimizers.contains_key(&id) { "backbone_down_projection" } else { "frozen_backbone" },
            "optimizer_step": self.optimizers.get(&id).map(TensorAdamState::step),
        })).collect::<Vec<_>>();
        serde_json::to_string(&serde_json::json!({"schema": "xlog.native-adaptation.v1", "entrypoint": self.entrypoint,
            "parameters": parameters, "adaptation": self.adaptation})).map_err(|error| invalid(error.to_string()))
    }

    /// The actual definitions available to a proposer, including older callable bodies.
    pub fn function_definitions(&self) -> Result<String> {
        serde_json::to_string(&self.functions).map_err(|error| invalid(error.to_string()))
    }
}

fn emit(program: &mut TensorProgram, operation: Op) -> usize {
    let id = program.operations.len();
    program.operations.push(operation);
    id
}

fn parameter_shape(parameter: &TensorParameter) -> Result<TensorValueShape> {
    let (rows, columns) = match parameter.shape.as_slice() {
        &[columns] => (1, columns),
        &[rows, columns] => (rows, columns),
        _ => return Err(invalid("acquired parameter must be a vector or matrix")),
    };
    Ok(TensorValueShape {
        rows,
        columns,
        float32: true,
    })
}
