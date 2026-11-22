//! Checkpoint-backed causal numerical programs and their retained outputs.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use xlog_core::{MemoryBudget, Result, XlogError};
use xlog_cuda::tensor::{
    LogitSelection, NativeTensor, TensorAdamState, TensorExecutor, TensorInputs, TensorLoss,
    TensorTrainingResult,
};
use xlog_cuda::{CudaProviderBuilder, DlpackManagedTensor};
use xlog_ir::tensor::{TensorProgram, TensorSignature, TensorValueShape};
use xlog_logic::function::FunctionRegistry;
use xlog_neural::causal_decoder::CausalDecoderCheckpoint;
use xlog_neural::causal_snapshot::{
    CausalDecoderSnapshot, CausalSnapshotDescription, NativeAdaptation, OptimizerSnapshot,
};

pub mod acquired;
pub mod grid;

fn native_executor(
    device: usize,
    memory_budget_bytes: u64,
    state_bytes: u64,
) -> Result<Arc<TensorExecutor>> {
    if memory_budget_bytes <= state_bytes {
        return Err(XlogError::Compilation(
            "memory budget must exceed actual native state bytes to admit execution workspace"
                .into(),
        ));
    }
    Ok(Arc::new(
        CudaProviderBuilder::new(device, MemoryBudget::with_limit(memory_budget_bytes))
            .build_tensor_executor()?,
    ))
}

/// Versioned weights and executable numerical definition on one native provider.
pub struct NativeCausalModel {
    executor: Arc<TensorExecutor>,
    parameters: Vec<NativeTensor>,
    functions: Arc<FunctionRegistry>,
    version: Arc<()>,
    entrypoint: String,
    checkpoint_identity: [u8; 32],
    device: usize,
    optimizers: BTreeMap<usize, TensorAdamState>,
    adaptation: NativeAdaptation,
}

/// One complete vocabulary row, retaining its storage and original provider.
pub struct NativeLogits {
    executor: Arc<TensorExecutor>,
    tensor: NativeTensor,
    entrypoint: String,
    disabled_definition: Option<String>,
    /// Calls to the selected definition replaced in this prefix, including scan steps.
    pub disabled_calls: usize,
    /// Zero-based input position whose next-token distribution was computed.
    pub prediction_position: usize,
    /// Width of the complete vocabulary, including reserved token IDs.
    pub vocabulary_size: usize,
    /// CUDA device ordinal of the retained output.
    pub device: usize,
}

impl NativeCausalModel {
    fn program(&self) -> &TensorProgram {
        self.functions
            .numerical(&self.entrypoint)
            .expect("validated immutable decoder definition")
    }

    /// Import local config.json and authentic SafeTensors shards, then publish
    /// the model only after every parameter and tied alias has been checked.
    pub fn from_checkpoint(
        directory: &Path,
        device: usize,
        memory_budget_bytes: u64,
        max_context: usize,
    ) -> Result<Self> {
        let mut checkpoint = CausalDecoderCheckpoint::open(directory, max_context)?;
        let weight_bytes =
            checkpoint
                .program()
                .parameters
                .iter()
                .try_fold(0u64, |total, parameter| {
                    total
                        .checked_add(parameter.bytes()? as u64)
                        .ok_or_else(|| XlogError::Compilation("checkpoint size overflow".into()))
                })?;
        let executor = native_executor(device, memory_budget_bytes, weight_bytes)?;
        let mut parameters = Vec::with_capacity(checkpoint.program().parameters.len());
        for parameter in &checkpoint.program().parameters {
            let (rows, columns) = match parameter.shape.as_slice() {
                &[columns] => (1, columns),
                &[rows, columns] => (rows, columns),
                _ => {
                    return Err(XlogError::Compilation(
                        "decoder parameter must be a vector or matrix".into(),
                    ))
                }
            };
            parameters.push(executor.allocate_parameter(rows, columns)?);
        }
        let checkpoint_identity = checkpoint
            .load(|id, offset, bytes| executor.write_tensor(&mut parameters[id], offset, bytes))?;
        Ok(Self {
            executor,
            parameters,
            functions: Arc::clone(checkpoint.functions()),
            version: Arc::new(()),
            entrypoint: checkpoint.entrypoint().into(),
            checkpoint_identity,
            device,
            optimizers: BTreeMap::new(),
            adaptation: NativeAdaptation::default(),
        })
    }

    /// Save actual program/parameter/optimizer bytes and a caller-owned immutable
    /// application payload. Existing paths are never overwritten. The returned
    /// identity is SHA256 of this entire self-contained SafeTensors snapshot.
    pub fn save_snapshot(&self, path: &Path, application_state: Vec<u8>) -> Result<[u8; 32]> {
        let optimizers = self
            .optimizers
            .iter()
            .map(|(parameter, state)| OptimizerSnapshot {
                parameter: *parameter,
                step: state.step(),
            })
            .collect();
        let description = CausalSnapshotDescription::new(
            Arc::clone(&self.functions),
            self.entrypoint.clone(),
            self.checkpoint_identity,
            optimizers,
            self.adaptation.clone(),
            application_state,
        );
        let mut tensors = self.parameters.iter().collect::<Vec<_>>();
        for state in self.optimizers.values() {
            tensors.extend(state.snapshot_tensors());
        }
        CausalDecoderSnapshot::write(path, &description, |id, offset, bytes| {
            self.executor.read_tensor(tensors[id], offset, bytes)
        })
    }

    /// Restore the exact saved bytes on a fresh native provider, without an
    /// external checkpoint. Application bytes are returned only after SHA256 and
    /// numerical optimizer checks succeed; their semantics remain caller-owned.
    pub fn from_snapshot(
        path: &Path,
        expected_identity: [u8; 32],
        device: usize,
        memory_budget_bytes: u64,
    ) -> Result<(Self, Vec<u8>)> {
        let snapshot = CausalDecoderSnapshot::open(path, expected_identity)?;
        let roster = snapshot.description().tensors()?;
        let state_bytes = roster.iter().try_fold(0u64, |total, tensor| {
            total
                .checked_add(tensor.bytes()? as u64)
                .ok_or_else(|| XlogError::Compilation("snapshot allocation extent overflow".into()))
        })?;
        let executor = native_executor(device, memory_budget_bytes, state_bytes)?;
        let mut tensors = roster
            .iter()
            .map(|tensor| {
                executor.allocate_tensor(TensorValueShape {
                    rows: tensor.rows,
                    columns: tensor.columns,
                    float32: tensor.float32,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let description = snapshot
            .load(|id, offset, bytes| executor.write_tensor(&mut tensors[id], offset, bytes))?;
        let state = tensors.split_off(description.program()?.parameters.len());
        let mut state = state.into_iter();
        let mut optimizers = BTreeMap::new();
        for optimizer in &description.optimizers {
            let mut take = || {
                state.next().ok_or_else(|| {
                    XlogError::Compilation("snapshot optimizer roster is incomplete".into())
                })
            };
            let restored = executor.restore_optimizer(
                optimizer.parameter,
                &tensors[optimizer.parameter],
                take()?,
                take()?,
                take()?,
                optimizer.step,
            )?;
            optimizers.insert(optimizer.parameter, restored);
        }
        Ok((
            Self {
                executor,
                parameters: tensors,
                functions: description.functions,
                version: Arc::new(()),
                entrypoint: description.entrypoint,
                checkpoint_identity: description.checkpoint_identity,
                device,
                optimizers,
                adaptation: description.adaptation,
            },
            description.application_state,
        ))
    }

    /// Fork a candidate with one trainable MLP down projection. Frozen weights
    /// and existing optimizer versions share storage until a successful update.
    pub fn fork_trainable(&self, down_projection_layer: usize) -> Result<Self> {
        let name = format!("model.layers.{down_projection_layer}.mlp.down_proj.weight");
        let parameter = self
            .program()
            .parameters
            .iter()
            .position(|entry| entry.name == name)
            .ok_or_else(|| XlogError::Compilation("trainable decoder layer is absent".into()))?;
        if !self.adaptation.insertions.is_empty() {
            if self.adaptation.down_projection != Some(parameter) {
                return Err(XlogError::Compilation(
                    "fork_candidate preserves an acquired model's actual trainable roles".into(),
                ));
            }
            return Ok(self.fork_candidate());
        }
        if self
            .adaptation
            .down_projection
            .is_some_and(|id| id != parameter)
        {
            return Err(XlogError::Compilation(
                "a candidate retains its selected down projection".into(),
            ));
        }
        let mut candidate = self.fork_candidate();
        if !candidate.optimizers.contains_key(&parameter) {
            candidate.optimizers.insert(
                parameter,
                self.executor
                    .optimizer(parameter, &self.parameters[parameter])?,
            );
        }
        candidate.adaptation.down_projection = Some(parameter);
        candidate.adaptation.trainable = vec![parameter];
        if let Some(grid) = &candidate.adaptation.grid {
            candidate
                .adaptation
                .trainable
                .extend([grid.sensor, grid.readout]);
        }
        Ok(candidate)
    }

    /// Continue an immutable model version with exactly the same active roles,
    /// definitions, physical aliases and optimizer states, including frozen controls.
    pub fn fork_candidate(&self) -> Self {
        Self {
            executor: Arc::clone(&self.executor),
            parameters: self.parameters.clone(),
            functions: Arc::clone(&self.functions),
            version: Arc::clone(&self.version),
            entrypoint: self.entrypoint.clone(),
            checkpoint_identity: self.checkpoint_identity,
            device: self.device,
            optimizers: self.optimizers.clone(),
            adaptation: self.adaptation.clone(),
        }
    }

    /// One mean-minibatch binary NLL update, using only the supplied original IDs.
    /// The caller owns data selection and the global attempt/update budget.
    pub fn fit_binary(
        &mut self,
        token_ids: &[Vec<u32>],
        prediction_positions: &[usize],
        outcomes: &[bool],
        negative_token_id: u32,
        positive_token_id: u32,
        history_spans: Option<&[(usize, usize)]>,
    ) -> Result<TensorTrainingResult> {
        if outcomes.len() != token_ids.len() {
            return Err(XlogError::Compilation(
                "one outcome is required per training prefix".into(),
            ));
        }
        let losses = outcomes
            .iter()
            .map(|&outcome| TensorLoss::Binary {
                negative: negative_token_id,
                positive: positive_token_id,
                outcome,
            })
            .collect::<Vec<_>>();
        self.fit(
            token_ids,
            prediction_positions,
            &losses,
            history_spans,
            false,
        )
    }

    /// One full-vocabulary next-token cross-entropy update. Every target is scored
    /// against all original vocabulary entries, including for editor training.
    /// Train only the permitted down projection unless train_acquired selects the
    /// candidate's current trainable set, leaving older insertions frozen.
    pub fn fit_tokens(
        &mut self,
        token_ids: &[Vec<u32>],
        prediction_positions: &[usize],
        target_ids: &[u32],
        history_spans: Option<&[(usize, usize)]>,
        train_acquired: bool,
    ) -> Result<TensorTrainingResult> {
        if target_ids.len() != token_ids.len() {
            return Err(XlogError::Compilation(
                "one next-token target is required per training prefix".into(),
            ));
        }
        let losses = target_ids
            .iter()
            .map(|&target| TensorLoss::Token { target })
            .collect::<Vec<_>>();
        self.fit(
            token_ids,
            prediction_positions,
            &losses,
            history_spans,
            !train_acquired,
        )
    }

    fn fit(
        &mut self,
        token_ids: &[Vec<u32>],
        prediction_positions: &[usize],
        losses: &[TensorLoss],
        history_spans: Option<&[(usize, usize)]>,
        down_only: bool,
    ) -> Result<TensorTrainingResult> {
        if token_ids.is_empty()
            || prediction_positions.len() != token_ids.len()
            || losses.len() != token_ids.len()
        {
            return Err(XlogError::Compilation(
                "training requires equally sized nonempty prefixes, positions and targets".into(),
            ));
        }
        if history_spans.is_some_and(|spans| spans.len() != token_ids.len()) {
            return Err(XlogError::Compilation(
                "one raw history span is required per prefix".into(),
            ));
        }
        let ids = if down_only {
            self.adaptation
                .down_projection
                .into_iter()
                .collect::<Vec<_>>()
        } else {
            self.adaptation.trainable.clone()
        };
        let mut programs = Vec::with_capacity(token_ids.len());
        for (index, (tokens, &position)) in token_ids.iter().zip(prediction_positions).enumerate() {
            let span = history_spans.map(|spans| spans[index]);
            self.validate_history_span(tokens.len(), span)?;
            programs.push(self.functions.compile_numerical(
                &self.entrypoint,
                tokens.len(),
                position,
                span,
            )?);
        }
        let batch = token_ids
            .iter()
            .zip(prediction_positions)
            .zip(losses)
            .enumerate()
            .map(|(index, ((tokens, &position), &loss))| {
                (
                    &programs[index],
                    TensorInputs::Tokens(tokens),
                    position,
                    history_spans.map(|spans| spans[index]),
                    loss,
                )
            })
            .collect::<Vec<_>>();
        self.fit_inputs(&ids, &batch)
    }

    fn fit_inputs(
        &mut self,
        ids: &[usize],
        batch: &[(
            &TensorProgram,
            TensorInputs<'_>,
            usize,
            Option<(usize, usize)>,
            TensorLoss,
        )],
    ) -> Result<TensorTrainingResult> {
        let states = ids
            .iter()
            .map(|id| {
                self.optimizers
                    .get(id)
                    .map(|state| {
                        (
                            state,
                            if self.program().parameters[*id].float32 {
                                1.0e-3
                            } else {
                                1.0e-5
                            },
                        )
                    })
                    .ok_or_else(|| {
                        XlogError::Compilation("trainable parameter has no optimizer".into())
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let next_step = self
            .adaptation
            .completed_updates
            .checked_add(1)
            .ok_or_else(|| XlogError::Compilation("completed update counter overflow".into()))?;
        let (updates, mut result) = self
            .executor
            .train_inputs(&self.parameters, &states, batch)?;
        // Publish only after every new parameter and optimizer version is finite.
        for (id, forward, state) in updates {
            self.parameters[id] = forward;
            self.optimizers.insert(id, state);
        }
        self.version = Arc::new(());
        self.adaptation.completed_updates = next_step;
        result.step = next_step;
        Ok(result)
    }

    fn validate_history_span(&self, tokens: usize, span: Option<(usize, usize)>) -> Result<()> {
        if (!self.adaptation.insertions.is_empty() && span.is_none())
            || span.is_some_and(|(start, end)| start > end || end > tokens)
        {
            return Err(XlogError::Compilation(
                "acquired execution requires an explicit raw history span within this prefix"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Recompute the supplied causal prefix and return its requested vocabulary row.
    /// With acquired residuals excluded, execute the original decoder definition
    /// using its current trained parameter prefix, without changing model state.
    pub fn forward(
        &self,
        token_ids: &[u32],
        prediction_position: usize,
        history_span: Option<(usize, usize)>,
        include_acquired: bool,
    ) -> Result<NativeLogits> {
        self.forward_with_intervention(
            token_ids,
            prediction_position,
            history_span,
            include_acquired,
            None,
        )
    }

    /// Execute a prefix while optionally disabling one named numerical callee
    /// in the active model. Other calls and their physical parameters are shared.
    pub fn forward_with_intervention(
        &self,
        token_ids: &[u32],
        prediction_position: usize,
        history_span: Option<(usize, usize)>,
        include_acquired: bool,
        disabled_definition: Option<&str>,
    ) -> Result<NativeLogits> {
        self.validate_history_span(token_ids.len(), history_span)?;
        let (entrypoint, parameters) = if include_acquired {
            (self.entrypoint.as_str(), self.parameters.as_slice())
        } else {
            if disabled_definition.is_some() {
                return Err(XlogError::Compilation(
                    "disabling a named definition requires the active decoder".into(),
                ));
            }
            let entrypoint = "model::forward";
            let backbone = self.functions.numerical(entrypoint)?;
            let active = self.program();
            if backbone.signature != TensorSignature::Decoder
                || backbone.vocabulary != active.vocabulary
                || backbone.max_context != active.max_context
                || active.parameters.get(..backbone.parameters.len())
                    != Some(backbone.parameters.as_slice())
            {
                return Err(XlogError::Compilation(
                    "original decoder contract or parameter prefix differs from the active model"
                        .into(),
                ));
            }
            let parameters = self
                .parameters
                .get(..backbone.parameters.len())
                .ok_or_else(|| {
                    XlogError::Compilation(
                        "original decoder parameter allocations are absent".into(),
                    )
                })?;
            (entrypoint, parameters)
        };
        let (program, disabled_calls) = self.functions.compile_numerical_with_intervention(
            entrypoint,
            token_ids.len(),
            prediction_position,
            history_span,
            disabled_definition,
        )?;
        let tensor = self.executor.execute(
            &program,
            parameters,
            token_ids,
            prediction_position,
            history_span,
        )?;
        Ok(NativeLogits {
            executor: Arc::clone(&self.executor),
            tensor,
            entrypoint: entrypoint.to_owned(),
            disabled_definition: disabled_definition.map(str::to_owned),
            disabled_calls,
            prediction_position,
            vocabulary_size: program.vocabulary,
            device: self.device,
        })
    }

    /// Digest of the actual imported configuration and canonical parameter bytes.
    pub fn checkpoint_identity(&self) -> &[u8; 32] {
        &self.checkpoint_identity
    }

    /// Canonical parameter names and completed native optimizer updates.
    /// Imported models return an empty map; frozen trained versions retain their counters.
    pub fn optimizer_steps(&self) -> BTreeMap<String, u64> {
        self.optimizers
            .iter()
            .map(|(parameter, state)| {
                (
                    self.program().parameters[*parameter].name.clone(),
                    state.step(),
                )
            })
            .collect()
    }

    /// Maximum accepted input length; longer input is rejected without truncation.
    pub fn max_context(&self) -> usize {
        self.program().max_context
    }

    /// Complete vocabulary width.
    pub fn vocabulary_size(&self) -> usize {
        self.program().vocabulary
    }
}

impl NativeLogits {
    /// Optional callee selected for this invocation, even when its call count is zero.
    pub fn disabled_definition(&self) -> Option<&str> {
        self.disabled_definition.as_deref()
    }

    /// Numerical entrypoint actually executed to produce this vocabulary row.
    pub fn entrypoint(&self) -> &str {
        &self.entrypoint
    }

    /// Read only the explicitly requested token logits, in caller order.
    pub fn read(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        self.executor.read_logits(&self.tensor, token_ids)
    }

    /// CUDA maxima with minimum-ID ties and the original full-row FP32 digest.
    /// A supplied mask is an exact little-endian bitset; absent means every token.
    pub fn select_greedy(&self, allowed_mask: Option<&[u8]>) -> Result<LogitSelection> {
        self.executor.select_greedy(&self.tensor, allowed_mask)
    }

    /// CUDA categorical draw at temperature one, with an explicit Philox address.
    /// Scores and the full-row digest refer to the original unperturbed logits.
    pub fn select_categorical(
        &self,
        allowed_mask: Option<&[u8]>,
        seed: u64,
        sequence: u64,
        position: u32,
    ) -> Result<LogitSelection> {
        self.executor
            .select_categorical(&self.tensor, allowed_mask, seed, sequence, position)
    }

    /// Export the native [1, vocabulary] FP32 allocation with retained ownership.
    pub fn export(&self) -> Result<DlpackManagedTensor> {
        self.tensor.export()
    }
}
