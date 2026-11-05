//! Checkpoint-backed causal numerical programs and their retained outputs.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use xlog_core::{MemoryBudget, Result, XlogError};
use xlog_cuda::tensor::{
    GreedySelection, NativeTensor, TensorAdamState, TensorExecutor, TensorLoss,
    TensorTrainingResult,
};
use xlog_cuda::{CudaKernelProvider, CudaProviderBuilder, DlpackManagedTensor};
use xlog_ir::tensor::{TensorProgram, TensorValueShape};
use xlog_logic::function::FunctionRegistry;
use xlog_neural::causal_decoder::CausalDecoderCheckpoint;
use xlog_neural::causal_snapshot::{
    CausalDecoderSnapshot, CausalSnapshotDescription, NativeAdaptation, OptimizerSnapshot,
};

mod acquired;

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
    let provider: Arc<CudaKernelProvider> = Arc::new(
        CudaProviderBuilder::new(device, MemoryBudget::with_limit(memory_budget_bytes)).build()?,
    );
    Ok(Arc::new(TensorExecutor::new(provider)?))
}

/// Versioned weights and executable numerical definition on one native provider.
pub struct NativeCausalModel {
    executor: Arc<TensorExecutor>,
    parameters: Vec<NativeTensor>,
    functions: Arc<FunctionRegistry>,
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
        Ok(candidate)
    }

    /// Continue an immutable model version with exactly the same active roles,
    /// definitions, physical aliases and optimizer states, including frozen controls.
    pub fn fork_candidate(&self) -> Self {
        Self {
            executor: Arc::clone(&self.executor),
            parameters: self.parameters.clone(),
            functions: Arc::clone(&self.functions),
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
    pub fn fit_tokens(
        &mut self,
        token_ids: &[Vec<u32>],
        prediction_positions: &[usize],
        target_ids: &[u32],
        history_spans: Option<&[(usize, usize)]>,
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
            true,
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
                    tokens.as_slice(),
                    position,
                    history_spans.map(|spans| spans[index]),
                    loss,
                )
            })
            .collect::<Vec<_>>();
        let (updates, mut result) =
            self.executor
                .train_parameters(&self.parameters, &states, &batch)?;
        // Publish only after every new parameter and optimizer version is finite.
        for (id, forward, state) in updates {
            self.parameters[id] = forward;
            self.optimizers.insert(id, state);
        }
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
    pub fn forward(
        &self,
        token_ids: &[u32],
        prediction_position: usize,
        history_span: Option<(usize, usize)>,
    ) -> Result<NativeLogits> {
        self.validate_history_span(token_ids.len(), history_span)?;
        let program = self.functions.compile_numerical(
            &self.entrypoint,
            token_ids.len(),
            prediction_position,
            history_span,
        )?;
        let tensor = self.executor.execute(
            &program,
            &self.parameters,
            token_ids,
            prediction_position,
            history_span,
        )?;
        Ok(NativeLogits {
            executor: Arc::clone(&self.executor),
            tensor,
            prediction_position,
            vocabulary_size: self.program().vocabulary,
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
    /// Read only the explicitly requested token logits, in caller order.
    pub fn read(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        self.executor.read_logits(&self.tensor, token_ids)
    }

    /// CUDA maxima with minimum-ID ties and the original full-row FP32 digest.
    /// A supplied mask is an exact little-endian bitset; absent means every token.
    pub fn select_greedy(&self, allowed_mask: Option<&[u8]>) -> Result<GreedySelection> {
        self.executor.select_greedy(&self.tensor, allowed_mask)
    }

    /// Export the native [1, vocabulary] FP32 allocation with retained ownership.
    pub fn export(&self) -> Result<DlpackManagedTensor> {
        self.tensor.export()
    }
}
