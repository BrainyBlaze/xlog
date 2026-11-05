//! Checkpoint-backed causal numerical programs and their retained outputs.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use xlog_core::{MemoryBudget, Result, XlogError};
use xlog_cuda::tensor::{
    NativeTensor, TensorAdamState, TensorExecutor, TensorLoss, TensorTrainingResult,
};
use xlog_cuda::{CudaKernelProvider, CudaProviderBuilder, DlpackManagedTensor};
use xlog_ir::tensor::{TensorProgram, TensorValueShape};
use xlog_neural::causal_decoder::CausalDecoderCheckpoint;
use xlog_neural::causal_snapshot::{
    CausalDecoderSnapshot, CausalSnapshotDescription, OptimizerSnapshot,
};

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
    program: TensorProgram,
    checkpoint_identity: [u8; 32],
    device: usize,
    optimizer: Option<(usize, TensorAdamState)>,
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
                .program
                .parameters
                .iter()
                .try_fold(0u64, |total, parameter| {
                    total
                        .checked_add(parameter.bytes()? as u64)
                        .ok_or_else(|| XlogError::Compilation("checkpoint size overflow".into()))
                })?;
        let executor = native_executor(device, memory_budget_bytes, weight_bytes)?;
        let mut parameters = Vec::with_capacity(checkpoint.program.parameters.len());
        for parameter in &checkpoint.program.parameters {
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
            program: checkpoint.program,
            checkpoint_identity,
            device,
            optimizer: None,
        })
    }

    /// Save actual program/parameter/optimizer bytes and a caller-owned immutable
    /// application payload. Existing paths are never overwritten. The returned
    /// identity is SHA256 of this entire self-contained SafeTensors snapshot.
    pub fn save_snapshot(&self, path: &Path, application_state: Vec<u8>) -> Result<[u8; 32]> {
        let optimizer = self
            .optimizer
            .as_ref()
            .map(|(parameter, state)| OptimizerSnapshot {
                parameter: *parameter,
                step: state.step(),
            });
        let description = CausalSnapshotDescription::new(
            self.program.clone(),
            self.checkpoint_identity,
            optimizer,
            application_state,
        );
        let mut tensors = self.parameters.iter().collect::<Vec<_>>();
        if let Some((_, state)) = &self.optimizer {
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
        let state = tensors.split_off(description.program.parameters.len());
        let optimizer = if let Some(optimizer) = description.optimizer {
            let [master, first, second]: [NativeTensor; 3] = state.try_into().map_err(|_| {
                XlogError::Compilation("snapshot optimizer roster is incomplete".into())
            })?;
            let state = executor.restore_linear_optimizer(
                optimizer.parameter,
                &tensors[optimizer.parameter],
                master,
                first,
                second,
                optimizer.step,
            )?;
            Some((optimizer.parameter, state))
        } else {
            None
        };
        Ok((
            Self {
                executor,
                parameters: tensors,
                program: description.program,
                checkpoint_identity: description.checkpoint_identity,
                device,
                optimizer,
            },
            description.application_state,
        ))
    }

    /// Fork a candidate with one trainable MLP down projection. Frozen weights
    /// and existing optimizer versions share storage until a successful update.
    pub fn fork_trainable(&self, down_projection_layer: usize) -> Result<Self> {
        let name = format!("model.layers.{down_projection_layer}.mlp.down_proj.weight");
        let parameter = self
            .program
            .parameters
            .iter()
            .position(|entry| entry.name == name)
            .ok_or_else(|| XlogError::Compilation("trainable decoder layer is absent".into()))?;
        let state = match &self.optimizer {
            Some((old_parameter, state)) if *old_parameter == parameter => state.clone(),
            Some(_) => {
                return Err(XlogError::Compilation(
                    "a trainable candidate retains its selected down projection".into(),
                ))
            }
            None => self
                .executor
                .linear_optimizer(parameter, &self.parameters[parameter])?,
        };
        Ok(Self {
            executor: Arc::clone(&self.executor),
            parameters: self.parameters.clone(),
            program: self.program.clone(),
            checkpoint_identity: self.checkpoint_identity,
            device: self.device,
            optimizer: Some((parameter, state)),
        })
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
        self.fit(token_ids, prediction_positions, &losses)
    }

    /// One full-vocabulary next-token cross-entropy update. Every target is scored
    /// against all original vocabulary entries, including for editor training.
    pub fn fit_tokens(
        &mut self,
        token_ids: &[Vec<u32>],
        prediction_positions: &[usize],
        target_ids: &[u32],
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
        self.fit(token_ids, prediction_positions, &losses)
    }

    fn fit(
        &mut self,
        token_ids: &[Vec<u32>],
        prediction_positions: &[usize],
        losses: &[TensorLoss],
    ) -> Result<TensorTrainingResult> {
        if token_ids.is_empty()
            || prediction_positions.len() != token_ids.len()
            || losses.len() != token_ids.len()
        {
            return Err(XlogError::Compilation(
                "training requires equally sized nonempty prefixes, positions and targets".into(),
            ));
        }
        let (parameter, state) = self.optimizer.as_ref().ok_or_else(|| {
            XlogError::Compilation(
                "fork a trainable candidate before updating model parameters".into(),
            )
        })?;
        let batch = token_ids
            .iter()
            .zip(prediction_positions)
            .zip(losses)
            .map(|((tokens, &position), &loss)| (tokens.as_slice(), position, loss))
            .collect::<Vec<_>>();
        let (forward, next, result) =
            self.executor
                .train_linear(&self.program, &self.parameters, state, &batch, 1.0e-5)?;
        // All allocations and numerical checks completed before this exclusive
        // publication. Old readers and forked models retain their original bytes.
        self.parameters[*parameter] = forward;
        self.optimizer = Some((*parameter, next));
        Ok(result)
    }

    /// Recompute the supplied causal prefix and return its requested vocabulary row.
    pub fn forward(&self, token_ids: &[u32], prediction_position: usize) -> Result<NativeLogits> {
        let tensor = self.executor.execute(
            &self.program,
            &self.parameters,
            token_ids,
            prediction_position,
        )?;
        Ok(NativeLogits {
            executor: Arc::clone(&self.executor),
            tensor,
            prediction_position,
            vocabulary_size: self.program.vocabulary,
            device: self.device,
        })
    }

    /// Digest of the actual imported configuration and canonical parameter bytes.
    pub fn checkpoint_identity(&self) -> &[u8; 32] {
        &self.checkpoint_identity
    }

    /// Canonical parameter names and completed native optimizer updates.
    /// Frozen models return an empty map; newly trainable parameters have step zero.
    pub fn optimizer_steps(&self) -> BTreeMap<String, u64> {
        self.optimizer
            .iter()
            .map(|(parameter, state)| {
                (
                    self.program.parameters[*parameter].name.clone(),
                    state.step(),
                )
            })
            .collect()
    }

    /// Maximum accepted input length; longer input is rejected without truncation.
    pub fn max_context(&self) -> usize {
        self.program.max_context
    }

    /// Complete vocabulary width.
    pub fn vocabulary_size(&self) -> usize {
        self.program.vocabulary
    }
}

impl NativeLogits {
    /// Read only the explicitly requested token logits, in caller order.
    pub fn read(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        self.executor.read_logits(&self.tensor, token_ids)
    }

    /// Export the native [1, vocabulary] FP32 allocation with retained ownership.
    pub fn export(&self) -> Result<DlpackManagedTensor> {
        self.tensor.export()
    }
}
