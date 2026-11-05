//! Checkpoint-backed causal numerical programs and their retained outputs.

use std::path::Path;
use std::sync::Arc;

use xlog_core::{MemoryBudget, Result, XlogError};
use xlog_cuda::tensor::{NativeTensor, TensorExecutor};
use xlog_cuda::{CudaKernelProvider, CudaProviderBuilder, DlpackManagedTensor};
use xlog_ir::tensor::TensorProgram;
use xlog_neural::causal_decoder::CausalDecoderCheckpoint;

/// Immutable weights and executable numerical definition on one native provider.
pub struct NativeCausalModel {
    executor: Arc<TensorExecutor>,
    parameters: Vec<NativeTensor>,
    program: TensorProgram,
    checkpoint_identity: [u8; 32],
    device: usize,
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
        if memory_budget_bytes <= weight_bytes {
            return Err(XlogError::Compilation("memory budget must exceed actual checkpoint weight bytes to admit execution workspace".into()));
        }
        let provider: Arc<CudaKernelProvider> = Arc::new(
            CudaProviderBuilder::new(device, MemoryBudget::with_limit(memory_budget_bytes))
                .build()?,
        );
        let executor = Arc::new(TensorExecutor::new(provider)?);
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
        let checkpoint_identity = checkpoint.load(|id, offset, bytes| {
            executor.write_parameter(&mut parameters[id], offset, bytes)
        })?;
        Ok(Self {
            executor,
            parameters,
            program: checkpoint.program,
            checkpoint_identity,
            device,
        })
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
