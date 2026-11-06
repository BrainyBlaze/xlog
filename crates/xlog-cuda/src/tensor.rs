//! Native dense programs on the provider's recorded streams and budgeted storage.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::cublas::{result as blas_result, sys as blas_sys, CudaBlas};
use sha2::{Digest, Sha256};
use xlog_core::{Result, XlogError};
use xlog_ir::tensor::{TensorOperation as Op, TensorProgram, TensorValueShape};

use crate::device_runtime::StreamId;
use crate::dlpack::{export_slice_managed_tensor, DLDataType, DlpackManagedTensor};
use crate::launch::{LaunchEnqueueError, LaunchRecorder};
use crate::memory::TrackedCudaSlice;
use crate::{CudaFunction, CudaKernelProvider, LaunchAsync, LaunchConfig};

const BLAS_WORKSPACE_BYTES: usize = 4 * 1024 * 1024;

mod functions;
mod training;
pub use training::{TensorAdamState, TensorLoss, TensorTrainingResult};

/// CUDA token selection and the digest of the complete original FP32 vocabulary row.
pub struct LogitSelection {
    /// Maximum over the whole vocabulary, with the lowest ID breaking ties.
    pub raw_argmax: u32,
    /// Original FP32 score of the unmasked maximum.
    pub raw_score: f32,
    /// Token chosen among allowed tokens by the requested selection law.
    pub selected: u32,
    /// Original FP32 score of the selected token, before masking.
    pub selected_score: f32,
    /// SHA256 of every original little-endian FP32 logit in vocabulary order.
    pub logits_sha256: [u8; 32],
}

fn failure(error: impl std::fmt::Display) -> XlogError {
    XlogError::Kernel(format!("native numerical execution: {error}"))
}

fn enqueue_error(error: LaunchEnqueueError<XlogError>) -> XlogError {
    match error {
        LaunchEnqueueError::Preparation(error) => failure(error),
        LaunchEnqueueError::PreparationAndCleanup {
            preparation,
            cleanup,
        } => failure(format!("{preparation}; reservation cleanup: {cleanup}")),
        LaunchEnqueueError::Operation(error) => error,
        LaunchEnqueueError::OperationAndCleanup { operation, cleanup } => {
            failure(format!("{operation}; reservation cleanup: {cleanup}"))
        }
    }
}

fn dimension(value: usize) -> Result<u32> {
    u32::try_from(value)
        .ok()
        .filter(|&v| v > 0 && v <= i32::MAX as u32)
        .ok_or_else(|| failure("dimension exceeds the CUDA integer contract"))
}

fn grid(blocks: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid_dim: (dimension(blocks)?, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// A retained allocation, including its exact logical extent and storage type.
/// Clones share storage; they do not make additional parameter copies.
#[derive(Clone)]
pub struct NativeTensor {
    storage: Arc<TrackedCudaSlice<u8>>,
    shape: TensorValueShape,
    device: usize,
}

impl NativeTensor {
    fn pointer(&self) -> u64 {
        self.storage.device_ptr_value()
    }
    fn elements(&self) -> usize {
        self.shape.rows * self.shape.columns
    }

    /// Export the actual retained producer allocation after its recorded writes.
    pub fn export(&self) -> Result<DlpackManagedTensor> {
        export_slice_managed_tensor(
            Arc::clone(&self.storage),
            self.device as i32,
            DLDataType {
                code: if self.shape.float32 { 2 } else { 4 },
                bits: if self.shape.float32 { 32 } else { 16 },
                lanes: 1,
            },
            self.shape.rows,
            self.shape.columns,
        )
    }
}

struct BlasState {
    handle: CudaBlas,
    workspace: TrackedCudaSlice<u8>,
}

/// Dense operation dispatcher sharing the canonical provider's ownership graph.
pub struct TensorExecutor {
    provider: Arc<CudaKernelProvider>,
    stream: StreamId,
    blas: Mutex<BlasState>,
}

impl TensorExecutor {
    /// Bind one cuBLAS handle and its budgeted workspace to a provider-owned stream.
    pub fn new(provider: Arc<CudaKernelProvider>) -> Result<Self> {
        if crate::provider::detect_compute_capability(provider.device())? < 80 {
            return Err(failure(
                "BF16 numerical programs require compute capability 8.0 or newer",
            ));
        }
        let runtime = provider
            .memory()
            .runtime()
            .ok_or_else(|| failure("provider has no device runtime"))?;
        let stream = runtime.stream_pool().acquire().map_err(failure)?;
        let cuda_stream = runtime
            .stream_pool()
            .resolve(stream)
            .ok_or_else(|| failure("unresolved stream"))?;
        cuda_stream.context().bind_to_thread().map_err(failure)?;
        let handle = CudaBlas::new(cuda_stream).map_err(failure)?;
        // SAFETY: a newly owned handle; the flag prevents split reductions from
        // narrowing accumulators to the BF16 output type before completion.
        let status = unsafe {
            blas_sys::cublasSetMathMode(
                *handle.handle(),
                blas_sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
            )
        };
        if status != blas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(failure(format!("cuBLAS accumulation mode: {status:?}")));
        }
        let workspace = provider.memory().alloc::<u8>(BLAS_WORKSPACE_BYTES)?;
        // SAFETY: the budgeted allocation is CUDA aligned, retained with this handle,
        // and every GEMM records it as writable on this same immutable stream.
        let status = unsafe {
            blas_sys::cublasSetWorkspace_v2(
                *handle.handle(),
                workspace.device_ptr_value() as usize as *mut c_void,
                BLAS_WORKSPACE_BYTES,
            )
        };
        if status != blas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(failure(format!("cuBLAS workspace: {status:?}")));
        }
        Ok(Self {
            provider,
            stream,
            blas: Mutex::new(BlasState { handle, workspace }),
        })
    }

    /// Allocate one BF16 parameter through the canonical memory manager.
    pub fn allocate_parameter(&self, rows: usize, columns: usize) -> Result<NativeTensor> {
        self.allocate_tensor(TensorValueShape {
            rows,
            columns,
            float32: false,
        })
    }

    /// Allocate checked BF16/FP32 native storage for execution or unpublished restore.
    pub fn allocate_tensor(&self, shape: TensorValueShape) -> Result<NativeTensor> {
        dimension(shape.rows)?;
        dimension(shape.columns)?;
        let bytes = shape
            .rows
            .checked_mul(shape.columns)
            .and_then(|n| n.checked_mul(if shape.float32 { 4 } else { 2 }))
            .ok_or_else(|| failure("tensor byte extent overflow"))?;
        Ok(NativeTensor {
            storage: Arc::new(self.provider.memory().alloc::<u8>(bytes)?),
            shape,
            device: self.provider.device().ordinal(),
        })
    }

    /// Fill an unpublished allocation from a checked checkpoint or snapshot byte range.
    pub fn write_tensor(
        &self,
        tensor: &mut NativeTensor,
        offset: usize,
        bytes: &[u8],
    ) -> Result<()> {
        if tensor.device != self.provider.device().ordinal() {
            return Err(failure("tensor storage does not belong to this device"));
        }
        let end = offset
            .checked_add(bytes.len())
            .filter(|&n| n <= tensor.storage.len())
            .ok_or_else(|| failure("checkpoint write exceeds parameter allocation"))?;
        let storage = Arc::get_mut(&mut tensor.storage)
            .ok_or_else(|| failure("cannot overwrite a published parameter alias"))?;
        let mut view = storage.slice_mut(offset..end);
        self.provider.htod_sync_copy_into_tracked(bytes, &mut view)
    }

    /// Read exact retained bytes through tracked host I/O for durable snapshots.
    pub fn read_tensor(
        &self,
        tensor: &NativeTensor,
        offset: usize,
        bytes: &mut [u8],
    ) -> Result<()> {
        let end = offset
            .checked_add(bytes.len())
            .filter(|&end| end <= tensor.storage.len())
            .ok_or_else(|| failure("snapshot read exceeds tensor allocation"))?;
        if tensor.device != self.provider.device().ordinal() {
            return Err(failure("snapshot tensor is on another device"));
        }
        self.provider
            .dtoh_sync_copy_into_tracked(&tensor.storage.slice(offset..end), bytes)
    }

    fn recorder(&self, inputs: &[&NativeTensor], output: &NativeTensor) -> LaunchRecorder {
        let mut recorder = LaunchRecorder::new_strict(self.stream);
        for input in inputs {
            recorder.read(input.storage.as_ref());
        }
        recorder.write(output.storage.as_ref());
        recorder
    }

    fn launch<P>(
        &self,
        recorder: LaunchRecorder,
        name: &str,
        config: LaunchConfig,
        parameters: P,
    ) -> Result<()>
    where
        CudaFunction: LaunchAsync<P>,
    {
        let function = self
            .provider
            .device()
            .inner()
            .get_func("xlog_tensor", name)
            .ok_or_else(|| failure(format!("missing kernel {name}")))?;
        let runtime = self
            .provider
            .memory()
            .runtime()
            .ok_or_else(|| failure("missing runtime"))?;
        // SAFETY: all callers validate shapes before allocation/launch, use the ABI
        // in tensor.cu, and register every accessed allocation in this recorder.
        // The synchronous closure enqueues only on the admitted stream.
        unsafe {
            recorder.enqueue(runtime, |enqueue| {
                function
                    .launch_in(enqueue, config, parameters)
                    .map_err(failure)
            })
        }
        .map_err(enqueue_error)?
        .commit()
        .map_err(failure)
    }

    fn gemm(
        &self,
        a: &NativeTensor,
        b: &NativeTensor,
        output: &NativeTensor,
        transpose_a: bool,
        transpose_b: bool,
        m: usize,
        n: usize,
        k: usize,
        batches: usize,
    ) -> Result<()> {
        let (m, n, k, batches) = (
            dimension(m)? as i32,
            dimension(n)? as i32,
            dimension(k)? as i32,
            dimension(batches)? as i32,
        );
        let state = self
            .blas
            .lock()
            .map_err(|_| failure("cuBLAS ownership lock poisoned"))?;
        let mut recorder = self.recorder(&[a, b], output);
        recorder.write(&state.workspace);
        let runtime = self
            .provider
            .memory()
            .runtime()
            .ok_or_else(|| failure("missing runtime"))?;
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let lda = if transpose_a { k } else { m };
        let transa = if transpose_a {
            blas_sys::cublasOperation_t::CUBLAS_OP_T
        } else {
            blas_sys::cublasOperation_t::CUBLAS_OP_N
        };
        let transb = if transpose_b {
            blas_sys::cublasOperation_t::CUBLAS_OP_T
        } else {
            blas_sys::cublasOperation_t::CUBLAS_OP_N
        };
        let extent = |rows: i32, columns: i32| {
            (rows as usize)
                .checked_mul(columns as usize)
                .and_then(|n| n.checked_mul(batches as usize))
        };
        if a.shape.float32 != output.shape.float32
            || b.shape.float32 != output.shape.float32
            || Some(a.elements()) != extent(m, k)
            || Some(b.elements()) != extent(k, n)
            || Some(output.elements()) != extent(m, n)
        {
            return Err(failure("GEMM storage type or extent mismatch"));
        }
        let dtype = if output.shape.float32 {
            blas_sys::cudaDataType::CUDA_R_32F
        } else {
            blas_sys::cudaDataType::CUDA_R_16BF
        };
        // SAFETY: execute() has checked every matrix extent and type. This handle
        // is bound permanently to the recorder's stream. Inputs, output, and
        // workspace are retained/registered; host alpha/beta live until enqueue.
        unsafe {
            recorder.enqueue(runtime, |enqueue| {
                enqueue
                    .stream()
                    .context()
                    .bind_to_thread()
                    .map_err(failure)?;
                blas_result::gemm_strided_batched_ex(
                    *state.handle.handle(),
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    (&alpha as *const f32).cast(),
                    a.pointer() as usize as *const c_void,
                    dtype,
                    lda,
                    i64::from(m) * i64::from(k),
                    b.pointer() as usize as *const c_void,
                    dtype,
                    if transpose_b { n } else { k },
                    i64::from(k) * i64::from(n),
                    (&beta as *const f32).cast(),
                    output.pointer() as usize as *mut c_void,
                    dtype,
                    m,
                    i64::from(m) * i64::from(n),
                    batches,
                    if output.shape.float32 {
                        blas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC
                    } else {
                        blas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
                    },
                    blas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
                .map_err(failure)
            })
        }
        .map_err(enqueue_error)?
        .commit()
        .map_err(failure)
    }

    fn attention(
        &self,
        query: &NativeTensor,
        key: &NativeTensor,
        value: &NativeTensor,
        output: &NativeTensor,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        let sequence = query.shape.rows;
        let packed_shape = TensorValueShape {
            rows: heads * sequence,
            columns: head_dim,
            float32: false,
        };
        let q = self.attention_pack(query, heads, heads, head_dim)?;
        let k = self.attention_pack(key, heads, kv_heads, head_dim)?;
        let v = self.attention_pack(value, heads, kv_heads, head_dim)?;
        let scores = self.allocate_tensor(TensorValueShape {
            rows: heads * sequence,
            columns: sequence,
            float32: false,
        })?;
        self.gemm(
            &k, &q, &scores, true, false, sequence, sequence, head_dim, heads,
        )?;
        drop(q);
        drop(k);
        let probabilities = self.attention_probabilities(&scores, sequence, head_dim)?;
        drop(scores);
        let attended = self.allocate_tensor(packed_shape)?;
        self.gemm(
            &v,
            &probabilities,
            &attended,
            false,
            false,
            head_dim,
            sequence,
            sequence,
            heads,
        )?;
        self.launch(
            self.recorder(&[&attended], output),
            "tensor_attention_unpack_bf16",
            grid(output.elements().div_ceil(256))?,
            (
                attended.pointer(),
                output.pointer(),
                output.elements() as u64,
                sequence as u32,
                heads as u32,
                head_dim as u32,
            ),
        )
    }

    /// Execute the validated graph, freeing intermediates after their last consumer.
    /// Input IDs are the caller's original IDs; no text/tokenizer conversion occurs.
    pub fn execute(
        &self,
        program: &TensorProgram,
        parameters: &[NativeTensor],
        token_ids: &[u32],
        prediction_position: usize,
        history_span: Option<(usize, usize)>,
    ) -> Result<NativeTensor> {
        let mut values = self.execute_retaining(
            program,
            parameters,
            token_ids,
            prediction_position,
            history_span,
            &[],
        )?;
        values[program.output]
            .take()
            .ok_or_else(|| failure("missing numerical output"))
    }

    fn execute_retaining(
        &self,
        program: &TensorProgram,
        parameters: &[NativeTensor],
        token_ids: &[u32],
        prediction_position: usize,
        history_span: Option<(usize, usize)>,
        retained: &[bool],
    ) -> Result<Vec<Option<NativeTensor>>> {
        let shapes = program.shapes(token_ids.len(), prediction_position)?;
        if parameters.len() != program.parameters.len()
            || token_ids
                .iter()
                .any(|&id| id as usize >= program.vocabulary)
        {
            return Err(failure(
                "parameter roster or token ID is outside the numerical program",
            ));
        }
        for (parameter, expected) in parameters.iter().zip(&program.parameters) {
            if parameter.storage.len() != expected.bytes()?
                || parameter.shape.float32 != expected.float32
                || parameter.device != self.provider.device().ordinal()
            {
                return Err(failure(
                    "parameter allocation differs from its numerical program",
                ));
            }
        }
        let mut tokens = self.provider.memory().alloc::<u32>(token_ids.len())?;
        self.provider
            .htod_sync_copy_into_tracked(token_ids, &mut tokens)?;
        let mut uses = vec![0usize; program.operations.len()];
        uses[program.output] = 1;
        for (id, &keep) in retained.iter().enumerate() {
            uses[id] += usize::from(keep);
        }
        for operation in &program.operations {
            for input in operation.inputs() {
                uses[input] += 1;
            }
        }
        let mut values: Vec<Option<NativeTensor>> = vec![None; program.operations.len()];
        for (id, operation) in program.operations.iter().enumerate() {
            let output = self.allocate_tensor(shapes[id])?;
            let input = |index: usize| -> Result<&NativeTensor> {
                values[index]
                    .as_ref()
                    .ok_or_else(|| failure("missing live numerical value"))
            };
            let count = output.elements() as u64;
            let vector_grid = grid(output.elements().div_ceil(256))?;
            match *operation {
                Op::Embedding { weight } => {
                    let weight = &parameters[weight];
                    let mut recorder = self.recorder(&[weight], &output);
                    recorder.read(&tokens);
                    self.launch(
                        recorder,
                        "tensor_embedding_bf16",
                        vector_grid,
                        (
                            tokens.device_ptr_value(),
                            weight.pointer(),
                            output.pointer(),
                            count,
                            shapes[id].columns as u32,
                        ),
                    )?;
                }
                Op::Linear {
                    input: source,
                    weight,
                } => {
                    let source = input(source)?;
                    self.gemm(
                        &parameters[weight],
                        source,
                        &output,
                        true,
                        false,
                        output.shape.columns,
                        source.shape.rows,
                        source.shape.columns,
                        1,
                    )?;
                }
                Op::RmsNorm {
                    input: source,
                    weight,
                    epsilon,
                } => {
                    let source = input(source)?;
                    let weight = &parameters[weight];
                    let width = weight.elements();
                    self.launch(
                        self.recorder(&[source, weight], &output),
                        "tensor_rms_norm_bf16",
                        grid(output.elements() / width)?,
                        (
                            source.pointer(),
                            weight.pointer(),
                            output.pointer(),
                            width as u32,
                            epsilon,
                        ),
                    )?;
                }
                Op::Rotary {
                    input: source,
                    heads,
                    head_dim,
                    theta,
                } => {
                    let source = input(source)?;
                    self.launch(
                        self.recorder(&[source], &output),
                        "tensor_rotary_bf16",
                        vector_grid,
                        (
                            source.pointer(),
                            output.pointer(),
                            count,
                            heads as u32,
                            head_dim as u32,
                            theta,
                        ),
                    )?;
                }
                Op::Attention {
                    query,
                    key,
                    value,
                    heads,
                    kv_heads,
                    head_dim,
                } => {
                    self.attention(
                        input(query)?,
                        input(key)?,
                        input(value)?,
                        &output,
                        heads,
                        kv_heads,
                        head_dim,
                    )?;
                }
                Op::Add { left, right }
                | Op::SiluGate {
                    gate: left,
                    value: right,
                } => {
                    let (left, right) = (input(left)?, input(right)?);
                    let name = if matches!(operation, Op::Add { .. }) {
                        if output.shape.float32 {
                            "tensor_accumulate_f32"
                        } else {
                            "tensor_add_bf16"
                        }
                    } else {
                        "tensor_silu_gate_bf16"
                    };
                    self.launch(
                        self.recorder(&[left, right], &output),
                        name,
                        vector_grid,
                        (left.pointer(), right.pointer(), output.pointer(), count),
                    )?;
                }
                Op::PredictionRow { input: source } => {
                    let source = input(source)?;
                    self.launch(
                        self.recorder(&[source], &output),
                        "tensor_prediction_row_bf16",
                        vector_grid,
                        (
                            source.pointer(),
                            output.pointer(),
                            output.shape.columns as u32,
                            prediction_position as u32,
                        ),
                    )?;
                }
                Op::Float32 { input: source } => {
                    let source = input(source)?;
                    self.launch(
                        self.recorder(&[source], &output),
                        "tensor_bf16_to_f32",
                        vector_grid,
                        (source.pointer(), output.pointer(), count),
                    )?;
                }
                _ => self.function_operation(
                    operation,
                    parameters,
                    &values,
                    &tokens,
                    history_span,
                    &output,
                )?,
            }
            values[id] = Some(output);
            for source in operation.inputs() {
                uses[source] -= 1;
                if uses[source] == 0 {
                    values[source] = None;
                }
            }
        }
        let output = values[program.output]
            .as_ref()
            .ok_or_else(|| failure("missing numerical output"))?;
        self.require_finite(output)?;
        Ok(values)
    }

    fn require_finite(&self, tensor: &NativeTensor) -> Result<()> {
        let mut invalid = self.provider.memory().alloc::<u32>(1)?;
        self.provider
            .htod_sync_copy_into_tracked(&[0u32], &mut invalid)?;
        let mut recorder = LaunchRecorder::new_strict(self.stream);
        recorder.read(tensor.storage.as_ref());
        recorder.read_write(&invalid);
        self.launch(
            recorder,
            "tensor_check_finite_f32",
            grid(tensor.elements().div_ceil(256))?,
            (
                tensor.pointer(),
                invalid.device_ptr_value(),
                tensor.elements() as u64,
            ),
        )?;
        let mut flag = [0u32];
        self.provider
            .dtoh_sync_copy_into_tracked(&invalid, &mut flag)?;
        if flag[0] != 0 {
            return Err(failure(
                "numerical execution produced nonfinite FP32 values",
            ));
        }
        Ok(())
    }

    /// Read requested output coordinates, preserving order and repeated indices.
    pub fn read_logits(&self, tensor: &NativeTensor, indices: &[u32]) -> Result<Vec<f32>> {
        if !tensor.shape.float32
            || tensor.shape.rows != 1
            || tensor.device != self.provider.device().ordinal()
            || indices
                .iter()
                .any(|&index| index as usize >= tensor.shape.columns)
        {
            return Err(failure(
                "logit read requires this device's FP32 output and in-range indices",
            ));
        }
        if indices.is_empty() {
            return Ok(Vec::new());
        }
        let count = dimension(indices.len())?;
        let mut selection = self.provider.memory().alloc::<u32>(indices.len())?;
        self.provider
            .htod_sync_copy_into_tracked(indices, &mut selection)?;
        let output = self.provider.memory().alloc::<f32>(indices.len())?;
        let mut recorder = LaunchRecorder::new_strict(self.stream);
        recorder
            .read(tensor.storage.as_ref())
            .read(&selection)
            .write(&output);
        self.launch(
            recorder,
            "tensor_select_f32",
            grid(indices.len().div_ceil(256))?,
            (
                tensor.pointer(),
                selection.device_ptr_value(),
                output.device_ptr_value(),
                count,
            ),
        )?;
        let mut values = vec![0.0f32; indices.len()];
        self.provider
            .dtoh_sync_copy_into_tracked(&output, &mut values)?;
        Ok(values)
    }

    /// Select raw and masked maxima on CUDA, then hash the retained row's exact
    /// little-endian FP32 bytes through tracked host I/O. No indices or Python
    /// vocabulary vector are needed; the complete row still crosses to the host.
    pub fn select_greedy(
        &self,
        tensor: &NativeTensor,
        allowed_mask: Option<&[u8]>,
    ) -> Result<LogitSelection> {
        self.select_logits(tensor, allowed_mask, None)
    }

    /// Temperature-one categorical selection with Philox-addressed Gumbel noise.
    /// The address is supplied by the caller; no device RNG state is advanced.
    pub fn select_categorical(
        &self,
        tensor: &NativeTensor,
        allowed_mask: Option<&[u8]>,
        seed: u64,
        sequence: u64,
        position: u32,
    ) -> Result<LogitSelection> {
        self.select_logits(tensor, allowed_mask, Some((seed, sequence, position)))
    }

    fn select_logits(
        &self,
        tensor: &NativeTensor,
        allowed_mask: Option<&[u8]>,
        sampling: Option<(u64, u64, u32)>,
    ) -> Result<LogitSelection> {
        let law = if sampling.is_some() {
            "categorical"
        } else {
            "greedy"
        };
        if !tensor.shape.float32
            || tensor.shape.rows != 1
            || tensor.device != self.provider.device().ordinal()
        {
            return Err(failure(format!(
                "{law} selection requires this device's FP32 vocabulary row"
            )));
        }
        let vocabulary = dimension(tensor.shape.columns)?;
        if let Some(mask) = allowed_mask {
            let remainder = vocabulary % 8;
            if mask.len() != tensor.shape.columns.div_ceil(8)
                || (remainder != 0 && mask[mask.len() - 1] >> remainder != 0)
                || !mask.iter().any(|&byte| byte != 0)
            {
                return Err(failure(format!(
                    "{law} mask must contain in-range tokens at the exact vocabulary extent"
                )));
            }
        }
        let mask = allowed_mask
            .map(|bytes| {
                let mut mask = self.provider.memory().alloc::<u8>(bytes.len())?;
                self.provider
                    .htod_sync_copy_into_tracked(bytes, &mut mask)?;
                Ok::<_, XlogError>(mask)
            })
            .transpose()?;
        let groups = tensor.shape.columns.div_ceil(256).min(1024);
        let partials = self.provider.memory().alloc::<u32>(groups * 5)?;
        let output = self.provider.memory().alloc::<u32>(5)?;
        let mut recorder = LaunchRecorder::new_strict(self.stream);
        recorder.read(tensor.storage.as_ref()).write(&partials);
        if let Some(mask) = &mask {
            recorder.read(mask);
        }
        self.launch(
            recorder,
            "tensor_greedy_partial_f32",
            grid(groups)?,
            (
                tensor.pointer(),
                mask.as_ref().map_or(0, |mask| mask.device_ptr_value()),
                partials.device_ptr_value(),
                vocabulary,
            ),
        )?;
        let mut recorder = LaunchRecorder::new_strict(self.stream);
        recorder.read(&partials).write(&output);
        self.launch(
            recorder,
            "tensor_greedy_reduce_f32",
            grid(1)?,
            (
                partials.device_ptr_value(),
                output.device_ptr_value(),
                groups as u32,
            ),
        )?;
        let sampled = if let Some((seed, sequence, position)) = sampling {
            let samples = self.provider.memory().alloc::<u64>(groups * 2)?;
            let selected = self.provider.memory().alloc::<u32>(5)?;
            let mut recorder = LaunchRecorder::new_strict(self.stream);
            recorder
                .read(tensor.storage.as_ref())
                .read(&output)
                .write(&samples);
            if let Some(mask) = &mask {
                recorder.read(mask);
            }
            self.launch(
                recorder,
                "tensor_categorical_partial_f32",
                grid(groups)?,
                (
                    tensor.pointer(),
                    mask.as_ref().map_or(0, |mask| mask.device_ptr_value()),
                    output.device_ptr_value(),
                    samples.device_ptr_value(),
                    vocabulary,
                    seed,
                    sequence,
                    position,
                ),
            )?;
            let mut recorder = LaunchRecorder::new_strict(self.stream);
            recorder
                .read(tensor.storage.as_ref())
                .read(&output)
                .read(&samples)
                .write(&selected);
            self.launch(
                recorder,
                "tensor_categorical_reduce_f32",
                grid(1)?,
                (
                    tensor.pointer(),
                    output.device_ptr_value(),
                    samples.device_ptr_value(),
                    selected.device_ptr_value(),
                    groups as u32,
                ),
            )?;
            // Retain both allocations through the final result synchronization.
            Some((samples, selected))
        } else {
            None
        };
        let mut result = [0u32; 5];
        self.provider.dtoh_sync_copy_into_tracked(
            sampled.as_ref().map_or(&output, |(_, selected)| selected),
            &mut result,
        )?;
        if result[4] != 0 {
            return Err(failure(format!(
                "{law} selection requires a finite full-vocabulary row"
            )));
        }
        if result[0] >= vocabulary || result[2] >= vocabulary {
            return Err(failure(format!(
                "{law} selection produced no admissible token"
            )));
        }
        let mut bytes = vec![0u8; tensor.shape.columns * 4];
        self.read_tensor(tensor, 0, &mut bytes)?;
        Ok(LogitSelection {
            raw_argmax: result[0],
            raw_score: f32::from_bits(result[1]),
            selected: result[2],
            selected_score: f32::from_bits(result[3]),
            logits_sha256: Sha256::digest(&bytes).into(),
        })
    }
}
