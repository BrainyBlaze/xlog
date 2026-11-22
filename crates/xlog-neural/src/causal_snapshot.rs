//! Self-contained native decoder state with bounded I/O and exact file identity.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use safetensors::tensor::{Dtype, Metadata, TensorInfo};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use xlog_core::Result;
use xlog_ir::tensor::TensorProgram;
use xlog_logic::function::FunctionRegistry;

use crate::tensor_archive::{
    invalid, io_error, TensorArchive, MAX_HEADER_BYTES, TRANSFER_CHUNK_BYTES,
};

const SCHEMA: &str = "xlog.native-causal-snapshot.v3";
static NEXT_TEMPORARY: AtomicU64 = AtomicU64::new(0);

/// A physical parameter and its completed optimizer update count.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizerSnapshot {
    /// Index into the numerical program's shared parameter table.
    pub parameter: usize,
    /// Completed Adam steps; all snapshots occur between complete updates.
    pub step: u64,
}

/// A retained acquired definition and its exact insertion in the canonical decoder.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcquiredInsertion {
    /// Registry name of the feature-producing definition.
    pub function: String,
    /// Zero-based decoder layer after which the residual is inserted.
    pub layer: usize,
    /// Shared physical ID of the FP32 projection.
    pub projection: usize,
    /// Shared physical IDs created by this definition bundle, excluding projection.
    pub parameters: Vec<usize>,
    /// Actual residual operation index in the entry definition.
    pub output: usize,
}

/// Trainable sensor and readout surrounding the checkpoint's causal layers.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GridAdaptation {
    /// FP32 projection from positioned observation features to the residual width.
    pub sensor: usize,
    /// FP32 projection from the normalized query state to the prediction row.
    pub readout: usize,
    /// Frozen log colour transitions and progress intercept. Its presence selects target-centred queries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_prior: Option<usize>,
    /// Zero-based layer after which acquired features enter the residual.
    pub insertion_layer: usize,
}

impl GridAdaptation {
    /// The extra prior input is the action-query indicator, independent of all cell features.
    pub fn has_action_prefix(&self, program: &TensorProgram) -> bool {
        self.color_prior
            .and_then(|id| program.parameters.get(id))
            .is_some_and(|parameter| parameter.shape == [17, 17])
    }

    /// Hidden query state shared by the observation update and the prediction head.
    pub fn prediction_state(&self, program: &TensorProgram) -> Result<usize> {
        use xlog_ir::tensor::TensorOperation as Op;
        let mut projection = program.output;
        if let Some(prior) = self.color_prior {
            let Some(Op::SoftplusColumn { input, column: 16 }) = program.operations.get(projection)
            else {
                return Err(invalid(
                    "grid progress link differs from its calibrated output",
                ));
            };
            let Some(Op::Add { left, right }) = program.operations.get(*input) else {
                return Err(invalid("grid output lacks its colour prior"));
            };
            if !matches!(program.operations.get(*right), Some(Op::Linear { weight, .. }) if *weight == prior)
            {
                return Err(invalid(
                    "grid colour prior differs from its numerical caller",
                ));
            }
            projection = *left;
            if self.has_action_prefix(program) {
                let Some(Op::Select {
                    positive, negative, ..
                }) = program.operations.get(projection)
                else {
                    return Err(invalid("grid output lacks its action-specific readout"));
                };
                let Some(Op::Linear {
                    input: action,
                    weight,
                }) = program.operations.get(*positive)
                else {
                    return Err(invalid("grid progress projection is absent"));
                };
                if *weight != self.readout
                    || !matches!(
                        program.operations.get(*action),
                        Some(Op::Row { row: 0, .. })
                    )
                {
                    return Err(invalid("grid progress must read the first causal position"));
                }
                projection = *negative;
            }
        }
        match program.operations.get(projection) {
            Some(Op::Linear { input, weight }) if *weight == self.readout => Ok(*input),
            _ => Err(invalid("grid state readout differs from its physical role")),
        }
    }
}

/// Parameter ownership of the default editor and its immutable acquisition rules.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditAdaptation {
    /// Name of the ordinary score function, shared by both rule interventions.
    pub base: String,
    /// Parameter IDs for the ordinary editor.
    pub base_parameters: Vec<usize>,
    /// Names and newly owned optimizer slots of all retained search rules.
    pub rules: std::collections::BTreeMap<String, Vec<usize>>,
}

/// The residual connection fitted in a matched comparison and its actual callee.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GridFunctionComparison {
    /// Physical output projection shared at initialization by the two alternatives.
    pub projection: usize,
    /// Feature function called at the selected insertion in this branch.
    pub function: String,
}

/// Continuation roles are explicit and independent of per-parameter Adam counters.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeAdaptation {
    /// Registered editor and its versioned score functions, if initialized.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub editor: Option<EditAdaptation>,
    /// Installed numerical observation boundary, absent in a token checkpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grid: Option<GridAdaptation>,
    /// Selected function and its independently fitted connection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comparison: Option<GridFunctionComparison>,
    /// All retained insertion generations, including frozen predecessors.
    pub insertions: Vec<AcquiredInsertion>,
    /// Current parameters for outcome fitting and optional acquired token training.
    pub trainable: Vec<usize>,
    /// Permitted down projection, selected alone by default token training.
    pub down_projection: Option<usize>,
    /// Number of successfully published minibatch updates on this model lineage.
    pub completed_updates: u64,
}

impl NativeAdaptation {
    fn validate(
        &self,
        program: &TensorProgram,
        functions: &FunctionRegistry,
        optimizer_ids: &std::collections::BTreeSet<usize>,
    ) -> Result<()> {
        use xlog_ir::tensor::{TensorOperation as Op, TensorSignature};
        let mut owned = std::collections::BTreeSet::new();
        let mut outputs = std::collections::BTreeSet::new();
        if let Some(grid) = &self.grid {
            let Some(sensor) = program.parameters.get(grid.sensor) else {
                return Err(invalid("grid sensor is absent"));
            };
            let Some(readout) = program.parameters.get(grid.readout) else {
                return Err(invalid("grid readout is absent"));
            };
            let TensorSignature::Observations {
                input_widths,
                output_width,
                output_rows: 1,
            } = &program.signature
            else {
                return Err(invalid("grid requires a numerical observation entrypoint"));
            };
            let [width, hidden] = input_widths.as_slice() else {
                return Err(invalid("invalid grid inputs"));
            };
            grid.prediction_state(program)?;
            if let Some(id) = grid.color_prior {
                if !program
                    .parameters
                    .get(id)
                    .is_some_and(|p| p.float32 && (p.shape == [17, 16] || p.shape == [17, 17]))
                    || optimizer_ids.contains(&id)
                    || *output_width != 17
                {
                    return Err(invalid(
                        "grid prior requires sixteen colour inputs and an optional action intercept",
                    ));
                }
            }
            if !sensor.float32 || sensor.shape != [*hidden, *width] || !readout.float32
                || readout.shape != [*output_width, *hidden]
                || !program.operations.iter().any(|op| matches!(op, Op::Linear { weight, input } if *weight == grid.sensor && matches!(program.operations.get(*input), Some(Op::Argument { argument: 0 }))))
                || grid.insertion_layer.saturating_add(1) >= program.operations.iter().filter(|op| matches!(op, Op::Attention { .. })).count()
                || !owned.insert(grid.sensor) || !owned.insert(grid.readout)
            { return Err(invalid("grid observation roles differ from their numerical caller")); }
        } else if matches!(program.signature, TensorSignature::Observations { .. }) {
            return Err(invalid("observation entrypoint lacks continuation roles"));
        }
        if let Some(editor) = &self.editor {
            if self.grid.is_none() || editor.rules.is_empty() {
                return Err(invalid("editor has no observation boundary or search rule"));
            }
            let base = &functions.numerical(&editor.base)?.signature;
            if !matches!(base, TensorSignature::Pure { input_widths, output_width: 1 } if input_widths.len() == 2)
            {
                return Err(invalid(
                    "editor score requires state and edit arguments and a scalar result",
                ));
            }
            for (name, ids) in
                std::iter::once((&editor.base, &editor.base_parameters)).chain(&editor.rules)
            {
                if &functions.numerical(name)?.signature != base {
                    return Err(invalid("search rule has a different calling convention"));
                }
                for &id in ids {
                    if !owned.insert(id) || !program.parameters.get(id).is_some_and(|p| p.float32) {
                        return Err(invalid("editor parameter identity or type differs"));
                    }
                }
            }
        }
        for insertion in &self.insertions {
            let projection = program
                .parameters
                .get(insertion.projection)
                .ok_or_else(|| invalid("insertion projection is absent"))?;
            let &[hidden, width] = projection.shape.as_slice() else {
                return Err(invalid("insertion projection must be a matrix"));
            };
            if !projection.float32
                || width != 16
                || !owned.insert(insertion.projection)
                || !outputs.insert(insertion.output)
                || functions.numerical(&insertion.function)?.signature
                    != (TensorSignature::Causal {
                        input_width: hidden,
                        output_width: width,
                    })
            {
                return Err(invalid("insertion type or physical identity differs"));
            }
            let offset = insertion
                .output
                .checked_sub(4)
                .ok_or_else(|| invalid("insertion body is incomplete"))?;
            let body = program
                .operations
                .get(offset..=insertion.output)
                .ok_or_else(|| invalid("insertion body is absent"))?;
            let [Op::Float32 { input }, Op::Call {
                function,
                arguments,
            }, Op::Linear {
                input: feature,
                weight,
            }, Op::Bfloat16 { input: delta }, Op::Add { left, right }] = body
            else {
                return Err(invalid(
                    "insertion does not match its actual numerical call path",
                ));
            };
            let selected = self
                .comparison
                .as_ref()
                .filter(|c| c.projection == insertion.projection);
            let expected_function = selected.map_or(&insertion.function, |c| &c.function);
            if expected_function != &insertion.function
                && functions.numerical(expected_function)?.signature
                    != (TensorSignature::Recurrent {
                        input_width: hidden,
                        output_width: width,
                    })
            {
                return Err(invalid(
                    "comparison recurrence has a different numerical signature",
                ));
            }
            if function != expected_function
                || arguments != &[offset]
                || *feature != offset + 1
                || *weight != insertion.projection
                || *delta != offset + 2
                || *left != *input
                || *right != offset + 3
            {
                return Err(invalid("insertion operands differ from the saved roles"));
            }
            if let Some(previous) = self
                .insertions
                .iter()
                .find(|previous| previous.output == *input)
            {
                if previous.layer != insertion.layer || previous.output >= insertion.output {
                    return Err(invalid("insertion layer chain is inconsistent"));
                }
            } else {
                let Some(Op::Add {
                    right: projected, ..
                }) = program.operations.get(*input)
                else {
                    return Err(invalid(
                        "insertion does not follow a decoder layer residual",
                    ));
                };
                let Some(Op::Linear { weight, .. }) = program.operations.get(*projected) else {
                    return Err(invalid("insertion residual has no down projection"));
                };
                if program.parameters[*weight].name
                    != format!("model.layers.{}.mlp.down_proj.weight", insertion.layer)
                {
                    return Err(invalid(
                        "insertion's declared layer differs from its actual caller",
                    ));
                }
            }
            for &id in &insertion.parameters {
                if !owned.insert(id)
                    || !program
                        .parameters
                        .get(id)
                        .is_some_and(|parameter| parameter.float32)
                {
                    return Err(invalid(
                        "acquired parameter ownership is duplicated or has a different type",
                    ));
                }
            }
        }
        for &id in optimizer_ids {
            let parameter = &program.parameters[id];
            if parameter.float32 {
                if !owned.contains(&id) {
                    return Err(invalid("FP32 optimizer has no acquired owner"));
                }
            } else if !parameter.name.starts_with("model.layers.")
                || !parameter.name.ends_with(".mlp.down_proj.weight")
                || !program.operations.iter().any(
                    |operation| matches!(operation, Op::Linear { weight, .. } if *weight == id),
                )
            {
                return Err(invalid(
                    "BF16 optimizer does not own a decoder down projection",
                ));
            }
        }
        if owned.iter().any(|id| !optimizer_ids.contains(id)) {
            return Err(invalid("acquired optimizer state is incomplete"));
        }
        let mut expected = std::collections::BTreeSet::new();
        if let Some(grid) = &self.grid {
            expected.extend([grid.sensor, grid.readout]);
        }
        if let Some(last) = self.insertions.last() {
            expected.extend(last.parameters.iter().copied());
            expected.insert(last.projection);
        }
        if let Some(id) = self.down_projection {
            if !optimizer_ids.contains(&id) || program.parameters[id].float32 {
                return Err(invalid("trainable down projection role is invalid"));
            }
            expected.insert(id);
        }
        if let Some(comparison) = &self.comparison {
            if self.grid.is_none()
                || self.down_projection.is_some()
                || !self
                    .insertions
                    .iter()
                    .any(|entry| entry.projection == comparison.projection)
            {
                return Err(invalid(
                    "comparison requires an installed grid feature connection",
                ));
            }
            expected = std::collections::BTreeSet::from([comparison.projection]);
        }
        let actual = self
            .trainable
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        if actual.len() != self.trainable.len()
            || expected != actual
            || actual.iter().any(|id| !optimizer_ids.contains(id))
        {
            return Err(invalid(
                "active trainable IDs differ from their retained roles",
            ));
        }
        Ok(())
    }
}

/// All native definitions and caller-owned continuation bytes in one snapshot.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CausalSnapshotDescription {
    schema: String,
    /// The actual canonical definitions, including numerical bodies and signatures.
    pub functions: Arc<FunctionRegistry>,
    /// Name resolved in the common function registry for forward and differentiation.
    pub entrypoint: String,
    /// Provenance of the original import, distinct from this file's identity.
    pub checkpoint_identity: [u8; 32],
    /// All retained optimizer owners, including frozen predecessor definitions.
    pub optimizers: Vec<OptimizerSnapshot>,
    /// Actual insertion and update roles for continuation and consumer inspection.
    pub adaptation: NativeAdaptation,
    /// Application-owned history, RNG, cursors and budgets, interpreted by that owner.
    pub application_state: Vec<u8>,
}

impl CausalSnapshotDescription {
    /// Capture a cold model version and the application's immutable byte payload.
    pub fn new(
        functions: Arc<FunctionRegistry>,
        entrypoint: String,
        checkpoint_identity: [u8; 32],
        optimizers: Vec<OptimizerSnapshot>,
        adaptation: NativeAdaptation,
        application_state: Vec<u8>,
    ) -> Self {
        Self {
            schema: SCHEMA.into(),
            functions,
            entrypoint,
            checkpoint_identity,
            optimizers,
            adaptation,
            application_state,
        }
    }

    /// Resolve the saved entry point instead of trusting a second plan copy.
    pub fn program(&self) -> Result<&TensorProgram> {
        self.functions.numerical(&self.entrypoint)
    }

    /// Storage roster in exactly the order passed to archive readers and writers.
    pub fn tensors(&self) -> Result<Vec<SnapshotTensor>> {
        if self.schema != SCHEMA {
            return Err(invalid(
                "unsupported native snapshot schema or numerical convention",
            ));
        }
        self.functions.validate()?;
        let program = self.program()?;
        self.functions
            .compile_numerical(&self.entrypoint, 1, 0, Some((0, 1)))?;
        let mut tensors = Vec::with_capacity(program.parameters.len() + 3);
        for (id, parameter) in program.parameters.iter().enumerate() {
            let (rows, columns) = match parameter.shape.as_slice() {
                &[columns] => (1, columns),
                &[rows, columns] => (rows, columns),
                _ => return Err(invalid("snapshot parameter must be a vector or matrix")),
            };
            tensors.push(SnapshotTensor {
                name: format!("parameter.{id}"),
                rows,
                columns,
                float32: parameter.float32,
            });
        }
        let mut optimizer_ids = std::collections::BTreeSet::new();
        for optimizer in &self.optimizers {
            if !optimizer_ids.insert(optimizer.parameter)
                || optimizer.step > self.adaptation.completed_updates
            {
                return Err(invalid(
                    "snapshot optimizer identities or counters are inconsistent",
                ));
            }
            let parameter = program
                .parameters
                .get(optimizer.parameter)
                .ok_or_else(|| invalid("snapshot optimizer parameter is absent"))?;
            let (rows, columns) = match parameter.shape.as_slice() {
                &[columns] => (1, columns),
                &[rows, columns] => (rows, columns),
                _ => return Err(invalid("snapshot optimizer requires a vector or matrix")),
            };
            for name in [
                "optimizer.master",
                "optimizer.first_moment",
                "optimizer.second_moment",
            ] {
                tensors.push(SnapshotTensor {
                    name: format!("{name}.{}", optimizer.parameter),
                    rows,
                    columns,
                    float32: true,
                });
            }
        }
        self.adaptation
            .validate(program, &self.functions, &optimizer_ids)?;
        Ok(tensors)
    }
}

/// Physical layout of one tensor in the snapshot's ordered storage roster.
pub struct SnapshotTensor {
    name: String,
    /// Contiguous row count.
    pub rows: usize,
    /// Contiguous column count.
    pub columns: usize,
    /// FP32 optimizer state, or BF16 model parameters when false.
    pub float32: bool,
}

impl SnapshotTensor {
    /// Checked allocation and payload extent.
    pub fn bytes(&self) -> Result<usize> {
        self.rows
            .checked_mul(self.columns)
            .and_then(|n| n.checked_mul(if self.float32 { 4 } else { 2 }))
            .ok_or_else(|| invalid("snapshot tensor byte extent overflow"))
    }
}

/// A validated description and the still-unpublished stream of native tensor bytes.
pub struct CausalDecoderSnapshot {
    path: PathBuf,
    description: CausalSnapshotDescription,
    archive: TensorArchive,
    expected_identity: [u8; 32],
}

impl CausalDecoderSnapshot {
    /// Immutable description used to admit storage before reading payload bytes.
    pub fn description(&self) -> &CausalSnapshotDescription {
        &self.description
    }

    /// Validate the archive roster before native allocation. Content identity is
    /// checked by load after all bytes have actually been consumed.
    pub fn open(path: &Path, expected_identity: [u8; 32]) -> Result<Self> {
        let archive = TensorArchive::open(path)?;
        let metadata = archive
            .metadata
            .metadata()
            .as_ref()
            .filter(|metadata| metadata.len() == 1)
            .and_then(|metadata| metadata.get("xlog"))
            .ok_or_else(|| invalid("snapshot requires its exact native description"))?;
        let description: CausalSnapshotDescription = serde_json::from_str(metadata)
            .map_err(|error| invalid(format!("native snapshot description: {error}")))?;
        let tensors = description.tensors()?;
        let names = archive.metadata.offset_keys();
        if names.len() != tensors.len() {
            return Err(invalid(
                "snapshot tensor roster is incomplete or has extra entries",
            ));
        }
        for (name, tensor) in names.iter().zip(&tensors) {
            let info = archive
                .metadata
                .info(name)
                .ok_or_else(|| invalid("snapshot tensor metadata is absent"))?;
            if name != &tensor.name
                || info.shape != [tensor.rows, tensor.columns]
                || info.dtype
                    != if tensor.float32 {
                        Dtype::F32
                    } else {
                        Dtype::BF16
                    }
            {
                return Err(invalid(
                    "snapshot tensor order, shape or dtype differs from its program",
                ));
            }
        }
        Ok(Self {
            path: path.to_path_buf(),
            description,
            archive,
            expected_identity,
        })
    }

    /// Stream bytes into unpublished allocations, then verify the entire file's
    /// SHA256, including definitions and application state. No model is published here.
    /// A mismatch reports the source path and both expected and observed digests.
    pub fn load<F>(mut self, mut write: F) -> Result<CausalSnapshotDescription>
    where
        F: FnMut(usize, usize, &[u8]) -> Result<()>,
    {
        let mut digest = self.archive.header_digest();
        for (id, name) in self.archive.metadata.offset_keys().iter().enumerate() {
            self.archive.read_tensor(name, |offset, bytes| {
                digest.update(bytes);
                write(id, offset, bytes)
            })?;
        }
        let actual: [u8; 32] = digest.finalize().into();
        if actual != self.expected_identity {
            return Err(invalid(format!(
                "snapshot SHA256 differs from the expected saved state for {}: expected {}, actual {}",
                self.path.display(),
                hex(&self.expected_identity),
                hex(&actual)
            )));
        }
        Ok(self.description)
    }

    /// Stream a complete model version into a new file and atomically publish it.
    /// Existing destinations are never replaced; the returned digest covers all bytes.
    pub fn write<F>(
        path: &Path,
        description: &CausalSnapshotDescription,
        mut read: F,
    ) -> Result<[u8; 32]>
    where
        F: FnMut(usize, usize, &mut [u8]) -> Result<()>,
    {
        let tensors = description.tensors()?;
        let mut offset = 0usize;
        let mut entries = Vec::with_capacity(tensors.len());
        for tensor in &tensors {
            let end = offset
                .checked_add(tensor.bytes()?)
                .ok_or_else(|| invalid("snapshot payload extent overflow"))?;
            entries.push((
                tensor.name.clone(),
                TensorInfo {
                    dtype: if tensor.float32 {
                        Dtype::F32
                    } else {
                        Dtype::BF16
                    },
                    shape: vec![tensor.rows, tensor.columns],
                    data_offsets: (offset, end),
                },
            ));
            offset = end;
        }
        let description =
            serde_json::to_string(description).map_err(|error| invalid(error.to_string()))?;
        let metadata = Metadata::new(Some(HashMap::from([("xlog".into(), description)])), entries)
            .map_err(|error| invalid(error.to_string()))?;
        let mut header =
            serde_json::to_vec(&metadata).map_err(|error| invalid(error.to_string()))?;
        header.resize(header.len().next_multiple_of(8), b' ');
        if header.len() as u64 > MAX_HEADER_BYTES {
            return Err(invalid(
                "native snapshot description exceeds the header bound",
            ));
        }
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        if path.file_name().is_none() {
            return Err(invalid("snapshot destination must name a new file"));
        }
        match fs::symlink_metadata(path) {
            Ok(_) => return Err(invalid("snapshot destination already exists")),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
            Err(error) => return Err(io_error(error)),
        }
        let epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| invalid(error.to_string()))?
            .as_nanos();
        let temporary = parent.join(format!(
            ".xlog-snapshot-{}-{epoch}-{}.tmp",
            std::process::id(),
            NEXT_TEMPORARY.fetch_add(1, Ordering::Relaxed)
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temporary).map_err(io_error)?;
        let result = (|| -> Result<[u8; 32]> {
            let length = (header.len() as u64).to_le_bytes();
            file.write_all(&length).map_err(io_error)?;
            file.write_all(&header).map_err(io_error)?;
            let mut digest = Sha256::new();
            digest.update(length);
            digest.update(&header);
            let mut buffer = vec![0u8; TRANSFER_CHUNK_BYTES];
            for (id, tensor) in tensors.iter().enumerate() {
                let bytes = tensor.bytes()?;
                let mut offset = 0;
                while offset < bytes {
                    let count = buffer.len().min(bytes - offset);
                    read(id, offset, &mut buffer[..count])?;
                    file.write_all(&buffer[..count]).map_err(io_error)?;
                    digest.update(&buffer[..count]);
                    offset += count;
                }
            }
            file.sync_all().map_err(io_error)?;
            // Linking a same-directory temporary is atomic and refuses an existing
            // destination, unlike rename, which would silently replace it.
            fs::hard_link(&temporary, path).map_err(io_error)?;
            Ok(digest.finalize().into())
        })();
        drop(file);
        let cleanup = fs::remove_file(&temporary);
        match (result, cleanup) {
            (Ok(identity), Ok(())) => {
                File::open(parent).and_then(|directory| directory.sync_all()).map_err(|error| invalid(format!(
                    "snapshot published at {} with SHA256 {}; directory sync failed: {error}", path.display(), hex(&identity))))?;
                Ok(identity)
            }
            (Err(error), Ok(())) => Err(error),
            (Ok(identity), Err(error)) => Err(invalid(format!(
                "snapshot published at {} with SHA256 {}; temporary {} cleanup failed: {error}",
                path.display(),
                hex(&identity),
                temporary.display()
            ))),
            (Err(error), Err(cleanup)) => Err(invalid(format!(
                "{error}; temporary {} cleanup failed: {cleanup}",
                temporary.display()
            ))),
        }
    }
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
