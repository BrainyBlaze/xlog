//! Self-contained native decoder state with bounded I/O and exact file identity.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
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

const SCHEMA: &str = "xlog.native-causal-snapshot.v2";
static NEXT_TEMPORARY: AtomicU64 = AtomicU64::new(0);

/// The physical down projection and its completed optimizer update count.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizerSnapshot {
    /// Index into the numerical program's shared parameter table.
    pub parameter: usize,
    /// Completed Adam steps; all snapshots occur between complete updates.
    pub step: u64,
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
    /// Present only for a trainable candidate.
    pub optimizer: Option<OptimizerSnapshot>,
    /// Application-owned history, RNG, cursors and budgets, interpreted by that owner.
    pub application_state: Vec<u8>,
}

impl CausalSnapshotDescription {
    /// Capture a cold model version and the application's immutable byte payload.
    pub fn new(
        functions: Arc<FunctionRegistry>,
        entrypoint: String,
        checkpoint_identity: [u8; 32],
        optimizer: Option<OptimizerSnapshot>,
        application_state: Vec<u8>,
    ) -> Self {
        Self {
            schema: SCHEMA.into(),
            functions,
            entrypoint,
            checkpoint_identity,
            optimizer,
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
        program.shapes(1, 0)?;
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
                float32: false,
            });
        }
        if let Some(optimizer) = &self.optimizer {
            let parameter = program
                .parameters
                .get(optimizer.parameter)
                .ok_or_else(|| invalid("snapshot optimizer parameter is absent"))?;
            // Restore the same trainable role accepted by fork_trainable.
            let layer = parameter
                .name
                .strip_prefix("model.layers.")
                .and_then(|name| name.strip_suffix(".mlp.down_proj.weight"))
                .and_then(|name| name.parse::<usize>().ok())
                .ok_or_else(|| invalid("snapshot optimizer must name a decoder down projection"))?;
            if parameter.name != format!("model.layers.{layer}.mlp.down_proj.weight") {
                return Err(invalid(
                    "snapshot optimizer parameter name is not canonical",
                ));
            }
            let &[rows, columns] = parameter.shape.as_slice() else {
                return Err(invalid("snapshot optimizer requires a matrix parameter"));
            };
            for name in [
                "optimizer.master",
                "optimizer.first_moment",
                "optimizer.second_moment",
            ] {
                tensors.push(SnapshotTensor {
                    name: name.into(),
                    rows,
                    columns,
                    float32: true,
                });
            }
        }
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
            description,
            archive,
            expected_identity,
        })
    }

    /// Stream bytes into unpublished allocations, then verify the entire file's
    /// SHA256, including definitions and application state. No model is published here.
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
            return Err(invalid(
                "snapshot SHA256 differs from the expected saved state",
            ));
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
