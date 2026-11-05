//! Shared bounded SafeTensors input for checkpoint import and state restoration.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use safetensors::tensor::Metadata;
use sha2::{Digest, Sha256};
use xlog_core::{Result, XlogError};

pub(crate) const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;
pub(crate) const TRANSFER_CHUNK_BYTES: usize = 16 * 1024 * 1024;

pub(crate) fn invalid(message: impl Into<String>) -> XlogError {
    XlogError::Compilation(message.into())
}

pub(crate) fn io_error(error: std::io::Error) -> XlogError {
    invalid(format!("native tensor archive I/O: {error}"))
}

pub(crate) struct TensorArchive {
    file: File,
    pub(crate) metadata: Metadata,
    header: Vec<u8>,
}

impl TensorArchive {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).map_err(io_error)?;
        let file_bytes = file.metadata().map_err(io_error)?.len();
        let mut length = [0u8; 8];
        file.read_exact(&mut length).map_err(io_error)?;
        let header_bytes = u64::from_le_bytes(length);
        if header_bytes > MAX_HEADER_BYTES
            || header_bytes.checked_add(8).is_none_or(|n| n > file_bytes)
        {
            return Err(invalid("invalid or oversized safetensors header"));
        }
        let mut header = vec![0u8; header_bytes as usize];
        file.read_exact(&mut header).map_err(io_error)?;
        // The official metadata parser checks shapes, dtypes and contiguous offsets.
        let metadata: Metadata = serde_json::from_slice(&header)
            .map_err(|error| invalid(format!("safetensors metadata: {error}")))?;
        if (8 + header_bytes).checked_add(metadata.data_len() as u64) != Some(file_bytes) {
            return Err(invalid(
                "safetensors payload length differs from its validated metadata",
            ));
        }
        Ok(Self {
            file,
            metadata,
            header,
        })
    }

    pub(crate) fn header_digest(&self) -> Sha256 {
        let mut digest = Sha256::new();
        digest.update((self.header.len() as u64).to_le_bytes());
        digest.update(&self.header);
        digest
    }

    pub(crate) fn read_tensor<F>(&mut self, name: &str, mut write: F) -> Result<[u8; 32]>
    where
        F: FnMut(usize, &[u8]) -> Result<()>,
    {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| invalid("archive tensor is absent"))?;
        let (begin, end) = info.data_offsets;
        self.file
            .seek(SeekFrom::Start(8 + self.header.len() as u64 + begin as u64))
            .map_err(io_error)?;
        let mut buffer = vec![0u8; TRANSFER_CHUNK_BYTES.min(end - begin)];
        let mut offset = 0;
        let mut digest = Sha256::new();
        while offset < end - begin {
            let bytes = buffer.len().min(end - begin - offset);
            self.file
                .read_exact(&mut buffer[..bytes])
                .map_err(io_error)?;
            digest.update(&buffer[..bytes]);
            write(offset, &buffer[..bytes])?;
            offset += bytes;
        }
        Ok(digest.finalize().into())
    }
}
