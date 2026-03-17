//! On-disk circuit artifact cache.
//!
//! Caches compiled circuit topology so d-DNNF compilation is skipped on warm starts.
//! Stores topology and metadata only -- weights change per query.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use xlog_core::{Result, XlogError};

const MAGIC: u32 = 0x584C4743; // "XLGC"
const FORMAT_VERSION: u32 = 1;

/// Header size in bytes:
/// magic(4) + version(4) + cnf_hash(8) + config_hash(8) + random_vars_hash(8) +
/// sm(4) + num_nodes(4) + num_edges(4) + num_levels(4) + root(4) + max_var(4) +
/// has_free_var_mask(1) + padding(3) = 60
const HEADER_SIZE: usize = 60;

#[derive(Debug, Clone)]
pub(crate) struct CircuitCacheKey {
    pub cnf_hash: u64,
    pub config_hash: u64,
    pub random_vars_hash: u64,
    pub sm: u32,
}

#[derive(Debug)]
pub(crate) struct CircuitArtifact {
    pub num_nodes: u32,
    pub num_edges: u32,
    pub num_levels: u32,
    pub root: u32,
    pub max_var: u32,
    pub has_free_var_mask: bool,
    pub node_type: Vec<u8>,
    pub child_offsets: Vec<u32>,
    pub child_indices: Vec<u32>,
    pub lit: Vec<i32>,
    pub decision_var: Vec<u32>,
    pub decision_child_false: Vec<u32>,
    pub decision_child_true: Vec<u32>,
    pub level_nodes: Vec<u32>,
    pub level_offsets: Vec<u32>,
    pub free_var_mask: Vec<u8>,
}

/// Resolve the cache directory.
///
/// Priority: `XLOG_CIRCUIT_CACHE_DIR` env var > `XDG_CACHE_HOME`/xlog/circuits >
/// `HOME`/.cache/xlog/circuits.
pub(crate) fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("XLOG_CIRCUIT_CACHE_DIR") {
        PathBuf::from(dir)
    } else if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        PathBuf::from(xdg).join("xlog").join("circuits")
    } else {
        PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join(".cache")
            .join("xlog")
            .join("circuits")
    }
}

fn artifact_filename(key: &CircuitCacheKey) -> String {
    format!(
        "{:016x}_{:016x}_{:016x}_{:08x}_{:08x}.bin",
        key.cnf_hash, key.config_hash, key.random_vars_hash, key.sm, FORMAT_VERSION,
    )
}

/// Write a circuit artifact to the on-disk cache.
///
/// Uses atomic rename (write to `.tmp`, then rename) so readers never see a partial file.
pub(crate) fn write_artifact(key: &CircuitCacheKey, artifact: &CircuitArtifact) -> Result<()> {
    write_artifact_to(&cache_dir(), key, artifact)
}

/// Read a circuit artifact from the on-disk cache.
///
/// Returns `Ok(None)` if the cache file does not exist or fails validation (stale entry).
/// Returns `Err` only on genuine IO errors after the file has been opened.
pub(crate) fn read_artifact(key: &CircuitCacheKey) -> Result<Option<CircuitArtifact>> {
    read_artifact_from(&cache_dir(), key)
}

// ---------------------------------------------------------------------------
// Internal implementations that accept an explicit directory (testable).
// ---------------------------------------------------------------------------

fn io_err(e: std::io::Error) -> XlogError {
    XlogError::Compilation(format!("circuit cache IO error: {}", e))
}

/// Read `count` little-endian u32 values from `data` starting at `offset`.
///
/// Unlike `bytemuck::cast_slice`, this does not require 4-byte alignment of the
/// source slice, which matters when the preceding u8 array (`node_type`) has a
/// length that is not a multiple of 4.
fn read_u32_vec(data: &[u8], offset: usize, count: usize) -> Vec<u32> {
    (0..count)
        .map(|i| {
            let s = offset + i * 4;
            u32::from_le_bytes([data[s], data[s + 1], data[s + 2], data[s + 3]])
        })
        .collect()
}

/// Read `count` little-endian i32 values from `data` starting at `offset`.
fn read_i32_vec(data: &[u8], offset: usize, count: usize) -> Vec<i32> {
    (0..count)
        .map(|i| {
            let s = offset + i * 4;
            i32::from_le_bytes([data[s], data[s + 1], data[s + 2], data[s + 3]])
        })
        .collect()
}

fn write_artifact_to(dir: &Path, key: &CircuitCacheKey, artifact: &CircuitArtifact) -> Result<()> {
    fs::create_dir_all(dir).map_err(io_err)?;

    let path = dir.join(artifact_filename(key));
    let tmp = path.with_extension("tmp");
    let mut f = fs::File::create(&tmp).map_err(io_err)?;

    // Write header
    f.write_all(&MAGIC.to_le_bytes()).map_err(io_err)?;
    f.write_all(&FORMAT_VERSION.to_le_bytes()).map_err(io_err)?;
    f.write_all(&key.cnf_hash.to_le_bytes()).map_err(io_err)?;
    f.write_all(&key.config_hash.to_le_bytes())
        .map_err(io_err)?;
    f.write_all(&key.random_vars_hash.to_le_bytes())
        .map_err(io_err)?;
    f.write_all(&key.sm.to_le_bytes()).map_err(io_err)?;
    f.write_all(&artifact.num_nodes.to_le_bytes())
        .map_err(io_err)?;
    f.write_all(&artifact.num_edges.to_le_bytes())
        .map_err(io_err)?;
    f.write_all(&artifact.num_levels.to_le_bytes())
        .map_err(io_err)?;
    f.write_all(&artifact.root.to_le_bytes()).map_err(io_err)?;
    f.write_all(&artifact.max_var.to_le_bytes())
        .map_err(io_err)?;
    f.write_all(&[artifact.has_free_var_mask as u8])
        .map_err(io_err)?;
    f.write_all(&[0u8; 3]).map_err(io_err)?; // padding

    // Write arrays as raw bytes
    f.write_all(&artifact.node_type).map_err(io_err)?;
    f.write_all(bytemuck::cast_slice(&artifact.child_offsets))
        .map_err(io_err)?;
    f.write_all(bytemuck::cast_slice(&artifact.child_indices))
        .map_err(io_err)?;
    f.write_all(bytemuck::cast_slice(&artifact.lit))
        .map_err(io_err)?;
    f.write_all(bytemuck::cast_slice(&artifact.decision_var))
        .map_err(io_err)?;
    f.write_all(bytemuck::cast_slice(&artifact.decision_child_false))
        .map_err(io_err)?;
    f.write_all(bytemuck::cast_slice(&artifact.decision_child_true))
        .map_err(io_err)?;
    f.write_all(bytemuck::cast_slice(&artifact.level_nodes))
        .map_err(io_err)?;
    f.write_all(bytemuck::cast_slice(&artifact.level_offsets))
        .map_err(io_err)?;
    f.write_all(&artifact.free_var_mask).map_err(io_err)?;

    drop(f);
    fs::rename(&tmp, &path).map_err(io_err)?;

    evict_if_needed_in(dir)?;
    Ok(())
}

fn read_artifact_from(dir: &Path, key: &CircuitCacheKey) -> Result<Option<CircuitArtifact>> {
    let path = dir.join(artifact_filename(key));

    let data = match fs::read(&path) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(XlogError::Compilation(format!(
                "Failed to read cache file: {}",
                e
            )))
        }
    };

    parse_artifact(&data, key)
}

/// Parse a circuit artifact from raw file bytes, validating header and key.
fn parse_artifact(data: &[u8], key: &CircuitCacheKey) -> Result<Option<CircuitArtifact>> {
    // 1. Check minimum header size
    if data.len() < HEADER_SIZE {
        return Ok(None);
    }

    let mut cursor = 0usize;

    macro_rules! read_u32 {
        () => {{
            let val = u32::from_le_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
            ]);
            cursor += 4;
            val
        }};
    }

    macro_rules! read_u64 {
        () => {{
            let val = u64::from_le_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
                data[cursor + 4],
                data[cursor + 5],
                data[cursor + 6],
                data[cursor + 7],
            ]);
            cursor += 8;
            val
        }};
    }

    // 2. Validate magic
    let magic = read_u32!();
    if magic != MAGIC {
        return Ok(None);
    }

    // 3. Validate format version
    let version = read_u32!();
    if version != FORMAT_VERSION {
        return Ok(None);
    }

    // 4. Validate key fields
    let cnf_hash = read_u64!();
    let config_hash = read_u64!();
    let random_vars_hash = read_u64!();
    let sm = read_u32!();

    if cnf_hash != key.cnf_hash
        || config_hash != key.config_hash
        || random_vars_hash != key.random_vars_hash
        || sm != key.sm
    {
        return Ok(None);
    }

    // 5. Parse metadata from header
    let num_nodes = read_u32!();
    let num_edges = read_u32!();
    let num_levels = read_u32!();
    let root = read_u32!();
    let max_var = read_u32!();
    let has_free_var_mask = data[cursor] != 0;
    cursor += 1;
    cursor += 3; // skip padding

    debug_assert_eq!(cursor, HEADER_SIZE);

    // 6. Calculate expected array sizes
    let node_type_bytes = num_nodes as usize;
    let child_offsets_elems = (num_nodes as usize) + 1;
    let child_offsets_bytes = child_offsets_elems * 4;
    let child_indices_bytes = (num_edges as usize) * 4;
    let lit_bytes = (num_nodes as usize) * 4;
    let decision_var_bytes = (num_nodes as usize) * 4;
    let decision_child_false_bytes = (num_nodes as usize) * 4;
    let decision_child_true_bytes = (num_nodes as usize) * 4;
    let level_nodes_bytes = (num_nodes as usize) * 4;
    let level_offsets_elems = (num_levels as usize) + 1;
    let level_offsets_bytes = level_offsets_elems * 4;
    let free_var_mask_bytes = (max_var as usize) + 1;

    let expected_total = HEADER_SIZE
        + node_type_bytes
        + child_offsets_bytes
        + child_indices_bytes
        + lit_bytes
        + decision_var_bytes
        + decision_child_false_bytes
        + decision_child_true_bytes
        + level_nodes_bytes
        + level_offsets_bytes
        + free_var_mask_bytes;

    if data.len() < expected_total {
        return Ok(None);
    }

    // 7. Parse arrays from raw bytes.
    //
    // We use from_le_bytes helpers instead of bytemuck::cast_slice because
    // the cursor may not be 4-byte aligned after reading the u8 node_type
    // array (e.g. when num_nodes is not a multiple of 4).
    let node_type = data[cursor..cursor + node_type_bytes].to_vec();
    cursor += node_type_bytes;

    let child_offsets = read_u32_vec(data, cursor, child_offsets_elems);
    cursor += child_offsets_bytes;

    let child_indices = read_u32_vec(data, cursor, num_edges as usize);
    cursor += child_indices_bytes;

    let lit = read_i32_vec(data, cursor, num_nodes as usize);
    cursor += lit_bytes;

    let decision_var = read_u32_vec(data, cursor, num_nodes as usize);
    cursor += decision_var_bytes;

    let decision_child_false = read_u32_vec(data, cursor, num_nodes as usize);
    cursor += decision_child_false_bytes;

    let decision_child_true = read_u32_vec(data, cursor, num_nodes as usize);
    cursor += decision_child_true_bytes;

    let level_nodes = read_u32_vec(data, cursor, num_nodes as usize);
    cursor += level_nodes_bytes;

    let level_offsets = read_u32_vec(data, cursor, level_offsets_elems);
    cursor += level_offsets_bytes;

    let free_var_mask = data[cursor..cursor + free_var_mask_bytes].to_vec();
    // cursor += free_var_mask_bytes; // not needed after last read

    Ok(Some(CircuitArtifact {
        num_nodes,
        num_edges,
        num_levels,
        root,
        max_var,
        has_free_var_mask,
        node_type,
        child_offsets,
        child_indices,
        lit,
        decision_var,
        decision_child_false,
        decision_child_true,
        level_nodes,
        level_offsets,
        free_var_mask,
    }))
}

/// Evict oldest cache entries when total cache size exceeds the limit.
///
/// Default limit is 512 MB, configurable via `XLOG_CIRCUIT_CACHE_MAX_MB`.
fn evict_if_needed_in(dir: &Path) -> Result<()> {
    let max_mb: u64 = std::env::var("XLOG_CIRCUIT_CACHE_MAX_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    evict_if_needed_in_with_limit(dir, max_mb)
}

fn evict_if_needed_in_with_limit(dir: &Path, max_mb: u64) -> Result<()> {
    let max_bytes = max_mb * 1024 * 1024;

    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Ok(()), // directory gone or unreadable, nothing to evict
    };

    // Collect .bin files with their size and mtime.
    let mut files: Vec<(PathBuf, u64, std::time::SystemTime)> = Vec::new();
    let mut total_size: u64 = 0;

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("bin") {
            continue;
        }
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        if !meta.is_file() {
            continue;
        }
        let size = meta.len();
        let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
        total_size += size;
        files.push((path, size, mtime));
    }

    if total_size <= max_bytes {
        return Ok(());
    }

    // Sort by mtime ascending (oldest first)
    files.sort_by_key(|&(_, _, mtime)| mtime);

    for (path, size, _) in &files {
        if total_size <= max_bytes {
            break;
        }
        // Best-effort: ignore errors when removing individual files during eviction.
        if fs::remove_file(path).is_ok() {
            total_size = total_size.saturating_sub(*size);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    // Use a unique directory per test to avoid interference between parallel tests.
    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn test_cache_dir() -> PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let dir = std::env::temp_dir()
            .join("xlog_disk_cache_test")
            .join(format!("{}_{}", pid, id));
        fs::create_dir_all(&dir).expect("create test cache dir");
        dir
    }

    fn make_key(cnf_hash: u64) -> CircuitCacheKey {
        CircuitCacheKey {
            cnf_hash,
            config_hash: 0xDEADBEEF,
            random_vars_hash: 0xCAFEBABE,
            sm: 89,
        }
    }

    fn make_artifact() -> CircuitArtifact {
        // A small circuit: 4 nodes, 3 edges, 2 levels, max_var=2
        CircuitArtifact {
            num_nodes: 4,
            num_edges: 3,
            num_levels: 2,
            root: 0,
            max_var: 2,
            has_free_var_mask: true,
            node_type: vec![1, 2, 3, 4],
            child_offsets: vec![0, 1, 2, 3, 3], // num_nodes + 1 = 5
            child_indices: vec![1, 2, 3],       // num_edges = 3
            lit: vec![0, 1, -1, 2],             // num_nodes = 4
            decision_var: vec![0, 1, 2, 0],
            decision_child_false: vec![0, 2, 3, 0],
            decision_child_true: vec![0, 1, 3, 0],
            level_nodes: vec![0, 1, 2, 3], // num_nodes = 4
            level_offsets: vec![0, 1, 4],  // num_levels + 1 = 3
            free_var_mask: vec![0, 1, 0],  // max_var + 1 = 3
        }
    }

    #[test]
    fn test_roundtrip() {
        let dir = test_cache_dir();

        let key = make_key(0x1234);
        let artifact = make_artifact();

        write_artifact_to(&dir, &key, &artifact).expect("write should succeed");

        let loaded = read_artifact_from(&dir, &key)
            .expect("read should not error")
            .expect("read should return Some");

        assert_eq!(loaded.num_nodes, artifact.num_nodes);
        assert_eq!(loaded.num_edges, artifact.num_edges);
        assert_eq!(loaded.num_levels, artifact.num_levels);
        assert_eq!(loaded.root, artifact.root);
        assert_eq!(loaded.max_var, artifact.max_var);
        assert_eq!(loaded.has_free_var_mask, artifact.has_free_var_mask);
        assert_eq!(loaded.node_type, artifact.node_type);
        assert_eq!(loaded.child_offsets, artifact.child_offsets);
        assert_eq!(loaded.child_indices, artifact.child_indices);
        assert_eq!(loaded.lit, artifact.lit);
        assert_eq!(loaded.decision_var, artifact.decision_var);
        assert_eq!(loaded.decision_child_false, artifact.decision_child_false);
        assert_eq!(loaded.decision_child_true, artifact.decision_child_true);
        assert_eq!(loaded.level_nodes, artifact.level_nodes);
        assert_eq!(loaded.level_offsets, artifact.level_offsets);
        assert_eq!(loaded.free_var_mask, artifact.free_var_mask);

        let _ = fs::remove_dir_all(&dir);
    }

    /// Regression test: num_nodes=5 is not a multiple of 4, so the u32
    /// arrays after node_type start at a non-4-byte-aligned offset.
    /// The old bytemuck::cast_slice read path would panic here.
    #[test]
    fn test_roundtrip_unaligned_num_nodes() {
        let dir = test_cache_dir();

        let key = make_key(0x5555);
        // 5 nodes, 4 edges, 3 levels, max_var=3
        let artifact = CircuitArtifact {
            num_nodes: 5,
            num_edges: 4,
            num_levels: 3,
            root: 0,
            max_var: 3,
            has_free_var_mask: false,
            node_type: vec![1, 2, 3, 4, 5],
            child_offsets: vec![0, 1, 2, 3, 4, 4], // num_nodes + 1 = 6
            child_indices: vec![1, 2, 3, 4],       // num_edges = 4
            lit: vec![0, 1, -1, 2, -2],            // num_nodes = 5
            decision_var: vec![0, 1, 2, 3, 0],
            decision_child_false: vec![0, 2, 3, 4, 0],
            decision_child_true: vec![0, 1, 3, 4, 0],
            level_nodes: vec![0, 1, 2, 3, 4], // num_nodes = 5
            level_offsets: vec![0, 1, 3, 5],  // num_levels + 1 = 4
            free_var_mask: vec![0, 1, 0, 1],  // max_var + 1 = 4
        };

        write_artifact_to(&dir, &key, &artifact).expect("write should succeed");

        let loaded = read_artifact_from(&dir, &key)
            .expect("read should not error")
            .expect("read should return Some");

        assert_eq!(loaded.num_nodes, artifact.num_nodes);
        assert_eq!(loaded.num_edges, artifact.num_edges);
        assert_eq!(loaded.num_levels, artifact.num_levels);
        assert_eq!(loaded.root, artifact.root);
        assert_eq!(loaded.max_var, artifact.max_var);
        assert_eq!(loaded.has_free_var_mask, artifact.has_free_var_mask);
        assert_eq!(loaded.node_type, artifact.node_type);
        assert_eq!(loaded.child_offsets, artifact.child_offsets);
        assert_eq!(loaded.child_indices, artifact.child_indices);
        assert_eq!(loaded.lit, artifact.lit);
        assert_eq!(loaded.decision_var, artifact.decision_var);
        assert_eq!(loaded.decision_child_false, artifact.decision_child_false);
        assert_eq!(loaded.decision_child_true, artifact.decision_child_true);
        assert_eq!(loaded.level_nodes, artifact.level_nodes);
        assert_eq!(loaded.level_offsets, artifact.level_offsets);
        assert_eq!(loaded.free_var_mask, artifact.free_var_mask);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_mismatched_key_returns_none() {
        let dir = test_cache_dir();

        let key = make_key(0xAAAA);
        let artifact = make_artifact();

        write_artifact_to(&dir, &key, &artifact).expect("write should succeed");

        // Different cnf_hash => different file path, so returns None (file not found)
        let different_key = make_key(0xBBBB);
        let result = read_artifact_from(&dir, &different_key).expect("read should not error");
        assert!(result.is_none(), "mismatched key should return None");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_missing_file_returns_none() {
        let dir = test_cache_dir();

        let key = make_key(0x9999);
        let result = read_artifact_from(&dir, &key).expect("read should not error");
        assert!(result.is_none(), "missing file should return None");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_truncated_file_returns_none() {
        let dir = test_cache_dir();

        let key = make_key(0x7777);
        let artifact = make_artifact();
        write_artifact_to(&dir, &key, &artifact).expect("write should succeed");

        // Truncate the file to half its header
        let path = dir.join(artifact_filename(&key));
        let data = fs::read(&path).unwrap();
        fs::write(&path, &data[..HEADER_SIZE / 2]).unwrap();

        let result = read_artifact_from(&dir, &key).expect("read should not error");
        assert!(result.is_none(), "truncated file should return None");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_corrupted_magic_returns_none() {
        let dir = test_cache_dir();

        let key = make_key(0x6666);
        let artifact = make_artifact();
        write_artifact_to(&dir, &key, &artifact).expect("write should succeed");

        // Corrupt the magic bytes
        let path = dir.join(artifact_filename(&key));
        let mut data = fs::read(&path).unwrap();
        data[0] = 0xFF;
        data[1] = 0xFF;
        fs::write(&path, &data).unwrap();

        let result = read_artifact_from(&dir, &key).expect("read should not error");
        assert!(result.is_none(), "corrupted magic should return None");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_eviction() {
        let dir = test_cache_dir();

        let key1 = make_key(0x1111);
        let key2 = make_key(0x2222);
        let artifact = make_artifact();

        // Write two artifacts without eviction (using the internal write_artifact_to
        // which calls evict_if_needed_in with the default 512MB limit).
        write_artifact_to(&dir, &key1, &artifact).expect("write 1 should succeed");
        write_artifact_to(&dir, &key2, &artifact).expect("write 2 should succeed");

        // Both should be readable
        assert!(read_artifact_from(&dir, &key1).unwrap().is_some());
        assert!(read_artifact_from(&dir, &key2).unwrap().is_some());

        // Now force eviction with 0 MB limit
        evict_if_needed_in_with_limit(&dir, 0).expect("eviction should succeed");

        // All files should have been evicted
        let r1 = read_artifact_from(&dir, &key1).unwrap();
        let r2 = read_artifact_from(&dir, &key2).unwrap();
        assert!(r1.is_none(), "key1 should have been evicted");
        assert!(r2.is_none(), "key2 should have been evicted");

        let _ = fs::remove_dir_all(&dir);
    }
}
