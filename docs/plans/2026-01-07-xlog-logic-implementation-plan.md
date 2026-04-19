# XLOG-Logic Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the foundational xlog-logic MVP: a GPU-native Datalog engine supporting n-ary relations, recursive fixpoint computation, stratified negation, and aggregates.

**Architecture:** Rust workspace with 5 crates (xlog-core, xlog-ir, xlog-cuda, xlog-runtime, xlog-logic). CUDA kernels compiled via CMake to PTX, loaded at runtime via cudarc. Semi-naive evaluation with delta/full relation maintenance.

**Tech Stack:** Rust 1.75+, cudarc, pest (parser), thiserror, CUDA 12.x, CMake

---

## Phase 0: Skeleton Setup

### Task 1: Initialize Cargo Workspace

**Files:**
- Create: `Cargo.toml`
- Create: `crates/xlog-core/Cargo.toml`
- Create: `crates/xlog-core/src/lib.rs`
- Create: `crates/xlog-ir/Cargo.toml`
- Create: `crates/xlog-ir/src/lib.rs`
- Create: `crates/xlog-cuda/Cargo.toml`
- Create: `crates/xlog-cuda/src/lib.rs`
- Create: `crates/xlog-runtime/Cargo.toml`
- Create: `crates/xlog-runtime/src/lib.rs`
- Create: `crates/xlog-logic/Cargo.toml`
- Create: `crates/xlog-logic/src/lib.rs`

**Step 1: Create workspace Cargo.toml**

```toml
[workspace]
resolver = "2"
members = [
    "crates/xlog-core",
    "crates/xlog-ir",
    "crates/xlog-cuda",
    "crates/xlog-runtime",
    "crates/xlog-logic",
]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
repository = "https://github.com/BrainyBlaze/xlog"

[workspace.dependencies]
thiserror = "1.0"
cudarc = "0.12"
pest = "2.7"
pest_derive = "2.7"
```

**Step 2: Create xlog-core crate**

`crates/xlog-core/Cargo.toml`:
```toml
[package]
name = "xlog-core"
version.workspace = true
edition.workspace = true

[dependencies]
thiserror.workspace = true
```

`crates/xlog-core/src/lib.rs`:
```rust
//! Core types and traits for XLOG

pub mod error;
pub mod config;
pub mod types;
pub mod traits;

pub use error::{XlogError, Result};
pub use config::{MemoryBudget, RuntimeConfig};
pub use types::{ScalarType, Schema};
```

**Step 3: Create xlog-ir crate**

`crates/xlog-ir/Cargo.toml`:
```toml
[package]
name = "xlog-ir"
version.workspace = true
edition.workspace = true

[dependencies]
xlog-core = { path = "../xlog-core" }
```

`crates/xlog-ir/src/lib.rs`:
```rust
//! Intermediate representations for XLOG

pub mod rir;
pub mod metadata;
pub mod plan;

pub use rir::RirNode;
pub use metadata::RirMeta;
```

**Step 4: Create xlog-cuda crate**

`crates/xlog-cuda/Cargo.toml`:
```toml
[package]
name = "xlog-cuda"
version.workspace = true
edition.workspace = true

[dependencies]
xlog-core = { path = "../xlog-core" }
cudarc.workspace = true
```

`crates/xlog-cuda/src/lib.rs`:
```rust
//! GPU kernel provider for XLOG

pub mod device;
pub mod memory;
pub mod provider;
pub mod kernels;
```

**Step 5: Create xlog-runtime crate**

`crates/xlog-runtime/Cargo.toml`:
```toml
[package]
name = "xlog-runtime"
version.workspace = true
edition.workspace = true

[dependencies]
xlog-core = { path = "../xlog-core" }
xlog-ir = { path = "../xlog-ir" }
xlog-cuda = { path = "../xlog-cuda" }
```

`crates/xlog-runtime/src/lib.rs`:
```rust
//! Execution engine for XLOG

pub mod relation;
pub mod executor;
pub mod profiler;
```

**Step 6: Create xlog-logic crate**

`crates/xlog-logic/Cargo.toml`:
```toml
[package]
name = "xlog-logic"
version.workspace = true
edition.workspace = true

[dependencies]
xlog-core = { path = "../xlog-core" }
xlog-ir = { path = "../xlog-ir" }
xlog-runtime = { path = "../xlog-runtime" }
pest.workspace = true
pest_derive.workspace = true
```

`crates/xlog-logic/src/lib.rs`:
```rust
//! Datalog frontend for XLOG

pub mod parser;
pub mod ast;
pub mod stratify;
pub mod lower;
pub mod compile;
```

**Step 7: Verify workspace builds**

Run: `cargo check --workspace`
Expected: Successful compilation with no errors

**Step 8: Commit**

```bash
git add Cargo.toml crates/
git commit -m "$(cat <<'EOF'
feat: initialize cargo workspace with 5 crates

Set up monorepo structure for xlog-logic MVP:
- xlog-core: foundational types & traits
- xlog-ir: intermediate representations
- xlog-cuda: GPU kernel provider
- xlog-runtime: execution engine
- xlog-logic: Datalog frontend

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 1: xlog-core Implementation

### Task 2: Implement XlogError

**Files:**
- Create: `crates/xlog-core/src/error.rs`
- Test: `crates/xlog-core/src/error.rs` (inline tests)

**Step 1: Write the failing test**

Add to `crates/xlog-core/src/error.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_error_display() {
        let err = XlogError::Parse("unexpected token".to_string());
        assert_eq!(err.to_string(), "Parse error: unexpected token");
    }

    #[test]
    fn test_stratification_cycle_display() {
        let err = XlogError::StratificationCycle(vec!["foo".to_string(), "bar".to_string()]);
        assert!(err.to_string().contains("foo"));
        assert!(err.to_string().contains("bar"));
    }

    #[test]
    fn test_resource_exhausted_display() {
        let err = XlogError::ResourceExhausted {
            context: "join operation".to_string(),
            estimated_bytes: 1024,
            budget_bytes: 512,
        };
        assert!(err.to_string().contains("1024"));
        assert!(err.to_string().contains("512"));
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-core`
Expected: FAIL with "cannot find type `XlogError`"

**Step 3: Write minimal implementation**

Complete `crates/xlog-core/src/error.rs`:
```rust
//! Error types for XLOG

use thiserror::Error;

/// Primary error type for XLOG operations
#[derive(Debug, Error)]
pub enum XlogError {
    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Stratification failed: cycle through negation involving {0:?}")]
    StratificationCycle(Vec<String>),

    #[error("Domain safety: variable {0} not bound in positive literal")]
    UnsafeVariable(String),

    #[error("Resource exhausted: {context}, estimated {estimated_bytes} bytes, budget {budget_bytes} bytes")]
    ResourceExhausted {
        context: String,
        estimated_bytes: u64,
        budget_bytes: u64,
    },

    #[error("Kernel error: {0}")]
    Kernel(String),

    #[error("Type error: {0}")]
    Type(String),

    #[error("Compilation error: {0}")]
    Compilation(String),
}

/// Result alias using XlogError
pub type Result<T> = std::result::Result<T, XlogError>;
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-core`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add crates/xlog-core/src/error.rs
git commit -m "$(cat <<'EOF'
feat(xlog-core): implement XlogError with thiserror

Error variants cover parsing, stratification, domain safety,
resource exhaustion, kernel errors, type errors, and compilation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Implement ScalarType and Schema

**Files:**
- Create: `crates/xlog-core/src/types.rs`
- Test: `crates/xlog-core/src/types.rs` (inline tests)

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_type_size() {
        assert_eq!(ScalarType::U32.size_bytes(), 4);
        assert_eq!(ScalarType::U64.size_bytes(), 8);
        assert_eq!(ScalarType::Bool.size_bytes(), 1);
    }

    #[test]
    fn test_schema_total_row_size() {
        let schema = Schema {
            columns: vec![
                ("a".to_string(), ScalarType::U32),
                ("b".to_string(), ScalarType::U64),
            ],
            key_columns: vec![0],
        };
        assert_eq!(schema.row_size_bytes(), 12); // 4 + 8
    }

    #[test]
    fn test_schema_arity() {
        let schema = Schema {
            columns: vec![
                ("x".to_string(), ScalarType::U32),
                ("y".to_string(), ScalarType::U32),
                ("z".to_string(), ScalarType::U32),
            ],
            key_columns: vec![0, 1, 2],
        };
        assert_eq!(schema.arity(), 3);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-core`
Expected: FAIL with "cannot find type `ScalarType`"

**Step 3: Write minimal implementation**

`crates/xlog-core/src/types.rs`:
```rust
//! Core types for XLOG schemas and data

/// Supported scalar types in XLOG relations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    U32,
    U64,
    I32,
    I64,
    F32,
    F64,
    Bool,
    /// Dictionary-encoded string
    Symbol,
}

impl ScalarType {
    /// Returns the size in bytes of this scalar type
    pub fn size_bytes(&self) -> usize {
        match self {
            ScalarType::U32 | ScalarType::I32 | ScalarType::F32 | ScalarType::Symbol => 4,
            ScalarType::U64 | ScalarType::I64 | ScalarType::F64 => 8,
            ScalarType::Bool => 1,
        }
    }

    /// Returns true if this is a numeric type
    pub fn is_numeric(&self) -> bool {
        !matches!(self, ScalarType::Bool | ScalarType::Symbol)
    }
}

/// Schema describing a relation's columns
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    /// Column names and their types
    pub columns: Vec<(String, ScalarType)>,
    /// Indices of columns that form the key (for dedup/indexing)
    pub key_columns: Vec<usize>,
}

impl Schema {
    /// Create a new schema with all columns as keys
    pub fn new(columns: Vec<(String, ScalarType)>) -> Self {
        let key_columns = (0..columns.len()).collect();
        Self { columns, key_columns }
    }

    /// Number of columns
    pub fn arity(&self) -> usize {
        self.columns.len()
    }

    /// Total size of one row in bytes
    pub fn row_size_bytes(&self) -> usize {
        self.columns.iter().map(|(_, ty)| ty.size_bytes()).sum()
    }

    /// Get column type by index
    pub fn column_type(&self, index: usize) -> Option<ScalarType> {
        self.columns.get(index).map(|(_, ty)| *ty)
    }

    /// Get column index by name
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|(n, _)| n == name)
    }
}

/// Unique identifier for a relation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RelId(pub u32);

/// Aggregation operations supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggOp {
    Count,
    Sum,
    Min,
    Max,
    LogSumExp,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_type_size() {
        assert_eq!(ScalarType::U32.size_bytes(), 4);
        assert_eq!(ScalarType::U64.size_bytes(), 8);
        assert_eq!(ScalarType::Bool.size_bytes(), 1);
    }

    #[test]
    fn test_schema_total_row_size() {
        let schema = Schema {
            columns: vec![
                ("a".to_string(), ScalarType::U32),
                ("b".to_string(), ScalarType::U64),
            ],
            key_columns: vec![0],
        };
        assert_eq!(schema.row_size_bytes(), 12);
    }

    #[test]
    fn test_schema_arity() {
        let schema = Schema {
            columns: vec![
                ("x".to_string(), ScalarType::U32),
                ("y".to_string(), ScalarType::U32),
                ("z".to_string(), ScalarType::U32),
            ],
            key_columns: vec![0, 1, 2],
        };
        assert_eq!(schema.arity(), 3);
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-core`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-core/src/types.rs
git commit -m "$(cat <<'EOF'
feat(xlog-core): implement ScalarType, Schema, RelId, AggOp

Core type system for XLOG relations with support for:
- 8 scalar types (u32, u64, i32, i64, f32, f64, bool, symbol)
- Schema with column names, types, and key designation
- Aggregation operations (count, sum, min, max, logsumexp)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Implement MemoryBudget and RuntimeConfig

**Files:**
- Create: `crates/xlog-core/src/config.rs`
- Test: `crates/xlog-core/src/config.rs` (inline tests)

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_budget_default() {
        let budget = MemoryBudget::default();
        assert!(!budget.allow_ooc);
        assert!(budget.abort_on_exceed);
    }

    #[test]
    fn test_runtime_config_default() {
        let config = RuntimeConfig::default();
        assert!(config.deterministic);
        assert!(!config.profile);
    }

    #[test]
    fn test_memory_budget_from_device() {
        let budget = MemoryBudget::from_device_memory(10_000_000_000); // 10GB
        assert_eq!(budget.device_bytes, 8_000_000_000); // 80%
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-core`
Expected: FAIL with "cannot find type `MemoryBudget`"

**Step 3: Write minimal implementation**

`crates/xlog-core/src/config.rs`:
```rust
//! Configuration types for XLOG runtime

/// GPU memory budget configuration
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Maximum device memory to use in bytes
    pub device_bytes: u64,
    /// Allow out-of-core execution (spill to host)
    pub allow_ooc: bool,
    /// Abort on memory budget exceeded (vs try to continue)
    pub abort_on_exceed: bool,
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self {
            device_bytes: 0, // Will be set from device query
            allow_ooc: false,
            abort_on_exceed: true,
        }
    }
}

impl MemoryBudget {
    /// Create a budget using 80% of available device memory
    pub fn from_device_memory(total_bytes: u64) -> Self {
        Self {
            device_bytes: (total_bytes as f64 * 0.8) as u64,
            allow_ooc: false,
            abort_on_exceed: true,
        }
    }

    /// Create a budget with explicit byte limit
    pub fn with_limit(device_bytes: u64) -> Self {
        Self {
            device_bytes,
            allow_ooc: false,
            abort_on_exceed: true,
        }
    }

    /// Enable out-of-core mode
    pub fn with_ooc(mut self) -> Self {
        self.allow_ooc = true;
        self
    }
}

/// Runtime configuration for XLOG execution
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Memory budget settings
    pub memory: MemoryBudget,
    /// Use deterministic execution (may be slower)
    pub deterministic: bool,
    /// Enable profiling (row counts, memory tracking)
    pub profile: bool,
    /// Maximum fixpoint iterations before abort
    pub max_iterations: u32,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            memory: MemoryBudget::default(),
            deterministic: true,
            profile: false,
            max_iterations: 1_000_000,
        }
    }
}

impl RuntimeConfig {
    /// Enable profiling
    pub fn with_profiling(mut self) -> Self {
        self.profile = true;
        self
    }

    /// Set memory budget
    pub fn with_memory(mut self, memory: MemoryBudget) -> Self {
        self.memory = memory;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_budget_default() {
        let budget = MemoryBudget::default();
        assert!(!budget.allow_ooc);
        assert!(budget.abort_on_exceed);
    }

    #[test]
    fn test_runtime_config_default() {
        let config = RuntimeConfig::default();
        assert!(config.deterministic);
        assert!(!config.profile);
    }

    #[test]
    fn test_memory_budget_from_device() {
        let budget = MemoryBudget::from_device_memory(10_000_000_000);
        assert_eq!(budget.device_bytes, 8_000_000_000);
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-core`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-core/src/config.rs
git commit -m "$(cat <<'EOF'
feat(xlog-core): implement MemoryBudget and RuntimeConfig

Configuration types for GPU memory management:
- MemoryBudget with device limit, OOC toggle, abort policy
- RuntimeConfig with determinism, profiling, iteration limits
- Builder pattern for easy configuration

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Implement KernelProvider Trait

**Files:**
- Create: `crates/xlog-core/src/traits.rs`
- Test: `crates/xlog-core/src/traits.rs` (inline tests)

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Schema, ScalarType, AggOp};

    struct MockProvider;

    impl KernelProvider for MockProvider {
        fn hash_join(
            &self,
            _left: &GpuBuffer,
            _right: &GpuBuffer,
            _left_keys: &[usize],
            _right_keys: &[usize],
        ) -> crate::Result<GpuBuffer> {
            Ok(GpuBuffer::empty())
        }

        fn dedup(&self, _input: &GpuBuffer, _key_cols: &[usize]) -> crate::Result<GpuBuffer> {
            Ok(GpuBuffer::empty())
        }

        fn union(&self, _a: &GpuBuffer, _b: &GpuBuffer) -> crate::Result<GpuBuffer> {
            Ok(GpuBuffer::empty())
        }

        fn diff(&self, _a: &GpuBuffer, _b: &GpuBuffer) -> crate::Result<GpuBuffer> {
            Ok(GpuBuffer::empty())
        }

        fn groupby_agg(
            &self,
            _input: &GpuBuffer,
            _key_cols: &[usize],
            _agg: AggOp,
            _value_col: usize,
        ) -> crate::Result<GpuBuffer> {
            Ok(GpuBuffer::empty())
        }
    }

    #[test]
    fn test_mock_provider_compiles() {
        let provider = MockProvider;
        let empty = GpuBuffer::empty();
        assert!(provider.dedup(&empty, &[0]).is_ok());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-core`
Expected: FAIL with "cannot find trait `KernelProvider`"

**Step 3: Write minimal implementation**

`crates/xlog-core/src/traits.rs`:
```rust
//! Core traits for XLOG extensibility

use crate::types::{AggOp, Schema, ScalarType};
use crate::Result;

/// Opaque handle to GPU memory buffer
/// Actual implementation lives in xlog-cuda
#[derive(Debug)]
pub struct GpuBuffer {
    /// Number of rows in this buffer
    pub num_rows: u64,
    /// Schema of this buffer
    pub schema: Schema,
    /// Opaque handle (will be CudaSlice in xlog-cuda)
    handle: GpuBufferHandle,
}

#[derive(Debug)]
enum GpuBufferHandle {
    Empty,
    // Cuda(cudarc::driver::CudaSlice<u8>) - added in xlog-cuda
}

impl GpuBuffer {
    /// Create an empty buffer
    pub fn empty() -> Self {
        Self {
            num_rows: 0,
            schema: Schema::new(vec![]),
            handle: GpuBufferHandle::Empty,
        }
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.num_rows == 0
    }

    /// Estimated memory usage in bytes
    pub fn estimated_bytes(&self) -> u64 {
        self.num_rows * self.schema.row_size_bytes() as u64
    }
}

/// Trait for GPU kernel execution providers
///
/// This abstraction allows swapping CUDA for other backends (HIP, SYCL)
pub trait KernelProvider: Send + Sync {
    /// Perform a hash join between two buffers
    fn hash_join(
        &self,
        left: &GpuBuffer,
        right: &GpuBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
    ) -> Result<GpuBuffer>;

    /// Remove duplicate rows based on key columns
    fn dedup(&self, input: &GpuBuffer, key_cols: &[usize]) -> Result<GpuBuffer>;

    /// Compute union of two buffers
    fn union(&self, a: &GpuBuffer, b: &GpuBuffer) -> Result<GpuBuffer>;

    /// Compute set difference (a - b)
    fn diff(&self, a: &GpuBuffer, b: &GpuBuffer) -> Result<GpuBuffer>;

    /// Perform groupby aggregation
    fn groupby_agg(
        &self,
        input: &GpuBuffer,
        key_cols: &[usize],
        agg: AggOp,
        value_col: usize,
    ) -> Result<GpuBuffer>;
}

/// Trait for relation storage backends
pub trait RelationStore: Send + Sync {
    /// Get a relation by ID
    fn get(&self, name: &str) -> Option<&GpuBuffer>;

    /// Store a relation
    fn put(&mut self, name: &str, buffer: GpuBuffer);

    /// Check if relation exists
    fn contains(&self, name: &str) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockProvider;

    impl KernelProvider for MockProvider {
        fn hash_join(
            &self,
            _left: &GpuBuffer,
            _right: &GpuBuffer,
            _left_keys: &[usize],
            _right_keys: &[usize],
        ) -> Result<GpuBuffer> {
            Ok(GpuBuffer::empty())
        }

        fn dedup(&self, _input: &GpuBuffer, _key_cols: &[usize]) -> Result<GpuBuffer> {
            Ok(GpuBuffer::empty())
        }

        fn union(&self, _a: &GpuBuffer, _b: &GpuBuffer) -> Result<GpuBuffer> {
            Ok(GpuBuffer::empty())
        }

        fn diff(&self, _a: &GpuBuffer, _b: &GpuBuffer) -> Result<GpuBuffer> {
            Ok(GpuBuffer::empty())
        }

        fn groupby_agg(
            &self,
            _input: &GpuBuffer,
            _key_cols: &[usize],
            _agg: AggOp,
            _value_col: usize,
        ) -> Result<GpuBuffer> {
            Ok(GpuBuffer::empty())
        }
    }

    #[test]
    fn test_mock_provider_compiles() {
        let provider = MockProvider;
        let empty = GpuBuffer::empty();
        assert!(provider.dedup(&empty, &[0]).is_ok());
    }

    #[test]
    fn test_gpu_buffer_empty() {
        let buf = GpuBuffer::empty();
        assert!(buf.is_empty());
        assert_eq!(buf.estimated_bytes(), 0);
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-core`
Expected: PASS

**Step 5: Update lib.rs exports**

Update `crates/xlog-core/src/lib.rs`:
```rust
//! Core types and traits for XLOG

pub mod error;
pub mod config;
pub mod types;
pub mod traits;

pub use error::{XlogError, Result};
pub use config::{MemoryBudget, RuntimeConfig};
pub use types::{ScalarType, Schema, RelId, AggOp};
pub use traits::{GpuBuffer, KernelProvider, RelationStore};
```

**Step 6: Run all tests**

Run: `cargo test -p xlog-core`
Expected: PASS (all tests)

**Step 7: Commit**

```bash
git add crates/xlog-core/
git commit -m "$(cat <<'EOF'
feat(xlog-core): implement KernelProvider and RelationStore traits

Core abstractions for GPU kernel execution:
- GpuBuffer: opaque handle to GPU memory with schema
- KernelProvider: trait for join, dedup, union, diff, groupby
- RelationStore: trait for relation storage backends

Enables backend abstraction (CUDA/HIP/SYCL) at trait boundary.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: xlog-ir Implementation

### Task 6: Implement RirMeta (Node Metadata)

**Files:**
- Create: `crates/xlog-ir/src/metadata.rs`
- Test: `crates/xlog-ir/src/metadata.rs` (inline tests)

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::{Schema, ScalarType};

    #[test]
    fn test_rir_meta_default() {
        let meta = RirMeta::default();
        assert_eq!(meta.est_rows, (0, 0));
        assert!(meta.deterministic);
    }

    #[test]
    fn test_layout_hint_default() {
        let hint = LayoutHint::default();
        assert_eq!(hint, LayoutHint::CudfTable);
    }

    #[test]
    fn test_skew_signature() {
        let sig = SkewSignature {
            hot_keys: vec![42, 100],
            entropy: 2.5,
        };
        assert!(sig.is_skewed()); // entropy < 3.0 threshold
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-ir`
Expected: FAIL with "cannot find type `RirMeta`"

**Step 3: Write minimal implementation**

`crates/xlog-ir/src/metadata.rs`:
```rust
//! Metadata for RIR nodes (cardinality, memory estimates, skew)

use xlog_core::Schema;

/// Hint for physical layout selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayoutHint {
    /// Standard cuDF table (baseline)
    #[default]
    CudfTable,
    /// HISA-style indexed storage for recursion
    HisaIndexed,
    /// VFLog-style columnar for bandwidth workloads
    VflogColumnar,
}

/// Signature of data skew for join optimization
#[derive(Debug, Clone)]
pub struct SkewSignature {
    /// Top-k hot keys
    pub hot_keys: Vec<u64>,
    /// Shannon entropy of key distribution
    pub entropy: f64,
}

impl SkewSignature {
    /// Check if data is considered skewed (entropy below threshold)
    pub fn is_skewed(&self) -> bool {
        self.entropy < 3.0 // bits
    }
}

/// Metadata attached to each RIR node
#[derive(Debug, Clone)]
pub struct RirMeta {
    /// Schema of output relation
    pub schema: Schema,
    /// Estimated row count range (min, max)
    pub est_rows: (u64, u64),
    /// Estimated memory bytes range (min, max)
    pub est_bytes: (u64, u64),
    /// Optional skew signature
    pub skew: Option<SkewSignature>,
    /// Whether this node produces deterministic output
    pub deterministic: bool,
    /// Layout hint for physical storage
    pub layout_hint: LayoutHint,
}

impl Default for RirMeta {
    fn default() -> Self {
        Self {
            schema: Schema::new(vec![]),
            est_rows: (0, 0),
            est_bytes: (0, 0),
            skew: None,
            deterministic: true,
            layout_hint: LayoutHint::default(),
        }
    }
}

impl RirMeta {
    /// Create metadata with schema
    pub fn with_schema(schema: Schema) -> Self {
        Self {
            schema,
            ..Default::default()
        }
    }

    /// Set estimated rows
    pub fn with_rows(mut self, min: u64, max: u64) -> Self {
        self.est_rows = (min, max);
        self.est_bytes = (
            min * self.schema.row_size_bytes() as u64,
            max * self.schema.row_size_bytes() as u64,
        );
        self
    }

    /// Set layout hint
    pub fn with_layout(mut self, hint: LayoutHint) -> Self {
        self.layout_hint = hint;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::ScalarType;

    #[test]
    fn test_rir_meta_default() {
        let meta = RirMeta::default();
        assert_eq!(meta.est_rows, (0, 0));
        assert!(meta.deterministic);
    }

    #[test]
    fn test_layout_hint_default() {
        let hint = LayoutHint::default();
        assert_eq!(hint, LayoutHint::CudfTable);
    }

    #[test]
    fn test_skew_signature() {
        let sig = SkewSignature {
            hot_keys: vec![42, 100],
            entropy: 2.5,
        };
        assert!(sig.is_skewed());
    }

    #[test]
    fn test_meta_with_rows() {
        let schema = Schema::new(vec![
            ("a".to_string(), ScalarType::U32),
            ("b".to_string(), ScalarType::U32),
        ]);
        let meta = RirMeta::with_schema(schema).with_rows(100, 200);
        assert_eq!(meta.est_rows, (100, 200));
        assert_eq!(meta.est_bytes, (800, 1600)); // 8 bytes per row
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-ir`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-ir/src/metadata.rs
git commit -m "$(cat <<'EOF'
feat(xlog-ir): implement RirMeta node metadata

Metadata for cost-based optimization:
- LayoutHint: CudfTable, HisaIndexed, VflogColumnar
- SkewSignature: hot keys and entropy for join optimization
- RirMeta: schema, row/byte estimates, determinism flags

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Implement RirNode (Relational IR)

**Files:**
- Create: `crates/xlog-ir/src/rir.rs`
- Test: `crates/xlog-ir/src/rir.rs` (inline tests)

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::RelId;

    #[test]
    fn test_scan_node() {
        let node = RirNode::Scan { rel: RelId(1) };
        assert!(matches!(node, RirNode::Scan { rel: RelId(1) }));
    }

    #[test]
    fn test_join_node() {
        let left = Box::new(RirNode::Scan { rel: RelId(1) });
        let right = Box::new(RirNode::Scan { rel: RelId(2) });
        let join = RirNode::Join {
            left,
            right,
            left_keys: vec![0],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        assert!(matches!(join, RirNode::Join { .. }));
    }

    #[test]
    fn test_fixpoint_node() {
        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(2) });
        let fp = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(3),
            full_rel: RelId(4),
        };
        assert!(matches!(fp, RirNode::Fixpoint { scc_id: 0, .. }));
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-ir`
Expected: FAIL with "cannot find type `RirNode`"

**Step 3: Write minimal implementation**

`crates/xlog-ir/src/rir.rs`:
```rust
//! Relational IR node definitions

use xlog_core::{AggOp, RelId};

/// Join type variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Standard inner join
    Inner,
    /// Left outer join
    LeftOuter,
    /// Semi join (exists check)
    Semi,
    /// Anti join (not exists / negation)
    Anti,
}

/// Expression in filter predicates
#[derive(Debug, Clone)]
pub enum Expr {
    /// Column reference by index
    Column(usize),
    /// Constant value
    Const(ConstValue),
    /// Binary comparison
    Compare {
        left: Box<Expr>,
        op: CompareOp,
        right: Box<Expr>,
    },
    /// Logical AND
    And(Vec<Expr>),
    /// Logical OR
    Or(Vec<Expr>),
    /// Logical NOT
    Not(Box<Expr>),
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Constant values in expressions
#[derive(Debug, Clone)]
pub enum ConstValue {
    U32(u32),
    U64(u64),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Symbol(String),
}

/// Relational IR node types
#[derive(Debug, Clone)]
pub enum RirNode {
    /// Scan a base relation
    Scan {
        rel: RelId,
    },

    /// Filter rows by predicate
    Filter {
        input: Box<RirNode>,
        predicate: Expr,
    },

    /// Project specific columns
    Project {
        input: Box<RirNode>,
        columns: Vec<usize>,
    },

    /// Join two relations
    Join {
        left: Box<RirNode>,
        right: Box<RirNode>,
        left_keys: Vec<usize>,
        right_keys: Vec<usize>,
        join_type: JoinType,
    },

    /// Group by with aggregation
    GroupBy {
        input: Box<RirNode>,
        key_cols: Vec<usize>,
        /// (value_column, aggregation_op)
        aggs: Vec<(usize, AggOp)>,
    },

    /// Union multiple inputs
    Union {
        inputs: Vec<RirNode>,
    },

    /// Remove duplicates
    Distinct {
        input: Box<RirNode>,
        key_cols: Vec<usize>,
    },

    /// Set difference (left - right)
    Diff {
        left: Box<RirNode>,
        right: Box<RirNode>,
    },

    /// Fixpoint iteration for recursion
    Fixpoint {
        /// SCC identifier
        scc_id: u32,
        /// Base case computation
        base: Box<RirNode>,
        /// Recursive step computation
        recursive: Box<RirNode>,
        /// Relation for delta (new tuples)
        delta_rel: RelId,
        /// Relation for full result
        full_rel: RelId,
    },
}

impl RirNode {
    /// Check if this node is a leaf (Scan)
    pub fn is_leaf(&self) -> bool {
        matches!(self, RirNode::Scan { .. })
    }

    /// Get all relation IDs referenced in this subtree
    pub fn referenced_relations(&self) -> Vec<RelId> {
        let mut rels = Vec::new();
        self.collect_relations(&mut rels);
        rels
    }

    fn collect_relations(&self, rels: &mut Vec<RelId>) {
        match self {
            RirNode::Scan { rel } => rels.push(*rel),
            RirNode::Filter { input, .. } | RirNode::Project { input, .. } => {
                input.collect_relations(rels);
            }
            RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
                left.collect_relations(rels);
                right.collect_relations(rels);
            }
            RirNode::Union { inputs } => {
                for input in inputs {
                    input.collect_relations(rels);
                }
            }
            RirNode::GroupBy { input, .. } | RirNode::Distinct { input, .. } => {
                input.collect_relations(rels);
            }
            RirNode::Fixpoint {
                base,
                recursive,
                delta_rel,
                full_rel,
                ..
            } => {
                base.collect_relations(rels);
                recursive.collect_relations(rels);
                rels.push(*delta_rel);
                rels.push(*full_rel);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_node() {
        let node = RirNode::Scan { rel: RelId(1) };
        assert!(matches!(node, RirNode::Scan { rel: RelId(1) }));
        assert!(node.is_leaf());
    }

    #[test]
    fn test_join_node() {
        let left = Box::new(RirNode::Scan { rel: RelId(1) });
        let right = Box::new(RirNode::Scan { rel: RelId(2) });
        let join = RirNode::Join {
            left,
            right,
            left_keys: vec![0],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        assert!(matches!(join, RirNode::Join { .. }));
        let rels = join.referenced_relations();
        assert!(rels.contains(&RelId(1)));
        assert!(rels.contains(&RelId(2)));
    }

    #[test]
    fn test_fixpoint_node() {
        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(2) });
        let fp = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(3),
            full_rel: RelId(4),
        };
        assert!(matches!(fp, RirNode::Fixpoint { scc_id: 0, .. }));
    }

    #[test]
    fn test_anti_join() {
        let left = Box::new(RirNode::Scan { rel: RelId(1) });
        let right = Box::new(RirNode::Scan { rel: RelId(2) });
        let anti = RirNode::Join {
            left,
            right,
            left_keys: vec![0],
            right_keys: vec![0],
            join_type: JoinType::Anti,
        };
        if let RirNode::Join { join_type, .. } = anti {
            assert_eq!(join_type, JoinType::Anti);
        }
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-ir`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-ir/src/rir.rs
git commit -m "$(cat <<'EOF'
feat(xlog-ir): implement RirNode relational IR

Complete relational algebra node types:
- Scan, Filter, Project for basic ops
- Join with Inner/LeftOuter/Semi/Anti variants
- GroupBy with aggregation support
- Union, Distinct, Diff for set operations
- Fixpoint for semi-naive recursive evaluation

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Implement Execution Plan

**Files:**
- Create: `crates/xlog-ir/src/plan.rs`
- Test: `crates/xlog-ir/src/plan.rs` (inline tests)

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scc_ordering() {
        let sccs = vec![
            Scc { id: 0, predicates: vec!["edge".into()], is_recursive: false },
            Scc { id: 1, predicates: vec!["reach".into()], is_recursive: true },
        ];
        let plan = ExecutionPlan::new(sccs);
        assert_eq!(plan.sccs.len(), 2);
        assert!(!plan.sccs[0].is_recursive);
        assert!(plan.sccs[1].is_recursive);
    }

    #[test]
    fn test_stratum_assignment() {
        let strata = vec![
            Stratum { id: 0, sccs: vec![0, 1] },
            Stratum { id: 1, sccs: vec![2] },
        ];
        assert_eq!(strata[0].sccs.len(), 2);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-ir`
Expected: FAIL with "cannot find type `Scc`"

**Step 3: Write minimal implementation**

`crates/xlog-ir/src/plan.rs`:
```rust
//! Execution plan representation

use crate::rir::RirNode;
use crate::metadata::RirMeta;

/// Strongly Connected Component in the dependency graph
#[derive(Debug, Clone)]
pub struct Scc {
    /// Unique SCC identifier
    pub id: u32,
    /// Predicate names in this SCC
    pub predicates: Vec<String>,
    /// Whether this SCC contains recursion
    pub is_recursive: bool,
}

/// Stratum in stratified evaluation
#[derive(Debug, Clone)]
pub struct Stratum {
    /// Stratum number (0 = base)
    pub id: u32,
    /// SCCs in this stratum (topologically ordered)
    pub sccs: Vec<u32>,
}

/// Compiled rule ready for execution
#[derive(Debug, Clone)]
pub struct CompiledRule {
    /// Head predicate name
    pub head: String,
    /// RIR tree for rule body
    pub body: RirNode,
    /// Metadata for cost estimation
    pub meta: RirMeta,
}

/// Complete execution plan for a program
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// SCCs in dependency order
    pub sccs: Vec<Scc>,
    /// Strata for negation ordering
    pub strata: Vec<Stratum>,
    /// Compiled rules grouped by SCC
    pub rules_by_scc: Vec<Vec<CompiledRule>>,
    /// Total estimated memory peak (bytes)
    pub est_memory_peak: u64,
}

impl ExecutionPlan {
    /// Create a new execution plan from SCCs
    pub fn new(sccs: Vec<Scc>) -> Self {
        Self {
            sccs,
            strata: vec![],
            rules_by_scc: vec![],
            est_memory_peak: 0,
        }
    }

    /// Add strata to the plan
    pub fn with_strata(mut self, strata: Vec<Stratum>) -> Self {
        self.strata = strata;
        self
    }

    /// Get the number of recursive SCCs
    pub fn recursive_scc_count(&self) -> usize {
        self.sccs.iter().filter(|s| s.is_recursive).count()
    }

    /// Check if this plan has any recursion
    pub fn has_recursion(&self) -> bool {
        self.sccs.iter().any(|s| s.is_recursive)
    }
}

/// Builder for execution plans
#[derive(Debug, Default)]
pub struct PlanBuilder {
    sccs: Vec<Scc>,
    strata: Vec<Stratum>,
    rules: Vec<Vec<CompiledRule>>,
}

impl PlanBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_scc(&mut self, scc: Scc) -> &mut Self {
        self.sccs.push(scc);
        self.rules.push(vec![]);
        self
    }

    pub fn add_rule(&mut self, scc_id: u32, rule: CompiledRule) -> &mut Self {
        if let Some(rules) = self.rules.get_mut(scc_id as usize) {
            rules.push(rule);
        }
        self
    }

    pub fn add_stratum(&mut self, stratum: Stratum) -> &mut Self {
        self.strata.push(stratum);
        self
    }

    pub fn build(self) -> ExecutionPlan {
        ExecutionPlan {
            sccs: self.sccs,
            strata: self.strata,
            rules_by_scc: self.rules,
            est_memory_peak: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scc_ordering() {
        let sccs = vec![
            Scc {
                id: 0,
                predicates: vec!["edge".into()],
                is_recursive: false,
            },
            Scc {
                id: 1,
                predicates: vec!["reach".into()],
                is_recursive: true,
            },
        ];
        let plan = ExecutionPlan::new(sccs);
        assert_eq!(plan.sccs.len(), 2);
        assert!(!plan.sccs[0].is_recursive);
        assert!(plan.sccs[1].is_recursive);
    }

    #[test]
    fn test_stratum_assignment() {
        let strata = vec![
            Stratum { id: 0, sccs: vec![0, 1] },
            Stratum { id: 1, sccs: vec![2] },
        ];
        assert_eq!(strata[0].sccs.len(), 2);
    }

    #[test]
    fn test_plan_builder() {
        let mut builder = PlanBuilder::new();
        builder.add_scc(Scc {
            id: 0,
            predicates: vec!["p".into()],
            is_recursive: false,
        });
        builder.add_stratum(Stratum { id: 0, sccs: vec![0] });
        let plan = builder.build();
        assert_eq!(plan.sccs.len(), 1);
        assert_eq!(plan.strata.len(), 1);
    }

    #[test]
    fn test_has_recursion() {
        let non_recursive = ExecutionPlan::new(vec![Scc {
            id: 0,
            predicates: vec!["p".into()],
            is_recursive: false,
        }]);
        assert!(!non_recursive.has_recursion());

        let recursive = ExecutionPlan::new(vec![Scc {
            id: 0,
            predicates: vec!["reach".into()],
            is_recursive: true,
        }]);
        assert!(recursive.has_recursion());
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-ir`
Expected: PASS

**Step 5: Update xlog-ir lib.rs**

`crates/xlog-ir/src/lib.rs`:
```rust
//! Intermediate representations for XLOG

pub mod rir;
pub mod metadata;
pub mod plan;

pub use rir::{RirNode, JoinType, Expr, CompareOp, ConstValue};
pub use metadata::{RirMeta, LayoutHint, SkewSignature};
pub use plan::{ExecutionPlan, Scc, Stratum, CompiledRule, PlanBuilder};
```

**Step 6: Run all tests**

Run: `cargo test -p xlog-ir`
Expected: PASS

**Step 7: Commit**

```bash
git add crates/xlog-ir/
git commit -m "$(cat <<'EOF'
feat(xlog-ir): implement ExecutionPlan with SCC and Stratum

Execution planning structures:
- Scc: strongly connected components with recursion flag
- Stratum: stratification layers for negation
- CompiledRule: RIR tree with metadata
- ExecutionPlan: complete plan with memory estimates
- PlanBuilder: fluent API for plan construction

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3: CUDA Build Setup

### Task 9: Create CUDA Kernel Stubs and CMake Build

**Files:**
- Create: `kernels/CMakeLists.txt`
- Create: `kernels/join.cu`
- Create: `kernels/dedup.cu`
- Create: `kernels/groupby.cu`
- Create: `build.rs` (workspace root)

**Step 1: Create CMakeLists.txt**

`kernels/CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.18)
project(xlog_kernels CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Generate PTX for common architectures
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Compile to PTX only (no object files)
set(CUDA_PTXAS_FLAGS "-v")

# Kernel sources
set(KERNEL_SOURCES
    join.cu
    dedup.cu
    groupby.cu
)

# Create PTX for each kernel
foreach(KERNEL_SRC ${KERNEL_SOURCES})
    get_filename_component(KERNEL_NAME ${KERNEL_SRC} NAME_WE)
    add_library(${KERNEL_NAME}_ptx OBJECT ${KERNEL_SRC})
    set_target_properties(${KERNEL_NAME}_ptx PROPERTIES
        CUDA_PTX_COMPILATION ON
    )
endforeach()

# Install PTX files
install(FILES
    $<TARGET_OBJECTS:join_ptx>
    $<TARGET_OBJECTS:dedup_ptx>
    $<TARGET_OBJECTS:groupby_ptx>
    DESTINATION ptx
)
```

**Step 2: Create join.cu stub**

`kernels/join.cu`:
```cuda
// XLOG GPU Join Kernels
// Hash join with linked-list collision handling

#include <cstdint>

// Hash function for join keys
__device__ __forceinline__ uint32_t hash_key(uint32_t key) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key;
}

// Build phase: insert keys into hash table with linked lists
extern "C" __global__ void hash_join_build(
    const uint32_t* __restrict__ keys,      // Input keys
    const uint32_t* __restrict__ payloads,  // Input payloads (row indices)
    uint32_t num_rows,
    uint32_t* __restrict__ hash_table,      // Hash table (key, payload, next)
    uint32_t* __restrict__ next_ptrs,       // Next pointers for linked lists
    uint32_t hash_table_size
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    uint32_t key = keys[tid];
    uint32_t payload = payloads[tid];
    uint32_t hash = hash_key(key) % hash_table_size;

    // Atomic linked list insertion
    uint32_t old = atomicExch(&hash_table[hash * 3 + 2], tid);
    next_ptrs[tid] = old;
    hash_table[hash * 3] = key;
    hash_table[hash * 3 + 1] = payload;
}

// Probe phase: find matches and output join results
extern "C" __global__ void hash_join_probe(
    const uint32_t* __restrict__ probe_keys,
    const uint32_t* __restrict__ probe_payloads,
    uint32_t num_probe_rows,
    const uint32_t* __restrict__ hash_table,
    const uint32_t* __restrict__ build_keys,
    const uint32_t* __restrict__ build_payloads,
    const uint32_t* __restrict__ next_ptrs,
    uint32_t hash_table_size,
    uint32_t* __restrict__ output_left,     // Output: left row indices
    uint32_t* __restrict__ output_right,    // Output: right row indices
    uint32_t* __restrict__ output_count     // Atomic counter for output
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_probe_rows) return;

    uint32_t key = probe_keys[tid];
    uint32_t hash = hash_key(key) % hash_table_size;

    // Walk the linked list
    uint32_t current = hash_table[hash * 3 + 2];
    while (current != 0xFFFFFFFF) {
        if (build_keys[current] == key) {
            uint32_t out_idx = atomicAdd(output_count, 1);
            output_left[out_idx] = probe_payloads[tid];
            output_right[out_idx] = build_payloads[current];
        }
        current = next_ptrs[current];
    }
}
```

**Step 3: Create dedup.cu stub**

`kernels/dedup.cu`:
```cuda
// XLOG GPU Deduplication Kernels
// Sort-based deduplication with prefix sum compaction

#include <cstdint>

// Mark duplicates in a sorted array
// Output: 1 if row is unique (first occurrence), 0 if duplicate
extern "C" __global__ void mark_duplicates(
    const uint32_t* __restrict__ sorted_keys,
    uint32_t num_rows,
    uint32_t num_key_cols,
    uint32_t row_stride,  // Number of columns per row
    uint8_t* __restrict__ unique_mask
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    if (tid == 0) {
        // First row is always unique
        unique_mask[0] = 1;
        return;
    }

    // Compare with previous row
    bool is_duplicate = true;
    for (uint32_t k = 0; k < num_key_cols && is_duplicate; k++) {
        uint32_t curr = sorted_keys[tid * row_stride + k];
        uint32_t prev = sorted_keys[(tid - 1) * row_stride + k];
        if (curr != prev) {
            is_duplicate = false;
        }
    }

    unique_mask[tid] = is_duplicate ? 0 : 1;
}

// Compact rows based on unique mask using prefix sum offsets
extern "C" __global__ void compact_rows(
    const uint32_t* __restrict__ input,
    const uint8_t* __restrict__ unique_mask,
    const uint32_t* __restrict__ prefix_sum,  // Exclusive prefix sum of mask
    uint32_t num_rows,
    uint32_t row_stride,
    uint32_t* __restrict__ output
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    if (unique_mask[tid]) {
        uint32_t out_idx = prefix_sum[tid];
        for (uint32_t c = 0; c < row_stride; c++) {
            output[out_idx * row_stride + c] = input[tid * row_stride + c];
        }
    }
}
```

**Step 4: Create groupby.cu stub**

`kernels/groupby.cu`:
```cuda
// XLOG GPU GroupBy Kernels
// Sorted-input group aggregation

#include <cstdint>
#include <cfloat>

// Detect group boundaries in sorted data
extern "C" __global__ void detect_group_boundaries(
    const uint32_t* __restrict__ sorted_keys,
    uint32_t num_rows,
    uint32_t num_key_cols,
    uint32_t row_stride,
    uint8_t* __restrict__ is_boundary  // 1 if start of new group
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    if (tid == 0) {
        is_boundary[0] = 1;
        return;
    }

    bool boundary = false;
    for (uint32_t k = 0; k < num_key_cols; k++) {
        uint32_t curr = sorted_keys[tid * row_stride + k];
        uint32_t prev = sorted_keys[(tid - 1) * row_stride + k];
        if (curr != prev) {
            boundary = true;
            break;
        }
    }

    is_boundary[tid] = boundary ? 1 : 0;
}

// Count aggregation per group
extern "C" __global__ void groupby_count(
    const uint8_t* __restrict__ is_boundary,
    const uint32_t* __restrict__ group_ids,  // Prefix sum of is_boundary
    uint32_t num_rows,
    uint32_t* __restrict__ counts
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    uint32_t group = group_ids[tid];
    atomicAdd(&counts[group], 1);
}

// Sum aggregation per group
extern "C" __global__ void groupby_sum(
    const uint32_t* __restrict__ values,
    const uint32_t* __restrict__ group_ids,
    uint32_t num_rows,
    uint64_t* __restrict__ sums
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    uint32_t group = group_ids[tid];
    atomicAdd((unsigned long long*)&sums[group], (unsigned long long)values[tid]);
}

// Min aggregation per group
extern "C" __global__ void groupby_min(
    const uint32_t* __restrict__ values,
    const uint32_t* __restrict__ group_ids,
    uint32_t num_rows,
    uint32_t* __restrict__ mins
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    uint32_t group = group_ids[tid];
    atomicMin(&mins[group], values[tid]);
}

// Max aggregation per group
extern "C" __global__ void groupby_max(
    const uint32_t* __restrict__ values,
    const uint32_t* __restrict__ group_ids,
    uint32_t num_rows,
    uint32_t* __restrict__ maxs
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    uint32_t group = group_ids[tid];
    atomicMax(&maxs[group], values[tid]);
}
```

**Step 5: Create .gitignore for build artifacts**

`kernels/.gitignore`:
```
build/
*.ptx
```

**Step 6: Verify CUDA files are syntactically valid**

Run: `ls kernels/`
Expected: CMakeLists.txt, join.cu, dedup.cu, groupby.cu

**Step 7: Commit**

```bash
git add kernels/
git commit -m "$(cat <<'EOF'
feat: add CUDA kernel stubs with CMake build

GPU kernels for core relational operations:
- join.cu: hash join with linked-list collision handling
- dedup.cu: sort-based deduplication with compaction
- groupby.cu: sorted-input group aggregation (count/sum/min/max)

CMake configured to compile to PTX for runtime loading.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4: xlog-logic Parser

### Task 10: Create Pest Grammar

**Files:**
- Create: `crates/xlog-logic/src/grammar.pest`
- Modify: `crates/xlog-logic/src/parser.rs`
- Test: inline tests

**Step 1: Write the failing test**

In `crates/xlog-logic/src/parser.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fact() {
        let input = "edge(1, 2).";
        let result = parse_program(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_rule() {
        let input = "reach(X, Y) :- edge(X, Y).";
        let result = parse_program(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_recursive_rule() {
        let input = "reach(X, Z) :- reach(X, Y), edge(Y, Z).";
        let result = parse_program(input);
        assert!(result.is_ok());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-logic`
Expected: FAIL with "cannot find function `parse_program`"

**Step 3: Create the Pest grammar**

`crates/xlog-logic/src/grammar.pest`:
```pest
// XLOG Grammar for Datalog-style logic programs

WHITESPACE = _{ " " | "\t" | "\r" | "\n" }
COMMENT = _{ "//" ~ (!"\n" ~ ANY)* }

// Identifiers
ident = @{ ASCII_ALPHA_LOWER ~ (ASCII_ALPHANUMERIC | "_")* }
variable = @{ ASCII_ALPHA_UPPER ~ (ASCII_ALPHANUMERIC | "_")* }

// Literals
integer = @{ "-"? ~ ASCII_DIGIT+ }
float_num = @{ "-"? ~ ASCII_DIGIT+ ~ "." ~ ASCII_DIGIT+ }
string_lit = @{ "\"" ~ (!"\"" ~ ANY)* ~ "\"" }

// Terms
term = { variable | integer | float_num | string_lit | ident }

// Atoms
atom = { ident ~ "(" ~ term_list? ~ ")" }
term_list = { term ~ ("," ~ term)* }

// Aggregate expressions
agg_op = { "count" | "sum" | "min" | "max" }
aggregate = { agg_op ~ "(" ~ variable ~ ")" }
agg_term = { aggregate | term }

// Comparison operators
cmp_op = { "==" | "!=" | "<=" | ">=" | "<" | ">" | "=" }
comparison = { term ~ cmp_op ~ term }

// Body literals
negated_atom = { "not" ~ atom }
body_literal = { negated_atom | atom | comparison }
body = { body_literal ~ ("," ~ body_literal)* }

// Head with optional aggregate
head_term = { agg_term | term }
head_term_list = { head_term ~ ("," ~ head_term)* }
head = { ident ~ "(" ~ head_term_list? ~ ")" }

// Rules and facts
rule_def = { head ~ ":-" ~ body ~ "." }
fact = { atom ~ "." }
constraint = { ":-" ~ body ~ "." }

// Queries
query = { "?-" ~ atom ~ "." }

// Domain declarations
domain_decl = { "domain" ~ ident ~ ":" ~ type_spec ~ "." }
type_spec = { "u32" | "u64" | "i32" | "i64" | "f32" | "f64" | "bool" | "symbol" }

// Predicate declarations
pred_decl = { "pred" ~ ident ~ "(" ~ type_list? ~ ")" ~ "." }
type_list = { type_spec ~ ("," ~ type_spec)* }

// Program structure
statement = { domain_decl | pred_decl | rule_def | fact | constraint | query }
program = { SOI ~ statement* ~ EOI }
```

**Step 4: Implement the parser module**

`crates/xlog-logic/src/parser.rs`:
```rust
//! Parser for XLOG programs using Pest

use pest::Parser;
use pest_derive::Parser;
use xlog_core::{XlogError, Result};

#[derive(Parser)]
#[grammar = "grammar.pest"]
pub struct XlogParser;

/// Parse result containing the parsed pairs
pub type ParseResult<'a> = pest::iterators::Pairs<'a, Rule>;

/// Parse an XLOG program string
pub fn parse_program(input: &str) -> Result<ParseResult<'_>> {
    XlogParser::parse(Rule::program, input)
        .map_err(|e| XlogError::Parse(e.to_string()))
}

/// Parse a single statement
pub fn parse_statement(input: &str) -> Result<ParseResult<'_>> {
    XlogParser::parse(Rule::statement, input)
        .map_err(|e| XlogError::Parse(e.to_string()))
}

/// Parse a single atom
pub fn parse_atom(input: &str) -> Result<ParseResult<'_>> {
    XlogParser::parse(Rule::atom, input)
        .map_err(|e| XlogError::Parse(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fact() {
        let input = "edge(1, 2).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse fact: {:?}", result.err());
    }

    #[test]
    fn test_parse_rule() {
        let input = "reach(X, Y) :- edge(X, Y).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse rule: {:?}", result.err());
    }

    #[test]
    fn test_parse_recursive_rule() {
        let input = "reach(X, Z) :- reach(X, Y), edge(Y, Z).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse recursive rule: {:?}", result.err());
    }

    #[test]
    fn test_parse_negation() {
        let input = "isolated(X) :- node(X), not edge(X, Y).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse negation: {:?}", result.err());
    }

    #[test]
    fn test_parse_aggregate() {
        let input = "out_degree(X, count(Y)) :- edge(X, Y).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse aggregate: {:?}", result.err());
    }

    #[test]
    fn test_parse_constraint() {
        let input = ":- reach(X, X).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse constraint: {:?}", result.err());
    }

    #[test]
    fn test_parse_query() {
        let input = "?- reach(1, N).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse query: {:?}", result.err());
    }

    #[test]
    fn test_parse_full_program() {
        let input = r#"
            edge(1, 2).
            edge(2, 3).
            edge(3, 4).
            reach(X, Y) :- edge(X, Y).
            reach(X, Z) :- reach(X, Y), edge(Y, Z).
            ?- reach(1, N).
        "#;
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse full program: {:?}", result.err());
    }

    #[test]
    fn test_parse_comparison() {
        let input = "small(X) :- value(X), X < 10.";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse comparison: {:?}", result.err());
    }

    #[test]
    fn test_parse_pred_decl() {
        let input = "pred edge(u32, u32).";
        let result = parse_program(input);
        assert!(result.is_ok(), "Failed to parse pred decl: {:?}", result.err());
    }
}
```

**Step 5: Run test to verify it passes**

Run: `cargo test -p xlog-logic`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/xlog-logic/src/grammar.pest crates/xlog-logic/src/parser.rs
git commit -m "$(cat <<'EOF'
feat(xlog-logic): implement Pest parser for Datalog

Complete grammar supporting:
- Facts and rules with n-ary predicates
- Stratified negation (not)
- Aggregates (count, sum, min, max)
- Comparison operators
- Constraints and queries
- Domain and predicate declarations

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Implement AST Types

**Files:**
- Create: `crates/xlog-logic/src/ast.rs`
- Test: inline tests

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_variable() {
        let term = Term::Variable("X".to_string());
        assert!(term.is_variable());
    }

    #[test]
    fn test_atom_arity() {
        let atom = Atom {
            predicate: "edge".to_string(),
            terms: vec![Term::Integer(1), Term::Integer(2)],
        };
        assert_eq!(atom.arity(), 2);
    }

    #[test]
    fn test_rule_is_fact() {
        let fact = Rule {
            head: Atom {
                predicate: "edge".to_string(),
                terms: vec![Term::Integer(1), Term::Integer(2)],
            },
            body: vec![],
        };
        assert!(fact.is_fact());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-logic`
Expected: FAIL with "cannot find type `Term`"

**Step 3: Write the AST implementation**

`crates/xlog-logic/src/ast.rs`:
```rust
//! Abstract Syntax Tree for XLOG programs

use xlog_core::ScalarType;

/// A term in an atom
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    /// Variable (starts with uppercase)
    Variable(String),
    /// Integer constant
    Integer(i64),
    /// Float constant
    Float(f64),
    /// String constant
    String(String),
    /// Symbol (lowercase identifier)
    Symbol(String),
    /// Aggregate expression
    Aggregate(AggExpr),
}

impl Term {
    pub fn is_variable(&self) -> bool {
        matches!(self, Term::Variable(_))
    }

    pub fn is_constant(&self) -> bool {
        !self.is_variable() && !matches!(self, Term::Aggregate(_))
    }

    pub fn variable_name(&self) -> Option<&str> {
        match self {
            Term::Variable(name) => Some(name),
            _ => None,
        }
    }
}

/// Aggregate expression
#[derive(Debug, Clone, PartialEq)]
pub struct AggExpr {
    pub op: AggOp,
    pub variable: String,
}

/// Aggregation operator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggOp {
    Count,
    Sum,
    Min,
    Max,
}

/// An atom (predicate applied to terms)
#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    pub predicate: String,
    pub terms: Vec<Term>,
}

impl Atom {
    pub fn arity(&self) -> usize {
        self.terms.len()
    }

    /// Get all variables in this atom
    pub fn variables(&self) -> Vec<&str> {
        self.terms
            .iter()
            .filter_map(|t| t.variable_name())
            .collect()
    }
}

/// Comparison operator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// A comparison expression
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    pub left: Term,
    pub op: CompOp,
    pub right: Term,
}

/// A literal in the body of a rule
#[derive(Debug, Clone, PartialEq)]
pub enum BodyLiteral {
    /// Positive atom
    Positive(Atom),
    /// Negated atom (negation as failure)
    Negated(Atom),
    /// Comparison
    Comparison(Comparison),
}

impl BodyLiteral {
    pub fn is_positive(&self) -> bool {
        matches!(self, BodyLiteral::Positive(_))
    }

    pub fn is_negated(&self) -> bool {
        matches!(self, BodyLiteral::Negated(_))
    }

    /// Get the atom if this is an atom literal
    pub fn atom(&self) -> Option<&Atom> {
        match self {
            BodyLiteral::Positive(a) | BodyLiteral::Negated(a) => Some(a),
            BodyLiteral::Comparison(_) => None,
        }
    }

    /// Get all variables in this literal
    pub fn variables(&self) -> Vec<&str> {
        match self {
            BodyLiteral::Positive(a) | BodyLiteral::Negated(a) => a.variables(),
            BodyLiteral::Comparison(c) => {
                let mut vars = vec![];
                if let Some(v) = c.left.variable_name() {
                    vars.push(v);
                }
                if let Some(v) = c.right.variable_name() {
                    vars.push(v);
                }
                vars
            }
        }
    }
}

/// A rule (head :- body)
#[derive(Debug, Clone, PartialEq)]
pub struct Rule {
    pub head: Atom,
    pub body: Vec<BodyLiteral>,
}

impl Rule {
    /// Check if this is a fact (empty body)
    pub fn is_fact(&self) -> bool {
        self.body.is_empty()
    }

    /// Check if this rule has negation
    pub fn has_negation(&self) -> bool {
        self.body.iter().any(|l| l.is_negated())
    }

    /// Check if this rule has aggregation in head
    pub fn has_aggregation(&self) -> bool {
        self.head.terms.iter().any(|t| matches!(t, Term::Aggregate(_)))
    }

    /// Get all predicates used in the body
    pub fn body_predicates(&self) -> Vec<&str> {
        self.body
            .iter()
            .filter_map(|l| l.atom().map(|a| a.predicate.as_str()))
            .collect()
    }

    /// Get all variables in head
    pub fn head_variables(&self) -> Vec<&str> {
        self.head.variables()
    }

    /// Get all variables in body
    pub fn body_variables(&self) -> Vec<&str> {
        self.body.iter().flat_map(|l| l.variables()).collect()
    }
}

/// A constraint (:- body)
#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    pub body: Vec<BodyLiteral>,
}

/// A query (?- atom)
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    pub atom: Atom,
}

/// Domain declaration
#[derive(Debug, Clone, PartialEq)]
pub struct DomainDecl {
    pub name: String,
    pub typ: ScalarType,
}

/// Predicate declaration
#[derive(Debug, Clone, PartialEq)]
pub struct PredDecl {
    pub name: String,
    pub types: Vec<ScalarType>,
}

/// A complete XLOG program
#[derive(Debug, Clone, Default)]
pub struct Program {
    pub domains: Vec<DomainDecl>,
    pub predicates: Vec<PredDecl>,
    pub rules: Vec<Rule>,
    pub constraints: Vec<Constraint>,
    pub queries: Vec<Query>,
}

impl Program {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get all facts (rules with empty body)
    pub fn facts(&self) -> impl Iterator<Item = &Rule> {
        self.rules.iter().filter(|r| r.is_fact())
    }

    /// Get all proper rules (non-facts)
    pub fn proper_rules(&self) -> impl Iterator<Item = &Rule> {
        self.rules.iter().filter(|r| !r.is_fact())
    }

    /// Get all predicates defined in this program
    pub fn defined_predicates(&self) -> Vec<&str> {
        self.rules
            .iter()
            .map(|r| r.head.predicate.as_str())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_variable() {
        let term = Term::Variable("X".to_string());
        assert!(term.is_variable());
        assert!(!term.is_constant());
    }

    #[test]
    fn test_term_constant() {
        let term = Term::Integer(42);
        assert!(!term.is_variable());
        assert!(term.is_constant());
    }

    #[test]
    fn test_atom_arity() {
        let atom = Atom {
            predicate: "edge".to_string(),
            terms: vec![Term::Integer(1), Term::Integer(2)],
        };
        assert_eq!(atom.arity(), 2);
    }

    #[test]
    fn test_atom_variables() {
        let atom = Atom {
            predicate: "edge".to_string(),
            terms: vec![Term::Variable("X".to_string()), Term::Integer(2)],
        };
        let vars = atom.variables();
        assert_eq!(vars, vec!["X"]);
    }

    #[test]
    fn test_rule_is_fact() {
        let fact = Rule {
            head: Atom {
                predicate: "edge".to_string(),
                terms: vec![Term::Integer(1), Term::Integer(2)],
            },
            body: vec![],
        };
        assert!(fact.is_fact());
    }

    #[test]
    fn test_rule_has_negation() {
        let rule = Rule {
            head: Atom {
                predicate: "isolated".to_string(),
                terms: vec![Term::Variable("X".to_string())],
            },
            body: vec![
                BodyLiteral::Positive(Atom {
                    predicate: "node".to_string(),
                    terms: vec![Term::Variable("X".to_string())],
                }),
                BodyLiteral::Negated(Atom {
                    predicate: "edge".to_string(),
                    terms: vec![
                        Term::Variable("X".to_string()),
                        Term::Variable("Y".to_string()),
                    ],
                }),
            ],
        };
        assert!(rule.has_negation());
    }

    #[test]
    fn test_program_facts() {
        let mut program = Program::new();
        program.rules.push(Rule {
            head: Atom {
                predicate: "edge".to_string(),
                terms: vec![Term::Integer(1), Term::Integer(2)],
            },
            body: vec![],
        });
        program.rules.push(Rule {
            head: Atom {
                predicate: "reach".to_string(),
                terms: vec![Term::Variable("X".to_string()), Term::Variable("Y".to_string())],
            },
            body: vec![BodyLiteral::Positive(Atom {
                predicate: "edge".to_string(),
                terms: vec![Term::Variable("X".to_string()), Term::Variable("Y".to_string())],
            })],
        });

        assert_eq!(program.facts().count(), 1);
        assert_eq!(program.proper_rules().count(), 1);
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-logic`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-logic/src/ast.rs
git commit -m "$(cat <<'EOF'
feat(xlog-logic): implement AST types for Datalog programs

Complete AST representation:
- Term: Variable, Integer, Float, String, Symbol, Aggregate
- Atom: predicate with terms
- BodyLiteral: Positive, Negated, Comparison
- Rule: head :- body with fact/negation detection
- Program: domains, predicates, rules, constraints, queries

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Implement Stratification

**Files:**
- Create: `crates/xlog-logic/src/stratify.rs`
- Test: inline tests

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

    #[test]
    fn test_stratify_simple() {
        // edge -> reach (no negation)
        let program = create_tc_program();
        let result = stratify(&program);
        assert!(result.is_ok());
    }

    #[test]
    fn test_stratify_with_negation() {
        // node, edge -> isolated (with negation)
        let program = create_isolated_program();
        let result = stratify(&program);
        assert!(result.is_ok());
        let strata = result.unwrap();
        assert!(strata.len() >= 2); // at least 2 strata due to negation
    }

    #[test]
    fn test_stratify_cycle_through_negation() {
        // p :- not q. q :- not p. (cycle through negation)
        let program = create_unstratifiable_program();
        let result = stratify(&program);
        assert!(result.is_err());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-logic`
Expected: FAIL with "cannot find function `stratify`"

**Step 3: Implement stratification**

`crates/xlog-logic/src/stratify.rs`:
```rust
//! Stratification analysis for negation and aggregation

use std::collections::{HashMap, HashSet};
use xlog_core::{XlogError, Result};
use crate::ast::{Program, Rule, BodyLiteral};

/// Dependency edge type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepType {
    /// Positive dependency (no stratum constraint)
    Positive,
    /// Negative dependency (must be in lower stratum)
    Negative,
    /// Aggregate dependency (must be in lower stratum)
    Aggregate,
}

/// Dependency graph edge
#[derive(Debug, Clone)]
pub struct DepEdge {
    pub from: String,
    pub to: String,
    pub dep_type: DepType,
}

/// Dependency graph for stratification analysis
#[derive(Debug, Default)]
pub struct DependencyGraph {
    /// All predicates
    pub predicates: HashSet<String>,
    /// Edges: from predicate depends on to predicate
    pub edges: Vec<DepEdge>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_predicate(&mut self, name: String) {
        self.predicates.insert(name);
    }

    pub fn add_edge(&mut self, from: String, to: String, dep_type: DepType) {
        self.predicates.insert(from.clone());
        self.predicates.insert(to.clone());
        self.edges.push(DepEdge { from, to, dep_type });
    }

    /// Get all outgoing edges from a predicate
    pub fn outgoing(&self, pred: &str) -> Vec<&DepEdge> {
        self.edges.iter().filter(|e| e.from == pred).collect()
    }
}

/// Build dependency graph from program
pub fn build_dependency_graph(program: &Program) -> DependencyGraph {
    let mut graph = DependencyGraph::new();

    for rule in &program.rules {
        let head = &rule.head.predicate;
        graph.add_predicate(head.clone());

        for lit in &rule.body {
            match lit {
                BodyLiteral::Positive(atom) => {
                    graph.add_edge(head.clone(), atom.predicate.clone(), DepType::Positive);
                }
                BodyLiteral::Negated(atom) => {
                    graph.add_edge(head.clone(), atom.predicate.clone(), DepType::Negative);
                }
                BodyLiteral::Comparison(_) => {}
            }
        }

        // Check for aggregation in head
        if rule.has_aggregation() {
            // Aggregation creates a dependency similar to negation
            for lit in &rule.body {
                if let BodyLiteral::Positive(atom) = lit {
                    graph.add_edge(head.clone(), atom.predicate.clone(), DepType::Aggregate);
                }
            }
        }
    }

    graph
}

/// Find strongly connected components using Tarjan's algorithm
fn find_sccs(graph: &DependencyGraph) -> Vec<Vec<String>> {
    let mut index_counter = 0;
    let mut stack = Vec::new();
    let mut indices: HashMap<String, usize> = HashMap::new();
    let mut lowlinks: HashMap<String, usize> = HashMap::new();
    let mut on_stack: HashSet<String> = HashSet::new();
    let mut sccs: Vec<Vec<String>> = Vec::new();

    fn strongconnect(
        v: &str,
        graph: &DependencyGraph,
        index_counter: &mut usize,
        stack: &mut Vec<String>,
        indices: &mut HashMap<String, usize>,
        lowlinks: &mut HashMap<String, usize>,
        on_stack: &mut HashSet<String>,
        sccs: &mut Vec<Vec<String>>,
    ) {
        indices.insert(v.to_string(), *index_counter);
        lowlinks.insert(v.to_string(), *index_counter);
        *index_counter += 1;
        stack.push(v.to_string());
        on_stack.insert(v.to_string());

        for edge in graph.outgoing(v) {
            let w = &edge.to;
            if !indices.contains_key(w) {
                strongconnect(w, graph, index_counter, stack, indices, lowlinks, on_stack, sccs);
                let low_v = *lowlinks.get(v).unwrap();
                let low_w = *lowlinks.get(w).unwrap();
                lowlinks.insert(v.to_string(), low_v.min(low_w));
            } else if on_stack.contains(w) {
                let low_v = *lowlinks.get(v).unwrap();
                let idx_w = *indices.get(w).unwrap();
                lowlinks.insert(v.to_string(), low_v.min(idx_w));
            }
        }

        let low_v = *lowlinks.get(v).unwrap();
        let idx_v = *indices.get(v).unwrap();
        if low_v == idx_v {
            let mut scc = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.remove(&w);
                scc.push(w.clone());
                if w == v {
                    break;
                }
            }
            sccs.push(scc);
        }
    }

    for pred in &graph.predicates {
        if !indices.contains_key(pred) {
            strongconnect(
                pred,
                graph,
                &mut index_counter,
                &mut stack,
                &mut indices,
                &mut lowlinks,
                &mut on_stack,
                &mut sccs,
            );
        }
    }

    sccs
}

/// Check for cycles through negation/aggregation in an SCC
fn check_scc_for_negation_cycle(scc: &[String], graph: &DependencyGraph) -> Option<Vec<String>> {
    if scc.len() == 1 {
        // Single predicate SCC - check for self-reference through negation
        let pred = &scc[0];
        for edge in graph.outgoing(pred) {
            if edge.to == *pred && edge.dep_type != DepType::Positive {
                return Some(vec![pred.clone()]);
            }
        }
        return None;
    }

    // Multi-predicate SCC - any negative edge within SCC is a problem
    let scc_set: HashSet<&str> = scc.iter().map(|s| s.as_str()).collect();
    for pred in scc {
        for edge in graph.outgoing(pred) {
            if scc_set.contains(edge.to.as_str()) && edge.dep_type != DepType::Positive {
                return Some(scc.to_vec());
            }
        }
    }
    None
}

/// Stratum assignment result
#[derive(Debug, Clone)]
pub struct Stratum {
    pub id: usize,
    pub predicates: Vec<String>,
}

/// Perform stratification analysis
pub fn stratify(program: &Program) -> Result<Vec<Stratum>> {
    let graph = build_dependency_graph(program);
    let sccs = find_sccs(&graph);

    // Check for cycles through negation
    for scc in &sccs {
        if let Some(cycle) = check_scc_for_negation_cycle(scc, &graph) {
            return Err(XlogError::StratificationCycle(cycle));
        }
    }

    // Assign strata based on dependencies
    let mut stratum_map: HashMap<String, usize> = HashMap::new();
    let mut max_stratum = 0;

    // Process SCCs in reverse topological order
    for scc in sccs.iter().rev() {
        let mut min_stratum = 0;

        // Find minimum stratum based on dependencies
        for pred in scc {
            for edge in graph.outgoing(pred) {
                if let Some(&dep_stratum) = stratum_map.get(&edge.to) {
                    let required = match edge.dep_type {
                        DepType::Positive => dep_stratum,
                        DepType::Negative | DepType::Aggregate => dep_stratum + 1,
                    };
                    min_stratum = min_stratum.max(required);
                }
            }
        }

        // Assign this stratum to all predicates in SCC
        for pred in scc {
            stratum_map.insert(pred.clone(), min_stratum);
        }
        max_stratum = max_stratum.max(min_stratum);
    }

    // Group predicates by stratum
    let mut strata: Vec<Stratum> = (0..=max_stratum)
        .map(|id| Stratum {
            id,
            predicates: vec![],
        })
        .collect();

    for (pred, stratum) in stratum_map {
        strata[stratum].predicates.push(pred);
    }

    // Remove empty strata
    strata.retain(|s| !s.predicates.is_empty());

    // Renumber
    for (i, stratum) in strata.iter_mut().enumerate() {
        stratum.id = i;
    }

    Ok(strata)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

    fn create_tc_program() -> Program {
        let mut program = Program::new();

        // edge(1,2). edge(2,3).
        program.rules.push(Rule {
            head: Atom { predicate: "edge".into(), terms: vec![Term::Integer(1), Term::Integer(2)] },
            body: vec![],
        });

        // reach(X,Y) :- edge(X,Y).
        program.rules.push(Rule {
            head: Atom {
                predicate: "reach".into(),
                terms: vec![Term::Variable("X".into()), Term::Variable("Y".into())],
            },
            body: vec![BodyLiteral::Positive(Atom {
                predicate: "edge".into(),
                terms: vec![Term::Variable("X".into()), Term::Variable("Y".into())],
            })],
        });

        // reach(X,Z) :- reach(X,Y), edge(Y,Z).
        program.rules.push(Rule {
            head: Atom {
                predicate: "reach".into(),
                terms: vec![Term::Variable("X".into()), Term::Variable("Z".into())],
            },
            body: vec![
                BodyLiteral::Positive(Atom {
                    predicate: "reach".into(),
                    terms: vec![Term::Variable("X".into()), Term::Variable("Y".into())],
                }),
                BodyLiteral::Positive(Atom {
                    predicate: "edge".into(),
                    terms: vec![Term::Variable("Y".into()), Term::Variable("Z".into())],
                }),
            ],
        });

        program
    }

    fn create_isolated_program() -> Program {
        let mut program = Program::new();

        // node(1). node(2). node(3).
        for i in 1..=3 {
            program.rules.push(Rule {
                head: Atom { predicate: "node".into(), terms: vec![Term::Integer(i)] },
                body: vec![],
            });
        }

        // edge(1, 2).
        program.rules.push(Rule {
            head: Atom { predicate: "edge".into(), terms: vec![Term::Integer(1), Term::Integer(2)] },
            body: vec![],
        });

        // isolated(X) :- node(X), not edge(X, _), not edge(_, X).
        program.rules.push(Rule {
            head: Atom {
                predicate: "isolated".into(),
                terms: vec![Term::Variable("X".into())],
            },
            body: vec![
                BodyLiteral::Positive(Atom {
                    predicate: "node".into(),
                    terms: vec![Term::Variable("X".into())],
                }),
                BodyLiteral::Negated(Atom {
                    predicate: "edge".into(),
                    terms: vec![Term::Variable("X".into()), Term::Variable("Y".into())],
                }),
            ],
        });

        program
    }

    fn create_unstratifiable_program() -> Program {
        let mut program = Program::new();

        // p :- not q.
        program.rules.push(Rule {
            head: Atom { predicate: "p".into(), terms: vec![] },
            body: vec![BodyLiteral::Negated(Atom { predicate: "q".into(), terms: vec![] })],
        });

        // q :- not p.
        program.rules.push(Rule {
            head: Atom { predicate: "q".into(), terms: vec![] },
            body: vec![BodyLiteral::Negated(Atom { predicate: "p".into(), terms: vec![] })],
        });

        program
    }

    #[test]
    fn test_stratify_simple() {
        let program = create_tc_program();
        let result = stratify(&program);
        assert!(result.is_ok(), "Stratification failed: {:?}", result.err());
    }

    #[test]
    fn test_stratify_with_negation() {
        let program = create_isolated_program();
        let result = stratify(&program);
        assert!(result.is_ok(), "Stratification failed: {:?}", result.err());
        let strata = result.unwrap();
        assert!(strata.len() >= 2, "Expected at least 2 strata, got {}", strata.len());
    }

    #[test]
    fn test_stratify_cycle_through_negation() {
        let program = create_unstratifiable_program();
        let result = stratify(&program);
        assert!(result.is_err(), "Should fail with cycle through negation");
        if let Err(XlogError::StratificationCycle(preds)) = result {
            assert!(preds.contains(&"p".to_string()) || preds.contains(&"q".to_string()));
        }
    }

    #[test]
    fn test_dependency_graph_construction() {
        let program = create_tc_program();
        let graph = build_dependency_graph(&program);

        assert!(graph.predicates.contains("edge"));
        assert!(graph.predicates.contains("reach"));

        let reach_deps: Vec<_> = graph.outgoing("reach").collect();
        assert!(!reach_deps.is_empty());
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-logic`
Expected: PASS

**Step 5: Update lib.rs**

Update `crates/xlog-logic/src/lib.rs`:
```rust
//! Datalog frontend for XLOG

pub mod parser;
pub mod ast;
pub mod stratify;
pub mod lower;
pub mod compile;

pub use parser::{parse_program, parse_statement};
pub use ast::{Program, Rule, Atom, Term, BodyLiteral};
pub use stratify::{stratify, Stratum, DependencyGraph};
```

**Step 6: Run all tests**

Run: `cargo test -p xlog-logic`
Expected: PASS

**Step 7: Commit**

```bash
git add crates/xlog-logic/
git commit -m "$(cat <<'EOF'
feat(xlog-logic): implement stratification with Tarjan's SCC

Stratification analysis for negation and aggregation:
- Build dependency graph from program rules
- Find SCCs using Tarjan's algorithm
- Check for cycles through negation/aggregation
- Assign strata in topological order
- Return StratificationCycle error for unstratifiable programs

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5: Integration Tests (Success Criteria)

### Task 13: Create Integration Test for Transitive Closure

**Files:**
- Create: `tests/logic/tc.xlog`
- Create: `tests/logic/mod.rs`
- Create: `tests/integration_tests.rs`

**Step 1: Create test program file**

`tests/logic/tc.xlog`:
```prolog
% Transitive closure test program
edge(1, 2).
edge(2, 3).
edge(3, 4).

reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).

?- reach(1, N).
```

**Step 2: Create stratified negation test**

`tests/logic/stratified.xlog`:
```prolog
% Stratified negation test program
node(1).
node(2).
node(3).
edge(1, 2).

isolated(X) :- node(X), not edge(X, Y), not edge(Y, X).

?- isolated(N).
```

**Step 3: Create aggregate test**

`tests/logic/aggregates.xlog`:
```prolog
% Aggregate test program
edge(1, 2).
edge(1, 3).
edge(2, 4).

out_degree(X, count(Y)) :- edge(X, Y).

?- out_degree(1, N).
```

**Step 4: Create integration test**

`tests/integration_tests.rs`:
```rust
//! Integration tests for xlog-logic

use xlog_logic::{parse_program, stratify};

#[test]
fn test_parse_tc_program() {
    let input = include_str!("logic/tc.xlog");
    let result = parse_program(input);
    assert!(result.is_ok(), "Failed to parse TC program: {:?}", result.err());
}

#[test]
fn test_parse_stratified_program() {
    let input = include_str!("logic/stratified.xlog");
    let result = parse_program(input);
    assert!(result.is_ok(), "Failed to parse stratified program: {:?}", result.err());
}

#[test]
fn test_parse_aggregate_program() {
    let input = include_str!("logic/aggregates.xlog");
    let result = parse_program(input);
    assert!(result.is_ok(), "Failed to parse aggregate program: {:?}", result.err());
}

// Note: Full execution tests require xlog-cuda and xlog-runtime
// which depend on CUDA hardware. These will be added in later tasks.
```

**Step 5: Run integration tests**

Run: `cargo test --test integration_tests`
Expected: PASS

**Step 6: Commit**

```bash
git add tests/
git commit -m "$(cat <<'EOF'
test: add integration tests for xlog-logic parsing

Test programs matching success criteria:
- tc.xlog: transitive closure (recursion)
- stratified.xlog: stratified negation
- aggregates.xlog: count aggregation

Full execution tests pending CUDA runtime implementation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

This plan implements the xlog-logic MVP in bite-sized TDD steps:

1. **Phase 0**: Workspace skeleton with 5 crates
2. **Phase 1**: xlog-core (errors, types, config, traits)
3. **Phase 2**: xlog-ir (metadata, RIR nodes, execution plans)
4. **Phase 3**: CUDA kernel stubs with CMake
5. **Phase 4**: xlog-logic parser, AST, stratification
6. **Phase 5**: Integration tests

**Remaining tasks** (not detailed above, follow same pattern):
- Task 14-17: AST-to-Parse conversion, lowering to RIR
- Task 18-21: xlog-cuda device/memory/provider implementation
- Task 22-25: xlog-runtime executor and fixpoint loop
- Task 26-30: Full CUDA kernel implementations
- Task 31-35: End-to-end execution tests with CUDA

---

Plan complete and saved to `docs/plans/2026-01-07-xlog-logic-implementation-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
