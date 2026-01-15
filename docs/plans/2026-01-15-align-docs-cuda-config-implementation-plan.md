# Align Negation Docs, CUDA Packaging, and Runtime Iteration Limits Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align documentation negation syntax with the parser, ensure CUDA PTX packaging is consistent with runtime loading, and honor runtime-configured SCC iteration limits.

**Architecture:** Update the validation report plan to use `not` instead of `\+`, compile `circuit` and `mc_sample` kernels in `build.rs`, and store `RuntimeConfig` on the executor so recursive SCC evaluation uses the configured limit.

**Tech Stack:** Rust (xlog-runtime, xlog-cuda, xlog-logic), CUDA PTX build via `nvcc`, Markdown documentation.

### Task 1: Align validation report negation syntax

**Files:**
- Modify: `docs/plans/2026-01-11-full-system-validation-report.md`

**Step 1: Update negation examples to `not`**

Edit the three `\+` occurrences:

```text
| Negation | `not atom` | `safe(X) :- node(X), not danger(X).` |

safe(X) :- node(X), not danger(X).
```

**Step 2: Verify no `\+` remains**

Run: `rg -n -F "\\+" docs/plans/2026-01-11-full-system-validation-report.md`

Expected: no output.

**Step 3: Commit**

```bash
git add docs/plans/2026-01-11-full-system-validation-report.md
git commit -m "docs: align negation syntax in validation report plan"
```

### Task 2: Ensure build script compiles circuit and mc_sample kernels

**Files:**
- Create: `crates/xlog-cuda/tests/build_script_tests.rs`
- Modify: `crates/xlog-cuda/build.rs`

**Step 1: Write the failing test**

```rust
use std::fs;
use std::path::PathBuf;

#[test]
fn test_build_script_includes_circuit_and_mc_sample() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let build_rs = manifest_dir.join("build.rs");
    let contents = fs::read_to_string(&build_rs).expect("read build.rs");

    assert!(
        contents.contains("\"circuit\""),
        "build.rs should list circuit kernel for PTX compilation"
    );
    assert!(
        contents.contains("\"mc_sample\""),
        "build.rs should list mc_sample kernel for PTX compilation"
    );
}
```

**Step 2: Run the test to verify failure**

Run: `cargo test -p xlog-cuda --test build_script_tests`

Expected: FAIL with assertion "build.rs should list circuit kernel for PTX compilation".

**Step 3: Update build script kernel list**

In `crates/xlog-cuda/build.rs`, update the kernel list to:

```rust
let kernels = [
    "join", "dedup", "groupby", "scan", "sort", "filter", "pack", "set_ops",
    "circuit", "mc_sample",
];
```

**Step 4: Re-run the test to verify pass**

Run: `cargo test -p xlog-cuda --test build_script_tests`

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/xlog-cuda/build.rs crates/xlog-cuda/tests/build_script_tests.rs
git commit -m "build: compile circuit and mc_sample kernels"
```

### Task 3: Honor RuntimeConfig.max_iterations in recursive SCC evaluation

**Files:**
- Modify: `crates/xlog-runtime/Cargo.toml`
- Create: `crates/xlog-runtime/tests/executor_config_tests.rs`
- Modify: `crates/xlog-runtime/src/executor.rs`

**Step 1: Add dev-dependency and write the failing test**

Add to `crates/xlog-runtime/Cargo.toml`:

```toml
[dev-dependencies]
xlog-logic = { path = "../xlog-logic" }
```

Create `crates/xlog-runtime/tests/executor_config_tests.rs`:

```rust
use std::sync::Arc;

use xlog_core::{MemoryBudget, RuntimeConfig, ScalarType, Schema, XlogError};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_logic::Compiler;
use xlog_runtime::Executor;

fn has_cuda_device() -> bool {
    CudaDevice::new(0).is_ok()
}

fn create_executor_with_config(
    config: RuntimeConfig,
) -> Option<(Executor, Arc<CudaKernelProvider>)> {
    if !has_cuda_device() {
        return None;
    }

    let device = Arc::new(CudaDevice::new(0).ok()?);
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    let provider = Arc::new(CudaKernelProvider::new(device, memory).ok()?);
    let executor = Executor::new_with_config(provider.clone(), config);

    Some((executor, provider))
}

fn create_edge_buffer(
    provider: &CudaKernelProvider,
    edges: &[(u32, u32)],
) -> CudaBuffer {
    let schema = Schema::new(vec![
        ("c0".to_string(), ScalarType::U32),
        ("c1".to_string(), ScalarType::U32),
    ]);

    if edges.is_empty() {
        let col0 = provider.memory().alloc::<u8>(0).expect("alloc");
        let col1 = provider.memory().alloc::<u8>(0).expect("alloc");
        return CudaBuffer::from_columns(vec![col0.into(), col1.into()], 0, schema);
    }

    let col0_bytes: Vec<u8> = edges.iter().flat_map(|(from, _)| from.to_le_bytes()).collect();
    let col1_bytes: Vec<u8> = edges.iter().flat_map(|(_, to)| to.to_le_bytes()).collect();

    let mut col0 = provider.memory().alloc::<u8>(col0_bytes.len()).expect("alloc");
    let mut col1 = provider.memory().alloc::<u8>(col1_bytes.len()).expect("alloc");

    provider.device().inner().htod_sync_copy_into(&col0_bytes, &mut col0).expect("htod");
    provider.device().inner().htod_sync_copy_into(&col1_bytes, &mut col1).expect("htod");

    CudaBuffer::from_columns(vec![col0.into(), col1.into()], edges.len() as u64, schema)
}

fn setup_executor_with_facts(
    executor: &mut Executor,
    compiler: &Compiler,
    facts: Vec<(&str, CudaBuffer)>,
) {
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }

    for (name, buffer) in facts {
        executor.store_mut().put(name, buffer);
    }
}

#[test]
fn test_executor_respects_max_iterations() {
    let mut config = RuntimeConfig::default();
    config.max_iterations = 1;

    let (mut executor, provider) = match create_executor_with_config(config) {
        Some(e) => e,
        None => {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }
    };

    let source = r#"
        edge(1, 2).
        edge(2, 3).
        edge(3, 4).
        reach(X, Y) :- edge(X, Y).
        reach(X, Z) :- reach(X, Y), edge(Y, Z).
    "#;

    let mut compiler = Compiler::new();
    let plan = compiler.compile(source).expect("Compilation failed");
    assert!(plan.has_recursion(), "Expected recursive plan");

    let edge_buffer = create_edge_buffer(&provider, &[(1, 2), (2, 3), (3, 4)]);
    setup_executor_with_facts(&mut executor, &compiler, vec![("edge", edge_buffer)]);

    let err = executor
        .execute_plan(&plan)
        .expect_err("expected iteration cap error");
    let msg = err.to_string();
    assert!(
        msg.contains("iteration limit (1)"),
        "unexpected error message: {msg}"
    );
}
```

**Step 2: Run the test to verify failure**

Run: `cargo test -p xlog-runtime --test executor_config_tests`

Expected: FAIL to compile with `no function or associated item named \`new_with_config\``.

**Step 3: Implement config-aware executor**

In `crates/xlog-runtime/src/executor.rs`:

- Add `RuntimeConfig` to the import list.
- Add a `config: RuntimeConfig` field to `Executor`.
- Add constructor:

```rust
pub fn new_with_config(provider: Arc<CudaKernelProvider>, config: RuntimeConfig) -> Self {
    const DEFAULT_JOIN_INDEX_CACHE_BYTES: u64 = 256 * 1024 * 1024;
    let max_index_cache_bytes = (provider.memory().budget().device_bytes / 4)
        .min(DEFAULT_JOIN_INDEX_CACHE_BYTES);
    Self {
        provider,
        store: RelationStore::new(),
        rel_names: HashMap::new(),
        name_to_rel: HashMap::new(),
        stats: StatsManager::new(),
        join_index_cache: JoinIndexCache::new(max_index_cache_bytes),
        config,
    }
}
```

- Update `Executor::new` to delegate to the default config:

```rust
pub fn new(provider: Arc<CudaKernelProvider>) -> Self {
    Self::new_with_config(provider, RuntimeConfig::default())
}
```

- Replace the SCC loop bound to use the config:

```rust
let max_iterations = self.config.max_iterations as usize;
for _iteration in 0..max_iterations {
    // existing loop body
}

if !reached_fixpoint {
    return Err(XlogError::Execution(format!(
        "Recursive SCC iteration limit ({}) exceeded",
        self.config.max_iterations
    )));
}
```

**Step 4: Re-run the test to verify pass**

Run: `cargo test -p xlog-runtime --test executor_config_tests`

Expected: PASS (or skipped if no CUDA device).

**Step 5: Commit**

```bash
git add crates/xlog-runtime/Cargo.toml crates/xlog-runtime/src/executor.rs crates/xlog-runtime/tests/executor_config_tests.rs
git commit -m "feat: honor runtime max_iterations in executor"
```
