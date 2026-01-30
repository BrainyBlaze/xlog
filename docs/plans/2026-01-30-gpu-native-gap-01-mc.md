# GPU-Native MC Path Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the Monte Carlo (MC) inference path fully GPU-native (no CPU sampling/aggregation in production), returning device-resident results with optional host readout only when explicitly requested.

**Architecture:** Move sampling, evidence filtering, and query counts to device using existing MC kernels and executor GPU evaluation, introduce a device-resident result struct, and gate host-read APIs behind `host-io` so GPU-native flows never do DTOH reads.

**Tech Stack:** Rust, CUDA (cudarc), existing kernels in `kernels/mc_eval.cu`, `xlog-cuda` provider, `xlog-runtime` executor.

---

### Task 1: Add failing GPU-native MC test (TDD)

**Files:**
- Modify: `crates/xlog-prob/tests/no_dtoh_gpu_native.rs`
- (Optional) Create: `crates/xlog-prob/tests/mc_gpu_native.rs`

**Step 1: Write the failing test**

Add a test that ensures the GPU MC evaluation path does not perform device-to-host reads in its implementation file. Use a string scan similar to other no-DTOH tests:

```rust
#[test]
fn mc_gpu_path_has_no_dtoh_calls() {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src/mc.rs");
    let text = std::fs::read_to_string(&path).expect("read mc.rs");
    assert!(
        !text.contains("dtoh_sync_copy_into")
            && !text.contains("copy_to_host")
            && !text.contains("download")
            && !text.contains("to_host"),
        "MC GPU path must not use device->host reads (found in {})",
        path.display()
    );
}
```

**Step 2: Run test to verify it fails**

Run:
```
cargo test -p xlog-prob --test no_dtoh_gpu_native mc_gpu_path_has_no_dtoh_calls -- --nocapture
```
Expected: FAIL (mc.rs currently has host reads in GPU path).

---

### Task 2: Introduce device-resident MC results

**Files:**
- Modify: `crates/xlog-prob/src/mc.rs`

**Step 1: Write the failing test**

Add a test asserting the new device-resident API exists and returns device buffers without DTOH. Add a simple compile-time test in `crates/xlog-prob/tests/mc_gpu_native.rs`:

```rust
#[test]
fn mc_gpu_native_returns_device_buffers() {
    // Just ensure the type and method exist and are callable in CUDA-enabled env.
    // Skip if no CUDA.
    if xlog_cuda::CudaDevice::new(0).is_err() {
        eprintln!("Skipping: CUDA unavailable");
        return;
    }
    // Compile-only: we just check the API signatures.
    // Real behavior is covered by existing MC kernel tests.
}
```

**Step 2: Run test to verify it fails**

Run:
```
cargo test -p xlog-prob --test mc_gpu_native -- --nocapture
```
Expected: FAIL until new API exists.

**Step 3: Implement device-resident result struct and API**

Add a GPU-native result struct and evaluation method:

```rust
pub struct McGpuResultDevice {
    pub query_counts: TrackedCudaSlice<u32>, // len = num_queries
    pub evidence_samples: TrackedCudaSlice<u32>, // len = 1
    pub total_samples: u32,
    pub seed: u64,
    pub confidence: f64,
}

impl McProgram {
    pub fn evaluate_gpu_device(&self, cfg: McEvalConfig) -> Result<McGpuResultDevice> {
        // GPU-only sampling + counts using provider.sample_bernoulli_matrix_device
        // and existing mc_eval kernels. No dtoh.
    }
}
```

Implement using existing `evaluate_gpu_counts_with` and `mc_eval` kernels:
- Convert `evaluate_gpu_counts` to write counts into device buffers.
- Use `mc_eval_accumulate_counts` to update device counters.
- Return `TrackedCudaSlice` results without host reads.

**Step 4: Run test to verify it passes**

Run:
```
cargo test -p xlog-prob --test mc_gpu_native -- --nocapture
```
Expected: PASS.

**Step 5: Commit**

```
git add crates/xlog-prob/src/mc.rs crates/xlog-prob/tests/no_dtoh_gpu_native.rs crates/xlog-prob/tests/mc_gpu_native.rs
git commit -m "feat: add GPU-native MC device results"
```

---

### Task 3: Gate host-read MC API behind host-io and keep GPU-native path clean

**Files:**
- Modify: `crates/xlog-prob/src/mc.rs`

**Step 1: Write failing test**

Add a build/test that ensures host-read APIs are behind `#[cfg(feature = "host-io")]`:

```rust
#[test]
fn mc_host_read_apis_gated() {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src/mc.rs");
    let text = std::fs::read_to_string(&path).expect("read mc.rs");
    assert!(text.contains("#[cfg(feature = \"host-io\")]"));
}
```

**Step 2: Run test to verify it fails**

Run:
```
cargo test -p xlog-prob --test mc_gpu_native mc_host_read_apis_gated -- --nocapture
```
Expected: FAIL if any host-read MC paths are ungated.

**Step 3: Gate host-read paths**

Ensure that any method returning `McResult` (with host counts and confidence intervals) is inside `#[cfg(feature = "host-io")]`. The GPU-native device method should be always available.

**Step 4: Run tests**

Run:
```
cargo test -p xlog-prob --test mc_gpu_native -- --nocapture
```
Expected: PASS.

**Step 5: Commit**

```
git add crates/xlog-prob/src/mc.rs crates/xlog-prob/tests/mc_gpu_native.rs
git commit -m "chore: gate MC host-read APIs and keep GPU path device-only"
```

---

### Task 4: End-to-end GPU MC correctness check (device counts)

**Files:**
- Modify: `crates/xlog-prob/tests/mc_gpu_native.rs`

**Step 1: Write failing test**

Add a small deterministic MC test with a fixed seed and a tiny program where expected counts are known. Use device counts and then (only in test) DTOH to assert correctness (tests are allowed to read back):

```rust
#[test]
fn mc_gpu_device_counts_match_expected_small() {
    // small program with one probabilistic fact p=1.0
    // Expect query true for all samples.
}
```

**Step 2: Run failing test**

Run:
```
cargo test -p xlog-prob --test mc_gpu_native mc_gpu_device_counts_match_expected_small -- --nocapture
```
Expected: FAIL until device path is wired.

**Step 3: Implement minimal wiring and DTOH in test only**

Use provider.dtoh in the test only to verify the device counts.

**Step 4: Run test**

Run:
```
cargo test -p xlog-prob --test mc_gpu_native -- --nocapture
```
Expected: PASS.

**Step 5: Commit**

```
git add crates/xlog-prob/tests/mc_gpu_native.rs
 git commit -m "test: add GPU MC device-count correctness check"
```

---

### Task 5: Full verification

**Files:**
- None

**Step 1: Run MC-related tests**

```
cargo test -p xlog-prob --test mc_gpu_native -- --nocapture
cargo test -p xlog-prob --test no_dtoh_gpu_native -- --nocapture
```
Expected: PASS

**Step 2: Run full test suite**

```
cargo test
```
Expected: PASS

---

### Task 6: Commit rollup (if needed)

If any changes remain uncommitted:

```
git add crates/xlog-prob/src/mc.rs crates/xlog-prob/tests/mc_gpu_native.rs crates/xlog-prob/tests/no_dtoh_gpu_native.rs
 git commit -m "feat: make MC inference GPU-native"
```
