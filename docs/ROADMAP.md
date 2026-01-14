# XLOG Development Roadmap

**Last Updated:** January 14, 2026
**Current Status:** Phase 4 Complete (integrated `xlog-prob` + Python `xlog-gpu`; CUDA certification passing)

---

## Core Language & Compiler (xlog-logic)

### Implemented
- Datalog parsing and compilation. [v0.2.0]
- Stratified negation analysis. [v0.2.0]
- `.xlog` queries (`?-`) and constraints (`:-`) via compilation desugaring + runner enforcement. [v0.2.0]
- `symbol` values are currently represented as a `u32` hash (MVP, not reversible). [v0.2.0]

### Planned

## Runtime & Execution (xlog-runtime)

### Implemented
- Semi-naive fixpoint iteration. [v0.2.0]
- E2E transitive closure queries. [v0.2.0]
- P2.2 Float Comparisons in Filters — runtime filter predicate evaluation supports F32/F64 comparisons. Files: `crates/xlog-runtime/src/executor.rs`. [v0.2.0]
- Some aggregate value-type restrictions remain (see `docs/ARCHITECTURE.md`). [v0.2.0]

### Planned
- GPU-resident execution fully on GPU (currently partial; arithmetic/utility helpers remain on CPU). [v0.3.x]
- CLI/REPL interface. [v0.3.x]

## GPU Backend & Kernels (xlog-cuda + kernels)

### Implemented
- GPU hash joins (inner/semi/anti/left-outer; on-device materialization; key verification; byte-hashed scalar keys). [v0.2.0]
- Sort correctness for all scalar key types (stable GPU permutation + GPU reorder). [v0.2.0]
- GPU set operations (union, diff). [v0.2.0]
- GPU aggregations (count, sum, min, max) including multi-key groupby. [v0.2.0]
- P2.3 Implement LogSumExp — Files: `kernels/groupby.cu`, `crates/xlog-cuda/src/provider.rs`. [v0.2.0]
- CUDA kernels: join, dedup, groupby, scan, filter, pack, sort, set_ops, circuit, mc_sample. [v0.2.0]
- P1.1 Join Correctness (Hash Collision Safety) — Join v2 verifies key bytes after hash match (no false positives). Files: `kernels/join.cu`, `crates/xlog-cuda/src/provider.rs`. [v0.2.0]
- P1.2 Aggregation Sum Overflow/Truncation — Sum aggregates return `U64` and the schema reflects it. Files: `crates/xlog-cuda/src/provider.rs`. [v0.2.0]
- P1.3 Large-Input Filter/Compaction — Multi-block scan works beyond 256 elements fully on GPU (recursive scan of block sums). Files: `kernels/scan.cu`, `crates/xlog-cuda/src/provider.rs`. [v0.2.0]
- P1.4 GPU Memory Budget Enforcement — Atomic budget reservation + RAII accounting for frees. Files: `crates/xlog-cuda/src/memory.rs`, `crates/xlog-cuda/src/multi_gpu_memory.rs`. [v0.2.0]
- P1.5 Remove CPU Roundtrips in Dedup — Issue: Dedup previously downloaded keys to host for boundary detection. Solution: GPU columnar boundary detection + GPU scan/compact (mask/prefix/compact stays on-device). Remaining: None (P3.3 removed the CPU sort permutation). Files: `crates/xlog-cuda/src/provider.rs`, `kernels/dedup.cu`. [v0.2.0]
- P2.1 Extend Type Support — `union_gpu`/`diff_gpu` support all scalar types (and multi-column schemas) without CPU row-materialization; GPU sort permutation is now fully on-device (P3.3). Goal: uniform scalar support across join/sort/filter/dedup/set-ops (I32/I64/U64/F32/F64/Bool/Symbol). Files: `crates/xlog-runtime/src/executor.rs`, `crates/xlog-cuda/src/provider.rs`, `kernels/*.cu`. Effort: 1-2 weeks. [v0.2.0]
- P2.4 Increase Join Output Limit — Join output buffers are sized from a count pass (no silent truncation); optional cap still supported. Files: `crates/xlog-cuda/src/provider.rs`. [v0.2.0]
- P2.5 Key Verification Mode — Join v2 performs key verification by default after hash match. Goal: optional unsafe "hash-only" fast path for performance experiments (explicitly opt-in). Files: `kernels/join.cu`, `crates/xlog-cuda/src/provider.rs`. [v0.2.0]
- P3.1 Optimize Radix Scatter — Precompute per-digit per-block offsets via GPU prefix sums (no per-element block loop). [v0.2.0]
- P3.2 Coalesced Memory Access — Cache-friendly bucket layout (CSR buckets: counts + offsets + contiguous entries + hashes). Effort: 1 week. [v0.2.0]
- P3.3 Multi-Column GPU Sort — Stable GPU radix sort generates the permutation on-device for all scalar key types and multi-column lexicographic keys (no host roundtrip). [v0.2.0]
- P3.4 Kernel Fusion — Fuse filter compare+scan and dedup unique+scan; remove count-kernel pass by computing output counts from final prefix element. Effort: 1-2 weeks. [v0.2.0]

### Planned

## Optimizer & Stats (xlog-solve + xlog-stats)

### Implemented
- P4.2 Query Optimizer — predicate pushdown + cost-based join planning. Includes bushy DP join trees for small bodies (≤10 atoms) with build/probe cost model; greedy bushy join planning fallback for large bodies; cartesian joins supported via constant-key join (avoids empty-key GPU join errors); compiler can seed optimizer from `xlog_stats::StatsSnapshot` (runtime → compiler feedback loop); stats snapshots include predicate names for safe `RelId` remapping and join ordering. [v0.2.0]

### Planned

## Incremental Maintenance & Adaptive Indexing

### Implemented
- P4.3 Incremental Maintenance — semi-naive SCC evaluation + delta application API; runtime recursive SCC evaluation uses semi-naive deltas with per-scan occurrence rewriting; `Executor::apply_deltas_and_recompute` supports insert-only incremental updates for monotone SCCs and recompute for non-monotone SCCs and dependents; deletes recompute affected SCC closure. [v0.2.0]
- P4.4 Adaptive Indexing (HISA) — runtime `Executor` maintains `xlog_stats::StatsManager` for scan heat/cardinality/bytes and join selectivities; join index cache with LRU eviction, invalidation on relation updates, and budget-aware sizing/heuristics (build-side hash reuse when right side is a hot Scan relation). [v0.2.0]

### Planned

## Interop (Arrow/DLPack/CuDF)

### Implemented
- P4.1 CuDF Integration — Arrow RecordBatch export/import (`CudaKernelProvider::{to_arrow_record_batch, from_arrow_record_batch}`); Arrow IPC stream helpers (`CudaKernelProvider::{to_arrow_ipc_stream, from_arrow_ipc_stream, write_arrow_ipc_stream_file, read_arrow_ipc_stream_file}`); DLPack export for zero-copy GPU handoff (`CudaKernelProvider::to_dlpack_table`); DLPack import for zero-copy GPU ingestion (`CudaKernelProvider::{from_dlpack_tensors, from_dlpack_tensors_with_schema}`); interop notes + Python DLPack examples in `docs/architecture/cudf-interop.md` and `examples/python/`. Effort: 2-3 weeks. [v0.2.0]

### Planned

## Probabilistic Reasoning (xlog-prob)

### Implemented
- Phase 4 deliverables: PIR (Provenance IR) implementation; XGCF (GPU Circuit Format) for WMC; D4 backend integration for knowledge compilation (vendored; built in-repo); forward/backward circuit evaluation; neural predicate support (DLPack-first; torch optional); explicit P3 gating for non-monotone recursion unless `prob_engine=mc` is requested. [v0.2.0]
- Phase 4 prerequisites: CUDA correctness suite passing (P1.1-P1.4) and P2.3 LogSumExp implemented. [v0.2.0]
- Phase 4 plans: design `docs/plans/2026-01-13-phase4-integrated-design.md`, implementation `docs/plans/2026-01-13-phase4-integrated-implementation-plan.md`, architecture `docs/architecture/xlog-prob.md`. [v0.2.0]
- Phase 4 effort estimate: 2-3 months. [v0.2.0]

### Planned

## Python Interop (xlog-gpu-py)

### Implemented
- User-visible Python package `xlog-gpu` (PyO3 + maturin; returns DLPack capsules). [v0.2.0]

### Planned

## CUDA Certification & Validation

### Implemented
- CUDA certification suite + PTX validation. [v0.2.0]
- CUDA certification results: 140/140 (100%) in `docs/plans/2026-01-14-cuda-certification-results.md`. [v0.2.0]

### Planned

## Epistemic Logic (xlog-elp)

### Implemented

### Planned
- Phase 5 deliverables: EIR (Epistemic IR) implementation; G91 semantics (compatibility mode); FAEEL semantics (default); Generate-Propagate-Test algorithm; Epistemic splitting. [v0.4-0.5]
- Phase 5 prerequisites: Phase 4 complete; solver integration. [v0.4-0.5]
- Phase 5 effort estimate: 3-4 months. [v0.4-0.5]

## Scaling & Distributed

### Implemented

### Planned
- Phase 6 deliverables: Multi-GPU support (single node); Distributed execution; Production CLI/REPL; Comprehensive benchmarks; Documentation and tutorials. [v0.6+]
- Phase 6 prerequisites: Phases 4-5 complete; P3-P4 optimizations. [v0.6+]
- Phase 6 effort estimate: 4-6 months. [v0.6+]

## Quality & Readiness

### Implemented
- Current status: Phase 4 Complete (integrated `xlog-prob` + Python `xlog-gpu`; CUDA certification passing). [v0.2.0]
- Workspace tests passing (`cargo test --workspace --all-targets --release`). [v0.2.0]
- CUDA certification: 140/140 (100%) — see `docs/plans/2026-01-14-cuda-certification-results.md`. [v0.2.0]
- Crates: 11 (`xlog-core`, `xlog-ir`, `xlog-logic`, `xlog-runtime`, `xlog-cuda`, `xlog-solve`, `xlog-stats`, `xlog-cuda-tests`, `xlog-prob`, `xlog-gpu`, `xlog-gpu-py`). [v0.2.0]
- CUDA Kernels: 10 (join, dedup, groupby, scan, filter, pack, sort, set_ops, circuit, mc_sample). [v0.2.0]
- Phase: 4 of 6 complete. [v0.2.0]
- No remaining P1-P3 roadmap blockers (performance tuning continues under Phase 6 benchmarks). [v0.2.0]
- Phase 4 Complete metrics: workspace tests passing; E2E Datalog queries work; CUDA certification suite passes (140/140); no known critical correctness bugs in CUDA provider; xlog-prob implemented (exact `exact_ddnnf` + approximate `mc`); Python `xlog-gpu` implemented (PyO3 + DLPack). [v0.2.0]
- Production Ready (After P1+P2): all critical CUDA correctness fixes applied; memory budget enforced; no 256-row filter/compact limit. [v0.2.0]
- Feature Complete (After P4): CuDF integration; Query optimizer; Incremental maintenance. [v0.2.0]
- Full Vision (After Phase 6): xlog-prob working. [v0.2.0]
- P1 remaining effort: 0 (P1 complete). [v0.2.0]
- Total P2 Effort: ~2 weeks. [v0.2.0]
- Total P3 Effort: ~4 weeks. [v0.2.0]
- Total P4 Effort: ~3 months. [v0.2.0]

### Planned
- GPU-resident execution fully on GPU (currently partial; core joins/sort/dedup/set-ops stay on-GPU; remaining CPU paths mostly in arithmetic/utility helpers). [v0.3.x]
- Float predicate support complete (runtime + CUDA). [v0.3.x]
- Benchmarks documented. [v0.3.x]
- CLI/REPL interface. [v0.3.x]
- xlog-elp working. [v0.4-0.5]
- Multi-GPU support. [v0.6+]
- Production documentation. [v0.6+]
- Recommended Execution Order: Now: P4.1-P4.4 (Interop/optimizer/incremental/indexing); Month 1-3: Phase 4 (xlog-prob exact path + `xlog-gpu` MVP) in parallel with remaining P4 work; Month 4-6: Phase 5 (xlog-elp); Month 7+: Phase 6 (Scaling). [v0.2.0]

## Reliability & Risk

### Implemented
- Risk Assessment: GPU-native dedup/unique complexity (Likelihood: Medium, Impact: High, Mitigation: Start with single-key, then multi-key; keep CPU fallback). [v0.2.0]
- Risk Assessment: Float predicate semantics (NaN/total order) (Likelihood: Medium, Impact: Medium, Mitigation: Define semantics; mirror CUDA `total_cmp` where needed). [v0.2.0]
- Risk Assessment: D4 integration challenges (Likelihood: High, Impact: Medium, Mitigation: Plan fallback KC backends). [v0.2.0]
- Risk Assessment: ELP complexity explosion (Likelihood: High, Impact: High, Mitigation: Strict tier bounds). [v0.2.0]
- Risk Assessment: Multi-GPU synchronization (Likelihood: Medium, Impact: High, Mitigation: Start single-node first). [v0.2.0]

### Planned

## Documentation & References

### Implemented
- Resources (Documentation): `docs/spec.md`, `docs/spec-v1.1.md`, `docs/ARCHITECTURE.md`, `examples/README.md`, `docs/VALIDATION_REPORT.md`, `docs/plans/2026-01-14-cuda-certification-results.md`. [v0.2.0]
- Runnable `.xlog` runner example (`crates/xlog-logic/examples/xlog_run.rs`) + example programs (`examples/xlog/`). [v0.2.0]
- Resources (Key Papers): GPUlog (HISA indexes, parallel fixpoint); VFLog (columnar GPU Datalog); ProbLog (knowledge compilation); FAEEL (epistemic semantics). [v0.2.0]
- Resources (External Dependencies): cudarc 0.12 (CUDA bindings); D4 (knowledge compiler; vendored, built in-repo) - Phase 4; PyTorch (optional; via DLPack interop) - Phase 4. [v0.2.0]

### Planned
