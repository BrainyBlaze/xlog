# XLOG Development Roadmap

**Last Updated:** January 12, 2026
**Current Status:** Phase 3 Complete (CUDA certification passed); Phase 4 in progress

---

## Current State

| Metric | Value |
|--------|-------|
| Workspace Tests Passing | 717 (`cargo test --workspace --all-targets --release`) |
| CUDA Certification | **133/133 (100%)** - see `docs/plans/2026-01-12-cuda-certification-results.md` |
| Crates | 8 (core, ir, logic, runtime, cuda, solve, stats, cuda-tests) |
| CUDA Kernels | 8 (join, dedup, groupby, scan, filter, pack, sort, set_ops) |
| Phase | 3 of 6 complete |

### What Works
- ✅ Datalog parsing and compilation
- ✅ Stratified negation analysis
- ✅ Semi-naive fixpoint iteration
- ✅ GPU hash joins (inner/semi/anti/left-outer; on-device materialization; key verification; byte-hashed scalar keys)
- ✅ Sort correctness for all scalar key types (stable GPU permutation + GPU reorder)
- ✅ GPU set operations (union, diff)
- ✅ GPU aggregations (count, sum, min, max)
- ✅ LogSumExp aggregation (F64)
- ✅ E2E transitive closure queries
- ✅ CUDA certification suite + PTX validation

### What's Broken/Limited
- ✅ No remaining P1-P3 roadmap blockers (performance tuning continues under Phase 6 benchmarks)

---

## Priority 1: Production Blockers

### P1.1 Join Correctness (Hash Collision Safety) - DONE
**Solution:** Join v2 verifies key bytes after hash match (no false positives).
**Files:** `kernels/join.cu`, `crates/xlog-cuda/src/provider.rs`

### P1.2 Aggregation Sum Overflow/Truncation - DONE
**Solution:** Sum aggregates return `U64` and the schema reflects it.
**Files:** `crates/xlog-cuda/src/provider.rs`

### P1.3 Large-Input Filter/Compaction - DONE
**Solution:** Multi-block scan works beyond 256 elements fully on GPU (recursive scan of block sums).
**Files:** `kernels/scan.cu`, `crates/xlog-cuda/src/provider.rs`

### P1.4 GPU Memory Budget Enforcement - DONE
**Solution:** Atomic budget reservation + RAII accounting for frees.
**Files:** `crates/xlog-cuda/src/memory.rs`, `crates/xlog-cuda/src/multi_gpu_memory.rs`

### P1.5 Remove CPU Roundtrips in Dedup - DONE
**Issue:** Dedup previously downloaded keys to host for boundary detection.
**Solution:** GPU columnar boundary detection + GPU scan/compact (mask/prefix/compact stays on-device).
**Remaining:** None (P3.3 removed the CPU sort permutation).
**Files:** `crates/xlog-cuda/src/provider.rs`, `kernels/dedup.cu`

**Remaining P1 Effort:** 0 (P1 complete)

---

## Priority 2: Important Improvements

### P2.1 Extend Type Support
**Status:** DONE
**Change:** `union_gpu`/`diff_gpu` support all scalar types (and multi-column schemas) without CPU row-materialization; GPU sort permutation is now fully on-device (P3.3).
**Goal:** Uniform scalar support across join/sort/filter/dedup/set-ops (including I32/I64/U64/F32/F64/Bool/Symbol).
**Files:** `crates/xlog-runtime/src/executor.rs`, `crates/xlog-cuda/src/provider.rs`, `kernels/*.cu`
**Effort:** 1-2 weeks

### P2.2 Float Comparisons in Filters
**Status:** DONE
**Change:** Runtime filter predicate evaluation supports F32/F64 comparisons.
**Files:** `crates/xlog-runtime/src/executor.rs`

### P2.3 Implement LogSumExp - DONE
**Files:** `kernels/groupby.cu`, `crates/xlog-cuda/src/provider.rs`

### P2.4 Increase Join Output Limit
**Status:** DONE (v2 join)
**Change:** Join output buffers are sized from a count pass (no silent truncation); optional cap still supported.
**Files:** `crates/xlog-cuda/src/provider.rs`

### P2.5 Key Verification Mode - DONE (Default)
**Current:** Join v2 performs key verification by default after hash match.
**Goal:** Optional unsafe "hash-only" fast path for performance experiments (explicitly opt-in).
**Files:** `kernels/join.cu`, `crates/xlog-cuda/src/provider.rs`

**Total P2 Effort:** ~2 weeks

---

## Priority 3: Performance Optimizations

### P3.1 Optimize Radix Scatter
**Status:** DONE
**Issue:** O(grid_size) loop in scatter phase
**Solution:** Precompute per-digit per-block offsets via GPU prefix sums (no per-element block loop)

### P3.2 Coalesced Memory Access
**Issue:** Hash table probe causes cache misses
**Status:** DONE
**Solution:** Cache-friendly bucket layout (CSR buckets: counts + offsets + contiguous entries + hashes)
**Effort:** 1 week

### P3.3 Multi-Column GPU Sort
**Status:** DONE
**Change:** Stable GPU radix sort generates the permutation on-device for all scalar key types and multi-column lexicographic keys (no host roundtrip).

### P3.4 Kernel Fusion
**Issue:** Multiple kernel launches for compound operations
**Status:** DONE
**Solution:** Fuse filter compare+scan and dedup unique+scan; remove count-kernel pass by computing output counts from final prefix element
**Effort:** 1-2 weeks

**Total P3 Effort:** ~4 weeks

---

## Priority 4: Feature Completeness

### P4.1 CuDF Integration
**Goal:** Interoperability with RAPIDS ecosystem
**Status:** IN PROGRESS (Arrow interop + IPC stream)
**Implemented:**
- Arrow RecordBatch export/import (`CudaKernelProvider::{to_arrow_record_batch, from_arrow_record_batch}`)
- Arrow IPC stream helpers (`CudaKernelProvider::{to_arrow_ipc_stream, from_arrow_ipc_stream, write_arrow_ipc_stream_file, read_arrow_ipc_stream_file}`)
- Notes + Python cuDF example: `docs/architecture/cudf-interop.md`
**Next:**
- Zero-copy interchange (likely DLPack or CUDA-aware Arrow memory)
**Effort:** 2-3 weeks

### P4.2 Query Optimizer
**Goal:** Cost-based join ordering, predicate pushdown
**Status:** IN PROGRESS (predicate pushdown enabled; join ordering heuristic)
**Implemented:**
- Compiler runs optimizer pass (predicate pushdown) on all compiled rule bodies
- Lowering-time greedy join atom ordering using estimated cardinalities + bound-variable preference
**Next:**
- Cost-based join ordering across full join tree shapes (beyond left-deep greedy)
- Broader stats feedback loop (runtime → compiler)
**Effort:** 3-4 weeks

### P4.3 Incremental Maintenance
**Goal:** Delta updates without full recomputation
**Status:** IN PROGRESS (semi-naive SCC evaluation)
**Implemented:**
- Runtime recursive SCC evaluation uses semi-naive deltas with per-scan occurrence rewriting (supports mutual recursion + self-joins)
**Next:**
- Incremental maintenance across EDB updates (insert/delete) without recompilation
**Effort:** 2-3 weeks

### P4.4 Adaptive Indexing (HISA)
**Goal:** Build indexes dynamically for hot relations
**Status:** IN PROGRESS (runtime stats wiring)
**Implemented:**
- Runtime `Executor` maintains `xlog_stats::StatsManager` and records:
  - Scan heat + cardinality/bytes
  - Join selectivities (when both sides are base relations)
**Next:**
- Index manager + memory-budget-aware caching/eviction (hash build-side reuse keyed by observed join keys)
**Effort:** 4-6 weeks

**Total P4 Effort:** ~3 months

---

## Phase 4: xlog-prob (Probabilistic Reasoning)

### Deliverables
- PIR (Provenance IR) implementation
- XGCF (GPU Circuit Format) for WMC
- D4 backend integration for knowledge compilation
- Forward/backward circuit evaluation
- Neural predicate support (PyTorch integration)

### Prerequisites
- CUDA correctness suite passing (P1.1-P1.4)
- P2.3 LogSumExp implemented

### Effort Estimate: 2-3 months

---

## Phase 5: xlog-elp (Epistemic Logic)

### Deliverables
- EIR (Epistemic IR) implementation
- G91 semantics (compatibility mode)
- FAEEL semantics (default)
- Generate-Propagate-Test algorithm
- Epistemic splitting

### Prerequisites
- Phase 4 complete
- Solver integration

### Effort Estimate: 3-4 months

---

## Phase 6: Scaling & Production

### Deliverables
- Multi-GPU support (single node)
- Distributed execution
- Production CLI/REPL
- Comprehensive benchmarks
- Documentation and tutorials

### Prerequisites
- Phases 4-5 complete
- P3-P4 optimizations

### Effort Estimate: 4-6 months

---

## Recommended Execution Order

```
Week 1-4:   P3.1-P3.4 (Performance)
Month 2-3:  P4.1-P4.4 (Features)
Month 5-7:  Phase 4 (xlog-prob)
Month 8-11: Phase 5 (xlog-elp)
Month 12+:  Phase 6 (Scaling)
```

---

## Success Metrics

### Phase 3 Complete (Current)
- [x] 717 workspace tests passing (release)
- [x] E2E Datalog queries work
- [x] CUDA certification suite passes (133/133)
- [x] No known critical correctness bugs in CUDA provider
- [ ] GPU-resident execution ← **Partial** (core joins/sort/dedup/set-ops stay on-GPU; remaining CPU paths mostly in arithmetic/utility helpers)

### Production Ready (After P1+P2)
- [x] All critical CUDA correctness fixes applied
- [ ] Float predicate support complete (runtime + CUDA)
- [x] Memory budget enforced
- [x] No 256-row filter/compact limit
- [ ] Benchmarks documented

### Feature Complete (After P4)
- [ ] CuDF integration
- [ ] Query optimizer
- [ ] Incremental maintenance
- [ ] CLI/REPL interface

### Full Vision (After Phase 6)
- [ ] xlog-prob working
- [ ] xlog-elp working
- [ ] Multi-GPU support
- [ ] Production documentation

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GPU-native dedup/unique complexity | Medium | High | Start with single-key, then multi-key; keep CPU fallback |
| Float predicate semantics (NaN/total order) | Medium | Medium | Define semantics; mirror CUDA `total_cmp` where needed |
| D4 integration challenges | High | Medium | Plan fallback KC backends |
| ELP complexity explosion | High | High | Strict tier bounds |
| Multi-GPU synchronization | Medium | High | Start single-node first |

---

## Resources

### Documentation
- `docs/spec.md` - Full system specification
- `docs/spec-v1.1.md` - Revised design
- `docs/ARCHITECTURE.md` - System architecture
- `docs/VALIDATION_REPORT.md` - Current validation
- `docs/plans/2026-01-12-cuda-certification-results.md` - CUDA certification results

### Key Papers
- GPUlog (HISA indexes, parallel fixpoint)
- VFLog (columnar GPU Datalog)
- ProbLog (knowledge compilation)
- FAEEL (epistemic semantics)

### External Dependencies
- cudarc 0.12 (CUDA bindings)
- D4 (knowledge compiler) - Phase 4
- PyTorch (neural predicates) - Phase 4
