# XLOG Development Roadmap

**Last Updated:** January 9, 2026
**Current Status:** Phase 3 Complete (with critical issues)

---

## Current State

| Metric | Value |
|--------|-------|
| Tests Passing | 275 |
| Crates | 5 (core, ir, logic, runtime, cuda) |
| CUDA Kernels | 7 (join, dedup, filter, sort, groupby, scan, set_ops) |
| Phase | 3 of 6 complete |

### What Works
- ✅ Datalog parsing and compilation
- ✅ Stratified negation analysis
- ✅ Semi-naive fixpoint iteration
- ✅ GPU hash joins (single/multi-column)
- ✅ GPU radix sort (U32 keys)
- ✅ GPU set operations (union, diff)
- ✅ GPU aggregations (count, sum, min, max)
- ✅ E2E transitive closure queries

### What's Broken/Limited
- ❌ Join uses hash-only comparison (no key verification)
- ❌ Sum aggregation truncates u64 to u32
- ❌ Filter/compact limited to 256 rows
- ❌ Dedup uses CPU sort (not GPU)
- ❌ Memory budget not enforced
- ❌ No float support in predicates
- ❌ LogSumExp not implemented

---

## Priority 1: Critical Fixes (Required for Production)

### P1.1 Fix Join Correctness
**Issue:** Hash-only comparison allows false positives
**Impact:** Incorrect query results
**Solution:** Add key byte comparison in probe phase
**Files:** `kernels/join.cu`, `provider.rs`
**Effort:** 2-3 days

### P1.2 Fix Aggregation Overflow
**Issue:** Sum computed as u64, truncated to u32
**Impact:** Silent data corruption
**Solution:** Return u64 sum, update schema handling
**Files:** `provider.rs:1592`
**Effort:** 1 day

### P1.3 Implement Multi-Block Prefix Sum
**Issue:** Current 256-element limit blocks all large filters
**Impact:** System unusable for real data
**Solution:** Implement hierarchical Blelloch scan
**Files:** `kernels/scan.cu`, `provider.rs`
**Effort:** 3-5 days

### P1.4 Enforce Memory Budget
**Issue:** Allocator tracks but doesn't enforce budget
**Impact:** OOM crashes instead of graceful errors
**Solution:** Add budget check before allocation
**Files:** `memory.rs`
**Effort:** 0.5 days

### P1.5 GPU Sort in Dedup
**Issue:** Dedup uses CPU sort, causing host roundtrip
**Impact:** Performance bottleneck, violates GPU-residency goal
**Solution:** Use existing `sort()` method in `dedup()`
**Files:** `provider.rs:507-591`
**Effort:** 1 day

**Total P1 Effort:** ~2 weeks

---

## Priority 2: Important Improvements

### P2.1 Extend Type Support
**Current:** Joins/set-ops only support U32
**Goal:** Support I32, I64, U64, F32, F64
**Files:** `provider.rs`, `kernels/*.cu`
**Effort:** 1 week

### P2.2 Float Comparisons in Filters
**Current:** Float predicates return error
**Goal:** Support F32/F64 in filter expressions
**Files:** `executor.rs:522-525`, `filter.cu`
**Effort:** 2-3 days

### P2.3 Implement LogSumExp
**Current:** Returns "not implemented"
**Goal:** Numerically stable log-sum-exp for probabilistic tier
**Files:** `groupby.cu`, `provider.rs`
**Effort:** 2-3 days

### P2.4 Increase Join Output Limit
**Current:** Clamped at 1M results
**Goal:** Dynamic allocation or chunked output
**Files:** `provider.rs:2825`
**Effort:** 2 days

### P2.5 Add Key Verification Mode
**Current:** Hash-only comparison everywhere
**Goal:** Optional full key verification for correctness-critical use
**Files:** `provider.rs`, `join.cu`
**Effort:** 2 days

**Total P2 Effort:** ~2 weeks

---

## Priority 3: Performance Optimizations

### P3.1 Optimize Radix Scatter
**Issue:** O(grid_size) loop in scatter phase
**Solution:** Use multi-level prefix sum for offsets
**Effort:** 3-4 days

### P3.2 Coalesced Memory Access
**Issue:** Hash table probe causes cache misses
**Solution:** Cache-friendly bucket layout
**Effort:** 1 week

### P3.3 Multi-Column GPU Sort
**Current:** Only U32 keys
**Goal:** Composite key sort on GPU
**Effort:** 1 week

### P3.4 Kernel Fusion
**Issue:** Multiple kernel launches for compound operations
**Solution:** Fuse filter+compact, sort+dedup
**Effort:** 1-2 weeks

**Total P3 Effort:** ~4 weeks

---

## Priority 4: Feature Completeness

### P4.1 CuDF Integration
**Goal:** Interoperability with RAPIDS ecosystem
**Effort:** 2-3 weeks

### P4.2 Query Optimizer
**Goal:** Cost-based join ordering, predicate pushdown
**Effort:** 3-4 weeks

### P4.3 Incremental Maintenance
**Goal:** Delta updates without full recomputation
**Effort:** 2-3 weeks

### P4.4 Adaptive Indexing (HISA)
**Goal:** Build indexes dynamically for hot relations
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
- P1 fixes complete
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
Week 1-2:   P1.1-P1.5 (Critical fixes)
Week 3-4:   P2.1-P2.5 (Important improvements)
Week 5-8:   P3.1-P3.4 (Performance)
Month 3-4:  P4.1-P4.4 (Features)
Month 5-7:  Phase 4 (xlog-prob)
Month 8-11: Phase 5 (xlog-elp)
Month 12+:  Phase 6 (Scaling)
```

---

## Success Metrics

### Phase 3 Complete (Current)
- [x] 275 tests passing
- [x] E2E Datalog queries work
- [ ] No critical correctness bugs ← **BLOCKED**
- [ ] GPU-resident execution ← **Partial**

### Production Ready (After P1+P2)
- [ ] All critical fixes applied
- [ ] Float support complete
- [ ] Memory budget enforced
- [ ] No 256-row limit
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
| Multi-block scan complexity | Medium | High | Research existing implementations |
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

### Key Papers
- GPUlog (HISA indexes, parallel fixpoint)
- VFLog (columnar GPU Datalog)
- ProbLog (knowledge compilation)
- FAEEL (epistemic semantics)

### External Dependencies
- cudarc 0.12 (CUDA bindings)
- D4 (knowledge compiler) - Phase 4
- PyTorch (neural predicates) - Phase 4
