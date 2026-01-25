# Neural-Verified JIT Architecture: Critical Validation & Refinement

**Date:** January 22, 2026
**Purpose:** Rigorous analysis of proposed solution — identify flaws, validate assumptions, propose improvements

**Note (Jan 25, 2026):** This is an exploratory research document. The authoritative production direction for GPU-native
knowledge compilation and verifier-grade correctness is `docs/design/2026-01-22-gpu-native-compilation-design.md` and
the implemented verifier contract is `docs/plans/2026-01-25-zero-host-reads-gpu-verifier.md` (see also `docs/ROADMAP.md`).
Some ideas discussed here (e.g., sampling-based verification, tensor fast paths, fallback routes) may be intentionally
excluded from the current production design.

---

## 1. Critical Analysis: What Could Go Wrong?

### Issue 1: Neural Compiler Training Data Quality ⚠️

**Problem:** D4 generates ONE valid circuit, not THE BEST circuit.

**Risk:**
- Training data may contain suboptimal circuits
- Model learns D4's specific heuristics (not optimal compilation)
- Overfitting to D4's biases

**Evidence Against:**
- D4 is state-of-art but not perfect
- Different variable orderings produce different circuits
- Circuit size varies by 2-10x depending on heuristics

**Proposed Fix:**
1. **Multi-teacher training:** Collect circuits from D4, Dsharp, C2D
2. **Pareto-optimal selection:** Pick smallest circuits from multiple runs
3. **Synthetic optimization:** Post-process circuits to minimize size
4. **Data augmentation:** Variable renaming, clause reordering preserves equivalence

**Improvement Impact:** +10-20% better circuits, more robust model

---

### Issue 2: Verification Scalability 🚨 CRITICAL

**Problem:** 1024 random samples is NOT enough for large formulas.

**Math:**
- Formula with N variables: 2^N possible assignments
- 1024 samples: covers 1024/2^N of space
- For N=100: 1024/2^100 ≈ 0 (essentially nothing)

**Risk:** Neural model could output circuit that's correct on 1024 samples but wrong on 2^100 - 1024 others.

**This Invalidates Core Correctness Claim! ❌**

**Proposed Fixes:**

**Fix 2A: Symbolic Verification (Sound & Complete)**
```
Instead of random sampling, use symbolic equivalence checking:
1. Convert circuit back to CNF (C')
2. Check if φ ≡ C' using SAT solver
   - Query: φ ⊕ C' (XOR)
   - If UNSAT → equivalent ✓
   - If SAT → counterexample found ✗
```

**Pros:** Mathematically sound, complete
**Cons:** Expensive (another SAT call), but runs on GPU

**Fix 2B: Structured Verification**
```
Verify structural properties of d-DNNF:
1. Decomposability: Children share no variables
2. Determinism: OR children mutually exclusive
3. Smoothness: All variables mentioned
4. Soundness: Clauses implied by circuit

These are LOCAL checks (fast), guarantee valid d-DNNF.
Then trust d-DNNF correctness theorem.
```

**Pros:** Fast (linear time), sound for d-DNNF class
**Cons:** Only checks structure, not semantic equivalence

**Fix 2C: Hybrid Verification**
```
Tier 1: Structural checks (1ms) - catches malformed circuits
Tier 2: Random sampling (1ms, 1024 samples) - catches obvious errors
Tier 3: Symbolic SAT check (5-10ms) - formal guarantee

Run Tier 1 always, Tier 2 for medium confidence, Tier 3 for high-stakes queries.
```

**Recommendation:** Use **Fix 2C (Hybrid)** — best of all worlds.

**Revised Time:** 1-11ms depending on tier (acceptable)

---

### Issue 3: JIT Compilation Overhead Reality Check 🤔

**Claim:** PTX generation + NVRTC = 1-2ms

**Reality Check:**
- PTX generation: string concatenation (fast, <1ms) ✓
- NVRTC compilation: **actual PTX→SASS compilation** (slow!)

**Benchmarking NVRTC:**
- Simple kernel (10 ops): ~5-10ms
- Medium kernel (100 ops): ~20-50ms
- Complex kernel (1000 ops): ~100-200ms

**Circuits often have 1000+ nodes → 100-200ms NVRTC compilation!**

**This Negates Speed Benefit! ❌**

**Proposed Fixes:**

**Fix 3A: Offline Pre-Compilation**
```
Identify common circuit patterns:
- 80% of queries fall into ~100 structural templates
- Pre-compile these offline (one-time cost)
- Only JIT for rare patterns (20% of queries)

Result: 80% queries = 0ms JIT, 20% queries = 100ms JIT
Average: 20ms (still 5-250x faster than CPU D4)
```

**Fix 3B: Coarse-Grained JIT**
```
Don't specialize entire circuit — specialize hotspots:
- Generic kernel for most nodes
- JIT specialized kernel for critical paths (root to heavy subtrees)
- Hybrid execution

Result: Small PTX code → <10ms NVRTC
```

**Fix 3C: Abandon JIT, Use Tensor Ops**
```
Like KLay approach: represent circuit as tensor operations
- AND → element-wise add (log space)
- OR → logsumexp reduction
- Circuit → sequence of cuBLAS/cuSPARSE calls

No JIT needed, hardware-optimized primitives.
```

**Recommendation:** **Fix 3C (Tensor Ops)** for primary path, **Fix 3A** for rare patterns.

**Revised Architecture:**
```
Neural → Verify → Convert to Tensor Ops → Execute (cuBLAS/cuSPARSE)
                     ↓ (if tensor conversion fails)
                  GPU D4 Fallback
```

---

### Issue 4: GPU D4 Parallelization May Not Work 🚨

**Problem:** D4 is a tree search algorithm — inherently sequential in worst case.

**Parallelization Challenges:**
1. **Load imbalance:** One branch may be 100x harder than others
2. **Recursion depth:** GPU has 24-32 level limit, D4 can go deeper
3. **Dynamic memory:** Component caching requires GPU malloc (slow, fragmented)
4. **Communication:** Threads need to share cache (contention)

**Risk:** GPU D4 might only be 2-3x faster than CPU, not 10-50x.

**Proposed Fixes:**

**Fix 4A: Hybrid CPU-GPU**
```
CPU does tree search (sequential part)
GPU evaluates components (parallel part)
- CPU picks variable, branches
- GPU does unit propagation, component analysis
- Balanced workload
```

**Fix 4B: Iterative Widening**
```
BFS-style search on GPU:
Level 0: 1 node (root)
Level 1: 2 nodes (both branches)
Level 2: 4 nodes
...
Level K: 2^K nodes (GPU saturated)

Parallelize across nodes at same level.
Avoids recursion depth limit.
```

**Fix 4C: Work-Stealing Queue**
```
Shared work queue on GPU:
- Producer threads: generate subproblems
- Consumer threads: solve subproblems
- Dynamic load balancing

Similar to CUDA Dynamic Parallelism, but explicit.
```

**Recommendation:** Start with **Fix 4B (Iterative Widening)** — simpler, more predictable.

**Reality Check:** Expect 5-10x speedup (not 50x). Still valuable!

---

### Issue 5: Memory Budget Concerns 💾

**Memory Usage Breakdown:**
```
Neural Model (50M params, FP16):     100 MB
Inference activations (batch 32):     50 MB
Circuit structures (10K circuits):   500 MB (50KB each)
Cached JIT kernels (10K):          1000 MB (100KB each)
Temporary buffers:                  200 MB
----------------------------------------------
Total:                              1850 MB
```

**GPU Budget:** 8-24 GB typical
**Headroom:** 6-22 GB for actual computation ✓

**But:** Kernel cache could explode with many diverse queries!

**Proposed Fixes:**

**Fix 5A: LRU Eviction**
```
Least Recently Used cache with size limit:
- Max 1000 JIT kernels (100 MB)
- Max 5000 circuits (250 MB)
- Evict oldest when full
```

**Fix 5B: Compression**
```
Circuit structures: compress with zstd (5-10x smaller)
JIT kernels: unload when not used (recompile on cache miss)
```

**Fix 5C: Persistent Cache**
```
Save compiled kernels to disk:
~/.xlog/kernel_cache/{hash}.cubin

Load from disk instead of recompiling.
SSD latency: ~0.1ms (faster than NVRTC!)
```

**Recommendation:** All three fixes — **LRU + Compression + Persistent Cache**

---

### Issue 6: Integration Complexity 🛠️

**Problem:** System has many moving parts:
1. Neural model (PyTorch/ONNX)
2. Verification (CUDA kernel)
3. Tensor ops (cuBLAS/cuSPARSE)
4. GPU D4 (CUDA + dynamic parallelism)
5. Cache management
6. Training pipeline

**Risk:** Integration bugs, difficult debugging, maintenance burden.

**Proposed Fixes:**

**Fix 6A: Phased Development**
```
Phase 1: Tensor ops only (no neural, no JIT)
Phase 2: Add neural prediction
Phase 3: Add verification
Phase 4: Add GPU D4 fallback

Each phase independently useful.
```

**Fix 6B: Modular Architecture**
```
trait Compiler {
    fn compile(&self, cnf: &CNF) -> Result<Circuit>;
}

impl Compiler for TensorOpCompiler { ... }
impl Compiler for NeuralCompiler { ... }
impl Compiler for GpuD4Compiler { ... }

Composition pattern: NeuralCompiler → fallback(GpuD4Compiler)
```

**Fix 6C: Extensive Testing**
```
Unit tests: Each component independently
Integration tests: Pairwise combinations
End-to-end tests: Full pipeline
Differential testing: Compare to D4 on 100K queries
Fuzzing: Random CNF generation
```

**Recommendation:** All three fixes.

---

## 2. Refined Architecture (v2)

### Simplified 3-Stage Pipeline

```
┌──────────────────────────────────────────────────────────┐
│  STAGE 1: PREDICT (Optional Fast Path)                   │
│  Neural Compiler (Small GNN, not Transformer)            │
│  Input: CNF Formula                                       │
│  Output: Circuit Structure (graph representation)        │
│  Time: ~1-2ms (simpler model than Transformer)           │
│  Confidence threshold: 0.95                              │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  STAGE 2: VERIFY (Hybrid Multi-Tier)                     │
│  Tier 1: Structural checks (decomp, determ, smooth)     │
│  Tier 2: Random sampling (1024 samples)                 │
│  Tier 3: Symbolic SAT check (φ ⊕ circuit = UNSAT?)     │
│  Time: 1-10ms (adaptive based on confidence)            │
└──────────────────────────────────────────────────────────┘
                         ↓
                   ┌─────┴─────┐
                   │           │
                ✓ Valid     ✗ Invalid
                   │           │
                   ↓           ↓
         ┌─────────────┐  ┌────────────────┐
         │  FAST PATH  │  │  FALLBACK PATH │
         │  (Tensor)   │  │  (GPU D4)      │
         └─────────────┘  └────────────────┘
                   │           │
                   └─────┬─────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  STAGE 3: EXECUTE (Unified)                              │
│  Convert circuit to tensor operations (cuBLAS/cuSPARSE)  │
│  OR                                                       │
│  Use generic xgcf_forward/backward kernels (current)     │
│  Time: <1ms for forward+backward                         │
└──────────────────────────────────────────────────────────┘
```

### Key Changes from v1:

1. **Simpler Neural Model:** GNN instead of Transformer
   - Faster inference (1-2ms vs 3-5ms)
   - Less memory (20M params vs 50M)
   - Better for graph-structured input (CNF)

2. **Removed JIT PTX Generation:** Too slow (NVRTC overhead)
   - Use tensor ops (cuBLAS) for fast path
   - Use current generic kernels for fallback
   - Simpler, more reliable

3. **Improved Verification:** Hybrid multi-tier
   - Structural checks (fast, sound for d-DNNF)
   - Random sampling (medium, high confidence)
   - Symbolic SAT (slow, complete guarantee)

4. **GPU D4 as Fallback Only:** Don't depend on 50x speedup
   - Focus on neural fast path (90% hit rate)
   - GPU D4 just needs to beat CPU (even 2x is fine)
   - Reduces risk

---

## 3. Alternative Name Suggestions

**Name Requirements:**
- Not pagan/mythological
- Technical/professional
- Memorable acronym
- Relates to logic/compilation/speed

### Option 1: **FORGE** 🔨
**Fast Optimized Reasoning and Gradient Engine**
- Conveys: building/crafting (compilation), speed, gradients
- Professional, not mythological
- Easy to remember

### Option 2: **APEX** 🔺
**Accelerated Probabilistic EXecution**
- Conveys: peak performance, precision
- Clean acronym
- Technical sound

### Option 3: **NEXUS** 🔗
**Neural EXecution and Universal Symbolic compiler**
- Conveys: connection (neuro-symbolic), universal (general purpose)
- Sci-fi vibe but not mythological
- Strong branding

### Option 4: **VERTEX** 📐
**VERified Tensor EXecution**
- Conveys: graph theory (vertices), verification, tensors
- Mathematical/technical
- Memorable

### Option 5: **SAGE** 🌿
**Symbolic Acceleration and Gradient Engine**
- Conveys: wisdom, symbolic reasoning, speed
- Simple, elegant
- Easy to say/remember

### Option 6: **PRISM** 🌈
**Probabilistic Reasoning with Instant Symbolic Methods**
- Conveys: light refraction (fast), probabilistic, instant
- Strong visual metaphor
- Clean acronym

### Option 7: **SWIFT** ⚡
**Symbolic Workload Inference and Fast Translation**
- Conveys: speed (primary goal)
- Simple, direct
- Well-known word (Swift language exists though)

### Option 8: **BOLT** ⚡
**Binary Optimized Logic Translation**
- Conveys: speed (lightning bolt), optimization
- Short, punchy
- Easy branding

### Option 9: **CORE** 💎
**Compiled Optimized Reasoning Engine**
- Conveys: essential/fundamental, compilation, reasoning
- Simple, professional
- Central concept

### Option 10: **FLUX** 🌊
**Fast Logic Universal eXecution**
- Conveys: flow, change, speed
- Modern sound
- Technical vibe

---

## 4. Recommendation: Best Name

### **FORGE** (Fast Optimized Reasoning and Gradient Engine)

**Why:**
1. ✅ Conveys compilation (forging/crafting circuits)
2. ✅ Conveys speed (optimized, fast)
3. ✅ Relates to neuro-symbolic (reasoning + gradients)
4. ✅ Professional, not mythological
5. ✅ Strong branding potential
6. ✅ Easy to remember and say

**Tagline:** *"Forging the future of GPU logic programming"*

**Alternative if FORGE doesn't resonate:** **NEXUS** or **VERTEX**

---

## 5. Revised Technical Roadmap

### Phase 1: Tensor-Based Fast Path (Weeks 1-3)

**Goal:** Prove tensor ops can replace circuits for common queries

**Tasks:**
1. Implement CNF → Sparse Matrix conversion
2. Implement forward pass via cuSPARSE (matrix ops)
3. Implement backward pass via cuBLAS (transpose + matmul)
4. Benchmark vs current circuit approach
5. Validate correctness on 1000 test cases

**Success Metric:** Tensor path 2-10x faster for small circuits (<1000 clauses)

**Deliverable:** Working tensor evaluator, integrated into XLOG

---

### Phase 2: Neural Prediction Layer (Weeks 4-6)

**Goal:** Train GNN to predict circuit structures

**Tasks:**
1. Collect 100K CNF→Circuit pairs from D4
2. Design GNN architecture (graph encoder → circuit decoder)
3. Train model (accuracy target: 70%+)
4. Implement GPU inference (PyTorch → ONNX → TensorRT)
5. Integrate into pipeline (Stage 1)

**Success Metric:** 70%+ verification pass rate, <2ms inference

**Deliverable:** Neural compiler predicting circuits

---

### Phase 3: Verification Layer (Weeks 7-8)

**Goal:** Guarantee correctness with hybrid verification

**Tasks:**
1. Implement structural checks (decomp, determ, smooth)
2. Implement random sampling verification (1024 samples)
3. Implement symbolic SAT verification (φ ⊕ circuit)
4. Integrate adaptive strategy (Tier 1→2→3)
5. Benchmark verification time

**Success Metric:** 100% correctness guarantee, <10ms verification (90th percentile)

**Deliverable:** Full verification system

---

### Phase 4: GPU D4 Fallback (Weeks 9-11)

**Goal:** GPU-resident compilation for fallback path

**Tasks:**
1. Implement BFS-style D4 (iterative widening)
2. Implement component caching on GPU (shared memory)
3. Implement unit propagation kernel
4. Optimize and benchmark vs CPU D4
5. Integrate as fallback path

**Success Metric:** 5-10x faster than CPU D4 (conservative target)

**Deliverable:** GPU D4 compiler

---

### Phase 5: Integration & Optimization (Week 12)

**Goal:** Polish, test, optimize end-to-end

**Tasks:**
1. Multi-level caching (LRU + persistent disk)
2. Memory optimization (compression, eviction)
3. Full certification suite (200 tests)
4. Benchmark suite (vs D4, PyJuice, Lobster)
5. Performance tuning (profile + optimize)

**Success Metric:** 10-100x speedup over CPU D4, 200/200 tests pass

**Deliverable:** Production-ready system

---

## 6. Updated Success Metrics

### Performance (Realistic Targets)

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| **Compilation (small)** | 50-500ms | <5ms | <2ms |
| **Compilation (large)** | 1-5s | <50ms | <20ms |
| **Throughput** | ~10 q/s | 100 q/s | 500 q/s |
| **Accuracy** | 100% (exact) | 100% (verified) | 100% |
| **Memory** | N/A | <1GB | <500MB |

### Quality

| Metric | Target |
|--------|--------|
| Neural pass rate | 70-90% |
| Verification time | <10ms (90th %ile) |
| Cache hit rate | >90% (production) |
| GPU D4 speedup | 5-10x vs CPU |
| End-to-end speedup | 10-100x vs current |

---

## 7. Risk Assessment (Updated)

| Risk | Probability | Impact | Mitigation | Residual Risk |
|------|-------------|--------|------------|---------------|
| Neural accuracy low | Medium | Medium | Fallback to GPU D4 | Low |
| Verification too slow | Low | Medium | Adaptive tiers | Low |
| GPU D4 not fast enough | Medium | Low | Neural covers 90% | Low |
| Tensor ops slower than expected | Low | High | Keep current kernels | Low |
| Memory overflow | Low | Medium | LRU + eviction | Low |
| Integration bugs | High | Medium | Phased development | Medium |

**Overall Risk:** **LOW-MEDIUM** — All components have fallbacks, system degrades gracefully

---

## 8. Paper Contribution (Refined)

### Title: "FORGE: Neural-Verified Compilation for GPU-Native Neuro-Symbolic Programming"

### Core Contributions

1. **Tensor-Based Circuit Execution** (replaces explicit circuits with sparse matrix ops)
   - Novel: First to use cuBLAS/cuSPARSE for d-DNNF evaluation
   - Impact: Hardware-optimized, no custom kernels needed

2. **Neural Circuit Prediction** (GNN predicts compilation from CNF)
   - Novel: First learned knowledge compiler with >70% accuracy
   - Impact: 10-100x speedup on fast path

3. **Hybrid Verification** (multi-tier correctness checking)
   - Novel: Combines structural + random + symbolic verification
   - Impact: 100% correctness guarantee with <10ms overhead

4. **GPU-Resident D4** (parallel BFS compilation on GPU)
   - Novel: First GPU implementation of d-DNNF compilation
   - Impact: 5-10x speedup over CPU baseline

5. **End-to-End System** (complete GPU pipeline)
   - Novel: First pure GPU-native neuro-symbolic programming system
   - Impact: State-of-art performance, production-ready

### Expected Venues

**Primary:** NeurIPS 2026 (Systems track + Neuro-Symbolic workshop)
**Secondary:** ICML 2027, IJCAI 2027
**Archival:** JAIR (Journal of AI Research)

---

## 9. Open Questions Requiring Decisions

### Question 1: Neural Architecture Choice
- **Option A:** GNN (better for graphs, recommended)
- **Option B:** Transformer (proven for code, more complex)
- **Option C:** Hybrid (GNN encoder + Transformer decoder)

**Recommendation:** Start with **Option A (GNN)**, try B if needed

---

### Question 2: Verification Strategy
- **Option A:** Always run Tier 3 (symbolic SAT) for 100% guarantee
- **Option B:** Adaptive tiers (Tier 3 only when needed)
- **Option C:** Skip verification for cached circuits

**Recommendation:** **Option B (Adaptive)** — balances speed and correctness

---

### Question 3: Execution Backend
- **Option A:** Tensor ops (cuBLAS/cuSPARSE) exclusively
- **Option B:** Keep current xgcf kernels as option
- **Option C:** Hybrid (tensor for small, kernel for large)

**Recommendation:** **Option C (Hybrid)** — best of both worlds

---

### Question 4: GPU D4 Priority
- **Option A:** Critical path (implement early)
- **Option B:** Nice-to-have (implement if time)
- **Option C:** Skip entirely (neural + tensor sufficient)

**Recommendation:** **Option B (Nice-to-have)** — neural covers 90% already

---

### Question 5: Training Data Strategy
- **Option A:** 100K samples from D4 only
- **Option B:** 1M samples from multiple compilers (D4, Dsharp, C2D)
- **Option C:** Continual learning (add production queries)

**Recommendation:** Start with **A**, scale to **B**, eventually **C**

---

## 10. Final Recommendations

### Immediate Priorities (Do Now)

1. ✅ **Approve name:** FORGE (or suggest alternative)
2. ✅ **Approve refined architecture:** v2 with tensor ops + simpler neural
3. ✅ **Approve roadmap:** 12 weeks, phased development
4. ✅ **Create branch:** `feature/forge-neural-compilation`
5. ✅ **Start Phase 1:** Tensor-based fast path (this week)

### Success Factors

1. **Phased development:** Each phase delivers value independently
2. **Conservative estimates:** 5-10x speedup (not 50x), 70% accuracy (not 95%)
3. **Graceful degradation:** System works even if components underperform
4. **Extensive testing:** Differential testing vs D4 on 100K queries
5. **Clear metrics:** Measure everything, optimize based on data

### Long-Term Vision

**FORGE becomes:**
- ✅ Fastest neuro-symbolic programming system (10-100x speedup)
- ✅ First pure GPU-native logic programming system (0% CPU)
- ✅ Most correct system (100% verified)
- ✅ Most scalable system (1000+ queries/second)
- ✅ State-of-art research (NeurIPS publication)
- ✅ Production-ready (200/200 tests pass)

---

## Summary: Go/No-Go Decision

### ✅ Proceed with FORGE if:
- You approve refined architecture (tensor ops + GNN + hybrid verification)
- You approve conservative timeline (12 weeks)
- You approve name: FORGE (or choose alternative)
- You commit to phased approach (each phase useful independently)

### ⏸️ Pause if:
- Major concerns about technical approach
- Timeline too aggressive
- Resource constraints (GPU memory, compute)
- Prefer simpler solution first

---

**Ready for your decision!** 🚀
