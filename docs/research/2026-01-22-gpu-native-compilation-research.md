# GPU-Native Knowledge Compilation Research
**Date:** January 22, 2026
**Focus:** Eliminating CPU bottlenecks in neuro-symbolic logic programming

**Note (Jan 25, 2026):** This is an exploratory research document. The authoritative production design for GPU-native
knowledge compilation is `docs/design/2026-01-22-gpu-native-compilation-design.md`, and the verifier implementation/contract
is `docs/plans/2026-01-25-zero-host-reads-gpu-verifier.md` (see also `docs/ROADMAP.md`). Not all surveyed approaches or
system proposals here are part of the current production roadmap.

---

## Executive Summary

Current XLOG architecture uses **CPU-based D4 compilation** to convert CNF formulas into GPU-executable circuits. This creates a fundamental bottleneck: every new query requires CPU roundtrip for compilation. Research reveals **4 revolutionary directions** that could eliminate this bottleneck entirely and make XLOG the world's first **pure GPU-native logic programming system**.

**Key Finding:** State-of-the-art systems (2025-2026) achieve **1-4 orders of magnitude speedups** by moving computation to GPU. XLOG can do better by moving *compilation itself* to GPU.

---

## 1. Current State-of-the-Art (2025-2026)

### 1.1 PyJuice: GPU Probabilistic Circuits (2025)

**Source:** [Scaling Tractable Probabilistic Circuits: A Systems Perspective](https://arxiv.org/abs/2406.00766)

**Key Innovation:** Compilation process converts PC into compact representation for block-based GPU parallelization.

**Performance:**
- **1-2 orders of magnitude faster** than existing PC systems
- **2-5x less GPU memory**
- Leverages Tensor Cores on modern GPUs

**Relevance:** Shows that *how you compile* for GPU matters as much as what you compile.

---

### 1.2 KLay: Sparse Arithmetic Circuits (2025)

**Source:** [KLay: Accelerating Sparse Arithmetic Circuits](https://pedrozudo.github.io/assets/documents/publications/2025/maene2025klaycolorai/maene2025klaycolorai.paper.pdf)

**Key Innovation:** Reduces arithmetic circuit evaluation to **index and scatter operations** — primitives already in tensor libraries.

**Impact:**
- Hardware-agnostic (works on any GPU)
- Leverages compiler stacks of open-source tensor libraries
- Significant performance improvements

**Relevance:** We don't need custom PTX kernels for everything — tensor ops can represent circuits!

---

### 1.3 Boolean Matrix Logic Programming (2024)

**Source:** [Boolean Matrix Logic Programming on the GPU](https://arxiv.org/html/2408.10369)

**Key Innovation:** Represents logic programs as **Boolean matrices** instead of trees/circuits.

**Performance:**
- GPU and CPU implementations outperform state-of-art by **1-4 orders of magnitude**
- Two algorithms for evaluating linear recursive Datalog on GPUs

**Relevance:** Matrix representation may be fundamentally better than circuits for GPU execution.

---

### 1.4 Lobster: GPU Neurosymbolic Framework (2025)

**Source:** [Lobster: GPU-Accelerated Framework for Neurosymbolic Programming](https://arxiv.org/abs/2503.21937)

**Key Innovation:** Unified framework mapping Datalog to GPU, supporting discrete, probabilistic, and **differentiable** reasoning.

**Performance:**
- **3.9x average speedup** over Scallop (state-of-art neurosymbolic)
- End-to-end GPU pipeline

**Relevance:** Shows complete neuro-symbolic systems can run entirely on GPU.

---

### 1.5 GPU-Native Compilation (December 2025)

**Source:** [Theoretical Foundations of GPU-Native Compilation](https://arxiv.org/html/2512.11200v1)

**Key Problem:** Current systems generate code on GPU, transfer to CPU for compilation, then return to GPU. This is **90-99% of iteration time**.

**Three Approaches:**
1. **Parallel traditional compilation** adapted for GPU execution
2. **Neural compilation** using learned sequence-to-sequence translation
3. **Hybrid architectures** combining both

**Potential Speedup:** **10-100x** for code iteration cycles

**Relevance:** This is EXACTLY our problem — we're doing compilation on CPU!

---

## 2. Knowledge Compilation Background

### 2.1 Current Tools: D4, C2D, miniC2D

**Source:** [An Improved Decision-DNNF Compiler](https://www.ijcai.org/proceedings/2017/0093.pdf)

**Compilation Targets:**
- **d-DNNF** (decomposable negation normal form) — deterministic, structured
- **SDD** (sentential decision diagrams) — generalizes OBDDs
- **d-SDNNFs** — structured d-DNNF

**Current Best:** D4 compiler outperforms C2D and Dsharp with lower compilation times and smaller representations.

**Problem:** All are **CPU-only** tools. No GPU-native knowledge compilation exists.

---

### 2.2 Differentiable Logic

**Sources:**
- [Differentiable Probabilistic Logic Networks](https://arxiv.org/abs/1907.04592)
- [Convolutional Differentiable Logic Gate Networks](https://arxiv.org/html/2411.04732v1)

**Key Insight:** Logic gates (AND, OR, NOT) can be made **differentiable** using:
- Soft logic operators (product t-norm, Łukasiewicz t-norm)
- Gumbel-Softmax for discrete decisions
- Straight-through estimators

**Benefit:** Enables **gradient descent directly on logic** — no need for fixed compilation!

**Relevance:** We could backprop through the compilation process itself.

---

## 3. GPU JIT Compilation Technologies

### 3.1 CUDA Runtime Compilation (NVRTC)

**Sources:**
- [CUDA JIT Compilation](https://medium.com/gpgpu/cuda-jit-compilation-1fb4950c67bb)
- [Jitify: Simplifying NVRTC](https://github.com/NVIDIA/jitify)

**Capability:** Compile CUDA code to PTX **at runtime on device**.

**Use Cases:**
- On-the-fly kernel generation based on runtime configuration
- Optimization for specific GPU architecture
- Dynamic kernel specialization

**Relevance:** We could generate circuit evaluation kernels on-the-fly for each query!

---

### 3.2 PTX JIT Compilation

**Source:** [Utilize PTX Just-In-Time Compilation](https://amirsojoodi.github.io/posts/JIT-PTX/)

**Process:**
1. Generate PTX (assembly-like) code at runtime
2. CUDA driver compiles PTX to machine code
3. Kernel executes immediately

**Benefit:** No CPU roundtrip if PTX generation happens on GPU!

---

## 4. Critical Analysis: Where XLOG Stands

### 4.1 Current Architecture (v0.4.0-alpha)

```
Query → CNF (CPU) → D4 Compilation (CPU) → XGCF Circuit (CPU)
  → Upload to GPU → Execute Forward/Backward (GPU) → Download gradients (CPU)
```

**Bottlenecks:**
1. ❌ D4 compilation on CPU (can take seconds for complex queries)
2. ❌ Circuit structure upload (once per query type, but still transfer)
3. ✅ Weight upload (minimal: 2*num_vars*8 bytes)
4. ✅ Forward/backward execution (pure GPU, fast)
5. ❌ Gradient download (2*num_vars*8 bytes, but CPU roundtrip)

**Cache helps but:** Only works for structurally identical queries. New query patterns require full CPU compilation.

---

### 4.2 What We Do Well

✅ **Forward/Backward Kernels:** Our PTX kernels are production-grade (200/200 tests pass)
✅ **Transfer Efficiency:** Weights/gradients are minimal (proportional to num_vars, not num_nodes)
✅ **Numerical Stability:** Log-space arithmetic, 1e-10 accuracy
✅ **Cache Speedup:** >100x for repeated query patterns

---

### 4.3 What We Must Fix

❌ **CPU-Dependent Compilation:** D4 runs on CPU, blocks GPU pipeline
❌ **Cold Start Problem:** First query of new pattern has large latency
❌ **Scalability Limit:** Can't compile 1000s of diverse queries/second
❌ **Training Bottleneck:** Online learning with dynamic queries stalls on compilation

---

## 5. Revolutionary Directions

### Direction 1: GPU-Resident Knowledge Compilation

**Vision:** Implement CNF→d-DNNF compilation **entirely on GPU**.

**Approach:**
- Port D4's top-down tree search to CUDA kernels
- Use GPU-parallel DPLL/CDCL solving
- Generate circuit structure in GPU memory (no transfer)

**Challenges:**
- D4 uses recursive tree search (hard to parallelize)
- Memory allocation during compilation (need dynamic GPU malloc)
- Determinism (search heuristics must be reproducible)

**Potential:** **10-100x compilation speedup** + zero transfer cost

**Precedent:** GPU-native compilation paper shows 10-100x speedup for general compilation

---

### Direction 2: Differentiable Circuit Compilation

**Vision:** Make compilation **differentiable** — backprop through compiler.

**Approach:**
- Soft logic during compilation (fuzzy CNF satisfaction)
- Gumbel-Softmax for circuit structure decisions
- End-to-end gradient from loss → circuit structure → CNF

**Benefits:**
- **Learned compilation:** Optimize compilation strategy for common queries
- **Adaptive circuits:** Circuit structure adapts to training data
- **Zero CPU:** Compilation becomes part of forward pass

**Challenges:**
- How to make tree search differentiable?
- Stability during training
- Biasing towards suboptimal circuit structures

**Precedent:** Differentiable Logic Gate Networks, Neural Architecture Search

---

### Direction 3: Boolean Matrix Representation

**Vision:** Replace tree circuits with **Boolean matrix operations**.

**Approach:**
- Represent CNF as sparse Boolean matrices
- Evaluation = matrix multiplication (cuBLAS/cuSPARSE)
- Gradients = transpose operations (BLAS primitives)

**Benefits:**
- **No compilation:** CNF directly becomes matrices
- **Hardware optimized:** Tensor cores accelerate matrix ops
- **Simple kernels:** Use existing CUDA libraries

**Challenges:**
- Matrix size scales with clause count (memory)
- May be less efficient for small circuits
- Integration with existing codebase

**Precedent:** Boolean Matrix Logic Programming (1-4 orders of magnitude speedup)

---

### Direction 4: JIT Circuit Kernel Generation

**Vision:** Generate **specialized PTX kernels** for each query pattern at runtime.

**Approach:**
- CNF → Circuit structure (CPU or GPU)
- Circuit structure → Custom PTX code (template-based)
- NVRTC compiles PTX → machine code on GPU
- Execute immediately, cache for reuse

**Benefits:**
- **Optimal code:** Each circuit gets hand-tuned kernel
- **Zero transfer:** PTX generation on GPU
- **Cache friendly:** Generated code reusable like current cache

**Challenges:**
- PTX generation complexity
- Compilation time (NVRTC still has overhead)
- Code size explosion for many diverse queries

**Precedent:** CUDA JIT used in cuTENSOR, Numba, custom kernels

---

### Direction 5: Hybrid Neural-Symbolic Compiler

**Vision:** Use **learned neural compiler** to predict optimal circuit structure.

**Approach:**
- Train transformer model: CNF → Circuit structure
- Model learns compilation strategies from D4 traces
- Inference on GPU (pure tensor ops), deterministic
- Fallback to D4 for unknown patterns

**Benefits:**
- **GPU inference:** Neural model runs on GPU (fast)
- **Learned optimization:** Better than hand-crafted heuristics
- **Smooth degradation:** Falls back when uncertain

**Challenges:**
- Training data (need CNF→Circuit pairs from D4)
- Correctness guarantees (neural outputs may be invalid)
- Model size and inference latency

**Precedent:** Neural compilation in GPU-native compilation paper, CUDA-LLM generating kernels

---

### Direction 6: Incremental Circuit Updates

**Vision:** Don't recompile from scratch — **update existing circuits** when query changes.

**Approach:**
- Detect structural similarity between queries
- Apply delta updates to GPU circuit (add/remove nodes)
- Reuse most of existing structure

**Benefits:**
- **Near-zero compilation:** Only compile differences
- **Online learning friendly:** Adapt to query distribution
- **Cache augmentation:** Extends current cache approach

**Challenges:**
- Detecting similarity (hashing, structural matching)
- Efficient GPU memory reallocation
- Correctness of incremental updates

**Precedent:** Incremental SAT solving, dynamic programming on GPU

---

## 6. Comparative Analysis

| Approach | Compilation Time | GPU Native | Correctness | Complexity | Novelty |
|----------|-----------------|------------|-------------|------------|---------|
| **Status Quo (D4 + Cache)** | Seconds (CPU) | ❌ | ✅ Verified | Low | ❌ Standard |
| **1. GPU-Resident D4** | 10-100x faster | ✅ | ✅ Same algorithm | High | ⭐⭐ Novel |
| **2. Differentiable Compilation** | Forward pass | ✅ | ⚠️ Approximate | Very High | ⭐⭐⭐ Groundbreaking |
| **3. Boolean Matrices** | Zero (direct) | ✅ | ✅ Verified | Medium | ⭐⭐ Novel |
| **4. JIT Kernel Generation** | ms (on GPU) | ✅ | ✅ Verified | High | ⭐⭐⭐ Groundbreaking |
| **5. Neural Compiler** | Forward pass | ✅ | ⚠️ Learned | Very High | ⭐⭐⭐ Groundbreaking |
| **6. Incremental Updates** | Near-zero | ✅ | ✅ Verified | High | ⭐⭐ Novel |

---

## 7. Recommended Hybrid Strategy

**Phase 1: Quick Wins (1-2 weeks)**
- Implement **Direction 3: Boolean Matrix Representation** for small queries (<1000 clauses)
- Keeps D4 for large queries, uses matrices for fast path
- Validates approach with minimal risk

**Phase 2: GPU Compilation (1-2 months)**
- Implement **Direction 1: GPU-Resident D4**
- Pure CUDA port of D4 algorithm
- Benchmark against CPU D4

**Phase 3: JIT Generation (2-3 months)**
- Implement **Direction 4: JIT Circuit Kernel Generation**
- Template-based PTX generation for common patterns
- NVRTC integration for runtime compilation

**Phase 4: Research Innovation (3-6 months)**
- Explore **Direction 2 or 5:** Differentiable or Neural Compilation
- Publish research paper
- Patent novel approach

---

## 8. Key Research Questions

1. **Can we achieve <1ms compilation for typical queries on GPU?**
   - D4 on CPU: seconds
   - Target: sub-millisecond on GPU

2. **What's the crossover point for matrices vs circuits?**
   - When do matrix ops beat specialized kernels?
   - Memory vs compute tradeoff

3. **Can compilation be made differentiable without losing correctness?**
   - Soft logic during search?
   - Probabilistic circuit structure?

4. **How much does JIT compilation overhead cost?**
   - PTX generation: ?ms
   - NVRTC compilation: ?ms
   - Amortized by cache

5. **Can a neural compiler match D4 correctness?**
   - Verification mechanisms?
   - Confidence thresholds?

---

## 9. Implementation Priorities

### Critical Path: Zero CPU Bottleneck

**Milestone 1:** Prove GPU-native compilation is possible
→ Implement Boolean Matrix fast path

**Milestone 2:** Scale to complex queries
→ GPU-Resident D4 or JIT generation

**Milestone 3:** Optimize for training workloads
→ Incremental updates + caching

**Milestone 4:** Push research boundaries
→ Differentiable/Neural compilation

---

## 10. Success Metrics

**Performance:**
- Compilation time: <1ms for 90% of queries (vs seconds on CPU)
- End-to-end latency: <10ms for forward+backward (vs 100ms+ with CPU compilation)
- Throughput: >1000 diverse queries/second (vs <10 with D4)

**Quality:**
- Correctness: 100% match with D4 (for verified approaches)
- Memory: <2GB GPU memory for 10K concurrent circuits
- Scalability: Handle 100K+ clause CNFs

**Innovation:**
- First pure GPU-native logic programming system
- Publishable research (NeurIPS, ICML, IJCAI)
- Patent-worthy novel compilation approach

---

## 11. Literature Sources

### Core Papers
1. [Scaling Tractable Probabilistic Circuits: A Systems Perspective](https://arxiv.org/abs/2406.00766) (PyJuice, 2024)
2. [Boolean Matrix Logic Programming on the GPU](https://arxiv.org/html/2408.10369) (2024)
3. [Lobster: A GPU-Accelerated Framework for Neurosymbolic Programming](https://arxiv.org/abs/2503.21937) (2025)
4. [Theoretical Foundations of GPU-Native Compilation](https://arxiv.org/html/2512.11200v1) (2025)
5. [KLay: Accelerating Sparse Arithmetic Circuits](https://pedrozudo.github.io/assets/documents/publications/2025/maene2025klaycolorai/maene2025klaycolorai.paper.pdf) (2025)

### Knowledge Compilation
6. [An Improved Decision-DNNF Compiler (D4)](https://www.ijcai.org/proceedings/2017/0093.pdf) (2017)
7. [A Top-Down Compiler for Sentential Decision Diagrams](https://dl.acm.org/doi/10.5555/2832581.2832687) (2015)

### Differentiable Logic
8. [Differentiable Probabilistic Logic Networks](https://arxiv.org/abs/1907.04592) (2019)
9. [Convolutional Differentiable Logic Gate Networks](https://arxiv.org/html/2411.04732v1) (2024)

### GPU JIT Compilation
10. [CUDA JIT Compilation](https://medium.com/gpgpu/cuda-jit-compilation-1fb4950c67bb)
11. [Jitify: Simplifying NVRTC](https://github.com/NVIDIA/jitify) (NVIDIA)
12. [Utilize PTX Just-In-Time Compilation](https://amirsojoodi.github.io/posts/JIT-PTX/)

### LLM-Driven Kernel Generation
13. [CUDA-LLM: LLMs Can Write Efficient CUDA Kernels](https://arxiv.org/abs/2506.09092) (2025)

---

## Next Steps

**Immediate (Today):**
1. Brainstorm session on these 6 directions
2. Select primary approach (recommend: Matrix + GPU D4 hybrid)
3. Design prototype architecture

**Week 1:**
1. Implement Boolean matrix fast path
2. Benchmark against D4+cache
3. Validate correctness on test suite

**Month 1:**
1. Begin GPU-Resident D4 implementation
2. Profile and optimize
3. Research paper outline

---

**Document Status:** Research Complete — Ready for Brainstorming
