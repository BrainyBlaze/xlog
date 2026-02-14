# XLoG Full-Train Neural Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run XLoG neural examples with full training on real datasets and full real evaluation, then compare speed/performance against DeepProbLog baseline evidence.

**Architecture:** Align release-mode behavior to true full-data semantics, execute examples sequentially with timed commands, and publish a report that captures metrics, runtime, and baseline deltas for comparable tasks.

**Tech Stack:** Python (pyxlog examples), PyTorch/torchvision, CUDA runtime, markdown reporting.

### Task 1: Enforce Full-Data Release Semantics

**Files:**
- Modify: `examples/neural/03_mnist_multidigit/train.py`
- Modify: `examples/neural/04_hwf/train.py`
- Modify: `examples/neural/05_poker/train.py`
- Modify: `examples/neural/06_clutrr/train.py`

**Steps:**
1. Remove non-CI sample caps in release mode (`2048`/`1024` style limits).
2. Keep CI subset behavior unchanged.
3. Keep dev behavior bounded (existing limits acceptable) but ensure release uses full data.
4. In poker release mode, use full train query set each epoch (no `512` query cap).

### Task 2: Add Real Held-Out Evaluation for 01_minimal

**Files:**
- Modify: `examples/neural/01_minimal/train.py`

**Steps:**
1. Add deterministic held-out evaluation on full MNIST test split.
2. Report both digit held-out accuracy and addition held-out accuracy.
3. Emit a stable metric line (`FINAL_METRIC`) for release comparison and log parsing.
4. Keep existing training path and defaults intact.

### Task 3: Run Sequential Full-Train Benchmarks

**Files:**
- Create logs under: `examples/neural/results/xlog_gpu_fulltrain_sequential/`

**Steps:**
1. Run `01_minimal` with full train and full test held-out eval.
2. Run `02_coins` in release mode.
3. Run `03_mnist_multidigit` in release mode.
4. Run `04_hwf` in release mode.
5. Run `05_poker` in release mode.
6. Run `06_clutrr` in release mode.
7. Record `/usr/bin/time` runtime and final metric for each.

### Task 4: Publish Comparison Report

**Files:**
- Create: `docs/reports/2026-02-12-xlog-neural-fulltrain-sequential.md`

**Steps:**
1. Document exact commands and log paths per example.
2. Summarize XLoG metric/runtime table.
3. Compare against DeepProbLog baseline where datasets/tasks overlap:
   - `01_minimal` vs baseline `01_minimal`
   - `02_coins` vs baseline `04_coins`
   - `04_hwf` vs baseline `10_hwf`
   - `05_poker` vs baseline `05_poker`
   - `06_clutrr` vs baseline `06_clutrr`
4. Mark non-overlapping tasks as non-comparable instead of forcing a claim.

### Task 5: Verify and Commit

**Steps:**
1. Run `python -m py_compile` on modified training scripts.
2. Validate report references and metric lines with `rg`.
3. Commit all changes with a benchmark-specific message.
