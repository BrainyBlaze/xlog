# XLOG

XLOG is a **GPU-accelerated Datalog engine** written in Rust with CUDA kernels. It compiles `.xlog` programs into relational plans and executes them efficiently on NVIDIA GPUs.

## Status (`main`)

- Deterministic `xlog-logic` tier: **production-ready** (Phase 3 complete).
- CUDA certification suite: **140/140 passing** (see `docs/plans/2026-01-14-cuda-certification-results.md`).
- Phase 4 complete: `xlog-prob` (exact `exact_ddnnf` + approximate `mc`) and Python `xlog_gpu` (PyO3 + DLPack) are implemented on `main`.

## What Works

- Datalog rules + facts, recursion (semi-naive fixpoint), stratified negation
- Comparisons (`= != < <= > >=`) and Prolog-style arithmetic (`is`) with builtins (`abs/min/max/pow/cast`)
- GPU relational operators: hash joins (inner/semi/anti/left-outer), sort, filter/compact, dedup/distinct, set ops (union/diff), groupby aggregates (count/sum/min/max/logsumexp)
- Interop: Arrow IPC (host copy) and DLPack (zero-copy per GPU column)

## Quickstart

### Build + test

Requires Linux + NVIDIA CUDA GPU (compute capability **sm_70+**).

```bash
cargo test --workspace --all-targets --release
```

### Run a `.xlog` example

```bash
cargo run -p xlog-logic --example xlog_run -- examples/xlog/00-basics/01_tc_reachability.xlog
```

See `examples/README.md` for the full example suite and runner flags.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [xlog-prob architecture](docs/architecture/xlog-prob.md)
- [Roadmap](docs/ROADMAP.md)
- [Examples](examples/)
- [CUDA certification results](docs/plans/2026-01-14-cuda-certification-results.md)
- [Current validation summary](docs/VALIDATION_REPORT.md)
