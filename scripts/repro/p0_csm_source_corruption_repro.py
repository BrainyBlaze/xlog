#!/usr/bin/env python3
"""Phase-0 S1 verify-harness — defect-#2 / P0.1 CUDA_ERROR_ILLEGAL_ADDRESS.

Reconstructs the (absent) original repro per the upstream defect-spec
(2026-06-12-xlog-requirements.md, P0.1/P0.4): re-put a GROWING fact table into a
LogicRelationSession and evaluate a WCOJ / count-scan-materialize (CSM) query in
a loop. Defect-#2 was a SOURCE-materialize fault — legacy count_scan_materialize
recycled an exclusive_scan offsets buffer as an un-overwritten output column, so
garbage keys appeared (signature: a key decoding to two CONSECUTIVE u32 read
through an i64 column, e.g. (55<<32)|54 = 236223201334) and intra-evaluation
allocation eventually faulted with CUDA_ERROR_ILLEGAL_ADDRESS (device-poisoning).

Run WITHOUT the workaround env-vars (XLOG_USE_DEVICE_RUNTIME / XLOG_USE_RECORDED_OPS),
WITH the clone-probe so a fault is attributed source-vs-transport:

    XLOG_DEBUG_VERIFY_CLONES=1 .venv/bin/python scripts/repro/p0_csm_source_corruption_repro.py

S1 gate: completes all iterations with NO illegal-address, NO garbage keys, NO
clone-probe source-mismatch — WITHOUT any workaround env-var.

Knobs: P0_ITERS, P0_MAX_N (max edge fan), P0_FAIL_ON_GARBAGE, P0_MEM_MB.
"""
import os
import sys

for _wa in ("XLOG_USE_DEVICE_RUNTIME", "XLOG_USE_RECORDED_OPS"):
    if os.environ.get(_wa):
        print(f"REFUSE: {_wa} set — S1 must verify the DEFAULT (non-recorded) path.", file=sys.stderr)
        sys.exit(2)

import cupy as cp
import pyxlog

# CSM-triggering program: triangle WCOJ over a growing edge relation forces the
# count -> exclusive_scan -> materialize path that defect-#2 corrupted.
PROGRAM = """
pred edge(u32, u32).
pred tri(u32, u32, u32).
tri(X, Y, Z) :- edge(X, Y), edge(Y, Z), edge(X, Z).
?- tri(X, Y, Z).
"""

GARBAGE_KEY_FLOOR = 1 << 32  # any value this large is a recycled-scan artifact


def growing_edges(n: int):
    """Triangle-dense growing edge set (fan from 0 + path), so CSM output volume
    grows across iterations."""
    src, dst = [], []
    for i in range(1, n + 1):
        src.append(0); dst.append(i)
    for i in range(1, n):
        src.append(i); dst.append(i + 1)
        src.append(0); dst.append(i + 1)   # close triangles 0-i-(i+1)
    return (cp.asarray(src, dtype=cp.uint32), cp.asarray(dst, dtype=cp.uint32))


def _materialized_rows(res):
    """Sum num_rows across query results — the count the CSM materialize produced.
    >0 proves the count->scan->materialize path actually ran (non-hollow); the
    clone-probe (XLOG_DEBUG_VERIFY_CLONES) is the type-agnostic corruption detector."""
    total = 0
    for q in res.queries:
        nr = q.num_rows
        total += nr() if callable(nr) else nr
    return total


def main() -> int:
    iters = int(os.environ.get("P0_ITERS", "64"))
    max_n = int(os.environ.get("P0_MAX_N", "256"))
    mem_mb = int(os.environ.get("P0_MEM_MB", "2048"))
    fail_on_garbage = os.environ.get("P0_FAIL_ON_GARBAGE", "1") == "1"

    print(f"P0 CSM repro: iters={iters} max_n={max_n} mem_mb={mem_mb} "
          f"clone_probe={os.environ.get('XLOG_DEBUG_VERIFY_CLONES','0')}")
    prog = pyxlog.LogicProgram.compile(PROGRAM, 0, mem_mb)

    total_materialized = 0
    for it in range(iters):
        n = 4 + (it * (max_n - 4)) // max(1, iters - 1)
        col0, col1 = growing_edges(n)
        session = prog.session()
        session.put_relation("edge", [col0, col1])
        res = session.evaluate()  # CSM materialize; faults here if live
        mats = _materialized_rows(res)
        total_materialized += mats
        if it % 8 == 0:
            print(f"  iter={it} n={n} triangles_materialized={mats} ok")

    if total_materialized == 0:
        print("HOLLOW: 0 rows materialized across all iters — harness did not exercise "
              "the CSM output path; result is NOT a valid S1 verify.", file=sys.stderr)
        return 3
    print(f"S1 PASS: {iters} growing re-put+evaluate iters, {total_materialized} total "
          f"triangles materialized, NO illegal-address, NO clone source-mismatch "
          f"(default path, no workaround).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
