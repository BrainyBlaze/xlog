#!/usr/bin/env python
"""Repro harness for external-consumer session buffer-cap faults.

Mirrors the consumer usage shape recorded in
the external consumer put schedule:

  compile the external session workload (3 modules, 14 queries, count aggregates)
  loop N times:
      put 7 fixture relations (fact relation growing 64 -> ~2900 rows)
      put fusion_contested_input / value_contrariety_input (0-3 rows)
      evaluate()
      read every query result

Arms:
  --mode persistent   one session for the whole loop (no rotation).
                      Expected to hit the loud persistent-session fault:
                      "compact_buffer_by_device_mask_device_count:
                       mask len N > row cap M"
  --mode rotate       fresh session per evaluate + replay of all puts
                      (the consumer's workaround). Hunts the
                      nondeterministic rotate-mode CUDA_ERROR_ILLEGAL_ADDRESS.

Out-of-domain exported-key detection: every exported key column is checked
against the generated id domain; invalid values are reported.

Usage:
  .venv/bin/python scripts/repro/external_consumer_session_fuzz.py \
      --fixture /path/to/external-consumer/fixtures/session-workload.xlog \
      --mode persistent --iters 60
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

TORCH_UINT32 = getattr(torch, "uint32", torch.int32)

DEVICE = "cuda"

# Domain bounds used both for generation and out-of-domain exported-key checks.
CLAIM_PRED_BASE = 5_000_000
N_PREDS = 40
N_SUBJECTS = 25
N_VALUES = 7


def col(data, dtype):
    return torch.tensor(data, device=DEVICE, dtype=dtype)


def make_external_consumer_fixture_relations(n_facts: int, seed: int) -> dict[str, list]:
    """Generate the 7 fixture relations + 2 auxiliary inputs for n_facts rows.

    Shapes follow the external put schedule; values are arranged so the
    census queries produce non-empty, growing outputs:
      - claim preds (P >= 5_000_000) in (P, A0) groups of growing size
      - Pro >= 0.5 so belnap_supported holds
      - one evidence row per fact so source_backed_fact holds
      - State 1 (committed) for most rows, 2 (quarantined) for some
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = list(range(n_facts))

    # (P, A0) groups of size n/N_PREDS with varying values inside each
    # group — drives the quadratic census query (contrariety_candidate_pair)
    # so query OUTPUT grows superlinearly while input grows linearly.
    pred = [CLAIM_PRED_BASE + (i % N_PREDS) for i in ids]
    subj = [i % N_PREDS for i in ids]
    val = [(i // N_PREDS) % N_VALUES for i in ids]
    state = [2 if i % 5 == 0 else 1 for i in ids]
    pro = [0.9] * n_facts
    contra = [0.01 if i % 3 else 0.06 for i in ids]

    fixture_fact = [
        col(ids, torch.int64),                      # F
        col(pred, torch.int64),                     # P
        col(subj, torch.int64),                     # A0 (subject)
        col(val, torch.int64),                      # A1 (value)
        col(pro, torch.float32),                    # Pro
        col(contra, torch.float32),                 # Contra
        col(state, TORCH_UINT32),                   # State
        col([1] * n_facts, torch.int32),            # Epoch
        col([0] * n_facts, TORCH_UINT32),           # Origin
        col([7] * n_facts, TORCH_UINT32),           # PS  (shared in-group)
        col(subj, TORCH_UINT32),                    # A0S (shared in-group)
        col([0] * n_facts, TORCH_UINT32),           # A1S (shared in-group)
    ]

    # ~240-row-scale evidence: one row per fact (grows with facts).
    fixture_evidence = [
        col(ids, torch.int64),                      # E
        col(ids, torch.int64),                      # F
        col([0] * n_facts, TORCH_UINT32),           # Modality
        col([0] * n_facts, torch.int64),            # Trace
        col([0] * n_facts, torch.int32),            # Code
        col([0] * n_facts, torch.int32),            # Dist
        col([0] * n_facts, torch.int32),            # Snap
        col([0] * n_facts, torch.int32),            # Layer
        col([0] * n_facts, torch.int32),            # Head
        col([0] * n_facts, torch.int32),            # Start
        col([5] * n_facts, torch.int32),            # End (> Start)
    ]

    # Justification edges ~1.3x facts: premises reference earlier facts.
    n_edges = max(1, (n_facts * 13) // 10)
    e_ids = list(range(n_edges))
    e_fact = [(i * 7) % n_facts for i in e_ids]
    e_premise = [(i * 3) % n_facts for i in e_ids]
    fixture_justification_edge = [
        col(e_ids, torch.int64),                    # J
        col(e_fact, torch.int64),                   # F
        col([(i % 6) for i in e_ids], torch.int64), # Rule
        col(e_premise, torch.int64),                # Premise
        col([0] * n_edges, torch.int32),            # Rank
        col([0] * n_edges, torch.int64),            # Group
    ]

    fixture_rule = [
        col(list(range(6)), torch.int64),
        col([0] * 6, torch.int64),
        col([0] * 6, TORCH_UINT32),
        col([0.5] * 6, torch.float32),
        col([0] * 6, TORCH_UINT32),
    ]
    fixture_rule_body = [
        col(list(range(12)), torch.int64),
        col([i % 3 for i in range(12)], torch.int32),
        col([i % 6 for i in range(12)], torch.int64),
        col([0] * 12, TORCH_UINT32),
    ]
    fixture_rule_binding = [
        col(list(range(36)), torch.int64),
        col([0] * 36, TORCH_UINT32),
        col([i % 4 for i in range(36)], torch.int32),
        col([0] * 36, TORCH_UINT32),
        col([0] * 36, torch.int32),
    ]
    n_viol = int(torch.randint(0, 11, (1,), generator=g).item())
    fixture_violation = [
        col(list(range(n_viol)), torch.int64),
        col([(i * 11) % max(1, n_facts) for i in range(n_viol)], torch.int64),
        col([0] * n_viol, TORCH_UINT32),
        col([0.1] * n_viol, torch.float32),
        col([0] * n_viol, torch.int32),
    ]

    n_contested = int(torch.randint(0, 4, (1,), generator=g).item())
    contested = [i * 5 for i in range(n_contested) if i * 5 < n_facts]
    fusion_contested_input = [col(contested, torch.int64)]

    n_vc = int(torch.randint(0, 3, (1,), generator=g).item())
    vc_pairs = [(i % N_VALUES, (i + 1) % N_VALUES) for i in range(n_vc)]
    value_contrariety_input = [
        col([p[0] for p in vc_pairs], torch.int64),
        col([p[1] for p in vc_pairs], torch.int64),
    ]

    # Exact relation keys are required by the external workload schema.
    return {
        "wmir_fact": fixture_fact,
        "wmir_evidence": fixture_evidence,
        "wmir_justification_edge": fixture_justification_edge,
        "wmir_rule": fixture_rule,
        "wmir_rule_body": fixture_rule_body,
        "wmir_rule_binding": fixture_rule_binding,
        "wmir_violation": fixture_violation,
        "fusion_contested_input": fusion_contested_input,
        "value_contrariety_input": value_contrariety_input,
    }


def check_out_of_domain_keys(qr, n_facts: int) -> list[str]:
    """Detect exported key columns outside the generated domain."""
    problems = []
    for name, capsule in zip(qr.columns, qr.tensors):
        try:
            # clone() immediately: pins values independent of capsule
            # lifetime so a stale-export bug cannot masquerade as (or hide)
            # device-memory corruption.
            t = torch.from_dlpack(capsule).clone()
            torch.cuda.synchronize()
        except Exception:
            continue
        if t.dtype not in (torch.int64, torch.int32):
            continue
        hi = int(t.max().item()) if t.numel() else 0
        lo = int(t.min().item()) if t.numel() else 0
        # Generous domain bound: ids, values, preds, fixed-point masses.
        if hi > CLAIM_PRED_BASE + N_PREDS or lo < -1:
            bad = ((t > CLAIM_PRED_BASE + N_PREDS) | (t < -1)).nonzero().flatten()
            idxs = bad.tolist()
            vals = t[bad][:8].tolist()
            contiguous_tail = bool(idxs) and idxs[-1] == t.numel() - 1 and (
                len(idxs) == idxs[-1] - idxs[0] + 1
            )
            problems.append(
                f"{qr.relation_name}.{name}: out-of-domain [{lo}, {hi}]"
                f" (id domain < {n_facts}, pred domain"
                f" < {CLAIM_PRED_BASE + N_PREDS}); rows={t.numel()}"
                f" bad={len(idxs)} first_idx={idxs[0]} last_idx={idxs[-1]}"
                f" contiguous_tail={contiguous_tail} sample_vals={vals}"
                f" sample_idxs={idxs[:8]}"
            )
    return problems


def run(args) -> int:
    import pyxlog

    src = open(args.fixture).read()
    program = pyxlog.LogicProgram.compile(src)
    session = program.session()
    print(f"compiled {args.fixture}; mode={args.mode} iters={args.iters}")

    fact_lo, fact_hi = 64, 2900
    step = max(1, (fact_hi - fact_lo) // max(1, args.iters - 1))

    prev_counts: dict[str, int] | None = None
    keepalive: list = []
    for it in range(args.iters):
        n_facts = min(fact_hi, fact_lo + it * step)
        rels = make_external_consumer_fixture_relations(n_facts, seed=args.seed + it)
        if args.keepalive:
            keepalive.append(rels)

        if args.mode == "rotate" and it > 0:
            session = program.session()

        try:
            if args.sync_before_put:
                # Hypothesis probe: if xlog ingests DLPack tensors on a
                # different CUDA stream than torch wrote them, syncing here
                # closes the race and strikes should vanish.
                torch.cuda.synchronize()
            for name, columns in rels.items():
                session.put_relation(name, columns)
            result = session.evaluate()
            if args.sync_after_evaluate:
                # Hypothesis probe: if evaluate() returns with async
                # copies still in flight on xlog's stream, a consumer
                # reading the DLPack capsules from another stream races
                # them. A device-wide sync here must close that race.
                torch.cuda.synchronize()
        except Exception as exc:  # the strike we are hunting
            print(
                f"\nSTRIKE iter={it} n_facts={n_facts} mode={args.mode}\n"
                f"  {type(exc).__name__}: {exc}",
                flush=True,
            )
            return 1

        # Query results come back positionally (__xlog_query_N) in source
        # order; the census query contrariety_candidate_pair is index 11.
        counts = {i: qr.num_rows for i, qr in enumerate(result.queries)}
        out_of_domain_keys: list[str] = []
        for qr in result.queries:
            out_of_domain_keys += check_out_of_domain_keys(qr, n_facts)
        if out_of_domain_keys:
            print(f"\nOUT-OF-DOMAIN KEYS iter={it} n_facts={n_facts}:")
            for p in out_of_domain_keys:
                print(f"  {p}")
            return 2

        if it % 10 == 0 or it == args.iters - 1:
            total = sum(counts.values())
            print(
                f"iter={it:4d} n_facts={n_facts:5d} "
                f"total_query_rows={total:7d} "
                f"census={counts.get(11, 0)}",
                flush=True,
            )
        prev_counts = counts

    print("clean: no strike, no out-of-domain keys")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--mode", choices=["persistent", "rotate"], default="persistent")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260612)
    ap.add_argument("--sync-before-put", action="store_true")
    ap.add_argument("--sync-after-evaluate", action="store_true")
    ap.add_argument(
        "--keepalive",
        action="store_true",
        help="Keep every generated torch tensor alive for the whole run, "
        "ruling out torch caching-allocator reuse under live xlog views.",
    )
    args = ap.parse_args()
    t0 = time.time()
    rc = run(args)
    print(f"elapsed {time.time() - t0:.1f}s rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
