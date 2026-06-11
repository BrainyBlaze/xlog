# S1 — Aggregate-Fused WCOJ Triangle Group-By-Root Count

Gate (research plan §5): fused >= 5x vs unfused (materialize + groupby
count) on skewed hub fixtures; <= 1.1x regression on small uniform.

Host: local dev GPU (WSL2, same box as W5.2 evidence). Command:

    cargo test -p xlog-cuda-tests --test test_wcoj_groupby_root_count \
      --release -- --ignored --nocapture

Measured 2026-06-11 (median of 5 reps, warmup excluded; parity vs host
brute-force oracle asserted before timing):

| fixture | unfused median | fused median | speedup | gate |
|---|---|---|---|---|
| hub_10k_z16 (n_xy=11000, n_yz=161000) | 11.665 ms | 1.927 ms | 6.05x | PASS (>=5x) |
| hub_50k_z16 (n_xy=51000, n_yz=801000) | 14.635 ms | 2.728 ms | 5.37x | PASS (>=5x) |
| small_uniform_200 | 4.885 ms | 1.960 ms | 2.49x | PASS (speedup, regression bound unused) |

Verdict: S1 GATE PASSED. Production executor wiring authorized per
docs/plans/2026-06-11-factorized-agg-wcoj-plan.md step 4.

Correctness evidence: groupby_root_count_matches_oracle_{small,skewed_hub}
and groupby_root_count_empty_intersection_roots_are_absent (3/3 green,
fused == oracle == unfused baseline).
