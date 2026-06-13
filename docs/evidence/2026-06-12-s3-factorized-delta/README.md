# D3 S3 Gate Evidence — Factorized Recursive Delta (2026-06-12)

Design: `docs/plans/2026-06-12-d3-factorized-delta-design.md`
Source: git archive of `feat/d3-factorized-delta` HEAD `0ae50b41`
Raw log: `runpod-s3-gate.log` (this directory)

## Verdict: **PASS** — both scales, every rep, with ~8×–12× margin on the memory gate

| Scale | Baseline (production executor) | Spike (factorized delta) | Peak ratio (gate ≥5×) | Wall-clock ratio (gate ≤1.2×) |
|---|---|---|---|---|
| primary k=4 b=256 (n=1024, \|TC\|=1,048,576) | median 2403.5–2429.8 ms / **2332.0 MiB** peak | median 208.6–211.3 ms / **58.3 MiB** peak | **40.03×** | **0.087×** (11.5× faster) |
| secondary k=4 b=384 (n=1536, \|TC\|=2,359,296) | median 8112.4 ms / **7857.0 MiB** peak | median 572.2 ms / **131.1 MiB** peak | **59.95×** | **0.071×** (14.2× faster) |

The ratio *grows* with scale (40× → 60×), as predicted by the design's `raw/|R| = b/k`
analysis — the baseline's witness-multiplied raw join + per-iteration sorts scale faster
than the spike's bitmap + novel terms.

## Measurement discipline

- Host: ephemeral RunPod RTX A4000 16 GB (driver 580.119.02, nvcc 12.4, rustc 1.96.0),
  community cloud, pod `claude-agent-xlog-d3-s3-gate-20260612` (`tvra5q58pnwt2b`) —
  **deleted after the run, deletion confirmed** (404 + pod list).
- Build: `XLOG_CUBIN_ARCHS=sm_86`, release profile.
- On-pod parity suite first: 4/4 green (step CPU-oracle parity incl. raw emit-order,
  spike TC loop vs oracle vs production executor on an irregular cyclic digraph,
  fail-closed out-of-domain, empty/saturated steps).
- 3 isolated serial gate runs (`--test-threads=1`, fresh process each) × 3 reps inside,
  fresh fixture (own memory manager) per rep; SM clocks/temp recorded around each run
  (1560–1920 MHz, 36–64 °C). Secondary scale: 1 isolated run × 3 reps.
- Peak memory: `GpuMemoryManager::peak_bytes()` high-water mark (`fetch_max` at both
  reservation funnels), `reset_peak()` after fixture upload, before the measured region.
  Peaks are bit-identical across reps (deterministic allocation sequence).
- Row-set parity asserted in every rep: row count == n²; full downloaded row-set size
  checked on rep 0 of each engine.
- Baseline = the **unmodified production executor** executing
  `q(X,Y) :- edge(X,Y). q(X,Z) :- q(X,Y), edge(Y,Z).` (semi-naive
  `execute_recursive_scc`: hash_join_v2 → diff_gpu → union_gpu). Spike shares
  `union_gpu`, so the measured difference isolates the join+diff+dedup delta pipeline.

## What the numbers mean

Per design §2/§3: the baseline materializes one flat row per derivation witness
(67M rows/iteration at primary scale; 226M at secondary), sorts that raw buffer inside
`diff_gpu`, and re-sorts all of R every iteration. The factorized step evaluates
`novel[x] = (∪_y edge[y]) \ R[x]` over a dense-domain characteristic bitvector
(128 KB / 288 KB) and flattens only surviving novel tuples, already lex-sorted and
deduped. Both the memory blowup AND the sort wall-clock disappear.

## Failed attempts / cleanup status (honesty record)

- 4 pods were wasted earlier the same day (~$0.70): 3 because `--startSSH` was missing
  from the create command (container runs, sshd never starts — looks like a dead pod),
  1 (`ypz7qeq2cp16ew`) killed by another agent as apparently-idle for the same reason.
  Recipe lesson recorded in project memory.
- First run on the measuring pod failed parity: the LaunchRecorder fix (conditional
  R-column reads recorded after preflight) was applied locally but uncommitted, so the
  pod tarball didn't have it. Committed as `0ae50b41`; the measured run is entirely from
  that commit.
- A mid-run pod rename (`runpodctl pod update --name`) reset the container and wiped the
  first build — rename pods before starting work, never mid-run.
- All pods created for this gate are deleted; the remaining pods on the account
  (`m46-supplement-arms`, `ase-ogbl-biokg-*`) belong to other owners and were not touched.

## Gate decision

S3 **PASS** → per the design (§6 step 6), D3 proceeds to a Phase B production-integration
plan (promoter recognition of linear recursive rules, kill switch, sparse-domain strategy,
cost-model gating). Phase B is a separate plan with its own gates; nothing in this
evidence claims production integration.
