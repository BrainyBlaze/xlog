# Supervisor Goal 031 — W3.4 Kernel Fusion (Bench Spike + Conditional Implementation)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G30 (Path C bundle expansion) DONE; merged to `main @ 7bb56e4d`. W3.7+W3.8+W3.9 added; W3.3 stays OPEN under original ≥ 2.0× gate. G31 begins Path C execution roadmap.
**Date:** 2026-05-13.

---

## Context

W3.4 row from `docs/v065-closure-board.md @ 7bb56e4d`:

> | W3.4 | ROADMAP item #11 | OPEN | — | Kernel fusion where benchmarks show materialization overhead dominates. Targeted candidates: layout + count, or count + materialize. | Bench: fused kernel shows **≥ 1.3× speedup** vs. 2-kernel sequence on a fixture where materialize is the long pole; deterministic. No regression on a small fixture where fusion penalty exceeds savings (must auto-disable below threshold). |

Per locked discipline `feedback_perf_bench_spike_first.md`: any perf item with a quantitative gate MUST start with a minimum-viable bench spike that measures the gate target before any full implementation commits. Failed spikes stay unmerged as evidence; only spike-validated approaches proceed to full implementation.

### Current WCOJ kernel pipeline (target for fusion)

Per `crates/xlog-cuda/kernels/wcoj.cu` and `crates/xlog-cuda/src/provider/wcoj.rs`:

1. **layout** — sort columns to enable binary-search intersection (`wcoj_layout_*` family).
2. **count** — per-row intersection-size pass (output counts per source row).
3. **exclusive scan** — prefix sum on counts → output offsets.
4. **materialize** — emit triples to the per-row offsets from #3.

The two fusion candidates and their data-dependency profiles:

* **(A) layout + count fusion** — both phases are per-row, no inter-phase global data dependency. Lower implementation complexity; expected gain bounded by layout phase share of total kernel time (likely sub-dominant on superhub-50K where intersection-emit is the long pole).
* **(B) count + materialize fusion** — the principled paper-§4 cut. Challenge: materialize requires per-row output offsets (from exclusive scan on count's output) before it can emit. Single-pass fusion requires either:
  * worst-case output buffer pre-sizing + atomic-counter for emit positions (memory cost + atomic contention),
  * two-pass single-launch design (count → on-device exclusive scan → materialize, all in one kernel via cooperative-groups grid sync),
  * persistent-threads in-place reduction with atomic emit (path-aligned with G27 architecture).

W3.4 acceptance gate (≥ 1.3× speedup "where materialize is the long pole") points to **(B)** as the principled target — fusing the long-pole phase with its predecessor is where the gain lives.

---

## G31 — W3.4 fusion: minimum-viable bench spike (candidate selection + measurement)

### Goal

Cut `bench-spike/w34-kernel-fusion` from `main @ 7bb56e4d`. Investigate both fusion candidates; pick the more promising one based on current kernel profiling; implement minimum-viable fused version; measure speedup at `superhub-50K` (and `superhub-200K` if available in existing bench harness) under V3 sample_size(200) + iters=1 paired-batching. Report ≥ 1.3× verdict.

**If spike achieves ≥ 1.3× at any tested scale:** G31 emits closure-readiness verdict + recommendation for G31-impl (full W3.4 implementation goal with dispatch gating + threshold auto-disable).

**If spike fails at all tested scales:** branch stays unmerged as evidence (per `feedback_perf_bench_spike_first.md`); G31 emits empirical-gap finding + recommendation (re-scope fusion candidate, defer W3.4 dependency until W3.9 production-scale fixture, etc.).

### Strategies (GQM+Strategies)

* **S31.1** Cut `bench-spike/w34-kernel-fusion` from `main @ 7bb56e4d`. Worktree at `.worktrees/w34-kernel-fusion`.

* **S31.2 — Candidate selection investigation.** Before any kernel write, run a profiling pass on the current 4-phase pipeline at `superhub-50K`:
  * Use `cudaEvent_t` start/stop pairs around each of the 4 phases (layout, count, scan, materialize) inside `CudaKernelProvider::wcoj_triangle_*` (or equivalent for whatever the existing canonical fixture exercises).
  * Gate the profiling instrumentation behind a temporary cargo feature `w34-spike-profiling` so production paths stay clean.
  * Capture per-phase median timings across ≥ 20 iterations at the canonical fixture.
  * Decide candidate (A) or (B) based on:
    * If `materialize` dominates ≥ 60% of total kernel time → candidate **(B) count + materialize**.
    * If `layout` + `count` together dominate ≥ 60% → candidate **(A) layout + count**.
    * Otherwise → pick the larger-share-of-total candidate; document the decision rationale.

* **S31.3 — Minimum-viable fused kernel implementation.** Based on S31.2 decision:
  * Write one new fused kernel in `crates/xlog-cuda/kernels/wcoj.cu` named `wcoj_triangle_fused_<candidate>` (e.g., `wcoj_triangle_fused_cm` for count+materialize).
  * For candidate (B) count+materialize, use the **two-pass single-launch** design: count phase writes per-row counts to a temporary buffer; on-device exclusive scan via cooperative-groups grid sync (or CUB device-wide scan launched as a child kernel); materialize phase reads offsets from the scan output and emits to a single output buffer.
  * Preserve determinism: output ordering must be row-major (canonical), bit-identical to the unfused baseline. Row-equality verified by `download_triples` comparison before any timing measurement.
  * No production code-path rerouting in this spike — keep the existing 4-phase pipeline as the live dispatch path; the fused kernel is an additional path callable only from the spike bench harness.

* **S31.4 — Bench harness wiring.** Add a new Criterion bench file `crates/xlog-integration/benches/wcoj_fusion_bench.rs`:
  * Two benchmark groups: `wcoj_unfused_baseline` and `wcoj_fused_<candidate>`.
  * V3 protocol: `sample_size(200)`, `iters=1` per measurement, paired-batching (alternate baseline + fused per iteration to share warm-cache state).
  * Fixtures: `superhub-50K` (canonical) + `superhub-200K` if present in existing bench harness; do NOT introduce new fixtures.
  * Row-equality assertion BEFORE first timing measurement: panic if `download_triples` from fused ≠ unfused.

* **S31.5 — Measurement protocol.** Run `cargo bench --bench wcoj_fusion_bench --features w34-spike-profiling` and capture:
  * Per-scale: baseline median + 95% CI, fused median + 95% CI, paired delta µs + 95% CI, paired delta %, speedup ratio (`baseline / fused`).
  * ≥ 1.3× gate verdict at each measured scale.
  * Per-phase timing breakdown (from S31.2 profiling) attached as evidence appendix.

* **S31.6 — Evidence README.** `docs/evidence/2026-05-13-w34-kernel-fusion-spike/README.md` MUST contain:
  * Parent SHA `7bb56e4d` explicit.
  * Branch HEAD SHA.
  * S31.2 candidate selection: per-phase timings + decision rationale.
  * S31.3 implementation summary (which candidate, which design, key design decisions).
  * S31.4 row-equality PASS confirmation (with row-count sample).
  * S31.5 per-scale measurement table.
  * ≥ 1.3× verdict per scale.
  * Closure-readiness verdict:
    * If ANY scale clears ≥ 1.3× → "W3.4 closure-ready: recommend G31-impl full implementation with dispatch gating".
    * If NO scale clears → "W3.4 spike empirically insufficient: recommend re-scope or defer pending W3.9".

* **S31.7** Branch UNMERGED to main + all G30 ancestor branches. No FF-merge, no push, no tag.

* **S31.8 — Final gates BEFORE the G31 commit.**
  * `cargo fmt --check --all` EXIT 0.
  * `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog --features w34-spike-profiling` EXIT 0.
  * `cargo test -p xlog-cuda-tests --test certification_suite --release` 1/1.
  * `cargo bench --no-run --bench wcoj_fusion_bench --features w34-spike-profiling` EXIT 0.

* **S31.9 — Forbidden behaviors.**
  * No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
  * No FF-merge into main.
  * No `docs/v065-closure-board.md` edit (closure-board edits only via approved closure proposals).
  * No `v0.6.6` references.
  * **No production code-path rerouting.** The existing 4-phase pipeline is the live dispatch path; the fused kernel is only called from the spike bench. Pinned by: `git diff main..HEAD -- crates/xlog-cuda/src/provider/wcoj.rs` MUST NOT include any change to live dispatch flow.
  * No closure proposal in this goal (G31 is spike-only).
  * No W3.4 marked DONE.
  * No relaxing the ≥ 1.3× gate.

* **S31.10** Single bundled commit subject `spike(w34): kernel-fusion bench spike (<candidate>) on superhub-50K[+200K]`.

### Questions

* **Q31.1** Branch HEAD SHA?
* **Q31.2** Candidate selected (A or B) + per-phase profiling timings + decision rationale?
* **Q31.3** Fused kernel implementation summary (file paths + design choices, especially exclusive-scan strategy for candidate B)?
* **Q31.4** Row-equality PASS at all measured scales BEFORE timing?
* **Q31.5** Per-scale: baseline median + 95% CI + fused median + 95% CI + paired delta + speedup ratio + ≥ 1.3× verdict?
* **Q31.6** Closure-readiness verdict + recommendation (G31-impl or re-scope)?
* **Q31.7** Branch unmerged from main + G30 ancestor branches?
* **Q31.8** Final gates (fmt, warnings build, CUDA cert, bench compile) all EXIT 0?

### Metrics

* **M31.1** `bench-spike/w34-kernel-fusion` exists; HEAD reachable from neither `main` nor G30 ancestor branches.
* **M31.2** Evidence README exists with all sections.
* **M31.3** Row-equality PASS at all measured scales.
* **M31.4** Per-scale measurement tables populated with medians + CI + ratios.
* **M31.5** Closure-readiness verdict explicit + grounded in measurement data.
* **M31.6** `git diff main..HEAD -- crates/xlog-cuda/src/provider/wcoj.rs` shows NO live-dispatch-flow change (the dispatch entry to fused kernel exists only behind the spike feature gate or as an additional method).
* **M31.7** Final gates all EXIT 0 captured in evidence README or commit message.
* **M31.8** `git tag --points-at HEAD` empty; `git ls-remote --heads origin "bench-spike*"` empty.
* **M31.9** Branch unmerged from main.

### Supervisor validation per locked protocol

* Read evidence README end-to-end.
* `git rev-parse <branch>` ≠ `main`; `git merge-base --is-ancestor <branch> main` returns false.
* Verify M31.6 no-live-dispatch-flow-change.
* Verify per-scale measurements + ≥ 1.3× verdicts.
* Verify closure-readiness recommendation grounded in data.
* Run final gates from supervisor session.
* Verify branch unmerged + no tag + no origin push.

### Decision branching after G31

* **If spike clears ≥ 1.3× at ≥ 1 scale:**
  * G32 = full W3.4 implementation goal (production dispatch gating + threshold auto-disable + acceptance cert grid + closure proposal).
  * G33+ = continue Path C roadmap: W3.5 shared-memory, W3.6 warp primitives, W3.7 helper-relation splitting, W3.8 stream multiplexing, W3.9 production-scale bench, G37 full-bundle integration, G38 W3.3 closure proposal.

* **If spike fails at all scales:**
  * G32 = present empirical-gap finding to user + decision options:
    * (a) Re-scope W3.4 candidate (try the other fusion shape).
    * (b) Defer W3.4 dependency until W3.9 production-scale fixture is built (spike on production scale may yield different ratios).
    * (c) Amend W3.4 gate (analogous to G29 — only with explicit user authorization and empirical justification).
  * The empirical chain accumulates; the gap is documented; no W3.4 DONE claim.

Proceed: cut bench-spike branch from main, profile current pipeline, pick fusion candidate, implement minimum-viable fused kernel, measure under V3 paired-batching, write evidence README with closure-readiness verdict, single bundled commit. Emit REVIEW REQUEST with HEAD SHA + per-scale ratios + closure-readiness recommendation.
