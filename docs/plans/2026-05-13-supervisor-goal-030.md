# Supervisor Goal 030 — Path C: Full WCOJ Bundle per SRDatalog Paper (REJECT G29 amendment; expand v0.6.5 scope)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G29 closure proposal (commit `2088c4c8` on `feat/w33-closure-proposal-iteration-1`) explicitly **REJECTED** by user. The amendment-to-variance-reduction-proxy is NOT accepted. W3.3 remains OPEN under the ORIGINAL ≥ 2.0× wall-time speedup gate.
**Date:** 2026-05-13.

---

## User directive (verbatim 2026-05-13)

> *"path c, no deffers, no simplification, document all findings and dispatch path c to codex with attached paper, requirements, and goals. the goal is to deliver FULL implementation and achieve target performance metrics according paper and xlogs architecture specifics. do not claim done until the goal is truly achieved. no excuse, no violations, no negotiating."*

This goal initiates the **Path C** v0.6.5 scope expansion: implement the FULL SRDatalog paper bundle of WCOJ mechanisms, then re-evaluate D7a on production-scale workloads. The G29 amendment is rejected; no metric is relaxed; no gate is negotiated; closure requires actually meeting the paper's performance envelope.

---

## Paper attachment + requirements (load-bearing)

**Paper:** Sun, Qi, Gilray, Kumar, Micinski. *"Scaling Worst-Case Optimal Datalog to GPUs"* arXiv:[2604.20073](https://arxiv.org/abs/2604.20073). HTML: https://arxiv.org/html/2604.20073. PDF: https://arxiv.org/pdf/2604.20073.

### What the paper claims (verbatim from §6)

> *"These massive gains reflect the combined synergy of SRDatalog's architectural mechanisms: raw GPU memory bandwidth, the columnar WCOJ execution model, flat-array delta merges, stream-order rule multiplexing, and histogram-guided skew mitigation."*

Geometric-mean speedups: **21× over Ascent, 14× over FlowLog, 26× over Soufflé** on **17 datasets across 7 workload classes; input sizes 977K to 126.9M tuples; iteration counts 24 to 2,322**. Hardware: **NVIDIA RTX 6000 Ada (48 GB)** vs **AMD EPYC 9655** CPU baseline.

### Five paper mechanisms vs v0.6.5 board coverage

| # | Paper mechanism | Paper section | v0.6.5 board ID | Status |
|---|---|---|---|---|
| 1 | Raw GPU memory bandwidth | (hardware) | n/a | n/a |
| 2 | Columnar WCOJ execution model | §4 | W3.1 + W3.2 | ✓ DONE |
| 3 | Flat-array delta merges | §4 | W4.1 | ✓ DONE |
| 4 | **Stream-aligned rule multiplexing** | §6 (CALM justification) | **MISSING** | **NEW: W3.8** |
| 5 | Histogram-guided skew mitigation | §5 | W3.3 | OPEN — paper-faithful G27 impl preserved on `feat/w33-persistent-threads-work-stealing @ d986cf10` |

**Additionally missing from board:** Helper-relation splitting from §5 (rule-rewriter that elevates buried inner-level skew to the root, enabling histograms to address deep join trees). The paper's 35.8× peak case (HeapAllocHelper) is attributed specifically to this. → **NEW: W3.7**

**Also missing:** Production-scale benchmark suite. Our current bench is `superhub-50K` (synthetic, 3-variable, single-rule, 50K tuples). The paper's smallest tested workload is **977K tuples** — 20× larger. The board needs a DOOP-class production benchmark fixture to validate against paper claims. → **NEW: W3.9**

### Paper limitations the original W3.3 gate ignored

Direct quotes from paper §6 and §7 (limitations):

> *"Across their extensive fixpoint chains, the vast majority of iterations generate only a microscopic trickle of new delta tuples. This severe structural sparsity inherently bounds peak GPU efficiency."*

> *"Root-level histogram only balances outermost variable; deep inner skew requires surgical helper-relation splitting."*

> *Single-rule workloads (Andersen, ddisasm, polonius): negligible (<3%) gains from stream multiplexing.*

Our `superhub-50K` benchmark is structurally **single-rule, shallow-join, small-fixture** — exactly the regime the paper's authors acknowledge as **inherently bounded**. The original W3.3 ≥ 2.0× gate calibrated against the paper's headline speedup was inappropriate for this benchmark.

---

## G30 — Path C scope expansion + Codex execution dispatch

### Goal

Cut `feat/v065-path-c-bundle-expansion` from `main @ 3f8e5d4c`. Three substantive deliverables:

**A. Findings documentation (paper grounding + 19-iteration empirical chain summary).**

**B. v0.6.5 closure board scope expansion (W3.7 + W3.8 + W3.9 added; W3.3 status notes updated to reflect Path C plan).**

**C. Path C execution roadmap (concrete supervisor-goal sequence for the implementation work).**

Per W1.1 process rule #1: the board mutations are STAGED on this branch; user approval is required before they take effect on main. However the user has pre-authorized Path C in the thread, so this goal commits the board expansion artifact pending the user's final review of the specific text + sequencing.

### Strategies (GQM+Strategies)

* **S30.1** Cut `feat/v065-path-c-bundle-expansion` from `main @ 3f8e5d4c`. Worktree at `.worktrees/v065-path-c`.

* **S30.2 — Findings documentation.** Create `docs/plans/2026-05-13-path-c-paper-grounded-findings.md` containing:
  * **§1 Paper grounding** — verbatim quotes of paper §6 attribution ("combined synergy of mechanisms"); paper Figure 5 ablation (histogram-alone 1.1-35.8× range, HeapAllocHelper 35.8× from helper-relation splitting); paper §5 Algorithm 2 (HG-WCOJ kernel); paper limitations on single-rule + sparse-fixpoint + shallow-join workloads.
  * **§2 19-iteration empirical chain summary** — G11 through G29 with commit anchors, key findings, what worked, what didn't. Reference the G11-G28 evidence READMEs.
  * **§3 Gap analysis** — paper-vs-board mapping table; identifies W3.7 (helper-relation splitting) and W3.8 (stream multiplexing) as MISSING; identifies W3.9 (production-scale bench) as MISSING.
  * **§4 G29 rejection rationale** — user directive verbatim; explanation that variance-reduction proxy is not accepted; closure requires meeting the ORIGINAL ≥ 2.0× gate empirically.
  * **§5 Path C execution roadmap** — supervisor-goal sequence (see S30.4).
  * **§6 Acceptance criteria** — W3.3 DONE requires D7a ≥ 2.0× measured on the new W3.9 production-scale benchmark fixture under the FULL bundle (W3.3 + W3.4 + W3.5 + W3.6 + W3.7 + W3.8). No partial closure, no proxy substitute, no scale-emergence claim without empirical demonstration.

* **S30.3 — Board expansion.** Update `docs/v065-closure-board.md`:
  * **Add row W3.7** in Wave 3:
    ```
    | W3.7 | Paper §5 | OPEN | — | Helper-relation splitting: AOT rule-rewriter that elevates buried inner-level skew (deep join trees) to the outermost variable by surgically materializing strategic sub-relation boundaries, enabling W3.3 histogram-guided slicing to address skew. Per paper Figure 3 (CallGraphEdge example) and Figure 5 (HeapAllocHelper 35.8× attribution). | Bench: a 6+-variable deep-join fixture with inner-level skew (DOOP-class CallGraphEdge analog) shows ≥ 2× speedup when rule-rewriter elevates the buried key to root vs. no-rewrite baseline; deterministic output preserved. |
    ```
  * **Add row W3.8** in Wave 3:
    ```
    | W3.8 | Paper §6 | OPEN | — | Stream-aligned rule multiplexing: AOT compiler pass that groups Count and Materialize kernels across independent rules by phase, dispatching them onto separate CUDA streams to overlap execution and memory allocation. CALM-justified per paper §6. | Bench: a rule-rich stratum fixture (3+ independent rules, DOOP datasets-class) shows ≥ 1.27× speedup vs. sequential dispatch on the canonical multi-rule fixture; deterministic output preserved across all rules; no regression on single-rule workloads (within ±3%). |
    ```
  * **Add row W3.9** in Wave 3:
    ```
    | W3.9 | Path C requirement | OPEN | — | Production-scale WCOJ benchmark suite: at-or-above paper's minimum tested scale (≥ 977K tuples, DOOP-class graph structure with naturally heavy keys); harness exercises full W3.3+W3.4+W3.5+W3.6+W3.7+W3.8 bundle. | Bench harness committed; at least 3 paper-class fixtures (CallGraphEdge analog, Andersen-class, ddisasm-class); harness runs full bundle + uniform-baseline pairs; reports ratio vs uniform baseline per fixture. |
    ```
  * **Update W3.3 row notes** to reflect Path C plan (status stays OPEN; required deliverable unchanged at ≥ 2.0× speedup; closure now explicitly depends on W3.4+W3.5+W3.6+W3.7+W3.8+W3.9 being DONE; reference `docs/plans/2026-05-13-path-c-paper-grounded-findings.md`).
  * **Update Status Tally:**
    ```
    | DONE | 13 (W2.4, W2.2, W2.1, W2.3, W2.5, W2.6, W3.1, W3.2, W4.1, W4.2, W4.3, W5.1, W5.2) |
    | IN-PROGRESS | 1 (W1.1) |
    | BLOCKED | 0 (—) |
    | OPEN | 12 (W3.3, W3.4, W3.5, W3.6, W3.7, W3.8, W3.9, W5.3, W5.4, W6.1, W6.2, W7.1) |
    | **Total** | **26** |
    ```
  * Update the "21 open items required for tag" header to "24 open items required for tag" (accounting for W3.7+W3.8+W3.9 additions).

* **S30.4 — Path C execution roadmap (S30.2 §5 content).** Document the supervisor-goal sequence:
  * **G31 — W3.4 Kernel fusion** (count+materialize OR layout+count single-kernel; auto-disable below threshold; per existing W3.4 acceptance gate ≥ 1.3×).
  * **G32 — W3.5 Shared-memory optimization** (small-relation `__shared__` path with threshold; per existing W3.5 acceptance gate ≥ 1.5×).
  * **G33 — W3.6 Warp-level `__shfl_*` primitives** (depends on W3.5 threshold; per existing W3.6 acceptance gate ≥ 1.3× below W3.5 threshold).
  * **G34 — W3.7 Helper-relation splitting AOT rule rewriter** (per paper §5; new acceptance gate per S30.3 W3.7 row).
  * **G35 — W3.8 Stream-aligned rule multiplexing AOT compiler pass** (per paper §6; new acceptance gate per S30.3 W3.8 row).
  * **G36 — W3.9 Production-scale benchmark suite** (DOOP-class fixtures; new acceptance gate per S30.3 W3.9 row).
  * **G37 — W3.3 full-bundle integration testing** (compose W3.3 + W3.4 + W3.5 + W3.6 + W3.7 + W3.8 on W3.9 fixtures; iterate until D7a ≥ 2.0× empirically achieved; correctness + D7b preservation maintained throughout).
  * **G38 — W3.3 closure proposal** (paper-faithful; meets ORIGINAL ≥ 2.0× gate; full bundle in place; user approval per W1.1 rule #1).
  * Each Gn is its own supervisor goal artifact in `docs/plans/2026-05-1{3,4,…}-supervisor-goal-N.md`. Each requires its own user-approved closure proposal.

* **S30.5 — G29 rejection cleanup.** The G29 closure-proposal branch (`feat/w33-closure-proposal-iteration-1 @ 2088c4c8`) is preserved as historical evidence per the `feedback_perf_bench_spike_first.md` discipline (failed/superseded branches stay unmerged). NO FF-merge of G29 to main. NO board edit applied. The G29 amendment text remains on the branch for the v0.6.6 record if W3.3 ever closes by amendment in the future.

* **S30.6 — Final gates BEFORE the G30 commit.**
  * `cargo fmt --check --all` EXIT 0.
  * `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` EXIT 0.
  * `cargo test -p xlog-cuda-tests --test certification_suite --release` 1/1.
  * `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` EXIT 0.

* **S30.7** Branch UNMERGED to all 20 parents (main + G11–G29). No FF-merge, no push, no tag.

* **S30.8 — Forbidden behaviors:**
  * No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
  * No FF-merge into main.
  * No `v0.6.6` references in code or board (per W1.1 rule #5).
  * No production-impl change (G30 is plan + board + findings documentation only).
  * **No claim of W3.3 DONE.** W3.3 stays OPEN until G37 + G38 produce empirical ≥ 2.0× under full bundle on production-scale workload.
  * No relaxing the D7a target. No proxy substitution. No scale-emergence handwave. No averaging. No measurement-surface shopping. The gate is wall-time speedup ≥ 2.0× on the W3.9 canonical fixture under the full bundle.
  * No marking ANY new item (W3.7, W3.8, W3.9) DONE without explicit user approval per W1.1 rule #1 after empirical demonstration.

* **S30.9** Single bundled commit subject `docs(v065): Path C bundle expansion — add W3.7 W3.8 W3.9 + paper-grounded findings + execution roadmap (rejects G29 amendment)`.

### Questions

* **Q30.1** Branch HEAD SHA?
* **Q30.2** Findings document path + sections present (§1 paper grounding, §2 empirical chain, §3 gap analysis, §4 G29 rejection, §5 execution roadmap, §6 acceptance criteria)?
* **Q30.3** Board edit: W3.7 + W3.8 + W3.9 rows added; W3.3 notes updated; tally updated to 26 total / 12 OPEN?
* **Q30.4** G29 branch preserved unmerged (no FF-merge into main)?
* **Q30.5** Final gates green (fmt + warnings build + CUDA cert + workspace tests)?
* **Q30.6** Branch unmerged from all 20 parents?

### Metrics

* **M30.1** `feat/v065-path-c-bundle-expansion` exists; HEAD reachable from none of 20 parents (main + G11-G29).
* **M30.2** `docs/plans/2026-05-13-path-c-paper-grounded-findings.md` exists with all 6 sections.
* **M30.3** `docs/v065-closure-board.md` updated: W3.7 + W3.8 + W3.9 rows present; W3.3 notes reference Path C plan; tally arithmetic correct (13 + 1 + 0 + 12 = 26).
* **M30.4** `git diff main..HEAD -- docs/plans/2026-05-13-w33-closure-proposal.md` byte-empty (G29 amendment text NOT merged into Path C branch).
* **M30.5** Final gates EXIT 0 captured inline in the findings document or commit message.
* **M30.6** `git tag --points-at HEAD` empty; `git ls-remote --heads origin "feat/v065*"` empty.
* **M30.7** Branch unmerged from all 20 parents.

### Supervisor validation per locked protocol

* Read findings document end-to-end.
* `git rev-parse feat/v065-path-c-bundle-expansion` ≠ all 20 parent SHAs.
* Verify board edit: W3.7 + W3.8 + W3.9 rows present with proper acceptance gates.
* Verify tally arithmetic 13 + 1 + 0 + 12 = 26.
* Verify W3.3 row still OPEN with original ≥ 2.0× gate intact (NO amendment applied).
* Verify G29 branch preserved unmerged.
* Run final gates from supervisor session.
* Verify branch unmerged + no tag + no origin push.

**If all gates green:** present G30 to the user for explicit approval of:
1. Board expansion (W3.7 + W3.8 + W3.9 additions)
2. G29 amendment rejection (no W3.3 OPEN→DONE)
3. Path C execution roadmap (G31-G38 sequence)
4. Authorization to proceed with G31 dispatch

Per W1.1 rule #1, the board change is staged until user accepts in thread. The user has pre-authorized Path C in the directive, so explicit approval here just confirms the specific text + sequencing before the FF-merge to main.

### Why this is the honest closure path

The paper claims 21-47× speedups via combined-synergy of 5 mechanisms. We've implemented 3 of them (#2 columnar, #3 delta merges, #5 histogram-guided in isolation per G27). The remaining 2 (#4 stream multiplexing, helper-relation splitting) plus production-scale benchmarking are required to honestly evaluate whether the paper's performance envelope is achievable in xlog. Path C commits to building all of it.

User directive verbatim: *"the goal is to deliver FULL implementation and achieve target performance metrics according paper and xlogs architecture specifics. do not claim done until the goal is truly achieved. no excuse, no violations, no negotiating."*

This goal honors that directive precisely.

Proceed: cut Path C branch from main, write findings document with paper grounding + empirical chain + gap analysis + G29 rejection + execution roadmap + acceptance criteria, stage board expansion with W3.7+W3.8+W3.9, leave G29 branch as historical evidence, run final gates, single bundled commit. Emit REVIEW REQUEST with HEAD SHA + findings document path + board diff summary + tally arithmetic + final gate results.
