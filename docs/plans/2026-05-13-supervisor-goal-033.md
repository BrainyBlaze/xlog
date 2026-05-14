# Supervisor Goal 033 — W3.4 Closure Proposal (stage OPEN→DONE for user approval)

**Supervisor:** Claude Code.
**Implementer:** Codex CLI on tmux session `codex-xlog`.
**Predecessor:** G32 W3.4 production implementation on `feat/w34-kernel-fusion-impl @ 70d2cf5e` PASSED all gates: superhub-50K **1.590×** (≥1.3×), superhub-1K **1.305×** (≥0.95×), cert grid 5/5, no W3.x regression, no board/plan diff, no `v0.6.6` strings, branch unmerged.
**Date:** 2026-05-13.

---

## Context

W3.4 acceptance contract (from `docs/v065-closure-board.md @ 7bb56e4d`):

> Bench: fused kernel shows **≥ 1.3× speedup** vs. 2-kernel sequence on a fixture where materialize is the long pole; deterministic. No regression on a small fixture where fusion penalty exceeds savings (must auto-disable below threshold).

Both gates satisfied by the G31+G32 evidence chain:

**Performance gate (≥ 1.3×)** — G32 measurement on superhub-50K under live threshold dispatch: **1.590×** (paired delta -670 µs, CI tight on both arms).

**Auto-disable gate (no regression on small fixture)** — G32 measurement on superhub-1K (input rows 2,730 < threshold 4,096): routes to unfused; ratio **1.305×** (≥0.95× preserved; routing verified by `wcoj_triangle_fused_dispatch_count` counter staying at 0 and `wcoj_triangle_unfused_dispatch_count` advancing).

W1.1 process rule #1: "No item is marked DONE without explicit user approval in the thread." G33 stages the closure artifact + board edit; **user approval is the gate for the FF-merge into main**.

---

## G33 — W3.4 closure proposal

### Goal

Cut `feat/w34-closure-proposal-iteration-1` from `feat/w34-kernel-fusion-impl @ 70d2cf5e` (the G32 production-impl HEAD; closure proposals are cut from the IMPL branch they propose to close, so the proposal commit carries the impl + the closure in a clean lineage).

Deliver three artifacts:

1. `docs/plans/2026-05-13-w34-closure-proposal.md` — closure proposal document with Response options 1/2/3.
2. Staged edit of `docs/v065-closure-board.md` — W3.4 row OPEN→DONE with closure note linking to evidence.
3. Tally update: DONE 13→14, OPEN 12→11, Total 26, "24 open items required for tag" → "23 open items required for tag".

Branch stays UNMERGED to main. The FF-merge requires explicit user approval per W1.1 rule #1.

### Strategies (GQM+Strategies)

* **S33.1** Cut `feat/w34-closure-proposal-iteration-1` from `feat/w34-kernel-fusion-impl @ 70d2cf5e`. Worktree at `.worktrees/w34-closure-proposal`.

* **S33.2 — Closure proposal document.** Write `docs/plans/2026-05-13-w34-closure-proposal.md` with sections:
  * **§1 Status & predecessor chain** — G30 (Path C bundle expansion, on main `@7bb56e4d`) + G31 (spike `@0276fd8d`, unmerged) + G32 (impl `@70d2cf5e`, unmerged) + this closure proposal (`@<HEAD>`).
  * **§2 Acceptance evidence** —
    * Perf gate: G32 superhub-50K bench 1.590× (table with baseline median + routed median + paired delta + CI + ratio + verdict).
    * Auto-disable gate: G32 superhub-1K bench 1.305× routed-to-unfused (table with routing verification via counters).
    * Cert grid: 5/5 PASS (A above-threshold→fused · B below-threshold→unfused · C1 env→u32::MAX forces unfused · C2 env→0 forces fused · D/E default=4096 + W2.5 prior).
    * Prior W3.x regression sweep: W2.5 7/0 + W3.1 9/0 + W3.2 K5 3/0 + W3.2 K6 3/0 + W2.4 3/0 + W4 recursive 8/0 all PASS.
  * **§3 Implementation summary** — file-by-file diff summary (the LOC table from G32's evidence README) + threshold value `4096` + env override `XLOG_WCOJ_W34_THRESHOLD` + production fork scope (canonical 4-byte triangle WCOJ + default `var_order` only; U64/rotated-leader/4-cycle/K5/K6 untouched).
  * **§4 Scope discipline** — what W3.4 closes vs what stays open. Closing W3.4 does NOT close W3.3 (still OPEN under ≥2.0× gate per Path C); does NOT promise fused-path on non-triangle shapes; does NOT promise fused-path under non-default `var_order`. Future extensions belong to W3.5/W3.6/G37 full-bundle integration.
  * **§5 Final gates** — fmt + warnings build + CUDA cert 1/1 + W3.4 certs 5/5 + bench compile, all EXIT 0 (captured from G32 evidence README + re-run on proposal branch HEAD).
  * **§6 Branch state** — `git rev-list --count main..HEAD = 2` (G32 commit + this closure-proposal commit); branch unmerged to main; no tag; no origin push.
  * **§7 Response options** —
    * **Response 1 — Accept as DONE** (recommended): FF-merge `feat/w34-closure-proposal-iteration-1` to main; W3.4 marked DONE on main; tally on main becomes 14 DONE + 1 IN-PROGRESS + 11 OPEN = 26. G31 + G32 commits land on main with G33 closure commit.
    * **Response 2 — Reject closure** (W3.4 stays OPEN; specify revised acceptance or evidence requirements).
    * **Response 3 — Defer closure** (W3.4 stays OPEN; closure deferred to a later iteration with stated reason).

* **S33.3 — Staged board edit.** Modify `docs/v065-closure-board.md`:
  * W3.4 row: status `OPEN` → `DONE`; closure date `2026-05-13`; notes append:
    ```
    DONE — Path C G31 spike + G32 impl + this closure proposal. G31 `bench-spike/w34-kernel-fusion @ 0276fd8d` (1.491× spike on superhub-50K). G32 `feat/w34-kernel-fusion-impl @ 70d2cf5e` (production threshold dispatch, layout+count fusion, threshold=4096 with `XLOG_WCOJ_W34_THRESHOLD` env override, cert grid 5/5, prior W3.x regressions 0). Final V3: superhub-50K 1.590× (≥1.3×); superhub-1K 1.305× routed-to-unfused (auto-disable proven). Scope: canonical 4-byte triangle WCOJ default `var_order` path only; U64/rotated/4-cycle/K5/K6 untouched. User-approved DONE in thread.
    ```
  * Status Tally update:
    ```
    | DONE | 14 (W2.4, W2.2, W2.1, W2.3, W2.5, W2.6, W3.1, W3.2, W3.4, W4.1, W4.2, W4.3, W5.1, W5.2) |
    | IN-PROGRESS | 1 (W1.1) |
    | BLOCKED | 0 (—) |
    | OPEN | 11 (W3.3, W3.5, W3.6, W3.7, W3.8, W3.9, W5.3, W5.4, W6.1, W6.2, W7.1) |
    | **Total** | **26** |
    ```
  * Header line: "24 open items required for tag" → "23 open items required for tag".

* **S33.4 — Final gates BEFORE the G33 commit.** Run on the closure-proposal branch HEAD (which has G32's full impl plus the closure-proposal additions):
  * `cargo fmt --check --all` EXIT 0.
  * `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` EXIT 0.
  * `cargo test -p xlog-cuda-tests --test certification_suite --release` 1/1.
  * `cargo test -p xlog-cuda-tests --test test_wcoj_w34_fusion --release` 5/5.
  * Prior W3.x regression sweep: identical to G32 §"Prior Regressions" section — same commands, all EXIT 0.

* **S33.5** Branch UNMERGED to main + G30. (Branch IS reachable from G32 by construction since it's branched from G32 — that's expected and required for FF-merge.) No FF-merge to main, no push, no tag.

* **S33.6 — Forbidden behaviors.**
  * No `git push`, no `git tag`, no `--force`, no `--no-verify`, no `--dangerously-bypass`.
  * **No FF-merge into main.** Main remains at `7bb56e4d` until user approves Response 1 in thread.
  * No `v0.6.6` references in any new file, commit message, or board text (per W1.1 rule #5; W3.4 closure does NOT clear the v0.6.6 lockout — only zero-OPEN does).
  * No modification to G32's impl commit (don't amend; don't rebase). G33 adds exactly TWO new things: the closure proposal doc + the staged board edit. The G32 impl commit is preserved bit-identical.
  * No `RuntimeConfig`/`CompilerConfig`/`CostModel` change (G33 is doc-only on top of G32 impl).
  * No W3.4 mark-DONE on main until user approves.
  * **No production code change in G33's diff.** Verified by: `git diff feat/w34-kernel-fusion-impl..HEAD -- crates/` MUST be byte-empty (G33 only adds the proposal doc + board edit, on top of G32's already-staged impl commit).

* **S33.7** Single bundled commit subject `docs(w34): closure proposal — W3.4 OPEN→DONE staged for user approval (1.590× / cert grid 5/5)`.

### Questions

* **Q33.1** Branch HEAD SHA + commits-on-branch count (`git rev-list --count main..HEAD`, should be 2)?
* **Q33.2** Closure proposal doc path + all 7 sections present?
* **Q33.3** Board edit: W3.4 row status `OPEN→DONE`, notes appended, tally updated, header updated?
* **Q33.4** Final gates (fmt + warnings + CUDA cert + W3.4 certs 5/5 + prior W3.x regressions) all EXIT 0 on proposal HEAD?
* **Q33.5** No production code change in G33's own commit (`git diff feat/w34-kernel-fusion-impl..HEAD -- crates/` byte-empty)?
* **Q33.6** Branch unmerged from main (`git merge-base --is-ancestor HEAD main` returns false)?
* **Q33.7** No tag, no origin push?

### Metrics

* **M33.1** `feat/w34-closure-proposal-iteration-1` exists; HEAD reachable from G32 but NOT from main.
* **M33.2** `git rev-list --count main..HEAD` = 2.
* **M33.3** Closure proposal doc exists with 7 sections + Response options 1/2/3.
* **M33.4** Board edit staged: W3.4 row OPEN→DONE; tally 14 DONE + 1 IN-PROGRESS + 0 BLOCKED + 11 OPEN = 26; header "23 open items required for tag".
* **M33.5** `git diff feat/w34-kernel-fusion-impl..HEAD -- crates/` byte-empty.
* **M33.6** Final gates all EXIT 0 captured in proposal doc §5.
* **M33.7** No tag at HEAD; no `feat/w34*` remote ref.
* **M33.8** Branch unmerged from main.

### Supervisor validation per locked protocol

* Read closure proposal doc end-to-end.
* `git rev-parse <branch>` ≠ `main`.
* `git merge-base --is-ancestor <branch> main` returns false.
* `git rev-list --count main..HEAD` returns 2.
* Verify M33.5 no-crates-diff in G33's own commit.
* Inspect board edit: W3.4 row + tally + header all updated consistently.
* Re-run final gates from supervisor session.
* Verify branch unmerged + no tag + no origin push.

**If all gates green:** present G33 to the user via AskUserQuestion with Response options 1 (Accept DONE + FF-merge) / 2 (Reject) / 3 (Defer / Amend).

### Why this is the closure path

W3.4 has two acceptance gates (perf + auto-disable). G31 proved perf in isolation (spike). G32 built the production threshold dispatch + cert grid satisfying BOTH gates. G33 packages the evidence + stages the board flip for user approval. The closure is paper-aligned (layout+count fusion is structurally analogous to paper §4 pipeline shape), production-grounded (threshold-routed live dispatch, env-overridable), and contract-pinned (5-cert grid enforces routing + correctness + env override + W3.x prior preservation).

Proceed: cut closure-proposal branch from G32, write closure proposal doc with Response options 1/2/3, stage board edit (W3.4 OPEN→DONE + tally + header), run final gates, single bundled commit on top of G32. Emit REVIEW REQUEST with HEAD SHA + commit-count + scope-checks + final-gate results + recommendation.
