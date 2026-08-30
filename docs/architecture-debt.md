# Architecture debt ledger — XLOG

Audit 1, 2026-08-10, in two waves: cross-crate structure, then the whitepaper and the three
hotspots. Diagnose-only — no code was modified by the audit itself.

## Audit frame — read this before using any entry

The audit ran against the **working tree**: branch `whitepaper-source` @ `9516a1c1` (2026-07-10,
v0.10.0). That is **73 commits behind** `origin/main` @ `a9c2ed17` (2026-08-07, v0.12.0), with 37
commits of its own not upstream. Every entry carries an `on origin/main:` line. Ten entries were
re-checked against `origin/main`; the compact rows were not.

One finding changed materially between the audited tree and `origin/main` — see SD-001.

**Scope of wave 1:** cross-crate structure only. Six dimensions were probed (crate graph and
layering; duplication across crates; the Rust-Python boundary; resource/state/lifecycle ownership;
contract drift; gates/config/release). Intra-crate detail was out of scope by design.
Excluded from all measurement: `examples/neural/baseline/**` (89,943 lines of vendored DeepProbLog
and Scallop), `target/`, `.worktrees/`, `docs/whitepaper/artifacts/**`.

## Trend row

| date | scope line | del/add (window) | open | accepted | fixed since last | new | worse |
|---|---|---|---|---|---|---|---|
| 2026-08-10 (audit 1, wave 1) | all tracked source \| ext=rs,py \| since=1 year ago \| vendored excluded | 0.174 | 18 | 0 | — | 18 | — |

Monthly shape at that scope: 2026-02 `0.02`, 03 `0.67`, 04 `0.06`, 05 `0.15`, 06 `0.18`,
07 `3.59` (net −7,461 lines). Punctuated, not flat: campaign months exist, so pruning is somebody's
job here at least occasionally. The level is not a grade; only direction at an identical scope is.

---

# Entries

### SD-001 — Rust tests never run on a pull request; they run only after merge, on a runner nobody can see
status: open   severity: high   first-seen: 2026-08-10   confidence: confirmed
on origin/main: **partly fixed, still open**

Cost:      196k lines of Rust tests, including `test_epistemic_gpu_wcoj_execution.rs` (30,474
           lines, 206 tests, 5,879 asserts, churn 200/yr, the #1 hotspot by churn x size), gate
           nothing before merge. A regression is caught after it is on `main`, by a runner whose
           existence cannot be confirmed from the repository, or by a consumer. This is the
           mechanism by which that file reached 30k lines.
Evidence:  audited tree: `cargo test` appears once in all eight workflows, at `cuda-ci.yml:13`,
           and `cuda-ci.yml` was `on: workflow_dispatch` only, set by `4b3d53f2` (2026-03-04,
           "ci: disable automatic workflow triggers"), which stripped triggers from bench,
           cuda-ci and fuzz in one commit. `pytest`: zero hits in `.github/` and `Makefile`.
           origin/main: `cuda-ci.yml` now triggers on `push: branches:[main]` with a paths filter,
           and `ci.yml:92-122` installs pytest and runs a provenance-contract test plus the CAVIAR
           example suites. But `ci.yml` still has no `cargo test` — a PR runs fmt, a three-rule
           clippy, a build without `--all-targets`, and those Python checks.
Verified:  confirmed. I read all eight workflow files on both refs and ran the greps myself.
Remedy:    one job in `ci.yml`: `cargo test --workspace --all-targets --exclude pyxlog --exclude
           xlog-cuda-tests` on a GPU-less runner (device-requiring tests already self-skip via
           `require_cuda_guard.rs`). 1 session to wire, 1-2 to triage what goes red.
Leave it:  post-merge detection stays the norm, and the cost of touching the hot files stays
           "until someone runs it by hand on a GPU box".

### SD-002 — a benchmark whose subject is multiway joins is built the one way that disables them
status: open   severity: high   first-seen: 2026-08-10   confidence: confirmed (construction) / suspected (numeric effect)
on origin/main: alive — `logic_bench.rs` still calls `GpuMemoryManager::new`; still 14 copies

Cost:      `docs/BENCHMARKS.md:64` documents `crates/xlog-gpu/benches/logic_bench.rs` as an
           official benchmark category; the file contains `bench_multiway_join`. Two of the 14
           hand-copied `make_provider` bodies omit `XlogDeviceRuntime`; this is one of them. The
           CLI copy documents what that means: "The plain `GpuMemoryManager::new` path leaves
           `memory().runtime() == None`, so those dispatches silently fall back to binary joins."
           Numbers from this file are not comparable with the `xlog-integration` benches, which
           all build the provider correctly.
Evidence:  `crates/xlog-cli/src/main.rs:1450-1456` (comment and correct construction);
           `crates/xlog-gpu/benches/logic_bench.rs:25-37` (the incorrect one);
           `crates/xlog-runtime/src/executor/wcoj_dispatch.rs:1048` ("Without a runtime-backed
           manager, the recorded WCOJ primitives can't run — fall back silently", returns
           `Ok(None)`); 14 copies enumerated, exactly two with zero `XlogDeviceRuntime`
           references, the other being `crates/xlog-cuda/src/provider/ilp_exact.rs`.
Verified:  I read the CLI comment, the bench and the dispatcher line, and enumerated all 14
           copies. NOT verified: that `bench_multiway_join` actually reaches the recorded-
           primitive dispatcher. That needs reading `xlog_gpu::logic::LogicProgram` end to end,
           and nothing was built or run. If it does, published numbers are affected.
Remedy:    replace two `GpuMemoryManager::new` calls with `with_runtime`, which already rejects a
           runtime-less manager — 2 lines. Then export the canonical builder
           (`crates/pyxlog/src/lib.rs:212 provider_from_config`) as `xlog_cuda::build_provider`
           and collapse the copies. 1 session for the first, 2 for the second.
Leave it:  every new bench, test or example is a coin flip between two execution modes
           distinguishable only by the numbers they produce.

### SD-003 — the GPU silently returns an under-computed probability where the CPU raises
status: open   severity: high   first-seen: 2026-08-10   confidence: confirmed
on origin/main: alive

Cost:      same monotone SCC fixpoint under Monte Carlo. CPU: `MAX_ITERS = 1024` and a hard
           `XlogError::Execution("Monotone SCC fixpoint iteration limit (1024) exceeded")`. GPU:
           the kernel writes `converged_flags[w] = converged` and then unconditionally counts
           queries and evidence from the non-converged state — there is no not-converged branch.
           The flag reaches the host as `resident_status_flags`; its only reader in the workspace
           is a test. A loud error on one path is a plausible number on the other, and the
           diagnostic sits next to it unread. Same for `sparse_overflow_flags`.
Evidence:  `crates/xlog-prob/src/mc/results.rs:220` and `:298` (CPU limit and error);
           `crates/xlog-cuda/kernels/mc_resident.cu:541` (flag written) and `:545-557` (queries
           counted from `cur` regardless); `crates/xlog-prob/src/mc/resident.rs:1108`;
           `crates/xlog-prob/tests/mc_resident.rs:104` (only reader).
Verified:  confirmed — I read the kernel region and grepped every reader of the flag myself.
Remedy:    read the flags on the host after the result copy and return the same
           `XlogError::Execution` the CPU path returns. One place, near `resident.rs:1108`.
           1 session.
Leave it:  the one branch where the GPU hands back a wrong probability instead of an error, in an
           engine whose numbers go into a paper.

### SD-004 — `XLOG_USE_RECORDED_OPS=false` turns the recorded path on
status: open   severity: high   first-seen: 2026-08-10   confidence: suspected
on origin/main: alive — the predicate is unchanged

Cost:      three incompatible boolean parsers decide which execution path runs.
           `CudaKernelProvider::env_flag` treats any non-empty value other than `"0"` as true, so
           `=false`, `=off` and `=no` all enable. The certification harness accepts only
           `"1"|"true"|"TRUE"|"True"`, where `tRue` disables. `wcoj_dispatch` uses a third rule.
           These variables select the execution path, so one binary behaves as several engines
           and a GPU bug report without a full environment snapshot is unusable. An external
           consumer reading the variable name will fall into this.
Evidence:  `crates/xlog-cuda/src/provider/mod.rs:1417-1421` (`!v.is_empty() && v != "0"`), gates at
           `:1435,1445,1453,1462,1473,1484,1493`;
           `crates/xlog-runtime/src/executor/wcoj_dispatch.rs:134-138`;
           `crates/xlog-cuda-tests/src/harness/provider.rs:109-116`.
Verified:  suspected — reported with line-level citations by the resource-ownership probe. I
           re-checked that the `v != "0"` predicate still exists on `origin/main`, but did not
           read all three parsers myself. Ten minutes of reading closes this.
Remedy:    one `xlog_core::env_bool(name) -> Option<bool>` with fixed semantics, read once into a
           `DispatchFlags` struct; delete the three local parsers. 1 session. It breaks existing
           CI strings such as `XLOG_USE_RECORDED_OPS=all`.
Leave it:  a trap with the sign inverted, in the variable most likely to be reached for during an
           incident.

### SD-005 — the CPU oracle for GPU certification has no callers, and the warning was silenced by widening the API
status: open   severity: high   first-seen: 2026-08-10   confidence: confirmed
on origin/main: alive

Cost:      `crates/xlog-cuda-tests/src/harness/validators.rs` is 705 lines headed "CPU reference
           implementations for validating GPU results": `hash_join_u32`, `semi_join_u32`,
           `radix_sort_u32`, the groupby family, scans, set operations, plus ULP and permutation
           assertions. 27 of its 33 `pub fn` have no consumer; the only occurrence of the word
           `validators` in the crate is the module declaration. The 25-category GPU certification
           suite therefore never compares a kernel against a reference. The history is the
           finding: `a17bf06e` (2026-03-28) made these `pub(crate)`; `fc2693fc` (2026-04-22,
           "Tighten workspace warning hygiene") reverted exactly that — 70 lines, `pub(crate) fn`
           to `pub fn` — because `pub(crate)` on an uncalled item produces `dead_code` and `pub`
           in a lib crate does not. The signal that the oracle was dead was removed by making the
           API wider.
Evidence:  `crates/xlog-cuda-tests/src/harness/validators.rs:1`;
           `crates/xlog-cuda-tests/src/harness/mod.rs:6` (sole mention);
           `git show fc2693fc -- .../validators.rs` gives 70 visibility-only changes;
           `git grep 'validators::' -- crates/` gives 0.
Verified:  confirmed — I ran the history check and the consumer grep myself.
Remedy:    decide: wire `reference::` into the 3-4 categories where the oracle is cheap
           (join, sort, scan, `c22_algorithms`), or delete the file. 0.5 session to decide,
           1 to wire.
Leave it:  705 lines that read as insurance and are not. Worse than absence: a reviewer sees "CPU
           reference implementations" in the tree and believes it.

### SD-006 — CPU and GPU disagree on float edge cases, and the GPU contradicts itself
status: open   severity: high   first-seen: 2026-08-10   confidence: suspected
on origin/main: not re-checked at source

Cost:      comparison in a rule body: CPU is IEEE (`a < b`), the GPU filter kernel uses a total
           order (`-NaN < -Inf < ... < -0.0 < +0.0 < ... < +Inf < +NaN`). So `X > 1.0` with
           `X = NaN` drops the row on CPU and keeps it on GPU; `X < 0.0` with `X = -0.0` is false
           on CPU and true on GPU. The GPU also disagrees with itself: `OP_EQ` and `OP_NE` stayed
           IEEE, so on GPU `-0.0 == 0.0` and `-0.0 < 0.0` hold simultaneously. For `logsumexp`,
           the CPU catches NaN and errors while the GPU has no guard, and `if (val <= old_val)
           break;` is false for NaN, so NaN wins the atomic max and becomes the group's answer. On
           two `+Inf` the CPU returns NaN and the GPU returns `+Inf`. This surfaces only on data
           containing NaN, +-Inf or -0.0 — that is, not on test fixtures, on somebody else's data.
Evidence:  `crates/xlog-prob/src/provenance.rs:2074-2080` against
           `crates/xlog-cuda/kernels/filter.cu:137-140` (with `filter.cu:37` documenting the total
           order and `:135-136` the IEEE remnant);
           `crates/xlog-prob/src/aggregates.rs:127,141` against
           `crates/xlog-cuda/kernels/groupby.cu:236,287`.
Verified:  suspected — the duplication probe read all four files and cited line-level; I did not
           open the kernels. Nothing was executed, so no divergence was reproduced.
Remedy:    write down which semantics is normative first (IEEE is already used in two places of
           three), then align the GPU and add one parameterised test over `{NaN, +-Inf, +-0.0}`.
           1 session to decide, 1-2 for the test.
Leave it:  the divergence exists and is undocumented; every new aggregate copies the asymmetry.

### SD-007 — `cudarc` is the de-facto sixteenth crate: eight crates reach the raw driver directly, past a seam built for exactly this
status: open   severity: med   first-seen: 2026-08-10   confidence: confirmed
on origin/main: alive — `cudarc` in 8 crate manifests

Cost:      `xlog-cuda` is documented as the single point of contact with CUDA. `cuda_compat.rs`
           exists and re-exports precisely the leaking types (`LaunchConfig`, `DeviceSlice`,
           `DeviceRepr`, `CudaStream`, `sys`); four places use it and 41 files bypass it. The
           incident already happened: `b93e797a` (the CUDA 13 migration) created `cuda_compat.rs`
           and in the same commit edited 20 files outside `xlog-cuda` — 14 in `xlog-prob`, plus
           runtime, solve and cuda-tests. The design doc for that commit predicted it: "cudarc API
           drift ... may require code changes outside simple manifest edits". The cure was built
           and not applied. The next driver bump repeats those 20 files.
Evidence:  `crates/xlog-cuda/src/cuda_compat.rs:5-8`;
           `crates/xlog-prob/src/compilation/gpu_cache.rs:5` (direct `use cudarc::driver::...` for
           two types the seam already exports); `git show --name-only b93e797a`.
Verified:  confirmed for the manifest count on `origin/main`, which I ran. The seam-bypass counts
           are the layering probe's, cited with commands.
Remedy:    mechanical: `use cudarc::driver::X` to `use xlog_cuda::X` in about 25 files, drop
           `cudarc` from seven manifests (dev-only in two), re-export the few missing symbols.
           1 session — the compiler points at every site.
Leave it:  the next driver migration is again about 20 files across four crates instead of one.

### SD-008 — `KernelProvider`, the documented backend abstraction, has no implementation outside a test mock
status: open   severity: med   first-seen: 2026-08-10   confidence: confirmed
on origin/main: alive — zero `impl KernelProvider` outside `xlog-core`

Cost:      `docs/ARCHITECTURE.md` presents `xlog-core` as holding the traits, and the trait carries
           "This abstraction allows swapping CUDA for other backends (HIP, SYCL)". The only
           implementation is `MockProvider` inside `#[cfg(test)]`. The real interface between
           layers is the concrete `xlog_cuda::CudaKernelProvider` with 218 `pub fn` across 11
           `impl` blocks in 9 files. The graph has no seam where a CPU stub could be substituted,
           which is why logic semantics can only be checked on a GPU runner and why a scheduler
           regression test cannot be written without hardware. `traits.rs` has exactly one commit,
           2026-02-04, while 310k lines grew around it.
Evidence:  `crates/xlog-core/src/traits.rs:36-39` and `:86`; `git log -- .../traits.rs` gives one
           commit; `git grep KernelProvider | grep -v CudaKernelProvider` gives 3 hits, all inside
           `xlog-core`.
Verified:  I confirmed the zero-implementation count on `origin/main` myself. The 218-method count
           is the probe's, with its command.
Remedy:    do not extract an interface, that is months. Stop the map from lying: delete
           `KernelProvider`, `RelationStore` and `GpuBuffer` from `xlog-core` (129 of the 133 lines
           of `traits.rs`) and fix the ARCHITECTURE line. 0.5 session. Then "do we want a backend
           seam" becomes an explicit choice instead of something that looks already done.
Leave it:  cheap in code, expensive in people: every new reader sees a pluggable backend that does
           not exist. Also `RelationStore` in `xlog-core` collides by name with the real
           `xlog_runtime::RelationStore`.

### SD-009 — the files that tell agents how to work are untracked on `main`, and the two copies have disjoint rules
status: open   severity: med   first-seen: 2026-08-10   confidence: confirmed
on origin/main: alive — both files are in `.gitignore` and absent from the tree

Cost:      95% of this codebase was written by agents following `AGENTS.md` and `CLAUDE.md`.
           Commit `8b66ce21` ("chore: untrack internal agent-instruction files") added both to
           `.gitignore`. A fresh clone of `main` gives an agent no rules at all; edits to the rules
           pass no review, appear in no PR, and drift silently between machines. For a repository
           where the agent instruction is the build process, this is the artifact that most needs
           version control. Worse, the two copies that ought to be one are disjoint: `AGENTS.md`
           has "Source Clarity and Artifact Hygiene" and no "Commit and Release Rules";
           `CLAUDE.md` has the reverse. So the Codex-side agent has never read "Never run
           `git tag`", "Never edit `[workspace.package].version`", "Never hand-edit
           `CHANGELOG.md`" — the most dangerous rules in the file.
Evidence:  `origin/main:.gitignore` (both entries); `git cat-file -e origin/main:AGENTS.md` fails;
           local `AGENTS.md:16-24` against `CLAUDE.md:51-91`.
Verified:  confirmed — I checked the ignore entries and the absence of the blob on `origin/main`.
Remedy:    put one file back under git, make the second a stub that references it, drop the two
           `.gitignore` lines. The sections merge mechanically; there is no content conflict, only
           omissions. 1 session.
Leave it:  rules are tightened locally and lost at the next clone, and the release prohibitions
           stay invisible to exactly the agent most likely to break them.

### SD-010 — a documented emergency kill switch does not exist in the code
status: open   severity: med   first-seen: 2026-08-10   confidence: confirmed
on origin/main: alive — 0 hits in `crates/*/src`, 2 doc pages still describe it

Cost:      `XLOG_DISABLE_WCOJ_TRIANGLE` is documented twice as a "hard kill switch — pins all
           triangle WCOJ off; beats every other triangle flag". The dispatcher reads only
           `XLOG_USE_WCOJ_TRIANGLE_U32`; the three neighbouring switches (groupby fusion, free
           join, factorized delta) do have resolvers. Anyone who sets it during an incident to
           move a hot query off the WCOJ path gets a silent no-op. It has already cost a live GPU
           run: a head-to-head against Soufflé on a rented 4090 was killed at 529 s against 11.7 s
           without being able to say whether WCOJ was on. Symmetrically, the page the README calls
           "the complete flag reference" omits `--wcoj`, the switch that actually enables it. The
           non-existent lever is documented twice, the real one not at all.
Evidence:  `docs/guides/wcoj-tuning.mdx:138`, `docs/guides/benchmarking.mdx:257`;
           `crates/xlog-runtime/src/executor/wcoj_dispatch.rs:97-107`;
           `git grep XLOG_DISABLE_WCOJ_TRIANGLE origin/main -- crates/` gives 3 hits, all doc
           comments in tests; `docs/superpowers/specs/2026-06-25-benchmark-evidence-tier1/
           T1.2-NOTE.md:21-28`.
Verified:  confirmed on `origin/main` by grep (0 production hits, 2 doc files). I did not open the
           incident note.
Remedy:    either add the env resolver next to its three neighbours (about 6 lines, the pattern
           exists) or delete the row from both tables and keep only the builder. Plus one line for
           `--wcoj` in the CLI reference. 0.5 session.
Leave it:  every future WCOJ benchmark risks repeating that incident, and the emergency lever
           stays a no-op.

---

## Compact entries

Real, evidence-backed, below the recommended cut. Same ID space; promote a row when it becomes
actionable. All `suspected` unless noted — they come from dimension probes and I did not
personally re-read every site.

| ID | Title | Sev | Conf | Evidence anchor | Cost in one line |
|---|---|---|---|---|---|
| SD-011 | GPU circuit cache is keyed on CNF bytes alone while the disk cache is keyed on (cnf, config, random_vars, sm); the table is built for reuse | high | suspected | `xlog-prob/src/compilation/gpu_cache.rs:219-256`, `compilation/mod.rs:244,271-283` | the first commit that makes the table long-lived — the stated warmup goal — returns a circuit compiled for different random variables: a different probability, no error |
| SD-012 | `#pragma prob_*` directives are read by the CLI and ignored by Python | high | suspected | `xlog-cli/src/main.rs:1634` against `pyxlog/src/program.rs:587,256` | the same file with `prob_samples 1000000` runs 1M samples from the CLI and 10k from Python, silently |
| SD-013 | `wcoj_paper_class`, the bench whose numbers are "paper-class", prints its VRAM gate without asserting it, and its peak accumulates across the whole run | high | suspected | `xlog-integration/benches/wcoj_paper_class.rs:91,424` against `skewed_multiway_bench.rs:79,135` | the published peak-memory number does not mean what is printed next to it; the fix went into the forked copy and never came back |
| SD-014 | a poisoned CUDA primary context is process-terminal and nothing observes it | high | suspected | `xlog-cuda/src/device.rs:459-470`, `device_runtime/runtime.rs:171`, commit `0e9b452d` | after one launch failure every later call in a long session fails with a different masking error, and there is no reset path |
| SD-015 | ten device-creation points across five crates; the designated single owner `XlogDeviceRuntime::try_get` has zero production callers; two incompatible allocation modes coexist on one card | high | suspected | `device_runtime/runtime.rs:49,127`, `xlog-prob/src/mc/mod.rs:824-837` | "OOM or illegal address after N queries" is unreproducible because the number of live providers depends on program shape; the P0 repro script toggles exactly these two modes |
| SD-016 | `_native.pyi` is the boundary's only contract and is checked by grepping its own text; 113 exported against 105 declared | med | confirmed | `python/tests/test_bridge_source.py:6-45` and 14 similar files; `forward_backward_grouped` and `register_domain_tensor_source` absent yet called from the package's own Python | mypy --strict gives false coverage on paths that have no types; the stub drifts about one method per release |
| SD-017 | the ILP path releases the GIL, the probabilistic and logic paths hold it; 5 `allow_threads` against 223 functions | med | confirmed | `pyxlog/src/ilp.rs:2382-2392` against `program.rs:542` (parameter deliberately named `_py`), `logic.rs:140,252` | any second Python thread blocks for the whole GPU fixpoint and a 10k-sample MC run; the correct pattern is already in the same crate, so the next entry point is a coin flip |
| SD-018 | a second `.xlog` dialect (`trainable_rule`, `train`) parsed by a hand-written Python lexer that must stay bug-compatible with the pest grammar | med | suspected | `pyxlog/python/pyxlog/ilp/neurosymbolic.py:1350-1444` against `xlog-logic/src/grammar.pest:4,26` | any grammar change silently breaks Python statement splitting; already divergent, since Python treats `'` as a quote and the grammar has only `"`; zero of 322 `.xlog` files exercise the dialect |

---

## Not debt — investigated and cleared

These make the next audit cheaper. No `SD-` ids; not to be confused with `accepted`.

- **ND-001** — `DiscardSink` looked like 11 copies in census section 7. It is 79, and all are
  identical no-ops; the only difference is `std::result::Result` spelling in 4 of 79. Not
  duplication.
- **ND-002** — `crates/xlog-cuda/src/wcoj_metadata.rs` against `provider/wcoj_metadata.rs`, 86%
  co-change, looked like a forked file. It is not: 143 lines of types against 5,200 lines of
  `impl`. The co-change is explained by "a new kernel needs both a type and an implementation".
- **ND-003** — the three kernel-list enumerations (`KERNEL_CU_NAMES`, `KernelModuleSpec.module_name`,
  25 `*_MODULE` constants) are byte-identical today and guarded by
  `const _: () = assert!(... == 25)` at `provider/mod.rs:439`. Only defect: a stale comment
  "All 24 modules listed" at `kernel_manifest_data.rs:10`. One line.
- **ND-004** — 31 `#[ignore]` tests: almost all carry an explicit reason ("measurement, run
  explicitly", "run on RunPod, never locally"), and one deliberately documents intended unsafe
  behaviour ("corruption is the intended outcome here"). Good engineering. Eight bare `#[ignore]`
  without a reason remain, which is minor.
- **ND-005** — panic through the FFI boundary is not UB here: `panic = "abort"` is not set and
  pyo3 0.24.2 converts unwinds in `#[pymethods]` into `PanicException`. Capsule destructors and
  `Drop for DlpackManagedTensor` are null-safe, and `TrackedCudaSlice` holds an `Arc` to the
  memory manager, so there is no leak or use-after-free on that path.
- **ND-006** — no background work exists in the engine (no `thread::spawn`, tokio or rayon in
  production code), so "who observes async failure" is currently vacuous. Recorded because the
  answer becomes "nobody" the moment the first background task appears.
- **Dimension 11 (client and presentation)** — not applicable beyond `docs-site`; not probed.
- **Dimension 12 (test-suite structure)** — not assigned to a probe in wave 1; partially covered
  by hand, see the coverage statement.

---

## Coverage statement — what wave 1 did not do

- **Frame.** Everything was measured on a tree 73 commits and two minor versions behind
  `origin/main`. Ten entries were re-checked against `origin/main` mechanically; SD-006 and the
  compact rows were not re-read at source there.
- **Nothing was built or executed.** No `cargo build/test/clippy`, no `pytest`, no `maturin`, no
  GPU. Every claim is read from source. No divergence in this ledger has been reproduced.
- **Intra-crate depth was out of scope by design.** In particular
  `crates/xlog-cuda/src/provider/relational.rs` (12,699 lines, churn 44),
  `crates/xlog-runtime/src/executor/epistemic_workspace.rs` (churn 75) and `wcoj_dispatch.rs`
  (churn 67) — the top hotspots — were read only around specific findings.
  `crates/xlog-integration/tests/test_epistemic_gpu_wcoj_execution.rs` (30,474 lines, the #1
  hotspot) was not opened by anyone.
- **Two of twelve dimensions were not probed:** client/presentation, and test-suite structure.
- **Kernels** (`kernels/*.cu`) were read only where a finding pointed at them.
- **GitHub Actions run history is not in the repository.** Whether a self-hosted CUDA runner
  exists, and whether branch protection makes any check required, could not be determined. If no
  required checks are configured, SD-001 is worse than stated.
- **Whitepaper claims about capability** (epistemic semantics, GPU CDCL equivalence certification,
  zero-host-transfer MC) were not checked against the code. That is the most expensive remaining
  dimension and the one most likely to matter.
- **Not read at all:** `xlog-induce`, `xlog-neural`, `xlog-stats` internals; the contents of
  `python/tests`; `fuzz/`, `tools/`, `build.rs`; and 20-plus active feature branches.

---

# Wave 2 — whitepaper claims, and inside the three hotspots

Audited **`origin/main` @ a9c2ed17** directly, not the stale checkout — except the whitepaper, which
exists only on `whitepaper-source`. Four probes: whitepaper claims against code; inside
`provider/relational.rs`; inside `executor/`; test-suite structure.

No code changed between waves, so the trend row is unchanged. Wave 2 adds 10 full and 11 compact
entries, and corrects one wave-1 entry.

### SD-019 — the headline cost table has no committed artifact, in either edition
status: open   severity: high   confidence: confirmed   reframed: 2026-08-10
NOTE: first written as "disagrees with its own artifact by 74x". That was wrong. The artifact I
compared against is `20260218T_postfix_v040alpha_batched` — v0.4.0-alpha, February, six minor
versions before the paper. Comparing a claim to a stale artifact produces a false finding of the
most damaging kind. The real finding is below and it is narrower.

Cost:      Section 8 opens with "Every quantitative claim is tied to an artifact", and Section 8.1
           states that all measurements were collected on an RTX PRO 3000. Table 2 — the only
           published absolute training cost, and the basis for the claim that the symbolic layer is
           nearly free once compiled — has no backing artifact in either edition. The live v0.12
           edition ships seven artifacts, none of which contains a training field. The only
           committed run on that hardware is v0.4.0-alpha from February, six minor versions older,
           and is not what the table reports. So the table cannot be checked by a reader, and the
           section states a standard it does not meet for its own headline number.
Evidence:  `docs/whitepaper/sections/08_evaluation.tex:7,46-62`;
           `examples/neural/results/track_a/20260218T_postfix_v040alpha_batched/01_minimal/seed_7/metrics.json`
           and the `seed_42` sibling on origin/main;
           `docs/whitepaper/artifacts/head-to-head/mnist_addition_vs_scallop.json`.
Verified:  confirmed — I read the table in both editions, listed the artifacts of the live edition,
           and confirmed none carries training fields. The v0.4.0-alpha vintage of the older run was
           pointed out by the author, not found by me.
Remedy:    commit the run the table came from, on the stated hardware and with the stated three
           seeds. If that run no longer exists, re-run it or re-attribute the table. Not a code
           change and not a number change — a missing artifact. 0.5 session plus the run.
Leave it:  the paper asserts artifact-backing as a property of itself, and a reviewer who checks the
           headline table finds nothing behind it.

### SD-020 — the WCOJ workload the paper describes is not the workload the harness measures
status: open   severity: high   confidence: confirmed

Cost:      this is contribution #2 of the paper and the only numeric support for the WCOJ speedup.
           The text promises four paper-scale fixtures with 4.19M input rows each, multi-rule
           recursive, "median over 10 runs with coefficient of variation at most 0.05". The harness
           sets SCALE to 1024, and 4,194,304 equals 1024 x 64 x 64 — the *output* triangle count,
           which the harness itself prints as `rows=`. The four fixtures are parameter variants of
           one generator and coincide at that scale, so the 26.6-29.6x spread is repeat noise on a
           single input rather than stability across workloads. There is no recursion and no
           multi-rule evaluation. And `summarize` returns the mean, not the median; the coefficient
           of variation is computed and never asserted, as are the geomean gate and a 38 GiB VRAM
           gate on a 12 GB device. The figure's 27.96 is hardcoded in `figures/make_results.py`,
           citing `docs/BENCHMARKS.md`, which does not exist on origin/main.
Evidence:  `docs/whitepaper/sections/08_evaluation.tex:29,36`;
           `crates/xlog-integration/benches/wcoj_paper_class.rs:26,28,348-361,457`;
           `docs/whitepaper/figures/make_results.py:16-18`.
Verified:  confirmed for SCALE, for mean-not-median, for the VRAM gate constant and for the claim
           text, all of which I read. The 196,608-input-row arithmetic is the probe's, derived from
           the fixture body; I did not re-derive it line by line.
Remedy:    paper edit: state the real shape — one triangle query over three 65,536-row relations
           producing 4.19M output triples — drop "multi-rule recursive", say "mean", and either
           assert the coefficient of variation in the harness or stop claiming it. 1 session.
Leave it:  not an option; it is a claimed contribution.

### SD-021 — a performance decision becomes a hard correctness error in another subsystem, at default config
status: open   severity: high   confidence: confirmed

Cost:      `summarize_runtime_routes` statically predicts which WCOJ routes will fire, turns the
           prediction into an obligation, and compares it against the real counters after the run; a
           mismatch raises `MissingRequiredWcojDispatch` on the hot path. But 4-cycle dispatch is off
           by default on both axes, so an epistemic program containing a canonical 4-cycle fails
           certification at a fully default configuration. Triangle has the same shape: the cost
           model returns false whenever any of three slots lacks cardinality statistics, and base
           relation scans do not populate `StatsManager`. And `CostModelKind::SkewClassifier` — a
           legal value of `XLOG_WCOJ_COST_MODEL` — hardcodes both predicates to false, so selecting
           it guarantees the error. Free Join is handled correctly: its route count is subtracted
           with an explicit note that the route is opportunistic. The problem was solved for one
           route and not generalised.
Evidence:  `crates/xlog-runtime/src/executor/epistemic_workspace.rs:6607-6616,3259-3271,2282-2298`;
           `crates/xlog-runtime/src/executor/wcoj_dispatch.rs:2853-2859,305-312`;
           `crates/xlog-runtime/src/executor/wcoj_cost_model.rs:117-123,159-170`; the correct
           treatment at `epistemic_workspace.rs:3219-3236`.
Verified:  confirmed for the `SkewClassifier` hardcoded false, which I read. The rest is the probe's,
           cited line by line. Not verified: that this has fired on a live program — no fixture with
           a canonical 4-cycle inside an epistemic reduction was run.
Remedy:    apply the Free Join treatment — obligate only routes the promoter marked as required
           (`MultiwayPlan::WcojWithPlan`), not routes that merely matched by shape. 0.5 session.
Leave it:  the only finding in either wave with a plausible user-visible failure at default config.

### SD-022 — the fail-closed epistemic check guards a counter nobody increments, and its output is published as evidence
status: open   severity: high   confidence: confirmed
supersedes: wave 1 recorded this mechanism as a model of correct discipline. That was wrong.

Cost:      `EpistemicCpuFallbackCounters` is constructed exactly once in production, through
           `::default()` with all fields zero, and none of its four fields is ever incremented
           anywhere in `crates/*/src`. So `cpu_fallbacks.is_zero()` is vacuously true, and the
           contract "no CPU fallbacks on the accepted hot path" is enforced by nothing. Worse than
           absent: the result is serialised outward as `"cpu_fallback_is_zero": true` and asserted by
           a test, so a check that cannot fail is consumed downstream as evidence that it passed.
Evidence:  `crates/xlog-ir/src/epistemic_plan.rs:55-75,337`;
           `crates/xlog-runtime/src/executor/epistemic_workspace.rs:2040-2046`;
           `crates/xlog-gpu/src/logic.rs:4024`; `crates/xlog-gpu/tests/logic_runner.rs:1108`.
Verified:  confirmed — I grepped every write to the four fields across production code. Zero `+=`;
           the four apparent matches are a format string and two local bindings.
Remedy:    make the silence visible to the compiler: drop `pub` on the fields, add `record_*`
           methods, and let dead-code analysis report that nothing calls them. 0.5 session. Then
           decide whether real CPU fallbacks exist on that path at all.
Leave it:  no — this one manufactures false evidence rather than merely missing a signal.

### SD-023 — the split that lasted a week: the extracted file is now larger than the module it was cut from
status: open   severity: med   confidence: confirmed

Cost:      `provider/relational.rs` was created on 2026-03-10 by commit `e2254760`, which cut 4,512
           lines out of a roughly 12,000-line `provider/mod.rs` into six domain submodules. Five
           months later it is 12,934 lines — larger than the god module it was extracted from — and
           `mod.rs` has regrown from 3,748 to 4,491. Most of the growth landed on 28-29 April: 4,963
           to 10,967 lines in eleven commits across two days. There is no second split attempt in the
           file's history and no technical barrier to one: every function is a method on a single
           `impl`, so moving them changes no signature. The March commit proved that mechanically.
           This is Lehman's second law in a single file — the counter-force was applied once and
           never again.
Evidence:  `git show --stat e2254760`; file length recomputed at each of the 48 revisions,
           2026-03-10 at 4,512 and 2026-08-07 at 12,934.
Verified:  confirmed — I ran the stat and both endpoint measurements myself.
Remedy:    the seam is already marked by a section comment at line 6286. Move lines 6286-12934, the
           22 recorded and on-stream functions, into `provider/relational_recorded.rs`. No signature
           changes. About 1 session.
Leave it:  every feature slice adds 200-700 lines to the same file and the same `impl`; merge
           conflicts between parallel join work are guaranteed.

### SD-024 — the flagship test runs under no automatic gate, and 204 of its 206 tests pass without a GPU
status: open   severity: high   confidence: confirmed

Cost:      `test_epistemic_gpu_wcoj_execution.rs`, 30,474 lines with 206 tests and 5,879 assertions,
           is run by no pull-request gate, no push gate, and not by the release script. In
           `cuda-ci.yml` the `rust-tests` job is gated on `workflow_dispatch`, while the neighbouring
           `python-wheel` job did receive a `push` trigger in commit `a5ebe90a` on 2026-08-05 — a
           deliberate asymmetry. Clippy still compiles the file on every pull request, so it costs
           compile time and returns no runtime signal. Separately, every test opens with a fixture
           probe that returns early when CUDA is unavailable, so on a machine without a GPU the file
           reports 206 green tests and zero checks. The guard against exactly this exists —
           `require_cuda_guard.rs`, whose doc block describes the hazard — but `XLOG_REQUIRE_CUDA` is
           exported only by `validate_release_gpu.sh`, which does not run `xlog-integration` at all.
           Repo-wide, 1,039 of 2,204 integration tests self-skip, and `xlog-runtime` with 100 such
           tests and `xlog-prob` with 60 have no guard file whatsoever.
Evidence:  `.github/workflows/cuda-ci.yml:36-39,61-64`; `.github/workflows/ci.yml`, which contains no
           `cargo test`; `scripts/validate_release_gpu.sh:98,111,132,134`;
           `crates/xlog-integration/tests/require_cuda_guard.rs:7-13`; 204 occurrences of the skip
           line in the test file.
Verified:  confirmed — I counted the 204 skip sites, confirmed the absence of `cargo test` from
           `ci.yml`, and confirmed that only one manifest mentions `xlog-cuda-tests`.
Remedy:    one line — add the push event to the `rust-tests` job condition — and copy
           `require_cuda_guard.rs` into `xlog-runtime/tests` and `xlog-prob/tests`, 13 lines each.
           0.5 session.
Leave it:  47% of the test suite protects nobody while creating the impression of protection.

### SD-025 — the shared test harness is unreachable, which is why the wave-1 CPU oracle is dead
status: open   severity: med   confidence: confirmed

Cost:      `xlog-cuda-tests` contains a harness — provider, validators, generators, diagnostics — and
           no other manifest in the repository depends on it. So 79 files independently declare
           `DiscardSink`, and 168 files hand-build the same chain from device through stream pool,
           resource, budget, runtime and memory manager to provider. Changing any link's signature is
           a 168-file edit. This is also the mechanism behind SD-005: the CPU reference oracle is not
           dead because someone disabled it, but because nothing can reach the crate that holds it.
Evidence:  `git grep -l xlog-cuda-tests -- '*/Cargo.toml'` returns one file, its own manifest;
           `crates/xlog-integration/tests/test_external_consumer_surface_preservation.rs:18-57`
           against `test_epistemic_gpu_wcoj_execution.rs:49-98`.
Verified:  confirmed — I ran the manifest grep myself.
Remedy:    add an `xlog-test-support` crate, or expose the existing harness as a dev-dependency, and
           migrate the two largest files as a pilot. 1 session for the pilot.
Leave it:  tolerable while the initialisation chain is stable; the first refactor of it turns this
           into a forced 168-file edit.

### SD-026 — roughly 155 silent declines and one counter, which counts the wrong thing
status: open   severity: high   confidence: suspected

Cost:      a comment in the executor names the incident directly: "rather than silently falling back
           to binary joins (the failure mode that wasted a full GPU benchmark cycle)". The fix that
           incident produced exports positive counters. There is no negative one:
           `wcoj_error_decline_count` counts kernel and layout errors only — its own doc string says
           structural declines stay silent — while gate-off, shape mismatch, missing buffer, key
           width mismatch, no runtime and cost-model-says-no all return `Ok(None)` without a trace.
           That is about 155 sites. The same benchmark cycle burns again if the cause is a shape
           mismatch: the counters show zero dispatches, zero declines and no explanation.
           `execution_stats` additionally exports 5 of 13 counters.
Evidence:  `crates/xlog-runtime/src/executor/wcoj_dispatch.rs:236-250`;
           `crates/xlog-runtime/src/executor/mod.rs:523-533,280-333`.
Verified:  suspected — the counts and the doc string are the probe's, cited line by line. I did not
           re-count the 155 sites.
Remedy:    one `decline(primitive, reason)` helper that returns `Ok(None)` and increments a
           per-primitive, per-reason table, plus its export in `ExecutionStats`. Replacing the returns
           is mechanical. 1 session.
Leave it:  no — the cheapest item on the list with the most direct link to an incident that already
           happened.

### SD-027 — an overflow flag written by five copies and read by nobody, so truncation is silent
status: open   severity: med   confidence: confirmed

Cost:      `d_overflow` is allocated, zeroed, registered on two launch recorders and passed to the
           kernel — 54 mentions in `relational.rs` — and never read back to the host anywhere in
           `xlog-cuda`. The comment promises a future helper to inspect it; that was 3.5 months ago.
           Meanwhile, when `max_output` is smaller than the real match count, the join returns a
           truncated result and the only signal of truncation dies in device memory. This is the
           same buffer whose lifetime cost the project PR #89.
Evidence:  `crates/xlog-cuda/src/provider/relational.rs:8148-8152` for the future-helper comment, and
           the flag's full lifecycle at `:7946,7953,7974,8027,8109,8169,8198`; the only host-side
           overflow read in the crate concerns a different buffer, at
           `provider/fj_delta_sparse.rs:446`.
Verified:  confirmed for the mention count of 54 and for there being exactly one host read in the
           crate, both of which I measured.
Remedy:    either add a `read_overflow_flag()` next to the existing metadata reader and return an
           error when it is set, or delete the flag from four host paths and from the kernel
           signature. The second is cheaper and removes the class. 1 session.
Leave it:  acceptable only if silent truncation at `max_output` is intended semantics — in which case
           it belongs in the rustdoc, not in a dead device flag.

### SD-028 — the flagship epistemic test never enters the multi-world regime it exists to test
status: open   severity: high   confidence: confirmed

Cost:      `max_worlds: 1` appears in all 149 occurrences in the file, with zero occurrences of any
           larger value; `max_candidates: 2` in 173 of 183. The modal operators `possible` and
           `not know` are only meaningful with more than one world view. So 206 tests confirm the
           single-world, two-candidate path and say nothing about aggregation across world views:
           priority between worlds, model deduplication, `max_models_per_reduction` overflow,
           reduction order. Multi-world cases exist, but in other crates and in far smaller volume.
Evidence:  `crates/xlog-integration/tests/test_epistemic_gpu_wcoj_execution.rs:812-816,17190-17194`;
           `crates/xlog-prob/tests/epistemic_prob_gpu_accepted_evidence.rs:2517,2916`;
           `crates/xlog-runtime/tests/test_epistemic_gpu_workspace.rs:154`.
Verified:  confirmed — I ran both counts, 149 and 0, myself.
Remedy:    add three to five tests at `max_worlds: 4` and `max_candidates: 16` on the existing
           fixtures, checked against the CPU oracle `run_generate_propagate_test` that the file
           already imports and already uses for 25 element-wise comparisons. 0.5 to 1 session.
Leave it:  no — this is a hole in precisely the subject the file names as its own.

## Compact entries — wave 2

| ID | Title | Sev | Conf | Evidence anchor | Cost in one line |
|---|---|---|---|---|---|
| SD-029 | the residency sweep drops the one cell that contradicts its conclusion | high | suspected | `artifacts/head-to-head/residency_scale_ablation.json`, cell handoffs=8 at −8.9%, against `08_evaluation.tex:96` | "a stable 40-56 us round-trip" holds only after removing a point where the forced round-trip was faster, and each point is a single measurement |
| SD-030 | "the solver never reads its status back to the host" — it does, twice per solve | med | suspected | `05_probabilistic.tex:39` against `xlog-solve/src/gpu_cdcl.rs:1577,195` | the device-side trap is additional, not instead, and the host read is explicitly exempted from the D2H counter the paper cites |
| SD-031 | the certified circuit is not the executed circuit | high | suspected | `00_abstract.tex:3` and `alg:verifiedkc` against `xlog-prob/src/compilation/mod.rs:411,430,460,469` | equivalence is proven on the pre-smoothing base circuit while forward and backward run on the post-smoothing one, so the smoothing pass is assumed rather than machine-checked |
| SD-032 | ~~section 7 claims device residency for all three modes; WFS is entirely host-side~~ **FIXED in the live v0.12 edition — the claim is gone** | — | confirmed fixed | `07_epistemic.tex:3` against `xlog-prob/src/wfs.rs`, 1,387 lines with no CUDA imports | a reviewer checking the residency claim finds one of three modes running on the host over grounded PIR |
| SD-033 | sections 5.3 and 8.5 attribute cold-start cost to opposite stages | med | suspected | `05_probabilistic.tex:63` against `08_evaluation.tex:94` and `artifacts/head-to-head/verify_overhead_isolation.json` | one says D4 dominates at about a minute, the other says D4 stays under 17 ms and CDCL is 96-100%, and the artifact reports `d4_compile_ms` of exactly 0.0 in five cells of six |
| SD-034 | "materialization exhausts the device" — the artifact shows a decline at 17% of budget | med | suspected | `00_abstract.tex:3` and `08_evaluation.tex:88` against `artifacts/head-to-head/triangle_counting_vs_souffle.json`, 3.23 GB against an 18.87 GB budget | the climax of the external comparison may be cumulative peak accounting rather than real exhaustion, and the run script is not committed |
| SD-035 | the group-by-root fusion matrix has holes and the empty cells materialize silently | high | suspected | `wcoj_dispatch.rs:3723,2277-2301` | a count group-by over a K7 clique returns `Ok(None)` and fully materializes the clique — the exact blow-up the fused path exists to prevent — with nothing in the log |
| SD-036 | two fixpoint engines, one blind to every WCOJ dispatcher and ignoring the configured iteration limit | med | suspected | `executor/recursive.rs:612-619,1016,1055-1059`; `node_dispatch.rs:327` | `RirNode::Fixpoint` always runs binary joins and caps at a hardcoded 1000, so `config.max_iterations` silently does not apply there |
| SD-037 | the join cache is honest; its background mode adds the full build cost and reports two counters equal by construction | med | suspected | `node_dispatch.rs:449-468,503-508`; `relational.rs:3718-3733` | the background build is synchronous, its result is discarded for that call, and the always-equal complete and deferred counters already participate in proving a speedup gate |
| SD-038 | 97% of the flagship test's assertions check telemetry rather than the answer | med | suspected | 162 of 5,618 assert sites compare a materialized result; 113 assert on error-message substrings | it cannot catch a wrong join answer with correct counters, and it freezes error wording into a contract |
| SD-039 | 71% of the flagship test file is duplicated preamble | med | confirmed | 194 of its 200 commits fall on two consecutive days, one per generated test | adding one axis costs a cartesian product of tests; a macro would remove about 18k lines without losing a single check |

## Not debt — added by wave 2

- **ND-007** — buffer registration on `LaunchRecorder` is currently complete in all 38 blocks. A probe
  wrote a scanner comparing registered buffers against kernel parameters per block and found 0 of 38
  violations. The PR #89 class is closed today, but closed by hand, with no gate. That scanner is the
  gate that does not exist; it is roughly 30 lines in CI.
- **ND-008** — no manual lifetime manipulation in `relational.rs`: zero `mem::forget`, `into_raw`,
  `ManuallyDrop`, explicit `drop(`, `Box::leak` or `transmute` across 201 allocations. Ownership is
  entirely RAII.
- **ND-009** — zero `TODO`, `FIXME`, `HACK` or `XXX` markers in the largest file in the project.
  Incident history is written as prose inside ordinary comments, so a marker search returns emptiness
  and creates a false impression of cleanliness. Do not use marker counts on this codebase.
- **ND-010** — the flagship test's CPU oracle is real: `run_generate_propagate_test` is an independent
  host implementation of generate, propagate and test, and 25 tests compare the GPU trace against it
  element by element. That is the best 12% of the file.
- **ND-011** — Rust tests do not grep source text; the Python-side disease of 15 files reading `.rs`
  and `.pyi` as text did not cross into the Rust suite. Its functional analogue did: 636 assertions
  repo-wide match on error-message substrings.
- **Confirmed whitepaper claims**, which is a result too: the 2.74x circuit-cache ablation with its
  confidence interval; the entire Scallop head-to-head across accuracy, epoch time and total; ProbLog2
  including the paper's honest refusal to claim speed; all three Soufflé sizes at 12.6, 23.9 and 42.5x
  together with the triangle counts; 1.1-1.5x on moderate skew. Verification really is on the
  production path and really is fail-closed, re-checking even a disk-cached circuit. The on-device
  resolution certificate exists. Total ordering in the filter kernel, k-clique 5 through 8, the FAEEL
  mode, the Monte Carlo megakernel and the 33 certification categories all check out.

## Coverage statement — wave 2

- Audited `origin/main` directly. The whitepaper was read from `whitepaper-source`, where it lives.
  Newer editions exist under `.claude/worktrees/` and, per internal notes, Table 2 is already
  corrected there while the 27.96x figure was restored byte-identical. **Verify against the edition
  that will be submitted, not this one.**
- Still nothing built or run. No divergence reproduced, no benchmark re-run, no GPU touched.
- Not covered: the promoter that constructs `MultiwayPlan` in `xlog-logic` and `xlog-ir`, which holds
  roughly half of the execution-path decisions; `executor/rewrite.rs`; `provider/wcoj_metadata.rs` at
  208 KB, the second-largest file in its directory; the CUDA kernels themselves; the second and third
  largest test files; the contents of `python/tests`; and whether the 206 flagship tests pass on a GPU.
- Section 8.7 of the paper, claiming 3.21x for the persistent index and 5.58x for the chain scorer,
  was not checked; an internal project note already records that the backing JSON is missing.


---

## Correction, 2026-08-10 — which edition the whitepaper findings apply to

Wave 2 audited `docs/whitepaper/` on `whitepaper-source`, stamped v0.10.0, last touched 2026-07-07.
The edition that ships is `.claude/worktrees/docs-realign-v0.12/paper/` on branch
`docs/realign-v0.12`, last touched 2026-08-09.

Re-checked against the live edition:

- **Section 8 is byte-identical between the two.** The paper grew from 580 to 705 lines overall, but
  the evaluation section — where every whitepaper finding lives — did not change at all. Nine of the
  ten claims survive word for word, including Table 2 unaltered.
- **SD-032 is fixed**: the phrase promising device residency for all three modes is gone.
- **SD-019 is reframed**, see the entry. The numbers are not shown to be wrong; the artifact that
  would let anyone check them is not committed.
- SD-020, SD-029, SD-030, SD-031, SD-033, SD-034 stand as written. SD-020 in particular does not
  depend on the edition at all: it is about the harness code on `origin/main`.

Method note for the next audit: establishing which *code* is current was already a rule after wave 1.
It now covers documents and artifacts too, and adds the distinction between a claim that is wrong and
a claim that is merely unbacked. Both changes are in the skill.

---

## Продолжение — аудит 2 (волна 3), 2026-08-24

Этот файл закрыт на записи SD-039. Аудит 2 выполнен против релизного рефа `origin/main`
@ `6478c884` (v0.12.0) и живёт в отдельном файле:

**[`xlog-architecture-debt-audit2-2026-08-24.md`](xlog-architecture-debt-audit2-2026-08-24.md)**

Он содержит: статус каждой из 39 записей выше на текущем `main` (4 fixed, 1 accepted,
6 worse, остальные open), 12 новых записей SD-040…SD-051, 5 снятых сигналов ND-012…ND-016
и заявление о покрытии. Нумерация `SD-nnn` сквозная и не переиспользуется — следующая
свободная `SD-052`.
