# Path C Paper-Grounded Findings And Execution Roadmap

## Status

Branch: `feat/v065-path-c-bundle-expansion`

Base: `main @ 3f8e5d4c6eeccc4f738056806c62f31ed0941413`

Predecessor: G29 closure proposal `2088c4c8` is rejected. Its D7a amendment-to-variance-proxy is preserved on `feat/w33-closure-proposal-iteration-1` as historical evidence and is not merged here. W3.3 remains OPEN under the original wall-time speedup gate.

User directive, 2026-05-13:

> "path c, no deffers, no simplification, document all findings and dispatch path c to codex with attached paper, requirements, and goals. the goal is to deliver FULL implementation and achieve target performance metrics according paper and xlogs architecture specifics. do not claim done until the goal is truly achieved. no excuse, no violations, no negotiating."

## §1 Paper Grounding

Primary source: Sun, Qi, Gilray, Kumar, Micinski, "Scaling Worst-Case Optimal Datalog to GPUs", arXiv:2604.20073v2. The arXiv abstract states that SRDatalog combines WCOJ, flat columnar storage, two-phase deterministic memory allocation, root-level histogram load balancing, structural helper-relation splitting, and stream-aligned rule multiplexing; it reports geometric-mean speedups of 21x to 47x on real program-analysis workloads.

Paper §6 attributes the headline result to the "combined synergy" of multiple mechanisms, not to W3.3 histogram scheduling in isolation. The evaluation table reports 17 datasets across seven workload classes, with input sizes from 977K to 126.9M tuples and iteration counts from 24 to 2,322. The hardware comparison is an NVIDIA RTX 6000 Ada GPU path against an AMD EPYC 9655 CPU baseline.

Paper Figure 5's skew ablation is not a single universal histogram result: the reported histogram-guided range is "1.1x to 35.8x". The high end is HeapAllocHelper, where helper-relation splitting exposes inner skew to the root-level histogram. The low end occurs when the chosen delta/root column is close to uniform or the workload is too small to amortize scheduling overhead.

Paper §5 Algorithm 2 is the "HG-WCOJ Kernel": it consumes prefix-summed root-key work units, maps each block to a slice of the flattened 1-D work space, binary-searches prefix sums to recover the owning root key, then runs the normal cooperative WCOJ inner traversal over flat columns. This confirms that W3.3 alone is one mechanism in a larger kernel pipeline, not the full performance story.

Paper §6 also explains why single-rule or rule-sparse workloads underrepresent the paper's speedups. Later fixpoint iterations can produce a "microscopic trickle" of deltas, and stream multiplexing gives negligible gains on rule-sparse workloads; the stream ablation says DOOP-class rule-rich strata benefit, while Andersen, ddisasm, and polonius show about +/-3% variance.

## §2 Empirical Chain Summary

| Goal | Commit | Finding |
|---|---|---|
| G11 | `a4c299fd` | Paper-aligned W3.3 plan iteration approved: histogram state belongs with `CudaBuffer`, refresh in Merge, consume at launch. |
| G12 | `3490fd09` | First spike exposed a noisy Criterion surface and stopped before superhub. |
| G13 | `d2a2fca5` | Phase attribution showed the initial noisy delta was mostly harness/measurement residual, not intrinsic P3/P5 cost. |
| G14 | `24c51bda` | Isolated harness removed most G12 noise but left a cross-harness residual. |
| G15 | `775902ed` | Expanded phase probes did not find fixable implementation overhead in the design buckets. |
| G16 | `4a8031ef` | Harness parity showed same-process Instant and phase probes agree; Criterion aggregation was the outlier. |
| G17 | `38dcc7fa` | Criterion audit confirmed batch amortization as the aggregation phantom source. |
| G18 | `d217a9c5` | Fixed harness passed D7b on uniform and showed superhub-50K still did not meet D7a. |
| G19 | `822aeb99` | First scale sweep stopped on 50K instability before 200K/1M. |
| G20 | `43dc0b4a` | V3 `sample_size(200)` established a stable production-bench measurement protocol. |
| G21 | `19d322fc` | V3 scale sweep kept W3.3 in measurement-failure territory; no closure. |
| G22 | `258bddc6` | Design RCA identified RC1+RC3: metadata discarded and histogram was not real skew data. |
| G23 | `dcb556db` | First true slice-aware kernels achieved real per-block variance reduction but not wall-time speedup. |
| G24 | `429c2cca` | Static 468-block slice-aware path measured `0.554991x` at 50K and stopped. |
| User fix | `6595b969` | Device-side slice-prefix computation reduced refresh overhead but did not change the core launch/scheduling tradeoff. |
| G25 | `7eb94bc2` | Partial Phase A launch-overhead probes were superseded after restart. |
| G26 | `2aeb74b4` | Static 117-block grid-stride reduced block count but collapsed balancing and regressed D7b. |
| G27 | `d986cf10` | Persistent work stealing preserved correctness and D7b, improved variance by 63.77%, but measured `0.203492x` at 50K. |
| G28 | `b0589101` | Scale validation rejected scale-threshold closure: 50K `0.204665x`, 200K `0.049343x`, 1M merge-resident timed out over one hour. |
| G29 | `2088c4c8` | Proposed variance-proxy amendment; user rejected it and selected Path C. |

Reference evidence READMEs are preserved on their unmerged branches, especially G27 `docs/evidence/2026-05-13-w33-persistent-threads-work-stealing/README.md` and G28 `docs/evidence/2026-05-13-w33-persistent-threads-scale-validation/README.md`.

What worked: W3.3 now has a paper-faithful launch path, deterministic row equality, adaptive uniform routing, CUDA cert preservation, and a measured 63.77% work-balancing effect.

What did not work: none of the isolated W3.3 scheduling architectures met the original `>= 2.0x` wall-time gate on the existing synthetic superhub fixture, and scaling that fixture worsened the ratio.

## §3 Gap Analysis

| Paper mechanism | Current board coverage | Gap |
|---|---|---|
| Raw GPU memory bandwidth | Not a board item | Hardware property only. |
| Columnar WCOJ execution model | W3.1 + W3.2 DONE | Covered for sorted accessors and k=5/k=6 WCOJ template. |
| Flat-array delta merges | W4.1 DONE | Covered for multi-recursive WCOJ correctness. |
| Histogram-guided skew mitigation | W3.3 OPEN | Implementation exists on G27 branch but original D7a still fails in isolation. |
| Helper-relation splitting | Missing | Add W3.7: AOT rule rewriting to expose buried inner skew to root-level histograms. |
| Stream-aligned rule multiplexing | Missing | Add W3.8: phase-aligned Count/Materialize dispatch across CUDA streams for independent rules. |
| Production-scale benchmark suite | Missing | Add W3.9: paper-class fixtures at or above the paper's minimum scale and full-bundle measurement surface. |

The original W3.3 superhub-50K fixture is single-rule, shallow, synthetic, and much smaller than the paper's smallest reported workload. That fixture was useful for RCA and correctness, but it cannot stand in for the paper's production performance envelope.

## §4 G29 Rejection Rationale

G29 proposed substituting a measured variance-reduction proxy for the original W3.3 `>= 2.0x` wall-time speedup gate. The user rejected that path and explicitly selected Path C. Therefore:

- W3.3 remains OPEN.
- The original `>= 2.0x` wall-time speedup gate remains binding.
- The G29 branch is preserved unmerged as historical evidence.
- This branch adds the missing paper mechanisms and production benchmark requirement instead of relaxing D7a.

## §5 Path C Execution Roadmap

| Goal | Scope | Acceptance anchor |
|---|---|---|
| G31 | W3.4 Kernel fusion: count+materialize or layout+count single-kernel, with auto-disable below threshold. | Existing W3.4 gate: `>= 1.3x` on materialization-long-pole fixture, deterministic, no small-fixture regression. |
| G32 | W3.5 Shared-memory optimization for small WCOJ relations. | Existing W3.5 gate: `>= 1.5x` below threshold, deterministic, no above-threshold regression. |
| G33 | W3.6 Warp-level `__shfl_*` primitives. | Existing W3.6 gate: `>= 1.3x` below W3.5 threshold, deterministic, no above-threshold regression. |
| G34 | W3.7 Helper-relation splitting AOT rule rewriter. | New W3.7 gate: 6+-variable deep-join inner-skew fixture shows `>= 2x` vs no-rewrite baseline, deterministic. |
| G35 | W3.8 Stream-aligned rule multiplexing AOT compiler pass. | New W3.8 gate: 3+ independent-rule fixture shows `>= 1.27x` vs sequential dispatch; single-rule workloads stay within +/-3%. |
| G36 | W3.9 Production-scale WCOJ benchmark suite. | New W3.9 gate: at least three paper-class fixtures, full-bundle and uniform-baseline pairs, ratios reported. |
| G37 | W3.3 full-bundle integration testing. | Compose W3.3+W3.4+W3.5+W3.6+W3.7+W3.8 on W3.9 fixtures until D7a `>= 2.0x` is empirically achieved with correctness and D7b preserved. |
| G38 | W3.3 closure proposal. | Paper-faithful closure proposal only after the original D7a gate is met under the full bundle and user approval is requested. |

Each G31-G38 item is a separate supervisor goal artifact and its own user-approved closure gate. No item is marked DONE from this planning commit.

## §6 Acceptance Criteria

W3.3 DONE requires all of the following:

- W3.4, W3.5, W3.6, W3.7, W3.8, and W3.9 are DONE through their own evidence and user approvals.
- The full bundle meets W3.3 D7a: `>= 2.0x` wall-time speedup vs uniform baseline on the canonical W3.9 production-scale fixture.
- Deterministic output is preserved via row-for-row equality against the baseline run.
- Uniform-fixture D7b remains within +/-5%.
- CUDA cert suite and workspace release gates pass.
- No proxy substitution, scale-emergence extrapolation, or measurement-surface shopping is used as closure evidence.

## G30 Verification

| Gate | Result |
|---|---|
| `cargo fmt --check --all` | EXIT 0 |
| `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` | EXIT 0 |
| `cargo test -p xlog-cuda-tests --test certification_suite --release` | EXIT 0; 1 passed / 0 failed |
| `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` | EXIT 0 |

## Review Request

G30 requests explicit user approval of:

1. Board expansion adding W3.7, W3.8, and W3.9.
2. G29 amendment rejection: no W3.3 OPEN->DONE transition and no variance-proxy substitution.
3. Path C execution roadmap G31-G38.
4. Authorization to proceed with G31 after this staged board expansion is accepted.
