# v0.9.2 Epistemic Closure Validation Checklist

Status: pending. This file is a validation runbook, not passing evidence.

Purpose: after the GPU-native WFS and C2 evidence expansions, the branch must still be proven against the original HOLD_FOR_FIXES blockers with fresh current-state gates. Do not use stale green runs or source-only edits as completion evidence.

## Fixture families that must stay in validation scope

The focused gates below must include every fixture family added for closure. A passing run over only
the original examples is insufficient.

| Family | Required scope | Why it exists |
|---|---|---|
| `13f*` / `13fw*` | Interior-negation target `{present,absent}` x mode `{FAEEL,G91}` | Proves the named C2 pilot is not a one-case present-target check. |
| `13g`-`13v` | All 64 two-operator negation cells under FAEEL | Proves finite nested-chain parity/duality over the base mode. |
| `13w*` | Same 64 two-operator negation cells under G91 | Proves the normalization is not accidentally FAEEL-only. |
| `33` / `33a*` | Canonical and mode/operator cyclic negated-modal WFS | Proves accepted cyclic negated-modal recursion routes to GPU WFS. |
| `33c*` | Mode x operator x seed `{present,absent}` cyclic WFS | Proves WFS truth output tracks seed support, not a hardcoded tuple. |
| `33b` / `33d*` | WFS plus ordinary EDB negation in the reduced SCC | Proves mixed modal/ordinary negation routes through the same GPU WFS plan. |
| `33e*` | WFS plus load-bearing EDB target state `{allowed,banned}` | Proves ordinary `not banned(Y)` has runtime effect, not just plan presence. |
| `42a*` / `42b*` | Same-name multi-arity single-literal and cross-arity matrices | Protects the same-name/arity disambiguation closure surface. |
| `44a*` | Single-modal truth table across mode, operator, and tuple state | Protects the base modal truth table in both modes. |

## Original blockers mapped to required evidence

### 1. Accepted WFS path must be GPU-native, not host WFS

Required source evidence:
- `crates/xlog-gpu/Cargo.toml` has no `xlog-prob` dependency.
- `crates/xlog-gpu/src/logic.rs` has no old host-WFS solver tokens: `xlog_prob`, `evaluate_wfs_rules`, `ground_wfs_program`, `LogicExecutionPlan::EpistemicWfs(`, `WfsRule`, `WfsLiteral`, `WfsConfig`, or `PirGraph`.
- WFS plan JSON reports `plan_kind:"epistemic_wfs_gpu"`, `reduction:"wfs_gpu_recursive"`, `host_wfs_fallback_allowed:false`, `wfs_gpu_passes:["overapprox","lower","upper"]`, deterministic `wfs_fixed_relations`, and explicit `wfs_convergence_predicates`.

Required focused gates:
```bash
cargo test -p xlog-gpu --test logic_runner test_xlog_gpu_manifest_does_not_depend_on_xlog_prob_host_wfs -- --nocapture
cargo test -p xlog-gpu --test logic_runner test_xlog_gpu_logic_source_does_not_reintroduce_host_wfs_solver -- --nocapture
cargo test -p xlog-gpu --test logic_runner test_wall_a1_wfs_plan_kind_matrix_compiles_without_cuda -- --nocapture
cargo test -p xlog-gpu --test logic_runner test_wall_a1_wfs_plan_clamps_zero_iteration_bound_without_cuda -- --nocapture
cargo test -p xlog-gpu --test logic_runner test_wall_a1_wfs_plan_exposes_multiple_fixed_relation_maps_without_cuda -- --nocapture
cargo test -p xlog-gpu --test logic_runner test_wall_a1_wfs_plan_exposes_fixed_relation_for_ordinary_edb_negation_without_cuda -- --nocapture
XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-gpu --test logic_runner test_wall_a1_negated_modal_cycle_routes_to_gpu_wfs_matrix -- --nocapture
XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-gpu --test logic_runner test_wall_a1_wfs_fixed_relation_names_avoid_user_collisions -- --nocapture
```

Required production-example gates:
```bash
XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-cli --test run_cli_tests test_xlog_run_negated_modal_through_recursion_uses_gpu_wfs_engine -- --nocapture
```

The WFS CLI gate must cover ex33, `33a*`, `33c*`, `33b`, `33d*`, and `33e*`. Passing only
the canonical ex33 fixture is not enough. The `33e*` cells are specifically load-bearing:
`allowed` cells must keep exactly `reach(1,2)`, while `banned` cells must emit no reach rows.

### 2. Release docs must not contradict closure status

Required evidence:
- Release docs must state GPU-native WFS candidate implemented, gates pending.
- Release docs must not say G91 possible recursion or accepted cyclic WFS is undone.
- Release docs must distinguish implemented source from unverified merge evidence.

Suggested stale-language scan:
```bash
rg -n "missing GPU-native WFS|WFS gap|host-side CPU|wfs-rejected|13f-nested-modal-interior-negation-rejected|G91 possible.*undone|cyclic WFS cases without a GPU-native executor" docs CHANGELOG.md ROADMAP.md examples/epistemic crates/xlog-cli/tests crates/xlog-gpu/tests
```

Expected result: no matches except intentional historical quotations that are explicitly marked superseded or rejected evidence.

### 3. C2 nested interior-negation pilot must be clean evidence

Required source evidence:
- `examples/epistemic/13f-nested-modal-interior-negation.xlog` declares `p()` and asserts `p().` for the present-target case.
- 13f companion cells cover target `{present,absent}` x mode `{FAEEL,G91}`.
- 13g-13v cover the 64 two-operator negation cells under FAEEL.
- 13w* replays the same finite two-operator negation cells under explicit G91.

The named 13f mini-matrix must include these exact files:
- `13f-nested-modal-interior-negation.xlog` (FAEEL, target present, `q` empty)
- `13f-nested-modal-interior-negation-absent.xlog` (FAEEL, target absent, `q` derived)
- `13fw-nested-modal-interior-negation-g91-present.xlog` (G91, target present, `q` empty)
- `13fw-nested-modal-interior-negation-g91-absent.xlog` (G91, target absent, `q` derived)

Required focused gates:
```bash
XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-cli --test run_cli_tests test_xlog_run_epistemic_examples -- --nocapture
XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-cli --test run_cli_tests test_xlog_run_nested_modal_negation_matrix_g91_companion -- --nocapture
```

The first gate must include the 13f present/absent and G91 companion entries in the example table. Passing only the original present-target fixture is not enough.

### 4. Branch must be current-main integrated

Required evidence:
- Current local main is merged or rebased into the worktree branch.
- The final validation SHA is recorded after integration, not before it.
- Any docs/config changes from main are reconciled without regressing the source and example surfaces above.

Suggested pre-merge/integration checks:
```bash
git merge-base --is-ancestor main HEAD
git status --short
git log --oneline --left-right main...HEAD
```

Do not claim this blocker resolved from source edits alone. It requires current git evidence after integration.

## Full release gates after focused gates

Run only after the focused gates above pass on the integrated branch:

```bash
cargo build -p xlog-prob --tests
cargo build -p xlog-prob --tests --features host-io
cargo check --workspace --all-targets
cargo test -p xlog-logic --test test_epistemic_eir
cargo test -p xlog-logic --test test_epistemic_splitting
cargo test -p xlog-logic --test test_epistemic_gpt
XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-gpu --test logic_runner
XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-cli --test run_cli_tests
XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-integration --test epistemic_wcoj
cargo fmt --check
git diff --check
rg -n "<<<<<<<|=======|>>>>>>>" .
```

If any command fails, record the failure exactly and fix the root cause before rerunning the relevant focused gate. Do not replace a failing runtime gate with a source audit or documentation update.
