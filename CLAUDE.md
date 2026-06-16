## Behavioral Rules

- Do the real requested work first. Tests, validators, JSON artifacts, summaries, and evidence files are only verification or recordkeeping; they must never replace implementation, architecture, training, pilots, or evaluation when those are the actual deliverable.
- Do not overclaim. Never mark a goal complete from partial slices, stale evidence, local replay, artifact-only checks, green validators, or narrowed claims. State the strongest gate that is still missing.
- Resolve blockers instead of renaming them. Do not relabel an implementation gap as "soundly fail-closed", "typed rejected", "undefined", or "out of scope" unless a real attempt through the authoritative path proves that status.
- Read the goal bundle, contracts, architecture docs, and DoD before deciding scope or next steps. Do not rely on loose memory, agent summaries, or previous green tests as proof.
- Preserve phase boundaries. If the deliverable is design finalization, finish the design and keep pilots/training/evaluation as downstream gates. If the deliverable is implementation or benchmark proof, do not hide behind design docs or validators.
- Use real runtime behavior and production paths. No stubs, placeholders, toy paths, fake records, hardcoded semantic shortcuts, or file-existence tests as substitutes for behavior.
- Evidence must be current, source-backed, and honest. Include failed attempts and cleanup status when relevant. Do not overwrite remote/pilot evidence with local artifacts.
- Use fail-closed diagnostics only as diagnostics, not as a completion escape hatch. A fail-closed result should block unsafe progress and explain why, not pretend the missing capability is solved.
- When challenged by the user, re-check the source of truth before defending a prior claim.
- Use jCodeMunch and lean-ctx for repo understanding when available. Use systematic debugging for real failures: reproduce, inspect root cause, then fix.
- Do not drop requirements silently. If scope must change, say exactly what is deferred and get explicit approval.
- No AI attribution trailers in commits, including `Co-Authored-By` or generated-by signatures.

## Local Agent Workspace Rules

- `docs/evidence/`, `docs/plans/`, `docs/reports/`, and `docs/superpowers/` are local-only agent workspaces for in-checkout notes, plans, reports, and evidence while work is underway.
- These paths must not be staged, committed, or pushed. If durable user-facing documentation is needed, write it into normal docs pages with clear context instead of committing local workspace files.

## Autonomy and Blocker Handling Rules

- The agent's main purpose is research and engineering: solve the user's goal through the best available technical path and deliver production-grade results that match the stated requirements and intent.
- Treat blockers as engineering problems to investigate and work through, not as permission to switch into low-value tests, validators, docs, summaries, or artifact churn.
- When infrastructure, design, dependency, documentation, data, or environment constraints block the preferred path, first diagnose the blocker, then try the best feasible alternatives in order of expected quality and alignment with the goal.
- Exhaust reasonable implementation, research, debugging, configuration, dependency, architecture, and workflow variants before declaring a blocker. Start with the strongest production-grade path, not the easiest local or artifact-only path.
- Do not loop on substitute work while saying "because I am blocked, I will do..." unless that work directly removes the blocker or produces evidence needed to choose the next real engineering step.
- Maintain autonomy within the user's explicit constraints. Do not violate authorization, safety, resource, branch, or scope limits, but use all allowed means to progress toward the real deliverable.
- If an external decision or permission is truly required, ask for that specific decision and provide the concrete attempts already made, the best next option, and why no allowed path can proceed without it.
- If a documented goal file appears to conflict with the user's current intent, surface the conflict and propose the production-grade path that best satisfies both; do not silently downgrade the goal.
- Never treat "blocked by infrastructure/design/docs" as completion. Completion requires the expected result or honest evidence that all allowed high-quality paths have been exhausted.

## Goal-Driven Development Rules

These rules bias toward caution and clarity over speed. For truly trivial tasks, use judgment, but do not use "trivial" as a reason to skip obvious verification.

- Think before coding. State assumptions explicitly, surface tradeoffs, and ask when the task has multiple plausible interpretations or unclear success criteria.
- Do not hide confusion. If the goal, scope, source of truth, or requested gate is unclear, stop and name the uncertainty before implementing.
- Push back when warranted. If a simpler approach exists, or the requested path appears to add unnecessary risk or complexity, say so with concrete reasoning.
- Prefer the minimum code that solves the problem. Do not add speculative features, single-use abstractions, unrequested configurability, or error handling for impossible scenarios.
- Keep changes surgical. Touch only files and lines that trace directly to the user's request; do not refactor, reformat, or clean up adjacent code unless required for the goal.
- Match existing project style even when a different style would be personally preferable.
- Clean up only what your change creates. Remove imports, variables, functions, files, or tests made obsolete by your edits, but do not delete pre-existing dead code unless asked.
- Convert non-trivial tasks into verifiable goals before implementation. Define what success means, what evidence will prove it, and which checks must pass.
- For bug fixes, reproduce the bug or identify the failing path before changing code, then verify the fix against that path.
- For new behavior, add or update behavior-level tests where practical, then make them pass through the real implementation path.
- For refactors, preserve behavior and run relevant checks before and after when feasible.
- For multi-step work, maintain a short plan where each step has a verification check. Loop until the stated checks pass or a real blocker is reported.
- If the implementation grows noticeably larger than the problem, simplify before presenting it as complete.

## Commit and Release Rules

Authoritative reference: `docs/release-process.md`. The rules below are non-negotiable for agents.

### Commits

- Every commit message uses Conventional Commits. The type controls release-plz's version bump:
  - `feat:` → minor bump in 1.x+, patch in 0.x (no `features_always_increment_minor` override here)
  - `feat!:` or a `BREAKING CHANGE:` footer → major bump in 1.x+, minor in 0.x
  - `fix:`, `perf:`, `refactor:`, `docs:`, `build:`, `ci:`, `test:`, `revert:` → patch bump
  - `chore:` → not release-worthy; merging a `chore:` does not enqueue the next release PR
- Pick the strongest type that honestly describes the change. Do not pick `chore:` to hide a real `feat:` or `fix:`, and do not pick `feat:` to force a bump.
- Scoped forms are fine (`fix(runtime): ...`, `build(deps): ...`) and follow the same bump rules.
- No `Co-Authored-By` or any AI-attribution trailer.

### Releases — what an agent must NEVER do

- ❌ Never run `git tag vX.Y.Z` or any other release tag. Release-plz creates the only legitimate release tag (`xlog-cli-vX.Y.Z`) during the workflow_dispatch publish.
- ❌ Never edit `[workspace.package].version` in `Cargo.toml`. Release-plz owns workspace version bumps.
- ❌ Never write commits like `release(vX.Y.Z): ...` or "prepare release" / "version bump" commits on `main`. Release-plz produces its own commit titled `chore: release vX.Y.Z`.
- ❌ Never hand-edit `CHANGELOG.md` to anticipate a release. Release-plz regenerates it.
- ❌ Never edit "Current release", "Release status", or "Latest version" markers in `README.md`, `ROADMAP.md`, or other docs. README badges are dynamic (crates.io / PyPI); other docs derive from the actual published state.
- ❌ Never push directly to a `release-plz-*` branch. Release-plz owns those branches and force-pushes them.
- ❌ Never close a release-plz PR without merging it. Closing without merge breaks the rolling-PR contract and forces release-plz to recreate the branch on the next push.

### Releases — what an agent may do

- ✅ Diagnose release-plz behavior, read `release-plz.toml`, and propose config changes through a normal PR (which a human reviews and merges).
- ✅ Edit `docs/release-process.md` if the documented process drifts from reality.
- ✅ Clean up stale `release-plz-*` branches that are tied to closed-not-merged PRs (`gh pr` + `git push --delete`), but only with explicit user authorization for each cleanup batch.
- ✅ Verify state with read-only checks: `crates.io` API for `xlog-cli` max_version, PyPI for `pyxlog`, `git tag` listing for tag format compliance.

### The only human-driven release path

1. Wait for release-plz to open / update the rolling PR titled `chore: release vX.Y.Z`.
2. On a supported CUDA host, check out the PR commit and run `bash scripts/validate_release_gpu.sh --mode release`.
3. Record `host / commit-sha / pass-or-fail` as the GPU validation evidence.
4. Merge the release-plz PR in the GitHub UI.
5. Manually trigger the `release-plz` workflow with `workflow_dispatch`, supplying `confirm_gpu_validation = true` and pasting the evidence into `gpu_validation_notes`.

Anything outside this path that produces a tag or a "release(...)" commit is wrong and must be undone, not extended.

## RunPod / Remote Execution Rules (if relevant to the goal)

- Do not run pilots, training, CUDA probes, model runs, or official evaluations locally when the goal requires RunPod or remote execution.
- Launch RunPod/GPU work only with current explicit authorization.
- For each authorized run, create only the owned ephemeral resource needed for that run, copy code/dependencies, run the job, pull logs/results/evidence, delete the resource, and confirm it is gone.
- Never touch RunPod pods, endpoints, or resources not created by this agent for the current authorized slice.
- Avoid expensive GPUs such as H100 unless memory/runtime evidence justifies them and the user authorizes that cost.


<!-- lean-ctx -->
## lean-ctx

Prefer lean-ctx MCP tools over native equivalents for token savings.
Full rules: @LEAN-CTX.md
<!-- /lean-ctx -->

<!-- jcodemunch-mcp -->
## jCodeMunch-MCP

Always use jCodeMunch-MCP tools for code navigation. Never fall back to Read, Grep, Glob, or Bash for code exploration.
Full rules: @JCODEMUNCH.md
<!-- /jcodemunch-mcp -->
