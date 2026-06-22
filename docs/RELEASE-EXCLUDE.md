# Release Exclusion Manifest

Internal/process documentation that **must not ship in the public release** but is
**kept in the repository for now** (development history, reading, and understanding).

This file is the single source of truth for what release packaging/tooling should
strip. Nothing here is deleted yet — these paths are retained locally and excluded
only at packaging time.

Status as of: v0.8.6 (2026-05).

## Excluded paths

| Path | Approx files | What it is | Reason for exclusion |
|---|---|---|---|
| `docs/superpowers/specs/` | 3 | Design/brainstorm specs for agentic (Claude Code) workflows | Internal process artifacts; not user-facing |
| `docs/superpowers/plans/` | 4 | Implementation plans for release/tooling prep | Internal process artifacts; not user-facing |
| `docs/plans/` | ~88 | supervisor-goal-NNN, *-closure-proposal, *-plan, design docs | Goal-driven development infrastructure; development history, not reference |
| `docs/evidence/` | ~159 (72 dirs) | Dated acceptance evidence (README + measurements.json per slice) | Internal acceptance-gate audits; consumed by release tooling, not consumers |
| `docs/v065-closure-board.md` | 1 | Legacy release closure board (the filename uses the old release-board shorthand) | Internal release-tracking infrastructure |

## Archive candidates (historical, superseded)

These pre-date v0.5.0 and are superseded by current certification (207/207). Recommend
moving to an archive location or excluding from release:

| Path | Reason |
|---|---|
| `docs/certification/2026-01-12-cuda-certification-results.md` | v0.4.0-alpha era (133 tests) |
| `docs/certification/2026-01-14-cuda-certification-results.md` | v0.4.0-alpha era (140 tests) |
| `docs/certification/2026-01-22-neural-symbolic-certification-report.md` | Untagged neural-symbolic certification report |
| `docs/certification/neural-symbolic-gpu-certification-spec.md` | Historical neural-symbolic certification spec |
| `clippy-report.txt` (repo root) | Empty build artifact |

## Notes

- `.gitignore` keeps `!docs/evidence/**` tracked deliberately so git-based release
  tooling can read it. That tooling is internal CI/automation, not consumer docs.
- When release packaging is implemented, it should read this manifest and strip the
  listed paths from the distributed artifact.
