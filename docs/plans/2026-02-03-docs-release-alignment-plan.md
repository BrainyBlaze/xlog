# Docs Release Alignment (v0.3.2) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Align README/ROADMAP/CHANGELOG and supporting docs to the repo’s actual release state (latest tagged release is `v0.3.2`; `v0.4.0-alpha` is not yet achieved).

**Architecture:** Treat release tags (`git tag`) as the source of truth for “Released” claims. Treat `main` as “Unreleased / ahead of v0.3.2” and document what is implemented vs what is release-gated (especially neural examples + example validation).

**Tech Stack:** Markdown docs (`README.md`, `CHANGELOG.md`, `docs/*.md`), git history, ripgrep.

---

## Task 0: Worktree + Baseline Verification

**Files:** none

**Step 1: Create a docs-only worktree**

Run:
```bash
git check-ignore -q .worktrees
git worktree add .worktrees/docs-alignment -b docs-alignment
```

**Step 2: Verify baseline is clean**

Run:
```bash
cd .worktrees/docs-alignment
LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-} cargo test --workspace --all-targets --exclude pyxlog -- --nocapture
```

Expected: exit 0.

---

## Task 1: Inventory “Release Claims” and Drift

**Files:**
- Inspect: `README.md`
- Inspect: `CHANGELOG.md`
- Inspect: `docs/ROADMAP.md`
- Inspect: `docs/ARCHITECTURE.md`
- Inspect: `docs/architecture/python-bindings.md`
- Inspect: `docs/architecture/cuda-certification.md`
- Inspect: `docs/certification/*.md`

**Step 1: Confirm tag reality**

Run:
```bash
git tag -l
```

Expected: latest release tag is `v0.3.2` (no `v0.4.0-alpha` tag).

**Step 2: Find “v0.4.0-alpha Released” claims**

Run:
```bash
rg -n "v0\\.4\\.0-alpha" README.md CHANGELOG.md docs -S
```

Expected: locate files that must stop claiming “released”.

---

## Task 2: Update Release Framing (README + ROADMAP + ARCHITECTURE)

**Files:**
- Modify: `README.md`
- Modify: `docs/ROADMAP.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/architecture/python-bindings.md`

**Step 1: README version + section headers**
- Change the badge/wording to reflect: latest release `v0.3.2`; `main` includes unreleased work.
- Rename “Neural-Symbolic Training (v0.4.0-alpha)” to “Neural-Symbolic Training (unreleased; target v0.4.0-alpha)”.
- Ensure no phrasing implies v0.4.0-alpha is shipped.

**Step 2: ROADMAP header + version table**
- Set “Current Version” to `v0.3.2 (Released)`.
- Represent `v0.4.0-alpha` as “Planned / target”, with explicit release gates:
  - validate *all* examples end-to-end
  - implement missing neural examples beyond `examples/neural/01_minimal`
- Update CUDA certification numbers to current suite.
- Fix GPU-native KC “planned” items that are already implemented (e.g., SAT/CDCL certification categories).

**Step 3: ARCHITECTURE + python-bindings**
- Replace “(v0.4.0-alpha)” labels with “(unreleased; target v0.4.0-alpha)” where appropriate.

**Step 4: Verify no stale claims remain**

Run:
```bash
rg -n "v0\\.4\\.0-alpha\\s*\\(Released\\)|v0\\.4\\.0-alpha complete|\\*\\*Released\\*\\*" docs README.md CHANGELOG.md -S
```

Expected: no results (or only historical context clearly marked as historical).

---

## Task 3: Fix Certification Docs (Counts, Categories, Links)

**Files:**
- Modify: `docs/architecture/cuda-certification.md`
- Modify: `docs/certification/2026-01-22-v0.4.0-alpha-certification-report.md`
- Modify: `docs/certification/v0.4.0-alpha-certification-spec.md`
- Modify: `README.md`
- Modify: `docs/ROADMAP.md`

**Step 1: Update certification suite facts**
- Update total tests/categories/PTX module count to match the current repo suite (C01–C25 + G01–G08).
- Ensure docs describe “current HEAD” certification, not a historical 140-test suite.

**Step 2: Historical certification reports**
- Keep historical results, but add explicit “historical snapshot / internal milestone” notes and link to current suite.

**Step 3: Verify via command output**

Run:
```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-} cargo test -p xlog-cuda-tests --test certification_suite -- --nocapture
```

Expected: PASS, and the output shows the current counts.

---

## Task 4: Restore a Single Source of Truth for Validation

**Files:**
- Create: `docs/VALIDATION_REPORT.md`
- Modify (remove broken refs): `docs/plans/2026-01-09-validation-report.md`
- Modify (remove broken refs): `docs/plans/2026-01-11-full-system-validation-report.md`

**Step 1: Create `docs/VALIDATION_REPORT.md`**
Include:
- Latest tagged release (`v0.3.2`)
- Current `main` status (“unreleased; ahead of v0.3.2”)
- Concrete verification commands and the most recent run date
- Explicit note that example validation is incomplete (release gate for v0.4.0-alpha)

**Step 2: Update docs that reference the missing file**
- Point to `docs/VALIDATION_REPORT.md` and ensure links resolve.

**Step 3: Link verification**

Run:
```bash
rg -n "docs/VALIDATION_REPORT\\.md" docs -S
test -f docs/VALIDATION_REPORT.md
```

Expected: file exists and references are consistent.

---

## Task 5: Final Consistency Sweep

**Files:** documentation only

**Step 1: Search for known stale tokens**

Run:
```bash
rg -n "140/140|200/200|Current Version: v0\\.4\\.0-alpha|\\bv0\\.4\\.0-alpha\\b.*Released" README.md docs CHANGELOG.md -S
```

Expected: only historical docs show historical numbers; no “v0.4.0-alpha Released” claims remain.

**Step 2: Ensure doc edits didn’t introduce TODO/placeholder language**

Run:
```bash
rg -n "\\bTODO\\b|PLACEHOLDER|TBD" README.md docs CHANGELOG.md -S
```

Expected: no new TODO/placeholder content added by this change.

---

## Task 6: Commit

**Files:** docs only

Run:
```bash
git status
git add README.md CHANGELOG.md docs/ROADMAP.md docs/ARCHITECTURE.md docs/architecture/python-bindings.md docs/architecture/cuda-certification.md docs/certification docs/VALIDATION_REPORT.md docs/plans/2026-01-09-validation-report.md docs/plans/2026-01-11-full-system-validation-report.md
git commit -m "docs: align release status to v0.3.2; update roadmap/changelog/certification"
```

