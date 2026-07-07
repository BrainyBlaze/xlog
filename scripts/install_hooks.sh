#!/usr/bin/env bash
# Installs a pre-commit hook that blocks internal development task-codes from
# entering committed code, comments, tests, and examples. Behavioral language
# only: describe what the code does, not which task/slice produced it.
# See the naming rule in AGENTS.md. Idempotent; re-run any time.
#
# Bypass for a genuine, dated historical record: git commit --no-verify
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
hook="$repo_root/.git/hooks/pre-commit"

cat > "$hook" <<'HOOK'
#!/usr/bin/env bash
# Block internal task-codes in staged additions. Bypass: git commit --no-verify
set -euo pipefail

# Task-code vocabulary (added lines only). Strong codes that are never legitimate
# as live explanatory text. Behavioral names instead, per AGENTS.md.
pattern='(ST-?TRC|EGB-[0-9]|design-[AB][[:space:]>)]|Shape-[0-9]|surface-[0-9]|Arm-[AB][[:space:]>)]|Phase-1[ab][[:space:]>)]|SLICE-[0-9]|v[0-9]+\.[0-9]+\.[0-9]+[[:space:]]+slice[[:space:]]+[0-9]|@(xlog|dts)-?(claude|dlm))'
# Legitimate identifiers that must not trip the gate.
allow='(_slice[0-9]|\.slice|as_slice|slice::|reshape|landscape|shape0|within_set|surface_|\.shape|deslice)'

# Only inspect added lines in staged source/comment files (skip gitignored workspaces).
offenders="$(git diff --cached -U0 --no-color -- 'crates/**' 'python/**' 'examples/**' 'src/**' \
  | grep -E '^\+' | grep -Ev '^\+\+\+' \
  | grep -En "$pattern" | grep -Ev "$allow" || true)"

if [ -n "$offenders" ]; then
  echo "pre-commit: internal task-code detected in staged additions." >&2
  echo "Use behavioral language (what the code does), per the AGENTS.md naming rule." >&2
  echo "Offending additions:" >&2
  echo "$offenders" | sed 's/^/  /' >&2
  echo "If this is a genuine dated historical record, bypass with: git commit --no-verify" >&2
  exit 1
fi
HOOK

chmod +x "$hook"
echo "Installed pre-commit task-code gate at $hook"
