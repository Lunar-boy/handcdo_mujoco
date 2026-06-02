#!/usr/bin/env bash
set -euo pipefail

BASE_BRANCH="${BASE_BRANCH:-main}"
TASK_FILE="${1:-}"
TITLE="${2:-}"

if [[ -z "$TASK_FILE" || -z "$TITLE" ]]; then
  echo "Usage: $0 <task-file> <pr-title>"
  echo "Example: $0 .codex/pr-prompts/backend-abstraction.md 'Add simulator backend abstraction'"
  exit 1
fi

if [[ ! -f "$TASK_FILE" ]]; then
  echo "Task file not found: $TASK_FILE"
  exit 1
fi

git rev-parse --is-inside-work-tree >/dev/null

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is not clean. Commit or stash your changes first."
  git status --short
  exit 1
fi

gh auth status >/dev/null

SAFE_TITLE="$(echo "$TITLE" \
  | tr '[:upper:]' '[:lower:]' \
  | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g' \
  | cut -c1-60)"

TS="$(date +%Y%m%d-%H%M%S)"
BRANCH="codex/${SAFE_TITLE}-${TS}"

git fetch origin "$BASE_BRANCH"
git checkout -B "$BRANCH" "origin/$BASE_BRANCH"

mkdir -p .codex/pr-runs
RUN_DIR=".codex/pr-runs/${SAFE_TITLE}-${TS}"
mkdir -p "$RUN_DIR"

echo "Running Codex on branch: $BRANCH"

CODEX_SUMMARY="$RUN_DIR/codex-summary.md"
CODEX_LOG="$RUN_DIR/codex-events.jsonl"

# Codex edits files, but does not push or create the PR.
# -a never is appropriate here because the script itself controls the final commit/push/PR.
# Keep sandbox at workspace-write, not danger-full-access.
codex --ask-for-approval never exec \
  --sandbox workspace-write \
  --json \
  "$(cat "$TASK_FILE")" \
  > "$CODEX_LOG"

# Extract a readable final message if jq is available; otherwise keep JSONL only.
if command -v jq >/dev/null 2>&1; then
  jq -r 'select(.type=="turn.completed") // empty' "$CODEX_LOG" > "$CODEX_SUMMARY" || true
fi

echo "Changed files:"
git status --short

if [[ -z "$(git status --porcelain)" ]]; then
  echo "Codex made no changes. No PR created."
  exit 0
fi

echo "Checking whitespace..."
git diff --check

echo "Running tests..."
TEST_LOG="$RUN_DIR/test.log"
set +e
pytest -q 2>&1 | tee "$TEST_LOG"
TEST_STATUS="${PIPESTATUS[0]}"
set -e

if [[ "$TEST_STATUS" -ne 0 ]]; then
  echo "Tests failed. Leaving branch locally for inspection: $BRANCH"
  echo "Fix manually or rerun Codex. No commit/PR created."
  exit 1
fi

git add -A

COMMIT_MSG="$TITLE"
git commit -m "$COMMIT_MSG"

git push -u origin "$BRANCH"

BODY_FILE="$RUN_DIR/pr-body.md"
{
  echo "## Summary"
  echo
  echo "Automated Codex-assisted PR."
  echo
  echo "Task file: \`$TASK_FILE\`"
  echo
  echo "## Validation"
  echo
  echo "- \`git diff --check\`: passed"
  echo "- \`pytest -q\`: passed"
  echo
  echo "## Notes"
  echo
  echo "Please review the diff carefully before merging."
} > "$BODY_FILE"

PR_URL="$(gh pr create \
  --base "$BASE_BRANCH" \
  --head "$BRANCH" \
  --title "$TITLE" \
  --body-file "$BODY_FILE" \
  --draft)"

echo "Created draft PR:"
echo "$PR_URL"
