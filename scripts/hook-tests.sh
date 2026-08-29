#!/usr/bin/env bash
# scripts/hook-tests.sh — deterministic tests for .claude/hooks/*.
#
# The hooks are the only thing standing between an agent and a bad push, so
# they get tested like code. No model involved: pure stdin/exit-code checks,
# runs in about a second, safe to gate CI on.
#
# The "false positives" block matters as much as the denials. A guard that
# fires on `grep 'git push'` is a guard people learn to ignore.
set -uo pipefail
cd "$(dirname "$0")/.."
export CLAUDE_PROJECT_DIR="$PWD"

# Throwaway lockfile so a real push approval is never consumed by the tests.
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export SPINDLL_PUSH_LOCK="$TMP/push.allowed"

HOOK=".claude/hooks/guard-git.py"
PASS=0
FAIL=0

# want <expected-exit> <name> <command-string>
want() {
  local expect="$1" name="$2" cmd="$3" payload got
  payload=$(python3 -c \
    'import json,sys; print(json.dumps({"tool_name":"Bash","tool_input":{"command":sys.argv[1]}}))' \
    "$cmd")
  printf '%s' "$payload" | python3 "$HOOK" >/dev/null 2>&1
  got=$?
  if [ "$got" = "$expect" ]; then
    PASS=$((PASS + 1))
    printf '  ok   %s\n' "$name"
  else
    FAIL=$((FAIL + 1))
    printf '  FAIL %s (want exit %s, got %s)\n' "$name" "$expect" "$got" >&2
  fi
}

echo "== denials =="
want 2 "push without approval"           'git push origin HEAD'
want 2 "push, no refspec"                'git push'
want 2 "gh pr create"                    'gh pr create --fill'
want 2 "gh pr merge"                     'gh pr merge 86'
want 2 "gh release create"               'gh release create v9.9.9'
want 2 "--no-verify"                     'git commit --no-verify -m wip'
want 2 "push chained after a real cmd"   'cargo test && git push origin HEAD'
want 2 "push inside a subshell"          '(cd /tmp/wt && git push origin main)'
want 2 "push with env prefix"            'GIT_TRACE=1 git push origin HEAD'
want 2 "force-push to main"              'git push --force origin main'
want 2 "force-push shorthand to next"    'git push -f origin next'
want 2 "force-with-lease to main"        'git push --force-with-lease origin HEAD:main'
want 2 "git add -f PUNCHLIST"            'git add -f docs/PUNCHLIST.md'
want 2 "git add --force .refs"           'git add --force .refs/review/x.md'

echo "== allowances =="
want 0 "plain commit"                    'git commit -m "feat: thing"'
want 0 "status"                          'git status --short'
want 0 "normal add"                      'git add src/http.rs'
want 0 "add -f on a normal path"         'git add -f target/keep.txt'
want 0 "cargo build"                     'cargo build --features cli,http'
want 0 "fetch"                           'git fetch origin'
want 0 "log"                             'git log --oneline -5'

echo "== false positives the string-matching version got wrong =="
want 0 "grep for the phrase git push"    "grep -rn 'git push' docs/"
want 0 "echo mentioning git push"        'echo "remember: git push needs approval"'
want 0 "heredoc body mentioning push"    "cat > /tmp/a.md <<'EOF'
run git push --force origin main
EOF"
want 0 "test name containing next"       'cargo test next_token'
want 0 "branch named next-thing"         'git checkout -b feat/next-thing'
want 0 "no-verify inside a quoted docstring" 'echo "never pass --no-verify"'

echo "== the lockfile is consumed exactly once =="
touch "$SPINDLL_PUSH_LOCK"
want 0 "push with approval"              'git push origin HEAD'
want 2 "next push needs a fresh touch"   'git push origin HEAD'

echo
printf 'hook-tests: %d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
