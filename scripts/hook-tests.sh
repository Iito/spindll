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
want 2 "push after git -C <dir>"         'git -C /tmp/wt push origin HEAD'
want 2 "push after git -c <k=v>"         'git -c core.sshCommand=ssh push origin HEAD'
want 2 "push after --git-dir=<path>"     'git --git-dir=.git push origin HEAD'
want 2 "push after --work-tree <path>"   'git --work-tree /tmp/wt push'
want 2 "push after stacked git options"  'git --no-pager -C . -c x.y=z push'
want 2 "git add -f PUNCHLIST after -C"   'git -C . add -f docs/PUNCHLIST.md'
want 2 "gh pr create after -R <repo>"    'gh pr -R Iito/spindll create --fill'
want 2 "gh pr merge after --repo=<repo>" 'gh pr --repo=Iito/spindll merge 86'
want 2 "gh release create after -R"      'gh release -R Iito/spindll create v9.9.9'
want 2 "push after env -i"               'env -i git push origin HEAD'
want 2 "push after env -u NAME"          'env -u GIT_DIR git push origin HEAD'
want 2 "push after env -S <cmd>"         "env -S 'git push origin HEAD'"
want 2 "push after command -p"           'command -p git push origin HEAD'
want 2 "push after time -p"              'time -p git push origin HEAD'
want 2 "push after sudo -u <user>"       'sudo -u mansa git push origin HEAD'
want 2 "push after timeout <secs>"       'timeout 60 git push origin HEAD'
want 2 "push after timeout -k <d> <d>"   'timeout -k 5 60 git push'
want 2 "push after nice -n <n>"          'nice -n 10 git push'
want 2 "push inside bash -c"             "bash -c 'git push origin HEAD'"
want 2 "push inside sh -lc"              "sh -lc 'cd /tmp/wt && git push'"
want 2 "push inside zsh -o <opt> -c"     "zsh -o errexit -c 'git push'"
want 2 "push inside eval"                "eval 'git push origin HEAD'"
want 2 "push inside env -i bash -c"      "env -i bash -c 'git push'"
want 2 "push in a here-string to bash"   "bash <<< 'git push'"
want 2 "push in a heredoc fed to bash"   "bash <<'EOF'
git push origin HEAD
EOF"
want 2 "push in a heredoc piped to sh"   "cat <<'EOF' | sh
git push
EOF"
want 2 "push via /usr/bin/git"           '/usr/bin/git push origin HEAD'
want 2 "push via /usr/bin/env git"       '/usr/bin/env git push'
want 2 "gh pr create via a full path"    '/opt/homebrew/bin/gh pr create --fill'
want 2 "push on the line after cd"       "cd /tmp/wt
git push origin HEAD"
want 2 "push after a commented line"     "cargo test # don't push yet
git push"
want 2 "push across a line continuation" 'git \
  push origin HEAD'
want 2 "commit -n"                       'git commit -n -m wip'
want 2 "commit -n inside a cluster"      'git commit -anm wip'
want 2 "abbreviated --no-verify"         'git commit --no-veri -m wip'
want 2 "-c core.hooksPath on commit"     'git -c core.hooksPath=/dev/null commit -m wip'
want 2 "-c core.hooksPath in any case"   'git -c CORE.HOOKSPATH=/tmp commit -m wip'
want 2 "--config-env core.hooksPath"     'git --config-env=core.hooksPath=H commit -m wip'
want 2 "git config sets core.hooksPath"  'git config core.hooksPath /dev/null'
want 2 "git config unsets core.hooksPath" 'git config --unset core.hooksPath'

echo "== allowances =="
want 0 "plain commit"                    'git commit -m "feat: thing"'
want 0 "status"                          'git status --short'
want 0 "normal add"                      'git add src/http.rs'
want 0 "add -f on a normal path"         'git add -f target/keep.txt'
want 0 "cargo build"                     'cargo build --features cli,http'
want 0 "fetch"                           'git fetch origin'
want 0 "log"                             'git log --oneline -5'
want 0 "status after git -C <dir>"       'git -C /tmp/wt status --short'
want 0 "gh pr list after -R <repo>"      'gh pr -R Iito/spindll list'
want 0 "env -i without git"              'env -i cargo test'
want 0 "timeout without git"             'timeout 60 cargo test'
want 0 "bash -c without git"             "bash -c 'cargo test'"
want 0 "bash running a script file"      'bash scripts/ratchet.sh'
want 0 "lines without git"               "cd /tmp
ls -la"
want 0 "git log -n"                      'git log -n 5'
want 0 "read core.hooksPath"             'git config --get core.hooksPath'
want 0 "legacy read of core.hooksPath"   'git config core.hooksPath'

echo "== false positives the string-matching version got wrong =="
want 0 "grep for the phrase git push"    "grep -rn 'git push' docs/"
want 0 "echo mentioning git push"        'echo "remember: git push needs approval"'
want 0 "heredoc body mentioning push"    "cat > /tmp/a.md <<'EOF'
run git push --force origin main
EOF"
want 0 "test name containing next"       'cargo test next_token'
want 0 "branch named next-thing"         'git checkout -b feat/next-thing'
want 0 "no-verify inside a quoted docstring" 'echo "never pass --no-verify"'
want 0 "-C into a directory named push"  'git -C push status'
want 0 "-c value mentioning push"        'git -c alias.p=push log --oneline -1'
want 0 "commit message after -m"         'git commit -m "-n flag docs"'
want 0 "commit -mn is the message n"     'git commit -mn'
want 0 "quoted newline before push text" 'echo "a
git push"'
want 0 "comment line mentioning push"    "# remember: git push needs approval
cargo test"
want 0 "heredoc to python mentioning push" "python3 - <<'EOF'
print('git push')
EOF"

echo "== the lockfile is consumed exactly once =="
touch "$SPINDLL_PUSH_LOCK"
want 0 "push with approval"              'git push origin HEAD'
want 2 "next push needs a fresh touch"   'git push origin HEAD'
touch "$SPINDLL_PUSH_LOCK"
want 0 "push after -C with approval"     'git -C /tmp/wt push origin HEAD'
want 2 "and that consumed it too"        'git -C /tmp/wt push origin HEAD'
touch "$SPINDLL_PUSH_LOCK"
want 0 "push in bash -c with approval"   "bash -c 'git push origin HEAD'"
want 2 "and that consumed it as well"    "bash -c 'git push origin HEAD'"

echo
printf 'hook-tests: %d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
