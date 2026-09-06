#!/usr/bin/env python3
"""scripts/harness-lint.py — assert the harness docs and the harness agree.

Every finding this file checks for is one that actually happened here:

  * AGENTS.md documented a committed `nightshift.yml` that was never in the repo.
  * AGENTS.md said the ratchet ran `clippy -- -D warnings` while the script had
    been warn-only for months.
  * settings.local.json pre-approved `git push` while AGENTS.md called it the
    number one forbidden action.

Prose drifts from enforcement silently, and nobody notices until an agent acts on
the prose. So the agreement gets tested like code: deterministic, no model, under
a second, safe to gate CI and the ratchet on.

Run: python3 scripts/harness-lint.py
"""

import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

PASS, FAIL = 0, 0
FAILURES = []


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print("  ok   " + name)
    else:
        FAIL += 1
        FAILURES.append((name, detail))
        print("  FAIL %s%s" % (name, (" — " + detail) if detail else ""), file=sys.stderr)


def read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""


AGENTS = read("AGENTS.md")
RATCHET = read("scripts/ratchet.sh")
FANOUT = read("scripts/review-fanout.sh")

print("== AGENTS.md stays a front page ==")
lines = AGENTS.count("\n")
check(
    "AGENTS.md is under 120 lines (is %d)" % lines,
    0 < lines <= 120,
    "detail belongs in .claude/skills/, not here",
)

HARNESS_DIRS = {"scripts", ".claude", "docs", ".github", ".codex", "proto", "src", "bench"}

# Paths the docs name that must NOT exist in a clean checkout. The punchlist and
# worklog are per-host and untracked on purpose — AGENTS.md's own hard rule says
# never to commit them — so asserting they exist fails in CI and on every fresh
# clone, which is exactly what it did until this exemption landed. They get a
# real check below instead: that nobody has committed them.
LOCAL_ONLY = {"docs/PUNCHLIST.md", "docs/WORKLOG.md"}

print("== every path AGENTS.md names actually exists ==")
# Backticked things that look like repo paths: have a slash or a known suffix,
# no spaces, no glob beyond the skills wildcard.
CANDIDATE = re.compile(r"`([A-Za-z0-9_./*-]+\.(?:md|sh|py|toml|yml|rs|json))`")
seen = set()
for doc in ["AGENTS.md", "REVIEW.md"] + sorted(
    os.path.join(d, "SKILL.md")
    for d in (
        os.path.join(".claude/skills", x) for x in os.listdir(".claude/skills")
    )
    if os.path.isdir(d)
):
    for m in CANDIDATE.finditer(read(doc)):
        path = m.group(1)
        if path.startswith(("http", "~")) or "*" in path:
            continue
        # Only police the harness's own surface. Paths outside it are usually
        # files the docs tell you to *create* (a worktree-local .cargo/config.toml,
        # a param grid), and linting those turns the check into noise.
        if path in LOCAL_ONLY:
            continue
        first = path.split("/")[0]
        if "/" in path and first not in HARNESS_DIRS:
            continue
        if "/" not in path and not path.isupper() and not path.endswith(".md"):
            continue
        key = (doc, path)
        if key in seen:
            continue
        seen.add(key)
        check("%s -> %s" % (doc, path), os.path.exists(path), "referenced but missing")

print("== the per-host files stay per-host ==")
try:
    tracked = subprocess.run(
        ["git", "ls-files", "--"] + sorted(LOCAL_ONLY),
        capture_output=True, text=True, check=False,
    ).stdout.split()
except OSError:
    tracked = []  # no git here; the CI job and the ratchet both have one
check(
    "punchlist and worklog are not committed",
    not tracked,
    "tracked: %s — each host keeps its own copy, and a checkout of an older "
    "commit silently overwrites it" % ", ".join(tracked),
)

print("== the ratchet does what the docs say it does ==")
check(
    "ratchet runs clippy with -D warnings",
    re.search(r"cargo clippy[^\n]*-D warnings", RATCHET) is not None,
    "clippy is warn-only again — that is how ratchet=green stopped meaning anything",
)
check(
    "ratchet does not swallow clippy failure",
    not re.search(r"cargo clippy[^\n]*\|\|", RATCHET),
    "a `|| true` or `|| echo` after clippy reopens the gate",
)
check(
    "ratchet runs the hook tests",
    "hook-tests.sh" in RATCHET,
    "the hooks are the enforcement layer; untested they rot silently",
)
check(
    "ratchet cap is not raised above 90s",
    re.search(r'RATCHET_CAP:-(\d+)', RATCHET) is not None
    and int(re.search(r'RATCHET_CAP:-(\d+)', RATCHET).group(1)) <= 90,
    "trim the test filter instead",
)

print("== the review contract is a file, not a string in a script ==")
check("REVIEW.md exists", os.path.isfile("REVIEW.md"))
check(
    "review-fanout.sh reads REVIEW.md",
    "REVIEW.md" in FANOUT,
    "the prompt drifted back into the script",
)
check(
    "REVIEW.md caps nits",
    re.search(r"five nits|at most five", read("REVIEW.md"), re.I) is not None,
    "an uncapped review buries its own crit findings",
)

print("== hooks are wired, present and syntactically valid ==")
settings = {}
try:
    settings = json.loads(read(".claude/settings.json"))
except ValueError as exc:
    check("settings.json parses", False, str(exc))
else:
    check("settings.json parses", True)

hook_cmds = []
for event, entries in (settings.get("hooks") or {}).items():
    for entry in entries:
        for hook in entry.get("hooks", []):
            hook_cmds.append(hook.get("command", ""))

check("at least one PreToolUse hook is configured", bool(hook_cmds),
      "AGENTS.md claims the hard rules are enforced by hooks")

for cmd in hook_cmds:
    m = re.search(r"\.claude/hooks/([A-Za-z0-9_.-]+)", cmd)
    if not m:
        continue
    path = ".claude/hooks/" + m.group(1)
    check("hook exists: " + path, os.path.isfile(path))
    if path.endswith(".py") and os.path.isfile(path):
        try:
            compile(read(path), path, "exec")
            check("hook compiles: " + path, True)
        except SyntaxError as exc:
            check("hook compiles: " + path, False, str(exc))

print("== permissions do not pre-approve what AGENTS.md forbids ==")
for label, path in [
    ("settings.json", ".claude/settings.json"),
    ("settings.local.json", ".claude/settings.local.json"),
]:
    if not os.path.isfile(path):
        continue
    try:
        allow = json.loads(read(path)).get("permissions", {}).get("allow", [])
    except ValueError:
        continue
    bad = [
        e
        for e in allow
        if re.match(r"^Bash\(git (\*|push|reset --hard|filter-branch)", e)
    ]
    check(
        "%s does not auto-approve push/destructive git" % label,
        not bad,
        "found: %s" % ", ".join(bad),
    )

print("== every slash command AGENTS.md advertises exists ==")
for m in re.finditer(r"^\| `/([a-z-]+)`", AGENTS, re.M):
    name = m.group(1)
    check("/%s" % name, os.path.isfile(".claude/commands/%s.md" % name))

print("== skills are addressable ==")
for entry in sorted(os.listdir(".claude/skills")):
    skill = os.path.join(".claude/skills", entry, "SKILL.md")
    if not os.path.isfile(skill):
        continue
    body = read(skill)
    check(
        "%s has name+description frontmatter" % entry,
        body.startswith("---") and "name:" in body and "description:" in body,
        "without frontmatter the skill is never surfaced",
    )
    check(
        "%s is linked from AGENTS.md" % entry,
        entry in AGENTS,
        "an unlinked skill is a file nobody opens",
    )

print()
print("harness-lint: %d passed, %d failed" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
