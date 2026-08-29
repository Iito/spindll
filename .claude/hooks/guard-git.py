#!/usr/bin/env python3
"""PreToolUse(Bash) guard.

Turns the AGENTS.md hard rules from prose an agent is trusted to remember into
gates it cannot walk past. Deny = exit 2 with the reason on stderr; Claude Code
feeds that back to the model instead of running the command.

Precision is the whole point. The first version of this matched the raw command
string and cheerfully blocked `grep 'git push'`, which is how you train someone
to stop reading guard output. So the command is tokenised with quotes honoured
and heredoc bodies dropped, and rules only fire on segments that actually
invoke git or gh.

Deliberately NOT enforced here: which branch you commit on. Both quickfix->main
and feature->next are real flows in this repo, so that stays a judgment call.
"""

import json
import os
import re
import shlex
import subprocess
import sys

LOCK = os.environ.get(
    "SPINDLL_PUSH_LOCK",
    os.path.expanduser("~/.local/state/spindll-harness/push.allowed"),
)

HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
WRAPPERS = {"env", "sudo", "command", "nohup", "time", "exec"}
SEPARATORS = set(";&|()")
FORCE_FLAGS = {"--force", "-f", "--force-with-lease"}
PROTECTED = {"main", "next"}
LOCAL_ONLY = ("PUNCHLIST", "WORKLOG", ".refs")


def deny(msg):
    print("guard-git: " + msg, file=sys.stderr)
    sys.exit(2)


def strip_heredocs(cmd):
    """Drop heredoc bodies. Their text is data, not commands to police."""
    lines = cmd.split("\n")
    out, i = [], 0
    while i < len(lines):
        m = HEREDOC.search(lines[i])
        out.append(HEREDOC.sub("", lines[i]) if m else lines[i])
        i += 1
        if m:
            delim = m.group(2)
            while i < len(lines) and lines[i].strip() != delim:
                i += 1
            i += 1
    return "\n".join(out)


def segments(cmd):
    """Split into command segments, honouring quotes. None if unparseable."""
    lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    try:
        tokens = list(lex)
    except ValueError:
        return None
    segs, cur = [], []
    for tok in tokens:
        if tok and set(tok) <= SEPARATORS:
            if cur:
                segs.append(cur)
                cur = []
        else:
            cur.append(tok)
    if cur:
        segs.append(cur)
    return segs


def head(seg):
    """(command, args) with leading VAR=val assignments and wrappers skipped."""
    for i, tok in enumerate(seg):
        if ASSIGN.match(tok) or tok in WRAPPERS:
            continue
        return tok, seg[i + 1 :]
    return None, []


def git(*args):
    try:
        return subprocess.run(
            ("git",) + args, capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        return ""


def require_approval(what):
    if not os.path.isfile(LOCK):
        deny(
            "%s needs explicit approval. Ask the user to run:\n"
            "  mkdir -p %s && touch %s\n"
            "One touch authorises one publish; this hook consumes it."
            % (what, os.path.dirname(LOCK), LOCK)
        )
    os.remove(LOCK)
    print("guard-git: consumed the publish approval at " + LOCK, file=sys.stderr)


def check(cmd, args):
    argset = set(args)

    if "--no-verify" in argset:
        deny(
            "--no-verify is banned. The pre-commit hook enforces this repo's "
            "commit identity, and stepping around it is how the wrong author "
            "lands in history."
        )

    if cmd == "git" and args[:1] == ["push"]:
        if argset & FORCE_FLAGS:
            current = git("branch", "--show-current")
            named = {a.rsplit(":", 1)[-1] for a in args}
            if (named & PROTECTED) or current in PROTECTED:
                deny(
                    "force-push touching main/next is refused outright. "
                    "Run it in your own terminal if you truly mean it."
                )
        require_approval("push")

    if cmd == "gh" and args[:2] in (["pr", "create"], ["pr", "merge"]):
        require_approval("opening/merging a PR")

    if cmd == "gh" and args[:2] == ["release", "create"]:
        require_approval("cutting a release")

    if cmd == "git" and args[:1] == ["commit"] and "--amend" in argset:
        on_remote = git("branch", "-r", "--contains", "HEAD")
        if on_remote:
            deny(
                "HEAD is already published on: %s. Never amend a pushed "
                "commit — add a new one."
                % ", ".join(on_remote.split())
            )

    if cmd == "git" and args[:1] == ["add"] and (argset & {"-f", "--force"}):
        if any(marker in a for a in args for marker in LOCAL_ONLY):
            deny(
                "docs/PUNCHLIST.md, docs/WORKLOG.md and .refs/ are per-host "
                "local files. They are never committed."
            )


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    if payload.get("tool_name") != "Bash":
        sys.exit(0)
    raw = payload.get("tool_input", {}).get("command", "")
    if not raw.strip():
        sys.exit(0)

    os.chdir(os.environ.get("CLAUDE_PROJECT_DIR", "."))

    segs = segments(strip_heredocs(raw))
    if segs is None:
        # Unbalanced quotes. Keep only the highest-stakes gate closed rather
        # than guessing at the rest.
        if re.search(r"\bgit\s+push\b|\bgh\s+pr\s+(create|merge)\b", raw):
            require_approval("push")
        sys.exit(0)

    for seg in segs:
        cmd, args = head(seg)
        if cmd in ("git", "gh"):
            check(cmd, args)

    sys.exit(0)


if __name__ == "__main__":
    main()
