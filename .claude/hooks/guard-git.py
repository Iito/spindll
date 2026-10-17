#!/usr/bin/env python3
"""PreToolUse(Bash) guard.

Turns the AGENTS.md hard rules from prose an agent is trusted to remember into
gates it cannot walk past. Deny = exit 2 with the reason on stderr; Claude Code
feeds that back to the model instead of running the command.

Precision is the whole point. The first version of this matched the raw command
string and cheerfully blocked `grep 'git push'`, which is how you train someone
to stop reading guard output. So the command is read the way the shell reads
it: quotes honoured, comments and heredoc bodies dropped (unless a shell runs
the body as its script), one command per line, wrappers like `env` and
`timeout` looked through, and nested `sh -c` / `eval` strings read in turn.
Rules only fire on segments that actually invoke git or gh.

Deliberately NOT enforced here: which branch you commit on. Both quickfix->main
and feature->next are real flows in this repo, so that stays a judgment call.
Nor can it see commands assembled at runtime, inside scripts, behind aliases or
in raw `gh api` calls: it stops accidents, it is not a sandbox.
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
# Commands that run the command after them, each with its options that take a
# value, so `env -u NAME git push` and `sudo -u me git push` still read as git.
WRAPPERS = {
    "env": {"-u", "--unset", "-C", "--chdir", "-P"},
    "sudo": {"-u", "--user", "-g", "--group", "-C", "-D", "--chdir", "-h",
             "--host", "-p", "--prompt", "-r", "-t", "-T", "-U", "-R"},
    "command": set(),
    "nohup": set(),
    "time": {"-o", "-f"},
    "exec": {"-a"},
    "nice": {"-n"},
    "timeout": {"-s", "--signal", "-k", "--kill-after"},
    "caffeinate": {"-t", "-w"},
}
SHELLS = {"sh", "bash", "zsh", "dash", "ksh"}
SHELL_VALUE_OPTS = {"-o", "+o", "-O", "+O", "--rcfile", "--init-file"}
SEPARATORS = set(";&|()")
FORCE_FLAGS = {"--force", "-f", "--force-with-lease"}
PROTECTED = {"main", "next"}
LOCAL_ONLY = ("PUNCHLIST", "WORKLOG", ".refs")
# Options that can sit between the command and the subcommand the rules key
# on, and that take the next token as their value: `git -C <dir> push`,
# `gh pr -R <repo> create`. Any other leading `-...` token (`--no-pager`,
# `--git-dir=<path>`) is self-contained.
GIT_VALUE_OPTS = {
    "-C", "-c", "--git-dir", "--work-tree", "--namespace",
    "--config-env", "--super-prefix", "--attr-source",
}
GH_VALUE_OPTS = {"-R", "--repo"}
# `git commit` options whose value could otherwise pass for -n: `-mn` is the
# message "n", `-m -n` the message "-n". -S and -u take an optional attached one.
COMMIT_VALUE_SHORTS = set("mFCct")
COMMIT_OPTIONAL_SHORTS = set("Su")
COMMIT_VALUE_LONGS = {
    "--message", "--file", "--reuse-message", "--reedit-message", "--template",
    "--author", "--date", "--cleanup", "--fixup", "--squash", "--trailer",
    "--pathspec-from-file",
}
CONFIG_WRITE_FLAGS = {"--unset", "--unset-all", "--replace-all", "--add"}
CONFIG_READ_FLAGS = {
    "--get", "--get-all", "--get-regexp", "--get-urlmatch", "-l", "--list",
}
NO_VERIFY_WHY = (
    "The pre-commit hook enforces this repo's commit identity, and stepping "
    "around it is how the wrong author lands in history."
)


def deny(msg):
    print("guard-git: " + msg, file=sys.stderr)
    sys.exit(2)


def strip_heredocs(cmd):
    """Drop heredoc bodies: their text is data, not commands to police.

    Returns the remaining text and the bodies a shell reads as its script
    (`bash <<EOF`, `cat <<EOF | sh`), which are commands after all.
    """
    lines = cmd.split("\n")
    out, scripts, i = [], [], 0
    while i < len(lines):
        m = HEREDOC.search(lines[i])
        line = HEREDOC.sub("", lines[i]) if m else lines[i]
        out.append(line)
        i += 1
        if m:
            delim, body = m.group(2), []
            while i < len(lines) and lines[i].strip() != delim:
                body.append(lines[i])
                i += 1
            i += 1
            if feeds_shell(line):
                scripts.append("\n".join(body))
    return "\n".join(out), scripts


def logical_lines(cmd):
    """Split where the shell ends a command at a newline.

    Not inside quotes, not after a backslash (a backslash-newline joins the
    lines), and not inside a comment, which is dropped here.
    """
    lines, cur, quote, comment, i = [], [], None, False, 0
    while i < len(cmd):
        c = cmd[i]
        if comment:
            if c == "\n":
                comment = False
                lines.append("".join(cur))
                cur = []
        elif quote:
            cur.append(c)
            if c == "\\" and quote == '"' and i + 1 < len(cmd):
                i += 1
                cur.append(cmd[i])
            elif c == quote:
                quote = None
        elif c == "\\" and i + 1 < len(cmd):
            i += 1
            if cmd[i] != "\n":
                cur += [c, cmd[i]]
        elif c in "'\"":
            quote = c
            cur.append(c)
        elif c == "#" and (not cur or cur[-1] in " \t;&|()"):
            comment = True
        elif c == "\n":
            lines.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    lines.append("".join(cur))
    return lines


def segments(cmd):
    """Split into command segments, honouring quotes. None if unparseable."""
    segs = []
    for line in logical_lines(cmd):
        lex = shlex.shlex(line, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        # logical_lines already dropped real comments. shlex would also treat
        # a `#` inside a word as one and swallow the rest of the line with it.
        lex.commenters = ""
        try:
            tokens = list(lex)
        except ValueError:
            return None
        cur = []
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
    """(command, args) past leading VAR=val assignments and wrappers.

    A wrapper's own options go with it, and the value of those that take one,
    so `env -u NAME git push` and `timeout 60 git push` read as git. The
    command line `env -S '<cmd>'` splits out is handed on as an eval. Commands
    compare by basename, so `/usr/bin/git` is git.
    """
    i = 0
    while i < len(seg):
        if ASSIGN.match(seg[i]):
            i += 1
            continue
        name = os.path.basename(seg[i])
        if name not in WRAPPERS:
            return name, seg[i + 1 :]
        i += 1
        while i < len(seg) and seg[i].startswith("-"):
            opt = seg[i]
            if name == "env" and opt in ("-S", "--split-string"):
                return "eval", seg[i + 1 :]
            if name == "env" and opt.startswith("--split-string="):
                return "eval", [opt.split("=", 1)[1]] + seg[i + 1 :]
            if name == "env" and opt.startswith("-S"):
                return "eval", [opt[2:]] + seg[i + 1 :]
            i += 2 if opt in WRAPPERS[name] else 1
        if name == "timeout":
            i += 1  # the duration comes before the command
    return None, []


def shell_scripts(args):
    """Command strings a shell runs from its arguments.

    `-c '<cmd>'`, whatever flags sit around it, and a `<<< '<cmd>'` here-string.
    """
    scripts, has_c, skip = [], False, False
    for tok in args:
        if skip:
            skip = False
        elif tok in SHELL_VALUE_OPTS:
            skip = True
        elif tok.startswith("--"):
            continue
        elif tok[:1] in ("-", "+") and len(tok) > 1:
            has_c = has_c or "c" in tok[1:]
        else:
            if has_c:
                scripts.append(tok)
            break
    if "<<<" in args:
        at = args.index("<<<")
        scripts += args[at + 1 : at + 2]
    return scripts


def shell_reads_stdin(args):
    """Whether a shell takes its script from stdin.

    That means no `-c`, and no script file unless `-s` makes the operands
    arguments.
    """
    skip = False
    for tok in args:
        if skip:
            skip = False
        elif tok in SHELL_VALUE_OPTS:
            skip = True
        elif tok.startswith("--"):
            continue
        elif tok[:1] in ("-", "+") and len(tok) > 1:
            if "c" in tok[1:]:
                return False
            if "s" in tok[1:]:
                return True
        else:
            return False
    return True


def feeds_shell(line):
    """Whether a heredoc on this line ends up as a shell's script."""
    for seg in segments(line) or []:
        cmd, args = head(seg)
        if cmd in SHELLS and shell_reads_stdin(args):
            return True
    return False


def strip_options(args, value_opts):
    """args with leading options dropped; those in value_opts take the next token too."""
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in value_opts else 1
    return args[i:]


def git_config_keys(args):
    """Config keys set by git's own leading `-c <k>=<v>` / `--config-env`.

    Lowercased: config section and key names are case-insensitive.
    """
    keys, i = [], 0
    while i < len(args) and args[i].startswith("-"):
        opt = args[i]
        if opt in ("-c", "--config-env") and i + 1 < len(args):
            keys.append(args[i + 1].split("=", 1)[0].lower())
        elif opt.startswith("--config-env="):
            keys.append(opt[len("--config-env=") :].split("=", 1)[0].lower())
        i += 2 if opt in GIT_VALUE_OPTS else 1
    return keys


def commit_skips_hooks(args):
    """Whether `git commit` args carry -n, the short form of --no-verify.

    Alone or in a cluster like `-anm`, but not as the value of another option.
    """
    skip = False
    for a in args:
        if skip:
            skip = False
        elif a == "--":
            return False
        elif a.startswith("--"):
            skip = a in COMMIT_VALUE_LONGS
        elif a.startswith("-") and len(a) > 1:
            for k, ch in enumerate(a[1:], 1):
                if ch == "n":
                    return True
                if ch in COMMIT_VALUE_SHORTS:
                    skip = k == len(a) - 1  # no attached value: it is the next token
                    break
                if ch in COMMIT_OPTIONAL_SHORTS:
                    break
    return False


def writes_hooks_path(args):
    """Whether `git config` args set, unset or replace core.hooksPath."""
    lowered = [a.lower() for a in args]
    if "core.hookspath" not in lowered:
        return False
    if args[:1] in (["set"], ["unset"]) or set(args) & CONFIG_WRITE_FLAGS:
        return True
    if args[:1] in (["get"], ["list"]) or set(args) & CONFIG_READ_FLAGS:
        return False
    # Legacy `git config <key> [<value>]`: a value after the key is a write.
    after = args[lowered.index("core.hookspath") + 1 :]
    return any(not a.startswith("-") for a in after)


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
    # git accepts any unambiguous prefix of a long option; --no-veri is the
    # shortest one that is not also --no-verbose.
    if any(a.startswith("--no-veri") and "--no-verify".startswith(a) for a in args):
        deny("--no-verify is banned. " + NO_VERIFY_WHY)

    if cmd == "git" and "core.hookspath" in git_config_keys(args):
        deny("Overriding core.hooksPath skips the pre-commit hook just like "
             "--no-verify. " + NO_VERIFY_WHY)

    # The rules below key on the subcommand's position, so read it past the
    # options allowed in front of it. Otherwise `git -C <dir> push` is not a
    # push and walks straight past the approval gate.
    if cmd == "git":
        args = strip_options(args, GIT_VALUE_OPTS)
    elif cmd == "gh":
        args = args[:1] + strip_options(args[1:], GH_VALUE_OPTS)
    argset = set(args)

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

    if cmd == "git" and args[:1] == ["commit"] and commit_skips_hooks(args[1:]):
        deny("git commit -n is --no-verify, which is banned. " + NO_VERIFY_WHY)

    if cmd == "git" and args[:1] == ["commit"] and "--amend" in argset:
        on_remote = git("branch", "-r", "--contains", "HEAD")
        if on_remote:
            deny(
                "HEAD is already published on: %s. Never amend a pushed "
                "commit — add a new one."
                % ", ".join(on_remote.split())
            )

    if cmd == "git" and args[:1] == ["config"] and writes_hooks_path(args[1:]):
        deny("Changing core.hooksPath skips the pre-commit hook just like "
             "--no-verify. " + NO_VERIFY_WHY)

    if cmd == "git" and args[:1] == ["add"] and (argset & {"-f", "--force"}):
        if any(marker in a for a in args for marker in LOCAL_ONLY):
            deny(
                "docs/PUNCHLIST.md, docs/WORKLOG.md and .refs/ are per-host "
                "local files. They are never committed."
            )


def scan(raw):
    """Police every git/gh invocation in a command line, nested ones included."""
    text, scripts = strip_heredocs(raw)
    segs = segments(text)
    if segs is None:
        # Unbalanced quotes. Keep only the highest-stakes gate closed rather
        # than guessing at the rest.
        if re.search(r"\bgit\s+push\b|\bgh\s+pr\s+(create|merge)\b", raw):
            require_approval("push")
        return

    for seg in segs:
        cmd, args = head(seg)
        if cmd in ("git", "gh"):
            check(cmd, args)
        elif cmd == "eval":
            scan(" ".join(args))
        elif cmd in SHELLS:
            for script in shell_scripts(args):
                scan(script)
    for script in scripts:
        scan(script)


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
    scan(raw)
    sys.exit(0)


if __name__ == "__main__":
    main()
