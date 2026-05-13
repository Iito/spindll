---
description: Karpathy autoloop. Sweep a perf metric across a parameter grid, keep winners, log all trials.
allowed-tools: Bash, Read, Write
argument-hint: <metric> <grid.jsonl>
---

Karpathy autoresearch loop. Source: `AGENTS.md` "Karpathy autoloop", PDF slide 18.

Metric (arg 1): one of `prompt_eval_tps`, `decode_tps`, `peak_rss_mb`, `p50_ms`, `p95_ms`.
Grid (arg 2): JSONL at `.refs/autoloop/grids/<name>.jsonl`. Each line: `{"label": "...", "patch_cmd": "...", "env": {"K":"V"}}`.

Run: `bash scripts/autoloop.sh "$1" "$2"`.

What it does (already implemented in the script — just kick it):

```
baseline = bench(median of 3)
for row in grid:
  apply patch
  cargo build --release --features cli,http,$BACKEND
  trial = bench(median of 3)
  if trial improves on baseline by >= +2 %:
    keep + commit on autoloop/<date> branch
  else:
    revert
  append row + trial to .refs/autoloop/log-<date>.jsonl
```

After the loop ends (grid exhausted or `WALL_CAP_SEC` hit), summarize:
- Baseline → final delta.
- Top 3 winners by delta.
- Any reverts that crashed the build (those become new `docs/PUNCHLIST.md` items).

Never push the autoloop branch automatically. User decides whether a winner ships.

Caveat: this loop must run on **ubuntu** (the script refuses to run on Darwin). Mac is reserved for non-agent work + MLX validation.
