#!/usr/bin/env python3
"""scripts/bands-check.py — the deterministic half of the maintain stage.

Records perf samples and decides, from numbers alone, whether a run has left its
control band and at what tier. An agent is invoked only *after* this says so; it
never gets a vote on whether a regression happened.

Bands live in bands.yaml. Samples live in .refs/bands/<metric>.jsonl (a local
sink — never committed).

    scripts/bands-check.py record decode_tps 38.8
    scripts/bands-check.py check decode_tps
    scripts/bands-check.py check --json
    scripts/bands-check.py bench llama3.2:1b        # run the bench, record all metrics, check
    scripts/bands-check.py check --fail-at 2        # non-zero exit at tier 2+, for a scheduler

Exit status is 0 unless --fail-at is given (or something actually broke), so a
plain `check` in a cron job does not look like a failure.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANDS = os.path.join(ROOT, "bands.yaml")
STORE = os.path.join(ROOT, ".refs", "bands")


def load_bands():
    try:
        import yaml
    except ImportError:
        sys.exit(
            "bands-check: PyYAML is required to read bands.yaml.\n"
            "  pip install pyyaml   (or apt install python3-yaml)"
        )
    if not os.path.isfile(BANDS):
        sys.exit("bands-check: bands.yaml is missing — nothing defines the bands")
    with open(BANDS) as fh:
        cfg = yaml.safe_load(fh)
    cfg["tiers"] = sorted(cfg.get("tiers", []), key=lambda t: t["sigma"], reverse=True)
    return cfg


def path_for(metric):
    return os.path.join(STORE, "%s.jsonl" % metric)


def record(metric, value, label=None):
    os.makedirs(STORE, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "value": float(value),
    }
    if label:
        row["label"] = label
    with open(path_for(metric), "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def samples(metric):
    out = []
    try:
        with open(path_for(metric)) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def evaluate(metric, cfg):
    """Judge the newest sample against the ones before it."""
    spec = cfg["metrics"].get(metric)
    if spec is None:
        return {"metric": metric, "status": "unknown-metric"}

    rows = samples(metric)
    if not rows:
        return {"metric": metric, "status": "no-samples"}

    latest = rows[-1]["value"]
    history = [r["value"] for r in rows[:-1]][-cfg["window"] :]

    if len(history) < cfg["min_samples"]:
        return {
            "metric": metric,
            "status": "baselining",
            "latest": latest,
            "have": len(history),
            "need": cfg["min_samples"],
        }

    median = statistics.median(history)
    stdev = statistics.pstdev(history)

    # A flat history has zero spread; any move is infinitely many sigma, which
    # is useless. Fall back to the relative gate alone.
    if stdev == 0:
        sigma = float("inf") if latest != median else 0.0
    else:
        sigma = (latest - median) / stdev

    # Point sigma in the direction of "worse", so a big improvement never pages.
    if spec["direction"] == "higher_is_better":
        sigma = -sigma
    delta_rel = abs(latest - median) / median if median else 0.0

    tier = 0
    action = "none"
    for t in cfg["tiers"]:
        if sigma >= t["sigma"] and delta_rel >= cfg["min_relative_change"]:
            tier = t["sigma"]
            action = t["action"]
            break

    return {
        "metric": metric,
        "status": "ok",
        "latest": latest,
        "median": round(median, 4),
        "stdev": round(stdev, 4),
        "sigma": round(sigma, 2) if sigma != float("inf") else "inf",
        "relative_change": round(delta_rel, 4),
        "unit": spec.get("unit", ""),
        "direction": spec["direction"],
        "tier": tier,
        "action": action,
        "samples": len(history) + 1,
    }


def render(v):
    if v["status"] == "no-samples":
        return "  %-16s no samples yet" % v["metric"]
    if v["status"] == "baselining":
        return "  %-16s baselining (%d/%d samples)" % (
            v["metric"],
            v["have"],
            v["need"],
        )
    if v["status"] == "unknown-metric":
        return "  %-16s not defined in bands.yaml" % v["metric"]
    mark = "ok  " if v["tier"] == 0 else "TIER%d" % v["tier"]
    return "  %-16s %s  %.4g %s (median %.4g, %s sigma worse, %.1f%%) -> %s" % (
        v["metric"],
        mark,
        v["latest"],
        v["unit"],
        v["median"],
        v["sigma"],
        v["relative_change"] * 100,
        v["action"],
    )


def run_bench(model, cfg):
    binary = os.path.join(ROOT, "target", "release", "spindll")
    if not os.path.isfile(binary):
        sys.exit("bands-check: %s not built — cargo build --release first" % binary)
    proc = subprocess.run(
        [binary, "bench", model, "--json"], capture_output=True, text=True
    )
    if proc.returncode != 0:
        sys.exit("bands-check: bench failed:\n" + proc.stderr.strip())
    try:
        data = json.loads(proc.stdout)
    except ValueError:
        sys.exit("bands-check: bench did not emit JSON:\n" + proc.stdout[:400])
    recorded = []
    for metric in cfg["metrics"]:
        if metric in data and data[metric] is not None:
            record(metric, data[metric], label=model)
            recorded.append(metric)
    return recorded


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("record", help="append one sample")
    p_rec.add_argument("metric")
    p_rec.add_argument("value", type=float)
    p_rec.add_argument("--label")

    p_chk = sub.add_parser("check", help="evaluate the newest sample")
    p_chk.add_argument("metric", nargs="?")
    p_chk.add_argument("--json", action="store_true")
    p_chk.add_argument(
        "--fail-at",
        type=int,
        default=0,
        metavar="N",
        help="exit non-zero when any metric reaches tier N or above",
    )

    p_bch = sub.add_parser("bench", help="run the bench, record every metric, then check")
    p_bch.add_argument("model")
    p_bch.add_argument("--json", action="store_true")
    p_bch.add_argument("--fail-at", type=int, default=0, metavar="N")

    args = ap.parse_args()
    cfg = load_bands()

    if args.cmd == "record":
        row = record(args.metric, args.value, args.label)
        print("recorded %s = %s at %s" % (args.metric, row["value"], row["ts"]))
        return 0

    if args.cmd == "bench":
        done = run_bench(args.model, cfg)
        print("recorded: %s" % (", ".join(done) or "nothing"))
        metrics = done
    else:
        metrics = [args.metric] if args.metric else list(cfg["metrics"])

    verdicts = [evaluate(m, cfg) for m in metrics]

    if args.json:
        print(json.dumps(verdicts, indent=2))
    else:
        print("== control bands ==")
        for v in verdicts:
            print(render(v))
        worst = max((v.get("tier", 0) for v in verdicts), default=0)
        if worst == 0:
            print("all metrics inside their bands")
        else:
            action = next(v["action"] for v in verdicts if v.get("tier") == worst)
            print("highest breach: tier %d -> %s (see bands.yaml)" % (worst, action))

    if args.fail_at:
        worst = max((v.get("tier", 0) for v in verdicts), default=0)
        if worst >= args.fail_at:
            return worst
    return 0


if __name__ == "__main__":
    sys.exit(main())
