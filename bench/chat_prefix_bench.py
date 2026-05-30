#!/usr/bin/env python3
"""Measure the MLX prompt-cache prefix-reuse win on a multi-turn chat workload.

The old prompt cache only hit on an *identical* token sequence, so every new
turn in a conversation triggered a full cold prefill of the whole prompt. With
prefix reuse, turn N reuses the state from turn N-1 and only prefills the new
turn's tokens.

For each rep this script:
  - sends turn 1 of a conversation whose (large) system prompt carries a rep-
    unique marker at the very front, so nothing is cached  → "cold": full prefill
  - sends turn 2 of the same conversation                  → "warm": prefix hit,
    prefill only the new turn (a1 + user2 + header)

and times TTFT (time to first streamed token) for each.

Run a spindll server first, e.g.:
    cargo build --features cli,http,mlx --bin spindll
    ./target/debug/spindll serve --http-port 8091 --budget 8G

Usage:
    python3 bench/chat_prefix_bench.py <model> [--url ...] [--reps 6] [--ctx-tokens 1800]
"""
import argparse, json, statistics, sys, time, urllib.request

# ~one "sentence" ≈ ~14 tokens; repeated to reach the target prompt size.
SENT = "Conventions agreed earlier in this project that you must always follow when answering. "


def ttft_ms(url, model, messages, max_tokens=4):
    body = json.dumps({
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data:") and line != "data: [DONE]":
                try:
                    delta = json.loads(line[5:].strip())["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta.get("content"):
                    return (time.perf_counter() - start) * 1000.0
            elif line.startswith("{"):
                return (time.perf_counter() - start) * 1000.0
    return (time.perf_counter() - start) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--url", default="http://localhost:8091")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--ctx-tokens", type=int, default=1800)
    args = ap.parse_args()

    filler = SENT * max(1, args.ctx_tokens // 14)

    sys.stderr.write(f"warming up {args.model} ...\n")
    ttft_ms(args.url, args.model, [{"role": "user", "content": "hi"}])

    cold, warm = [], []
    for rep in range(args.reps):
        # rep-unique marker AT THE FRONT → shares no prefix with anything cached.
        system = f"[bench-rep-{rep}-{time.time_ns()}] " + filler
        turn1 = [{"role": "system", "content": system},
                 {"role": "user", "content": "Name a fact about the number one."}]
        turn2 = turn1 + [{"role": "assistant", "content": "One is the first positive integer."},
                         {"role": "user", "content": "Name a fact about the number two."}]
        cold.append(ttft_ms(args.url, args.model, turn1))   # cold: full prefill
        warm.append(ttft_ms(args.url, args.model, turn2))   # warm: prefix hit
        sys.stderr.write(f"  rep {rep}: cold {cold[-1]:.0f}ms  warm {warm[-1]:.0f}ms\n")

    def stats(xs):
        return statistics.mean(xs), statistics.median(xs), min(xs), max(xs)

    cm, cmd, cmin, cmax = stats(cold)
    wm, wmd, wmin, wmax = stats(warm)
    print(f"\nmodel: {args.model}   ~{args.ctx_tokens}-token system prompt   reps: {args.reps}\n")
    print(f"{'series':<26}{'mean':>10}{'median':>10}{'min':>10}{'max':>10}")
    print(f"{'cold (full prefill)':<26}{cm:>9.0f}ms{cmd:>9.0f}ms{cmin:>9.0f}ms{cmax:>9.0f}ms")
    print(f"{'warm (prefix reuse)':<26}{wm:>9.0f}ms{wmd:>9.0f}ms{wmin:>9.0f}ms{wmax:>9.0f}ms")
    if cmd > 0:
        print(f"\nTTFT (median): {cmd:.0f}ms → {wmd:.0f}ms   ({(wmd - cmd) / cmd * 100:+.1f}%)")


if __name__ == "__main__":
    main()
