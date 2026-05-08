# Benchmarking spindll

`bench/run.sh` orchestrates engine lifecycle and delegates all timing to the
native Rust benchmark client (`src/bin/bench.rs`).

## Quick start

### 1. Build the Rust benchmark client

```bash
cargo build --release --bin bench --features cli
```

### 2. Install dependencies

```bash
brew install jq   # JSON processing for markdown reports
```

---

## run.sh — external comparison

### spindll only (HTTP + gRPC)

The simplest mode. The script starts and stops spindll automatically:

```bash
bench/run.sh \
    --model "mlx-community/Llama-3.2-1B-Instruct-4bit" \
    --mode spindll
```

### Compare with mlx-engine (lm-studio)

Runs two phases with automatic lifecycle management — starts lm-studio,
benchmarks it, stops it, cools down, starts spindll, benchmarks it, stops it:

```bash
bench/run.sh \
    --model "mlx-community/Llama-3.2-1B-Instruct-4bit" \
    --mode compare
```

Requires the `lms` CLI (installed with LM Studio).

---

## Workload options

| Flag | Default | Description |
|------|---------|-------------|
| `--runs N` | 10 | Measured runs per engine |
| `--warmup N` | 3 | Warmup runs discarded per engine |
| `--max-tokens N` | 200 | Max completion tokens |
| `--temperature F` | 0.0 | Sampling temperature (0.0 = greedy) |
| `--top-p F` | 0.95 | Nucleus sampling threshold |
| `--top-k N` | 40 | Must remain `40`; HTTP benchmark paths cannot override `top_k` |
| `--seed N` | 42 | RNG seed |
| `--prompts P,P,...` | built-in pool | Comma-separated prompts to cycle across runs |
| `--cooldown N` | 5 | Seconds between phases (thermal/memory settle) |

When neither `--prompts` is given, the bench binary cycles through four
built-in prompts. With 10 runs, the first 3 are cold cache misses and runs
4-10 are warm cache hits — which is why mean and median diverge in TTFT results.

`top_p` is applied across the HTTP and gRPC benchmark paths. `top_k` is fixed
at `40`, because the OpenAI-compatible HTTP endpoints used by the benchmark do
not expose a `top_k` override.

---

## Endpoints

| Flag | Default | Description |
|------|---------|-------------|
| `--url-spin` | `http://localhost:8080` | spindll HTTP base URL |
| `--grpc-host` | `localhost` | spindll gRPC host |
| `--grpc-port` | `50051` | spindll gRPC port |

---

## Understanding the results

**TTFT (time to first token)** — includes prompt processing (prefill). With
the prompt KV cache enabled, repeated prompts skip full prefill; only the
first occurrence of each prompt in a run is a cold miss.

**Tok/s** — decode-only throughput, computed as `(completion_tokens - 1) /
(total - TTFT)`. The first token is generated during the TTFT/prefill window,
so it is excluded from both the numerator and the time interval. This isolates
decode speed from prefill effects.

**Total** — wall time for the full streamed response.

**Mean vs median** — when prompts cycle, mean includes cold-miss runs; median
better represents steady-state (warm) performance.

**gRPC vs HTTP** — gRPC TTFT is consistently lower because it avoids HTTP
framing and JSON serialization overhead on the transport path.

**Cooldown** — the default 5s pause between phases lets thermals and memory
settle, reducing systematic bias from running engines sequentially.
