#!/usr/bin/env bash
# bench/run.sh — orchestrate inference benchmarks with automated engine lifecycle.
#
# Delegates all timing to the Rust bench binary (src/bin/bench.rs).
# This script handles server start/stop, health checks, cooldown, and markdown
# report generation.
#
# Modes:
#   compare        mlx-engine (lm-studio) vs spindll HTTP + gRPC  [default]
#   spindll        spindll HTTP + gRPC only
#
# For before/after comparisons between two spindll builds, see internal_bench.sh.
#
# Examples:
#   bench/run.sh --model mlx-community/Llama-3.2-3B-Instruct-4bit
#   bench/run.sh --model ... --mode spindll
#
# Prerequisites:
#   cargo build --release --bin bench --features cli
#   brew install jq

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_BIN="$ROOT/target/release/bench"
RESULTS_DIR="$SCRIPT_DIR/results"

# ── Defaults ──────────────────────────────────────────────────────────────────

MODEL=""
RUNS=10
WARMUP=3
MAX_TOKENS=200
TEMPERATURE=0.0
TOP_P=0.95
TOP_K=40
SEED=42
COOLDOWN=5
MODE=compare

URL_MLX="http://localhost:1234"
URL_SPIN="http://localhost:8080"
GRPC_HOST=localhost
GRPC_PORT=50051
HTTP_PORT=8080
SPIN_PORT=50051

SPINDLL_BIN="$ROOT/target/release/spindll"

OUTPUT=""
PROMPTS=""

# ── State ─────────────────────────────────────────────────────────────────────

SPINDLL_PID=""
LMS_MANAGED=false
WORK=""

# ── Helpers ───────────────────────────────────────────────────────────────────

die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
info() { printf '  %s\n' "$*"; }
sep()  { printf '  %s\n' "$(printf '─%.0s' {1..58})"; }

slug() { printf '%s' "${1##*/}" | tr -c 'a-zA-Z0-9._-' '-'; }

cleanup() {
    if [[ -n "$SPINDLL_PID" ]]; then
        kill "$SPINDLL_PID" 2>/dev/null
        wait "$SPINDLL_PID" 2>/dev/null || true
    fi
    if $LMS_MANAGED; then
        lms server stop 2>/dev/null || true
    fi
    if [[ -n "$WORK" ]]; then
        rm -rf "$WORK"
    fi
}
trap cleanup EXIT INT TERM

# ── Health check ──────────────────────────────────────────────────────────────

wait_http() {
    local url="$1" label="$2" timeout="${3:-30}"
    printf '  waiting for %s ' "$label"
    for ((i = 0; i < timeout; i++)); do
        if curl -so /dev/null --max-time 2 "${url}/v1/models" 2>/dev/null; then
            printf ' ready (%ds)\n' "$i"
            return 0
        fi
        printf '.'
        sleep 1
    done
    printf ' timeout\n' >&2
    return 1
}

cooldown() {
    ((COOLDOWN > 0)) && { info "cooldown ${COOLDOWN}s ..."; sleep "$COOLDOWN"; }
}

# ── Engine lifecycle ──────────────────────────────────────────────────────────

# Convert spindll model name to the lms identifier lms load expects.
# e.g. mlx-community/Llama-3.2-1B-Instruct-4bit -> llama-3.2-1b-instruct
lms_model_name() {
    local basename="${MODEL##*/}"           # drop org prefix
    local stripped="${basename%-[0-9]*bit}" # drop -4bit / -8bit / etc.
    echo "$stripped" | tr '[:upper:]' '[:lower:]'
}

# Symlink the model from spindll's store into ~/.lmstudio/models/ so lms can
# see it without a separate download.
# If the symlink is newly created, stop any running server so it rescans
# the models directory on the next start.
ensure_lmstudio_symlink() {
    local org="${MODEL%%/*}"   # e.g. mlx-community
    local name="${MODEL##*/}"  # e.g. Llama-3.2-1B-Instruct-4bit
    local src="$HOME/.spindll/models/$org/$name"
    local dst_dir="$HOME/.lmstudio/models/$org"
    local dst="$dst_dir/$name"

    [[ -d "$src" ]] || die "model not in spindll store: $src  (run: spindll pull $MODEL)"

    if [[ -L "$dst" ]] || [[ -d "$dst" ]]; then
        return
    fi

    mkdir -p "$dst_dir"
    ln -s "$src" "$dst"
    info "symlinked $dst -> $src"
    info "stopping lm-studio so it rescans models on next start ..."
    lms server stop 2>/dev/null || true
    sleep 2
}

start_mlx() {
    command -v lms >/dev/null || die "lms CLI not found (install LM Studio)"
    ensure_lmstudio_symlink
    info "starting lm-studio server"
    lms server start 2>/dev/null || true
    LMS_MANAGED=true
    wait_http "$URL_MLX" "lm-studio" 30
    local lms_name
    lms_name="$(lms_model_name)"
    info "loading model: $lms_name"
    local i
    for ((i = 1; i <= 5; i++)); do
        if lms load "$lms_name" 2>/dev/null; then
            break
        fi
        ((i < 5)) || die "lms load '$lms_name' failed after 5 attempts"
        info "  lms load failed, retrying in 3s ($i/5) ..."
        sleep 3
    done
    sleep 2
}

stop_mlx() {
    local lms_name
    lms_name="$(lms_model_name)"
    info "unloading model: $lms_name"
    lms unload "$lms_name" 2>/dev/null || true
    info "stopping lm-studio"
    lms server stop 2>/dev/null || true
    LMS_MANAGED=false
}

start_spindll() {
    local bin="${1:-$SPINDLL_BIN}"
    [[ -x "$bin" ]] || die "spindll not found: $bin"
    info "starting: $bin serve --port $SPIN_PORT --http-port $HTTP_PORT"
    "$bin" serve --port "$SPIN_PORT" --http-port "$HTTP_PORT" \
        >"$WORK/spindll.log" 2>&1 &
    SPINDLL_PID=$!
    wait_http "$URL_SPIN" "spindll" 60
}

stop_spindll() {
    if [[ -n "$SPINDLL_PID" ]]; then
        info "stopping spindll (pid $SPINDLL_PID)"
        kill "$SPINDLL_PID" 2>/dev/null || true
        wait "$SPINDLL_PID" 2>/dev/null || true
        SPINDLL_PID=""
        sleep 1
    fi
}

# ── Bench runner ──────────────────────────────────────────────────────────────

ensure_bench() {
    if [[ ! -x "$BENCH_BIN" ]]; then
        info "building bench binary ..."
        (cd "$ROOT" && cargo build --release --bin bench --features cli)
    fi
}

run_bench() {
    local phase="$1"
    local cmd=(
        "$BENCH_BIN"
        --phase "$phase"
        --model "$MODEL"
        --runs  "$RUNS"
        --warmup "$WARMUP"
        --max-tokens "$MAX_TOKENS"
        --temperature "$TEMPERATURE"
        --top-p "$TOP_P"
        --top-k "$TOP_K"
        --seed  "$SEED"
        --url-mlx  "$URL_MLX"
        --url-spin "$URL_SPIN"
        --grpc-host "$GRPC_HOST"
        --grpc-port "$GRPC_PORT"
        --json
    )
    [[ -n "$PROMPTS" ]] && cmd+=(--prompts "$PROMPTS")
    "${cmd[@]}"
}

# ── Markdown report ───────────────────────────────────────────────────────────

prompt_description() {
    if [[ -n "$PROMPTS" ]]; then
        IFS=',' read -ra ps <<< "$PROMPTS"
        if ((${#ps[@]} == 1)); then
            printf '**Prompt:** %s  ' "${ps[0]}"
        else
            printf '**Prompts (%d, cycling):**' "${#ps[@]}"
            local first=true
            for p in "${ps[@]}"; do
                $first || printf ' &nbsp;·&nbsp;'
                printf ' _%s_' "$p"
                first=false
            done
            printf '  '
        fi
    else
        printf '**Prompts (4, cycling):** _Explain how transformers work in simple terms._ &nbsp;·&nbsp; _What are the key differences between Python and Rust?_ &nbsp;·&nbsp; _Write a haiku about machine learning._ &nbsp;·&nbsp; _Describe the process of photosynthesis._  '
    fi
}

generate_report() {
    local json="$1" output="$2" title="$3" pairs_json="$4" extra="${5:-}"

    mkdir -p "$(dirname "$output")"

    local prompt_desc
    prompt_desc="$(prompt_description)"

    jq -r \
        --arg title "$title" \
        --arg date "$(date '+%Y-%m-%d %H:%M')" \
        --arg model "$MODEL" \
        --arg prompt_desc "$prompt_desc" \
        --arg runs "$RUNS" \
        --arg warmup "$WARMUP" \
        --arg max_tokens "$MAX_TOKENS" \
        --arg temp "$TEMPERATURE" \
        --arg seed "$SEED" \
        --arg extra "$extra" \
        --argjson pairs "$pairs_json" \
    '
    # ── helpers ──
    def stats:
      sort | . as $v | length as $n |
      if $n == 0 then {mean:0, median:0, min:0, max:0}
      else {
        mean:   ($v | add / $n),
        median: (if $n % 2 == 1 then $v[$n/2|floor]
                 else ($v[$n/2-1] + $v[$n/2]) / 2 end),
        min: $v[0],
        max: $v[$n-1]
      } end;
    def fmt($d):
      . * pow(10;$d) | round / pow(10;$d) | tostring;
    def pct:
      if .[1] == 0 then "n/a"
      else ((.[0] - .[1]) / .[1] * 100) |
           (if . >= 0 then "+" else "" end) + fmt(1) + "%"
      end;
    def md_row:  "| " + join(" | ") + " |";
    def md_sep:  [range(length) | "---"] | md_row;

    # ── compute stats per engine ──
    .engines | map({
      engine: .engine,
      ttft_ms:           ([.runs[].ttft_ms]           | stats),
      tok_per_sec:       ([.runs[].tok_per_sec]       | stats),
      total_ms:          ([.runs[].total_ms]          | stats),
      completion_tokens: ([.runs[].completion_tokens] | stats)
    }) as $st |

    # ── header ──
    "# \($title)", "",
    "**Date:** \($date)  ",
    "**Model:** `\($model)`  ",
    $prompt_desc,
    "**Runs:** \($runs) (+ \($warmup) warmup) &nbsp;|&nbsp; max_tokens: \($max_tokens) &nbsp;|&nbsp; temp: \($temp) &nbsp;|&nbsp; seed: \($seed)",
    (if $extra != "" then $extra else empty end),
    "",

    # ── metric tables ──
    (
      [["TTFT (ms)","ttft_ms"," ms",1],
       ["Tok/s","tok_per_sec","",1],
       ["Total (ms)","total_ms"," ms",1],
       ["Tokens","completion_tokens","",0]][]
      | . as [$t,$f,$u,$d] |
      "## \($t)", "",
      (["Engine","mean","median","min","max"] | md_row),
      (["Engine","mean","median","min","max"] | md_sep),
      ($st[] | [
        .engine,
        (.[$f].mean   | fmt($d)) + $u,
        (.[$f].median | fmt($d)) + $u,
        (.[$f].min    | fmt($d)) + $u,
        (.[$f].max    | fmt($d)) + $u
      ] | md_row),
      ""
    ),

    # ── deltas ──
    "## Deltas (mean)", "",
    "> Negative TTFT/total = faster. Positive tok/s = higher throughput.", "",
    (["Comparison","TTFT","tok/s","total"] | md_row),
    (["Comparison","TTFT","tok/s","total"] | md_sep),
    (
      $pairs[] as [$a,$b] |
      ($st | map(select(.engine==$a))[0]) as $ea |
      ($st | map(select(.engine==$b))[0]) as $eb |
      [
        "\($a) vs \($b)",
        ([$ea.ttft_ms.mean,     $eb.ttft_ms.mean]     | pct),
        ([$ea.tok_per_sec.mean, $eb.tok_per_sec.mean]  | pct),
        ([$ea.total_ms.mean,    $eb.total_ms.mean]     | pct)
      ] | md_row
    ),
    "",
    "---",
    "",
    "> **Note on token counts:** `completion_tokens` may differ by ±1 between engines.",
    "> mlx-engine applies `max_tokens` as an exclusive bound; spindll as inclusive.",
    "> The Python MLX vs Swift MLX tokenizers may also split text differently at word",
    "> boundaries. This introduces a ~0.5% systematic bias in tok/s and total time",
    "> at the token cap — TTFT is unaffected.",
    ""
    ' "$json" > "$output"

    info "results -> $output"
}

write_text_file() {
    local json="$1" base="$2"   # base = output path without .md
    local out="${base}.txt"
    jq -r '
      .engines[] |
      "# \(.engine)\n",
      (.runs | to_entries[] |
        "=== Run \(.key + 1) ===\nPrompt: \(.value.prompt)\n\n\(.value.text)\n"
      )
    ' "$json" > "$out"
    info "text   -> $out"
}

# ── Modes ─────────────────────────────────────────────────────────────────────

mode_compare() {
    sep
    info "Phase 1/3 -- mlx-engine"
    start_mlx
    run_bench mlx > "$WORK/mlx.json"
    stop_mlx
    cooldown

    sep
    info "Phase 2/3 -- spindll HTTP"
    start_spindll
    run_bench spin-http > "$WORK/spin-http.json"
    stop_spindll
    cooldown

    sep
    info "Phase 3/3 -- spindll gRPC"
    start_spindll
    run_bench spin-grpc > "$WORK/spin-grpc.json"
    stop_spindll

    jq -s '{engines: [.[].engines[]]}' \
        "$WORK/mlx.json" "$WORK/spin-http.json" "$WORK/spin-grpc.json" > "$WORK/merged.json"

    local out="${OUTPUT:-$RESULTS_DIR/$(date +%Y-%m-%d)_$(slug "$MODEL").md}"
    local pairs='[["spindll-http","mlx-engine"],["spindll-grpc","mlx-engine"],["spindll-grpc","spindll-http"]]'
    generate_report "$WORK/merged.json" "$out" "Inference Engine Benchmark" "$pairs"
    write_text_file "$WORK/merged.json" "${out%.md}"
}

mode_spindll() {
    sep
    info "Phase 1/2 -- spindll HTTP"
    start_spindll
    run_bench spin-http > "$WORK/spin-http.json"
    stop_spindll
    cooldown

    sep
    info "Phase 2/2 -- spindll gRPC"
    start_spindll
    run_bench spin-grpc > "$WORK/spin-grpc.json"
    stop_spindll

    jq -s '{engines: [.[].engines[]]}' \
        "$WORK/spin-http.json" "$WORK/spin-grpc.json" > "$WORK/spin.json"

    local out="${OUTPUT:-$RESULTS_DIR/$(date +%Y-%m-%d)_$(slug "$MODEL").md}"
    local pairs='[["spindll-grpc","spindll-http"]]'
    generate_report "$WORK/spin.json" "$out" "spindll Benchmark" "$pairs"
    write_text_file "$WORK/spin.json" "${out%.md}"
}

# ── CLI ───────────────────────────────────────────────────────────────────────

usage() {
    cat <<'EOF'
Usage: bench/run.sh --model MODEL [options]

Modes:
  --mode compare        mlx-engine vs spindll (default)
  --mode spindll        spindll HTTP + gRPC only

For before/after comparisons, use bench/internal_bench.sh instead.

Required:
  --model MODEL         model name sent to all engines

Engine:
  --spindll-bin PATH    spindll binary (default: target/release/spindll)
  --url-mlx URL         mlx-engine URL (default: http://localhost:1234)
  --url-spin URL        spindll HTTP URL (default: http://localhost:8080)
  --grpc-host HOST      spindll gRPC host (default: localhost)
  --grpc-port PORT      spindll gRPC port (default: 50051)
  --http-port PORT      spindll HTTP serve port (default: 8080)
  --spin-port PORT      spindll gRPC serve port (default: 50051)

Workload:
  --runs N              measured runs per engine (default: 10)
  --warmup N            warmup runs discarded (default: 3)
  --max-tokens N        max completion tokens (default: 200)
  --temperature F       sampling temperature (default: 0.0)
  --top-p F             nucleus sampling (default: 0.95)
  --top-k N             top-k sampling (default: 40)
  --seed N              RNG seed (default: 42)
  --prompts P,P,...     comma-separated prompts (default: 4-prompt built-in pool)
  --cooldown N          seconds between phases (default: 5)

Output:
  --output PATH         markdown output path (default: bench/results/...)
EOF
    exit 0
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)       MODEL="$2";       shift 2;;
            --mode)        MODE="$2";        shift 2;;
            --runs)        RUNS="$2";        shift 2;;
            --warmup)      WARMUP="$2";      shift 2;;
            --max-tokens)  MAX_TOKENS="$2";  shift 2;;
            --temperature) TEMPERATURE="$2"; shift 2;;
            --top-p)       TOP_P="$2";       shift 2;;
            --top-k)       TOP_K="$2";       shift 2;;
            --seed)        SEED="$2";        shift 2;;
            --cooldown)    COOLDOWN="$2";    shift 2;;
            --prompts)     PROMPTS="$2";     shift 2;;
            --url-mlx)     URL_MLX="$2";     shift 2;;
            --url-spin)    URL_SPIN="$2";    shift 2;;
            --grpc-host)   GRPC_HOST="$2";   shift 2;;
            --grpc-port)   GRPC_PORT="$2";   shift 2;;
            --http-port)   HTTP_PORT="$2";   shift 2;;
            --spin-port)   SPIN_PORT="$2";   shift 2;;
            --spindll-bin) SPINDLL_BIN="$2"; shift 2;;
            --output)      OUTPUT="$2";      shift 2;;
            -h|--help)     usage;;
            *)             die "unknown option: $1";;
        esac
    done
    [[ -n "$MODEL" ]] || die "--model is required (try --help)"
}

# ── Main ──────────────────────────────────────────────────────────────────────

main() {
    parse_args "$@"

    command -v jq   >/dev/null || die "jq required: brew install jq"
    command -v curl >/dev/null || die "curl required"

    WORK="$(mktemp -d)"
    ensure_bench

    echo
    echo "  model:    $MODEL"
    echo "  mode:     $MODE"
    echo "  runs:     $RUNS (+ $WARMUP warmup)"
    echo "  cooldown: ${COOLDOWN}s between phases"
    sep

    case "$MODE" in
        compare)      mode_compare;;
        spindll)      mode_spindll;;
        *)            die "unknown mode: $MODE (use compare or spindll)";;
    esac
}

main "$@"
