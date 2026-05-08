//! Native benchmark client: measures mlx-engine HTTP, spindll HTTP, and spindll gRPC
//! without Python client overhead.
//!
//! Build:  cargo build --release --bin bench --features cli
//!
//! Standalone:
//!   ./target/release/bench --model mlx-community/Llama-3.2-1B-Instruct-4bit --runs 5
//!
//! Called by bench/run.sh (JSON mode, one phase at a time):
//!   bench --model <m> --phase mlx  --json   # runs mlx-engine HTTP leg
//!   bench --model <m> --phase spin --json   # runs spindll HTTP + gRPC legs

use std::time::Instant;

use clap::{Parser, ValueEnum};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use spindll::proto::{spindll_client::SpindllClient, ChatRequest, GenerateParams, Message};
use tokio_stream::StreamExt;

// ── Default prompt pool ───────────────────────────────────────────────────────

const BUILTIN_PROMPTS: &[&str] = &[
    "Explain how transformers work in simple terms.",
    "What are the key differences between Python and Rust?",
    "Write a haiku about machine learning.",
    "Describe the process of photosynthesis.",
];

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Clone, ValueEnum, PartialEq)]
enum Phase {
    /// Benchmark mlx-engine HTTP only
    Mlx,
    /// Benchmark spindll HTTP only (use with a fresh server start for isolation)
    SpinHttp,
    /// Benchmark spindll gRPC only (use with a fresh server start for isolation)
    SpinGrpc,
    /// Benchmark spindll HTTP + gRPC against the same running instance
    Spin,
    /// Benchmark all three (both servers must be running)
    All,
}

#[derive(Parser)]
#[command(about = "Native benchmark client for spindll vs mlx-engine")]
struct Args {
    #[arg(long)]
    model: String,

    /// Single prompt (used as a one-element pool). Ignored when --prompts is set.
    #[arg(long, default_value = "")]
    prompt: String,

    /// Comma-separated prompts to cycle across runs.
    /// Default when neither --prompt nor --prompts is given: 4-prompt built-in pool.
    #[arg(long, value_delimiter = ',')]
    prompts: Vec<String>,

    #[arg(long, default_value_t = 10)]
    runs: u32,

    #[arg(long, default_value_t = 1)]
    warmup: u32,

    #[arg(long, default_value_t = 200)]
    max_tokens: i32,

    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    #[arg(long, default_value_t = 40)]
    top_k: i32,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// mlx-engine HTTP base URL
    #[arg(long, default_value = "http://localhost:1234")]
    url_mlx: String,

    /// spindll HTTP base URL
    #[arg(long, default_value = "http://localhost:8080")]
    url_spin: String,

    /// spindll gRPC host
    #[arg(long, default_value = "localhost")]
    grpc_host: String,

    /// spindll gRPC port
    #[arg(long, default_value_t = 50051)]
    grpc_port: u16,

    /// Which phase to run
    #[arg(long, value_enum, default_value = "all")]
    phase: Phase,

    /// Output JSON to stdout (for bench/run.sh integration)
    #[arg(long)]
    json: bool,
}

// ── Prompt resolution ─────────────────────────────────────────────────────────

fn resolve_prompts(args: &Args) -> Vec<&str> {
    if !args.prompts.is_empty() {
        args.prompts.iter().map(|s| s.as_str()).collect()
    } else if !args.prompt.is_empty() {
        vec![args.prompt.as_str()]
    } else {
        BUILTIN_PROMPTS.to_vec()
    }
}

// ── Result types ──────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone)]
struct RunResult {
    ttft_ms: f64,
    completion_tokens: u32,
    total_ms: f64,
    tok_per_sec: f64,
    prompt: String,
    text: String,
}

#[derive(Serialize)]
struct PhaseOutput {
    /// "mlx-engine", "spindll-http", "spindll-grpc"
    engine: String,
    runs: Vec<RunResult>,
}

#[derive(Serialize)]
struct BenchOutput {
    engines: Vec<PhaseOutput>,
}

// ── HTTP runner ───────────────────────────────────────────────────────────────

async fn run_once_http(
    client: &Client,
    url: &str,
    model: &str,
    prompt: &str,
    max_tokens: i32,
    args: &Args,
) -> anyhow::Result<RunResult> {
    let payload = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": true,
        "max_tokens": max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "stream_options": {"include_usage": true},
    });

    let t_start = Instant::now();
    let mut ttft: Option<f64> = None;
    let mut delta_chunks: u32 = 0;
    let mut tokens_from_usage: Option<u32> = None;
    let mut text = String::new();

    let resp = client
        .post(format!("{url}/v1/chat/completions"))
        .json(&payload)
        .send()
        .await?
        .error_for_status()?;

    let mut body = resp.bytes_stream();
    let mut buf = String::new();

    while let Some(chunk) = body.next().await {
        buf.push_str(&String::from_utf8_lossy(&chunk?));
        // Process all complete lines in the buffer
        while let Some(newline) = buf.find('\n') {
            let line: String = buf.drain(..=newline).collect();
            let line = line.trim();
            if !line.starts_with("data: ") {
                continue;
            }
            let raw = &line[6..];
            if raw == "[DONE]" {
                break;
            }
            let Ok(chunk): Result<serde_json::Value, _> = serde_json::from_str(raw) else {
                continue;
            };
            if let Some(ct) = chunk["usage"]["completion_tokens"].as_u64() {
                tokens_from_usage = Some(ct as u32);
            }
            let content = chunk["choices"][0]["delta"]["content"]
                .as_str()
                .unwrap_or("");
            if !content.is_empty() {
                if ttft.is_none() {
                    ttft = Some(t_start.elapsed().as_secs_f64() * 1000.0);
                }
                delta_chunks += 1;
                text.push_str(content);
            }
        }
    }

    let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = ttft.unwrap_or(total_ms);
    let completion_tokens = tokens_from_usage.unwrap_or(delta_chunks);
    let tok_per_sec = completion_tokens as f64 / (total_ms / 1000.0);

    Ok(RunResult { ttft_ms, completion_tokens, total_ms, tok_per_sec, prompt: prompt.to_string(), text })
}

// ── gRPC runner ───────────────────────────────────────────────────────────────

async fn run_once_grpc(
    client: &mut SpindllClient<tonic::transport::Channel>,
    prompt: &str,
    max_tokens: i32,
    args: &Args,
) -> anyhow::Result<RunResult> {
    let request = ChatRequest {
        model: args.model.clone(),
        messages: vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
        params: Some(GenerateParams {
            temperature: Some(args.temperature),
            top_p:       Some(args.top_p),
            top_k:       Some(args.top_k),
            max_tokens:  Some(max_tokens),
            seed:        Some(args.seed),
            stop:        vec![],
        }),
        encryption_key: vec![],
    };

    let t_start = Instant::now();
    let mut ttft: Option<f64> = None;
    let mut delta_count: u32 = 0;
    let mut tokens_from_usage: Option<u32> = None;
    let mut text = String::new();

    let mut stream = client.chat(request).await?.into_inner();
    while let Some(resp) = stream.message().await? {
        if !resp.token.is_empty() {
            if ttft.is_none() {
                ttft = Some(t_start.elapsed().as_secs_f64() * 1000.0);
            }
            delta_count += 1;
            text.push_str(&resp.token);
        }
        if resp.done {
            if let Some(usage) = resp.usage {
                if usage.completion_tokens > 0 {
                    tokens_from_usage = Some(usage.completion_tokens as u32);
                }
            }
            break;
        }
    }

    let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = ttft.unwrap_or(total_ms);
    let completion_tokens = tokens_from_usage.unwrap_or(delta_count);
    let tok_per_sec = completion_tokens as f64 / (total_ms / 1000.0);

    Ok(RunResult { ttft_ms, completion_tokens, total_ms, tok_per_sec, prompt: prompt.to_string(), text })
}

// ── Model pre-load ────────────────────────────────────────────────────────────

/// Ensure the model is resident in spindll before benchmark runs start.
/// mlx-engine (lm-studio) always has the model pre-loaded; calling this levels
/// the playing field so warmup runs measure pure inference, not model loading.
async fn preload_model(client: &Client, url: &str, model: &str) -> anyhow::Result<()> {
    eprint!("  [spindll] loading model '{model}' ...");
    let resp = client
        .post(format!("{url}/load"))
        .json(&serde_json::json!({ "model": model }))
        .timeout(std::time::Duration::from_secs(300))
        .send()
        .await?
        .error_for_status()?;
    let body: serde_json::Value = resp.json().await?;
    if body["already_loaded"].as_bool().unwrap_or(false) {
        eprintln!(" already loaded");
    } else {
        eprintln!(" done");
    }
    Ok(())
}

// ── Run loop ──────────────────────────────────────────────────────────────────

async fn bench_http(
    label: &str,
    client: &Client,
    url: &str,
    model: &str,
    prompts: &[&str],
    max_tokens: i32,
    args: &Args,
) -> Vec<RunResult> {
    let mut results = Vec::new();
    for i in 0..(args.warmup + args.runs) {
        let is_warmup = i < args.warmup;
        let run_num = if is_warmup { i + 1 } else { i - args.warmup + 1 };
        let total = if is_warmup { args.warmup } else { args.runs };
        let tag = if is_warmup { format!("warmup {run_num}/{total}") } else { format!("run {run_num}/{total}") };
        let prompt = prompts[i as usize % prompts.len()];
        eprint!("  [{label}] {tag} ...");
        match run_once_http(client, url, model, prompt, max_tokens, args).await {
            Ok(r) => {
                eprintln!(" {:.1} tok/s  TTFT {:.0} ms", r.tok_per_sec, r.ttft_ms);
                if !is_warmup { results.push(r); }
            }
            Err(e) => eprintln!(" ERROR: {e}"),
        }
    }
    results
}

async fn bench_grpc(
    client: &mut SpindllClient<tonic::transport::Channel>,
    prompts: &[&str],
    max_tokens: i32,
    args: &Args,
) -> Vec<RunResult> {
    let mut results = Vec::new();
    for i in 0..(args.warmup + args.runs) {
        let is_warmup = i < args.warmup;
        let run_num = if is_warmup { i + 1 } else { i - args.warmup + 1 };
        let total = if is_warmup { args.warmup } else { args.runs };
        let tag = if is_warmup { format!("warmup {run_num}/{total}") } else { format!("run {run_num}/{total}") };
        let prompt = prompts[i as usize % prompts.len()];
        eprint!("  [spindll-grpc] {tag} ...");
        match run_once_grpc(client, prompt, max_tokens, args).await {
            Ok(r) => {
                eprintln!(" {:.1} tok/s  TTFT {:.0} ms", r.tok_per_sec, r.ttft_ms);
                if !is_warmup { results.push(r); }
            }
            Err(e) => eprintln!(" ERROR: {e}"),
        }
    }
    results
}

// ── Stats + table ─────────────────────────────────────────────────────────────

fn stats(values: &[f64]) -> (f64, f64, f64, f64) {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let mean = v.iter().sum::<f64>() / n as f64;
    let median = if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 };
    (mean, median, v[0], v[n - 1])
}

fn print_table(engines: &[PhaseOutput]) {
    let sep = "─".repeat(78);
    for (title, field) in &[("TTFT (ms)", "ttft"), ("Tok/s", "tps"), ("Total (ms)", "total")] {
        println!("\n  {title}");
        println!("  {:16}  {:>10}  {:>10}  {:>10}  {:>10}", "Engine", "mean", "median", "min", "max");
        println!("  {}", &sep[..74]);
        for e in engines {
            let vals: Vec<f64> = e.runs.iter().map(|r| match *field {
                "ttft"  => r.ttft_ms,
                "tps"   => r.tok_per_sec,
                _       => r.total_ms,
            }).collect();
            let (mean, median, min, max) = stats(&vals);
            let unit = if *field == "tps" { "   " } else { " ms" };
            println!("  {:<16}  {:>9.1}{unit}  {:>9.1}{unit}  {:>9.1}{unit}  {:>9.1}{unit}",
                     e.engine, mean, median, min, max);
        }
    }

    // Deltas
    if engines.len() >= 2 {
        println!("\n{sep}");
        println!("  Deltas (mean) — lower TTFT/total is better, higher tok/s is better");
        println!("{sep}");
        let pairs: Vec<(&str, usize, usize)> = match engines.len() {
            2 => vec![(&engines[1].engine, 1, 0)],
            _ => vec![
                (&engines[1].engine, 1, 0),
                (&engines[2].engine, 2, 0),
                ("spindll-grpc vs spindll-http", 2, 1),
            ],
        };
        for (label, a, b) in pairs {
            let pct = |fa: f64, fb: f64| -> String {
                if fb == 0.0 { return "n/a".to_string(); }
                let d = (fa - fb) / fb * 100.0;
                format!("{}{:.1}%", if d >= 0.0 { "+" } else { "" }, d)
            };
            let (ea, eb) = (&engines[a], &engines[b]);
            let ttft_a = stats(&ea.runs.iter().map(|r| r.ttft_ms).collect::<Vec<_>>()).0;
            let ttft_b = stats(&eb.runs.iter().map(|r| r.ttft_ms).collect::<Vec<_>>()).0;
            let tps_a  = stats(&ea.runs.iter().map(|r| r.tok_per_sec).collect::<Vec<_>>()).0;
            let tps_b  = stats(&eb.runs.iter().map(|r| r.tok_per_sec).collect::<Vec<_>>()).0;
            let tot_a  = stats(&ea.runs.iter().map(|r| r.total_ms).collect::<Vec<_>>()).0;
            let tot_b  = stats(&eb.runs.iter().map(|r| r.total_ms).collect::<Vec<_>>()).0;
            println!("  {label:<38}  TTFT {:>8}  tok/s {:>8}  total {:>8}",
                     pct(ttft_a, ttft_b), pct(tps_a, tps_b), pct(tot_a, tot_b));
        }
        println!("{sep}");
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let http = Client::new();
    let mut output = BenchOutput { engines: vec![] };
    let prompts = resolve_prompts(&args);

    if args.phase == Phase::Mlx || args.phase == Phase::All {
        let results = bench_http("mlx-engine", &http, &args.url_mlx, &args.model, &prompts, args.max_tokens, &args).await;
        output.engines.push(PhaseOutput { engine: "mlx-engine".to_string(), runs: results });
    }

    if args.phase == Phase::SpinHttp || args.phase == Phase::Spin || args.phase == Phase::All {
        if let Err(e) = preload_model(&http, &args.url_spin, &args.model).await {
            eprintln!("  [spindll] preload warning: {e} — continuing anyway");
        }
        let http_results = bench_http("spindll-http", &http, &args.url_spin, &args.model, &prompts, args.max_tokens, &args).await;
        output.engines.push(PhaseOutput { engine: "spindll-http".to_string(), runs: http_results });
    }

    if args.phase == Phase::SpinGrpc || args.phase == Phase::Spin || args.phase == Phase::All {
        if let Err(e) = preload_model(&http, &args.url_spin, &args.model).await {
            eprintln!("  [spindll] preload warning: {e} — continuing anyway");
        }
        let grpc_endpoint = format!("http://{}:{}", args.grpc_host, args.grpc_port);
        let mut grpc = SpindllClient::connect(grpc_endpoint).await
            .map_err(|e| anyhow::anyhow!("gRPC connect failed: {e}"))?;
        let grpc_results = bench_grpc(&mut grpc, &prompts, args.max_tokens, &args).await;
        output.engines.push(PhaseOutput { engine: "spindll-grpc".to_string(), runs: grpc_results });
    }

    if args.json {
        println!("{}", serde_json::to_string(&output)?);
    } else {
        print_table(&output.engines);
    }

    Ok(())
}
