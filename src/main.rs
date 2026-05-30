use clap::{Parser, Subcommand};

/// Spindll — a Rust-native GGUF inference engine.
#[derive(Parser)]
#[command(name = "spindll", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Pull a model (e.g. "llama3.1:8b" or "TheBloke/Llama-3-8B-GGUF")
    Pull {
        /// Model name — Ollama format (llama3.1:8b) or HuggingFace (owner/model)
        model: String,

        /// Quantization filter (HuggingFace GGUF or MLX quant like "4bit")
        #[arg(long)]
        quant: Option<String>,

        /// Force GGUF format (skip MLX resolution on Apple Silicon)
        #[arg(long, conflicts_with = "mlx")]
        gguf: bool,

        /// Force MLX format (error if no MLX equivalent found)
        #[arg(long, conflicts_with = "gguf")]
        mlx: bool,
    },

    /// List local models
    List,

    /// Delete a local model
    Rm {
        /// Model name to delete
        model: String,

        /// Skip confirmation prompts when removing externally-imported models
        #[arg(long)]
        purge: bool,
    },

    /// Start the gRPC server (models loaded dynamically via Load RPC)
    Serve {
        /// Port to listen on
        #[arg(long, default_value = "50051")]
        port: u16,

        /// Default context size for loaded models
        #[arg(long, default_value = "2048")]
        ctx_size: u32,

        /// Default GPU layers (omit to auto-detect)
        #[arg(long)]
        gpu_layers: Option<u32>,

        /// Memory budget for loaded models (e.g. "8G", omit for full live availability)
        #[arg(long)]
        budget: Option<String>,

        /// Disk KV cache size for prompt prefixes (e.g. "2G", default 2G)
        #[arg(long, conflicts_with = "no_kv_cache")]
        kv_cache: Option<String>,

        /// Disable disk KV cache
        #[arg(long, conflicts_with = "kv_cache")]
        no_kv_cache: bool,

        /// In-memory KV cache budget. llama.cpp requires exact prompt matches; MLX reuses prefixes internally.
        #[arg(long, conflicts_with = "no_kv_ram_cache")]
        kv_ram_cache: Option<String>,

        /// Disable the in-memory KV state cache
        #[arg(long, conflicts_with = "kv_ram_cache")]
        no_kv_ram_cache: bool,

        /// Concurrent sequence slots per model for batch scheduling (0 = disabled)
        #[arg(long, default_value = "0")]
        batch_slots: usize,

        /// RAM cache for recently-evicted models (e.g. "8G", default 4G when enabled)
        #[arg(long)]
        ram_cache: Option<Option<String>>,

        /// HTTP/SSE server port (requires --features http, 0 = disabled)
        #[arg(long, default_value = "8080")]
        http_port: u16,
    },

    /// One-shot inference (no server needed)
    Run {
        /// Model to use
        model: String,

        /// Prompt text
        prompt: String,

        /// System prompt (default: "You are a helpful assistant.")
        #[arg(long)]
        system: Option<String>,

        /// Max tokens to generate (default: 512 from library)
        #[arg(long)]
        max_tokens: Option<u32>,

        /// Context size for the model (default 2048)
        #[arg(long, default_value = "2048")]
        ctx_size: u32,

        /// Memory budget (e.g. "8G", omit=live RAM, "0"=total RAM)
        #[arg(long)]
        budget: Option<String>,

        /// Enable KV cache for prompt prefixes (e.g. "2G", default 2G when enabled)
        #[arg(long)]
        kv_cache: Option<Option<String>>,
    },

    /// Benchmark one or two models (GGUF vs GGUF, MLX vs MLX, or mixed)
    #[cfg(feature = "bench")]
    Bench {
        /// Model to benchmark
        model: String,

        /// Optional second model to compare against (any format)
        against: Option<String>,

        /// Number of measured runs (plus 1 warmup)
        #[arg(long, default_value = "3")]
        runs: u32,

        /// Max tokens to generate per run
        #[arg(long, default_value = "100")]
        max_tokens: u32,

        /// KV cache context size for GGUF models (MLX handles context dynamically)
        #[arg(long, default_value = "2048")]
        ctx_size: u32,

        /// Prompt to use for all runs
        #[arg(long)]
        prompt: Option<String>,

        /// Output JSON instead of a table
        #[arg(long)]
        json: bool,
    },

    /// Import models from Ollama, HuggingFace cache, or local file
    Import {
        /// Path to a GGUF or MLX model to import
        path: Option<String>,

        /// Import from local Ollama cache
        #[arg(long)]
        from_ollama: bool,

        /// Import from local HuggingFace cache
        #[arg(long)]
        from_hf: bool,
    },

    /// Search for models across HuggingFace and Ollama
    Search {
        /// Search query (e.g. "qwen2.5", "llama 8b", "codestral")
        query: String,

        /// Maximum results to show
        #[arg(long, default_value = "20")]
        limit: usize,

        /// Filter by format
        #[arg(long, value_parser = ["gguf", "mlx"])]
        format: Option<String>,

        /// Sort order (default: hardware-aware ranking)
        #[arg(long, value_parser = ["downloads", "size", "name"])]
        sort: Option<String>,
    },

    /// Show server status
    Status {
        /// Port of the running server (auto-detected from lockfile if omitted)
        #[arg(long)]
        port: Option<u16>,
    },
}

/// Parse a human-readable size like "2G", "512M" into bytes. Defaults to 2GB.
fn parse_size_bytes(s: Option<&str>) -> u64 {
    const DEFAULT: u64 = 2 * 1_073_741_824; // 2 GB
    let s = match s {
        Some(s) => s.trim(),
        None => return DEFAULT,
    };
    if s.is_empty() {
        return DEFAULT;
    }

    let (num, mult) = if s.ends_with('G') || s.ends_with('g') {
        (&s[..s.len() - 1], 1_073_741_824u64)
    } else if s.ends_with('M') || s.ends_with('m') {
        (&s[..s.len() - 1], 1_048_576u64)
    } else {
        (s, 1u64) // raw bytes
    };

    num.parse::<f64>()
        .map(|n| (n * mult as f64) as u64)
        .unwrap_or(DEFAULT)
}

fn parse_size_arg(s: &str) -> Option<u64> {
    let s = s.trim();
    if s == "0" {
        return Some(0);
    }
    if s.is_empty() {
        return None;
    }
    Some(parse_size_bytes(Some(s)))
}

fn manager_memory_budget(raw_budget: Option<&str>, detected_budget: u64) -> u64 {
    if raw_budget.and_then(parse_size_arg) == Some(0) {
        0
    } else {
        detected_budget
    }
}

// ---------------------------------------------------------------------------
// Backend dispatch
// ---------------------------------------------------------------------------

fn backend_for_format(
    format: &spindll::model_store::registry::ModelFormat,
) -> anyhow::Result<Box<dyn spindll::backend::InferenceBackend>> {
    use spindll::model_store::registry::ModelFormat;
    match format {
        ModelFormat::Gguf => Ok(Box::new(
            spindll::backend::llamacpp::LlamaCppBackend::new()?,
        )),
        ModelFormat::Mlx => {
            #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
            return Ok(Box::new(spindll::backend::mlx_swift::MlxBackend));
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "mlx")))]
            anyhow::bail!("MLX backend requires Apple Silicon and --features mlx");
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark helpers (debug only — excluded from release builds)
// ---------------------------------------------------------------------------

#[cfg(feature = "bench")]
#[derive(serde::Serialize)]
struct BenchResult {
    format_name: &'static str,
    ttft_ms: f64,
    /// Decode-only throughput: (completion_tokens - 1) / (total - ttft).
    tok_per_sec: f64,
    total_ms: f64,
    completion_tokens: u32,
    mem_peak_mb: f64,
}

/// Resident memory of this process in MB. Uses `memory-stats` which wraps
/// task_info on macOS, /proc on Linux, GetProcessMemoryInfo on Windows --
/// ~370 ns/call, no self-pollution. (sysinfo::System::new_all takes ~5 ms
/// and inflates RSS by ~5 MB per call: see tools/mem_bench/.)
#[cfg(feature = "bench")]
fn phys_footprint_mb() -> f64 {
    memory_stats::memory_stats()
        .map(|s| s.physical_mem as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0)
}

/// Decode-only tok/s: the first token is produced during the TTFT window,
/// so only (tokens - 1) fall in the decode interval (total - ttft).
#[cfg(feature = "bench")]
fn decode_tok_per_sec(completion_tokens: u32, ttft_ms: f64, total_ms: f64) -> f64 {
    if completion_tokens < 2 {
        return 0.0;
    }
    let decode_ms = total_ms - ttft_ms;
    if decode_ms <= 0.0 {
        return 0.0;
    }
    (completion_tokens - 1) as f64 / (decode_ms / 1000.0)
}

#[cfg(feature = "bench")]
fn bench_by_format(
    path: &std::path::Path,
    format: spindll::model_store::registry::ModelFormat,
    prompt: &str,
    max_tokens: u32,
    runs: u32,
    ctx_size: u32,
) -> anyhow::Result<BenchResult> {
    use spindll::model_store::registry::ModelFormat;

    if runs == 0 {
        anyhow::bail!("--runs must be greater than 0");
    }

    let format_name = match format {
        ModelFormat::Gguf => "GGUF",
        ModelFormat::Mlx => "MLX",
    };

    let backend = backend_for_format(&format)?;
    let load_params = spindll::backend::BackendLoadParams {
        n_ctx: ctx_size,
        n_gpu_layers: None,
        memory_budget: 0,
    };
    let model = backend.load_model(path, load_params)?;
    let params = spindll::engine::GenerateParams {
        max_tokens,
        ..Default::default()
    };

    // warmup
    model.generate(prompt, &params, &mut |_| true)?;

    let mut ttft_sum = 0.0f64;
    let mut tps_sum = 0.0f64;
    let mut total_ms_sum = 0.0f64;
    let mut last_tokens = 0u32;
    let mut mem_peak = 0.0f64;

    for _ in 0..runs {
        let start = std::time::Instant::now();
        let mut first = true;
        let mut ttft = 0.0f64;
        let result = model.generate(prompt, &params, &mut |_token| {
            if first {
                ttft = start.elapsed().as_secs_f64() * 1000.0;
                first = false;
            }
            true
        })?;
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
        ttft_sum += ttft;
        tps_sum += decode_tok_per_sec(result.completion_tokens, ttft, total_ms);
        total_ms_sum += total_ms;
        last_tokens = result.completion_tokens;
        let sample = phys_footprint_mb();
        if sample > mem_peak {
            mem_peak = sample;
        }
    }

    Ok(BenchResult {
        format_name,
        ttft_ms: ttft_sum / runs as f64,
        tok_per_sec: tps_sum / runs as f64,
        total_ms: total_ms_sum / runs as f64,
        completion_tokens: last_tokens,
        mem_peak_mb: mem_peak,
    })
}

#[cfg(feature = "bench")]
fn format_mem(mb: f64) -> String {
    if mb <= 0.0 {
        return "  —".to_string();
    }
    if mb >= 1024.0 {
        format!("{:.1}G", mb / 1024.0)
    } else {
        format!("{:.0}M", mb)
    }
}

#[cfg(feature = "bench")]
fn print_bench_row(label: &str, r: &BenchResult) {
    let label = if label.len() > 40 { &label[..40] } else { label };
    println!(
        "{:<40} {:>4} {:>8.0}ms {:>8.1} {:>7.2}s {:>5} {:>6}",
        label,
        r.format_name,
        r.ttft_ms,
        r.tok_per_sec,
        r.total_ms / 1000.0,
        r.completion_tokens,
        format_mem(r.mem_peak_mb),
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "spindll=info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Pull { model, quant, gguf, mlx } => {
            let format_pref = if gguf {
                spindll::model_store::FormatPreference::Gguf
            } else if mlx {
                spindll::model_store::FormatPreference::Mlx
            } else {
                spindll::model_store::FormatPreference::Auto
            };
            let store = spindll::model_store::ModelStore::new(None);
            let path = store.pull(&model, quant.as_deref(), format_pref)?;
            println!("model ready: {}", path.display());
        }
        Commands::List => {
            let store = spindll::model_store::ModelStore::new(None);
            store.list()?;
        }
        Commands::Rm { model, purge } => {
            let store = spindll::model_store::ModelStore::new(None);
            store.remove(&model, purge)?;
        }
        Commands::Serve {
            port,
            ctx_size,
            gpu_layers,
            budget,
            kv_cache,
            no_kv_cache,
            kv_ram_cache,
            no_kv_ram_cache,
            batch_slots,
            ram_cache,
            http_port,
        } => {
            let mem = spindll::scheduler::budget::MemoryBudget::detect(budget.as_deref());
            const GB: f64 = 1_073_741_824.0;
            println!(
                "memory budget: {:.1} GB cap (system: {:.1} GB total, {:.1} GB free)",
                mem.budget as f64 / GB,
                mem.total_ram as f64 / GB,
                mem.available_ram as f64 / GB,
            );
            let mut manager =
                spindll::engine::ModelManager::new(ctx_size, gpu_layers, mem.budget)?;

            if !no_kv_cache {
                let bytes = kv_cache
                    .as_deref()
                    .map(|s| parse_size_bytes(Some(s)))
                    .unwrap_or(2 * 1_073_741_824);
                manager.enable_kv_cache(bytes);
                println!("kv cache: {:.1} GB max", bytes as f64 / 1_073_741_824.0);

                #[allow(unused_variables)]
                let ram_bytes: u64 = if !no_kv_ram_cache {
                    let b = kv_ram_cache
                        .as_deref()
                        .map(|s| parse_size_bytes(Some(s)))
                        .unwrap_or(512 * 1_048_576);
                    manager.enable_kv_ram_cache(b);
                    println!("kv ram cache: {:.0} MB max", b as f64 / 1_048_576.0);
                    b
                } else {
                    0
                };

                // Mirror the same budgets into the MLX prompt cache (Apple
                // Silicon + --features mlx only). Same flags drive both backends.
                #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
                spindll::backend::mlx_swift::set_cache_budgets(ram_bytes, bytes);
            } else {
                #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
                spindll::backend::mlx_swift::set_cache_budgets(0, 0);
            }

            if let Some(cache_size) = ram_cache {
                let bytes = match cache_size.as_deref() {
                    Some(s) => parse_size_bytes(Some(s)),
                    None => 4 * 1_073_741_824, // 4 GB default
                };
                if !cfg!(target_os = "macos") {
                    manager.enable_ram_cache(bytes);
                    println!("ram cache: {:.1} GB max", bytes as f64 / 1_073_741_824.0);
                } else {
                    println!("ram cache: disabled (unified memory)");
                }
            }

            if batch_slots > 0 {
                manager.set_batch_slots(batch_slots);
                println!("batch scheduling: {batch_slots} concurrent slots per model");
            }

            let manager = manager.into_arc();
            let store = std::sync::Arc::new(spindll::model_store::ModelStore::new(None));

            #[cfg(feature = "http")]
            if http_port > 0 {
                let http_mgr = manager.clone();
                let http_store = store.clone();
                tokio::spawn(async move {
                    if let Err(e) =
                        spindll::http::start_http_server(http_port, http_mgr, http_store).await
                    {
                        tracing::error!("HTTP server error: {e}");
                    }
                });
            }
            #[cfg(not(feature = "http"))]
            let _ = http_port;

            let effective_http_port = if cfg!(feature = "http") { http_port } else { 0 };
            spindll::lockfile::Lockfile::write(port, effective_http_port)?;
            let result = spindll::grpc::start_server(port, manager, store).await;
            spindll::lockfile::Lockfile::remove();
            result?;
        }
        Commands::Run {
            model,
            prompt,
            system,
            max_tokens,
            ctx_size,
            budget,
            kv_cache,
        } => {
            let store = spindll::model_store::ModelStore::new(None);
            let model_path = store.resolve_model_path(&model)?;
            let digest = store.resolve_model_digest(&model).unwrap_or_default();

            let mem = spindll::scheduler::budget::MemoryBudget::detect(budget.as_deref());
            let mgr_budget = manager_memory_budget(budget.as_deref(), mem.budget);
            let mut manager =
                spindll::engine::ModelManager::new(ctx_size, None, mgr_budget)?;

            if let Some(cache_size) = kv_cache {
                let bytes = parse_size_bytes(cache_size.as_deref());
                manager.enable_kv_cache(bytes);
            }

            manager.load_model_with_digest(&model, &model_path, None, digest)?;

            let system_prompt = system.unwrap_or_else(|| "You are a helpful assistant.".to_string());
            let messages = vec![
                ("system".to_string(), system_prompt),
                ("user".to_string(), prompt.clone()),
            ];

            let mut params = spindll::engine::GenerateParams::default();
            if let Some(max) = max_tokens {
                params.max_tokens = max;
            }

            manager.generate_chat(&model, &messages, &params, None, |token| {
                use std::io::Write;
                print!("{token}");
                std::io::stdout().flush().ok();
                true
            })?;
            println!();
        }
        #[cfg(feature = "bench")]
        Commands::Bench {
            model,
            against,
            runs,
            max_tokens,
            ctx_size,
            prompt,
            json,
        } => {
            let store = spindll::model_store::ModelStore::new(None);
            let default_prompt =
                "Explain how transformers work in machine learning, step by step.";
            let prompt_str = prompt.as_deref().unwrap_or(default_prompt);

            let path1 = store.resolve_model_path(&model)?;
            let fmt1 = store.resolve_model_format(&model)?;
            let r1 = bench_by_format(&path1, fmt1, prompt_str, max_tokens, runs, ctx_size)?;

            let r2 = if let Some(ref against) = against {
                let path2 = store.resolve_model_path(against)?;
                let fmt2 = store.resolve_model_format(against)?;
                Some(bench_by_format(&path2, fmt2, prompt_str, max_tokens, runs, ctx_size)?)
            } else {
                None
            };

            use std::io::Write;
            std::io::stderr().flush().ok();

            if json {
                let mut results = serde_json::json!([{
                    "model": &model,
                    "format": r1.format_name,
                    "ttft_ms": r1.ttft_ms,
                    "tok_per_sec": r1.tok_per_sec,
                    "total_ms": r1.total_ms,
                    "completion_tokens": r1.completion_tokens,
                    "mem_peak_mb": r1.mem_peak_mb,
                }]);
                if let Some(ref r2) = r2 {
                    results.as_array_mut().unwrap().push(serde_json::json!({
                        "model": against.as_ref().unwrap(),
                        "format": r2.format_name,
                        "ttft_ms": r2.ttft_ms,
                        "tok_per_sec": r2.tok_per_sec,
                        "total_ms": r2.total_ms,
                        "completion_tokens": r2.completion_tokens,
                        "mem_peak_mb": r2.mem_peak_mb,
                    }));
                }
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else {
                println!("prompt    {:?}", prompt_str);
                println!(
                    "runs      {} (+ 1 warmup)  max-tokens={}  ctx-size={} (GGUF)",
                    runs, max_tokens, ctx_size
                );
                println!("{}", "─".repeat(86));
                println!(
                    "{:<40} {:>4} {:>9} {:>9} {:>8} {:>5} {:>6}",
                    "MODEL", "FMT", "TTFT", "TOK/S", "TOTAL", "TOKS", "MEM"
                );
                println!("{}", "─".repeat(86));
                print_bench_row(&model, &r1);
                if let Some(ref r2) = r2 {
                    print_bench_row(against.as_ref().unwrap(), r2);
                }
                println!("{}", "─".repeat(86));

                if let Some(ref r2) = r2 {
                    let against = against.as_ref().unwrap();
                    let tps_ratio = r2.tok_per_sec / r1.tok_per_sec;
                    let ttft_ratio = r1.ttft_ms / r2.ttft_ms;

                    let (faster_label, slower_label, tps_pct, ttft_pct) = if tps_ratio >= 1.0 {
                        (
                            against.as_str(),
                            model.as_str(),
                            (tps_ratio - 1.0) * 100.0,
                            (ttft_ratio - 1.0) * 100.0,
                        )
                    } else {
                        (
                            model.as_str(),
                            against.as_str(),
                            (1.0 / tps_ratio - 1.0) * 100.0,
                            (1.0 / ttft_ratio - 1.0) * 100.0,
                        )
                    };

                    let fl = if faster_label.len() > 30 {
                        &faster_label[..30]
                    } else {
                        faster_label
                    };
                    let sl = if slower_label.len() > 30 {
                        &slower_label[..30]
                    } else {
                        slower_label
                    };
                    println!(
                        "{} is {:.0}% faster throughput, {:.0}% faster TTFT vs {}",
                        fl, tps_pct, ttft_pct, sl,
                    );
                }
            }
        }
        Commands::Search { query, limit, format, sort } => {
            use spindll::model_store::search;
            use spindll::model_store::registry::ModelFormat;

            let format_filter = match format.as_deref() {
                Some("gguf") => Some(ModelFormat::Gguf),
                Some("mlx") => Some(ModelFormat::Mlx),
                _ => None,
            };
            let sort_order = match sort.as_deref() {
                Some("downloads") => search::SortOrder::Downloads,
                Some("size") => search::SortOrder::Size,
                Some("name") => search::SortOrder::Name,
                _ => search::SortOrder::Default,
            };
            let opts = search::SearchOptions {
                limit,
                format_filter,
                sort: sort_order,
            };

            let mem = spindll::scheduler::budget::MemoryBudget::detect(None);
            let inference_mem = search::detect_inference_memory(mem.total_ram);

            let spinner = indicatif::ProgressBar::new_spinner();
            spinner.set_message(format!("Searching for \"{query}\"..."));
            spinner.enable_steady_tick(std::time::Duration::from_millis(80));

            let q = query.clone();
            let results = tokio::task::spawn_blocking(move || {
                search::search_models(&q, &opts, inference_mem)
            })
            .await??;

            spinner.finish_and_clear();
            println!();

            if results.is_empty() {
                println!("  No models found.");
                return Ok(());
            }

            let max_name = 50;
            let name_w = results
                .iter()
                .map(|r| r.name.len())
                .max()
                .unwrap_or(5)
                .max(5)
                .min(max_name);

            println!(
                "  {:<nw$}  {:<6}  {:<4}  {:>10}  {:<4}  {:>10}",
                "MODEL", "SOURCE", "FMT", "EST. SIZE", "FITS", "DOWNLOADS",
                nw = name_w,
            );
            println!("  {}", "-".repeat(name_w + 44));

            for r in &results {
                let size = match r.estimated_bytes {
                    Some(b) => format!("~{}", search::format_size(b)),
                    None => "-".into(),
                };
                let fits = match r.estimated_bytes {
                    Some(b) if b < inference_mem => " \u{2713}  ",
                    Some(_) => " \u{2717}  ",
                    None => " ?  ",
                };
                let dl = if r.downloads > 0 {
                    search::format_downloads(r.downloads)
                } else {
                    "-".into()
                };
                let fmt = match r.format {
                    ModelFormat::Gguf => "gguf",
                    ModelFormat::Mlx => "mlx",
                };
                let display_name = if r.name.len() > max_name {
                    format!("{}...", &r.name[..max_name - 3])
                } else {
                    r.name.clone()
                };
                println!(
                    "  {:<nw$}  {:<6}  {:<4}  {:>10} {} {:>10}",
                    display_name, r.source, fmt, size, fits, dl,
                    nw = name_w,
                );
            }

            let prefers = if spindll::model_store::platform_prefers_mlx() {
                "mlx (Apple Silicon)"
            } else {
                "gguf"
            };
            let mem_label = if inference_mem < mem.total_ram {
                format!("VRAM: ~{}", search::format_size(inference_mem))
            } else {
                format!("System RAM: ~{}", search::format_size(mem.total_ram))
            };
            println!(
                "\n  {}  Preferred format: {}",
                mem_label, prefers,
            );
            println!("  Pull: spindll pull <model>");
        }
        Commands::Import { path, from_ollama, from_hf } => {
            let store = spindll::model_store::ModelStore::new(None);

            match (path, from_ollama, from_hf) {
                (Some(p), false, false) => {
                    store.import_from_path(&p)?;
                    println!("imported model from {p}");
                }
                (None, true, false) => {
                    let count = store.import_from_ollama()?;
                    println!("imported {count} model(s) from ollama");
                }
                (None, false, true) => {
                    let count = store.import_from_hf()?;
                    println!("imported {count} model(s) from huggingface cache");
                }
                (Some(_), true, _) | (Some(_), _, true) => {
                    anyhow::bail!("cannot specify both a path and --from-ollama/--from-hf");
                }
                (None, false, false) => {
                    anyhow::bail!("specify either a path, --from-ollama, or --from-hf");
                }
                _ => {
                    anyhow::bail!("--from-ollama and --from-hf are mutually exclusive");
                }
            }
        }
        Commands::Status { port } => {
            let port = match port {
                Some(p) => p,
                None => match spindll::lockfile::Lockfile::read() {
                    Some(lf) => lf.grpc_port,
                    None => {
                        anyhow::bail!("no running server found (no lockfile); specify --port")
                    }
                },
            };
            let addr = format!("http://localhost:{port}");
            let mut client =
                spindll::proto::spindll_client::SpindllClient::connect(addr)
                    .await
                    .map_err(|e| {
                        anyhow::anyhow!("cannot connect to server on port {port}: {e}")
                    })?;

            let resp = client
                .status(spindll::proto::StatusRequest {})
                .await?
                .into_inner();

            if resp.models.is_empty() {
                println!("no models loaded");
            } else {
                println!(
                    "{:<35} {:>10} {:>6}  {}",
                    "MODEL", "MEMORY", "GPU", "DIGEST"
                );
                println!("{}", "-".repeat(75));
                for m in &resp.models {
                    let size = if m.memory_used >= 1_073_741_824 {
                        format!("{:.1} GB", m.memory_used as f64 / 1_073_741_824.0)
                    } else {
                        format!("{:.0} MB", m.memory_used as f64 / 1_048_576.0)
                    };
                    let digest_short = if m.digest.len() > 19 {
                        &m.digest[..19]
                    } else {
                        &m.digest
                    };
                    println!(
                        "{:<35} {:>10} {:>4}L  {}",
                        m.name, size, m.gpu_layers, digest_short
                    );
                }
            }

            if let Some(mem) = &resp.memory {
                println!();
                println!(
                    "memory: {:.1} GB used / {:.1} GB available / {:.1} GB total",
                    (mem.total_ram - mem.available_ram) as f64 / 1_073_741_824.0,
                    mem.available_ram as f64 / 1_073_741_824.0,
                    mem.total_ram as f64 / 1_073_741_824.0,
                );
            }

            if let Some(m) = &resp.metrics {
                println!();
                println!(
                    "requests: {}  errors: {}  cache: {}/{} ({:.0}%)  avg: {:.1} tok/s",
                    m.generate_requests,
                    m.generate_errors,
                    m.cache_hits,
                    m.cache_hits + m.cache_misses,
                    m.cache_hit_rate * 100.0,
                    m.avg_tokens_per_second,
                );
            }

            if !resp.devices.is_empty() {
                println!();
                println!("devices: {}", resp.devices.join(", "));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manager_memory_budget_zero_passthrough() {
        assert_eq!(manager_memory_budget(Some("0"), 99), 0);
    }

    #[test]
    fn manager_memory_budget_none_uses_detected() {
        assert_eq!(manager_memory_budget(None, 42), 42);
    }

    #[test]
    fn manager_memory_budget_uses_detected_cap_for_explicit_size() {
        assert_eq!(manager_memory_budget(Some("8G"), 123), 123);
    }

    #[cfg(feature = "bench")]
    #[test]
    fn decode_tok_per_sec_excludes_first_token() {
        let tps = super::decode_tok_per_sec(128, 50.0, 5050.0);
        assert!((tps - 25.4).abs() < 0.1); // 127 tokens / 5.0s
    }

    #[cfg(feature = "bench")]
    #[test]
    fn decode_tok_per_sec_edge_cases() {
        assert_eq!(super::decode_tok_per_sec(0, 10.0, 100.0), 0.0);
        assert_eq!(super::decode_tok_per_sec(1, 10.0, 100.0), 0.0);
        assert_eq!(super::decode_tok_per_sec(10, 100.0, 50.0), 0.0);
    }
}
