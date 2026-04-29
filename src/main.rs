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

        /// Memory budget (e.g. "8G"; omit=live RAM, "0"=total RAM)
        #[arg(long)]
        budget: Option<String>,

        /// Enable KV cache for prompt prefixes (e.g. "2G", default 2G when enabled)
        #[arg(long)]
        kv_cache: Option<Option<String>>,

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

        /// GGUF context size (0 = budget-safe auto)
        #[arg(long, default_value = "0")]
        ctx_size: u32,

        /// Memory budget (e.g. "8G"; omit=live RAM, "0"=total RAM)
        #[arg(long)]
        budget: Option<String>,

        /// Enable KV cache for prompt prefixes (e.g. "2G", default 2G when enabled)
        #[arg(long)]
        kv_cache: Option<Option<String>>,
    },

    /// Benchmark one or two models (any format: GGUF, MLX, or mixed)
    Bench {
        /// First model
        model: String,

        /// Optional second model to compare against
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
    },

    /// Import models from Ollama
    Import {
        /// Source to import from
        #[arg(long = "from-ollama")]
        from_ollama: bool,
    },

    /// Show server status
    Status {
        /// Port of the running server (auto-detected from lockfile if omitted)
        #[arg(long)]
        port: Option<u16>,
    },
}

fn parse_size_bytes(s: Option<&str>) -> u64 {
    const DEFAULT: u64 = 2 * 1_073_741_824; // 2 GB
    s.and_then(parse_size_arg).unwrap_or(DEFAULT)
}

fn parse_size_arg(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    let (num, mult) = if s.ends_with('G') || s.ends_with('g') {
        (&s[..s.len() - 1], 1_073_741_824u64)
    } else if s.ends_with('M') || s.ends_with('m') {
        (&s[..s.len() - 1], 1_048_576u64)
    } else {
        (s, 1u64)
    };

    num.parse::<f64>().ok().map(|n| (n * mult as f64) as u64)
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
// Benchmark helpers
// ---------------------------------------------------------------------------

struct BenchResult {
    format_name: &'static str,
    prompt_tokens: u32,
    completion_tokens: u32,
    ttft_ms: f64,
    decode_tokens_per_sec: f64,
    total_tokens_per_sec: f64,
    total_ms: f64,
    mem_peak_mb: f64,
}

#[cfg(target_os = "macos")]
fn phys_footprint_mb() -> f64 {
    use std::mem;
    unsafe {
        let mut info: libc::mach_task_basic_info = mem::zeroed();
        let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
        let ret = libc::task_info(
            libc::mach_task_self(),
            libc::MACH_TASK_BASIC_INFO as libc::task_flavor_t,
            &mut info as *mut libc::mach_task_basic_info as libc::task_info_t,
            &mut count,
        );
        if ret == libc::KERN_SUCCESS {
            info.resident_size as f64 / (1024.0 * 1024.0)
        } else {
            0.0
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn phys_footprint_mb() -> f64 {
    0.0
}

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
    let mut decode_tps_sum = 0.0f64;
    let mut total_tps_sum = 0.0f64;
    let mut total_ms_sum = 0.0f64;
    let mut prompt_token_sum = 0u64;
    let mut completion_token_sum = 0u64;
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
        let elapsed = start.elapsed().as_secs_f64();
        let (decode_tps, total_tps) = bench_sample_rates(result.completion_tokens, elapsed, ttft);
        ttft_sum += ttft;
        decode_tps_sum += decode_tps;
        total_tps_sum += total_tps;
        total_ms_sum += elapsed * 1000.0;
        prompt_token_sum += u64::from(result.prompt_tokens);
        completion_token_sum += u64::from(result.completion_tokens);
        let sample = phys_footprint_mb();
        if sample > mem_peak {
            mem_peak = sample;
        }
    }

    Ok(BenchResult {
        format_name,
        prompt_tokens: average_tokens(prompt_token_sum, runs),
        completion_tokens: average_tokens(completion_token_sum, runs),
        ttft_ms: ttft_sum / runs as f64,
        decode_tokens_per_sec: decode_tps_sum / runs as f64,
        total_tokens_per_sec: total_tps_sum / runs as f64,
        total_ms: total_ms_sum / runs as f64,
        mem_peak_mb: mem_peak,
    })
}

fn bench_sample_rates(completion_tokens: u32, total_seconds: f64, ttft_ms: f64) -> (f64, f64) {
    if completion_tokens == 0 || total_seconds <= 0.0 {
        return (0.0, 0.0);
    }

    let tokens = completion_tokens as f64;
    let total_tps = tokens / total_seconds;
    let decode_seconds = total_seconds - ttft_ms / 1000.0;
    let decode_tps = if decode_seconds > 0.0 {
        tokens / decode_seconds
    } else {
        0.0
    };
    (decode_tps, total_tps)
}

fn average_tokens(sum: u64, runs: u32) -> u32 {
    ((sum as f64 / runs as f64).round()).min(u32::MAX as f64) as u32
}

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

fn print_bench_row(label: &str, r: &BenchResult) {
    let label = if label.len() > 36 { &label[..36] } else { label };
    let tokens = format!("{}/{}", r.prompt_tokens, r.completion_tokens);
    println!(
        "{:<36} {:>4} {:>9} {:>7.0}ms {:>9.1} {:>9.1} {:>7.2}s {:>6}",
        label,
        r.format_name,
        tokens,
        r.ttft_ms,
        r.decode_tokens_per_sec,
        r.total_tokens_per_sec,
        r.total_ms / 1000.0,
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
        Commands::Rm { model } => {
            let store = spindll::model_store::ModelStore::new(None);
            store.remove(&model)?;
        }
        Commands::Serve {
            port,
            ctx_size,
            gpu_layers,
            budget,
            kv_cache,
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
            let manager_budget = manager_memory_budget(budget.as_deref(), mem.budget);
            let mut manager =
                spindll::engine::ModelManager::new(ctx_size, gpu_layers, manager_budget)?;

            if let Some(cache_size) = kv_cache {
                let bytes = parse_size_bytes(cache_size.as_deref());
                manager.enable_kv_cache(bytes);
                println!("kv cache: {:.1} GB max", bytes as f64 / 1_073_741_824.0);
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

            let manager = std::sync::Arc::new(manager);
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
            ctx_size,
            budget,
            kv_cache,
        } => {
            let store = spindll::model_store::ModelStore::new(None);
            let model_path = store.resolve_model_path(&model)?;
            let digest = store.resolve_model_digest(&model).unwrap_or_default();
            let params = spindll::engine::GenerateParams::default();
            let mem = spindll::scheduler::budget::MemoryBudget::detect(budget.as_deref());
            let manager_budget = manager_memory_budget(budget.as_deref(), mem.budget);
            let mut manager = spindll::engine::ModelManager::new(ctx_size, None, manager_budget)?;

            if let Some(cache_size) = kv_cache {
                let bytes = parse_size_bytes(cache_size.as_deref());
                manager.enable_kv_cache(bytes);
            }

            manager.load_model_with_digest(&model, &model_path, None, digest)?;
            manager.generate(&model, &prompt, &params, None, |token| {
                use std::io::Write;
                print!("{token}");
                std::io::stdout().flush().ok();
                true
            })?;
            println!();
        }
        Commands::Bench {
            model,
            against,
            runs,
            max_tokens,
            ctx_size,
            prompt,
        } => {
            let store = spindll::model_store::ModelStore::new(None);
            let default_prompt =
                "Explain how transformers work in machine learning, step by step.";
            let prompt_str = prompt.as_deref().unwrap_or(default_prompt);

            println!("prompt    {:?}", prompt_str);
            println!(
                "runs      {} (+ 1 warmup)  max-tokens={}  ctx-size={} (GGUF)",
                runs, max_tokens, ctx_size
            );
            println!("{}", "─".repeat(80));
            println!(
                "{:<36} {:>4} {:>9} {:>8} {:>9} {:>9} {:>8} {:>6}",
                "MODEL", "FMT", "P/C TOK", "TTFT", "DEC/S", "TOTAL/S", "TOTAL", "MEM"
            );
            println!("{}", "─".repeat(80));

            let path1 = store.resolve_model_path(&model)?;
            let fmt1 = store.resolve_model_format(&model)?;
            let r1 = bench_by_format(&path1, fmt1, prompt_str, max_tokens, runs, ctx_size)?;
            print_bench_row(&model, &r1);

            let Some(against) = against else {
                println!("{}", "─".repeat(80));
                return Ok(());
            };

            let path2 = store.resolve_model_path(&against)?;
            let fmt2 = store.resolve_model_format(&against)?;
            let r2 = bench_by_format(&path2, fmt2, prompt_str, max_tokens, runs, ctx_size)?;
            print_bench_row(&against, &r2);

            println!("{}", "─".repeat(80));

            let tps_ratio = r2.total_tokens_per_sec / r1.total_tokens_per_sec;
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
                "{} is {:.0}% faster total throughput, {:.0}% faster TTFT vs {}",
                fl, tps_pct, ttft_pct, sl,
            );
        }
        Commands::Import { from_ollama } => {
            if from_ollama {
                let store = spindll::model_store::ModelStore::new(None);
                let count = store.import_from_ollama()?;
                println!("imported {count} model(s) from ollama");
            } else {
                anyhow::bail!("specify --from-ollama");
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
                    "{:<35} {:>10} {:>6}  DIGEST",
                    "MODEL", "MEMORY", "GPU"
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
    fn manager_memory_budget_preserves_zero_as_total_ram_mode() {
        assert_eq!(manager_memory_budget(Some("0"), 123), 0);
    }

    #[test]
    fn manager_memory_budget_uses_detected_live_budget_when_omitted() {
        assert_eq!(manager_memory_budget(None, 123), 123);
    }

    #[test]
    fn manager_memory_budget_uses_detected_cap_for_explicit_size() {
        assert_eq!(manager_memory_budget(Some("8G"), 123), 123);
    }

    #[test]
    fn bench_sample_rates_split_decode_from_total_elapsed() {
        let (decode_tps, total_tps) = bench_sample_rates(128, 5.859_230_309, 42.624_710);
        assert!((total_tps - 21.85).abs() < 0.01);
        assert!((decode_tps - 22.01).abs() < 0.01);
    }

    #[test]
    fn average_tokens_rounds_across_runs() {
        assert_eq!(average_tokens(385, 3), 128);
    }
}
