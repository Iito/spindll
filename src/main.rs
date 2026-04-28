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

        /// Quantization filter (HuggingFace only)
        #[arg(long)]
        quant: Option<String>,
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

        /// Memory budget for loaded models (e.g. "8G", omit for 80% of available RAM)
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

        /// Enable KV cache for prompt prefixes (e.g. "2G", default 2G when enabled)
        #[arg(long)]
        kv_cache: Option<Option<String>>,
    },

    /// Benchmark two models side-by-side (any format: GGUF, MLX, or mixed)
    Bench {
        /// First model
        model: String,

        /// Second model to compare against
        against: String,

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
    ttft_ms: f64,
    tokens_per_sec: f64,
    total_ms: f64,
    tokens: u32,
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

    let format_name = match format {
        ModelFormat::Gguf => "GGUF",
        ModelFormat::Mlx => "MLX",
    };

    let backend = backend_for_format(&format)?;
    let load_params = spindll::backend::BackendLoadParams {
        n_ctx: ctx_size,
        n_gpu_layers: None,
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
        let elapsed = start.elapsed().as_secs_f64();
        ttft_sum += ttft;
        tps_sum += result.completion_tokens as f64 / elapsed;
        last_tokens = result.completion_tokens;
        let sample = phys_footprint_mb();
        if sample > mem_peak {
            mem_peak = sample;
        }
    }

    let avg_tps = tps_sum / runs as f64;
    Ok(BenchResult {
        format_name,
        ttft_ms: ttft_sum / runs as f64,
        tokens_per_sec: avg_tps,
        total_ms: last_tokens as f64 / avg_tps * 1000.0,
        tokens: last_tokens,
        mem_peak_mb: mem_peak,
    })
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
    let label = if label.len() > 40 { &label[..40] } else { label };
    println!(
        "{:<40} {:>4} {:>8.0}ms {:>8.1} {:>7.2}s {:>6}",
        label,
        r.format_name,
        r.ttft_ms,
        r.tokens_per_sec,
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
        Commands::Pull { model, quant } => {
            let store = spindll::model_store::ModelStore::new(None);
            let path = store.pull(&model, quant.as_deref())?;
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
            println!(
                "memory budget: {:.1} GB / {:.1} GB available",
                mem.budget as f64 / 1_073_741_824.0,
                mem.available_ram as f64 / 1_073_741_824.0
            );
            let mut manager =
                spindll::engine::ModelManager::new(ctx_size, gpu_layers, mem.budget)?;

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
            kv_cache,
        } => {
            let store = spindll::model_store::ModelStore::new(None);
            let format = store.resolve_model_format(&model)?;
            let model_path = store.resolve_model_path(&model)?;
            let params = spindll::engine::GenerateParams::default();

            use spindll::model_store::registry::ModelFormat;
            if kv_cache.is_some() && format == ModelFormat::Gguf {
                // KV cache path: use Engine directly (GGUF-only feature)
                let mut engine = spindll::engine::Engine::load(&model_path, None, 2048)?;
                if let Some(cache_size) = kv_cache {
                    let bytes = parse_size_bytes(cache_size.as_deref());
                    let digest = store.resolve_model_digest(&model).unwrap_or_default();
                    engine.set_model_digest(digest);
                    engine.enable_kv_cache(bytes);
                }
                engine.generate(&prompt, &params, |token| {
                    use std::io::Write;
                    print!("{token}");
                    std::io::stdout().flush().ok();
                    true
                })?;
            } else {
                let backend = backend_for_format(&format)?;
                let backend_model = backend.load_model(
                    &model_path,
                    spindll::backend::BackendLoadParams {
                        n_ctx: 2048,
                        n_gpu_layers: None,
                    },
                )?;
                backend_model.generate(&prompt, &params, &mut |token| {
                    use std::io::Write;
                    print!("{token}");
                    std::io::stdout().flush().ok();
                    true
                })?;
            }
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
                "{:<40} {:>4} {:>9} {:>9} {:>8} {:>6}",
                "MODEL", "FMT", "TTFT", "TOK/S", "TOTAL", "MEM"
            );
            println!("{}", "─".repeat(80));

            let path1 = store.resolve_model_path(&model)?;
            let fmt1 = store.resolve_model_format(&model)?;
            let r1 = bench_by_format(&path1, fmt1, prompt_str, max_tokens, runs, ctx_size)?;
            print_bench_row(&model, &r1);

            let path2 = store.resolve_model_path(&against)?;
            let fmt2 = store.resolve_model_format(&against)?;
            let r2 = bench_by_format(&path2, fmt2, prompt_str, max_tokens, runs, ctx_size)?;
            print_bench_row(&against, &r2);

            println!("{}", "─".repeat(80));

            let tps_ratio = r2.tokens_per_sec / r1.tokens_per_sec;
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
