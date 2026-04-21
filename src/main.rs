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

    /// Import models from Ollama
    Import {
        /// Source to import from
        #[arg(long = "from-ollama")]
        from_ollama: bool,
    },

    /// Show server status
    Status {
        /// Port of the running server
        #[arg(long, default_value = "50051")]
        port: u16,
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
            http_port,
        } => {
            let mem = spindll::scheduler::budget::MemoryBudget::detect(budget.as_deref());
            println!(
                "memory budget: {:.1} GB / {:.1} GB available",
                mem.budget as f64 / 1_073_741_824.0,
                mem.available_ram as f64 / 1_073_741_824.0
            );
            let mut manager = spindll::engine::ModelManager::new(ctx_size, gpu_layers, mem.budget)?;

            if let Some(cache_size) = kv_cache {
                let bytes = parse_size_bytes(cache_size.as_deref());
                manager.enable_kv_cache(bytes);
                println!("kv cache: {:.1} GB max", bytes as f64 / 1_073_741_824.0);
            }

            if batch_slots > 0 {
                manager.set_batch_slots(batch_slots);
                println!("batch scheduling: {batch_slots} concurrent slots per model");
            }

            let manager = std::sync::Arc::new(manager);
            let store = std::sync::Arc::new(
                spindll::model_store::ModelStore::new(None),
            );

            #[cfg(feature = "http")]
            if http_port > 0 {
                let http_mgr = manager.clone();
                let http_store = store.clone();
                tokio::spawn(async move {
                    if let Err(e) = spindll::http::start_http_server(http_port, http_mgr, http_store).await {
                        tracing::error!("HTTP server error: {e}");
                    }
                });
            }

            spindll::grpc::start_server(port, manager, store).await?;
        }
        Commands::Run { model, prompt, kv_cache } => {
            let store = spindll::model_store::ModelStore::new(None);
            let model_path = store.resolve_model_path(&model)?;

            let mut engine = spindll::engine::Engine::load(&model_path, None, 2048)?;

            if let Some(cache_size) = kv_cache {
                let bytes = parse_size_bytes(cache_size.as_deref());
                let digest = store.resolve_model_digest(&model).unwrap_or_default();
                engine.set_model_digest(digest);
                engine.enable_kv_cache(bytes);
            }

            let params = spindll::engine::GenerateParams::default();

            engine.generate(&prompt, &params, |token| {
                use std::io::Write;
                print!("{token}");
                std::io::stdout().flush().ok();
                true
            })?;
            println!();
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
            let addr = format!("http://localhost:{port}");
            let mut client = spindll::proto::spindll_client::SpindllClient::connect(addr).await
                .map_err(|e| anyhow::anyhow!("cannot connect to server on port {port}: {e}"))?;

            let resp = client.status(spindll::proto::StatusRequest {}).await?
                .into_inner();

            // Models
            if resp.models.is_empty() {
                println!("no models loaded");
            } else {
                println!("{:<35} {:>10} {:>6}  {}", "MODEL", "MEMORY", "GPU", "DIGEST");
                println!("{}", "-".repeat(75));
                for m in &resp.models {
                    let size = if m.memory_used >= 1_073_741_824 {
                        format!("{:.1} GB", m.memory_used as f64 / 1_073_741_824.0)
                    } else {
                        format!("{:.0} MB", m.memory_used as f64 / 1_048_576.0)
                    };
                    let digest_short = if m.digest.len() > 19 { &m.digest[..19] } else { &m.digest };
                    println!("{:<35} {:>10} {:>4}L  {}", m.name, size, m.gpu_layers, digest_short);
                }
            }

            // Memory
            if let Some(mem) = &resp.memory {
                println!();
                println!(
                    "memory: {:.1} GB used / {:.1} GB available / {:.1} GB total",
                    (mem.total_ram - mem.available_ram) as f64 / 1_073_741_824.0,
                    mem.available_ram as f64 / 1_073_741_824.0,
                    mem.total_ram as f64 / 1_073_741_824.0,
                );
            }

            // Metrics
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

            // Devices
            if !resp.devices.is_empty() {
                println!();
                println!("devices: {}", resp.devices.join(", "));
            }
        }
    }

    Ok(())
}
