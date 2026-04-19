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

    /// Start the gRPC server
    Serve {
        /// Model to load (registry key from `spindll list`)
        model: String,

        /// Port to listen on
        #[arg(long, default_value = "50051")]
        port: u16,

        /// Context size
        #[arg(long, default_value = "2048")]
        ctx_size: u32,

        /// GPU layers (omit to auto-detect)
        #[arg(long)]
        gpu_layers: Option<u32>,

        /// Memory budget (e.g. "8G")
        #[arg(long)]
        budget: Option<String>,
    },

    /// One-shot inference (no server needed)
    Run {
        /// Model to use
        model: String,

        /// Prompt text
        prompt: String,
    },

    /// Import models from Ollama
    Import {
        /// Source to import from
        #[arg(long = "from-ollama")]
        from_ollama: bool,
    },

    /// Show server status
    Status,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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
            model,
            port,
            ctx_size,
            gpu_layers,
            budget,
        } => {
            let store = spindll::model_store::ModelStore::new(None);
            let model_path = store.resolve_model_path(&model)?;

            // Check memory budget before loading
            let mem = spindll::scheduler::budget::MemoryBudget::detect(budget.as_deref());
            let model_size = std::fs::metadata(&model_path)?.len();
            if !mem.can_fit(model_size) {
                anyhow::bail!(
                    "model is {:.1} GB but memory budget is {:.1} GB",
                    model_size as f64 / 1_073_741_824.0,
                    mem.budget as f64 / 1_073_741_824.0
                );
            }

            let engine = spindll::engine::Engine::load(&model_path, gpu_layers, ctx_size)?;
            let engine = std::sync::Arc::new(engine);
            let store = std::sync::Arc::new(store);

            spindll::grpc::start_server(port, engine, store).await?;
        }
        Commands::Run { model, prompt } => {
            println!("run: model={model}, prompt={prompt}");
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
        Commands::Status => {
            println!("status: querying server");
        }
    }

    Ok(())
}
