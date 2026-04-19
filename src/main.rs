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
    /// Download a model from HuggingFace
    Pull {
        /// HuggingFace repo (e.g. "TheBloke/Llama-3-8B-GGUF")
        repo: String,

        /// Quantization filter (e.g. "Q4_K_M")
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
        Commands::Pull { repo, quant } => {
            let store = spindll::model_store::ModelStore::new(None);
            let path = store.pull(&repo, quant.as_deref())?;
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
        } => {
            let store = spindll::model_store::ModelStore::new(None);
            let model_path = store.resolve_model_path(&model)?;

            let engine = spindll::engine::Engine::load(&model_path, gpu_layers, ctx_size)?;
            let engine = std::sync::Arc::new(engine);
            let store = std::sync::Arc::new(store);

            spindll::grpc::start_server(port, engine, store).await?;
        }
        Commands::Run { model, prompt } => {
            println!("run: model={model}, prompt={prompt}");
        }
        Commands::Import { from_ollama } => {
            println!("import: from_ollama={from_ollama}");
        }
        Commands::Status => {
            println!("status: querying server");
        }
    }

    Ok(())
}
