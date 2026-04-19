// main.rs — the entry point, like Python's `if __name__ == "__main__"`.
//
// Uses `clap` to parse CLI arguments. The #[derive(Parser)] macro
// auto-generates the argument parser from struct definitions —
// similar to Python's argparse, but type-safe and checked at compile time.

use clap::{Parser, Subcommand};

/// Spindll — a Rust-native GGUF inference engine.
#[derive(Parser)]
#[command(name = "spindll", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Each variant here becomes a subcommand (like `spindll pull`, `spindll serve`).
/// Think of it as an enum where each variant holds its own arguments.
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
        /// Port to listen on
        #[arg(long, default_value = "50051")]
        port: u16,

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

fn main() {
    let cli = Cli::parse();

    // For now, just print which command was invoked.
    // Each arm will call into the real implementation in Phase 1.
    match cli.command {
        Commands::Pull { repo, quant } => {
            println!("pull: repo={repo}, quant={quant:?}");
        }
        Commands::List => {
            println!("list: showing local models");
        }
        Commands::Rm { model } => {
            println!("rm: deleting {model}");
        }
        Commands::Serve { port, budget } => {
            println!("serve: port={port}, budget={budget:?}");
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
}
