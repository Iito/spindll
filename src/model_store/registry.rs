use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Metadata for a single model file tracked in the registry.
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Source repo identifier (e.g. `"ollama/llama3.1"` or `"TheBloke/Llama-3-8B-GGUF"`).
    pub repo: String,
    /// GGUF filename on disk (e.g. `"8b.gguf"`).
    pub filename: String,
    /// Absolute path to the GGUF file (may be a symlink).
    pub path: PathBuf,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Unix timestamp when the model was downloaded or imported.
    pub downloaded_at: u64,
    /// SHA-256 content digest (e.g. `"sha256:abcdef..."`).
    #[serde(default)]
    pub digest: String,
    /// Human-readable model name from GGUF `general.name` metadata.
    #[serde(default)]
    pub model_name: String,
    /// Description from GGUF `general.description` metadata.
    #[serde(default)]
    pub description: String,
    /// Model architecture from GGUF `general.architecture` metadata (e.g. `"llama"`).
    #[serde(default)]
    pub architecture: String,
    /// Trained context length from GGUF metadata (0 if unknown).
    #[serde(default)]
    pub context_length: u32,
    /// Whether GGUF metadata has been read for this entry.
    #[serde(default)]
    pub metadata_read: bool,
}

/// Read GGUF metadata from a file without loading tensor weights.
///
/// Returns `(general.name, general.description, general.architecture, context_length)`.
pub fn read_gguf_metadata(path: &Path) -> (String, String, String, u32) {
    let ctx = match llama_cpp_2::gguf::GgufContext::from_file(path) {
        Some(c) => c,
        None => return (String::new(), String::new(), String::new(), 0),
    };

    let read_str = |key: &str| -> String {
        let idx = ctx.find_key(key);
        if idx >= 0 {
            ctx.val_str(idx).unwrap_or_default().to_string()
        } else {
            String::new()
        }
    };

    let arch = read_str("general.architecture");

    // Try <arch>.context_length first, then fall back to llama.context_length.
    let ctx_len = {
        let key = format!("{arch}.context_length");
        let idx = ctx.find_key(&key);
        if idx >= 0 {
            ctx.val_u32(idx)
        } else {
            let idx = ctx.find_key("llama.context_length");
            if idx >= 0 { ctx.val_u32(idx) } else { 0 }
        }
    };

    (
        read_str("general.name"),
        read_str("general.description"),
        arch,
        ctx_len,
    )
}

/// JSON-backed registry mapping model keys to their metadata.
///
/// Keys follow the format `"ollama/<name>/<tag>.gguf"` or `"<org>/<repo>/<file>.gguf"`.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Registry {
    /// Map from registry key to model metadata.
    pub models: HashMap<String, ModelEntry>,
}

impl Registry {
    /// Load the registry from a JSON file, or return an empty registry if the file doesn't exist.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        if path.exists() {
            let data = std::fs::read_to_string(path)?;
            let reg = serde_json::from_str(&data)?;
            Ok(reg)
        } else {
            Ok(Self::default())
        }
    }

    /// Persist the registry to a JSON file.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Insert or replace a model entry under the given key.
    pub fn add(&mut self, key: String, entry: ModelEntry) {
        self.models.insert(key, entry);
    }

    /// Backfill any entries missing GGUF metadata by reading the file header.
    /// Returns `true` if any entries were updated.
    pub fn backfill_metadata(&mut self) -> bool {
        let mut changed = false;
        for entry in self.models.values_mut() {
            if !entry.metadata_read && entry.path.exists() {
                let (name, desc, arch, ctx_len) = read_gguf_metadata(&entry.path);
                entry.model_name = name;
                entry.description = desc;
                entry.architecture = arch;
                entry.context_length = ctx_len;
                entry.metadata_read = true;
                changed = true;
            }
        }
        changed
    }

    /// Remove a model entry by key, returning it if it existed.
    pub fn remove(&mut self, key: &str) -> Option<ModelEntry> {
        self.models.remove(key)
    }
}
