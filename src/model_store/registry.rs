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

    /// Remove a model entry by key, returning it if it existed.
    pub fn remove(&mut self, key: &str) -> Option<ModelEntry> {
        self.models.remove(key)
    }
}
