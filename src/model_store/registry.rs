use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const CURRENT_VERSION: u32 = 1;

/// On-disk format of a model entry.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    /// Single GGUF file (llama.cpp backend).
    #[default]
    Gguf,
    /// Directory of safetensors + config.json (MLX Swift backend).
    Mlx,
}

/// Metadata for a single model tracked in the registry.
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
    /// Whether metadata has been read for this entry.
    #[serde(default)]
    pub metadata_read: bool,
    /// On-disk format: GGUF (single file) or MLX (safetensors directory).
    /// Defaults to `Gguf` so existing registry entries are unaffected.
    #[serde(default)]
    pub format: ModelFormat,
    /// Canonical base model identity (e.g. "Meta-Llama-3.1-8B-Instruct"),
    /// used for cross-format matching.
    #[serde(default)]
    pub base_model: String,
}

/// Sum the sizes of all files in a directory (non-recursive, follows symlinks).
fn dir_size(dir: &std::path::Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        total += std::fs::metadata(entry.path())?.len();
    }
    Ok(total)
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
    #[serde(default)]
    pub version: u32,
    /// Map from registry key to model metadata.
    pub models: HashMap<String, ModelEntry>,
}

impl Registry {
    /// Load the registry from a JSON file, or return an empty registry if the file doesn't exist.
    ///
    /// Automatically migrates older registry versions and warns if the file
    /// was written by a newer spindll.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let mut reg: Self = if path.exists() {
            let data = std::fs::read_to_string(path)?;
            serde_json::from_str(&data)?
        } else {
            return Ok(Self { version: CURRENT_VERSION, ..Self::default() });
        };

        if reg.version > CURRENT_VERSION {
            tracing::warn!(
                file_version = reg.version,
                binary_version = CURRENT_VERSION,
                "registry was written by a newer spindll — proceeding read-only"
            );
            return Ok(reg);
        }

        if reg.migrate() {
            reg.save(path)?;
        }

        Ok(reg)
    }

    /// Persist the registry to a JSON file.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Run all pending migrations and return true if anything changed.
    pub fn migrate(&mut self) -> bool {
        let mut changed = false;

        if self.version < 1 {
            self.backfill_metadata();
            self.version = 1;
            changed = true;
        }

        changed
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
            if (!entry.metadata_read || entry.context_length == 0)
                && entry.path.exists()
                && entry.format == ModelFormat::Gguf
            {
                let (name, desc, arch, ctx_len) = read_gguf_metadata(&entry.path);
                entry.model_name = name;
                entry.description = desc;
                entry.architecture = arch;
                entry.context_length = ctx_len;
                entry.metadata_read = true;
                changed = true;
            }

            // Backfill missing size for MLX directory entries (size was stored as 0 due to
            // symlink_metadata not following HF hub symlinks).
            if entry.size_bytes == 0 && entry.format == ModelFormat::Mlx && entry.path.is_dir() {
                if let Ok(size) = dir_size(&entry.path) {
                    entry.size_bytes = size;
                    changed = true;
                }
            }
        }
        changed
    }

    /// Remove a model entry by key, returning it if it existed.
    pub fn remove(&mut self, key: &str) -> Option<ModelEntry> {
        self.models.remove(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_v0_migrates_to_current() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");
        // v0: no version field
        std::fs::write(&path, r#"{"models":{}}"#).unwrap();

        let reg = Registry::load(&path).unwrap();
        assert_eq!(reg.version, CURRENT_VERSION);

        // Re-read: should already be current, no migration
        let reg2 = Registry::load(&path).unwrap();
        assert_eq!(reg2.version, CURRENT_VERSION);
    }

    #[test]
    fn load_current_version_no_migration() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");
        let reg = Registry { version: CURRENT_VERSION, ..Default::default() };
        reg.save(&path).unwrap();

        let data_before = std::fs::read_to_string(&path).unwrap();
        let _ = Registry::load(&path).unwrap();
        let data_after = std::fs::read_to_string(&path).unwrap();
        assert_eq!(data_before, data_after);
    }

    #[test]
    fn load_future_version_no_crash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");
        std::fs::write(&path, r#"{"version":99,"models":{}}"#).unwrap();

        let reg = Registry::load(&path).unwrap();
        assert_eq!(reg.version, 99);
    }

    #[test]
    fn roundtrip_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");

        let mut reg = Registry { version: CURRENT_VERSION, ..Default::default() };
        reg.add("test/model.gguf".into(), ModelEntry {
            repo: "test".into(),
            filename: "model.gguf".into(),
            path: PathBuf::from("/tmp/model.gguf"),
            size_bytes: 1000,
            downloaded_at: 12345,
            digest: "sha256:abc".into(),
            model_name: "Test Model".into(),
            description: String::new(),
            architecture: "llama".into(),
            context_length: 4096,
            metadata_read: true,
            format: ModelFormat::Gguf,
            base_model: "Test-Model-7B".into(),
        });
        reg.save(&path).unwrap();

        let loaded = Registry::load(&path).unwrap();
        assert_eq!(loaded.version, CURRENT_VERSION);
        assert_eq!(loaded.models.len(), 1);
        let entry = &loaded.models["test/model.gguf"];
        assert_eq!(entry.base_model, "Test-Model-7B");
        assert_eq!(entry.format, ModelFormat::Gguf);
    }
}
