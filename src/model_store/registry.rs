use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const CURRENT_VERSION: u32 = 2;

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

/// How the model entered Spindll's registry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSource {
    /// Downloaded from Ollama registry via `spindll pull "ollama/model:tag"`.
    OllamaSourceDownloaded,
    /// Downloaded from HuggingFace via `spindll pull "owner/repo"`.
    HfSourceDownloaded,
    /// Imported from local Ollama cache via `spindll import --from-ollama`.
    OllamaImported,
    /// Imported from local HuggingFace cache via `spindll import --from-hf`.
    HfImported,
    /// Imported from arbitrary path via `spindll import "/path/to/model"`.
    ManuallyImported,
}

impl Default for ModelSource {
    fn default() -> Self {
        Self::OllamaSourceDownloaded
    }
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
    /// How the model entered Spindll (pull vs import source).
    /// Defaults to OllamaSourceDownloaded for backward compatibility.
    #[serde(default)]
    pub source: ModelSource,
}

/// Sum the sizes of all files in a directory (non-recursive, follows symlinks).
pub(crate) fn dir_size(dir: &std::path::Path) -> std::io::Result<u64> {
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
    ///
    /// Refuses to overwrite a registry written by a newer spindll.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if path.exists() {
            let data = std::fs::read_to_string(path)?;
            if let Ok(on_disk) = serde_json::from_str::<Self>(&data) {
                if on_disk.version > CURRENT_VERSION {
                    anyhow::bail!(
                        "registry was written by a newer spindll (version {}); \
                         refusing to overwrite — upgrade spindll first",
                        on_disk.version
                    );
                }
            }
        }
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

        if self.version < 2 {
            self.infer_model_sources();
            self.version = 2;
            changed = true;
        }

        changed
    }

    /// Infer model sources for registry v0->v1 upgrade.
    /// Detects whether models were downloaded by Spindll or imported from external sources.
    fn infer_model_sources(&mut self) {
        let home = std::env::var("HOME").ok();
        let ollama_blobs = home.as_ref().map(|h| {
            std::path::PathBuf::from(h).join(".ollama/models/blobs")
        });
        let hf_cache = home.as_ref().map(|h| {
            std::path::PathBuf::from(h).join(".cache/huggingface/hub")
        });

        for entry in self.models.values_mut() {
            // Only set source if it's still at default
            if entry.source == ModelSource::OllamaSourceDownloaded {
                // Check if it's a symlink to external location
                if let Ok(target) = std::fs::read_link(&entry.path) {
                    // Check Ollama cache
                    if let Some(ref blobs) = ollama_blobs {
                        if target.starts_with(blobs) {
                            entry.source = ModelSource::OllamaImported;
                            continue;
                        }
                    }
                    // Check HuggingFace cache
                    if let Some(ref hf) = hf_cache {
                        if target.starts_with(hf) {
                            entry.source = ModelSource::HfImported;
                            continue;
                        }
                    }
                }

                // Infer from repo name for non-symlinks
                if entry.repo.starts_with("ollama/") {
                    entry.source = ModelSource::OllamaSourceDownloaded;
                } else if entry.repo.contains('/') {
                    entry.source = ModelSource::HfSourceDownloaded;
                }
            }
        }
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
        // v0: no version field, no source field
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
    fn save_refuses_to_overwrite_future_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.json");
        std::fs::write(&path, r#"{"version":99,"models":{}}"#).unwrap();

        let mut reg = Registry::load(&path).unwrap();
        reg.add("test/model.gguf".into(), ModelEntry {
            repo: "test".into(),
            filename: "model.gguf".into(),
            path: PathBuf::from("/tmp/nonexistent"),
            size_bytes: 100,
            downloaded_at: 1,
            digest: String::new(),
            model_name: String::new(),
            description: String::new(),
            architecture: String::new(),
            context_length: 0,
            metadata_read: false,
            format: ModelFormat::Gguf,
            base_model: String::new(),
            source: ModelSource::OllamaSourceDownloaded,
        });

        let result = reg.save(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("newer spindll"));

        let on_disk: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(on_disk["version"], 99);
        assert_eq!(on_disk["models"], serde_json::json!({}));
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
            source: ModelSource::OllamaSourceDownloaded,
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
