//! Model store — download, import, and resolve GGUF model files.
//!
//! Supports pulling from HuggingFace repos and the Ollama registry, importing
//! existing Ollama models via symlink, and resolving flexible model name formats
//! to on-disk GGUF paths.

pub mod download;
pub mod registry;
pub mod import;
pub mod ollama_pull;

use std::path::PathBuf;

/// Local model store backed by `~/.spindll` (or a custom directory).
///
/// Manages a registry of downloaded/imported GGUF models and provides
/// name resolution so callers can refer to models by short names like
/// `"llama3.1:8b"` instead of full paths.
pub struct ModelStore {
    base_dir: PathBuf,
}

impl ModelStore {
    /// Create a store rooted at the given directory, or `~/.spindll` if `None`.
    pub fn new(base_dir: Option<PathBuf>) -> Self {
        let base_dir = base_dir.unwrap_or_else(|| {
            let home = std::env::var("HOME").expect("HOME not set");
            PathBuf::from(home).join(".spindll")
        });
        Self { base_dir }
    }

    /// Path to the directory containing all downloaded model files.
    pub fn models_dir(&self) -> PathBuf {
        self.base_dir.join("models")
    }

    /// Path to the subdirectory for a specific repo (e.g. `models/ollama/llama3.1`).
    pub fn model_dir(&self, repo: &str) -> PathBuf {
        self.models_dir().join(repo)
    }

    /// Path to the `registry.json` file that tracks all known models.
    pub fn registry_path(&self) -> PathBuf {
        self.base_dir.join("registry.json")
    }

    /// Create the models directory tree if it doesn't exist.
    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(self.models_dir())
    }

    /// Pull a model. Auto-detects source:
    /// - Contains "/" → HuggingFace (e.g. "TheBloke/Llama-3-8B-GGUF")
    /// - Otherwise → Ollama registry (e.g. "llama3.1:8b")
    pub fn pull(&self, model: &str, quant: Option<&str>) -> anyhow::Result<PathBuf> {
        self.ensure_dirs()?;

        let is_hf = model.contains('/');

        let (path, size_bytes, key, digest) = if is_hf {
            let dest_dir = self.model_dir(model);
            let path = download::download_gguf(model, quant, &dest_dir)?;
            let size = std::fs::symlink_metadata(&path)?.len();
            let filename = path.file_name().unwrap().to_string_lossy();
            let key = format!("{}/{}", model, filename);
            let digest = download::sha256_file(&path)?;
            (path, size, key, digest)
        } else {
            let (name, _tag) = ollama_pull::parse_model_ref(model);
            let dest_dir = self.model_dir(&format!("ollama/{name}"));
            let (path, size, digest) = ollama_pull::pull_from_registry(model, &dest_dir)?;
            let filename = path.file_name().unwrap().to_string_lossy();
            let key = format!("ollama/{name}/{filename}");
            (path, size, key, digest)
        };

        download::validate_gguf(&path)?;

        let (model_name, description, architecture, context_length) = registry::read_gguf_metadata(&path);
        let filename = path.file_name().unwrap().to_string_lossy().to_string();
        let mut reg = registry::Registry::load(&self.registry_path())?;
        reg.add(key, registry::ModelEntry {
            repo: model.to_string(),
            filename,
            path: path.clone(),
            size_bytes,
            downloaded_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            digest,
            model_name,
            description,
            architecture,
            context_length,
            metadata_read: true,
        });
        reg.save(&self.registry_path())?;

        Ok(path)
    }

    /// Print all registered models to stdout in a tabular format.
    pub fn list(&self) -> anyhow::Result<()> {
        let mut reg = registry::Registry::load(&self.registry_path())?;
        if reg.backfill_metadata() {
            reg.save(&self.registry_path())?;
        }
        if reg.models.is_empty() {
            println!("no models downloaded");
            return Ok(());
        }

        println!("{:<35} {:>10}  {:<10}  {}", "MODEL", "SIZE", "ARCH", "DESCRIPTION");
        println!("{}", "-".repeat(85));
        let mut entries: Vec<_> = reg.models.iter().collect();
        entries.sort_by_key(|(k, _)| (*k).clone());
        for (key, entry) in entries {
            let display_name = format_model_name(key);
            let size = format_size(entry.size_bytes);
            let arch = if entry.architecture.is_empty() { "-" } else { &entry.architecture };
            let desc = if entry.description.is_empty() {
                entry.model_name.as_str()
            } else {
                &entry.description
            };
            println!("{:<35} {:>10}  {:<10}  {}", display_name, size, arch, desc);
        }
        Ok(())
    }

    /// Resolve any model name format to its canonical registry key.
    ///
    /// Accepted formats (tried in order):
    ///   - Exact registry key:   `ollama/llama3.1/8b.gguf`
    ///   - Ollama name+tag:      `llama3.1:8b`  → `ollama/llama3.1/8b.gguf`
    ///   - Ollama name only:     `llama3.1`     → first matching `ollama/llama3.1/*.gguf`
    ///   - HuggingFace repo:     `TheBloke/Llama-3-8B-GGUF` → first matching key
    pub fn resolve_key(&self, model: &str) -> anyhow::Result<String> {
        let reg = registry::Registry::load(&self.registry_path())?;

        // 1. Exact match
        if reg.models.contains_key(model) {
            return Ok(model.to_string());
        }

        // 2. Ollama name:tag  →  ollama/name/tag.gguf
        if let Some((name, tag)) = model.split_once(':') {
            let key = format!("ollama/{name}/{tag}.gguf");
            if reg.models.contains_key(&key) {
                return Ok(key);
            }
        }

        // 3. Bare name  →  first ollama/name/*.gguf entry
        let prefix = format!("ollama/{model}/");
        if let Some(key) = reg.models.keys().find(|k| k.starts_with(&prefix)) {
            return Ok(key.clone());
        }

        // 4. HuggingFace repo prefix
        let hf_prefix = format!("{model}/");
        if let Some(key) = reg.models.keys().find(|k| k.starts_with(&hf_prefix)) {
            return Ok(key.clone());
        }

        anyhow::bail!(
            "model '{}' not found in registry — run: spindll pull {}",
            model, model
        )
    }

    /// Look up a model key in the registry and return the path to the GGUF file.
    /// Accepts any format that `resolve_key` understands.
    pub fn resolve_model_path(&self, model: &str) -> anyhow::Result<PathBuf> {
        let key = self.resolve_key(model)?;
        let reg = registry::Registry::load(&self.registry_path())?;
        let path = &reg.models[&key].path;
        std::fs::canonicalize(path)
            .map_err(|_| anyhow::anyhow!("model file missing: {}", path.display()))
    }

    /// Look up a model's digest from the registry.
    pub fn resolve_model_digest(&self, model: &str) -> anyhow::Result<String> {
        let key = self.resolve_key(model)?;
        let reg = registry::Registry::load(&self.registry_path())?;
        Ok(reg.models[&key].digest.clone())
    }

    /// Import all models from Ollama's local storage.
    pub fn import_from_ollama(&self) -> anyhow::Result<u32> {
        self.ensure_dirs()?;
        let ollama = import::ollama_dir();
        let models = import::discover_models(&ollama)?;

        if models.is_empty() {
            println!("no ollama models found");
            return Ok(0);
        }

        let mut reg = registry::Registry::load(&self.registry_path())?;
        let mut imported = 0u32;

        for (name, tag, manifest_path) in &models {
            let manifest = match import::parse_manifest(manifest_path) {
                Ok(m) => m,
                Err(e) => {
                    tracing::warn!(name, tag, error = %e, "skipping model: manifest parse error");
                    continue;
                }
            };

            let layer = match manifest.model_layer() {
                Some(l) => l,
                None => {
                    tracing::warn!(name, tag, "skipping model: no model layer found");
                    continue;
                }
            };

            let blob_path = import::digest_to_blob_path(&ollama, &layer.digest);
            if !blob_path.exists() {
                tracing::warn!(name, tag, path = %blob_path.display(), "skipping model: blob missing");
                continue;
            }

            // Symlink into spindll store
            let dest_dir = self.model_dir(&format!("ollama/{name}"));
            std::fs::create_dir_all(&dest_dir)?;
            let filename = format!("{tag}.gguf");
            let dest = dest_dir.join(&filename);

            if !dest.exists() {
                std::os::unix::fs::symlink(&blob_path, &dest)?;
            }

            let key = format!("ollama/{name}/{filename}");
            if !reg.models.contains_key(&key) {
                let (gguf_name, gguf_desc, gguf_arch, gguf_ctx) = registry::read_gguf_metadata(&dest);
                reg.add(
                    key.clone(),
                    registry::ModelEntry {
                        repo: format!("ollama/{name}"),
                        filename: filename.clone(),
                        path: dest,
                        size_bytes: layer.size,
                        downloaded_at: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        digest: layer.digest.clone(),
                        model_name: gguf_name,
                        description: gguf_desc,
                        architecture: gguf_arch,
                        context_length: gguf_ctx,
                        metadata_read: true,
                    },
                );
                println!("imported {name}:{tag} ({:.1} GB)", layer.size as f64 / 1_073_741_824.0);
                imported += 1;
            } else {
                println!("already imported {name}:{tag}");
            }
        }

        reg.save(&self.registry_path())?;
        Ok(imported)
    }

    /// Remove a model from the registry and delete its file on disk.
    pub fn remove(&self, model: &str) -> anyhow::Result<()> {
        let mut reg = registry::Registry::load(&self.registry_path())?;
        let entry = reg.remove(model)
            .ok_or_else(|| anyhow::anyhow!("model '{}' not found", model))?;

        if entry.path.exists() {
            std::fs::remove_file(&entry.path)?;
        }

        reg.save(&self.registry_path())?;
        println!("deleted {}", model);
        Ok(())
    }
}

/// Convert a registry key to a friendly display name.
///
/// - `ollama/nemotron-3-nano/4b.gguf` → `nemotron-3-nano:4b`
/// - `TheBloke/Llama-3-8B-GGUF/model.gguf` → `TheBloke/Llama-3-8B-GGUF:model`
fn format_model_name(key: &str) -> String {
    let parts: Vec<&str> = key.splitn(3, '/').collect();
    match parts.as_slice() {
        [provider, name, file] if *provider == "ollama" => {
            let tag = file.strip_suffix(".gguf").unwrap_or(file);
            format!("{name}:{tag}")
        }
        [org, repo, file] => {
            let tag = file.strip_suffix(".gguf").unwrap_or(file);
            format!("{org}/{repo}:{tag}")
        }
        _ => key.to_string(),
    }
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else {
        format!("{} KB", bytes / 1024)
    }
}
