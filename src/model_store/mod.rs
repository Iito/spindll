//! Model store — download, import, and resolve model files.
//!
//! Supports pulling from HuggingFace repos and the Ollama registry, importing
//! existing Ollama models via symlink, and resolving flexible model name formats
//! to on-disk paths. On Apple Silicon, automatically resolves to MLX-format
//! models when available.

pub mod download;
pub mod registry;
pub mod import;
pub mod mlx_resolve;
pub mod ollama_pull;

use std::path::PathBuf;

/// Caller-specified format preference for `pull()`.
#[derive(Debug, Clone, PartialEq)]
pub enum FormatPreference {
    /// Let the platform decide: MLX on Apple Silicon, GGUF elsewhere.
    Auto,
    /// Force GGUF regardless of platform.
    Gguf,
    /// Force MLX — error if not found.
    Mlx,
}

impl Default for FormatPreference {
    fn default() -> Self {
        Self::Auto
    }
}

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

    /// Pull a model with format-aware resolution.
    ///
    /// On Apple Silicon with `FormatPreference::Auto`, attempts to find an MLX-format
    /// model on HuggingFace before falling back to GGUF. HuggingFace repos
    /// are auto-detected as GGUF or MLX from their contents; Ollama-style
    /// names (e.g. `"llama3.1:8b"`) always pull GGUF unless an MLX
    /// equivalent is resolvable.
    pub fn pull(
        &self,
        model: &str,
        quant: Option<&str>,
        format_pref: FormatPreference,
    ) -> anyhow::Result<PathBuf> {
        self.ensure_dirs()?;

        let want_mlx = match format_pref {
            FormatPreference::Mlx => true,
            FormatPreference::Gguf => false,
            FormatPreference::Auto => platform_prefers_mlx(),
        };

        // If we want MLX, try to resolve an MLX repo before downloading GGUF.
        if want_mlx {
            let mlx_quant = quant.unwrap_or("4bit");
            match self.try_pull_mlx(model, mlx_quant) {
                Ok(path) => return Ok(path),
                Err(e) => {
                    if format_pref == FormatPreference::Mlx {
                        return Err(e.context("no MLX model found and --mlx was specified"));
                    }
                    tracing::info!("no MLX version found, falling back to GGUF: {e:#}");
                }
            }
        }

        let strict_gguf = format_pref == FormatPreference::Gguf;
        self.pull_gguf(model, quant, strict_gguf)
    }

    /// Resolve an MLX equivalent for `model` and download it. Errors if no
    /// matching `mlx-community/...` repo is found on HuggingFace.
    fn try_pull_mlx(&self, model: &str, mlx_quant: &str) -> anyhow::Result<PathBuf> {
        // If the user passed a full HF repo path, try it directly as an MLX
        // repo before falling back to mlx-community resolution. Lets
        // `--mlx other-org/some-mlx-repo` actually pull from other-org instead
        // of silently resolving to a mlx-community lookalike (Codex #2).
        let candidate = if model.contains('/') {
            mlx_resolve::probe_repo(model)?
                .map(Ok)
                .unwrap_or_else(|| {
                    mlx_resolve::find_mlx_repo(model, mlx_quant)?
                        .ok_or_else(|| anyhow::anyhow!("no MLX equivalent found for '{model}'"))
                })?
        } else {
            mlx_resolve::find_mlx_repo(model, mlx_quant)?
                .ok_or_else(|| anyhow::anyhow!("no MLX equivalent found for '{model}'"))?
        };

        tracing::info!(repo = %candidate.repo_id, "resolved MLX model");

        let dest_dir = self.model_dir(&candidate.repo_id);
        let (path, size_bytes, digest) = match download::download_hf_auto(&candidate.repo_id, None, &dest_dir)? {
            download::HfDownload::Mlx { dir, size, digest } => (dir, size, digest),
            download::HfDownload::Gguf { .. } => {
                anyhow::bail!(
                    "resolved repo '{}' contains GGUF, not MLX safetensors",
                    candidate.repo_id
                );
            }
        };

        let (architecture, model_name) = download::read_mlx_metadata(&path);
        // For Ollama-style aliases ("llama3.1:8b") we stamp the normalized alias
        // ("llama3.1-8b") as base_model so resolve_key's step-5 alias match
        // ("model.replace(':', '-')") finds this MLX entry. Without this, the
        // alias is unresolvable after pull because the registry key is the
        // mlx-community/... repo and resolve_key has no other path back.
        let base_model = if !model.contains('/') && model.contains(':') {
            model.replace(':', "-")
        } else {
            derive_base_model(&model_name, model)
        };
        let key = candidate.repo_id.clone();

        let mut reg = registry::Registry::load(&self.registry_path())?;
        reg.add(key, registry::ModelEntry {
            repo: candidate.repo_id,
            filename: String::new(),
            path: path.clone(),
            size_bytes,
            downloaded_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            digest,
            model_name,
            description: String::new(),
            architecture,
            context_length: 0,
            metadata_read: true,
            format: registry::ModelFormat::Mlx,
            base_model,
        });
        reg.save(&self.registry_path())?;

        Ok(path)
    }

    /// Pull a GGUF model (the original path).
    ///
    /// `strict_gguf = true` rejects HF repos that resolve to MLX-only contents,
    /// so `--gguf` does not silently land an MLX directory (Codex #2).
    fn pull_gguf(&self, model: &str, quant: Option<&str>, strict_gguf: bool) -> anyhow::Result<PathBuf> {
        let is_hf = model.contains('/');

        // --- Download & detect format ---
        let (path, size_bytes, key, digest, format) = if is_hf {
            let dest_dir = self.model_dir(model);
            match download::download_hf_auto(model, quant, &dest_dir)? {
                download::HfDownload::Gguf { path, filename, size, digest } => {
                    download::validate_gguf(&path)?;
                    let key = format!("{}/{}", model, filename);
                    (path, size, key, digest, registry::ModelFormat::Gguf)
                }
                download::HfDownload::Mlx { dir, size, digest } => {
                    if strict_gguf {
                        anyhow::bail!(
                            "'{model}' contains MLX safetensors, not GGUF — drop --gguf or pass --mlx"
                        );
                    }
                    // Registry key is just the repo ID — no filename suffix.
                    let key = model.to_string();
                    (dir, size, key, digest, registry::ModelFormat::Mlx)
                }
            }
        } else {
            let (name, _tag) = ollama_pull::parse_model_ref(model);
            let dest_dir = self.model_dir(&format!("ollama/{name}"));
            let (path, size, digest) = ollama_pull::pull_from_registry(model, &dest_dir)?;
            download::validate_gguf(&path)?;
            let filename = path.file_name().unwrap().to_string_lossy();
            let key = format!("ollama/{name}/{filename}");
            (path, size, key, digest, registry::ModelFormat::Gguf)
        };

        // --- Read metadata ---
        let (model_name, description, architecture, context_length) = match format {
            registry::ModelFormat::Gguf => registry::read_gguf_metadata(&path),
            registry::ModelFormat::Mlx  => {
                let (arch, name) = download::read_mlx_metadata(&path);
                (name, String::new(), arch, 0u32)
            }
        };

        // --- Register ---
        let base_model = derive_base_model(&model_name, model);
        let filename = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
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
            format,
            base_model,
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

        let mut entries: Vec<_> = reg.models.iter().collect();
        entries.sort_by_key(|(k, _)| (*k).clone());

        // Pre-compute rows so we can size MODEL and ARCH columns to the
        // longest entry. mlx-community paths blow past 35 chars; static
        // widths either truncated or wasted space.
        let rows: Vec<_> = entries
            .iter()
            .map(|(key, entry)| {
                let display = display_name(key, entry);
                let fmt = match entry.format {
                    registry::ModelFormat::Gguf => "gguf",
                    registry::ModelFormat::Mlx => "mlx",
                };
                let size = format_size(entry.size_bytes);
                let arch = if entry.architecture.is_empty() {
                    "-".to_string()
                } else {
                    entry.architecture.clone()
                };
                let desc = if entry.description.is_empty() {
                    entry.model_name.clone()
                } else {
                    entry.description.clone()
                };
                (display, fmt, size, arch, desc)
            })
            .collect();

        const PADDING: usize = 2;
        let model_w = rows.iter().map(|r| r.0.len()).max().unwrap_or(0).max("MODEL".len()) + PADDING;
        let arch_w  = rows.iter().map(|r| r.3.len()).max().unwrap_or(0).max("ARCH".len()) + PADDING;

        println!(
            "{:<model_w$} {:<5} {:>10}  {:<arch_w$}  {}",
            "MODEL", "FMT", "SIZE", "ARCH", "DESCRIPTION"
        );
        let total_w = model_w + 1 + 5 + 1 + 10 + 2 + arch_w + 2 + "DESCRIPTION".len();
        println!("{}", "-".repeat(total_w));
        for (model, fmt, size, arch, desc) in rows {
            println!(
                "{:<model_w$} {:<5} {:>10}  {:<arch_w$}  {}",
                model, fmt, size, arch, desc
            );
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

        // 5. Match by base_model (finds MLX entries for Ollama-style names)
        let normalized = model.replace(':', "-").replace(' ', "-");
        if let Some((key, _)) = reg.models.iter().find(|(_, e)| {
            !e.base_model.is_empty() && e.base_model.eq_ignore_ascii_case(&normalized)
        }) {
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

    /// Look up a model's on-disk format (GGUF or MLX) from the registry.
    pub fn resolve_model_format(&self, model: &str) -> anyhow::Result<registry::ModelFormat> {
        let key = self.resolve_key(model)?;
        let reg = registry::Registry::load(&self.registry_path())?;
        Ok(reg.models[&key].format.clone())
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
                #[cfg(unix)]
                std::os::unix::fs::symlink(&blob_path, &dest)?;
                #[cfg(windows)]
                if std::fs::hard_link(&blob_path, &dest).is_err() {
                    std::fs::copy(&blob_path, &dest)?;
                }
            }

            let key = format!("ollama/{name}/{filename}");
            if !reg.models.contains_key(&key) {
                let (gguf_name, gguf_desc, gguf_arch, gguf_ctx) = registry::read_gguf_metadata(&dest);
                let base_model = derive_base_model(&gguf_name, &format!("{name}:{tag}"));
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
                        format: registry::ModelFormat::Gguf,
                        base_model,
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

    /// Remove a model from the registry and delete its file or directory on disk.
    pub fn remove(&self, model: &str) -> anyhow::Result<()> {
        let mut reg = registry::Registry::load(&self.registry_path())?;
        let entry = reg.remove(model)
            .ok_or_else(|| anyhow::anyhow!("model '{}' not found", model))?;

        if entry.path.exists() {
            // MLX entries are directories of safetensors shards + config; GGUF is a single file.
            match entry.format {
                registry::ModelFormat::Mlx => std::fs::remove_dir_all(&entry.path)?,
                registry::ModelFormat::Gguf => std::fs::remove_file(&entry.path)?,
            }
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
/// Derive a canonical base model name from GGUF metadata or the user-provided model string.
///
/// Prefers `general.name` from GGUF metadata (most reliable), falling back to
/// cleaning up the repo/model string by stripping GGUF-specific suffixes and org prefixes.
fn derive_base_model(gguf_name: &str, model: &str) -> String {
    // Use GGUF general.name if available — normalize spaces to hyphens.
    if !gguf_name.is_empty() {
        return gguf_name.replace(' ', "-");
    }

    // HuggingFace GGUF repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF" → "Meta-Llama-3.1-8B-Instruct"
    if model.contains('/') {
        let repo_part = model.rsplit('/').next().unwrap_or(model);
        let stripped = repo_part
            .strip_suffix("-GGUF")
            .or_else(|| repo_part.strip_suffix("-gguf"))
            .or_else(|| repo_part.strip_suffix("-quantized"))
            .unwrap_or(repo_part);
        return stripped.to_string();
    }

    // Ollama name — just return as-is for now, HF search is fuzzy enough
    model.to_string()
}

/// Human-readable display name for a registry entry.
///
/// Disambiguates by quant when the same repo holds multiple GGUF variants:
/// `Qwen/Qwen2.5-3B-Instruct-GGUF` becomes `Qwen/Qwen2.5-3B-Instruct-GGUF (q4_k_m)`.
/// Ollama entries keep their `name:tag` form (already disambiguated by tag).
/// MLX entries return `repo` as-is — mlx-community names already encode
/// quant in the repo string (`...-4bit`).
pub fn display_name(key: &str, entry: &registry::ModelEntry) -> String {
    match entry.format {
        registry::ModelFormat::Mlx => {
            if entry.repo.is_empty() { key.to_string() } else { entry.repo.clone() }
        }
        registry::ModelFormat::Gguf => {
            // Ollama: registry key is `ollama/<name>/<tag>.gguf` → `<name>:<tag>`.
            let parts: Vec<&str> = key.splitn(3, '/').collect();
            if let [provider, name, file] = parts.as_slice() {
                if *provider == "ollama" {
                    let tag = file.strip_suffix(".gguf").unwrap_or(file);
                    return format!("{name}:{tag}");
                }
            }
            // HF: `<repo> (<quant>)` when we can detect the quant, else just repo.
            let base = if entry.repo.is_empty() { key } else { entry.repo.as_str() };
            match download::extract_quant(&entry.filename) {
                Some(q) => format!("{base} ({q})"),
                None => base.to_string(),
            }
        }
    }
}

/// Returns true if this platform should prefer MLX over GGUF.
pub fn platform_prefers_mlx() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_store::registry::{ModelEntry, ModelFormat, Registry};

    fn write_entry(path: &std::path::Path, key: &str, entry: ModelEntry) {
        let mut reg = Registry::load(path).unwrap();
        reg.add(key.to_string(), entry);
        reg.save(path).unwrap();
    }

    fn mlx_entry(repo: &str, base_model: &str) -> ModelEntry {
        ModelEntry {
            repo: repo.to_string(),
            filename: String::new(),
            path: std::path::PathBuf::from("/tmp/nonexistent"),
            size_bytes: 0,
            downloaded_at: 0,
            digest: String::new(),
            model_name: String::new(),
            description: String::new(),
            architecture: String::new(),
            context_length: 0,
            metadata_read: true,
            format: ModelFormat::Mlx,
            base_model: base_model.to_string(),
        }
    }

    /// Regression for Codex finding #1: pulling `llama3.1:8b --mlx` registers
    /// under the mlx-community key, and `spindll run llama3.1:8b` must still
    /// find it. The MLX pull path now stamps the normalized alias as
    /// `base_model` so resolve_key step 5 hits.
    #[test]
    fn resolve_key_finds_mlx_by_ollama_alias() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        std::fs::create_dir_all(store.models_dir()).unwrap();
        write_entry(
            &store.registry_path(),
            "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            mlx_entry("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "llama3.1-8b"),
        );

        let resolved = store.resolve_key("llama3.1:8b").unwrap();
        assert_eq!(resolved, "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit");
    }

    /// Regression for Codex finding #5: removing an MLX entry must use
    /// remove_dir_all, not remove_file (which errors on directories).
    #[test]
    fn remove_mlx_handles_directory() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        std::fs::create_dir_all(store.models_dir()).unwrap();

        // Real on-disk MLX layout: dir with a config + safetensors shard.
        let model_dir = store.models_dir().join("mlx-community/test-4bit");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("config.json"), "{}").unwrap();
        std::fs::write(model_dir.join("model.safetensors"), b"fake").unwrap();

        let mut entry = mlx_entry("mlx-community/test-4bit", "test-4bit");
        entry.path = model_dir.clone();
        write_entry(
            &store.registry_path(),
            "mlx-community/test-4bit",
            entry,
        );

        store.remove("mlx-community/test-4bit").expect("remove should succeed for MLX dir");
        assert!(!model_dir.exists(), "MLX dir should be deleted");
        let reg = Registry::load(&store.registry_path()).unwrap();
        assert!(!reg.models.contains_key("mlx-community/test-4bit"));
    }
}
