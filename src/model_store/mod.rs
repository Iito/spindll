// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Iito <https://github.com/Iito> and sarmientoF <https://github.com/sarmientoF>

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
pub mod search;

use std::path::PathBuf;

/// Caller-specified format preference for `pull()`.
#[derive(Debug, Clone, PartialEq)]
#[derive(Default)]
pub enum FormatPreference {
    /// Let the platform decide: MLX on Apple Silicon, GGUF elsewhere.
    #[default]
    Auto,
    /// Force GGUF regardless of platform.
    Gguf,
    /// Force MLX — error if not found.
    Mlx,
}


/// Local model store backed by `~/.spindll` (or a custom directory).
///
/// Manages a registry of downloaded/imported GGUF models and provides
/// name resolution so callers can refer to models by short names like
/// `"llama3.1:8b"` instead of full paths.
pub struct ModelStore {
    base_dir: PathBuf,
}

/// A user-supplied model name resolved against the registry. See
/// [`ModelStore::resolve`].
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// Canonical registry key — the name to load, unload, and report under.
    pub key: String,
    pub path: PathBuf,
    pub digest: String,
    pub format: registry::ModelFormat,
    #[cfg(feature = "vision")]
    pub mmproj_path: Option<PathBuf>,
}

impl ModelStore {
    /// Create a store rooted at the given directory, or `~/.spindll` if `None`.
    pub fn new(base_dir: Option<PathBuf>) -> Self {
        let base_dir = base_dir.unwrap_or_else(|| {
            import::home_dir()
                .expect("cannot determine home directory (HOME/USERPROFILE unset)")
                .join(".spindll")
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
        // Direct probe first when input is a full HF repo path -- lets
        // `--mlx other-org/foo` pull from other-org, not just mlx-community.
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
        // Stamp normalized alias ("llama3.1:8b" -> "llama3.1-8b") as base_model
        // so resolve_key step 5 matches. Otherwise alias unresolvable post-pull.
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
            source: registry::ModelSource::HfSourceDownloaded,
            mmproj_path: None,
        });
        reg.save(&self.registry_path())?;

        Ok(path)
    }

    /// Pull a GGUF model. `strict_gguf=true` rejects MLX-only repos so
    /// `--gguf` does not silently land MLX safetensors.
    fn pull_gguf(&self, model: &str, quant: Option<&str>, strict_gguf: bool) -> anyhow::Result<PathBuf> {
        let is_hf = model.contains('/');

        // --- Download & detect format ---
        let (path, size_bytes, key, digest, format, mmproj_path) = if is_hf {
            let dest_dir = self.model_dir(model);
            match download::download_hf_auto(model, quant, &dest_dir)? {
                download::HfDownload::Gguf { path, filename, size, digest, mmproj_path } => {
                    download::validate_gguf(&path)?;
                    let key = format!("{}/{}", model, filename);
                    (path, size, key, digest, registry::ModelFormat::Gguf, mmproj_path)
                }
                download::HfDownload::Mlx { dir, size, digest } => {
                    if strict_gguf {
                        anyhow::bail!(
                            "'{model}' contains MLX safetensors, not GGUF — drop --gguf or pass --mlx"
                        );
                    }
                    if !platform_prefers_mlx() {
                        anyhow::bail!(
                            "'{model}' contains only MLX safetensors, which this build cannot run — \
                             look for a GGUF version instead"
                        );
                    }
                    let key = model.to_string();
                    (dir, size, key, digest, registry::ModelFormat::Mlx, None)
                }
            }
        } else {
            let (name, _tag) = ollama_pull::parse_model_ref(model);
            let dest_dir = self.model_dir(&format!("ollama/{name}"));
            let (path, size, digest) = ollama_pull::pull_from_registry(model, &dest_dir)?;
            download::validate_gguf(&path)?;
            let filename = path.file_name().unwrap().to_string_lossy();
            let key = format!("ollama/{name}/{filename}");
            (path, size, key, digest, registry::ModelFormat::Gguf, None)
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
        let source = if is_hf {
            registry::ModelSource::HfSourceDownloaded
        } else {
            registry::ModelSource::OllamaSourceDownloaded
        };
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
            source,
            mmproj_path,
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
            "{:<model_w$} {:<5} {:>10}  {:<arch_w$}  DESCRIPTION",
            "MODEL", "FMT", "SIZE", "ARCH"
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
    ///   - Ollama name only:     `llama3.1`     → preferred `ollama/llama3.1/*.gguf`
    ///   - HuggingFace repo:     `TheBloke/Llama-3-8B-GGUF` → preferred variant
    ///
    /// When a name matches several variants of one repo, the quant the pull
    /// path would have downloaded wins. Destructive callers must not use this:
    /// see [`Self::resolve_key_unique`].
    pub fn resolve_key(&self, model: &str) -> anyhow::Result<String> {
        let candidates = self.resolve_key_candidates(model)?;
        if candidates.len() < 2 {
            return candidates
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("model '{model}' not found in registry"));
        }

        // Several variants of one repo match. Rank them the way `pull` ranks a
        // repo's files: `QUANT_PRIORITY` puts q4_k_m first and pushes
        // fp16/bf16/f32 last, because they are research-precision and 3-4x the
        // size. Picking the lowest-sorting key instead would land `run <repo>`
        // on `…-bf16.gguf` — the one variant `pull` would never choose. Key
        // order breaks ties, so the choice stays identical on every run.
        let reg = registry::Registry::load(&self.registry_path())?;
        let rank = |key: &String| {
            reg.models
                .get(key)
                .map_or(usize::MAX, |e| download::rank_quant(&e.filename))
        };
        candidates
            .into_iter()
            .min_by(|a, b| rank(a).cmp(&rank(b)).then_with(|| a.cmp(b)))
            .ok_or_else(|| anyhow::anyhow!("model '{model}' not found in registry"))
    }

    /// Resolve `model` to exactly one registry key, refusing ambiguous names.
    ///
    /// Destructive callers must use this instead of [`Self::resolve_key`]: the
    /// prefix rules can match several entries, and silently picking one deletes
    /// a model the user never named.
    pub fn resolve_key_unique(&self, model: &str) -> anyhow::Result<String> {
        let candidates = self.resolve_key_candidates(model)?;
        if candidates.len() > 1 {
            anyhow::bail!(
                "'{}' matches {} models — name one exactly:\n  {}",
                model,
                candidates.len(),
                candidates.join("\n  ")
            );
        }
        candidates
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("model '{model}' not found in registry"))
    }

    /// Every registry key `model` could name, taken from the first naming rule
    /// that matches anything. Sorted, so the list a caller sees is the same on
    /// every run — `models` is a `HashMap` and its order is not stable. Sort
    /// order is presentation only: [`Self::resolve_key`] picks by quant
    /// preference, not by taking the first element.
    ///
    /// A result longer than one element means the name is ambiguous.
    fn resolve_key_candidates(&self, model: &str) -> anyhow::Result<Vec<String>> {
        let reg = registry::Registry::load(&self.registry_path())?;

        // 1. Exact match
        if reg.models.contains_key(model) {
            return Ok(vec![model.to_string()]);
        }

        // 1b. Display-name form: `<repo> (<quant>)` as printed by `spindll list`
        //     for repos holding multiple GGUF variants — match on repo prefix,
        //     disambiguated by each entry's quant tag.
        if let Some((base, rest)) = model.rsplit_once(" (")
            && let Some(quant) = rest.strip_suffix(')') {
                let prefix = format!("{base}/");
                let matches = sorted_keys(reg.models.iter().filter_map(|(k, e)| {
                    (k.starts_with(&prefix)
                        && download::extract_quant(&e.filename) == Some(quant))
                        .then_some(k)
                }));
                if !matches.is_empty() {
                    return Ok(matches);
                }
            }

        // 2. Ollama name:tag  →  ollama/name/tag.gguf
        if let Some((name, tag)) = model.split_once(':') {
            let key = format!("ollama/{name}/{tag}.gguf");
            if reg.models.contains_key(&key) {
                return Ok(vec![key]);
            }
        }

        // 3. Bare name  →  ollama/name/*.gguf entries
        let prefix = format!("ollama/{model}/");
        let matches = sorted_keys(reg.models.keys().filter(|k| k.starts_with(&prefix)));
        if !matches.is_empty() {
            return Ok(matches);
        }

        // 4. HuggingFace repo prefix
        let hf_prefix = format!("{model}/");
        let matches = sorted_keys(reg.models.keys().filter(|k| k.starts_with(&hf_prefix)));
        if !matches.is_empty() {
            return Ok(matches);
        }

        // 5. Match by base_model (finds MLX entries for Ollama-style names)
        let normalized = model.replace([':', ' '], "-");
        let matches = sorted_keys(reg.models.iter().filter_map(|(k, e)| {
            (!e.base_model.is_empty() && e.base_model.eq_ignore_ascii_case(&normalized))
                .then_some(k)
        }));
        if !matches.is_empty() {
            return Ok(matches);
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

    /// Resolve the mmproj path for a model.
    ///
    /// Returns the stored `mmproj_path` from the registry entry if present,
    /// otherwise scans the model's directory for `*mmproj*.gguf` files.
    pub fn resolve_mmproj_path(&self, model: &str) -> anyhow::Result<Option<PathBuf>> {
        let key = self.resolve_key(model)?;
        let reg = registry::Registry::load(&self.registry_path())?;
        let entry = &reg.models[&key];

        // Return stored path if present and still exists on disk.
        if let Some(ref stored) = entry.mmproj_path
            && stored.exists() {
                return Ok(Some(stored.clone()));
            }

        // Auto-discover: scan the directory the model lives in.
        if let Some(search_dir) = projector_search_dir(entry)
            && search_dir.is_dir()
            && let Some(found) = discover_mmproj(&search_dir) {
                return Ok(Some(found));
            }

        Ok(None)
    }

    /// Everything a load needs about a model, from one registry read.
    ///
    /// Callers must load and unload under [`ResolvedModel::key`], never the
    /// string the user typed: the alias `llama3.1:8b` and the canonical id
    /// `/v1/models` advertises have to name the same resident model, or the
    /// same weights end up loaded twice under two keys.
    pub fn resolve(&self, model: &str) -> anyhow::Result<ResolvedModel> {
        let key = self.resolve_key(model)?;
        let reg = registry::Registry::load(&self.registry_path())?;
        let entry = &reg.models[&key];

        let path = std::fs::canonicalize(&entry.path)
            .map_err(|_| anyhow::anyhow!("model file missing: {}", entry.path.display()))?;

        Ok(ResolvedModel {
            digest: entry.digest.clone(),
            format: entry.format.clone(),
            #[cfg(feature = "vision")]
            mmproj_path: self.resolve_mmproj_path(&key).unwrap_or(None),
            key,
            path,
        })
    }

    /// Import all models from Ollama's local storage.
    pub fn import_from_ollama(&self) -> anyhow::Result<u32> {
        self.ensure_dirs()?;
        let Some(ollama) = import::ollama_dir() else {
            anyhow::bail!("cannot determine home directory (HOME/USERPROFILE unset)");
        };
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
                        source: registry::ModelSource::OllamaImported,
                        mmproj_path: None,
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

    /// Import all models from HuggingFace's local cache.
    pub fn import_from_hf(&self) -> anyhow::Result<u32> {
        self.ensure_dirs()?;
        let Some(hf_cache) = import::hf_cache_dir() else {
            anyhow::bail!("cannot determine home directory (HOME/USERPROFILE unset)");
        };
        let models = import::discover_hf_models(&hf_cache)?;

        if models.is_empty() {
            println!("no huggingface models found in cache");
            return Ok(0);
        }

        let mut reg = registry::Registry::load(&self.registry_path())?;
        let mut imported = 0u32;

        for (repo_id, snapshot_path) in models {
            // Find GGUF or MLX models in this snapshot
            let gguf_files: Vec<_> = std::fs::read_dir(&snapshot_path)?
                .filter_map(|e| e.ok())
                .filter(|e| is_primary_gguf(&e.path()))
                .collect();
            let mmproj_path = discover_mmproj(&snapshot_path);

            let is_mlx_dir = snapshot_path.join("model.safetensors").exists()
                && snapshot_path.join("config.json").exists();

            if !gguf_files.is_empty() {
                // GGUF model with potentially multiple quants
                for gguf_entry in gguf_files {
                    let gguf_path = gguf_entry.path();
                    let filename = gguf_entry.file_name().to_string_lossy().to_string();
                    let dest_dir = self.model_dir(&repo_id);
                    std::fs::create_dir_all(&dest_dir)?;
                    let dest = dest_dir.join(&filename);

                    link_or_copy_if_missing(&gguf_path, &dest)?;
                    let imported_mmproj_path =
                        materialize_mmproj(mmproj_path.as_deref(), &dest_dir)?;

                    let key = format!("{}/{}", repo_id, filename);
                    if let Some(entry) = reg.models.get_mut(&key) {
                        if entry.format == registry::ModelFormat::Gguf
                            && entry.source == registry::ModelSource::HfImported
                            && imported_mmproj_path.is_some()
                        {
                            entry.mmproj_path = imported_mmproj_path;
                        }
                    } else {
                        let (gguf_name, gguf_desc, gguf_arch, gguf_ctx) =
                            registry::read_gguf_metadata(&dest);
                        let base_model = derive_base_model(&gguf_name, &repo_id);
                        let size = std::fs::metadata(&gguf_path)?.len();

                        reg.add(
                            key,
                            registry::ModelEntry {
                                repo: repo_id.clone(),
                                filename: filename.clone(),
                                path: dest,
                                size_bytes: size,
                                downloaded_at: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap()
                                    .as_secs(),
                                digest: String::new(),
                                model_name: gguf_name,
                                description: gguf_desc,
                                architecture: gguf_arch,
                                context_length: gguf_ctx,
                                metadata_read: true,
                                format: registry::ModelFormat::Gguf,
                                base_model,
                                source: registry::ModelSource::HfImported,
                                mmproj_path: imported_mmproj_path,
                            },
                        );
                        println!("imported {}/{}", repo_id, filename);
                        imported += 1;
                    }
                }
            } else if is_mlx_dir {
                // MLX model
                let dest_dir = self.model_dir(&repo_id);
                if !dest_dir.exists() {
                    link_or_copy_path(&snapshot_path, &dest_dir)?;
                }

                let key = repo_id.clone();
                if !reg.models.contains_key(&key) {
                    let size = registry::dir_size(&snapshot_path)?;

                    reg.add(
                        key,
                        registry::ModelEntry {
                            repo: repo_id.clone(),
                            filename: String::new(),
                            path: dest_dir,
                            size_bytes: size,
                            downloaded_at: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
                            digest: String::new(),
                            model_name: repo_id.clone(),
                            description: String::new(),
                            architecture: String::new(),
                            context_length: 0,
                            metadata_read: false,
                            format: registry::ModelFormat::Mlx,
                            base_model: repo_id.clone(),
                            source: registry::ModelSource::HfImported,
                            mmproj_path: None,
                        },
                    );
                    println!("imported mlx model: {}", repo_id);
                    imported += 1;
                }
            }
        }

        reg.save(&self.registry_path())?;
        Ok(imported)
    }

    /// Import a model from an arbitrary path: a `.gguf` file or an MLX
    /// directory containing `config.json` plus one or more `*.safetensors`
    /// files (single-file or sharded).
    pub fn import_from_path(&self, path_str: &str) -> anyhow::Result<()> {
        self.ensure_dirs()?;
        let path = std::fs::canonicalize(path_str)
            .map_err(|_| anyhow::anyhow!("path does not exist: {}", path_str))?;

        let is_gguf = path.is_file()
            && path
                .extension()
                .map(|e| e.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false);
        let is_mlx_dir = path.is_dir()
            && path.join("config.json").exists()
            && dir_has_safetensors(&path);

        let format = if is_gguf {
            registry::read_gguf_metadata(&path); // validate it parses as GGUF
            registry::ModelFormat::Gguf
        } else if is_mlx_dir {
            registry::ModelFormat::Mlx
        } else {
            anyhow::bail!(
                "unsupported model at {}: expected a .gguf file or an MLX directory (config.json + *.safetensors)",
                path.display()
            );
        };

        let filename = path
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("path has no file name: {}", path.display()))?
            .to_string_lossy();
        let key = format!("manual/{}", filename);

        let dest_dir = self.model_dir("manual");
        std::fs::create_dir_all(&dest_dir)?;
        let dest = dest_dir.join(filename.as_ref());

        // Symlink the source into the manual namespace. On unix this covers
        // both a GGUF file and an MLX directory; windows falls back to a copy.
        if !dest.exists() {
            link_or_copy_path(&path, &dest)?;
        }

        let (model_name, description, architecture, context_length, size) = match format {
            registry::ModelFormat::Gguf => {
                let (name, desc, arch, ctx) = registry::read_gguf_metadata(&dest);
                (name, desc, arch, ctx, std::fs::metadata(&path)?.len())
            }
            registry::ModelFormat::Mlx => {
                let (arch, name) = download::read_mlx_metadata(&path);
                (name, String::new(), arch, 0, registry::dir_size(&path)?)
            }
        };
        let base_model = derive_base_model(&model_name, &filename);

        let mut reg = registry::Registry::load(&self.registry_path())?;
        if reg.models.contains_key(&key) {
            println!("model already imported");
            return Ok(());
        }
        reg.add(
            key,
            registry::ModelEntry {
                repo: "manual".to_string(),
                filename: filename.to_string(),
                path: dest,
                size_bytes: size,
                downloaded_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                digest: String::new(),
                model_name,
                description,
                architecture,
                context_length,
                metadata_read: true,
                format,
                base_model,
                source: registry::ModelSource::ManuallyImported,
                mmproj_path: None,
            },
        );
        reg.save(&self.registry_path())?;
        Ok(())
    }

    /// Remove a model from the registry with source-aware cleanup.
    ///
    /// - OllamaSourceDownloaded / HfSourceDownloaded: auto-delete (Spindll owns the file)
    /// - OllamaImported / HfImported / ManuallyImported: prompt for confirmation (external ownership)
    /// - With purge=true: skip prompts, delete symlinks only for external sources
    pub fn remove(&self, model: &str, purge: bool) -> anyhow::Result<()> {
        // Accept every name form `resolve_key` understands — including the
        // `<repo> (<quant>)` display form `spindll list` prints, which users
        // paste straight back into `spindll rm`.
        // `resolve_key_unique`, not `resolve_key`: a name that prefix-matches
        // several variants must stop the deletion, not pick one of them.
        let key = self.resolve_key_unique(model)?;
        let mut reg = registry::Registry::load(&self.registry_path())?;
        let entry = reg.models.remove(&key)
            .ok_or_else(|| anyhow::anyhow!("model '{}' not found", key))?;

        let should_delete = match &entry.source {
            registry::ModelSource::OllamaSourceDownloaded
            | registry::ModelSource::HfSourceDownloaded => true,
            registry::ModelSource::OllamaImported
            | registry::ModelSource::HfImported
            | registry::ModelSource::ManuallyImported => {
                if purge {
                    true
                } else {
                    eprintln!(
                        "warning: '{}' is managed externally ({:?})",
                        key, entry.source
                    );
                    eprint!("delete symlink? (y/N) ");
                    use std::io::Write;
                    std::io::stdout().flush()?;

                    let mut response = String::new();
                    std::io::stdin().read_line(&mut response)?;
                    response.trim().eq_ignore_ascii_case("y")
                        || response.trim().eq_ignore_ascii_case("yes")
                }
            }
        };

        if should_delete {
            // symlink_metadata: a dangling symlink reports !exists() but still
            // occupies the path — remove it rather than leak it.
            if entry.path.symlink_metadata().is_ok() {
                // MLX = dir, GGUF = file.
                match entry.format {
                    registry::ModelFormat::Mlx => std::fs::remove_dir_all(&entry.path)?,
                    registry::ModelFormat::Gguf => std::fs::remove_file(&entry.path)?,
                }
            }
            let models_dir = self.models_dir();
            // A projector materialized into the store belongs to this model;
            // one living outside (imported in place) is not ours to delete.
            //
            // It is also shared: `download.rs` writes one mmproj per repo, so
            // every variant of that repo can use it — the ones recording the
            // path, and the ones recording `None` that let
            // `resolve_mmproj_path` discover it by scanning their directory.
            // A recorded path is therefore not the only kind of reference: any
            // surviving entry that searches this projector's directory is one
            // too. Deleting it under either strips vision from a model the user
            // kept, with nothing on disk left for the fallback to find. `reg`
            // no longer holds this entry, so any hit is a keeper.
            if let Some(mmproj) = &entry.mmproj_path
                && mmproj.starts_with(&models_dir)
                && !reg.models.values().any(|e| {
                    e.mmproj_path.as_deref() == Some(mmproj.as_path())
                        || projector_search_dir(e).as_deref() == mmproj.parent()
                })
                && mmproj.symlink_metadata().is_ok()
                && let Err(err) = std::fs::remove_file(mmproj)
            {
                // Not `?`: the weights are already gone and `reg.save` is still
                // ahead of us, so returning here would leave the registry
                // listing a model with no file behind it. Leak the projector
                // and keep the two consistent — the dir-prune loop below takes
                // the same way out.
                eprintln!(
                    "warning: could not delete projector {}: {err}",
                    mmproj.display()
                );
            }
            // Prune directories the deletion emptied (repo dir, then org dir).
            // remove_dir refuses non-empty dirs, so this stops at the first
            // level still holding other variants or an unrelated model.
            let mut dir = entry.path.parent();
            while let Some(d) = dir {
                if d == models_dir.as_path() || !d.starts_with(&models_dir) {
                    break;
                }
                if std::fs::remove_dir(d).is_err() {
                    break;
                }
                dir = d.parent();
            }
        } else {
            // User said no, put the entry back in the registry
            reg.models.insert(key.clone(), entry);
        }

        reg.save(&self.registry_path())?;
        if should_delete {
            println!("deleted {}", key);
        } else {
            println!("kept {}", key);
        }
        Ok(())
    }
}

/// The directory an entry's projector is looked for in: the model directory
/// itself for MLX, the containing directory for a GGUF file.
///
/// [`ModelStore::resolve_mmproj_path`] scans it when an entry records no
/// `mmproj_path`, so `remove` has to read it as a live reference to whatever
/// projector sits there. Both callers go through this function; if they drift
/// apart, `rm` starts deleting projectors that models still use.
fn projector_search_dir(entry: &registry::ModelEntry) -> Option<PathBuf> {
    match entry.format {
        registry::ModelFormat::Mlx => Some(entry.path.clone()),
        registry::ModelFormat::Gguf => entry.path.parent().map(|p| p.to_path_buf()),
    }
}

/// Collect registry keys into a stable, sorted list.
///
/// `Registry::models` is a `HashMap`, so iteration order varies per process.
/// Anything user-visible — a chosen key, an "ambiguous name" candidate list —
/// has to be ordered here or it changes run to run.
fn sorted_keys<'a>(keys: impl Iterator<Item = &'a String>) -> Vec<String> {
    let mut sorted: Vec<String> = keys.cloned().collect();
    sorted.sort();
    sorted
}

/// True when `dir` holds at least one `*.safetensors` file — single-file
/// (`model.safetensors`) or sharded (`model-00001-of-00002.safetensors` plus
/// an index json), both of which the MLX serve path already loads.
fn dir_has_safetensors(dir: &std::path::Path) -> bool {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries.filter_map(|e| e.ok()).any(|e| {
                e.path()
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
            })
        })
        .unwrap_or(false)
}

/// Convert a registry key to a friendly display name.
///
/// - `ollama/nemotron-3-nano/4b.gguf` → `nemotron-3-nano:4b`
/// - `TheBloke/Llama-3-8B-GGUF/model.gguf` → `TheBloke/Llama-3-8B-GGUF:model`
///
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
            if let [provider, name, file] = parts.as_slice()
                && *provider == "ollama" {
                    let tag = file.strip_suffix(".gguf").unwrap_or(file);
                    return format!("{name}:{tag}");
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

/// Scan a directory for a multimodal projector GGUF file (`*mmproj*.gguf`).
///
/// Returns the path to the first match, or `None` if no mmproj file is found.
pub fn discover_mmproj(dir: &std::path::Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_lower = name.to_string_lossy().to_lowercase();
        if name_lower.contains("mmproj") && name_lower.ends_with(".gguf") {
            return Some(entry.path());
        }
    }
    None
}

fn is_primary_gguf(path: &std::path::Path) -> bool {
    path.extension().is_some_and(|ext| ext == "gguf")
        && !path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().to_lowercase().contains("mmproj"))
}

fn materialize_mmproj(
    src: Option<&std::path::Path>,
    dest_dir: &std::path::Path,
) -> anyhow::Result<Option<PathBuf>> {
    let Some(src) = src else {
        return Ok(None);
    };
    let file_name = src
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("mmproj path has no file name: {}", src.display()))?;
    let dest = dest_dir.join(file_name);
    link_or_copy_if_missing(src, &dest)?;
    Ok(Some(dest))
}

fn link_or_copy_if_missing(
    src: &std::path::Path,
    dest: &std::path::Path,
) -> anyhow::Result<()> {
    let should_link = match std::fs::symlink_metadata(dest) {
        Ok(meta) if meta.file_type().is_symlink() && !dest.exists() => {
            std::fs::remove_file(dest)?;
            true
        }
        Ok(_) => false,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => true,
        Err(err) => return Err(err.into()),
    };
    if should_link {
        link_or_copy_path(src, dest)?;
    }
    Ok(())
}

fn link_or_copy_path(src: &std::path::Path, dest: &std::path::Path) -> anyhow::Result<()> {
    // `models/<owner>` may not exist yet — symlink, hard_link and copy all fail
    // with ENOENT when the parent is missing.
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(src, dest)?;
    }
    #[cfg(windows)]
    {
        if src.is_dir() {
            copy_dir_all(src, dest)?;
        } else if std::fs::hard_link(src, dest).is_err() {
            std::fs::copy(src, dest)?;
        }
    }
    Ok(())
}

#[cfg(windows)]
fn copy_dir_all(src: &std::path::Path, dest: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dest)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let target = dest.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_all(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
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
    use crate::model_store::registry::{ModelEntry, ModelFormat, ModelSource, Registry};

    fn write_entry(path: &std::path::Path, key: &str, entry: ModelEntry) {
        let mut reg = Registry::load(path).unwrap();
        reg.add(key.to_string(), entry);
        reg.save(path).unwrap();
    }

    #[test]
    fn primary_gguf_filter_skips_mmproj() {
        assert!(is_primary_gguf(std::path::Path::new("model-q4_k_m.gguf")));
        assert!(!is_primary_gguf(std::path::Path::new("mmproj-model-f16.gguf")));
    }

    #[test]
    fn materialize_mmproj_stores_projector_next_to_model() {
        let dir = tempfile::tempdir().unwrap();
        let hf_cache = dir.path().join("hf-cache");
        let store_dir = dir.path().join("models/repo");
        std::fs::create_dir_all(&hf_cache).unwrap();
        std::fs::create_dir_all(&store_dir).unwrap();
        let src = hf_cache.join("mmproj-model-f16.gguf");
        std::fs::write(&src, b"fake").unwrap();

        let stored = materialize_mmproj(Some(&src), &store_dir)
            .unwrap()
            .unwrap();

        assert_eq!(stored, store_dir.join("mmproj-model-f16.gguf"));
        assert!(stored.exists());
    }

    #[cfg(unix)]
    #[test]
    fn materialize_mmproj_replaces_dangling_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let hf_cache = dir.path().join("hf-cache");
        let store_dir = dir.path().join("models/repo");
        std::fs::create_dir_all(&hf_cache).unwrap();
        std::fs::create_dir_all(&store_dir).unwrap();
        let src = hf_cache.join("mmproj-model-f16.gguf");
        std::fs::write(&src, b"fake").unwrap();
        let dest = store_dir.join("mmproj-model-f16.gguf");
        std::os::unix::fs::symlink(dir.path().join("deleted"), &dest).unwrap();

        let stored = materialize_mmproj(Some(&src), &store_dir)
            .unwrap()
            .unwrap();

        assert_eq!(stored, dest);
        assert_eq!(std::fs::read(stored).unwrap(), b"fake");
    }

    #[cfg(unix)]
    #[test]
    fn link_or_copy_if_missing_replaces_dangling_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("model.gguf");
        let dest = dir.path().join("store/model.gguf");
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        std::fs::write(&src, b"fake").unwrap();
        std::os::unix::fs::symlink(dir.path().join("deleted"), &dest).unwrap();

        link_or_copy_if_missing(&src, &dest).unwrap();

        assert_eq!(std::fs::read(dest).unwrap(), b"fake");
    }

    #[test]
    fn link_or_copy_path_creates_missing_parent() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("snapshot");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("model.safetensors"), b"fake").unwrap();

        // models/<owner>/<repo> — <owner> does not exist yet, as on a fresh store.
        let dest = dir.path().join("models/mlx-community/some-model-4bit");

        link_or_copy_path(&src, &dest).unwrap();

        assert_eq!(
            std::fs::read(dest.join("model.safetensors")).unwrap(),
            b"fake"
        );
    }

    #[cfg(windows)]
    #[test]
    fn link_or_copy_path_copies_directory_on_windows() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        let nested = src.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("model.safetensors"), b"fake").unwrap();
        let dest = dir.path().join("dest");

        link_or_copy_path(&src, &dest).unwrap();
        assert_eq!(std::fs::read(dest.join("nested/model.safetensors")).unwrap(), b"fake");
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
            source: registry::ModelSource::HfSourceDownloaded,
            mmproj_path: None,
        }
    }

    /// Regression: alias must resolve when registry key is mlx-community/...
    /// Pull stamps normalized alias as base_model; this exercises the read side.
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

    /// An alias must resolve to the canonical key, so callers register the
    /// model under the same name `/v1/models` advertises instead of loading a
    /// second copy under the alias.
    #[test]
    fn resolve_returns_the_canonical_key_for_an_alias() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        std::fs::create_dir_all(store.models_dir()).unwrap();
        // resolve() canonicalizes, so the entry needs a real path on disk.
        let model_dir = store.models_dir().join("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit");
        std::fs::create_dir_all(&model_dir).unwrap();
        let mut entry =
            mlx_entry("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "llama3.1-8b");
        entry.path = model_dir;
        write_entry(
            &store.registry_path(),
            "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            entry,
        );

        let resolved = store.resolve("llama3.1:8b").unwrap();
        assert_eq!(resolved.key, "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit");
        assert!(resolved.path.exists());
        // Resolving the canonical key is a fixed point.
        assert_eq!(store.resolve(&resolved.key).unwrap().key, resolved.key);
    }

    #[test]
    fn resolve_reports_an_unknown_model() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        std::fs::create_dir_all(store.models_dir()).unwrap();

        let err = store.resolve("nope:7b").unwrap_err().to_string();
        assert!(err.contains("not found in registry"), "got: {err}");
    }

    /// Regression: MLX entry uses remove_dir_all (remove_file errors on dirs).
    #[test]
    fn remove_mlx_handles_directory() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        std::fs::create_dir_all(store.models_dir()).unwrap();

        // Real MLX layout: config + safetensors shard.
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

        store.remove("mlx-community/test-4bit", false).expect("remove should succeed for MLX dir");
        assert!(!model_dir.exists(), "MLX dir should be deleted");
        let reg = Registry::load(&store.registry_path()).unwrap();
        assert!(!reg.models.contains_key("mlx-community/test-4bit"));
    }

    #[test]
    fn resolve_key_accepts_list_display_name_with_quant() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        std::fs::create_dir_all(store.models_dir()).unwrap();
        write_entry(
            &store.registry_path(),
            "cjpais/llava-1.6-mistral-7b-gguf/llava-q8_0.gguf",
            gguf_entry("cjpais/llava-1.6-mistral-7b-gguf", "llava-q8_0.gguf"),
        );
        write_entry(
            &store.registry_path(),
            "cjpais/llava-1.6-mistral-7b-gguf/llava-q4_k_m.gguf",
            gguf_entry("cjpais/llava-1.6-mistral-7b-gguf", "llava-q4_k_m.gguf"),
        );

        let q8 = store.resolve_key("cjpais/llava-1.6-mistral-7b-gguf (q8_0)").unwrap();
        assert_eq!(q8, "cjpais/llava-1.6-mistral-7b-gguf/llava-q8_0.gguf");
        let q4 = store.resolve_key("cjpais/llava-1.6-mistral-7b-gguf (q4_k_m)").unwrap();
        assert_eq!(q4, "cjpais/llava-1.6-mistral-7b-gguf/llava-q4_k_m.gguf");
    }

    /// Regression: `spindll rm` on the name `spindll list` prints (quant
    /// suffix included) must resolve, delete the artifact AND its
    /// store-materialized mmproj, and prune the emptied repo/org dirs.
    #[test]
    fn remove_display_name_deletes_artifact_mmproj_and_empty_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        std::fs::create_dir_all(store.models_dir()).unwrap();

        let repo_dir = store.models_dir().join("cjpais/llava-1.6-mistral-7b-gguf");
        std::fs::create_dir_all(&repo_dir).unwrap();
        let model_file = repo_dir.join("llava-q8_0.gguf");
        let mmproj_file = repo_dir.join("mmproj-model-f16.gguf");
        std::fs::write(&model_file, b"fake-gguf").unwrap();
        std::fs::write(&mmproj_file, b"fake-mmproj").unwrap();

        let mut entry = gguf_entry("cjpais/llava-1.6-mistral-7b-gguf", "llava-q8_0.gguf");
        entry.path = model_file.clone();
        entry.mmproj_path = Some(mmproj_file.clone());
        write_entry(
            &store.registry_path(),
            "cjpais/llava-1.6-mistral-7b-gguf/llava-q8_0.gguf",
            entry,
        );

        store.remove("cjpais/llava-1.6-mistral-7b-gguf (q8_0)", false)
            .expect("display-name remove should succeed");

        assert!(!model_file.exists(), "artifact should be deleted");
        assert!(!mmproj_file.exists(), "store-materialized mmproj should be deleted");
        assert!(!repo_dir.exists(), "emptied repo dir should be pruned");
        assert!(!store.models_dir().join("cjpais").exists(), "emptied org dir should be pruned");
        assert!(store.models_dir().exists(), "models root must survive");
        let reg = Registry::load(&store.registry_path()).unwrap();
        assert!(reg.models.is_empty());
    }

    /// A second variant in the same repo keeps the shared dirs alive.
    #[test]
    fn remove_keeps_repo_dir_holding_other_variants() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        std::fs::create_dir_all(store.models_dir()).unwrap();

        let repo_dir = store.models_dir().join("TheBloke/Llama-GGUF");
        std::fs::create_dir_all(&repo_dir).unwrap();
        let q8 = repo_dir.join("llama-q8_0.gguf");
        let q4 = repo_dir.join("llama-q4_k_m.gguf");
        // `download.rs` materializes the projector into the repo dir, not a
        // per-variant dir, so every quant of the repo shares this one file.
        let mmproj = repo_dir.join("mmproj-llama-f16.gguf");
        std::fs::write(&q8, b"a").unwrap();
        std::fs::write(&q4, b"b").unwrap();
        std::fs::write(&mmproj, b"p").unwrap();

        let mut e8 = gguf_entry("TheBloke/Llama-GGUF", "llama-q8_0.gguf");
        e8.path = q8.clone();
        e8.mmproj_path = Some(mmproj.clone());
        let mut e4 = gguf_entry("TheBloke/Llama-GGUF", "llama-q4_k_m.gguf");
        e4.path = q4.clone();
        e4.mmproj_path = Some(mmproj.clone());
        write_entry(&store.registry_path(), "TheBloke/Llama-GGUF/llama-q8_0.gguf", e8);
        write_entry(&store.registry_path(), "TheBloke/Llama-GGUF/llama-q4_k_m.gguf", e4);

        store.remove("TheBloke/Llama-GGUF (q8_0)", false).unwrap();

        assert!(!q8.exists());
        assert!(q4.exists(), "sibling variant must survive");
        assert!(
            mmproj.exists(),
            "projector the surviving variant still points at must survive"
        );
        assert!(repo_dir.exists(), "repo dir still holding a variant must survive");
    }

    #[test]
    fn remove_deletes_mmproj_once_no_variant_references_it() {
        // The other half of the shared-projector rule: the last entry holding a
        // reference must still clean it up, or the store leaks the file forever.
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        std::fs::create_dir_all(store.models_dir()).unwrap();

        let repo_dir = store.models_dir().join("TheBloke/Llama-GGUF");
        std::fs::create_dir_all(&repo_dir).unwrap();
        let q8 = repo_dir.join("llama-q8_0.gguf");
        let q4 = repo_dir.join("llama-q4_k_m.gguf");
        let mmproj = repo_dir.join("mmproj-llama-f16.gguf");
        std::fs::write(&q8, b"a").unwrap();
        std::fs::write(&q4, b"b").unwrap();
        std::fs::write(&mmproj, b"p").unwrap();

        let mut e8 = gguf_entry("TheBloke/Llama-GGUF", "llama-q8_0.gguf");
        e8.path = q8.clone();
        e8.mmproj_path = Some(mmproj.clone());
        let mut e4 = gguf_entry("TheBloke/Llama-GGUF", "llama-q4_k_m.gguf");
        e4.path = q4.clone();
        e4.mmproj_path = Some(mmproj.clone());
        write_entry(&store.registry_path(), "TheBloke/Llama-GGUF/llama-q8_0.gguf", e8);
        write_entry(&store.registry_path(), "TheBloke/Llama-GGUF/llama-q4_k_m.gguf", e4);

        store.remove("TheBloke/Llama-GGUF (q8_0)", false).unwrap();
        // Asserting only the end state passes with no guard at all: the
        // projector has to be observed surviving the first removal.
        assert!(
            mmproj.exists(),
            "projector must survive while q4_k_m still references it"
        );

        store.remove("TheBloke/Llama-GGUF (q4_k_m)", false).unwrap();

        assert!(!mmproj.exists(), "last reference gone — projector must be deleted");
        assert!(!repo_dir.exists(), "emptied repo dir must be pruned");
    }

    /// `resolve_mmproj_path` falls back to scanning the model's directory, so a
    /// variant recording no `mmproj_path` still uses the repo's shared
    /// projector. Deleting the one entry that records the path must not strip
    /// vision from that sibling.
    #[test]
    fn remove_keeps_projector_a_sibling_discovers_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        std::fs::create_dir_all(store.models_dir()).unwrap();

        let repo_dir = store.models_dir().join("TheBloke/Llama-GGUF");
        std::fs::create_dir_all(&repo_dir).unwrap();
        let q8 = repo_dir.join("llama-q8_0.gguf");
        let q4 = repo_dir.join("llama-q4_k_m.gguf");
        let mmproj = repo_dir.join("mmproj-llama-f16.gguf");
        std::fs::write(&q8, b"a").unwrap();
        std::fs::write(&q4, b"b").unwrap();
        std::fs::write(&mmproj, b"p").unwrap();

        let mut e8 = gguf_entry("TheBloke/Llama-GGUF", "llama-q8_0.gguf");
        e8.path = q8.clone();
        e8.mmproj_path = Some(mmproj.clone());
        // `import`, `add_manual` and the GGUF pull path all record `None`.
        let mut e4 = gguf_entry("TheBloke/Llama-GGUF", "llama-q4_k_m.gguf");
        e4.path = q4.clone();
        e4.mmproj_path = None;
        write_entry(&store.registry_path(), "TheBloke/Llama-GGUF/llama-q8_0.gguf", e8);
        write_entry(&store.registry_path(), "TheBloke/Llama-GGUF/llama-q4_k_m.gguf", e4);

        store.remove("TheBloke/Llama-GGUF (q8_0)", false).unwrap();

        assert!(
            mmproj.exists(),
            "projector the surviving variant discovers on disk must survive"
        );
        assert_eq!(
            store
                .resolve_mmproj_path("TheBloke/Llama-GGUF/llama-q4_k_m.gguf")
                .unwrap(),
            Some(mmproj),
            "surviving variant must keep vision"
        );
    }

    /// A projector that cannot be deleted must not abort the removal before the
    /// registry is written: the weights are already gone, so an early return
    /// leaves the registry advertising a model with no file behind it.
    #[cfg(unix)]
    #[test]
    fn remove_saves_registry_when_projector_delete_fails() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        std::fs::create_dir_all(store.models_dir()).unwrap();

        let repo_dir = store.models_dir().join("TheBloke/Llama-GGUF");
        std::fs::create_dir_all(&repo_dir).unwrap();
        let q8 = repo_dir.join("llama-q8_0.gguf");
        std::fs::write(&q8, b"a").unwrap();

        // Unlink needs write permission on the parent directory, so a projector
        // in a read-only directory fails to delete while the weight file, in a
        // writable one, goes away.
        let locked = store.models_dir().join("locked");
        std::fs::create_dir_all(&locked).unwrap();
        let mmproj = locked.join("mmproj-llama-f16.gguf");
        std::fs::write(&mmproj, b"p").unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o555)).unwrap();

        // root ignores the mode bits — skip rather than assert a false pass.
        if std::fs::write(locked.join("probe"), b"x").is_ok() {
            let _ = std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755));
            return;
        }

        let key = "TheBloke/Llama-GGUF/llama-q8_0.gguf";
        let mut e8 = gguf_entry("TheBloke/Llama-GGUF", "llama-q8_0.gguf");
        e8.path = q8.clone();
        e8.mmproj_path = Some(mmproj.clone());
        write_entry(&store.registry_path(), key, e8);

        let result = store.remove("TheBloke/Llama-GGUF (q8_0)", false);
        let _ = std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755));

        assert!(
            result.is_ok(),
            "an undeletable projector must not fail the removal: {result:?}"
        );
        assert!(!q8.exists(), "weights must be deleted");
        let reg = Registry::load(&store.registry_path()).unwrap();
        assert!(
            !reg.models.contains_key(key),
            "registry must not list a model whose weights are gone"
        );
    }

    /// `spindll run <repo>` must land on the variant the pull path would have
    /// downloaded, not on whichever filename sorts first.
    #[test]
    fn resolve_prefers_quant_priority_over_lexicographic_order() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        std::fs::create_dir_all(store.models_dir()).unwrap();

        let repo_dir = store.models_dir().join("TheBloke/Llama-GGUF");
        std::fs::create_dir_all(&repo_dir).unwrap();
        // "llama-bf16.gguf" sorts before "llama-q4_k_m.gguf", but QUANT_PRIORITY
        // deprioritizes research-precision weights at 3-4x the size.
        let bf16 = repo_dir.join("llama-bf16.gguf");
        let q4 = repo_dir.join("llama-q4_k_m.gguf");
        std::fs::write(&bf16, b"a").unwrap();
        std::fs::write(&q4, b"b").unwrap();

        let mut ebf = gguf_entry("TheBloke/Llama-GGUF", "llama-bf16.gguf");
        ebf.path = bf16;
        let mut e4 = gguf_entry("TheBloke/Llama-GGUF", "llama-q4_k_m.gguf");
        e4.path = q4.clone();
        write_entry(&store.registry_path(), "TheBloke/Llama-GGUF/llama-bf16.gguf", ebf);
        write_entry(&store.registry_path(), "TheBloke/Llama-GGUF/llama-q4_k_m.gguf", e4);

        assert_eq!(
            store.resolve_key("TheBloke/Llama-GGUF").unwrap(),
            "TheBloke/Llama-GGUF/llama-q4_k_m.gguf",
            "bare repo name must resolve to the preferred quant"
        );

        // The whole load path, not just the key.
        let resolved = store.resolve("TheBloke/Llama-GGUF").unwrap();
        assert_eq!(resolved.key, "TheBloke/Llama-GGUF/llama-q4_k_m.gguf");
        assert_eq!(resolved.path, std::fs::canonicalize(&q4).unwrap());
    }

    #[test]
    fn remove_refuses_ambiguous_repo_prefix() {
        // `TheBloke/Llama-GGUF` prefix-matches two registry keys. Picking one by
        // HashMap iteration order would delete a model the user never named.
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        std::fs::create_dir_all(store.models_dir()).unwrap();

        let repo_dir = store.models_dir().join("TheBloke/Llama-GGUF");
        std::fs::create_dir_all(&repo_dir).unwrap();
        let q8 = repo_dir.join("llama-q8_0.gguf");
        let q4 = repo_dir.join("llama-q4_k_m.gguf");
        std::fs::write(&q8, b"a").unwrap();
        std::fs::write(&q4, b"b").unwrap();

        let mut e8 = gguf_entry("TheBloke/Llama-GGUF", "llama-q8_0.gguf");
        e8.path = q8.clone();
        let mut e4 = gguf_entry("TheBloke/Llama-GGUF", "llama-q4_k_m.gguf");
        e4.path = q4.clone();
        write_entry(&store.registry_path(), "TheBloke/Llama-GGUF/llama-q8_0.gguf", e8);
        write_entry(&store.registry_path(), "TheBloke/Llama-GGUF/llama-q4_k_m.gguf", e4);

        let err = store
            .remove("TheBloke/Llama-GGUF", false)
            .expect_err("ambiguous name must not delete anything");
        let msg = err.to_string();
        assert!(msg.contains("llama-q8_0.gguf"), "error must list candidates: {msg}");
        assert!(msg.contains("llama-q4_k_m.gguf"), "error must list candidates: {msg}");

        assert!(q8.exists(), "no model may be deleted for an ambiguous name");
        assert!(q4.exists(), "no model may be deleted for an ambiguous name");
    }

    #[test]
    fn resolve_key_returns_err_when_alias_unknown() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        std::fs::create_dir_all(store.models_dir()).unwrap();
        write_entry(
            &store.registry_path(),
            "TheBloke/Llama-3-8B-GGUF/llama-3-8b-q4_k_m.gguf",
            gguf_entry("TheBloke/Llama-3-8B-GGUF", "llama-3-8b-q4_k_m.gguf"),
        );

        let result = store.resolve_key("llama3.1:8b");
        assert!(result.is_err(), "unrelated GGUF entry must not match an Ollama alias");
    }

    #[test]
    fn resolve_key_prefers_exact_match_over_alias() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        std::fs::create_dir_all(store.models_dir()).unwrap();

        let gguf_key = "meta-llama/Meta-Llama-3.1-8B-Instruct/llama-q4_k_m.gguf";
        write_entry(&store.registry_path(), gguf_key, gguf_entry("meta-llama/Meta-Llama-3.1-8B-Instruct", "llama-q4_k_m.gguf"));

        let mut mlx = mlx_entry("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "llama3.1-8b");
        mlx.base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct/llama-q4_k_m.gguf".to_string();
        write_entry(&store.registry_path(), "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", mlx);

        let resolved = store.resolve_key(gguf_key).unwrap();
        assert_eq!(resolved, gguf_key, "exact key match (step 1) must beat base_model alias (step 5)");
    }

    // --- Item 14: display_name() — values surfaced in gRPC ListResponse ---

    fn gguf_entry(repo: &str, filename: &str) -> ModelEntry {
        ModelEntry {
            repo: repo.to_string(),
            filename: filename.to_string(),
            path: std::path::PathBuf::from("/tmp/nonexistent"),
            size_bytes: 0,
            downloaded_at: 0,
            digest: String::new(),
            model_name: String::new(),
            description: String::new(),
            architecture: String::new(),
            context_length: 0,
            metadata_read: true,
            format: ModelFormat::Gguf,
            base_model: String::new(),
            source: registry::ModelSource::HfSourceDownloaded,
            mmproj_path: None,
        }
    }

    #[test]
    fn display_name_hf_gguf_with_detectable_quant() {
        // Closes #12: picker UIs must see "(q4_k_m)" suffix to disambiguate.
        let entry = gguf_entry("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-q4_k_m.gguf");
        let name = display_name("Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-q4_k_m.gguf", &entry);
        assert_eq!(name, "Qwen/Qwen2.5-3B-Instruct-GGUF (q4_k_m)");
    }

    #[test]
    fn display_name_hf_gguf_fp16_no_quant_tag() {
        // fp16 files have no quant tag — display just the repo name.
        let entry = gguf_entry("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-f16.gguf");
        let name = display_name("Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-f16.gguf", &entry);
        assert_eq!(name, "Qwen/Qwen2.5-3B-Instruct-GGUF");
    }

    #[test]
    fn display_name_two_quants_same_repo_are_distinct() {
        // Core disambiguation requirement from #12.
        let e_q4 = gguf_entry("TheBloke/Llama-GGUF", "llama-q4_k_m.gguf");
        let e_fp16 = gguf_entry("TheBloke/Llama-GGUF", "llama-fp16.gguf");
        let n_q4 = display_name("TheBloke/Llama-GGUF/llama-q4_k_m.gguf", &e_q4);
        let n_fp16 = display_name("TheBloke/Llama-GGUF/llama-fp16.gguf", &e_fp16);
        assert_ne!(n_q4, n_fp16, "two variants of the same repo must produce distinct display names");
        assert!(n_q4.contains("q4_k_m"));
    }

    #[test]
    fn display_name_ollama_entry_uses_name_tag_form() {
        // Ollama keys look like "ollama/<name>/<tag>.gguf".
        let entry = gguf_entry("ollama/llama3.1", "8b.gguf");
        let name = display_name("ollama/llama3.1/8b.gguf", &entry);
        assert_eq!(name, "llama3.1:8b");
    }

    #[test]
    fn display_name_mlx_returns_repo() {
        // MLX repos already encode quant in their name (e.g. "-4bit").
        let entry = mlx_entry("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "llama3.1-8b");
        let name = display_name("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", &entry);
        assert_eq!(name, "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit");
    }

    #[test]
    fn display_name_mlx_empty_repo_falls_back_to_key() {
        let mut entry = mlx_entry("", "test");
        entry.format = ModelFormat::Mlx;
        let name = display_name("some-registry-key", &entry);
        assert_eq!(name, "some-registry-key");
    }

    // Item 14: prefer_format field in ListResponse.
    #[test]
    fn platform_prefers_mlx_is_false_on_non_apple_silicon() {
        // On Linux CI and Windows, this must be false so GGUF is the default.
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "mlx")))]
        assert!(!platform_prefers_mlx());
    }

    #[test]
    fn import_from_path_registers_gguf_as_manual() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        let src = dir.path().join("tiny.gguf");
        std::fs::write(&src, b"placeholder gguf bytes").unwrap();

        store
            .import_from_path(src.to_str().unwrap())
            .expect("gguf file import should succeed");

        let reg = Registry::load(&store.registry_path()).unwrap();
        let entry = reg.models.get("manual/tiny.gguf").expect("entry registered");
        assert_eq!(entry.source, ModelSource::ManuallyImported);
    }

    // MLX import symlinks a directory; windows hard_link/copy cannot, so these
    // exercise the unix path (same limitation as import_from_hf).
    #[cfg(unix)]
    #[test]
    fn import_from_path_accepts_mlx_directory() {
        // Regression: an MLX directory must import, not bail "not a file".
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        let src = dir.path().join("my-mlx-model");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("config.json"), b"{}").unwrap();
        std::fs::write(src.join("model.safetensors"), vec![0u8; 2048]).unwrap();

        store
            .import_from_path(src.to_str().unwrap())
            .expect("mlx directory import should succeed");

        let reg = Registry::load(&store.registry_path()).unwrap();
        let entry = reg.models.get("manual/my-mlx-model").expect("mlx entry registered");
        assert_eq!(entry.format, ModelFormat::Mlx);
    }

    #[cfg(unix)]
    #[test]
    fn import_from_path_mlx_size_sums_directory_files() {
        // size_bytes must reflect the model files, not the directory inode.
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        let src = dir.path().join("sized-mlx");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("config.json"), vec![0u8; 100]).unwrap();
        std::fs::write(src.join("model.safetensors"), vec![0u8; 4096]).unwrap();

        store.import_from_path(src.to_str().unwrap()).unwrap();

        let reg = Registry::load(&store.registry_path()).unwrap();
        let entry = reg.models.get("manual/sized-mlx").unwrap();
        assert_eq!(entry.size_bytes, 4196, "size must sum directory files, not stat the dir");
    }

    #[test]
    fn import_from_path_rejects_unsupported_file() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        let src = dir.path().join("notes.txt");
        std::fs::write(&src, b"not a model").unwrap();

        assert!(store.import_from_path(src.to_str().unwrap()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn import_from_path_accepts_sharded_mlx_directory() {
        // Issue #75 friction: sharded models serve fine, but import rejected
        // them for lacking a single-file model.safetensors.
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        let src = dir.path().join("sharded-mlx");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("config.json"), b"{}").unwrap();
        std::fs::write(src.join("model-00001-of-00002.safetensors"), vec![0u8; 1024]).unwrap();
        std::fs::write(src.join("model-00002-of-00002.safetensors"), vec![0u8; 1024]).unwrap();
        std::fs::write(src.join("model.safetensors.index.json"), b"{}").unwrap();

        store
            .import_from_path(src.to_str().unwrap())
            .expect("sharded mlx directory import should succeed");

        let reg = Registry::load(&store.registry_path()).unwrap();
        let entry = reg.models.get("manual/sharded-mlx").expect("sharded mlx entry registered");
        assert_eq!(entry.format, ModelFormat::Mlx);
    }

    #[test]
    fn import_from_path_rejects_dir_without_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        let src = dir.path().join("configs-only");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("config.json"), b"{}").unwrap();

        let err = store.import_from_path(src.to_str().unwrap()).unwrap_err().to_string();
        assert!(err.contains("*.safetensors"), "{err}");
    }
}
