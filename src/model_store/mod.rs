pub mod download;
pub mod registry;
pub mod import;
pub mod ollama_pull;

use std::path::PathBuf;

pub struct ModelStore {
    base_dir: PathBuf,
}

impl ModelStore {
    pub fn new(base_dir: Option<PathBuf>) -> Self {
        let base_dir = base_dir.unwrap_or_else(|| {
            let home = std::env::var("HOME").expect("HOME not set");
            PathBuf::from(home).join(".spindll")
        });
        Self { base_dir }
    }

    pub fn models_dir(&self) -> PathBuf {
        self.base_dir.join("models")
    }

    pub fn model_dir(&self, repo: &str) -> PathBuf {
        self.models_dir().join(repo)
    }

    pub fn registry_path(&self) -> PathBuf {
        self.base_dir.join("registry.json")
    }

    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(self.models_dir())
    }

    /// Pull a model from Ollama's registry (default) or HuggingFace.
    pub fn pull(&self, model: &str, quant: Option<&str>, from_hf: bool) -> anyhow::Result<PathBuf> {
        self.ensure_dirs()?;

        let (path, size_bytes) = if from_hf {
            let dest_dir = self.model_dir(model);
            let path = download::download_gguf(model, quant, &dest_dir)?;
            let size = std::fs::symlink_metadata(&path)?.len();
            (path, size)
        } else {
            let (name, _tag) = ollama_pull::parse_model_ref(model);
            let dest_dir = self.model_dir(&format!("ollama/{name}"));
            ollama_pull::pull_from_registry(model, &dest_dir)?
        };

        // Validate GGUF
        download::validate_gguf(&path)?;

        // Register
        let filename = path.file_name().unwrap().to_string_lossy().to_string();
        let key = if from_hf {
            format!("{}/{}", model, filename)
        } else {
            let (name, _tag) = ollama_pull::parse_model_ref(model);
            format!("ollama/{name}/{filename}")
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
        });
        reg.save(&self.registry_path())?;

        Ok(path)
    }

    pub fn list(&self) -> anyhow::Result<()> {
        let reg = registry::Registry::load(&self.registry_path())?;
        if reg.models.is_empty() {
            println!("no models downloaded");
            return Ok(());
        }

        println!("{:<50} {:>10}", "MODEL", "SIZE");
        println!("{}", "-".repeat(62));
        for (key, entry) in &reg.models {
            let size = format_size(entry.size_bytes);
            println!("{:<50} {:>10}", key, size);
        }
        Ok(())
    }

    /// Look up a model key in the registry and return the path to the GGUF file.
    pub fn resolve_model_path(&self, model: &str) -> anyhow::Result<PathBuf> {
        let reg = registry::Registry::load(&self.registry_path())?;
        let entry = reg
            .models
            .get(model)
            .ok_or_else(|| anyhow::anyhow!("model '{}' not found in registry", model))?;

        let path = &entry.path;
        // Resolve symlink to actual file
        let real = std::fs::canonicalize(path)
            .map_err(|_| anyhow::anyhow!("model file missing: {}", path.display()))?;
        Ok(real)
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
                    eprintln!("skipping {name}:{tag}: {e}");
                    continue;
                }
            };

            let layer = match manifest.model_layer() {
                Some(l) => l,
                None => {
                    eprintln!("skipping {name}:{tag}: no model layer found");
                    continue;
                }
            };

            let blob_path = import::digest_to_blob_path(&ollama, &layer.digest);
            if !blob_path.exists() {
                eprintln!("skipping {name}:{tag}: blob missing at {}", blob_path.display());
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

fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else {
        format!("{} KB", bytes / 1024)
    }
}
