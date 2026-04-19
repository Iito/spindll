pub mod download;
pub mod registry;
pub mod import;

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

    pub fn pull(&self, repo_id: &str, quant: Option<&str>) -> anyhow::Result<PathBuf> {
        self.ensure_dirs()?;
        let dest_dir = self.model_dir(repo_id);
        let path = download::download_gguf(repo_id, quant, &dest_dir)?;

        // Record in registry
        let metadata = std::fs::symlink_metadata(&path)?;
        let filename = path.file_name().unwrap().to_string_lossy().to_string();
        let key = format!("{}/{}", repo_id, filename);

        let mut reg = registry::Registry::load(&self.registry_path())?;
        reg.add(key, registry::ModelEntry {
            repo: repo_id.to_string(),
            filename,
            path: path.clone(),
            size_bytes: metadata.len(),
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
