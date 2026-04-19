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
        download::download_gguf(repo_id, quant, &dest_dir)
    }
}
