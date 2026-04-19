use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelEntry {
    pub repo: String,
    pub filename: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub downloaded_at: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Registry {
    pub models: HashMap<String, ModelEntry>,
}

impl Registry {
    pub fn load(path: &Path) -> Self {
        if path.exists() {
            let data = std::fs::read_to_string(path).unwrap();
            serde_json::from_str(&data).unwrap()
        } else {
            Self::default()
        }
    }

    pub fn save(&self, path: &Path) {
        let data = serde_json::to_string_pretty(self).unwrap();
        std::fs::write(path, data).unwrap();
    }

    pub fn add(&mut self, key: String, entry: ModelEntry) {
        self.models.insert(key, entry);
    }

    pub fn remove(&mut self, key: &str) -> Option<ModelEntry> {
        self.models.remove(key)
    }
}
