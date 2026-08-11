// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Iito <https://github.com/Iito> and sarmientoF <https://github.com/sarmientoF>

use serde::Deserialize;
use std::path::{Path, PathBuf};

const OLLAMA_MODELS_DIR: &str = ".ollama/models";
const MANIFESTS_DIR: &str = "manifests/registry.ollama.ai/library";
const BLOBS_DIR: &str = "blobs";
const HF_CACHE_DIR: &str = ".cache/huggingface/hub";

/// Parsed Ollama manifest describing a model's layers (blobs).
#[derive(Debug, Deserialize)]
pub struct OllamaManifest {
    /// The layers that make up this model (model weights, template, license, etc.).
    pub layers: Vec<OllamaLayer>,
}

/// A single layer (blob) within an Ollama manifest.
#[derive(Debug, Deserialize)]
pub struct OllamaLayer {
    /// MIME type identifying the layer's role (e.g. `"application/vnd.ollama.image.model"`).
    #[serde(rename = "mediaType")]
    pub media_type: String,
    /// Content-addressable digest (e.g. `"sha256:abc123..."`).
    pub digest: String,
    /// Layer size in bytes.
    pub size: u64,
}

impl OllamaManifest {
    /// Find the model layer (the GGUF blob).
    pub fn model_layer(&self) -> Option<&OllamaLayer> {
        self.layers
            .iter()
            .find(|l| l.media_type == "application/vnd.ollama.image.model")
    }
}

/// Return the path to Ollama's model directory, or `None` when no home
/// directory can be determined.
pub fn ollama_dir() -> Option<PathBuf> {
    Some(home_dir()?.join(OLLAMA_MODELS_DIR))
}

/// Parse an Ollama manifest file.
pub fn parse_manifest(path: &Path) -> anyhow::Result<OllamaManifest> {
    let data = std::fs::read_to_string(path)?;
    let manifest: OllamaManifest = serde_json::from_str(&data)?;
    Ok(manifest)
}

/// Convert a digest like "sha256:abc123..." to the blob filename "sha256-abc123..."
pub fn digest_to_blob_path(ollama_dir: &Path, digest: &str) -> PathBuf {
    let blob_name = digest.replace(':', "-");
    ollama_dir.join(BLOBS_DIR).join(blob_name)
}

/// Discover all Ollama models by scanning the manifests directory.
/// Returns (model_name, tag, manifest_path) tuples.
pub fn discover_models(ollama_dir: &Path) -> anyhow::Result<Vec<(String, String, PathBuf)>> {
    let manifests_dir = ollama_dir.join(MANIFESTS_DIR);
    if !manifests_dir.exists() {
        anyhow::bail!("ollama manifests not found at {}", manifests_dir.display());
    }

    let mut models = Vec::new();

    for model_entry in std::fs::read_dir(&manifests_dir)? {
        let model_entry = model_entry?;
        if !model_entry.file_type()?.is_dir() {
            continue;
        }
        let model_name = model_entry.file_name().to_string_lossy().to_string();

        for tag_entry in std::fs::read_dir(model_entry.path())? {
            let tag_entry = tag_entry?;
            let tag = tag_entry.file_name().to_string_lossy().to_string();
            models.push((model_name.clone(), tag, tag_entry.path()));
        }
    }

    Ok(models)
}

/// Return the path to HuggingFace's cache directory, or `None` when no home
/// directory can be determined.
pub fn hf_cache_dir() -> Option<PathBuf> {
    Some(home_dir()?.join(HF_CACHE_DIR))
}

/// Resolve the user's home directory from `HOME`, falling back to
/// `USERPROFILE` (Windows shells typically set only the latter).
pub(crate) fn home_dir() -> Option<PathBuf> {
    home_from(std::env::var_os("HOME"), std::env::var_os("USERPROFILE"))
}

fn home_from(
    home: Option<std::ffi::OsString>,
    userprofile: Option<std::ffi::OsString>,
) -> Option<PathBuf> {
    home.filter(|v| !v.is_empty())
        .or(userprofile.filter(|v| !v.is_empty()))
        .map(PathBuf::from)
}

/// Discover all GGUF and MLX models in the HuggingFace cache.
/// Returns (repo_id, model_path) tuples where repo_id is like "owner/repo".
pub fn discover_hf_models(hf_cache_dir: &Path) -> anyhow::Result<Vec<(String, PathBuf)>> {
    if !hf_cache_dir.exists() {
        anyhow::bail!("huggingface cache not found at {}", hf_cache_dir.display());
    }

    let mut models = Vec::new();

    for entry in std::fs::read_dir(hf_cache_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();

        // HF cache uses names like "models--owner--repo"
        if !name.starts_with("models--") {
            continue;
        }

        // Convert "models--owner--repo" to "owner/repo"
        let repo_id = name
            .strip_prefix("models--")
            .unwrap()
            .replace("--", "/");

        let path = entry.path();
        let snapshots = path.join("snapshots");
        if !snapshots.exists() {
            continue;
        }

        // Find the latest snapshot (just pick the first one for now)
        if let Ok(snapshots_iter) = std::fs::read_dir(&snapshots) {
            for snapshot_entry in snapshots_iter.flatten() {
                let snapshot_path = snapshot_entry.path();
                if snapshot_entry.file_type()?.is_dir() {
                    models.push((repo_id.clone(), snapshot_path));
                    break; // Use only the first snapshot
                }
            }
        }
    }

    Ok(models)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    fn os(s: &str) -> Option<OsString> {
        Some(OsString::from(s))
    }

    #[test]
    fn home_prefers_home_over_userprofile() {
        assert_eq!(
            home_from(os("/home/a"), os("C:\\Users\\a")),
            Some(PathBuf::from("/home/a"))
        );
    }

    #[test]
    fn home_falls_back_to_userprofile() {
        assert_eq!(
            home_from(None, os("C:\\Users\\a")),
            Some(PathBuf::from("C:\\Users\\a"))
        );
    }

    #[test]
    fn home_ignores_empty_values() {
        assert_eq!(
            home_from(os(""), os("C:\\Users\\a")),
            Some(PathBuf::from("C:\\Users\\a"))
        );
        assert_eq!(home_from(os(""), os("")), None);
        assert_eq!(home_from(None, None), None);
    }
}
