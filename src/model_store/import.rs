use serde::Deserialize;
use std::path::{Path, PathBuf};

const OLLAMA_MODELS_DIR: &str = ".ollama/models";
const MANIFESTS_DIR: &str = "manifests/registry.ollama.ai/library";
const BLOBS_DIR: &str = "blobs";

#[derive(Debug, Deserialize)]
pub struct OllamaManifest {
    pub layers: Vec<OllamaLayer>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaLayer {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub digest: String,
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

/// Return the path to Ollama's model directory.
pub fn ollama_dir() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME not set");
    PathBuf::from(home).join(OLLAMA_MODELS_DIR)
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
