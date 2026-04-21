use crate::model_store::import::{OllamaManifest, OllamaLayer};
use indicatif::{ProgressBar, ProgressStyle};
use std::io::Write;
use std::path::{Path, PathBuf};

const REGISTRY_BASE: &str = "https://registry.ollama.ai";

/// Parse a model reference like "llama3.1:8b" into (name, tag).
/// Defaults to "latest" if no tag is specified.
pub fn parse_model_ref(model: &str) -> (&str, &str) {
    match model.split_once(':') {
        Some((name, tag)) => (name, tag),
        None => (model, "latest"),
    }
}

/// Pull a model from Ollama's registry.
/// Returns (path, size, digest).
pub fn pull_from_registry(
    model: &str,
    dest_dir: &Path,
) -> anyhow::Result<(PathBuf, u64, String)> {
    let (name, tag) = parse_model_ref(model);
    let client = reqwest::blocking::Client::new();

    // Fetch manifest
    let manifest_url = format!("{REGISTRY_BASE}/v2/library/{name}/manifests/{tag}");
    tracing::info!(name, tag, "pulling model from registry");

    let resp = client.get(&manifest_url).send()?;
    if !resp.status().is_success() {
        anyhow::bail!(
            "failed to fetch manifest for {name}:{tag}: HTTP {}",
            resp.status()
        );
    }

    let manifest: OllamaManifest = resp.json()?;
    let layer = manifest
        .model_layer()
        .ok_or_else(|| anyhow::anyhow!("no model layer in manifest for {name}:{tag}"))?;

    // Download the blob
    let digest = layer.digest.clone();
    let blob_url = format!("{REGISTRY_BASE}/v2/library/{name}/blobs/{}", layer.digest);
    let dest = download_blob(&client, &blob_url, layer, dest_dir, name, tag)?;

    Ok((dest, layer.size, digest))
}

fn download_blob(
    client: &reqwest::blocking::Client,
    url: &str,
    layer: &OllamaLayer,
    dest_dir: &Path,
    name: &str,
    tag: &str,
) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dest_dir)?;
    let filename = format!("{tag}.gguf");
    let dest = dest_dir.join(&filename);

    if dest.exists() {
        let meta = std::fs::metadata(&dest)?;
        if meta.len() == layer.size {
            tracing::debug!(path = %dest.display(), "already downloaded");
            return Ok(dest);
        }
    }

    let mut resp = client.get(url).send()?;
    if !resp.status().is_success() {
        anyhow::bail!("failed to download blob: HTTP {}", resp.status());
    }

    let pb = ProgressBar::new(layer.size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40}] {bytes}/{total_bytes} ({eta})")?
            .progress_chars("=> "),
    );
    pb.set_message(format!("{name}:{tag}"));

    let mut file = std::fs::File::create(&dest)?;
    let mut downloaded = 0u64;
    let mut buf = vec![0u8; 8192];

    loop {
        let n = std::io::Read::read(&mut resp, &mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        downloaded += n as u64;
        pb.set_position(downloaded);
    }

    pb.finish_with_message("done");
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_ref() {
        assert_eq!(parse_model_ref("llama3.1:8b"), ("llama3.1", "8b"));
        assert_eq!(parse_model_ref("llama3.1"), ("llama3.1", "latest"));
        assert_eq!(parse_model_ref("qwen2:0.5b"), ("qwen2", "0.5b"));
    }
}
