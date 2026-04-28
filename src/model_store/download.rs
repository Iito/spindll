use hf_hub::api::sync::Api;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Download a GGUF model from HuggingFace and symlink it into the local store.
pub fn download_gguf(repo_id: &str, quant: Option<&str>, dest_dir: &Path) -> anyhow::Result<PathBuf> {
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());

    let info = repo.info()?;
    let gguf_files: Vec<_> = info
        .siblings
        .iter()
        .filter(|s| s.rfilename.ends_with(".gguf"))
        .collect();

    if gguf_files.is_empty() {
        anyhow::bail!("no GGUF files found in {repo_id}");
    }

    let target = if let Some(q) = quant {
        gguf_files
            .iter()
            .find(|s| s.rfilename.contains(q))
            .ok_or_else(|| anyhow::anyhow!("no file matching quantization '{q}' in {repo_id}"))?
    } else {
        &gguf_files[0]
    };

    tracing::info!(file = %target.rfilename, "downloading");
    let cached_path = repo.get(&target.rfilename)?;

    // Link from hf cache into our model store.
    // Unix: symlink. Windows: hard_link (no privilege required), falling back
    // to copy if the cache and store are on different volumes.
    std::fs::create_dir_all(dest_dir)?;
    let dest = dest_dir.join(&target.rfilename);
    if !dest.exists() {
        #[cfg(unix)]
        std::os::unix::fs::symlink(&cached_path, &dest)?;
        #[cfg(windows)]
        if std::fs::hard_link(&cached_path, &dest).is_err() {
            std::fs::copy(&cached_path, &dest)?;
        }
    }

    validate_gguf(&cached_path)?;

    tracing::info!(path = %dest.display(), "download complete");
    Ok(dest)
}

/// Download an MLX model (safetensors + config) from HuggingFace and link into the local store.
///
/// Returns `(dir_path, total_size, config_digest)`.
pub fn download_mlx(repo_id: &str, dest_dir: &Path) -> anyhow::Result<(PathBuf, u64, String)> {
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());
    let info = repo.info()?;

    let mlx_files: Vec<_> = info
        .siblings
        .iter()
        .filter(|s| {
            s.rfilename.ends_with(".safetensors")
                || s.rfilename.ends_with(".json")
                || s.rfilename.ends_with(".txt")
                || s.rfilename.ends_with(".model")
                || s.rfilename == "tokenizer.model"
        })
        .collect();

    let has_safetensors = mlx_files.iter().any(|s| s.rfilename.ends_with(".safetensors"));
    let has_config = mlx_files.iter().any(|s| s.rfilename == "config.json");

    if !has_safetensors || !has_config {
        anyhow::bail!("repo {repo_id} does not contain MLX model files (need safetensors + config.json)");
    }

    std::fs::create_dir_all(dest_dir)?;
    let mut total_size = 0u64;

    for file in &mlx_files {
        tracing::info!(file = %file.rfilename, "downloading");
        let cached_path = repo.get(&file.rfilename)?;

        let dest = dest_dir.join(&file.rfilename);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if !dest.exists() {
            link_or_copy(&cached_path, &dest)?;
        }
        total_size += std::fs::symlink_metadata(&dest)?.len();
    }

    let config_path = dest_dir.join("config.json");
    let digest = sha256_file(&config_path)?;

    tracing::info!(path = %dest_dir.display(), files = mlx_files.len(), "MLX download complete");
    Ok((dest_dir.to_path_buf(), total_size, digest))
}

/// Read MLX model metadata from config.json.
///
/// Returns `(architecture, model_name)`.
pub fn read_mlx_metadata(dir: &Path) -> (String, String) {
    let config_path = dir.join("config.json");
    let data = match std::fs::read_to_string(&config_path) {
        Ok(d) => d,
        Err(_) => return (String::new(), String::new()),
    };
    let json: serde_json::Value = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(_) => return (String::new(), String::new()),
    };

    let arch = json.get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let name = json.get("_name_or_path")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    (arch, name)
}

fn link_or_copy(src: &Path, dest: &Path) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(src, dest)?;
    }
    #[cfg(windows)]
    {
        if std::fs::hard_link(src, dest).is_err() {
            std::fs::copy(src, dest)?;
        }
    }
    Ok(())
}

/// Compute SHA-256 digest of a file, returned as `"sha256:<hex>"`.
pub fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

/// Check that a file starts with the GGUF magic bytes.
pub fn validate_gguf(path: &Path) -> anyhow::Result<()> {
    let mut f = std::fs::File::open(path)?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;

    if &magic != b"GGUF" {
        anyhow::bail!(
            "not a valid GGUF file (magic: {:02x}{:02x}{:02x}{:02x}): {}",
            magic[0], magic[1], magic[2], magic[3],
            path.display()
        );
    }
    Ok(())
}
