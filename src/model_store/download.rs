use hf_hub::api::sync::Api;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Combined HF downloader (auto-detects GGUF vs MLX from repo contents)
// ---------------------------------------------------------------------------

/// Result of an auto-detected HuggingFace download.
pub enum HfDownload {
    /// A single GGUF file was downloaded.
    Gguf {
        /// Path to the downloaded (or symlinked) GGUF file.
        path: PathBuf,
        /// Filename portion, used as the registry key suffix.
        filename: String,
        /// File size in bytes.
        size: u64,
        /// SHA-256 digest.
        digest: String,
    },
    /// An MLX safetensors directory was downloaded.
    Mlx {
        /// Path to the directory containing config.json + safetensors shards.
        dir: PathBuf,
        /// Sum of all downloaded file sizes.
        size: u64,
        /// SHA-256 digest of `config.json` (model version identifier).
        digest: String,
    },
}

/// Download a model from HuggingFace, auto-detecting the format:
/// - Repo contains `.gguf` files → downloads the best-matching GGUF.
/// - Repo contains `config.json` + `.safetensors` but no GGUF → downloads MLX directory.
pub fn download_hf_auto(
    repo_id: &str,
    quant: Option<&str>,
    dest_dir: &Path,
) -> anyhow::Result<HfDownload> {
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());
    let info = repo.info()?;

    let gguf_files: Vec<_> = info
        .siblings
        .iter()
        .filter(|s| s.rfilename.ends_with(".gguf"))
        .collect();

    if !gguf_files.is_empty() {
        // --- GGUF path ---
        let target = if let Some(q) = quant {
            gguf_files
                .iter()
                .find(|s| s.rfilename.contains(q))
                .ok_or_else(|| anyhow::anyhow!("no file matching '{q}' in {repo_id}"))?
        } else {
            // Prefer Q4_K_M as a sensible default; fall back to first entry.
            gguf_files
                .iter()
                .find(|s| s.rfilename.contains("Q4_K_M"))
                .unwrap_or(&gguf_files[0])
        };

        tracing::info!(file = %target.rfilename, "downloading GGUF");
        let cached = repo.get(&target.rfilename)?;

        std::fs::create_dir_all(dest_dir)?;
        let dest = dest_dir.join(&target.rfilename);
        if !dest.exists() {
            link_or_copy(&cached, &dest)?;
        }

        let size   = std::fs::metadata(&cached)?.len();
        let digest = sha256_file(&cached)?;

        tracing::info!(path = %dest.display(), "GGUF download complete");
        return Ok(HfDownload::Gguf {
            path: dest,
            filename: target.rfilename.clone(),
            size,
            digest,
        });
    }

    // --- MLX path ---
    let mlx_files: Vec<_> = info
        .siblings
        .iter()
        .filter(|s| {
            let f = &s.rfilename;
            f.ends_with(".safetensors")
                || f.ends_with(".json")
                || f.ends_with(".txt")
                || f.ends_with(".model")
        })
        .collect();

    let has_safetensors = mlx_files.iter().any(|s| s.rfilename.ends_with(".safetensors"));
    let has_config      = mlx_files.iter().any(|s| s.rfilename == "config.json");

    if !has_safetensors || !has_config {
        anyhow::bail!(
            "no GGUF files found and no MLX safetensors + config.json in '{repo_id}'"
        );
    }

    std::fs::create_dir_all(dest_dir)?;
    let mut total_size = 0u64;

    for sibling in &mlx_files {
        tracing::info!(file = %sibling.rfilename, "downloading MLX");
        let cached = repo.get(&sibling.rfilename)?;

        let dest = dest_dir.join(&sibling.rfilename);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if !dest.exists() {
            link_or_copy(&cached, &dest)?;
        }

        total_size += std::fs::metadata(&cached).map(|m| m.len()).unwrap_or(0);
    }

    let config_path = dest_dir.join("config.json");
    let digest = sha256_file(&config_path)?;

    tracing::info!(
        path  = %dest_dir.display(),
        files = mlx_files.len(),
        "MLX download complete"
    );
    Ok(HfDownload::Mlx {
        dir: dest_dir.to_path_buf(),
        size: total_size,
        digest,
    })
}

/// Read `model_type` and `_name_or_path` from an MLX model's `config.json`.
///
/// Returns `(architecture, model_name)`.
pub fn read_mlx_metadata(dir: &Path) -> (String, String) {
    let Ok(content) = std::fs::read_to_string(dir.join("config.json")) else {
        return (String::new(), String::new());
    };
    let Ok(json): Result<serde_json::Value, _> = serde_json::from_str(&content) else {
        return (String::new(), String::new());
    };

    let arch = json
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let name = json
        .get("_name_or_path")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    (arch, name)
}

// ---------------------------------------------------------------------------
// Kept for backward compatibility (used by the Ollama pull path)
// ---------------------------------------------------------------------------

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

    std::fs::create_dir_all(dest_dir)?;
    let dest = dest_dir.join(&target.rfilename);
    if !dest.exists() {
        link_or_copy(&cached_path, &dest)?;
    }

    validate_gguf(&cached_path)?;
    tracing::info!(path = %dest.display(), "download complete");
    Ok(dest)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Symlink on Unix; hard-link (with copy fallback) on Windows.
fn link_or_copy(src: &Path, dest: &Path) -> anyhow::Result<()> {
    #[cfg(unix)]
    std::os::unix::fs::symlink(src, dest)?;
    #[cfg(windows)]
    if std::fs::hard_link(src, dest).is_err() {
        std::fs::copy(src, dest)?;
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
