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

    // Symlink from hf cache into our model store
    std::fs::create_dir_all(dest_dir)?;
    let dest = dest_dir.join(&target.rfilename);
    if !dest.exists() {
        std::os::unix::fs::symlink(&cached_path, &dest)?;
    }

    validate_gguf(&cached_path)?;

    tracing::info!(path = %dest.display(), "download complete");
    Ok(dest)
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
