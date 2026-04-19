use hf_hub::api::sync::Api;
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

    println!("downloading {}", target.rfilename);
    let cached_path = repo.get(&target.rfilename)?;

    // Symlink from hf cache into our model store
    std::fs::create_dir_all(dest_dir)?;
    let dest = dest_dir.join(&target.rfilename);
    if !dest.exists() {
        std::os::unix::fs::symlink(&cached_path, &dest)?;
    }

    println!("done: {}", dest.display());
    Ok(dest)
}
