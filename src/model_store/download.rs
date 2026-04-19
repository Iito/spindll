use hf_hub::api::sync::Api;
use std::path::PathBuf;

/// Download a GGUF model from HuggingFace.
/// Returns the path to the downloaded file.
pub fn download_gguf(repo_id: &str, quant: Option<&str>) -> anyhow::Result<PathBuf> {
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
    let path = repo.get(&target.rfilename)?;

    println!("done: {}", path.display());
    Ok(path)
}
