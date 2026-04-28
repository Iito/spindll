use hf_hub::api::sync::Api;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Quant preference order for default GGUF selection. Lower index = more
/// preferred. q4_k_m is the de-facto local-inference standard; fp16/bf16/f32
/// are research-precision and 3–4× the size, so they're deprioritized.
/// Override with `--quant` to pick a specific variant.
const QUANT_PRIORITY: &[&str] = &[
    "q4_k_m", "q5_k_m", "q4_k_s", "q5_k_s",
    "q4_0", "q5_0",
    "q3_k_m", "q3_k_s",
    "q8_0", "q2_k",
];

/// Lower rank = more preferred. Files that don't match any known quant
/// fall between the priority list and full-precision; fp16/bf16/f32 sort
/// last so we don't accidentally pull a 6 GB research weight when a 2 GB
/// inference quant exists alongside it.
fn rank_quant(filename: &str) -> usize {
    let lower = filename.to_lowercase();
    for (i, q) in QUANT_PRIORITY.iter().enumerate() {
        if lower.contains(q) {
            return i;
        }
    }
    if lower.contains("fp16") || lower.contains("bf16") || lower.contains("f32") {
        return usize::MAX;
    }
    QUANT_PRIORITY.len()
}

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
        // No quant specified — pick the lowest-ranked variant. Stable: ties
        // resolve in API order via min_by_key's first-wins semantics.
        let picked = gguf_files
            .iter()
            .min_by_key(|s| rank_quant(&s.rfilename))
            .expect("non-empty checked above");
        tracing::info!(
            file = %picked.rfilename,
            "no --quant specified, picked default by quant priority (q4_k_m > q5_k_m > q4_0 > … > fp16); pass --quant to override"
        );
        picked
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_k_m_beats_fp16_and_q8() {
        let files = [
            "qwen2.5-3b-instruct-fp16-00001-of-00002.gguf",
            "qwen2.5-3b-instruct-q4_k_m.gguf",
            "qwen2.5-3b-instruct-q8_0.gguf",
        ];
        let picked = files.iter().min_by_key(|f| rank_quant(f)).unwrap();
        assert_eq!(*picked, "qwen2.5-3b-instruct-q4_k_m.gguf");
    }

    #[test]
    fn q5_k_m_beats_q8_when_no_q4() {
        let files = [
            "model-q8_0.gguf",
            "model-q5_k_m.gguf",
            "model-fp16.gguf",
        ];
        let picked = files.iter().min_by_key(|f| rank_quant(f)).unwrap();
        assert_eq!(*picked, "model-q5_k_m.gguf");
    }

    #[test]
    fn unknown_quant_beats_fp16() {
        let files = ["model-iq4_xs.gguf", "model-fp16.gguf"];
        let picked = files.iter().min_by_key(|f| rank_quant(f)).unwrap();
        assert_eq!(*picked, "model-iq4_xs.gguf");
    }

    #[test]
    fn case_insensitive() {
        assert_eq!(rank_quant("Model-Q4_K_M.gguf"), 0);
        assert_eq!(rank_quant("Model-FP16.gguf"), usize::MAX);
    }

    #[test]
    fn fp16_only_repo_picks_first() {
        // Pathological case: only fp16 variants available. Ties go to API
        // order — caller still gets *something*.
        let files = ["model-fp16-2.gguf", "model-fp16-1.gguf"];
        let picked = files.iter().min_by_key(|f| rank_quant(f)).unwrap();
        assert_eq!(*picked, "model-fp16-2.gguf");
    }
}
