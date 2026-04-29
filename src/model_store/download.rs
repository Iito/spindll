use hf_hub::api::sync::Api;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Quant selection
// ---------------------------------------------------------------------------

/// Quant preference order for default GGUF selection. Lower index = more
/// preferred. q4_k_m is the de-facto local-inference standard; fp16/bf16/f32
/// are research-precision and 3–4× the size, so they're deprioritized.
/// Override with `--quant` to pick a specific variant.
pub(crate) const QUANT_PRIORITY: &[&str] = &[
    "q4_k_m", "q5_k_m", "q4_k_s", "q5_k_s",
    "q4_0", "q5_0",
    "q3_k_m", "q3_k_s",
    "q8_0", "q2_k",
];

/// Full-precision tags used as fallback quant labels when no quantized
/// variant string appears in the filename.
const FULL_PRECISION: &[&str] = &["fp16", "bf16", "f32"];

/// Extract a quant tag from a GGUF filename, e.g.
///   "qwen2.5-3b-instruct-q4_k_m.gguf" -> Some("q4_k_m")
///   "qwen2.5-3b-instruct-fp16-00001-of-00002.gguf" -> Some("fp16")
///   "model.gguf" -> None
pub(crate) fn extract_quant(filename: &str) -> Option<&'static str> {
    let lower = filename.to_lowercase();
    for q in QUANT_PRIORITY.iter().chain(FULL_PRECISION.iter()) {
        if lower.contains(q) {
            return Some(q);
        }
    }
    None
}

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

/// Per-token KV cache bytes for an MLX model from its `config.json`.
/// `2 (K+V) × n_layers × n_kv_heads × head_dim × 2 (fp16)`. Returns 0 if
/// any required field is missing.
pub fn kv_bpt_from_mlx_config(config: &serde_json::Value) -> u64 {
    fn calc(c: &serde_json::Value) -> Option<u64> {
        let n_layers = c.get("num_hidden_layers")?.as_u64()?;
        let n_heads = c.get("num_attention_heads")?.as_u64()?;
        let n_kv_heads = c
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(n_heads);
        let hidden = c.get("hidden_size")?.as_u64()?;
        let head_dim = hidden.checked_div(n_heads).filter(|&d| d > 0)?;
        Some(2 * n_layers * n_kv_heads * head_dim * 2)
    }
    calc(config).unwrap_or(0)
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

    #[test]
    fn extract_quant_finds_q4_k_m() {
        assert_eq!(extract_quant("qwen2.5-3b-instruct-q4_k_m.gguf"), Some("q4_k_m"));
    }

    #[test]
    fn extract_quant_finds_fp16_in_sharded_name() {
        assert_eq!(
            extract_quant("qwen2.5-3b-instruct-fp16-00001-of-00002.gguf"),
            Some("fp16")
        );
    }

    #[test]
    fn extract_quant_case_insensitive() {
        assert_eq!(extract_quant("Model-Q4_K_M.gguf"), Some("q4_k_m"));
        assert_eq!(extract_quant("Model-FP16.gguf"), Some("fp16"));
    }

    #[test]
    fn extract_quant_returns_none_when_no_match() {
        assert_eq!(extract_quant("model.gguf"), None);
    }

    /// Locks down the KV-bytes-per-token formula against accidental
    /// "simplification" — the manager's eviction sizing depends on it.
    /// Numbers come from `mlx-community/Qwen3-0.6B-4bit/config.json`.
    #[test]
    fn kv_bpt_matches_qwen3_0_6b_shape() {
        let cfg = serde_json::json!({
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "hidden_size": 1024,
        });
        // 2 (K+V) × 28 layers × 8 kv-heads × (1024/16=64 head_dim) × 2 (fp16) = 57344
        assert_eq!(kv_bpt_from_mlx_config(&cfg), 57344);
    }

    #[test]
    fn kv_bpt_returns_zero_when_field_missing() {
        let cfg = serde_json::json!({
            "num_hidden_layers": 28,
            // num_attention_heads missing
            "hidden_size": 1024,
        });
        assert_eq!(kv_bpt_from_mlx_config(&cfg), 0);
    }
}
