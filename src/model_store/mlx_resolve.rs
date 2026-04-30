use hf_hub::api::sync::Api;

pub struct MlxCandidate {
    pub repo_id: String,
    pub size_estimate: u64,
}

/// Try to find an MLX-format repo on HuggingFace for the given model.
///
/// Attempts a direct repo probe first (fast), then falls back to the HF search API.
/// Returns `None` if no suitable candidate is found.
pub fn find_mlx_repo(model_name: &str, quant: &str) -> anyhow::Result<Option<MlxCandidate>> {
    let base = resolve_base_name(model_name);

    // Strategy 1: direct repo probe — most MLX repos follow a predictable naming convention.
    let direct = format!("mlx-community/{base}-{quant}");
    if let Some(candidate) = probe_repo(&direct)? {
        return Ok(Some(candidate));
    }
    // Try without quant suffix (fp16 repos have no suffix).
    if let Some(candidate) = probe_repo(&format!("mlx-community/{base}"))? {
        return Ok(Some(candidate));
    }

    // Strategy 2: HF model search API.
    if let Some(candidate) = search_hf(&base, quant)? {
        return Ok(Some(candidate));
    }

    Ok(None)
}

/// Map well-known Ollama model names to their HuggingFace base model name.
pub fn ollama_to_hf_base(name: &str, tag: &str) -> Option<&'static str> {
    match (name, tag) {
        ("llama3.1", "8b") => Some("Meta-Llama-3.1-8B-Instruct"),
        ("llama3.1", "70b") => Some("Meta-Llama-3.1-70B-Instruct"),
        ("llama3.1", "405b") => Some("Meta-Llama-3.1-405B-Instruct"),
        ("llama3.2", "1b") => Some("Llama-3.2-1B-Instruct"),
        ("llama3.2", "3b") => Some("Llama-3.2-3B-Instruct"),
        ("llama3.3", "70b") => Some("Llama-3.3-70B-Instruct"),
        ("mistral", "7b") => Some("Mistral-7B-Instruct-v0.3"),
        ("gemma2", "2b") => Some("gemma-2-2b-it"),
        ("gemma2", "9b") => Some("gemma-2-9b-it"),
        ("gemma2", "27b") => Some("gemma-2-27b-it"),
        ("qwen2.5", "0.5b") => Some("Qwen2.5-0.5B-Instruct"),
        ("qwen2.5", "1.5b") => Some("Qwen2.5-1.5B-Instruct"),
        ("qwen2.5", "3b") => Some("Qwen2.5-3B-Instruct"),
        ("qwen2.5", "7b") => Some("Qwen2.5-7B-Instruct"),
        ("qwen2.5", "14b") => Some("Qwen2.5-14B-Instruct"),
        ("qwen2.5", "32b") => Some("Qwen2.5-32B-Instruct"),
        ("qwen2.5", "72b") => Some("Qwen2.5-72B-Instruct"),
        ("phi3", "3.8b") => Some("Phi-3-mini-4k-instruct"),
        ("phi3", "14b") => Some("Phi-3-medium-4k-instruct"),
        ("deepseek-r1", "1.5b") => Some("DeepSeek-R1-Distill-Qwen-1.5B"),
        ("deepseek-r1", "7b") => Some("DeepSeek-R1-Distill-Qwen-7B"),
        ("deepseek-r1", "8b") => Some("DeepSeek-R1-Distill-Llama-8B"),
        ("deepseek-r1", "14b") => Some("DeepSeek-R1-Distill-Qwen-14B"),
        ("deepseek-r1", "32b") => Some("DeepSeek-R1-Distill-Qwen-32B"),
        ("deepseek-r1", "70b") => Some("DeepSeek-R1-Distill-Llama-70B"),
        _ => None,
    }
}

/// Resolve a user-provided model name to a base HF name suitable for MLX search.
fn resolve_base_name(model_name: &str) -> String {
    // Try Ollama name:tag mapping first.
    if let Some((name, tag)) = model_name.split_once(':') {
        if let Some(base) = ollama_to_hf_base(name, tag) {
            return base.to_string();
        }
        // Unknown Ollama model — return name and tag joined for fuzzy search.
        return format!("{name}-{tag}");
    }

    // Bare Ollama name without tag — try "latest" equivalent.
    if !model_name.contains('/') {
        if let Some(base) = ollama_to_hf_base(model_name, "latest") {
            return base.to_string();
        }
        return model_name.to_string();
    }

    // HuggingFace GGUF repo — strip org and GGUF suffixes.
    let repo_part = model_name.rsplit('/').next().unwrap_or(model_name);
    repo_part
        .strip_suffix("-GGUF")
        .or_else(|| repo_part.strip_suffix("-gguf"))
        .or_else(|| repo_part.strip_suffix("-quantized"))
        .unwrap_or(repo_part)
        .to_string()
}

/// Check if a specific repo exists on HuggingFace and looks like a valid MLX model.
pub fn probe_repo(repo_id: &str) -> anyhow::Result<Option<MlxCandidate>> {
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());

    let info = match repo.info() {
        Ok(i) => i,
        Err(_) => return Ok(None),
    };

    let has_safetensors = info.siblings.iter().any(|s| s.rfilename.ends_with(".safetensors"));
    let has_config = info.siblings.iter().any(|s| s.rfilename == "config.json");

    if !has_safetensors || !has_config {
        return Ok(None);
    }

    Ok(Some(MlxCandidate {
        repo_id: repo_id.to_string(),
        size_estimate: 0,
    }))
}

/// Search HuggingFace for MLX models matching the base name.
fn search_hf(base_name: &str, quant: &str) -> anyhow::Result<Option<MlxCandidate>> {
    let url = format!(
        "https://huggingface.co/api/models?author=mlx-community&search={}&sort=downloads&direction=-1&limit=20",
        urlencoding::encode(base_name)
    );

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    let resp = match client.get(&url).send() {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!("HF search failed: {e}");
            return Ok(None);
        }
    };

    if !resp.status().is_success() {
        return Ok(None);
    }

    let models: Vec<HfModelInfo> = resp.json()?;
    if models.is_empty() {
        return Ok(None);
    }

    // Prefer the quant variant (e.g., "-4bit"), then fall back to any match.
    let quant_suffix = format!("-{quant}");
    let base_lower = base_name.to_lowercase();

    let best = models.iter()
        .filter(|m| {
            let name = m.id.strip_prefix("mlx-community/").unwrap_or(&m.id).to_lowercase();
            let stripped = name
                .strip_suffix("-4bit")
                .or_else(|| name.strip_suffix("-8bit"))
                .or_else(|| name.strip_suffix("-bf16"))
                .unwrap_or(&name);
            stripped == base_lower || name.contains(&base_lower)
        })
        .min_by_key(|m| {
            let has_quant = m.id.to_lowercase().ends_with(&quant_suffix);
            (!has_quant as u8, u64::MAX - m.downloads)
        });

    match best {
        Some(m) => Ok(Some(MlxCandidate {
            repo_id: m.id.clone(),
            size_estimate: 0,
        })),
        None => Ok(None),
    }
}

#[derive(serde::Deserialize)]
struct HfModelInfo {
    #[serde(rename = "modelId", alias = "id")]
    id: String,
    #[serde(default)]
    downloads: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_map_covers_common_models() {
        assert_eq!(ollama_to_hf_base("llama3.1", "8b"), Some("Meta-Llama-3.1-8B-Instruct"));
        assert_eq!(ollama_to_hf_base("gemma2", "9b"), Some("gemma-2-9b-it"));
        assert_eq!(ollama_to_hf_base("qwen2.5", "7b"), Some("Qwen2.5-7B-Instruct"));
        assert_eq!(ollama_to_hf_base("unknown", "7b"), None);
    }

    #[test]
    fn resolve_base_name_from_ollama() {
        assert_eq!(resolve_base_name("llama3.1:8b"), "Meta-Llama-3.1-8B-Instruct");
        assert_eq!(resolve_base_name("unknown:7b"), "unknown-7b");
    }

    #[test]
    fn resolve_base_name_from_hf_repo() {
        assert_eq!(
            resolve_base_name("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"),
            "Meta-Llama-3.1-8B-Instruct"
        );
        assert_eq!(
            resolve_base_name("TheBloke/some-model-gguf"),
            "some-model"
        );
    }
}
