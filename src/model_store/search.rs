use serde::Deserialize;

use super::registry::ModelFormat;

#[derive(Debug, Clone, PartialEq)]
pub enum SearchSource {
    HuggingFace,
    Ollama,
}

impl std::fmt::Display for SearchSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HuggingFace => f.pad("HF"),
            Self::Ollama => f.pad("Ollama"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub name: String,
    pub source: SearchSource,
    pub format: ModelFormat,
    pub downloads: u64,
    pub estimated_bytes: Option<u64>,
}

#[derive(Deserialize)]
struct HfModel {
    #[serde(rename = "id")]
    id: String,
    #[serde(default)]
    downloads: u64,
}

pub fn search_models(query: &str, limit: usize) -> anyhow::Result<Vec<SearchResult>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    let mut results = Vec::new();

    match search_hf_gguf(&client, query) {
        Ok(gguf) => results.extend(gguf),
        Err(e) => tracing::warn!("HuggingFace GGUF search failed: {e:#}"),
    }
    match search_hf_mlx(&client, query) {
        Ok(mlx) => results.extend(mlx),
        Err(e) => tracing::warn!("HuggingFace MLX search failed: {e:#}"),
    }
    match probe_ollama(&client, query) {
        Ok(Some(ollama)) => results.push(ollama),
        Ok(None) => {}
        Err(e) => tracing::warn!("Ollama probe failed: {e:#}"),
    }

    let prefers_mlx = super::platform_prefers_mlx();
    let mem = crate::scheduler::budget::MemoryBudget::detect(None);
    rank_results(&mut results, prefers_mlx, mem.available_ram);

    results.truncate(limit);
    Ok(results)
}

fn search_hf_gguf(
    client: &reqwest::blocking::Client,
    query: &str,
) -> anyhow::Result<Vec<SearchResult>> {
    let url = format!(
        "https://huggingface.co/api/models?search={}&filter=gguf&sort=downloads&direction=-1&limit=20",
        urlencoding::encode(query),
    );
    let resp = client.get(&url).send()?;
    if !resp.status().is_success() {
        anyhow::bail!("HuggingFace API returned {}", resp.status());
    }
    let models: Vec<HfModel> = resp.json()?;

    Ok(models
        .into_iter()
        .map(|m| {
            let estimated = estimate_model_bytes(&m.id, &ModelFormat::Gguf);
            SearchResult {
                name: m.id,
                source: SearchSource::HuggingFace,
                format: ModelFormat::Gguf,
                downloads: m.downloads,
                estimated_bytes: estimated,
            }
        })
        .collect())
}

fn search_hf_mlx(
    client: &reqwest::blocking::Client,
    query: &str,
) -> anyhow::Result<Vec<SearchResult>> {
    let url = format!(
        "https://huggingface.co/api/models?author=mlx-community&search={}&sort=downloads&direction=-1&limit=20",
        urlencoding::encode(query),
    );
    let resp = client.get(&url).send()?;
    if !resp.status().is_success() {
        anyhow::bail!("HuggingFace API returned {}", resp.status());
    }
    let models: Vec<HfModel> = resp.json()?;

    Ok(models
        .into_iter()
        .map(|m| {
            let estimated = estimate_model_bytes(&m.id, &ModelFormat::Mlx);
            SearchResult {
                name: m.id,
                source: SearchSource::HuggingFace,
                format: ModelFormat::Mlx,
                downloads: m.downloads,
                estimated_bytes: estimated,
            }
        })
        .collect())
}

fn probe_ollama(
    client: &reqwest::blocking::Client,
    query: &str,
) -> anyhow::Result<Option<SearchResult>> {
    if query.contains('/') {
        return Ok(None);
    }
    let (name, tag) = super::ollama_pull::parse_model_ref(query);
    let url = format!(
        "https://registry.ollama.ai/v2/library/{name}/manifests/{tag}"
    );
    match client.get(&url).send() {
        Ok(resp) if resp.status().is_success() => Ok(Some(SearchResult {
            name: format!("{name}:{tag}"),
            source: SearchSource::Ollama,
            format: ModelFormat::Gguf,
            downloads: 0,
            estimated_bytes: None,
        })),
        _ => Ok(None),
    }
}

fn rank_results(results: &mut Vec<SearchResult>, prefers_mlx: bool, available_mem: u64) {
    results.sort_by(|a, b| {
        let a_preferred = matches!(a.format, ModelFormat::Mlx) == prefers_mlx;
        let b_preferred = matches!(b.format, ModelFormat::Mlx) == prefers_mlx;

        let a_fits = a.estimated_bytes.map_or(true, |s| s < available_mem);
        let b_fits = b.estimated_bytes.map_or(true, |s| s < available_mem);

        b_preferred
            .cmp(&a_preferred)
            .then_with(|| b_fits.cmp(&a_fits))
            .then_with(|| b.downloads.cmp(&a.downloads))
    });
}

fn estimate_model_bytes(name: &str, format: &ModelFormat) -> Option<u64> {
    let params = extract_param_billions(name)?;
    let bytes_per_param = match format {
        ModelFormat::Gguf => 0.55,
        ModelFormat::Mlx => {
            let lower = name.to_lowercase();
            if lower.contains("8bit") || lower.contains("8-bit") {
                1.1
            } else if lower.contains("bf16") || lower.contains("fp16") {
                2.0
            } else {
                0.55
            }
        }
    };
    Some((params * bytes_per_param * 1e9) as u64)
}

fn extract_param_billions(name: &str) -> Option<f64> {
    let bytes = name.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        if bytes[i].is_ascii_digit() || bytes[i] == b'.' {
            let start = i;
            while i < len && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                i += 1;
            }
            if i < len && (bytes[i] == b'B' || bytes[i] == b'b') {
                let after = if i + 1 < len { bytes[i + 1] } else { b'-' };
                if !after.is_ascii_alphabetic() {
                    let s = &name[start..i];
                    if let Ok(n) = s.parse::<f64>() {
                        if n > 0.0 && n < 1000.0 {
                            return Some(n);
                        }
                    }
                }
            }
        }
        i += 1;
    }
    None
}

pub fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else {
        format!("{:.0} MB", bytes as f64 / 1_048_576.0)
    }
}

pub fn format_downloads(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_params_standard() {
        assert_eq!(extract_param_billions("Qwen2.5-7B-Instruct-GGUF"), Some(7.0));
    }

    #[test]
    fn extract_params_decimal() {
        assert_eq!(extract_param_billions("Qwen2.5-0.5B-Instruct"), Some(0.5));
    }

    #[test]
    fn extract_params_lowercase() {
        assert_eq!(extract_param_billions("gemma-2-2b-it"), Some(2.0));
    }

    #[test]
    fn extract_params_none() {
        assert_eq!(extract_param_billions("Phi-3-mini-4k-instruct"), None);
    }

    #[test]
    fn extract_skips_bit_suffix() {
        assert_eq!(
            extract_param_billions("mlx-community/Qwen2.5-7B-Instruct-4bit"),
            Some(7.0),
        );
    }

    #[test]
    fn estimate_gguf_7b() {
        let est = estimate_model_bytes("Model-7B-GGUF", &ModelFormat::Gguf).unwrap();
        let gb = est as f64 / 1_073_741_824.0;
        assert!(gb > 3.0 && gb < 5.0, "expected ~3.8 GB, got {gb:.1} GB");
    }

    #[test]
    fn estimate_mlx_4bit() {
        let est = estimate_model_bytes("Model-7B-Instruct-4bit", &ModelFormat::Mlx).unwrap();
        let gb = est as f64 / 1_073_741_824.0;
        assert!(gb > 3.0 && gb < 5.0, "expected ~3.8 GB, got {gb:.1} GB");
    }

    #[test]
    fn estimate_mlx_fp16() {
        let est = estimate_model_bytes("Model-7B-Instruct-fp16", &ModelFormat::Mlx).unwrap();
        let gb = est as f64 / 1_073_741_824.0;
        assert!(gb > 12.0 && gb < 16.0, "expected ~14 GB, got {gb:.1} GB");
    }

    #[test]
    fn rank_prefers_format() {
        let mut results = vec![
            SearchResult {
                name: "gguf-model".into(),
                source: SearchSource::HuggingFace,
                format: ModelFormat::Gguf,
                downloads: 100_000,
                estimated_bytes: Some(4_000_000_000),
            },
            SearchResult {
                name: "mlx-model".into(),
                source: SearchSource::HuggingFace,
                format: ModelFormat::Mlx,
                downloads: 1_000,
                estimated_bytes: Some(4_000_000_000),
            },
        ];
        rank_results(&mut results, true, 16_000_000_000);
        assert_eq!(results[0].format, ModelFormat::Mlx);

        rank_results(&mut results, false, 16_000_000_000);
        assert_eq!(results[0].format, ModelFormat::Gguf);
    }

    #[test]
    fn rank_fits_before_too_large() {
        let mut results = vec![
            SearchResult {
                name: "big".into(),
                source: SearchSource::HuggingFace,
                format: ModelFormat::Mlx,
                downloads: 100_000,
                estimated_bytes: Some(20_000_000_000),
            },
            SearchResult {
                name: "small".into(),
                source: SearchSource::HuggingFace,
                format: ModelFormat::Mlx,
                downloads: 1_000,
                estimated_bytes: Some(4_000_000_000),
            },
        ];
        rank_results(&mut results, true, 8_000_000_000);
        assert_eq!(results[0].name, "small");
    }
}
