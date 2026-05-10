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
    #[serde(default)]
    safetensors: Option<SafetensorsInfo>,
}

#[derive(Deserialize)]
struct SafetensorsInfo {
    #[serde(default)]
    total: u64,
}

#[derive(Deserialize)]
struct HfRepoSibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
}

#[derive(Deserialize)]
struct HfRepoInfo {
    #[serde(default)]
    siblings: Vec<HfRepoSibling>,
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

    backfill_sizes(&client, &mut results);

    let prefers_mlx = super::platform_prefers_mlx();
    let mem = crate::scheduler::budget::MemoryBudget::detect(None);
    let inference_mem = detect_inference_memory(mem.total_ram);
    rank_results(&mut results, prefers_mlx, inference_mem);

    results.truncate(limit);
    Ok(results)
}

fn search_hf_gguf(
    client: &reqwest::blocking::Client,
    query: &str,
) -> anyhow::Result<Vec<SearchResult>> {
    let mut seen = std::collections::HashSet::new();
    let mut results = Vec::new();

    let urls = [
        format!(
            "https://huggingface.co/api/models?search={}&filter=gguf&sort=downloads&direction=-1&limit=20",
            urlencoding::encode(query),
        ),
        format!(
            "https://huggingface.co/api/models?search={}+GGUF&sort=downloads&direction=-1&limit=20",
            urlencoding::encode(query),
        ),
    ];

    for url in &urls {
        let resp = match client.get(url).send() {
            Ok(r) if r.status().is_success() => r,
            _ => continue,
        };
        let models: Vec<HfModel> = match resp.json() {
            Ok(m) => m,
            Err(_) => continue,
        };
        for m in models {
            if m.id.starts_with("mlx-community/") || !seen.insert(m.id.clone()) {
                continue;
            }
            let estimated = estimate_from_api(&m, &ModelFormat::Gguf);
            results.push(SearchResult {
                name: m.id,
                source: SearchSource::HuggingFace,
                format: ModelFormat::Gguf,
                downloads: m.downloads,
                estimated_bytes: estimated,
            });
        }
    }

    Ok(results)
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
            let estimated = estimate_from_api(&m, &ModelFormat::Mlx);
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

/// Returns the memory that constrains inference: VRAM on machines with a
/// dedicated GPU, total system RAM on unified-memory platforms (Apple Silicon).
pub fn detect_inference_memory(total_ram: u64) -> u64 {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return total_ram;
    }
    detect_vram_nvidia().unwrap_or(total_ram)
}

fn detect_vram_nvidia() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = stdout.lines().next()?.trim();
    let mib: u64 = first_line.parse().ok()?;
    Some(mib * 1_048_576)
}

fn estimate_from_api(m: &HfModel, format: &ModelFormat) -> Option<u64> {
    if let Some(ref st) = m.safetensors {
        if st.total > 0 {
            let params_b = st.total as f64 / 1e9;
            let bpp = match format {
                ModelFormat::Gguf => 0.55,
                ModelFormat::Mlx => {
                    let lower = m.id.to_lowercase();
                    if lower.contains("8bit") || lower.contains("8-bit") {
                        1.1
                    } else if lower.contains("bf16") || lower.contains("fp16") {
                        2.0
                    } else {
                        0.55
                    }
                }
            };
            return Some((params_b * bpp * 1e9) as u64);
        }
    }
    estimate_model_bytes(&m.id, format)
}

fn backfill_sizes(client: &reqwest::blocking::Client, results: &mut [SearchResult]) {
    for r in results.iter_mut() {
        if r.estimated_bytes.is_some() || r.source != SearchSource::HuggingFace {
            continue;
        }
        if let Some(size) = fetch_repo_size(client, &r.name, &r.format) {
            r.estimated_bytes = Some(size);
        }
    }
}

fn fetch_repo_size(
    client: &reqwest::blocking::Client,
    repo_id: &str,
    format: &ModelFormat,
) -> Option<u64> {
    let url = format!("https://huggingface.co/api/models/{repo_id}?blobs=true");
    let resp = client.get(&url).send().ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let info: HfRepoInfo = resp.json().ok()?;

    match format {
        ModelFormat::Gguf => {
            let mut groups: std::collections::HashMap<String, u64> =
                std::collections::HashMap::new();
            for s in &info.siblings {
                if !s.rfilename.ends_with(".gguf") {
                    continue;
                }
                let key = match s.rfilename.find('/') {
                    Some(i) => s.rfilename[..i].to_string(),
                    None => s.rfilename.clone(),
                };
                *groups.entry(key).or_default() += s.size.unwrap_or(0);
            }
            let (_key, total) = groups
                .iter()
                .filter(|(_, sz)| **sz > 0)
                .min_by_key(|(k, _)| crate::model_store::download::rank_quant(k))?;
            Some(*total)
        }
        ModelFormat::Mlx => {
            let total: u64 = info
                .siblings
                .iter()
                .filter(|s| {
                    s.rfilename.ends_with(".safetensors")
                        || s.rfilename.ends_with(".json")
                        || s.rfilename.ends_with(".model")
                })
                .filter_map(|s| s.size)
                .sum();
            if total > 0 { Some(total) } else { None }
        }
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
    fn rank_by_vram_on_dedicated_gpu() {
        // Simulate a machine with 64 GB system RAM but only 8 GB VRAM.
        // A 12 GB model won't fit in VRAM, so a smaller 4 GB model with
        // fewer downloads should rank higher.
        let vram: u64 = 8_000_000_000;
        let mut results = vec![
            SearchResult {
                name: "Popular-14B-GGUF".into(),
                source: SearchSource::HuggingFace,
                format: ModelFormat::Gguf,
                downloads: 500_000,
                estimated_bytes: Some(12_000_000_000),
            },
            SearchResult {
                name: "Niche-7B-GGUF".into(),
                source: SearchSource::HuggingFace,
                format: ModelFormat::Gguf,
                downloads: 5_000,
                estimated_bytes: Some(4_000_000_000),
            },
        ];
        rank_results(&mut results, false, vram);
        assert_eq!(
            results[0].name, "Niche-7B-GGUF",
            "model that fits in VRAM should rank above one that doesn't, regardless of downloads",
        );
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
