# Smart Format Selection on Pull

When a user pulls a model by name (e.g., `spindll pull llama3.1:8b`), download
the optimal format for their platform — MLX on Apple Silicon, GGUF everywhere else.
One model on disk, best backend picked automatically.

Depends on: MLX Swift bridge working end-to-end, backend trait abstraction (issue #1)

---

## The problem

Today `spindll pull llama3.1:8b` always downloads GGUF. On Apple Silicon, the same
model runs faster through MLX with unified memory — but the user would need to
know the exact `mlx-community/` repo name and pull it manually.

The fix: make `pull` platform-aware. The user thinks in model names, not formats.

---

## Desired UX

### Apple Silicon (mlx feature enabled)

```
$ spindll pull llama3.1:8b

Resolving llama3.1:8b for Apple Silicon...
Found: mlx-community/Meta-Llama-3.1-8B-Instruct-4bit (4.5 GB)
  ████████████████████████████████ 4.5 GB  done

Model ready (MLX, Metal-native).
```

One download. The user runs `spindll run llama3.1:8b` and it goes through MLX.

### Apple Silicon — no MLX equivalent exists

```
$ spindll pull obscure-model:7b

Resolving obscure-model:7b for Apple Silicon...
No MLX version found. Falling back to GGUF.
Pulling from Ollama registry...
  ████████████████████████████████ 4.1 GB  done

Model ready (GGUF, llama.cpp).
```

Graceful fallback. Still works, just through llama.cpp.

### Linux / non-Apple

```
$ spindll pull llama3.1:8b

Pulling llama3.1:8b from Ollama registry...
  ████████████████████████████████ 4.7 GB  done

Model ready (GGUF).
```

No change from today. MLX search never runs.

### Explicit format override

```
$ spindll pull --gguf llama3.1:8b    # force GGUF even on Apple Silicon
$ spindll pull --mlx  llama3.1:8b    # force MLX (error if not found)
```

---

## How MLX Community repos are named

The `mlx-community` org on HuggingFace mirrors popular models in MLX safetensors
format. Naming pattern:

| Original | MLX repo |
|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` |
| `mistralai/Mistral-7B-Instruct-v0.3` | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` |
| `google/gemma-2-9b-it` | `mlx-community/gemma-2-9b-it-4bit` |
| `Qwen/Qwen2.5-7B-Instruct` | `mlx-community/Qwen2.5-7B-Instruct-4bit` |

Pattern: `mlx-community/{model-name}[-{quant}bit]`

Quant suffixes: `-4bit`, `-8bit`, `-bf16`, or none (fp16). Default preference: `4bit`.

---

## Resolution flow

`pull()` gains an early resolution phase before any download begins.

### Phase 1 — Determine preferred format

```rust
fn preferred_format() -> ModelFormat {
    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    return ModelFormat::Mlx;

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "mlx")))]
    return ModelFormat::Gguf;
}
```

Overridden by `--gguf` or `--mlx` flags.

### Phase 2 — Resolve model name to a downloadable source

If preferred format is MLX, try to find the MLX repo before touching the network
for a GGUF download:

```
Input: "llama3.1:8b", preferred: MLX

1. Check hardcoded Ollama→HF map        → "Meta-Llama-3.1-8B-Instruct"
2. Search HF: mlx-community/{name}-4bit → found
3. Validate: fetch config.json, check model_type
4. Download MLX repo
```

If preferred format is GGUF, or MLX resolution fails at any step, use the
existing Ollama/HF GGUF download path unchanged.

```
Input: "llama3.1:8b", preferred: MLX

1. Check hardcoded map                  → "Meta-Llama-3.1-8B-Instruct"
2. Search HF                            → no results
3. Fallback: pull GGUF from Ollama      → success
```

### Phase 3 — Download and register

Unchanged from today. The selected source (HF MLX repo or Ollama GGUF) goes
through the existing `download_hf_auto()` or `ollama_pull::pull_from_registry()`
paths. `ModelEntry` is registered with the correct `ModelFormat`.

---

## MLX repo discovery

### New module: `src/model_store/mlx_resolve.rs`

```rust
pub struct MlxCandidate {
    pub repo_id: String,       // "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
    pub size_estimate: u64,
    pub downloads: u64,
}

/// Try to find an MLX repo for the given model name.
/// Returns None if no suitable candidate is found.
pub fn find_mlx_repo(
    model_name: &str,
    quant: &str,           // "4bit" default
) -> anyhow::Result<Option<MlxCandidate>>
```

### Strategy 1 — Direct repo probe (fast path)

Many MLX repos follow a predictable naming convention. Try a direct
`hf_hub` repo lookup before falling back to search:

```rust
// For Ollama names, use the hardcoded map
let base = ollama_to_hf_base("llama3.1", "8b")?;  // → "Meta-Llama-3.1-8B-Instruct"

// Try the obvious repo name first
let candidate = format!("mlx-community/{base}-{quant}");
if repo_exists(&candidate) {
    return Ok(Some(candidate));
}
```

This avoids the search API entirely for well-known models and is a single
HTTP HEAD or repo info call.

### Strategy 2 — HF model search (fallback)

If the direct probe misses:

```
GET https://huggingface.co/api/models?author=mlx-community&search={base_name}&limit=20
```

From results, rank by:

1. **Name match** — candidate repo name (minus `mlx-community/` and quant suffix)
   must be a close match to `base_name`. Use case-insensitive substring matching.
2. **Quant preference** — prefer the requested quant (`4bit` default)
3. **Downloads** — break ties with popularity

### Strategy 3 — Give up

If search returns nothing or all candidates fail validation, return `None`.
The caller falls back to GGUF.

### Validation

Before accepting any candidate, fetch its `config.json` (~1KB) and verify:
- `model_type` field exists (confirms it's an LLM, not a vision model etc.)
- For Ollama sources where we know the architecture, check it matches

---

## Base model name extraction

The core challenge: mapping user-facing model names to HF repo names.

### Ollama names → HF base names

Hardcoded map for popular models (fast, reliable):

```rust
fn ollama_to_hf_base(name: &str, tag: &str) -> Option<&'static str> {
    match (name, tag) {
        ("llama3.1", "8b")    => Some("Meta-Llama-3.1-8B-Instruct"),
        ("llama3.1", "70b")   => Some("Meta-Llama-3.1-70B-Instruct"),
        ("llama3.2", "1b")    => Some("Llama-3.2-1B-Instruct"),
        ("llama3.2", "3b")    => Some("Llama-3.2-3B-Instruct"),
        ("mistral", "7b")     => Some("Mistral-7B-Instruct-v0.3"),
        ("gemma2", "9b")      => Some("gemma-2-9b-it"),
        ("gemma2", "27b")     => Some("gemma-2-27b-it"),
        ("qwen2.5", "7b")     => Some("Qwen2.5-7B-Instruct"),
        ("qwen2.5", "14b")    => Some("Qwen2.5-14B-Instruct"),
        ("phi3", "3.8b")      => Some("Phi-3-mini-4k-instruct"),
        ("deepseek-r1", _)    => Some("DeepSeek-R1-Distill-Llama-8B"),
        _ => None,
    }
}
```

When the map misses, fall back to the HF search API with the Ollama name
as a search query. The search is fuzzy enough to handle minor mismatches.

### HuggingFace GGUF repos → HF base names

If the user pulls from a HF GGUF repo directly (e.g.,
`bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`):

1. Strip known GGUF suffixes: `-GGUF`, `-gguf`, `-quantized`
2. Strip the org prefix: `bartowski/` → `Meta-Llama-3.1-8B-Instruct`
3. Use that as the search query for `mlx-community`

---

## Registry changes

### New field on `ModelEntry`

```rust
pub struct ModelEntry {
    // ... existing fields ...
    #[serde(default)]
    pub base_model: String,  // canonical identity, e.g., "Meta-Llama-3.1-8B-Instruct"
}
```

Populated at pull time. Used for:
- Display grouping in `spindll list`
- Future re-resolution if user wants to switch formats later

No `companion_key` needed — there's only one entry per model now.

### Name resolution changes

`resolve_key()` needs to handle the case where the user asks for `llama3.1:8b`
but the registry has it stored under an MLX key:

```
"mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
```

Add a resolution path that matches by `base_model`:

```rust
// Existing: exact key, ollama name:tag, ollama bare name, HF prefix
// New: match by base_model field
if let Some((key, _)) = registry.models.iter().find(|(_, e)| {
    e.base_model.eq_ignore_ascii_case(normalized_name)
}) {
    return Ok(key.clone());
}
```

This means `spindll run llama3.1:8b` works regardless of whether the
on-disk model is GGUF or MLX. The user doesn't need to know.

---

## Display

### `spindll list`

Add a format column:

```
MODEL                              FMT    SIZE       ARCH
──────────────────────────────────────────────────────────
llama3.1:8b                        mlx    4.5 GB     llama
nemotron-3-nano:4b                 gguf   2.1 GB     llama
gemma2:9b                          mlx    5.2 GB     gemma2
custom-finetune:latest             gguf   3.8 GB     llama
```

The format column tells the user what they got. No ambiguity.

### `spindll pull` output

Show the resolution decision:

```
$ spindll pull llama3.1:8b
Resolving llama3.1:8b → mlx-community/Meta-Llama-3.1-8B-Instruct-4bit (MLX, 4bit)
  ████████████████████████████████ 4.5 GB  done
```

Or on fallback:

```
$ spindll pull obscure-model:7b
Resolving obscure-model:7b → no MLX found, using GGUF
Pulling from Ollama registry...
  ████████████████████████████████ 4.1 GB  done
```

---

## CLI surface

Flags on `pull`:

| Flag | Effect |
|---|---|
| `--gguf` | Force GGUF download, skip MLX resolution |
| `--mlx` | Force MLX download, error if not found (no GGUF fallback) |
| `--mlx-quant <4bit\|8bit\|fp16>` | Override default 4bit quant for MLX |

No new subcommands needed.

---

## Edge cases

**No MLX equivalent exists** — Fall back to GGUF silently (with one log line).
The user still gets a working model.

**User already has GGUF, wants MLX** — `spindll pull --mlx llama3.1:8b` resolves
the MLX repo, downloads it, and **replaces** the registry entry. The old GGUF
file stays on disk (user can clean up manually or we `rm` it). The model name
still resolves to `llama3.1:8b`.

**User on Apple Silicon wants GGUF** — `spindll pull --gguf llama3.1:8b` skips
MLX resolution entirely. Useful for benchmarking or when MLX has a bug.

**HF API unreachable** — Fall back to GGUF. The MLX resolution is best-effort
and must never block the pull.

**Large models (70B+)** — MLX 4bit versions of 70B models are ~40GB. The size
is shown before download starts. No special handling needed.

**HF repo is private / gated** — `hf_hub` handles auth via `HF_TOKEN`. If the
MLX repo requires acceptance (like Llama), the error message should say so.
Fall back to GGUF if auth fails.

**Ollama import (`spindll import`)** — The import path always brings in GGUF
from the local Ollama store. No change here — the user already has the file
locally. They can `spindll pull --mlx <name>` afterward if they want MLX.

---

## Implementation order

1. **`base_model` field** — Add to `ModelEntry`, populate during `pull()` from
   GGUF metadata / HF repo name. Backfill existing entries in `backfill_metadata()`.

2. **`mlx_resolve.rs`** — Implement the hardcoded map + HF search + validation.
   Test against 5-10 known models to tune the matching.

3. **Wire into `pull()`** — Add the resolution phase before download. Preferred
   format selection based on platform + feature flags. Fallback path.

4. **Name resolution update** — `resolve_key()` matches by `base_model` so
   `llama3.1:8b` finds the model regardless of on-disk format.

5. **`--gguf` / `--mlx` flags** — Override the platform default.

6. **Display updates** — Format column in `list`, resolution trace in `pull`.

Steps 1-3 are the core. Steps 4-6 are fast follow-ups in the same PR or next.
