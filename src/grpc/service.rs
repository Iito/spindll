// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Iito <https://github.com/Iito> and sarmientoF <https://github.com/sarmientoF>

use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::engine::{EvictionPriority, GenerateParams, LoadOptions, ModelManager};
use crate::model_store::ModelStore;
use crate::proto::spindll_server::Spindll;
use crate::proto::*;

/// Tonic service implementation for the spindll gRPC protocol.
///
/// Bridges gRPC requests to the [`ModelManager`] for inference and
/// [`ModelStore`] for model resolution and pulling.
pub struct SpindllService {
    manager: Arc<ModelManager>,
    model_store: Arc<ModelStore>,
}

impl SpindllService {
    /// Create a new service backed by the given manager and model store.
    pub fn new(manager: Arc<ModelManager>, model_store: Arc<ModelStore>) -> Self {
        Self { manager, model_store }
    }
}

fn proto_params_to_engine(p: Option<crate::proto::GenerateParams>) -> GenerateParams {
    match p {
        Some(p) => GenerateParams {
            max_tokens:  p.max_tokens .map(|v| v as u32).unwrap_or(512),
            temperature: p.temperature.unwrap_or(0.8),
            top_p:       p.top_p      .unwrap_or(0.95),
            top_k:       p.top_k      .unwrap_or(40),
            seed:        p.seed       .map(|v| v as u32).unwrap_or(42),
            prefill_only: false,
        },
        None => GenerateParams::default(),
    }
}

fn send_usage(
    stats: crate::engine::GenerateResult,
    elapsed: f32,
) -> UsageStats {
    UsageStats {
        prompt_tokens: stats.prompt_tokens as i32,
        completion_tokens: stats.completion_tokens as i32,
        tokens_per_second: if elapsed > 0.0 {
            stats.completion_tokens as f32 / elapsed
        } else {
            0.0
        },
    }
}

/// Effective text of a proto `Message`. Per the proto contract, a non-empty
/// `parts` list replaces `content`, so flatten the text parts (ignoring images);
/// otherwise fall back to `content`. Without this, a text-only message sent via
/// `parts` (with empty `content`) would be silently dropped on the text path.
fn proto_message_text(m: &crate::proto::Message) -> String {
    if m.parts.is_empty() {
        m.content.clone()
    } else {
        m.parts
            .iter()
            .filter(|p| p.r#type != "image")
            .map(|p| p.text.as_str())
            .collect()
    }
}

/// A streaming (non-final) chat token frame.
fn token_resp(token: &str) -> ChatResponse {
    ChatResponse {
        token: token.to_string(),
        done: false,
        usage: None,
        tool_calls: Vec::new(),
        finish_reason: String::new(),
    }
}

/// Map a gRPC `tool_choice` string onto the shared [`ToolChoice`].
fn grpc_tool_choice(s: &str) -> crate::engine::tools::ToolChoice {
    use crate::engine::tools::ToolChoice;
    match s {
        "" | "auto" => ToolChoice::Auto,
        "none" => ToolChoice::None,
        "required" => ToolChoice::Required,
        other => ToolChoice::Named(other.to_string()),
    }
}

/// Convert proto `Message` list with `parts` into engine `MultimodalMessage` list.
///
/// Enforces the same per-image byte cap as the HTTP path so the gRPC entry
/// point can't be used to bypass the DoS guard.
#[cfg(feature = "vision")]
fn proto_to_multimodal(messages: &[crate::proto::Message]) -> anyhow::Result<Vec<crate::engine::MultimodalMessage>> {
    use crate::engine::multimodal::{check_image_len, ContentPart, MultimodalMessage};

    messages.iter().map(|m| {
        if m.parts.is_empty() {
            // Text-only message — wrap content as a single Text part.
            Ok(MultimodalMessage {
                role: m.role.clone(),
                content: vec![ContentPart::Text(m.content.clone())],
            })
        } else {
            let content = m.parts.iter().map(|p| {
                if p.r#type == "image" {
                    check_image_len(p.image_data.len())?;
                    Ok(ContentPart::ImageBytes {
                        data: p.image_data.clone(),
                        media_type: if p.media_type.is_empty() { None } else { Some(p.media_type.clone()) },
                    })
                } else {
                    Ok(ContentPart::Text(p.text.clone()))
                }
            }).collect::<anyhow::Result<Vec<_>>>()?;
            Ok(MultimodalMessage { role: m.role.clone(), content })
        }
    }).collect()
}

#[tonic::async_trait]
impl Spindll for SpindllService {
    type GenerateStream = ReceiverStream<Result<GenerateResponse, Status>>;
    type ChatStream = ReceiverStream<Result<ChatResponse, Status>>;
    type PullStream = ReceiverStream<Result<PullProgress, Status>>;

    #[tracing::instrument(skip_all, fields(model))]
    async fn generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let req = request.into_inner();
        tracing::Span::current().record("model", req.model.as_str());
        let mgr = self.manager.clone();
        let (tx, rx) = mpsc::channel(32);

        tokio::task::spawn_blocking(move || {
            let params = proto_params_to_engine(req.params);
            let start = std::time::Instant::now();

            let result = mgr.generate(&req.model, &req.prompt, &params, None, |token| {
                let resp = GenerateResponse {
                    token: token.to_string(),
                    done: false,
                    usage: None,
                };
                tx.blocking_send(Ok(resp)).is_ok()
            });

            match result {
                Err(e) => {
                    let _ = tx.blocking_send(Err(Status::internal(e.to_string())));
                }
                Ok(stats) => {
                    let _ = tx.blocking_send(Ok(GenerateResponse {
                        token: String::new(),
                        done: true,
                        usage: Some(send_usage(stats, start.elapsed().as_secs_f32())),
                    }));
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    #[tracing::instrument(skip_all, fields(model))]
    async fn chat(
        &self,
        request: Request<ChatRequest>,
    ) -> Result<Response<Self::ChatStream>, Status> {
        let req = request.into_inner();
        tracing::Span::current().record("model", req.model.as_str());
        let mgr = self.manager.clone();
        let store = self.model_store.clone();
        let (tx, rx) = mpsc::channel(32);

        tokio::task::spawn_blocking(move || {
            // Auto-load the model if it isn't already in the manager.
            if !mgr.is_loaded(&req.model) {
                let path = match store.resolve_model_path(&req.model) {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(Status::not_found(
                            format!("model '{}' not found in store: {e}", req.model)
                        )));
                        return;
                    }
                };
                let digest = store.resolve_model_digest(&req.model).unwrap_or_default();
                // mmproj_path on autoload → first image req has vision.
                #[cfg(feature = "vision")]
                let mmproj_path = store.resolve_mmproj_path(&req.model).ok().flatten();
                #[cfg(feature = "vision")]
                let opts = crate::engine::manager::LoadOptions {
                    digest,
                    mmproj_path,
                    ..Default::default()
                };
                #[cfg(not(feature = "vision"))]
                let opts = crate::engine::manager::LoadOptions {
                    digest,
                    ..Default::default()
                };
                if let Err(e) = mgr.load_model_with_options(&req.model, &path, opts) {
                    let _ = tx.blocking_send(Err(Status::internal(
                        format!("failed to load model '{}': {e}", req.model)
                    )));
                    return;
                }
            }

            let params = proto_params_to_engine(req.params);
            let start = std::time::Instant::now();

            // Tool calling (prompt injection — mirrors the HTTP `/v1` surface; no
            // model-native grammar on llama-cpp-2 0.1.150). When active, the text
            // path buffers output so calls can be parsed from the full response.
            let tool_specs: Vec<crate::engine::tools::ToolSpec> = req
                .tools
                .iter()
                .map(|t| crate::engine::tools::ToolSpec {
                    name: t.name.clone(),
                    description: (!t.description.is_empty()).then(|| t.description.clone()),
                    parameters: serde_json::from_str(&t.parameters_json).ok(),
                })
                .collect();
            let tool_choice = grpc_tool_choice(&req.tool_choice);
            let has_tools = !tool_specs.is_empty()
                && !matches!(tool_choice, crate::engine::tools::ToolChoice::None);
            let mut output = String::new();

            // Text messages (role, content); tool preamble merged into the system
            // turn when active.
            let mut messages: Vec<(String, String)> = req
                .messages
                .iter()
                .map(|m| (m.role.clone(), proto_message_text(m)))
                .collect();
            // Rendered once; reused for the text turn here and the vision path below.
            let preamble = crate::engine::tools::tools_to_prompt(&tool_specs, &tool_choice);
            if let Some(ref preamble) = preamble {
                match messages.iter_mut().find(|(r, _)| r == "system") {
                    Some(sys) => sys.1 = format!("{}\n\n{}", sys.1, preamble),
                    None => messages.insert(0, ("system".to_string(), preamble.clone())),
                }
            }
            let enc_key: Option<[u8; 32]> = (req.encryption_key.len() == 32).then(|| {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&req.encryption_key);
                arr
            });

            // Vision path only if ≥1 image part. Text-only `parts` stay on text path.
            #[cfg(feature = "vision")]
            let has_image = req
                .messages
                .iter()
                .any(|m| m.parts.iter().any(|p| p.r#type == "image"));

            #[cfg(feature = "vision")]
            let result = if has_image {
                let mut mm_messages = match proto_to_multimodal(&req.messages) {
                    Ok(m) => m,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(Status::invalid_argument(e.to_string())));
                        return;
                    }
                };
                // Inject the tool preamble so vision + tools works like the text path,
                // and buffer output when tools are active so calls can be parsed.
                if let Some(ref preamble) = preamble {
                    crate::engine::multimodal::inject_system_text(&mut mm_messages, preamble);
                }
                mgr.generate_chat_multimodal(&req.model, &mm_messages, &params, |token| {
                    if has_tools {
                        output.push_str(token);
                        return true;
                    }
                    tx.blocking_send(Ok(token_resp(token))).is_ok()
                })
            } else {
                mgr.generate_chat(&req.model, &messages, &params, enc_key.as_ref(), |token| {
                    if has_tools {
                        output.push_str(token);
                        return true;
                    }
                    tx.blocking_send(Ok(token_resp(token))).is_ok()
                })
            };

            #[cfg(not(feature = "vision"))]
            let result =
                mgr.generate_chat(&req.model, &messages, &params, enc_key.as_ref(), |token| {
                    if has_tools {
                        output.push_str(token);
                        return true;
                    }
                    tx.blocking_send(Ok(token_resp(token))).is_ok()
                });

            match result {
                Err(e) => {
                    let _ = tx.blocking_send(Err(Status::internal(e.to_string())));
                }
                Ok(stats) => {
                    // When tools are active, parse calls from the buffered output and
                    // emit them (plus any prose) on the final frame.
                    let (tool_calls, finish_reason) = if has_tools {
                        let (calls, remaining) = crate::engine::tools::parse_tool_calls(&output);
                        if !remaining.is_empty() {
                            let _ = tx.blocking_send(Ok(token_resp(&remaining)));
                        }
                        let finish = if calls.is_empty() { "stop" } else { "tool_calls" };
                        let proto_calls = calls
                            .into_iter()
                            .map(|c| crate::proto::ToolCall {
                                id: c.id,
                                name: c.name,
                                arguments: c.arguments,
                            })
                            .collect();
                        (proto_calls, finish.to_string())
                    } else {
                        (Vec::new(), "stop".to_string())
                    };
                    let _ = tx.blocking_send(Ok(ChatResponse {
                        token: String::new(),
                        done: true,
                        usage: Some(send_usage(stats, start.elapsed().as_secs_f32())),
                        tool_calls,
                        finish_reason,
                    }));
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn pull(
        &self,
        request: Request<PullRequest>,
    ) -> Result<Response<Self::PullStream>, Status> {
        let req = request.into_inner();
        let store = self.model_store.clone();
        let (tx, rx) = mpsc::channel(4);

        tokio::task::spawn_blocking(move || {
            let quant = if req.quantization.is_empty() { None } else { Some(req.quantization.as_str()) };

            // Signal that the pull has started.
            let _ = tx.blocking_send(Ok(PullProgress {
                file: req.repo.clone(),
                downloaded: 0,
                total: 0,
                done: false,
            }));

            match store.pull(&req.repo, quant, crate::model_store::FormatPreference::Auto) {
                Ok(path) => {
                    let filename = path.file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    let _ = tx.blocking_send(Ok(PullProgress {
                        file: filename,
                        downloaded: size,
                        total: size,
                        done: true,
                    }));
                }
                Err(e) => {
                    let _ = tx.blocking_send(Err(Status::internal(e.to_string())));
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn list(
        &self,
        _request: Request<ListRequest>,
    ) -> Result<Response<ListResponse>, Status> {
        let mut reg = crate::model_store::registry::Registry::load(&self.model_store.registry_path())
            .map_err(|e| Status::internal(e.to_string()))?;
        if reg.backfill_metadata() {
            let _ = reg.save(&self.model_store.registry_path());
        }

        let models = reg
            .models
            .iter()
            .map(|(key, entry)| {
                let format = match entry.format {
                    crate::model_store::registry::ModelFormat::Gguf => "gguf",
                    crate::model_store::registry::ModelFormat::Mlx => "mlx",
                };
                ModelInfo {
                    name: key.clone(),
                    repo: entry.repo.clone(),
                    file: entry.filename.clone(),
                    quantization: String::new(),
                    size_bytes: entry.size_bytes,
                    last_used: String::new(),
                    digest: entry.digest.clone(),
                    model_name: entry.model_name.clone(),
                    description: entry.description.clone(),
                    architecture: entry.architecture.clone(),
                    context_length: entry.context_length,
                    format: format.to_string(),
                    base_model: entry.base_model.clone(),
                    display_name: crate::model_store::display_name(key, entry),
                }
            })
            .collect();

        let prefer_format = if crate::model_store::platform_prefers_mlx() {
            "mlx"
        } else {
            "gguf"
        };

        Ok(Response::new(ListResponse {
            models,
            prefer_format: prefer_format.to_string(),
        }))
    }

    #[tracing::instrument(skip_all, fields(model))]
    async fn load(
        &self,
        request: Request<LoadRequest>,
    ) -> Result<Response<LoadResponse>, Status> {
        let req = request.into_inner();
        tracing::Span::current().record("model", req.model.as_str());

        if self.manager.is_loaded(&req.model) {
            return Ok(Response::new(LoadResponse {
                success: true,
                message: format!("{} already loaded", req.model),
                already_loaded: true,
            }));
        }

        let model_path = self.model_store
            .resolve_model_path(&req.model)
            .map_err(|e| Status::not_found(e.to_string()))?;
        let digest = self.model_store
            .resolve_model_digest(&req.model)
            .unwrap_or_default();

        let gpu_layers = if req.gpu_layers < 0 { None } else { Some(req.gpu_layers as u32) };

        let priority = match crate::proto::EvictionPriority::try_from(req.priority) {
            Ok(crate::proto::EvictionPriority::PriorityLow) => EvictionPriority::Low,
            Ok(crate::proto::EvictionPriority::PriorityHigh) => EvictionPriority::High,
            _ => EvictionPriority::Normal,
        };
        let idle_reload = if req.idle_reload_secs == 0 {
            None
        } else {
            Some(std::time::Duration::from_secs(req.idle_reload_secs as u64))
        };

        self.manager
            .load_model_with_options(
                &req.model,
                &model_path,
                LoadOptions {
                    gpu_layers,
                    digest,
                    priority,
                    idle_reload,
                    #[cfg(feature = "vision")]
                    mmproj_path: self.model_store.resolve_mmproj_path(&req.model).unwrap_or(None),
                },
            )
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(LoadResponse {
            success: true,
            message: format!("loaded {}", req.model),
            already_loaded: false,
        }))
    }

    #[tracing::instrument(skip_all, fields(model))]
    async fn unload(
        &self,
        request: Request<UnloadRequest>,
    ) -> Result<Response<UnloadResponse>, Status> {
        let req = request.into_inner();
        tracing::Span::current().record("model", req.model.as_str());
        self.manager
            .unload_model(&req.model)
            .map_err(|e| Status::not_found(e.to_string()))?;

        Ok(Response::new(UnloadResponse { success: true }))
    }

    #[tracing::instrument(skip_all, fields(model))]
    async fn prefill(
        &self,
        request: Request<PrefillRequest>,
    ) -> Result<Response<PrefillResponse>, Status> {
        let req = request.into_inner();
        tracing::Span::current().record("model", req.model.as_str());
        let mgr = self.manager.clone();
        let store = self.model_store.clone();

        // The closure returns Result<_, tonic::Status>; Status is large by design.
        #[allow(clippy::result_large_err)]
        let result = tokio::task::spawn_blocking(move || {
            // Auto-load the model if not already loaded.
            if !mgr.is_loaded(&req.model) {
                let path = store
                    .resolve_model_path(&req.model)
                    .map_err(|e| Status::not_found(format!("model '{}' not found in store: {e}", req.model)))?;
                let digest = store.resolve_model_digest(&req.model).unwrap_or_default();
                mgr.load_model_with_digest(&req.model, &path, None, digest)
                    .map_err(|e| Status::internal(format!("failed to load model '{}': {e}", req.model)))?;
            }

            let messages: Vec<_> = req.messages.iter()
                .map(|m| (m.role.clone(), m.content.clone()))
                .collect();
            let enc_key: Option<[u8; 32]> = if req.encryption_key.len() == 32 {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&req.encryption_key);
                Some(arr)
            } else {
                None
            };

            let params = GenerateParams {
                prefill_only: true,
                ..GenerateParams::default()
            };

            let stats = mgr.generate_chat(&req.model, &messages, &params, enc_key.as_ref(), |_| true)
                .map_err(|e| Status::internal(e.to_string()))?;

            Ok::<_, Status>(PrefillResponse {
                tokens_cached: stats.prompt_tokens,
            })
        })
        .await
        .map_err(|e| Status::internal(format!("task join error: {e}")))?;

        result.map(Response::new)
    }

    async fn delete(
        &self,
        _request: Request<DeleteRequest>,
    ) -> Result<Response<DeleteResponse>, Status> {
        Err(Status::unimplemented("delete not yet implemented"))
    }

    async fn status(
        &self,
        _request: Request<StatusRequest>,
    ) -> Result<Response<StatusResponse>, Status> {
        let mem = crate::scheduler::budget::MemoryBudget::detect(None);

        let models = self.manager.loaded_models().iter()
            .map(|(name, size, layers, digest, n_ctx, _)| LoadedModel {
                name: name.clone(),
                memory_used: *size,
                gpu_layers: *layers as i32,
                digest: digest.clone(),
                context_length: *n_ctx,
            })
            .collect();

        let devices = {
            let mut d = vec!["CPU".to_string()];
            if cfg!(target_os = "macos") || cfg!(feature = "metal") {
                d.insert(0, "Metal".to_string());
            }
            if cfg!(feature = "cuda") {
                d.insert(0, "CUDA".to_string());
            }
            if cfg!(feature = "vulkan") {
                d.insert(0, "Vulkan".to_string());
            }
            d
        };

        Ok(Response::new(StatusResponse {
            models,
            memory: Some(MemoryInfo {
                total_ram: mem.total_ram,
                used_ram: mem.total_ram.saturating_sub(mem.available_ram),
                available_ram: mem.available_ram,
                total_vram: 0,
                used_vram: 0,
                available_vram: 0,
            }),
            devices,
            metrics: {
                let snap = self.manager.metrics().snapshot();
                Some(EngineMetrics {
                    cache_hits: snap.cache_hits,
                    cache_misses: snap.cache_misses,
                    cache_hit_rate: snap.cache_hit_rate(),
                    total_prompt_tokens: snap.total_prompt_tokens,
                    total_completion_tokens: snap.total_completion_tokens,
                    avg_tokens_per_second: snap.avg_tokens_per_second(),
                    generate_requests: snap.generate_requests,
                    generate_errors: snap.generate_errors,
                })
            },
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendLoadParams, BackendModel, InferenceBackend};
    use crate::engine::streaming::{GenerateParams as EngineParams, GenerateResult};
    use crate::model_store::registry::{ModelEntry, ModelFormat, ModelSource, Registry};

    struct FakeBackend;
    impl InferenceBackend for FakeBackend {
        fn load_model(&self, _: &std::path::Path, _: BackendLoadParams) -> anyhow::Result<Box<dyn BackendModel>> {
            Ok(Box::new(FakeModel))
        }
        fn name(&self) -> &str { "llamacpp" }
    }
    struct FakeModel;
    impl BackendModel for FakeModel {
        fn generate(&self, _: &str, _: &EngineParams, _: &mut dyn FnMut(&str) -> bool) -> anyhow::Result<GenerateResult> { Ok(GenerateResult::default()) }
        fn apply_chat_template(&self, _: &[(String, String)]) -> anyhow::Result<String> { Ok(String::new()) }
        fn n_ctx(&self) -> u32 { 2048 }
        fn size_bytes(&self) -> u64 { 100 }
        fn kv_bytes_per_token(&self) -> u64 { 1 }
        fn as_any(&self) -> &dyn std::any::Any { self }
    }

    #[tokio::test]
    async fn list_response_populates_format_base_model_display_name() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(Some(dir.path().to_path_buf()));
        std::fs::create_dir_all(store.models_dir()).unwrap();

        let mut reg = Registry::load(&store.registry_path()).unwrap();
        reg.add("TheBloke/Llama-GGUF/llama-q4_k_m.gguf".into(), ModelEntry {
            repo: "TheBloke/Llama-GGUF".into(),
            filename: "llama-q4_k_m.gguf".into(),
            path: "/tmp/nonexistent".into(),
            size_bytes: 4_000_000,
            downloaded_at: 1,
            digest: "sha256:abc".into(),
            model_name: "Llama".into(),
            description: String::new(),
            architecture: "llama".into(),
            context_length: 4096,
            metadata_read: true,
            format: ModelFormat::Gguf,
            base_model: String::new(),
            source: ModelSource::HfSourceDownloaded,
            mmproj_path: None,
        });
        reg.add("mlx-community/Llama-3.1-8B-4bit".into(), ModelEntry {
            repo: "mlx-community/Llama-3.1-8B-4bit".into(),
            filename: String::new(),
            path: "/tmp/nonexistent".into(),
            size_bytes: 4_200_000,
            downloaded_at: 2,
            digest: "sha256:def".into(),
            model_name: String::new(),
            description: String::new(),
            architecture: String::new(),
            context_length: 0,
            metadata_read: true,
            format: ModelFormat::Mlx,
            base_model: "llama3.1-8b".into(),
            source: ModelSource::HfSourceDownloaded,
            mmproj_path: None,
        });
        reg.save(&store.registry_path()).unwrap();

        let mgr = Arc::new(ModelManager::with_backends(vec![Box::new(FakeBackend)], 0));
        let svc = SpindllService::new(mgr, Arc::new(store));

        let resp = svc.list(Request::new(ListRequest {})).await.unwrap().into_inner();

        assert_eq!(resp.models.len(), 2);

        let gguf = resp.models.iter().find(|m| m.format == "gguf").unwrap();
        assert_eq!(gguf.display_name, "TheBloke/Llama-GGUF (q4_k_m)");
        assert!(gguf.base_model.is_empty());

        let mlx = resp.models.iter().find(|m| m.format == "mlx").unwrap();
        assert_eq!(mlx.display_name, "mlx-community/Llama-3.1-8B-4bit");
        assert_eq!(mlx.base_model, "llama3.1-8b");

        assert!(!resp.prefer_format.is_empty());
    }
}

#[cfg(all(test, feature = "vision"))]
mod proto_to_multimodal_tests {
    use super::*;
    use crate::engine::multimodal::{ContentPart as MmPart, MAX_IMAGE_BYTES};

    fn image_msg(image_data: Vec<u8>) -> crate::proto::Message {
        crate::proto::Message {
            role: "user".into(),
            content: String::new(),
            parts: vec![crate::proto::ContentPart {
                r#type: "image".into(),
                text: String::new(),
                image_data,
                media_type: "image/png".into(),
            }],
        }
    }

    #[test]
    fn maps_image_part_within_cap() {
        let out = proto_to_multimodal(&[image_msg(vec![1, 2, 3])]).unwrap();
        match &out[0].content[..] {
            [MmPart::ImageBytes { data, media_type }] => {
                assert_eq!(data.as_slice(), &[1, 2, 3]);
                assert_eq!(media_type.as_deref(), Some("image/png"));
            }
            other => panic!("expected image part, got {other:?}"),
        }
    }

    #[test]
    fn rejects_oversized_image() {
        let err = proto_to_multimodal(&[image_msg(vec![0u8; MAX_IMAGE_BYTES + 1])]).unwrap_err();
        assert!(err.to_string().contains("exceeds"), "got: {err}");
    }
}

#[cfg(test)]
mod proto_message_text_tests {
    use super::*;

    fn text_part(text: &str) -> crate::proto::ContentPart {
        crate::proto::ContentPart {
            r#type: "text".into(),
            text: text.into(),
            image_data: Vec::new(),
            media_type: String::new(),
        }
    }

    #[test]
    fn falls_back_to_content_when_no_parts() {
        let m = crate::proto::Message {
            role: "user".into(),
            content: "hello".into(),
            parts: vec![],
        };
        assert_eq!(proto_message_text(&m), "hello");
    }

    #[test]
    fn flattens_text_parts_when_content_empty() {
        // Regression: text carried in `parts` with empty `content` must not be dropped.
        let m = crate::proto::Message {
            role: "user".into(),
            content: String::new(),
            parts: vec![text_part("foo "), text_part("bar")],
        };
        assert_eq!(proto_message_text(&m), "foo bar");
    }

    #[test]
    fn parts_override_content_and_skip_images() {
        let m = crate::proto::Message {
            role: "user".into(),
            content: "ignored".into(),
            parts: vec![
                text_part("see "),
                crate::proto::ContentPart {
                    r#type: "image".into(),
                    text: "alt".into(),
                    image_data: vec![1, 2, 3],
                    media_type: "image/png".into(),
                },
                text_part("this"),
            ],
        };
        assert_eq!(proto_message_text(&m), "see this");
    }
}
