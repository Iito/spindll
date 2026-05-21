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
            draft_model_name: None,
            n_draft: 0,
            n_gram_draft: 0,
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
                if let Err(e) = mgr.load_model_with_digest(&req.model, &path, None, digest) {
                    let _ = tx.blocking_send(Err(Status::internal(
                        format!("failed to load model '{}': {e}", req.model)
                    )));
                    return;
                }
            }

            let messages: Vec<_> = req.messages.iter()
                .map(|m| (m.role.clone(), m.content.clone()))
                .collect();
            let params = proto_params_to_engine(req.params);
            let enc_key: Option<[u8; 32]> = if req.encryption_key.len() == 32 {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&req.encryption_key);
                Some(arr)
            } else {
                None
            };
            let start = std::time::Instant::now();

            let result = mgr.generate_chat(&req.model, &messages, &params, enc_key.as_ref(), |token| {
                let resp = ChatResponse {
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
                    let _ = tx.blocking_send(Ok(ChatResponse {
                        token: String::new(),
                        done: true,
                        usage: Some(send_usage(stats, start.elapsed().as_secs_f32())),
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
                LoadOptions { gpu_layers, digest, priority, idle_reload },
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
    use crate::model_store::registry::{ModelEntry, ModelFormat, Registry};


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
