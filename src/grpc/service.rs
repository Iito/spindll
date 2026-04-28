use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::engine::{GenerateParams, ModelManager};
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
            max_tokens: if p.max_tokens > 0 { p.max_tokens as u32 } else { 512 },
            temperature: if p.temperature > 0.0 { p.temperature } else { 0.8 },
            top_p: if p.top_p > 0.0 { p.top_p } else { 0.95 },
            top_k: if p.top_k > 0 { p.top_k } else { 40 },
            seed: if p.seed > 0 { p.seed as u32 } else { 42 },
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
            let prompt = match mgr.apply_chat_template(&req.model, &messages) {
                Ok(p) => p,
                Err(e) => {
                    let _ = tx.blocking_send(Err(Status::internal(
                        format!("chat template error: {e}")
                    )));
                    return;
                }
            };
            let params = proto_params_to_engine(req.params);
            let enc_key: Option<[u8; 32]> = if req.encryption_key.len() == 32 {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&req.encryption_key);
                Some(arr)
            } else {
                None
            };
            let start = std::time::Instant::now();

            let result = mgr.generate(&req.model, &prompt, &params, enc_key.as_ref(), |token| {
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
            .map(|(key, entry)| ModelInfo {
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
            })
            .collect();

        Ok(Response::new(ListResponse { models }))
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

        self.manager
            .load_model_with_digest(&req.model, &model_path, gpu_layers, digest)
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
            let prompt = mgr.apply_chat_template(&req.model, &messages)
                .map_err(|e| Status::internal(format!("chat template error: {e}")))?;

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

            let stats = mgr.generate(&req.model, &prompt, &params, enc_key.as_ref(), |_| true)
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
