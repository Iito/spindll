use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::engine::{Engine, GenerateParams};
use crate::model_store::ModelStore;
use crate::proto::spindll_server::Spindll;
use crate::proto::*;

pub struct SpindllService {
    engine: Arc<Engine>,
    model_store: Arc<ModelStore>,
}

impl SpindllService {
    pub fn new(engine: Arc<Engine>, model_store: Arc<ModelStore>) -> Self {
        Self { engine, model_store }
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
        },
        None => GenerateParams::default(),
    }
}

#[tonic::async_trait]
impl Spindll for SpindllService {
    type GenerateStream = ReceiverStream<Result<GenerateResponse, Status>>;
    type ChatStream = ReceiverStream<Result<ChatResponse, Status>>;
    type PullStream = ReceiverStream<Result<PullProgress, Status>>;

    async fn generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let req = request.into_inner();
        let engine = self.engine.clone();
        let (tx, rx) = mpsc::channel(32);

        tokio::task::spawn_blocking(move || {
            let params = proto_params_to_engine(req.params);
            let start = std::time::Instant::now();

            let result = engine.generate(&req.prompt, &params, |token| {
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
                    let elapsed = start.elapsed().as_secs_f32();
                    let _ = tx.blocking_send(Ok(GenerateResponse {
                        token: String::new(),
                        done: true,
                        usage: Some(UsageStats {
                            prompt_tokens: stats.prompt_tokens as i32,
                            completion_tokens: stats.completion_tokens as i32,
                            tokens_per_second: if elapsed > 0.0 {
                                stats.completion_tokens as f32 / elapsed
                            } else {
                                0.0
                            },
                        }),
                    }));
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn chat(
        &self,
        request: Request<ChatRequest>,
    ) -> Result<Response<Self::ChatStream>, Status> {
        let req = request.into_inner();
        let engine = self.engine.clone();
        let (tx, rx) = mpsc::channel(32);

        tokio::task::spawn_blocking(move || {
            let messages: Vec<_> = req.messages.iter()
                .map(|m| (m.role.clone(), m.content.clone()))
                .collect();
            let prompt = match engine.apply_chat_template(&messages) {
                Ok(p) => p,
                Err(e) => {
                    let _ = tx.blocking_send(Err(Status::internal(
                        format!("chat template error: {e}")
                    )));
                    return;
                }
            };
            let params = proto_params_to_engine(req.params);
            let start = std::time::Instant::now();

            let result = engine.generate(&prompt, &params, |token| {
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
                    let elapsed = start.elapsed().as_secs_f32();
                    let _ = tx.blocking_send(Ok(ChatResponse {
                        token: String::new(),
                        done: true,
                        usage: Some(UsageStats {
                            prompt_tokens: stats.prompt_tokens as i32,
                            completion_tokens: stats.completion_tokens as i32,
                            tokens_per_second: if elapsed > 0.0 {
                                stats.completion_tokens as f32 / elapsed
                            } else {
                                0.0
                            },
                        }),
                    }));
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn pull(
        &self,
        _request: Request<PullRequest>,
    ) -> Result<Response<Self::PullStream>, Status> {
        Err(Status::unimplemented("pull rpc not yet implemented"))
    }

    async fn list(
        &self,
        _request: Request<ListRequest>,
    ) -> Result<Response<ListResponse>, Status> {
        let reg = crate::model_store::registry::Registry::load(&self.model_store.registry_path())
            .map_err(|e| Status::internal(e.to_string()))?;

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
            })
            .collect();

        Ok(Response::new(ListResponse { models }))
    }

    async fn load(
        &self,
        _request: Request<LoadRequest>,
    ) -> Result<Response<LoadResponse>, Status> {
        Err(Status::unimplemented("load not yet implemented"))
    }

    async fn unload(
        &self,
        _request: Request<UnloadRequest>,
    ) -> Result<Response<UnloadResponse>, Status> {
        Err(Status::unimplemented("unload not yet implemented"))
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

        let devices = if cfg!(target_os = "macos") {
            vec!["Metal".to_string(), "CPU".to_string()]
        } else {
            vec!["CPU".to_string()]
        };

        Ok(Response::new(StatusResponse {
            models: vec![],
            memory: Some(MemoryInfo {
                total_ram: mem.total_ram,
                used_ram: mem.total_ram.saturating_sub(mem.available_ram),
                available_ram: mem.available_ram,
                total_vram: 0,
                used_vram: 0,
                available_vram: 0,
            }),
            devices,
        }))
    }
}
