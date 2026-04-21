//! gRPC server exposing spindll's inference and model management RPCs.

pub mod service;

use std::sync::Arc;
use crate::engine::ModelManager;
use crate::model_store::ModelStore;
use crate::proto::spindll_server::SpindllServer;

/// Start the gRPC server on the given port.
///
/// Binds to `0.0.0.0:<port>` and serves until the process exits.
/// The server exposes generate, chat, load/unload, pull, list, status,
/// prefill, and delete RPCs.
pub async fn start_server(
    port: u16,
    manager: Arc<ModelManager>,
    model_store: Arc<ModelStore>,
) -> anyhow::Result<()> {
    let addr = format!("0.0.0.0:{port}").parse()?;
    let svc = service::SpindllService::new(manager, model_store);

    tracing::info!(%addr, "gRPC server listening");

    tonic::transport::Server::builder()
        .add_service(SpindllServer::new(svc))
        .serve(addr)
        .await?;

    Ok(())
}
