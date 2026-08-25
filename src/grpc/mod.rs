// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Iito <https://github.com/Iito> and sarmientoF <https://github.com/sarmientoF>

//! gRPC server exposing spindll's inference and model management RPCs.

pub mod service;

use std::sync::Arc;
use crate::engine::ModelManager;
use crate::model_store::ModelStore;
use crate::proto::spindll_server::SpindllServer;

/// Start the gRPC server on the given port.
///
/// Binds to `0.0.0.0:<port>` and serves until the process exits or a shutdown
/// signal arrives. The server exposes generate, chat, load/unload, pull, list,
/// status, prefill, and delete RPCs.
///
/// Returns on SIGINT/SIGTERM so the caller can clean up — without that the
/// lockfile survives every ordinary Ctrl-C, and the next client command reads
/// a dead server's ports out of it.
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
        .serve_with_shutdown(addr, shutdown_signal())
        .await?;

    Ok(())
}

/// Resolves on the first SIGINT or SIGTERM.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            // Without SIGTERM we still have Ctrl-C; never fire spuriously.
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    tracing::info!("shutdown signal received");
}
