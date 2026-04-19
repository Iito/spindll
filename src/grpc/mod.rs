pub mod service;

use std::sync::Arc;
use crate::engine::Engine;
use crate::model_store::ModelStore;
use crate::proto::spindll_server::SpindllServer;

pub async fn start_server(
    port: u16,
    engine: Arc<Engine>,
    model_store: Arc<ModelStore>,
) -> anyhow::Result<()> {
    let addr = format!("0.0.0.0:{port}").parse()?;
    let svc = service::SpindllService::new(engine, model_store);

    println!("spindll serving on {addr}");

    tonic::transport::Server::builder()
        .add_service(SpindllServer::new(svc))
        .serve(addr)
        .await?;

    Ok(())
}
