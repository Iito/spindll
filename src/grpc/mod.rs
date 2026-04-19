//! gRPC server — exposes Spindll's functionality over the network.
//!
//! Uses tonic to serve the protobuf-defined API with
//! bidirectional streaming.

pub mod service;
