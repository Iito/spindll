// build.rs — runs BEFORE the Rust compiler.
// Think of it like a Makefile step that generates code.
//
// This compiles proto/spindll.proto into Rust types and a gRPC
// client/server. The generated code lands in target/ and is
// included via tonic::include_proto!() in our source.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos("proto/spindll.proto")?;
    Ok(())
}
