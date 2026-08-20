// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Iito <https://github.com/Iito> and sarmientoF <https://github.com/sarmientoF>

//! Hand-written FFI declarations for llama.cpp's RPC backend (`ggml-rpc.h`).
//!
//! `llama-cpp-sys-2` never enables `GGML_RPC`, so it neither compiles the
//! backend nor bindgens this header. Spindll flips the CMake option through
//! the sys crate's `GGML_*` env-var forwarding (`.cargo/config.toml`), which
//! compiles the RPC backend into the same static library these declarations
//! link against. Signatures must match `ggml/include/ggml-rpc.h` at the rev
//! pinned in `[patch.crates-io]` (0.1.154 / RPC protocol 5.0.0); revisit on
//! every llama-cpp-2 bump.

use std::ffi::c_char;

use llama_cpp_sys_2::{ggml_backend_dev_t, ggml_backend_reg_t};

unsafe extern "C" {
    /// Registers a remote RPC server's devices into the global ggml backend
    /// registry; models loaded afterwards may offload layers onto them.
    pub fn ggml_backend_rpc_add_server(endpoint: *const c_char) -> ggml_backend_reg_t;

    /// Serves this process's devices to remote coordinators. Blocks forever;
    /// run it on a dedicated thread.
    pub fn ggml_backend_rpc_start_server(
        endpoint: *const c_char,
        cache_dir: *const c_char,
        n_threads: usize,
        n_devices: usize,
        devices: *mut ggml_backend_dev_t,
    );

    /// Queries free/total memory of one device behind an endpoint.
    pub fn ggml_backend_rpc_get_device_memory(
        endpoint: *const c_char,
        device: u32,
        free: *mut usize,
        total: *mut usize,
    );

    /// Returns the RPC backend registration. Points at static storage —
    /// never null, no init required, touches no network or device state.
    pub fn ggml_backend_rpc_reg() -> ggml_backend_reg_t;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Link smoke test: passes only when the RPC backend was actually
    /// compiled into the llama.cpp static library and its symbols resolve.
    #[test]
    fn rpc_backend_links_and_registers() {
        let reg = unsafe { ggml_backend_rpc_reg() };
        assert!(!reg.is_null(), "ggml_backend_rpc_reg() returned null");
    }
}
