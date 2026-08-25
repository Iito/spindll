// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Iito <https://github.com/Iito> and sarmientoF <https://github.com/sarmientoF>

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Lockfile {
    pub pid: u32,
    pub grpc_port: u16,
    pub http_port: u16,
}

fn lockfile_path() -> PathBuf {
    std::env::temp_dir().join("spindll.lock")
}

impl Lockfile {
    pub fn write(grpc_port: u16, http_port: u16) -> std::io::Result<()> {
        let lock = Lockfile {
            pid: std::process::id(),
            grpc_port,
            http_port,
        };
        let json = serde_json::to_string(&lock)
            .map_err(std::io::Error::other)?;
        std::fs::write(lockfile_path(), json)
    }

    pub fn read() -> Option<Lockfile> {
        let data = std::fs::read_to_string(lockfile_path()).ok()?;
        let lock: Lockfile = serde_json::from_str(&data).ok()?;
        if process_alive(lock.pid) {
            Some(lock)
        } else {
            std::fs::remove_file(lockfile_path()).ok();
            None
        }
    }

    /// Remove the lockfile, but only when it is ours.
    ///
    /// A second `spindll serve` that fails to bind would otherwise erase the
    /// record of the healthy server already holding the port, leaving
    /// `spindll status` reporting no server while one is serving.
    pub fn remove() {
        let ours = std::fs::read_to_string(lockfile_path())
            .ok()
            .and_then(|d| serde_json::from_str::<Lockfile>(&d).ok())
            .is_some_and(|l| l.pid == std::process::id());
        if ours {
            std::fs::remove_file(lockfile_path()).ok();
        }
    }
}

fn process_alive(pid: u32) -> bool {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[sysinfo::Pid::from_u32(pid)]), true);
    sys.process(sysinfo::Pid::from_u32(pid)).is_some()
}
