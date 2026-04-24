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
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
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

    pub fn remove() {
        std::fs::remove_file(lockfile_path()).ok();
    }
}

fn process_alive(pid: u32) -> bool {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[sysinfo::Pid::from_u32(pid)]), true);
    sys.process(sysinfo::Pid::from_u32(pid)).is_some()
}
