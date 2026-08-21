// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Iito <https://github.com/Iito> and sarmientoF <https://github.com/sarmientoF>

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Proto codegen — always runs.
    println!("cargo:rerun-if-changed=proto/spindll.proto");
    println!("cargo:rerun-if-changed=proto");
    tonic_build::compile_protos("proto/spindll.proto")?;

    // MLX Swift bridge — only on aarch64 macOS with `--features mlx`.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if std::env::var("CARGO_FEATURE_MLX").is_ok() {
        build_mlx_bridge()?;
    }

    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn build_mlx_bridge() -> Result<(), Box<dyn std::error::Error>> {
    use std::process::Command;

    println!("cargo:rerun-if-changed=mlx_bridge/Sources");
    println!("cargo:rerun-if-changed=mlx_bridge/Package.swift");

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;

    // Fail fast on a missing Metal Toolchain BEFORE the multi-minute Swift
    // build, so the actionable hint is the first thing the build says.
    preflight_metal_toolchain()?;

    // mlx-swift 0.31.5+ declares swift-tools 6.3, so SwiftPM must be at least
    // that new to parse the dependency manifests — newer than the SwiftPM in
    // Xcode <= 26.3. Honor an explicit $TOOLCHAINS, else use the default
    // `swift` when new enough, else probe for an installed swift.org
    // toolchain. Applied to the SwiftPM child process only — metallib
    // compilation below stays on the xcode-select'd toolchain.
    let toolchain_id = select_swift_toolchain()?;

    // Build the Swift package as a release static library.
    let mut swift_build = Command::new("swift");
    swift_build.args([
        "build",
        "--package-path", "mlx_bridge",
        "--configuration", "release",
        "--arch", "arm64",
    ]);
    if let Some(id) = &toolchain_id {
        println!("cargo:warning=mlx_bridge: default swift is older than 6.3, using toolchain {id}");
        swift_build.env("TOOLCHAINS", id);
    }
    let status = swift_build.status()?;

    if !status.success() {
        return Err("swift build failed for mlx_bridge".into());
    }

    // Compile Metal shaders → mlx.metallib so MLX can find its GPU kernels.
    // SwiftPM cannot compile .metal files; we do it here with xcrun metal + metallib.
    compile_mlx_metallib(&manifest_dir)?;

    // Swift SPM outputs: mlx_bridge/.build/release/libMlxBridge.a
    let lib_dir = format!("{manifest_dir}/mlx_bridge/.build/release");
    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib=static=MlxBridge");

    // Locate the Xcode developer directory via xcode-select so this works on any Mac.
    let dev_dir = Command::new("xcode-select")
        .arg("-p")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "/Applications/Xcode.app/Contents/Developer".to_string());

    let toolchain = format!("{dev_dir}/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift");

    // Swift objects auto-link back-deployment shims (e.g. the
    // swift_coroFrameAlloc compatibility archive) from the toolchain that
    // compiled them — when a newer toolchain was selected above, its static
    // libs must be searched at LINK time ahead of the xcode-select'd
    // toolchain's older ones, which predate those shims and fail the link.
    // Never put it on the runtime rpath: its dylibs expect a newer
    // libswiftCore than the OS ships and dyld aborts at load.
    if let Some(dir) = toolchain_id.as_deref().and_then(toolchain_swift_lib_dir) {
        println!("cargo:rustc-link-search=native={dir}/macosx");
    }

    // Static compatibility shims (auto-linked from compiled Swift objects).
    println!("cargo:rustc-link-search=native={toolchain}/macosx");

    // Resolve @rpath/libswift_Concurrency.dylib at runtime.
    // Toolchain path covers Xcode installs; /usr/lib/swift covers the dyld shared cache.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{toolchain}/macosx");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");

    // System frameworks required by MLX.
    for fw in &["Foundation", "Metal", "Accelerate", "MetalPerformanceShaders"] {
        println!("cargo:rustc-link-lib=framework={fw}");
    }

    Ok(())
}

/// Destination of the compiled Metal library, next to the Rust binary.
/// OUT_DIR = target/{profile}/build/spindll-{hash}/out  →  ../../.. = target/{profile}/
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn metallib_dest_path() -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let out_dir = std::env::var("OUT_DIR")?;
    let bin_dir = std::path::Path::new(&out_dir)
        .ancestors()
        .nth(3)
        .ok_or("cannot derive bin dir from OUT_DIR")?
        .to_path_buf();
    Ok(bin_dir.join("mlx.metallib"))
}

/// Issue #75 friction: when `mlx.metallib` will need compiling this build,
/// verify the Metal Toolchain exists up front instead of surfacing the error
/// after the long Swift build. A cached metallib skips the check entirely
/// (matching `compile_mlx_metallib`'s skip); any other `xcrun` hiccup is left
/// for the real compile step to report with full context.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn preflight_metal_toolchain() -> Result<(), Box<dyn std::error::Error>> {
    use std::process::Command;

    if metallib_dest_path()?.exists() {
        return Ok(());
    }
    let Ok(out) = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "--version"])
        .output()
    else {
        return Ok(());
    };
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() && stderr.contains("Metal Toolchain") {
        return Err(
            "Metal Toolchain not installed — the MLX build needs it to compile mlx.metallib.\n\
             Run: xcodebuild -downloadComponent MetalToolchain\n\
             Then rebuild with: cargo build --features cli,mlx".into()
        );
    }
    if !out.status.success() {
        // Only visible with `cargo build -vv`; the real compile step reports.
        println!("preflight: xcrun metal failed for a non-toolchain reason: {stderr}");
    }
    Ok(())
}

/// Compile MLX's pre-generated Metal shaders into `mlx.metallib` and copy it
/// next to the Rust binary so `load_colocated_library("mlx")` in device.cpp finds it.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn compile_mlx_metallib(manifest_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    let metal_src = PathBuf::from(manifest_dir)
        .join("mlx_bridge/.build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal");

    let out_dir = std::env::var("OUT_DIR")?;
    let metallib_dest = metallib_dest_path()?;
    let metallib_out = Path::new(&out_dir).join("mlx.metallib");

    // Skip recompilation if the metallib is already in the binary dir.
    // The build script re-runs whenever mlx_bridge/Sources changes (rerun-if-changed above),
    // so the metallib will be refreshed when shaders change.
    if metallib_dest.exists() {
        // Ensure OUT_DIR has a copy for include_bytes!() even when the
        // build-script hash changes across rebuilds.
        if !metallib_out.exists() {
            std::fs::copy(&metallib_dest, &metallib_out)?;
        }
        return Ok(());
    }

    // Collect all .metal files recursively under mlx-generated/metal/.
    let metal_files = collect_metal_files(&metal_src)?;
    if metal_files.is_empty() {
        return Err("no .metal files found in mlx-generated/metal/".into());
    }

    // Compile each .metal → .air in OUT_DIR.
    let mut air_files: Vec<PathBuf> = Vec::new();
    for metal_file in &metal_files {
        // Disambiguate files with the same stem in different subdirs.
        let rel = metal_file.strip_prefix(&metal_src)?.to_string_lossy();
        let safe_name = rel.replace(['/', '\\', '.'], "_");
        let air_file = Path::new(&out_dir).join(format!("{safe_name}.air"));

        let out = Command::new("xcrun")
            .args([
                "-sdk", "macosx", "metal",
                "-O2",
                "-c", metal_file.to_str().unwrap(),
                "-o", air_file.to_str().unwrap(),
                "-I", metal_src.to_str().unwrap(),
            ])
            .output()?;

        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            if stderr.contains("missing Metal Toolchain") {
                return Err(
                    "Metal Toolchain not installed.\n\
                     Run: xcodebuild -downloadComponent MetalToolchain\n\
                     Then rebuild with: cargo build --features cli,mlx".into()
                );
            }
            return Err(format!(
                "metal compilation failed for {}:\n{}",
                metal_file.display(),
                stderr
            ).into());
        }
        air_files.push(air_file);
    }

    // Link all .air files into mlx.metallib in OUT_DIR, then copy to the binary dir.
    let mut args = vec![
        "-sdk".to_string(), "macosx".to_string(),
        "metallib".to_string(),
        "-o".to_string(), metallib_out.to_str().unwrap().to_string(),
    ];
    args.extend(air_files.iter().map(|p| p.to_str().unwrap().to_string()));

    let status = Command::new("xcrun").args(&args).status()?;
    if !status.success() {
        return Err("metallib link failed".into());
    }

    std::fs::copy(&metallib_out, &metallib_dest)?;
    println!("cargo:warning=compiled mlx.metallib → {}", metallib_dest.display());

    Ok(())
}

/// `usr/lib/swift` of the toolchain `id`, located via its swiftc.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn toolchain_swift_lib_dir(id: &str) -> Option<String> {
    let out = std::process::Command::new("xcrun")
        .args(["--toolchain", id, "--find", "swiftc"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let swiftc = String::from_utf8_lossy(&out.stdout).trim().to_string();
    // <toolchain>/usr/bin/swiftc → <toolchain>/usr/lib/swift
    let usr = std::path::Path::new(&swiftc).parent()?.parent()?;
    let dir = usr.join("lib/swift");
    dir.exists().then(|| dir.display().to_string())
}

/// Minimum Swift needed to parse mlx_bridge's dependency manifests
/// (mlx-swift 0.31.5+ declares `swift-tools-version: 6.3`).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const MIN_SWIFT_FOR_BRIDGE: (u32, u32) = (6, 3);

/// `swift --version` → (major, minor), optionally under a TOOLCHAINS selection.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn swift_version(toolchain_id: Option<&str>) -> Option<(u32, u32)> {
    let mut cmd = std::process::Command::new("swift");
    cmd.arg("--version");
    if let Some(id) = toolchain_id {
        cmd.env("TOOLCHAINS", id);
    }
    let out = cmd.output().ok()?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let rest = &text[text.find("Swift version ")? + "Swift version ".len()..];
    let mut nums = rest.split(|c: char| !c.is_ascii_digit());
    Some((nums.next()?.parse().ok()?, nums.next()?.parse().ok()?))
}

/// How to invoke SwiftPM: `None` = default toolchain is new enough,
/// `Some(id)` = set `TOOLCHAINS=<id>`. Errors when nothing installed can
/// parse tools-6.3 manifests, with install hints.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn select_swift_toolchain() -> Result<Option<String>, Box<dyn std::error::Error>> {
    // An explicit user choice always wins.
    if let Ok(id) = std::env::var("TOOLCHAINS") {
        return Ok(Some(id));
    }
    if swift_version(None).is_some_and(|v| v >= MIN_SWIFT_FOR_BRIDGE) {
        return Ok(None);
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let roots = [
        format!("{home}/Library/Developer/Toolchains"),
        "/Library/Developer/Toolchains".to_string(),
    ];
    for root in &roots {
        // Try the swift-latest symlink first, then every toolchain newest-first.
        let mut candidates: Vec<String> = Vec::new();
        let latest = format!("{root}/swift-latest.xctoolchain/Info.plist");
        if std::path::Path::new(&latest).exists() {
            candidates.push(latest);
        }
        if let Ok(rd) = std::fs::read_dir(root) {
            let mut found: Vec<String> = rd
                .flatten()
                .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("xctoolchain"))
                .map(|e| format!("{}/Info.plist", e.path().display()))
                .collect();
            found.sort();
            found.reverse();
            candidates.extend(found);
        }
        for plist in candidates {
            let Ok(out) = std::process::Command::new("/usr/libexec/PlistBuddy")
                .args(["-c", "Print CFBundleIdentifier", &plist])
                .output()
            else {
                continue;
            };
            if !out.status.success() {
                continue;
            }
            let id = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !id.is_empty() && swift_version(Some(&id)).is_some_and(|v| v >= MIN_SWIFT_FOR_BRIDGE) {
                return Ok(Some(id));
            }
        }
    }
    Err(format!(
        "mlx_bridge needs Swift {}.{}+ (mlx-swift 0.31.5+ declares swift-tools 6.3), but the \
         selected toolchain's swift is older and no newer toolchain was found under \
         ~/Library/Developer/Toolchains or /Library/Developer/Toolchains.\n\
         Fix one of:\n\
         - install a Swift 6.3+ toolchain from https://www.swift.org/install/macos/\n\
         - select an Xcode 26.4+ install: sudo xcode-select -s /Applications/Xcode.app\n\
         - set TOOLCHAINS=<bundle id> to a specific toolchain",
        MIN_SWIFT_FOR_BRIDGE.0, MIN_SWIFT_FOR_BRIDGE.1
    )
    .into())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn collect_metal_files(dir: &std::path::Path) -> Result<Vec<std::path::PathBuf>, Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            files.extend(collect_metal_files(&path)?);
        } else if path.extension().and_then(|e| e.to_str()) == Some("metal") {
            files.push(path);
        }
    }
    Ok(files)
}
