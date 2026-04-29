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

    // Build the Swift package as a release static library.
    let status = Command::new("swift")
        .args([
            "build",
            "--package-path", "mlx_bridge",
            "--configuration", "release",
            "--arch", "arm64",
        ])
        .status()?;

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

/// Compile MLX's pre-generated Metal shaders into `mlx.metallib` and copy it
/// next to the Rust binary so `load_colocated_library("mlx")` in device.cpp finds it.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn compile_mlx_metallib(manifest_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    let metal_src = PathBuf::from(manifest_dir)
        .join("mlx_bridge/.build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal");

    // Derive the binary output directory from OUT_DIR:
    // OUT_DIR = target/{profile}/build/spindll-{hash}/out  →  ../../.. = target/{profile}/
    let out_dir = std::env::var("OUT_DIR")?;
    let bin_dir = Path::new(&out_dir)
        .ancestors()
        .nth(3)
        .ok_or("cannot derive bin dir from OUT_DIR")?
        .to_path_buf();

    let metallib_dest = bin_dir.join("mlx.metallib");

    // Skip recompilation if the metallib is already in the binary dir.
    // The build script re-runs whenever mlx_bridge/Sources changes (rerun-if-changed above),
    // so the metallib will be refreshed when shaders change.
    if metallib_dest.exists() {
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
        let stem = metal_file
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or("invalid metal filename")?;
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
    let metallib_tmp = Path::new(&out_dir).join("mlx.metallib");
    let mut args = vec![
        "-sdk".to_string(), "macosx".to_string(),
        "metallib".to_string(),
        "-o".to_string(), metallib_tmp.to_str().unwrap().to_string(),
    ];
    args.extend(air_files.iter().map(|p| p.to_str().unwrap().to_string()));

    let status = Command::new("xcrun").args(&args).status()?;
    if !status.success() {
        return Err("metallib link failed".into());
    }

    std::fs::copy(&metallib_tmp, &metallib_dest)?;
    println!("cargo:warning=compiled mlx.metallib → {}", metallib_dest.display());

    Ok(())
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
