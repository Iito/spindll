# iOS

spindll's library builds for iPhone and the iOS Simulator. The numbers below
are from an iPhone 16 Pro Max (iOS 27.2) running Gemma 3 270M Q4_0 (97 MB) with
`n_ctx` 256, one fresh process per measurement.

## Build

```sh
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
cargo build --lib --release --target aarch64-apple-ios       # iPhone
cargo build --lib --release --target aarch64-apple-ios-sim   # Simulator
```

The MLX bridge is macOS-only for now: with `--features mlx`, a build for any
other target skips it with a warning.

## Keyboard extensions: build without Metal

llama.cpp compiles Metal in by default on Apple, and its setup costs memory as
soon as the backend starts, even when every layer runs on the CPU:

| llama.cpp build | Backend start | Total after load and 8 tokens |
|---|---|---|
| Metal compiled in | 55 MB | 82 MB |
| Metal compiled out | 2 MB | 32 MB |

A keyboard extension gets about 76 MB on an iPhone 15, so build it without
Metal. `llama-cpp-sys-2` passes any `GGML_*` environment variable to CMake:

```sh
GGML_METAL=OFF cargo build --lib --release --target aarch64-apple-ios
```

The model file itself barely counts: weights are memory-mapped, and clean
file-backed pages don't add to the footprint iOS enforces. Loading the 97 MB
model costs about 12 MB.

## Defaults on iOS

- **CPU unless the `metal` feature is on.** Small models are faster on the
  CPU: 83 tok/s against 37 tok/s on Metal for the model above.
- **Always CPU on the Simulator.** Its Metal returns garbage for llama.cpp's
  kernels, which is why llama.cpp's own iOS example runs it on the CPU too.
- **The memory budget comes from `os_proc_available_memory()`.** iOS
  terminates a process at its jetsam limit, far below physical RAM, so the
  budget is the headroom under that limit, not total RAM.
