// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MlxBridge",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "MlxBridge", type: .static, targets: ["MlxBridge"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm",
            // Pinned to a main revision (not a tag) for reproducible builds:
            // upstream's tags lag main by weeks, and this is the newest
            // revision our toolchain can build. Ceiling, in order:
            //  - #369 (2026-07-15) onward calls mlx-swift 0.31.5+ APIs
            //    (MLXArray.maskFill, DType.greatestFiniteMagnitudeArray);
            //  - mlx-swift 0.31.5+ declares swift-tools 6.3, newer than any
            //    installed toolchain (Xcode 16.4 = 6.1, Xcode 26.3 = 6.2.4),
            //    so resolution keeps mlx-swift at 0.31.4;
            //  - lm main @ 2026-08-11+ (#519) additionally declares 6.2.
            // This revision (main @ 2026-07-15, parent of #369) picks up the
            // tool round-trip ordering fix (#409), ChatSession cancellation
            // fixes (#389/#413/#423), the safetensors-index fix (#408), and
            // Qwen3.5 windowed prefill (#399). The guided-generation
            // required-properties fix (#465) and prompt-cache persistence
            // (#475) land after #369 — revisit once a Swift 6.3 toolchain is
            // on this mac and the macos-15 CI runner.
            revision: "d2424294a6c3bbd0de37a0761d80efc05e6813dd"
        ),
        // Needed so `import Tokenizers` is in scope when `#huggingFaceTokenizerLoader()` expands.
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            .upToNextMinor(from: "1.3.0")
        ),
    ],
    targets: [
        .target(
            name: "MlxBridge",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/MlxBridge",
            publicHeadersPath: "include",
            // The FFI bridge intentionally shares DispatchSemaphore/Box across Task
            // boundaries — suppress Swift 6 strict region isolation for this target only.
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
