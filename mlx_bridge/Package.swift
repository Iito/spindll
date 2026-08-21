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
            // upstream's tags lag main by weeks.
            // This revision (main @ 2026-08-17) picks up the guided-generation
            // required-properties fix (#465), the tool round-trip ordering fix
            // (#409), prompt-cache persistence (#475), thinking-budget
            // enforcement (#521), Qwen3.5 decode perf (#442/#467/#468), and
            // tool call parser hardening (#531).
            // Toolchain floor: Swift 6.3 (mlx-swift 0.31.5+ declares
            // swift-tools 6.3). Locally that means a swift.org 6.3+ toolchain
            // (build.rs auto-detects one under ~/Library/Developer/Toolchains)
            // or Xcode 26.4+; CI uses Xcode 26.6 on the macos-26 runner.
            revision: "7871b09b2eda7500bc2acad51125ebd772cbaffe"
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
