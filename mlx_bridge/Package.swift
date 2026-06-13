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
            // Pinned to a revision (not branch: "main") for reproducible builds.
            // Must include upstream #268, which adds `@preconcurrency import
            // CoreImage` so MLXVLM compiles under Swift 6 strict concurrency.
            revision: "a47894a1e7e963b24bd48c030f5fc1d1627e60e9"
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
