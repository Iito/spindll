// swift-tools-version: 6.1
// 6.1 is the first tools version that can set a dependency's traits.
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
            // 3.33.0 (2026-11-17) runs on mlx-swift 0.33.
            // Toolchain floor: Swift 6.3 (mlx-swift 0.31.5+ declares
            // swift-tools 6.3). Locally that means a swift.org 6.3+ toolchain
            // (build.rs auto-detects one under ~/Library/Developer/Toolchains)
            // or Xcode 26.4+; CI uses Xcode 26.6 on the macos-26 runner.
            exact: "3.33.0",
            // Turn off the default FoundationModelsIntegration trait. It builds
            // an adapter for Apple's FoundationModels framework, which this
            // bridge never uses, and that adapter has broken on Xcode 27 SDKs
            // (upstream #537, #543).
            traits: []
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
