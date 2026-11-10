// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SpindllKit",
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "SpindllKit", targets: ["SpindllKit"]),
    ],
    targets: [
        // Built by scripts/build-ios-xcframework.sh; not checked in.
        .binaryTarget(name: "SpindllFFI", path: "SpindllFFI.xcframework"),
        .target(
            name: "SpindllKit",
            dependencies: ["SpindllFFI"],
            // What the Rust static library links against.
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Foundation"),
                .linkedFramework("Security"),
                .linkedLibrary("c++"),
                .linkedLibrary("iconv"),
            ]
        ),
    ]
)
