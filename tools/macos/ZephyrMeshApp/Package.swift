// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "ZephyrMeshApp",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "ZephyrMeshApp", targets: ["ZephyrMeshApp"])
    ],
    targets: [
        .executableTarget(
            name: "ZephyrMeshApp",
            resources: [.process("Resources")]
        ),
        .testTarget(name: "ZephyrMeshAppTests", dependencies: ["ZephyrMeshApp"], resources: [.process("Fixtures")])
    ]
)
