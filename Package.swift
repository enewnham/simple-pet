// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SimplePet",
    platforms: [.macOS(.v12)],
    products: [
        .executable(name: "pet-overlay", targets: ["PetOverlay"])
    ],
    targets: [
        .target(name: "PetCore"),
        .executableTarget(name: "PetOverlay", dependencies: ["PetCore"]),
        .executableTarget(name: "AssetImporter", path: "scripts",
                          exclude: ["pet-hook", "fetch-assets.sh", "unpack-assets.py", "check-desktop.py"],
                          sources: ["import-assets.swift"]),
        .testTarget(name: "PetCoreTests", dependencies: ["PetCore"]),
        .testTarget(name: "AssetImportTests", dependencies: ["AssetImporter"]),
        .testTarget(name: "PetOverlayTests", dependencies: ["PetOverlay", "PetCore"])
    ],
    swiftLanguageModes: [.v5]
)
