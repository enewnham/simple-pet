// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SimplePet",
    platforms: [.macOS(.v12)],
    products: [
        .executable(name: "pet-overlay", targets: ["PetOverlay"]),
        .executable(name: "pet-speech-similarity", targets: ["PetSpeechSimilarity"])
    ],
    targets: [
        .target(name: "PetCore"),
        .executableTarget(name: "PetOverlay", dependencies: ["PetCore"]),
        .executableTarget(name: "PetSpeechSimilarity")
    ]
)
