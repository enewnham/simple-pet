import AppKit
import Foundation

struct CheckFailure: Error { let message: String }
func require(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    if try !condition() { throw CheckFailure(message: message) }
}
let manager = FileManager.default
let temporary = manager.temporaryDirectory.appendingPathComponent("pet-asset-check-\(UUID().uuidString)")
try manager.createDirectory(at: temporary, withIntermediateDirectories: true)
defer { try? manager.removeItem(at: temporary) }
guard CommandLine.arguments.count == 2 else { fatalError("Usage: asset-import-tests PATH_TO_IMPORTER") }
let importer = URL(fileURLWithPath: CommandLine.arguments[1]).standardizedFileURL
let source = temporary.appendingPathComponent("source")
let pet = source.appendingPathComponent("assets/pets/fixture")
let output = temporary.appendingPathComponent("output")
try manager.createDirectory(at: pet, withIntermediateDirectories: true)

func invoke(_ selectedPet: String = "fixture", root: URL? = nil, origin: String? = nil) throws -> Int32 {
    let process = Process()
    process.executableURL = importer
    process.arguments = [(root ?? source).path, output.path, selectedPet, "fixture-commit"]
    if let origin { process.arguments!.append(origin) }
    process.standardOutput = FileHandle.nullDevice
    process.standardError = FileHandle.nullDevice
    try process.run()
    process.waitUntilExit()
    return process.terminationStatus
}
func writeObject(_ object: [String: Any], to url: URL) throws {
    try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]).write(to: url)
}
func readManifest() throws -> [String: Any] {
    try JSONSerialization.jsonObject(with: Data(contentsOf: output.appendingPathComponent("manifest.json"))) as! [String: Any]
}
func writeSheet(height: Int) throws {
    let width = 1536
    var pixels = [UInt8](repeating: 0, count: width * height * 4)
    for y in 0..<height {
        for x in 0..<width {
            let index = (y * width + x) * 4
            pixels[index] = UInt8((y / 208) * 20)
            pixels[index + 1] = UInt8((x / 192) * 20)
            pixels[index + 2] = 100
            pixels[index + 3] = 255
        }
    }
    let provider = CGDataProvider(data: Data(pixels) as CFData)!
    let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                        bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    try NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:])!
        .write(to: pet.appendingPathComponent("spritesheet.png"))
}
func pixel(_ path: String) throws -> [Int] {
    let bitmap = NSBitmapImageRep(data: try Data(contentsOf: output.appendingPathComponent(path)))!
    try require(bitmap.pixelsWide == 192 && bitmap.pixelsHigh == 208, "Keep authored frame dimensions.")
    var channels = [Int](repeating: 0, count: bitmap.samplesPerPixel)
    bitmap.getPixel(&channels, atX: 10, y: 10)
    return channels
}

try writeSheet(height: 1872)
try require(try invoke() != 0, "Reject sheets without the V2 gaze rows.")
try writeSheet(height: 2288)
try require(try invoke() == 0, "Import a V2 sheet without version metadata.")
let expected: [(String, Int, Int, String, Double)] = [
    ("idle", 6, 0, "rest", 5.5), ("running-right", 8, 20, "walk-right", 1.06),
    ("running-left", 8, 40, "walk-left", 1.06), ("waving", 4, 60, "greeting", 0.7),
    ("jumping", 5, 80, "celebrate", 0.84), ("failed", 8, 100, "error", 1.22),
    ("waiting", 6, 120, "attention", 1.01), ("running", 6, 140, "work", 0.82),
    ("review", 6, 160, "attention", 1.03),
]
let sheet = try readManifest()
let sheetClips = sheet["clips"] as! [String: [String: Any]]
try require(sheetClips.count == 25 && (sheet["gaze"] as! [String]).count == 16, "Import nine animations and sixteen gaze poses.")
for (id, count, red, role, duration) in expected {
    let clip = sheetClips[id]!
    let paths = clip["frames"] as! [String]
    try require(paths.count == count, "Keep \(id)'s authored frame count.")
    try require(clip["role"] as? String == role, "Declare the \(id) semantic role.")
    try require(abs((clip["fps"] as! Double) * duration - Double(count)) < 0.0001, "Keep \(id)'s authored timing.")
    for (index, path) in paths.enumerated() {
        let channels = try pixel(path)
        try require(channels[0] == red && channels[1] == index * 20, "Select the \(id) row and preserve frame order.")
    }
}
try require(sheet["commit"] as? String == "fixture-commit", "Record source provenance.")
try "keep me".write(to: output.appendingPathComponent("unrelated.txt"), atomically: true, encoding: .utf8)

try writeSheet(height: 2288)
try writeObject(["spriteVersionNumber": 2, "spritesheetPath": "spritesheet.png", "gazeAnchor": ["x": 0.4, "y": 0.3]], to: pet.appendingPathComponent("pet.json"))
try require(try invoke() == 0, "Import a V2 package using its declared sheet path.")
let v2 = try readManifest()
let gaze = v2["gaze"] as! [String]
let v2Clips = v2["clips"] as! [String: [String: Any]]
try require(gaze.count == 16 && v2Clips.count == 25, "Import sixteen optional V2 gaze poses.")
for (index, id) in gaze.enumerated() {
    let channels = try pixel((v2Clips[id]!["frames"] as! [String])[0])
    try require(channels[0] == (9 + index / 8) * 20 && channels[1] == (index % 8) * 20, "Keep the clockwise gaze atlas order.")
}
try require((v2["gazeAnchor"] as! [String: Double])["x"] == 0.4, "Preserve the authored gaze anchor.")

let named = source.appendingPathComponent("assets/pets/small")
try manager.createDirectory(at: named.appendingPathComponent("idle"), withIntermediateDirectories: true)
for (name, frameIndex) in [("frame10.png", 2), ("frame2.png", 1), ("frame1.png", 0)] {
    try manager.copyItem(at: output.appendingPathComponent(String(format: "clips/idle/%03d.png", frameIndex)),
                         to: named.appendingPathComponent("idle/\(name)"))
}
try require(try invoke("small") == 0, "Accept a rest-only pet from a sprite folder.")
let sparse = try readManifest()
let sparseClips = sparse["clips"] as! [String: [String: Any]]
try require(sparseClips.count == 1 && sparse["gaze"] == nil, "Omit unavailable walking and gesture capabilities.")
for (index, path) in (sparseClips["idle"]!["frames"] as! [String]).enumerated() {
    try require(try pixel(path)[1] == index * 20, "Sort named frames numerically.")
}
try require(!manager.fileExists(atPath: output.appendingPathComponent("clips/running").path), "Remove stale generated clips when replacing a pack.")
try require(try String(contentsOf: output.appendingPathComponent("unrelated.txt"), encoding: .utf8) == "keep me", "Preserve unrelated output files.")

let custom = temporary.appendingPathComponent("custom")
try manager.createDirectory(at: custom, withIntermediateDirectories: true)
try manager.copyItem(at: output.appendingPathComponent("clips/idle/000.png"), to: custom.appendingPathComponent("pose.png"))
let customManifest: [String: Any] = ["pet": "custom", "rest": "calm", "clips": [
    "calm": ["frames": ["pose.png"], "fps": 1.0, "role": "rest"],
    "bounce": ["frames": ["pose.png"], "fps": 2.0, "role": "celebrate", "max_repeats": 2],
    "snooze": ["frames": ["pose.png"], "fps": 1.0, "role": "sleep"],
]]
try writeObject(customManifest, to: custom.appendingPathComponent("manifest.json"))
try require(try invoke("custom", root: custom) == 0, "Accept direct custom packs with explicit semantic metadata.")
let installed = try readManifest()["clips"] as! [String: [String: Any]]
try require(installed["bounce"]?["max_repeats"] as? Int == 2, "Preserve custom repetition limits.")
try require(installed["snooze"]?["role"] as? String == "sleep", "Preserve authored sleep clips.")
try require(try invoke("custom", root: custom, origin: "/downloads/custom-pet.zip") == 0, "Accept an original local ZIP path as provenance.")
try require(try readManifest()["source"] as? String == "/downloads/custom-pet.zip", "Preserve the original ZIP path rather than its extraction directory.")
let saved = try Data(contentsOf: output.appendingPathComponent("manifest.json"))
try manager.removeItem(at: custom.appendingPathComponent("pose.png"))
try require(try invoke("custom", root: custom) != 0, "Reject custom packs with missing referenced frames.")
try require(try Data(contentsOf: output.appendingPathComponent("manifest.json")) == saved, "Preserve the installed pack after failed validation.")
try require(manager.fileExists(atPath: output.appendingPathComponent("clips/bounce/000.png").path), "Preserve installed frames after failed validation.")
var escaping = customManifest
escaping["clips"] = ["calm": ["frames": ["../output/clips/calm/000.png"], "fps": 1.0, "role": "rest"]]
try writeObject(escaping, to: custom.appendingPathComponent("manifest.json"))
try require(try invoke("custom", root: custom) != 0, "Reject frame references outside the source pet.")
let store = temporary.appendingPathComponent("store-pet")
try manager.createDirectory(at: store, withIntermediateDirectories: true)
try manager.copyItem(at: pet.appendingPathComponent("spritesheet.png"), to: store.appendingPathComponent("spritesheet.png"))
let storeMetadata: [String: Any] = [
    "id": "store-pet", "displayName": "Store Pet", "description": "A downloaded companion",
    "spritesheetPath": "spritesheet.png", "spriteVersionNumber": 2,
    "sourceUrl": "https://pets.example/pets/store-pet",
]
try writeObject(storeMetadata, to: store.appendingPathComponent("pet.json"))
try require(try invoke("store-pet", root: store) == 0, "Import a store-style package containing only metadata and its sheet.")
let storeManifest = try readManifest()
let preservedPackage = storeManifest["package"] as! [String: Any]
try require(preservedPackage["id"] as? String == "store-pet", "Preserve the package identity in manifest provenance.")
try require(preservedPackage["displayName"] as? String == "Store Pet", "Preserve the package display name.")
try require(storeManifest["source"] as? String == "https://pets.example/pets/store-pet", "Use the package's declared source when no origin is supplied.")
try require(storeManifest["sourcePath"] as? String == ".", "Keep the direct package source path relative to its package root.")
let zipOrigin = "/downloads/store-pet.zip"
try require(try invoke("store-pet", root: store, origin: zipOrigin) == 0, "Accept the original ZIP path as source origin.")
let downloadedManifest = try readManifest()
try require(downloadedManifest["source"] as? String == zipOrigin, "Use the original ZIP path rather than a temporary extraction path.")
try require((downloadedManifest["package"] as! [String: Any])["sourceUrl"] as? String == "https://pets.example/pets/store-pet", "Preserve the package's own URL alongside the ZIP origin.")
let savedStore = try Data(contentsOf: output.appendingPathComponent("manifest.json"))
try manager.removeItem(at: store.appendingPathComponent("spritesheet.png"))
try require(try invoke("store-pet", root: store, origin: zipOrigin) != 0, "Reject a store package with a missing sheet.")
try require(try Data(contentsOf: output.appendingPathComponent("manifest.json")) == savedStore, "Preserve the installed pack after a store package fails validation.")
print("Asset import checks passed: V2 sheets, sparse packs, custom packs, store packages, provenance, and failed replacement.")
