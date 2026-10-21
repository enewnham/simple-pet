import AppKit
import Foundation
import Testing

struct AssetImportTests {
    @Test func rejectsSheetsWithoutGazeRows() throws {
        let fixture = try AssetFixture()
        try fixture.writeSheet(height: 1872)
        #expect(try fixture.invoke(expectedSuccess: false) != 0)
    }

    @Test func importsAuthoredAnimationFramesAndTiming() throws {
        let fixture = try AssetFixture()
        try fixture.writeSheet()
        try fixture.invoke()
        let manifest = try fixture.manifest()
        let clips = try #require(manifest["clips"] as? [String: [String: Any]])
        let gaze = try #require(manifest["gaze"] as? [String])
        #expect(clips.count == 25)
        #expect(gaze.count == 16)
        #expect(manifest["commit"] as? String == "fixture-commit")
        let expected: [(String, Int, Int, String, Double)] = [
            ("idle", 6, 0, "rest", 5.5), ("running-right", 8, 20, "walk-right", 1.06),
            ("running-left", 8, 40, "walk-left", 1.06), ("waving", 4, 60, "greeting", 0.7),
            ("jumping", 5, 80, "celebrate", 0.84), ("failed", 8, 100, "error", 1.22),
            ("waiting", 6, 120, "attention", 1.01), ("running", 6, 140, "work", 0.82),
            ("review", 6, 160, "attention", 1.03),
        ]
        for (id, count, red, role, duration) in expected {
            let clip = try #require(clips[id])
            let paths = try #require(clip["frames"] as? [String])
            let fps = try #require(clip["fps"] as? Double)
            #expect(paths.count == count, "Frame count for \(id)")
            #expect(clip["role"] as? String == role, "Semantic role for \(id)")
            #expect(abs(fps * duration - Double(count)) < 0.0001, "Duration for \(id)")
            for (index, path) in paths.enumerated() {
                let channels = try fixture.pixel(path)
                #expect(channels[0] == red, "Sheet row for \(id)")
                #expect(channels[1] == index * 20, "Frame order for \(id)")
            }
        }
    }

    @Test func preservesGazeOrderAndAnchorFromPackageMetadata() throws {
        let fixture = try AssetFixture()
        try fixture.writeSheet()
        try fixture.writeObject([
            "spriteVersionNumber": 2, "spritesheetPath": "spritesheet.png",
            "gazeAnchor": ["x": 0.4, "y": 0.3],
        ], to: fixture.pet.appendingPathComponent("pet.json"))
        try fixture.invoke()
        let manifest = try fixture.manifest()
        let gaze = try #require(manifest["gaze"] as? [String])
        let clips = try #require(manifest["clips"] as? [String: [String: Any]])
        #expect(gaze.count == 16)
        #expect(clips.count == 25)
        for (index, id) in gaze.enumerated() {
            let frames = try #require(clips[id]?["frames"] as? [String])
            let path = try #require(frames.first)
            let channels = try fixture.pixel(path)
            #expect(channels[0] == (9 + index / 8) * 20)
            #expect(channels[1] == (index % 8) * 20)
        }
        let anchor = try #require(manifest["gazeAnchor"] as? [String: Double])
        #expect(anchor == ["x": 0.4, "y": 0.3])
    }

    @Test func replacesSheetWithNumericallySortedFolderFrames() throws {
        let fixture = try AssetFixture()
        try fixture.writeSheet()
        try fixture.invoke()
        try "keep me".write(to: fixture.output.appendingPathComponent("unrelated.txt"),
                            atomically: true, encoding: .utf8)
        let named = fixture.source.appendingPathComponent("assets/pets/small/idle")
        try FileManager.default.createDirectory(at: named, withIntermediateDirectories: true)
        for (name, frameIndex) in [("frame10.png", 2), ("frame2.png", 1), ("frame1.png", 0)] {
            try FileManager.default.copyItem(
                at: fixture.output.appendingPathComponent(String(format: "clips/idle/%03d.png", frameIndex)),
                to: named.appendingPathComponent(name))
        }
        try fixture.invoke("small")
        let manifest = try fixture.manifest()
        let clips = try #require(manifest["clips"] as? [String: [String: Any]])
        #expect(clips.count == 1)
        #expect(manifest["gaze"] == nil)
        let frames = try #require(clips["idle"]?["frames"] as? [String])
        #expect(frames.count == 3)
        for (index, path) in frames.enumerated() {
            #expect(try fixture.pixel(path)[1] == index * 20)
        }
        #expect(!FileManager.default.fileExists(atPath: fixture.output.appendingPathComponent("clips/running").path))
        #expect(try String(contentsOf: fixture.output.appendingPathComponent("unrelated.txt"), encoding: .utf8) == "keep me")
    }

    @Test func preservesCustomMetadataAndInstalledPackAfterMissingFrame() throws {
        let fixture = try AssetFixture()
        let custom = try fixture.customPack()
        try fixture.invoke("custom", root: custom)
        let manifest = try fixture.manifest()
        let clips = try #require(manifest["clips"] as? [String: [String: Any]])
        #expect(clips["bounce"]?["max_repeats"] as? Int == 2)
        #expect(clips["snooze"]?["role"] as? String == "sleep")
        try fixture.invoke("custom", root: custom, origin: "/downloads/custom-pet.zip")
        #expect(try fixture.manifest()["source"] as? String == "/downloads/custom-pet.zip")
        let saved = try fixture.manifestData()
        let savedFrame = try Data(contentsOf: fixture.output.appendingPathComponent("clips/bounce/000.png"))
        try FileManager.default.removeItem(at: custom.appendingPathComponent("pose.png"))
        #expect(try fixture.invoke("custom", root: custom, expectedSuccess: false) != 0)
        #expect(try fixture.manifestData() == saved)
        #expect(try Data(contentsOf: fixture.output.appendingPathComponent("clips/bounce/000.png")) == savedFrame)
    }

    @Test func rejectsFrameReferencesOutsideCustomPack() throws {
        let fixture = try AssetFixture()
        let custom = try fixture.customPack()
        try fixture.invoke("custom", root: custom)
        let manifest: [String: Any] = ["pet": "custom", "rest": "calm", "clips": [
            "calm": ["frames": ["../output/clips/calm/000.png"], "fps": 1.0, "role": "rest"],
        ]]
        try fixture.writeObject(manifest, to: custom.appendingPathComponent("manifest.json"))
        #expect(try fixture.invoke("custom", root: custom, expectedSuccess: false) != 0)
    }

    @Test func preservesStoreProvenanceAndInstalledPackAfterMissingSheet() throws {
        let fixture = try AssetFixture()
        let store = fixture.temporary.appendingPathComponent("store-pet")
        try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
        try fixture.writeSheet(to: store.appendingPathComponent("spritesheet.png"))
        try fixture.writeObject([
            "id": "store-pet", "displayName": "Store Pet", "description": "A downloaded companion",
            "spritesheetPath": "spritesheet.png", "spriteVersionNumber": 2,
            "sourceUrl": "https://pets.example/pets/store-pet",
        ], to: store.appendingPathComponent("pet.json"))
        try fixture.invoke("store-pet", root: store)
        let manifest = try fixture.manifest()
        let package = try #require(manifest["package"] as? [String: Any])
        #expect(package["id"] as? String == "store-pet")
        #expect(package["displayName"] as? String == "Store Pet")
        #expect(manifest["source"] as? String == "https://pets.example/pets/store-pet")
        #expect(manifest["sourcePath"] as? String == ".")
        let origin = "/downloads/store-pet.zip"
        try fixture.invoke("store-pet", root: store, origin: origin)
        let downloaded = try fixture.manifest()
        let downloadedPackage = try #require(downloaded["package"] as? [String: Any])
        #expect(downloaded["source"] as? String == origin)
        #expect(downloadedPackage["sourceUrl"] as? String == "https://pets.example/pets/store-pet")
        let saved = try fixture.manifestData()
        try FileManager.default.removeItem(at: store.appendingPathComponent("spritesheet.png"))
        #expect(try fixture.invoke("store-pet", root: store, origin: origin, expectedSuccess: false) != 0)
        #expect(try fixture.manifestData() == saved)
    }
}

private final class AssetTestBundle: NSObject {}

private final class AssetFixture {
    let temporary: URL
    let source: URL
    let pet: URL
    let output: URL

    init() throws {
        temporary = FileManager.default.temporaryDirectory.appendingPathComponent("pet-asset-test-\(UUID().uuidString)")
        source = temporary.appendingPathComponent("source")
        pet = source.appendingPathComponent("assets/pets/fixture")
        output = temporary.appendingPathComponent("output")
        try FileManager.default.createDirectory(at: pet, withIntermediateDirectories: true)
    }

    deinit { try? FileManager.default.removeItem(at: temporary) }

    @discardableResult
    func invoke(_ selectedPet: String = "fixture", root: URL? = nil, origin: String? = nil,
                expectedSuccess: Bool = true, sourceLocation: SourceLocation = #_sourceLocation) throws -> Int32 {
        let process = Process()
        let testBundle = Bundle(for: AssetTestBundle.self)
        process.executableURL = testBundle.bundleURL.deletingLastPathComponent().appendingPathComponent("AssetImporter")
        var arguments = [(root ?? source).path, output.path, selectedPet, "fixture-commit"]
        if let origin { arguments.append(origin) }
        process.arguments = arguments
        process.standardOutput = FileHandle.nullDevice
        let errorPipe = Pipe()
        process.standardError = errorPipe
        try process.run()
        let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        if expectedSuccess {
            let diagnostic = String(decoding: errorData, as: UTF8.self)
            try #require(process.terminationStatus == 0, Comment(rawValue: diagnostic), sourceLocation: sourceLocation)
        }
        return process.terminationStatus
    }

    func writeObject(_ object: [String: Any], to url: URL) throws {
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]).write(to: url)
    }

    func manifestData() throws -> Data {
        try Data(contentsOf: output.appendingPathComponent("manifest.json"))
    }

    func manifest() throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: manifestData()) as? [String: Any])
    }

    func customPack() throws -> URL {
        let custom = temporary.appendingPathComponent("custom")
        try FileManager.default.createDirectory(at: custom, withIntermediateDirectories: true)
        try writeImage(width: 192, height: 208, to: custom.appendingPathComponent("pose.png"))
        try writeObject(["pet": "custom", "rest": "calm", "clips": [
            "calm": ["frames": ["pose.png"], "fps": 1.0, "role": "rest"],
            "bounce": ["frames": ["pose.png"], "fps": 2.0, "role": "celebrate", "max_repeats": 2],
            "snooze": ["frames": ["pose.png"], "fps": 1.0, "role": "sleep"],
        ]], to: custom.appendingPathComponent("manifest.json"))
        return custom
    }

    func writeSheet(height: Int = 2288, to url: URL? = nil) throws {
        try writeImage(width: 1536, height: height, to: url ?? pet.appendingPathComponent("spritesheet.png"))
    }

    func writeImage(width: Int, height: Int, to url: URL) throws {
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
        let provider = try #require(CGDataProvider(data: Data(pixels) as CFData))
        let image = try #require(CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let png = try #require(NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:]))
        try png.write(to: url)
    }

    func pixel(_ path: String) throws -> [Int] {
        let data = try Data(contentsOf: output.appendingPathComponent(path))
        let bitmap = try #require(NSBitmapImageRep(data: data))
        #expect(bitmap.pixelsWide == 192)
        #expect(bitmap.pixelsHigh == 208)
        var channels = [Int](repeating: 0, count: bitmap.samplesPerPixel)
        bitmap.getPixel(&channels, atX: 10, y: 10)
        try #require(channels.count >= 2)
        return channels
    }
}
