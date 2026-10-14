import AppKit
import Foundation

struct AssetError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

struct Animation {
    let id: String
    let role: String
    let row: Int
    let count: Int
    let duration: Double
}

let animations = [
    Animation(id: "idle", role: "rest", row: 0, count: 6, duration: 5.5),
    Animation(id: "running-right", role: "walk-right", row: 1, count: 8, duration: 1.06),
    Animation(id: "running-left", role: "walk-left", row: 2, count: 8, duration: 1.06),
    Animation(id: "waving", role: "greeting", row: 3, count: 4, duration: 0.7),
    Animation(id: "jumping", role: "celebrate", row: 4, count: 5, duration: 0.84),
    Animation(id: "failed", role: "error", row: 5, count: 8, duration: 1.22),
    Animation(id: "waiting", role: "attention", row: 6, count: 6, duration: 1.01),
    Animation(id: "running", role: "work", row: 7, count: 6, duration: 0.82),
    Animation(id: "review", role: "attention", row: 8, count: 6, duration: 1.03),
]
let aliases: [String: String] = [
    "idle": "rest", "rest": "rest", "thinking": "attention", "alert": "attention",
    "ponder": "attention", "review": "attention", "waiting": "attention",
    "executing": "work", "busy": "work", "work": "work", "run": "work",
    "walk": "work", "running": "work", "success": "celebrate", "happy": "celebrate",
    "celebrate": "celebrate", "celebrating": "celebrate", "jumping": "celebrate",
    "error": "error", "shock": "error", "down": "error", "failed": "error",
    "waving": "greeting", "running-left": "walk-left", "running-right": "walk-right",
    "sleep": "sleep", "sleeping": "sleep", "nap": "sleep",
]
let roles = Set(["rest", "sleep", "work", "attention", "celebrate", "error", "greeting", "walk-left", "walk-right", "gaze", "gesture"])
let manager = FileManager.default

func decodeImage(_ url: URL) throws -> CGImage {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw AssetError("Cannot decode image: \(url.path)")
    }
    return image
}

func writePNG(_ image: CGImage, to url: URL) throws {
    guard let data = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:]) else {
        throw AssetError("Cannot encode PNG: \(url.path)")
    }
    try data.write(to: url)
}

func readObject(_ url: URL) throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
        throw AssetError("Expected a JSON object: \(url.path)")
    }
    return object
}

func safeName(_ name: String) -> Bool {
    !name.isEmpty && !name.contains("/") && !name.contains("\\") && name != "." && name != ".."
}

func resolveFrame(_ path: String, in source: URL) throws -> URL {
    let root = source.resolvingSymlinksInPath().standardizedFileURL
    let frame = source.appendingPathComponent(path).resolvingSymlinksInPath().standardizedFileURL
    guard !path.hasPrefix("/"), frame.path.hasPrefix(root.path + "/") else {
        throw AssetError("Frame path escapes the pet directory: \(path)")
    }
    return frame
}

func saveClip(_ id: String, images: [CGImage], fps: Double, role: String, staging: URL,
              maxRepeats: Int? = nil) throws -> [String: Any] {
    guard safeName(id), !images.isEmpty, fps.isFinite, fps > 0, roles.contains(role),
          maxRepeats == nil || maxRepeats! > 0 else {
        throw AssetError("Invalid clip: \(id)")
    }
    let relative = "clips/\(id)"
    let directory = staging.appendingPathComponent(relative)
    try manager.createDirectory(at: directory, withIntermediateDirectories: true)
    var paths: [String] = []
    for (index, image) in images.enumerated() {
        let name = String(format: "%03d.png", index)
        try writePNG(image, to: directory.appendingPathComponent(name))
        paths.append("\(relative)/\(name)")
    }
    var clip: [String: Any] = ["frames": paths, "fps": fps, "role": role]
    if let maxRepeats { clip["max_repeats"] = maxRepeats }
    return clip
}

func validateExtras(_ metadata: [String: Any], clips: [String: [String: Any]]) throws {
    if let gazeValue = metadata["gaze"] {
        guard let gaze = gazeValue as? [String], gaze.count == 16,
              gaze.allSatisfy({ clips[$0]?["role"] as? String == "gaze" }) else {
            throw AssetError("Gaze must reference sixteen gaze clips in clockwise order from up.")
        }
    }
    if let anchorValue = metadata["gazeAnchor"] {
        guard let anchor = anchorValue as? [String: Double], let x = anchor["x"], let y = anchor["y"],
              x.isFinite, y.isFinite, (0...1).contains(x), (0...1).contains(y) else {
            throw AssetError("gazeAnchor must contain normalized x and y coordinates.")
        }
    }
}

func extract(source: URL, staging: URL) throws -> [String: Any] {
    let manifestURL = source.appendingPathComponent("manifest.json")
    if manager.fileExists(atPath: manifestURL.path) {
        let manifest = try readObject(manifestURL)
        guard let declared = manifest["clips"] as? [String: [String: Any]],
              let rest = manifest["rest"] as? String, !declared.isEmpty else {
            throw AssetError("A custom manifest must declare clips and a rest clip.")
        }
        var clips: [String: [String: Any]] = [:]
        for (id, clip) in declared {
            guard let paths = clip["frames"] as? [String], let fps = clip["fps"] as? Double,
                  let role = clip["role"] as? String else { throw AssetError("Invalid clip: \(id)") }
            var limit: Int?
            if let value = clip["max_repeats"] {
                guard let number = value as? NSNumber, number.doubleValue == Double(number.intValue), number.intValue > 0 else {
                    throw AssetError("max_repeats must be a positive integer: \(id)")
                }
                limit = number.intValue
            }
            let images = try paths.map { try decodeImage(resolveFrame($0, in: source)) }
            clips[id] = try saveClip(id, images: images, fps: fps, role: role, staging: staging, maxRepeats: limit)
        }
        guard clips[rest]?["role"] as? String == "rest" else { throw AssetError("The rest clip must have the rest role.") }
        try validateExtras(manifest, clips: clips)
        var result = manifest
        result["clips"] = clips
        return result
    }

    let petURL = source.appendingPathComponent("pet.json")
    let pet = manager.fileExists(atPath: petURL.path) ? try readObject(petURL) : [:]
    let sheetURL: URL?
    if let relative = pet["spritesheetPath"] as? String {
        sheetURL = try resolveFrame(relative, in: source)
    } else {
        sheetURL = ["spritesheet.webp", "spritesheet.png", "default-pet-spritesheet.webp"]
            .map { source.appendingPathComponent($0) }.first { manager.fileExists(atPath: $0.path) }
    }
    var clips: [String: [String: Any]] = [:]
    var result: [String: Any] = ["rest": "idle"]
    if let sheetURL {
        let image = try decodeImage(sheetURL)
        if let value = pet["spriteVersionNumber"] {
            guard let number = value as? Int, number == 2 else { throw AssetError("spriteVersionNumber must be 2 when present.") }
        }
        guard image.width == 1536, image.height == 2288 else {
            throw AssetError("Expected a V2 1536x2288 sheet, got \(image.width)x\(image.height).")
        }
        for animation in animations {
            let images = try (0..<animation.count).map { index -> CGImage in
                guard let frame = image.cropping(to: CGRect(x: index * 192, y: animation.row * 208, width: 192, height: 208)) else {
                    throw AssetError("Cannot crop \(animation.id) frame \(index).")
                }
                return frame
            }
            clips[animation.id] = try saveClip(animation.id, images: images, fps: Double(animation.count) / animation.duration,
                                               role: animation.role, staging: staging)
        }
        var gaze: [String] = []
        for index in 0..<16 {
            let id = String(format: "gaze-%02d", index)
            let rect = CGRect(x: (index % 8) * 192, y: (9 + index / 8) * 208, width: 192, height: 208)
            guard let frame = image.cropping(to: rect) else { throw AssetError("Cannot crop \(id).") }
            clips[id] = try saveClip(id, images: [frame], fps: 1, role: "gaze", staging: staging, maxRepeats: 1)
            gaze.append(id)
        }
        result["gaze"] = gaze
        if let anchor = pet["gazeAnchor"] { result["gazeAnchor"] = anchor }
    } else {
        for entry in try manager.contentsOfDirectory(at: source, includingPropertiesForKeys: [.isDirectoryKey]) {
            let name = entry.pathExtension.lowercased() == "png" ? entry.deletingPathExtension().lastPathComponent : entry.lastPathComponent
            guard let role = aliases[name] else { continue }
            let candidates: [URL]
            if try entry.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true {
                candidates = try manager.contentsOfDirectory(at: entry, includingPropertiesForKeys: nil)
                    .filter { $0.pathExtension.lowercased() == "png" }
                    .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
            } else {
                candidates = entry.pathExtension.lowercased() == "png" ? [entry] : []
            }
            guard !candidates.isEmpty else { continue }
            guard clips[name] == nil else { throw AssetError("Duplicate static and animated clip: \(name)") }
            let root = source.resolvingSymlinksInPath().standardizedFileURL
            let images = try candidates.map { candidate -> CGImage in
                let frame = candidate.resolvingSymlinksInPath().standardizedFileURL
                guard frame.path.hasPrefix(root.path + "/") else { throw AssetError("Frame path escapes the pet directory: \(frame.path)") }
                return try decodeImage(frame)
            }
            clips[name] = try saveClip(name, images: images, fps: 8, role: role, staging: staging)
        }
        guard let rest = clips.keys.sorted().first(where: { clips[$0]?["role"] as? String == "rest" }) else {
            throw AssetError("The pet must supply an idle or rest PNG clip.")
        }
        result["rest"] = rest
    }
    try validateExtras(result, clips: clips)
    result["clips"] = clips
    return result
}

func run() throws {
    guard (5...6).contains(CommandLine.arguments.count) else { throw AssetError("Usage: import-assets SOURCE OUTPUT PET COMMIT [SOURCE_ORIGIN]") }
    let root = URL(fileURLWithPath: CommandLine.arguments[1]).standardizedFileURL
    let output = URL(fileURLWithPath: CommandLine.arguments[2]).standardizedFileURL
    let pet = CommandLine.arguments[3]
    let commit = CommandLine.arguments[4]
    let origin = CommandLine.arguments.count == 6 && !CommandLine.arguments[5].isEmpty ? CommandLine.arguments[5] : nil
    guard safeName(pet) else { throw AssetError("Use a pet directory name without slashes.") }
    let candidates = ["assets/pets", "apps/desktop/assets/pets", "pets"].map { root.appendingPathComponent($0).appendingPathComponent(pet) } + [root]
    guard let source = candidates.first(where: { candidate in
        manager.fileExists(atPath: candidate.path) &&
            (candidate != root || ["manifest.json", "pet.json", "idle", "idle.png", "rest", "rest.png", "spritesheet.webp", "spritesheet.png"]
                .contains { manager.fileExists(atPath: candidate.appendingPathComponent($0).path) })
    }) else { throw AssetError("Pet '\(pet)' is absent from the source.") }
    let resolvedSource = source.resolvingSymlinksInPath().standardizedFileURL
    let resolvedOutput = output.resolvingSymlinksInPath().standardizedFileURL
    guard !resolvedSource.path.hasPrefix(resolvedOutput.path + "/"), resolvedSource != resolvedOutput else {
        throw AssetError("Output must be separate from the source pet.")
    }
    let parent = output.deletingLastPathComponent()
    try manager.createDirectory(at: parent, withIntermediateDirectories: true)
    let staging = parent.appendingPathComponent(".pet-assets-\(UUID().uuidString)")
    let backup = parent.appendingPathComponent(".pet-backup-\(UUID().uuidString)")
    try manager.createDirectory(at: staging, withIntermediateDirectories: true)
    defer { try? manager.removeItem(at: staging) }
    var metadata = try extract(source: source, staging: staging)
    metadata["pet"] = pet
    let packageURL = source.appendingPathComponent("pet.json")
    let package = manager.fileExists(atPath: packageURL.path) ? try readObject(packageURL) : [:]
    if !package.isEmpty { metadata["package"] = package }
    metadata["source"] = origin ?? metadata["source"] ?? package["sourceUrl"] ?? package["sourceURL"] ?? package["source"] ??
        (source == root ? root.path : "https://github.com/OpenPetsHQ/openpets")
    metadata["commit"] = commit
    metadata["sourcePath"] = source == root ? "." : String(source.path.dropFirst(root.path.count + 1))
    let json = try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys])
    try json.write(to: staging.appendingPathComponent("manifest.json"))
    let generated = Set(["clips", "manifest.json"])
    if manager.fileExists(atPath: output.path) {
        for entry in try manager.contentsOfDirectory(at: output, includingPropertiesForKeys: nil) where !generated.contains(entry.lastPathComponent) {
            try manager.copyItem(at: entry, to: staging.appendingPathComponent(entry.lastPathComponent))
        }
        try manager.moveItem(at: output, to: backup)
    }
    do { try manager.moveItem(at: staging, to: output) }
    catch {
        if manager.fileExists(atPath: backup.path) { try? manager.moveItem(at: backup, to: output) }
        throw error
    }
    if manager.fileExists(atPath: backup.path) { try manager.removeItem(at: backup) }
    let clips = metadata["clips"] as! [String: [String: Any]]
    for id in clips.keys.sorted() { print("\(id): \((clips[id]!["frames"] as! [String]).count) frames (\(clips[id]!["role"]!))") }
    print("Assets written to \(output.path)")
}

do { try run() }
catch { fputs("Asset import failed: \(error)\n", stderr); exit(1) }
