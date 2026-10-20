import CoreGraphics
import Foundation

public enum PetState: String, Codable, CaseIterable {
    case idle, working, error
}

public struct AnimationCommand: Decodable {
    public let id: String
    public let repeats: Int?
    public let commandID: Int
    enum CodingKeys: String, CodingKey { case id, repeats; case commandID = "command_id" }
}

public struct MotionCommand: Decodable {
    public let x: Double
    public let y: Double
    public let seconds: Double
    public let commandID: Int
    enum CodingKeys: String, CodingKey { case x, y, seconds; case commandID = "command_id" }
}

public struct PetEvent: Decodable {
    public let state: PetState?
    public let text: String?
    public let ttl: Double?
    public let animation: AnimationCommand?
    public let motion: MotionCommand?
    public let stopMotion: Bool?
    public let sleeping: Bool?
    public let deliveryID: String?
    enum CodingKeys: String, CodingKey {
        case state, text, ttl, animation, motion, sleeping
        case stopMotion = "stop_motion"
        case deliveryID = "delivery_id"
    }

    public static func decode(_ data: Data) throws -> PetEvent {
        let event = try JSONDecoder().decode(PetEvent.self, from: data)
        guard event.state != nil || event.text != nil || event.animation != nil || event.motion != nil || event.stopMotion == true || event.sleeping != nil else {
            throw EventError.invalid
        }
        if let ttl = event.ttl, !ttl.isFinite || ttl < 0 { throw EventError.invalid }
        if let deliveryID = event.deliveryID, UUID(uuidString: deliveryID) == nil { throw EventError.invalid }
        if let animation = event.animation {
            guard !animation.id.isEmpty, animation.commandID >= 0 else { throw EventError.invalid }
            if let repeats = animation.repeats, repeats <= 0 { throw EventError.invalid }
        }
        if let motion = event.motion {
            guard motion.x.isFinite, motion.y.isFinite, motion.seconds.isFinite,
                  motion.seconds > 0, motion.commandID >= 0 else { throw EventError.invalid }
        }
        return event
    }
    private enum EventError: Error { case invalid }
}

public final class PetModel {
    public private(set) var state: PetState = .idle
    public private(set) var text: String?
    public private(set) var sleeping = false
    public var onChange: (() -> Void)?
    private var expiration: DispatchWorkItem?
    private var acceptedDeliveries = Set<String>()
    private var deliveryOrder: [String] = []

    public init() {}

    @discardableResult
    public func apply(_ event: PetEvent) -> String? {
        dispatchPrecondition(condition: .onQueue(.main))
        if let state = event.state { self.state = state }
        if let sleeping = event.sleeping { self.sleeping = sleeping }
        if let deliveryID = event.deliveryID, acceptedDeliveries.contains(deliveryID) {
            onChange?()
            return deliveryID
        }
        var accepted: String?
        if let input = event.text {
            expiration?.cancel()
            expiration = nil
            text = input.trimmingCharacters(in: .whitespacesAndNewlines)
            if text?.isEmpty == true || event.ttl == 0 { text = nil }
            if text != nil {
                if let deliveryID = event.deliveryID {
                    acceptedDeliveries.insert(deliveryID)
                    deliveryOrder.append(deliveryID)
                    if deliveryOrder.count > 256 {
                        acceptedDeliveries.remove(deliveryOrder.removeFirst())
                    }
                    accepted = deliveryID
                }
                let work = DispatchWorkItem { [weak self] in
                    self?.text = nil
                    self?.onChange?()
                }
                expiration = work
                DispatchQueue.main.asyncAfter(deadline: .now() + (event.ttl ?? 15), execute: work)
            }
        }
        onChange?()
        return accepted
    }
    deinit { expiration?.cancel() }
}

public struct SpriteClip: Decodable {
    public let frames: [String]
    public let fps: Double
    public let role: String
    public let maxRepeats: Int?
    enum CodingKeys: String, CodingKey { case frames, fps, role; case maxRepeats = "max_repeats" }
}

public struct SpriteManifest: Decodable {
    public let pet: String
    public let rest: String
    public let clips: [String: SpriteClip]
    public let gaze: [String]?

    public static func load(directory: URL) throws -> SpriteManifest {
        let root = directory.standardizedFileURL.resolvingSymlinksInPath()
        let manifest = try JSONDecoder().decode(Self.self, from: Data(contentsOf: root.appendingPathComponent("manifest.json")))
        guard !manifest.pet.isEmpty, manifest.clips[manifest.rest] != nil else { throw PackError.invalid }
        for (id, clip) in manifest.clips {
            guard !id.isEmpty, !clip.frames.isEmpty, clip.fps.isFinite, clip.fps > 0,
                  clip.maxRepeats == nil || clip.maxRepeats! > 0 else { throw PackError.invalid }
            for frame in clip.frames {
                let file = root.appendingPathComponent(frame).standardizedFileURL.resolvingSymlinksInPath()
                guard !frame.hasPrefix("/"), file.path.hasPrefix(root.path + "/"),
                      file.pathExtension.lowercased() == "png",
                      FileManager.default.fileExists(atPath: file.path) else { throw PackError.invalid }
            }
        }
        if let gaze = manifest.gaze {
            guard gaze.count == 16, gaze.allSatisfy({ manifest.clips[$0] != nil }) else { throw PackError.invalid }
        }
        return manifest
    }
    private enum PackError: LocalizedError {
        case invalid
        var errorDescription: String? { "Sprite pack requires valid clips, frame paths, and a rest pose" }
    }
}

public struct AnimationPlayback {
    public let commandID: Int
    public let clipID: String
    private let count: Int
    private let fps: Double
    private let repeats: Int?
    private let start: Double
    public private(set) var finished = false

    public init(commandID: Int, clipID: String, count: Int, fps: Double, repeats: Int?, start: Double) {
        self.commandID = commandID
        self.clipID = clipID
        self.count = count
        self.fps = fps
        self.repeats = repeats
        self.start = start
    }

    public mutating func sample(at time: Double) -> (frame: Int, completed: Bool) {
        let elapsedFrames = max(0, floor((time - start) * fps))
        if let repeats = repeats, elapsedFrames >= Double(count) * Double(repeats) {
            let completed = !finished
            finished = true
            return (count - 1, completed)
        }
        return (Int(elapsedFrames.truncatingRemainder(dividingBy: Double(count))), false)
    }
}

public struct WindowMotion {
    public let commandID: Int
    private let start: CGPoint
    private let target: CGPoint
    private let started: Double
    private let seconds: Double

    public init(command: MotionCommand, origin: CGPoint, size: CGSize, bounds: CGRect, started: Double) {
        commandID = command.commandID
        start = origin
        target = Self.clamp(CGPoint(x: command.x, y: command.y), size: size, bounds: bounds)
        self.started = started
        seconds = command.seconds
    }

    public func sample(at time: Double, size: CGSize, bounds: CGRect) -> (origin: CGPoint, finished: Bool) {
        let progress = min(1, max(0, (time - started) / seconds))
        let point = CGPoint(x: start.x + (target.x - start.x) * progress, y: start.y + (target.y - start.y) * progress)
        return (Self.clamp(point, size: size, bounds: bounds), progress == 1)
    }

    public static func clamp(_ point: CGPoint, size: CGSize, bounds: CGRect) -> CGPoint {
        CGPoint(x: min(max(point.x, bounds.minX), max(bounds.minX, bounds.maxX - size.width)),
                y: min(max(point.y, bounds.minY), max(bounds.minY, bounds.maxY - size.height)))
    }
}
