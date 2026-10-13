import AppKit
import PetCore

final class PetView: NSView {
    private let model: PetModel
    let manifest: SpriteManifest
    private let frames: [String: [NSImage]]
    private var playback: AnimationPlayback
    private var frameIndex = 0
    private var timer: Timer?
    private var latestCommand = -1
    var onPlayback: ((Int, String) -> Void)?
    var onInteraction: ((Bool, Bool) -> Void)?
    var onModelChange: (() -> Void)?

    init(model: PetModel, assets: URL) throws {
        self.model = model
        manifest = try SpriteManifest.load(directory: assets)
        var loaded: [String: [NSImage]] = [:]
        for (id, clip) in manifest.clips {
            loaded[id] = try clip.frames.map { file in
                guard let image = NSImage(contentsOf: assets.appendingPathComponent(file)),
                      image.size.width > 0, image.size.height > 0 else {
                    throw NSError(domain: "PetSprite", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot decode sprite frame: \(file)"])
                }
                return image
            }
        }
        frames = loaded
        let rest = manifest.clips[manifest.rest]!
        playback = AnimationPlayback(commandID: -1, clipID: manifest.rest, count: rest.frames.count,
                                     fps: rest.fps, repeats: nil, start: ProcessInfo.processInfo.systemUptime)
        super.init(frame: NSRect(x: 0, y: 0, width: 200, height: 160))
        model.onChange = { [weak self] in
            guard let self = self else { return }
            self.setAccessibilityLabel("Pet: \(self.model.state.rawValue). \(self.model.text ?? "")")
            self.needsDisplay = true
            self.onModelChange?()
        }
        setAccessibilityElement(true)
        setAccessibilityRole(.image)
        scheduleAnimation()
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) is unsupported") }
    override var mouseDownCanMoveWindow: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    func play(_ command: AnimationCommand) {
        guard command.commandID > latestCommand else { return }
        latestCommand = command.commandID
        guard let clip = manifest.clips[command.id] else {
            onPlayback?(command.commandID, "interrupted")
            return
        }
        if playback.commandID >= 0 && !playback.finished {
            onPlayback?(playback.commandID, "interrupted")
        }
        let repeats = command.repeats.map { min($0, clip.maxRepeats ?? $0) }
        playback = AnimationPlayback(commandID: command.commandID, clipID: command.id,
                                     count: clip.frames.count, fps: clip.fps, repeats: repeats,
                                     start: ProcessInfo.processInfo.systemUptime)
        frameIndex = 0
        scheduleAnimation(finite: repeats != nil)
        needsDisplay = true
    }

    private func scheduleAnimation(finite: Bool = false) {
        timer?.invalidate()
        let clip = manifest.clips[playback.clipID]!
        guard clip.frames.count > 1 || finite else { return }
        let timer = Timer(timeInterval: 1 / min(clip.fps, 60), repeats: true) { [weak self] _ in
            guard let self = self else { return }
            let sample = self.playback.sample(at: ProcessInfo.processInfo.systemUptime)
            if self.frameIndex != sample.frame { self.needsDisplay = true }
            self.frameIndex = sample.frame
            if sample.completed {
                self.timer?.invalidate()
                self.timer = nil
                self.onPlayback?(self.playback.commandID, "finished")
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        let spriteRect = NSRect(x: 52, y: 4, width: 96, height: 96)
        let image = frames[playback.clipID]![frameIndex]
        let scale = min(spriteRect.width / image.size.width, spriteRect.height / image.size.height)
        let size = NSSize(width: image.size.width * scale, height: image.size.height * scale)
        let rect = NSRect(x: spriteRect.midX - size.width / 2, y: spriteRect.minY, width: size.width, height: size.height)
        NSGraphicsContext.current?.imageInterpolation = .none
        image.draw(in: rect, from: .zero, operation: .sourceOver, fraction: 1)
        if let text = model.text { drawSpeech(text) }
    }

    private func drawSpeech(_ text: String) {
        let style = NSMutableParagraphStyle()
        style.alignment = .center
        style.lineBreakMode = .byWordWrapping
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 12), .foregroundColor: NSColor.labelColor, .paragraphStyle: style
        ]
        let string = NSAttributedString(string: text, attributes: attributes)
        let measured = string.boundingRect(with: NSSize(width: 168, height: 1000), options: [.usesLineFragmentOrigin])
        let height = min(60, max(30, ceil(measured.height) + 12))
        let rect = NSRect(x: 8, y: bounds.height - height - 2, width: 184, height: height)
        let bubble = NSBezierPath(roundedRect: rect, xRadius: 9, yRadius: 9)
        NSColor.windowBackgroundColor.withAlphaComponent(0.93).setFill()
        bubble.fill()
        NSColor.separatorColor.withAlphaComponent(0.7).setStroke()
        bubble.lineWidth = 1
        bubble.stroke()
        string.draw(with: rect.insetBy(dx: 8, dy: 6), options: [.usesLineFragmentOrigin, .truncatesLastVisibleLine])
    }

    override func mouseDown(with event: NSEvent) {
        guard let window = window else { return }
        onInteraction?(true, false)
        defer { onInteraction?(false, false) }
        let initialPointer = NSEvent.mouseLocation
        let initialOrigin = window.frame.origin
        while let next = window.nextEvent(matching: [.leftMouseDragged, .leftMouseUp], until: .distantFuture, inMode: .eventTracking, dequeue: true) {
            if next.type == .leftMouseUp { break }
            let pointer = NSEvent.mouseLocation
            window.setFrameOrigin(NSPoint(x: initialOrigin.x + pointer.x - initialPointer.x,
                                          y: initialOrigin.y + pointer.y - initialPointer.y))
        }
    }

    override func rightMouseDown(with event: NSEvent) {
        onInteraction?(false, true)
        defer { onInteraction?(false, false) }
        let menu = NSMenu()
        menu.addItem(withTitle: "Quit Pet", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        NSMenu.popUpContextMenu(menu, with: event, for: self)
    }
    deinit { timer?.invalidate() }
}
