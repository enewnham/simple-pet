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
    private var spriteRect = NSRect(x: 52, y: 4, width: 96, height: 96)
    private var speechRect: NSRect?
    private let speechScroll = NSScrollView()
    private let speechText = NSTextView()
    var onPlayback: ((Int, String) -> Void)?
    var onInteraction: ((Bool, Bool) -> Void)?
    var onClick: (() -> Void)?
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
        speechScroll.drawsBackground = false
        speechScroll.borderType = .noBorder
        speechScroll.hasVerticalScroller = true
        speechScroll.autohidesScrollers = true
        speechScroll.scrollerStyle = .overlay
        speechScroll.isHidden = true
        speechText.isEditable = false
        speechText.isSelectable = true
        speechText.drawsBackground = false
        speechText.font = .systemFont(ofSize: 13)
        speechText.textColor = .labelColor
        speechText.textContainerInset = .zero
        speechText.textContainer?.lineFragmentPadding = 0
        speechText.textContainer?.widthTracksTextView = true
        speechText.isVerticallyResizable = true
        speechText.isHorizontallyResizable = false
        speechScroll.documentView = speechText
        addSubview(speechScroll)
        model.onChange = { [weak self] in
            guard let self = self else { return }
            let activity = self.model.sleeping ? "sleeping" : self.model.state.rawValue
            self.setAccessibilityLabel("Pet: \(activity). \(self.model.text ?? "")")
            self.layoutSpeech()
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

    var petFrameInScreen: NSRect {
        let origin = window?.frame.origin ?? .zero
        return NSRect(x: origin.x + spriteRect.minX - 52, y: origin.y + spriteRect.minY - 4,
                      width: 200, height: 160)
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        layoutSpeech()
    }

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
        let image = frames[playback.clipID]![frameIndex]
        let scale = min(spriteRect.width / image.size.width, spriteRect.height / image.size.height)
        let size = NSSize(width: image.size.width * scale, height: image.size.height * scale)
        let rect = NSRect(x: spriteRect.midX - size.width / 2, y: spriteRect.minY, width: size.width, height: size.height)
        NSGraphicsContext.current?.imageInterpolation = .none
        image.draw(in: rect, from: .zero, operation: .sourceOver, fraction: 1)
        if model.sleeping {
            let sleep = NSAttributedString(string: "Zzz", attributes: [.font: NSFont.systemFont(ofSize: 15, weight: .medium),
                                                                       .foregroundColor: NSColor.secondaryLabelColor])
            sleep.draw(at: NSPoint(x: spriteRect.maxX - 12, y: spriteRect.maxY - 20))
        }
        if let rect = speechRect {
            let bubble = NSBezierPath(roundedRect: rect, xRadius: 9, yRadius: 9)
            NSColor.windowBackgroundColor.withAlphaComponent(0.96).setFill()
            bubble.fill()
            NSColor.separatorColor.withAlphaComponent(0.7).setStroke()
            bubble.lineWidth = 1
            bubble.stroke()
        }
    }

    private func layoutSpeech(spriteOrigin: NSPoint? = nil) {
        guard let window = window else { return }
        let origin = spriteOrigin ?? NSPoint(x: window.frame.minX + spriteRect.minX,
                                             y: window.frame.minY + spriteRect.minY)
        let sprite = NSRect(origin: origin, size: spriteRect.size)
        let petFrame = NSRect(x: sprite.minX - 52, y: sprite.minY - 4, width: 200, height: 160)
        var frame = petFrame
        var bubble: NSRect?
        let changedText = speechText.string != (model.text ?? "")
        if let text = model.text,
           let screen = NSScreen.screens.first(where: { $0.frame.contains(NSPoint(x: sprite.midX, y: sprite.midY)) }) ?? window.screen ?? NSScreen.main {
            let visible = screen.visibleFrame.insetBy(dx: 8, dy: 8)
            let attributed = NSAttributedString(string: text, attributes: [.font: speechText.font!])
            let measured = attributed.boundingRect(with: NSSize(width: min(344, visible.width - 16), height: .greatestFiniteMagnitude),
                                                   options: [.usesLineFragmentOrigin, .usesFontLeading])
            let width = min(360, visible.width, max(184, ceil(measured.width) + 16))
            if changedText { speechText.string = text }
            speechText.setFrameSize(NSSize(width: width - 16, height: speechText.frame.height))
            speechText.textContainer?.containerSize = NSSize(width: width - 16, height: .greatestFiniteMagnitude)
            let manager = speechText.layoutManager!
            let container = speechText.textContainer!
            manager.ensureLayout(for: container)
            let textHeight = ceil(manager.usedRect(for: container).height)
            let above = min(visible.height, max(0, visible.maxY - sprite.maxY - 8))
            let below = min(visible.height, max(0, sprite.minY - visible.minY - 8))
            let preferredHeight = max(30, textHeight + 12)
            let placeAbove = above >= preferredHeight || above >= below
            let height = min(preferredHeight, placeAbove ? above : below)
            let x = min(max(sprite.midX - width / 2, visible.minX), visible.maxX - width)
            let y = placeAbove ? sprite.maxY + 8 : sprite.minY - 8 - height
            bubble = NSRect(x: x, y: min(max(y, visible.minY), visible.maxY - height), width: width, height: height)
            frame = petFrame.union(bubble!)
            speechText.setFrameSize(NSSize(width: width - 16, height: max(textHeight, height - 12)))
        }
        // Preserve the sprite's screen position while the speech panel changes size.
        spriteRect.origin = NSPoint(x: sprite.minX - frame.minX, y: sprite.minY - frame.minY)
        window.setFrame(frame, display: false)
        speechRect = bubble.map { NSRect(x: $0.minX - frame.minX, y: $0.minY - frame.minY, width: $0.width, height: $0.height) }
        speechScroll.isHidden = speechRect == nil
        if let speechRect = speechRect {
            speechScroll.frame = speechRect.insetBy(dx: 8, dy: 6)
            if changedText { speechText.scrollToBeginningOfDocument(nil) }
        } else {
            speechText.string = ""
        }
        needsDisplay = true
    }

    override func mouseDown(with event: NSEvent) {
        guard let window = window else { return }
        let clickedSprite = spriteRect.contains(convert(event.locationInWindow, from: nil))
        var dragged = false
        onInteraction?(true, false)
        defer {
            onInteraction?(false, false)
            if clickedSprite && !dragged { onClick?() }
        }
        let initialPointer = window.convertPoint(toScreen: event.locationInWindow)
        let initialOrigin = NSPoint(x: window.frame.minX + spriteRect.minX, y: window.frame.minY + spriteRect.minY)
        while let next = window.nextEvent(matching: [.leftMouseDragged, .leftMouseUp], until: .distantFuture, inMode: .eventTracking, dequeue: true) {
            if next.type == .leftMouseUp { break }
            let pointer = window.convertPoint(toScreen: next.locationInWindow)
            if hypot(pointer.x - initialPointer.x, pointer.y - initialPointer.y) >= 4 { dragged = true }
            guard dragged else { continue }
            layoutSpeech(spriteOrigin: NSPoint(x: initialOrigin.x + pointer.x - initialPointer.x,
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
