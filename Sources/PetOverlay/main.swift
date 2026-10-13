import AppKit
import Darwin
import PetCore

struct Options {
    var socket = "/tmp/pet.sock"
    var daemonSocket = "/tmp/pet-daemon.sock"
    var assets: URL
    var pointerRadius = 100.0

    init() throws {
        assets = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("assets")
        var arguments = Array(CommandLine.arguments.dropFirst())
        while !arguments.isEmpty {
            let option = arguments.removeFirst()
            if option == "--help" { print(Self.usage); exit(0) }
            guard !arguments.isEmpty else { throw OptionError.invalid }
            let value = arguments.removeFirst()
            switch option {
            case "--socket": socket = value
            case "--daemon-socket": daemonSocket = value
            case "--assets": assets = URL(fileURLWithPath: value)
            case "--pointer-radius":
                guard let radius = Double(value), radius.isFinite, radius >= 0 else { throw OptionError.invalid }
                pointerRadius = radius
            default: throw OptionError.invalid
            }
        }
    }
    static let usage = "Usage: pet-overlay --assets PACK [--socket PATH] [--daemon-socket PATH] [--pointer-radius PIXELS]"
    private enum OptionError: LocalizedError {
        case invalid
        var errorDescription: String? { Options.usage }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private let options: Options
    private let model = PetModel()
    private var panel: NSPanel?
    private var view: PetView?
    private var server: SocketServer?
    private var sender: SocketSender?
    private var signals: [DispatchSourceSignal] = []
    private var statusTimer: Timer?
    private var motionTimer: Timer?
    private var motion: WindowMotion?
    private var latestMotion = -1
    private var dragging = false
    private var menu = false
    private var previousPointer = NSEvent.mouseLocation
    private var pointerMovedAt = 0.0
    private var lastStatus: Data?
    private var lastStatusTime = 0.0

    init(options: Options) { self.options = options }

    func applicationDidFinishLaunching(_ notification: Notification) {
        do {
            let view = try PetView(model: model, assets: options.assets)
            self.view = view
            sender = try SocketSender(path: options.daemonSocket)
            server = try SocketServer(path: options.socket) { [weak self] event in self?.receive(event) }
            let window = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 200, height: 160),
                                 styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
            window.isOpaque = false
            window.backgroundColor = .clear
            window.hasShadow = false
            window.level = .floating
            window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
            window.isMovableByWindowBackground = true
            window.hidesOnDeactivate = false
            window.isReleasedWhenClosed = false
            window.contentView = view
            if let screen = NSScreen.main ?? NSScreen.screens.first {
                let visible = screen.visibleFrame
                window.setFrameOrigin(NSPoint(x: visible.maxX - 220, y: visible.minY + 20))
            }
            panel = window
            view.onPlayback = { [weak self] id, result in self?.complete("animation", id: id, result: result) }
            view.onModelChange = { [weak self] in
                self?.stopIfPaused()
                self?.status(force: true)
            }
            view.onInteraction = { [weak self] dragging, menu in
                self?.dragging = dragging
                self?.menu = menu
                self?.stopIfPaused()
                self?.status(force: true)
            }
            window.orderFrontRegardless()
            let timer = Timer(timeInterval: 0.1, repeats: true) { [weak self] _ in
                self?.stopIfPaused()
                self?.status()
            }
            RunLoop.main.add(timer, forMode: .common)
            statusTimer = timer
            status(force: true)
            for number in [SIGINT, SIGTERM] {
                signal(number, SIG_IGN)
                let source = DispatchSource.makeSignalSource(signal: number, queue: .main)
                source.setEventHandler { NSApp.terminate(nil) }
                source.resume()
                signals.append(source)
            }
        } catch {
            fputs("pet-overlay: \(error.localizedDescription)\n", stderr)
            server?.stop()
            exit(EXIT_FAILURE)
        }
    }

    private var screenBounds: CGRect {
        (panel?.screen ?? NSScreen.main ?? NSScreen.screens.first)?.visibleFrame ?? .zero
    }

    private var paused: Bool {
        guard let panel = panel else { return true }
        let pointer = NSEvent.mouseLocation
        let rect = panel.frame
        let dx = pointer.x - rect.midX
        let dy = pointer.y - (rect.minY + 52)
        return dragging || menu || model.text != nil || model.state == .error || hypot(dx, dy) <= options.pointerRadius
    }

    private func receive(_ event: PetEvent) {
        model.apply(event)
        if event.stopMotion == true { stopMotion() }
        if let animation = event.animation { view?.play(animation) }
        if let command = event.motion, command.commandID > latestMotion {
            latestMotion = command.commandID
            stopMotion()
            guard !paused, let panel = panel else {
                complete("motion", id: command.commandID, result: "interrupted")
                return
            }
            motion = WindowMotion(command: command, origin: panel.frame.origin, size: panel.frame.size,
                                  bounds: screenBounds, started: ProcessInfo.processInfo.systemUptime)
            let timer = Timer(timeInterval: 1 / 30, repeats: true) { [weak self] _ in self?.advanceMotion() }
            RunLoop.main.add(timer, forMode: .common)
            motionTimer = timer
        }
    }

    private func advanceMotion() {
        guard !paused else { stopMotion(); return }
        guard let motion = motion, let panel = panel else { return }
        let sample = motion.sample(at: ProcessInfo.processInfo.systemUptime, size: panel.frame.size, bounds: screenBounds)
        panel.setFrameOrigin(sample.origin)
        if sample.finished {
            self.motion = nil
            motionTimer?.invalidate()
            motionTimer = nil
            complete("motion", id: motion.commandID, result: "finished")
            status(force: true)
        }
    }

    private func stopIfPaused() { if paused { stopMotion() } }

    private func stopMotion() {
        guard let motion = motion else { return }
        self.motion = nil
        motionTimer?.invalidate()
        motionTimer = nil
        complete("motion", id: motion.commandID, result: "interrupted")
    }

    private func complete(_ event: String, id: Int, result: String) {
        sender?.send(["kind": "renderer", "event": event, "command_id": id, "result": result])
    }

    private func status(force: Bool = false) {
        guard let panel = panel else { return }
        let now = ProcessInfo.processInfo.systemUptime
        let pointer = NSEvent.mouseLocation
        if pointer != previousPointer { pointerMovedAt = now }
        previousPointer = pointer
        let bounds = screenBounds
        let payload: [String: Any] = [
            "kind": "renderer", "event": "status", "x": panel.frame.minX, "y": panel.frame.minY,
            "width": panel.frame.width, "height": panel.frame.height,
            "screen": ["x": bounds.minX, "y": bounds.minY, "width": bounds.width, "height": bounds.height],
            "pointer": ["x": pointer.x, "y": pointer.y, "moving": now - pointerMovedAt < 0.3],
            "dragging": dragging, "menu": menu, "speech": model.text != nil
        ]
        let encoded = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
        if now - lastStatusTime >= 0.1 && (force || encoded != lastStatus || now - lastStatusTime >= 1) {
            sender?.send(payload)
            lastStatus = encoded
            lastStatusTime = now
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        statusTimer?.invalidate()
        stopMotion()
        server?.stop()
    }
}

do {
    let options = try Options()
    let application = NSApplication.shared
    let delegate = AppDelegate(options: options)
    application.setActivationPolicy(.accessory)
    application.delegate = delegate
    withExtendedLifetime(delegate) { application.run() }
} catch {
    fputs("pet-overlay: \(error.localizedDescription)\n", stderr)
    exit(EXIT_FAILURE)
}
