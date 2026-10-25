import AppKit
import PetCore

final class ConversationController {
    let session = ConversationSession()
    let panel: ConversationPanel
    let content = ConversationContent()
    private let anchor: () -> NSRect
    private let send: ([String: Any]) -> Void
    private let clock: () -> TimeInterval
    private var timer: Timer?
    private var monitors: [Any] = []
    private var observers: [NSObjectProtocol] = []
    private var wantsFocus = false
    private var roundsOrigin = NSPoint.zero
    var onVisibilityChange: (() -> Void)?
    var isVisible: Bool { session.current != nil }

    init(anchor: @escaping () -> NSRect, send: @escaping ([String: Any]) -> Void,
         clock: @escaping () -> TimeInterval = { ProcessInfo.processInfo.systemUptime }) {
        self.anchor = anchor
        self.send = send
        self.clock = clock
        panel = ConversationPanel(contentRect: NSRect(x: 0, y: 0, width: 380, height: 320),
                                  styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
        configurePanel()
        content.onAction = { [weak self] in self?.performAction($0) }
        content.onDismiss = { [weak self] in self?.dismiss() }
        panel.onDismiss = { [weak self] in self?.dismiss() }
        installMonitoring()
    }

    private func configurePanel() {
        panel.contentView = content
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.hidesOnDeactivate = false
        panel.isReleasedWhenClosed = false
        panel.isMovable = false
        panel.title = "Pet conversation"
        panel.setAccessibilityLabel("Pet conversation")
    }

    func requestFocus() { wantsFocus = true }

    @discardableResult
    func receive(_ event: PetEvent) -> Bool {
        if event.dismissConversation == true || event.conversation == nil && event.text != nil {
            dismiss(notify: false)
        }
        guard let conversation = event.conversation else { return true }
        let prior = session.current
        guard session.receive(conversation, at: clock()) else { return false }
        if prior?.mode == .rounds { roundsOrigin = content.scroll.contentView.bounds.origin }
        if prior?.id != conversation.id { roundsOrigin = .zero }
        if let prior = prior, prior.id != conversation.id {
            send(["kind": "renderer", "event": "interaction_state", "interaction_id": prior.id, "engaged": false])
        }
        content.render(conversation, pending: false)
        position()
        let origin = conversation.mode == .rounds ? roundsOrigin : .zero
        content.scroll.contentView.scroll(to: origin)
        content.scroll.reflectScrolledClipView(content.scroll.contentView)
        panel.orderFrontRegardless()
        if wantsFocus && conversation.persistent {
            panel.makeKeyAndOrderFront(nil)
            panel.selectNextKeyView(nil)
            wantsFocus = false
        }
        onVisibilityChange?()
        return true
    }

    func pulse() {
        guard let conversation = session.current else { return }
        let engaged = panel.frame.contains(NSEvent.mouseLocation) || panel.isKeyWindow
        if engaged != session.isEngaged {
            session.setEngaged(engaged, at: clock())
            send(["kind": "renderer", "event": "interaction_state", "interaction_id": conversation.id, "engaged": engaged])
        }
        session.advance(at: clock())
        if session.shouldExpire { dismiss(); return }
        position()
    }

    func dismiss(notify: Bool = true) {
        guard let conversation = session.dismiss() else { return }
        panel.orderOut(nil)
        wantsFocus = false
        roundsOrigin = .zero
        if notify {
            send(["kind": "renderer", "event": "interaction_dismissed",
                  "interaction_id": conversation.id, "revision": conversation.revision])
        }
        onVisibilityChange?()
    }

    func performAction(_ id: String) {
        guard let conversation = session.current, let action = session.beginAction(id) else { return }
        content.setPending(true)
        position()
        send(["kind": "renderer", "event": "interaction_action", "interaction_id": conversation.id,
              "revision": conversation.revision, "action_id": action.id])
        if action.kind == .dismiss { dismiss(notify: false) }
    }

    private func position() {
        guard session.current != nil else { return }
        let pet = anchor()
        let screen = NSScreen.screens.first { $0.frame.contains(NSPoint(x: pet.midX, y: pet.midY)) } ?? NSScreen.main
        guard let screen = screen else { return }
        let width = min(380, screen.visibleFrame.width - 24)
        let preferred = NSSize(width: width, height: content.preferredHeight(width: width))
        let frame = Self.frame(anchor: pet, visible: screen.visibleFrame, preferred: preferred)
        if panel.frame != frame { panel.setFrame(frame, display: false) }
        content.layoutContent()
    }

    static func frame(anchor: NSRect, visible: NSRect, preferred: NSSize) -> NSRect {
        let bounds = visible.insetBy(dx: 12, dy: 12)
        let width = min(preferred.width, bounds.width)
        let above = max(0, bounds.maxY - anchor.maxY - 10)
        let below = max(0, anchor.minY - bounds.minY - 10)
        let placeAbove = above >= preferred.height || above >= below
        let available = placeAbove ? above : below
        let height = min(preferred.height, max(180, available), bounds.height)
        let x = min(max(anchor.midX - width / 2, bounds.minX), bounds.maxX - width)
        let y = placeAbove ? anchor.maxY + 10 : anchor.minY - height - 10
        return NSRect(x: x, y: min(max(y, bounds.minY), bounds.maxY - height), width: width, height: height)
    }

    private func installMonitoring() {
        let mask: NSEvent.EventTypeMask = [.leftMouseDown, .rightMouseDown]
        if let monitor = NSEvent.addGlobalMonitorForEvents(matching: mask, handler: { [weak self] _ in self?.outsideClick(at: NSEvent.mouseLocation) }) {
            monitors.append(monitor)
        }
        if let monitor = NSEvent.addLocalMonitorForEvents(matching: mask, handler: { [weak self] event in
            self?.outsideClick(at: NSEvent.mouseLocation)
            return event
        }) { monitors.append(monitor) }
        let notifications = NSWorkspace.shared.notificationCenter
        observers.append(notifications.addObserver(forName: NSWorkspace.accessibilityDisplayOptionsDidChangeNotification,
                                                   object: nil, queue: .main) { [weak self] _ in self?.content.updateContrast() })
        let timer = Timer(timeInterval: 0.1, repeats: true) { [weak self] _ in self?.pulse() }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    func outsideClick(at pointer: NSPoint) {
        guard isVisible else { return }
        if !panel.frame.contains(pointer) && !anchor().contains(pointer) { dismiss() }
    }

    func stop() {
        timer?.invalidate()
        timer = nil
        for monitor in monitors { NSEvent.removeMonitor(monitor) }
        monitors.removeAll()
        for observer in observers { NSWorkspace.shared.notificationCenter.removeObserver(observer) }
        observers.removeAll()
        panel.orderOut(nil)
    }

    deinit { stop() }
}

final class ConversationPanel: NSPanel {
    var onDismiss: (() -> Void)?
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
    override func cancelOperation(_ sender: Any?) { onDismiss?() }
}
