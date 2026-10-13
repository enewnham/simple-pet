import AppKit
import PetCore

@main
@MainActor
struct PetOverlayChecks {
    static func main() throws {
        _ = NSApplication.shared
        guard let screen = NSScreen.main ?? NSScreen.screens.first else {
            throw CheckFailure(message: "Run overlay checks in a logged-in macOS desktop")
        }
        let assets = FileManager.default.temporaryDirectory.appendingPathComponent("pet-overlay-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: assets, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: assets) }
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 1, pixelsHigh: 1,
                                      bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
                                      isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 4, bitsPerPixel: 32)!
        bitmap.setColor(.black, atX: 0, y: 0)
        try bitmap.representation(using: .png, properties: [:])!.write(to: assets.appendingPathComponent("rest.png"))
        let manifest = "{\"pet\":\"quiet\",\"rest\":\"rest\",\"clips\":{\"rest\":{\"frames\":[\"rest.png\"],\"fps\":1,\"role\":\"rest\"}}}"
        try Data(manifest.utf8).write(to: assets.appendingPathComponent("manifest.json"))
        let model = PetModel()
        let view = try PetView(model: model, assets: assets)
        let panel = NSPanel(contentRect: NSRect(x: screen.visibleFrame.minX + 20, y: screen.visibleFrame.minY + 20,
                                               width: 200, height: 160),
                            styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
        panel.isReleasedWhenClosed = false
        panel.contentView = view
        defer { panel.close() }
        let scroll = view.subviews.compactMap { $0 as? NSScrollView }.first!
        let text = scroll.documentView as! NSTextView
        let originalPet = view.petFrameInScreen
        try apply(["text": "Hello 🐱", "ttl": 60], to: model)
        try require(text.string == "Hello 🐱", "Short speech is complete")
        try require(view.petFrameInScreen == originalPet, "Showing speech preserves the pet position")
        let shortHeight = scroll.frame.height
        let speech = String(repeating: "Readable speech 猫 🐱 with a line of details.\n", count: 300) + "Final sentence."
        try apply(["text": speech, "ttl": 60], to: model)
        try require(text.string == speech, "Long Unicode speech is complete")
        try require(scroll.frame.height > shortHeight, "The bubble grows with the speech")
        try require(text.frame.height > scroll.contentSize.height, "Overflow remains in a scrollable document")
        try require(view.petFrameInScreen == originalPet, "Growing speech preserves the pet position")
        text.scrollRangeToVisible(NSRange(location: (speech as NSString).length - 1, length: 1))
        try require(scroll.documentVisibleRect.maxY >= text.frame.height - 2, "The last sentence can be scrolled into view")
        let scrolledOrigin = scroll.documentVisibleRect.origin
        try apply(["state": "working"], to: model)
        try require(scroll.documentVisibleRect.origin == scrolledOrigin, "State updates preserve the scroll position")
        try apply(["text": "New speech", "ttl": 60], to: model)
        try require(scroll.documentVisibleRect.minY == 0, "New speech starts at the beginning")
        try apply(["text": ""], to: model)
        try require(scroll.isHidden && panel.frame.size == NSSize(width: 200, height: 160), "Dismissed speech restores the pet footprint")
        try require(view.petFrameInScreen == originalPet, "Dismissing speech preserves the pet position")
        let visible = screen.visibleFrame
        for origin in [NSPoint(x: visible.minX, y: visible.minY),
                       NSPoint(x: visible.maxX - 200, y: visible.minY),
                       NSPoint(x: visible.minX, y: visible.maxY - 160),
                       NSPoint(x: visible.maxX - 200, y: visible.maxY - 160)] {
            panel.setFrameOrigin(origin)
            let pet = view.petFrameInScreen
            try apply(["text": speech, "ttl": 60], to: model)
            let rect = view.convert(scroll.frame.insetBy(dx: -8, dy: -6), to: nil)
            let bubble = rect.offsetBy(dx: panel.frame.minX, dy: panel.frame.minY)
            try require(visible.contains(bubble), "Speech stays within the monitor at each corner")
            try require(view.petFrameInScreen == pet, "Edge layout preserves the pet position")
            try apply(["text": ""], to: model)
        }
        var clicks = 0
        var interactions: [Bool] = []
        view.onClick = { clicks += 1 }
        view.onInteraction = { dragging, _ in interactions.append(dragging) }
        let spritePoint = NSPoint(x: 100, y: 52)
        let beforeClick = view.petFrameInScreen
        NSApp.postEvent(mouseEvent(.leftMouseUp, at: spritePoint, in: panel), atStart: true)
        view.mouseDown(with: mouseEvent(.leftMouseDown, at: spritePoint, in: panel))
        try require(clicks == 1, "Clicking the sprite requests speech")
        try require(interactions == [true, false], "A click ends interaction before requesting speech")
        try require(view.petFrameInScreen == beforeClick, "Clicking preserves the pet position")
        let dragPoint = NSPoint(x: spritePoint.x - 30, y: spritePoint.y - 10)
        NSApp.postEvent(mouseEvent(.leftMouseUp, at: dragPoint, in: panel), atStart: true)
        NSApp.postEvent(mouseEvent(.leftMouseDragged, at: dragPoint, in: panel), atStart: true)
        view.mouseDown(with: mouseEvent(.leftMouseDown, at: spritePoint, in: panel))
        try require(clicks == 1, "Dragging the sprite does not request speech")
        try require(view.petFrameInScreen.origin == NSPoint(x: beforeClick.minX - 30, y: beforeClick.minY - 10),
                        "Dragging repositions the pet")
        print("Pet overlay checks passed (speech, scrolling, monitor edges, clicks, and dragging).")
    }

    private static func apply(_ payload: [String: Any], to model: PetModel) throws {
        model.apply(try PetEvent.decode(JSONSerialization.data(withJSONObject: payload)))
    }

    private static func mouseEvent(_ type: NSEvent.EventType, at point: NSPoint, in window: NSWindow) -> NSEvent {
        NSEvent.mouseEvent(with: type, location: point, modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                          windowNumber: window.windowNumber, context: nil, eventNumber: 0, clickCount: 1, pressure: 1)!
    }

    private static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw CheckFailure(message: message) }
    }
}

private struct CheckFailure: Error { let message: String }
