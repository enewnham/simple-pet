import AppKit
import PetCore
import Testing
@testable import PetOverlay

@Suite(.serialized)
@MainActor
struct PetOverlayTests {
    @Test func speechPreservesUnicodeAndPetPosition() throws {
        try withFixture { fixture in
            let originalPet = fixture.view.petFrameInScreen
            try fixture.apply(["text": "Hello 🐱", "ttl": 60])
            #expect(fixture.text.string == "Hello 🐱")
            #expect(fixture.view.petFrameInScreen == originalPet)
            let shortHeight = fixture.scroll.frame.height
            try fixture.apply(["text": Self.longSpeech, "ttl": 60])
            #expect(fixture.text.string == Self.longSpeech)
            #expect(fixture.scroll.frame.height > shortHeight)
            #expect(fixture.text.frame.height > fixture.scroll.contentSize.height)
            #expect(fixture.view.petFrameInScreen == originalPet)
            fixture.text.scrollRangeToVisible(NSRange(location: (Self.longSpeech as NSString).length - 1, length: 1))
            #expect(fixture.scroll.documentVisibleRect.maxY >= fixture.text.frame.height - 2)
        }
    }

    @Test func activityPreservesScrollAndNewSpeechResetsIt() throws {
        try withFixture { fixture in
            try fixture.apply(["text": Self.longSpeech, "ttl": 60])
            fixture.text.scrollRangeToVisible(NSRange(location: (Self.longSpeech as NSString).length - 1, length: 1))
            let scrolledOrigin = fixture.scroll.documentVisibleRect.origin
            try #require(scrolledOrigin.y > 0)
            try fixture.apply(["state": "working"])
            #expect(fixture.scroll.documentVisibleRect.origin == scrolledOrigin)
            try fixture.apply(["text": "New speech", "ttl": 60])
            #expect(fixture.text.string == "New speech")
            #expect(fixture.scroll.documentVisibleRect.minY == 0)
        }
    }

    @Test func dismissingSpeechRestoresPetFootprintAndPosition() throws {
        try withFixture { fixture in
            let originalPet = fixture.view.petFrameInScreen
            try fixture.apply(["text": Self.longSpeech, "ttl": 60])
            try #require(!fixture.scroll.isHidden)
            try fixture.apply(["text": ""])
            #expect(fixture.scroll.isHidden)
            #expect(fixture.panel.frame.size == NSSize(width: 200, height: 160))
            #expect(fixture.view.petFrameInScreen == originalPet)
        }
    }

    @Test(arguments: Corner.allCases)
    func speechStaysOnScreenAtMonitorCorners(corner: Corner) throws {
        try withFixture { fixture in
            let visible = fixture.screen.visibleFrame
            fixture.panel.setFrameOrigin(corner.origin(in: visible))
            let originalPet = fixture.view.petFrameInScreen
            try fixture.apply(["text": Self.longSpeech, "ttl": 60])
            let rect = fixture.view.convert(fixture.scroll.frame.insetBy(dx: -8, dy: -6), to: nil)
            let bubble = rect.offsetBy(dx: fixture.panel.frame.minX, dy: fixture.panel.frame.minY)
            #expect(visible.contains(bubble))
            #expect(fixture.view.petFrameInScreen == originalPet)
        }
    }

    @Test func napsExposeSleepAndRestoreActivityWithoutMovingPet() throws {
        try withFixture { fixture in
            try fixture.apply(["state": "working"])
            let originalPet = fixture.view.petFrameInScreen
            #expect(fixture.view.accessibilityLabel() == "Pet: working. ")
            try fixture.apply(["sleeping": true])
            #expect(fixture.view.accessibilityLabel() == "Pet: sleeping. ")
            #expect(fixture.view.petFrameInScreen == originalPet)
            try fixture.apply(["sleeping": false])
            #expect(fixture.view.accessibilityLabel() == "Pet: working. ")
            #expect(fixture.view.petFrameInScreen == originalPet)
        }
    }

    @Test func clickingEndsInteractionBeforeRequestingSpeech() throws {
        try withFixture { fixture in
            var events: [String] = []
            fixture.view.onClick = { events.append("click") }
            fixture.view.onInteraction = { dragging, _ in events.append(dragging ? "begin" : "end") }
            let point = NSPoint(x: 100, y: 52)
            let originalPet = fixture.view.petFrameInScreen
            fixture.panel.events = [try fixture.mouseEvent(.leftMouseUp, at: point)]
            fixture.view.mouseDown(with: try fixture.mouseEvent(.leftMouseDown, at: point))
            #expect(events == ["begin", "end", "click"])
            #expect(fixture.view.petFrameInScreen == originalPet)
        }
    }

    @Test func draggingMovesPetWithoutRequestingSpeech() throws {
        try withFixture { fixture in
            var events: [String] = []
            fixture.view.onClick = { events.append("click") }
            fixture.view.onInteraction = { dragging, _ in events.append(dragging ? "begin" : "end") }
            let point = NSPoint(x: 100, y: 52)
            let dragPoint = NSPoint(x: point.x - 30, y: point.y - 10)
            let originalPet = fixture.view.petFrameInScreen
            fixture.panel.events = [try fixture.mouseEvent(.leftMouseDragged, at: dragPoint),
                                    try fixture.mouseEvent(.leftMouseUp, at: dragPoint)]
            fixture.view.mouseDown(with: try fixture.mouseEvent(.leftMouseDown, at: point))
            #expect(events == ["begin", "end"])
            #expect(fixture.view.petFrameInScreen.origin == NSPoint(x: originalPet.minX - 30, y: originalPet.minY - 10))
        }
    }

    enum Corner: CaseIterable, Sendable {
        case bottomLeft, bottomRight, topLeft, topRight

        func origin(in visible: NSRect) -> NSPoint {
            switch self {
            case .bottomLeft: NSPoint(x: visible.minX, y: visible.minY)
            case .bottomRight: NSPoint(x: visible.maxX - 200, y: visible.minY)
            case .topLeft: NSPoint(x: visible.minX, y: visible.maxY - 160)
            case .topRight: NSPoint(x: visible.maxX - 200, y: visible.maxY - 160)
            }
        }
    }

    private static let longSpeech = String(repeating: "Readable speech 猫 🐱 with a line of details.\n", count: 300) + "Final sentence."

    private func withFixture(_ body: (Fixture) throws -> Void) throws {
        _ = NSApplication.shared
        let screen = try #require(NSScreen.main ?? NSScreen.screens.first, "Run overlay tests in a logged-in macOS desktop")
        let assets = FileManager.default.temporaryDirectory.appendingPathComponent("pet-overlay-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: assets, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: assets) }
        let bitmap = try #require(NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 1, pixelsHigh: 1,
                                                 bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
                                                 isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 4, bitsPerPixel: 32))
        bitmap.setColor(.black, atX: 0, y: 0)
        let png = try #require(bitmap.representation(using: .png, properties: [:]))
        try png.write(to: assets.appendingPathComponent("rest.png"))
        let manifest = "{\"pet\":\"quiet\",\"rest\":\"rest\",\"clips\":{\"rest\":{\"frames\":[\"rest.png\"],\"fps\":1,\"role\":\"rest\"}}}"
        try Data(manifest.utf8).write(to: assets.appendingPathComponent("manifest.json"))
        let model = PetModel()
        let view = try PetView(model: model, assets: assets)
        let panel = EventPanel(contentRect: NSRect(x: screen.visibleFrame.minX + 20, y: screen.visibleFrame.minY + 20,
                                               width: 200, height: 160),
                            styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
        panel.isReleasedWhenClosed = false
        panel.contentView = view
        defer { panel.close() }
        let scroll = try #require(view.subviews.compactMap { $0 as? NSScrollView }.first)
        let text = try #require(scroll.documentView as? NSTextView)
        try body(Fixture(screen: screen, model: model, view: view, panel: panel, scroll: scroll, text: text))
    }

    @MainActor
    private struct Fixture {
        let screen: NSScreen
        let model: PetModel
        let view: PetView
        let panel: EventPanel
        let scroll: NSScrollView
        let text: NSTextView

        func apply(_ payload: [String: Any]) throws {
            model.apply(try PetEvent.decode(JSONSerialization.data(withJSONObject: payload)))
        }

        func mouseEvent(_ type: NSEvent.EventType, at point: NSPoint) throws -> NSEvent {
            try #require(NSEvent.mouseEvent(with: type, location: point, modifierFlags: [],
                                           timestamp: ProcessInfo.processInfo.systemUptime,
                                           windowNumber: panel.windowNumber, context: nil, eventNumber: 0,
                                           clickCount: 1, pressure: 1))
        }
    }

    @MainActor
    private final class EventPanel: NSPanel {
        var events: [NSEvent] = []

        override func nextEvent(matching mask: NSEvent.EventTypeMask, until expiration: Date?,
                                inMode mode: RunLoop.Mode, dequeue deqFlag: Bool) -> NSEvent? {
            // Supply pointer events without nesting the test host's main run loop.
            guard let event = events.first else { return nil }
            if deqFlag { events.removeFirst() }
            return event
        }
    }
}
