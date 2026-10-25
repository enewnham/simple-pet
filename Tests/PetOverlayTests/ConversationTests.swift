import AppKit
import PetCore
import Testing
@testable import PetOverlay

@Suite(.serialized)
@MainActor
struct ConversationOverlayTests {
    @Test func actionsSendTheDisplayedIdentityAndDisableRepeatedActivation() throws {
        try withFixture { controller, packets in
            try controller.receive(event(mode: "thread"))
            #expect(controller.isVisible)
            #expect(!controller.panel.isKeyWindow)
            controller.performAction("open")
            controller.performAction("open")
            let sent = packets.values.filter { $0["event"] as? String == "interaction_action" }
            #expect(sent.count == 1)
            #expect(sent.first?["interaction_id"] as? String == Self.identifier)
            #expect(sent.first?["revision"] as? Int == 1)
            #expect(sent.first?["action_id"] as? String == "open")
            #expect(controller.session.pendingAction == "open")
            #expect(descendants(controller.content).compactMap { $0 as? NSButton }
                .filter { $0.identifier?.rawValue == "open" }.allSatisfy { !$0.isEnabled })
        }
    }

    @Test func updatedFailureRestoresActionsAndPreservesThePetAnchor() throws {
        try withFixture { controller, _ in
            let anchor = Self.anchor
            try controller.receive(event(mode: "thread"))
            controller.performAction("open")
            try controller.receive(event(mode: "thread", revision: 2, error: "Could not open this work. Try again."))
            #expect(controller.session.pendingAction == nil)
            let labels = descendants(controller.content).compactMap { ($0 as? NSTextField)?.stringValue }
            #expect(labels.contains("Could not open this work. Try again."))
            #expect(controller.panel.frame.intersects(anchor) == false)
        }
    }

    @Test func generatedHeadlinesLeadTheCardWithoutRepeatingTheRawTitle() throws {
        try withFixture { controller, _ in
            let headline = "Zellij may be keeping its old colors"
            try controller.receive(event(mode: "thread", headline: headline))
            let labels = descendants(controller.content).compactMap { ($0 as? NSTextField)?.stringValue }
            #expect(labels.filter { $0 == headline }.count == 1)
            #expect(!labels.contains("A small improvement"))
            #expect(!labels.contains("A moment together"))
            #expect(!descendants(controller.content).contains { ($0 as? NSBox)?.boxType == .separator })
        }
    }

    @Test func aHeadlineAndRemarkDisplayWithoutAnItem() throws {
        try withFixture { controller, _ in
            try controller.receive(event(mode: "thread", headline: "A small thought for later", itemCount: 0))
            let labels = descendants(controller.content).compactMap { ($0 as? NSTextField)?.stringValue }
            #expect(labels.contains("A small thought for later"))
            #expect(labels.contains("A little company."))
            #expect(controller.panel.frame.height < 220)
        }
    }

    @Test func shortCardsFitTheirContentAndLongContextScrollsWithinTheScreen() throws {
        try withFixture { controller, _ in
            try controller.receive(event(mode: "thread", headline: "Zagent tweak is wrapped up"))
            let shortHeight = controller.panel.frame.height
            #expect(shortHeight < 280)
            #expect(controller.content.scroll.documentView!.frame.height <= controller.content.scroll.contentSize.height + 1)
            let detail = String(repeating: "The saved report describes the findings and the next step.\n", count: 80)
            try controller.receive(event(mode: "context", revision: 2, headline: "Zagent tweak is wrapped up", detail: detail))
            #expect(controller.panel.frame.height > shortHeight)
            #expect(controller.content.scroll.documentView!.frame.height > controller.content.scroll.contentSize.height)
            #expect(controller.panel.screen!.visibleFrame.contains(controller.panel.frame))
            let labels = descendants(controller.content).compactMap { $0 as? NSTextField }
            for label in labels where !label.stringValue.isEmpty {
                #expect(label.frame.width > 0)
                #expect(label.frame.height >= label.font!.boundingRectForFont.height - 2)
            }
        }
    }

    @Test(arguments: ["more", "rounds"])
    func recentWorkMenuDispatchesTheSelectedBackendAction(action: String) throws {
        try withFixture { controller, packets in
            let actions = [["id": "more", "label": "Another thread", "kind": "more"],
                           ["id": "rounds", "label": "Make the rounds", "kind": "rounds"],
                           ["id": "dismiss", "label": "Dismiss", "kind": "dismiss"]]
            try controller.receive(event(mode: "thread", footerActions: actions))
            let popup = try #require(descendants(controller.content).compactMap { $0 as? NSPopUpButton }.first)
            #expect(popup.title == "Recent work")
            let item = try #require(popup.menu?.items.first { $0.identifier?.rawValue == action })
            #expect(NSApp.sendAction(try #require(item.action), to: item.target, from: item))
            #expect(packets.values.last?["action_id"] as? String == action)
            #expect(!popup.isHidden)
            #expect(!popup.isEnabled)
            #expect(!descendants(controller.content).compactMap { $0 as? NSButton }.contains { $0.title == "Dismiss" })
        }
    }

    @Test func outsideClicksDismissWhilePetAndPanelClicksKeepTheConversation() throws {
        try withFixture { controller, packets in
            try controller.receive(event(mode: "thread"))
            controller.outsideClick(at: NSPoint(x: Self.anchor.midX, y: Self.anchor.midY))
            #expect(controller.isVisible)
            let frame = controller.panel.frame
            controller.outsideClick(at: NSPoint(x: frame.midX, y: frame.midY))
            #expect(controller.isVisible)
            controller.outsideClick(at: NSPoint(x: frame.minX - 20, y: frame.maxY + 20))
            #expect(!controller.isVisible)
            #expect(packets.values.filter { $0["event"] as? String == "interaction_dismissed" }.count == 1)
        }
    }

    @Test func roundsRestoreScrollAfterReadingContext() throws {
        try withFixture { controller, _ in
            try controller.receive(event(mode: "rounds", long: true))
            let scroll = controller.content.scroll
            scroll.contentView.scroll(to: NSPoint(x: 0, y: 120))
            scroll.reflectScrolledClipView(scroll.contentView)
            let origin = scroll.contentView.bounds.origin
            try #require(origin.y > 0)
            try controller.receive(event(mode: "context", revision: 2, long: true))
            #expect(scroll.contentView.bounds.origin.y == 0)
            try controller.receive(event(mode: "rounds", revision: 3, long: true))
            #expect(scroll.contentView.bounds.origin == origin)
        }
    }

    @Test func legacySpeechClosesTheConversationAndDisplaysItsOwnText() throws {
        try withFixture { controller, packets in
            let snapshot = try event(mode: "thread")
            controller.receive(snapshot)
            let speech = try PetEvent.decode(Data(#"{"text":"Another little thought.","ttl":15}"#.utf8))
            controller.receive(speech)
            let model = PetModel()
            model.apply(speech)
            #expect(!controller.isVisible)
            #expect(!controller.panel.isVisible)
            #expect(model.text == "Another little thought.")
            #expect(packets.values.filter { $0["event"] as? String == "interaction_dismissed" }.isEmpty)
            controller.receive(snapshot)
            #expect(!controller.isVisible)
        }
    }

    @Test func rejectedConversationsAcknowledgeOnlyDeliveriesThatWereDisplayed() throws {
        try withFixture { controller, _ in
            let model = PetModel()
            let displayedID = UUID().uuidString
            let displayed = try event(mode: "thread", deliveryID: displayedID)
            let accepted = controller.receive(displayed)
            #expect(accepted)
            #expect(model.apply(displayed, acceptsSpeech: accepted) == displayedID)
            let stale = try event(mode: "thread", revision: 0, deliveryID: UUID().uuidString)
            let ignored = controller.receive(stale)
            #expect(!ignored)
            #expect(model.apply(stale, acceptsSpeech: ignored) == nil)
            controller.dismiss()
            let retried = controller.receive(displayed)
            #expect(!retried)
            #expect(model.apply(displayed, acceptsSpeech: retried) == displayedID)
            let unseen = try event(mode: "thread", revision: 2, deliveryID: UUID().uuidString)
            let rejected = controller.receive(unseen)
            #expect(!rejected)
            #expect(model.apply(unseen, acceptsSpeech: rejected) == nil)
            #expect(!controller.isVisible)
        }
    }

    @Test func escapeDismissesOnceAndDeliveryRetryStaysClosed() throws {
        try withFixture { controller, packets in
            let snapshot = try event(mode: "thread")
            controller.receive(snapshot)
            controller.panel.cancelOperation(nil)
            controller.panel.cancelOperation(nil)
            controller.receive(snapshot)
            #expect(!controller.isVisible)
            #expect(!controller.panel.isVisible)
            #expect(packets.values.filter { $0["event"] as? String == "interaction_dismissed" }.count == 1)
        }
    }

    @Test(arguments: ["loading", "empty", "thread", "context", "rounds"])
    func modesExposeNativeReadableContent(mode: String) throws {
        try withFixture { controller, _ in
            try controller.receive(event(mode: mode))
            let labels = descendants(controller.content).compactMap { ($0 as? NSTextField)?.stringValue }
            #expect(labels.contains("Ready to review 🐱"))
            #expect(controller.content.scroll.documentView != nil)
            #expect(controller.panel.accessibilityLabel() == "Pet conversation")
        }
    }

    @Test func anchoredPanelsStayWithinScreenCorners() {
        let screen = NSRect(x: -1000, y: 40, width: 1000, height: 760)
        for x in [screen.minX, screen.maxX - 96] {
            for y in [screen.minY, screen.maxY - 96] {
                let frame = ConversationController.frame(anchor: NSRect(x: x, y: y, width: 96, height: 96),
                                                         visible: screen, preferred: NSSize(width: 380, height: 460))
                #expect(screen.contains(frame))
            }
        }
    }

    private static let identifier = "79A3412C-28E2-4F93-800F-7C46506A49F9"
    private static let anchor = NSRect(x: 200, y: 100, width: 96, height: 96)

    private func withFixture(_ body: (ConversationController, Packets) throws -> Void) throws {
        _ = NSApplication.shared
        let packets = Packets()
        let controller = ConversationController(anchor: { Self.anchor }, send: { packets.values.append($0) }, clock: { 10 })
        defer { controller.stop() }
        try body(controller, packets)
    }

    private func event(mode: String, revision: Int = 1, error: String? = nil,
                       long: Bool = false, deliveryID: String? = nil, headline: String? = nil,
                       detail: String? = nil, footerActions: [[String: String]] = [], itemCount: Int = 1) throws -> PetEvent {
        let items = (0..<(long ? 12 : itemCount)).map { index in
            ["id": "work-\(index)", "title": "A small improvement", "text": "Ready to review 🐱",
             "context": detail ?? "The build passed. There is enough context to take the next step.",
             "actions": [["id": index == 0 ? "open" : "open-\(index)", "label": "Open work", "kind": "navigate"]]] as [String: Any]
        }
        var snapshot: [String: Any] = ["id": Self.identifier, "revision": revision, "mode": mode,
                                       "persistent": true, "ttl": 15, "text": "A little company.",
                                       "items": items, "actions": footerActions]
        if let error = error { snapshot["error"] = error }
        if let headline = headline { snapshot["headline"] = headline }
        var payload: [String: Any] = ["conversation": snapshot]
        if let deliveryID = deliveryID { payload["delivery_id"] = deliveryID }
        return try PetEvent.decode(JSONSerialization.data(withJSONObject: payload))
    }

    private func descendants(_ view: NSView) -> [NSView] {
        view.subviews.flatMap { [$0] + descendants($0) }
    }

    private final class Packets {
        var values: [[String: Any]] = []
    }
}
