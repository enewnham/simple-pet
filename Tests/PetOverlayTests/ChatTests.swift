import AppKit
import PetCore
import Testing
@testable import PetOverlay

@Suite(.serialized)
@MainActor
struct ChatOverlayTests {
    @Test func submittedTextKeepsItsIdentityAndDraftAcrossPassiveReplies() throws {
        _ = NSApplication.shared
        var packets: [[String: Any]] = []
        let controller = ConversationController(anchor: { NSRect(x: 200, y: 100, width: 96, height: 96) }, send: { packets.append($0) })
        defer { controller.stop() }
        controller.receive(try event())
        #expect(!controller.panel.isKeyWindow)
        controller.content.composer.stringValue = "Watch these checks"
        controller.content.onDraftChange?(controller.content.composer.stringValue)
        controller.submitMessage()
        controller.submitMessage()
        let submitted = packets.filter { $0["event"] as? String == "conversation_message" }
        #expect(submitted.count == 1)
        let messageID = try #require(submitted.first?["message_id"] as? String)
        #expect(submitted.first?["conversation_id"] as? String == Self.conversationID)
        #expect(submitted.first?["interaction_id"] as? String == Self.presentationID)
        controller.receive(try event(revision: 2, pending: messageID, messages: [["id": messageID, "role": "user", "text": "Watch these checks"]]))
        #expect(controller.content.composer.stringValue == "Watch these checks")
        controller.content.composer.stringValue = "Explain another finding"
        controller.content.onDraftChange?(controller.content.composer.stringValue)
        controller.receive(try event(revision: 3, messages: [["id": messageID, "role": "user", "text": "Watch these checks"],
                                                           ["id": "answer", "role": "assistant", "text": "I can watch those checks."]]))
        #expect(controller.content.composer.stringValue == "Explain another finding")
        #expect(!controller.panel.isKeyWindow)
        let labels = descendants(controller.content).compactMap { ($0 as? NSTextField)?.stringValue }
        #expect(labels.contains("I can watch those checks."))
    }

    @Test func ambientUpdatesPreserveAnOpenChatAndItsInput() throws {
        _ = NSApplication.shared
        let controller = ConversationController(anchor: { NSRect(x: 200, y: 100, width: 96, height: 96) }, send: { _ in })
        defer { controller.stop() }
        controller.receive(try event())
        controller.content.composer.stringValue = "Still typing"
        controller.content.onDraftChange?("Still typing")
        let speech = try PetEvent.decode(Data(#"{"text":"An ambient thought","ttl":15}"#.utf8))
        #expect(!controller.receive(speech))
        #expect(controller.isVisible)
        #expect(controller.content.composer.stringValue == "Still typing")
        #expect(controller.session.current?.conversationID == Self.conversationID)
    }

    @Test func lostRetryAndFinalSnapshotsRecoverThroughStableCommandsAndResync() throws {
        _ = NSApplication.shared
        var packets: [[String: Any]] = []
        var time: TimeInterval = 0
        let controller = ConversationController(anchor: { NSRect(x: 200, y: 100, width: 96, height: 96) },
                                                send: { packets.append($0) }, clock: { time })
        defer { controller.stop() }
        let messageID = UUID().uuidString
        let messages = [["id": messageID, "role": "user", "text": "Explain the evidence"]]
        let retry = [["id": "retry", "label": "Retry reply", "kind": "retry"]]
        controller.receive(try event(messages: messages, extra: ["failed_message_id": messageID, "actions": retry]))
        controller.performAction("retry")
        let first = try #require(packets.last)
        #expect(first["event"] as? String == "conversation_retry")
        #expect(!controller.composer.isAccepted)
        time = 1.1
        controller.pulse()
        let attempts = packets.filter { $0["event"] as? String == "conversation_retry" }
        #expect(attempts.count == 2)
        #expect(attempts[0]["command_id"] as? String == attempts[1]["command_id"] as? String)
        #expect(attempts[0]["message_id"] as? String == messageID)
        let commandID = try #require(first["command_id"] as? String)
        controller.receive(try event(revision: 2, pending: messageID, messages: messages,
                                     extra: ["acknowledged_command_id": commandID]))
        #expect(controller.composer.isAccepted)
        time = 3.2
        controller.pulse()
        #expect(packets.contains { $0["event"] as? String == "conversation_sync" })
        #expect(packets.filter { $0["event"] as? String == "conversation_retry" }.count == 2)
        controller.receive(try event(revision: 3, messages: messages + [["id": "answer", "role": "assistant", "text": "Here is the evidence."]],
                                     extra: ["acknowledged_command_id": commandID]))
        #expect(controller.composer.submission == nil)
        #expect(controller.content.composer.stringValue.isEmpty)
    }

    @Test func permanentRejectionKeepsTheDraftEditableAndStopsResending() throws {
        _ = NSApplication.shared
        var packets: [[String: Any]] = []
        var time: TimeInterval = 0
        let controller = ConversationController(anchor: { NSRect(x: 200, y: 100, width: 96, height: 96) },
                                                send: { packets.append($0) }, clock: { time })
        defer { controller.stop() }
        controller.receive(try event())
        controller.composer.draft = "Keep this question"
        controller.submitMessage()
        let messageID = try #require(controller.composer.submission?.id)
        controller.receive(try event(revision: 2, extra: ["rejected_message_id": messageID, "error": "Please edit that message."]))
        #expect(controller.composer.submission == nil)
        #expect(controller.content.composer.stringValue == "Keep this question")
        time = 3
        controller.pulse()
        #expect(packets.filter { $0["event"] as? String == "conversation_message" }.count == 1)
        controller.composer.draft = String(repeating: "x", count: 16001)
        controller.submitMessage()
        #expect(controller.composer.submission == nil)
        #expect(controller.composer.validationError != nil)
        controller.composer.draft = "A shorter question"
        controller.submitMessage()
        #expect(packets.filter { $0["event"] as? String == "conversation_message" }.count == 2)
    }

    @Test func activityShowsEvidenceAndKeepsTaskControlsAvailableDuringAReply() throws {
        _ = NSApplication.shared
        var packets: [[String: Any]] = []
        let controller = ConversationController(anchor: { NSRect(x: 200, y: 100, width: 96, height: 96) }, send: { packets.append($0) })
        defer { controller.stop() }
        let messageID = UUID().uuidString
        let entry: [String: Any] = ["id": "owned", "category": "session", "title": "Validate rollback", "text": "Expected result: validation evidence",
                                    "state": "running", "reason": "Harness report remains unverified", "revisit_at": "Tomorrow at 9",
                                    "sources": [["id": "meeting", "title": "Rollback discussion", "detail": "Source date: unknown. Origin: meeting-export."]],
                                    "actions": [["id": "stop", "label": "Stop this task", "kind": "stop"]]]
        controller.receive(try event(pending: messageID, messages: [["id": messageID, "role": "user", "text": "Explain the result"]],
                                     extra: ["mode": "activity", "activity": [entry], "actions": [["id": "pause", "label": "Pause coordination", "kind": "pause"]]]))
        #expect(!controller.panel.isKeyWindow)
        let labels = descendants(controller.content).compactMap { ($0 as? NSTextField)?.stringValue }
        #expect(labels.contains("Source date: unknown. Origin: meeting-export."))
        #expect(labels.contains("Harness report remains unverified"))
        #expect(labels.contains("Revisit: Tomorrow at 9"))
        let stop = try #require(descendants(controller.content).compactMap { $0 as? NSButton }.first { $0.title == "Stop this task" })
        #expect(stop.isEnabled)
        controller.performAction("stop")
        #expect(packets.last?["action_id"] as? String == "stop")
        #expect(controller.content.composer.stringValue == "Explain the result")
    }

    private static let conversationID = "30B80556-CDA8-4097-A1B7-7A8722C011BC"
    private static let presentationID = "79A3412C-28E2-4F93-800F-7C46506A49F9"

    private func event(revision: Int = 1, pending: String? = nil, messages: [[String: String]] = [], extra: [String: Any] = [:]) throws -> PetEvent {
        var snapshot: [String: Any] = ["id": Self.presentationID, "revision": revision, "mode": "chat",
                                       "persistent": true, "ttl": 0, "text": "", "items": [], "actions": [],
                                       "conversation_id": Self.conversationID, "messages": messages]
        if let pending = pending { snapshot["pending_message_id"] = pending }
        snapshot.merge(extra) { _, value in value }
        return try PetEvent.decode(JSONSerialization.data(withJSONObject: ["conversation": snapshot]))
    }

    private func descendants(_ view: NSView) -> [NSView] { [view] + view.subviews.flatMap(descendants) }
}
