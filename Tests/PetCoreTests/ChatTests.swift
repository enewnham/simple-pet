import Foundation
import Testing
@testable import PetCore

struct ChatTests {
    @Test func passiveSnapshotsPreserveDraftAndDurableMessageIdentity() throws {
        let composer = ConversationComposer()
        let initial = try snapshot()
        composer.receive(initial)
        composer.draft = "Watch these checks"
        let submission = try #require(composer.submit(to: initial))
        #expect(composer.submit(to: initial) == nil)
        composer.receive(try snapshot(revision: 2, messages: [message(submission.id, text: submission.text)], pending: submission.id))
        #expect(composer.isAccepted)
        #expect(composer.submission?.id == submission.id)
        #expect(composer.draft == "Watch these checks")
        composer.draft = "Another question in progress"
        composer.receive(try snapshot(revision: 3, messages: [message(submission.id, text: submission.text), message("answer", role: "assistant", text: "I can watch them.")]))
        #expect(composer.submission == nil)
        #expect(composer.draft == "Another question in progress")
    }

    @Test func failedRepliesKeepInputAndRetryTheAcceptedMessage() throws {
        let composer = ConversationComposer()
        let initial = try snapshot()
        composer.receive(initial)
        composer.draft = "Explain the source"
        let submission = try #require(composer.submit(to: initial))
        let messages = [message(submission.id, text: submission.text)]
        let failed = try snapshot(revision: 2, messages: messages, failed: submission.id)
        composer.receive(failed)
        #expect(composer.draft == submission.text)
        #expect(composer.retry(failed) == submission.id)
        composer.receive(try snapshot(revision: 3, messages: messages, pending: submission.id))
        #expect(composer.submission?.id == submission.id)
        composer.receive(try snapshot(revision: 4, messages: messages + [message("answer", role: "assistant", text: "The report supports that claim.")]))
        #expect(composer.draft.isEmpty)
    }

    @Test func reopeningRestoresPendingInputWithAFreshPresentation() throws {
        let composer = ConversationComposer()
        let pending = try snapshot(messages: [message("pending", text: "Watch the build")], pending: "pending")
        composer.receive(pending)
        #expect(composer.draft == "Watch the build")
        #expect(composer.isAccepted)
        #expect(composer.submit(to: pending) == nil)
        let session = ConversationSession()
        session.receive(pending, at: 0)
        session.dismiss()
        let reopened = try snapshot(presentation: UUID().uuidString, messages: [message("pending", text: "Watch the build")], pending: "pending")
        #expect(session.receive(reopened, at: 1))
        #expect(session.current?.conversationID == pending.conversationID)
    }

    @Test func chatRejectsInvalidIdentityAndPendingRoles() throws {
        #expect(throws: (any Error).self) { try snapshot(messages: [message("pending", role: "assistant", text: "Answer")], pending: "pending") }
        #expect(throws: (any Error).self) { try snapshot(conversation: "invalid") }
    }

    private func message(_ id: String, role: String = "user", text: String) -> [String: String] {
        ["id": id, "role": role, "text": text]
    }

    private func snapshot(revision: Int = 1, presentation: String = UUID().uuidString,
                          conversation: String = "30B80556-CDA8-4097-A1B7-7A8722C011BC",
                          messages: [[String: String]] = [], pending: String? = nil, failed: String? = nil) throws -> Conversation {
        var payload: [String: Any] = ["id": presentation, "revision": revision, "mode": "chat", "persistent": true,
                                      "ttl": 0, "text": "", "items": [], "actions": [], "conversation_id": conversation, "messages": messages]
        if let pending = pending { payload["pending_message_id"] = pending }
        if let failed = failed { payload["failed_message_id"] = failed }
        return try #require(PetEvent.decode(JSONSerialization.data(withJSONObject: ["conversation": payload])).conversation)
    }
}
