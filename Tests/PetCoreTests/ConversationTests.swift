import Foundation
import Testing
@testable import PetCore

struct ConversationTests {
    @Test(arguments: ["thread", "context", "rounds", "loading", "empty"])
    func decodesConversationModes(mode: String) throws {
        let snapshot = try fixture(mode: mode)
        #expect(snapshot.mode.rawValue == mode)
        #expect(snapshot.allActions.map(\.id) == ["rounds", "open"])
    }

    @Test(arguments: [
        ["id": "invalid"], ["revision": -1], ["ttl": -1], ["mode": "unknown"],
        ["actions": [["id": "open", "label": "Repeated", "kind": "navigate"]]],
        ["actions": [["id": "", "label": "Open", "kind": "navigate"]]],
        ["actions": [["id": "close", "label": " ", "kind": "dismiss"]]]
    ] as [[String: Any]])
    func rejectsInvalidConversationFields(fields: [String: Any]) {
        #expect(throws: (any Error).self) { try fixture(fields: fields) }
    }

    @Test func decodesAnOptionalModelHeadline() throws {
        #expect(try fixture().headline == nil)
        #expect(try fixture(fields: ["headline": "Zagent tweak is wrapped up"]).headline == "Zagent tweak is wrapped up")
    }

    @Test func revisionsPreservePendingUntilAnUpdatedSnapshotArrives() throws {
        let session = ConversationSession()
        #expect(session.receive(try fixture(), at: 10))
        #expect(session.beginAction("open")?.kind == .navigate)
        #expect(session.beginAction("rounds") == nil)
        #expect(!session.receive(try fixture(), at: 11))
        #expect(session.pendingAction == "open")
        #expect(session.receive(try fixture(fields: ["revision": 2, "error": "Could not open the work."]), at: 12))
        #expect(session.pendingAction == nil)
        #expect(session.current?.error == "Could not open the work.")
    }

    @Test func engagementPausesTheRemainingLifetime() throws {
        let session = ConversationSession()
        session.receive(try fixture(fields: ["ttl": 5]), at: 10)
        session.setEngaged(true, at: 12)
        session.advance(at: 100)
        #expect(!session.shouldExpire)
        session.setEngaged(false, at: 100)
        session.advance(at: 102.9)
        #expect(!session.shouldExpire)
        session.advance(at: 103)
        #expect(session.shouldExpire)
    }

    @Test func dismissalSuppressesDeliveryRetriesAndLaterRevisions() throws {
        let session = ConversationSession()
        let snapshot = try fixture()
        session.receive(snapshot, at: 0)
        #expect(session.dismiss() == snapshot)
        #expect(!session.receive(snapshot, at: 1))
        #expect(!session.receive(try fixture(fields: ["revision": 2]), at: 2))
        #expect(session.current == nil)
        #expect(session.receive(try fixture(fields: ["id": UUID().uuidString]), at: 3))
    }

    @Test func supersededConversationsStayClosedWhenAnOldDeliveryArrives() throws {
        let session = ConversationSession()
        let first = try fixture()
        let second = try fixture(fields: ["id": UUID().uuidString])
        session.receive(first, at: 0)
        session.receive(second, at: 1)
        #expect(!session.receive(first, at: 2))
        #expect(session.current == second)
    }

    @Test func persistentConversationsAndPendingActionsSurviveIdleTime() throws {
        let session = ConversationSession()
        session.receive(try fixture(fields: ["persistent": true, "ttl": 0]), at: 0)
        session.advance(at: 100)
        #expect(!session.shouldExpire)
        session.receive(try fixture(fields: ["revision": 2, "ttl": 1]), at: 100)
        #expect(session.beginAction("open") != nil)
        session.advance(at: 200)
        #expect(!session.shouldExpire)
    }

    @Test @MainActor func conversationDeliveryAcknowledgesWithoutLegacySpeech() throws {
        let payload = try conversationPayload()
        let identifier = UUID().uuidString
        let event = try PetEvent.decode(JSONSerialization.data(withJSONObject: [
            "conversation": payload, "text": "Shared speech", "ttl": 15, "delivery_id": identifier
        ]))
        let model = PetModel()
        model.apply(try PetEvent.decode(Data(#"{"text":"Earlier speech"}"#.utf8)))
        #expect(model.text == "Earlier speech")
        #expect(model.apply(event) == identifier)
        #expect(model.text == nil)
        #expect(model.apply(event) == identifier)
        #expect(model.text == nil)
        #expect(try PetEvent.decode(Data(#"{"dismiss_conversation":true}"#.utf8)).dismissConversation == true)
    }

    @Test @MainActor func rejectedSpeechUpdatesActivityAndAcknowledgesOnlyAcceptedRetries() throws {
        let identifier = UUID().uuidString
        let snapshot = try conversationPayload()
        let event = try PetEvent.decode(JSONSerialization.data(withJSONObject: [
            "conversation": snapshot, "state": "working", "delivery_id": identifier
        ]))
        let model = PetModel()
        model.apply(try PetEvent.decode(Data(#"{"text":"Visible speech"}"#.utf8)))
        #expect(model.apply(event, acceptsSpeech: false) == nil)
        #expect(model.state == .working)
        #expect(model.text == "Visible speech")
        #expect(model.apply(event) == identifier)
        #expect(model.text == nil)
        #expect(model.apply(event, acceptsSpeech: false) == identifier)
    }

    private func fixture(mode: String = "thread", fields: [String: Any] = [:]) throws -> Conversation {
        var payload = try conversationPayload()
        payload["mode"] = mode
        payload.merge(fields) { _, new in new }
        let event = try PetEvent.decode(JSONSerialization.data(withJSONObject: ["conversation": payload]))
        return try #require(event.conversation)
    }

    private func conversationPayload() throws -> [String: Any] {
        let data = Data(#"{"id":"79A3412C-28E2-4F93-800F-7C46506A49F9","revision":1,"mode":"thread","persistent":false,"ttl":5,"text":"Your work has a thread.","items":[{"id":"work","title":"Small improvement","text":"Ready to review.","context":"The build passed.","actions":[{"id":"open","label":"Open work","kind":"navigate"}]}],"actions":[{"id":"rounds","label":"Make the rounds","kind":"rounds"}],"error":null}"#.utf8)
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
