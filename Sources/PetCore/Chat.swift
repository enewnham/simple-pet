import Foundation

public enum ChatRole: String, Decodable {
    case user, assistant
}

public struct ChatMessage: Decodable, Equatable {
    public let id: String
    public let role: ChatRole
    public let text: String
    public let sources: [ConversationSource]?
    public let excerpt: String?
}

public struct ConversationSource: Decodable, Equatable {
    public let id: String
    public let title: String
    public let detail: String?
}

public enum ActivityCategory: String, Decodable {
    case responsibility, investigation, session, result, deferred, decision
}

public struct ActivityEntry: Decodable, Equatable {
    public let id: String
    public let category: ActivityCategory
    public let title: String
    public let text: String
    public let state: String
    public let reason: String?
    public let revisitAt: String?
    public let sources: [ConversationSource]
    public let actions: [ConversationAction]

    private enum CodingKeys: String, CodingKey {
        case id, category, title, text, state, reason, sources, actions
        case revisitAt = "revisit_at"
    }
}

public struct ConversationSubmission: Equatable {
    public let id: String
    public let conversationID: String
    public let text: String
}

public final class ConversationComposer {
    public var draft = ""
    public private(set) var submission: ConversationSubmission?
    public private(set) var isAccepted = false
    public private(set) var retryCommandID: String?
    public private(set) var validationError: String?
    private var conversationID: String?
    private var drafts: [String: String] = [:]

    public init() {}

    public func receive(_ conversation: Conversation) {
        guard let id = conversation.conversationID else { return }
        if conversationID != id {
            if let prior = conversationID { drafts[prior] = draft }
            draft = drafts[id] ?? ""
            submission = nil
            isAccepted = false
            retryCommandID = nil
            conversationID = id
        }
        let activeID = conversation.pendingMessageID ?? conversation.failedMessageID
        if submission == nil, draft.isEmpty, let activeID = activeID,
           let message = conversation.messages?.first(where: { $0.id == activeID }) {
            draft = message.text
            submission = ConversationSubmission(id: message.id, conversationID: id, text: message.text)
        }
        guard let submission = submission else { return }
        if conversation.rejectedMessageID == submission.id {
            self.submission = nil
            retryCommandID = nil
            isAccepted = false
            return
        }
        if let commandID = retryCommandID,
           conversation.acknowledgedCommandID != commandID,
           conversation.pendingMessageID != submission.id,
           conversation.failedMessageID == submission.id { return }
        let messages = conversation.messages ?? []
        isAccepted = messages.contains { $0.id == submission.id } || conversation.lastUserMessageID == submission.id
        if conversation.failedMessageID == submission.id {
            self.submission = nil
            isAccepted = false
            retryCommandID = nil
            return
        }
        let hasFinished = isAccepted && conversation.pendingMessageID != submission.id && conversation.failedMessageID != submission.id
        if hasFinished {
            if draft == submission.text { draft = "" }
            self.submission = nil
            isAccepted = false
            retryCommandID = nil
        }
    }

    public func submit(to conversation: Conversation) -> ConversationSubmission? {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        validationError = text.unicodeScalars.count > 16000 ? "Write a message of up to 16,000 characters." : nil
        guard conversation.mode == .chat, let id = conversation.conversationID,
              conversation.pendingMessageID == nil, submission == nil, !text.isEmpty, validationError == nil else { return nil }
        let submission = ConversationSubmission(id: UUID().uuidString, conversationID: id, text: text)
        self.submission = submission
        isAccepted = false
        retryCommandID = nil
        return submission
    }

    public func retry(_ conversation: Conversation) -> String? {
        guard conversation.pendingMessageID == nil, let id = conversation.failedMessageID,
              let conversationID = conversation.conversationID,
              let message = conversation.messages?.first(where: { $0.id == id }) else { return nil }
        submission = ConversationSubmission(id: id, conversationID: conversationID, text: message.text)
        isAccepted = false
        retryCommandID = UUID().uuidString
        return id
    }
}
