import Foundation

public enum ConversationMode: String, Decodable {
    case thread, context, rounds, loading, empty
}

public enum ConversationActionKind: String, Decodable {
    case navigate, context, tomorrow, handled, more, rounds, back, dismiss, thread
}

public struct ConversationAction: Decodable, Equatable {
    public let id: String
    public let label: String
    public let kind: ConversationActionKind
}

public struct ConversationItem: Decodable, Equatable {
    public let id: String
    public let title: String
    public let text: String
    public let context: String
    public let actions: [ConversationAction]
}

public struct Conversation: Decodable, Equatable {
    public let id: String
    public let revision: Int
    public let mode: ConversationMode
    public let persistent: Bool
    public let ttl: TimeInterval
    public let text: String
    public let headline: String?
    public let items: [ConversationItem]
    public let actions: [ConversationAction]
    public let error: String?

    public var allActions: [ConversationAction] { actions + items.flatMap(\.actions) }

    func validate() throws {
        guard UUID(uuidString: id) != nil, revision >= 0, ttl.isFinite, ttl >= 0 else {
            throw ConversationError.invalid
        }
        let itemIDs = items.map(\.id)
        guard itemIDs.allSatisfy({ !$0.isEmpty }), Set(itemIDs).count == itemIDs.count else {
            throw ConversationError.invalid
        }
        let actionIDs = allActions.map(\.id)
        guard Set(actionIDs).count == actionIDs.count,
              allActions.allSatisfy({ !$0.id.isEmpty && !$0.label.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else {
            throw ConversationError.invalid
        }
    }

    private enum ConversationError: Error { case invalid }
}

public final class ConversationSession {
    public private(set) var current: Conversation?
    public private(set) var pendingAction: String?
    public private(set) var isEngaged = false
    private var dismissed = Set<String>()
    private var dismissedOrder: [String] = []
    private var remaining: TimeInterval = 0
    private var lastTick: TimeInterval = 0

    public init() {}

    @discardableResult
    public func receive(_ conversation: Conversation, at time: TimeInterval) -> Bool {
        guard !dismissed.contains(conversation.id) else { return false }
        if let current = current, current.id == conversation.id, conversation.revision <= current.revision {
            return false
        }
        if let current = current, current.id != conversation.id {
            rememberDismissed(current.id)
            isEngaged = false
        }
        current = conversation
        pendingAction = nil
        remaining = conversation.ttl
        lastTick = time
        return true
    }

    public func setEngaged(_ engaged: Bool, at time: TimeInterval) {
        advance(at: time)
        isEngaged = engaged
    }

    public func advance(at time: TimeInterval) {
        if !isEngaged && pendingAction == nil { remaining -= max(0, time - lastTick) }
        lastTick = time
    }

    public var shouldExpire: Bool {
        guard let current = current else { return false }
        return !current.persistent && !isEngaged && pendingAction == nil && remaining <= 0
    }

    public func beginAction(_ id: String) -> ConversationAction? {
        guard pendingAction == nil, let action = current?.allActions.first(where: { $0.id == id }) else { return nil }
        pendingAction = id
        return action
    }

    @discardableResult
    public func dismiss() -> Conversation? {
        guard let current = current else { return nil }
        rememberDismissed(current.id)
        self.current = nil
        pendingAction = nil
        isEngaged = false
        return current
    }

    private func rememberDismissed(_ id: String) {
        dismissed.insert(id)
        dismissedOrder.append(id)
        if dismissedOrder.count > 256 { dismissed.remove(dismissedOrder.removeFirst()) }
    }
}
