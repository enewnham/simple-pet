import AppKit
import PetCore

final class ConversationContent: NSVisualEffectView {
    let scroll = NSScrollView()
    private let document = ConversationDocument()
    private let body = NSStackView()
    private let footer = NSStackView()
    private let close = ConversationButton(image: NSImage(systemSymbolName: "xmark", accessibilityDescription: "Close conversation")!,
                                           target: nil, action: nil)
    private let progress = NSProgressIndicator()
    private let pendingLabel = NSTextField(labelWithString: "Working…")
    private var heading: NSTextField?
    private var labels: [NSTextField] = []
    private var buttons: [NSButton] = []
    private var responses: [NSStackView] = []
    private var bodyWidth: NSLayoutConstraint!
    var onAction: ((String) -> Void)?
    var onDismiss: (() -> Void)?

    init() {
        super.init(frame: .zero)
        material = .popover
        blendingMode = .behindWindow
        state = .active
        wantsLayer = true
        layer?.cornerRadius = 16
        layer?.masksToBounds = true
        setAccessibilityLabel("Pet conversation")
        configureContent()
        configureFooter()
        close.target = self
        close.action = #selector(closeConversation)
        close.bezelStyle = .inline
        close.isBordered = false
        close.setAccessibilityLabel("Close conversation")
        addSubview(close)
        updateContrast()
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) is unsupported") }
    override var needsPanelToBecomeKey: Bool { true }

    func render(_ conversation: Conversation, pending: Bool) {
        buttons.removeAll()
        labels.removeAll()
        responses.removeAll()
        clear(body)
        clear(footer)
        let title = title(for: conversation)
        let heading = appendText(title, to: body, size: 17, inset: 28)
        heading.font = .systemFont(ofSize: 17, weight: .semibold)
        self.heading = heading
        if !conversation.text.isEmpty { appendText(conversation.text, to: body, size: 14) }
        appendState(conversation)
        if let error = conversation.error, !error.isEmpty {
            appendText(error, to: body, size: 13, color: .systemRed)
        }
        for (index, item) in conversation.items.enumerated() {
            if index > 0 { appendSeparator() }
            appendItem(item, conversation: conversation)
        }
        appendNavigation(conversation.actions)
        setPending(pending)
        setAccessibilityLabel("Pet conversation: \(title)")
        window?.recalculateKeyViewLoop()
    }

    private func appendState(_ conversation: Conversation) {
        if conversation.mode == .loading && conversation.text.isEmpty {
            appendText("Gathering the threads…", to: body, size: 14)
        }
        if conversation.mode == .empty && conversation.text.isEmpty {
            appendText("No recent work to show.", to: body, size: 14)
        }
    }

    private func appendItem(_ item: ConversationItem, conversation: Conversation) {
        let card = NSStackView()
        card.orientation = .vertical
        card.alignment = .leading
        card.spacing = 10
        let singleSubject = conversation.items.count == 1 && [.thread, .context].contains(conversation.mode)
        if !singleSubject && !item.title.isEmpty {
            let title = appendText(item.title, to: card, size: 14)
            title.font = .systemFont(ofSize: 14, weight: .semibold)
        }
        if !item.text.isEmpty && item.text != conversation.text {
            appendText(item.text, to: card, size: 14)
        }
        if conversation.mode == .context && !item.context.isEmpty {
            appendText(item.context, to: card, size: 13, color: .secondaryLabelColor)
        }
        appendResponses(item.actions, to: card)
        body.addArrangedSubview(card)
        card.widthAnchor.constraint(equalTo: body.widthAnchor).isActive = true
    }

    private func appendResponses(_ actions: [ConversationAction], to stack: NSStackView) {
        guard !actions.isEmpty else { return }
        let group = NSStackView(views: actions.map { button(for: $0) })
        group.spacing = 8
        group.alignment = .centerY
        stack.addArrangedSubview(group)
        group.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        responses.append(group)
    }

    private func appendNavigation(_ actions: [ConversationAction]) {
        let visible = actions.filter { $0.kind != .dismiss }
        let recent = visible.filter { [.more, .rounds].contains($0.kind) }
        if recent.count > 1 {
            let popup = NSPopUpButton(frame: .zero, pullsDown: true)
            popup.bezelStyle = .inline
            popup.isBordered = false
            popup.font = .systemFont(ofSize: 12)
            popup.contentTintColor = .secondaryLabelColor
            popup.setAccessibilityLabel("Recent work")
            let menu = NSMenu(title: "Recent work")
            menu.autoenablesItems = false
            menu.addItem(withTitle: "Recent work", action: nil, keyEquivalent: "")
            for action in recent {
                let item = NSMenuItem(title: action.label, action: #selector(performMenuAction(_:)), keyEquivalent: "")
                item.identifier = NSUserInterfaceItemIdentifier(action.id)
                item.target = self
                menu.addItem(item)
            }
            popup.menu = menu
            footer.addArrangedSubview(popup)
            buttons.append(popup)
        }
        for action in visible where !recent.contains(action) || recent.count == 1 {
            let button = button(for: action)
            button.bezelStyle = .inline
            button.isBordered = false
            button.font = .systemFont(ofSize: 12)
            button.contentTintColor = .secondaryLabelColor
            footer.addArrangedSubview(button)
        }
    }

    @discardableResult
    private func appendText(_ text: String, to stack: NSStackView, size: CGFloat,
                            color: NSColor = .labelColor, inset: CGFloat = 0) -> NSTextField {
        let label = NSTextField(wrappingLabelWithString: text)
        label.font = .systemFont(ofSize: size)
        label.textColor = color
        label.isSelectable = true
        label.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        stack.addArrangedSubview(label)
        label.widthAnchor.constraint(equalTo: stack.widthAnchor, constant: -inset).isActive = true
        labels.append(label)
        return label
    }

    private func appendSeparator() {
        let separator = NSBox()
        separator.boxType = .separator
        body.addArrangedSubview(separator)
        separator.widthAnchor.constraint(equalTo: body.widthAnchor).isActive = true
    }

    private func button(for action: ConversationAction) -> NSButton {
        let button = ConversationButton(title: action.label, target: self, action: #selector(performAction(_:)))
        button.identifier = NSUserInterfaceItemIdentifier(action.id)
        button.bezelStyle = .rounded
        button.controlSize = .regular
        button.font = .systemFont(ofSize: 13)
        button.setAccessibilityLabel(action.label)
        buttons.append(button)
        return button
    }

    func setPending(_ pending: Bool) {
        for button in buttons { button.isEnabled = !pending }
        progress.removeFromSuperview()
        pendingLabel.removeFromSuperview()
        if pending {
            footer.addArrangedSubview(progress)
            footer.addArrangedSubview(pendingLabel)
            if !NSWorkspace.shared.accessibilityDisplayShouldReduceMotion { progress.startAnimation(nil) }
        } else {
            progress.stopAnimation(nil)
        }
    }

    func preferredHeight(width: CGFloat) -> CGFloat {
        measureDocument(width: max(1, width - 40))
        return ceil(body.fittingSize.height) + 40 + footerHeight + footerGap
    }

    private var footerHeight: CGFloat { footer.arrangedSubviews.isEmpty ? 0 : max(22, footer.fittingSize.height) }
    private var footerGap: CGFloat { footer.arrangedSubviews.isEmpty ? 0 : 14 }

    func layoutContent() {
        let footerHeight = footerHeight
        footer.frame = NSRect(x: 20, y: 20, width: bounds.width - 40, height: footerHeight)
        scroll.frame = NSRect(x: 20, y: 20 + footerHeight + footerGap, width: max(1, bounds.width - 40),
                              height: max(1, bounds.height - 40 - footerHeight - footerGap))
        close.frame = NSRect(x: bounds.width - 38, y: bounds.height - 36, width: 20, height: 20)
        measureDocument(width: scroll.contentSize.width)
    }

    private func measureDocument(width: CGFloat) {
        bodyWidth.constant = width
        for label in labels { label.preferredMaxLayoutWidth = max(1, width - (label === heading ? 28 : 0)) }
        for group in responses {
            let required = group.arrangedSubviews.reduce(CGFloat(0)) { $0 + $1.fittingSize.width } + CGFloat(max(0, group.arrangedSubviews.count - 1)) * 8
            group.orientation = required <= width ? .horizontal : .vertical
            group.alignment = group.orientation == .horizontal ? .centerY : .leading
        }
        document.setFrameSize(NSSize(width: width, height: max(1, document.frame.height)))
        document.layoutSubtreeIfNeeded()
        document.setFrameSize(NSSize(width: width, height: max(1, ceil(body.fittingSize.height))))
        document.layoutSubtreeIfNeeded()
    }

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        updateContrast()
    }

    func updateContrast() {
        let workspace = NSWorkspace.shared
        let solid = workspace.accessibilityDisplayShouldReduceTransparency || workspace.accessibilityDisplayShouldIncreaseContrast
        blendingMode = solid ? .withinWindow : .behindWindow
        layer?.backgroundColor = solid ? NSColor.windowBackgroundColor.usingColorSpace(.deviceRGB)?.cgColor : nil
        layer?.borderWidth = workspace.accessibilityDisplayShouldIncreaseContrast ? 2 : 0.5
        layer?.borderColor = NSColor.separatorColor.usingColorSpace(.deviceRGB)?.cgColor
    }

    private func title(for conversation: Conversation) -> String {
        switch conversation.mode {
        case .rounds: return "Make the rounds"
        case .empty: return "No fresh notes"
        case .loading: return conversation.headline ?? "One moment"
        case .thread, .context:
            return conversation.headline ?? conversation.items.first?.title ?? "A note from Pet"
        }
    }

    private func clear(_ stack: NSStackView) {
        for view in stack.arrangedSubviews { view.removeFromSuperview() }
    }

    @objc private func performAction(_ sender: NSButton) {
        guard let id = sender.identifier?.rawValue else { return }
        onAction?(id)
    }

    @objc private func performMenuAction(_ sender: NSMenuItem) {
        guard let id = sender.identifier?.rawValue else { return }
        onAction?(id)
    }

    @objc private func closeConversation() { onDismiss?() }
}

private extension ConversationContent {
    func configureContent() {
        scroll.drawsBackground = false
        scroll.borderType = .noBorder
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        scroll.scrollerStyle = .overlay
        scroll.documentView = document
        body.orientation = .vertical
        body.alignment = .leading
        body.spacing = 12
        body.translatesAutoresizingMaskIntoConstraints = false
        document.addSubview(body)
        bodyWidth = body.widthAnchor.constraint(equalToConstant: 340)
        NSLayoutConstraint.activate([
            body.leadingAnchor.constraint(equalTo: document.leadingAnchor),
            body.topAnchor.constraint(equalTo: document.topAnchor),
            bodyWidth
        ])
        addSubview(scroll)
    }

    func configureFooter() {
        footer.orientation = .horizontal
        footer.alignment = .centerY
        footer.spacing = 10
        addSubview(footer)
        progress.style = .spinning
        progress.controlSize = .small
        progress.isDisplayedWhenStopped = false
        progress.widthAnchor.constraint(equalToConstant: 16).isActive = true
        progress.heightAnchor.constraint(equalToConstant: 16).isActive = true
        pendingLabel.font = .systemFont(ofSize: 12)
        pendingLabel.textColor = .secondaryLabelColor
    }

}

private final class ConversationDocument: NSView {
    override var isFlipped: Bool { true }
}

private final class ConversationButton: NSButton {
    override var needsPanelToBecomeKey: Bool { true }
}
