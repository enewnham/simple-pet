# Conversation and Activity

- Status: Implemented
- Date: 2026-11-05
- Implementation: Integrated feature under verification

This specification defines the human-facing conversation and coordination surfaces for the desktop coordinator. It is part of the [desktop coordinator plan](overview.md), alongside [responsibilities and evidence](responsibilities-and-evidence.md), [SDK workflows](sdk-workflows.md), and [desktop sessions](desktop-sessions.md). The [implementation ledger](implementation.md) records completed checks and remaining acceptance.

## Current starting points

The renderer already displays ambient speech and a compact AppKit conversation panel. [`ConversationController`](../../../Sources/PetOverlay/ConversationController.swift) positions it near the pet, receives versioned snapshots, tracks engagement, and sends typed actions. [`ConversationContent`](../../../Sources/PetOverlay/ConversationContent.swift) renders conversation items, context, actions, pending state, and errors. [`ConversationSession`](../../../Sources/PetCore/Conversation.swift) rejects stale revisions and tracks presentation lifetime.

The daemon's [ambient interactions](../../../pet_daemon/interactions.py) build snapshots and handle navigation, context, Tomorrow, Handled, and dismissal. The [durable conversation controller](../../../pet_daemon/conversation.py) accepts typed messages, runs SDK replies, restores history, and projects [Activity](../../../pet_daemon/activity.py). [Application routing](../../../pet_daemon/application.py) and the [socket receiver](../../../pet_daemon/daemon.py) connect renderer datagrams to these owners. Canonical chat and Activity records retain accepted state independently of presentation and delivery acknowledgement.

## Product behavior

The pet may surface a short ambient prompt when it has useful, evidence-backed work to share. An ambient prompt remains transient and never takes keyboard focus. Clicking the pet or its prompt opens a compact conversation that stays available while the user reads, types, or chooses an action. An explicit click can focus the conversation. Passive updates cannot move focus or replace text the user is composing. The panel offers suggested conversational continuations and concrete actions, such as inspecting evidence, opening a work location, or asking for a clarification. Labels distinguish explaining a connection from starting work in a harness.

The Activity surface shows work the coordinator is watching or investigating, harnesses it launched, their observable results and decisions, and intentions deferred for later. It distinguishes observations, coordinator conclusions, user decisions, and task outcomes. A launched coding harness has an owned session and visible status. The coordinator can explain what it observed and link to the source evidence or relevant repository location.

Activity can explain why a candidate was selected, deferred, or left quiet, and what condition or review time will make the pet reconsider it. Delays caused by a busy slot, limited budget, or an explicit priority override remain visible. Consideration history is separate from speech history: deciding to wait can be recorded without interrupting the user. The [coordination loop](responsibilities-and-evidence.md#ongoing-coordination-loop) owns this policy.

The conversation answers questions about selected work, evidence, and available actions. It can show a source report, relevant session or repository facts, and navigation to that source. A completion claim cites evidence satisfying the adopted outcome. Ownership or a turn ending alone does not establish completion. User-marked handling remains identifiable as human feedback.

The first conversation begins with a prepared brief of available current and recent work. It includes relevant requests, decisions, open questions, related work references, deferred intentions, and source availability. Configured historical imports can establish useful initial context before new hooks arrive. The pet can search beyond the brief, expand source passages, and connect recent work without first requiring an adopted responsibility. Consequential actions retain their separate authority checks.

Context search and ranking return useful excerpts and source references for the agent to reason about. The conversation can explain why material was retrieved and distinguish a tentative connection from a confirmed one. Corrections such as "these are different projects" update working memory and dependent retrieval. Model reasoning and note-taking use the configured Haiku SDK workflow described in [SDK workflows](sdk-workflows.md).

When the user asks what the pet remembers or why it believes a connection matters, show the note's origin, applicable scope, support, and relevant source dates. A direct user correction remains authoritative at its stated scope. Agent hypotheses remain tentative, source decisions retain their date and supersession, and reported harness results remain reports. Contradictions can become a clarification question. The [note lifecycle](responsibilities-and-evidence.md#working-note-lifecycle) preserves these distinctions in the brief and retrieval results.

## Companion voice and attribution

The pet is the consistent companion the user interacts with. It coordinates
implementation through desktop harnesses and describes outcomes according to the
evidence available. Execution details remain inspectable without dominating every
conversation.

First-person participation can describe work the pet investigates or coordinates,
such as "I've been working on the migration check." Claims that affect a decision
still distinguish a harness report, supporting verification, and unresolved
uncertainty. Natural narration does not imply that the internal pet edited code or
independently verified a reported result. Source and native-session navigation remain
available when the user wants to inspect or take over the work.

Personality shapes curiosity, wording, timing, and the form of help offered. Establish
supported meaning before expression. The pet can be playful, pleased, curious, or
quiet without inventing urgency or treating every completed action as a reason to
speak. Relaxed presence is part of the experience.

When a session reports completion and newly accessible notes mention a possible
requirement, preserve the uncertain timeline. For example:

> The session reports that the CI check is finished. I also found a meeting
> discussion about staging rollback that may affect the requirement. I'll check
> when that discussion happened and whether the implementation addresses it.

If the meeting date or decision status cannot be established, describe a possible
conflict. The notes' arrival time establishes availability to the pet, not the
time of the underlying discussion or decision.

## Identity and state

The implementation keeps four identity scopes separate:

- A durable conversation ID identifies the user's chat across turns, reopening, and application restarts.
- Durable message IDs identify each accepted user or assistant message and support duplicate suppression.
- A presentation ID and revision identify a transient or updated renderer snapshot. They control stale UI actions and acknowledgements, and may expire with the panel.
- SDK request IDs and harness/session IDs identify external work. They remain distinct from conversation and presentation IDs, even when a chat turn launches a harness.

The daemon owns chat history, message acceptance, pending turns, and links to referenced work. The Swift renderer owns input, focus, rendering, and event delivery. Before invoking the SDK, the daemon validates a message against the conversation and current turn. Repeated delivery of one message ID produces at most one accepted turn. Stale revisions cannot act on newer content.

While a turn is pending, the panel shows that state and preserves the user's text. A failure includes a retry or recovery action and keeps the input. Reopening restores messages and pending status with a fresh presentation ID. Sleep affects transient presentation and ambient delivery, but preserves history and does not mark pending work successful. On wake, Activity reconciles owned-session status from evidence.

## Actions and controls

Feedback actions apply to one selected evidence item or work connection. **Later** defers that item until a stated time. The existing Tomorrow action retains its stated next-morning meaning. **Handled** records that it needs no more attention. **Not relevant** suppresses the suggested connection in its current context. It does not rewrite a source-established association without an explicit correction. Each action is idempotent and remains visible in Activity with its scope and time. Clarifications can offer **Yes**, **No**, and **Not sure**, with room to type an explanation.

The user can inspect a claim's evidence and navigate to a repository, report, or session. The SDK can explain supplied evidence and ask simple clarifications. It cannot add unsupported facts.

Pause coordination and stop task are separate controls. Pausing coordination stops proactive review, prompts, and new coordinator-initiated work while preserving chat history and already launched harnesses. Stopping a task targets one owned harness session and records the request and observed terminal outcome. Stopping a task does not pause coordination; pausing coordination does not stop a task. Activity clearly identifies the selected task and effect of the Stop action without adding a required second confirmation.

SDK internal reasoning and research operate on supplied evidence and coordination records. Coding runs through the user's configured desktop adapters and produces an owned session with observable status and results. The pet does not implement coding tools or claim results on behalf of an adapter.

## Invariants

- Ambient prompts and passive updates never take keyboard focus.
- Only an explicit user interaction can direct keyboard input to the conversation.
- Durable conversation and message identity survives presentation replacement and application restart.
- Presentation revisions, SDK requests, and harness sessions cannot be mistaken for durable chat messages.
- A duplicate message or action cannot launch duplicate work or record duplicate feedback.
- Every recommendation and claimed result links to available evidence or is labeled as uncertain.
- Feedback changes only the selected item or connection.
- Coordination pause and task stop have independent state and effects.
- Activity separates a requested action from its confirmed outcome.
- A deferred candidate retains an inspectable reason and revisit condition or time.
- Compressing the conversation or brief cannot turn a tentative hypothesis into a fact or silently remove an applicable user correction.

## Acceptance scenarios

1. **Watch PR checks:** the user asks the pet to watch a specified pull request until its checks pass. The adopted watch survives restart. A fresh successful observation produces one report with supporting links. The user can ask a clarification or open the source. Ambient updates remain unfocused, and the user's explicit click opens the conversation.
2. **Conversation and retry:** the user submits one message with a durable message ID. The daemon accepts it once, shows pending state, and returns a grounded answer with suggested continuations and executable actions. A transport retry does not repeat the SDK turn. A failed turn retains the input and exposes retry. A stale panel action cannot affect the response.
3. **Feedback scope:** the user marks one report connection Later, Handled, or Not relevant. Activity records the decision against that connection. Other items in the conversation keep their prior state.
4. **Inspect and launch:** the user opens evidence, then asks the coordinator to start a coding task through the configured adapter. Activity distinguishes dispatch accepted, workspace preparing, and observed running execution, then displays the result and links. A planned task is never shown as complete before evidence confirms it.
5. **Separate controls:** with one harness running, the user pauses coordination. Ambient review and new coordinator launches stop, while that harness continues. The user can later resume coordination. A separate stop request targets only the selected harness and Activity records its observed outcome.
6. **Sleep and reopen:** a conversation remains available after closing and reopening the panel and after desktop sleep. The reopened panel has a fresh presentation identity, restored durable messages, and accurate pending or session state.
7. **Final cross-source handoff:** when a task result cites a pull request, report, and harness session, the conversation and Activity distinguish each source and link to all available evidence. The user can inspect a source, ask how it supports the result, and see any unresolved uncertainty without the assistant treating the handoff as proof beyond its recorded outcome.
8. **First-use orientation and retrieval:** with configured current work and historical records available, the first reply starts from that work. A question about an open issue retrieves a relevant earlier decision and its surrounding passage. The user can correct a proposed connection, and subsequent retrieval respects that correction. Missing sources remain visible without making the conversation wait for days of new activity.
9. **Why this work:** a noisy project changes repeatedly while another responsibility reaches its review time. Activity explains the selected focus and any deferred work. A budget-limited review stays pending rather than appearing completed, and inspecting that state does not require an ambient prompt.
10. **Remembered interpretation:** an agent hypothesis is summarized more than once, then conflicts with a scoped user correction. The conversation still identifies the hypothesis as tentative, preserves the correction, and can show the original supporting sources. It asks about a material contradiction without presenting repeated summaries as independent confirmation.
11. **Natural participation and uncertain chronology:** the pet describes its migration investigation in first person, retains the distinction between a session's report and verification, and offers source/session inspection. A meeting export arriving after a session ends does not establish a later meeting or changed decision. If date or decision status stays unknown, the conversation preserves the possible conflict and its uncertainty.

## Delivery sequence

The [overview](overview.md) organizes one complete implementation into internal work packages, followed by a direct trunk merge. Package 1 documents the design, package 2 establishes durable records, and package 3 supplies prepared context, bootstrap, and retrieval. Package 4 adds the composer, and package 5 connects SDK reasoning, retrieval, and notes. Integrate these with the ticket and meeting adapters in packages 6 and 7 at the context checkpoint. Package 8 adds an adopted watch, package 9 supplies desktop launch, and package 10 adds owned-session continuation. Package 12 completes ambient prompts and Activity. Each package includes focused interaction checks. Complete the combined feature before merging it to trunk.
