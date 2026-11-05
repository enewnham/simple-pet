# SDK conversations and workflows

- Status: Implemented
- Date: 2026-11-05
- Implementation: Integrated checks and live SDK acceptance passed

This document specifies the internal agent in the
[desktop coordinator plan](overview.md). [Conversation](conversation.md) defines
the chat surface. [Responsibilities and evidence](responsibilities-and-evidence.md)
defines durable work and provenance. [Desktop sessions](desktop-sessions.md)
defines delegation to coding harnesses.

## Context and decision

Simple Pet's Python daemon collects coding activity and generates ambient
commentary through an Anthropic model client. A desktop coordinator also needs
conversations that retrieve evidence, investigate questions, and carry adopted
responsibilities through ordinary next steps.

Use the Claude Agent SDK for Python for pet conversation, investigation, and
workflow calls. The configured coordinator default is Haiku with extended
thinking enabled and medium effort. Context engineering establishes the relevant
work, evidence, and available questions before invocation. The agent can retrieve
more evidence through application tools and selected MCP integrations. Meaning
finding belongs to the reasoning agent. The existing ambient expression path can
remain separate.

Substantive implementation goes to a configurable external desktop harness
through the delegation adapter. Claude, Codex, and Cursor remain valid external
harnesses independently of the internal SDK vendor. The coordinator runs locally
while the computer is awake.

## Interfaces and ownership

The application owns canonical project records, adopted responsibilities,
conversations, source evidence, delegated tasks, and exact operation state in SQLite.
Compact working notes retain supported findings, uncertain hypotheses, source
pointers, and corrections. Notes orient reasoning while canonical records
determine action state. An SDK adapter owns client connections, request streaming,
interruption, session resumption, usage, and conversion of SDK messages into
application events. A narrow adapter makes the SDK effect testable without
introducing interchangeable agent runtimes.

Each invocation receives its conversation, inquiry, or responsibility ID, the
current instruction revision, a focused purpose, a work brief, available tools,
and a resource budget. Scheduled passes also carry pass and candidate IDs,
applicable responsibility and source revisions, and any application-reserved
focus. Results identify the candidates they address and their dispositions,
with a short supported reason and source references. Raw chains of thought are
outside the application contract. The application validates and persists accepted
findings, actions, and terminal outcomes before presenting them as committed work.

## Ongoing work loop

The application queues and coalesces candidates from source events, due followups,
quiet review, and reconciliation. Before invoking Haiku, it applies timing,
priority, fairness, concurrency, budgets, freshness, and permissions. Code reserves
focus when overdue fair service or explicit user ordering requires it. Otherwise
it offers eligible routine peers for the model to choose among. Follow the canonical
[ongoing coordination loop](responsibilities-and-evidence.md#ongoing-coordination-loop).

Pass eligible bounded candidate briefs, source changes, prior considerations,
current revisions, available coding capacity, and any reserved focus to the agent. Haiku judges meaning,
selects among offered routine candidates when no focus is reserved, and chooses a
bounded investigation, eligible continuation, human question, wait, or no finding.
It returns dispositions for considered candidates and must address any reserved
focus. It can suggest a
wake condition or review time. The application validates and bounds actual
timing. Model output cannot invent operational states or urgency, override
explicit user ordering or fairness, or bypass a required reserved focus.

For a finite goal, the agent can return a completion assessment against the exact
recorded condition and responsibility revision. Observed assessments require
current independent leaf evidence. Harness reports, hypotheses, and derived notes
cannot supply independent verification. Reported or inferred assessments retain
their uncertainty. Exact machine watches remain application-owned. Explicit human
confirmation uses a foreground tool tied to the accepted user event, with a separate
human-confirmed basis.

The application records considered and serviced separately. Budget-blocked work
remains unconsidered. Persist waits, wake conditions, and no-finding results so
the scheduler avoids repeated model polls. Waiting on a human resumes after a
reply or material change. Wake and restart reconciliation coalesce pending
candidates before scheduling. Inquiry, action, and history records remain
independent of attention and speech decisions. Finishing work never forces speech.

## Work brief and retrieval

Before the first invocation, bootstrap a bounded view of existing work from
configured historical harness records and available sources. Preserve their
provenance and coverage. A historical harness record supplies evidence without
conferring ownership of that harness session. First use must be useful before the
pet accumulates its own conversation history.

Build the work brief for the current purpose. Include relevant intents, decisions,
progress, unresolved questions, references, active and deferred tasks, changes
since the relevant checkpoint, and decisions flagged for human attention. State
source availability, freshness, and missing coverage. Supply a source map of
aliases, topics, and collections so the agent can choose where to investigate.
Keep detail proportional to the question. Avoid ingesting or preinterpreting all
provider data or constructing a giant fixed prompt.

Prepared briefs and retrieval results include applicable user corrections as
authoritative within their scope, label tentative hypotheses as reviewable, and
flag source conflicts. Source decisions retain their date, scope, and
supersession. Reports retain their reported status through notes and summaries.

Expose scoped semantic tools for current work context, federated context search,
passage reading with neighboring text, and relevant changes. Search spans enabled
sources, pet notes, and artifacts. Passage results include excerpts, source,
date, retrieval reason, and references for expansion. Tools explain their scope,
coverage, and useful questions without prescribing a mandatory sequence of
thought or observations. Selected basic provider MCP operations remain available
for precise lookups. The agent can search beyond the initial brief and follow
references to resolve a question.

Catalog preparation and direct retrieval are complementary. Accessible source text
can inform inquiry before the application classifies it as a decision. Tools preserve
source dates separately from observation/import time. An unknown discussion date or
decision status stays unknown in reasoning and in the result.

Retrieval selects sources and expands queries using the question, source map,
and work references. Independent searches can run in parallel. Deduplicate
results and rank exact references and confirmed links alongside keyword and
semantic relevance. Prefer passages that carry meaningful decisions over stray
mentions. Account for supersession, freshness, user feedback, and source
diversity. Return enough neighboring text to interpret a passage faithfully.
The reasoning agent judges meaning across these results and records supported
findings or uncertain hypotheses through the note tools. Epistemic confidence,
authority to act, and retrieval relevance remain separate judgments. Ranking a
note highly never grants it authority or strengthens its support.

Application tools also propose or perform authorized workflow steps and request
external delegation. The SDK supports application-defined tools through an
in-process MCP server. See its
[custom tools guide](https://code.claude.com/docs/en/agent-sdk/custom-tools).

Source adapters supply session activity, Git, and GitHub evidence. A generic
read-only ticket-tracking interface, initially implemented by Linear, and a local
transcript adapter contribute evidence when enabled and relevant to the question.
Ticket tools consume the provider-qualified contract and optional capabilities
defined in [responsibilities and evidence](responsibilities-and-evidence.md). An
adapter can use MCP or a direct API for exact watches. MCP research can begin
without a custom normalized collector for each source.

Tool results preserve whether a statement was observed, reported, or inferred,
including source time and reference. SDK-produced text never establishes an
external fact by itself. A model summary retains its supporting references and
uncertainty.

## Note tools and provenance

Use explicit note operations following the canonical
[working note lifecycle](responsibilities-and-evidence.md#working-note-lifecycle).
Recording a finding or hypothesis carries actor, origin, kind, support status,
scope, original date when supplied, observed and checked times, leaf evidence
references and revisions, and derived lineage. The application assigns origin from the actual
event and validates these fields. Revise, recheck, refute, supersede, and archive
operations address an existing note and retain its origin and history.

Direct instructions and corrections enter canonical state from actual human
events. The SDK cannot assign itself user origin or promote a note's confidence.
New independent evidence or direct user confirmation can request a validated
support transition. Repeated model mentions, retrieval, and compaction cannot
strengthen a claim or reset its review age. Notes and summaries cannot serve as
their own evidence.

Summaries preserve provenance, support, scope, and uncertainty. A synthesized
claim needs references to underlying leaf evidence, including when that evidence
was reached through another note. Corrections update affected notes and subsequent
briefs through validated transitions while retaining what was corrected.

## Tools and configuration

The published repository exposes model selection, enabled MCP integrations,
resource budgets, and invocation frequency as configuration. Coalesce relevant
changes before invoking the reasoner. Budgets bound model usage, elapsed work,
and tool activity. The application checks them during execution and records why
a run stopped. It keeps credentials outside durable conversation records.

The coordinator's default SDK configuration enables extended thinking through
adaptive thinking on Haiku 5.5 with medium effort:

```python
ClaudeAgentOptions(
    model="claude-haiku-5-5",
    thinking={"type": "adaptive"},
    effort="medium",
)
```

[Haiku 5.5](https://platform.claude.com/docs/en/models/haiku-5-5/overview)
supports adaptive thinking. Its
[thinking configuration](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)
uses adaptive thinking and effort rather than the legacy enabled mode and token
budget. Adaptive thinking can decide that a request needs no explicit thinking
block. Coordinator model, thinking, effort, and sufficient response budget have
their own settings, separate from ambient commentary. Do not silently upgrade
the model or effort.

Configure the actual tool surface explicitly. The internal pet receives no code
editing or unrestricted shell tools. Repository inspection uses scoped read
operations. A request to implement a change becomes a delegation request carrying
the agreed task and evidence to an external harness.

The SDK's `allowed_tools` controls automatic approval. It does not restrict tool
availability. Configure built-in tools and registered MCP servers separately,
then apply permission rules and application validation. Review the pinned SDK's
behavior against its [Python reference](https://code.claude.com/docs/en/agent-sdk/python)
and [permissions guide](https://code.claude.com/docs/en/agent-sdk/permissions)
when implementing this boundary.

Select settings sources and MCP connections explicitly. An invocation must not
inherit project tools, hooks, plugins, or instructions merely because its working
directory is a checkout. Startup verifies the effective configuration and fails
clearly when a required integration is unavailable. Optional integrations remain
visible as unavailable in the relevant workflow.

## Durable state and lifetime

SDK session/request IDs identify reusable model working context and invocations.
Durable conversation/message IDs identify accepted human dialogue. Presentation
IDs and revisions identify renderer snapshots. Delivery IDs identify messages
offered for renderer acceptance and remain stable across their retries. Desktop
task, command, and native session IDs identify delegated execution. Store explicit
associations between these identifiers. An SDK transcript can be discarded or
rebuilt without losing adopted responsibilities, evidence, conversation history,
working notes, or task outcomes.
The SDK supports resuming a specific session by ID, as described in
[sessions](https://code.claude.com/docs/en/agent-sdk/sessions).

Foreground conversation and background investigation have independent lifetimes.
The conversation stays responsive while an investigation retrieves evidence.
Each run captures its instruction, responsibility, and source revisions. Tool
calls carry their pass and candidate IDs when scheduled. The application rejects
stale candidate results before they can act or change current conclusions.
A new instruction that changes scope invalidates affected runs, cancels their
pending work, and retains relevant findings only as attributed historical evidence.

Cancellation stops model work and pending application operations where possible.
Sleep and shutdown interrupt active runs and persist their known status. Restart
reconciles pending operations against durable records and external observations.
It resumes only work whose purpose, authority, and revision remain current. An
uncertain external outcome requires reconciliation before another attempt.
Resuming SDK context never blindly replays operations.

## Action validation and attention

For each requested effect, the application checks the current instruction and
applicable responsibility revisions, source freshness, ownership, permitted
operation, resource budget, and recorded operation identity. It commits acceptance
and outcome with the affected durable state. These checks belong at the effect
boundary even when an SDK permission rule approves a tool call.

Active recent work and goals can guide proactive read-only discovery without an
adopted responsibility. Consequential task actions require adoption. An explicitly
adopted responsibility authorizes its ordinary next steps within the recorded
scope. A priority change, competing ownership, or proposed scope expansion becomes
a human decision in the conversation. Model wording and pet persona cannot
establish authorization, urgency, or ownership.

Background findings enter existing attention and delivery decisions. Routine
progress can remain in the conversation or Activity surface. A background run
cannot force a speech interruption. Selected speech receives an established
meaning and communicative purpose before persona generation. Claims about
persona quality require actual generated responses.

Apply the [companion narration rule](conversation.md#companion-voice-and-attribution)
when expressing outcomes. First-person participation can describe coordinated work
without making every reply a report about the harness. Material distinctions between
reported results, supporting verification, and uncertainty remain clear, with source
and native-session details available for inspection.

## Workflow scenarios

**Investigate existing work.** On first use, the work brief identifies an active
PR from configured historical records. The user asks why it is waiting. Context
search finds related decisions, passage reading expands their meaning, and a
precise GitHub lookup checks current status. The agent distinguishes a reported
fix from an observed passing check and answers with references. Missing access
produces a precise uncertainty. This inquiry needs no adopted watch.

**Investigate a possible rollback conflict.** A session reports that a CI check is
finished. Newly accessible meeting notes discuss staging rollback. The agent reads
the passage directly and checks its date, decision status, and relevance to the
implementation. Import time cannot establish a later decision. Unknown chronology
or status remains part of a possible-conflict finding. Any external investigation
requires applicable authority and an owned session, as described in the
[migration walkthrough](overview.md#verification-and-release-acceptance).

**Follow an adopted responsibility.** The user asks the pet to watch a PR through
review. A changed check triggers a bounded investigation. An ordinary authorized
step proceeds through its validated tool. A requested product decision returns
to the user. New user instructions invalidate conflicting pending actions.

**Delegate implementation.** An investigation identifies a substantive fix. The
pet prepares a bounded task for the external harness adapter and records its
task association. A launch acknowledgment establishes acceptance and may describe
pending workspace preparation. Running execution and completion require corresponding
observations tied to that task.

## Invariants

- SQLite preserves accepted work independently of SDK context.
- Working notes retain support, uncertainty, and corrections.
- Model repetition cannot change note origin, support, or review age.
- Every effect passes current application validation.
- Superseded runs cannot act on their captured revision.
- Evidence retains source, time, scope, and uncertainty through summaries.
- External implementation uses the desktop harness adapter.

## Alternatives and consequences

A provider-neutral agent framework adds configuration and abstraction before a
second internal runtime is required. Direct model calls remain suitable for
focused ambient expression, but require application-built conversation and tool
loops for coordinator work. Giving the internal agent coding tools duplicates
the desktop harness's implementation role and obscures delegated task ownership.

The chosen SDK creates an explicit vendor dependency for internal workflows.
Maintainers must pin and verify SDK behavior, account for its working files, and
test effective permissions alongside application checks.

## Verification and delivery

Use a fake SDK at the adapter effect boundary to test retrieval, tool arguments,
streaming, errors, and cancellation. First-invocation fixtures must produce a
useful work brief from configured history, show source availability, and preserve
harness ownership. Retrieval fixtures cover aliases, expanded queries, parallel
queries, deduplication, decision passages, neighboring text, source diversity,
freshness, supersession, and expansion beyond the initial selection. A correction
must change working notes and the next brief while retaining provenance.

Verify that tool guidance explains useful questions without requiring a rigid
reasoning sequence. Verify stale revisions reject effects, read-only inquiry
works before adoption, consequential actions require adoption, budget exhaustion
stops work, forbidden tools are absent, and unrelated settings do not load.
Restart must reconcile uncertain operations without duplicate launch. Verify
responsive chat during investigation, attention-controlled background delivery,
and effective coordinator model configuration.

Have the fake SDK attempt to forge a user correction and reject it. Repeated
hypothesis summaries must preserve support and review age. A stale candidate
cannot act, and model selection cannot bypass a required reserved focus. Without
one, the model can select a relevant routine candidate and explain its choice. Persist wait
and wake dispositions across restart, leave budget-blocked candidates
unconsidered, and persist none or no-finding outcomes without producing speech.

Ship established retrieval and SDK patterns through normal automated checks and
manual acceptance, then iterate. Inspect generated conversations for grounded
claims, preserved uncertainty, useful investigations, appropriate next steps,
and persona expression. The [implementation ledger](implementation.md) records
automated checks, and [verification](verification.md) records bounded live model
acceptance. Remaining integration and desktop checks stay explicit there.

The [overview](overview.md) defines thirteen internal work packages for one
integrated implementation run and direct merge to `main`. Package 1 specifies
documentation and retrieval contracts. Package 2 introduces minimal durable state and working notes.
Package 3 supplies the brief, bootstrap, retrieval foundation, and MCP/provider
fixtures. Package 4 adds typed conversation. Package 5 connects SDK conversation
to the brief, context search, passage reading, changes, workflow tools, and notes.

Packages 6 and 7 depend on packages 2 and 3. They add the generic ticket interface
with its first Linear adapter and local meeting exports. Integrate packages 3
through 7 at the context checkpoint while worker streams progress in parallel.
Package 8 adds adopted watches and followups, package 9 external launch, and
package 10 owned follow-through. Package 11 broadens ongoing inquiry scheduling.
Broad context inquiry is available in package 5. Package 12 connects ambient
delivery and Activity. Package 13 verifies the complete integration before merge.
