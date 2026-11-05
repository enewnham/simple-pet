# Responsibilities and evidence

- Status: Implemented
- Date: 2026-11-05
- Scope: Prepared context, working notes, coordination, responsibilities, and authority

## Context

The desktop coordinator helps a person carry work across projects while their Mac
is awake. Its first conversation needs useful context about recent requests,
decisions, progress, and unresolved questions. Configured sources already contain
much of that history. The pet can investigate relevant work before the user adopts
a responsibility or new session hooks arrive.

The current [shared work model](../../../pet_daemon/work/shared_work.py) identifies
subjects through agent sessions:

```python
class WorkSubject(BaseModel):
    id: SessionIdentity
```

[Work records](../../../pet_daemon/work/models.py) preserve reports, captured
references, observations, and provenance.
[SQLite storage](../../../pet_daemon/storage/work.py) persists those records. They
supply current work evidence. Durable project and responsibility identities extend
beyond individual sessions.

The [overview](overview.md) owns delivery order.
[Conversation](conversation.md) owns typed chat, suggested actions, ambient prompts,
and Activity. [SDK workflows](sdk-workflows.md) owns production conversation and
research through the Claude Agent SDK. [Desktop sessions](desktop-sessions.md)
owns the external adapter that launches and follows Claude, Codex, and Cursor
harnesses. The pet coordinates implementation through those harnesses and never
implements code directly.

## Decision

Use Claude Haiku 5.5 with adaptive thinking and medium effort for internal SDK
coordination. Engineer prepared context and MCP retrieval to support useful
reasoning with that configuration. Build directly on established brief, search,
passage reading, change tracking, and transactional state patterns. Product
implementation and focused acceptance checks proceed together.

Keep these state layers distinct:

| Layer | Meaning and authority |
| --- | --- |
| Canonical application state | Exact user instructions, adopted responsibilities, priorities, task and operation identities, authority, pause state, and accepted transitions. SQLite is authoritative. |
| Compact working memory | Work briefs, hypotheses, open questions, useful corrections, confirmed links, and source pointers. Stored notes support continuity and remain revisable against instructions and evidence. |
| Retrieved source data | Provider observations, agent reports, passages, source revisions, freshness, and cached retrieval results. These support claims and never grant control. |

The person owns the work. The daemon records accepted transitions and enforces
authority. SDK sessions receive snapshots and return findings, notes, or proposals.
SDK conversation memory cannot establish canonical application state. Summaries
retain the distinction between a source fact, an agent's report, and a tentative
interpretation.

## Prepared work context

Before first invocation, startup conversation, or post-wake conversation, prepare
a useful bounded brief from available state and configured sources. Include current
sessions and their requests, decisions, progress, and open questions. Include known
repository, PR, and ticket references, adopted responsibilities, deferred work,
human-input flags, and meaningful changes since the last checkpoint. Show freshness,
unavailable sources, and uncertainty where they affect interpretation.

Bootstrap bounded configured harness history and available notes so the first
conversation can discuss existing work. Import captures source provenance and
relevant passages without replaying historical actions or acquiring ownership of
sessions. Missing history remains unknown. Bootstrap limits, source selection, and
brief size are configurable. A failed source does not prevent a useful partial brief
whose omissions remain visible.

Supply a source map alongside the brief. It identifies available collections,
configured access scope, known projects and aliases, useful identifiers, supported
search/read/change operations, and known routes to source material. The brief
provides orientation. Source pointers and retrieval provide supporting detail.
Updating a compact brief preserves unresolved questions and the citations needed
to inspect its claims. Select notes whose scope applies and whose support remains
valid under the working note lifecycle. Include applicable user corrections as
constraints. Label tentative or historical material, and recheck affected claims
before an action relies on them. An old brief supplies orientation and source
pointers, never the sole evidence for a new claim.

Prepared context supports proactive read-only discovery across recent work and goals.
The pet can investigate a possible dependency, retrieve a past decision, or compare
meeting context with current implementation work under configured read permissions.
Adoption governs commitments and actions. Learning what the user's work means
can precede adoption. Suggested questions guide investigation without requiring a
finding, an adopted responsibility, or a universal project graph.

## Retrieval contract

The application exposes four core semantic capabilities through the SDK's tool
facade. Provider MCP operations remain available within configured permissions for
precise provider-specific lookups alongside the curated retrieval capabilities.

Cataloged identities and explicit links orient the brief, while direct search and
reading supply additional context. An accessible passage can support inquiry before
it is classified as a decision or linked to a structured project record. Preserve
its source and uncertainty when using it, and retain findings that become useful.

| Capability | Returned meaning |
| --- | --- |
| Work context | A prepared brief and source map for the current conversation or selected work, with references, freshness, and unknowns. |
| Contextual cross-source search | Ranked citable passages and observations for a question, selected work, and supplied identifiers, with source coverage and availability. |
| Passage expansion | The source passage around a hit or citation, its revision, surrounding context, canonical location, and current availability or supersession. |
| Changes since a checkpoint | Relevant source or application changes since a recorded boundary, including edits, removals, failures, and unchanged observations where material. |

The retrieval coordinator expands a question using known identifiers, aliases,
project terms, confirmed links, and the decision being investigated. For example,
a question about a PR's requirement can search its ticket key, project vocabulary,
and the reason for the requirement. Relevant provider queries run in parallel.
Expansion stays within configured read scope, and provider failures remain visible
in the combined result.

Deduplicate provider-qualified identities, source revisions, and overlapping
passages while preserving distinct evidence. Rank using exact references and
confirmed links, keyword and semantic relevance, decision substance, supersession,
freshness, user feedback, and source diversity. An older accepted decision can
outrank a recent status update when it answers the question. A useful result set
includes complementary sources rather than repeated copies of one report. Query
and ranking limits remain configurable.

Apply scoped user corrections before ranking or query expansion so rejected links
cannot be reintroduced through aliases or summaries. Return tentative connections
with explicit labels and limited evidentiary weight. Historical sources remain
retrievable with their original scope and supersession status.

Ranking determines what to inspect first. Evidence validity determines what a claim
can establish. A high-ranking agent report still establishes what the agent said.
A superseded decision may explain history while failing to establish the current
requirement. Freshness cannot turn an unconfirmed association into a fact. Uncertain
cross-source connections remain hypotheses until supported or user-confirmed.

Retain used passages and useful retrieval caches with provider identity, revision,
observation time, canonical links, and query or source provenance. Evidence retention
preserves support for active work, accepted decisions, completion reports, and their
citations. Compact working memory records useful confirmed links and corrections
under the lifecycle below. New conflicting evidence can justify a question about a
binding correction, while that correction remains in force until the user changes
it or its stated scope ends.

## Working note lifecycle

Every note records its origin separately from its kind. Origin identifies a direct
user statement, fetched source, harness report, or agent derivation. Kind identifies
an instruction, correction, decision, observation, hypothesis, synthesis, or source
pointer. Preserve stated scope, the original date when supplied, observation and
check dates, and leaf evidence identities and revisions. Leaf evidence means the
underlying user statement, source passage, or observation supporting a derived
claim. Repeated mention and compaction do not refresh original or observed evidence
dates or add support. An actual recheck records its check time separately.

Epistemic support records what the evidence establishes and its uncertainty.
Action authority records what the user permits. These are separate properties.
A well-supported source decision cannot authorize a harness launch. A hypothesis
can help choose a read-only inquiry while remaining tentative. Numerical confidence
is unnecessary for these distinctions.

Notes have active, needs-recheck, superseded, rejected, or archived status. Changes
record their reason and lineage. Revision changes the note while retaining history.
Refutation records evidence against a claim. Recheck obtains current support.
Supersession records the applicable replacement. Archive removes a note from routine
context while preserving history and citations required by active work. These
operations follow the note's origin, kind, and scope:

| Note | Revision and retention policy |
| --- | --- |
| Direct user instruction or correction | Retain the canonical user record. Apply it within its stated scope until superseded, retracted, or that scope ends. Contrary imports trigger a question and cannot replace it. Age never demotes it below a hypothesis. Archive its working copy only when no active scope or dependent work needs it. |
| Fetched decision or observation | Preserve original date, scope, source revision, and support. Recheck affected claims after edits or deletion, marking changed or unavailable support. Record supersession when evidence establishes a replacement. A newer mention alone does not supersede a decision, and deletion alone does not refute it. Archive when no active work or citation requires it. |
| Harness report | Keep its reported status and origin through revision, recheck, and compression. Separate validation records establish an independently checked outcome. Refutation or supersession retains what the harness reported and the contrary evidence. Archive when no active work or citation requires it. |
| Agent hypothesis | Keep it tentative and revisitable. Repetition or summary provides no additional support or new evidence date. Independent evidence or direct confirmation records an explicit transition and its basis. Refute or reject unsupported connections, recheck after relevant change, and archive when their question or scope ends. |
| Agent synthesis or brief | Derive support from leaf evidence and applicable corrections. Revise after dependent changes and recheck affected claims before acting. Preserve origin, scope, uncertainty, and source labels through compression. Supersede stale versions, and archive versions no active work or citation needs. |
| Source pointer | Preserve the source identity and route. Recheck availability and the referenced revision when used. Revise moved routes, mark unavailable targets, and supersede or archive obsolete pointers without asserting their source claims are false. |

A synthesis cannot support itself or turn a hypothesis into an observed fact.
Following another summary's citation must reach leaf evidence with its original
epistemic status. Validation of a harness report adds an observation and a supported
conclusion, preserving the report's origin. Conditional user corrections apply only
within their stated condition. When that condition ends, the note leaves active
constraints with its history intact. An unresolved condition is a reason to recheck
or ask, without silently dropping the correction.

Compaction preserves active corrections and necessary leaf evidence. Notes follow
kind-specific revision and retention rules rather than identical time decay. The
repository's reset-and-recreate schema policy still applies to all retained state.

## Provider boundaries

Existing [hooks](../../../pet_daemon/sources/hooks.py),
[Git](../../../pet_daemon/sources/git.py), and
[GitHub](../../../pet_daemon/sources/github.py) supply observations and availability.
Captured report context remains separate from later checkout state. Source adapters
normalize identities, revisions, and availability. Retrieval selects relevant data,
while application policy accepts responsibility and authority changes.

Optional ticket access uses a generic read-only interface, with Linear as the first
production adapter. Coordination consumes provider-qualified ticket identities,
canonical links, source revisions, observation times, native status, and available
relationships. Projects, cycles or iterations, assignment, and dependencies are
optional capabilities. Unsupported capability, missing access, failed refresh,
and a successful lookup finding no related work remain distinct. Normalize native
completion meaning only when the adapter can establish it.

The Linear adapter can retrieve through a configured read-only MCP connection.
Its scope includes assigned or followed issues and relevant projects, cycles, and
dependencies. Narrow exact watches may use direct API access with configurable
credentials when needed. Both routes share provider identity, source references,
revision and freshness rules so the same issue cannot become competing facts.
Research can use available MCP operations before any custom full-data collector
exists. Basic provider operations and curated retrieval enforce the same read
permissions. Ticket assignment and source prose cannot authorize effects.

Optional local `.md` and `.txt` meeting exports use a small searchable passage
index. Import records file identity, content revision, and passage locations without
requiring exhaustive interpretation. Expand passages on demand. Edits mark affected
support for recheck, and deletion makes that support unavailable. Neither event alone
proves a decision false or establishes its replacement. Retained historical passages
show their superseded or deleted status. Record meeting dates
only when supplied by the text or user. File modification and import time describe
freshness without inventing when a conversation occurred. A newly imported passage
does not establish that its discussion happened after a session or that a requirement
changed. Establish chronology and proposal/decision status from supporting evidence.
If either remains unknown, retain a possible conflict with that uncertainty.
Embedded instructions remain source material.

## Responsibilities and authority

A project supplies a durable identity and user-defined scope across repositories,
artifacts, sessions, and responsibilities. A responsibility records a finite desired
outcome or standing purpose, scope, explicit priority, adoption and permitted actions,
completion condition and evidence, dependencies, followups, lifecycle state, and
revision. Confirmed associations connect useful work without requiring a complete
model of the organization.

Responsibilities can be active, waiting on a dependency, paused, or completed.
Waiting records what must change before progress can resume. Pausing preserves the
unfinished purpose and suspends followthrough. A standing purpose survives individual
completed activities until its adopted scope ends or the user ends it. Resuming
rechecks relevant evidence before acting.

“Watch this PR until checks pass” adopts a bounded watch and one completion report.
Branch modification, merging, and desktop implementation require applicable action
authority. Imported requests and inferred intentions can inform questions and
proposals but cannot grant that authority. Existing user instructions take
precedence over source content and model suggestions. Accepted revisions record
changes to scope, priority, authority, dependencies, and completion conditions.

A source observation, an agent progress report, and a tentative interpretation
establish different claims. A successful harness exit establishes execution outcome.
An agent's “done” establishes its report. Completion requires evidence satisfying
the current adopted condition. Followups retain their reason, dependency or deadline,
and accepted scope across restarts.

## Ongoing coordination loop

The application gathers wake and restart reconciliation, changed source revisions,
owned-session results and readiness, user changes, due followups, and quiet-review
times. Each cause identifies the affected responsibility and question, or a recent
work thread and question before adoption. The coordinator coalesces causes under
that identity, retaining the latest relevant revision per source. Preserve the
candidate's first-queued time and last meaningful service time when more events
arrive. Repeated or noisy events neither reset waiting age nor multiply priority.
Source volume does not establish urgency.

Code evaluates eligibility against awake and pause state, due conditions, evidence
freshness, scope, current operations, available time, concurrency, and budget. Perform
exact checks in code where possible, including watch conditions and operation
identity. Reconcile unknown operations before issuing further effects for the
affected work. A blocked candidate leaves unrelated eligible work available.
Retain one background investigation and one active pet-initiated coding execution
under the configured limits. Preparation, active execution, native approval waits,
and uncertain consequential dispatch reserve coding capacity. An observed idle
READY session remains reusable and releases active execution capacity. Launch and
continue share the same effect-boundary capacity check. Fair selection does not
preempt a running harness.

Quiet eligible work receives a configurable review due time and maximum deferral
from its last meaningful consideration, or initial queue time before its first
service. Meaningful consideration is a completed assessment with a disposition.
Its service timestamp supplies the origin for the next review and deferral bound.
Polling, eligibility inspection, or inclusion in a model's candidate list alone
does not count. Overdue quiet work receives fair service ahead of ordinary
change-only work. An explicit do-this-next or exclusive user direction, or an
evidenced time-critical obligation, can override that ordering with a recorded
reason and revisit condition. Standing relative priority orders routine peers and
cannot repeatedly override overdue quiet review. Preserve visible overdue state
during an override. This service bound applies when work is eligible
and resources permit, without an absolute wall-clock promise while paused or out
of budget.

For routine peers, rotate projects by least recent meaningful service. Within a
project, respect explicit priorities and due urgency, then oldest waiting time and
stable identity. Overdue reviews apply within projects too, so a stream of routine
high-priority changes cannot indefinitely postpone quiet eligible work. If explicit
commitments conflict beyond this scheduling policy, ask the user. Recording an
override preserves the displaced work's waiting age and next review.

A reasoning pass receives a bounded set of eligible candidates, their reasons,
changed references, prior dispositions, and available coding capacity across
projects. The scheduler reserves a focus when
overdue fair service or an explicit user direction requires it. Otherwise the model
can select among the offered routine candidates and explain what deserves inquiry.
Return dispositions for the candidates actually assessed, including any reserved
focus, choosing investigation, authorized continuation, a question, waiting,
completion assessment, or no
finding, with reasons. Unassessed candidates remain pending.
Code validates authority, freshness, operation identity, and eligibility before
applying any choice. Selection can satisfy a review through a reasoned wait or
no-finding disposition. Merely skipping a candidate does not service it.

Persist each candidate's disposition, considered and serviced timestamps,
instruction and evidence revisions, selected action and deduplication identifiers,
and next review or wake condition. Considered time records actual assessment,
including an explicit deferral. Serviced time records a completed meaningful
assessment with a disposition for the question, whether model reasoning or an exact
application check. A no-action result still checkpoints the assessed revisions and
reason. A candidate skipped because budget ran out remains pending, without a false
consideration or service record. A candidate awaiting a human decision waits for
the reply unless changed scope or evidence warrants revisiting the question.
An explicit scheduling deferral can update considered time but cannot reset service
age. A reasoned wait answering the selected question counts as meaningful service.

Unchanged watch observations do not cause another model investigation. A due
followup can refresh the source, and exact checks can update the watch without
model interpretation. New meaningful evidence or a due review can reopen a
checkpointed question. Failed refresh records unavailability and cannot establish
completion. On wake or restart, reconcile and coalesce queued causes before
selection, preserving history and deduplication identifiers rather than producing
a burst of repeated inquiries or actions.

Each asynchronous inquiry captures applicable instruction, responsibility, and
evidence revisions. A correction, source change, pause, or authority change can
supersede its proposed transition or presentation. Revalidate affected results
before applying them, and record accepted state, citations, and reasons together.
Keep superseded results as historical inquiry records where retained. Establish
supported meaning and the reason to speak before persona expression.

Activity exposes pending and overdue work, dispositions, overrides, accepted
changes, source failures, and evidence. Chat supports inspection, correction, and
adoption. Feedback applies to its named claim, association, or suggestion under
the note lifecycle. Dismissal does not silently cancel a responsibility or suppress
unrelated work.

Finite completion assessment compares current leaf evidence with the recorded
condition and captured responsibility revision. Code checks scope, authority,
freshness, and source kinds before accepting the model's meaning judgment.
Reported or inferred success remains unresolved. Exact PR checks retain their
machine watch. An accepted completion, its evidence, consideration, Activity, and
pending notice commit together. Renderer acknowledgement records notice delivery.
A direct user confirmation records its actual human event and remains distinct
from independent verification. It can also end a standing purpose. Goal completion
never claims that a native harness was stopped.

## Lifecycle invariants

- Durable identities and accepted authority remain independent of SDK memory and
  provider conversation IDs. Importing or retrieving text never grants control.
- Source removal, supersession, and stale support remain visible in presented
  claims. Ranking cannot override evidence validity.
- Accepted completion cites evidence satisfying the current adopted condition.
  Failed refresh, silence, and process exit cannot supply a missing condition.
- Sleep suspends local coordination. Wake reconciles sources and due followups and
  prepares context before current-state claims. Owned external sessions may continue.
- Restarts preserve canonical state and working memory within a schema. Following
  the repository's [schema policy](../../../pet_daemon/storage/migrations.py),
  schema changes drop and recreate storage. Reset clears authority and pending
  followthrough even if external sessions remain. Fresh adoption is required, and
  no data-preserving migration is introduced.

## Alternatives

SDK memory alone cannot enforce exact authority or reconstruct evidence behind an
accepted transition. Restricting all discovery to adopted responsibilities delays
useful understanding of existing work. A required normalized company model or
universal association graph adds interpretation before focused retrieval can help.
Prepared briefs, compact memory, source adapters, and on-demand passages support
the product directly.

## Delivery

The [overview](overview.md) defines thirteen internal work packages within one
coordinated implementation run and direct merge to `main`. Package 2 establishes
minimal durable state, note origin and revision history, references, and Activity.
Package 3 supplies prepared briefs that filter notes by validity and scope, history
bootstrap, contextual retrieval, and an MCP facade against existing work and
provider fixtures. Package 5 connects that foundation to SDK conversation, typed
note operations, and package 4's composer. Packages 6 and 7 add the ticket interface
with Linear and meeting exports. Exercise their combined context at the information
integration checkpoint. Package 8 supplies adopted watches and basic watch timers.
Package 11 completes the ongoing scheduler, consideration history, and fair review
policy. Complete all dependencies as part of the same delivery.

## Consequences

This design requires freshness, cache invalidation, and correction handling alongside
conversation. Focused checks and actual generated responses verify useful context,
retrieval quality, grounding, and authority during delivery.

## Acceptance scenarios

1. **Seeded first invocation:** Configured harness history and notes produce a brief
   with recent requests, decisions, open questions, references, and human-input
   flags before new events arrive. Unavailable history remains explicit.
2. **Relevant older decision:** A current PR question expands through its ticket
   and project aliases. An older applicable decision outranks a fresh status update,
   and passage expansion supplies the decision's scope and revision.
3. **Cross-source coverage:** MCP and direct observations of one ticket deduplicate
   under one identity. Search preserves complementary meeting and session evidence
   with distinct provenance and reports a failed provider separately.
4. **Correction retained:** The user rejects a proposed project connection. Working
   memory retains the scoped correction, and unchanged evidence does not generate
   the same suggestion on the next conversation.
5. **Read discovery before adoption:** The pet investigates a possible dependency
   using configured sources and presents a supported hypothesis. It cannot launch
   a harness task without applicable action authority.
6. **Useful retrieval without a graph:** A small meeting index, existing session
   records, and configured provider search answer a question before any exhaustive
   company model or universal association graph exists.
7. **Watch completion:** An adopted PR watch reports once after a fresh observation
   satisfies the condition for the relevant revision. Unchanged polls remain quiet,
   and failed refresh after an earlier success cannot complete it.
8. **Superseded inquiry:** A user correction or source edit arrives during inquiry.
   The outdated result cannot update state or describe superseded scope as current.
9. **Transcript revision:** Edited or deleted passages withdraw current support for
   an inferred deadline. A supplied meeting date remains distinct from import time,
   and embedded instructions never expand authority.
10. **Waiting and standing work:** A dependency change wakes waiting work. A standing
    purpose survives one completed activity. Pausing preserves unfinished work.
11. **Restart and wake:** State survives restart within the schema. Wake prepares a
    refreshed brief. Schema reset clears authority despite surviving external sessions.
12. **Ticket contract:** A fake adapter with different native identifiers and no
    cycle support verifies that retrieval and coordination depend on the shared
    contract. Unsupported capability remains distinct from an empty result.
13. **Quiet work amid noise:** A busy project and a quiet project have eligible
    questions. Repeated source events coalesce, preserving waiting age. The quiet
    project's overdue review receives fair service ahead of routine changes. The
    same rule prevents quiet work within the busy project from starving.
14. **Priority and deadlines:** An evidenced time-critical obligation or explicit
    do-this-next direction overrides an overdue review with a reason and revisit
    condition. Activity keeps displaced work visibly overdue. A standing high
    relative priority or routine source volume cannot create repeated overrides.
15. **Budget and blocked work:** Candidates omitted after budget exhaustion remain
    pending without invented consideration or service. An uncertain operation
    blocks further effects for its work while other eligible work proceeds. Fair
    selection never interrupts an already running harness.
16. **Wake deduplication:** Repeated events across restart and wake merge under the
    same question, preserve first-queued age and prior dispositions, and reconcile
    unknown operations without duplicate launches or a burst of inquiries.
17. **Repeated hypothesis:** Several summaries repeat an uncertain dependency. Its
    support and evidence dates do not strengthen. Direct confirmation or independent
    evidence records a supported transition with preserved lineage.
18. **Binding correction:** A corrected project link does not recur through search,
    aliases, or a compact brief. A contradictory import prompts clarification and
    cannot demote the direct user correction because it is older.
19. **Decision continuity:** A newer mention of an old decision preserves its
    original scope and date. An explicit applicable replacement records supersession.
    Editing or deleting the source marks affected support for recheck rather than
    proving the decision false.
20. **Compacted synthesis:** A brief derived from a hypothesis keeps its tentative
    label and leaf citations through compaction. Another brief cannot cite that
    synthesis as independent validation. A harness report remains reported until a
    separate validation record establishes the outcome.
21. **Conditional scope:** A user correction applies while its stated condition
    holds. When that condition ends, its active constraint ends with history retained.
    If the condition is unresolved, recheck or ask before changing its applicability.
22. **Meaningful choice:** With no reserved fair or explicit focus, the model can
    choose among eligible routine candidates using their meaning and explain that
    choice. An overdue reserved focus must receive an assessed disposition, even
    when the outcome is a supported wait rather than a new investigation.
23. **Discussion and arrival time:** A meeting export becomes available after a
    session reports completion. The pet reads the passage before it has a structured
    decision record, preserves unknown discussion date or proposal status, and
    investigates a possible conflict. It does not infer a newer decision or changed
    requirement from import time alone.
