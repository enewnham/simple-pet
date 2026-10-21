# Connected work and commentary

## Purpose

The pet embodies the agent swarm: what we are attempting, what we have produced, and
where the developer's involvement could help. Source observations connect work before
attention selects a subject and the model expresses it in the pet's voice.

## Source observations and relationships

Agent hooks record requests, execution, progress, and reported outcomes automatically.
Explicit requests create provisional work. Follow-ups refine that work and remain attributed
user evidence. Ending a session or reporting a successful turn does not complete the work.

The accepted session's checkout starts Git discovery. Git establishes the worktree root,
repository identity, checked-out branch, and revision. Exact repository and branch identities
resolve a GitHub PR. A fresh session requires a fresh Git observation, so a cached branch
cannot establish a new session's association after a checkout switch.

Several attempts can share artifacts without sharing intent or claiming authorship of every
change. Branch and PR identities support those relationships even when their revisions differ.
Known PRs remain independently observable after a worktree is pruned. Missing source links
remain unresolved. Failed refreshes preserve the age and uncertainty of successful observations.

Native GitHub stack metadata supplies exact related PR links. Each member retains its own
state and revision, and linked PRs refresh independently. A merged member does not complete
its stack. Issue tracking is deferred until a Linear data source is available.

## Work and attention

SQLite stores work, session associations, artifacts, attributed evidence, opportunity versions,
and surfacing outcomes. Source adapters feed the work store. One work view combines session
activity and derived opportunities for attention and commentary. GitHub does not schedule
separate remarks.

Derived actions include reviewing an explicit review request, inspecting failed checks on
the current PR revision, and considering branch output for PR preparation. Preparation requires
observed output, an agent progress report, and a successful lookup finding no associated PR.
It does not establish implementation completion or readiness. Draft state alone supplies
context without selecting an implementation or review action.

Known revisions determine whether checks and interpretations remain current. Changed output
requires reassessment. Terminal PR state resolves that artifact's obsolete opportunities
without completing the intended work. Repeated identical polls and changed file counts do
not create fresh interpretations by themselves.

## Commentary

The model receives a compact account of intent, attributed agent progress, PR subject,
observed state, relevant checks, and supported next steps. Exact IDs, Git revisions, branch
names, and raw file or commit counts stay in relationship resolution and freshness checks.
Actionable PR links remain available when they help identify the work.

Work in progress, possible attention, explicit input requests, and a person waiting for an
answer remain distinct. Unchecked lists, draft status, and failed checks establish no social
dependency or urgency. Only an explicit question, permission request, review request, or other
direct evidence supports a claim that input is required. Model instructions preserve these
boundaries and distinguish observed facts from agent reports.

Session summaries describe intent and reported results. Recent requests and current progress
take precedence. Request size is a soft target for adding optional context, with required
context preserved even when the request exceeds the target.

## Delivery

Speech for an opportunity carries a delivery ID. The renderer acknowledges accepting it
for display before the domain records it as surfaced. A socket send or animation completion
does not establish display, and display does not establish that the developer read it.

The delivery layer retries an unacknowledged packet within its runtime budget using the same
ID and content. Socket attempt state stays in memory. Durable surfacing outcomes suppress
repeated unchanged suggestions across restart. Manual report commands, decision workflows,
and a persistent socket delivery ledger are outside this implementation.

## Automatic integration example

`Tests/test_work_runtime.py` runs a fresh daemon against real temporary Git data and native
hook datagrams. GitHub CLI and model HTTP responses are simulated at their external boundaries.
The test does not seed work records or manually attach artifacts.

1. An intent requests session persistence and identifies its checkout. A later native turn
   report says implementation and validation succeeded.
2. Git discovers the repository and branch. GitHub supplies a draft PR titled
   “Persist coding session context” with a failing check on its current head.
3. The shared work view selects inspecting the failed check. Model input preserves both
   the agent's validation report and the independent failing check, along with draft status.
   It omits raw revision and count fields.
4. The renderer accepts the speech. An unchanged source refresh does not repeat the remark,
   and its surfaced version survives daemon restart.
5. A new head requires reassessment. Passing checks remove the check opportunity without
   establishing readiness. An explicit review request supplies a separate review opportunity.
6. A merge arriving during generation rejects the obsolete response and resolves the check
   opportunity without marking the work item complete.

The fixture establishes collection, association, selection, context, invalidation, and delivery.
It does not establish the quality of unsimulated model prose or desktop visual behavior.

## Code layout and verification

- `application.py` ingests observations, reconciles collection targets, and coordinates commentary.
- `agent_activity.py` coordinates accepted hooks, activity reactions, and commentary triggers.
- `sources/` collects hook, Git, PR, and stack observations.
- `work/` connects facts and derives opportunities and shared work views.
- `attention.py` selects subjects. `commentary/context.py` projects their meaningful facts.
- `commentary/cognition.py` requests commentary for the supplied selection.
- `storage/` persists work, selection history, and atomic message acceptance.
- `delivery.py` owns renderer acknowledgment and runtime retries.
- `sessions.py` owns execution. `episodes.py` retains separate reaction bookkeeping.

Schema version 7 preserves version 4/5/6 history and records attention history and message
outcomes. Speech history, message acceptance, and captured-version surfacing commit in one
transaction. Clear-memory includes work records and message outcomes.
`make check` covers Python, Swift core, assets, and shell syntax. `make build` compiles the
renderer and speech similarity helper. Desktop visual checks require a logged-in session.
