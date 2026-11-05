# Desktop coordinator implementation

- Status: Complete
- Date: 2026-11-05
- Checkout: `/Users/elliot/sc/simple-pet`
- Integration branch: `desktop-coordinator`
- Base: `8475c30` on `main`
- Delivery: One integrated implementation run, local checkpoints, direct merge to `main`

## User decisions

Follow the [overview](overview.md) and four owning specifications. Use Haiku 5.5
with adaptive thinking and medium effort. Build prepared work context and retrieval
before ongoing scheduling. The pet coordinates coding through an external adapter.
Read-only discovery can precede adoption. Task actions require canonical authority.
Use the existing reset-and-recreate schema policy. Keep user instructions, note
support, action authority, source freshness, and reported execution distinct.

The authorization covers the complete implementation, live model and desktop
acceptance, and direct merge. Live Haiku conversation and scheduled inquiry passed
the bounded acceptance in [verification](verification.md). The primary personal
harness is Claude Code in Kitty, with hook callbacks for observations and an owned
Kitty socket for terminal control. Recheck captured hook/readiness revisions before
serialized input and correlate callbacks with the requested command. Native atomic
comparison APIs are optional. Terminal delivery does not establish exactly-once
execution. Codex remains an optional backend.

## Active ownership

| Stream | Owned files and integration boundary |
| --- | --- |
| Foundation and integration | `pet_daemon/coordination/`, `pet_daemon/storage/coordination.py`, migrations, Memory facade, top-level config, application/daemon wiring, foundation tests, and this ledger. |
| Conversation | `pet_daemon/conversation.py`, Swift conversation/event/panel files, and focused conversation tests. Consumes `memory.coordination` and the SDK reply callable. |
| SDK and retrieval | `pet_daemon/companion/`, provider adapters and source-index codec, dependency manifests, and focused companion tests. Consumes canonical records and the Memory work snapshot APIs. |
| Desktop and Activity | `pet_daemon/desktop/`, explicit delegation hook fields, public adapter examples, Activity projection, and focused adapter/interaction tests. Consumes shared task, operation, and scheduler contracts. |

Workers share the checkout and preserve one another's purposeful edits. The
foundation worker integrates lifecycle wiring. Source-index schema belongs to the
foundation migration, and source payload codecs belong to the companion worker.

## Packages and acceptance

All 13 work packages are complete. The table records their acceptance evidence.

| Package | Depends on | Acceptance evidence |
| --- | --- | --- |
| 1: Contracts and fixtures | None | Specifications, canonical domain contracts, and serialized desktop protocol fixtures present. Focused contract checks passed. |
| 2: Durable foundation | 1 | Frozen records, transactional repository, schema 27, Memory facade, source-index tables, and config integration implemented. Thirteen focused persistence/reset checks and integrated consumers passed. |
| 3: Prepared context and retrieval | 2 | Brief, bounded native history bootstrap, current accepted human inputs, source map, ranking, passage expansion, freshness, changes, and selected MCP proxy implemented. Source/SDK suite passed 30 checks, including imported-input distinction and restart. |
| 4: Typed conversation | 2 | Controller, bounded history pages, renderer sync, and Swift composer implemented. Seven conversation transport checks, application-to-SDK checks, and nine Swift Chat checks passed. |
| 5: SDK conversation and notes | 3, 4 | SDK callable, typed note tools, and host authority context implemented. SDK correctness, application fake-SDK integration, and bounded live conversation/inquiry acceptance passed. |
| 6: Ticket interface and Linear | 2, 3 | Generic read-only contract, direct Linear adapter, and selected MCP normalization implemented. Fixtures and real stdio/HTTP MCP transport checks passed. Optional live Linear account retrieval remains untested. |
| 7: Meeting exports | 2, 3 | Searchable passages, complete snapshot edits/removals, bounded files, dates, and freshness implemented. Source suite passed. |
| 8: Adopted responsibility/watch | 5 | Accepted user ingress, adoption/revision, exact checks, scoped feedback, evidence-based finite completion, explicit human confirmation, and durable notice outbox implemented. Watch/workflow, completion, and no-active-session delivery/ack/restart checks passed. |
| 9: Desktop launch and recovery | 8 | Public process adapter, intent-before-effect, receipts, exact hook association, and reconciliation implemented. Adapter/hook/notify checks and fresh Claude Code/Kitty launch with real hooks passed. |
| 10: Owned follow-through | 9, 11 | Current authority, captured task/user/hook revisions, readiness, continuation, stop/open, and lost-receipt recovery implemented. Twenty-four public desktop checks and 40 private adapter tests passed. Actual continuation and active interruption passed with exact callback/native evidence. |
| 11: Ongoing scheduling | 5, 6, 7, 8 | Coalescing, bounded reasoning, quiet fairness, scoped unknown operations, source refresh, disposition/budget history, and available coding capacity implemented. Eleven loop regressions and integrated app/daemon checks passed. Actual accepted replies wake waiting-human questions without resetting their age. |
| 12: Full Activity/ambient | 10, 11 | Activity projection, canonical controls, source inspection, and notice delivery bridge implemented. Twelve Activity/conversation checks and 31 overlay checks passed. Isolated release acceptance confirmed passive speech, durable chat/reopen, Activity protocol delivery, and native conversation panels. Physical input/capture and visual Activity inspection were unavailable. |
| 13: Integrated release checks | 12 | Full automated check, final follow-up lint/Python checks, release build, generated-response assessment, independent reviews, actual desktop acceptance, and installed configuration/readback passed. |

## Published foundation contracts

`pet_daemon.coordination.models` defines frozen canonical records and closed
lifecycle enums. `Memory.coordination` supplies transactional chat acceptance,
duplicate suppression, stale completion rejection, failure/retry, note history,
instructions, responsibilities, task operations, candidates, consideration history,
pause state, and Activity. `Memory.work_facts()` and bounded
`Memory.recent_work_reports(limit)` expose existing evidence for context bootstrap.

The companion source index uses migration-owned tables and passed SQLite storage.
Its document identity and revision codec remains in the companion module. Top-level
configuration imports the companion's typed configuration and resolves configured
working and meeting paths relative to the config file.

## Validation

Branch creation preserved the untracked specifications. The focused command
`make test-python ARGS='Tests/test_coordination.py Tests/test_memory.py'` passed
60 tests and five subtests. Coverage includes restart, chat deduplication and retry,
stale replies, user correction history, hypothesis/report distinctions, rejected
import authority, unknown operations, stale consideration, pet-scoped clearing,
destructive schema reset, and future-schema rejection without writes. The test
script required process access outside the sandbox. An initial command naming a
nonexistent config test collected no tests, then the corrected command passed.

Existing ambient behavior and complete application-to-SDK conversation flow remain
under scoped regression checks. The command
`make test-python ARGS='Tests/test_coordination.py Tests/test_application.py'` passed
86 tests and four subtests. It exercises renderer input through the real controller,
prepared SDK invocation with a fake external stream, cited durable response,
duplicate transport, ambient suppression, and sleep interruption with explicit retry.
Scoped `make lint-python PYTHON_PATHS=...` passed lint, formatting, and type checks
for canonical records, storage, configuration, application, daemon, and their tests.
`PYTHON_PATHS` defaults to the full repository and supports exclusive worker checks.

The responsibility, scheduler, watch, and application run passed 104 tests and four
subtests. Separate application/runtime regressions passed the watch report across
restart before delivery, no active coding session, renderer acknowledgement, and
restart without reannouncement. The enabled daemon socket test used a fake SDK
stream and verified typed chat and sleep/wake without replay. The focused
`Tests/test_coordination_loop.py Tests/test_watches.py` run passed 23 checks after
review fixes for inspection failures, max deferral, explicit Later, effect
revalidation, finite watch conditions, cancellation/restart, failed control outcomes,
and explicit user correction supersession. Scoped Python lint/type/format checks
and repository Python complexity checks passed.

Schema 27 recreates schema 26 storage, clearing instructions and adopted authority.
Same-schema restarts preserve accepted state. Future schemas are rejected before
writes. Review findings added guarded failed-attempt revisions, immutable note
actors/evidence dates, and durable retry command receipts. UI review fixes bound
presentation history while retaining the complete canonical transcript, and add
renderer reconciliation. Companion review fixes revalidate expanded source passages.
The final source/SDK suite passed 30 checks. Legacy ambient runtime and desktop
fixtures disable the coordinator explicitly. Enabled-coordinator tests inject the
SDK query effect, so broad automated tests cannot make accidental live model calls.

Local checkpoints are `ea499e0` for specifications, `e43b4dd` for canonical
records/storage, `304124e` for integrated conversation, retrieval, scheduling,
desktop operations, and Activity, and `a4075f5` for current human direction and
shared coding capacity. Full `make check` passed, including Python 935
tests and 791 subtests, Swift tests, lint, type checks, complexity, and shell syntax.
`make build` passed and produced `.build/pet-overlay` for isolated acceptance.
The final follow-up gate passed `make lint` and the complete Python suite with 937
tests and 791 subtests in 28.14 seconds. It includes active-execution capacity,
bounded current human inputs, imported-input distinctions, and the hook observation
payload. The accepted-user reply wake regression passed through the actual
application/controller/SDK boundary, preserving queue/service ages and duplicate
suppression. Swift source remains covered by the complete check and release build.

Independent reviews verified operational source timestamps, valid empty retrieval,
finite completion, responsibility scope revision, and exact hook association.
Observed dated decision evidence completed a finite goal through the real model and
coordinator with one completion record and notice; a report-only CI claim remained
incomplete.

The personal Claude Code adapter passed 40 private tests and independent review
without critical findings. Authentication, fresh Kitty launch, real hook
correlation, continuation, and active interruption passed. Matching native
interruption evidence establishes the Stop result independently of the later
SessionEnd or user closure. [Verification](verification.md) records these evidence
boundaries, the release overlay checks, physical input/capture limitations, and
optional account coverage. The reviewed private adapter and validated configuration
are installed at persistent user-local locations. A fresh service recovered the
same task/native session through the installed wrapper after restart. Production
pet configuration, autostart, authentication, and home environment were unchanged.

## Direct integration readiness

The fetched `origin/main` and local `main` both point to `8475c30`. The feature
source checkpoint `a4075f5` has no divergent remote commits. Delivery uses a direct
fast-forward integration to `main`, with no PR workflow. Preserve the
native-acceptance worktrees and their branches. Remote publication is outside this
delivery request.
