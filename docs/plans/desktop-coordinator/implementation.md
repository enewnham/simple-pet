# Desktop coordinator implementation

- Status: In progress
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
the bounded acceptance in [verification](verification.md). Personal Codex acceptance
uses launch, inspect, open, and stop. Its installed native API lacks an interaction
revision comparison for continuation, so automatic Codex continuation remains
disabled. Public automatic continuation retains its adapter revision requirement.

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

Package completion requires its focused checks and integrated consumers. The table
records implementation status without declaring unrun acceptance complete.

| Package | Depends on | Status and remaining acceptance |
| --- | --- | --- |
| 1: Contracts and fixtures | None | Specifications, canonical domain contracts, and serialized desktop protocol fixtures present. Focused contract checks passed. |
| 2: Durable foundation | 1 | Frozen records, transactional repository, schema 27, Memory facade, source-index tables, and config integration implemented. Thirteen focused persistence/reset checks and integrated consumers passed. |
| 3: Prepared context and retrieval | 2 | Brief, bounded native history bootstrap, source map, ranking, passage expansion, freshness, changes, and selected MCP proxy implemented. Source/SDK suite passed 25 checks. |
| 4: Typed conversation | 2 | Controller, bounded history pages, renderer sync, and Swift composer implemented. Seven conversation transport checks, application-to-SDK checks, and nine Swift Chat checks passed. |
| 5: SDK conversation and notes | 3, 4 | SDK callable, typed note tools, and host authority context implemented. Ten SDK correctness checks, application fake-SDK integration, and bounded live conversation/inquiry acceptance passed. Source integration continues. |
| 6: Ticket interface and Linear | 2, 3 | Generic read-only contract, direct Linear adapter, and selected MCP normalization implemented. Fixtures and real stdio/HTTP MCP transport checks passed in the 25-check source/SDK suite. Live account retrieval is optional. |
| 7: Meeting exports | 2, 3 | Searchable passages, complete snapshot edits/removals, bounded files, dates, and freshness implemented. Source suite passed. |
| 8: Adopted responsibility/watch | 5 | Accepted user ingress, adoption, exact check timers, scoped feedback, and durable one-report outbox implemented. Ten watch/workflow cases passed, plus no-active-session delivery/ack/restart proof. |
| 9: Desktop launch and recovery | 8 | Public process adapter, intent-before-effect, receipts, hook association, and reconciliation implemented. Adapter/hook/notify suite passed 62 checks and 93 subtests, with twelve desktop cases. Personal Codex acceptance pending. |
| 10: Owned follow-through | 9, 11 | Current authority, task/user/native interaction revisions, readiness, continuation, stop/open, lost-receipt recovery implemented and fake-adapter checks passed. Actual Codex automatic continuation unsupported and disabled. |
| 11: Ongoing scheduling | 5, 6, 7, 8 | Coalescing, bounded reasoning, quiet fairness, scoped unknown-operation handling, source refresh, and disposition/budget history implemented. Ten loop regressions passed, including review fixes. Shared app/daemon wiring complete. |
| 12: Full Activity/ambient | 10, 11 | Activity projection, canonical controls, source inspection, and notice delivery bridge implemented. Eleven Activity/conversation checks and nine Swift Chat checks passed. Full integrated acceptance pending. |
| 13: Integrated release checks | 12 | Full automated checks, generated-response assessment, real desktop acceptance, configuration guidance, and direct merge pending. |

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
The frozen source/SDK suite passed 25 checks. Legacy ambient runtime and desktop
fixtures disable the coordinator explicitly. Enabled-coordinator tests inject the
SDK query effect, so broad automated tests cannot make accidental live model calls.

Local checkpoints are `ea499e0` for specifications and `e43b4dd` for canonical
records/storage. The source/UI/desktop snapshot is frozen for its integrated
checkpoint. Full `make check`, private desktop acceptance, final
review, and direct merge remain pending. Production daemon restart has not run.
