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

The authorization covers the complete implementation, eventual live model and
desktop acceptance, and direct merge. The orchestrator defers live effects until
the automated checkpoint is ready and the first personal harness is selected.
Real personal-adapter acceptance remains pending configuration.

## Active ownership

| Stream | Owned files and integration boundary |
| --- | --- |
| Foundation and integration | `pet_daemon/coordination/`, `pet_daemon/storage/coordination.py`, migrations, Memory facade, top-level config, application/daemon wiring, foundation tests, and this ledger. |
| Conversation | `pet_daemon/conversation.py`, Swift conversation/event/panel files, and focused conversation tests. Consumes `memory.coordination` and the SDK reply callable. |
| SDK and retrieval | `pet_daemon/companion/`, provider adapters and source-index codec, dependency manifests, and focused companion tests. Consumes canonical records and the Memory work snapshot APIs. |

Workers share the checkout and preserve one another's purposeful edits. The
foundation worker integrates lifecycle wiring. Source-index schema belongs to the
foundation migration, and source payload codecs belong to the companion worker.

## Packages and acceptance

Package completion requires its focused checks and integrated consumers. The table
records implementation status without declaring unrun acceptance complete.

| Package | Depends on | Status and remaining acceptance |
| --- | --- | --- |
| 1: Contracts and fixtures | None | Specifications present. Canonical domain contracts published. Serialized fixtures and checks in progress. |
| 2: Durable foundation | 1 | Frozen records, transactional repository, schema 27, Memory facade, source-index tables, and config integration implemented. Focused persistence/reset checks passed. Complete integrated consumers before signoff. |
| 3: Prepared context and retrieval | 2 | Companion worker implementing brief, bootstrap, source map, ranking, passage expansion, and changes tools. |
| 4: Typed conversation | 2 | Controller and Swift composer implemented. Three Python controller checks, 23 existing Swift checks, six new Swift checks, Swift lint, and complexity passed. Application-to-SDK fake-boundary checks passed. Review fixes for bounded snapshots, renderer sync, and retry command identity in progress. |
| 5: SDK conversation and notes | 3, 4 | SDK callable, typed note tools, and host authority context implemented. Five companion checks and application fake-SDK integration passed. Retrieval/source freshness review fixes and full provider integration continue. |
| 6: Ticket interface and Linear | 2, 3 | SDK worker implementing generic read-only contract and first adapter. Scoped provider checks pending. |
| 7: Meeting exports | 2, 3 | SDK worker implementing passage index and revision/removal handling. Provider checks pending. |
| 8: Adopted responsibility/watch | 5 | Foundation owns implementation after shared state and conversation integration. Basic timers and one-time completion pending. |
| 9: Desktop launch and recovery | 8 | Public adapter and intent-before-effect integration pending. Personal acceptance pending configured harness. |
| 10: Owned follow-through | 9, 11 | Session readiness, authority, continuation, stop/open controls, and recovery pending. |
| 11: Ongoing scheduling | 5, 6, 7, 8 | Foundation owns candidate policy, fairness, disposition history, and runtime facade. SDK supplies reasoning callable, desktop adapter supplies validated effects. No separate desktop dispatch loop. |
| 12: Full Activity/ambient | 10, 11 | Minimal Activity records present. Presentation, controls, and interruption integration pending. |
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

Schema 27 recreates schema 26 storage, clearing instructions and adopted authority.
Same-schema restarts preserve accepted state. Future schemas are rejected before
writes. Review findings added guarded failed-attempt revisions, immutable note
actors/evidence dates, and durable retry command receipts. UI review fixes bound
presentation history while retaining the complete canonical transcript, and add
renderer reconciliation. Companion review fixes revalidate expanded source passages.
Typed provider and UI checks continue in their streams.

No daemon restart, live API request, or desktop harness launch has been performed.
